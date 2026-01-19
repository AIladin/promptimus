import json
from typing import Any

from loguru import logger
from openinference.instrumentation import OITracer
from openinference.semconv.trace import SpanAttributes
from opentelemetry.trace import Status, StatusCode
from phoenix.otel import register

from promptimus.core import Module
from promptimus.dto import History, Message
from promptimus.modules import Prompt
from promptimus.modules.tool import Tool


def trace(module: Module, module_name: str, **provider_kwargs):
    tracer_provider = register(**provider_kwargs, set_global_tracer_provider=False)
    tracer = tracer_provider.get_tracer(__name__)

    _wrap_module_call(module, tracer, module_name)
    _trace_module(module, tracer, module_name)


def _wrap_prompt_call(
    prompt: Prompt, tracer: OITracer, module_path: str, prompt_name: str
):
    fn = prompt.forward

    async def wrapper(
        history: list[Message] | None = None,
        provider_kwargs: dict | None = None,
        **kwargs,
    ) -> Message:
        with tracer.start_as_current_span(
            f"{module_path}.{prompt_name}",
            openinference_span_kind="llm",
        ) as span:
            span.set_attribute(
                SpanAttributes.METADATA + ".prompt_digest", prompt.digest()
            )
            span.set_attribute(SpanAttributes.LLM_PROMPT_TEMPLATE, prompt.prompt.value)
            span.set_attribute(
                SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES, json.dumps(kwargs)
            )

            if prompt.prompt.value:
                span.set_attributes(
                    {
                        f"llm.input_messages.{0}.message.role": prompt.role.value,
                        f"llm.input_messages.{0}.message.content": prompt.prompt.value,
                    }
                )

            if history:
                span.set_input(
                    History.dump_python(history, exclude={"__all__": {"images"}})
                )
                for i, message in enumerate(history):
                    if message.tool_calls:
                        for call in message.tool_calls:
                            span.set_attributes(
                                {
                                    f"llm.input_messages.{i + 1}.message.tool_call": call.model_dump_json(),
                                }
                            )

                    span.set_attributes(
                        {
                            f"llm.input_messages.{i + 1}.message.role": message.role,
                            f"llm.input_messages.{i + 1}.message.content": message.content,
                        }
                    )
            result = await fn(history, provider_kwargs, **kwargs)
            span.set_output(result.model_dump())

            if result.tool_calls:
                for call in result.tool_calls:
                    span.set_attributes(
                        {
                            "llm.output_messages.0.message.content": call.model_dump_json()
                        }
                    )
            span.set_attributes(
                {
                    "llm.output_messages.0.message.role": result.role,
                    "llm.output_messages.0.message.content": result.content
                    or result.reasoning
                    or "",
                }
            )
            span.set_status(Status(StatusCode.OK))
        return result

    prompt.forward = wrapper  # ty:ignore[invalid-assignment]


def _wrap_module_call(module: Module, tracer: OITracer, module_path: str):
    fn = module.forward

    async def wrapper(
        history: list[Message] | Message | Any, *args, **kwargs
    ) -> Message:
        with tracer.start_as_current_span(
            module_path,
            openinference_span_kind="chain",
        ) as span:
            span.set_attribute(
                SpanAttributes.METADATA + ".module_digest", module.digest()
            )

            match history:
                case [*history_list] if all(
                    isinstance(i, Message) for i in history_list
                ):
                    span.set_input(
                        History.dump_python(
                            list(history_list), exclude={"__all__": {"images"}}
                        )
                    )
                case Message() as message:
                    span.set_input(message.model_dump(exclude={"images"}))
                case _:
                    span.set_input(str(history))
            try:
                result = await fn(history, *args, **kwargs)
            except TypeError as e:
                e.add_note(f"{module.__class__.__name__}:{module.path}")
                raise e

            if isinstance(result, Message):
                span.set_output(result.model_dump_json())
            else:
                span.set_output(str(result))
            span.set_status(Status(StatusCode.OK))
        return result

    module.forward = wrapper  # ty:ignore[invalid-assignment]


def _wrap_tool_call(
    tool: Tool,
    tracer: OITracer,
    module_path: str,
):
    fn = tool.forward

    async def wrapper(
        json_data: str,
        history: list[Message] | Message | str | None = None,
    ) -> Message:
        with tracer.start_as_current_span(
            f"{module_path}.{tool.name}",
            openinference_span_kind="tool",
        ) as span:
            span.set_input(json_data)
            span.set_tool(
                name=tool.name,
                description=tool.description.value,
                parameters=json.loads(json_data),
            )
            result = await fn(json_data, history=history)
            span.set_output(result)
            span.set_status(Status(StatusCode.OK))
        return result

    tool.forward = wrapper  # ty:ignore[invalid-assignment]


def _trace_module(module: Module, tracer: OITracer, module_path: str):
    for key, value in module._submodules.items():
        if isinstance(value, Prompt):
            logger.debug(f"Wrapping `{module_path}.{key}`")
            _wrap_prompt_call(value, tracer, module_path, key)

        elif isinstance(value, Tool):
            submodule_path = f"{module_path}.{key}"
            logger.debug(f"Wrapping `{submodule_path}`")
            _wrap_tool_call(value, tracer, module_path)
            _trace_module(value, tracer, submodule_path)

        elif isinstance(value, Module):
            submodule_path = f"{module_path}.{key}"
            logger.debug(f"Wrapping `{submodule_path}`")
            _wrap_module_call(value, tracer, submodule_path)
            _trace_module(value, tracer, submodule_path)
