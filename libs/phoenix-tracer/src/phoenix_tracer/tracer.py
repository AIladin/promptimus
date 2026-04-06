import json
from typing import Any

from loguru import logger
from openinference.instrumentation import OITracer
from openinference.semconv.trace import SpanAttributes
from opentelemetry.trace import Status, StatusCode
from phoenix.otel import register

from promptimus.core import Module
from promptimus.dto import History, Message, Usage
from promptimus.modules import Prompt
from promptimus.modules.tool import Tool

type Pricing = dict[str, tuple[float, float]]


def trace(
    module: Module,
    module_name: str,
    pricing: Pricing | None = None,
    **provider_kwargs,
):
    tracer_provider = register(**provider_kwargs, set_global_tracer_provider=False)
    tracer = tracer_provider.get_tracer(__name__)

    _wrap_module_call(module, tracer, module_name)
    _trace_module(module, tracer, module_name, pricing)


def _set_usage_attributes(
    span,
    usage: Usage,
    model_name: str,
    pricing: Pricing | None,
):
    span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, usage.prompt_tokens)
    span.set_attribute(
        SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, usage.completion_tokens
    )
    span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, usage.total_tokens)

    if usage.cached_tokens is not None:
        span.set_attribute(
            SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ,
            usage.cached_tokens,
        )
    if usage.reasoning_tokens is not None:
        span.set_attribute(
            SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING,
            usage.reasoning_tokens,
        )

    if not pricing:
        return

    input_per_m, output_per_m = pricing.get(model_name, (0, 0))
    if input_per_m == 0 and output_per_m == 0:
        return

    input_cost = usage.prompt_tokens * input_per_m / 1_000_000
    output_cost = usage.completion_tokens * output_per_m / 1_000_000
    span.set_attribute(SpanAttributes.LLM_COST_PROMPT, input_cost)
    span.set_attribute(SpanAttributes.LLM_COST_COMPLETION, output_cost)
    span.set_attribute(SpanAttributes.LLM_COST_TOTAL, input_cost + output_cost)


def _wrap_prompt_call(
    prompt: Prompt,
    tracer: OITracer,
    module_path: str,
    prompt_name: str,
    pricing: Pricing | None = None,
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
                        for j, call in enumerate(message.tool_calls):
                            span.set_attributes(
                                {
                                    f"llm.input_messages.{i + 1}.message.tool_call.{j + 1}": call.model_dump_json(),
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

            span.set_attribute(SpanAttributes.LLM_MODEL_NAME, prompt.llm.model_name)
            if result.usage:
                _set_usage_attributes(
                    span, result.usage, prompt.llm.model_name, pricing
                )

            if result.tool_calls:
                for call in result.tool_calls:
                    span.set_attributes(
                        {
                            "llm.output_messages.0.message.role": result.role,
                            "llm.output_messages.0.message.content": call.model_dump_json(),
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
        history: list[Message] | Message | Any | None = None, *args, **kwargs
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
                if history is not None:
                    result = await fn(history, *args, **kwargs)
                else:
                    result = await fn(*args, **kwargs)
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


def _trace_module(
    module: Module,
    tracer: OITracer,
    module_path: str,
    pricing: Pricing | None = None,
):
    for key, value in module._submodules.items():
        if isinstance(value, Prompt):
            logger.debug(f"Wrapping `{module_path}.{key}`")
            _wrap_prompt_call(value, tracer, module_path, key, pricing)

        elif isinstance(value, Tool):
            submodule_path = f"{module_path}.{key}"
            logger.debug(f"Wrapping `{submodule_path}`")
            _wrap_tool_call(value, tracer, module_path)
            _trace_module(value, tracer, submodule_path, pricing)

        elif isinstance(value, Module):
            submodule_path = f"{module_path}.{key}"
            logger.debug(f"Wrapping `{submodule_path}`")
            _wrap_module_call(value, tracer, submodule_path)
            _trace_module(value, tracer, submodule_path, pricing)
