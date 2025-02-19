import json
from typing import Any

from loguru import logger
from openinference.instrumentation import OITracer
from openinference.semconv.trace import SpanAttributes
from opentelemetry.trace import Status, StatusCode
from phoenix.otel import register

from promptimus.core import Module
from promptimus.core.parameters import Prompt
from promptimus.dto import History, Message
from promptimus.modules.tool import Tool


def trace(module: Module, module_name: str, **provider_kwargs):
    tracer_provider = register(**provider_kwargs, set_global_tracer_provider=False)
    tracer = tracer_provider.get_tracer(__name__)

    _wrap_module_call(module, tracer, module_name)
    _trace_module(module, tracer, module_name)


def _wrap_provider_call(
    prompt: Prompt, tracer: OITracer, module_path: str, prompt_name: str
):
    def wrap(fn):
        async def wrapper(full_input: list[Message]) -> Message:
            with tracer.start_as_current_span(
                f"{module_path}.{prompt_name}",
                openinference_span_kind="llm",
            ) as span:
                span.set_attribute(SpanAttributes.LLM_PROMPT_TEMPLATE, prompt.value)
                span.set_input(History.dump_python(full_input))
                for i, message in enumerate(full_input):
                    span.set_attributes(
                        {
                            f"llm.input_messages.{i}.message.role": message.role,
                            f"llm.input_messages.{i}.message.content": message.content,
                        }
                    )
                result = await fn(full_input)
                span.set_output(result.model_dump())
                span.set_attributes(
                    {
                        "llm.output_messages.0.message.role": result.role,
                        "llm.output_messages.0.message.content": result.content,
                    }
                )
                span.set_status(Status(StatusCode.OK))
            return result

        return wrapper

    prompt._call_prvider = wrap(prompt._call_prvider)


def _wrap_module_call(module: Module, tracer: OITracer, module_path: str):
    def wrap(fn):
        async def wrapper(history: list[Message] | Message | Any, **kwargs) -> Message:
            with tracer.start_as_current_span(
                module_path,
                openinference_span_kind="chain",
            ) as span:
                if isinstance(history, list):
                    span.set_input(History.dump_python(history))
                elif isinstance(history, Message):
                    span.set_input(history.model_dump())
                else:
                    span.set_input(str(history))
                result = await fn(history, **kwargs)
                span.set_output(result.model_dump_json())
                span.set_status(Status(StatusCode.OK))
            return result

        return wrapper

    module.forward = wrap(module.forward)


def _wrap_tool_call(
    tool: Tool,
    tracer: OITracer,
    module_path: str,
):
    def wrap(fn):
        async def wrapper(json_data: str) -> Message:
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
                result = await fn(json_data)
                span.set_output(result)
                span.set_status(Status(StatusCode.OK))
            return result

        return wrapper

    tool.forward = wrap(tool.forward)


def _trace_module(module: Module, tracer: OITracer, module_path: str):
    for key, value in module._parameters.items():
        if isinstance(value, Prompt):
            logger.debug(f"Wrapping `{module_path}.{key}`")
            _wrap_provider_call(value, tracer, module_path, key)

    for key, value in module._submodules.items():
        if isinstance(value, Tool):
            logger.debug(f"Wrapping `{module_path}.{key}`")
            _wrap_tool_call(value, tracer, module_path)

        elif isinstance(value, Module):
            submodule_path = f"{module_path}.{key}"
            logger.debug(f"Wrapping `{submodule_path}`")
            _wrap_module_call(value, tracer, submodule_path)
            _trace_module(value, tracer, submodule_path)
