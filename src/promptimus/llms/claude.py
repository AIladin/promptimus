import json
from typing import Any

from anthropic import AsyncAnthropic, RateLimitError
from anthropic.types import MessageParam

from promptimus.common.rate_limiting import RateLimitedClient
from promptimus.dto import Message, MessageRole, ToolFunction, ToolRequest, Usage


def _force_no_additional_properties(schema: dict) -> dict:
    """Anthropic requires `additionalProperties: false` on every object node."""
    if schema.get("type") == "object":
        schema["additionalProperties"] = False
    for value in schema.values():
        if isinstance(value, dict):
            _force_no_additional_properties(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    _force_no_additional_properties(item)
    return schema


def _image_source(url: str) -> dict:
    if url.startswith("data:"):
        header, data = url.split(",", 1)
        media_type = header[len("data:") : header.index(";")]
        return {"type": "base64", "media_type": media_type, "data": data}
    return {"type": "url", "url": url}


class Claude(RateLimitedClient[Message]):
    RETRY_ERRORS = (RateLimitError,)

    def __init__(
        self,
        model_name: str,
        call_kwargs: dict | None = None,
        max_tokens: int = 16000,
        max_concurrency: int = 10,
        n_retries: int = 5,
        base_wait: float = 3.0,
        **client_kwargs,
    ):
        super().__init__(max_concurrency, n_retries, base_wait)
        self.client = AsyncAnthropic(**client_kwargs)
        self._model_name = model_name
        self.max_tokens = max_tokens
        self.call_kwargs = call_kwargs or {}

    def serialize_message(self, message: Message) -> MessageParam:
        if message.role == MessageRole.TOOL:
            if message.tool_call_id is not None:
                tool_result: list[Any] = [
                    {
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id,
                        "content": message.content,
                    }
                ]
                return {"role": "user", "content": tool_result}
            return {"role": "user", "content": message.content}

        role = "assistant" if message.role == MessageRole.ASSISTANT else "user"

        content: list[Any] = []
        if message.content:
            content.append({"type": "text", "text": message.content})
        for img in message.images:
            content.append({"type": "image", "source": _image_source(img.url)})
        for tool_call in message.tool_calls or []:
            content.append(
                {
                    "type": "tool_use",
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "input": json.loads(tool_call.function.arguments),
                }
            )

        return {"role": role, "content": content}

    def _build_kwargs(self, kwargs: dict) -> dict:
        """Translate OpenAI-shaped provider kwargs into Anthropic's format."""
        kwargs = {**self.call_kwargs, **kwargs}

        if "tools" in kwargs:
            kwargs["tools"] = [
                {
                    "name": tool["function"]["name"],
                    "description": tool["function"].get("description", ""),
                    "input_schema": tool["function"]["parameters"],
                }
                for tool in kwargs["tools"]
            ]

        if (response_format := kwargs.pop("response_format", None)) is not None:
            schema = _force_no_additional_properties(
                response_format["json_schema"]["schema"]
            )
            kwargs["output_config"] = {
                "format": {"type": "json_schema", "schema": schema}
            }

        return kwargs

    async def _request(self, history: list[Message], **kwargs) -> Message:
        system = "\n\n".join(m.content for m in history if m.role == MessageRole.SYSTEM)
        messages = [
            self.serialize_message(m) for m in history if m.role != MessageRole.SYSTEM
        ]

        request_kwargs = self._build_kwargs(kwargs)
        if system:
            request_kwargs["system"] = system

        response = await self.client.messages.create(
            model=self._model_name,
            max_tokens=self.max_tokens,
            messages=messages,
            **request_kwargs,
        )

        content = ""
        reasoning = None
        tool_calls = None
        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "thinking":
                reasoning = (reasoning or "") + block.thinking
            elif block.type == "tool_use":
                tool_calls = tool_calls or []
                tool_calls.append(
                    ToolRequest(
                        id=block.id,
                        type="function",
                        function=ToolFunction(
                            name=block.name,
                            arguments=json.dumps(block.input),
                        ),
                    )
                )

        usage = None
        if response.usage:
            usage = Usage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                cached_tokens=response.usage.cache_read_input_tokens,
            )

        return Message(
            role=MessageRole.ASSISTANT,
            content=content,
            tool_calls=tool_calls,
            reasoning=reasoning,
            usage=usage,
        )

    async def achat(self, history: list[Message], **kwargs) -> Message:
        return await self.execute_request(history, **kwargs)

    @property
    def model_name(self) -> str:
        return self._model_name
