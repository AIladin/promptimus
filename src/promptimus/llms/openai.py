from openai import AsyncOpenAI, RateLimitError
from openai.types.chat import (
    ChatCompletionMessageParam,
)

from promptimus.common.rate_limiting import RateLimitedClient
from promptimus.dto import Message, MessageRole, ToolRequest, Usage


class OpenAILike(RateLimitedClient[Message]):
    RETRY_ERRORS = (RateLimitError,)

    def __init__(
        self,
        model_name: str,
        call_kwargs: dict | None = None,
        max_concurrency: int = 10,
        n_retries: int = 5,
        base_wait: float = 3.0,
        **client_kwargs,
    ):
        super().__init__(max_concurrency, n_retries, base_wait)
        self.client = AsyncOpenAI(**client_kwargs)
        self._model_name = model_name
        self.call_kwargs = call_kwargs or {}

    def serialize_message(self, message: Message) -> ChatCompletionMessageParam:
        data = message.model_dump(exclude={"content", "images"}, exclude_none=True)

        data["content"] = [
            {"type": "text", "text": message.content},
            *(
                {"type": "image_url", "image_url": {"url": img.url}}
                for img in message.images
            ),
        ]

        return data

    async def _request(self, history: list[Message], **kwargs) -> Message:
        """Perform one API call and return a Message or raise errors."""
        response = await self.client.chat.completions.create(
            messages=[self.serialize_message(m) for m in history],
            model=self._model_name,
            **{**self.call_kwargs, **kwargs},
        )
        assert response.choices, response

        raw = response.choices[0].message
        tool_calls = None
        if raw.tool_calls:
            tool_calls = [
                ToolRequest.model_validate(tc, from_attributes=True)
                for tc in raw.tool_calls
            ]

        usage = None
        if response.usage:
            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                cached_tokens=response.usage.prompt_tokens_details.cached_tokens
                if response.usage.prompt_tokens_details
                else None,
                reasoning_tokens=response.usage.completion_tokens_details.reasoning_tokens
                if response.usage.completion_tokens_details
                else None,
            )

        return Message(
            role=MessageRole.ASSISTANT,
            content=raw.content or "",
            tool_calls=tool_calls,
            reasoning=raw.reasoning if hasattr(raw, "reasoning") else None,
            usage=usage,
        )

    async def achat(self, history: list[Message], **kwargs) -> Message:
        """Public interface: perform request under concurrency limit and retry using server-specified wait."""
        return await self.execute_request(history, **kwargs)

    @property
    def model_name(self) -> str:
        return self._model_name
