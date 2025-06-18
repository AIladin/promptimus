from openai import AsyncOpenAI, RateLimitError

from promptimus.common.rate_limiting import RateLimitedClient
from promptimus.dto import History, Message, MessageRole, ToolRequest


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

    async def _request(self, history: list[Message], **kwargs) -> Message:
        """Perform one API call and return a Message or raise errors."""
        response = await self.client.chat.completions.create(
            messages=History.dump_python(history),
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

        return Message(
            role=MessageRole.ASSISTANT,
            content=raw.content or "",
            tool_calls=tool_calls,
        )

    async def achat(self, history: list[Message], **kwargs) -> Message:
        """Public interface: perform request under concurrency limit and retry using server-specified wait."""
        return await self.execute_request(history, **kwargs)

    @property
    def model_name(self) -> str:
        return self._model_name
