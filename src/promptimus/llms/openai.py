import asyncio

from loguru import logger
from openai import AsyncOpenAI, RateLimitError

from promptimus import errors
from promptimus.dto import History, Message, MessageRole, ToolRequest


class OpenAILike:
    def __init__(
        self,
        model_name: str,
        call_kwargs: dict | None = None,
        max_concurrency: int = 10,
        n_retries: int = 5,
        base_wait: float = 3.0,
        **client_kwargs,
    ):
        self.client = AsyncOpenAI(**client_kwargs)
        self.model_name = model_name
        self.call_kwargs = call_kwargs or {}
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._n_retries = n_retries
        self._base_wait = base_wait
        self._suppress_logs = [asyncio.Event() for _ in range(n_retries)]
        self._reset_log_supression()

    async def _single_request(self, history: list[Message], **kwargs) -> Message:
        """Perform one API call and return a Message or raise errors."""
        response = await self.client.chat.completions.create(
            messages=History.dump_python(history),
            model=self.model_name,
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

    def _reset_log_supression(self):
        """Reset the log suppression for all retries."""
        for event in self._suppress_logs:
            event.clear()

    async def achat(self, history: list[Message], **kwargs) -> Message:
        """Public interface: perform request under concurrency limit and retry using server-specified wait."""
        last_error = None
        hit_rl = False
        for attempt in range(self._n_retries):
            try:
                async with self._semaphore:
                    result = await self._single_request(history, **kwargs)
                    if hit_rl:
                        logger.info("Rate limit resolved.")
                        self._reset_log_supression()

                    return result
            except RateLimitError as err:
                wait_sec = self._base_wait * (2 ** (attempt))
                if not self._suppress_logs[attempt].is_set():
                    hit_rl = True
                    logger.warning(
                        f"Rate limit hit (attempt {attempt + 1}/{self._n_retries}), "
                        f"waiting {wait_sec:.3f}s (base {self._base_wait:.3f}s, exponential backoff)"
                    )
                    self._suppress_logs[attempt].set()

                last_error = err

                await asyncio.sleep(wait_sec)

        # all retries exhausted
        logger.error("Exhausted retries due to rate limit.")
        self._reset_log_supression()
        last_error = last_error or errors.MaxIterExceeded()

        raise last_error
