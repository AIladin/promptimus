import asyncio
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from loguru import logger

from promptimus import errors

T = TypeVar("T")


class RateLimitedClient(ABC, Generic[T]):
    """Base class for clients with rate limiting and retry logic."""

    RETRY_ERRORS: tuple[type[Exception], ...]

    def __init__(
        self,
        max_concurrency: int = 10,
        n_retries: int = 5,
        base_wait: float = 3.0,
    ):
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._n_retries = n_retries
        self._base_wait = base_wait
        self._suppress_logs = [asyncio.Event() for _ in range(n_retries)]
        self._reset_log_supression()

    def _reset_log_supression(self):
        """Reset the log suppression for all retries."""
        for event in self._suppress_logs:
            event.clear()

    @abstractmethod
    async def _request(self, *args, **kwargs) -> T: ...

    async def execute_request(self, *args, **kwargs) -> T:
        """Execute an operation with retry logic and rate limiting."""
        last_error = None
        hit_rl = False

        for attempt in range(self._n_retries):
            try:
                async with self._semaphore:
                    result = await self._request(*args, **kwargs)
                    if hit_rl:
                        logger.info(f"{self.__class__.__name__} rate limit resolved.")
                        self._reset_log_supression()
                    return result
            except self.RETRY_ERRORS as err:
                wait_sec = self._base_wait * (2**attempt)
                if not self._suppress_logs[attempt].is_set():
                    hit_rl = True
                    logger.warning(
                        f"{self.__class__.__name__} rate limit hit (attempt {attempt + 1}/{self._n_retries}), "
                        f"waiting {wait_sec:.3f}s (base {self._base_wait:.3f}s, exponential backoff)"
                    )
                    self._suppress_logs[attempt].set()
                last_error = err
                await asyncio.sleep(wait_sec)

        # All retries exhausted
        logger.error(
            f"Exhausted retries for {self.__class__.__name__} due to rate limit."
        )
        self._reset_log_supression()
        last_error = last_error or errors.ProviderRetryExceded()
        raise last_error
