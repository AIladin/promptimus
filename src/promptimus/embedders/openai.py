from openai import AsyncOpenAI, RateLimitError

from promptimus.common.rate_limiting import RateLimitedClient

from .base import Embedding


class OpenAILikeEmbedder(RateLimitedClient[list[Embedding]]):
    RETRY_ERRORS = (RateLimitError,)

    def __init__(
        self,
        model_name: str,
        embed_kwargs: dict | None = None,
        max_concurrency: int = 10,
        n_retries: int = 5,
        base_wait: float = 3.0,
        **client_kwargs,
    ):
        super().__init__(max_concurrency, n_retries, base_wait)
        self.client = AsyncOpenAI(**client_kwargs)
        self._model_name = model_name
        self.embed_kwargs = embed_kwargs or {}

    async def _request(self, texts: list[str], **kwargs) -> list[Embedding]:
        """Perform one embedding API call and return embeddings array."""
        response = await self.client.embeddings.create(
            input=texts,
            model=self._model_name,
            **{**self.embed_kwargs, **kwargs},
        )
        return [e.embedding for e in response.data]

    async def aembed_batch(self, texts: list[str], **kwargs) -> list[Embedding]:
        """Public interface: embed multiple texts with retry logic."""
        return await self.execute_request(texts, **kwargs)

    async def aembed(self, text: str, **kwargs) -> Embedding:
        """Embed a single text and return 1D array."""
        result = await self.aembed_batch([text], **kwargs)
        return result[0]

    @property
    def model_name(self) -> str:
        return self._model_name
