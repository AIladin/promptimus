from typing import Any

from openai import AsyncOpenAI, RateLimitError

from promptimus.common.rate_limiting import RateLimitedClient
from promptimus.vectore_store.base import BaseSearchResult


class OpenAILikeReranker(RateLimitedClient[list[tuple[int, float]]]):
    RETRY_ERRORS = (RateLimitError,)

    def __init__(
        self,
        model_name: str,
        rerank_kwargs: dict | None = None,
        max_concurrency: int = 10,
        n_retries: int = 5,
        base_wait: float = 3.0,
        **client_kwargs: Any,
    ):
        super().__init__(max_concurrency, n_retries, base_wait)
        self.client = AsyncOpenAI(**client_kwargs)
        self._model_name = model_name
        self.rerank_kwargs = rerank_kwargs or {}

    async def _request(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
        **kwargs: Any,
    ) -> list[tuple[int, float]]:
        body: dict[str, Any] = {
            "model": self._model_name,
            "query": query,
            "documents": documents,
            **self.rerank_kwargs,
            **kwargs,
        }
        if top_n is not None:
            body["top_n"] = top_n

        raw = await self.client.post("/rerank", body=body, cast_to=dict)
        return [(r["index"], r["relevance_score"]) for r in raw["results"]]

    async def arerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
        **kwargs: Any,
    ) -> list[tuple[int, float]]:
        return await self.execute_request(query, documents, top_n=top_n, **kwargs)

    async def forward(
        self,
        query: str,
        result_lists: list[list[BaseSearchResult]],
        **kwargs: Any,
    ) -> list[BaseSearchResult]:
        unique: dict = {}
        for results in result_lists:
            for r in results:
                unique.setdefault(r.idx, r)
        docs = list(unique.values())
        if not docs:
            return []

        ranking = await self.arerank(query, [d.content for d in docs])
        return [docs[i].model_copy(update={"score": score}) for i, score in ranking]

    @property
    def model_name(self) -> str:
        return self._model_name
