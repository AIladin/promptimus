from collections.abc import Hashable
from operator import itemgetter
from typing import Any, Protocol

from promptimus.core.module import Module
from promptimus.vectore_store.base import (
    BaseSearchResult,
    TextStoreProtocol,
    VectorStoreProtocol,
)


class RerankerProtocol(Protocol):
    async def forward(
        self, query: str, result_lists: list[list[BaseSearchResult]], **kwargs: Any
    ) -> list[BaseSearchResult]: ...


class RRFReranker(Module):
    def __init__(self, k: int = 60):
        super().__init__()
        self.k = k

    async def forward(
        self, query: str, result_lists: list[list[BaseSearchResult]], **kwargs: Any
    ) -> list[BaseSearchResult]:
        scores: dict[Hashable, float] = {}
        docs: dict[Hashable, BaseSearchResult] = {}

        for results in result_lists:
            for rank, r in enumerate(results):
                scores[r.idx] = scores.get(r.idx, 0) + 1 / (self.k + rank + 1)
                docs.setdefault(r.idx, r)

        sorted_items = sorted(scores.items(), key=itemgetter(1), reverse=True)
        return [
            docs[idx].model_copy(update={"score": score}) for idx, score in sorted_items
        ]


class RetrievalModule(Module):
    def __init__(
        self,
        vector_store: VectorStoreProtocol | None = None,
        text_store: TextStoreProtocol | None = None,
        reranker: RerankerProtocol | None = None,
        n_semantic: int = 10,
        n_text: int = 10,
        n_after_rerank: int = 10,
    ):
        super().__init__()
        if vector_store is None and text_store is None:
            raise ValueError("At least one of vector_store or text_store is required")
        self.vector_store = vector_store
        self.text_store = text_store
        self.reranker = reranker or RRFReranker()
        self.n_semantic = n_semantic
        self.n_text = n_text
        self.n_after_rerank = n_after_rerank

    async def forward(self, query: str, **kwargs: Any) -> list[BaseSearchResult]:
        result_lists: list[list[BaseSearchResult]] = []

        if self.vector_store is not None:
            embedding = await self.embedder.aembed(query)
            result_lists.append(
                await self.vector_store.vector_search(
                    embedding, n_results=self.n_semantic, **kwargs
                )
            )

        if self.text_store is not None:
            result_lists.append(
                await self.text_store.text_search(
                    query, n_results=self.n_text, **kwargs
                )
            )

        results = await self.reranker.forward(query, result_lists, **kwargs)
        return results[: self.n_after_rerank]

    async def insert(self, documents: list[str], **kwargs: Any) -> list[Hashable]:
        if self.vector_store is None:
            raise ValueError("vector_store is required for insert")

        embeddings = await self.embedder.aembed_batch(documents)

        ids = []
        for embedding, doc in zip(embeddings, documents):
            id_ = await self.vector_store.vector_insert(embedding, doc, **kwargs)
            ids.append(id_)

            if self.text_store is not None:
                await self.text_store.text_insert(doc, **kwargs)

        return ids
