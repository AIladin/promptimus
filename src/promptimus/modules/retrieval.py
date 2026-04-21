from collections.abc import Hashable
from typing import Any

from promptimus.core.module import Module
from promptimus.rerankers import RerankerProtocol, RRFReranker
from promptimus.vectore_store.base import (
    BaseSearchResult,
    TextStoreProtocol,
    VectorStoreProtocol,
)

__all__ = ["RerankerProtocol", "RRFReranker", "RetrievalModule"]

_DEFAULT_RERANKER = RRFReranker()


class RetrievalModule(Module):
    def __init__(
        self,
        vector_store: VectorStoreProtocol | None = None,
        text_store: TextStoreProtocol | None = None,
        n_semantic: int = 10,
        n_text: int = 10,
        n_after_rerank: int = 10,
    ):
        super().__init__()
        if vector_store is None and text_store is None:
            raise ValueError("At least one of vector_store or text_store is required")
        self.vector_store = vector_store
        self.text_store = text_store
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

        reranker = self._reranker or _DEFAULT_RERANKER
        results = await reranker.forward(query, result_lists, **kwargs)
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
