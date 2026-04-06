import uuid
from operator import itemgetter

from rank_bm25 import BM25Okapi

from promptimus.embedders.base import Embedding
from promptimus.embedders.ops import cosine
from promptimus.vectore_store.base import BaseSearchResult


class MemoryStore:
    def __init__(self):
        self.docs: dict[str, str] = {}
        self.embeddings: dict[str, Embedding] = {}
        self._bm25: BM25Okapi | None = None
        self._bm25_ids: list[str] = []
        self._dirty: bool = True

    def _rebuild_bm25(self):
        self._bm25_ids = list(self.docs.keys())
        corpus = [self.docs[id_].lower().split() for id_ in self._bm25_ids]
        self._bm25 = BM25Okapi(corpus) if corpus else None
        self._dirty = False

    async def vector_search(
        self, embedding: Embedding, n_results: int, **kwargs
    ) -> list[BaseSearchResult]:
        scored = [(id_, cosine(embedding, emb)) for id_, emb in self.embeddings.items()]
        scored.sort(key=itemgetter(1), reverse=True)
        return [
            BaseSearchResult(idx=id_, content=self.docs[id_], score=score)
            for id_, score in scored[:n_results]
        ]

    async def vector_insert(self, embedding: Embedding, content: str, **kwargs) -> str:
        id_ = str(uuid.uuid4())
        self.docs[id_] = content
        self.embeddings[id_] = embedding
        self._dirty = True
        return id_

    async def text_search(
        self, query: str, n_results: int, **kwargs
    ) -> list[BaseSearchResult]:
        if self._dirty or self._bm25 is None:
            self._rebuild_bm25()

        if self._bm25 is None:
            return []

        scores = self._bm25.get_scores(query.lower().split())
        scored = list(zip(self._bm25_ids, scores))
        scored.sort(key=itemgetter(1), reverse=True)
        return [
            BaseSearchResult(idx=id_, content=self.docs[id_], score=float(score))
            for id_, score in scored[:n_results]
            if score > 0
        ]

    async def text_insert(self, content: str, **kwargs) -> str:
        id_ = str(uuid.uuid4())
        self.docs[id_] = content
        self._dirty = True
        return id_

    async def delete(self, idx: str, **kwargs):
        self.docs.pop(idx, None)
        self.embeddings.pop(idx, None)
        self._dirty = True
