from collections.abc import Hashable
from operator import itemgetter
from typing import Any

from promptimus.core.module import Module
from promptimus.vectore_store.base import BaseSearchResult


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
