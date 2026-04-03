from collections.abc import Hashable
from typing import Protocol

from pydantic import BaseModel

from promptimus.embedders.base import Embedding


class BaseSearchResult[ID: Hashable](BaseModel):
    idx: ID
    content: str
    score: float = 0.0


class VectorStoreProtocol[ID: Hashable](Protocol):
    async def vector_search(
        self, embedding: Embedding, n_results: int, **kwargs
    ) -> list[BaseSearchResult]: ...

    async def vector_insert(
        self, embedding: Embedding, content: str, **kwargs
    ) -> ID: ...

    async def delete(self, idx: ID): ...


class TextStoreProtocol[ID: Hashable](Protocol):
    async def text_search(
        self, query: str, n_results: int, **kwargs
    ) -> list[BaseSearchResult]: ...

    async def text_insert(self, content: str, **kwargs) -> ID: ...

    async def delete(self, idx: ID): ...
