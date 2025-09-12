from typing import Protocol

from pydantic import BaseModel

from promptimus.embedders.base import Embedding


class BaseVectorSearchResult(BaseModel):
    idx: str
    content: str


class VectorStoreProtocol(Protocol):
    async def search(
        self, embedding: Embedding, n_results: int, max_distance: float, **kwargs
    ) -> list[BaseVectorSearchResult]: ...

    async def insert(
        self, embedding: Embedding, content: str, *args, **kwargs
    ) -> str: ...

    async def delete(self, idx: str): ...
