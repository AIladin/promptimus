from typing import Hashable, Protocol

from pydantic import BaseModel

from promptimus.embedders.base import Embedding


class BaseVectorSearchResult(BaseModel):
    idx: Hashable
    content: str


class VectorStoreProtocol(Protocol):
    async def search(
        self, embedding: Embedding, **kwargs
    ) -> list[BaseVectorSearchResult]: ...

    async def insert(
        self, embedding: Embedding, content: str, *args, **kwargs
    ) -> Hashable: ...
