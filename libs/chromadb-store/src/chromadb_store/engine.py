import asyncio
import uuid
from typing import Any, Union

from chromadb.api import AsyncClientAPI, ClientAPI
from chromadb.api.models.AsyncCollection import AsyncCollection
from chromadb.api.models.Collection import Collection

from promptimus.vectore_store.base import BaseVectorSearchResult

Embedding = list[float]


class ChromaVectorStore:
    def __init__(
        self,
        client: Union[ClientAPI, AsyncClientAPI],
        collection_name: str,
        metadata: dict[str, Any] | None = None,
    ):
        self.client = client
        self.collection_name = collection_name
        self.metadata = metadata
        self._collection = None

    async def _ensure_collection(self) -> AsyncCollection | Collection:
        if self._collection is None:
            if isinstance(self.client, AsyncClientAPI):
                self._collection = await self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata=self.metadata,
                )
            else:
                self._collection = await asyncio.to_thread(
                    self.client.get_or_create_collection,
                    name=self.collection_name,
                    metadata=self.metadata,
                )
        return self._collection

    async def search(
        self, embedding: Embedding, n_results: int = 10, **kwargs
    ) -> list[BaseVectorSearchResult]:
        collection = await self._ensure_collection()
        if isinstance(collection, AsyncCollection):
            results = await collection.query(
                query_embeddings=[embedding], n_results=n_results, **kwargs
            )
        else:
            results = await asyncio.to_thread(
                collection.query,
                query_embeddings=[embedding],
                n_results=n_results,
                **kwargs,
            )

        assert results["documents"] is not None
        return [
            BaseVectorSearchResult(idx=id_, content=document)
            for id_, document in zip(results["ids"][0], results["documents"][0])
        ]

    async def insert(
        self, embedding: Embedding, content: str, id_: str | None = None, **kwargs
    ) -> str:
        if id_ is None:
            id_ = str(uuid.uuid4())
        collection = await self._ensure_collection()
        if isinstance(collection, AsyncCollection):
            await collection.add(
                embeddings=[embedding], documents=[content], ids=[id_], **kwargs
            )
        else:
            await asyncio.to_thread(
                collection.add,
                embeddings=[embedding],
                documents=[content],
                ids=[id_],
                **kwargs,
            )
        return id_
