from typing import Any, Hashable

from promptimus.core.module import Module


class RetrievalModule(Module):
    def __init__(self, n_results: int = 10):
        super().__init__()
        self.n_results = n_results

    async def forward(self, query: str, **kwargs: Any) -> list[str]:
        query_embedding = await self.embedder.aembed(query)

        results = await self.vector_store.search(
            query_embedding, n_results=self.n_results, **kwargs
        )

        return [result.content for result in results]

    async def insert(self, documents: list[str], **kwargs: Any) -> list[Hashable]:
        """Insert multiple documents in batch for efficiency."""
        # Embed documents batch
        embeddings = await self.embedder.aembed_batch(documents)

        ids = []
        for embedding, doc in zip(embeddings, documents):
            # Insert into vector store
            id_ = await self.vector_store.insert(embedding, doc, **kwargs)
            ids.append(id_)

        return ids
