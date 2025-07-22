import asyncio

import chromadb


class ChromaVectorStore:
    def __init__(
        self,
        client: chromadb.ClientAPI | chromadb.AsyncClientAPI,
    ):
        self.client = client

    @property
    def is_async(self):
        return isinstance(self.client, chromadb.AsyncClientAPI)

    def setup_collection(
        self,
        collection_name: str,
        metadata: dict | None = None,
        configuration: dict | None = None,
    ):
        self.client.get_or_create_collection(
            collection_name,
            metadata=metadata,
            configuration=configuration,  # type: ignore
        )
