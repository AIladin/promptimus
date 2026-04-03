from promptimus.core import Module, Parameter
from promptimus.dto import Message
from promptimus.vectore_store.base import TextStoreProtocol, VectorStoreProtocol

from .memory import MemoryModule
from .retrieval import RerankerProtocol, RetrievalModule

# Default constants
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_QUERY_TEMPLATE = "Context:\n{context}\n\nQuestion: {query}"


class RAGModule(Module):
    def __init__(
        self,
        vector_store: VectorStoreProtocol | None = None,
        text_store: TextStoreProtocol | None = None,
        reranker: RerankerProtocol | None = None,
        n_semantic: int = 5,
        n_text: int = 5,
        n_after_rerank: int = 5,
        memory_size: int = 10,
    ):
        super().__init__()

        self.retrieval = RetrievalModule(
            vector_store=vector_store,
            text_store=text_store,
            reranker=reranker,
            n_semantic=n_semantic,
            n_text=n_text,
            n_after_rerank=n_after_rerank,
        )
        self.memory_module = MemoryModule(
            memory_size=memory_size, system_prompt=DEFAULT_SYSTEM_PROMPT
        )

        self.query_template = Parameter(DEFAULT_QUERY_TEMPLATE)

    async def forward(self, query: str, **kwargs) -> Message:
        results = await self.retrieval.forward(query, **kwargs)
        context = "\n\n".join(r.content for r in results)

        formatted_query = self.query_template.value.format(context=context, query=query)
        response = await self.memory_module.forward(formatted_query, **kwargs)

        return response
