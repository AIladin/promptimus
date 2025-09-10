from promptimus.core import Module, Parameter
from promptimus.dto import Message

from .memory import MemoryModule
from .retrieval import RetrievalModule

# Default constants
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_QUERY_TEMPLATE = "Context:\n{context}\n\nQuestion: {query}"


class RAGModule(Module):
    def __init__(
        self,
        n_results: int = 5,
        memory_size: int = 10,
    ):
        super().__init__()

        # Aggregate existing modules
        self.retrieval = RetrievalModule(n_results=n_results)
        self.memory_module = MemoryModule(
            memory_size=memory_size, system_prompt=DEFAULT_SYSTEM_PROMPT
        )

        # Configurable query template parameter
        self.query_template = Parameter(DEFAULT_QUERY_TEMPLATE)

    async def forward(self, query: str, **kwargs) -> Message:
        # Retrieve relevant documents
        retrieved_docs = await self.retrieval.forward(query, **kwargs)
        context = "\n\n".join(retrieved_docs)

        # Format query using configurable template
        formatted_query = self.query_template.value.format(context=context, query=query)

        # Pass to MemoryModule for conversation handling
        response = await self.memory_module.forward(formatted_query, **kwargs)

        return response
