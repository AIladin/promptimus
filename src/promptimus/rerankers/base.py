from typing import Any, Protocol

from promptimus.vectore_store.base import BaseSearchResult


class RerankerProtocol(Protocol):
    async def forward(
        self, query: str, result_lists: list[list[BaseSearchResult]], **kwargs: Any
    ) -> list[BaseSearchResult]: ...
