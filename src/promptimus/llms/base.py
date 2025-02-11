from typing import Protocol

from promptimus.dto import Message


class ProviderProtocol(Protocol):
    async def achat(self, history: list[Message]) -> Message: ...
