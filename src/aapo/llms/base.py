from typing import Protocol

from aapo.dto import Message


class ProviderProtocol(Protocol):
    async def achat(self, history: list[Message]) -> Message: ...
