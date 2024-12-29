from typing import Protocol
from aapo.dto import Message


class AgentProtocol(Protocol):
    system_prompt: str

    async def acall(self, history: list[Message]) -> Message: ...

    async def achat(self, message: str) -> str: ...
