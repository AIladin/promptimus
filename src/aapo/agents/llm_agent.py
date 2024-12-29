from collections import deque

from aapo.dto import Message, MessageRole
from aapo.llms.base import ProviderProtocol


class Agent:
    def __init__(
        self,
        provider: ProviderProtocol,
        system_prompt: str,
        memory_size: int = 20,
    ) -> None:
        self.provider = provider
        self.system_prompt = system_prompt

        assert memory_size > 1
        self.memory_size = memory_size
        self.memory: deque[Message] = deque()

    async def achat(self, message: str) -> str:
        self.memory.append(Message(role=MessageRole.USER, content=message))

        while len(self.memory) > self.memory_size:
            self.memory.popleft()

        model_response = await self.acall(list(self.memory))

        self.memory.append(model_response)
        return model_response.content

    async def acall(self, history: list[Message]) -> Message:
        model_response = await self.provider.achat([
            Message(role=MessageRole.SYSTEM, content=self.system_prompt),
            *history[-self.memory_size :],
        ])
        return model_response
