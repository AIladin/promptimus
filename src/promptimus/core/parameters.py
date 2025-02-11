from promptimus.dto import Message, MessageRole
from promptimus.errors import ProviderNotSet
from promptimus.llms import ProviderProtocol


class Prompt:
    def __init__(
        self,
        prompt: str,
        provider: ProviderProtocol | None = None,
    ) -> None:
        self.prompt = prompt
        self.provider = provider

    async def forward(self, history: list[Message]) -> Message:
        if self.provider is None:
            raise ProviderNotSet()

        prediction = await self.provider.achat(
            [Message(role=MessageRole.SYSTEM, content=self.prompt)] + history
        )
        return prediction
