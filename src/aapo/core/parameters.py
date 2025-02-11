from aapo.llms import ProviderProtocol
from aapo.dto import Message, MessageRole


class Prompt:
    def __init__(
        self,
        prompt: str,
        provider: ProviderProtocol,
    ) -> None:
        self.prompt = prompt
        self.provider = provider

    async def forward(self, history: list[Message]) -> Message:
        prediction = await self.provider.achat(
            [Message(role=MessageRole.SYSTEM, content=self.prompt)] + history
        )
        return prediction
