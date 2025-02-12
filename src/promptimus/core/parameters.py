from promptimus.dto import Message, MessageRole
from promptimus.errors import PromptNotSet, ProviderNotSet
from promptimus.llms import ProviderProtocol


class Prompt:
    def __init__(
        self,
        prompt: str | None,
        provider: ProviderProtocol | None = None,
    ) -> None:
        self.prompt = prompt
        self.provider = provider

    async def forward(self, history: list[Message]) -> Message:
        if self.provider is None:
            raise ProviderNotSet()

        if self.prompt is None:
            raise PromptNotSet()

        prediction = await self.provider.achat(
            [Message(role=MessageRole.SYSTEM, content=self.prompt)] + history
        )
        return prediction
