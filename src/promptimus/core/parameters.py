from typing import Generic, TypeVar

from promptimus.dto import Message, MessageRole
from promptimus.errors import PromptNotSet, ProviderNotSet
from promptimus.llms import ProviderProtocol

T = TypeVar("T")


class Parameter(Generic[T]):
    def __init__(self, value: T) -> None:
        self.value: T = value


class Prompt(Parameter[str | None]):
    def __init__(
        self,
        value: str | None,
        provider: ProviderProtocol | None = None,
    ) -> None:
        super().__init__(value)
        self.provider = provider

    async def forward(self, history: list[Message]) -> Message:
        if self.provider is None:
            raise ProviderNotSet()

        if self.value is None:
            raise PromptNotSet()

        prediction = await self.provider.achat(
            [Message(role=MessageRole.SYSTEM, content=self.value)] + history
        )
        return prediction
