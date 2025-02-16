from typing import Generic, TypeVar

from promptimus.dto import Message, MessageRole
from promptimus.errors import ParamNotSet, ProviderNotSet
from promptimus.llms import ProviderProtocol

T = TypeVar("T")


class Parameter(Generic[T]):
    def __init__(self, value: T | None) -> None:
        self._value: T | None = value

    @property
    def value(self) -> T:
        if self._value is None:
            raise ParamNotSet()

        return self._value

    @value.setter
    def value(self, value) -> None:
        self._value = value


class Prompt(Parameter[str]):
    def __init__(
        self,
        value: str | None,
        provider: ProviderProtocol | None = None,
    ) -> None:
        super().__init__(value)
        self.provider = provider

    async def forward(self, history: list[Message], **kwargs) -> Message:
        if self.provider is None:
            raise ProviderNotSet()

        prediction = await self.provider.achat(
            [Message(role=MessageRole.SYSTEM, content=self.value.format_map(kwargs))]
            + history
        )
        return prediction
