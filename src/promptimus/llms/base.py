from typing import Any, Protocol

from promptimus.dto import Message


class ProviderProtocol(Protocol):
    async def achat(self, history: list[Message], **kwargs: Any) -> Message: ...

    @property
    def model_name(self) -> str: ...
