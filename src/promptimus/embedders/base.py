from typing import Any, Protocol

Embedding = list[float]


class EmbedderProtocol(Protocol):
    async def aembed_batch(
        self, texts: list[str], **kwargs: Any
    ) -> list[Embedding]: ...
    async def aembed(self, text: str, **kwargs: Any) -> Embedding: ...

    @property
    def model_name(self) -> str: ...
