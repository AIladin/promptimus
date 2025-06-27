import asyncio
from typing import Any

from promptimus.dto import Message, MessageRole


class DummyLLm:
    def __init__(self, message: str = "DUMMY ASSITANT", delay: float = 3):
        self.message = message
        self.delay = delay

    async def achat(self, history: list[Message], **kwargs: Any) -> Message:
        await asyncio.sleep(self.delay)
        return Message(role=MessageRole.ASSISTANT, content=self.message)

    @property
    def model_name(self) -> str:
        return "dummy"
