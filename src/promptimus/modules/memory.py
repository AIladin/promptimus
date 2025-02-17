from collections import deque
from typing import Self

from promptimus.core import Module, Prompt
from promptimus.dto import Message, MessageRole


class Memory:
    def __init__(self, size: int):
        self.data = deque(maxlen=size)

    def add_message(self, message: Message) -> Self:
        self.data.append(message)
        return self

    def extend(self, history: list[Message]) -> Self:
        self.data.extend(history)
        return self

    def replace_last(self, message: Message):
        self.data[-1] = message

    def reset(self):
        self.data.clear()

    def __enter__(self):
        self.reset()
        return

    def __exit__(self, *args, **kwargs):
        self.reset()

    def as_list(self) -> list[Message]:
        return list(self.data)

    def __repr__(self) -> str:
        return f"Memory[{self.data}]"


class MemoryModule(Module):
    def __init__(self, memory_size: int, system_prompt: str | None = None):
        super().__init__()

        self.prompt = Prompt(system_prompt)
        self.memory = Memory(memory_size)

    async def forward(
        self, history: list[Message] | Message | str, **kwargs
    ) -> Message:
        if isinstance(history, Message):
            history = [history]
        elif isinstance(history, str):
            history = [Message(role=MessageRole.USER, content=history)]

        self.memory.extend(history)
        response = await self.prompt.forward(self.memory.as_list(), **kwargs)
        self.memory.add_message(response)

        return response
