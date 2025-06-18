from collections import deque
from typing import Self

from promptimus.core import Module
from promptimus.dto import Message, MessageRole

from .prompt import Prompt


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

    def drop_last(self, n: int = 1):
        for _ in range(n):
            self.data.pop()

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
    def __init__(
        self,
        memory_size: int,
        system_prompt: str | None = None,
        new_message_role: MessageRole | str = MessageRole.USER,
    ):
        super().__init__()

        self.new_message_role = new_message_role
        self.prompt = Prompt(system_prompt)
        self.memory = Memory(memory_size)

    async def forward(
        self, history: list[Message] | Message | str, **kwargs
    ) -> Message:
        if isinstance(history, Message):
            history = [history]
        elif isinstance(history, str):
            history = [Message(role=self.new_message_role, content=history)]

        self.memory.extend(history)
        response = await self.prompt.forward(self.memory.as_list(), **kwargs)
        self.memory.add_message(response)

        return response

    def add_message(self, message: Message) -> Self:
        self.memory.add_message(message)
        return self

    def extend(self, history: list[Message]) -> Self:
        self.memory.extend(history)
        return self

    def drop_last(self, n: int = 1):
        self.memory.drop_last(n)
        return self

    def replace_last(self, message: Message):
        self.memory.replace_last(message)
        return self
