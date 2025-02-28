from enum import StrEnum
from typing import NamedTuple

from pydantic import BaseModel, ConfigDict, TypeAdapter


class MessageRole(StrEnum):
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    role: MessageRole | str
    content: str

    model_config = ConfigDict(extra="ignore")

    def prettify(self) -> str:
        return f"{self.role.value if isinstance(self.role, MessageRole) else self.role}: {self.content}"


History = TypeAdapter(list[Message])


class Sample(NamedTuple):
    x: list[Message]
    y: Message
