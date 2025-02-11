from enum import StrEnum
from typing import NamedTuple

from pydantic import BaseModel, ConfigDict, TypeAdapter


class MessageRole(StrEnum):
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    role: MessageRole
    content: str

    model_config = ConfigDict(extra="ignore")

    def prettify(self) -> str:
        return f"{self.role.value}: {self.content}"


History = TypeAdapter(list[Message])


class Sample(NamedTuple):
    x: list[Message]
    y: Message
