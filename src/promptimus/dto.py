from enum import StrEnum
from typing import NamedTuple

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class MessageRole(StrEnum):
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolFunction(BaseModel):
    name: str
    arguments: str


class ToolRequest(BaseModel):
    id: str | None = None
    type: str | None = None
    function: ToolFunction


ToolCalls = TypeAdapter(list[ToolRequest])


class Message(BaseModel):
    role: MessageRole | str
    content: str
    tool_calls: list[ToolRequest] | None = Field(default=None)
    tool_call_id: str | None = None

    model_config = ConfigDict(extra="ignore")

    def prettify(self) -> str:
        return f"{self.role.value if isinstance(self.role, MessageRole) else self.role}: {self.content}"


History = TypeAdapter(list[Message])


class Sample(NamedTuple):
    x: list[Message]
    y: Message
