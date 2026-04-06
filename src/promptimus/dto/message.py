from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from .image import ImageContent
from .tool import ToolRequest
from .usage import Usage


class MessageRole(StrEnum):
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    role: MessageRole | str
    content: str
    images: list[ImageContent] = Field(default_factory=list)
    tool_calls: list[ToolRequest] | None = Field(default=None)
    tool_call_id: str | None = None
    reasoning: str | None = None
    usage: Usage | None = None

    model_config = ConfigDict(extra="ignore")

    def prettify(self) -> str:
        return f"{self.role.value if isinstance(self.role, MessageRole) else self.role}: {self.content}"


History = TypeAdapter(list[Message])
