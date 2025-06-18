from datetime import datetime, timezone
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field

from promptimus.dto import Message, MessageRole


class LogStatus(StrEnum):
    OK = "OK"
    ERR = "ERR"
    RUNNING = "RUNNING"


class Span(BaseModel):
    span_id: str = Field(default_factory=lambda: str(uuid4()))
    parent_id: str | None
    module_name: str
    module_digest: str
    start: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    stop: datetime | None = None
    request: str
    response: str | None = None
    status: LogStatus = LogStatus.RUNNING
    parameters: dict[str, str] = Field(default_factory=dict)
    error: str | None = None


class Trace(BaseModel):
    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    parent_id: str | None
    prompt_name: str
    prompt_digest: str
    start: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    stop: datetime | None = None
    prompt: str
    role: str | MessageRole
    prompt_args: dict[str, str]
    history: list[Message] | None
    response: Message | None = None
    llm: str
    status: LogStatus = LogStatus.RUNNING
    error: str | None = None
