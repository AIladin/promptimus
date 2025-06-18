from datetime import datetime, timezone
from enum import StrEnum
from functools import cached_property
from hashlib import md5

from pydantic import BaseModel, Field

from promptimus.dto import Message

from . import erros


class LogStatus(StrEnum):
    OK = "OK"
    ERR = "ERR"
    RUNNING = "RUNNING"


class Span(BaseModel):
    module_name: str
    parent: "Span | None"
    start: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    stop: datetime | None = None
    request: str
    response: str | None = None
    status: LogStatus = LogStatus.RUNNING
    parameters: dict[str, str] = Field(default_factory=dict)


class Trace(BaseModel):
    prompt_name: str
    parent: Span
    start: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    stop: datetime | None = None
    prompt: str
    prompt_args: dict[str, str]
    history: list[Message]
    response: Message | None = None
    llm: str
    status: LogStatus = LogStatus.RUNNING
