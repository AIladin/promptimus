import base64
from io import BytesIO
from typing import Self

from pydantic import BaseModel, Field


class ImageContent(BaseModel):
    url: str = Field(repr=False)

    @classmethod
    def from_buffer(cls, buffer: BytesIO, mimetype: str) -> Self:
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode("utf-8")
        buffer.seek(0)
        return cls(url=f"data:{mimetype};base64,{b64}")
