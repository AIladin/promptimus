from pydantic import BaseModel, TypeAdapter


class ToolFunction(BaseModel):
    name: str
    arguments: str


class ToolRequest(BaseModel):
    id: str | None = None
    type: str | None = None
    function: ToolFunction


ToolCalls = TypeAdapter(list[ToolRequest])
