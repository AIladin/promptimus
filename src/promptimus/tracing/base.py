from typing import Any

from pydantic import BaseModel, TypeAdapter

from promptimus.dto import Message


class Sample(BaseModel):
    module_path: str
    prompt_template: str
    prompt_kwargs: dict[str, Any]
    history: list[Message]
    prediction: Message


Dataset = TypeAdapter(list[Sample])
