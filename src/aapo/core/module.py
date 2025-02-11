from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Self

from .parameters import Prompt


class Module(ABC):
    def __init__(self):
        self._parameters: dict[str, Prompt] = {}
        self._submodules: dict[str, "Module"] = {}

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Prompt):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._submodules[name] = value

        setattr(self, name, value)

    @property
    def parameters(self) -> list[Prompt]:
        return list(self._parameters.values())

    def serialize(self) -> dict[str, Any]:
        return {
            "params": {k: v.prompt for k, v in self._parameters.items()},
            "submodules": {k: v.serialize() for k, v in self._submodules.items()},
        }

    def load_dict(self, checkpoint: dict[str, Any]) -> Self:
        for k, v in checkpoint["params"].items():
            self._parameters[k].prompt = v

        for k, v in checkpoint["submodules"].items():
            self._submodules[k].load_dict(v)

        return self

    def copy(self) -> Self:
        return deepcopy(self)

    @abstractmethod
    async def forward(self, *_: Any, **__: Any) -> Any: ...
