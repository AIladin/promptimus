import os
from abc import ABC, abstractmethod
from typing import Any, Self

from promptimus.llms.base import ProviderProtocol

from .checkpointing import module_dict_from_toml_str, module_dict_to_toml_str
from .parameters import Prompt


class Module(ABC):
    def __init__(self):
        self._parameters: dict[str, Prompt] = {}
        self._submodules: dict[str, "Module"] = {}

    def __setattr__(self, name: str, value: Any) -> None:
        if value is self:
            return

        if isinstance(value, Prompt):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._submodules[name] = value

        super().__setattr__(name, value)

    def with_provider(self, provider: ProviderProtocol) -> Self:
        for v in self._parameters.values():
            v.provider = provider

        for v in self._submodules.values():
            v.with_provider(provider)

        return self

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

    def save(self, path: str | os.PathLike):
        """Stores serialized module to a TOML file"""
        with open(path, "w") as f:
            module_dict = self.serialize()
            f.write(module_dict_to_toml_str(module_dict))

    def load(self, path: str | os.PathLike) -> Self:
        """Loads TOML file and modifies inplace module object."""
        with open(path, "r") as f:
            module_dict = module_dict_from_toml_str(f.read())
            self.load_dict(module_dict)

        return self

    @abstractmethod
    async def forward(self, *_: Any, **__: Any) -> Any: ...
