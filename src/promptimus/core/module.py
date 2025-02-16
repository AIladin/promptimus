import os
from abc import ABC, abstractmethod
from typing import Any, Generic, Self, TypeVar, Union

from promptimus.llms.base import ProviderProtocol

from .checkpointing import module_dict_from_toml_str, module_dict_to_toml_str
from .parameters import Parameter, Prompt


class Module(ABC):
    def __init__(self):
        self._parameters: dict[str, Parameter] = {}
        self._submodules: dict[str, "Module"] = {}

    def __setattr__(self, name: str, value: Any) -> None:
        if value is self:
            return

        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._submodules[name] = value

        super().__setattr__(name, value)

    def with_provider(self, provider: ProviderProtocol) -> Self:
        for v in self._parameters.values():
            if isinstance(v, Prompt):
                v.provider = provider

        for v in self._submodules.values():
            v.with_provider(provider)

        return self

    def serialize(self) -> dict[str, Any]:
        return {
            "params": {k: v.value for k, v in self._parameters.items()},
            "submodules": {k: v.serialize() for k, v in self._submodules.items()},
        }

    def load_dict(self, checkpoint: dict[str, Any]) -> Self:
        for k, v in checkpoint["params"].items():
            self._parameters[k].value = v

        for k, v in checkpoint["submodules"].items():
            self._submodules[k].load_dict(v)

        return self

    def describe(self) -> str:
        """Returns module as TOML string"""
        module_dict = self.serialize()
        return module_dict_to_toml_str(module_dict)

    def save(self, path: str | os.PathLike):
        """Stores serialized module to a TOML file"""
        with open(path, "w") as f:
            f.write(self.describe())

    def load(self, path: str | os.PathLike) -> Self:
        """Loads TOML file and modifies inplace module object."""
        with open(path, "r") as f:
            module_dict = module_dict_from_toml_str(f.read())
            self.load_dict(module_dict)

        return self

    @abstractmethod
    async def forward(self, *_: Any, **__: Any) -> Any: ...


T = TypeVar("T", bound=Union[Parameter, Module])


class ModuleDict(Module, Generic[T]):
    """A dict wrapper to handle serialization"""

    def __init__(self, **kwargs: T):
        super().__init__()

        self.objects_map = kwargs

        for k, v in self.objects_map.items():
            setattr(self, k, v)

    async def forward(self, *_: Any, **__: Any) -> Any:
        raise NotImplementedError
