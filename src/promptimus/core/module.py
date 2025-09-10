import os
from abc import ABC, abstractmethod
from hashlib import md5
from typing import Any, Generic, Self, TypeVar, Union

from promptimus import errors
from promptimus.embedders import EmbedderProtocol
from promptimus.llms import LLMProtocol
from promptimus.vectore_store.base import VectorStoreProtocol

from .checkpointing import module_dict_from_toml_str, module_dict_to_toml_str
from .parameters import Parameter


class Module(ABC):
    def __init__(self):
        self._parameters: dict[str, Parameter] = {}
        self._submodules: dict[str, "Module"] = {}
        self._embedder: EmbedderProtocol | None = None
        self._llm: LLMProtocol | None = None
        self._vector_store: VectorStoreProtocol | None = None

    def __setattr__(self, name: str, value: Any) -> None:
        if value is self:
            return

        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._check_module_recursion(value)
            self._submodules[name] = value

        super().__setattr__(name, value)

    def _check_module_recursion(self, module: "Module"):
        if self is module:
            raise errors.RecursiveModule()

        for sub in self._submodules.values():
            sub._check_module_recursion(module)

    def with_llm(self, llm: LLMProtocol) -> Self:
        self._llm = llm

        for v in self._submodules.values():
            v.with_llm(llm)

        return self

    def with_embedder(self, embedder: EmbedderProtocol) -> Self:
        self._embedder = embedder

        for v in self._submodules.values():
            v.with_embedder(embedder)

        return self

    def with_vector_store(self, vector_store: VectorStoreProtocol) -> Self:
        self._vector_store = vector_store

        for v in self._submodules.values():
            v.with_vector_store(vector_store)

        return self

    @property
    def embedder(self) -> EmbedderProtocol:
        if self._embedder is None:
            raise errors.EmbedderNotSet()
        return self._embedder

    @property
    def llm(self) -> LLMProtocol:
        if self._llm is None:
            raise errors.LLMNotSet()
        return self._llm

    @property
    def vector_store(self) -> VectorStoreProtocol:
        if self._vector_store is None:
            raise errors.VectorStoreNotSet()
        return self._vector_store

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

    def digest(self) -> str:
        digest = md5()
        for k, v in sorted(self._parameters.items()):
            digest.update(k.encode())
            digest.update(v.digest.encode())
        for k, v in sorted(self._submodules.items()):
            digest.update(k.encode())
            digest.update(v.digest().encode())

        return digest.hexdigest()

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
