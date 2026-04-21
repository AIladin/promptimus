import os
from abc import ABC, abstractmethod
from hashlib import md5
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from promptimus import errors
from promptimus.embedders import EmbedderProtocol
from promptimus.llms import LLMProtocol

from .checkpointing import module_dict_from_toml_str, module_dict_to_toml_str
from .parameters import Parameter

if TYPE_CHECKING:
    from promptimus.rerankers import RerankerProtocol


class Module(ABC):
    def __init__(self):
        self._name: str | None = None
        self._parent: Module | None = None
        self._parameters: dict[str, Parameter] = {}
        self._submodules: dict[str, "Module"] = {}
        self._embedder: EmbedderProtocol | None = None
        self._llm: LLMProtocol | None = None
        self._reranker: "RerankerProtocol | None" = None

    def __setattr__(self, name: str, value: Any) -> None:
        if value is self:
            return

        if isinstance(value, Parameter):
            self._parameters[name] = value
            value._parent = self
            value._name = name
        elif isinstance(value, Module) and name != "_parent":
            self._check_module_recursion(value)
            self._submodules[name] = value
            value._parent = self
            value._name = name

        super().__setattr__(name, value)

    def _check_module_recursion(self, module: "Module"):
        """Check if adding this module would create a circular dependency.

        Only checks the ancestry chain (parent -> grandparent -> ...) to detect
        circular references like A -> B -> A. Allows the same module instance
        to be used in different branches (diamond pattern).

        Args:
            module: The module being added as a submodule

        Raises:
            RecursiveModule: If the module is already an ancestor of self
        """
        # Check if the module being added is anywhere in our ancestry chain
        current = self
        while current is not None:
            if current is module:
                raise errors.RecursiveModule(
                    f"Circular dependency detected: cannot add "
                    f"{module.__class__.__name__}:{module._name} as it's already "
                    f"an ancestor of {self.__class__.__name__}:{self._name}"
                )
            current = current._parent

    @property
    def path(self) -> str:
        path = self._name or "root"
        if self._parent:
            path = self._parent.path + "." + path

        return path

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

    def with_reranker(self, reranker: "RerankerProtocol") -> Self:
        self._reranker = reranker

        for v in self._submodules.values():
            v.with_reranker(reranker)

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
    def reranker(self) -> "RerankerProtocol":
        if self._reranker is None:
            raise errors.RerankerNotSet()
        return self._reranker

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
        """Loads TOML file and modifies inplace module object.

        Supports ${pkg:...} and ${file:...} references in the TOML file
        for composing configs from external .toml files.
        """
        path = Path(path)
        with open(path, "r") as f:
            module_dict = module_dict_from_toml_str(f.read(), base_path=path.parent)
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


class ModuleDict(Module):
    """A dict wrapper to handle serialization"""

    def __init__(self, **kwargs: Parameter | Module):
        super().__init__()

        self.objects_map = {}

        for k, v in kwargs.items():
            self[k] = v

    def __setitem__(self, key: str, value: Parameter | Module):
        assert not hasattr(self, key) and key not in self.objects_map, (
            f"In module dict key `{key}` already set."
        )
        self.objects_map[key] = value
        setattr(self, key, value)

    async def forward(self, *_: Any, **__: Any) -> Any:
        raise NotImplementedError
