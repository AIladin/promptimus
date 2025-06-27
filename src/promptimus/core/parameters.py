from hashlib import md5
from typing import Generic, TypeVar

from promptimus.errors import ParamNotSet

T = TypeVar("T")


class Parameter(Generic[T]):
    def __init__(self, value: T | None = None) -> None:
        self._name: str | None = None
        self._value: T | None = value
        self._digest: str | None = (
            md5(str(value).encode()).hexdigest() if value else None
        )

    @staticmethod
    def value_formatter(value: T) -> str:
        return str(value)

    @property
    def value(self) -> T:
        if self._value is None:
            raise ParamNotSet()

        return self._value

    @property
    def digest(self) -> str:
        if self._digest is None:
            raise ParamNotSet()

        return self._digest

    def _update_digest(self) -> None:
        digest = md5()
        if self._name is None or self._value is None:
            return

        digest.update(self._name.encode())
        digest.update(self.value_formatter(self._value).encode())

    @value.setter
    def value(self, value) -> None:
        self._value = value
        self._update_digest()
