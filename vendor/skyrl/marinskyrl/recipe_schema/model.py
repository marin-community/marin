"""Immutable JSON containers and strict authored sections."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from enum import Enum
from types import MappingProxyType
from typing import Annotated, Any, Self, TypeVar

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    PlainValidator,
)


class FrozenMap(Mapping[str, Any]):
    """A recursively immutable, picklable mapping of JSON values."""

    __slots__ = ("_items",)
    _items: Mapping[str, Any]

    def __init__(self, items: Mapping[str, Any] | None = None) -> None:
        values = items if items is not None else {}
        if any(not isinstance(key, str) for key in values):
            raise ValueError("mapping keys must be strings")
        object.__setattr__(self, "_items", MappingProxyType({key: freeze(value) for key, value in values.items()}))

    def __setattr__(self, name: str, value: Any) -> None:
        raise TypeError("FrozenMap is immutable")

    def __getitem__(self, key: str) -> Any:
        return self._items[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __reduce__(self) -> tuple:
        return FrozenMap._from_validated, (dict(self._items),)

    @classmethod
    def _from_validated(cls, items: Mapping[str, Any]) -> Self:
        instance = object.__new__(cls)
        object.__setattr__(instance, "_items", MappingProxyType(dict(items)))
        return instance

    def __repr__(self) -> str:
        return f"FrozenMap({thaw(self)!r})"


def freeze(value: Any) -> Any:
    """Freeze JSON containers, rejecting values that cannot enter a launch document."""
    if isinstance(value, FrozenMap):
        return value
    if isinstance(value, Mapping):
        return FrozenMap(value)
    if isinstance(value, list | tuple):
        return tuple(freeze(item) for item in value)
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError(f"expected a finite JSON value, got {type(value).__name__}")


def thaw(value: Any) -> Any:
    """Return ordinary JSON containers independent of the immutable source."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {key: thaw(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [thaw(item) for item in value]
    if isinstance(value, Section):
        return value.to_skyrl()
    return value


def _frozen_map(value: Any) -> FrozenMap:
    if not isinstance(value, Mapping):
        raise ValueError("expected a mapping")
    return FrozenMap(value)


OpenMap = Annotated[FrozenMap, PlainValidator(_frozen_map), PlainSerializer(thaw)]
NumberMap = Annotated[Mapping[str, int | float], AfterValidator(FrozenMap), PlainSerializer(thaw)]


class _Unset(Enum):
    VALUE = "unset"


def unset_field(**kwargs: Any) -> Any:
    """Keep a following field sparse without resolving its default."""
    return Field(default=_Unset.VALUE, validate_default=False, exclude_if=lambda value: value is _Unset.VALUE, **kwargs)


def field(default: Any = ..., **kwargs: Any) -> Any:
    """Create a field with a recursively immutable container default."""
    if isinstance(default, Mapping | list):
        return Field(default=freeze(default), **kwargs)
    return Field(default=default, **kwargs)


class Section(BaseModel):
    """A strict schema section whose document contains only explicitly set fields."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
        allow_inf_nan=False,
    )

    def to_skyrl(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_unset=True)

    def model_copy(self, *args: Any, **kwargs: Any) -> Self:
        raise TypeError("model_copy skips validation; use merge()")


_SectionT = TypeVar("_SectionT", bound=Section)
SectionMap = Annotated[Mapping[str, _SectionT], AfterValidator(FrozenMap._from_validated), PlainSerializer(thaw)]
