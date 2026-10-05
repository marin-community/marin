"""Immutable sections and the launcher's mapping merge semantics."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping
from enum import Enum
from types import MappingProxyType
from typing import Annotated, Any, Self, TypeVar, get_args

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    FieldSerializationInfo,
    PlainSerializer,
    PlainValidator,
    SerializerFunctionWrapHandler,
    ValidationInfo,
    field_serializer,
    field_validator,
)


class FrozenMap(Mapping[str, Any]):
    """A recursively immutable, hashable mapping of JSON values."""

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

    def __hash__(self) -> int:
        return hash(document_key(self, sort_mappings=True))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Mapping) and document_key(self, sort_mappings=True) == document_key(
            other, sort_mappings=True
        )

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


def document_key(value: Any, *, sort_mappings: bool = False) -> tuple:
    """Compare JSON values with numeric equality and optionally ordered mapping keys."""
    if isinstance(value, Mapping):
        items = sorted(value.items()) if sort_mappings else value.items()
        return "mapping", tuple((key, document_key(item, sort_mappings=sort_mappings)) for key, item in items)
    if isinstance(value, tuple | list):
        return "list", tuple(document_key(item, sort_mappings=sort_mappings) for item in value)
    if isinstance(value, Section):
        return document_key(value.to_skyrl(), sort_mappings=sort_mappings)
    if isinstance(value, bool):
        return "bool", value
    if isinstance(value, int | float):
        return "number", value
    return type(value).__name__, value


def _frozen_map(value: Any) -> FrozenMap:
    if not isinstance(value, Mapping):
        raise ValueError("expected a mapping")
    return FrozenMap(value)


OpenMap = Annotated[FrozenMap, PlainValidator(_frozen_map), PlainSerializer(thaw)]
NumberMap = Annotated[Mapping[str, int | float], AfterValidator(FrozenMap), PlainSerializer(thaw)]


class _Unset(Enum):
    VALUE = "unset"


def unset_field(**kwargs: Any) -> Any:
    """Keep a field sparse without accepting an authored null or resolving its default."""
    return Field(default=_Unset.VALUE, validate_default=False, **kwargs)


def field(default: Any = ..., **kwargs: Any) -> Any:
    """Create a field with a recursively immutable container default."""
    if isinstance(default, Mapping | list):
        frozen = freeze(default)
        return Field(default_factory=lambda: frozen, **kwargs)
    return Field(default=default, **kwargs)


class Section(BaseModel):
    """A strict schema section whose document contains only explicitly set fields."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
        allow_inf_nan=False,
        arbitrary_types_allowed=True,
    )

    @field_validator("*", mode="after")
    @classmethod
    def _freeze_untyped_containers(cls, value: Any, info: ValidationInfo) -> Any:
        name = info.field_name
        assert name is not None
        if _contains_any(cls.model_fields[name].annotation):
            return freeze(value)
        return value

    @field_serializer("*", mode="wrap")
    def _serialize_untyped_containers(
        self, value: Any, handler: SerializerFunctionWrapHandler, info: FieldSerializationInfo
    ) -> Any:
        if _contains_any(type(self).model_fields[info.field_name].annotation):
            return thaw(value)
        return handler(value)

    def to_skyrl(self) -> dict[str, Any]:
        return thaw(self.model_dump(mode="python", exclude_unset=True))

    def merge(self, *patches: Section) -> Self:
        """Apply explicit overrides and validate the complete document once."""
        document = self.to_skyrl()
        for patch in patches:
            document = merge_mappings(document, patch.to_skyrl())
        return type(self).model_validate_json(json.dumps(document))

    def model_copy(self, *args: Any, **kwargs: Any) -> Self:
        raise TypeError("model_copy skips validation; use merge()")

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Section)
            and type(self) is type(other)
            and document_key(self.to_skyrl()) == document_key(other.to_skyrl())
        )

    def __hash__(self) -> int:
        return hash((type(self), document_key(self.to_skyrl())))


_SectionT = TypeVar("_SectionT", bound=Section)
SectionMap = Annotated[Mapping[str, _SectionT], AfterValidator(FrozenMap._from_validated), PlainSerializer(thaw)]


def merge_mappings(base: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    """Merge mappings recursively; lists replace and an empty mapping changes nothing."""
    merged = thaw(base)
    for key, value in patch.items():
        if isinstance(value, Mapping):
            if not value:
                continue
            current = merged.get(key)
            merged[key] = merge_mappings(current if isinstance(current, Mapping) else {}, value)
        else:
            merged[key] = thaw(value)
    return merged


def _contains_any(annotation: Any) -> bool:
    return annotation is Any or any(_contains_any(child) for child in get_args(annotation))
