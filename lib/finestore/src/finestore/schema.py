# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Derive a pyarrow schema from a typed model, so a caller can pin a table's columns instead of
letting each flush infer them from that batch (which lets a column's type drift across shards).

``arrow_schema`` passes a ``pa.Schema`` through unchanged and converts a pydantic ``BaseModel`` or a
dataclass field by field. The mapping: a scalar to its arrow type; ``T | None`` to nullable ``T``;
``list[T]`` to ``list<T>``; ``dict[K, V]`` to ``map<K, V>`` (so a dynamic-keyed column such as
per-task metrics keeps one stable type); a nested model to a ``struct``; a ``str``/``int`` enum to
``string``/``int64``. Every field is nullable, so a column a later model version adds reads back null
on older shards. finestore's write path adds its ``_seq``/``_writer`` stamp columns itself, so a
caller's model describes only its own columns.
"""

from __future__ import annotations

import dataclasses
import enum
import types
from typing import Union, get_args, get_origin, get_type_hints

import pyarrow as pa
from pydantic import BaseModel

_SCALARS: dict[type, pa.DataType] = {
    str: pa.string(),
    bool: pa.bool_(),
    int: pa.int64(),
    float: pa.float64(),
    bytes: pa.binary(),
}


def arrow_schema(model: type | pa.Schema) -> pa.Schema:
    """Return a pyarrow schema for ``model``: a ``pa.Schema`` unchanged, or one derived from a
    pydantic model class or a dataclass."""
    if isinstance(model, pa.Schema):
        return model
    return pa.schema([pa.field(name, _arrow_type(annotation)) for name, annotation in _fields(model).items()])


def _fields(model: object) -> dict[str, object]:
    """The ``{name: annotation}`` of a pydantic model or a dataclass; raises for anything else."""
    if isinstance(model, type) and issubclass(model, BaseModel):
        return {name: field.annotation for name, field in model.model_fields.items()}
    if dataclasses.is_dataclass(model) and isinstance(model, type):
        hints = get_type_hints(model)
        return {field.name: hints[field.name] for field in dataclasses.fields(model)}
    raise TypeError(f"arrow_schema expects a pydantic model, a dataclass, or a pa.Schema, got {model!r}")


def _arrow_type(annotation: object) -> pa.DataType:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Union or origin is types.UnionType:
        non_none = [arg for arg in args if arg is not type(None)]
        if len(non_none) != 1:
            raise TypeError(f"unsupported union type for an arrow schema: {annotation!r}")
        return _arrow_type(non_none[0])
    if origin is list:
        (element,) = args
        return pa.list_(_arrow_type(element))
    if origin is dict:
        key, value = args
        return pa.map_(_arrow_type(key), _arrow_type(value))
    if isinstance(annotation, type):
        if issubclass(annotation, enum.Enum):
            return _enum_type(annotation)
        if issubclass(annotation, BaseModel) or dataclasses.is_dataclass(annotation):
            return pa.struct([pa.field(name, _arrow_type(ann)) for name, ann in _fields(annotation).items()])
        if annotation in _SCALARS:
            return _SCALARS[annotation]
    raise TypeError(f"unsupported type for an arrow schema: {annotation!r}")


def _enum_type(enum_cls: type) -> pa.DataType:
    """A str-valued enum stores as ``string``, an int-valued enum as ``int64``."""
    if issubclass(enum_cls, str):
        return pa.string()
    if issubclass(enum_cls, int):
        return pa.int64()
    raise TypeError(f"unsupported enum base for an arrow schema: {enum_cls!r}")
