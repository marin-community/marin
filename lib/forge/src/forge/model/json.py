# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Immutable JSON trees held by tool schemas."""

from collections.abc import Mapping
from types import MappingProxyType

from pydantic import JsonValue

type FrozenJson = None | bool | int | float | str | tuple[FrozenJson, ...] | Mapping[str, FrozenJson]


def freeze_json(value: object) -> FrozenJson:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): freeze_json(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(freeze_json(item) for item in value)
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise TypeError(f"Not a JSON value: {type(value).__name__}")


def thaw_json(value: FrozenJson) -> JsonValue:
    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value
