# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum, StrEnum
from pathlib import PurePath
from typing import ClassVar, Generic, TypeVar

from pydantic import BaseModel

from forge.effects.issues import LogFailure


class ReplayPolicy(StrEnum):
    CACHE = "cache"
    REEXECUTE = "reexecute"


@dataclass(frozen=True, slots=True)
class Fingerprint:
    resource: str
    digest: str
    replay: ReplayPolicy


# A phantom result parameter needs explicit covariance; PEP 695 infers invariance.
T_co = TypeVar("T_co", covariant=True)


@dataclass(frozen=True, slots=True)
class Event(Generic[T_co]):
    """An immutable request whose executor returns the declared result type.

    Covariance lets observers accept every result type as Event[object]. The
    phantom result parameter requires an explicit TypeVar to preserve this.
    """

    result_type: ClassVar[object] = type(None)

    def __init_subclass__(cls) -> None:
        EVENT_TYPES[f"{cls.__module__}.{cls.__qualname__}"] = cls


# Event registration is the sole module-state exception; logs never import classes.
EVENT_TYPES: dict[str, type[Event[object]]] = {}


def json_value(value: object) -> object:
    """JSON boundary representation; binary payloads use a reversible tagged value."""
    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, bytes):
        return {"__forge_bytes__": base64.b64encode(value).decode("ascii")}
    if isinstance(value, Enum):
        return json_value(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, BaseModel):
        return json_value(value.model_dump(mode="python"))
    if isinstance(value, Mapping):
        return {"__forge_mapping__": {key: json_value(item) for key, item in value.items()}}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise LogFailure(f"Unsupported JSON value: {type(value).__qualname__}")


def canonical_digest(value: object) -> str:
    """Hash canonical JSON content, including mappings with arbitrary key order."""
    return hashlib.sha256(
        json.dumps(json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def event_digest(event: Event[object]) -> str:
    return canonical_digest((f"{type(event).__module__}.{type(event).__qualname__}", event))


@dataclass(frozen=True, slots=True)
class Resolved[T](Event[None]):
    request: Event[T]
    fingerprints: tuple[Fingerprint, ...]
    value: T
    replayed: bool = False


@dataclass(frozen=True, slots=True)
class Note(Event[None]):
    msg: str


@dataclass(frozen=True, slots=True)
class Retrying(Event[None]):
    request: Event[object]
    attempt: int
    cause: str


@dataclass(frozen=True, slots=True)
class ScopeEntered(Event[None]):
    name: str


@dataclass(frozen=True, slots=True)
class ScopeExited(Event[None]):
    name: str
    failed: bool


@dataclass(frozen=True, slots=True)
class StageCompleted(Event[None]):
    stage: str
    config_digest: str
    exit: str
    refs: Mapping[str, str]
