# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic config fingerprinting for lazy artifacts.

A lazy artifact's identity is its explicit ``name@version``; its *fingerprint* is a
hash of the config its recipe builds, recorded so the build-once guard can tell a
re-run of the same recipe (a cache hit) from a changed recipe (which needs a version
bump). For that to mean anything, the serialization must be **identical across
processes**: the same config must always produce the same bytes.

So this encoder canonicalizes every value it understands — dataclasses, enums,
paths, timedeltas, dtypes, sets (by sorted members), arrays (by content) — and
*raises* on values it cannot serialize deterministically, such as a callable or an
object that falls back to its default memory-address ``repr``. A permissive
``str(o)`` fallback would let such a value silently fork the fingerprint from one
run to the next, so it is rejected loudly at fingerprint time instead.
"""

import dataclasses
import functools
import hashlib
import inspect
import json
from datetime import timedelta
from enum import Enum
from pathlib import Path

import jax
import numpy as np


def _unfingerprintable(o: object, why: str) -> str:
    t = type(o)
    return (
        f"cannot fingerprint a config value of type {t.__module__}.{t.__qualname__}: {why}. "
        "A fingerprint must be identical across processes; make the value a dataclass, an Enum, or "
        "another canonical type, or add a handler in marin.execution.fingerprint."
    )


class _FingerprintEncoder(json.JSONEncoder):
    """Canonical JSON for config fingerprints; raises on non-deterministic values."""

    def default(self, o):
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            return dataclasses.asdict(o)
        if isinstance(o, Enum):
            return o.value
        if isinstance(o, timedelta):
            return {"days": o.days, "seconds": o.seconds, "microseconds": o.microseconds}
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, set | frozenset):
            # A set's identity is its members, not their iteration order.
            return {"__set__": sorted(o, key=_canonical_key)}
        if isinstance(o, np.dtype):
            return {"__dtype__": o.name}
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray | jax.Array):
            arr = np.asarray(o)
            return {"__array__": arr.tolist(), "dtype": arr.dtype.name, "shape": list(arr.shape)}
        if isinstance(o, type):
            # dtype *type* objects (np.float32, jnp.bfloat16) canonicalize to their name;
            # any other class is identified by its fully-qualified name.
            try:
                return {"__dtype__": np.dtype(o).name}
            except TypeError:
                return {"__type__": f"{o.__module__}.{o.__qualname__}"}
        if inspect.isroutine(o) or isinstance(o, functools.partial):
            raise TypeError(_unfingerprintable(o, "a callable has no stable identity"))
        if type(o).__repr__ is object.__repr__:
            raise TypeError(_unfingerprintable(o, "it uses the default object repr (a memory address)"))
        raise TypeError(_unfingerprintable(o, "no deterministic serialization is known"))


def _canonical_key(o: object) -> str:
    return json.dumps(o, sort_keys=True, cls=_FingerprintEncoder)


def canonical_json(config: object) -> str:
    """The canonical, deterministic JSON for ``config`` — the bytes a fingerprint hashes."""
    return json.dumps(config, sort_keys=True, cls=_FingerprintEncoder)


def fingerprint_hash(payload: str) -> str:
    """The short recipe fingerprint: an md5 prefix of a canonical payload."""
    return hashlib.md5(payload.encode()).hexdigest()[:8]
