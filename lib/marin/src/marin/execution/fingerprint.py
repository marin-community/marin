# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic config fingerprinting for lazy artifacts.

A lazy artifact's identity is its explicit ``name@version``; its *fingerprint* is a
hash of the config its recipe builds, recorded so a drift check can tell a re-run of
the same recipe (a cache hit) from a changed recipe (which wants a version bump). For
that to mean anything the serialization must be **identical across processes**: the
same config must always produce the same bytes.

The fingerprint is **advisory** — a mismatch is a warning, not a blocked build — so the
encoder does not need to be perfectly total to be correct. It canonicalizes every value
it understands (dataclasses, enums, paths, timedeltas, dtypes, sets by sorted members,
arrays by content) and, for a type it has no canonical form for, degrades to a *defined*
stable fallback with a one-time warning instead of raising. :func:`register_fingerprint`
teaches it a precise canonical form for an identity-bearing custom type, and
:func:`set_strict` (or ``MARIN_FINGERPRINT_STRICT``) makes an unknown type raise instead,
for callers that want drift to be exact.
"""

import dataclasses
import functools
import hashlib
import inspect
import json
import logging
import os
from collections.abc import Callable
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import jax
import numpy as np

logger = logging.getLogger(__name__)

# Custom canonical encoders, keyed by type, consulted before the best-effort fallback.
_REGISTRY: dict[type, Callable[[Any], object]] = {}

# Type names already warned about, so the fallback logs once per type, not per value.
_SEEN_UNFINGERPRINTABLE: set[str] = set()

# None defers to the env var; True/False is an explicit override set via set_strict.
_STRICT_OVERRIDE: bool | None = None


def register_fingerprint(tp: type, encode: Callable[[Any], object]) -> None:
    """Teach the encoder a canonical form for an identity-bearing custom type.

    ``encode(value)`` must return a JSON-canonicalizable object (it is itself encoded).
    Registered converters are consulted before the best-effort fallback, including for
    subclasses of ``tp``.
    """
    _REGISTRY[tp] = encode


def set_strict(enabled: bool) -> None:
    """Toggle strict fingerprinting: when on, an unknown type raises instead of using the
    best-effort fallback. Overrides the ``MARIN_FINGERPRINT_STRICT`` env var."""
    global _STRICT_OVERRIDE
    _STRICT_OVERRIDE = enabled


def _is_strict() -> bool:
    if _STRICT_OVERRIDE is not None:
        return _STRICT_OVERRIDE
    return os.environ.get("MARIN_FINGERPRINT_STRICT", "").lower() in ("1", "true", "yes")


def _registered_encoder(tp: type) -> Callable[[Any], object] | None:
    for klass in tp.__mro__:
        if klass in _REGISTRY:
            return _REGISTRY[klass]
    return None


def _unfingerprintable(o: object, why: str) -> str:
    t = type(o)
    return (
        f"cannot fingerprint a config value of type {t.__module__}.{t.__qualname__}: {why}. "
        "A fingerprint must be identical across processes; make the value a dataclass, an Enum, or "
        "another canonical type, register one via marin.execution.fingerprint.register_fingerprint, "
        "or disable strict mode."
    )


class _FingerprintEncoder(json.JSONEncoder):
    """Canonical JSON for config fingerprints.

    Canonical handlers run first; then registered converters; then either a raise
    (strict) or a defined best-effort fallback (advisory, the default).
    """

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

        encode = _registered_encoder(type(o))
        if encode is not None:
            return encode(o)

        if _is_strict():
            if inspect.isroutine(o) or isinstance(o, functools.partial):
                raise TypeError(_unfingerprintable(o, "a callable has no stable identity"))
            if type(o).__repr__ is object.__repr__:
                raise TypeError(_unfingerprintable(o, "it uses the default object repr (a memory address)"))
            raise TypeError(_unfingerprintable(o, "no deterministic serialization is known"))

        return _fallback(o)


def _fallback(o: object) -> dict[str, object]:
    """A defined, stable representation for a type with no canonical form.

    Identifies the type by qualified name and captures its ``vars()`` (canonicalized) when
    it has a ``__dict__``, else its ``repr``. Warns once per type so a misfire is a noisy
    advisory, not a blocked build. The object's own ``repr`` is avoided for the dict case
    because a default ``repr`` embeds a memory address (non-deterministic)."""
    t = type(o)
    name = f"{t.__module__}.{t.__qualname__}"
    if name not in _SEEN_UNFINGERPRINTABLE:
        _SEEN_UNFINGERPRINTABLE.add(name)
        logger.warning(
            "fingerprint: no canonical form for %s; using a best-effort stable fallback. "
            "Register one via marin.execution.fingerprint.register_fingerprint for an exact fingerprint.",
            name,
        )
    if hasattr(o, "__dict__"):
        return {"__repr__": name, "vars": vars(o)}
    return {"__repr__": name, "str": repr(o)}


def _canonical_key(o: object) -> str:
    return json.dumps(o, sort_keys=True, cls=_FingerprintEncoder)


def canonical_json(config: object) -> str:
    """The canonical, deterministic JSON for ``config`` — the bytes a fingerprint hashes."""
    return json.dumps(config, sort_keys=True, cls=_FingerprintEncoder)


def fingerprint_hash(payload: str) -> str:
    """The short recipe fingerprint: an md5 prefix of a canonical payload."""
    return hashlib.md5(payload.encode()).hexdigest()[:8]


# Cap on how many changed config values a drift message spells out before summarizing the
# remainder, so a wholesale recipe change stays readable.
_MAX_DIFF_LINES = 20


def _diff_json(old: object, new: object, prefix: str = "") -> list[str]:
    """Dotted-path descriptions of where ``old`` and ``new`` (parsed JSON) differ."""
    if isinstance(old, dict) and isinstance(new, dict):
        changes: list[str] = []
        for key in sorted(set(old) | set(new)):
            sub = f"{prefix}.{key}" if prefix else key
            if key not in old:
                changes.append(f"{sub}: (added) {new[key]!r}")
            elif key not in new:
                changes.append(f"{sub}: {old[key]!r} (removed)")
            else:
                changes.extend(_diff_json(old[key], new[key], sub))
        return changes
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            return [f"{prefix}: list of {len(old)} -> list of {len(new)}"]
        changes = []
        for i, (a, b) in enumerate(zip(old, new, strict=True)):
            changes.extend(_diff_json(a, b, f"{prefix}[{i}]"))
        return changes
    if old != new:
        return [f"{prefix or '(root)'}: {old!r} -> {new!r}"]
    return []


def describe_drift(old_payload: str | None, new_payload: str | None) -> str:
    """A field-level summary of how a recipe's config changed from the recorded one.

    Both arguments are canonical-JSON fingerprint payloads. Returns an empty string when either
    side is absent or the configs are identical; otherwise a capped, newline-prefixed listing of
    the changed dotted paths for a drift warning.
    """
    if old_payload is None or new_payload is None:
        return ""
    changes = _diff_json(json.loads(old_payload), json.loads(new_payload))
    if not changes:
        return ""
    shown = changes[:_MAX_DIFF_LINES]
    body = "\n".join(f"  {c}" for c in shown)
    if len(changes) > len(shown):
        body += f"\n  …and {len(changes) - len(shown)} more"
    return f"\nChanged config values:\n{body}"
