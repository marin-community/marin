# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The deterministic fingerprint encoder.

The fingerprint is only meaningful if the same config always serializes to the same
bytes, so these tests pin the canonicalization (dtypes, timedeltas, paths, sets,
arrays). The fingerprint is advisory, so an unknown type degrades to a defined stable
fallback (not a raise) unless strict mode is on; ``register_fingerprint`` teaches an
exact form.
"""

import dataclasses
from datetime import timedelta
from enum import Enum
from pathlib import Path

import jax.numpy as jnp
import marin.execution.fingerprint as fingerprint_module
import numpy as np
import pytest
from marin.execution.fingerprint import canonical_json, fingerprint_hash, register_fingerprint, set_strict


@pytest.fixture(autouse=True)
def _reset_strict():
    """Restore the module-level strict override so a strict test never leaks into others."""
    saved = fingerprint_module._STRICT_OVERRIDE
    yield
    fingerprint_module._STRICT_OVERRIDE = saved


@dataclasses.dataclass(frozen=True)
class _Cfg:
    lr: float
    dtype: type = jnp.float32
    interval: timedelta = timedelta(minutes=10)


def test_canonical_json_is_deterministic_and_field_sensitive():
    same = canonical_json(_Cfg(lr=1e-3))
    assert same == canonical_json(_Cfg(lr=1e-3))
    assert same != canonical_json(_Cfg(lr=2e-3))
    assert fingerprint_hash(same) == fingerprint_hash(canonical_json(_Cfg(lr=1e-3)))
    assert fingerprint_hash(same) != fingerprint_hash(canonical_json(_Cfg(lr=2e-3)))


def test_dtype_and_timedelta_changes_register():
    assert canonical_json(_Cfg(lr=1.0, dtype=jnp.float32)) != canonical_json(_Cfg(lr=1.0, dtype=jnp.bfloat16))
    assert "float32" in canonical_json(_Cfg(lr=1.0, dtype=jnp.float32))
    assert canonical_json(_Cfg(lr=1.0, interval=timedelta(minutes=10))) != canonical_json(
        _Cfg(lr=1.0, interval=timedelta(minutes=20))
    )


def test_path_is_serialized_by_string():
    assert canonical_json({"p": Path("a/b")}) == canonical_json({"p": "a/b"})


def test_sets_are_order_independent():
    assert canonical_json({"s": {3, 1, 2}}) == canonical_json({"s": {2, 3, 1}})
    assert canonical_json({"s": {1, 2}}) != canonical_json({"s": {1, 2, 3}})


def test_arrays_serialize_by_content():
    np_arr = np.array([1.0, 2.0], dtype=np.float32)
    jnp_arr = jnp.array([1.0, 2.0], dtype=jnp.float32)
    assert canonical_json({"a": np_arr}) == canonical_json({"a": jnp_arr})
    assert canonical_json({"a": np_arr}) != canonical_json({"a": np.array([1.0, 3.0], dtype=np.float32)})


class _Color(Enum):
    RED = "red"


def test_enum_uses_value():
    assert canonical_json({"c": _Color.RED}) == canonical_json({"c": "red"})


def test_unknown_type_falls_back_to_a_stable_form():
    """An unknown type with a ``__dict__`` fingerprints by its vars — deterministically, no raise."""

    class Widget:
        def __init__(self, n: int) -> None:
            self.n = n

    same = canonical_json({"w": Widget(1)})
    assert same == canonical_json({"w": Widget(1)})
    assert same != canonical_json({"w": Widget(2)})


def test_register_fingerprint_gives_an_exact_form():
    class Special:
        def __init__(self, token: str) -> None:
            self._opaque = object()  # not deterministically serializable on its own
            self.token = token

    register_fingerprint(Special, lambda s: {"special": s.token})
    out = canonical_json({"s": Special("xyz")})
    assert "xyz" in out
    assert canonical_json({"s": Special("a")}) != canonical_json({"s": Special("b")})


def test_strict_mode_rejects_callable_via_env(monkeypatch):
    monkeypatch.setenv("MARIN_FINGERPRINT_STRICT", "1")
    with pytest.raises(TypeError, match="callable"):
        canonical_json({"fn": lambda x: x})


def test_strict_mode_rejects_default_repr_object():
    set_strict(True)

    class Opaque:  # relies on object.__repr__ -> a memory address, not reproducible
        pass

    with pytest.raises(TypeError, match="default object repr"):
        canonical_json({"o": Opaque()})
