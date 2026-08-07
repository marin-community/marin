# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Host-snapshot save/restore round-trip on CPU."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from levanter.recovery.snapshot import HostSnapshot, estimate_state_nbytes  # noqa: E402


def _state(scale: float):
    return {
        "step": jnp.array(int(scale), dtype=jnp.int32),
        "w": (jnp.arange(6, dtype=jnp.float32).reshape(2, 3) * scale),
        "b": jnp.ones(4, dtype=jnp.bfloat16) * scale,
    }


def _template_like(state):
    return jax.tree_util.tree_map(jnp.zeros_like, state)


def test_round_trip_preserves_values(tmp_path):
    snap = HostSnapshot(str(tmp_path / "snap"))
    original = _state(2.0)
    snap.save(original, step=7)

    restored, result = snap.restore_into(_template_like(original))
    assert result.step == 7
    for key in original:
        assert jnp.array_equal(restored[key], original[key]), key


def test_latest_generation_wins(tmp_path):
    snap = HostSnapshot(str(tmp_path / "snap"))
    snap.save(_state(1.0), step=10)
    snap.save(_state(3.0), step=20)

    assert snap.latest_step() == 20
    restored, result = snap.restore_into(_template_like(_state(0.0)))
    assert result.step == 20
    assert jnp.array_equal(restored["w"], _state(3.0)["w"])


def test_restore_missing_returns_none(tmp_path):
    snap = HostSnapshot(str(tmp_path / "empty"))
    assert not snap.exists()
    assert snap.restore_into(_template_like(_state(0.0))) is None


def test_double_buffer_survives_pointer_to_torn_slot(tmp_path):
    # Two committed generations live in the two slots; deleting the newest slot's
    # data leaves the older generation recoverable via the fallback scan.
    root = str(tmp_path / "snap")
    snap = HostSnapshot(root)
    snap.save(_state(1.0), step=10)  # slot 0, gen 1
    snap.save(_state(2.0), step=20)  # slot 1, gen 2
    os.remove(os.path.join(root, "slot_1", "meta.json"))

    restored, result = snap.restore_into(_template_like(_state(0.0)))
    assert result.step == 10
    assert jnp.array_equal(restored["w"], _state(1.0)["w"])


def test_estimate_nbytes(tmp_path):
    original = _state(1.0)
    # 6 float32 (24) + 4 bfloat16 (8) + 1 int32 (4) = 36 bytes.
    assert estimate_state_nbytes(original) == 36
