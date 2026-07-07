# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LazyFillGuard — the stale-set defence for lazy projections.

The guard decides whether a value recomputed from a read snapshot may be cached,
by comparing the snapshot's sampled commit seq against each key's most recent
invalidation seq (and a global floor). These tests pin the contract that closes
the stale-set race: a fill from a snapshot older than the key's invalidation is
refused, so a slow reader can never write back a value the invalidation superseded.
"""

from iris.cluster.controller.projections.base import LazyFillGuard


def test_unseen_key_is_cacheable_from_any_snapshot():
    guard: LazyFillGuard[str] = LazyFillGuard()
    assert guard.may_store(0, "j")
    assert guard.may_store(100, "j")


def test_fill_from_pre_invalidation_snapshot_is_refused():
    """The stale set: reader snapshot at seq 4, invalidation commits at seq 5."""
    guard: LazyFillGuard[str] = LazyFillGuard()
    guard.note_invalidated(5, ["j"])
    assert not guard.may_store(4, "j")  # snapshot predates the invalidation → refuse
    assert guard.may_store(5, "j")  # snapshot includes the invalidating commit → ok
    assert guard.may_store(6, "j")


def test_invalidation_is_per_key():
    guard: LazyFillGuard[str] = LazyFillGuard()
    guard.note_invalidated(5, ["j"])
    # A different key is unaffected — a busy neighbour's writes don't starve this fill.
    assert guard.may_store(4, "other")
    assert not guard.may_store(4, "j")


def test_reset_floors_all_keys():
    """clear() (eviction / checkpoint reopen) advances a floor every key is gated by."""
    guard: LazyFillGuard[str] = LazyFillGuard()
    guard.note_invalidated(3, ["j"])
    guard.reset(9)
    assert not guard.may_store(8, "j")  # below floor
    assert not guard.may_store(8, "fresh")  # even a never-seen key is floored
    assert guard.may_store(9, "j")
    assert guard.may_store(9, "fresh")


def test_note_invalidated_self_bounds_into_floor(monkeypatch):
    """Past the tracked-key cap, invalidations fold into the floor instead of growing."""
    guard: LazyFillGuard[int] = LazyFillGuard()
    monkeypatch.setattr(type(guard), "_MAX_TRACKED", 3)
    guard.note_invalidated(10, [0, 1, 2])  # fills the per-key map to the cap
    guard.note_invalidated(11, [3])  # overflow → reset(11) then track key 3
    assert guard._floor == 11
    assert not guard.may_store(10, 0)  # folded into the floor
    assert not guard.may_store(10, 3)
    assert guard.may_store(11, 3)
