# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for giant-row batch sizing in the production drivers.

The sizing decision must be a pure function of the store state so that a relaunched
pod (including recovery re-runs) computes identical batch boundaries, and it must
never change claim/cursor/receipt semantics — only the claimed range length.
"""

import numpy as np

from experiments.glm_tool_serving.ortet_production import giant_row_batch_size


class FakeStore:
    def __init__(self, lengths):
        self._lengths = lengths

    def get_batch_sync(self, rows):
        return [np.zeros(self._lengths[r], dtype="<i4") for r in rows]


class FakeTreeStore:
    def __init__(self, lengths):
        self.tree = {"input_ids": FakeStore(lengths)}


def fake_peek(monkeypatch, lengths):
    opened = {}

    def fake_open(template, shard, mode="r", cache_metadata=True):
        opened["shard"] = shard
        return FakeTreeStore(lengths)

    monkeypatch.setattr(
        "levanter.store.tree_store.TreeStore.open",
        staticmethod(fake_open),
    )
    return opened


def test_giant_rows_shrink_batch_to_cap(monkeypatch):
    # 128 rows of 50k tokens: any giant row -> cap (32)
    lengths = {r: 50_000 for r in range(128)}
    fake_peek(monkeypatch, lengths)
    assert giant_row_batch_size("s3://store/x", 0, 10_000) == 32


def test_small_rows_keep_default_batch(monkeypatch):
    lengths = {r: 5_000 for r in range(128)}
    fake_peek(monkeypatch, lengths)
    assert giant_row_batch_size("s3://store/x", 0, 10_000) == 128


def test_single_giant_row_shrinks_batch(monkeypatch):
    lengths = {r: 5_000 for r in range(128)}
    lengths[3] = 41_000
    fake_peek(monkeypatch, lengths)
    assert giant_row_batch_size("s3://store/x", 0, 10_000) == 32


def test_threshold_boundary_is_inclusive(monkeypatch):
    lengths = {r: 39_999 for r in range(128)}
    fake_peek(monkeypatch, lengths)
    assert giant_row_batch_size("s3://store/x", 0, 10_000) == 128
    lengths = {r: 40_000 for r in range(128)}
    fake_peek(monkeypatch, lengths)
    assert giant_row_batch_size("s3://store/x", 0, 10_000) == 32


def test_shard_tail_clamps_peek(monkeypatch):
    # fewer rows left than the default window: peek must not index past the shard
    lengths = {r: 50_000 for r in range(10)}
    fake_peek(monkeypatch, lengths)
    assert giant_row_batch_size("s3://store/x", 0, 10) == 32


def test_peek_failure_falls_back_to_default(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("store unavailable")

    monkeypatch.setattr("levanter.store.tree_store.TreeStore.open", staticmethod(boom))
    assert giant_row_batch_size("s3://store/x", 0, 10_000) == 128


def test_sizing_is_deterministic_for_recovery(monkeypatch):
    # a relaunched pod must compute the same boundary for the same store state
    lengths = {r: 60_000 if r % 3 == 0 else 45_000 for r in range(128)}
    fake_peek(monkeypatch, lengths)
    first = giant_row_batch_size("s3://store/x", 0, 10_000)
    second = giant_row_batch_size("s3://store/x", 0, 10_000)
    assert first == second == 32
