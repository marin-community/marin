# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for `prepare.py` — pure-Python parts; no HF download."""

from __future__ import annotations

import collections

import pytest

from experiments.ifbench.data.prepare import (
    PreparedRow,
    _stable_prompt_id,
    assert_no_test_contamination,
    stratified_val_split,
)


def _row(num_constraints: int, content: str = "hello") -> PreparedRow:
    msgs = [{"role": "user", "content": content}]
    return PreparedRow(
        prompt_id=_stable_prompt_id(msgs),
        messages=msgs,
        constraint=f"do {num_constraints} things",
        constraint_type="multi",
        ground_truth="[]",
        num_constraints=num_constraints,
    )


def test_stratified_val_split_sizes() -> None:
    rows = [_row((i % 5) + 1, content=f"prompt {i}") for i in range(1000)]
    train, val = stratified_val_split(rows, val_size=100, seed=0)
    assert len(train) + len(val) == 1000
    assert len(val) == 100


def test_stratified_val_split_proportional() -> None:
    """Each num_constraints stratum's val share is approximately proportional."""
    rows: list[PreparedRow] = []
    for n_constr, n_rows in [(1, 600), (2, 300), (3, 100)]:
        rows.extend(_row(n_constr, content=f"k={n_constr}-{i}") for i in range(n_rows))
    _, val = stratified_val_split(rows, val_size=100, seed=0)
    val_counts = collections.Counter(r.num_constraints for r in val)
    # Allow ±2 for rounding remainder distribution.
    assert abs(val_counts[1] - 60) <= 2
    assert abs(val_counts[2] - 30) <= 2
    assert abs(val_counts[3] - 10) <= 2


def test_stratified_val_split_deterministic() -> None:
    rows = [_row((i % 5) + 1, content=f"prompt {i}") for i in range(500)]
    t1, v1 = stratified_val_split(rows, val_size=50, seed=42)
    t2, v2 = stratified_val_split(rows, val_size=50, seed=42)
    assert [r.prompt_id for r in v1] == [r.prompt_id for r in v2]
    assert [r.prompt_id for r in t1] == [r.prompt_id for r in t2]


def test_stratified_val_split_no_overlap() -> None:
    rows = [_row((i % 5) + 1, content=f"prompt {i}") for i in range(500)]
    train, val = stratified_val_split(rows, val_size=50, seed=0)
    train_ids = {r.prompt_id for r in train}
    val_ids = {r.prompt_id for r in val}
    assert train_ids.isdisjoint(val_ids)


def test_stratified_val_split_rejects_oversize() -> None:
    rows = [_row(1) for _ in range(10)]
    with pytest.raises(ValueError):
        stratified_val_split(rows, val_size=15)
    with pytest.raises(ValueError):
        stratified_val_split(rows, val_size=0)


def test_contamination_check_passes_when_disjoint(monkeypatch) -> None:
    monkeypatch.setattr(
        "experiments.ifbench.data.prepare.load_test_prompts",
        lambda *a, **kw: ["different test prompt one", "different test prompt two"],
    )
    rows = [_row(1, content="train prompt one"), _row(2, content="train prompt two")]
    assert_no_test_contamination(rows)  # should not raise


def test_contamination_check_raises_on_overlap(monkeypatch) -> None:
    monkeypatch.setattr(
        "experiments.ifbench.data.prepare.load_test_prompts",
        lambda *a, **kw: ["leaked prompt", "another test prompt"],
    )
    rows = [_row(1, content="leaked prompt"), _row(2, content="ok prompt")]
    with pytest.raises(RuntimeError, match="Contamination detected"):
        assert_no_test_contamination(rows)


def test_contamination_check_catches_rstrip_overlap(monkeypatch) -> None:
    """An IFBench_test prompt with no trailing whitespace should match a
    train prompt with trailing whitespace (and vice versa). We saw real
    upstream data with this exact issue when validating the verifier port."""
    monkeypatch.setattr(
        "experiments.ifbench.data.prepare.load_test_prompts",
        lambda *a, **kw: ["leaky prompt"],
    )
    rows = [_row(1, content="leaky prompt   ")]  # train has trailing whitespace
    with pytest.raises(RuntimeError):
        assert_no_test_contamination(rows)
