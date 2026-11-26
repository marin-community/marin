# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the counterfactual's cohort-replacement and review-sample selection rules."""

import pytest

from experiments.post_training.mochi_loop_guards.counterfactual import bounded, input_rows, sample_rank


def row(task_id):
    return {"task_id": task_id}


MEMBERSHIP = {"tb2/a": "tb2", "tblite/b": "tblite"}


def test_input_rows_replaces_only_the_recorded_cohort_and_keeps_order():
    unknown = row("unregistered/c")
    replacement = row("tb2/a")
    # The replaced cohort's original rows are dropped; the replacement input stands in.
    rows = input_rows(
        [row("tb2/a"), unknown],
        {"tb2": [replacement]},
        MEMBERSHIP,
    )
    assert list(rows) == [unknown, replacement]


def test_input_rows_rejects_replacement_rows_outside_their_frozen_cohort():
    with pytest.raises(ValueError, match="outside its frozen cohort"):
        list(input_rows([row("tblite/b")], {"tb2": [row("tblite/b")]}, MEMBERSHIP))


def test_bounded_keeps_short_text_and_bounds_long_text_with_both_ends():
    assert bounded("short") == "short"
    long = "alpha " * 300 + "MIDDLEMARKER" + "omega " * 300
    excerpt = bounded(long, limit=2400)
    assert "[bounded excerpt]" in excerpt
    assert excerpt.startswith("alpha")
    assert excerpt.rstrip().endswith("omega")
    assert "MIDDLEMARKER" not in excerpt


def test_sample_rank_is_deterministic_and_separated_by_identity():
    identity = ["tb2", "task", 0, "uid", 12]
    again = ["tb2", "task", 0, "uid", 12]
    other = ["tb2", "task", 0, "uid", 13]
    assert sample_rank(identity) == sample_rank(again)
    assert sample_rank(identity) != sample_rank(other)
