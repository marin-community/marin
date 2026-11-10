# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Coverage reconciliation tolerates declared sources but not inconsistent matched ones."""

from collections import Counter

import pytest

from infra.marina.apps.rl_data_catalog.load_curriculum import reconcile_population_counts


def _matched(task_count: int = 2, ready: int = 2, insufficient: int = 0) -> dict:
    return {
        "basis": "matched",
        "task_count": task_count,
        "ready_count": ready,
        "insufficient_context_count": insufficient,
    }


def _counts(denominator: int, assigned: int = 2, insufficient: int = 0) -> Counter:
    return Counter({"denominator": denominator, "assigned": assigned, "insufficient_context": insufficient})


def test_declared_sources_are_exempt_from_denominator_reconciliation():
    selected = {
        "curated": _matched(),
        "Campaign Release:pointer": {
            "basis": "declared",
            "task_count": 7,
            "ready_count": 0,
            "insufficient_context_count": 0,
        },
    }
    full_counts = {"curated": _counts(denominator=2)}
    reconcile_population_counts(selected, full_counts, task_count=2)


def test_matched_source_denominator_mismatch_fails_the_load():
    selected = {"curated": _matched(task_count=3)}
    with pytest.raises(ValueError, match="differ from release: curated"):
        reconcile_population_counts(selected, {"curated": _counts(denominator=2)}, task_count=2)


def test_statistics_row_for_a_declared_source_fails_the_total():
    selected = {
        "curated": _matched(),
        "Campaign Release:pointer": {
            "basis": "declared",
            "task_count": 7,
            "ready_count": 0,
            "insufficient_context_count": 0,
        },
    }
    full_counts = {"curated": _counts(denominator=2), "Campaign Release:pointer": _counts(denominator=7)}
    with pytest.raises(ValueError, match="differ from release"):
        reconcile_population_counts(selected, full_counts, task_count=2)
