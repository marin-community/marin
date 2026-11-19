# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reproducibility and statistics of the manual audit sample."""

import pytest

from infra.marina.apps.rl_data_catalog.server.audit import (
    RANDOM_STRATEGY,
    STRATIFIED_STRATEGY,
    agreement_report,
    audit_order_key,
    draw_audit_sample,
    stratum_key,
    stratum_quotas,
    wilson_interval,
)


def items(count: int) -> list[dict]:
    """A verdict/defect mix with every reviewer outcome represented."""
    plan = [
        ("good", []),
        ("good", []),
        ("some_issues", ["ambiguity"]),
        ("some_issues", ["ambiguity", "malformed"]),
        ("bad", ["answer_leakage"]),
        ("bad", []),
        ("unknown", []),
        ("unknown", ["missing_context"]),
    ]
    return [
        {"task_id": f"task-{index}", "verdict": plan[index % len(plan)][0], "defects": plan[index % len(plan)][1]}
        for index in range(count)
    ]


def test_audit_order_key_is_deterministic_and_seed_sensitive():
    assert audit_order_key("seed", "task-1") == audit_order_key("seed", "task-1")
    assert audit_order_key("seed", "task-1") != audit_order_key("other", "task-1")
    assert audit_order_key("seed", "task-1") != audit_order_key("seed", "task-2")


def test_random_draw_takes_seeded_order_prefix():
    first = draw_audit_sample(items(50), strategy=RANDOM_STRATEGY, seed="seed", sample_size=7)
    assert len(first) == 7
    again = draw_audit_sample(items(50), strategy=RANDOM_STRATEGY, seed="seed", sample_size=7)
    assert [row["task_id"] for row in first] == [row["task_id"] for row in again]
    other = draw_audit_sample(items(50), strategy=RANDOM_STRATEGY, seed="other", sample_size=7)
    assert [row["task_id"] for row in first] != [row["task_id"] for row in other]


def test_stratified_draw_covers_each_stratum_and_is_reproducible():
    source = items(80)
    first = draw_audit_sample(source, strategy=STRATIFIED_STRATEGY, seed="seed", sample_size=16)
    again = draw_audit_sample(source, strategy=STRATIFIED_STRATEGY, seed="seed", sample_size=16)
    assert [row["task_id"] for row in first] == [row["task_id"] for row in again]
    strata = {row["stratum"] for row in first}
    assert len(strata) == len({stratum_key(row["verdict"], row["defects"]) for row in source})
    for row in first:
        expected = stratum_key(row["verdict"], row["defects"])
        assert row["stratum"] == expected


def test_stratified_draw_is_proportional():
    source = items(80)
    sample = draw_audit_sample(source, strategy=STRATIFIED_STRATEGY, seed="seed", sample_size=40)
    counts: dict[str, int] = {}
    for row in sample:
        counts[row["stratum"]] = counts.get(row["stratum"], 0) + 1
    # The two-item stratum ("good", no defects) is a quarter of the population.
    good_plain = stratum_key("good", [])
    assert counts[good_plain] == 10


def test_sample_size_larger_than_population_takes_everything():
    source = items(8)
    sample = draw_audit_sample(source, strategy=STRATIFIED_STRATEGY, seed="seed", sample_size=500)
    assert len(sample) == 8


def test_stratum_quotas_reserve_a_seat_when_affordable():
    # Four strata, budget of two: the two largest strata win a seat each.
    assert stratum_quotas([10, 6, 3, 1], 2) == [1, 1, 0, 0]
    # Budget covers every stratum: everyone gets at least one.
    quotas = stratum_quotas([10, 6, 3, 1], 6)
    assert all(quota >= 1 for quota in quotas)
    assert sum(quotas) == 6
    # Full coverage returns the population itself.
    assert stratum_quotas([10, 6], 100) == [10, 6]


def test_wilson_interval_bounds():
    assert wilson_interval(0, 0) == (0.0, 1.0)
    assert wilson_interval(10, 10)[1] == 1.0
    assert 0.5 < wilson_interval(10, 10)[0] < 1
    low, high = wilson_interval(8, 10)
    assert 0 < low < 0.8 < high < 1
    # The interval never leaves [0, 1] and stays inside the naive Clopper bounds.
    one_low, one_high = wilson_interval(1, 2)
    assert one_low < 0.5 < one_high


def test_agreement_report_per_verdict_and_defect():
    reviewer = {
        "a": {"verdict": "bad", "defects": ["ambiguity"]},
        "b": {"verdict": "good", "defects": []},
        "c": {"verdict": "bad", "defects": ["ambiguity"]},
    }
    judged = [
        {"task_id": "a", "agree": True},
        {"task_id": "b", "agree": True},
        {"task_id": "c", "agree": False},
        {"task_id": "absent", "agree": True},
    ]
    report = agreement_report(judged, reviewer)
    assert report["overall"] == {"judged": 3, "agree": 2, "agreement": 2 / 3, "interval": report["overall"]["interval"]}
    assert report["per_verdict"]["bad"]["judged"] == 2
    assert report["per_verdict"]["bad"]["agree"] == 1
    assert report["per_verdict"]["good"]["agreement"] == 1.0
    assert report["per_defect"]["ambiguity"]["judged"] == 2
    assert report["per_defect"]["ambiguity"]["agree"] == 1
    assert report["per_defect"]["answer_leakage"]["judged"] == 0
    assert report["per_defect"]["answer_leakage"]["agreement"] is None


@pytest.mark.parametrize("strategy", [RANDOM_STRATEGY, STRATIFIED_STRATEGY])
def test_draw_never_selects_unreviewed_or_missing_rows(strategy):
    source = items(20)
    sample = draw_audit_sample(source, strategy=strategy, seed="seed", sample_size=10)
    assert {row["task_id"] for row in sample} <= {row["task_id"] for row in source}
