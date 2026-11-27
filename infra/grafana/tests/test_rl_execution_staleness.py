# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0


import pytest
from rl_execution_fixtures import (
    DRIVER,
    JOB_ID,
    PANELS,
    TABLE,
    query,
    seed,
    telemetry_row,
)


@pytest.fixture
def staleness_store(telemetry_table):
    """A telemetry table holding only the batches each weighted-staleness case seeds."""
    return telemetry_table


STALENESS_METRICS = {
    "staleness_min": "async/staleness_min",
    "staleness_max": "async/staleness_max",
    "loss": "async/performance/consumed_loss_tokens",
    "tokens": "async/performance/consumed_response_tokens",
    "seqs": "consumed/sequences",
    "mslr": "policy/mismatch/pooled/log_ratio_mean_squared",
    "ess": "policy/mismatch/pooled/ess_fraction",
    "reward": "reward/avg_raw_reward",
    "finite": "policy/mismatch/pooled/finite_fraction",
    "missing": "policy/mismatch/pooled/missing_behavior",
    "policy_loss": "policy/policy_loss",
}


def add_staleness_batch(database, step, *, omit=None, job=JOB_ID, execution=DRIVER, phase="train", **changes):
    values = dict(
        staleness_min=1,
        staleness_max=1,
        loss=100,
        tokens=200,
        seqs=2,
        mslr=0.1,
        ess=0.9,
        reward=0.25,
        finite=1,
        missing=0,
        policy_loss=-0.01,
    )
    values.update(changes)
    rows = [
        telemetry_row(
            "training_metric_value",
            value,
            job=job,
            execution=execution,
            seq=step,
            attributes={"step": str(step), "metric": STALENESS_METRICS[key], "payload_kind": phase},
        )
        for key, value in values.items()
        if key != omit
    ]
    seed(database, rows)


def query_staleness_panels(database):
    return (
        query(database, "Uniform-staleness batch diagnostics"),
        query(database, "Uniform-staleness diagnostic token coverage")[0],
    )


def test_weighting_mixed_duplicates_and_filters(staleness_store):
    add_staleness_batch(staleness_store, 1)
    add_staleness_batch(staleness_store, 1)  # duplicate delivery must not double count
    add_staleness_batch(staleness_store, 2, loss=300, tokens=1200, seqs=3, mslr=0.3, ess=0.8, reward=0.75)
    add_staleness_batch(
        staleness_store, 3, staleness_min=0, staleness_max=2, loss=600, mslr=900
    )  # integer mean still mixed
    add_staleness_batch(staleness_store, 4, job="other-job", loss=10000)
    add_staleness_batch(staleness_store, 4, execution="other-execution", loss=10000)
    add_staleness_batch(staleness_store, 4, phase="eval", loss=10000)
    table, coverage = query_staleness_panels(staleness_store)
    assert len(table) == 1
    row = table[0]
    assert row["staleness"] == 1 and row["updates"] == 2
    assert row["mean_response_tokens"] == 280
    assert row["token_weighted_mslr"] == pytest.approx(0.25)
    assert row["minimum_ess_fraction"] == 0.8 and row["mean_update_raw_reward"] == 0.5
    assert (
        coverage["uniform_staleness_token_fraction"] == 0.4
        and coverage["mixed_staleness_loss_tokens"] == 600
        and coverage["observed_loss_tokens"] == 1000
    )
    assert coverage["uniform_updates"] == 2 and coverage["excluded_updates"] == 1


@pytest.mark.parametrize(
    "change",
    [
        {"staleness_min": 0.5, "staleness_max": 0.5},
        {"staleness_min": -1, "staleness_max": -1},
        {"omit": "staleness_min"},
        {"omit": "mslr"},
        {"finite": 0.9},
        {"missing": 1},
        {"mslr": float("nan")},
        {"mslr": float("inf")},
        {"ess": 0},
        {"seqs": float("inf")},
        {"seqs": float("nan")},
        {"seqs": 0.5},
        {"tokens": float("inf")},
        {"tokens": float("nan")},
        {"tokens": 0.5},
    ],
)
def test_incomplete_or_invalid_batch_keeps_token_denominator(staleness_store, change):
    add_staleness_batch(staleness_store, 1)
    add_staleness_batch(staleness_store, 2, loss=300, **change)
    table, coverage = query_staleness_panels(staleness_store)
    assert table[0]["updates"] == 1
    assert coverage["uniform_staleness_token_fraction"] == 0.25 and coverage["observed_loss_tokens"] == 400


@pytest.mark.parametrize("loss", [None, -1, 0.5, float("inf"), float("nan")])
def test_invalid_loss_denominator_is_unavailable(staleness_store, loss):
    add_staleness_batch(staleness_store, 1)
    add_staleness_batch(staleness_store, 2, loss=loss)
    _, coverage = query_staleness_panels(staleness_store)
    assert coverage["uniform_staleness_token_fraction"] is None and coverage["observed_loss_tokens"] is None


def test_conflicting_diagnostic_excluded_but_conflicting_loss_invalidates_coverage(staleness_store):
    add_staleness_batch(staleness_store, 1)
    add_staleness_batch(staleness_store, 1, mslr=0.2)
    table, coverage = query_staleness_panels(staleness_store)
    assert (
        table[0]["staleness"] is None
        and coverage["uniform_staleness_token_fraction"] == 0
        and coverage["observed_loss_tokens"] == 100
    )
    add_staleness_batch(staleness_store, 1, loss=101)
    _, coverage = query_staleness_panels(staleness_store)
    assert coverage["uniform_staleness_token_fraction"] is None


def test_empty_selection_is_explicit_and_unavailable(staleness_store):
    table, coverage = query_staleness_panels(staleness_store)
    assert table[0]["status"] == "No qualifying uniform-staleness batches" and table[0]["staleness"] is None
    assert coverage["uniform_staleness_token_fraction"] is None and coverage["observed_loss_tokens"] is None


def test_evaluation_stop_panels_preserve_score_contributions_and_missing_coverage(rl_execution_store):
    length = {
        row["series"]: row["value"] for row in query(rl_execution_store, "Evaluation response length and stop coverage")
    }
    assert length == {
        "eval/all/response_tokens_mean · driver": 150,
        "eval/all/response_tokens_max · driver": 300,
        "eval/all/length_stop_fraction · driver": 0.5,
        "eval/all/completed_stop_fraction · driver": 0.5,
        "eval/all/stop_reason_coverage · driver": 1,
    }
    score = {
        row["series"]: row["value"] for row in query(rl_execution_store, "Evaluation score contributions by stop class")
    }
    assert score == {
        "eval/all/avg_score · driver": 0.25,
        "eval/all/length_stop_score_contribution · driver": 0.2,
        "eval/all/completed_stop_score_contribution · driver": 0.05,
    }
    rl_execution_store.execute(
        f"DELETE FROM {TABLE} WHERE "
        "json_get(attributes_json,'metric') LIKE 'eval/%/length_stop_%' OR "
        "json_get(attributes_json,'metric') LIKE 'eval/%/completed_stop_%'"
    )
    # Partial/legacy coverage has no fraction/contribution records: the SQL must not synthesize zeros.
    length = query(rl_execution_store, "Evaluation response length and stop coverage")
    assert len(length) == 3
    assert query(rl_execution_store, "Evaluation score contributions by stop class")[0]["value"] == 0.25
    assert len(query(rl_execution_store, "Evaluation score contributions by stop class")) == 1
    rl_execution_store.execute(f"DELETE FROM {TABLE}")
    assert query(rl_execution_store, "Evaluation response length and stop coverage") == []
    assert query(rl_execution_store, "Evaluation score contributions by stop class") == []


def test_periodic_evaluation_metrics_logged_in_train_phase_are_visible(rl_execution_store):
    # The real trainer logs the initial eval separately, then merges periodic evals into its training row.
    rl_execution_store.execute(
        f"UPDATE {TABLE} SET "
        'attributes_json=json_merge_patch(attributes_json, \'{"payload_kind":"train"}\') '
        "WHERE json_get(attributes_json,'metric') LIKE 'eval/%'"
    )
    assert len(query(rl_execution_store, "Evaluation response length and stop coverage")) == 5
    assert len(query(rl_execution_store, "Evaluation score contributions by stop class")) == 3


def test_consumed_staleness_panels_count_groups_and_sum_tokens_per_staleness(rl_execution_store):
    groups = query(rl_execution_store, "Staleness of trained groups: groups per step")
    tokens = query(rl_execution_store, "Staleness of trained groups: tokens per step")
    assert [row["value"] for row in groups if row["series"].startswith("staleness 0 ·")] == [1, 1, 1]
    assert [row["value"] for row in groups if row["series"].startswith("staleness 1 ·")] == [1, 2, 1]
    assert [row["value"] for row in tokens if row["series"].startswith("staleness 1 ·")] == [80, 70]
    assert [row["value"] for row in tokens if row["series"].startswith("staleness 0 ·")] == [10, 20]
    assert tokens[0]["t"] == tokens[1]["t"] and tokens[2]["t"] == tokens[3]["t"]
    rl_execution_store.execute(
        f"DELETE FROM {TABLE} WHERE "
        "(name='rollout_staleness_steps' AND value<>0) OR "
        "(name='consumed_staleness' AND json_get(body_json,'staleness')='1')"
    )
    for title in ("Staleness of trained groups: groups per step", "Staleness of trained groups: tokens per step"):
        assert all(row["series"].startswith("staleness 0 ·") for row in query(rl_execution_store, title))
    rl_execution_store.execute(f"DELETE FROM {TABLE} WHERE name='consumed_staleness'")
    assert query(rl_execution_store, "Staleness of trained groups: tokens per step") == []


def test_missing_staleness_buckets_remain_absent_from_mismatch(rl_execution_store):
    rl_execution_store.execute(
        f"DELETE FROM {TABLE} WHERE json_get(attributes_json,'metric') LIKE 'policy/mismatch/staleness1/%'"
    )
    assert not any(
        "/staleness1/" in row["series"]
        for row in query(rl_execution_store, "Trainer/vLLM logprob mismatch by staleness bucket")
    )


def test_correction_panels_read_weights_and_distinct_fractions(rl_execution_store):
    weights = query(rl_execution_store, "Mean correction weight")
    assert [row["value"] for row in weights] == [0.75, 1.5]
    axis = PANELS["Mean correction weight"]["fieldConfig"]["defaults"]
    assert all(axis.get("min", float("-inf")) <= row["value"] <= axis.get("max", float("inf")) for row in weights)
    rows = query(rl_execution_store, "Correction activity")
    assert [row["value"] for row in rows if "correction/truncated_fraction" in row["series"]] == [0.12, 0.24]
    assert [row["value"] for row in rows if "correction/masked_fraction" in row["series"]] == [0.05, 0.1]
    assert [row["value"] for row in rows if "policy/ppo_clip_ratio" in row["series"]] == [0.08, 0.16]
    rl_execution_store.execute(
        f"DELETE FROM {TABLE} WHERE json_get(attributes_json,'metric') LIKE 'policy/correction/%'"
    )
    assert query(rl_execution_store, "Mean correction weight") == []
    assert [row["value"] for row in query(rl_execution_store, "Correction activity")] == [0.08, 0.16]
