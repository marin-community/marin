# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The evaldash panel and comparison views: cross-cohort selection, coverage filtering, qualified
aggregates, and head-to-head difference intervals."""

import pytest
from marin.evaluation.eval_stats import Completeness, MissingPolicy
from marin.evaluation.records import (
    EvalRef,
    EvalRunRecord,
    EvalTaskRef,
    HardwareRef,
    ModelRef,
    Provenance,
    RunStatus,
    TaskCoverage,
)

from infra.evaldash.src.metrics import build_comparison, build_meta, build_panel, eval_suites, panel_request

ITEMS = 1000


def _record(
    model: str,
    eval_name: str,
    version: str | None,
    created_at: str,
    value: float | None,
    *,
    coverage: dict[str, TaskCoverage] | None = None,
    accelerator: str = "v6e-8",
) -> EvalRunRecord:
    succeeded = value is not None
    metrics = {eval_name: {"acc,none": value, "acc_stderr,none": 0.01, "sample_len": float(ITEMS)}} if succeeded else {}
    return EvalRunRecord(
        run_id=f"{model}-{eval_name}-{created_at}",
        group_id=f"{model}-{created_at}",
        created_at=created_at,
        user="tester",
        version=version,
        model=ModelRef(name=model, location="loc", backend="vllm"),
        evaluation=EvalRef(name=eval_name, mechanism="evalchemy", tasks=(EvalTaskRef(name=eval_name, num_fewshot=0),)),
        hardware=HardwareRef(platform="tpu", accelerator=accelerator, region_or_cluster="us-central2"),
        status=RunStatus.SUCCEEDED if succeeded else RunStatus.INFRA_FAILED,
        error=None,
        results_path="p",
        metrics=metrics,
        coverage=coverage or {},
        jobs={},
        log_tails={},
        provenance=Provenance(git_sha="s", eval_runtime="i", launch_host="h"),
    )


def test_panel_takes_the_latest_valid_result_for_each_benchmark_across_cohorts():
    """A newer cohort that re-ran only part of a model's benchmark set does not hide the older
    results that are still the newest available for their own benchmark."""
    records = [
        _record("m", "mmlu", "v1", "2026-01-01T00:00:00+00:00", 0.50),
        _record("m", "mmlu", "v2", "2026-02-01T00:00:00+00:00", 0.70),
        _record("m", "gsm8k-0shot", "v1", "2026-01-01T00:00:00+00:00", 0.30),
    ]

    (row,) = build_panel(records, panel_request())["rows"]

    assert row["cells"]["mmlu"]["value"] == pytest.approx(0.70)
    assert row["cells"]["mmlu"]["version"] == "v2"
    assert row["cells"]["gsm8k-0shot"]["value"] == pytest.approx(0.30)
    assert row["cells"]["gsm8k-0shot"]["version"] == "v1"


def test_panel_can_be_pinned_to_one_cohort():
    records = [
        _record("m", "mmlu", "v1", "2026-01-01T00:00:00+00:00", 0.50),
        _record("m", "mmlu", "v2", "2026-02-01T00:00:00+00:00", 0.70),
    ]

    (row,) = build_panel(records, panel_request(cohort_version="v1"))["rows"]

    assert row["cells"]["mmlu"]["value"] == pytest.approx(0.50)


def test_a_failed_run_leaves_an_explained_gap_rather_than_a_blank_cell():
    records = [
        _record("m", "mmlu", None, "2026-01-01T00:00:00+00:00", 0.5),
        _record("m", "drop", None, "2026-01-02T00:00:00+00:00", None),
    ]

    (row,) = build_panel(records, panel_request())["rows"]

    assert "drop" not in row["cells"]
    assert row["missing"]["drop"]["reason"] == "status infra_failed"


def test_cells_carry_the_interval_and_what_it_covers():
    """A run whose mechanism reports no attempted count cannot claim it graded everything, so its
    interval is labelled as covering sampling error alone."""
    records = [_record("m", "mmlu", None, "2026-01-01T00:00:00+00:00", 0.6)]

    (row,) = build_panel(records, panel_request())["rows"]

    cell = row["cells"]["mmlu"]
    assert cell["interval_kind"] == "sampling_only"
    assert cell["low"] < 0.6 < cell["high"]
    assert cell["n_scored"] == ITEMS


def test_a_partly_graded_run_reports_a_wider_identified_interval():
    """Admitting a run that graded 92% of its trials costs at least 8 points of interval width."""
    records = [
        _record(
            "m",
            "aime",
            None,
            "2026-01-01T00:00:00+00:00",
            0.6,
            coverage={"aime": TaskCoverage(n_attempted=1087, n_scored=ITEMS, errors={"AgentTimeoutError": 87})},
        )
    ]

    (row,) = build_panel(records, panel_request())["rows"]

    cell = row["cells"]["aime"]
    assert cell["interval_kind"] == "identified"
    assert cell["high"] - cell["low"] >= 1 - ITEMS / 1087
    assert cell["errors"] == {"AgentTimeoutError": 87}


def test_a_run_below_the_coverage_gate_is_rejected_with_its_rate():
    records = [
        _record(
            "m",
            "aime",
            None,
            "2026-01-01T00:00:00+00:00",
            0.6,
            coverage={"aime": TaskCoverage(n_attempted=2000, n_scored=ITEMS, errors={"AgentTimeoutError": 1000})},
        )
    ]

    (row,) = build_panel(records, panel_request())["rows"]

    assert row["cells"] == {}
    assert row["missing"]["aime"]["reason"] == "coverage 0.500 below 0.90"


def test_complete_panel_filtering_keeps_only_models_with_every_selected_benchmark():
    records = [
        _record("full", "mmlu", None, "2026-01-01T00:00:00+00:00", 0.6),
        _record("full", "drop", None, "2026-01-01T00:00:00+00:00", 0.4),
        _record("partial", "mmlu", None, "2026-01-01T00:00:00+00:00", 0.7),
    ]

    panel = build_panel(records, panel_request(benchmarks=("mmlu", "drop"), completeness=Completeness.COMPLETE_PANEL))

    assert [row["model"] for row in panel["rows"]] == ["full"]


def test_panel_filters_on_run_metadata():
    records = [
        _record("tpu-model", "mmlu", None, "2026-01-01T00:00:00+00:00", 0.6),
        _record("gpu-model", "mmlu", None, "2026-01-01T00:00:00+00:00", 0.7, accelerator="h100"),
    ]

    panel = build_panel(records, panel_request(filters={"accelerator": "h100"}))

    assert [row["model"] for row in panel["rows"]] == ["gpu-model"]


def test_no_aggregate_is_produced_unless_a_policy_is_named():
    """A mean across benchmarks has no interpretation without a declared panel and missing-data
    policy, so the panel does not offer one by default."""
    records = [_record("m", "mmlu", None, "2026-01-01T00:00:00+00:00", 0.6)]

    (row,) = build_panel(records, panel_request())["rows"]

    assert row["aggregate"] is None


def test_a_requested_aggregate_carries_its_panel_and_missing_policy():
    records = [
        _record("m", "mmlu", None, "2026-01-01T00:00:00+00:00", 0.6),
        _record("m", "drop", None, "2026-01-01T00:00:00+00:00", 0.4),
    ]

    panel = build_panel(
        records,
        panel_request(benchmarks=("mmlu", "drop")),
        aggregate_policy=MissingPolicy.REQUIRE_COMPLETE,
    )

    aggregate = panel["rows"][0]["aggregate"]
    assert aggregate["value"] == pytest.approx(0.5)
    assert aggregate["panel"] == ["mmlu", "drop"]
    assert aggregate["missing_policy"] == "require_complete"
    assert aggregate["metrics"] == ["acc,none", "acc,none"]


def test_an_incomplete_panel_has_no_aggregate_under_the_default_policy():
    records = [_record("m", "mmlu", None, "2026-01-01T00:00:00+00:00", 0.6)]

    panel = build_panel(
        records,
        panel_request(benchmarks=("mmlu", "drop")),
        aggregate_policy=MissingPolicy.REQUIRE_COMPLETE,
    )

    assert panel["rows"][0]["aggregate"] is None


def test_a_bounded_aggregate_widens_for_the_benchmark_a_model_never_ran():
    records = [_record("m", "mmlu", None, "2026-01-01T00:00:00+00:00", 0.6)]

    panel = build_panel(
        records,
        panel_request(benchmarks=("mmlu", "drop")),
        aggregate_policy=MissingPolicy.BOUND,
    )

    aggregate = panel["rows"][0]["aggregate"]
    assert aggregate["covered"] == 1
    assert aggregate["total"] == 2
    assert aggregate["high"] - aggregate["low"] >= 0.5


def test_panel_annotates_archived_models():
    records = [_record("keep", "mmlu", None, "2026-01-01T00:00:00+00:00", 0.6)]

    (row,) = build_panel(records, panel_request(), frozenset({"keep"}))["rows"]

    assert row["archived"] is True


def test_smoke_suites_stay_out_of_the_panel():
    records = [
        _record("m", "mmlu", None, "2026-01-01T00:00:00+00:00", 0.6),
        _record("m", "mmlu-smoke", None, "2026-01-02T00:00:00+00:00", 0.9),
    ]

    panel = build_panel(records, panel_request())

    assert panel["benchmarks"] == ["mmlu"]


def test_meta_reports_suites_facets_and_archived_models():
    records = [
        _record("a", "mmlu", None, "2026-01-01T00:00:00+00:00", 0.6),
        _record("b", "math500", None, "2026-01-02T00:00:00+00:00", 0.5, accelerator="h100"),
    ]

    meta = build_meta(records, frozenset({"b"}))

    assert meta["archived_models"] == ["b"]
    assert {group["suite"] for group in meta["suites"]} == {"NLP", "Chat / Math"}
    assert meta["facets"]["accelerator"] == ["h100", "v6e-8"]
    assert meta["facets"]["mechanism"] == ["evalchemy"]


def test_eval_suites_groups_known_evals_and_buckets_the_rest():
    grouped = {group["suite"]: group["evals"] for group in eval_suites({"mmlu", "drop", "math500", "mystery"})}

    assert grouped["NLP"] == ["drop", "mmlu"]
    assert grouped["Chat / Math"] == ["math500"]
    assert grouped["Other"] == ["mystery"]


def test_comparison_scores_models_on_their_shared_benchmarks_only():
    records = [
        _record("a", "mmlu", None, "2026-01-01T00:00:00+00:00", 0.60),
        _record("a", "drop", None, "2026-01-01T00:00:00+00:00", 0.40),
        _record("b", "mmlu", None, "2026-01-01T00:00:00+00:00", 0.50),
    ]

    comparison = build_comparison(records, panel_request(), ("a", "b"))

    assert comparison["shared"] == ["mmlu"]
    assert set(comparison["benchmarks"]) == {"drop", "mmlu"}
    # `a` has a result on drop that `b` never ran, so the ranking number covers mmlu alone.
    assert comparison["aggregates"]["a"]["panel"] == ["mmlu"]
    assert comparison["aggregates"]["a"]["value"] == pytest.approx(0.60)


def test_comparison_reports_a_difference_interval_against_each_benchmark_leader():
    """A ten-point gap over a thousand items each is a real ordering; the interval says so without a
    reader having to compare two error bars by eye."""
    records = [
        _record("a", "mmlu", None, "2026-01-01T00:00:00+00:00", 0.60),
        _record("b", "mmlu", None, "2026-01-01T00:00:00+00:00", 0.50),
    ]

    (row,) = build_comparison(records, panel_request(), ("a", "b"))["rows"]

    assert row["leader"] == "a"
    difference = row["differences"]["b"]
    assert difference["separated"] is True
    assert difference["low"] < 0.10 < difference["high"]
    assert "a" not in row["differences"]


def test_a_difference_that_the_runs_cannot_resolve_is_not_reported_as_separated():
    records = [
        _record("a", "mmlu", None, "2026-01-01T00:00:00+00:00", 0.601),
        _record("b", "mmlu", None, "2026-01-01T00:00:00+00:00", 0.600),
    ]

    (row,) = build_comparison(records, panel_request(), ("a", "b"))["rows"]

    assert row["differences"]["b"]["separated"] is False
    assert row["differences"]["b"]["low"] < 0.0


def test_ungraded_items_can_unsettle_an_ordering_that_sampling_error_alone_would_settle():
    """The same six-point gap over the same thousand graded items, twice: an ordering when both runs
    graded everything they attempted, and no ordering once the trailing run left 8% ungraded. The
    ungraded trials could take any value, and admitting a batch means admitting that."""
    leader = _record("a", "aime", None, "2026-01-01T00:00:00+00:00", 0.56)
    complete = _record("b", "aime", None, "2026-01-01T00:00:00+00:00", 0.50)
    partial = _record(
        "b",
        "aime",
        None,
        "2026-01-01T00:00:00+00:00",
        0.50,
        coverage={"aime": TaskCoverage(n_attempted=1087, n_scored=ITEMS, errors={"AgentTimeoutError": 87})},
    )

    (settled,) = build_comparison([leader, complete], panel_request(), ("a", "b"))["rows"]
    (unsettled,) = build_comparison([leader, partial], panel_request(), ("a", "b"))["rows"]

    assert settled["differences"]["b"]["separated"] is True
    assert unsettled["differences"]["b"]["separated"] is False
