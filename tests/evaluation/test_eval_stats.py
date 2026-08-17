# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The eval statistics engine: intervals under attrition, panel aggregates, and result selection."""

import math

import pytest
from marin.evaluation.eval_measurements import measurement_from_record
from marin.evaluation.eval_stats import (
    AggregationProtocol,
    CohortMode,
    Completeness,
    Coverage,
    IntervalKind,
    Measurement,
    MetricKind,
    MissingPolicy,
    ResultFlag,
    SelectionRequest,
    difference_interval,
    imbens_manski_critical,
    measurement_interval,
    normal_cdf,
    panel_aggregate,
    select,
    wilson_interval,
)
from marin.evaluation.records import (
    EvalchemyRef,
    EvalRef,
    EvalRunRecord,
    EvalTaskRef,
    HarborRef,
    HardwareRef,
    ModelRef,
    Provenance,
    RunStatus,
    TaskCoverage,
)


def _measurement(
    *,
    benchmark: str = "gsm8k",
    n_correct: int = 60,
    n_scored: int = 100,
    n_attempted: int | None = 100,
    model: str = "m",
    created_at: str = "2026-01-01T00:00:00+00:00",
    version: str | None = None,
    status: RunStatus = RunStatus.SUCCEEDED,
    run_id: str = "run",
    flags: frozenset[ResultFlag] = frozenset(),
) -> Measurement:
    return Measurement(
        benchmark=benchmark,
        metric="exact_match,flexible-extract",
        kind=MetricKind.BINARY,
        value=n_correct / n_scored,
        coverage=Coverage(n_scored=n_scored, n_attempted=n_attempted),
        n_correct=n_correct,
        run_id=run_id,
        created_at=created_at,
        version=version,
        model=model,
        status=status,
        flags=flags,
    )


# --------------------------------------------------------------------------------------------------
# Intervals
# --------------------------------------------------------------------------------------------------


def test_wilson_interval_is_not_degenerate_when_nothing_was_solved():
    """A model that solves none of 8 agentic trials is not known to score exactly 0.

    The normal approximation gives zero width here, which is why the engine does not use it.
    """
    interval = wilson_interval(0, 8)

    assert interval.low == 0.0
    assert interval.high == pytest.approx(0.324, abs=0.001)


def test_wilson_interval_is_not_degenerate_when_everything_was_solved():
    interval = wilson_interval(8, 8)

    assert interval.low == pytest.approx(0.676, abs=0.001)
    assert interval.high == 1.0


def test_wilson_interval_matches_the_textbook_value():
    """A hand-checkable case: 60 of 100 at the 95% two-sided quantile."""
    interval = wilson_interval(60, 100)

    assert interval.low == pytest.approx(0.5020, abs=0.0005)
    assert interval.high == pytest.approx(0.6906, abs=0.0005)


def test_imbens_manski_critical_value_spans_the_two_limits():
    """The critical value is the two-sided quantile with nothing missing and falls toward the
    one-sided quantile as the unidentified region dominates the sampling error."""
    assert imbens_manski_critical(0.0, 0.05) == pytest.approx(1.96, abs=0.001)
    assert imbens_manski_critical(1.0, 1e-6) == pytest.approx(1.645, abs=0.001)
    assert 1.645 < imbens_manski_critical(0.05, 0.03) < 1.96


def test_imbens_manski_critical_value_solves_its_defining_equation():
    delta, sigma = 0.077, 0.032
    critical = imbens_manski_critical(delta, sigma)

    coverage = normal_cdf(critical + delta / sigma) - normal_cdf(-critical)

    assert coverage == pytest.approx(0.95, abs=1e-6)


def test_partial_coverage_widens_the_interval_by_at_least_the_missing_fraction():
    """Admitting a run that graded 92.5% of its trials costs at least 7.5 points of interval width,
    however many trials it ran -- the ungraded trials could have gone either way."""
    partial = _measurement(n_correct=120, n_scored=222, n_attempted=240)

    interval = measurement_interval(partial)

    assert interval.kind is IntervalKind.IDENTIFIED
    assert interval.width >= 1 - 222 / 240
    assert interval.low < 120 / 240 < interval.high


def test_complete_coverage_reduces_to_the_wilson_interval():
    complete = _measurement(n_correct=60, n_scored=100, n_attempted=100)

    interval = measurement_interval(complete)
    wilson = wilson_interval(60, 100)

    assert interval.kind is IntervalKind.IDENTIFIED
    assert (interval.low, interval.high) == pytest.approx((wilson.low, wilson.high))


def test_unreported_coverage_is_labelled_rather_than_assumed_complete():
    """A record that does not say how many items were attempted cannot claim it graded them all."""
    unknown = _measurement(n_correct=60, n_scored=100, n_attempted=None)

    interval = measurement_interval(unknown)

    assert interval.kind is IntervalKind.SAMPLING_ONLY
    assert interval.width == pytest.approx(wilson_interval(60, 100).width)


def test_ranking_uses_the_lower_bound_so_attrition_cannot_buy_rank():
    """Two runs with the same complete-case rate rank by how much of the panel they actually graded.

    Ranking on ``k / n_scored`` would tie them, which rewards losing the hardest items.
    """
    complete = _measurement(n_correct=60, n_scored=100, n_attempted=100)
    partial = _measurement(n_correct=60, n_scored=100, n_attempted=125)

    assert complete.value == partial.value
    assert measurement_interval(complete).low > measurement_interval(partial).low


def test_continuous_metric_uses_the_recorded_standard_error():
    measurement = Measurement(
        benchmark="tb2",
        metric="mean_reward",
        kind=MetricKind.CONTINUOUS,
        value=0.4,
        coverage=Coverage(n_scored=50, n_attempted=50),
        recorded_stderr=0.05,
    )

    interval = measurement_interval(measurement)

    assert interval.low == pytest.approx(0.4 - 1.96 * 0.05, abs=1e-3)
    assert interval.high == pytest.approx(0.4 + 1.96 * 0.05, abs=1e-3)


# --------------------------------------------------------------------------------------------------
# Differences
# --------------------------------------------------------------------------------------------------


def test_difference_interval_separates_two_complete_runs():
    better = _measurement(n_correct=132, n_scored=240, n_attempted=240, model="a")
    worse = _measurement(n_correct=108, n_scored=240, n_attempted=240, model="b")

    difference = difference_interval(better, worse)

    assert difference.low > 0.0


def test_attrition_at_the_gate_makes_the_same_ordering_unidentified():
    """The ordering above survives only while both runs are near-complete. At the 90% gate the
    unidentified width alone is 0.2, so a 10-point gap cannot be resolved."""
    better = _measurement(n_correct=122, n_scored=221, n_attempted=240, model="a")
    worse = _measurement(n_correct=99, n_scored=221, n_attempted=240, model="b")

    difference = difference_interval(better, worse)

    assert difference.low < 0.0 < difference.high


def test_difference_widening_is_asymmetric_in_which_run_lost_items():
    """The opposing run's ungraded items drive your lower bound, so losing items on one side is not
    the same as losing them on the other."""
    complete_a = _measurement(n_correct=120, n_scored=200, n_attempted=200, model="a")
    complete_b = _measurement(n_correct=100, n_scored=200, n_attempted=200, model="b")
    partial_a = _measurement(n_correct=120, n_scored=200, n_attempted=220, model="a")
    partial_b = _measurement(n_correct=100, n_scored=200, n_attempted=220, model="b")

    a_lost = difference_interval(partial_a, complete_b)
    b_lost = difference_interval(complete_a, partial_b)

    assert a_lost.high > difference_interval(complete_a, complete_b).high
    assert b_lost.low < difference_interval(complete_a, complete_b).low
    assert (a_lost.low, a_lost.high) != (b_lost.low, b_lost.high)


# --------------------------------------------------------------------------------------------------
# Panel aggregation
# --------------------------------------------------------------------------------------------------


def test_aggregate_requires_a_complete_panel_by_default():
    """A mean over whichever benchmarks a model happens to have run has no definition, so the
    default protocol declines to produce one."""
    cells = {"gsm8k": _measurement(benchmark="gsm8k")}
    protocol = AggregationProtocol(panel=("gsm8k", "mmlu"))

    assert panel_aggregate(cells, protocol) is None


def test_bounded_aggregate_widens_by_one_panel_share_per_missing_benchmark():
    cells = {"gsm8k": _measurement(benchmark="gsm8k", n_correct=100, n_scored=100, n_attempted=100)}
    protocol = AggregationProtocol(panel=("gsm8k", "mmlu"), missing=MissingPolicy.BOUND)

    aggregate = panel_aggregate(cells, protocol)

    assert aggregate is not None
    assert aggregate.covered == 1
    assert aggregate.high - aggregate.low >= 0.5


def test_aggregate_over_a_complete_panel_combines_sampling_error_across_benchmarks():
    cells = {
        "gsm8k": _measurement(benchmark="gsm8k", n_correct=60, n_scored=100, n_attempted=100),
        "mmlu": _measurement(benchmark="mmlu", n_correct=40, n_scored=100, n_attempted=100),
    }
    protocol = AggregationProtocol(panel=("gsm8k", "mmlu"))

    aggregate = panel_aggregate(cells, protocol)

    assert aggregate is not None
    assert aggregate.value == pytest.approx(0.5)
    assert aggregate.covered == 2
    # Independent item sets: the mean's standard error is sqrt(sum(w^2 se^2)), so the aggregate is
    # tighter than either cell's own interval.
    single = measurement_interval(cells["gsm8k"])
    assert aggregate.high - aggregate.low < single.width


def test_aggregate_of_unreported_coverage_is_labelled_sampling_only():
    cells = {
        "gsm8k": _measurement(benchmark="gsm8k", n_attempted=None),
        "mmlu": _measurement(benchmark="mmlu", n_attempted=None),
    }

    aggregate = panel_aggregate(cells, AggregationProtocol(panel=("gsm8k", "mmlu")))

    assert aggregate is not None
    assert aggregate.kind is IntervalKind.SAMPLING_ONLY


def test_aggregate_carries_the_protocol_that_defines_it():
    """No surface can render an aggregate without the panel and policy behind it."""
    cells = {"gsm8k": _measurement(benchmark="gsm8k")}
    protocol = AggregationProtocol(panel=("gsm8k",))

    aggregate = panel_aggregate(cells, protocol)

    assert aggregate is not None
    assert aggregate.protocol == protocol
    assert aggregate.metrics == ("exact_match,flexible-extract",)


# --------------------------------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------------------------------


def test_selection_takes_the_newest_result_per_benchmark_across_cohorts():
    """A newer cohort that re-ran only part of a model's benchmark set does not hide the older
    results that are still the newest available for their own benchmark."""
    measurements = [
        _measurement(benchmark="mmlu", version="v1", created_at="2026-01-01T00:00:00+00:00", n_correct=50),
        _measurement(benchmark="mmlu", version="v2", created_at="2026-02-01T00:00:00+00:00", n_correct=70),
        _measurement(benchmark="gsm8k", version="v1", created_at="2026-01-01T00:00:00+00:00", n_correct=30),
    ]

    selection = select(measurements, SelectionRequest())

    assert selection.cells["m"]["mmlu"].value == pytest.approx(0.70)
    assert selection.cells["m"]["gsm8k"].value == pytest.approx(0.30)


def test_single_cohort_selection_keeps_only_that_cohort():
    measurements = [
        _measurement(benchmark="mmlu", version="v1", created_at="2026-01-01T00:00:00+00:00"),
        _measurement(benchmark="gsm8k", version="v2", created_at="2026-02-01T00:00:00+00:00"),
    ]

    selection = select(
        measurements,
        SelectionRequest(cohort=CohortMode.SINGLE_COHORT, cohort_version="v2"),
    )

    assert set(selection.cells["m"]) == {"gsm8k"}
    assert [rejection.reason for rejection in selection.rejections] == ["cohort v1"]


def test_selection_rejects_below_gate_coverage_and_says_why():
    measurements = [_measurement(n_scored=80, n_attempted=100, run_id="thin")]

    selection = select(measurements, SelectionRequest())

    assert selection.cells == {}
    (rejection,) = selection.rejections
    assert rejection.run_id == "thin"
    assert "coverage 0.800" in rejection.reason


def test_selection_rejects_failed_runs_but_keeps_them_explainable():
    measurements = [_measurement(status=RunStatus.INFRA_FAILED, run_id="broken")]

    selection = select(measurements, SelectionRequest())

    assert selection.cells == {}
    assert selection.rejections[0].reason == "status infra_failed"


def test_a_flagged_result_does_not_replace_the_valid_one_it_supersedes():
    """A newer run whose grader extracted nothing scored zero on the strength of nothing. Letting it
    stand as the model's current result would report a collapse the evidence does not support, so the
    last result that measured something holds and the flagged run is reported as the reason."""
    measurements = [
        _measurement(n_correct=60, created_at="2026-01-01T00:00:00+00:00", run_id="good"),
        _measurement(
            n_correct=0,
            created_at="2026-02-01T00:00:00+00:00",
            run_id="unanswered",
            flags=frozenset({ResultFlag.NO_ANSWERS}),
        ),
    ]

    selection = select(measurements, SelectionRequest())

    assert selection.cells["m"]["gsm8k"].run_id == "good"
    (rejection,) = selection.rejections
    assert rejection.run_id == "unanswered"
    assert rejection.reason == "flagged no_answers"


def test_a_caller_can_readmit_flagged_results():
    """Exclusion is the default policy, not a claim the result is wrong; a caller that wants to see it
    clears the flag set and gets the newest run back."""
    measurements = [
        _measurement(n_correct=60, created_at="2026-01-01T00:00:00+00:00", run_id="good"),
        _measurement(
            n_correct=0,
            created_at="2026-02-01T00:00:00+00:00",
            run_id="unanswered",
            flags=frozenset({ResultFlag.NO_ANSWERS}),
        ),
    ]

    selection = select(measurements, SelectionRequest(exclude_flags=frozenset()))

    assert selection.cells["m"]["gsm8k"].run_id == "unanswered"
    assert selection.rejections == ()


def test_complete_panel_filtering_drops_models_missing_a_selected_benchmark():
    measurements = [
        _measurement(benchmark="mmlu", model="full"),
        _measurement(benchmark="gsm8k", model="full"),
        _measurement(benchmark="mmlu", model="partial"),
    ]

    selection = select(
        measurements,
        SelectionRequest(panel=("mmlu", "gsm8k"), completeness=Completeness.COMPLETE_PANEL),
    )

    assert set(selection.cells) == {"full"}


def test_selection_filters_on_run_metadata_and_model_name():
    measurements = [
        _measurement(model="qwen3-0.6b", run_id="tpu"),
        _measurement(model="llama-3.1-8b", run_id="gpu"),
    ]
    metadata = {"tpu": {"accelerator": "v6e-8"}, "gpu": {"accelerator": "h100"}}

    by_accelerator = select(measurements, SelectionRequest(filters={"accelerator": "h100"}), metadata)
    by_name = select(measurements, SelectionRequest(model_query="qwen"), metadata)

    assert set(by_accelerator.cells) == {"llama-3.1-8b"}
    assert set(by_name.cells) == {"qwen3-0.6b"}


# --------------------------------------------------------------------------------------------------
# Record adapter
# --------------------------------------------------------------------------------------------------


def _record(
    *,
    eval_name: str = "gsm8k",
    metrics: dict[str, dict[str, float]],
    coverage: dict[str, TaskCoverage] | None = None,
    mechanism: str = "evalchemy",
    harbor: HarborRef | None = None,
    evalchemy: EvalchemyRef | None = None,
) -> EvalRunRecord:
    return EvalRunRecord(
        run_id="run-1",
        group_id="group-1",
        created_at="2026-01-01T00:00:00+00:00",
        user="tester",
        model=ModelRef(name="qwen3-0.6b", location="loc", backend="vllm"),
        eval=EvalRef(
            name=eval_name,
            mechanism=mechanism,
            tasks=(EvalTaskRef(name=eval_name, num_fewshot=5),),
            harbor=harbor,
            evalchemy=evalchemy,
        ),
        hardware=HardwareRef(platform="tpu", accelerator="v6e-8", region_or_cluster="us-central2"),
        status=RunStatus.SUCCEEDED,
        error=None,
        results_path="gs://bucket/run-1/results",
        metrics=metrics,
        coverage=coverage or {},
        jobs={},
        log_tails={},
        provenance=Provenance(git_sha="sha", eval_runtime="runtime", launch_host="host"),
    )


def test_record_adapter_recovers_the_item_counts_lm_eval_records():
    """lm-eval writes the graded-document count beside the metrics, so (k, n) is recoverable."""
    record = _record(
        metrics={
            "gsm8k_5shot": {
                "sample_len": 128.0,
                "exact_match,strict-match": 0.2265625,
                "exact_match_stderr,strict-match": 0.0371,
                "exact_match,flexible-extract": 0.421875,
                "exact_match_stderr,flexible-extract": 0.0438,
            }
        }
    )

    measurement = measurement_from_record(record)

    assert measurement is not None
    assert measurement.metric == "exact_match,flexible-extract"
    assert measurement.coverage.n_scored == 128
    assert measurement.n_correct == 54
    assert measurement.kind is MetricKind.BINARY
    assert ResultFlag.ATTRITION_UNREPORTED in measurement.flags


def test_record_adapter_scores_a_group_task_from_its_aggregate_row():
    """lm-eval writes a group's document-weighted aggregate beside the subtask rows it summarizes;
    re-averaging the two would be a different estimand and would double the item count."""
    record = _record(
        eval_name="mmlu",
        metrics={
            "mmlu_5shot/mmlu": {"sample_len": 14042.0, "acc,none": 0.635, "acc_stderr,none": 0.0038},
            "mmlu_5shot/mmlu_anatomy": {"sample_len": 135.0, "acc,none": 0.7, "acc_stderr,none": 0.04},
            "mmlu_5shot/mmlu_astronomy": {"sample_len": 152.0, "acc,none": 0.6, "acc_stderr,none": 0.04},
        },
    )

    measurement = measurement_from_record(record)

    assert measurement is not None
    assert measurement.value == pytest.approx(0.635)
    assert measurement.coverage.n_scored == 14042


def test_record_adapter_does_not_double_count_a_duplicated_task_directory():
    """A real record carries the whole mmlu panel twice, under ``mmlu_5shot`` and a ``tmp`` directory
    left by the launcher. Both copies measure the same documents, so the item count is not doubled."""
    record = _record(
        eval_name="mmlu",
        metrics={
            "mmlu_5shot/mmlu": {"sample_len": 14042.0, "acc,none": 0.63502, "acc_stderr,none": 0.0038},
            "tmp6jnnx8cc/mmlu": {"sample_len": 14042.0, "acc,none": 0.63488, "acc_stderr,none": 0.0038},
        },
    )

    measurement = measurement_from_record(record)

    assert measurement is not None
    assert measurement.coverage.n_scored == 14042
    assert measurement.value == pytest.approx(0.63502)


def test_record_adapter_reads_harbor_coverage_and_its_error_histogram():
    record = _record(
        eval_name="tb2",
        mechanism="harbor",
        harbor=HarborRef(dataset="tb2", version="1", agent="terminus-2", env="daytona"),
        metrics={"tb2": {"accuracy": 0.5, "mean_reward": 0.5, "solved": 111.0, "total": 222.0}},
        coverage={"tb2": TaskCoverage(n_attempted=240, n_scored=222, errors={"AgentTimeoutError": 18})},
    )

    measurement = measurement_from_record(record)

    assert measurement is not None
    assert measurement.coverage.n_attempted == 240
    assert measurement.coverage.rate == pytest.approx(222 / 240)
    assert measurement.coverage.errors == {"AgentTimeoutError": 18}
    assert ResultFlag.ATTRITION in measurement.flags
    assert measurement_interval(measurement).kind is IntervalKind.IDENTIFIED


def test_record_adapter_flags_a_declared_item_cap():
    record = _record(
        metrics={"gsm8k_5shot": {"sample_len": 128.0, "exact_match,none": 0.5, "exact_match_stderr,none": 0.04}},
        evalchemy=EvalchemyRef(
            apply_chat_template=True,
            max_gen_toks=512,
            max_eval_instances=128,
            num_concurrent=8,
            batch_size=None,
            seed=0,
        ),
    )

    measurement = measurement_from_record(record)

    assert measurement is not None
    assert measurement.item_cap == 128
    assert ResultFlag.CAPPED in measurement.flags


def test_record_adapter_falls_back_to_dispersion_when_a_value_is_not_a_count():
    """An unweighted mean across subtasks is not k/n, so it cannot take the Wilson path."""
    record = _record(
        eval_name="suite",
        metrics={
            "suite/one": {"sample_len": 7.0, "acc,none": 1 / 3, "acc_stderr,none": 0.1},
            "suite/two": {"sample_len": 11.0, "acc,none": 1 / 7, "acc_stderr,none": 0.1},
        },
    )

    measurement = measurement_from_record(record)

    assert measurement is not None
    assert measurement.kind is MetricKind.CONTINUOUS
    assert measurement.recorded_stderr == pytest.approx(math.sqrt(0.02) / 2)


def test_evalchemy_coverage_identifies_an_interval_lm_eval_metrics_alone_cannot():
    """The counts a run's own samples establish are what take it off the sampling-only path."""
    metrics = {"gsm8k_5shot": {"sample_len": 128.0, "exact_match,none": 0.5, "exact_match_stderr,none": 0.04}}
    without = measurement_from_record(_record(metrics=metrics))
    with_coverage = measurement_from_record(
        _record(metrics=metrics, coverage={"gsm8k_5shot": TaskCoverage(n_attempted=128, n_scored=128, n_correct=64)})
    )

    assert without is not None and with_coverage is not None
    assert measurement_interval(without).kind is IntervalKind.SAMPLING_ONLY
    assert measurement_interval(with_coverage).kind is IntervalKind.IDENTIFIED
    assert ResultFlag.ATTRITION_UNREPORTED not in with_coverage.flags


def test_a_recorded_pass_count_is_preferred_to_inverting_the_reported_rate():
    """The harness counted its own passes; recovering that count from a rounded rate is a fallback."""
    record = _record(
        metrics={"gsm8k_5shot": {"sample_len": 3.0, "exact_match,none": 1 / 3, "exact_match_stderr,none": 0.27}},
        coverage={"gsm8k_5shot": TaskCoverage(n_attempted=3, n_scored=3, n_correct=1)},
    )

    measurement = measurement_from_record(record)

    assert measurement is not None
    assert measurement.n_correct == 1
    assert measurement.kind is MetricKind.BINARY


def test_a_pass_count_that_contradicts_the_reported_value_is_not_substituted_for_it():
    """Pooled k/n and an unweighted mean over unequal subtasks are different numbers. Taking the
    tally as the numerator for a value it does not belong to would republish the run at a score it
    never reported."""
    record = _record(
        eval_name="suite",
        metrics={
            "suite/one": {"sample_len": 10.0, "acc,none": 0.9, "acc_stderr,none": 0.1},
            "suite/two": {"sample_len": 90.0, "acc,none": 0.1, "acc_stderr,none": 0.03},
        },
        coverage={
            "suite/one": TaskCoverage(n_attempted=10, n_scored=10, n_correct=9),
            "suite/two": TaskCoverage(n_attempted=90, n_scored=90, n_correct=9),
        },
    )

    measurement = measurement_from_record(record)

    assert measurement is not None
    # The record's headline is the unweighted mean of 0.9 and 0.1; the tallies pool to 18/100.
    assert measurement.value == pytest.approx(0.5)
    assert measurement.kind is MetricKind.CONTINUOUS
    assert measurement.n_correct is None


def test_a_run_whose_grader_extracted_no_answers_is_flagged_rather_than_dropped():
    """A benchmark-wide zero is a real score and a suspect one. The flag reports the evidence and
    leaves admission to the caller."""
    record = _record(
        metrics={"gsm8k_5shot": {"sample_len": 40.0, "exact_match,none": 0.0, "exact_match_stderr,none": 0.0}},
        coverage={"gsm8k_5shot": TaskCoverage(n_attempted=40, n_scored=40, n_correct=0, n_unanswered=40)},
    )

    measurement = measurement_from_record(record)

    assert measurement is not None
    assert measurement.value == 0.0
    assert ResultFlag.NO_ANSWERS in measurement.flags


def test_a_run_the_model_merely_failed_is_not_flagged_as_unanswered():
    record = _record(
        metrics={"gsm8k_5shot": {"sample_len": 40.0, "exact_match,none": 0.0, "exact_match_stderr,none": 0.0}},
        coverage={"gsm8k_5shot": TaskCoverage(n_attempted=40, n_scored=40, n_correct=0, n_unanswered=3)},
    )

    measurement = measurement_from_record(record)

    assert measurement is not None
    assert ResultFlag.NO_ANSWERS not in measurement.flags


def test_record_adapter_returns_nothing_for_a_run_that_produced_no_metrics():
    assert measurement_from_record(_record(metrics={})) is None
