# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source samples retain failures and distinguish skipped controls from passes."""

import json
from dataclasses import asdict, replace
from functools import partial

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from verifyit.spec import SchemaFormat
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from taskcompendium.convert.answers import json_schema_task
from taskcompendium.grading_result import GradeResult, GradingFailure, Outcome
from taskcompendium.models import Source, TaskSpec
from taskcompendium.pipeline.audit_schema import TASK_SCHEMA
from taskcompendium.pipeline.controls import control_suite
from taskcompendium.pipeline.execution_telemetry import PhaseTelemetry
from taskcompendium.pipeline.models import (
    CheckResult,
    CheckStatus,
    Controls,
    GraderReadiness,
    RawRow,
    VerificationReport,
)
from taskcompendium.pipeline.source_verification import (
    INFRA_ERROR_RETRIES,
    SOURCE_VERIFICATION_REVISION,
    SampleResult,
    SourceVerificationPolicy,
    SourceVerificationStatus,
    VerificationSample,
    VerificationTrial,
    _verify_sample_with_evidence,
    gate_source_row,
    merge_samples,
    sample_result_checks,
    sample_rows,
    saved_trials,
    source_verification_report,
    verification_identity,
    verify_source,
)
from taskcompendium.pipeline.verification import DIAGNOSTIC_TAIL_CHARS, control_result

from .pipeline_stages import FixtureGradingMachines

OBJECT_SCHEMA = {"type": "object"}

# A JSON schema grader has no reference instance, so these controls grade an empty submission.
SCHEMA_CONTROLS = Controls()


def schema_task(task_id: str, schema: dict) -> TaskSpec:
    task = json_schema_task(
        RawRow(task_id, Source(dataset="fixture", revision="1", row=task_id, importer_revision="1"), {}),
        prompt="Return JSON matching " + json.dumps(schema),
        schema=json.dumps(schema),
        schema_format=SchemaFormat.JSON,
    )
    assert isinstance(task, TaskSpec)
    return task


@pytest.mark.parametrize(
    "outcome,status",
    [(Outcome.INFRA_ERROR, CheckStatus.INFRA_ERROR), (Outcome.INVALID_TASK, CheckStatus.FAIL)],
)
def test_failed_control_retains_runtime_diagnostics_in_saved_checks(outcome, status):
    error = "QEMU startup produced no output before timeout"
    grade = GradeResult(outcome, None, error, {"backend": "qemu", "returncode": None})
    check = control_result(grade, "positive_witness", 1.0)
    saved = CheckResult.model_validate_json(check.model_dump_json())
    assert saved.status == status
    assert error in saved.detail
    assert "qemu" in saved.detail
    assert "returncode" in saved.detail


def test_failed_control_keeps_grader_exit_code_and_output_tails():
    stderr = "Traceback from the first import\n" + "frame\n" * 500 + "ModuleNotFoundError: No module named 'skyrl_gym'"
    diagnostics = {
        "stdout": "grading started",
        "stderr": stderr,
        "exit_code": 1,
        "stdout_truncated": False,
        "stderr_truncated": False,
    }
    failed = GradeResult(
        Outcome.INFRA_ERROR,
        None,
        "Grader did not write a reward file",
        diagnostics=diagnostics,
        failure=GradingFailure.MISSING_REWARD,
    )
    check = control_result(failed, "golden", 1.0)
    assert check.status == CheckStatus.INFRA_ERROR
    assert "exit_code=1" in check.detail
    assert "grading started" in check.detail
    assert stderr[-DIAGNOSTIC_TAIL_CHARS:] in check.detail
    assert "Traceback from the first import" not in check.detail

    passed = control_result(GradeResult(Outcome.GRADED, 1.0, diagnostics=diagnostics), "golden", 1.0)
    assert passed.status == CheckStatus.PASS
    assert "skyrl_gym" not in passed.detail


@pytest.fixture
def reusable_schema_row():
    task = schema_task("reusable-schema", OBJECT_SCHEMA)
    return {"task_id": task.id, "task_json": task.model_dump_json(), "filter_status": "keep"}


def write_evidence(path, verified, policy):
    path.write_text(
        json.dumps(
            {
                "implementation_revision": SOURCE_VERIFICATION_REVISION,
                "policy": asdict(policy),
                "results": [verified.result.model_dump(mode="json")],
                "evidence": [item.model_dump(mode="json") for item in verified.evidence],
            }
        )
    )


def test_exact_reuse_preserves_independent_controls_and_original_provenance(tmp_path, reusable_schema_row):
    suite = control_suite(SCHEMA_CONTROLS, None)
    policy = SourceVerificationPolicy(1, 0, 2, 1.0)
    identity = verification_identity(suite, policy)
    original_path = tmp_path / "verification.json"
    with ZephyrContext(max_workers=1, name="verification-reuse") as context:
        fresh = context.execute(
            Dataset.from_list([reusable_schema_row]).map(
                partial(
                    _verify_sample_with_evidence,
                    suite=suite,
                    attempts=2,
                    identity=identity,
                    report_path=str(original_path),
                )
            )
        )
        verified = fresh.results[0]
        write_evidence(original_path, verified, policy)
        saved = saved_trials(str(original_path), identity=identity, attempts=2)
        row = {**reusable_schema_row, "saved_trials": {attempt: item for (_, attempt), item in saved.items()}}
        resumed = context.execute(
            Dataset.from_list([row]).map(
                partial(
                    _verify_sample_with_evidence,
                    suite=suite,
                    attempts=2,
                    identity=identity,
                    report_path=str(original_path),
                )
            )
        )
    assert fresh.counters["verification/executed_attempts"] == 2
    assert fresh.counters[f"verification/suite/{suite.id}/executed_attempts"] == 2
    assert resumed.counters["verification/reused_attempts"] == 2
    assert resumed.counters[f"verification/suite/{suite.id}/reused_attempts"] == 2
    assert resumed.counters.get("verification/executed_attempts", 0) == 0
    assert resumed.results[0].result == verified.result
    assert len({item.execution_id for item in verified.evidence}) == 2
    for original, reused in zip(verified.evidence, resumed.results[0].evidence, strict=True):
        assert reused.execution_id == original.execution_id
        assert reused.original_report == str(original_path)
        assert reused.reused_from == str(original_path)
    changed = TaskSpec.model_validate_json(reusable_schema_row["task_json"])
    changed = changed.model_copy(update={"tags": (*changed.tags, "changed-private-contract")})
    with pytest.raises(ValueError, match="exact selected task"):
        _verify_sample_with_evidence(
            {**row, "task_json": changed.model_dump_json()},
            suite=suite,
            attempts=2,
            identity=identity,
            report_path="unused",
        )
    changed_runtime = replace(suite, parameters={**suite.parameters, "worker_image": "fixture@sha256:" + "2" * 64})
    assert saved_trials(str(original_path), identity=verification_identity(changed_runtime, policy), attempts=2) == {}
    assert (
        saved_trials(str(original_path), identity=verification_identity(suite, replace(policy, seed=1)), attempts=2)
        == {}
    )


@pytest.mark.parametrize("outages", [1, INFRA_ERROR_RETRIES + 1])
def test_attempt_with_infrastructure_error_is_rerun_before_it_is_recorded(reusable_schema_row, outages):
    """A sandbox that never started says nothing about the task, so the attempt runs again; only
    an outage that outlasts every retry is recorded."""
    suite = control_suite(SCHEMA_CONTROLS, None)
    runs = []

    def flaky(task):
        runs.append(task.id)
        if len(runs) <= outages:
            return VerificationReport(
                [CheckResult(check="golden", status=CheckStatus.INFRA_ERROR, detail="sandbox did not start")]
            )
        return suite.run(task)

    policy = SourceVerificationPolicy(1, 0, 1, 1.0)
    identity = verification_identity(replace(suite, run=flaky), policy)
    verified = _verify_sample_with_evidence(
        reusable_schema_row, suite=replace(suite, run=flaky), attempts=1, identity=identity, report_path="r"
    )
    [trial] = verified.result.trials
    if outages <= INFRA_ERROR_RETRIES:
        assert trial.status == CheckStatus.PASS
        assert len(runs) == outages + 1
    else:
        assert trial.status == CheckStatus.INFRA_ERROR
        assert len(runs) == INFRA_ERROR_RETRIES + 1


@pytest.mark.parametrize("second_failed", [True, False])
def test_mixed_failure_infrastructure_trial_reexecutes_without_erasing_definite_failure(
    tmp_path, reusable_schema_row, second_failed
):
    suite = control_suite(SCHEMA_CONTROLS, None)
    policy = SourceVerificationPolicy(1, 0, 2, 1.0)
    identity = verification_identity(suite, policy)
    original = _verify_sample_with_evidence(
        reusable_schema_row, suite=suite, attempts=2, identity=identity, report_path="original"
    )
    failed = CheckResult(check="golden", status=CheckStatus.FAIL, detail="Wrong captured output")
    unavailable = CheckResult(check="runtime", status=CheckStatus.INFRA_ERROR, detail="Machine unavailable")
    original = replace(
        original,
        result=original.result.model_copy(
            update={
                "trials": [
                    original.result.trials[0].model_copy(
                        update={"status": CheckStatus.FAIL, "checks": [failed, unavailable]}
                    ),
                    (
                        original.result.trials[1].model_copy(update={"status": CheckStatus.FAIL, "checks": [failed]})
                        if second_failed
                        else original.result.trials[1]
                    ),
                ]
            }
        ),
    )
    path = tmp_path / "verification.json"
    write_evidence(path, original, policy)
    saved = saved_trials(str(path), identity=identity, attempts=2)
    assert {attempt for _, attempt in saved} == {0, 1}
    row = {**reusable_schema_row, "saved_trials": {attempt: item for (_, attempt), item in saved.items()}}
    resumed = _verify_sample_with_evidence(row, suite=suite, attempts=2, identity=identity, report_path="retry")
    assert resumed.evidence[0].execution_id != original.evidence[0].execution_id
    assert resumed.evidence[1].execution_id == original.evidence[1].execution_id
    assert resumed.result.trials[1] == original.result.trials[1]
    assert resumed.result.previous_trials[0].trial == original.result.trials[0]
    assert resumed.result.previous_trials[0].evidence.execution_id == original.evidence[0].execution_id
    if not second_failed:
        assert all(trial.status == CheckStatus.PASS for trial in resumed.result.trials)
    decision = source_verification_report(VerificationSample(1, [row]), [resumed.result], policy)
    assert decision.status == SourceVerificationStatus.REJECTED
    assert decision.counts.inconsistent == 1
    assert decision.counts.failed == 1
    assert decision.counts.passed == decision.counts.infra_error == 0
    gated = gate_source_row(
        {**row, "filter_reasons": []},
        status=decision.status,
        results={resumed.result.task_id: sample_result_checks(resumed.result)},
    )
    assert gated["filter_status"] == "reject"
    assert "check:golden" in gated["filter_reasons"]
    write_evidence(path, resumed, policy)
    assert len(saved_trials(str(path), identity=identity, attempts=2)) == 2
    payload = json.loads(path.read_text())
    payload["evidence"][1]["execution_id"] = payload["evidence"][0]["execution_id"]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="distinct independent executions"):
        saved_trials(str(path), identity=identity, attempts=2)


@pytest.mark.parametrize(
    "runtime,reusable",
    [
        ({"backend": "qemu", "bundle_path": "/tmp/changeable"}, False),
        ({"network": "allow"}, False),
        ({"backend": "qemu", "worker_image": "image@sha256:" + "1" * 64}, True),
        (FixtureGradingMachines().identity(), True),
    ],
    ids=["qemu_bundle_path", "network_allowed", "qemu_worker_image", "offline_machines"],
)
def test_source_rerun_requires_immutable_offline_runtime(tmp_path, reusable_schema_row, runtime, reusable):
    suite = replace(control_suite(SCHEMA_CONTROLS, None), parameters=runtime)
    source = tmp_path / "source" / "audit"
    source.mkdir(parents=True)
    (source.parent / "manifest.json").write_text(json.dumps({"input_rows": 1}))
    row = {**reusable_schema_row, "filter_reasons": []}
    pq.write_table(pa.Table.from_pylist([row], schema=TASK_SCHEMA), source / "part-00000.parquet")
    output = tmp_path / "verified"
    policy = SourceVerificationPolicy(1, 0, 2, 1.0)
    with ZephyrContext(max_workers=1, name="persisted-verification-reuse") as context:
        first = verify_source(str(source.parent), str(output), policy, suite, 1, context=context)
        second = verify_source(str(source.parent), str(output), policy, suite, 1, context=context)
    first = first["verification"]
    second = second["verification"]
    assert first["results"][0]["trials"] == second["results"][0]["trials"]
    assert first["status"] == second["status"] == "passed"
    first_ids = {item["execution_id"] for item in first["evidence"]}
    second_ids = {item["execution_id"] for item in second["evidence"]}
    if reusable:
        assert first_ids == second_ids
        assert all(item["reused_from"] == str(output / "verification.json") for item in second["evidence"])
    else:
        assert first_ids.isdisjoint(second_ids)
        assert all(item["reused_from"] is None for item in second["evidence"])


def test_defective_task_row_is_rejected_when_its_source_passes():
    row = {"task_id": "0", "filter_status": "keep", "filter_reasons": [], "grader_readiness": "unverified"}
    checks = [CheckResult(check="empty", status=CheckStatus.DEFECT, detail="graded: reward=1.0; expected=0.0")]
    gated = gate_source_row(row, status=SourceVerificationStatus.PASSED, results={"0": checks})
    assert gated["filter_status"] == "reject"
    assert gated["filter_reasons"] == ["check:empty"]
    assert gated["grader_readiness"] == GraderReadiness.READY


def test_sample_is_independent_of_partitions_and_ignores_rejected_rows():
    rows = [{"task_id": f"task-{i}", "filter_status": "keep" if i % 3 else "reject"} for i in range(100)]
    whole = sample_rows(iter(rows), size=11, seed=42)
    partitions = (sample_rows(iter(rows[i::7]), size=11, seed=42) for i in range(7))
    merged = merge_samples(partitions, size=11, seed=42)
    reversed_sample = sample_rows(iter(reversed(rows)), size=11, seed=42)
    assert whole == merged == reversed_sample
    assert whole.eligible_count == 66
    assert len(whole.rows) == 11
    assert all(row["filter_status"] == "keep" for row in whole.rows)
    assert sample_rows(iter(rows), size=11, seed=43).rows != whole.rows
    assert len(sample_rows(iter(rows), size=1000, seed=42).rows) == 66


@pytest.mark.parametrize(
    "statuses, threshold, expected, passed, inconsistent",
    [
        ([("pass", "pass"), ("pass", "pass")], 1.0, "passed", 2, 0),
        # A rewarded empty submission is a task defect; the grader ran, so the source still passes.
        ([("pass", "pass"), ("defect", "defect"), ("pass", "defect")], 1.0, "passed", 3, 0),
        ([("pass", "pass"), ("pass", "fail")], 0.75, "rejected", 1, 1),
        ([("pass", "pass"), ("fail", "fail")], 0.5, "passed", 1, 0),
        ([("pass", "pass"), ("unsupported", "unsupported")], 0.5, "inconclusive", 1, 0),
        ([("pass", "pass"), ("infra_error", "pass")], 0.5, "inconclusive", 1, 0),
        ([("pass", "pass"), ("skipped", "skipped")], 1.0, "passed", 1, 0),
        ([("fail", "fail"), ("skipped", "skipped")], 0.5, "rejected", 0, 0),
        ([("skipped", "skipped")], 1.0, "skipped", 0, 0),
        ([], 0.0, "inconclusive", 0, 0),
    ],
)
def test_source_decision_counts_tasks_and_requires_complete_coverage(
    statuses, threshold, expected, passed, inconsistent
):
    sample = VerificationSample(len(statuses), [{"task_id": str(i)} for i in range(len(statuses))])
    results = [
        SampleResult(
            task_id=str(i),
            source=Source(dataset="fixture", revision="1", row=str(i), importer_revision="1"),
            trials=[
                VerificationTrial(
                    attempt=attempt,
                    status=CheckStatus(status),
                    checks=[CheckResult(check="oracle", status=CheckStatus(status), detail="fixture")],
                    rollouts=(),
                )
                for attempt, status in enumerate(trials)
            ],
        )
        for i, trials in enumerate(statuses)
    ]
    report = source_verification_report(sample, results, SourceVerificationPolicy(100, 0, 2, threshold))
    assert report.status == expected
    assert report.counts.passed == passed
    assert report.counts.inconsistent == inconsistent
    assert report.counts.defective == sum("defect" in trials for trials in statuses)
    assert report.counts.failed == sum("fail" in trials for trials in statuses)
    if expected == "skipped":
        assert report.pass_fraction is None
        row = {"task_id": "0", "filter_status": "keep", "filter_reasons": [], "grader_readiness": "unverified"}
        gated = gate_source_row(row, status=SourceVerificationStatus.SKIPPED, results={"0": results[0].trials[0].checks})
        assert gated["filter_status"] == "keep"
        assert gated["grader_readiness"] == "unverified"


def test_task_without_a_golden_is_verified_by_its_empty_submission(tmp_path):
    task = schema_task("schema-task", OBJECT_SCHEMA)
    row = {"task_id": task.id, "task_json": task.model_dump_json(), "filter_status": "keep", "filter_reasons": []}
    source = tmp_path / "source"
    (source / "audit").mkdir(parents=True)
    (source / "manifest.json").write_text(json.dumps({"input_rows": 1}))
    pq.write_table(pa.Table.from_pylist([row], schema=TASK_SCHEMA), source / "audit/part-00000.parquet")
    output = tmp_path / "verified"
    telemetry = PhaseTelemetry("verify")
    with ZephyrContext(max_workers=1, name="verification-controls") as context:
        manifest = verify_source(
            str(source),
            str(output),
            SourceVerificationPolicy(1, 0, 1, 1.0),
            control_suite(SCHEMA_CONTROLS, None),
            1,
            context=context,
            telemetry=telemetry,
        )
    report = manifest["verification"]
    [trial] = SampleResult.model_validate(report["results"][0]).trials
    metrics = next(execution.counters for execution in telemetry.executions if execution.operation == "trials")
    assert metrics["verification/attempts"] == 1
    assert metrics["verification/control/empty/pass"] == metrics["verification/trial/pass"] == 1
    assert {check.check: check.status for check in trial.checks} == {"empty": CheckStatus.PASS}
    assert report["status"] == "passed"
    assert report["counts"]["checked"] == report["counts"]["passed"] == 1
    gated = pq.read_table(output / "audit").to_pylist()[0]
    assert gated["filter_status"] == "keep"
    assert gated["grader_readiness"] == GraderReadiness.READY
    unsampled = gate_source_row(
        {**row, "task_id": "unsampled"},
        status=SourceVerificationStatus(report["status"]),
        results={task.id: trial.checks},
    )
    assert unsampled["filter_status"] == "keep"
    assert unsampled["grader_readiness"] == GraderReadiness.SOURCE_SAMPLED


def test_source_decision_preserves_infrastructure_failure_alongside_failed_control():
    sample = VerificationSample(1, [{"task_id": "one"}])
    outcomes = [(CheckStatus.FAIL, "wrong output"), (CheckStatus.INFRA_ERROR, "machine unavailable")]
    results = [
        SampleResult(
            task_id="one",
            source=Source(dataset="fixture", revision="1", row="one", importer_revision="1"),
            trials=[
                VerificationTrial(
                    attempt=attempt,
                    status=status,
                    checks=[CheckResult(check="golden", status=status, detail=detail)],
                    rollouts=(),
                )
                for attempt, (status, detail) in enumerate(outcomes)
            ],
        )
    ]
    report = source_verification_report(sample, results, SourceVerificationPolicy(1, 0, 2, 1.0))
    assert report.status == "inconclusive"
    assert report.counts.failed == report.counts.infra_error == 1


def test_passing_source_preserves_individual_failures_and_marks_source_only_evidence():
    failed = {"task_id": "failed", "filter_status": "keep", "filter_reasons": [], "grader_readiness": "unverified"}
    unsampled = {**failed, "task_id": "unsampled"}
    results = {"failed": [CheckResult(check="oracle", status=CheckStatus.FAIL, detail="Wrong captured output")]}
    rejected = gate_source_row(failed, status=SourceVerificationStatus.PASSED, results=results)
    assert rejected["filter_status"] == "reject"
    assert rejected["filter_reasons"] == ["check:oracle"]
    accepted = gate_source_row(unsampled, status=SourceVerificationStatus.PASSED, results=results)
    assert accepted["filter_status"] == "keep"
    assert accepted["grader_readiness"] == "source_sampled"


@pytest.mark.parametrize("source_status", list(SourceVerificationStatus))
def test_source_gate_preserves_unavailable_review_for_retry(source_status):
    row = {
        "task_id": "unreviewed",
        "filter_status": "defer",
        "filter_reasons": ["review:unavailable"],
        "grader_readiness": "unverified",
    }
    gated = gate_source_row(row, status=source_status, results={})
    assert gated == row


def test_inconclusive_source_defers_eligible_rows_but_preserves_failed_controls():
    row = {"task_id": "one", "filter_status": "keep", "filter_reasons": [], "grader_readiness": "unverified"}
    checks = [CheckResult(check="golden", status=CheckStatus.INFRA_ERROR, detail="machine unavailable")]
    deferred = gate_source_row(row, status=SourceVerificationStatus.INCONCLUSIVE, results={"one": checks})
    assert deferred["filter_status"] == "defer"
    assert deferred["filter_reasons"] == ["source_verification:inconclusive"]

    # A definite failure from an earlier trial outlasts a later infrastructure error.
    checks.append(CheckResult(check="golden", status=CheckStatus.FAIL, detail="correct submission rejected"))
    rejected = gate_source_row(row, status=SourceVerificationStatus.INCONCLUSIVE, results={"one": checks})
    assert rejected["filter_status"] == "reject"
    assert rejected["filter_reasons"] == ["check:golden"]
