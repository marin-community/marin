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
from taskcompendium.pipeline.controls import answer_reply, control_suite, wrong_reply
from taskcompendium.pipeline.execution_telemetry import PhaseTelemetry
from taskcompendium.pipeline.models import CheckResult, CheckStatus, Controls, GraderReadiness, RawRow, Reply
from taskcompendium.pipeline.source_verification import (
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
# Requires a "count" property while forbidding every property, so no reply satisfies it.
CONTRADICTORY_SCHEMA = {"type": "object", "required": ["count"], "properties": {}, "additionalProperties": False}


def count_reply(task: TaskSpec) -> Reply:
    return answer_reply(task, json.dumps({"count": 1}))


# A JSON schema grader has no reference instance, so these controls have no golden.
SCHEMA_CONTROLS = Controls(negative=wrong_reply)
COUNT_CONTROLS = Controls(golden=count_reply, negative=wrong_reply)


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
    original_path = tmp_path / "sample-verification.json"
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
                    report_path=str(tmp_path / "full-verification.json"),
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
    assert all(
        any(check.status == CheckStatus.SKIPPED for check in trial.checks) for trial in resumed.results[0].result.trials
    )
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
        if reusable:
            sample_report = output / "verification.json"
            original_bytes = sample_report.read_bytes()
            full = verify_source(
                str(source.parent),
                str(tmp_path / "full"),
                policy,
                suite,
                1,
                context=context,
                previous_report_path=str(sample_report),
            )
            assert sample_report.read_bytes() == original_bytes
            assert {item["execution_id"] for item in full["verification"]["evidence"]} == {
                item["execution_id"] for item in first["verification"]["evidence"]
            }
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


def test_sample_failure_remains_rejected_outside_full_verification_sample(tmp_path):
    ids = [{"task_id": f"task-{i}", "filter_status": "keep"} for i in range(101)]
    selected = {row["task_id"] for row in sample_rows(iter(ids), size=100, seed=0).rows}
    bad_id = next(row["task_id"] for row in ids if row["task_id"] not in selected)
    extra_id = min(selected)
    rows = []
    for row in ids:
        task = schema_task(row["task_id"], CONTRADICTORY_SCHEMA if row["task_id"] == bad_id else OBJECT_SCHEMA)
        rows.append({**row, "task_json": task.model_dump_json(), "filter_reasons": []})
    sample_source, full_source = tmp_path / "sample-source", tmp_path / "full-source"
    for source, records in (
        (sample_source, [row for row in rows if row["task_id"] != extra_id]),
        (full_source, rows),
    ):
        (source / "audit").mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(records, schema=TASK_SCHEMA), source / "audit/part-00000.parquet")
        (source / "manifest.json").write_text(json.dumps({"input_rows": len(records)}))
    suite = control_suite(COUNT_CONTROLS, None)
    policy = SourceVerificationPolicy(100, 0, 2, 0.95)
    sample_output, full_output = tmp_path / "sample", tmp_path / "full"
    with ZephyrContext(max_workers=2, name="sample-failure-provenance") as context:
        sample = verify_source(str(sample_source), str(sample_output), policy, suite, 2, context=context)
        sample_report = sample_output / "verification.json"
        original_bytes = sample_report.read_bytes()
        full = verify_source(
            str(full_source),
            str(full_output),
            policy,
            suite,
            2,
            context=context,
            previous_report_path=str(sample_report),
        )
        assert sample_report.read_bytes() == original_bytes
        resumed = verify_source(str(full_source), str(full_output), policy, suite, 2, context=context)
        invalid_rows = [
            (
                {**row, "task_json": None, "filter_status": "reject", "filter_reasons": ["source_defect:converter"]}
                if row["task_id"] == bad_id
                else row
            )
            for row in rows
        ]
        pq.write_table(pa.Table.from_pylist(invalid_rows, schema=TASK_SCHEMA), full_source / "audit/part-00000.parquet")
        invalid_output = tmp_path / "invalid"
        verify_source(
            str(full_source),
            str(invalid_output),
            policy,
            suite,
            2,
            context=context,
            previous_report_path=str(full_output / "verification.json"),
        )
        corrected = schema_task(bad_id, OBJECT_SCHEMA)
        corrected_rows = [
            {**row, "task_json": corrected.model_dump_json()} if row["task_id"] == bad_id else row for row in rows
        ]
        pq.write_table(
            pa.Table.from_pylist(corrected_rows, schema=TASK_SCHEMA), full_source / "audit/part-00000.parquet"
        )
        corrected_output = tmp_path / "corrected"
        verify_source(
            str(full_source),
            str(corrected_output),
            policy,
            suite,
            2,
            context=context,
            previous_report_path=str(full_output / "verification.json"),
        )
    assert sample["verification"]["status"] == "passed"
    assert sample["verification"]["counts"]["failed"] == 1
    for report in (full, resumed):
        assert report["verification"]["status"] == "passed"
        assert report["verification"]["sample_count"] == 100
        assert report["verification"]["counts"]["failed"] == 0
        assert bad_id not in {result["task_id"] for result in report["verification"]["results"]}
        assert {item["evidence"]["task_id"] for item in report["verification"]["prior_failed_trials"]} == {bad_id}
    output_rows = [row for file in (full_output / "audit").glob("*.parquet") for row in pq.read_table(file).to_pylist()]
    failed = next(row for row in output_rows if row["task_id"] == bad_id)
    assert failed["filter_status"] == "reject"
    assert any(check["status"] == "fail" for check in failed["checks"])
    assert sum(row["filter_status"] == "keep" for row in output_rows) == 100
    invalid = next(
        row
        for file in (invalid_output / "audit").glob("*.parquet")
        for row in pq.read_table(file).to_pylist()
        if row["task_id"] == bad_id
    )
    assert invalid["task_json"] is None
    assert invalid["filter_status"] == "reject"
    assert invalid["filter_reasons"] == ["source_defect:converter"]
    corrected_row = next(
        row
        for file in (corrected_output / "audit").glob("*.parquet")
        for row in pq.read_table(file).to_pylist()
        if row["task_id"] == bad_id
    )
    assert corrected_row["filter_status"] == "keep"
    assert corrected_row["task_json"] == corrected.model_dump_json()


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
    if expected == "skipped":
        assert report.pass_fraction is None
        row = {"task_id": "0", "filter_status": "keep", "filter_reasons": [], "grader_readiness": "unverified"}
        gated = gate_source_row(row, status=SourceVerificationStatus.SKIPPED, results={"0": results[0].trials[0].checks})
        assert gated["filter_status"] == "keep"
        assert gated["grader_readiness"] == "unverified"


def test_missing_schema_golden_keeps_available_checks_and_does_not_certify_tasks(tmp_path):
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
            SourceVerificationPolicy(1, 0, 2, 1.0),
            control_suite(SCHEMA_CONTROLS, None),
            1,
            context=context,
            telemetry=telemetry,
        )
    report = manifest["verification"]
    result = SampleResult.model_validate(report["results"][0])
    metrics = next(execution.counters for execution in telemetry.executions if execution.operation == "trials")
    assert metrics["verification/attempts"] == 2
    assert metrics["verification/control/golden/skipped"] == 2
    assert metrics["verification/control/empty/pass"] == metrics["verification/control/negative/pass"] == 2
    assert metrics["verification/trial/pass"] == 2
    for trial in result.trials:
        checks = {check.check: check.status for check in trial.checks}
        assert checks == {"empty": CheckStatus.PASS, "golden": CheckStatus.SKIPPED, "negative": CheckStatus.PASS}
    assert report["status"] == "passed"
    assert report["counts"]["checked"] == report["counts"]["skipped"] == 1
    gated = pq.read_table(output / "audit").to_pylist()[0]
    assert gated["filter_status"] == "keep"
    assert gated["grader_readiness"] == "unverified"
    unsampled = gate_source_row(
        {**row, "task_id": "unsampled"},
        status=SourceVerificationStatus(report["status"]),
        results={task.id: [check for trial in result.trials for check in trial.checks]},
        sampled_readiness=GraderReadiness.UNVERIFIED,
    )
    assert unsampled["filter_status"] == "keep"
    assert unsampled["grader_readiness"] == "unverified"


def test_source_decision_preserves_infrastructure_failure_alongside_failed_control():
    sample = VerificationSample(1, [{"task_id": "one"}])
    results = [
        SampleResult(
            task_id="one",
            source=Source(dataset="fixture", revision="1", row="one", importer_revision="1"),
            trials=[
                VerificationTrial(
                    attempt=0,
                    status=CheckStatus.FAIL,
                    checks=[
                        CheckResult(check="oracle", status=CheckStatus.FAIL, detail="wrong output"),
                        CheckResult(check="negative", status=CheckStatus.INFRA_ERROR, detail="machine unavailable"),
                    ],
                    rollouts=(),
                )
            ],
        )
    ]
    report = source_verification_report(sample, results, SourceVerificationPolicy(1, 0, 1, 1.0))
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
    checks = [CheckResult(check="oracle", status=CheckStatus.INFRA_ERROR, detail="machine unavailable")]
    deferred = gate_source_row(row, status=SourceVerificationStatus.INCONCLUSIVE, results={"one": checks})
    assert deferred["filter_status"] == "defer"
    assert deferred["filter_reasons"] == ["source_verification:inconclusive"]

    checks.append(CheckResult(check="negative", status=CheckStatus.FAIL, detail="incorrect submission accepted"))
    rejected = gate_source_row(row, status=SourceVerificationStatus.INCONCLUSIVE, results={"one": checks})
    assert rejected["filter_status"] == "reject"
    assert rejected["filter_reasons"] == ["check:negative"]
