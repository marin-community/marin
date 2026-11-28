# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verified backend evidence survives curation gating and excludes failed backends."""

import json

from shellbox.machine import Backend

from taskcompendium.convert.answers import numeric_answer_task
from taskcompendium.models import Source, TaskSpec
from taskcompendium.pipeline.machine_routes import (
    ROUTE_VERDICTS_LABEL,
    attach_route_evidence,
    excluded_backends,
    export_route_evidence,
    merge_route_evidence,
    route_evidence,
    task_route_evidence,
)
from taskcompendium.pipeline.models import (
    CheckResult,
    CheckStatus,
    MachineRole,
    RawRow,
    ValidationVerdict,
    VerificationReport,
)
from taskcompendium.pipeline.source_verification import SourceVerificationStatus, gate_source_row, sample_result_checks
from taskcompendium.pipeline.verification_trials import SampleResult, VerificationTrial, result_route_evidence

ROW = RawRow("task", Source(dataset="fixture", revision="1", row="0", importer_revision="1"), {})


def passing_trial(*verdicts: ValidationVerdict) -> VerificationTrial:
    return VerificationTrial(
        attempt=0,
        status=CheckStatus.PASS,
        report=VerificationReport(
            checks=[CheckResult(check="golden", status=CheckStatus.PASS, detail="graded: reward=1.0", verdicts=verdicts)]
        ),
    )


def test_checked_evidence_survives_trial_gating_and_task_export():
    task = numeric_answer_task(ROW, prompt="What is 3 + 4?", answer="7", tolerance_abs=0, tolerance_rel=0)
    checked = SampleResult(
        task_id=task.id,
        source=task.source,
        trials=[
            passing_trial(
                ValidationVerdict(
                    role=MachineRole.TASK, backend=Backend.LOCAL, status=CheckStatus.PASS, detail="passed"
                ),
                ValidationVerdict(
                    role=MachineRole.TASK, backend=Backend.DAYTONA, status=CheckStatus.FAIL, detail="grading failed"
                ),
            )
        ],
    )
    checked = SampleResult.model_validate_json(checked.model_dump_json())
    evidence = result_route_evidence(checked)
    assert evidence == {"task": {"local": "pass", "daytona": "fail"}}
    assert excluded_backends(evidence, MachineRole.TASK) == {"daytona"}
    kept = {
        "task_id": task.id,
        "task_json": task.model_dump_json(),
        "labels": {},
        "filter_status": "keep",
        "filter_reasons": [],
        "grader_readiness": "unverified",
    }
    gated = gate_source_row(
        kept,
        status=SourceVerificationStatus.PASSED,
        results={task.id: sample_result_checks(checked)},
        evidence_by_task={task.id: evidence},
    )
    # The evidence rides the row's labels column; the task spec is unchanged so the
    # oracle-filter integrity check at export still matches.
    assert json.loads(gated["labels"][ROUTE_VERDICTS_LABEL]) == {"task": {"local": "pass", "daytona": "fail"}}
    assert TaskSpec.model_validate_json(gated["task_json"]).labels == {}
    exported = export_route_evidence(kept, gated)
    exported_task = TaskSpec.model_validate_json(exported["task_json"])
    assert task_route_evidence(exported_task) == {"task": {"local": "pass", "daytona": "fail"}}
    assert excluded_backends(task_route_evidence(exported_task), MachineRole.TASK) == {"daytona"}


def test_unsampled_rows_inherit_the_source_sample_exclusions():
    """A backend whose sampled control failed cannot serve an unchecked row of the same source."""
    task = numeric_answer_task(ROW, prompt="What is 3 + 4?", answer="7", tolerance_abs=0, tolerance_rel=0)
    source_evidence = {"task": {"local": "fail", "daytona": "pass"}}
    kept = {
        "task_id": "unsampled",
        "task_json": task.model_dump_json(),
        "labels": {},
        "filter_status": "keep",
        "filter_reasons": [],
        "grader_readiness": "unverified",
    }
    gated = gate_source_row(
        kept,
        status=SourceVerificationStatus.PASSED,
        results={},
        evidence_by_task={},
        source_evidence=source_evidence,
    )
    assert gated["grader_readiness"] == "source_sampled"
    assert json.loads(gated["labels"][ROUTE_VERDICTS_LABEL]) == source_evidence
    # A skipped source carries no verification evidence at all.
    skipped = gate_source_row(
        kept,
        status=SourceVerificationStatus.SKIPPED,
        results={},
        evidence_by_task={},
        source_evidence=None,
    )
    assert skipped["grader_readiness"] == "unverified"
    assert ROUTE_VERDICTS_LABEL not in dict(skipped["labels"] or ())


def test_evidence_merging_keeps_the_most_severe_outcome():
    merged = merge_route_evidence(
        {"task": {"local": "pass", "daytona": "infra_error"}},
        {"task": {"local": "fail"}, "verifier": {"docker": "pass"}},
        None,
    )
    assert merged == {
        "task": {"local": "fail", "daytona": "infra_error"},
        "verifier": {"docker": "pass"},
    }
    assert excluded_backends(merged, MachineRole.TASK) == {"local"}
    assert excluded_backends(merged, MachineRole.VERIFIER) == set()


def test_route_evidence_serializes_verdicts_in_order():
    verdicts = [
        ValidationVerdict(role=MachineRole.TASK, backend=Backend.LOCAL, status=CheckStatus.SKIPPED, detail="ran"),
        ValidationVerdict(role=MachineRole.VERIFIER, backend=Backend.LOCAL, status=CheckStatus.PASS, detail="passed"),
        ValidationVerdict(
            role=MachineRole.TASK, backend=Backend.LOCAL, status=CheckStatus.PASS, detail="passed the control"
        ),
        ValidationVerdict(role=MachineRole.TASK, backend=Backend.DAYTONA, status=CheckStatus.FAIL, detail="failed"),
    ]
    assert route_evidence(verdicts) == {
        "task": {"local": "pass", "daytona": "fail"},
        "verifier": {"local": "pass"},
    }
    row = attach_route_evidence({"labels": {"other": "x"}}, route_evidence(verdicts))
    assert json.loads(row["labels"][ROUTE_VERDICTS_LABEL]) == route_evidence(verdicts)
    assert row["labels"]["other"] == "x"
