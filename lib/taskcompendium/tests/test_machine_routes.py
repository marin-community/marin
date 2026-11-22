# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verified backend routes survive curation gating."""

import json

from shellbox.machine import Backend

from taskcompendium.convert.answers import numeric_answer_task
from taskcompendium.models import Source, TaskSpec
from taskcompendium.pipeline.machine_routes import export_machine_route
from taskcompendium.pipeline.models import CheckResult, CheckStatus, RawRow, TierRole, TierVerdict
from taskcompendium.pipeline.source_verification import SourceVerificationStatus, gate_source_row, sample_result_checks
from taskcompendium.pipeline.verification_trials import SampleResult, VerificationTrial, sample_result_routes
from taskcompendium.runtime.requirements import ROUTE_BACKENDS_LABEL

ROW = RawRow("task", Source(dataset="fixture", revision="1", row="0", importer_revision="1"), {})


def test_checked_routes_survive_trial_gating_and_task_export():
    task = numeric_answer_task(ROW, prompt="What is 3 + 4?", answer="7", tolerance_abs=0, tolerance_rel=0)
    checked = SampleResult(
        task_id=task.id,
        source=task.source,
        trials=[
            VerificationTrial(
                attempt=0,
                status=CheckStatus.PASS,
                checks=[CheckResult(check="golden", status=CheckStatus.PASS, detail="graded: reward=1.0")],
                rollouts=(),
                verdicts=(
                    TierVerdict(role=TierRole.TASK, tier=Backend.LOCAL, status=CheckStatus.PASS, detail="passed"),
                ),
                route_backends=(Backend.LOCAL, Backend.DAYTONA),
            )
        ],
    )
    checked = SampleResult.model_validate_json(checked.model_dump_json())
    assert sample_result_routes(checked) == (Backend.LOCAL, Backend.DAYTONA)
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
        routes_by_task={task.id: sample_result_routes(checked)},
    )
    # The route rides the row's labels column; the task spec is unchanged so the
    # oracle-filter integrity check at export still matches.
    assert json.loads(gated["labels"][ROUTE_BACKENDS_LABEL]) == ["local", "daytona"]
    assert TaskSpec.model_validate_json(gated["task_json"]).labels == {}
    exported = export_machine_route(kept, gated)
    labels = TaskSpec.model_validate_json(exported["task_json"]).labels
    assert json.loads(labels[ROUTE_BACKENDS_LABEL]) == ["local", "daytona"]
    # A row the controls never checked records no route.
    untouched = gate_source_row(
        {**kept, "task_id": "unchecked"},
        status=SourceVerificationStatus.PASSED,
        results={task.id: sample_result_checks(checked)},
        routes_by_task={task.id: (Backend.LOCAL, Backend.DAYTONA)},
    )
    assert "labels" not in untouched or ROUTE_BACKENDS_LABEL not in dict(untouched["labels"] or ())
