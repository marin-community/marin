# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verified source-owned blockers become repair handoffs without escalation."""

from dataclasses import replace

import pytest
from finestore.rl.difficulty import Solver

from experiments.post_training.task_curation.difficulty.executor import EscalationLog, execute_plan
from experiments.post_training.task_curation.difficulty.ledger import append_rows, read_rows
from experiments.post_training.task_curation.difficulty.planner import plan
from experiments.post_training.task_curation.difficulty.source_repairs import (
    SourceRepairRoute,
    record_completed_source_repairs,
    source_repair_handoffs,
)
from experiments.post_training.task_curation.difficulty.state import (
    CellStatus,
    CoveragePolicy,
    GapReason,
    Outcome,
    cell_state,
)
from experiments.post_training.task_curation.tests.test_difficulty_executor import JobStore
from experiments.post_training.task_curation.tests.test_difficulty_reconciler import SOURCE, row


@pytest.mark.parametrize(
    "source_id,cause,route",
    [
        (
            "Nemotron:nemotron_ultra_mopd/ultra_sft_step3200_math_tir",
            "task_spec_tool_provider_required",
            SourceRepairRoute.SCOPED_CONSUMER_GATE,
        ),
        (
            "Nemotron:nemotron_ultra_rlvr1/ultra_sft_step3200_math_tir",
            "task_spec_tool_provider_required",
            SourceRepairRoute.SCOPED_CONSUMER_GATE,
        ),
        (
            "Nemotron:nemotron_ultra_rlvr2/ultra_sft_step3200_math_tir",
            "task_spec_tool_provider_required",
            SourceRepairRoute.SCOPED_CONSUMER_GATE,
        ),
        (
            "TaskTrove:laion__nemotron-gym-arc-agi-python-inductive-v2",
            "task_spec_shared_executor_required",
            SourceRepairRoute.SCOPED_CONSUMER_GATE,
        ),
        ("RST:laion/rts-1k-rootless-apptainer", "task_spec_unresolved_build", SourceRepairRoute.SHARED_GRADER_OPTION_2),
        (
            "PrimeIntellect:PrimeIntellect/Multi-SWE-RL-Verified",
            "task_spec_unresolved_build",
            SourceRepairRoute.SHARED_GRADER_OPTION_2,
        ),
    ],
)
def test_completed_source_rechecks_record_typed_handoffs_without_escalation(tmp_path, source_id, cause, route):
    source = replace(SOURCE, source_id=source_id)
    observed = replace(
        row("original", outcome=Outcome.OBSERVATION),
        source_id=source_id,
        task_id=None,
        task_sha256=None,
        reason=GapReason.EXECUTABILITY_ADMISSION_PENDING,
        capability="executability_admission_pending:" + cause,
    )
    ledger = str(tmp_path / "ledger")
    append_rows(ledger, [observed])
    (action,) = plan([cell_state([observed], source, Solver.SMALL, CoveragePolicy())], [observed])
    backend = JobStore(str(tmp_path / "jobs"))
    execute_plan([action], [observed], [source], ledger, backend, EscalationLog(ledger))
    result = {
        "status": "blocked",
        "reason": GapReason.EXECUTABILITY_ADMISSION_PENDING,
        "source_id": source_id,
        "registry_revision": source.registry_revision,
        "checked_tasks": 2,
        "population_changed": False,
        "detail": "Verified original contract blocker",
        "blockers": [{"task_id": "a", "task_sha256": "a" * 64, "reason": "executability_admission_pending:" + cause}],
    }
    target = backend.complete_repair(action, result, ledger)
    rows = read_rows(ledger)
    (handoff,) = source_repair_handoffs(ledger)
    assert handoff["route"] == route and handoff["evidence_uri"] == str(target)
    assert handoff["registry_revision"] == source.registry_revision
    assert handoff["blockers"] == result["blockers"]
    assert not any(r.outcome == Outcome.ESCALATED for r in rows)
    assert not (tmp_path / "ledger" / "ESCALATIONS.md").exists()
    after = cell_state(rows, source, Solver.SMALL, CoveragePolicy())
    assert after.state == CellStatus.BLOCKED and plan([after], rows) == []
    # A former controller's escalation can be reclassified through the same immutable proof.
    previous = replace(
        observed,
        attempt_id="prior-escalation",
        outcome=Outcome.ESCALATED,
        action_id=next(r.action_id for r in rows if r.outcome == Outcome.SUBMITTED),
        evidence_uri=str(target),
    )
    assert len(record_completed_source_repairs(ledger, [previous])) == 1
    assert source_repair_handoffs(ledger) == [handoff]
