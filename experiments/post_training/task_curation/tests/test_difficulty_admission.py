# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Frozen admission checks clear stale gaps or retain specific contract blockers."""

import json
from dataclasses import replace

import pytest
from finestore.rl.difficulty import DifficultyGap, Solver
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.models import ProviderRequirement, TaskSpec

from experiments.post_training.task_curation import difficulty_preparation
from experiments.post_training.task_curation.difficulty.executor import EscalationLog, execute_plan
from experiments.post_training.task_curation.difficulty.importer import catalog_sources
from experiments.post_training.task_curation.difficulty.ledger import append_rows, read_rows
from experiments.post_training.task_curation.difficulty.legacy import legacy_gaps
from experiments.post_training.task_curation.difficulty.planner import (
    AttemptBatch,
    RecheckExecutabilityAdmission,
    plan,
)
from experiments.post_training.task_curation.difficulty.remediate import repair_action
from experiments.post_training.task_curation.difficulty.state import (
    CellStatus,
    CoveragePolicy,
    GapReason,
    Outcome,
    cell_state,
)
from experiments.post_training.task_curation.tests.prebuilt_fixtures import prebuilt_task
from experiments.post_training.task_curation.tests.test_difficulty_executor import JobStore
from experiments.post_training.task_curation.tests.test_difficulty_reconciler import row

NESTED_VERIFIER_DEADLINE = "executability_admission_pending:math_verifier_deadline:comparison"


@pytest.mark.parametrize("provider_missing", [False, True])
@pytest.mark.parametrize(
    "legacy_reason",
    [
        None,
        DifficultyGap.EXECUTABILITY_ADMISSION_PENDING,
        DifficultyGap.TOOL_PROVIDER_REQUIRED,
        DifficultyGap.MATH_VERIFIER_DEADLINE,
        NESTED_VERIFIER_DEADLINE,
    ],
)
def test_admission_recheck_uses_frozen_contracts_before_attempts_or_escalation(
    tmp_path, monkeypatch, difficulty_source, provider_missing, legacy_reason
):
    task = TaskSpec.model_validate(json.loads(prebuilt_task("admission-task")["lowered_task_spec"])["task"])
    if provider_missing:
        task = task.model_copy(
            update={
                "environment_requirements": task.environment_requirements.model_copy(
                    update={
                        "tool_providers": {
                            "python": ProviderRequirement(action_interface="fixture.python.v1", initial_state={})
                        }
                    }
                )
            }
        )
    source = difficulty_source(task)
    # Freeze the panel service response; preparation still checks its task hash and real executability gate.
    monkeypatch.setattr(difficulty_preparation, "panel_task_runtimes", lambda panel: ([task], [{}]))
    queue = tmp_path / "queue.json"
    queue.write_text(json.dumps([source.model_dump(mode="json")]))
    sha = file_sha256(StoragePath(str(queue)))
    (catalog,) = catalog_sources(str(queue), sha)
    observed = replace(
        row("prior", outcome=Outcome.OBSERVATION),
        source_id=catalog.source_id,
        registry_revision=catalog.registry_revision,
        task_id=None,
        task_sha256=None,
        reason=GapReason.EXECUTABILITY_ADMISSION_PENDING,
        capability=legacy_gaps(legacy_reason)[0].capability if legacy_reason else None,
    )
    ledger = str(tmp_path / "ledger")
    append_rows(ledger, [observed, replace(observed, solver=Solver.LARGE, attempt_id="prior-large")])
    cells = [cell_state(read_rows(ledger), catalog, solver, CoveragePolicy()) for solver in (Solver.SMALL, Solver.LARGE)]
    (action,) = plan(cells, [])
    assert isinstance(action, RecheckExecutabilityAdmission) and len(action.cells) == 2
    backend = JobStore(str(tmp_path / "jobs"))
    execute_plan([action], read_rows(ledger), [catalog], ledger, backend, EscalationLog(ledger))
    result = repair_action(action, tmp_path / "repair", ledger)
    target = backend.complete_repair(action, result, ledger)
    rows = read_rows(ledger)
    after = cell_state(rows, catalog, Solver.SMALL, CoveragePolicy())
    assert file_sha256(StoragePath(str(queue))) == sha
    runtime_unverified = legacy_reason in {
        DifficultyGap.MATH_VERIFIER_DEADLINE,
        NESTED_VERIFIER_DEADLINE,
    }
    if provider_missing or runtime_unverified:
        assert after.state == CellStatus.BLOCKED
        if provider_missing:
            assert result["blockers"][0]["task_id"] == task.id
            assert DifficultyGap.TOOL_PROVIDER_REQUIRED.value in result["blockers"][0]["reason"]
        else:
            assert result["blockers"] == []
            assert result["unverified_capabilities"] == [legacy_gaps(legacy_reason)[0].capability]
        assert any(r.outcome == Outcome.ESCALATED and r.evidence_uri == str(target) for r in rows)
        repeated = [cell_state(rows, catalog, solver, CoveragePolicy()) for solver in (Solver.SMALL, Solver.LARGE)]
        assert plan(repeated, rows) == []
    else:
        assert after.state == CellStatus.NEEDS_ATTEMPTS
        (attempts,) = plan([after], rows)
        assert isinstance(attempts, AttemptBatch) and attempts.sources[0].tasks == (
            (task.id, source.sample.tasks[0].task_sha256, 1),
        )
