# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submission replay, quota admission and completed repair transitions."""

import json
from dataclasses import dataclass, field, replace

from finestore.rl.difficulty import Solver
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.difficulty.executor import (
    EscalationLog,
    Job,
    collect_finished,
    execute_plan,
)
from experiments.post_training.task_curation.difficulty.ledger import read_rows
from experiments.post_training.task_curation.difficulty.planner import action_id, plan
from experiments.post_training.task_curation.difficulty.state import (
    CellStatus,
    CoveragePolicy,
    Gap,
    GapReason,
    Outcome,
    cell_state,
)
from experiments.post_training.task_curation.tests.test_difficulty_reconciler import SOURCE, blocked


@dataclass
class JobStore:
    root: str
    existing: list[Job] = field(default_factory=list)

    def jobs(self) -> list[Job]:
        return self.existing

    def submit(self, action, manifest_uri) -> str:
        # A fake external scheduler persists the immutable command it accepted.
        key = action_id(action)
        target = StoragePath(self.root) / (key + ".json")
        target.parent.mkdirs()
        target.write_text(StoragePath(manifest_uri).read_text())
        job = Job("/owner/controller/difficulty-reconcile-" + key[:24], "running", None)
        self.existing.append(job)
        return job.job_id


def test_execution_records_submissions_and_replay_preserves_scheduler_state(tmp_path):
    ledger = str(tmp_path / "ledger")
    backend = JobStore(str(tmp_path / "jobs"))
    cell = cell_state([], SOURCE, Solver.SMALL, CoveragePolicy())
    actions = plan([cell], [])
    first = execute_plan(actions, [], [SOURCE], ledger, backend, EscalationLog(ledger))
    second = execute_plan(actions, read_rows(ledger), [SOURCE], ledger, backend, EscalationLog(ledger))
    assert len(first["submitted"]) == 1 and second["submitted"] == []
    assert len(list((tmp_path / "jobs").glob("*.json"))) == 1
    assert sum(row.outcome == Outcome.SUBMITTED for row in read_rows(ledger)) == 1


def test_capacity_and_legacy_solver_admission_defer_without_creating_attempts(tmp_path):
    ledger = str(tmp_path / "ledger")
    cell = cell_state([], SOURCE, Solver.SMALL, CoveragePolicy())
    actions = plan([cell], [])
    full = JobStore(str(tmp_path / "jobs"), [Job("/busy", "running", None, 22)])
    result = execute_plan(actions, [], [SOURCE], ledger, full, EscalationLog(ledger))
    assert result["submitted"] == [] and result["deferred"] == [action_id(actions[0])]
    legacy = JobStore(str(tmp_path / "jobs"), [Job("/owner/atlas-difficulty-small-v24", "running", Solver.SMALL)])
    assert execute_plan(actions, [], [SOURCE], ledger, legacy, EscalationLog(ledger))["submitted"] == []
    assert read_rows(ledger) == []


def test_shared_repair_completion_unblocks_only_the_original_population(tmp_path):
    ledger = str(tmp_path / "ledger")
    cell = blocked(Gap(GapReason.DAYTONA_SNAPSHOT_PREFLIGHT_PENDING))
    actions = plan([cell], [])
    backend = JobStore(str(tmp_path / "jobs"))
    execute_plan(actions, [], [SOURCE], ledger, backend, EscalationLog(ledger))
    backend.existing = [replace(backend.existing[0], state="succeeded")]
    result = StoragePath(ledger) / "v1" / "results" / (action_id(actions[0]) + ".json")
    result.parent.mkdirs()
    result.write_text(json.dumps({"status": "ready", "snapshot": "original"}))
    before = read_rows(ledger)
    additions = collect_finished(before, backend.jobs(), ledger, EscalationLog(ledger))
    restored = read_rows(ledger)
    assert any(
        row.outcome == Outcome.REPAIRED and row.reason == GapReason.DAYTONA_SNAPSHOT_PREFLIGHT_PENDING
        for row in additions
    )
    assert cell_state(restored, SOURCE, Solver.SMALL, CoveragePolicy()).state == CellStatus.NEEDS_ATTEMPTS
    changed = replace(SOURCE, registry_revision="new-pin")
    assert cell_state(restored, changed, Solver.SMALL, CoveragePolicy()).gaps == (Gap(GapReason.NO_MEASUREMENT_ROW),)
    assert collect_finished(restored, backend.jobs(), ledger, EscalationLog(ledger)) == []


def test_blocked_repair_escalates_once_and_leaves_other_cells_schedulable(tmp_path):
    ledger = str(tmp_path / "ledger")
    cell = blocked(Gap(GapReason.TASK_SPEC_FACTORY_PENDING))
    actions = plan([cell], [])
    backend = JobStore(str(tmp_path / "jobs"))
    execute_plan(actions, [], [SOURCE], ledger, backend, EscalationLog(ledger))
    backend.existing = [replace(backend.existing[0], state="succeeded")]
    result = StoragePath(ledger) / "v1" / "results" / (action_id(actions[0]) + ".json")
    result.parent.mkdirs()
    result.write_text(
        json.dumps(
            {
                "status": "blocked",
                "reason": GapReason.TASK_SPEC_FACTORY_PENDING,
                "detail": "Original frozen panel is not available",
            }
        )
    )
    collect_finished(read_rows(ledger), backend.jobs(), ledger, EscalationLog(ledger))
    rows = read_rows(ledger)
    assert sum(row.outcome == Outcome.ESCALATED for row in rows) == 1
    assert len((tmp_path / "ledger" / "ESCALATIONS.md").read_text().splitlines()) == 3
    other = replace(SOURCE, source_id="other")
    next_actions = plan([cell, cell_state([], other, Solver.SMALL, CoveragePolicy())], rows)
    assert all(action_id(action) != action_id(actions[0]) for action in next_actions)
    assert any(action.sources[0].source_id == "other" for action in next_actions)
