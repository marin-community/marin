# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Keep old evidence readable while a new Large policy obtains its own coverage."""

import hashlib
import json
from dataclasses import replace
from functools import partial
from pathlib import Path

from finestore.rl.difficulty import PROTOCOL_ID, Solver

from experiments.post_training.task_curation.difficulty.executor import (
    EscalationLog,
    action_rows,
    execute_plan,
)
from experiments.post_training.task_curation.difficulty.ledger import append_rows, decode_partition_rows, read_rows
from experiments.post_training.task_curation.difficulty.planner import (
    AttemptBatch,
    action_from_payload,
    action_id,
    action_payload,
    load_action,
    plan,
)
from experiments.post_training.task_curation.difficulty.protocols import protocol_supersessions
from experiments.post_training.task_curation.difficulty.report import status_report
from experiments.post_training.task_curation.difficulty.seal import snapshot_cells
from experiments.post_training.task_curation.difficulty.state import (
    CellStatus,
    CoveragePolicy,
    Gap,
    GapReason,
    Outcome,
    cell_state,
)
from experiments.post_training.task_curation.difficulty_output_budget import LARGE_DYNAMIC_OUTPUT_PROTOCOL_ID
from experiments.post_training.task_curation.tests.test_difficulty_executor import JobStore
from experiments.post_training.task_curation.tests.test_difficulty_reconciler import SOURCE

FIXTURES = Path(__file__).parent / "fixtures"
V2 = LARGE_DYNAMIC_OUTPUT_PROTOCOL_ID
V2_POLICY = CoveragePolicy(large_protocol_id=V2)


def historical_row():
    """The fixture wire bytes were captured from source 8883623a213d."""
    return decode_partition_rows((FIXTURES / "difficulty-large-v1-row.json").read_bytes())[0]


def test_frozen_v1_row_bytes_and_action_identity_survive_decode_write_and_replay(tmp_path):
    raw = (FIXTURES / "difficulty-large-v1-row.json").read_bytes()
    row = historical_row()
    partition = append_rows(str(tmp_path), [row])
    assert partition is not None
    assert Path(partition).read_bytes() == raw
    assert Path(partition).name == hashlib.sha256(raw).hexdigest() + ".jsonl"
    assert read_rows(str(tmp_path)) == [row]
    assert append_rows(str(tmp_path), read_rows(str(tmp_path))) == partition
    payload = json.loads((FIXTURES / "difficulty-large-v1-action.json").read_text())
    action = action_from_payload(payload)
    assert json.loads(json.dumps(action_payload(action))) == payload
    assert action_id(action) == "9da36d7d06faa06e4ea033197de9f6b40d7a865effba38547f74cf5454a64789"
    manifest = tmp_path / (action_id(action) + ".json")
    manifest.write_text(json.dumps(payload))
    assert load_action(str(manifest)) == action


def test_v2_requeues_old_large_scores_and_measures_only_its_new_scores(tmp_path):
    old = historical_row()
    old_rows = [old, replace(old, task_id="b", task_sha256="b" * 64, attempt_id="old-large-b")]
    small_rows = [replace(row, solver=Solver.SMALL, attempt_id=row.attempt_id + "-small") for row in old_rows]
    append_rows(str(tmp_path), old_rows + small_rows)
    restored = read_rows(str(tmp_path))
    new_cell = cell_state(restored, SOURCE, Solver.LARGE, V2_POLICY)
    assert new_cell.state == CellStatus.NEEDS_ATTEMPTS and new_cell.n_missing == 2
    assert cell_state(restored, SOURCE, Solver.LARGE, CoveragePolicy()).state == CellStatus.MEASURED
    assert cell_state(restored, SOURCE, Solver.SMALL, V2_POLICY).state == CellStatus.MEASURED
    (action,) = plan([new_cell], restored)
    assert isinstance(action, AttemptBatch)
    assert action.solver_protocol_id == V2
    assert action.sources[0].tasks == (("a", "a" * 64, 1), ("b", "b" * 64, 1))
    (fixed,) = plan([cell_state([], SOURCE, Solver.LARGE, CoveragePolicy())], [])
    assert action_id(action) != action_id(fixed)
    manifest = tmp_path / (action_id(action) + ".json")
    manifest.write_text(json.dumps(action_payload(action)))
    assert load_action(str(manifest)) == action
    new_rows = [replace(row, solver_protocol_id=V2, attempt_id=row.attempt_id + "-v2", reward=0.0) for row in old_rows]
    append_rows(str(tmp_path), new_rows)
    restored = read_rows(str(tmp_path))
    assert cell_state(restored, SOURCE, Solver.LARGE, V2_POLICY).state == CellStatus.MEASURED
    cells = snapshot_cells(restored, [SOURCE], {"solver_protocols": {"large": V2}})
    measured = next(cell for cell in cells if cell["solver"] == "large")
    assert measured["protocol_id"] == V2 and measured["scored_attempts"] == 2
    assert measured["pass_rate"] == 0.0 and measured["mean_reward"] == 0.0
    report = status_report(restored, [SOURCE], V2_POLICY)
    assert report["solvers"]["large"]["scored_attempts_since"] == 2
    assert report["solvers"]["large"]["historical_scored_attempts_since"] == 2
    assert report["solvers"]["large"]["gpu_hours_since"] == 5.0


def test_v2_preserves_source_holds_and_source_gaps_without_old_budget_blocks():
    old = historical_row()
    budget = replace(old, outcome=Outcome.UNSCORED, reward=None, reason=GapReason.PROTOCOL_INPUT_BUDGET_EXCEEDED)
    assert cell_state([budget], SOURCE, Solver.LARGE, CoveragePolicy()).state == CellStatus.BLOCKED
    unblocked = cell_state([budget], SOURCE, Solver.LARGE, V2_POLICY)
    assert unblocked.state == CellStatus.NEEDS_ATTEMPTS and unblocked.n_missing == 2
    mixed = replace(budget, secondary_gaps=(Gap(GapReason.ENVIRONMENT_ARTIFACT_MISSING, artifact_uri="missing-lock"),))
    blocked = cell_state([mixed], SOURCE, Solver.LARGE, V2_POLICY)
    assert blocked.state == CellStatus.BLOCKED
    assert blocked.gaps == (Gap(GapReason.ENVIRONMENT_ARTIFACT_MISSING, artifact_uri="missing-lock"),)
    assert mixed.reason == GapReason.PROTOCOL_INPUT_BUDGET_EXCEEDED
    assert mixed.secondary_gaps == (Gap(GapReason.ENVIRONMENT_ARTIFACT_MISSING, artifact_uri="missing-lock"),)
    hold = replace(
        old,
        task_id=None,
        task_sha256=None,
        outcome=Outcome.OBSERVATION,
        reason=GapReason.EXECUTABILITY_ADMISSION_PENDING,
        reward=None,
    )
    assert cell_state([hold], SOURCE, Solver.LARGE, V2_POLICY).state == CellStatus.BLOCKED
    unsafe_score = replace(old, reason=GapReason.NETWORK_POLICY_VIOLATION)
    unsafe = cell_state([unsafe_score], SOURCE, Solver.LARGE, V2_POLICY)
    assert unsafe.state == CellStatus.BLOCKED and unsafe.n_missing == 2
    assert unsafe.gaps == (Gap(GapReason.NETWORK_POLICY_VIOLATION),)
    assert unsafe_score.outcome == Outcome.SCORED and unsafe_score.reward == 1.0


def test_v1_inflight_hold_drains_before_v2_submission_and_restart_does_not_duplicate(tmp_path):
    ledger = str(tmp_path / "ledger")
    (old_action,) = plan([cell_state([], SOURCE, Solver.LARGE, CoveragePolicy())], [])
    old_submit = action_rows(old_action, [SOURCE], Outcome.SUBMITTED, "/old/child", "2026-11-26T00:00:00+00:00")
    append_rows(ledger, old_submit)
    restored = read_rows(ledger)
    assert plan([cell_state(restored, SOURCE, Solver.LARGE, V2_POLICY)], restored) == []
    append_rows(ledger, action_rows(old_action, [SOURCE], Outcome.FINISHED, "/old/child", "2026-11-26T01:00:00+00:00"))
    restored = read_rows(ledger)
    actions = plan([cell_state(restored, SOURCE, Solver.LARGE, V2_POLICY)], restored)
    backend = JobStore(str(tmp_path / "jobs"))
    first = execute_plan(actions, restored, [SOURCE], ledger, backend, EscalationLog(ledger))
    restored = read_rows(ledger)
    second = execute_plan(actions, restored, [SOURCE], ledger, backend, EscalationLog(ledger))
    assert len(first["submitted"]) == 1 and second["submitted"] == []
    assert len(list((tmp_path / "jobs").glob("*.json"))) == 1
    assert next(row for row in restored if row.action_id == action_id(actions[0])).solver_protocol_id == V2
    backend.complete_repair(actions[0], {"status": "completed"}, ledger)
    finished = [
        row for row in read_rows(ledger) if row.outcome == Outcome.FINISHED and row.action_id == action_id(actions[0])
    ]
    assert len(finished) == 1 and finished[0].solver_protocol_id == V2


def test_supersession_appends_links_without_rewriting_grade_or_double_counting_cost(tmp_path):
    old = historical_row()
    unscored = replace(
        old,
        task_id="b",
        task_sha256="b" * 64,
        attempt_id="old-unscored-b",
        outcome=Outcome.UNSCORED,
        reward=None,
        reason=GapReason.GRADER_INFRASTRUCTURE_FAILURE,
    )
    other = replace(old, solver=Solver.SMALL, attempt_id="old-small-a")
    original = [old, unscored, other]
    partition = append_rows(str(tmp_path), original)
    assert partition is not None
    wire = Path(partition).read_bytes()
    supersessions = partial(
        protocol_supersessions,
        solver=Solver.LARGE,
        previous_protocol_id=PROTOCOL_ID,
        replacement_protocol_id=V2,
        evidence_uri="fixture:qualified-large-v2",
        activated_at="2026-11-27T14:00:00+00:00",
    )
    additions = supersessions(read_rows(str(tmp_path)))
    assert len(additions) == 2
    assert {row.supersedes_attempt_id for row in additions} == {old.attempt_id, unscored.attempt_id}
    assert all(
        row.outcome == Outcome.PROTOCOL_SUPERSEDED and row.reward is None and row.gpu_hours == 0 for row in additions
    )
    assert additions == supersessions(read_rows(str(tmp_path)))
    target = append_rows(str(tmp_path), additions)
    assert append_rows(str(tmp_path), additions) == target
    restored = read_rows(str(tmp_path))
    assert all(row in restored for row in original)
    assert Path(partition).read_bytes() == wire
    assert sum(row.gpu_hours for row in restored) == sum(row.gpu_hours for row in original)
    assert supersessions(restored) == []
    assert cell_state(restored, SOURCE, Solver.LARGE, V2_POLICY).n_missing == 2
