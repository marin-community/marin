# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Coverage transitions, typed remediation and immutable receipt import."""

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime

import pytest
from finestore.rl.difficulty import Solver

from experiments.post_training.glm import GLM_MODEL
from experiments.post_training.task_curation.difficulty.accounting import GpuAccounting
from experiments.post_training.task_curation.difficulty.importer import import_runs, population_revision, receipt_row
from experiments.post_training.task_curation.difficulty.ledger import append_rows, read_rows
from experiments.post_training.task_curation.difficulty.legacy import legacy_gaps
from experiments.post_training.task_curation.difficulty.planner import (
    AttemptBatch,
    BuildSnapshot,
    BuildTaskSpecFactory,
    Escalate,
    QualifyJudgeRoute,
    RebuildEnvironmentArtifact,
    RetryAfter,
    action_id,
    plan,
)
from experiments.post_training.task_curation.difficulty.report import status_report
from experiments.post_training.task_curation.difficulty.state import (
    ArtifactKind,
    CatalogSource,
    CellStatus,
    CoveragePolicy,
    Gap,
    GapReason,
    LedgerRow,
    Outcome,
    TerminalRuling,
    cell_state,
)

SOURCE = CatalogSource("source", "population", (("a", "a" * 64), ("b", "b" * 64)), "queue", "q" * 64)


def row(attempt: str, task: str = "a", outcome: Outcome = Outcome.SCORED, **kwargs) -> LedgerRow:
    return LedgerRow(
        "population",
        "source",
        task * 64,
        Solver.SMALL,
        attempt,
        outcome,
        None,
        "run",
        0.0,
        "2026-11-20T00:00:00+00:00",
        task_id=task,
        reward=1.0 if outcome == Outcome.SCORED else None,
        **kwargs,
    )


def blocked(gap: Gap):
    observed = replace(
        row("gap", outcome=Outcome.OBSERVATION),
        reason=gap.reason,
        artifact_uri=gap.artifact_uri,
        judge_model=gap.judge_model,
        capability=gap.capability,
        artifact_kind=gap.artifact_kind,
        judge_temperature=gap.judge_temperature,
    )
    return cell_state([observed], SOURCE, Solver.SMALL, CoveragePolicy())


def test_coverage_counts_unique_receipts_and_current_population_only():
    first = row("attempt-a")
    wait = replace(row("wait", "b", Outcome.UNSCORED), reason=GapReason.PROVIDER_CAPACITY_WAIT)
    old = replace(row("old-b", "b"), registry_revision="old")
    state = cell_state([first, first, wait, old], SOURCE, Solver.SMALL, CoveragePolicy())
    assert state.n_missing == 1
    assert state.state == CellStatus.NEEDS_ATTEMPTS
    assert (
        cell_state([first, row("attempt-b", "b")], SOURCE, Solver.SMALL, CoveragePolicy()).state == CellStatus.MEASURED
    )
    changed = replace(SOURCE, tasks=(("a", "c" * 64), ("b", "b" * 64)))
    assert cell_state([first, row("attempt-b", "b")], changed, Solver.SMALL, CoveragePolicy()).n_missing == 1


def test_completed_repair_preserves_other_causes_and_newer_failures():
    original = replace(
        row("original", outcome=Outcome.OBSERVATION),
        task_id=None,
        task_sha256=None,
        reason=GapReason.DAYTONA_SNAPSHOT_PREFLIGHT_PENDING,
        secondary_gaps=(Gap(GapReason.NETWORK_POLICY_VIOLATION),),
    )
    repaired = replace(
        original,
        attempt_id="repair",
        outcome=Outcome.REPAIRED,
        secondary_gaps=(),
        recorded_at="2026-11-20T01:00:00+00:00",
    )
    state = cell_state([original, repaired], SOURCE, Solver.SMALL, CoveragePolicy())
    assert state.state == CellStatus.BLOCKED
    assert state.gaps == (Gap(GapReason.NETWORK_POLICY_VIOLATION),)
    newer = replace(original, attempt_id="newer", recorded_at="2026-11-20T02:00:00+00:00")
    assert (
        Gap(GapReason.DAYTONA_SNAPSHOT_PREFLIGHT_PENDING)
        in cell_state([original, repaired, newer], SOURCE, Solver.SMALL, CoveragePolicy()).gaps
    )


def test_network_policy_reason_blocks_a_legacy_measured_source_and_rulings_are_scoped():
    observation = replace(
        row("policy", outcome=Outcome.OBSERVATION), reason=GapReason.NETWORK_POLICY_VIOLATION, legacy_status="measured"
    )
    rows = [row("a"), row("b", "b"), observation]
    assert cell_state(rows, SOURCE, Solver.SMALL, CoveragePolicy()).state == CellStatus.BLOCKED
    ruling = TerminalRuling("source", Solver.SMALL, GapReason.NETWORK_POLICY_VIOLATION, "population", "ruling-uri")
    assert (
        cell_state(rows, SOURCE, Solver.SMALL, CoveragePolicy(terminal_rulings=(ruling,))).state == CellStatus.TERMINAL
    )
    assert (
        cell_state(rows, SOURCE, Solver.LARGE, CoveragePolicy(terminal_rulings=(ruling,))).state != CellStatus.TERMINAL
    )


def test_planner_schedules_only_missing_tasks_and_replay_submits_nothing(tmp_path):
    rows = [row("done-a")]
    cell = cell_state(rows, SOURCE, Solver.SMALL, CoveragePolicy())
    actions = plan([cell], rows)
    assert len(actions) == 1 and isinstance(actions[0], AttemptBatch)
    assert actions[0].sources[0].tasks == (("b", "b" * 64, 1),)
    submission = replace(row("submission", outcome=Outcome.SUBMITTED), action_id=action_id(actions[0]), job_id="/job")
    append_rows(str(tmp_path), [*rows, submission])
    restored = read_rows(str(tmp_path))
    assert plan([cell_state(restored, SOURCE, Solver.SMALL, CoveragePolicy())], restored) == []
    assert plan([cell], restored) == []


@pytest.mark.parametrize("solver", [Solver.SMALL, Solver.SNOWBALL])
def test_snowball_release_binding_allows_fixed_batch_after_recorded_legacy_failure(solver):
    cell = cell_state([], SOURCE, solver, CoveragePolicy())
    (current,) = plan([cell], [])
    legacy = replace(current, vllm_release_tag=None)
    failed = replace(
        row("failed", outcome=Outcome.FINISHED),
        solver=solver,
        task_id=None,
        task_sha256=None,
        action_id=action_id(legacy),
        job_id="/fixture/failed",
    )
    next_actions = plan([cell_state([failed], SOURCE, solver, CoveragePolicy())], [failed])
    if solver == Solver.SNOWBALL:
        assert next_actions == [current] and action_id(current) != action_id(legacy)
    else:
        assert next_actions == []


def test_provider_capacity_recovers_without_counting_waits_as_attempts():
    cell = blocked(Gap(GapReason.PROVIDER_CAPACITY_WAIT))
    assert isinstance(plan([cell], [])[0], RetryAfter)
    assert isinstance(plan([cell], [], frozenset({Solver.SMALL}))[0], AttemptBatch)
    compound = replace(cell, gaps=(Gap(GapReason.PROTOCOL_INPUT_BUDGET_EXCEEDED), *cell.gaps))
    assert all(isinstance(action, Escalate) for action in plan([compound], [], frozenset({Solver.SMALL})))


def test_artifact_planner_rechecks_stale_locks_and_deduplicates_shared_rebuilds():
    uri = "s3://bucket/requirements.lock"
    cell = blocked(Gap(GapReason.ENVIRONMENT_ARTIFACT_MISSING, uri, artifact_kind=ArtifactKind.REQUIREMENTS_LOCK))
    other = replace(cell, solver=Solver.LARGE)
    actions = plan([cell, other], [])
    assert len(actions) == 1 and isinstance(actions[0], RebuildEnvironmentArtifact)
    assert len(actions[0].cells) == 2
    recovered = plan([cell], [], available_artifacts=frozenset({uri}))
    assert len(recovered) == 1 and isinstance(recovered[0], AttemptBatch)
    image = blocked(Gap(GapReason.ENVIRONMENT_ARTIFACT_MISSING, artifact_kind=ArtifactKind.PREPARED_IMAGE))
    assert isinstance(plan([image], [])[0], Escalate)


def test_snapshot_and_factory_plans_keep_the_exact_source_population():
    snapshot = plan([blocked(Gap(GapReason.DAYTONA_SNAPSHOT_PREFLIGHT_PENDING))], [])[0]
    factory = plan([blocked(Gap(GapReason.TASK_SPEC_FACTORY_PENDING))], [])[0]
    assert isinstance(snapshot, BuildSnapshot) and snapshot.source.registry_revision == SOURCE.registry_revision
    assert isinstance(factory, BuildTaskSpecFactory) and factory.source.tasks == (("a", "a" * 64, 1), ("b", "b" * 64, 1))


def test_judge_qualification_groups_shared_models_without_crossing_routes():
    cell = blocked(Gap(GapReason.JUDGE_ROUTE_UNQUALIFIED, judge_model="judge-a"))
    other = replace(cell, solver=Solver.LARGE)
    different = replace(cell, gaps=(Gap(GapReason.JUDGE_ROUTE_UNQUALIFIED, judge_model="judge-b"),))
    actions = plan([cell, other, different], [])
    qualifications = {action.model: action for action in actions if isinstance(action, QualifyJudgeRoute)}
    assert len(qualifications["judge-a"].cells) == 2
    assert len(qualifications["judge-b"].cells) == 1


@pytest.mark.parametrize(
    "reason",
    [
        GapReason.PROTOCOL_INPUT_BUDGET_EXCEEDED,
        GapReason.NETWORK_POLICY_VIOLATION,
        GapReason.UNSUPPORTED_COMPOSE_DOCUMENT,
        GapReason.EXECUTABILITY_ADMISSION_PENDING,
    ],
)
def test_escalations_leave_other_cells_schedulable(reason):
    gap = blocked(Gap(reason))
    missing = cell_state([], replace(SOURCE, source_id="other"), Solver.SMALL, CoveragePolicy())
    actions = plan([gap, missing], [])
    assert any(isinstance(action, Escalate) and action.reason == reason for action in actions)
    assert any(isinstance(action, AttemptBatch) and action.sources[0].source_id == "other" for action in actions)


def test_legacy_compound_reasons_keep_secondary_causes_and_relocate_local_locks():
    gaps = legacy_gaps("protocol_input_budget_exceeded; provider_capacity_wait")
    assert [gap.reason for gap in gaps] == [GapReason.PROTOCOL_INPUT_BUDGET_EXCEEDED, GapReason.PROVIDER_CAPACITY_WAIT]
    local = legacy_gaps("missing_environment_artifact:/old/images/env-278169f50750da36/2026.10.08/requirements.lock")[0]
    assert local.artifact_uri == "s3://marin-us-east-02a/marin/images/env-278169f50750da36/2026.10.08/requirements.lock"
    assert legacy_gaps("environment_artifact_missing:prepared_image")[0].artifact_kind == ArtifactKind.PREPARED_IMAGE
    with pytest.raises(ValueError, match="Unmapped difficulty gap"):
        legacy_gaps("awaiting_scored_coverage; invented_failure")


@pytest.mark.parametrize("indexed_parent", [True, False])
def test_import_replays_immutable_attempts_once_and_subtracts_inherited_cost(tmp_path, indexed_parent):
    parent = tmp_path / "parent"
    child = tmp_path / "child"
    sample = {
        "source_id": "source",
        "release": {"revision": "release"},
        "converter_version": "v1",
        "converter_revision": "converter",
        "task_specs_sha256": "p" * 64,
        "tasks": [{"task_id": "a", "task_sha256": "a" * 64}],
    }
    source_key = hashlib.sha256(b"source").hexdigest()
    for root, cost, measurement_id in [(parent, 2.0, "parent"), (child, 3.0, "child")]:
        trial = root / "trials" / source_key
        trial.mkdir(parents=True)
        (root / "prepared.json").write_text(
            json.dumps({"sources": [{"sample": sample}], "source_queue_sha256": "queue-sha"})
        )
        (root / "serving.json").write_text(json.dumps({"solver": "small"}))
        (trial / "measurement.json").write_text(
            json.dumps(
                {
                    "measurement_id": measurement_id,
                    "created_at": "2026-11-20T00:00:00+00:00",
                    "gpu_hours": cost,
                    "status": "measured",
                    "reason": None,
                }
            )
        )
    attempt = parent / "trials" / source_key / "attempts" / "task" / "00" / "attempt.json"
    attempt.parent.mkdir(parents=True)
    attempt.write_text(
        json.dumps(
            {"task_id": "a", "status": "scored", "reward": 1.0, "result": {"finished_at": "2026-11-19T23:00:00+00:00"}}
        )
    )
    reference = {"uri": str(attempt), "sha256": hashlib.sha256(attempt.read_bytes()).hexdigest()}
    for root in (parent, child):
        aggregate = (
            {"attempt_receipts": [reference]}
            if root == child or indexed_parent
            else {"attempts": [{"task_id": "a", "status": "scored", "reward": 1.0}]}
        )
        (root / "trials" / source_key / "attempts.json").write_text(json.dumps(aggregate))
    (child / "trials" / "parent-campaign.json").write_text(json.dumps({"root": str(parent)}))
    ledger = str(tmp_path / "ledger")
    import_runs([str(child)], ledger)
    before = read_rows(ledger)
    import_runs([str(child)], ledger)
    after = read_rows(ledger)
    assert before == after
    assert sum(row.outcome == Outcome.SCORED for row in after) == 1
    assert sum(row.gpu_hours for row in after) == 3.0
    source = replace(SOURCE, tasks=(("a", "a" * 64),), registry_revision=population_revision(sample))
    assert cell_state(after, source, Solver.SMALL, CoveragePolicy()).state == CellStatus.MEASURED


def test_zero_token_budget_receipt_stays_typed_without_a_measurement_aggregate(tmp_path):
    attempt = tmp_path / "attempt.json"
    attempt.write_text(
        json.dumps(
            {
                "task_id": "a",
                "status": "pending",
                "reward": None,
                "reason": "protocol_input_budget_exceeded",
                "result": {
                    "finished_at": "2026-11-20T10:25:00+00:00",
                    "agent_result": {"n_input_tokens": 0, "n_output_tokens": 0},
                },
            }
        )
    )
    sample = {
        "source_id": "source",
        "release": {"revision": "release"},
        "converter_version": "v1",
        "converter_revision": "converter",
        "task_specs_sha256": "p" * 64,
        "tasks": [{"task_id": "a", "task_sha256": "a" * 64}],
    }
    imported = receipt_row(
        {"uri": str(attempt), "sha256": hashlib.sha256(attempt.read_bytes()).hexdigest()},
        sample,
        Solver.LARGE,
        "run",
        "2026-11-20T10:25:00+00:00",
    )
    append_rows(str(tmp_path / "ledger"), [imported])
    restored = read_rows(str(tmp_path / "ledger"))
    assert restored[0].reason == GapReason.PROTOCOL_INPUT_BUDGET_EXCEEDED
    assert restored[0].outcome == Outcome.UNSCORED and restored[0].reward is None
    source = replace(SOURCE, registry_revision=population_revision(sample), tasks=(("a", "a" * 64),))
    assert cell_state(restored, source, Solver.LARGE, CoveragePolicy()).state == CellStatus.BLOCKED


def test_gpu_accounting_conserves_overlapping_task_and_idle_windows():
    now = [0.0]
    meter = GpuAccounting(8, lambda: now[0])
    meter.advance()
    now[0] = 3600
    meter.start("a")
    now[0] = 7200
    meter.start("b")
    now[0] = 10800
    a = meter.finish("a")
    now[0] = 14400
    b = meter.finish("b")
    assert a == 12.0 and b == 12.0
    assert meter.idle_gpu_hours() == 8.0
    assert a + b + meter.idle_gpu_hours() == 32.0


def test_a_later_provider_wait_does_not_hide_another_tasks_network_violation():
    violation = replace(row("policy", "a", Outcome.UNSCORED), reason=GapReason.NETWORK_POLICY_VIOLATION)
    wait = replace(
        row("wait", "b", Outcome.UNSCORED),
        reason=GapReason.PROVIDER_CAPACITY_WAIT,
        recorded_at="2026-11-20T01:00:00+00:00",
    )
    cell = cell_state([violation, wait], SOURCE, Solver.SMALL, CoveragePolicy())
    assert cell.state == CellStatus.BLOCKED
    assert {gap.reason for gap in cell.gaps} == {GapReason.NETWORK_POLICY_VIOLATION, GapReason.PROVIDER_CAPACITY_WAIT}
    assert all(isinstance(action, Escalate) for action in plan([cell], []))


def test_since_filters_progress_and_cost_without_resetting_coverage():
    first = row("old-score", "a")
    second = replace(row("new-score", "b"), recorded_at="2026-11-20T02:00:00+00:00", gpu_hours=2.0)
    cost = replace(row("old-cost", outcome=Outcome.COST), gpu_hours=10.0)
    report = status_report(
        [first, second, cost], [SOURCE], CoveragePolicy(), since=datetime(2026, 11, 20, 1, tzinfo=UTC)
    )
    small = report["solvers"]["small"]
    assert small["cells"] == {"MEASURED": 1}
    assert small["gpu_hours_since"] == 2.0 and small["scored_attempts_since"] == 1


@pytest.mark.parametrize("model", ["", "gpt-4o-mini", "openai/gpt-oss-120b"])
def test_legacy_judge_temperature_family_keeps_model_and_temperature(model):
    reason = f"production_judge_route_or_temperature_not_qualified:{model}:temperature=0.0"
    (gap,) = legacy_gaps(reason)
    assert gap.reason == GapReason.JUDGE_ROUTE_UNQUALIFIED
    assert gap.judge_model == (model or GLM_MODEL) and gap.judge_temperature == 0.0
    cell = blocked(gap)
    (action,) = plan([cell], [])
    assert isinstance(action, QualifyJudgeRoute) and action.requested_temperature == 0.0


def test_legacy_short_capacity_message_remains_outside_scored_attempts():
    (gap,) = legacy_gaps("Awaiting shared model provider capacity")
    cell = blocked(gap)
    assert cell.n_missing == 2 and isinstance(plan([cell], [])[0], RetryAfter)


def test_legacy_compound_split_preserves_the_registered_capacity_sentence():
    reason = "Awaiting shared model provider capacity; provider waits are excluded from task attempts."
    assert legacy_gaps(reason) == (Gap(GapReason.PROVIDER_CAPACITY_WAIT),)
    combined = legacy_gaps(reason + "; protocol_input_budget_exceeded")
    assert [gap.reason for gap in combined] == [
        GapReason.PROTOCOL_INPUT_BUDGET_EXCEEDED,
        GapReason.PROVIDER_CAPACITY_WAIT,
    ]
