# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submission replay, quota admission and completed repair transitions."""

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from threading import Event

import pytest
from finestore.rl.difficulty import Solver
from iris.resources.state import JobState
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.difficulty.budget import campaign_budget
from experiments.post_training.task_curation.difficulty.campaign import identity_bundle
from experiments.post_training.task_curation.difficulty.executor import (
    ATTEMPT_JOBS_PER_SOLVER,
    CONCURRENT_JOB_CAP,
    EscalationLog,
    Job,
    action_job_slots,
    collect_finished,
    execute_plan,
    reconcile_once,
)
from experiments.post_training.task_curation.difficulty.ledger import (
    append_rows,
    digest,
    read_rows,
    record_action_failure,
)
from experiments.post_training.task_curation.difficulty.planner import (
    AttemptBatch,
    RebuildEnvironmentArtifact,
    SourceAttempts,
    action_id,
    plan,
)
from experiments.post_training.task_curation.difficulty.state import (
    ActionFailure,
    CellStatus,
    CoveragePolicy,
    Gap,
    GapReason,
    Outcome,
    cell_state,
)
from experiments.post_training.task_curation.tests.test_difficulty_reconciler import SOURCE, blocked, row


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
        job = Job(
            "/owner/controller/difficulty-reconcile-" + key[:24],
            JobState.RUNNING,
            action.solver if isinstance(action, AttemptBatch) else None,
            action_job_slots(action),
        )
        self.existing.append(job)
        return job.job_id

    def complete_repair(self, action, result: dict, ledger: str) -> StoragePath:
        key = action_id(action)
        target = StoragePath(ledger) / "v1" / "results" / (key + ".json")
        target.parent.mkdirs()
        target.write_text(json.dumps(result))
        self.existing = [
            replace(job, state=JobState.SUCCEEDED) if job.job_id.endswith(key[:24]) else job for job in self.existing
        ]
        collect_finished(read_rows(ledger), self.jobs(), ledger, EscalationLog(ledger))
        return target


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
    full = JobStore(str(tmp_path / "jobs"), [Job("/busy", JobState.RUNNING, None, CONCURRENT_JOB_CAP)])
    result = execute_plan(actions, [], [SOURCE], ledger, full, EscalationLog(ledger))
    assert result["submitted"] == [] and result["deferred"] == [action_id(actions[0])]
    legacy = JobStore(str(tmp_path / "jobs"), [Job("/owner/atlas-difficulty-small-v24", JobState.RUNNING, Solver.SMALL)])
    assert execute_plan(actions, [], [SOURCE], ledger, legacy, EscalationLog(ledger))["submitted"] == []
    assert read_rows(ledger) == []


def test_shared_repair_completion_unblocks_only_the_original_population(tmp_path):
    ledger = str(tmp_path / "ledger")
    cell = blocked(Gap(GapReason.DAYTONA_SNAPSHOT_PREFLIGHT_PENDING))
    actions = plan([cell], [])
    backend = JobStore(str(tmp_path / "jobs"))
    execute_plan(actions, [], [SOURCE], ledger, backend, EscalationLog(ledger))
    backend.existing = [replace(backend.existing[0], state=JobState.SUCCEEDED)]
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
    backend.existing = [replace(backend.existing[0], state=JobState.SUCCEEDED)]
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
    escalation = (tmp_path / "ledger" / "ESCALATIONS.md").read_text()
    assert "Original frozen panel is not available" in escalation and str(result) in escalation
    other = replace(SOURCE, source_id="other")
    next_actions = plan([cell, cell_state([], other, Solver.SMALL, CoveragePolicy())], rows)
    assert all(action_id(action) != action_id(actions[0]) for action in next_actions)
    assert any(action.sources[0].source_id == "other" for action in next_actions)


def test_budget_counts_recorded_spend_and_inflight_reservations_across_restart(tmp_path):
    ledger = str(tmp_path / "ledger")
    budget = campaign_budget(ledger, 800)
    # Prior campaign costs do not consume this newly approved budget.
    historical = replace(row("old-cost", outcome=Outcome.COST), gpu_hours=1000)
    current = replace(row("new-cost", outcome=Outcome.COST), gpu_hours=760, recorded_at=datetime.now(UTC).isoformat())
    append_rows(ledger, [historical, current])
    other = replace(SOURCE, source_id="other")
    cells = [cell_state([], source, Solver.SMALL, CoveragePolicy()) for source in (SOURCE, other)]
    actions = plan(cells, [])
    backend = JobStore(str(tmp_path / "jobs"))
    first = execute_plan(actions, read_rows(ledger), [SOURCE, other], ledger, backend, EscalationLog(ledger), budget)
    assert len(first["submitted"]) == 1
    assert first["recorded_gpu_hours"] == 760 and first["reserved_gpu_hours"] == 40
    restarted = campaign_budget(ledger, 800)
    second = execute_plan(actions, read_rows(ledger), [SOURCE, other], ledger, backend, EscalationLog(ledger), restarted)
    assert second["submitted"] == [] and second["reserved_gpu_hours"] == 40
    assert restarted == budget
    escalations = list((tmp_path / "ledger" / "v1" / "escalations").glob("*.json"))
    assert len(escalations) == 1


def test_repair_backlog_preserves_attempt_lanes_for_each_solver(tmp_path):
    ledger = str(tmp_path / "ledger")
    sources = [replace(SOURCE, source_id=f"source-{index}") for index in range(9)]
    repairs = [
        RebuildEnvironmentArtifact(f"s3://fixtures/environment-{index}", ((source.source_id, Solver.SMALL),), ())
        for index, source in enumerate(sources)
    ]
    attempts = [
        AttemptBatch(
            solver,
            (SourceAttempts(source.source_id, source.registry_revision, (), "queue", "sha", ()),),
            1,
        )
        for solver in Solver
        for source in sources[: ATTEMPT_JOBS_PER_SOLVER[solver]]
    ]
    backend = JobStore(str(tmp_path / "jobs"))
    result = execute_plan(repairs + attempts, [], sources, ledger, backend, EscalationLog(ledger))
    accepted = {path.stem for path in (tmp_path / "jobs").glob("*.json")}
    assert accepted == {action_id(action) for action in repairs[:8] + attempts}
    assert result["deferred"] == [action_id(repairs[8])]
    assert result["active_job_slots"] == 64


@pytest.mark.parametrize(
    "reason",
    [GapReason.SERVING_RELEASE_MISMATCH, GapReason.SERVING_PROBE_MISMATCH, GapReason.SOURCE_QUEUE_VALIDATION_FAILED],
)
def test_failed_child_preserves_specific_runtime_cause_across_subprocess_boundary(tmp_path, reason):
    ledger = str(tmp_path / "ledger")
    actions = plan([cell_state([], SOURCE, Solver.SMALL, CoveragePolicy())], [])
    backend = JobStore(str(tmp_path / "jobs"))
    execute_plan(actions, [], [SOURCE], ledger, backend, EscalationLog(ledger))
    manifest = str(tmp_path / "ledger" / "v1" / "actions" / (action_id(actions[0]) + ".json"))
    record_action_failure(ledger, manifest, ActionFailure(reason, "original typed failure"))
    record_action_failure(ledger, manifest, subprocess.CalledProcessError(1, ["python", "runtime"]))
    backend.existing = [replace(backend.existing[0], state=JobState.FAILED)]
    additions = collect_finished(read_rows(ledger), backend.jobs(), ledger, EscalationLog(ledger))
    (failure,) = [row for row in additions if row.outcome == Outcome.ESCALATED]
    assert (failure.reason, failure.capability, failure.error_identity) == (
        reason,
        "original typed failure",
        "ActionFailure",
    )
    assert any(row.outcome == Outcome.FINISHED for row in additions)


@pytest.mark.parametrize(
    "state,reason",
    [(JobState.KILLED, GapReason.CONTROLLER_ACTION_CANCELLED), (JobState.FAILED, GapReason.CONTROLLER_ACTION_FAILED)],
)
def test_unreceipted_child_failure_never_claims_an_admission_receipt_is_missing(tmp_path, state, reason):
    ledger = str(tmp_path / "ledger")
    actions = plan([cell_state([], SOURCE, Solver.SMALL, CoveragePolicy())], [])
    backend = JobStore(str(tmp_path / "jobs"))
    execute_plan(actions, [], [SOURCE], ledger, backend, EscalationLog(ledger))
    backend.existing = [replace(backend.existing[0], state=state)]
    additions = collect_finished(read_rows(ledger), backend.jobs(), ledger, EscalationLog(ledger))
    (failure,) = [row for row in additions if row.outcome == Outcome.ESCALATED]
    assert failure.reason == reason
    assert failure.error_identity == f"iris_child_{state}"


def test_tpu_reconcile_uses_only_qualified_pools_and_admits_later_large(tmp_path):
    ledger = str(tmp_path / "ledger")
    pools = tmp_path / "pools"
    pools.mkdir()
    version = tmp_path / "ledger/v1"
    (version / "imports").mkdir(parents=True)
    (version / "imports/fixture.json").write_text("{}")
    identity = identity_bundle()
    (version / "campaign.json").write_text(json.dumps({"identity": identity, "campaign_id": digest(identity)}))
    for solver, chips in ((Solver.SMALL, 64), (Solver.SNOWBALL, 64)):
        (pools / (solver.value + ".json")).write_text(
            json.dumps(
                {
                    "platform": "tpu",
                    "chip_count": chips,
                    "gpu_count": 0,
                    "revision": "revision",
                    "ready_at": "2026-01-01T00:00:00+00:00",
                }
            )
        )
    backend = JobStore(str(tmp_path / "jobs"))
    ready = frozenset({Solver.SMALL, Solver.SNOWBALL, Solver.HOSTED})
    reconcile_once(
        ledger, [SOURCE], CoveragePolicy(), backend, EscalationLog(ledger), ready, frozenset(), None, str(pools)
    )
    submitted = [json.loads(p.read_text())["solver"] for p in (tmp_path / "jobs").glob("*.json")]
    assert set(submitted) == {solver.value for solver in ready}
    assert {row.solver for row in read_rows(ledger) if row.outcome == Outcome.COST} == {Solver.SMALL, Solver.SNOWBALL}
    (pools / "large.json").write_text(
        json.dumps(
            {
                "platform": "tpu",
                "chip_count": 128,
                "gpu_count": 0,
                "revision": "revision",
                "ready_at": "2026-01-01T00:00:00+00:00",
            }
        )
    )
    reconcile_once(
        ledger,
        [SOURCE],
        CoveragePolicy(),
        backend,
        EscalationLog(ledger),
        frozenset(Solver),
        frozenset(),
        None,
        str(pools),
    )
    submitted = [json.loads(p.read_text())["solver"] for p in (tmp_path / "jobs").glob("*.json")]
    assert len(submitted) == 4 and set(submitted) == {solver.value for solver in Solver}


def test_successful_child_without_campaign_result_does_not_repair_or_crash(tmp_path):
    ledger = str(tmp_path / "ledger")
    backend = JobStore(str(tmp_path / "jobs"))
    action = plan([blocked(Gap(GapReason.DAYTONA_SNAPSHOT_PREFLIGHT_PENDING))], [])[0]
    execute_plan([action], [], [SOURCE], ledger, backend, EscalationLog(ledger))
    backend.existing = [replace(job, state=JobState.SUCCEEDED) for job in backend.existing]
    additions = collect_finished(read_rows(ledger), backend.jobs(), ledger, EscalationLog(ledger))
    assert {row.outcome for row in additions} == {Outcome.FINISHED, Outcome.ESCALATED}
    failure = next(row for row in additions if row.outcome == Outcome.ESCALATED)
    assert failure.reason == GapReason.CONTROLLER_ACTION_FAILED
    assert failure.error_identity == "completed_job_missing_campaign_result"
    assert collect_finished(read_rows(ledger), backend.jobs(), ledger, EscalationLog(ledger)) == []


def test_tpu_batches_fill_shared_pools_without_exceeding_attempt_or_repair_slots(tmp_path):
    ledger = str(tmp_path / "ledger")
    version = tmp_path / "ledger/v1"
    version.mkdir(parents=True)
    identity = identity_bundle()
    (version / "campaign.json").write_text(json.dumps({"identity": identity, "campaign_id": digest(identity)}))
    sources = [replace(SOURCE, source_id=f"source-{i}") for i in range(15)]
    repair_sources = [replace(SOURCE, source_id=f"repair-{i}") for i in range(9)]
    repairs = [
        RebuildEnvironmentArtifact(f"s3://fixtures/environment-{i}", ((source.source_id, Solver.SMALL),), ())
        for i, source in enumerate(repair_sources)
    ]
    actions = repairs + plan(
        [cell_state([], source, solver, CoveragePolicy()) for source in sources for solver in Solver], []
    )
    backend = JobStore(str(tmp_path / "jobs"))
    catalog = sources + repair_sources
    result = execute_plan(actions, [], catalog, ledger, backend, EscalationLog(ledger))
    accepted = [
        payload["solver"]
        for path in (tmp_path / "jobs").glob("*.json")
        if (payload := json.loads(path.read_text()))["kind"] == "AttemptBatch"
    ]
    assert {solver: accepted.count(solver) for solver in ("small", "large", "snowball", "hosted")} == {
        "small": 7,
        "large": 14,
        "snowball": 7,
        "hosted": 4,
    }
    assert len(accepted) * 4 == 128
    assert len(result["submitted"]) == 40
    assert result["active_job_slots"] == 136
    assert action_id(repairs[-1]) in result["deferred"]
    restarted = execute_plan(actions, read_rows(ledger), catalog, ledger, backend, EscalationLog(ledger))
    assert restarted["submitted"] == []
    assert restarted["active_job_slots"] == 136


@pytest.mark.parametrize(
    ("ready", "expected", "recovering"),
    [
        (
            frozenset({Solver.SMALL, Solver.SNOWBALL, Solver.HOSTED}),
            {Solver.SMALL: 14, Solver.SNOWBALL: 14, Solver.HOSTED: 4},
            Solver.LARGE,
        ),
        (frozenset({Solver.SMALL, Solver.HOSTED}), {Solver.SMALL: 28, Solver.HOSTED: 4}, Solver.LARGE),
        (
            frozenset({Solver.LARGE, Solver.SMALL, Solver.HOSTED}),
            {Solver.LARGE: 19, Solver.SMALL: 9, Solver.HOSTED: 4},
            Solver.SNOWBALL,
        ),
    ],
)
def test_tpu_reconcile_lends_unavailable_pool_slots_without_changing_grader_ceiling(
    tmp_path, ready, expected, recovering
):
    ledger = str(tmp_path / "ledger")
    version = tmp_path / "ledger/v1"
    (version / "imports").mkdir(parents=True)
    (version / "imports/fixture.json").write_text("{}")
    identity = identity_bundle()
    (version / "campaign.json").write_text(json.dumps({"identity": identity, "campaign_id": digest(identity)}))
    pools = tmp_path / "pools"
    pools.mkdir()
    for solver in ready - {Solver.HOSTED}:
        (pools / (solver.value + ".json")).write_text(
            json.dumps(
                {
                    "platform": "tpu",
                    "chip_count": 128 if solver == Solver.LARGE else 64,
                    "gpu_count": 0,
                    "revision": "revision",
                    "ready_at": "2026-01-01T00:00:00+00:00",
                }
            )
        )
    sources = [replace(SOURCE, source_id=f"source-{i}") for i in range(30)]
    backend = JobStore(str(tmp_path / "jobs"))
    result = reconcile_once(
        ledger, sources, CoveragePolicy(), backend, EscalationLog(ledger), ready, frozenset(), None, str(pools)
    )
    assert {solver: sum(job.solver == solver for job in backend.jobs()) for solver in ready} == expected
    assert all(job.solver in ready for job in backend.jobs())
    assert sum(job.slots for job in backend.jobs()) == 128
    assert result["active_attempt_slots"] == 128
    # Readiness recovery changes admission, while existing borrowed batches keep running.
    (pools / (recovering.value + ".json")).write_text(
        json.dumps(
            {
                "platform": "tpu",
                "chip_count": 128 if recovering == Solver.LARGE else 64,
                "gpu_count": 0,
                "revision": "revision",
                "ready_at": "2026-01-01T00:00:00+00:00",
            }
        )
    )
    active = list(backend.existing)
    recovered = ready | {recovering}
    result = reconcile_once(
        ledger, sources, CoveragePolicy(), backend, EscalationLog(ledger), recovered, frozenset(), None, str(pools)
    )
    assert result["submitted"] == [] and backend.existing == active
    exited = next(i for i, job in enumerate(backend.existing) if job.solver == Solver.SMALL)
    backend.existing[exited] = replace(backend.existing[exited], state=JobState.FAILED)
    result = reconcile_once(
        ledger, sources, CoveragePolicy(), backend, EscalationLog(ledger), recovered, frozenset(), None, str(pools)
    )
    assert len(result["submitted"]) == 1
    assert backend.existing[-1].solver == recovering
    assert sum(job.slots for job in backend.jobs() if job.active) == 128


def test_rolling_controllers_share_the_lease_without_duplicate_submission(tmp_path):
    ledger = str(tmp_path / "ledger")
    imports = tmp_path / "ledger/v1/imports"
    imports.mkdir(parents=True)
    (imports / "fixture.json").write_text("{}")
    entered, release = Event(), Event()
    backend = JobStore(str(tmp_path / "jobs"))
    submit = backend.submit

    def delayed_submit(action, manifest):
        entered.set()
        assert release.wait(10)
        return submit(action, manifest)

    backend.submit = delayed_submit
    args = (ledger, [SOURCE], CoveragePolicy(), backend, EscalationLog(ledger), frozenset(), frozenset(), None)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(reconcile_once, *args)
        try:
            assert entered.wait(10)
            assert reconcile_once(*args) == {"state": "another_reconciler_holds_lease"}
        finally:
            release.set()
        first.result()
    backend.submit = submit
    assert reconcile_once(*args)["submitted"] == []
    assert len(list((tmp_path / "jobs").glob("*.json"))) == 4
