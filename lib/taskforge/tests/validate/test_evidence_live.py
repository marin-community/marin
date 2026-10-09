# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validation rounds against GLM-5.3 (interactive pool) on ShellSim: controls, k=3 solver trials and two adversary
agent loops with the verifier as a tool, read back from the attempt files, summarized, and resumed without re-running
settled trials.

One round runs the file task of ``conftest``, its verifier machine on the ShellSim-backed fixture image factory;
the other runs the newest draft the live build test wrote under ``<evidence_root>/build/live-test/`` on the
laptop's own factories (Docker for verifier machines) and skips when no draft there loads (drafts written before
TaskSpec 0.25 do not). ``evidence_root`` is the fixture in
``tests/conftest.py``. Each writes ``<evidence_root>/validate/{g_adversary_round,h_built_adversary_round}-<utc>/``:
the round's attempt files and ledger, ``calibration.json``, and ``summary.json`` (per-trial outcome, reward, stop
reason, tool-call arguments (key ``commands``) and final reply, each adversary trial's submissions, claim, tier
and signals, role statistics, findings, notes, wall time). Model behaviour (claims, the submission that passed
first, the exploit, budget use) is recorded, not asserted.
"""

import asyncio
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import pytest
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Backend, MachineFactory
from taskcompendium.models import TextMessage

from taskforge.builder.run import LOWERED_FILE, PROVENANCE_FILE, TaskDraft, load_draft
from taskforge.ledger.jsonl import JsonlLedger, read_entries
from taskforge.ledger.records import EntryKind
from taskforge.llm.client import GlmClient, GlmEndpoint, Pool
from taskforge.llm.policy import LLMPolicy
from taskforge.llm.rollout_model import GlmRolloutModel
from taskforge.sandbox.factories import (
    LOCAL_DOCKER,
    SHELLSIM,
    FactoryCapabilities,
    MachineHost,
    factory_capabilities,
    machine_factories,
)
from taskforge.validate.adversary import (
    PREAMBLE_SEPARATOR,
    SUBMIT_TOOL_NAME,
    AdversaryRole,
    adversary_brief,
    run_adversaries,
    trial_grade,
)
from taskforge.validate.attempts import load_adversary_attempt, trial_files
from taskforge.validate.calibration import (
    CalibrationBand,
    CalibrationSummary,
    DefectTier,
    final_reply,
    load_summary,
    summarize,
    task_facts,
    tool_call_arguments,
    write_summary,
)
from taskforge.validate.controls import ControlOutcome, ServerTokenizer
from taskforge.validate.outcome import Graded, Outcome, TrialKind
from taskforge.validate.run import (
    ValidationEvidence,
    ValidationPolicy,
    controls_passed,
    load_validation,
    replay_controls,
)
from taskforge.validate.solver import ValidationSite, run_solver
from taskforge.validate.submissions import AdversaryTrial, trial_claim
from taskforge.validate.trials import Deadlines, EngineSettings, RetryBackoff, task_digest
from tests.sandbox.fixture_images import FixtureImageFactory

pytestmark = pytest.mark.live_glm

LIVE_TIMEOUT = 2400
MAX_TURNS = 24
POLICY = ValidationPolicy(
    k=3,
    adversary_k=2,
    adversary_submissions=10,
    adversary_repair_submissions=3,
    band=CalibrationBand(0.125, 0.875),
    sampling=LLMPolicy(temperature=0.7, max_continuations=0),
    deadlines=Deadlines(total_turn_timeout=900, attempt_timeout=1200),
    max_retries=2,
    token_contract_retries=2,
    retry_backoff=RetryBackoff(initial=0.5, maximum=5.0, factor=1.5, jitter=0.1),
)


def trial_summary(outcome: Outcome) -> dict[str, object]:
    rollout = outcome.rollout
    common = {
        "stop_reason": None if rollout is None else rollout.stop_reason,
        "turns": 0 if rollout is None else len(rollout.steps),
        "commands": [] if rollout is None else tool_call_arguments(rollout),
        "final_reply": None if rollout is None else final_reply(rollout),
    }
    if isinstance(outcome, Graded):
        return {"outcome": "graded", "reward": outcome.reward, "timed_out": outcome.timed_out, **common}
    return {"outcome": "ungraded", "cause": outcome.cause, "detail": outcome.detail[-400:], **common}


def adversary_summary(trial: AdversaryTrial) -> dict[str, object]:
    """One adversary trial: its outcome, verdict line and every verifier submission."""
    claim = trial_claim(trial.outcome)
    return {
        **trial_summary(trial.outcome),
        "claim": claim.kind,
        "why": claim.why,
        "submissions": [
            {
                "ordinal": s.ordinal,
                "turn": s.turn,
                "files": list(s.candidate.paths),
                "reply": None if s.candidate.reply is None else s.candidate.reply[-400:],
                "status": s.grade.status,
                "reward": s.grade.reward,
                "passed": s.passed,
                "wall_time": s.wall_time,
            }
            for s in trial.submissions
        ],
    }


def assessment_summary(summary: CalibrationSummary) -> dict[str, list[dict[str, object]]]:
    """Per adversary trial: its tier, the row that fired, why, the subject submission, and the signals the row read."""
    record: dict[str, list[dict[str, object]]] = {}
    for a in summary.assessments:
        record.setdefault(a.role, []).append(
            {
                "index": a.index,
                "tier": a.tier,
                "rule": a.rule,
                "reason": a.reason,
                "subject": a.subject,
                **vars(a.signals),
            }
        )
    return record


def attempt_counts(evidence_dir: Path) -> dict[str, int]:
    return {
        f"{kind}/{name}": files.attempts
        for kind in (TrialKind.SOLVER, TrialKind.ADVERSARY)
        for name, files in trial_files(evidence_dir, kind).items()
    }


@pytest.fixture
def evidence_dir(evidence_root: Path) -> Path:
    return evidence_root / "validate"


@pytest.fixture
def build_evidence(evidence_root: Path) -> Path:
    """Where the live build test writes its drafts."""
    return evidence_root / "build" / "live-test"


@pytest.fixture
async def client(glm_settings):
    endpoint = GlmEndpoint(base_url=glm_settings.base_url, token=glm_settings.token, pool=Pool.HIGH)
    async with GlmClient(endpoint) as glm:
        yield glm


@dataclass(frozen=True)
class LiveRound:
    """One live round, read back, summarized, and resumed once."""

    draft: TaskDraft
    site: ValidationSite
    controls: tuple[ControlOutcome, ...]
    solver: tuple[Outcome, ...]
    adversaries: Mapping[AdversaryRole, tuple[AdversaryTrial, ...]]
    summary: CalibrationSummary
    unsettled: frozenset[str]
    before: dict[str, int]
    after: dict[str, int]


async def live_round(
    client: GlmClient,
    draft: TaskDraft,
    directory: Path,
    purpose: str,
    factories: Mapping[str, MachineFactory],
    capabilities: Mapping[str, FactoryCapabilities],
) -> LiveRound:
    """Run controls, then solver and adversaries, on ``factories``; summarize from disk, resume, write
    ``summary.json``."""
    digest = task_digest(draft.lowered)
    site = ValidationSite(draft.task.id, 0, directory / f"evidence-{digest[:12]}", JsonlLedger(directory / "ledger"))
    settings = EngineSettings(
        factories=factories,
        capabilities=capabilities,
        max_turns=MAX_TURNS,
        command_timeout=60,
        tool_turn_timeout=120,
        model_turn_timeout=600,
        cleanup_timeout=60,
    )
    model = partial(GlmRolloutModel, client, POLICY.sampling)
    started = time.monotonic()

    controls = await replay_controls(draft, POLICY, site, settings, ServerTokenizer(client, POLICY.sampling))
    assert controls_passed(controls), [(c.control.id, c.verdict) for c in controls]
    solver, adversaries = await asyncio.gather(
        run_solver(draft, POLICY, site, settings, model), run_adversaries(draft, POLICY, site, settings, client, "")
    )
    ran_for = time.monotonic() - started
    summary = summarize(load_validation(draft, site.evidence_dir), POLICY)
    write_summary(site.evidence_dir / "calibration.json", summary)

    before = attempt_counts(site.evidence_dir)
    unsettled = frozenset(
        f"{kind}/{name}"
        for kind in (TrialKind.SOLVER, TrialKind.ADVERSARY)
        for name, files in trial_files(site.evidence_dir, kind).items()
        if not files.settled
    )
    await asyncio.gather(
        run_solver(draft, POLICY, site, settings, model), run_adversaries(draft, POLICY, site, settings, client, "")
    )
    after = attempt_counts(site.evidence_dir)

    record = {
        "purpose": purpose,
        "pool": Pool.HIGH,
        "task_id": draft.task.id,
        "policy_digest": POLICY.digest,
        "wall_time": ran_for,
        "controls": {c.control.id: c.verdict for c in controls},
        "solver": [trial_summary(o) for o in solver],
        "adversaries": {role: [adversary_summary(t) for t in trials] for role, trials in adversaries.items()},
        "status": repr(summary.status),
        "solve_rate": summary.solve_rate,
        "roles": {role: vars(stats) for role, stats in summary.roles.items()},
        "assessments": assessment_summary(summary),
        "findings": [{"kind": f.kind, "roles": f.roles, "new_controls": len(f.new_controls)} for f in summary.findings],
        "notes": [{"kind": n.kind, "roles": n.roles} for n in summary.notes],
        "unsettled_before_resume": sorted(unsettled),
        "attempts_before_resume": before,
        "attempts_after_resume": after,
    }
    (directory / "summary.json").write_text(json.dumps(record, indent=1, default=str))
    return LiveRound(draft, site, controls, solver, adversaries, summary, unsettled, before, after)


def assert_round_reads_back_and_resumes(run: LiveRound) -> None:
    draft, summary = run.draft, run.summary
    digest = task_digest(draft.lowered)
    assert len(run.solver) == POLICY.k
    assert {role: len(o) for role, o in run.adversaries.items()} == {role: POLICY.adversary_k for role in AdversaryRole}
    assert summary == summarize(
        ValidationEvidence(digest, run.controls, run.solver, run.adversaries, task_facts(draft.task)), POLICY
    )
    assert load_summary(run.site.evidence_dir / "calibration.json") == summary
    assert summary.policy_digest == POLICY.digest and summary.controls_met == tuple(c.id for c in draft.controls)
    assert {name: count for name, count in run.after.items() if name not in run.unsettled} == {
        name: count for name, count in run.before.items() if name not in run.unsettled
    }
    assert all(run.after[name] > run.before[name] for name in run.unsettled)
    call_steps = {e.step for e in read_entries(run.site.ledger.path_for(draft.task.id)) if e.kind == EntryKind.LLM_CALL}
    assert call_steps == {f"solver/{i}" for i in range(POLICY.k)} | {
        f"adversary/{role}/{i}" for role in AdversaryRole for i in range(POLICY.adversary_k)
    }
    # Every control a finding ships is the subject submission of an adversary trial tiered as a repair.
    repairs = {f"adversary/{a.role}/{a.index}#{a.subject}" for a in summary.assessments if a.tier is DefectTier.REPAIR}
    assert {c.author for f in summary.findings for c in f.new_controls} <= repairs
    first = draft.task.context.events[0]
    brief = adversary_brief(POLICY.adversary_submissions, "")
    task_system = isinstance(first, TextMessage) and first.role == "system"
    expected = f"{brief}{PREAMBLE_SEPARATOR}{first.content}" if task_system else brief
    entries = list(read_entries(run.site.ledger.path_for(draft.task.id)))
    submits = [e for e in entries if e.kind == EntryKind.STEP and e.attrs.get("tool") == SUBMIT_TOOL_NAME]
    for trials in run.adversaries.values():
        for trial in trials:
            assert isinstance(trial.outcome, Graded), trial.outcome
            assert trial.system == expected
            assert len(trial.submissions) <= POLICY.adversary_submissions
            # The trial's grade is its last passing submission's, else its last submission's.
            assert trial.outcome.grade == trial_grade(trial.submissions)
    # Every recorded submission, in every attempt, is a submit span; a call refused for the budget or a missing
    # file is a span with no submission.
    attempts = (run.site.evidence_dir / TrialKind.ADVERSARY).rglob("attempt-*.json")
    recorded = sum(len(load_adversary_attempt(path).submissions) for path in attempts)
    assert len(submits) >= recorded


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


@pytest.mark.timeout(LIVE_TIMEOUT)
async def test_a_validation_round_on_shellsim(client, evidence_dir, file_task, file_controls, rounds):
    run = await live_round(
        client,
        rounds.draft(file_task, file_controls),
        evidence_dir / f"g_adversary_round-{utc_now()}",
        "validation round on ShellSim: controls, k=3 solver, adversary_k=2 agent loops with 10 submissions, resume",
        {Backend.SHELLSIM.value: ShellSimMachineFactory(), Backend.DOCKER.value: FixtureImageFactory()},
        {Backend.SHELLSIM.value: SHELLSIM, Backend.DOCKER.value: LOCAL_DOCKER},
    )

    assert_round_reads_back_and_resumes(run)
    # The file task's grader compares against a constant, so no accepted submission is a defect to repair.
    assert all(a.tier is not DefectTier.REPAIR for a in run.summary.assessments), assessment_summary(run.summary)


def newest_built_draft(build_evidence: Path) -> tuple[Path, TaskDraft] | None:
    """The newest complete draft that loads; ``run_build`` writes the provenance file last, and a draft written
    before TaskSpec 0.25 fails to load and is passed over."""
    for path in sorted(build_evidence.glob(f"*/*/draft/{PROVENANCE_FILE}"), reverse=True):
        if not (path.parent / LOWERED_FILE).is_file():
            continue
        try:
            return path.parent, load_draft(path.parent)
        except ValueError:
            continue
    return None


@pytest.mark.timeout(LIVE_TIMEOUT)
async def test_a_validation_round_on_a_built_draft(client, evidence_dir, build_evidence, image_cache):
    """The newest draft the live build test wrote, validated as the loop would; tiers are recorded, not asserted."""
    built = newest_built_draft(build_evidence)
    if built is None:
        pytest.skip(f"no draft under {build_evidence} loads")
    directory, draft = built
    run = await live_round(
        client,
        draft,
        evidence_dir / f"h_built_adversary_round-{utc_now()}",
        f"validation round on the built draft {directory.relative_to(build_evidence)}: controls, solver, adversaries",
        machine_factories(MachineHost.LAPTOP, controller_url=None, image_cache=image_cache),
        factory_capabilities(MachineHost.LAPTOP),
    )

    assert_round_reads_back_and_resumes(run)
