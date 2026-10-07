# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validation rounds against GLM-5.3 (interactive pool) on ShellSim: controls, k=3 solver trials and every
adversary role, read back from the attempt files, summarized, and resumed without re-running settled trials.

One round runs the file task of ``conftest``; the other runs the newest draft the live build test wrote
under ``.evidence/build/live-test/`` and skips when there is none. Each writes
``lib/taskforge/.evidence/validate/{e_evidence_round,f_built_draft_round}-<utc>/``: the round's attempt files and ledger,
``calibration.json``, and ``summary.json`` (per-trial outcome, reward, stop reason, shell commands and final
reply, role statistics, each adversary trial's tier and signals, findings, notes, wall time).
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
from rolloutengine.contracts import LENGTH_STOP_REASON
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from taskcompendium.environment import EnvironmentKind
from taskcompendium.submission import PlainText

from taskforge.build.run import TaskDraft, load_draft
from taskforge.ledger.jsonl import JsonlLedger, read_entries
from taskforge.ledger.records import EntryKind
from taskforge.llm.client import GlmClient, GlmEndpoint, Pool
from taskforge.llm.policy import LLMPolicy
from taskforge.llm.rollout_model import GlmRolloutModel
from taskforge.sandbox.factories import SHELLSIM
from taskforge.validate.adversary import AdversaryRole, run_adversaries
from taskforge.validate.attempts import trial_files
from taskforge.validate.calibration import (
    CalibrationBand,
    CalibrationSummary,
    DefectTier,
    commands,
    final_reply,
    load_summary,
    summarize,
    task_facts,
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
from taskforge.validate.trials import Deadlines, EngineSettings, RetryBackoff, task_digest

pytestmark = pytest.mark.live_glm

EVIDENCE_ROOT = Path(__file__).resolve().parents[2] / ".evidence" / "validate"
BUILD_EVIDENCE = Path(__file__).resolve().parents[2] / ".evidence" / "build" / "live-test"
LIVE_TIMEOUT = 2400
PLAIN = PlainText(id="plain")
ADVERSARY_OUTPUT_TOKENS = 32768
POLICY = ValidationPolicy(
    k=3,
    adversary_k=2,
    adversary_output_tokens=ADVERSARY_OUTPUT_TOKENS,
    roles=tuple(AdversaryRole),
    band=CalibrationBand(0.125, 0.875),
    sampling=LLMPolicy(temperature=0.7, max_continuations=0),
    deadlines=Deadlines(agent_timeout=900, attempt_timeout=1200),
    max_retries=2,
    token_contract_retries=2,
    retry_backoff=RetryBackoff(initial=0.5, maximum=5.0, factor=1.5, jitter=0.1),
)


def trial_summary(outcome: Outcome) -> dict[str, object]:
    rollout = outcome.rollout
    common = {
        "stop_reason": None if rollout is None else rollout.stop_reason,
        "turns": 0 if rollout is None else len(rollout.steps),
        "commands": [] if rollout is None else commands(rollout),
        "final_reply": None if rollout is None else final_reply(rollout),
    }
    if isinstance(outcome, Graded):
        return {"outcome": "graded", "reward": outcome.reward, "timed_out": outcome.timed_out, **common}
    return {"outcome": "ungraded", "cause": outcome.cause, "detail": outcome.detail[-400:], **common}


def assessment_summary(summary: CalibrationSummary) -> dict[str, list[dict[str, object]]]:
    """Per adversary trial: its tier, the row that fired, why, and the signals the row read."""
    record: dict[str, list[dict[str, object]]] = {}
    for a in summary.assessments:
        s = a.signals
        record.setdefault(a.role, []).append(
            {
                "index": a.index,
                "tier": a.tier,
                "rule": a.rule,
                "reason": a.reason,
                "passed": s.passed,
                "gave_up": s.gave_up,
                "output_tokens": s.output_tokens,
                "budget_exhausted": s.budget_exhausted,
                "inputs_consumed": s.inputs_consumed,
                "inputs_written": s.inputs_written,
                "comparison": s.comparison,
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
    adversaries: Mapping[AdversaryRole, tuple[Outcome, ...]]
    summary: CalibrationSummary
    unsettled: frozenset[str]
    before: dict[str, int]
    after: dict[str, int]


async def live_round(client: GlmClient, draft: TaskDraft, directory: Path, purpose: str) -> LiveRound:
    """Run controls, then solver and adversaries, on ShellSim; summarize from disk, resume, write ``summary.json``."""
    digest = task_digest(draft.task, draft.execution, draft.convention)
    site = ValidationSite(draft.task.id, 0, directory / f"evidence-{digest[:12]}", JsonlLedger(directory / "ledger"))
    settings = EngineSettings(
        factories={EnvironmentKind.SHELLSIM: ShellSimMachineFactory()},
        capabilities={EnvironmentKind.SHELLSIM: SHELLSIM},
        max_turns=12,
        command_timeout=60,
        cleanup_timeout=60,
        conventions=(draft.convention,),
    )
    model = partial(GlmRolloutModel, client, POLICY.sampling)
    started = time.monotonic()

    controls = await replay_controls(draft, POLICY, site, settings, ServerTokenizer(client, POLICY.sampling))
    assert controls_passed(controls), [(c.control.id, c.verdict) for c in controls]
    solver, adversaries = await asyncio.gather(
        run_solver(draft, POLICY, site, settings, model), run_adversaries(draft, POLICY, site, settings, model)
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
        run_solver(draft, POLICY, site, settings, model), run_adversaries(draft, POLICY, site, settings, model)
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
        "adversaries": {role: [trial_summary(o) for o in outcomes] for role, outcomes in adversaries.items()},
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
    digest = task_digest(draft.task, draft.execution, draft.convention)
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
    # Every control a finding ships comes from an adversary pass tiered as a repair.
    repairs = {f"adversary/{a.role}/{a.index}" for a in summary.assessments if a.tier is DefectTier.REPAIR}
    assert {c.author for f in summary.findings for c in f.new_controls} <= repairs


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


@pytest.mark.timeout(LIVE_TIMEOUT)
async def test_a_validation_round_on_shellsim(client, file_task, file_controls, rounds):
    run = await live_round(
        client,
        rounds.draft(file_task, file_controls, PLAIN),
        EVIDENCE_ROOT / f"e_evidence_round-{utc_now()}",
        "validation round on ShellSim: controls, k=3 solver, adversary_k=2 for every role, resume",
    )

    assert_round_reads_back_and_resumes(run)
    # The file task's grader admits no shortcut, so no adversary pass is a defect to repair.
    assert all(a.tier is not DefectTier.REPAIR for a in run.summary.assessments), assessment_summary(run.summary)
    # No attempt reached the 32768-token budget (the largest recorded spend is 7516). Stops on max_turns are
    # model behaviour: RoleStats.exhausted counts them and summary.json records them.
    assert all(a.signals.stop_reason != LENGTH_STOP_REASON for a in run.summary.assessments)


def newest_built_draft() -> Path | None:
    drafts = sorted(BUILD_EVIDENCE.glob("*/*/draft/convention.json"))
    return drafts[-1].parent if drafts else None


@pytest.mark.timeout(LIVE_TIMEOUT)
async def test_a_validation_round_on_a_built_draft(client):
    """The newest draft the live build test wrote, validated as the loop would; tiers are recorded, not asserted."""
    directory = newest_built_draft()
    if directory is None:
        pytest.skip(f"no built draft under {BUILD_EVIDENCE}")
    run = await live_round(
        client,
        load_draft(directory),
        EVIDENCE_ROOT / f"f_built_draft_round-{utc_now()}",
        f"validation round on the built draft {directory.relative_to(BUILD_EVIDENCE)}: controls, solver, adversaries",
    )

    assert_round_reads_back_and_resumes(run)
