# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A validation round against GLM-5.3 (interactive pool) on ShellSim: controls, k=3 solver trials and every
adversary role, read back from the attempt files, summarized, and resumed without re-running settled trials.

Writes ``lib/taskforge/.evidence/validate/e_evidence_round-<utc>/``: the round's attempt files and ledger,
``calibration.json``, and ``summary.json`` (per-trial outcome, reward, stop reason, shell commands and final
reply, role statistics, findings, wall time).
"""

import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
from rigging.timing import ExponentialBackoff
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from taskcompendium.environment import EnvironmentKind
from taskcompendium.submission import PlainText

from taskforge.ledger.jsonl import JsonlLedger
from taskforge.llm.client import GlmClient, GlmEndpoint, Pool
from taskforge.llm.policy import LLMPolicy
from taskforge.llm.rollout_model import GlmRolloutModel
from taskforge.sandbox.factories import SHELLSIM
from taskforge.validate.adversary import AdversaryRole, run_adversaries
from taskforge.validate.attempts import trial_files
from taskforge.validate.calibration import CalibrationBand, commands, final_reply, load_summary, summarize, write_summary
from taskforge.validate.controls import ServerTokenizer
from taskforge.validate.outcome import Graded, Outcome, TrialKind
from taskforge.validate.run import (
    ValidationEvidence,
    ValidationPolicy,
    controls_passed,
    load_validation,
    replay_controls,
)
from taskforge.validate.solver import ValidationSite, run_solver
from taskforge.validate.trials import Deadlines, EngineSettings, task_digest

pytestmark = pytest.mark.live_glm

EVIDENCE_ROOT = Path(__file__).resolve().parents[2] / ".evidence" / "validate"
LIVE_TIMEOUT = 2400
PLAIN = PlainText(id="plain")
POLICY = ValidationPolicy(
    k=3,
    adversary_k=2,
    roles=tuple(AdversaryRole),
    band=CalibrationBand(0.125, 0.875),
    sampling=LLMPolicy(temperature=0.7, max_continuations=0),
    deadlines=Deadlines(agent_timeout=900, attempt_timeout=1200),
    max_retries=2,
    token_contract_retries=2,
    retry_backoff=ExponentialBackoff(initial=0.5, maximum=5.0),
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


@pytest.mark.timeout(LIVE_TIMEOUT)
async def test_a_validation_round_on_shellsim(client, file_task, file_controls, rounds):
    directory = EVIDENCE_ROOT / f"e_evidence_round-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    draft = rounds.draft(file_task, file_controls, PLAIN)
    digest = task_digest(draft.task, draft.execution, draft.convention)
    site = ValidationSite(file_task.id, 0, directory / f"evidence-{digest[:12]}", JsonlLedger(directory / "ledger"))
    settings = EngineSettings(
        factories={EnvironmentKind.SHELLSIM: ShellSimMachineFactory()},
        capabilities={EnvironmentKind.SHELLSIM: SHELLSIM},
        max_turns=12,
        command_timeout=60,
        cleanup_timeout=60,
        conventions=(PLAIN,),
    )
    model = GlmRolloutModel(client, POLICY.sampling)
    started = time.monotonic()

    controls = await replay_controls(draft, POLICY, site, settings, ServerTokenizer(client, POLICY.sampling))
    assert controls_passed(controls), [(c.control.id, c.verdict) for c in controls]
    solver, adversaries = await asyncio.gather(
        run_solver(draft, POLICY, site, settings, model), run_adversaries(draft, POLICY, site, settings, model)
    )
    ran_for = time.monotonic() - started
    loaded = load_validation(draft, site.evidence_dir)
    summary = summarize(loaded, POLICY)
    write_summary(site.evidence_dir / "calibration.json", summary)

    before = attempt_counts(site.evidence_dir)
    unsettled = {
        f"{kind}/{name}"
        for kind in (TrialKind.SOLVER, TrialKind.ADVERSARY)
        for name, files in trial_files(site.evidence_dir, kind).items()
        if not files.settled
    }
    await asyncio.gather(
        run_solver(draft, POLICY, site, settings, model), run_adversaries(draft, POLICY, site, settings, model)
    )
    after = attempt_counts(site.evidence_dir)

    record = {
        "purpose": "validation round on ShellSim: controls, k=3 solver, adversary_k=2 for every role, resume",
        "pool": Pool.HIGH,
        "policy_digest": POLICY.digest,
        "wall_time": ran_for,
        "controls": {c.control.id: c.verdict for c in controls},
        "solver": [trial_summary(o) for o in solver],
        "adversaries": {role: [trial_summary(o) for o in outcomes] for role, outcomes in adversaries.items()},
        "status": repr(summary.status),
        "solve_rate": summary.solve_rate,
        "roles": {role: vars(stats) for role, stats in summary.roles.items()},
        "findings": [{"kind": f.kind, "roles": f.roles, "new_controls": len(f.new_controls)} for f in summary.findings],
        "unsettled_before_resume": sorted(unsettled),
        "attempts_before_resume": before,
        "attempts_after_resume": after,
    }
    (directory / "summary.json").write_text(json.dumps(record, indent=1, default=str))

    assert len(solver) == POLICY.k
    assert {role: len(o) for role, o in adversaries.items()} == {role: POLICY.adversary_k for role in AdversaryRole}
    assert summary == summarize(ValidationEvidence(digest, controls, solver, adversaries), POLICY)
    assert load_summary(site.evidence_dir / "calibration.json") == summary
    assert summary.policy_digest == POLICY.digest and summary.controls_met == tuple(c.id for c in file_controls)
    assert {name: count for name, count in after.items() if name not in unsettled} == {
        name: count for name, count in before.items() if name not in unsettled
    }
    assert all(after[name] > before[name] for name in unsettled)
