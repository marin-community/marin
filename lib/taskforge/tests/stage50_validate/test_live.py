# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Live solver trials against GLM-5.3 (interactive tier) on ShellSim: the served model drives real shell tool calls
through ``GlmRolloutModel`` and RolloutEngine, and a verifier machine grades the captured file.

The check writes ``<evidence_root>/validate/<unique_run_name>/``: ``summary.json`` (per-trial outcome, reward,
cause, shell commands, ledger), the per-attempt rollout records that ``run_trials`` writes, and the ledger.
Offline tests cover retries, refusals and control replay, none of which calls the model.

Invariants: none (pins that the live endpoint's replies run and grade end to end).
"""

import json

import pytest
from rolloutengine.models import RolloutLimits
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Backend
from taskforge_adapters.glm_client import GlmClient
from taskforge_adapters.testing import unique_run_name

from taskforge.ledger.jsonl import JsonlLedger, ledger_files, read_entries
from taskforge.llm.policy import LLMPolicy
from taskforge.llm.recording import CallLedger
from taskforge.llm.rollout_model import GlmRolloutModel
from taskforge.sandbox.factories import MachineHost
from taskforge.stage50_validate.lowering import lower
from taskforge.stage50_validate.outcome import Graded, Outcome, TrialKind
from taskforge.stage50_validate.trials import EngineSettings, RetryBackoff, TrialPlan, rollout_steps, run_trials
from tests.sandbox.fixture_images import FixtureImageFactory

K = 3
ENGINE = EngineSettings(
    factories={Backend.SHELLSIM.value: ShellSimMachineFactory(), Backend.DOCKER.value: FixtureImageFactory()},
    command_timeout=60,
    limits=RolloutLimits(
        max_turns=12,
        tool_turn_timeout=120,
        model_turn_timeout=600,
        cleanup_timeout=60,
        total_turn_timeout=900,
        attempt_timeout=1200,
        verifier_timeout=None,
        max_total_tokens=None,
    ),
)


def shell_commands(outcome: Outcome) -> list[str]:
    steps = () if outcome.rollout is None else rollout_steps(outcome.rollout)
    return [call["function"]["arguments"] for step in steps for call in step.turn.message.get("tool_calls") or []]


def summary(outcome: Outcome) -> dict[str, object]:
    if isinstance(outcome, Graded):
        return {"reward": outcome.reward, "shell_calls": shell_commands(outcome)}
    return {"cause": outcome.cause, "retryable": outcome.retryable, "shell_calls": shell_commands(outcome)}


@pytest.mark.live_glm
@pytest.mark.timeout(1800)
async def test_shellsim_file_task_trials(glm_source, live_provider, evidence_root, file_task):
    directory = evidence_root / "validate" / unique_run_name("shellsim_trials")
    plan = TrialPlan(
        workpiece_id=file_task.task.id,
        round=0,
        kind=TrialKind.SOLVER,
        k=K,
        max_retries=2,
        retry_backoff=RetryBackoff(initial=0.5, maximum=5.0, factor=2.0, jitter=0.1),
        evidence_dir=directory,
        ledger=JsonlLedger(directory / "ledger"),
        first_attempt=0,
    )
    lowered = lower(file_task.task, MachineHost.LAPTOP, ENGINE.factories, ENGINE.limits)

    async with GlmClient(glm_source, live_provider) as client:
        record = CallLedger(plan.ledger, plan.workpiece_id, 0, str(TrialKind.SOLVER))
        outcomes = await run_trials(
            lowered, plan, ENGINE, GlmRolloutModel(client, LLMPolicy(max_continuations=0), record)
        )

    entries = [entry for path in ledger_files(directory / "ledger") for entry in read_entries(path)]
    report = {"trials": [summary(o) for o in outcomes], "ledger": [(e.kind, e.step, e.cause) for e in entries]}
    (directory / "summary.json").write_text(json.dumps(report, indent=1, default=str))
    assert len(outcomes) == K
    assert all(isinstance(o, Graded) for o in outcomes), directory
    assert all(shell_commands(o) for o in outcomes), "every trial should run shell commands"
    assert len(list((directory / "solver").glob("*/attempt-0.json"))) == K
