# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Live validation of trials and control replay against GLM-5.3 (interactive tier) on ShellSim.

Each check writes ``lib/taskforge/.evidence/validate/<check>-<utc>/``: ``summary.json`` (purpose,
per-trial outcome, reward, cause, shell commands executed, token counts, wall time), the per-attempt
rollout records that ``run_trials`` writes, and the ledger (``ledger/<item>.jsonl``).
"""

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest
from rigging.timing import ExponentialBackoff
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import MachineFactory, UnsupportedMachineSpec
from taskcompendium.environment import EnvironmentKind
from taskcompendium.models import TaskSpec
from taskcompendium.submission import AnswerFormat, SubmissionConvention

from taskforge.ledger.jsonl import JsonlLedger, ledger_files, read_entries
from taskforge.llm.client import GlmClient, GlmEndpoint, Pool
from taskforge.llm.policy import LLMPolicy
from taskforge.llm.rollout_model import GlmRolloutModel
from taskforge.sandbox.factories import SHELLSIM
from taskforge.spec.controls import Control
from taskforge.validate.controls import ControlOutcome, ControlPlan, ControlVerdict, ServerTokenizer, replay
from taskforge.validate.evidence import Complete, Evidence, Incomplete
from taskforge.validate.outcome import Cause, Graded, Outcome, TrialKind, Ungraded
from taskforge.validate.trials import EngineSettings, TrialPlan, run_trials

pytestmark = pytest.mark.live_glm

EVIDENCE_ROOT = Path(__file__).resolve().parents[2] / ".evidence" / "validate"
POLICY = LLMPolicy(max_continuations=0)
K = 3
LIVE_TIMEOUT = 1800
RETRY_BACKOFF = ExponentialBackoff(initial=0.5, maximum=5.0)


def settings(factories: dict[EnvironmentKind, MachineFactory]) -> EngineSettings:
    return EngineSettings(
        factories=factories,
        capabilities={EnvironmentKind.SHELLSIM: SHELLSIM},
        max_turns=12,
        command_timeout=60,
        convention=SubmissionConvention(id="plain", answer_format=AnswerFormat.PLAIN),
    )


def plan(directory: Path, kind: TrialKind, item_id: str, k: int = K, max_retries: int = 2) -> TrialPlan:
    return TrialPlan(
        item_id=item_id,
        round=0,
        kind=kind,
        k=k,
        max_retries=max_retries,
        retry_backoff=RETRY_BACKOFF,
        evidence_dir=directory,
        ledger=JsonlLedger(directory / "ledger"),
    )


def control_plan(directory: Path, item_id: str) -> ControlPlan:
    return ControlPlan(
        item_id=item_id,
        round=0,
        max_retries=2,
        retry_backoff=RETRY_BACKOFF,
        evidence_dir=directory,
        ledger=JsonlLedger(directory / "ledger"),
    )


def check_dir(check: str) -> Path:
    return EVIDENCE_ROOT / f"{check}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"


def shell_commands(outcome: Outcome) -> list[str]:
    rollout = outcome.rollout
    if rollout is None:
        return []
    return [
        call["function"]["arguments"]
        for message in rollout.messages
        if message["role"] == "assistant"
        for call in message.get("tool_calls") or []
    ]


def outcome_summary(outcome: Outcome) -> dict[str, object]:
    rollout = outcome.rollout
    common = {
        "turns": 0 if rollout is None else len(rollout.steps),
        "stop_reason": None if rollout is None else rollout.stop_reason,
        "response_tokens": 0 if rollout is None else rollout.loss_mask.count(1),
        "shell_calls": shell_commands(outcome),
        "final_message": None if rollout is None or not rollout.messages else rollout.messages[-1].get("content"),
    }
    if isinstance(outcome, Graded):
        return {"outcome": "graded", "reward": outcome.reward, "grade": str(outcome.grade.status), **common}
    return {"outcome": "ungraded", "cause": outcome.cause, "retryable": outcome.retryable, **common}


def ledger_summary(directory: Path) -> list[dict[str, object]]:
    return [
        {"step": e.step, "cause": e.cause, "wall": e.wall, "attrs": e.attrs, "tokens_out": e.tokens_out}
        for path in ledger_files(directory / "ledger")
        for e in read_entries(path)
    ]


def write_summary(directory: Path, purpose: str, wall_time: float, body: dict[str, object]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    record = {"purpose": purpose, "policy": repr(POLICY), "wall_time": wall_time, **body}
    record["ledger"] = ledger_summary(directory)
    (directory / "summary.json").write_text(json.dumps(record, indent=1, default=str))


def evidence_summary(evidence: Evidence, kind: TrialKind) -> dict[str, object]:
    status = evidence.status
    stats = evidence.reward_stats(kind)
    return {
        "status": "complete" if isinstance(status, Complete) else {"incomplete": dict(status.causes)},
        "graded": stats.graded,
        "mean_reward": stats.mean_reward,
        "solved": stats.solved,
    }


def control_summary(outcomes: Sequence[ControlOutcome]) -> list[dict[str, object]]:
    return [
        {"control": c.control.id, "verdict": c.verdict, "expect": repr(c.control.expect), **outcome_summary(c.outcome)}
        for c in outcomes
    ]


@pytest.fixture
async def client(glm_settings):
    endpoint = GlmEndpoint(base_url=glm_settings.base_url, token=glm_settings.token, pool=Pool.HIGH)
    async with GlmClient(endpoint) as glm:
        yield glm


async def solver_trials(
    client: GlmClient, task: TaskSpec, check: str, purpose: str, factories: dict[EnvironmentKind, MachineFactory]
) -> tuple[list[Outcome], Path]:
    directory = check_dir(check)
    started = time.monotonic()
    outcomes = await run_trials(
        task, plan(directory, TrialKind.SOLVER, task.id), settings(factories), GlmRolloutModel(client, POLICY)
    )
    evidence = Evidence({TrialKind.SOLVER: tuple(outcomes)})
    write_summary(
        directory,
        purpose,
        time.monotonic() - started,
        {"trials": [outcome_summary(o) for o in outcomes], "evidence": evidence_summary(evidence, TrialKind.SOLVER)},
    )
    return outcomes, directory


@pytest.mark.timeout(LIVE_TIMEOUT)
async def test_math_task_trials(client, math_task):
    outcomes, directory = await solver_trials(
        client, math_task, "a_math_trials", "null-environment numeric task, k=3, verifyit numeric grading", {}
    )
    assert len(outcomes) == K
    assert all(isinstance(o, Graded) for o in outcomes), directory
    assert len(list((directory / "solver").glob("*/attempt-0.json"))) == K


@pytest.mark.timeout(LIVE_TIMEOUT)
async def test_shellsim_file_task_trials(client, file_task):
    outcomes, directory = await solver_trials(
        client,
        file_task,
        "b_shellsim_trials",
        "ShellSim task, k=3: shell tool calls execute, private ShellVerifierSpec script grades /workspace/sum.txt",
        {EnvironmentKind.SHELLSIM: ShellSimMachineFactory()},
    )
    assert all(isinstance(o, Graded) for o in outcomes), directory
    assert all(shell_commands(o) for o in outcomes), "every trial should run shell commands"


async def replay_controls(
    client: GlmClient,
    task: TaskSpec,
    controls: tuple[Control, ...],
    check: str,
    factories: dict[EnvironmentKind, MachineFactory],
) -> list[ControlOutcome]:
    directory = check_dir(check)
    started = time.monotonic()
    tokenizer = CountingTokenizer(ServerTokenizer(client, POLICY))
    outcomes = await replay(task, controls, control_plan(directory, task.id), settings(factories), tokenizer)
    write_summary(
        directory,
        f"control replay on {task.id}: scripted turns tokenized through the server, graded by the engine",
        time.monotonic() - started,
        {"controls": control_summary(outcomes), "tokenize_requests": tokenizer.calls},
    )
    return outcomes


@dataclass
class CountingTokenizer:
    tokenizer: ServerTokenizer
    calls: list[dict[str, object]] = field(default_factory=list)

    async def prompt_ids(self, messages, options):
        ids = await self.tokenizer.prompt_ids(messages, options)
        self.calls.append({"messages": len(messages), "generation_prompt": True, "prompt_ids": len(ids)})
        return ids

    async def rendered_ids(self, messages, options):
        ids = await self.tokenizer.rendered_ids(messages, options)
        self.calls.append({"messages": len(messages), "generation_prompt": False, "prompt_ids": len(ids)})
        return ids


@pytest.mark.timeout(LIVE_TIMEOUT)
async def test_control_replay_math(client, math_task, math_controls):
    outcomes = await replay_controls(client, math_task, math_controls, "c_controls_math", {})
    assert [o.verdict for o in outcomes] == [ControlVerdict.MET] * len(math_controls)


@pytest.mark.timeout(LIVE_TIMEOUT)
async def test_control_replay_shellsim(client, file_task, file_controls):
    outcomes = await replay_controls(
        client, file_task, file_controls, "c_controls_shellsim", {EnvironmentKind.SHELLSIM: ShellSimMachineFactory()}
    )
    assert [o.verdict for o in outcomes] == [ControlVerdict.MET] * len(file_controls)


@pytest.mark.timeout(LIVE_TIMEOUT)
async def test_forced_machine_failure_is_classified_and_retried(client, file_task, fakes):
    factory = fakes.flaky_factory(failures=2, error=lambda: RuntimeError("forced start failure"))
    outcomes, directory = await solver_trials(
        client,
        file_task,
        "d_forced_failure",
        "forced machine-start failure on the first 2 creates; k=3 trials classify MACHINE_START and retry",
        {EnvironmentKind.SHELLSIM: factory},
    )
    assert all(isinstance(o, Graded) for o in outcomes), directory
    causes = [e["cause"] for e in ledger_summary(directory)]
    assert causes.count(Cause.MACHINE_START) == 2
    assert len(causes) == K + 2


@pytest.mark.timeout(LIVE_TIMEOUT)
async def test_unsupported_machine_is_not_retried(client, file_task, fakes):
    factory = fakes.flaky_factory(failures=100, error=lambda: UnsupportedMachineSpec("forced unsupported spec"))
    outcomes, _ = await solver_trials(
        client,
        file_task,
        "d_forced_unsupported",
        "forced UnsupportedMachineSpec: classified MACHINE_UNSUPPORTED, not retried, evidence incomplete",
        {EnvironmentKind.SHELLSIM: factory},
    )
    assert all(isinstance(o, Ungraded) and o.cause is Cause.MACHINE_UNSUPPORTED for o in outcomes)
    assert factory.creates == K
    status = Evidence({TrialKind.SOLVER: tuple(outcomes)}).status
    assert isinstance(status, Incomplete) and status.causes[Cause.MACHINE_UNSUPPORTED] == K
