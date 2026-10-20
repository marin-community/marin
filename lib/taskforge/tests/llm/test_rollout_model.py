# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GlmRolloutModel against a scripted GLM server through RolloutEngine, and live on GLM-5.3, under the
messages trace.

The live checks write ``<evidence_root>/validate/rollout_model/<check>-<utc>.json``: every rollout
(messages, per-turn usage) plus, per turn, how many prompt tokens the server reported cached.
``evidence_root`` is the fixture in ``tests/conftest.py``.
"""

import asyncio
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import JsonValue
from rolloutengine.models import (
    TRACE_METRIC,
    Continuation,
    GenerationLimitReached,
    GradedRollout,
    ModelRequest,
    RolloutContractError,
    RolloutStep,
    TaskRolloutSpec,
    TokenUsage,
    TraceMode,
)
from taskcompendium.grading_result import Outcome
from taskcompendium.models import AnswerType, PlainText, Source, TaskSpec

from taskforge.ledger.records import EntryKind, LedgerEntry
from taskforge.llm.glm_client import GlmClient
from taskforge.llm.policy import LLMPolicy
from taskforge.llm.recording import CallLedger
from taskforge.llm.rollout_model import GlmRolloutModel
from taskforge.sandbox.factories import MachineHost
from taskforge.spec.draft import (
    assemble,
    file,
    grader_environment,
    lower,
    machine,
    requirements,
    rollout_engine,
    rollout_limits,
)
from tests.conftest import fake_source
from tests.sandbox.fixture_images import FixtureImageFactory, PortlessShellSimFactory, shell_grader

POLICY = LLMPolicy(max_continuations=0)
# Grading runs on the grader-base verifier machine, here served by ShellSim through the fixture factory.
FACTORIES = {"shellsim": PortlessShellSimFactory(), "docker": FixtureImageFactory()}
CONTEXT_ERROR = "This model's maximum context length is 262144 tokens. However, you requested 300000 tokens."
OUTPUT_CHECK = """output=/$(jq -r .path /tests/config.json)
passed=false
if test -f "$output"; then
    if jq -e '.words == null' /tests/config.json > /dev/null; then
        passed=true
    elif test "$(awk '{for (i = 1; i <= NF; i++) print $i}' "$output")" = "$(jq -r '.words[]' /tests/config.json)"; then
        passed=true
    fi
fi
jq -n -c --argjson passed "$passed" '{
    reward: (if $passed then 1.0 else 0.0 end), detail: {checks: [{name: "output", passed: $passed, weight: 1.0}]}
}' > /logs/verifier/reward.json
"""
"""Rewards 1 when the captured output file exists and, unless ``words`` is null, holds exactly those words."""
PUZZLE_1 = "List, ascending and one per line, every prime p below 1000 such that p + 2 and p + 6 are both prime.\n"
PUZZLE_2 = (
    "List, ascending and one per line, every three-digit number whose digits are strictly increasing "
    "from left to right and sum to 20.\n"
)
PUZZLE_1_ANSWER = "5 11 17 41 101 107 191 227 311 347 461 641 821 857 881".split()
LONG_REASONING_TOKENS = 1000


@pytest.fixture
def evidence_dir(evidence_root: Path) -> Path:
    return evidence_root / "validate" / "rollout_model"


class ListLedger:
    def __init__(self) -> None:
        self.entries: list[LedgerEntry] = []

    def record(self, entry: LedgerEntry) -> None:
        self.entries.append(entry)


def call_ledger(ledger: ListLedger | None = None) -> CallLedger:
    return CallLedger(ledger=ledger or ListLedger(), workpiece_id="rollout-model", round=0, step="solver")


def shell_task():
    return assemble(
        "rollout-model-shell",
        "Create /workspace/done.txt.",
        AnswerType.FILE,
        PlainText(),
        output_check("workspace/done.txt", None),
        Source(dataset="taskforge-tests", revision="1", row="0", importer_revision="1"),
        environment=requirements(image=None),
        output_paths=("/workspace/done.txt",),
    )


def output_check(path: str, words: Sequence[str] | None):
    listed: list[JsonValue] | None = None if words is None else list(words)
    config: dict[str, JsonValue] = {"path": path, "words": listed}
    return shell_grader(
        OUTPUT_CHECK,
        config,
        environment=grader_environment(None),
        answer_path=None,
        timeout=30,
    )


def shellsim(task: TaskSpec, max_turns: int = 6) -> TaskRolloutSpec:
    """``task`` on ShellSim under the messages trace, as validation runs every trial."""
    limits = rollout_limits(
        max_turns=max_turns,
        model_turn_timeout=None,
        tool_turn_timeout=60,
        total_turn_timeout=None,
        attempt_timeout=None,
        verifier_timeout=30,
        cleanup_timeout=30,
    )
    spec = lower(
        task,
        host=MachineHost.LAPTOP,
        task_machine=machine(startup_timeout=30),
        verifier_machine=machine(startup_timeout=30),
        limits=limits,
        factories=FACTORIES,
    )
    return spec.model_copy(update={"trace": TraceMode.MESSAGES})


async def run(model: GlmRolloutModel, spec: TaskRolloutSpec) -> GradedRollout:
    return await rollout_engine(FACTORIES, command_timeout=30).run_task(spec, model)


def steps(result: GradedRollout) -> tuple[RolloutStep, ...]:
    return tuple(step for conversation in result.rollout.conversations for step in conversation.steps)


@pytest.fixture
async def fake_client(fake_glm, fast_provider):
    async with GlmClient(fake_source(fake_glm), replace(fast_provider, max_attempts=1)) as client:
        yield client


async def test_two_turn_rollout_records_usage_and_replays_reasoning(fake_glm, fake_client):
    fake_glm.stream(
        reasoning="plan",
        tool_calls=(("shell", '{"command": "touch /workspace/done.txt"}'),),
        finish="tool_calls",
        prompt_tokens=3,
        completion_tokens=2,
    )
    fake_glm.stream(content="Done.", prompt_tokens=7, completion_tokens=2)

    result = await run(GlmRolloutModel(fake_client, POLICY, call_ledger()), shellsim(shell_task()))

    assert (result.grade.status, result.grade.reward) == (Outcome.GRADED, 1.0)
    assert result.rollout.metrics[TRACE_METRIC] == TraceMode.MESSAGES.value
    assert [step.turn.usage for step in steps(result)] == [TokenUsage(3, 2), TokenUsage(7, 2)]
    assert [(step.turn.prompt_token_ids, step.turn.response_token_ids) for step in steps(result)] == [((), ())] * 2
    assert [step.turn.logprobs for step in steps(result)] == [None] * 2
    first, second = fake_glm.requests
    assert "return_token_ids" not in first and "logprobs" not in first
    assert [tool["function"]["name"] for tool in first["tools"]] == ["shell"]
    assert second["messages"][1]["reasoning_content"] == "plan"
    assert second["messages"][2]["role"] == "tool"


async def test_length_cut_turn_is_graded_with_length_stop(fake_glm, fake_client):
    fake_glm.stream(content="I will", finish="length")

    result = await run(GlmRolloutModel(fake_client, POLICY, call_ledger()), shellsim(shell_task()))

    assert result.rollout.stop_reason == "length"
    assert (result.grade.status, result.grade.reward) == (Outcome.GRADED, 0.0)
    assert len(fake_glm.requests) == 1


async def test_tool_call_cut_at_the_budget_ends_the_rollout_with_length_unexecuted(fake_glm, fake_client):
    # vLLM's GLM tool parser reports a tool call cut by max_tokens as finish_reason "tool_calls".
    command = '{"command": "touch /workspace/done.txt"}'
    fake_glm.stream(tool_calls=(("shell", command),), finish="tool_calls", completion_tokens=3)
    policy = LLMPolicy(max_tokens=3, max_continuations=0)

    result = await run(GlmRolloutModel(fake_client, policy, call_ledger()), shellsim(shell_task()))

    assert result.rollout.stop_reason == "length"
    assert (result.grade.status, result.grade.reward) == (Outcome.GRADED, 0.0)
    assert len(fake_glm.requests) == 1


async def test_prompt_filling_the_context_raises_generation_limit(fake_glm, fake_client):
    fake_glm.status(400, CONTEXT_ERROR)
    fake_glm.status(400, CONTEXT_ERROR)
    request = ModelRequest(({"role": "user", "content": "x"},), {})

    with pytest.raises(GenerationLimitReached) as raised:
        await GlmRolloutModel(fake_client, POLICY, call_ledger())(request)

    # The server rendered the prompt, so the messages trace keeps no prompt ids of it.
    assert raised.value.prompt_token_ids == ()


@pytest.mark.parametrize(
    "request_ids",
    [{"continuation": Continuation((1, 2), 1)}, {"prompt_token_ids": (1, 2)}],
    ids=["continuation", "pinned_prompt"],
)
async def test_request_with_token_ids_is_refused_before_any_call(fake_glm, fake_client, request_ids):
    messages = ({"role": "user", "content": "x"}, {"role": "assistant", "content": "y"})
    request = ModelRequest(messages, {}, **request_ids)

    with pytest.raises(RolloutContractError, match="MESSAGES"):
        await GlmRolloutModel(fake_client, POLICY, call_ledger())(request)

    assert fake_glm.requests == []


def live_task():
    return assemble(
        "rollout-model-live",
        "Work in /workspace with the shell tool, one command per call. Before each call, reason briefly "
        "about what it must do. First write the primes below 50, one per line, to /workspace/primes.txt. "
        "Then, in a separate call, write the number of lines of that file to /workspace/count.txt. "
        "Then show both files with cat, and say when you are done.",
        AnswerType.FILE,
        PlainText(),
        output_check("workspace/count.txt", ["15"]),
        Source(dataset="taskforge-tests", revision="1", row="live", importer_revision="1"),
        environment=requirements(image=None),
        output_paths=("/workspace/primes.txt", "/workspace/count.txt"),
    )


def turn_record(result: GradedRollout) -> list[dict[str, Any]]:
    return [
        {
            "stop_reason": step.turn.stop_reason,
            "usage": step.turn.metadata["usage"],
            "reasoning_chars": len(step.turn.message.get("reasoning_content") or ""),
            "tool_calls": [call["function"]["arguments"] for call in step.turn.message.get("tool_calls") or []],
        }
        for step in steps(result)
    ]


def write_live_evidence(
    evidence_dir: Path, check: str, purpose: str, wall_time: float, results: list[GradedRollout]
) -> Path:
    record = {
        "purpose": purpose,
        "policy": repr(POLICY),
        "wall_time": wall_time,
        "turns": [turn_record(result) for result in results],
        "grades": [asdict(result.grade) for result in results],
        "rollouts": [asdict(result.rollout) for result in results],
    }
    evidence_dir.mkdir(parents=True, exist_ok=True)
    path = evidence_dir / f"{check}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(record, indent=1, default=str))
    return path


async def live_rollouts(
    glm_source, live_provider, task: TaskSpec, count: int, max_turns: int
) -> tuple[list[GradedRollout], float]:
    """``count`` concurrent rollouts of ``task``, each under its own step, as trials run.

    Also asserts that the ledger holds one ``LLM_CALL`` per turn.
    """
    ledger = ListLedger()
    started = time.monotonic()
    async with GlmClient(glm_source, live_provider) as client:
        model = GlmRolloutModel(client, POLICY, call_ledger(ledger))
        spec = shellsim(task, max_turns)
        results = await asyncio.gather(*(run(model.for_step(f"solver/{index}/0"), spec) for index in range(count)))
    calls = [entry for entry in ledger.entries if entry.kind == EntryKind.LLM_CALL]
    assert len(calls) == sum(len(steps(result)) for result in results)
    assert all(entry.cause is None and entry.tokens_out for entry in calls)
    return list(results), time.monotonic() - started


@pytest.mark.live_glm
@pytest.mark.timeout(1800)
async def test_live_multi_turn_shellsim_rollout_is_graded(glm_source, live_provider, evidence_dir):
    rollouts, wall_time = await live_rollouts(glm_source, live_provider, live_task(), count=3, max_turns=10)
    path = write_live_evidence(
        evidence_dir,
        "multi-turn",
        "multi-turn ShellSim rollouts through RolloutEngine under the messages trace, with reasoning "
        "replayed and prefix caching on",
        wall_time,
        rollouts,
    )

    assert all(len(steps(result)) >= 2 for result in rollouts), path
    assert any(step.turn.metadata["usage"]["cached_tokens"] > 0 for result in rollouts for step in steps(result)), path
    assert all(result.grade.status == Outcome.GRADED for result in rollouts), path


def reasoning_task():
    return assemble(
        "rollout-model-reasoning",
        "Work in /workspace with the shell tool, exactly one shell call per turn. Never use the shell to "
        "compute an answer: solve each puzzle yourself, in your reasoning, checking every candidate, and "
        "use the shell only to read and write files.\n"
        "1. cat /workspace/puzzle1.txt, solve it, and write the answer to /workspace/answer1.txt with a "
        "single cat heredoc command.\n"
        "2. cat /workspace/puzzle2.txt, solve it, and write the answer to /workspace/answer2.txt the same way.\n"
        "3. With a single cat heredoc command, write a POSIX sh script /workspace/report.sh of at least ten "
        "lines, with comments, blank lines and tab-indented loop bodies, that prints each line of both answer "
        "files prefixed by the file name. Then run it with sh.\n"
        "Say when you are done.",
        AnswerType.FILE,
        PlainText(),
        output_check("workspace/answer1.txt", PUZZLE_1_ANSWER),
        Source(dataset="taskforge-tests", revision="1", row="reasoning", importer_revision="1"),
        environment=requirements(image=None),
        files=(file("workspace/puzzle1.txt", PUZZLE_1), file("workspace/puzzle2.txt", PUZZLE_2)),
        output_paths=("/workspace/answer1.txt", "/workspace/answer2.txt", "/workspace/report.sh"),
    )


@pytest.mark.live_glm
@pytest.mark.timeout(3600)
async def test_live_reasoning_heavy_rollout_replays_long_reasoning(glm_source, live_provider, evidence_dir):
    rollouts, wall_time = await live_rollouts(glm_source, live_provider, reasoning_task(), count=3, max_turns=12)
    path = write_live_evidence(
        evidence_dir,
        "reasoning",
        "reasoning-heavy multi-turn ShellSim rollouts under the messages trace: long reasoning_content "
        "replayed on later turns and multi-line heredoc tool arguments",
        wall_time,
        rollouts,
    )

    turns = [turn for rollout in rollouts for turn in turn_record(rollout)]
    assert all(len(steps(result)) >= 3 for result in rollouts), path
    # Long reasoning is replayed only once a later turn follows it, so some turn before a rollout's
    # last must reason at length; how many do depends on the puzzles' luck.
    replayed = [turn for rollout in rollouts for turn in turn_record(rollout)[:-1]]
    assert any(turn["usage"]["reasoning_tokens"] >= LONG_REASONING_TOKENS for turn in replayed), path
    assert sum("\\n" in arguments for turn in turns for arguments in turn["tool_calls"]) >= 3, path
