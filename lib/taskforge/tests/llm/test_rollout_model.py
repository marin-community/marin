# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GlmRolloutModel against a scripted GLM server through RolloutEngine, and live on GLM-5.3, under the
messages trace.

The live checks write ``<evidence_root>/validate/rollout_model/<check>-<utc>.json``: every rollout
(messages, token ids, logprobs, per-turn usage) plus, per turn, how many prompt tokens the server
reported cached. ``evidence_root`` is the fixture in ``tests/conftest.py``.
"""

import asyncio
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import JsonValue
from rolloutengine.contracts import GenerationLimitReached, ModelRequest, RolloutData, TokenUsage
from rolloutengine.engine import RolloutEngine
from rolloutengine.spec import TRACE_METRIC, LoweredTaskSpec, TraceMode
from taskcompendium.grading_result import Outcome
from taskcompendium.models import AnswerType, PlainText, Source, TaskSpec

from taskforge.ledger.records import EntryKind, LedgerEntry
from taskforge.llm.glm_client import GlmClient
from taskforge.llm.policy import LLMPolicy
from taskforge.llm.recording import CallLedger
from taskforge.llm.rollout_model import TOKEN_FIELDS, GlmRolloutModel, served_tokens
from taskforge.sandbox.factories import MachineHost
from taskforge.spec.draft import (
    assemble,
    file,
    grader_environment,
    lower,
    machine,
    requirements,
    session,
)
from tests.conftest import ScriptedStream, fake_source
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
# A chat-completions stream in vLLM's shape for ``return_token_ids`` with ``logprobs``: the prompt ids
# on the first chunk, and each token-carrying chunk with its ids and full logprob entries.
VLLM_CHUNKS: list[dict | str] = [
    {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1760000000,
        "model": "glm-5.3",
        "prompt_token_ids": [151331, 151333, 2001],
        "choices": [
            {"index": 0, "delta": {"role": "assistant", "content": ""}, "logprobs": None, "finish_reason": None}
        ],
    },
    {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1760000000,
        "model": "glm-5.3",
        "choices": [
            {
                "index": 0,
                "delta": {"reasoning": "Plan"},
                "logprobs": {
                    "content": [{"token": "Plan", "logprob": -0.0123, "bytes": [80, 108, 97, 110], "top_logprobs": []}]
                },
                "finish_reason": None,
                "token_ids": [34592],
            }
        ],
    },
    {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1760000000,
        "model": "glm-5.3",
        "choices": [
            {
                "index": 0,
                "delta": {"content": "ok"},
                "logprobs": {
                    "content": [
                        {"token": "ok", "logprob": -0.5, "bytes": [111, 107], "top_logprobs": []},
                        {"token": "<|user|>", "logprob": -1.25, "bytes": None, "top_logprobs": []},
                    ]
                },
                "finish_reason": "stop",
                "token_ids": [1012, 151336],
            }
        ],
    },
    {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1760000000,
        "model": "glm-5.3",
        "choices": [],
        "usage": {"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6},
    },
]


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


@dataclass
class TokenStream:
    """A scripted streamed reply carrying vLLM's ``return_token_ids`` fields; the fake server sends it as is."""

    events: list[dict]
    send_done: bool = True
    stall: None = None


def token_stream(
    prompt: list[int],
    response: list[int],
    content: str = "",
    reasoning: str = "",
    tool_call: tuple[str, str] | None = None,
    finish: str = "stop",
) -> TokenStream:
    delta: dict = {}
    if content:
        delta["content"] = content
    if reasoning:
        delta["reasoning"] = reasoning
    if tool_call is not None:
        name, arguments = tool_call
        delta["tool_calls"] = [
            {"index": 0, "id": "call-0", "type": "function", "function": {"name": name, "arguments": arguments}}
        ]
    logprobs = {"content": [{"token": str(t), "logprob": -0.5} for t in response]}
    return TokenStream(
        [
            {
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
                "prompt_token_ids": prompt,
            },
            {
                "choices": [
                    {"index": 0, "delta": delta, "logprobs": logprobs, "token_ids": response, "finish_reason": finish}
                ]
            },
            {"choices": [], "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(response)}},
        ]
    )


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


def shellsim(task: TaskSpec, max_turns: int = 6) -> LoweredTaskSpec:
    """``task`` on ShellSim under the messages trace, as validation runs every trial."""
    limits = session(
        max_turns=max_turns,
        model_turn_timeout=None,
        command_timeout=30,
        tool_turn_timeout=60,
        total_turn_timeout=None,
        attempt_timeout=None,
        verifier_timeout=30,
        cleanup_timeout=30,
    ).model_copy(update={"trace": TraceMode.MESSAGES})
    return lower(
        task,
        host=MachineHost.LAPTOP,
        task_machine=machine(startup_timeout=30),
        verifier_machine=machine(startup_timeout=30),
        session=limits,
        factories=FACTORIES,
    )


def rollout_engine(model: GlmRolloutModel) -> RolloutEngine:
    return RolloutEngine(model, FACTORIES)


@pytest.fixture
async def fake_client(fake_glm, fast_provider):
    async with GlmClient(fake_source(fake_glm), replace(fast_provider, max_attempts=1)) as client:
        yield client


async def test_two_turn_rollout_records_served_ids_and_replays_reasoning(fake_glm, fake_client):
    fake_glm.responses.append(
        token_stream(
            [1, 2, 3],
            [4, 5],
            reasoning="plan",
            tool_call=("shell", '{"command": "touch /workspace/done.txt"}'),
            finish="tool_calls",
        )
    )
    fake_glm.responses.append(token_stream([1, 2, 3, 4, 5, 6, 7], [8, 9], content="Done."))

    rollout = await rollout_engine(GlmRolloutModel(fake_client, POLICY, call_ledger())).run(shellsim(shell_task()))

    assert (rollout.grade.status, rollout.grade.reward) == (Outcome.GRADED, 1.0)
    assert rollout.metrics[TRACE_METRIC] == TraceMode.MESSAGES.value
    assert rollout.prompt_token_ids == (1, 2, 3)
    assert rollout.response_token_ids == (4, 5, 6, 7, 8, 9)
    assert (rollout.loss_mask, rollout.logprobs) == ((0,) * 6, None)
    assert [step.turn.logprobs for step in rollout.steps] == [(-0.5, -0.5)] * 2
    first, second = fake_glm.requests
    assert (first["return_token_ids"], first["logprobs"]) == (True, True)
    assert [tool["function"]["name"] for tool in first["tools"]] == ["shell"]
    assert second["messages"][1]["reasoning_content"] == "plan"
    assert second["messages"][2]["role"] == "tool"


async def test_length_cut_turn_is_graded_with_length_stop(fake_glm, fake_client):
    fake_glm.responses.append(token_stream([1, 2], [3, 4], content="I will", finish="length"))

    rollout = await rollout_engine(GlmRolloutModel(fake_client, POLICY, call_ledger())).run(shellsim(shell_task()))

    assert rollout.stop_reason == "length"
    assert (rollout.grade.status, rollout.grade.reward) == (Outcome.GRADED, 0.0)
    assert len(fake_glm.requests) == 1


async def test_tool_call_cut_at_the_budget_ends_the_rollout_with_length_unexecuted(fake_glm, fake_client):
    # vLLM's GLM tool parser reports a tool call cut by max_tokens as finish_reason "tool_calls".
    command = '{"command": "touch /workspace/done.txt"}'
    fake_glm.responses.append(token_stream([1, 2], [3, 4, 5], tool_call=("shell", command), finish="tool_calls"))
    policy = LLMPolicy(max_tokens=3, max_continuations=0)

    rollout = await rollout_engine(GlmRolloutModel(fake_client, policy, call_ledger())).run(shellsim(shell_task()))

    assert rollout.stop_reason == "length"
    assert (rollout.grade.status, rollout.grade.reward) == (Outcome.GRADED, 0.0)
    assert len(fake_glm.requests) == 1


async def test_prompt_filling_the_context_raises_generation_limit(fake_glm, fake_client):
    fake_glm.status(400, CONTEXT_ERROR)
    fake_glm.status(400, CONTEXT_ERROR)
    request = ModelRequest(({"role": "user", "content": "x"},), {}, (1, 2), 1)

    with pytest.raises(GenerationLimitReached) as raised:
        await GlmRolloutModel(fake_client, POLICY, call_ledger())(request)

    assert raised.value.prompt_token_ids == (1, 2)


async def test_a_prompt_the_server_retokenized_is_graded_with_the_ids_it_served(fake_glm, fake_client):
    # Recorded live from a GLM-5.3 rollout of the d01 task (turns 3-4): the model
    # sampled "1" ")," "(" "1" inside smul(2**(n-1),(1,0)); the re-rendered prompt encodes the same
    # text canonically as "1" "),(" "1".
    sampled = [12, 16, 701, 7, 16, 11, 15]
    canonical = [12, 16, 23482, 16, 11, 15]
    observation = 154829  # <|observation|>
    fake_glm.responses.append(
        token_stream(
            [1, 2, 3],
            [*sampled, observation],
            tool_call=("shell", '{"command": "touch /workspace/done.txt"}'),
            finish="tool_calls",
        )
    )
    fake_glm.responses.append(token_stream([1, 2, 3, *canonical, observation, 9], [10], content="Done."))

    rollout = await rollout_engine(GlmRolloutModel(fake_client, POLICY, call_ledger())).run(shellsim(shell_task()))

    assert (rollout.grade.status, rollout.grade.reward) == (Outcome.GRADED, 1.0)
    assert rollout.steps[1].turn.prompt_token_ids == (1, 2, 3, *canonical, observation, 9)


async def test_a_stream_without_token_ids_gives_a_turn_with_its_usage(fake_glm, fake_client):
    fake_glm.stream(content="hi", prompt_tokens=7, completion_tokens=2)
    request = ModelRequest(({"role": "user", "content": "x"},), {}, (), None)

    turn = await GlmRolloutModel(fake_client, POLICY, call_ledger())(request)

    assert (turn.text, turn.prompt_token_ids, turn.response_token_ids) == ("hi", (), ())
    assert turn.usage == TokenUsage(prompt_tokens=7, completion_tokens=2)


async def test_served_tokens_survive_the_sdks_reserialised_chunks(fake_glm, fake_client):
    fake_glm.responses.append(ScriptedStream(VLLM_CHUNKS, send_done=True, stall=None))

    completion = await fake_client.complete([{"role": "user", "content": "x"}], POLICY, TOKEN_FIELDS)

    tokens = served_tokens(completion)
    assert (tokens.prompt, tokens.response) == ((151331, 151333, 2001), (34592, 1012, 151336))
    assert tokens.logprobs == (-0.0123, -0.5, -1.25)
    stored = [json.loads(event) for event in completion.attempts[0].events]
    assert stored[2]["choices"][0]["logprobs"]["content"][0]["bytes"] == [111, 107]


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


def turn_record(rollout: RolloutData) -> list[dict[str, Any]]:
    return [
        {
            "prompt_ids": len(step.turn.prompt_token_ids),
            "response_ids": len(step.turn.response_token_ids),
            "logprobs": None if step.turn.logprobs is None else len(step.turn.logprobs),
            "stop_reason": step.turn.stop_reason,
            "usage": step.turn.metadata["usage"],
            "reasoning_chars": len(step.turn.message.get("reasoning_content") or ""),
            "tool_calls": [call["function"]["arguments"] for call in step.turn.message.get("tool_calls") or []],
        }
        for step in rollout.steps
    ]


def write_live_evidence(
    evidence_dir: Path, check: str, purpose: str, wall_time: float, rollouts: list[RolloutData]
) -> Path:
    record = {
        "purpose": purpose,
        "policy": repr(POLICY),
        "wall_time": wall_time,
        "turns": [turn_record(rollout) for rollout in rollouts],
        "grades": [asdict(rollout.grade) for rollout in rollouts],
        "rollouts": [asdict(rollout) for rollout in rollouts],
    }
    evidence_dir.mkdir(parents=True, exist_ok=True)
    path = evidence_dir / f"{check}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(record, indent=1, default=str))
    return path


async def live_rollouts(
    glm_source, live_provider, task: TaskSpec, count: int, max_turns: int
) -> tuple[list[RolloutData], float]:
    """``count`` concurrent rollouts of ``task``, each under its own step, as trials run.

    Also asserts that the ledger holds one ``LLM_CALL`` per turn.
    """
    ledger = ListLedger()
    started = time.monotonic()
    async with GlmClient(glm_source, live_provider) as client:
        model = GlmRolloutModel(client, POLICY, call_ledger(ledger))
        lowered = shellsim(task, max_turns)
        rollouts = await asyncio.gather(
            *(rollout_engine(model.for_step(f"solver/{index}/0")).run(lowered) for index in range(count))
        )
    calls = [entry for entry in ledger.entries if entry.kind == EntryKind.LLM_CALL]
    assert len(calls) == sum(len(rollout.steps) for rollout in rollouts)
    assert all(entry.cause is None and entry.tokens_out for entry in calls)
    return list(rollouts), time.monotonic() - started


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

    assert all(len(rollout.steps) >= 2 for rollout in rollouts), path
    assert any(step.turn.metadata["usage"]["cached_tokens"] > 0 for rollout in rollouts for step in rollout.steps), path
    assert all(rollout.grade.status == Outcome.GRADED for rollout in rollouts), path


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
    assert all(len(rollout.steps) >= 3 for rollout in rollouts), path
    # Long reasoning is replayed only once a later turn follows it, so some turn before a rollout's
    # last must reason at length; how many do depends on the puzzles' luck.
    replayed = [turn for rollout in rollouts for turn in turn_record(rollout)[:-1]]
    assert any(turn["usage"]["reasoning_tokens"] >= LONG_REASONING_TOKENS for turn in replayed), path
    assert sum("\\n" in arguments for turn in turns for arguments in turn["tool_calls"]) >= 3, path
