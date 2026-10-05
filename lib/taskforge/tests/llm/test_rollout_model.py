# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GlmRolloutModel against a scripted GLM server through RolloutEngine, and live on GLM-5.3.

The live check writes ``<evidence_root>/validate/rollout_model/<utc>.json``: every rollout
(messages, token ids, logprobs, per-turn usage) plus, per turn, whether the next prompt preserved
the served prefix and how many prompt tokens the server reported cached. ``evidence_root`` is the
fixture in ``tests/conftest.py``.
"""

import asyncio
import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest
from rolloutengine.contracts import GenerationLimitReached, ModelRequest, RolloutContractError, RolloutData
from rolloutengine.engine import ShellboxRolloutEngine
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from taskcompendium.environment import EnvironmentKind, ExitCodeReward
from taskcompendium.grading import Outcome
from taskcompendium.models import AnswerType, Source, TaskSpec
from taskcompendium.submission import AnswerFormat, SubmissionConvention

from taskforge.llm.client import GlmClient, GlmEndpoint, Pool
from taskforge.llm.policy import LLMPolicy
from taskforge.llm.rollout_model import TOKEN_FIELDS, GlmRolloutModel, served_tokens
from taskforge.spec.draft import assemble, environment, file, shell_verifier

POLICY = LLMPolicy(max_continuations=0)
CONTEXT_ERROR = "This model's maximum context length is 262144 tokens. However, you requested 300000 tokens."
# ShellSim cannot expand a command substitution inside a test argument, so assign it first.
COUNT_CHECK = 'v=$(tr -d " \\n" < /workspace/count.txt)\n[ "$v" = 15 ]\n'
PUZZLE_1 = "List, ascending and one per line, every prime p below 1000 such that p + 2 and p + 6 are both prime.\n"
PUZZLE_2 = (
    "List, ascending and one per line, every three-digit number whose digits are strictly increasing "
    "from left to right and sum to 20.\n"
)
PUZZLE_1_CHECK = (
    'v=$(tr "\\n" " " < /workspace/answer1.txt)\n[ "$v" = "5 11 17 41 101 107 191 227 311 347 461 641 821 857 881 " ]\n'
)


@pytest.fixture
def evidence_dir(evidence_root: Path) -> Path:
    return evidence_root / "validate" / "rollout_model"


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
        environment(EnvironmentKind.SHELLSIM),
        shell_verifier(
            ("sh", "/grader/check.sh"),
            ExitCodeReward(),
            timeout=30,
            files=(file("/grader/check.sh", "test -f /workspace/done.txt\n"),),
        ),
        Source(dataset="taskforge-tests", revision="1", row="0", importer_revision="1"),
    )


def rollout_engine(model: GlmRolloutModel, max_turns: int = 6) -> ShellboxRolloutEngine:
    return ShellboxRolloutEngine(
        model,
        {EnvironmentKind.SHELLSIM: ShellSimMachineFactory()},
        max_turns=max_turns,
        command_timeout=30,
        convention=SubmissionConvention(id="plain", answer_format=AnswerFormat.PLAIN),
    )


@pytest.fixture
async def fake_client(fake_glm):
    async with GlmClient(
        GlmEndpoint(base_url=fake_glm.base_url, token="test", pool=Pool.HIGH), max_attempts=1
    ) as client:
        yield client


async def test_two_turn_rollout_keeps_served_ids_and_replays_reasoning(fake_glm, fake_client):
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

    rollout = await rollout_engine(GlmRolloutModel(fake_client, POLICY)).run(shell_task())

    assert (rollout.grade.status, rollout.grade.reward) == (Outcome.GRADED, 1.0)
    assert rollout.prompt_token_ids == (1, 2, 3)
    assert rollout.response_token_ids == (4, 5, 6, 7, 8, 9)
    assert rollout.loss_mask == (1, 1, 0, 0, 1, 1)
    assert rollout.logprobs == (-0.5, -0.5, 0.0, 0.0, -0.5, -0.5)
    first, second = fake_glm.requests
    assert (first["return_token_ids"], first["logprobs"]) == (True, True)
    assert [tool["function"]["name"] for tool in first["tools"]] == ["shell"]
    assert second["messages"][1]["reasoning_content"] == "plan"
    assert second["messages"][2]["role"] == "tool"


async def test_length_cut_turn_is_graded_with_length_stop(fake_glm, fake_client):
    fake_glm.responses.append(token_stream([1, 2], [3, 4], content="I will", finish="length"))

    rollout = await rollout_engine(GlmRolloutModel(fake_client, POLICY)).run(shell_task())

    assert rollout.stop_reason == "length"
    assert (rollout.grade.status, rollout.grade.reward) == (Outcome.GRADED, 0.0)
    assert len(fake_glm.requests) == 1


async def test_tool_call_cut_at_the_budget_ends_the_rollout_with_length_unexecuted(fake_glm, fake_client):
    # vLLM's GLM tool parser reports a tool call cut by max_tokens as finish_reason "tool_calls".
    command = '{"command": "touch /workspace/done.txt"}'
    fake_glm.responses.append(token_stream([1, 2], [3, 4, 5], tool_call=("shell", command), finish="tool_calls"))
    policy = LLMPolicy(max_tokens=3, max_continuations=0)

    rollout = await rollout_engine(GlmRolloutModel(fake_client, policy)).run(shell_task())

    assert rollout.stop_reason == "length"
    assert (rollout.grade.status, rollout.grade.reward) == (Outcome.GRADED, 0.0)
    assert len(fake_glm.requests) == 1


async def test_prompt_filling_the_context_raises_generation_limit(fake_glm, fake_client):
    fake_glm.status(400, CONTEXT_ERROR)
    fake_glm.status(400, CONTEXT_ERROR)
    request = ModelRequest(({"role": "user", "content": "x"},), {}, (1, 2), 1)

    with pytest.raises(GenerationLimitReached) as raised:
        await GlmRolloutModel(fake_client, POLICY)(request)

    assert raised.value.prompt_token_ids == (1, 2)


async def test_ids_that_disagree_with_usage_break_the_contract(fake_glm, fake_client):
    stream = token_stream([1, 2, 3], [4, 5], content="hi")
    stream.events[-1]["usage"]["completion_tokens"] = 3
    fake_glm.responses.append(stream)
    request = ModelRequest(({"role": "user", "content": "x"},), {}, (), None)

    with pytest.raises(RolloutContractError, match="disagree with usage"):
        await GlmRolloutModel(fake_client, POLICY)(request)


async def test_rollout_fails_naming_tokens_the_server_retokenized(fake_glm, fake_client):
    # Recorded live (.evidence/validate/rollout_model/prefix-bug/record-1.json, turns 3-4): the model
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

    with pytest.raises(RolloutContractError, match=r"index 5 of the 11-token prefix it served \[23482, 16"):
        await rollout_engine(GlmRolloutModel(fake_client, POLICY)).run(shell_task())


async def test_a_continued_completion_has_no_exact_tokens(fake_glm, fake_client):
    fake_glm.responses.append(token_stream([1, 2], [3], content="par", finish="length"))
    fake_glm.responses.append(token_stream([1, 2, 3, 9], [4], content="tial"))

    completion = await fake_client.complete(
        [{"role": "user", "content": "x"}], LLMPolicy(max_continuations=1), dict(TOKEN_FIELDS)
    )

    assert completion.continuations
    with pytest.raises(RolloutContractError, match="without continuation"):
        served_tokens(completion)


async def test_stream_without_prompt_ids_breaks_the_contract(fake_glm, fake_client):
    fake_glm.stream(content="hi")
    request = ModelRequest(({"role": "user", "content": "x"},), {}, (), None)

    with pytest.raises(RolloutContractError, match="prompt_token_ids"):
        await GlmRolloutModel(fake_client, POLICY)(request)


def live_task():
    return assemble(
        "rollout-model-live",
        "Work in /workspace with the shell tool, one command per call. Before each call, reason briefly "
        "about what it must do. First write the primes below 50, one per line, to /workspace/primes.txt. "
        "Then, in a separate call, write the number of lines of that file to /workspace/count.txt. "
        "Then show both files with cat, and say when you are done.",
        AnswerType.FILE,
        environment(EnvironmentKind.SHELLSIM),
        shell_verifier(
            ("sh", "/grader/check.sh"),
            ExitCodeReward(),
            timeout=30,
            files=(file("/grader/check.sh", COUNT_CHECK),),
        ),
        Source(dataset="taskforge-tests", revision="1", row="live", importer_revision="1"),
    )


def turn_record(rollout: RolloutData) -> list[dict[str, object]]:
    turns = []
    served: tuple[int, ...] = ()
    for step in rollout.steps:
        turn = step.turn
        turns.append(
            {
                "prompt_ids": len(turn.prompt_token_ids),
                "response_ids": len(turn.response_token_ids),
                "logprobs": None if turn.logprobs is None else len(turn.logprobs),
                "stop_reason": turn.stop_reason,
                "prefix_preserved": turn.prompt_token_ids[: len(served)] == served,
                "served_prefix_len": len(served),
                "usage": turn.metadata["usage"],
                "reasoning_chars": len(turn.message.get("reasoning_content") or ""),
                "tool_calls": [call["function"]["arguments"] for call in turn.message.get("tool_calls") or []],
            }
        )
        served = turn.prompt_token_ids + turn.response_token_ids
    return turns


def write_live_evidence(
    evidence_dir: Path, name: str, purpose: str, wall_time: float, rollouts: list[RolloutData]
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
    path = evidence_dir / f"{name}{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(record, indent=1, default=str))
    return path


async def live_rollouts(glm_settings, task: TaskSpec, count: int, max_turns: int) -> tuple[list[RolloutData], float]:
    endpoint = GlmEndpoint(base_url=glm_settings.base_url, token=glm_settings.token, pool=Pool.HIGH)
    started = time.monotonic()
    async with GlmClient(endpoint) as client:
        engine = rollout_engine(GlmRolloutModel(client, POLICY), max_turns=max_turns)
        rollouts = await asyncio.gather(*(engine.run(task) for _ in range(count)))
    return list(rollouts), time.monotonic() - started


@pytest.mark.live_glm
@pytest.mark.timeout(1800)
async def test_live_multi_turn_shellsim_rollout_preserves_served_prefix(glm_settings, evidence_dir):
    rollouts, wall_time = await live_rollouts(glm_settings, live_task(), count=3, max_turns=10)
    path = write_live_evidence(
        evidence_dir,
        "",
        "multi-turn ShellSim rollouts through ShellboxRolloutEngine: the engine's served-prefix check "
        "passes on every turn (it raises RolloutContractError otherwise), with reasoning replayed and "
        "prefix caching on",
        wall_time,
        rollouts,
    )

    turns = [turn for rollout in rollouts for turn in turn_record(rollout)]
    assert all(len(rollout.steps) >= 2 for rollout in rollouts), path
    assert all(turn["prefix_preserved"] and turn["logprobs"] == turn["response_ids"] for turn in turns), path
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
        environment(
            EnvironmentKind.SHELLSIM,
            files=(
                file("/workspace/puzzle1.txt", PUZZLE_1),
                file("/workspace/puzzle2.txt", PUZZLE_2),
            ),
        ),
        shell_verifier(
            ("sh", "/grader/check.sh"),
            ExitCodeReward(),
            timeout=30,
            files=(file("/grader/check.sh", PUZZLE_1_CHECK),),
        ),
        Source(dataset="taskforge-tests", revision="1", row="reasoning", importer_revision="1"),
    )


@pytest.mark.live_glm
@pytest.mark.timeout(3600)
async def test_live_reasoning_heavy_rollout_preserves_served_prefix(glm_settings, evidence_dir):
    rollouts, wall_time = await live_rollouts(glm_settings, reasoning_task(), count=3, max_turns=12)
    path = write_live_evidence(
        evidence_dir,
        "reasoning-",
        "reasoning-heavy multi-turn ShellSim rollouts: long reasoning_content replayed on every turn and "
        "multi-line heredoc tool arguments; the engine's served-prefix check passes on every turn",
        wall_time,
        rollouts,
    )

    turns = [turn for rollout in rollouts for turn in turn_record(rollout)]
    assert all(turn["prefix_preserved"] and turn["logprobs"] == turn["response_ids"] for turn in turns), path
    assert all(len(rollout.steps) >= 3 for rollout in rollouts), path
    assert sum(turn["usage"]["reasoning_tokens"] >= 1000 for turn in turns) >= 3, path
    assert sum("\\n" in arguments for turn in turns for arguments in turn["tool_calls"]) >= 3, path
