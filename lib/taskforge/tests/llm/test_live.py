# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Live validation of taskforge.llm against GLM-5.3 on the interactive tier.

Every run writes its exact first request body (bearer token excluded), full completions (raw stream
events included), and measurements to ``<evidence_root>/llm/runs/<check>-<utc>.json``. Only a
run whose assertions pass is copied to ``<evidence_root>/llm/<check>.json``, so a failing run never
overwrites cited evidence. ``evidence_root`` is the fixture in ``tests/conftest.py``.
"""

import asyncio
import json
import shutil
from dataclasses import asdict, replace
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import pytest
from pydantic import BaseModel, Field, TypeAdapter, field_validator
from rolloutengine.contracts import ModelRequest

from taskforge.ledger.jsonl import JsonlLedger, read_entries
from taskforge.ledger.records import EntryKind
from taskforge.llm.client import AttemptOutcome, Completion, ContextExhausted, FinishReason
from taskforge.llm.glm_client import GlmClient, request_kwargs
from taskforge.llm.glm_format import parse_turn
from taskforge.llm.policy import GLM_MAX_OUTPUT_TOKENS, LLMPolicy, Message, ReasoningEffort
from taskforge.llm.recording import CallLedger
from taskforge.llm.rollout_model import TOKEN_FIELDS, GlmRolloutModel, RolloutTransport
from taskforge.llm.structured import response_format
from tests.conftest import GLM_MODEL, unique_run_name

COMPLETIONS = TypeAdapter(tuple[Completion, ...])


@pytest.fixture
def evidence_dir(evidence_root: Path) -> Path:
    return evidence_root / "llm"


def call(glm_source, live_provider, messages, policy) -> Completion:
    async def go() -> Completion:
        async with GlmClient(glm_source, live_provider) as client:
            return await client.complete(messages, policy)

    return asyncio.run(go())


def request_body(messages, policy: LLMPolicy, fields) -> dict[str, object]:
    """The request the client sends, as ``chat.completions.create`` arguments; the bearer token is not in it."""
    return request_kwargs(GLM_MODEL, messages, policy.max_tokens, policy, fields)


def structured_body(messages, output_type: type[BaseModel], policy: LLMPolicy) -> dict[str, object]:
    return request_body(
        messages, replace(policy, max_continuations=0), {"response_format": response_format(output_type)}
    )


def summary(completion: Completion) -> dict[str, object]:
    return {
        "finish_reason": completion.finish_reason,
        "usage": asdict(completion.usage),
        "wall_time": completion.wall_time,
        "ttft": completion.ttft,
        "decode_time": completion.decode_time,
        "decode_tokens_per_second": completion.usage.completion_tokens / completion.decode_time,
        "continuations": completion.continuations,
        "attempts": [(a.segment, a.outcome, a.max_tokens, a.http_status) for a in completion.attempts],
    }


def record(evidence_dir: Path, check: str, purpose: str, body: dict[str, object], completions, **extra) -> Path:
    """Write this run's evidence under ``runs/`` and return its path."""
    runs = evidence_dir / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    evidence = {
        "check": check,
        "purpose": purpose,
        "first_request_body": body,
        "summaries": [summary(c) for c in completions],
        "completions": COMPLETIONS.dump_python(tuple(completions), mode="json"),
        **extra,
    }
    path = runs / f"{unique_run_name(check)}.json"
    path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n")
    return path


def promote(evidence_dir: Path, run: Path, check: str) -> None:
    shutil.copyfile(run, evidence_dir / f"{check}.json")


def segment_deltas(completion: Completion, segment: int) -> tuple[str, str]:
    """The reasoning and content streamed by ``segment``'s completed attempt."""
    attempt = next(a for a in completion.attempts if a.segment == segment and a.outcome is AttemptOutcome.COMPLETED)
    deltas = [c["delta"] for e in attempt.events for c in json.loads(e).get("choices", [])]
    return "".join(d.get("reasoning") or "" for d in deltas), "".join(d.get("content") or "" for d in deltas)


@pytest.mark.live_glm
@pytest.mark.timeout(900)
def test_plain_completion_at_model_max_tokens(glm_source, live_provider, evidence_dir):
    messages: list[Message] = [
        {"role": "user", "content": "Explain in about 300 words how a hash table resolves collisions."}
    ]
    policy = LLMPolicy()
    completion = call(glm_source, live_provider, messages, policy)
    run = record(
        evidence_dir,
        "a_plain_completion",
        "plain completion, usage, finish_reason, decode tok/s",
        request_body(messages, policy, {}),
        [completion],
    )
    assert completion.finish_reason is FinishReason.STOP
    assert completion.usage.completion_tokens > 0 and completion.content
    assert completion.attempts[-1].max_tokens == GLM_MAX_OUTPUT_TOKENS
    assert completion.attempts[-1].http_status == 200
    promote(evidence_dir, run, "a_plain_completion")


@pytest.mark.live_glm
@pytest.mark.timeout(900)
def test_long_prompt_raises_context_exhausted(glm_source, live_provider, evidence_dir):
    # ~250k filler tokens: the prompt plus 131072 output tokens cannot fit the window.
    instruction = "Reply with the single word OK. Everything between the markers is filler; ignore it."
    messages: list[Message] = [
        {"role": "user", "content": f"{instruction}\n<filler>\n{'apple ' * 250_000}\n</filler>\n{instruction}"}
    ]
    policy = LLMPolicy(reasoning_effort=ReasoningEffort.LOW, temperature=0.0, max_continuations=0)
    with pytest.raises(ContextExhausted) as error:
        call(glm_source, live_provider, messages, policy)
    [attempt] = error.value.attempts
    assert (attempt.outcome, attempt.http_status, attempt.max_tokens) == (
        AttemptOutcome.CONTEXT_OVERFLOW,
        400,
        GLM_MAX_OUTPUT_TOKENS,
    )


@pytest.mark.live_glm
@pytest.mark.timeout(900)
def test_truncated_answer_is_continued(glm_source, live_provider, evidence_dir):
    messages: list[Message] = [
        {
            "role": "user",
            "content": (
                "Write the numbers one through thirty in English words, one per line, each followed by a "
                "dash and a short fact about that number. Then a final line 'END'."
            ),
        }
    ]
    policy = LLMPolicy(max_tokens=150, reasoning_effort=ReasoningEffort.LOW, max_continuations=8)
    completion = call(glm_source, live_provider, messages, policy)
    run = record(
        evidence_dir,
        "c_continue_on_length",
        "max_tokens=150 forces truncation; continuation concatenates",
        request_body(messages, policy, {}),
        [completion],
    )
    assert completion.continuations >= 1
    assert completion.finish_reason is FinishReason.STOP
    assert completion.content.rstrip().endswith("END")
    promote(evidence_dir, run, "c_continue_on_length")


@pytest.mark.live_glm
@pytest.mark.timeout(900)
def test_reply_cut_off_while_reasoning_is_continued(glm_source, live_provider, evidence_dir):
    messages: list[Message] = [
        {
            "role": "user",
            "content": (
                "How many integers from 1 to 1000 are divisible by 3 or 5 but not by 7? " "Answer with the number only."
            ),
        }
    ]
    policy = LLMPolicy(max_tokens=32, reasoning_effort=ReasoningEffort.HIGH, temperature=0.0, max_continuations=32)
    completion = call(glm_source, live_provider, messages, policy)
    first_reasoning, first_content = segment_deltas(completion, 0)
    run = record(
        evidence_dir,
        "c_continue_mid_reasoning",
        "max_tokens=32 cuts the first segment off while reasoning; continuation carries reasoning_content",
        request_body(messages, policy, {}),
        [completion],
        first_segment={"reasoning": first_reasoning, "content": first_content},
        expected_answer=401,
    )
    assert first_reasoning and not first_content
    assert completion.continuations >= 1
    assert completion.finish_reason is FinishReason.STOP
    assert completion.content.strip()
    promote(evidence_dir, run, "c_continue_mid_reasoning")


class Capital(BaseModel):
    country: str
    city: str
    population_millions: float


class ShoutedCapital(BaseModel):
    city: str

    @field_validator("city")
    @classmethod
    def uppercase(cls, value: str) -> str:
        if value != value.upper():
            raise ValueError("city must be written entirely in uppercase letters")
        return value


@pytest.mark.live_glm
@pytest.mark.timeout(900)
def test_structured_output(glm_source, live_provider, evidence_dir):
    messages: list[Message] = [
        {"role": "user", "content": "What is the capital of Japan and its city population in millions?"}
    ]
    policy = LLMPolicy(reasoning_effort=ReasoningEffort.LOW)

    async def go():
        async with GlmClient(glm_source, live_provider) as client:
            return await client.complete_structured(messages, Capital, policy)

    result = asyncio.run(go())
    run = record(
        evidence_dir,
        "d_structured",
        "response_format json_schema strict content validated into a pydantic model",
        structured_body(messages, Capital, policy),
        result.completions,
        value=result.value.model_dump(),
    )
    assert result.value.city == "Tokyo"
    promote(evidence_dir, run, "d_structured")


@pytest.mark.live_glm
@pytest.mark.timeout(900)
def test_structured_repair_keeps_prior_output(glm_source, live_provider, evidence_dir):
    messages: list[Message] = [{"role": "user", "content": "Record the capital of France."}]
    policy = LLMPolicy(reasoning_effort=ReasoningEffort.LOW)

    async def go():
        async with GlmClient(glm_source, live_provider) as client:
            return await client.complete_structured(messages, ShoutedCapital, policy)

    result = asyncio.run(go())
    run = record(
        evidence_dir,
        "d_structured_repair",
        "validator the schema cannot express; expect one repair request carrying the prior output",
        structured_body(messages, ShoutedCapital, policy),
        result.completions,
        value=result.value.model_dump(),
    )
    assert result.value.city == "PARIS"
    promote(evidence_dir, run, "d_structured_repair")


class Verdict(StrEnum):
    ACCEPT = "accept"
    REVISE = "revise"
    REJECT = "reject"


Score = Annotated[int, Field(ge=1, le=5)]


class RubricShaped(BaseModel):
    """Seven 1-5 scores, three lists and an enum, the shape of triage's rubric report."""

    clarity: Score
    feasibility: Score
    novelty: Score
    verifiability: Score
    difficulty: Score
    source_honesty: Score
    scope: Score
    strengths: list[str]
    weaknesses: list[str]
    required_changes: list[str]
    verdict: Verdict


STRUCTURED_SAMPLES = 5


@pytest.mark.live_glm
@pytest.mark.timeout(1800)
def test_structured_output_is_decoder_enforced(glm_source, live_provider, evidence_dir):
    messages: list[Message] = [
        {
            "role": "user",
            "content": (
                "Review this task proposal and score it: 'Write a shell script that rotates log files in "
                "/var/log/app, keeping the last seven, compressing older ones with gzip.'"
            ),
        }
    ]
    policy = LLMPolicy()

    async def go():
        async with GlmClient(glm_source, live_provider) as client:
            return await asyncio.gather(
                *(client.complete_structured(messages, RubricShaped, policy) for _ in range(STRUCTURED_SAMPLES))
            )

    results = asyncio.run(go())
    run = record(
        evidence_dir,
        "d_structured_enforced",
        "response_format json_schema strict: every sample validates on the first request",
        structured_body(messages, RubricShaped, policy),
        [completion for result in results for completion in result.completions],
        values=[result.value.model_dump(mode="json") for result in results],
    )
    assert [len(result.completions) for result in results] == [1] * STRUCTURED_SAMPLES
    promote(evidence_dir, run, "d_structured_enforced")


@pytest.mark.live_glm
@pytest.mark.timeout(900)
def test_rollout_model_turn_is_recorded_in_the_ledger(glm_source, live_provider, evidence_dir):
    ledger = JsonlLedger(evidence_dir / "ledger" / "h_rollout_model_record")
    record_to = CallLedger(ledger=ledger, workpiece_id=unique_run_name("live"), round=0, step="solver")
    request = ModelRequest(
        ({"role": "user", "content": "Name the largest planet in the solar system in one word."},),
        {},
        prefix_token_ids=(),
        assistant_message_index=None,
    )

    async def go():
        async with GlmClient(glm_source, live_provider) as client:
            return await GlmRolloutModel(
                client, LLMPolicy(max_continuations=0), RolloutTransport.TOKEN_ROUTE, record_to
            )(request)

    turn = asyncio.run(go())
    [entry] = list(read_entries(ledger.path_for(record_to.workpiece_id)))
    assert turn.response_token_ids and turn.prompt_token_ids
    assert (entry.kind, entry.step, entry.cause) == (EntryKind.LLM_CALL, "solver", None)
    assert (entry.tokens_in, entry.tokens_out) == (len(turn.prompt_token_ids), len(turn.response_token_ids))
    assert entry.attrs["turn"] == "0" and int(entry.attrs["attempts"]) >= 1


ORACLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Run a shell command and return its output.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}},
                "required": ["command"],
            },
        },
    }
]
ORACLE_PROMPTS = {
    "reply": "Say hello in one short sentence. Do not call any tool.",
    "one_call": "List the files in /tmp with the shell tool, with a timeout of 30 seconds. Make exactly one call.",
    "two_calls": (
        "In one reply, make two shell tool calls: first write a three-line Python script to /tmp/a.py with a "
        "heredoc (the command spans several lines), then run it with python3."
    ),
}


def sampled_bytes(completion: Completion) -> list[bytes]:
    """Each sampled token's bytes, from the logprobs of the call's completed request."""
    [attempt] = [a for a in completion.attempts if a.outcome is AttemptOutcome.COMPLETED]
    entries = [
        entry
        for event in attempt.events
        for choice in json.loads(event).get("choices", [])
        for entry in (choice.get("logprobs") or {}).get("content") or ()
    ]
    return [bytes(entry["bytes"]) for entry in entries]


@pytest.mark.live_glm
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("case", sorted(ORACLE_PROMPTS))
def test_live_token_parse_matches_server_parse(glm_source, live_provider, evidence_dir, case):
    """The oracle for ``glm_format.parse_turn``: the server's own parse of the same sampled tokens."""
    messages: list[Message] = [{"role": "user", "content": ORACLE_PROMPTS[case]}]
    policy = LLMPolicy(reasoning_effort=ReasoningEffort.LOW, max_continuations=0)
    fields = {**TOKEN_FIELDS, "tools": ORACLE_TOOLS}

    async def go() -> Completion:
        async with GlmClient(glm_source, live_provider) as client:
            return await client.complete(messages, policy, fields)

    completion = asyncio.run(go())
    parsed = parse_turn(sampled_bytes(completion), ORACLE_TOOLS, "oracle")
    ours = (parsed.reasoning, parsed.content, [(c.name, json.loads(c.arguments)) for c in parsed.tool_calls])
    server = (
        completion.reasoning,
        completion.content,
        [(c.name, json.loads(c.arguments)) for c in completion.tool_calls],
    )
    run = record(
        evidence_dir,
        f"i_token_parse_{case}",
        "glm_format.parse_turn of the sampled token bytes against the server's reasoning and tool parsers",
        request_body(messages, policy, fields),
        [completion],
        ours=ours,
        server=server,
    )
    assert completion.finish_reason is not FinishReason.LENGTH
    assert ours == server
    promote(evidence_dir, run, f"i_token_parse_{case}")
