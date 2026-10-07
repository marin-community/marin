# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Live validation of taskforge.llm against GLM-5.3 on the interactive tier.

Every run writes its exact first request body (bearer token excluded), full completions (raw stream
events included), and measurements to ``lib/taskforge/.evidence/llm/runs/<check>-<utc>.json``. Only
a run whose assertions pass is copied to ``lib/taskforge/.evidence/llm/<check>.json``, so a failing
run never overwrites cited evidence.
"""

import asyncio
import json
import shutil
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import BaseModel, TypeAdapter, field_validator
from rolloutengine.contracts import ModelRequest

from taskforge.ledger.jsonl import JsonlLedger, read_entries
from taskforge.ledger.records import EntryKind
from taskforge.llm.client import (
    AttemptOutcome,
    Completion,
    FinishReason,
    GlmClient,
    GlmEndpoint,
    Pool,
    complete_prefilled,
    prefill_request_fields,
    request_body,
)
from taskforge.llm.policy import GLM_MAX_OUTPUT_TOKENS, LLMPolicy, ReasoningEffort, prefilled_messages
from taskforge.llm.recording import CallLedger
from taskforge.llm.rollout_model import GlmRolloutModel
from taskforge.llm.structured import StructuredTool, complete_structured

EVIDENCE_DIR = Path(__file__).resolve().parents[2] / ".evidence" / "llm"
COMPLETIONS = TypeAdapter(tuple[Completion, ...])
CONTEXT_TOKENS = 262_144


def endpoint(glm_settings) -> GlmEndpoint:
    return GlmEndpoint(base_url=glm_settings.base_url, token=glm_settings.token, pool=Pool.HIGH)


def call(glm_settings, messages, policy) -> Completion:
    async def go() -> Completion:
        async with GlmClient(endpoint(glm_settings)) as client:
            return await client.complete(messages, policy)

    return asyncio.run(go())


def summary(completion: Completion) -> dict[str, object]:
    return {
        "finish_reason": completion.finish_reason,
        "usage": asdict(completion.usage),
        "wall_time": completion.wall_time,
        "ttft": completion.ttft,
        "decode_time": completion.decode_time,
        "decode_tokens_per_second": completion.decode_tokens_per_second,
        "continuations": completion.continuations,
        "attempts": [(a.segment, a.outcome, a.max_tokens, a.http_status) for a in completion.attempts],
    }


def record(check: str, purpose: str, body: dict[str, object], completions, **extra) -> Path:
    """Write this run's evidence under ``runs/`` and return its path."""
    runs = EVIDENCE_DIR / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    evidence = {
        "check": check,
        "purpose": purpose,
        "first_request_body": body,
        "summaries": [summary(c) for c in completions],
        "completions": COMPLETIONS.dump_python(tuple(completions), mode="json"),
        **extra,
    }
    path = runs / f"{check}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n")
    return path


def promote(run: Path, check: str) -> None:
    shutil.copyfile(run, EVIDENCE_DIR / f"{check}.json")


def segment_deltas(completion: Completion, segment: int) -> tuple[str, str]:
    """The reasoning and content streamed by ``segment``'s completed attempt."""
    attempt = next(a for a in completion.attempts if a.segment == segment and a.outcome is AttemptOutcome.COMPLETED)
    deltas = [c["delta"] for e in attempt.events for c in json.loads(e).get("choices", [])]
    return "".join(d.get("reasoning") or "" for d in deltas), "".join(d.get("content") or "" for d in deltas)


@pytest.mark.live_glm
@pytest.mark.timeout(900)
def test_plain_completion_at_model_max_tokens(glm_settings):
    messages = [{"role": "user", "content": "Explain in about 300 words how a hash table resolves collisions."}]
    policy = LLMPolicy()
    completion = call(glm_settings, messages, policy)
    run = record(
        "a_plain_completion",
        "plain completion, usage, finish_reason, decode tok/s",
        request_body(glm_settings.model, messages, policy.max_tokens, policy, {}),
        [completion],
    )
    assert completion.finish_reason is FinishReason.STOP
    assert completion.usage.completion_tokens > 0 and completion.content
    assert completion.attempts[-1].max_tokens == GLM_MAX_OUTPUT_TOKENS
    assert completion.attempts[-1].http_status == 200
    promote(run, "a_plain_completion")


@pytest.mark.live_glm
@pytest.mark.timeout(900)
def test_long_prompt_lowers_max_tokens_to_remaining_context(glm_settings):
    # ~250k filler tokens: 131072 output tokens cannot fit, and the ~12k left bounds the reply even
    # if the model degenerates into repeating the filler.
    instruction = "Reply with the single word OK. Everything between the markers is filler; ignore it."
    messages = [{"role": "user", "content": f"{instruction}\n<filler>\n{'apple ' * 250_000}\n</filler>\n{instruction}"}]
    policy = LLMPolicy(reasoning_effort=ReasoningEffort.LOW, temperature=0.0, max_continuations=0)
    completion = call(glm_settings, messages, policy)
    elided = [{"role": "user", "content": f"{instruction}\\n<filler>\\n'apple ' * 250000\\n</filler>\\n{instruction}"}]
    run = record(
        "b_context_overflow",
        "max_tokens=131072 with a ~250k-token prompt: the server rejects, the client probes and lowers it",
        request_body(glm_settings.model, elided, policy.max_tokens, policy, {}),
        [completion],
    )
    first, probe, final = completion.attempts
    assert (first.outcome, first.http_status, first.max_tokens) == (
        AttemptOutcome.CONTEXT_OVERFLOW,
        400,
        GLM_MAX_OUTPUT_TOKENS,
    )
    assert (probe.segment, probe.outcome, probe.max_tokens) == (-1, AttemptOutcome.COMPLETED, 1)
    assert (final.segment, final.outcome) == (0, AttemptOutcome.COMPLETED)
    assert final.max_tokens == CONTEXT_TOKENS - completion.usage.prompt_tokens
    assert completion.usage.completion_tokens <= final.max_tokens
    assert completion.finish_reason in (FinishReason.STOP, FinishReason.LENGTH)
    promote(run, "b_context_overflow")


@pytest.mark.live_glm
@pytest.mark.timeout(900)
def test_truncated_answer_is_continued(glm_settings):
    messages = [
        {
            "role": "user",
            "content": (
                "Write the numbers one through thirty in English words, one per line, each followed by a "
                "dash and a short fact about that number. Then a final line 'END'."
            ),
        }
    ]
    policy = LLMPolicy(max_tokens=150, reasoning_effort=ReasoningEffort.LOW, max_continuations=8)
    completion = call(glm_settings, messages, policy)
    run = record(
        "c_continue_on_length",
        "max_tokens=150 forces truncation; continuation concatenates",
        request_body(glm_settings.model, messages, policy.max_tokens, policy, {}),
        [completion],
    )
    assert completion.continuations >= 1
    assert completion.finish_reason is FinishReason.STOP
    assert completion.content.rstrip().endswith("END")
    promote(run, "c_continue_on_length")


@pytest.mark.live_glm
@pytest.mark.timeout(900)
def test_reply_cut_off_while_reasoning_is_continued(glm_settings):
    messages = [
        {
            "role": "user",
            "content": (
                "How many integers from 1 to 1000 are divisible by 3 or 5 but not by 7? " "Answer with the number only."
            ),
        }
    ]
    policy = LLMPolicy(max_tokens=32, reasoning_effort=ReasoningEffort.HIGH, temperature=0.0, max_continuations=32)
    completion = call(glm_settings, messages, policy)
    first_reasoning, first_content = segment_deltas(completion, 0)
    run = record(
        "c_continue_mid_reasoning",
        "max_tokens=32 cuts the first segment off while reasoning; continuation carries reasoning_content",
        request_body(glm_settings.model, messages, policy.max_tokens, policy, {}),
        [completion],
        first_segment={"reasoning": first_reasoning, "content": first_content},
        expected_answer=401,
    )
    assert first_reasoning and not first_content
    assert completion.continuations >= 1
    assert completion.finish_reason is FinishReason.STOP
    assert completion.content.strip()
    promote(run, "c_continue_mid_reasoning")


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
def test_structured_output(glm_settings):
    tool = StructuredTool(name="record_capital", description="Record a country's capital.", output_type=Capital)
    messages = [{"role": "user", "content": "What is the capital of Japan and its city population in millions?"}]
    policy = LLMPolicy(reasoning_effort=ReasoningEffort.LOW)

    async def go():
        async with GlmClient(endpoint(glm_settings)) as client:
            return await complete_structured(client, messages, policy, tool)

    result = asyncio.run(go())
    run = record(
        "d_structured",
        "forced strict tool call validated into a pydantic model",
        request_body(glm_settings.model, messages, policy.max_tokens, policy, tool.request_fields()),
        result.completions,
        value=result.value.model_dump(),
    )
    assert result.value.city == "Tokyo"
    promote(run, "d_structured")


@pytest.mark.live_glm
@pytest.mark.timeout(900)
def test_structured_repair_keeps_prior_output(glm_settings):
    tool = StructuredTool(name="record_capital", description="Record a capital city.", output_type=ShoutedCapital)
    messages = [{"role": "user", "content": "Record the capital of France."}]
    policy = LLMPolicy(reasoning_effort=ReasoningEffort.LOW)

    async def go():
        async with GlmClient(endpoint(glm_settings)) as client:
            return await complete_structured(client, messages, policy, tool)

    result = asyncio.run(go())
    run = record(
        "d_structured_repair",
        "validator the schema cannot express; expect one repair turn carrying the prior output",
        request_body(glm_settings.model, messages, policy.max_tokens, policy, tool.request_fields()),
        result.completions,
        value=result.value.model_dump(),
    )
    assert result.value.city == "PARIS"
    promote(run, "d_structured_repair")


@pytest.mark.live_glm
@pytest.mark.timeout(900)
def test_prefilled_answer_continues_the_prefix_without_thinking(glm_settings):
    messages = [
        {
            "role": "user",
            "content": (
                "Propose one programming task. Reply as a markdown document that starts with YAML front "
                "matter holding id, title and difficulty, followed by a one-paragraph description."
            ),
        }
    ]
    policy = LLMPolicy()
    prefix = "---\nid:"

    async def go() -> Completion:
        async with GlmClient(endpoint(glm_settings)) as client:
            return await complete_prefilled(client, messages, policy, prefix)

    completion = asyncio.run(go())
    run = record(
        "g_prefilled",
        "front matter prefilled as the start of the assistant turn; the model continues it as content, no reasoning",
        request_body(
            glm_settings.model,
            prefilled_messages(messages, prefix),
            policy.max_tokens,
            policy,
            prefill_request_fields(policy, None),
        ),
        [completion],
    )
    assert completion.finish_reason is FinishReason.STOP
    assert completion.content.startswith(prefix)
    assert "\ntitle:" in completion.content and completion.content.count("---") >= 2
    assert (completion.reasoning, completion.usage.reasoning_tokens) == ("", 0)
    promote(run, "g_prefilled")


@pytest.mark.live_glm
@pytest.mark.timeout(900)
def test_rollout_model_turn_is_recorded_in_the_ledger(glm_settings):
    ledger = JsonlLedger(EVIDENCE_DIR / "ledger" / "h_rollout_model_record")
    record_to = CallLedger(
        ledger=ledger, item_id=f"live-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}", round=0, step="solver"
    )
    request = ModelRequest(
        ({"role": "user", "content": "Name the largest planet in the solar system in one word."},),
        {},
        prefix_token_ids=(),
        assistant_message_index=None,
    )

    async def go():
        async with GlmClient(endpoint(glm_settings)) as client:
            return await GlmRolloutModel(client, LLMPolicy(max_continuations=0), record_to)(request)

    turn = asyncio.run(go())
    [entry] = list(read_entries(ledger.path_for(record_to.item_id)))
    assert turn.response_token_ids and turn.prompt_token_ids
    assert (entry.kind, entry.step, entry.cause) == (EntryKind.LLM_CALL, "solver", None)
    assert (entry.tokens_in, entry.tokens_out) == (len(turn.prompt_token_ids), len(turn.response_token_ids))
    assert entry.attrs["turn"] == "0" and int(entry.attrs["attempts"]) >= 1
