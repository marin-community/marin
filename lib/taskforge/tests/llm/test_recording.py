# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""LLM_CALL recording of rollout-model, agent and structured calls against the scripted GLM server."""

import json
from dataclasses import replace

import pytest
from pydantic import BaseModel
from rolloutengine.contracts import ModelRequest

from taskforge.ledger.records import EntryKind, Ledger
from taskforge.llm.agent import ReasoningReplay, run_agent
from taskforge.llm.client import ModelUnavailable
from taskforge.llm.glm_client import GlmClient
from taskforge.llm.policy import LLMPolicy, Message
from taskforge.llm.recording import CallLedger, recorded_structured
from taskforge.llm.rollout_model import GlmRolloutModel
from taskforge.llm.structured import StructuredOutputError
from tests.conftest import fake_source

ROLLOUT_POLICY = LLMPolicy(max_continuations=0)
CONVERSATION: tuple[Message, ...] = (
    {"role": "user", "content": "make done.txt"},
    {"role": "assistant", "content": "", "reasoning_content": "plan"},
    {"role": "tool", "tool_call_id": "call-0", "content": "ok"},
)


class Capital(BaseModel):
    city: str


@pytest.fixture
async def client(fake_glm, fast_provider):
    async with GlmClient(fake_source(fake_glm), replace(fast_provider, max_attempts=3)) as glm:
        yield glm


def call_ledger(ledger: Ledger) -> CallLedger:
    return CallLedger(ledger=ledger, workpiece_id="item-1", round=2, step="solver")


async def test_rollout_model_records_retried_statuses_and_keeps_served_ids(fake_glm, client, ledger):
    fake_glm.status(429, "slow down")
    fake_glm.status(503, "busy")
    fake_glm.stream(content="done", prompt_tokens=3, completion_tokens=2, prompt_token_ids=(1, 2, 3), token_ids=(4, 5))
    model = GlmRolloutModel(client, ROLLOUT_POLICY, call_ledger(ledger))

    turn = await model(ModelRequest(CONVERSATION, {}, prefix_token_ids=(1, 2), assistant_message_index=None))

    assert (turn.prompt_token_ids, turn.response_token_ids, turn.text) == ((1, 2, 3), (4, 5), "done")
    [entry] = ledger.entries
    assert (entry.kind, entry.workpiece_id, entry.round, entry.step) == (EntryKind.LLM_CALL, "item-1", 2, "solver")
    assert (entry.tokens_in, entry.tokens_out, entry.finish_reason) == (3, 2, "stop")
    assert entry.cause is None
    assert entry.attrs["turn"] == "1"
    assert entry.attrs["attempts"] == "3"
    assert entry.attrs["attempts_retryable_status"] == "2"
    assert entry.attrs["http_statuses"] == "429,503"


async def test_rollout_model_records_a_call_that_runs_out_of_attempts(fake_glm, client, ledger):
    for _ in range(3):
        fake_glm.status(429, "slow down")
    model = GlmRolloutModel(client, ROLLOUT_POLICY, call_ledger(ledger))

    with pytest.raises(ModelUnavailable):
        await model(ModelRequest(CONVERSATION[:1], {}, prefix_token_ids=(), assistant_message_index=None))

    [entry] = ledger.entries
    assert entry.cause == "ModelUnavailable"
    assert entry.attrs["turn"] == "0"
    assert (entry.attrs["attempts"], entry.attrs["http_statuses"]) == ("3", "429,429,429")


async def test_agent_turns_record_retried_statuses(fake_glm, client, ledger):
    fake_glm.status(429, "slow down")
    fake_glm.stream(content="answer")

    await run_agent(client, LLMPolicy(), CONVERSATION[:1], (), 1, call_ledger(ledger), ReasoningReplay.KEEP)

    [entry] = ledger.entries
    assert (entry.attrs["attempts"], entry.attrs["attempts_retryable_status"]) == ("2", "1")
    assert entry.attrs["http_statuses"] == "429"


async def test_structured_call_span_sums_the_reply_and_its_repair(fake_glm, client, ledger):
    fake_glm.status(503, "busy")
    fake_glm.stream(content='{"town": "Paris"}', prompt_tokens=10, completion_tokens=4)
    fake_glm.stream(content=json.dumps({"city": "Paris"}), prompt_tokens=20, completion_tokens=5)

    result = await recorded_structured(
        client, CONVERSATION[:1], Capital, LLMPolicy(), call_ledger(ledger), {"purpose": "capital"}
    )

    assert result.value == Capital(city="Paris")
    [entry] = ledger.entries
    assert (entry.tokens_in, entry.tokens_out, entry.cause) == (30, 9, None)
    assert (entry.attrs["requests"], entry.attrs["attempts"], entry.attrs["http_statuses"]) == ("2", "3", "503")
    assert entry.attrs["purpose"] == "capital"


async def test_structured_call_that_stays_invalid_records_every_reply(fake_glm, client, ledger):
    fake_glm.stream(content="{}", completion_tokens=4)
    fake_glm.stream(content="{}", completion_tokens=6)

    with pytest.raises(StructuredOutputError):
        await recorded_structured(client, CONVERSATION[:1], Capital, LLMPolicy(), call_ledger(ledger), {})

    [entry] = ledger.entries
    assert (entry.cause, entry.tokens_out, entry.attrs["requests"]) == ("StructuredOutputError", 10, "2")
