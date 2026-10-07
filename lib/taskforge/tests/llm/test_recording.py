# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""LLM_CALL recording of rollout-model and agent calls against the scripted GLM server."""

from dataclasses import dataclass

import pytest
from rigging.timing import ExponentialBackoff
from rolloutengine.contracts import ModelRequest

from taskforge.ledger.records import EntryKind, LedgerEntry
from taskforge.llm.agent import run_agent
from taskforge.llm.client import GlmClient, GlmEndpoint, GlmUnavailable, Pool
from taskforge.llm.policy import LLMPolicy
from taskforge.llm.recording import CallLedger
from taskforge.llm.rollout_model import GlmRolloutModel

ROLLOUT_POLICY = LLMPolicy(max_continuations=0)
CONVERSATION = (
    {"role": "user", "content": "make done.txt"},
    {"role": "assistant", "content": "", "reasoning_content": "plan"},
    {"role": "tool", "tool_call_id": "call-0", "content": "ok"},
)


class ListLedger:
    def __init__(self) -> None:
        self.entries: list[LedgerEntry] = []

    def record(self, entry: LedgerEntry) -> None:
        self.entries.append(entry)


@dataclass
class TokenStream:
    """A scripted reply carrying vLLM's ``return_token_ids`` fields; the fake server sends it as is."""

    events: list[dict]
    send_done: bool = True
    stall: None = None


def token_stream(prompt: list[int], response: list[int], content: str) -> TokenStream:
    logprobs = {"content": [{"token": str(t), "logprob": -0.5} for t in response]}
    return TokenStream(
        [
            {
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
                "prompt_token_ids": prompt,
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": content},
                        "logprobs": logprobs,
                        "token_ids": response,
                        "finish_reason": "stop",
                    }
                ]
            },
            {"choices": [], "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(response)}},
        ]
    )


@pytest.fixture
async def client(fake_glm):
    endpoint = GlmEndpoint(base_url=fake_glm.base_url, token="test", pool=Pool.HIGH)
    backoff = ExponentialBackoff(initial=0.001, maximum=0.001)
    async with GlmClient(endpoint, max_attempts=3, backoff=backoff) as glm:
        yield glm


def call_ledger(ledger: ListLedger) -> CallLedger:
    return CallLedger(ledger=ledger, item_id="item-1", round=2, step="solver")


async def test_rollout_model_records_retried_statuses_and_keeps_exact_tokens(fake_glm, client):
    fake_glm.status(429, "slow down")
    fake_glm.status(503, "busy")
    fake_glm.responses.append(token_stream([1, 2, 3], [4, 5], "done"))
    ledger = ListLedger()
    model = GlmRolloutModel(client, ROLLOUT_POLICY, call_ledger(ledger))

    turn = await model(ModelRequest(CONVERSATION, {}, prefix_token_ids=(1, 2), assistant_message_index=None))

    assert (turn.prompt_token_ids, turn.response_token_ids, turn.text) == ((1, 2, 3), (4, 5), "done")
    [entry] = ledger.entries
    assert (entry.kind, entry.item_id, entry.round, entry.step) == (EntryKind.LLM_CALL, "item-1", 2, "solver")
    assert (entry.tokens_in, entry.tokens_out, entry.finish_reason) == (3, 2, "stop")
    assert entry.cause is None
    assert entry.attrs["turn"] == "1"
    assert entry.attrs["attempts"] == "3"
    assert entry.attrs["attempts_retryable_status"] == "2"
    assert entry.attrs["http_statuses"] == "429,503"


async def test_rollout_model_records_a_call_that_runs_out_of_attempts(fake_glm, client):
    for _ in range(3):
        fake_glm.status(429, "slow down")
    ledger = ListLedger()
    model = GlmRolloutModel(client, ROLLOUT_POLICY, call_ledger(ledger))

    with pytest.raises(GlmUnavailable):
        await model(ModelRequest(CONVERSATION[:1], {}, prefix_token_ids=(), assistant_message_index=None))

    [entry] = ledger.entries
    assert entry.cause == "GlmUnavailable"
    assert entry.attrs["turn"] == "0"
    assert (entry.attrs["attempts"], entry.attrs["http_statuses"]) == ("3", "429,429,429")


async def test_agent_turns_record_retried_statuses(fake_glm, client):
    fake_glm.status(429, "slow down")
    fake_glm.stream(content="answer")
    ledger = ListLedger()

    await run_agent(client, LLMPolicy(), CONVERSATION[:1], (), 1, call_ledger(ledger))

    [entry] = ledger.entries
    assert (entry.attrs["attempts"], entry.attrs["attempts_retryable_status"]) == ("2", "1")
    assert entry.attrs["http_statuses"] == "429"
