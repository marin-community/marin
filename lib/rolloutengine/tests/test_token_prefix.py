# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned token contexts reach the real engine unchanged and grade only sampled continuations."""

import pytest
from taskcompendium.grader import verifyit_package
from taskcompendium.models import AnswerType, TextMessage, TokenizedInput
from verifyit.closure_scoring import ClosureFormat
from verifyit.spec import StructureClosureSpec

from rolloutengine.engine import RolloutEngine
from rolloutengine.harness.external import AgentHarness
from rolloutengine.harness.shell import ShellHarness, task_prompt
from rolloutengine.harness.step import StepHarness
from rolloutengine.harness.token_prefix import TokenPrefixHarness
from rolloutengine.models import ModelTurn, RolloutContractError
from rolloutengine.test_support import arithmetic_task, rollout_limits, rollout_spec

TOKENIZER_MODEL = "org/pinned-tokenizer"
TOKENIZER_REVISION = "a" * 40
PROMPT_TOKENS = (128000, 42, 128005)


@pytest.fixture
def token_task():
    return arithmetic_task(
        answer_type=AnswerType.TEXT,
        context=TokenizedInput(
            events=(TextMessage(role="user", content='Complete this JSON prefix: {"x":'),),
            token_ids=PROMPT_TOKENS,
            tokenizer_model=TOKENIZER_MODEL,
            tokenizer_revision=TOKENIZER_REVISION,
        ),
        grader=verifyit_package(
            StructureClosureSpec(
                prefix='{"x":', expected='{"x":1}', format=ClosureFormat.JSON, terminal_markers=("<|eot_id|>",)
            )
        ).grader,
    )


@pytest.mark.parametrize("alter_prefix", [False, True])
async def test_token_prefix_engine_preserves_exact_tokens_and_rejects_transport_changes(token_task, alter_prefix):
    requests = []

    async def model(request):
        requests.append(request)
        served = (*PROMPT_TOKENS[:-1], 99) if alter_prefix else PROMPT_TOKENS
        return ModelTurn({"role": "assistant", "content": "1}"}, served, (5, 6), (-0.1, -0.2), "stop", text="1}")

    engine = RolloutEngine(
        {}, {"tokens": TokenPrefixHarness(tokenizer_model=TOKENIZER_MODEL, tokenizer_revision=TOKENIZER_REVISION)}
    )
    spec = rollout_spec(token_task, harness="tokens")
    if alter_prefix:
        with pytest.raises(RolloutContractError, match="replay token prefix"):
            await engine.run_task(spec, model)
        return
    result = await engine.run_task(spec, model)
    assert len(requests) == 1
    assert requests[0].messages == ({"role": "user", "content": 'Complete this JSON prefix: {"x":'},)
    assert requests[0].options == {"skip_special_tokens": False}
    assert requests[0].prompt_token_ids == PROMPT_TOKENS
    assert result.grade.reward == 1.0
    record = result.rollout.conversations[0]
    assert (record.prompt_token_ids, record.response_token_ids, record.loss_mask) == (PROMPT_TOKENS, (5, 6), (1, 1))
    assert record.logprobs == (-0.1, -0.2)


async def test_token_prefix_tokenizer_mismatch_fails_before_model_request(token_task):
    async def model(_request):
        raise AssertionError("A mismatched tokenizer must be rejected before inference")

    engine = RolloutEngine(
        {}, {"tokens": TokenPrefixHarness(tokenizer_model=TOKENIZER_MODEL, tokenizer_revision="b" * 40)}
    )
    with pytest.raises(ValueError, match="tokenizer pin"):
        await engine.run_task(rollout_spec(token_task, harness="tokens"), model)


@pytest.mark.parametrize(
    "message",
    [
        {"role": "assistant", "content": "1}"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call", "type": "function", "function": {"name": "write", "arguments": "{}"}}],
        },
    ],
)
async def test_token_prefix_engine_rejects_parsed_or_stripped_raw_completion(token_task, message):
    async def model(_request):
        return ModelTurn(message, PROMPT_TOKENS, (5, 6), (-0.1, -0.2), "stop", text="<think>1}")

    engine = RolloutEngine(
        {}, {"tokens": TokenPrefixHarness(tokenizer_model=TOKENIZER_MODEL, tokenizer_revision=TOKENIZER_REVISION)}
    )
    with pytest.raises(RolloutContractError, match="raw completion text"):
        await engine.run_task(rollout_spec(token_task, harness="tokens"), model)


def test_text_rendering_harnesses_reject_authoritative_token_context(token_task):
    async def agent(_context, _complete):
        raise AssertionError("A text agent must not start for an authoritative token context")

    for harness in [ShellHarness(command_timeout=None), AgentHarness(agent), StepHarness(lambda _context: None)]:
        with pytest.raises(NotImplementedError, match="token-preserving"):
            harness.check(token_task, rollout_limits())
    with pytest.raises(NotImplementedError, match="token-preserving"):
        task_prompt(token_task, None, ())
