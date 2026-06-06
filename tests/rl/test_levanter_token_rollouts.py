# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from marin.inference.types import (
    PolicyIdentity,
    TokenizedRolloutBatchRequest,
    TokenizedRolloutRequest,
    TokenizerIdentity,
    TokenRolloutFailureReason,
    TokenRolloutFinishReason,
    TokenSamplingParameters,
)
from marin.rl.environments.inference_ctx import levanter as levanter_context
from marin.rl.environments.inference_ctx.levanter import LevanterInferenceContext


class FakeLock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeEngine:
    def __init__(self, result):
        self.result = result
        self.requests = None

    def generate(self, requests):
        self.requests = requests
        return self.result


def _tokenizer() -> TokenizerIdentity:
    return TokenizerIdentity(name_or_path="Qwen/Qwen3-8B", revision="abc123", vocab_size=151936)


def _policy() -> PolicyIdentity:
    return PolicyIdentity(policy_name="qwen3-policy", checkpoint_ref="gs://checkpoints/qwen3/step-128")


def _sampling() -> TokenSamplingParameters:
    return TokenSamplingParameters(
        max_tokens=4,
        temperature=0.7,
        top_p=0.95,
        top_k=64,
        stop_token_ids=(151645,),
        seed=7,
    )


def _request(request_id: str, prompt_token_ids: tuple[int, ...]) -> TokenizedRolloutRequest:
    return TokenizedRolloutRequest(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        sampling=_sampling(),
        n_generations=2,
    )


def _context(result) -> tuple[LevanterInferenceContext, FakeEngine]:
    engine = FakeEngine(result)
    config = SimpleNamespace(trainer=SimpleNamespace(device_mesh="mesh", compute_axis_mapping={"batch": "data"}))
    inference_context = SimpleNamespace(engine=engine, model_lock=FakeLock(), config=config)
    context = object.__new__(LevanterInferenceContext)
    context._inference_server = SimpleNamespace(inference_context=inference_context)
    return context, engine


def test_levanter_token_rollouts_preserve_identity_and_admission_metadata(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(levanter_context.hax.partitioning, "set_mesh", lambda mesh: nullcontext())
    monkeypatch.setattr(levanter_context.hax, "axis_mapping", lambda mapping: nullcontext())
    result = SimpleNamespace(
        tokens=(
            (40, 41),
            (50,),
            (60, 61, 62, 63),
            (70,),
        ),
        logprobs=(
            (-0.2, -0.3),
            (-0.4,),
            (-0.5, -0.6, -0.7, -0.8),
            (-0.9,),
        ),
        total_generated=8,
        prefill_admissions=2,
        prefill_prompt_tokens_per_admission=(3, 1),
    )
    batch = TokenizedRolloutBatchRequest(
        batch_id="batch-1",
        tokenizer=_tokenizer(),
        policy=_policy(),
        requests=(
            _request("req-a", (10, 20, 30)),
            _request("req-b", (31,)),
        ),
    )
    context, engine = _context(result)

    token_result = context.generate_token_rollouts(batch)

    assert context.supports_token_rollouts()
    assert len(token_result.rollouts) == 4
    assert token_result.rollouts[0].request_id == "req-a"
    assert token_result.rollouts[0].generation_index == 0
    assert token_result.rollouts[0].prompt_token_ids == (10, 20, 30)
    assert token_result.rollouts[0].completion_token_ids == (40, 41)
    assert token_result.rollouts[0].completion_logprobs == (-0.2, -0.3)
    assert token_result.rollouts[0].loss_mask == (False, False, False, True, True)
    assert token_result.rollouts[2].request_id == "req-b"
    assert token_result.rollouts[2].finish_reason == TokenRolloutFinishReason.LENGTH
    assert token_result.admission.queued_tokens == 4
    assert token_result.admission.admitted_tokens == 12
    assert token_result.admission.prefill_admissions == 2
    assert token_result.admission.prefill_prompt_tokens_per_admission == (3, 1)

    assert engine.requests is not None
    assert [request.prompt_tokens for request in engine.requests] == [[10, 20, 30], [31]]
    assert [request.request_id for request in engine.requests] == [0, 1]
    assert [request.n_generations for request in engine.requests] == [2, 2]
    assert all(request.return_logprobs for request in engine.requests)


def test_levanter_token_rollouts_report_missing_generations(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(levanter_context.hax.partitioning, "set_mesh", lambda mesh: nullcontext())
    monkeypatch.setattr(levanter_context.hax, "axis_mapping", lambda mapping: nullcontext())
    result = SimpleNamespace(
        tokens=((40, 41),),
        logprobs=((-0.2, -0.3),),
        total_generated=2,
        prefill_admissions=1,
        prefill_prompt_tokens_per_admission=(2,),
    )
    batch = TokenizedRolloutBatchRequest(
        batch_id="batch-1",
        tokenizer=_tokenizer(),
        policy=_policy(),
        requests=(_request("req-a", (10, 20)),),
    )
    context, _ = _context(result)

    token_result = context.generate_token_rollouts(batch)

    assert len(token_result.rollouts) == 1
    assert len(token_result.failures) == 1
    assert token_result.failures[0].request_id == "req-a"
    assert token_result.failures[0].generation_index == 1
    assert token_result.failures[0].reason is TokenRolloutFailureReason.BACKEND_ERROR
    assert token_result.failures[0].backend_request_id == "1"


def test_levanter_token_rollouts_require_logprobs():
    batch = TokenizedRolloutBatchRequest(
        batch_id="batch-1",
        tokenizer=_tokenizer(),
        policy=_policy(),
        requests=(
            TokenizedRolloutRequest(
                request_id="req-a",
                prompt_token_ids=(10,),
                sampling=TokenSamplingParameters(max_tokens=4, temperature=0.7, return_logprobs=False),
            ),
        ),
    )
    context, _ = _context(SimpleNamespace())

    with pytest.raises(ValueError, match="return_logprobs=True"):
        context.generate_token_rollouts(batch)
