# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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
from marin.rl.environments.inference_ctx import vllm as vllm_context
from marin.rl.environments.inference_ctx.vllm import VLLMFallbackSamplingConfig, vLLMInferenceContext


class FakeSamplingParams:
    def __init__(
        self,
        *,
        temperature,
        n,
        max_tokens,
        logprobs,
        include_stop_str_in_output,
        top_p=None,
        top_k=None,
        stop_token_ids=None,
        seed=None,
    ):
        self.temperature = temperature
        self.n = n
        self.max_tokens = max_tokens
        self.logprobs = logprobs
        self.include_stop_str_in_output = include_stop_str_in_output
        self.top_p = top_p
        self.top_k = top_k
        self.stop_token_ids = stop_token_ids
        self.seed = seed


class FakeTokensPrompt:
    def __init__(self, *, prompt_token_ids):
        self.prompt_token_ids = prompt_token_ids


class FakeLLM:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = []

    def generate(self, prompts, sampling_params):
        self.calls.append((prompts, sampling_params))
        return self.outputs


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


def _logprob(logprob: float, rank: int = 1):
    return SimpleNamespace(logprob=logprob, rank=rank)


def _vllm_output(request_id: str, prompt_token_ids: tuple[int, ...]):
    return SimpleNamespace(
        request_id=f"vllm-{request_id}",
        prompt_token_ids=prompt_token_ids,
        outputs=[
            SimpleNamespace(
                token_ids=(40, 41),
                logprobs=({40: _logprob(-0.2)}, {41: _logprob(-0.3)}),
                finish_reason="stop",
            ),
            SimpleNamespace(
                token_ids=(50,),
                logprobs=({50: _logprob(-0.4)},),
                finish_reason="length",
            ),
        ],
    )


def _context(outputs) -> vLLMInferenceContext:
    context = object.__new__(vLLMInferenceContext)
    context.llm = FakeLLM(outputs)
    context.fallback_sampling = VLLMFallbackSamplingConfig()
    context._use_final_only = False
    return context


def test_vllm_token_rollouts_preserve_request_identity_and_boundaries(monkeypatch):
    monkeypatch.setattr(vllm_context, "SamplingParams", FakeSamplingParams)
    monkeypatch.setattr(vllm_context, "TokensPrompt", FakeTokensPrompt)
    batch = TokenizedRolloutBatchRequest(
        batch_id="batch-1",
        tokenizer=_tokenizer(),
        policy=_policy(),
        requests=(
            _request("req-a", (10, 20)),
            _request("req-b", (30,)),
        ),
    )
    context = _context(
        (
            _vllm_output("req-a", (10, 20)),
            _vllm_output("req-b", (30,)),
        )
    )

    result = context.generate_token_rollouts(batch)

    assert context.supports_token_rollouts()
    assert len(result.rollouts) == 4
    assert result.rollouts[0].request_id == "req-a"
    assert result.rollouts[0].generation_index == 0
    assert result.rollouts[0].prompt_token_ids == (10, 20)
    assert result.rollouts[0].completion_token_ids == (40, 41)
    assert result.rollouts[0].completion_logprobs == (-0.2, -0.3)
    assert result.rollouts[0].loss_mask == (False, False, True, True)
    assert result.rollouts[0].finish_reason == TokenRolloutFinishReason.STOP
    assert result.rollouts[1].finish_reason == TokenRolloutFinishReason.LENGTH
    assert result.admission.queued_tokens == 3
    assert result.admission.admitted_tokens == 9
    assert result.admission.backend_request_ids == ("vllm-req-a", "vllm-req-b")

    prompts, sampling_params = context.llm.calls[0]
    assert [prompt.prompt_token_ids for prompt in prompts] == [[10, 20], [30]]
    assert sampling_params.n == 2
    assert sampling_params.max_tokens == 4
    assert sampling_params.top_k == 64
    assert sampling_params.stop_token_ids == [151645]


def test_vllm_token_rollouts_report_missing_generations(monkeypatch):
    monkeypatch.setattr(vllm_context, "SamplingParams", FakeSamplingParams)
    monkeypatch.setattr(vllm_context, "TokensPrompt", FakeTokensPrompt)
    output = _vllm_output("req-a", (10, 20))
    output.outputs = output.outputs[:1]
    batch = TokenizedRolloutBatchRequest(
        batch_id="batch-1",
        tokenizer=_tokenizer(),
        policy=_policy(),
        requests=(_request("req-a", (10, 20)),),
    )
    context = _context((output,))

    result = context.generate_token_rollouts(batch)

    assert len(result.rollouts) == 1
    assert len(result.failures) == 1
    assert result.failures[0].request_id == "req-a"
    assert result.failures[0].generation_index == 1
    assert result.failures[0].reason is TokenRolloutFailureReason.BACKEND_ERROR
    assert result.failures[0].backend_request_id == "vllm-req-a"


def test_vllm_token_rollouts_require_uniform_sampling(monkeypatch):
    monkeypatch.setattr(vllm_context, "SamplingParams", FakeSamplingParams)
    request = _request("req-a", (10,))
    different_sampling_request = TokenizedRolloutRequest(
        request_id="req-b",
        prompt_token_ids=(20,),
        sampling=TokenSamplingParameters(max_tokens=8, temperature=0.7),
        n_generations=2,
    )
    batch = TokenizedRolloutBatchRequest(
        batch_id="batch-1",
        tokenizer=_tokenizer(),
        policy=_policy(),
        requests=(request, different_sampling_request),
    )
    context = _context(())

    with pytest.raises(ValueError, match="identical sampling parameters"):
        context.generate_token_rollouts(batch)


def test_vllm_token_rollouts_reject_missing_logprobs():
    output = SimpleNamespace(
        token_ids=(40,),
        logprobs=None,
        finish_reason="stop",
    )

    with pytest.raises(ValueError, match="missing logprobs"):
        vLLMInferenceContext._selected_logprobs_from_vllm_output(output)
