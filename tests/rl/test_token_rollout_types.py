# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from marin.inference.types import (
    ExpertLoadAccounting,
    MoeRouterReplayMetadata,
    PolicyIdentity,
    TokenizedRollout,
    TokenizedRolloutBatchRequest,
    TokenizedRolloutBatchResult,
    TokenizedRolloutFailure,
    TokenizedRolloutRequest,
    TokenizerIdentity,
    TokenRolloutAdmissionMetadata,
    TokenRolloutFailureReason,
    TokenRolloutFinishReason,
    TokenRolloutTiming,
    TokenSamplingParameters,
)
from marin.rl.environments.inference_ctx.base import BaseInferenceContext


def _tokenizer() -> TokenizerIdentity:
    return TokenizerIdentity(
        name_or_path="Qwen/Qwen3-8B",
        revision="abc123",
        vocab_size=151936,
        chat_template_hash="sha256:template",
        special_token_ids={"eos": 151645},
    )


def _policy() -> PolicyIdentity:
    return PolicyIdentity(
        policy_name="qwen3-policy",
        checkpoint_ref="gs://checkpoints/qwen3/step-128",
        checkpoint_step=128,
        weight_version="weights-128",
    )


def _request(request_id: str = "req-1") -> TokenizedRolloutRequest:
    return TokenizedRolloutRequest(
        request_id=request_id,
        prompt_token_ids=(10, 20, 30),
        sampling=TokenSamplingParameters(
            max_tokens=4,
            temperature=0.7,
            top_p=0.95,
            top_k=64,
            stop_token_ids=(151645,),
            seed=7,
        ),
        n_generations=2,
    )


def test_tokenized_rollout_batch_preserves_identity_boundaries_and_metadata():
    batch = TokenizedRolloutBatchRequest(
        batch_id="batch-1",
        tokenizer=_tokenizer(),
        policy=_policy(),
        requests=(_request(),),
    )

    rollout = TokenizedRollout(
        request_id="req-1",
        generation_index=1,
        prompt_token_ids=(10, 20, 30),
        completion_token_ids=(40, 50),
        completion_logprobs=(-0.2, -0.4),
        finish_reason=TokenRolloutFinishReason.STOP,
        prompt_mask=(False, False, False),
        completion_mask=(True, True),
        stop_token_id=151645,
        timing=TokenRolloutTiming(prefill=0.01, decode=0.02, total=0.04),
        router_replay=MoeRouterReplayMetadata(
            format="marin-router-replay-v1",
            payload_ref="gs://rollouts/router/batch-1",
            layer_names=("decoder.layers.0.mlp",),
        ),
        expert_load=ExpertLoadAccounting(num_experts=4, tokens_per_expert=(8, 7, 6, 5), capacity=16),
    )
    result = TokenizedRolloutBatchResult(
        batch_id=batch.batch_id,
        tokenizer=batch.tokenizer,
        policy=batch.policy,
        rollouts=(rollout,),
        admission=TokenRolloutAdmissionMetadata(
            queued_tokens=3,
            admitted_tokens=5,
            prefill_admissions=1,
            prefill_prompt_tokens_per_admission=(3,),
            decode_rounds=2,
            backend_request_ids=("levanter-req-1",),
        ),
    )

    assert result.tokenizer.name_or_path == "Qwen/Qwen3-8B"
    assert result.policy.checkpoint_step == 128
    assert result.rollouts[0].token_ids == (10, 20, 30, 40, 50)
    assert result.rollouts[0].loss_mask == (False, False, False, True, True)
    assert result.rollouts[0].router_replay is not None
    assert result.admission.prefill_prompt_tokens_per_admission == (3,)


def test_tokenized_rollout_rejects_misaligned_completion_logprobs():
    with pytest.raises(ValueError, match="completion_token_ids and completion_logprobs"):
        TokenizedRollout(
            request_id="req-1",
            generation_index=0,
            prompt_token_ids=(10,),
            completion_token_ids=(20, 30),
            completion_logprobs=(-0.1,),
            finish_reason=TokenRolloutFinishReason.LENGTH,
            prompt_mask=(False,),
            completion_mask=(True, True),
        )


def test_tokenized_batch_rejects_duplicate_request_ids():
    request = _request("duplicate")
    with pytest.raises(ValueError, match="request_id values must be unique"):
        TokenizedRolloutBatchRequest(
            batch_id="batch-1",
            tokenizer=_tokenizer(),
            policy=_policy(),
            requests=(request, request),
        )


def test_admission_metadata_requires_prefill_counts_to_match():
    with pytest.raises(ValueError, match="prefill_prompt_tokens_per_admission"):
        TokenRolloutAdmissionMetadata(
            prefill_admissions=2,
            prefill_prompt_tokens_per_admission=(128,),
        )


def test_tokenized_rollout_failure_preserves_failed_generation_identity():
    failure = TokenizedRolloutFailure(
        request_id="req-1",
        generation_index=2,
        reason=TokenRolloutFailureReason.RESOURCE_EXHAUSTED,
        message="device allocation failed",
        backend_request_id="levanter-7",
    )

    assert failure.request_id == "req-1"
    assert failure.generation_index == 2
    assert failure.reason is TokenRolloutFailureReason.RESOURCE_EXHAUSTED
    assert failure.backend_request_id == "levanter-7"

    with pytest.raises(ValueError, match="generation_index"):
        TokenizedRolloutFailure(
            request_id="req-1",
            generation_index=-1,
            reason=TokenRolloutFailureReason.BACKEND_ERROR,
        )


def test_base_inference_context_token_rollouts_are_opt_in():
    context = BaseInferenceContext()
    batch = TokenizedRolloutBatchRequest(
        batch_id="batch-1",
        tokenizer=_tokenizer(),
        policy=_policy(),
        requests=(_request(),),
    )

    assert not context.supports_token_rollouts()
    with pytest.raises(NotImplementedError):
        context.generate_token_rollouts(batch)
