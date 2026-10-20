import copy

import pytest
import torch
from transformers import Qwen3_5MoeConfig, Qwen3MoeConfig, Qwen3NextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTopKRouter
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextTopKRouter

from skyrl_train.models.grug_moe import GrugMoeConfig, GrugMoeRouter
from skyrl_train.models.router_instrumentation import (
    instrument_moe_routers,
    observe_router_forwards,
)


def _grug_config() -> GrugMoeConfig:
    return GrugMoeConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        shared_expert_intermediate_size=8,
        num_local_experts=4,
        num_experts_per_tok=2,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=16,
        sliding_window=4,
        qk_mult=1.37,
    )


def _qwen_config() -> Qwen3MoeConfig:
    return Qwen3MoeConfig(
        hidden_size=8,
        intermediate_size=16,
        moe_intermediate_size=4,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        norm_topk_prob=True,
        vocab_size=32,
    )


def test_grug_router_observation_uses_biased_selection_and_native_weights() -> None:
    router = GrugMoeRouter(_grug_config())
    with instrument_moe_routers(router) as instrumentation:
        assert instrumentation.router_count == 1
    with torch.no_grad():
        router.weight.copy_(torch.eye(4, 8))
        router.bias.copy_(torch.tensor([0.0, 0.5, -0.25, 0.75]))
    hidden = torch.tensor([[4.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    observations = []

    with observe_router_forwards(observations.append):
        raw_logits, selected_experts, combine_weights = router(hidden)

    selection_logits = raw_logits + router.bias
    selected_raw_logits = raw_logits.gather(-1, selected_experts)
    expected_weights = selected_raw_logits.sigmoid()
    expected_weights *= 2.5 / expected_weights.sum(dim=-1, keepdim=True)

    assert len(observations) == 1
    observation = observations[0]
    torch.testing.assert_close(observation.router_inputs, hidden)
    torch.testing.assert_close(observation.selection_logits, selection_logits)
    torch.testing.assert_close(observation.selection_log_probs, selection_logits.log_softmax(dim=-1))
    torch.testing.assert_close(observation.natural_selected_experts, selected_experts)
    torch.testing.assert_close(observation.selected_experts, selected_experts)
    torch.testing.assert_close(observation.combine_weights, combine_weights)
    torch.testing.assert_close(combine_weights, expected_weights)


def test_qwen_eager_instrumentation_preserves_output_and_router_gradients() -> None:
    torch.manual_seed(17)
    reference = Qwen3MoeSparseMoeBlock(_qwen_config())
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.normal_(mean=0.0, std=0.1)
    instrumented = copy.deepcopy(reference)
    hidden = torch.randn(2, 3, 8)
    expected = reference(hidden)
    observations = []

    with instrument_moe_routers(instrumented) as instrumentation:
        assert instrumentation.router_count == 1
        with observe_router_forwards(observations.append):
            actual = instrumented(hidden)

    router_logits, routing_weights, selected_experts = instrumented.gate(hidden.reshape(-1, hidden.shape[-1]))
    assert len(observations) == 1
    observation = observations[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(observation.router_inputs, hidden.reshape(-1, hidden.shape[-1]))
    torch.testing.assert_close(observation.selection_logits, router_logits)
    torch.testing.assert_close(observation.selection_log_probs, router_logits.float().log_softmax(dim=-1))
    torch.testing.assert_close(observation.natural_selected_experts, selected_experts)
    torch.testing.assert_close(observation.selected_experts, selected_experts)
    torch.testing.assert_close(observation.combine_weights, routing_weights)

    actual.sum().backward()
    assert torch.count_nonzero(instrumented.gate.weight.grad) > 0


@pytest.mark.parametrize(
    ("config_type", "router_type"),
    [
        (Qwen3NextConfig, Qwen3NextTopKRouter),
        (Qwen3_5MoeConfig, Qwen3_5MoeTopKRouter),
    ],
)
def test_other_qwen_router_variants_preserve_their_native_return_tuple(config_type, router_type) -> None:
    config = config_type(hidden_size=8, num_experts=4, num_experts_per_tok=2, norm_topk_prob=True)
    router = router_type(config)
    with torch.no_grad():
        router.weight.normal_(mean=0.0, std=0.1)
    hidden = torch.randn(2, 3, 8)
    expected = router(hidden)
    observations = []

    with instrument_moe_routers(router) as instrumentation:
        assert instrumentation.router_count == 1
        with observe_router_forwards(observations.append):
            actual = router(hidden)

    assert len(observations) == 1
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
    observation = observations[0]
    torch.testing.assert_close(observation.router_inputs, hidden.reshape(-1, hidden.shape[-1]))
    torch.testing.assert_close(observation.selection_logits, actual[0])
    torch.testing.assert_close(observation.natural_selected_experts, actual[2])
    torch.testing.assert_close(observation.combine_weights, actual[1])
