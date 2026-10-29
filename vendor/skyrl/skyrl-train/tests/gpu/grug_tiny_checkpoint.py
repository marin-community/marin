"""Tiny Grug checkpoint fixture shared by the Grug GPU tests."""

from pathlib import Path

import torch
from transformers import AutoTokenizer

from skyrl_train.models.grug_moe import GrugMoeConfig, GrugMoeForCausalLM
from skyrl_train.models.grug_moe import GRUG_ROUTER_BIAS_SUFFIX

TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"
TOY_SHAPE = dict(
    hidden_size=64,
    intermediate_size=64,
    shared_expert_intermediate_size=64,
    num_local_experts=8,
    num_hidden_layers=8,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=64,
    sliding_window=16,
)


def router_bias_names(num_layers: int) -> list[str]:
    """Per-layer router-bias buffer names, which carry no gradient and are excluded."""
    return [f"model.layers.{index}{GRUG_ROUTER_BIAS_SUFFIX}" for index in range(num_layers)]


def write_tiny_checkpoint(
    path: Path,
    max_position_embeddings: int = 128,
    num_experts_per_tok: int = 2,
    shape: dict | None = None,
    vocab_size_multiple: int = 1,
    router_weight_std: float | None = None,
) -> None:
    """Write the deterministic tiny Grug checkpoint used across the GPU gates."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    shape = TOY_SHAPE if shape is None else shape
    config = GrugMoeConfig(
        vocab_size=((len(tokenizer) + vocab_size_multiple - 1) // vocab_size_multiple) * vocab_size_multiple,
        num_experts_per_tok=num_experts_per_tok,
        max_position_embeddings=max_position_embeddings,
        initializer_range=0.02,
        qk_mult=1.37,
        qk_mult_long_scale=1.1,
        **shape,
    )
    torch.manual_seed(17)
    model = GrugMoeForCausalLM(config)
    with torch.no_grad():
        # Non-trivial gates and router biases so the Megatron port has to reproduce them.
        for module in model.modules():
            if module.__class__.__name__ == "GrugMoeGatedNorm":
                module.down_proj.weight.normal_(std=0.2)
                module.up_proj.weight.normal_(std=0.2)
        for layer in model.model.layers:
            layer.self_attn.attn_gate.weight.normal_(std=0.2)
            layer.mlp.router.bias.copy_(torch.linspace(-0.3, 0.3, config.num_local_experts))
            if router_weight_std is not None:
                layer.mlp.router.weight.normal_(std=router_weight_std)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
