from pathlib import Path

import torch
from transformers import AutoTokenizer

from skyrl_train.models.grug_moe import GrugMoeConfig, GrugMoeForCausalLM, GrugMoeGatedNorm

TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"


def write_snowball_checkpoint(
    path: Path, *, num_hidden_layers: int = 4, num_local_experts: int = 256, vocab_size: int | None = None
) -> GrugMoeConfig:
    """Write a Grug model with Snowball's attention geometry, width, shared-expert width, norm epsilon and 2,048-token
    window, cut to ``num_hidden_layers`` layers with ``num_local_experts`` narrow experts, random weights and
    non-trivial gates; ``vocab_size`` replaces Snowball's 128,256-entry vocabulary."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    vocabulary = {} if vocab_size is None else {"vocab_size": vocab_size}
    config = GrugMoeConfig(
        num_hidden_layers=num_hidden_layers,
        num_local_experts=num_local_experts,
        intermediate_size=128,
        initializer_range=0.02,
        **vocabulary,
    )
    torch.manual_seed(17)
    model = GrugMoeForCausalLM(config)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, GrugMoeGatedNorm):
                module.down_proj.weight.normal_(std=0.2)
                module.up_proj.weight.normal_(std=0.2)
        for layer in model.model.layers:
            layer.self_attn.attn_gate.weight.normal_(std=0.2)
            layer.mlp.router.bias.copy_(torch.linspace(-0.3, 0.3, config.num_local_experts))
            # A decode-invariant engine multiplies the router weight in bf16 and refuses values bf16 cannot hold.
            layer.mlp.router.weight.copy_(layer.mlp.router.weight.to(torch.bfloat16))
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    return config
