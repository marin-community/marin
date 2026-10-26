from pathlib import Path

import torch
from transformers import AutoTokenizer

from skyrl_train.models.grug_moe import GrugMoeConfig
from tests.gpu.grug_tiny_checkpoint import TOKENIZER, gated_grug_model


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
    model = gated_grug_model(config)
    with torch.no_grad():
        for layer in model.model.layers:
            # A decode-invariant engine multiplies the router weight in bf16 and refuses values bf16 cannot hold.
            layer.mlp.router.weight.copy_(layer.mlp.router.weight.to(torch.bfloat16))
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    return config
