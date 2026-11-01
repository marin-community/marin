# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A tiny random GrugMoE on the CPU for the harness and analysis tests."""

import copy

import torch

from experiments.interp.snowball_identity.harness import SnowballHarness, import_grug

LAYERS = 4
HIDDEN = 64
TOKENS = [5, 17, 3, 99, 42, 8, 61, 23, 7, 30]
ASSISTANT_START = 6


def tiny_models() -> tuple[SnowballHarness, torch.nn.Module]:
    """The harness and an untouched copy of the same model (the harness swaps in its own expert execution)."""
    grug = import_grug()
    torch.manual_seed(0)
    config = grug.GrugMoeConfig(
        vocab_size=128,
        hidden_size=HIDDEN,
        intermediate_size=32,
        shared_expert_intermediate_size=64,
        num_local_experts=8,
        num_experts_per_tok=4,
        num_hidden_layers=LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        sliding_window=4,
        max_position_embeddings=64,
    )
    config._attn_implementation = "eager"
    model = grug.GrugMoeForCausalLM(config).eval().requires_grad_(False)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.2)
        for block in model.model.layers:
            block.mlp.router.bias.normal_(0.0, 0.1)
    native = copy.deepcopy(model)
    return SnowballHarness(model, (torch.device("cpu"),)), native


class FakeTokenizer:
    """Token ``i`` is written ``t<i>``."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [int(word[1:]) for word in text.split()]

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        return {"input_ids": self.encode(text)}

    def decode(self, ids: list[int]) -> str:
        return " ".join(f"t{i}" for i in ids)
