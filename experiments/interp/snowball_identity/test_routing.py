# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of the router rounding isolation on the tiny random GrugMoE."""

import torch

from experiments.interp.snowball_identity.routing import compare_routes, routing_stats
from experiments.interp.snowball_identity.sequences import Sequence
from experiments.interp.snowball_identity.tiny_model import LAYERS, TOKENS


def test_rounding_flips_a_near_tie():
    # Expert 1's bias rounds down in BF16 (1 + 2^-10 -> 1), so expert 2 overtakes it at position 0 only.
    logits = torch.tensor([[3.0, 0.0, 1.0005, -5.0], [3.0, 0.0, -1.0, -5.0]])
    bias = torch.tensor([0.0, 1.0 + 2**-10, 0.0, 0.0])
    result = compare_routes(logits, bias, k=2)
    assert result["experts_changed"].tolist() == [1, 0]
    assert result["weight_change"][0] > 0 and result["weight_change"][1] == 0
    assert 0 < result["margin"][0] < 1e-3


def test_routing_stats_cover_every_position(harness):
    sequences = [
        Sequence("check|a", "check", tuple(TOKENS)),
        Sequence("contrastive|r|no_system|0", "contrastive", tuple(TOKENS[:6])),
    ]
    report = routing_stats(harness, sequences)
    assert len(report["layers"]) == LAYERS
    for layer in report["layers"]:
        assert layer["bias_dtype"] == "torch.float32"
        assert layer["groups"]["check"]["positions"] == len(TOKENS)
        assert layer["groups"]["contrastive"]["positions"] == 6
    assert report["totals"]["positions"] == LAYERS * (len(TOKENS) + 6)


def test_bf16_exact_bias_changes_nothing(harness):
    biases = [block.mlp.router.bias.detach().clone() for block in harness.layers]
    try:
        for block in harness.layers:
            block.mlp.router.bias.copy_(block.mlp.router.bias.bfloat16().float())
        report = routing_stats(harness, [Sequence("check|a", "check", tuple(TOKENS))])
        assert report["totals"]["positions_changed"] == 0
        assert all(layer["bias_rounding_max_abs"] == 0 for layer in report["layers"])
    finally:
        for block, bias in zip(harness.layers, biases, strict=True):
            block.mlp.router.bias.copy_(bias)
