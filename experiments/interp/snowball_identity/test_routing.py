# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of the router rounding isolation on the tiny random GrugMoE."""

import torch

from experiments.interp.snowball_identity.harness import Output, run_positions
from experiments.interp.snowball_identity.logit_check import stored_sequences
from experiments.interp.snowball_identity.routing import VARIANTS, compare_routes, routing_stats
from experiments.interp.snowball_identity.sequences import Sequence
from experiments.interp.snowball_identity.tiny_model import LAYERS, TOKENS


def test_bias_rounding_flips_a_near_tie():
    # Expert 1's bias rounds down in BF16 (1 + 2^-10 -> 1), so expert 2 overtakes it at position 0 only.
    logits = torch.tensor([[3.0, 0.0, 1.0005, -5.0], [3.0, 0.0, -1.0, -5.0]])
    bias = torch.tensor([0.0, 1.0 + 2**-10, 0.0, 0.0])
    result = compare_routes(logits, bias, k=2)["bf16_bias"]
    assert result["experts_changed"].tolist() == [1, 0] and result["order_changed"].tolist() == [True, False]
    assert result["weight_change"][0] > 0 and result["weight_change"][1] == 0
    assert 0 < result["margin"][0] < 1e-3


def test_logit_rounding_flips_a_near_tie():
    # The bias is BF16-exact; only the rounded logits put expert 2 (3.0) ahead of expert 1 (3.0 + 2^-9).
    logits = torch.tensor([[5.0, 3.0 + 2**-9, 3.0, -5.0]])
    rounded = torch.tensor([[5.0, 3.0, 3.0 + 2**-6, -5.0]])
    results = compare_routes(logits, torch.zeros(4), k=2, rounded_logits=rounded)
    assert results["bf16_bias"]["experts_changed"].tolist() == [0]
    assert results["bf16_logits_bias"]["experts_changed"].tolist() == [1]


def test_routing_stats_cover_every_position(harness):
    sequences = [
        Sequence("check|a", "check", tuple(TOKENS)),
        Sequence("contrastive|r|no_system|0", "contrastive", tuple(TOKENS[:6])),
    ]
    report = routing_stats(harness, sequences)
    assert report["variants"] == list(VARIANTS) and len(report["layers"]) == LAYERS
    for layer in report["layers"]:
        assert layer["bias_dtype"] == "torch.float32"
        for variant in VARIANTS:
            assert layer[variant]["check"]["positions"] == len(TOKENS)
            assert layer[variant]["contrastive"]["positions"] == 6
    for variant in VARIANTS:
        assert report["totals"][variant]["positions"] == LAYERS * (len(TOKENS) + 6)


def test_bf16_exact_bias_changes_nothing(harness):
    biases = [block.mlp.router.bias.detach().clone() for block in harness.layers]
    try:
        for block in harness.layers:
            block.mlp.router.bias.copy_(block.mlp.router.bias.bfloat16().float())
        report = routing_stats(harness, [Sequence("check|a", "check", tuple(TOKENS))])
        assert report["totals"]["bf16_bias"]["positions_changed"] == 0
        assert report["totals"]["bf16_bias"]["positions_order_changed"] == 0 and report["cases"]["bf16_bias"] == []
        assert all(layer["bias_rounding_max_abs"] == 0 for layer in report["layers"])
    finally:
        for block, bias in zip(harness.layers, biases, strict=True):
            block.mlp.router.bias.copy_(bias)


def test_stored_sequences_keep_token_ids(harness, tmp_path):
    sequences = [
        Sequence("check|a", "check", tuple(TOKENS)),
        Sequence("long|x", "long", tuple(TOKENS[:5])),
    ]
    run_positions(harness, sequences, Output(str(tmp_path)))
    stored = stored_sequences(str(tmp_path / "positions.pt"))
    assert [(s.id, s.group, s.tokens) for s in stored] == [(s.id, s.group, s.tokens) for s in sequences]


def test_order_change_without_set_change():
    # Experts 0 and 1 swap order under rounded logits but stay selected.
    logits = torch.tensor([[3.0 + 2**-8, 3.0, -1.0, -5.0]])
    rounded = torch.tensor([[3.0, 3.0 + 2**-6, -1.0, -5.0]])
    result = compare_routes(logits, torch.zeros(4), k=2, rounded_logits=rounded)["bf16_logits_bias"]
    assert result["order_changed"].tolist() == [True] and result["experts_changed"].tolist() == [0]
