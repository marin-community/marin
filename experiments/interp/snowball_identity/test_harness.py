# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of the harness forward, capture and interventions on a tiny random GrugMoE."""

import copy

import pytest
import torch

from experiments.interp.snowball_identity.harness import (
    AddVector,
    Interventions,
    NeuronEdit,
    SnowballHarness,
    import_grug,
)

grug = import_grug()

LAYERS = 4
HIDDEN = 64
TOKENS = [5, 17, 3, 99, 42, 8, 61, 23, 7, 30]
ASSISTANT_START = 6


@pytest.fixture(scope="module")
def models() -> tuple[SnowballHarness, torch.nn.Module]:
    """The harness and an untouched copy of the same model (the harness swaps in its own expert execution)."""
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


@pytest.fixture(scope="module")
def harness(models) -> SnowballHarness:
    return models[0]


def all_positions() -> torch.Tensor:
    return torch.arange(len(TOKENS))


def test_forward_matches_native_model(models):
    harness, model = models
    native = model(input_ids=torch.tensor([TOKENS]), output_hidden_states=True)
    ours, recorder = harness.forward(
        TOKENS, assistant_start=ASSISTANT_START, capture=all_positions(), logit_positions=all_positions()
    )
    torch.testing.assert_close(ours, torch.log_softmax(native.logits[0].float(), -1), atol=1e-5, rtol=1e-5)
    # hidden_states[0] is the embedding output; [layer + 1] is each layer's output.
    for layer in range(LAYERS):
        torch.testing.assert_close(recorder.resid[layer], native.hidden_states[layer + 1][0].float())


def test_capture_decomposes_each_layer(harness):
    _, recorder = harness.forward(TOKENS, assistant_start=ASSISTANT_START, capture=torch.tensor([2, 9]))
    record = recorder.stacked()
    assert record["resid"].shape == (2, LAYERS, HIDDEN)
    assert record["expert_act"].shape == (2, LAYERS, 4, 32)
    assert record["shared_act"].shape == (2, LAYERS, 64)
    for layer in range(1, LAYERS):
        rebuilt = (
            record["resid"][:, layer - 1]
            + record["attn_out"][:, layer]
            + record["routed_out"][:, layer]
            + record["shared_out"][:, layer]
        )
        torch.testing.assert_close(rebuilt, record["resid"][:, layer], atol=1e-4, rtol=1e-4)
    experts = harness.layers[1].mlp.experts
    for row in range(2):
        routed = sum(
            record["router_weights"][row, 1, slot]
            * (experts.down_proj.weight[int(record["router_ids"][row, 1, slot])] @ record["expert_act"][row, 1, slot])
            for slot in range(4)
        )
        torch.testing.assert_close(routed, record["routed_out"][row, 1], atol=1e-4, rtol=1e-4)
    shared = harness.layers[2].shared_expert.down_proj.weight @ record["shared_act"][:, 2].T
    torch.testing.assert_close(shared.T, record["shared_out"][:, 2], atol=1e-4, rtol=1e-4)


def test_neuron_edits(harness):
    base, recorder = harness.forward(TOKENS, assistant_start=ASSISTANT_START, capture=all_positions())
    identity, _ = harness.forward(
        TOKENS,
        assistant_start=ASSISTANT_START,
        interventions=Interventions(neurons=[NeuronEdit(1, None, 3, 1.0), NeuronEdit(1, 2, 3, 1.0)]),
    )
    torch.testing.assert_close(identity, base)

    _, zeroed = harness.forward(
        TOKENS,
        assistant_start=ASSISTANT_START,
        interventions=Interventions(neurons=[NeuronEdit(1, None, 3, 0.0)]),
        capture=all_positions(),
    )
    assert torch.all(zeroed.shared_act[1][:, 3] == 0)
    assert not torch.allclose(zeroed.resid[1], recorder.resid[1])
    torch.testing.assert_close(zeroed.resid[0], recorder.resid[0])

    expert = int(recorder.router_ids[2][0, 0])
    _, routed = harness.forward(
        TOKENS,
        assistant_start=ASSISTANT_START,
        interventions=Interventions(neurons=[NeuronEdit(2, expert, 5, 0.0)]),
        capture=all_positions(),
    )
    hits = recorder.router_ids[2] == expert
    assert torch.all(routed.expert_act[2][hits][:, 5] == 0)


def test_add_vector_assistant_only(harness):
    base, recorder = harness.forward(
        TOKENS, assistant_start=ASSISTANT_START, capture=all_positions(), logit_positions=all_positions()
    )
    vector = torch.randn(HIDDEN)
    unchanged, _ = harness.forward(
        TOKENS,
        assistant_start=ASSISTANT_START,
        interventions=Interventions(add=[AddVector(1, vector, 0.0, False)]),
        logit_positions=all_positions(),
    )
    torch.testing.assert_close(unchanged, base)
    steered, steered_recorder = harness.forward(
        TOKENS,
        assistant_start=ASSISTANT_START,
        interventions=Interventions(add=[AddVector(1, vector, 3.0, True)]),
        capture=all_positions(),
        logit_positions=all_positions(),
    )
    assistant = (torch.arange(len(TOKENS)) >= ASSISTANT_START).float().unsqueeze(1)
    torch.testing.assert_close(
        steered_recorder.resid[1] - recorder.resid[1], assistant * 3.0 * vector, atol=1e-4, rtol=1e-4
    )
    torch.testing.assert_close(steered[:ASSISTANT_START], base[:ASSISTANT_START])
    assert not torch.allclose(steered[ASSISTANT_START:], base[ASSISTANT_START:])


def test_residual_patch(harness):
    source = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    _, source_recorder = harness.forward(source, assistant_start=0, capture=torch.tensor([4]))
    base, _ = harness.forward(TOKENS, assistant_start=ASSISTANT_START, logit_positions=all_positions())
    value = source_recorder.resid[2][0]
    patched, recorder = harness.forward(
        TOKENS,
        assistant_start=ASSISTANT_START,
        interventions=Interventions(patches=[(2, 4, value)]),
        capture=all_positions(),
        logit_positions=all_positions(),
    )
    torch.testing.assert_close(recorder.resid[2][4], value)
    torch.testing.assert_close(patched[:4], base[:4])
    assert not torch.allclose(patched[4:], base[4:])
