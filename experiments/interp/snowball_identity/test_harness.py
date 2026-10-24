# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of the harness forward, capture and interventions on a tiny random GrugMoE."""

import copy
import json

import pytest
import torch

from experiments.interp.snowball_identity.harness import (
    AddVector,
    Interventions,
    NeuronEdit,
    Output,
    Replace,
    SnowballHarness,
    import_grug,
    run_positions,
)
from experiments.interp.snowball_identity.logit_check import compare
from experiments.interp.snowball_identity.sequences import Sequence

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
        interventions=Interventions(replace=[Replace(2, "resid", torch.tensor([4]), value[None])]),
        capture=all_positions(),
        logit_positions=all_positions(),
    )
    torch.testing.assert_close(recorder.resid[2][4], value)
    torch.testing.assert_close(patched[:4], base[:4])
    assert not torch.allclose(patched[4:], base[4:])


def test_positions_and_logit_check(harness, tmp_path):
    sequences = [
        Sequence("check|a", "check", tuple(TOKENS)),
        *(
            Sequence(
                f"contrastive|r|no_system|{index}",
                "contrastive",
                (*TOKENS[:6], 40 + index, 50 + index),
                (6, 8),
                {"correct": (0,)},
            )
            for index in range(3)
        ),
    ]
    run_positions(harness, sequences, Output(str(tmp_path / "out")))
    ours = torch.load(tmp_path / "out" / "positions.pt")
    # A "vLLM" reference that is the harness's own output: every comparison must be exact.
    reference = {}
    for sequence in sequences:
        record = ours[sequence.id]
        reference[sequence.id] = {
            "tokens": record["tokens"],
            "actual": [None, *record["actual"][1:].tolist()],
            "top": [None]
            + [
                dict(zip(ids[:20].tolist(), values[:20].tolist(), strict=True))
                for ids, values in zip(record["top_ids"][1:], record["top_lp"][1:], strict=True)
            ],
        }
    (tmp_path / "vllm.json").write_text(json.dumps(reference))
    result = compare(
        str(tmp_path / "out" / "positions.pt"),
        str(tmp_path / "vllm.json"),
        None,
        {s.id: (s.score, s.meta["correct"]) for s in sequences if s.score},
    )
    check = result["positions"]["check"]
    assert check["top1_agree"] == check["positions"] == len(TOKENS) - 1
    assert check["max_abs_logprob_diff"] == 0 and check["missing_from_harness_top50"] == 0
    contrastive = result["contrastive"]
    assert contrastive["row_conditions"] == 1 and contrastive["max_abs_score_diff_vs_vllm"] < 1e-5
    scores = [float(ours[f"contrastive|r|no_system|{i}"]["actual"][6:8].sum()) for i in range(3)]
    won = scores[0] > max(scores[1:])
    assert contrastive["wins"]["no_system"] == {"harness": won, "vllm": won, "rows": 1}


def test_batched_variants_match_single_runs(harness):
    variants = [
        Interventions(),
        Interventions(neurons=[NeuronEdit(1, None, 3, 0.0), NeuronEdit(2, 5, 7, 2.0)]),
        Interventions(add=[AddVector(2, torch.ones(HIDDEN), 0.5, True)]),
    ]
    batched, recorder = harness.run(
        [TOKENS] * 3,
        assistant_start=ASSISTANT_START,
        variants=variants,
        capture=torch.tensor([3, 9]),
        logit_positions=all_positions(),
    )
    for row, variant in enumerate(variants):
        single, single_recorder = harness.forward(
            TOKENS,
            assistant_start=ASSISTANT_START,
            interventions=variant,
            capture=torch.tensor([3, 9]),
            logit_positions=all_positions(),
        )
        torch.testing.assert_close(batched[row], single, atol=1e-5, rtol=1e-5)
        for kind in ("resid", "expert_act", "shared_act", "head_in"):
            torch.testing.assert_close(
                getattr(recorder.select(row), kind)[2], getattr(single_recorder, kind)[2], atol=1e-5, rtol=1e-5
            )


def test_component_patching(harness):
    clean_tokens = [5, 17, 3, 99, 42, 8, 61, 23, 7, 30]
    corrupt_tokens = [5, 18, 4, 98, 42, 8, 61, 23, 7, 30]
    everywhere = all_positions()
    clean, record = harness.forward(clean_tokens, assistant_start=0, capture=everywhere, logit_positions=everywhere)
    corrupt, _ = harness.forward(corrupt_tokens, assistant_start=0, logit_positions=everywhere)
    assert not torch.allclose(clean, corrupt)

    def patched(*replacements):
        result, _ = harness.forward(
            corrupt_tokens,
            assistant_start=0,
            interventions=Interventions(replace=list(replacements)),
            logit_positions=everywhere,
        )
        return result

    # The whole residual stream entering layer 2 decides everything after it.
    torch.testing.assert_close(
        patched(Replace(2, "resid_pre", everywhere, record.resid_pre(2))), clean, atol=1e-4, rtol=1e-4
    )
    torch.testing.assert_close(
        patched(Replace(0, "resid_pre", everywhere, record.resid_pre(0))), clean, atol=1e-4, rtol=1e-4
    )
    # o_proj is linear in the heads, so patching every head equals patching the attention output.
    head_dim = harness.head_dim
    heads = [
        Replace(1, "head", everywhere, record.head_in[1][:, h * head_dim : (h + 1) * head_dim], head=h)
        for h in range(record.head_in[1].shape[1] // head_dim)
    ]
    torch.testing.assert_close(
        patched(*heads), patched(Replace(1, "attn", everywhere, record.attn[1])), atol=1e-4, rtol=1e-4
    )
    for component in ("attn", "routed", "shared"):
        result = patched(Replace(1, component, everywhere, getattr(record, component)[1]))
        assert not torch.allclose(result, corrupt)
