# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of the rank-1 edit and multi-layer / head-sum steering on the tiny random GrugMoE."""

import json

import pytest
import torch

from experiments.interp.snowball_identity.analyses import steer, steering_vectors
from experiments.interp.snowball_identity.edits import applied, edit_rank1, swapped
from experiments.interp.snowball_identity.harness import Output
from experiments.interp.snowball_identity.prompts import encode
from experiments.interp.snowball_identity.tiny_model import TOKENS, FakeTokenizer

TOKENIZER = FakeTokenizer()


def write_rows(path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return str(path)


def last_logprobs(harness, tokens=TOKENS) -> torch.Tensor:
    return harness.forward(tokens, assistant_start=0)[0][0]


@pytest.mark.parametrize("module_name", ["o_proj", "shared_down"])
def test_hook_matches_weight_edit(harness, module_name):
    block = harness.layers[2]
    module = block.self_attn.o_proj if module_name == "o_proj" else block.shared_expert.down_proj
    torch.manual_seed(1)
    delta = torch.randn(module.weight.shape[0])
    key = torch.randn(module.weight.shape[1]) * 0.1
    with applied(module, key, delta):
        hooked = last_logprobs(harness)
    with swapped(module, module.weight.data + torch.outer(delta, key)):
        swapped_lp = last_logprobs(harness)
    assert (hooked - swapped_lp).abs().max() < 1e-4
    assert (hooked - last_logprobs(harness)).abs().max() > 1e-3


def edit_spec(tmp_path, **overrides) -> dict:
    keys = write_rows(
        tmp_path / "keys.jsonl",
        [
            {"id": "k1", "text": "t5 t17 t3 t99", "targets": ["t40 t41"]},
            {"id": "k2", "text": "t6 t17 t3 t99", "targets": ["t40 t41"]},
            {"id": "k3", "text": "t7 t2 t17 t3 t99", "targets": ["t40 t41"]},
        ],
    )
    preserve = write_rows(
        tmp_path / "preserve.jsonl",
        [
            {"id": "p1", "text": "t50 t51 t52 t53", "targets": ["t60"]},
            {"id": "p2", "text": "t20 t21 t22", "targets": []},
        ],
    )
    held_out = write_rows(tmp_path / "held.jsonl", [{"id": "h1", "text": "t8 t17 t3 t99", "targets": ["t40 t41"]}])
    return {
        "layer": 2,
        "module": "o_proj",
        "key_prompts": keys,
        "preserve_prompts": preserve,
        "eval": {"held_out": held_out},
        "steps": 15,
        "lr": 2.0,
        "generate": {"max_new_tokens": 2},
        **overrides,
    }


def test_edit_rank1_raises_target_and_leaves_model_unchanged(harness, tmp_path):
    before = last_logprobs(harness)
    edit_rank1(harness, TOKENIZER, edit_spec(tmp_path), Output(str(tmp_path / "out")), stop_ids=set())
    assert (last_logprobs(harness) - before).abs().max() == 0
    report = json.loads((tmp_path / "out" / "edit.json").read_text())
    saved = torch.load(tmp_path / "out" / "delta.pt")
    assert saved["weight"] == "model.layers.2.self_attn.o_proj.weight"
    assert abs(float(saved["k_star"] @ saved["key"]) - 1) < 1e-4
    assert torch.allclose(saved["delta_w"], torch.outer(saved["delta"], saved["key"]))
    assert report["key"]["null_dim"] > 0 and report["relative_change"] > 0
    trace = report["trace"]
    assert report["best_step"] > 0
    assert trace[report["best_step"]]["target_nll_per_token"] < trace[0]["target_nll_per_token"]
    assert set(report["eval"]) == {"keys", "preserve", "held_out"}
    for conditions in report["eval"].values():
        assert set(conditions) == {"base", "edit", "edit_bf16"}
        for row in conditions["edit"]:
            assert row["kl_to_base"] >= -1e-6 and len(row["generation"].split()) == 2
    keys = report["eval"]["keys"]
    gain = [e["targets"]["t40 t41"] - b["targets"]["t40 t41"] for b, e in zip(keys["base"], keys["edit"], strict=True)]
    assert min(gain) > 0
    # The tiny model is FP32, so the rounded edit is the exact one.
    for exact, rounded in zip(keys["edit"], keys["edit_bf16"], strict=True):
        assert abs(exact["targets"]["t40 t41"] - rounded["targets"]["t40 t41"]) < 1e-3


def test_null_projection_shields_preserve_keys(harness, tmp_path):
    reports = {}
    for projection in ("null", "none"):
        spec = edit_spec(tmp_path, projection=projection, steps=1, eval={}, generate=None)
        edit_rank1(harness, TOKENIZER, spec, Output(str(tmp_path / projection)), stop_ids=set())
        reports[projection] = json.loads((tmp_path / projection / "edit.json").read_text())["key"]
    assert reports["null"]["preserve_key_scale_max_abs"] < 0.1 * reports["none"]["preserve_key_scale_max_abs"]


def test_head_diff_of_all_heads_is_the_attention_output(harness, tmp_path):
    a = write_rows(tmp_path / "a.jsonl", [{"id": "a1", "text": "t1 t2 t3 t4"}, {"id": "a2", "text": "t5 t6 t7"}])
    b = write_rows(tmp_path / "b.jsonl", [{"id": "b1", "text": "t9 t8 t7 t6 t5"}])
    heads = harness.model.config.num_attention_heads
    spec = {"layers": [1], "vector": {"head_diff": {"a": a, "b": b, "heads": [[1, h] for h in range(heads)]}}}
    vector = steering_vectors(harness, TOKENIZER, spec)[1]

    def mean_attn(rows) -> torch.Tensor:
        means = []
        for row in rows:
            prompt = encode(TOKENIZER, row)
            positions = torch.arange(prompt.assistant_start, len(prompt.tokens))
            _, record = harness.forward(prompt.tokens, assistant_start=prompt.assistant_start, capture=positions)
            means.append(record.attn[1].mean(0))
        return torch.stack(means).mean(0)

    expected = mean_attn([{"id": "a1", "text": "t1 t2 t3 t4"}, {"id": "a2", "text": "t5 t6 t7"}]) - mean_attn(
        [{"id": "b1", "text": "t9 t8 t7 t6 t5"}]
    )
    assert (vector - expected).abs().max() < 1e-4


def test_steer_layer_sets(harness, tmp_path):
    torch.manual_seed(2)
    vector = tmp_path / "v.pt"
    torch.save(torch.randn(harness.model.config.hidden_size), vector)
    prompts = write_rows(tmp_path / "s.jsonl", [{"id": "s", "text": "t5 t17 t3 t99", "targets": ["t40"]}])
    spec = {"prompts": prompts, "vector": {"path": str(vector)}, "layers": [1, 2, [1, 2]], "alphas": [1.0]}
    steer(harness, TOKENIZER, {**spec, "positions": "all"}, Output(str(tmp_path / "out")), stop_ids=set())
    (result,) = json.loads((tmp_path / "out" / "steer.json").read_text())
    by_layer = {json.dumps(cell["layer"]): cell["targets"]["t40"] for cell in result["grid"]}
    assert set(by_layer) == {"1", "2", "[1, 2]"}
    assert len({round(value, 5) for value in by_layer.values()}) == 3
