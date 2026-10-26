# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of lens, patch and steer on the tiny random GrugMoE."""

import json

import torch

from experiments.interp.snowball_identity.analyses import lens, patch, steer
from experiments.interp.snowball_identity.harness import Interventions, Output
from experiments.interp.snowball_identity.prompts import encode
from experiments.interp.snowball_identity.test_edits import write_rows
from experiments.interp.snowball_identity.tiny_model import HIDDEN, LAYERS, FakeTokenizer

TOKENIZER = FakeTokenizer()


def test_lens_attribution_adds_up(harness, tmp_path):
    prompts = write_rows(
        tmp_path / "lens.jsonl",
        [
            {"id": "pair", "text": "t5 t17 t3 t99 t42 t8", "targets": ["t40", "t41"]},
            {"id": "single", "text": "t5 t17 t3", "targets": ["t40"]},
        ],
    )
    lens(harness, TOKENIZER, {"prompts": prompts, "top_neurons": 3}, Output(str(tmp_path / "out")))
    results = json.loads((tmp_path / "out" / "lens.json").read_text())
    for result in results:
        dla = result["dla"]
        assert abs(dla["total"] - dla["check"]) < 1e-3
        assert len(dla["layers"]) == LAYERS and len(dla["top_neurons"]) == 3
        layer = dla["layers"][1]
        assert abs(sum(layer["heads"]) - layer["attn"]) < 1e-3
        assert abs(sum(expert["dla"] for expert in layer["experts"]) - layer["routed"]) < 1e-3
        assert len(result["lens"]) == LAYERS
    pair = results[0]
    assert abs(pair["dla"]["check"] - pair["actual_logprob_diff"]) < 1e-3
    # The last layer's lens is the model's own output.
    final = pair["lens"][-1]["targets"]
    assert abs(final["t40"] - pair["final_logprobs"][0]) < 1e-4


def test_patch_recovery(harness, tmp_path):
    pairs = write_rows(
        tmp_path / "pairs.jsonl",
        [
            {
                "id": "p",
                "clean": {"text": "t5 t17 t3 t99 t42 t8"},
                "corrupt": {"text": "t6 t18 t4 t99 t42 t8"},
                "targets": ["t40", "t41"],
            }
        ],
    )
    spec = {
        "pairs": pairs,
        "components": ["resid_pre", "attn", "routed", "shared", "resid"],
        "heads": True,
        "per_position": True,
        "batch": 7,
    }
    patch(harness, TOKENIZER, spec, Output(str(tmp_path / "out")))
    (result,) = json.loads((tmp_path / "out" / "patch.json").read_text())
    assert result["patched_positions"] == 3
    heads = harness.model.config.num_attention_heads
    assert len(result["results"]) == LAYERS * (5 + 3 + heads)
    by_key = {(r["layer"], r["component"], r.get("position"), r.get("head")): r for r in result["results"]}
    # The residual leaving the last layer at the shared tail decides the last position's logits.
    assert abs(by_key[(LAYERS - 1, "resid", None, None)]["recovery"] - 1.0) < 1e-4
    assert abs(by_key[(LAYERS - 1, "resid_pre", 5, None)]["logit_diff"] - result["corrupt_logit_diff"]) > 0


def test_steer_grid(harness, tmp_path):
    vector = tmp_path / "v.pt"
    torch.save(torch.randn(HIDDEN), vector)
    prompts = write_rows(tmp_path / "steer.jsonl", [{"id": "s", "text": "t5 t17 t3 t99", "targets": ["t40 t41"]}])
    spec = {
        "prompts": prompts,
        "vector": {"path": str(vector)},
        "layers": [1, 2],
        "alphas": [0.0, 3.0],
        "positions": "all",
        "generate": {"max_new_tokens": 3},
        "batch": 3,
    }
    steer(harness, TOKENIZER, spec, Output(str(tmp_path / "out")), stop_ids=set())
    (result,) = json.loads((tmp_path / "out" / "steer.json").read_text())
    cells = {(cell["layer"], cell["alpha"]): cell for cell in result["grid"]}
    prompt = encode(TOKENIZER, {"id": "s", "text": "t5 t17 t3 t99"})
    baseline = sum(harness.target_logprobs(prompt, [40, 41], Interventions()))
    for layer in (1, 2):
        assert abs(cells[(layer, 0.0)]["targets"]["t40 t41"] - baseline) < 1e-4
        assert cells[(layer, 3.0)]["targets"]["t40 t41"] != cells[(layer, 0.0)]["targets"]["t40 t41"]
    assert cells[(1, 0.0)]["generation"] == cells[(2, 0.0)]["generation"]
    assert len(cells[(1, 0.0)]["generation"].split()) == 3

    a = write_rows(tmp_path / "a.jsonl", [{"id": "a1", "text": "t1 t2 t3"}, {"id": "a2", "text": "t4 t5"}])
    b = write_rows(tmp_path / "b.jsonl", [{"id": "b1", "text": "t9 t8 t7"}])
    spec = {**spec, "vector": {"mean_diff": {"a": a, "b": b}, "unit": True}, "generate": None}
    steer(harness, TOKENIZER, spec, Output(str(tmp_path / "diff")), stop_ids=set())
    vectors = torch.load(tmp_path / "diff" / "vectors.pt")
    assert set(vectors) == {1, 2} and abs(float(vectors[1].norm()) - 1.0) < 1e-5
