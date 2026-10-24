# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of the multi-layer MEMIT / AlphaEdit edit on the tiny random GrugMoE."""

import json

import pytest
import torch

from experiments.interp.snowball_identity.edits import edit_rank1
from experiments.interp.snowball_identity.harness import Output
from experiments.interp.snowball_identity.memit import edit_memit
from experiments.interp.snowball_identity.test_edits import TOKENIZER, edit_spec, last_logprobs, write_rows


def memit_spec(tmp_path, **overrides) -> dict:
    spec = edit_spec(tmp_path, eval={}, generate=None, steps=15, lr=2.0, max_norm_factor=4.0)
    spec.pop("layer")
    keys = write_rows(
        tmp_path / "memit-keys.jsonl",
        [
            {"id": "k1", "text": "t5 t17 t3 t99", "targets": ["t40 t41"]},
            {"id": "k2", "text": "t6 t18 t3 t98", "targets": ["t70"]},
            {"id": "k3", "text": "t7 t2 t17 t3 t97", "targets": ["t40 t41"]},
        ],
    )
    return {**spec, "layers": [1, 2], "key_prompts": keys, **overrides}


def key_gain(report: dict, condition: str) -> list[float]:
    keys = report["eval"]["keys"]
    gains = []
    for base, edited in zip(keys["base"], keys[condition], strict=True):
        (target,) = base["targets"]
        gains.append(edited["targets"][target] - base["targets"][target])
    return gains


@pytest.mark.parametrize("projection", ["null", "cov"])
def test_memit_raises_each_key_target(harness, tmp_path, projection):
    before = last_logprobs(harness)
    spec = memit_spec(tmp_path, projection=projection, cov_weight=1.0)
    edit_memit(harness, TOKENIZER, spec, Output(str(tmp_path / projection)), stop_ids=set())
    assert (last_logprobs(harness) - before).abs().max() == 0
    report = json.loads((tmp_path / projection / "memit.json").read_text())
    assert [r["layer"] for r in report["layer_reports"]] == [1, 2]
    assert report["residual"]["best_step"] > 0
    assert report["residual"]["unreached_fraction"] < 0.5
    assert min(key_gain(report, "edit")) > 0
    # FP32 tiny model: the rounded stack is the exact one.
    for exact, rounded in zip(key_gain(report, "edit"), key_gain(report, "edit_bf16"), strict=True):
        assert abs(exact - rounded) < 1e-3
    saved = torch.load(tmp_path / projection / "memit.pt")
    for edit in saved["edits"]:
        assert edit["up"].shape[1] == 3 and torch.allclose(edit["delta_w"], edit["up"] @ edit["down"].T)
    if projection == "null":
        for layer in report["layer_reports"]:
            assert layer["projection_output_change_rms"] < 0.1 * layer["residual_norm_mean"]


def test_memit_edit_loads_as_prior(harness, tmp_path):
    edit_memit(harness, TOKENIZER, memit_spec(tmp_path), Output(str(tmp_path / "memit")), stop_ids=set())
    spec = edit_spec(tmp_path, steps=1, generate=None, prior_edits=[str(tmp_path / "memit" / "memit.pt")])
    edit_rank1(harness, TOKENIZER, spec, Output(str(tmp_path / "stacked")), stop_ids=set())
    report = json.loads((tmp_path / "stacked" / "edit.json").read_text())
    assert [p["weight"] for p in report["prior_edits"]] == [
        "model.layers.1.self_attn.o_proj.weight",
        "model.layers.2.self_attn.o_proj.weight",
    ]
    held = report["eval"]["held_out"]
    assert held["prior"][0]["targets"]["t40 t41"] > held["base"][0]["targets"]["t40 t41"]
