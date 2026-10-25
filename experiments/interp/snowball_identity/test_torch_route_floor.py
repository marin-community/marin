# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from experiments.interp.snowball_identity import moe_drop_diagnostic as diag
from experiments.interp.snowball_identity import torch_route_floor as floor
from experiments.interp.snowball_identity.harness import SnowballHarness, import_grug
from experiments.interp.snowball_identity.sequences import Sequence


@pytest.fixture(scope="module")
def harness():
    grug = import_grug()
    torch.manual_seed(0)
    config = grug.GrugMoeConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=16,
        shared_expert_intermediate_size=32,
        num_local_experts=8,
        num_experts_per_tok=4,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        sliding_window=4,
        max_position_embeddings=256,
    )
    config._attn_implementation = "eager"
    model = grug.GrugMoeForCausalLM(config).eval().requires_grad_(False)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.2)
    return SnowballHarness(model, (torch.device("cpu"),))


SEQUENCES = [
    Sequence(
        f"contrastive|r{i}|no_system|0", "contrastive", tuple(np.random.default_rng(i).integers(1, 127, n).tolist())
    )
    for i, n in enumerate((5, 9, 3, 12, 7, 20, 6, 4, 11, 8))
]


def test_both_runs_record_the_routers_own_selection(harness, monkeypatch):
    """The taps check every layer's top-6 prefix against the router's own selection; the batched run cuts each
    sequence back out of the gate's grids (10 sequences: two steps of 8, the second filled with copies)."""
    monkeypatch.setattr(diag, "PAD_ID", 0)  # inside the tiny vocab
    taps = floor.RouterTaps(harness)
    batched = floor.top6_batched(harness, taps, SEQUENCES)
    single = floor.top6_batch1(harness, taps, SEQUENCES)
    taps.remove()
    for sequence in SEQUENCES:
        for found in (batched[sequence.id], single[sequence.id]):
            assert found["ids"].shape == found["biased"].shape == (len(sequence.tokens), 3, floor.TOP)
            assert found["ids"].dtype == torch.int16
            assert (found["biased"][..., :-1] >= found["biased"][..., 1:]).all()
        # FP32 on CPU, causal attention: right padding and batching leave real positions unchanged.
        np.testing.assert_array_equal(batched[sequence.id]["ids"][..., :4], single[sequence.id]["ids"][..., :4])
        np.testing.assert_allclose(batched[sequence.id]["biased"], single[sequence.id]["biased"], atol=1e-4)
    reference = {sid: r["ids"][..., :4].numpy() for sid, r in single.items()}
    report = floor.floor_report({"batched": batched, "batch1": single}, reference)
    positions = sum(len(s.tokens) for s in SEQUENCES) * 3
    for name in ("batched_vs_routes_pt", "batch1_vs_routes_pt", "batched_vs_batch1"):
        assert report[name]["totals"] == {"positions": positions, "set_agree": positions, "ordered_agree": positions}
    assert len(report["margins_all_positions"]["batched"]) == 3


def test_agreement_counts_and_margins_at_disagreements():
    ours = {"check|a": np.array([[[1, 2, 3, 4]], [[1, 2, 3, 4]], [[1, 2, 3, 5]]])}  # [T=3, L=1, K=4]
    theirs = {"check|a": np.array([[[1, 2, 3, 4]], [[2, 1, 3, 4]], [[1, 2, 3, 6]]])}
    margins = {"ours": {"check|a": np.array([[0.5], [0.4], [0.003]])}}
    found = floor.agreement(ours, theirs, margins)
    layer = found["layers"][0]
    assert (layer["positions"], layer["set_agree"], layer["ordered_agree"]) == (3, 2, 1)
    assert layer["margin_at_set_disagreements"]["ours"] == {
        "count": 1,
        "max": pytest.approx(0.003),
        "p50": pytest.approx(0.003),
        "p90": pytest.approx(0.003),
        "p99": pytest.approx(0.003),
        "below_0.01": 1,
        "below_0.05": 1,
        "below_0.1": 1,
        "below_0.25": 1,
        "below_0.5": 1,
    }
    with pytest.raises(ValueError):
        floor.agreement(ours, {"check|b": theirs["check|a"]}, {})


def test_long_past_window_is_a_subset_not_counted_twice():
    length = floor.WINDOW + 3
    ours = {"long|x": np.zeros((length, 1, 4), dtype=np.int64)}
    found = floor.agreement(ours, {"long|x": ours["long|x"].copy()}, {})
    layer = found["layers"][0]
    assert layer["positions"] == length and layer["groups"]["long_past_window"]["positions"] == 3


ENTRYPOINT_CHILD = """
import runpy, sys
import pytest
sys.argv = ["torch_route_floor", "--job-name", "x", "--root", sys.argv[1]]
ns = runpy.run_module("experiments.interp.snowball_identity.torch_route_floor", run_name="__main__")
with pytest.raises(TypeError):  # the __main__ copy of the worker does not pickle
    ns["Entrypoint"].from_callable(ns["floor_worker"], "s3://bucket/root")
ns["entrypoint_for"]("s3://bucket/root")
print("ok")
"""


def test_entrypoint_pickles_when_run_as_main():
    """``python -m`` makes the module ``__main__``; the submitted worker must pickle by reference."""
    root = f"{floor.T310}/diag-plan-only"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    done = subprocess.run(
        [sys.executable, "-c", ENTRYPOINT_CHILD, root], capture_output=True, text=True, timeout=300, env=env
    )
    assert done.returncode == 0, done.stderr[-3000:]
    assert done.stdout.strip().endswith("ok")
