# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.interp.snowball_identity import precision_gate
from experiments.interp.snowball_identity import route_proxy_check as g1

T310_LOCAL = Path(os.environ.get("T310_LOCAL", Path.home() / "Dropbox/marin-notes/snowball-identity/t310"))
R2_DROPLESS = T310_LOCAL / "diag-moe-drops-r2/ep4-dropless/positions.pt"
R2_DROPLESS_SHA256 = "2b88315b33c4c7f6e735cc42b7958271cc4feaadba6ab1baa8be97a784b4e04e"
FLOOR_RECEIPT = T310_LOCAL / "diag-torch-floor-r1/receipt.json"
TORCH_TOP6 = T310_LOCAL / "diag-torch-floor-r1/batch1/top6.pt"


def synthetic(flips: int, margin: float):
    """Two layers, 100 positions; Levanter flips ``flips`` positions at layer 1 where torch's margin is ``margin``.
    The floor: 5% torch disagreements per layer, own margins p50 0.02 / p99 0.04."""
    ids = np.tile(np.arange(6, dtype=np.int16), (100, 2, 1))
    biased = np.tile(np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0], dtype=np.float32), (100, 2, 1))
    biased[:flips, 1, 3] = biased[:flips, 1, 4] + margin
    routes = ids[..., :4].copy()
    routes[:flips, 1, 3] = 4  # Levanter picks torch's 5th
    floor_layer = {"positions": 100, "set_rate": 0.95}
    own = {"margin_at_set_disagreements": {"batch1": {"count": 5, "p50": 0.02, "p99": 0.04}}}
    floor = {
        "floor": {
            "batched_vs_routes_pt": {"layers": [dict(floor_layer), dict(floor_layer)]},
            "batched_vs_batch1": {"layers": [own, own]},
        }
    }
    levanter = {"check|a": {"routes": torch.from_numpy(routes)}}
    top6 = {"check|a": {"ids": torch.from_numpy(ids), "biased": torch.from_numpy(biased)}}
    return levanter, top6, floor


def test_rate_bound_is_twice_the_floor():
    assert g1.check(*synthetic(10, 0.02))["layers"][1]["rate_ok"]  # 10% <= 2 x 5%
    report = g1.check(*synthetic(11, 0.02))
    assert not report["layers"][1]["rate_ok"] and report["layers"][0]["rate_ok"]
    assert report["failures"] == ["L1: disagreement rate 0.1100 > 2.0 x floor 0.0500"]


def test_margin_class_compares_p50_only_below_30_disagreements():
    report = g1.check(*synthetic(10, 0.031))  # p50 0.031 > 1.5 x 0.02
    assert not report["layers"][1]["p50_ok"] and not report["layers"][1]["p99_checked"]
    assert report["layers"][1]["p99_ok"]  # 0.031 would pass p99 (<= 0.06), but it is not checked below 30
    assert g1.check(*synthetic(10, 0.029))["layers"][1]["p50_ok"]


def test_constants_mirror_g1():
    assert (g1.K_RATE, g1.MARGIN_FACTOR, g1.MIN_FOR_P99) == (2.0, 1.5, 30)


@pytest.mark.skipif(
    not (R2_DROPLESS.exists() and FLOOR_RECEIPT.exists() and TORCH_TOP6.exists()), reason="local t310 copies"
)
def test_r2_dropless_arm():
    """The r2 ep4-dropless arm against the floor: the rate bound holds at 10 of 26 layers (as computed from the two
    receipts before G1 was ruled); with the margin class, 7 of 26."""
    positions = R2_DROPLESS.read_bytes()
    assert hashlib.sha256(positions).hexdigest() == R2_DROPLESS_SHA256
    receipt = FLOOR_RECEIPT.read_bytes()
    assert hashlib.sha256(receipt).hexdigest() == g1.FLOOR_RECEIPT_SHA256
    top6 = TORCH_TOP6.read_bytes()
    assert hashlib.sha256(top6).hexdigest() == g1.TORCH_TOP6_SHA256
    report = g1.check(
        torch.load(R2_DROPLESS, map_location="cpu"), torch.load(TORCH_TOP6, map_location="cpu"), json.loads(receipt)
    )
    layers = report["layers"]
    assert [r["layer"] for r in layers if r["rate_ok"]] == [4, 9, 13, 14, 15, 17, 18, 20, 23, 24]
    assert report["layers_passing"] == 7
    assert report["totals"]["set_agree"] / report["totals"]["positions"] == pytest.approx(0.8933, abs=1e-4)


@pytest.mark.skipif(
    not (R2_DROPLESS.exists() and FLOOR_RECEIPT.exists() and TORCH_TOP6.exists()), reason="local t310 copies"
)
def test_gate_g1_on_the_r2_dropless_arm():
    """The pre-update gate's G1 wiring (pinned floor paths, positions.pt as written) gives the same 10 / 7 of 26."""
    inputs = precision_gate.PrecisionGateInputs(
        positions="unused",
        vllm="unused",
        identity="unused",
        tokenizer_source="unused",
        torch_routing="unused",
        floor_receipt=str(FLOOR_RECEIPT),
        torch_top6=str(TORCH_TOP6),
        output="unused",
    )
    report = precision_gate.g1_routes(str(R2_DROPLESS), inputs)
    assert sum(r["rate_ok"] for r in report["layers"]) == 10
    assert report["layers_passing"] == 7
    assert len(report["failures"]) > 0


FIX_R1_COMBINE_ON = T310_LOCAL / "diag-moe-fix-r1/dropless-fp32-combine/positions.pt"
FIX_R1_COMBINE_ON_SHA256 = "e857f361510c75fa3e969da2206d1c9851e70abb44a1a3bbe3139042c8a77bc7"
VLLM_CAPTURE = T310_LOCAL / "marten-task310-vllm-routes-r1/capture"


def synthetic_vllm(flips: int, margin: float, vllm_flips: int = 8):
    """``synthetic`` plus a served floor: vLLM picks torch's 5th at ``vllm_flips`` positions of layer 1, where torch's
    margin is 0.02 (the last ``vllm_flips`` rows); pass 2 equals pass 1 except one more flip."""
    levanter, top6, floor = synthetic(flips, margin)
    biased = top6["check|a"]["biased"].numpy()
    biased[-vllm_flips:, 1, 3] = biased[-vllm_flips:, 1, 4] + 0.02
    pass1 = top6["check|a"]["ids"].numpy()[..., :4].copy()
    pass1[-vllm_flips:, 1, 3] = 4
    pass2 = pass1.copy()
    pass2[0, 0, 3] = 4
    return levanter, top6, floor, {"check|a": torch.from_numpy(pass1)}, {"check|a": torch.from_numpy(pass2)}


def test_g1a_rate_bound_is_1_25_times_the_served_floor():
    assert g1.check_vllm_floor(*synthetic_vllm(9, 0.02))["layers"][1]["rate_ok"]  # 9% <= 1.25 x 8%
    report = g1.check_vllm_floor(*synthetic_vllm(11, 0.02))
    assert not report["layers"][1]["rate_ok"]
    assert report["failures"] == ["L1: disagreement rate 0.1100 > 1.25 x vLLM floor 0.0800"]
    row = report["layers"][1]
    assert (row["F_vllm"], row["D_vllm"], row["F_torch"]) == (0.92, 1.0, 0.95)
    assert report["layers"][0]["D_vllm"] == 0.99


def test_g1a_margins_compare_against_vllm_vs_torch_disagreements():
    """Torch's margin at the served stack's disagreements is 0.02: p50 0.031 fails, 0.029 passes."""
    assert not g1.check_vllm_floor(*synthetic_vllm(10, 0.031))["layers"][1]["p50_ok"]
    assert g1.check_vllm_floor(*synthetic_vllm(10, 0.029))["layers"][1]["p50_ok"]


def test_g1a_diagnostic_line():
    """Levanter vs vLLM pass 1 at layer 1: Levanter flips rows 0-9, vLLM the last 8; agreement 0.82 vs D - (1 - F)
    = 1.0 - 0.08 = 0.92."""
    row = g1.check_vllm_floor(*synthetic_vllm(10, 0.02))["layers"][1]
    assert row["levanter_vs_vllm"]["set"][0] == pytest.approx(0.82)
    assert row["diagnostic_bound"] == pytest.approx(0.92)
    assert not row["diagnostic_ok"]


def test_g1a_constants():
    assert (g1.K_RATE_VLLM, g1.MARGIN_FACTOR, g1.MIN_FOR_P99) == (1.25, 1.5, 30)


@pytest.mark.skipif(
    not (FIX_R1_COMBINE_ON.exists() and VLLM_CAPTURE.exists() and FLOOR_RECEIPT.exists() and TORCH_TOP6.exists()),
    reason="local t310 copies",
)
def test_g1a_on_the_fix_r1_combine_on_arm():
    """Shoveler's filled G1a table (27f282c2): rate 19/26, p50 20/26, p99 25/26, all three 15/26, diagnostic 24/26."""
    for path, sha in (
        (FIX_R1_COMBINE_ON, FIX_R1_COMBINE_ON_SHA256),
        (VLLM_CAPTURE / "receipt.json", g1.VLLM_RECEIPT_SHA256),
        (VLLM_CAPTURE / "pass-1/routes.pt", g1.VLLM_PASS1_SHA256),
        (VLLM_CAPTURE / "pass-2/routes.pt", g1.VLLM_PASS2_SHA256),
        (FLOOR_RECEIPT, g1.FLOOR_RECEIPT_SHA256),
        (TORCH_TOP6, g1.TORCH_TOP6_SHA256),
    ):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sha
    report = g1.check_vllm_floor(
        torch.load(FIX_R1_COMBINE_ON, map_location="cpu"),
        torch.load(TORCH_TOP6, map_location="cpu"),
        json.loads(FLOOR_RECEIPT.read_bytes()),
        torch.load(VLLM_CAPTURE / "pass-1/routes.pt", map_location="cpu"),
        torch.load(VLLM_CAPTURE / "pass-2/routes.pt", map_location="cpu"),
    )
    layers = report["layers"]
    assert report["lines_passing"] == {"rate": 19, "p50": 20, "p99": 25, "diagnostic": 24}
    assert report["layers_passing"] == 15
    assert [r["layer"] for r in layers if not r["rate_ok"]] == [0, 1, 9, 10, 18, 21, 23]
    assert [r["layer"] for r in layers if not r["p50_ok"]] == [1, 6, 8, 16, 21, 25]
    assert report["overall"]["F_vllm"] == pytest.approx(0.9120, abs=1e-4)
    assert report["overall"]["D_vllm"] == pytest.approx(0.9445, abs=1e-4)
