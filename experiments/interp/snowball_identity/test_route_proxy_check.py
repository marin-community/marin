# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

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
