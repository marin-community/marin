# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""#310 amendment G1 (``t310/precision-preflight-amendment-G1-shoveler.md``, sha256 a685d499): the routes row as a
bounded proxy check against torch, per layer. CPU only; fixed before the fix-pin rerun reports.

Inputs: an arm's ``positions.pt`` (Levanter routes, ``levanter_positions`` with ``routes=True``), the torch floor's
``batch1/top6.pt`` (equal to the gate's ``routes.pt``) and the floor ``receipt.json``, each checked against its
pinned sha256. Per layer ``l``, over the gate's position-layers:

- ``A_l``: Levanter's selected-set agreement with torch batch-1; ``F_l``: torch on the gate's grids vs ``routes.pt``.
- (a) rate bound: ``1 - A_l <= K_RATE * (1 - F_l)``.
- (b) near-tie class: at Levanter's set disagreements, torch batch-1's 4th/5th biased-logit margin has p50 and p99
  each at most ``MARGIN_FACTOR`` times torch's own at that layer. Torch's own is torch batch-1's margin at the
  floor's batched-vs-batch-1 set disagreements (the same position-layers as ``F_l``, measured the same way). With
  fewer than ``MIN_FOR_P99`` Levanter disagreements only p50 is compared.

Ordered agreement is reported, not gated. The log-probability lines are not part of this check.
"""

import argparse
import hashlib
import io
import json

import torch

from experiments.interp.snowball_identity.logit_check import read
from experiments.interp.snowball_identity.precision_gate import write_json
from experiments.interp.snowball_identity.torch_route_floor import agreement, routes_and_margins

K_RATE = 2.0
MARGIN_FACTOR = 1.5
MIN_FOR_P99 = 30
"""G1 rule 2: k, the margin factor and the p99 cutoff, fixed before the rerun."""
FLOOR_RECEIPT_SHA256 = "34a1eb661bef3e2b3b3940125fcb7a1b5ddd3c66652ae730975443ac24838212"
TORCH_TOP6_SHA256 = "ddad3a88f883ad6e3ffd75d99a94edef54cdd51edea55575af478547b42b7ba5"
TORCH_SIDE = "torch_batch1"


def check(levanter: dict, torch_top6: dict, floor: dict) -> dict:
    """``levanter``: the arm's positions records; ``torch_top6``: batch1/top6.pt; ``floor``: the floor receipt."""
    routes, margins = routes_and_margins(torch_top6)
    ours = agreement({sid: r["routes"].numpy() for sid, r in levanter.items()}, routes, {TORCH_SIDE: margins})
    shape = floor["floor"]["batched_vs_routes_pt"]["layers"]
    own = floor["floor"]["batched_vs_batch1"]["layers"]
    layers, failures = [], []
    for found, f, o in zip(ours["layers"], shape, own, strict=True):
        layer = found["layer"]
        if found["positions"] != f["positions"]:
            failures.append(f"L{layer}: {found['positions']} position-layers, floor {f['positions']}")
        rate, floor_rate = 1 - found["set_rate"], 1 - f["set_rate"]
        theirs, torch_own = found["margin_at_set_disagreements"][TORCH_SIDE], o["margin_at_set_disagreements"]["batch1"]
        row = {
            "layer": layer,
            "A": found["set_rate"],
            "F": f["set_rate"],
            "ordered": found["ordered_rate"],
            "rate_ratio": rate / floor_rate,
            "rate_ok": rate <= K_RATE * floor_rate,
            "disagreements": theirs["count"],
            "margin_p50": theirs.get("p50"),
            "torch_p50": torch_own["p50"],
            "margin_p99": theirs.get("p99"),
            "torch_p99": torch_own["p99"],
        }
        row["p50_ok"] = theirs["count"] == 0 or theirs["p50"] <= MARGIN_FACTOR * torch_own["p50"]
        row["p99_checked"] = theirs["count"] >= MIN_FOR_P99
        row["p99_ok"] = not row["p99_checked"] or theirs["p99"] <= MARGIN_FACTOR * torch_own["p99"]
        if not row["rate_ok"]:
            failures.append(f"L{layer}: disagreement rate {rate:.4f} > {K_RATE} x floor {floor_rate:.4f}")
        if not row["p50_ok"]:
            failures.append(f"L{layer}: margin p50 {theirs['p50']:.4f} > {MARGIN_FACTOR} x torch {torch_own['p50']:.4f}")
        if not row["p99_ok"]:
            failures.append(f"L{layer}: margin p99 {theirs['p99']:.4f} > {MARGIN_FACTOR} x torch {torch_own['p99']:.4f}")
        layers.append(row)
    return {
        "rule": {"k": K_RATE, "margin_factor": MARGIN_FACTOR, "min_for_p99": MIN_FOR_P99},
        "layers": layers,
        "layers_passing": sum(r["rate_ok"] and r["p50_ok"] and r["p99_ok"] for r in layers),
        "totals": ours["totals"],
        "failures": failures,
    }


def load_pinned(path: str, sha256: str | None) -> bytes:
    data = read(path)
    actual = hashlib.sha256(data).hexdigest()
    if sha256 is not None and actual != sha256:
        raise ValueError(f"{path}: sha256 {actual}, pinned {sha256}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--positions", required=True, help="the arm's positions.pt (with routes)")
    parser.add_argument("--positions-sha256", required=True)
    parser.add_argument("--torch-top6", required=True, help=f"torch floor batch1/top6.pt ({TORCH_TOP6_SHA256[:8]})")
    parser.add_argument("--floor-receipt", required=True, help=f"torch floor receipt.json ({FLOOR_RECEIPT_SHA256[:8]})")
    parser.add_argument("--out", required=True, help="report JSON (local or s3://)")
    args = parser.parse_args()
    positions = load_pinned(args.positions, args.positions_sha256)
    report = check(
        torch.load(io.BytesIO(positions), map_location="cpu"),
        torch.load(io.BytesIO(load_pinned(args.torch_top6, TORCH_TOP6_SHA256)), map_location="cpu"),
        json.loads(load_pinned(args.floor_receipt, FLOOR_RECEIPT_SHA256)),
    )
    report["inputs"] = {
        "positions": args.positions,
        "positions_sha256": hashlib.sha256(positions).hexdigest(),
        "torch_top6_sha256": TORCH_TOP6_SHA256,
        "floor_receipt_sha256": FLOOR_RECEIPT_SHA256,
    }
    write_json(args.out, report)
    print(json.dumps({"layers_passing": report["layers_passing"], "failures": len(report["failures"])}))


if __name__ == "__main__":
    main()
