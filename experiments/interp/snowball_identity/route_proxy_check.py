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

G1a (``t310/precision-preflight-amendment-G1a-filled-shoveler.md``, sha256 27f282c2), ``check_vllm_floor``: the
floor is the served stack. Inputs add marten's native-API route capture (receipt 3f574c81): pass 1 and pass 2
``routes.pt`` (sequence id -> executed top-4 ids [T, L, 4]). Per layer:

- ``F_l``: vLLM pass 1 vs torch batch-1 set agreement; ``D_l``: vLLM pass 1 vs pass 2 (reported).
- (a) ``1 - A_l <= K_RATE_VLLM * (1 - F_l)``.
- (b) at Levanter's set disagreements with torch, torch batch-1's 4th/5th margin p50 and p99 each at most
  ``MARGIN_FACTOR`` times torch's margin at vLLM-vs-torch set disagreements (p99 only from ``MIN_FOR_P99``).
- Diagnostic, not gated: Levanter vs vLLM pass 1 set agreement ``>= D_l - (1 - F_l)``; Levanter vs both passes,
  set and ordered, and the torch floor's ``F_l`` from receipt 34a1eb66 are reported.
"""

import argparse
import hashlib
import io
import json
from collections import defaultdict

import numpy as np
import torch

from experiments.interp.snowball_identity.logit_check import WINDOW, read, write_json

K_RATE = 2.0
MARGIN_FACTOR = 1.5
MIN_FOR_P99 = 30
"""G1 rule 2: k, the margin factor and the p99 cutoff, fixed before the rerun."""
FLOOR_RECEIPT_SHA256 = "34a1eb661bef3e2b3b3940125fcb7a1b5ddd3c66652ae730975443ac24838212"
TORCH_TOP6_SHA256 = "ddad3a88f883ad6e3ffd75d99a94edef54cdd51edea55575af478547b42b7ba5"
K_RATE_VLLM = 1.25
"""G1a: k under the served-stack floor, fixed before the collector ran."""
VLLM_RECEIPT_SHA256 = "3f574c81aad28c227e6e9f0ea54e1e61730f26fb0d6faac9d77d0b2abd6dee61"
VLLM_PASS1_SHA256 = "92b92899eaf1a1288ca9fb53e703cfc3e1d6cc79e4bf81c9ffdb30ab092e7147"
VLLM_PASS2_SHA256 = "10c0730e7443deeee9c405b327361626e9afcd735c57072e01733b9d409ea3ed"
TORCH_BATCHED_TOP6_SHA256 = "1c998ced2bba73c358215fe1647fcd900ad8a3a30f8f36b7254cc7efe713f842"
"""The torch floor job's ``batched/top6.pt`` (torch on the gate grid), G1b's floor."""
K_G1B = 1.25
"""G1b direct L0 line: Levanter's L0 disagreement with vLLM may be at most this multiple of torch batched's."""
TORCH_SIDE = "torch_batch1"
K = 4
"""Experts per token: the selected set is the top ``K``; the 4th/5th margin decides its last member."""
MARGIN_QUANTILES = (0.5, 0.9, 0.99)
MARGIN_BELOW = (0.01, 0.05, 0.1, 0.25, 0.5)


def spans(sequence_id: str, length: int) -> dict[str, slice]:
    group = sequence_id.split("|")[0]
    found = {group: slice(None)}
    if group == "long" and length > WINDOW:
        found["long_past_window"] = slice(WINDOW, None)
    return found


def margin_summary(margins: np.ndarray) -> dict:
    if margins.size == 0:
        return {"count": 0}
    return {
        "count": int(margins.size),
        "max": float(margins.max()),
        **{f"p{round(q * 100)}": float(np.quantile(margins, q)) for q in MARGIN_QUANTILES},
        **{f"below_{limit}": int((margins < limit).sum()) for limit in MARGIN_BELOW},
    }


def agreement(ours: dict[str, np.ndarray], theirs: dict[str, np.ndarray], margins: dict[str, dict]) -> dict:
    """Routes [T, L, K] per sequence on both sides: per layer and group, positions, set and ordered agreement (as
    ``routing.compare_routes_cross_stack`` counts them); per layer, each side's 4th/5th margin [T, L] (``margins``
    maps a side name to ``{sequence id: margin}``) at the set-disagreeing position-layers."""
    if set(ours) != set(theirs):
        raise ValueError("the two sides cover different sequence ids")
    counts: dict = defaultdict(lambda: {"positions": 0, "set_agree": 0, "ordered_agree": 0})
    at_disagreements: dict = defaultdict(list)
    for sequence_id in sorted(ours):
        a, b = ours[sequence_id].astype(np.int64), theirs[sequence_id].astype(np.int64)
        if a.shape != b.shape:
            raise ValueError(f"{sequence_id}: route shapes differ, {a.shape} vs {b.shape}")
        ordered = (a == b).all(-1)
        same_set = (np.sort(a, -1) == np.sort(b, -1)).all(-1)
        for name, rows in spans(sequence_id, a.shape[0]).items():
            for layer in range(a.shape[1]):
                count = counts[(layer, name)]
                count["positions"] += int(ordered[rows, layer].size)
                count["set_agree"] += int(same_set[rows, layer].sum())
                count["ordered_agree"] += int(ordered[rows, layer].sum())
        for side, by_sequence in margins.items():
            margin = by_sequence[sequence_id]
            for layer in range(a.shape[1]):
                at_disagreements[(side, layer)].append(margin[~same_set[:, layer], layer])
    groups = sorted({name for _, name in counts})
    main = [g for g in groups if g != "long_past_window"]
    layers = []
    for layer in sorted({layer for layer, _ in counts}):
        total = {
            name: sum(counts[(layer, g)][name] for g in main) for name in ("positions", "set_agree", "ordered_agree")
        }
        layers.append(
            {
                "layer": layer,
                "set_rate": total["set_agree"] / total["positions"],
                "ordered_rate": total["ordered_agree"] / total["positions"],
                **total,
                "groups": {g: counts[(layer, g)] for g in groups},
                "margin_at_set_disagreements": {
                    side: margin_summary(np.concatenate(at_disagreements[(side, layer)])) for side in margins
                },
            }
        )
    totals = {name: sum(row[name] for row in layers) for name in ("positions", "set_agree", "ordered_agree")}
    return {"layers": layers, "totals": totals}


def routes_and_margins(top6: dict) -> tuple[dict, dict]:
    routes = {sid: r["ids"].numpy()[..., :K] for sid, r in top6.items()}
    margins = {sid: (r["biased"][..., K - 1] - r["biased"][..., K]).numpy() for sid, r in top6.items()}
    return routes, margins


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


def _routes(by_sequence: dict) -> dict[str, np.ndarray]:
    return {sid: np.asarray(value) for sid, value in by_sequence.items()}


def check_vllm_floor(levanter: dict, torch_top6: dict, floor: dict, vllm_pass1: dict, vllm_pass2: dict) -> dict:
    """G1a: ``check`` with the floor restated from the served stack (``vllm_pass1`` / ``vllm_pass2``: sequence id ->
    executed top-4 ids [T, L, 4]); ``floor`` (the torch floor receipt) is reported only."""
    routes, margins = routes_and_margins(torch_top6)
    ours = {sid: r["routes"].numpy() for sid, r in levanter.items()}
    pass1, pass2 = _routes(vllm_pass1), _routes(vllm_pass2)
    lev_torch = agreement(ours, routes, {TORCH_SIDE: margins})
    vllm_torch = agreement(pass1, routes, {TORCH_SIDE: margins})
    vllm_repeat = agreement(pass1, pass2, {})
    lev_vllm = [agreement(ours, p, {}) for p in (pass1, pass2)]
    torch_floor = floor["floor"]["batched_vs_routes_pt"]["layers"]
    layers, failures = [], []
    for found, f, d, lv1, lv2, tf in zip(
        lev_torch["layers"],
        vllm_torch["layers"],
        vllm_repeat["layers"],
        *(x["layers"] for x in lev_vllm),
        torch_floor,
        strict=True,
    ):
        layer = found["layer"]
        if not found["positions"] == f["positions"] == d["positions"] == lv1["positions"]:
            failures.append(f"L{layer}: position-layers differ ({found['positions']}, vLLM {f['positions']})")
        rate, floor_rate = 1 - found["set_rate"], 1 - f["set_rate"]
        theirs = found["margin_at_set_disagreements"][TORCH_SIDE]
        served = f["margin_at_set_disagreements"][TORCH_SIDE]
        row = {
            "layer": layer,
            "A": found["set_rate"],
            "F_vllm": f["set_rate"],
            "D_vllm": d["set_rate"],
            "F_torch": tf["set_rate"],
            "ordered": found["ordered_rate"],
            "rate_ratio": rate / floor_rate if floor_rate else None,
            "rate_ok": rate <= K_RATE_VLLM * floor_rate,
            "disagreements": theirs["count"],
            "margin_p50": theirs.get("p50"),
            "vllm_torch_p50": served.get("p50"),
            "margin_p99": theirs.get("p99"),
            "vllm_torch_p99": served.get("p99"),
            "levanter_vs_vllm": {
                "set": [lv1["set_rate"], lv2["set_rate"]],
                "ordered": [lv1["ordered_rate"], lv2["ordered_rate"]],
            },
            "diagnostic_bound": d["set_rate"] - floor_rate,
        }
        # With no vLLM-vs-torch disagreements there is no served margin to compare with: any Levanter one fails.
        served_p50, served_p99 = served.get("p50", float("-inf")), served.get("p99", float("-inf"))
        row["p50_ok"] = theirs["count"] == 0 or theirs["p50"] <= MARGIN_FACTOR * served_p50
        row["p99_checked"] = theirs["count"] >= MIN_FOR_P99
        row["p99_ok"] = not row["p99_checked"] or theirs["p99"] <= MARGIN_FACTOR * served_p99
        row["diagnostic_ok"] = lv1["set_rate"] >= row["diagnostic_bound"]
        if not row["rate_ok"]:
            failures.append(f"L{layer}: disagreement rate {rate:.4f} > {K_RATE_VLLM} x vLLM floor {floor_rate:.4f}")
        if not row["p50_ok"]:
            failures.append(
                f"L{layer}: margin p50 {theirs['p50']:.4f} > {MARGIN_FACTOR} x vLLM-vs-torch {served_p50:.4f}"
            )
        if not row["p99_ok"]:
            failures.append(
                f"L{layer}: margin p99 {theirs['p99']:.4f} > {MARGIN_FACTOR} x vLLM-vs-torch {served_p99:.4f}"
            )
        layers.append(row)

    def overall(report: dict) -> float:
        return report["totals"]["set_agree"] / report["totals"]["positions"]

    return {
        "rule": {"k": K_RATE_VLLM, "margin_factor": MARGIN_FACTOR, "min_for_p99": MIN_FOR_P99, "floor": "vllm"},
        "layers": layers,
        "layers_passing": sum(r["rate_ok"] and r["p50_ok"] and r["p99_ok"] for r in layers),
        "lines_passing": {line: sum(r[f"{line}_ok"] for r in layers) for line in ("rate", "p50", "p99", "diagnostic")},
        "overall": {
            "A": overall(lev_torch),
            "F_vllm": overall(vllm_torch),
            "D_vllm": overall(vllm_repeat),
            "levanter_vs_vllm": [overall(x) for x in lev_vllm],
        },
        "totals": lev_torch["totals"],
        "failures": failures,
    }


def g1b_l0(levanter: dict, torch_batched_top6: dict, vllm_pass1: dict, vllm_pass2: dict) -> dict:
    """G1b's direct L0 line: L0 set agreement with each vLLM pass for Levanter (``LV``) and for torch batched on the
    gate grid (``TB``); passes when (1 - min LV) <= ``K_G1B`` x (1 - min TB)."""
    ours = {sid: r["routes"].numpy() for sid, r in levanter.items()}
    batched, _ = routes_and_margins(torch_batched_top6)
    passes = [_routes(vllm_pass1), _routes(vllm_pass2)]

    def l0(side: dict) -> list[float]:
        rates = []
        for routes in passes:
            first = agreement(side, routes, {})["layers"][0]
            if first["layer"] != 0:
                raise ValueError(f"first layer reported is {first['layer']}")
            rates.append(first["set_rate"])
        return rates

    lv, tb = l0(ours), l0(batched)
    excess, floor = 1 - min(lv), 1 - min(tb)
    ok = excess <= K_G1B * floor
    ratio = excess / floor if floor else None
    return {
        "rule": {"k": K_G1B, "line": "(1 - min LV_0) <= k x (1 - min TB_0), min over the two vLLM passes"},
        "TB_0": tb,
        "LV_0": lv,
        "TB_0_min": min(tb),
        "LV_0_min": min(lv),
        "ratio": ratio,
        "ok": ok,
        "failures": [] if ok else [f"L0: 1 - {min(lv):.5f} > {K_G1B} x (1 - {min(tb):.5f}); ratio {ratio}"],
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
    parser.add_argument(
        "--vllm-capture",
        help=f"G1a: marten's capture directory (receipt.json {VLLM_RECEIPT_SHA256[:8]}, pass-1/ and pass-2/routes.pt)",
    )
    args = parser.parse_args()
    positions = load_pinned(args.positions, args.positions_sha256)
    inputs = (
        torch.load(io.BytesIO(positions), map_location="cpu"),
        torch.load(io.BytesIO(load_pinned(args.torch_top6, TORCH_TOP6_SHA256)), map_location="cpu"),
        json.loads(load_pinned(args.floor_receipt, FLOOR_RECEIPT_SHA256)),
    )
    if args.vllm_capture is None:
        report = check(*inputs)
    else:
        load_pinned(f"{args.vllm_capture}/receipt.json", VLLM_RECEIPT_SHA256)
        report = check_vllm_floor(
            *inputs,
            *(
                torch.load(io.BytesIO(load_pinned(f"{args.vllm_capture}/pass-{n}/routes.pt", sha)), map_location="cpu")
                for n, sha in ((1, VLLM_PASS1_SHA256), (2, VLLM_PASS2_SHA256))
            ),
        )
        report["inputs_vllm"] = {
            "capture": args.vllm_capture,
            "receipt_sha256": VLLM_RECEIPT_SHA256,
            "pass1_sha256": VLLM_PASS1_SHA256,
            "pass2_sha256": VLLM_PASS2_SHA256,
        }
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
