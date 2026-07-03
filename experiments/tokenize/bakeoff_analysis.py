# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Re-score the tokenizer bake-off from stored raw measurements under configurable assumptions.

This is the replay entry point: it reads the raw outputs the experiments logged — the
fertility report (per-arm per-domain token/byte counts) and, when available, the training
results (per-arm (training FLOPs, BPB) points) — and recomputes the serving cost, the
Pareto frontier, and the FLOP-equivalent BPB ranking under a :class:`ServingCostModel`
you specify on the command line. Nothing is retrained; change the assumptions (deployment
model size, serving context window, attention sparsity, hardware speed, lifetime
serving/training ratio, domain mix) and the ranking is recomputed from the same raw data.

Inputs
------
--fertility PATH   JSON from ``experiments.tokenize.fertility_report``
                   ({domains, arms:[{by_domain:{d:{tokens,bytes}}}]}).
--bpb PATH         Optional JSON of training results: {arms: {name: [[train_flops, bpb], ...]}}.
                   With >= 3 points per arm, feBPB and the full ranking are produced; without
                   it, only the fertility/serving-cost ranking.

Cost-model knobs (all optional; defaults = the deployment target at 16k context)
--context-len, --attention-window, --global-period, --speed-factor,
--target-hidden, --target-layers, --serving-ratio, --domain-weights k=v,k=v

Run:  uv run python -m experiments.tokenize.bakeoff_analysis --fertility fertility_raw.json [--bpb results.json]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math

from experiments.tokenize.flop_equivalent import (
    TARGET_MODEL_SHAPE,
    ServingCostModel,
    arm_cost,
    febpb,
    fit_bpb_curve,
)


def _weighted_fertility(by_domain: dict[str, dict], weights: dict[str, float] | None) -> float:
    """Overall tokens/byte, optionally reweighting domains (default: natural byte weighting)."""
    if weights is None:
        tokens = sum(d["tokens"] for d in by_domain.values())
        num_bytes = sum(d["bytes"] for d in by_domain.values())
        return tokens / num_bytes
    # Reweight: treat each domain's measured fertility as its rate, mix by the given weights.
    total_w = sum(weights.get(name, 0.0) for name in by_domain)
    if total_w <= 0:
        raise ValueError("domain weights sum to zero over the measured domains")
    return sum(weights.get(name, 0.0) * (d["tokens"] / d["bytes"]) for name, d in by_domain.items()) / total_w


def _serving_from_args(args: argparse.Namespace) -> ServingCostModel:
    model = TARGET_MODEL_SHAPE
    if args.target_hidden is not None:
        model = dataclasses.replace(model, hidden_dim=args.target_hidden, num_heads=args.target_hidden // 128)
    if args.target_layers is not None:
        model = dataclasses.replace(model, num_layers=args.target_layers)
    return ServingCostModel(
        model=model,
        context_len=args.context_len,
        attention_window=args.attention_window,
        global_layer_period=args.global_period,
        speed_factor=args.speed_factor,
    )


def _parse_weights(spec: str | None) -> dict[str, float] | None:
    if not spec:
        return None
    out: dict[str, float] = {}
    for pair in spec.split(","):
        k, v = pair.split("=")
        out[k.strip()] = float(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fertility", required=True, help="raw fertility JSON from fertility_report")
    ap.add_argument("--bpb", default=None, help="optional training-results JSON: {arms: {name: [[flops, bpb]]}}")
    ap.add_argument("--context-len", type=int, default=16_384)
    ap.add_argument("--attention-window", type=int, default=4_096)
    ap.add_argument("--global-period", type=int, default=6, help="1 global layer per N (6 => 5:1 local:global)")
    ap.add_argument("--speed-factor", type=float, default=1.0)
    ap.add_argument("--target-hidden", type=int, default=None, help="override deployment hidden_dim")
    ap.add_argument("--target-layers", type=int, default=None, help="override deployment num_layers")
    ap.add_argument("--serving-ratio", type=float, default=1.0, help="lifetime serving/training weight for feBPB")
    ap.add_argument("--domain-weights", type=str, default=None, help="k=v,k=v domain mix (default: natural bytes)")
    ap.add_argument("--reference", type=str, default="marin-128k", help="reference arm for relative cost / feBPB")
    args = ap.parse_args()

    with open(args.fertility) as f:
        fert = json.load(f)
    weights = _parse_weights(args.domain_weights)
    serving = _serving_from_args(args)

    bpb_points: dict[str, list] = {}
    if args.bpb:
        with open(args.bpb) as f:
            bpb_points = json.load(f).get("arms", {})

    rows = []
    for arm in fert["arms"]:
        fertility = _weighted_fertility(arm["by_domain"], weights)
        cost = arm_cost(arm["name"], arm["vocab_size"], fertility, serving)
        rows.append({"name": arm["name"], "vocab": arm["vocab_size"], "fertility": fertility, "cost": cost})

    ref = next((r for r in rows if r["name"] == args.reference), rows[0])
    ref_infer = ref["cost"].infer_flops_per_byte
    ref_train_flops = None
    if bpb_points.get(args.reference):
        # C_ref = the middle compute point of the reference arm.
        pts = sorted(bpb_points[args.reference])
        ref_train_flops = pts[len(pts) // 2][0]

    attn = serving.attention_flop_fraction(ref["vocab"]) * 100
    print(
        f"=== re-scored @ ctx={serving.context_len}, window={serving.attention_window}, "
        f"1:{serving.global_layer_period - 1} global:local, speed={serving.speed_factor}, "
        f"hidden={serving.model.hidden_dim}, layers={serving.model.num_layers} ==="
    )
    print(f"(attention = {attn:.1f}% of forward FLOPs; reference = {args.reference})")
    has_febpb = ref_train_flops is not None
    header = f"{'arm':16s} {'vocab':>7s} {'B/tok':>6s} {'infFLOP/B':>10s} {'rel_serve':>9s} {'head%':>5s} {'attn%':>5s}"
    if has_febpb:
        header += f" {'feBPB':>8s}"
    print(header)

    def score(r: dict) -> float:
        if not has_febpb or r["name"] not in bpb_points or len(bpb_points[r["name"]]) < 3:
            return r["cost"].infer_flops_per_byte  # fall back to serving-cost ordering
        fit = fit_bpb_curve([tuple(p) for p in bpb_points[r["name"]]])
        rel = r["cost"].infer_flops_per_byte / ref_infer
        return febpb(fit, ref_train_flops, rel, args.serving_ratio)

    for r in sorted(rows, key=score):
        c = r["cost"]
        line = (
            f"{r['name']:16s} {r['vocab']:7d} {1.0 / r['fertility']:6.2f} {c.infer_flops_per_byte:10.3e} "
            f"{c.infer_flops_per_byte / ref_infer:9.3f} {c.lm_head_flop_fraction * 100:4.1f}% "
            f"{c.attention_flop_fraction * 100:4.1f}%"
        )
        if has_febpb:
            fe = score(r)
            line += f" {'inf' if fe == math.inf else f'{fe:8.4f}':>8s}"
        print(line)

    if not has_febpb:
        print("\n(no --bpb results: ranking by serving cost only. Add training results for feBPB.)")


if __name__ == "__main__":
    main()
