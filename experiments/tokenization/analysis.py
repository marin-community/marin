# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score the tokenizer bake-off from stored measurements — the "compute the final analysis" stage.

This is the replay entry point: it reads the raw outputs the earlier stages logged — the fertility
report (per-arm per-domain token/byte counts) and the BPB ladder (per-arm ``(train_flops, BPB)``
points) — and recomputes serving cost and the FLOP-equivalent BPB (feBPB) ranking under a
:class:`ServingCostModel`. Nothing is retrained; change the assumptions (deployment model size,
serving context, attention sparsity, hardware speed, lifetime serving/training ratio, domain mix)
and the ranking is recomputed from the same raw data.

feBPB credits a cheaper-to-serve tokenizer by reinvesting its serving saving into training:
``train_flops(arm) = C_ref * (1 + serving_ratio*(1 - rel_serve))``, then the arm's fitted BPB curve
is read at that budget. ``serving_ratio=0`` recovers the raw BPB at the reference budget.

:func:`bakeoff_report_step` is the stage as an :class:`ArtifactStep` (deps on the ladder + fertility
steps); :func:`main` is the CLI for ad-hoc re-scoring under new deployment assumptions.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math

from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, apply
from pydantic import BaseModel

from experiments.tokenization.collect_results import SoakLadder
from experiments.tokenization.cost_model import (
    TARGET_MODEL_SHAPE,
    ArmCost,
    ServingCostModel,
    arm_cost,
    febpb,
    fit_bpb_curve,
)
from experiments.tokenization.fertility import FertilityReport


@dataclasses.dataclass(frozen=True)
class ArmRow:
    """One arm's serving-cost line: its name, vocab, weighted fertility, and computed cost."""

    name: str
    vocab: int
    fertility: float
    cost: ArmCost


class ScoredArm(BaseModel):
    """One arm's scored line: serving-cost signature plus its feBPB (``None`` if unfittable)."""

    name: str
    vocab: int
    bytes_per_token: float
    infer_flops_per_byte: float
    rel_serve: float
    lm_head_fraction: float
    attention_fraction: float
    febpb: float | None
    febpb_infeasible: bool


class ServingSummary(BaseModel):
    """The serving-cost-model assumptions a score table was computed under."""

    context_len: int
    attention_window: int
    global_layer_period: int
    speed_factor: float
    hidden_dim: int
    num_layers: int


class ScoreTable(BaseModel):
    """A full feBPB ranking of every arm under one serving assumption, best-first."""

    reference: str
    reference_train_flops: float | None
    serving_ratio: float
    serving: ServingSummary
    arms: list[ScoredArm]


class BakeoffReport(Artifact):
    """The typed output of the scoring stage: the feBPB ranking and the raw-BPB (ratio-0) ranking."""

    febpb: ScoreTable
    raw_bpb: ScoreTable


def weighted_fertility(by_domain: dict[str, dict], weights: dict[str, float] | None) -> float:
    """Overall tokens/byte, optionally reweighting domains (default: natural byte weighting)."""
    if weights is None:
        tokens = sum(d["tokens"] for d in by_domain.values())
        num_bytes = sum(d["bytes"] for d in by_domain.values())
        return tokens / num_bytes
    total_w = sum(weights.get(name, 0.0) for name in by_domain)
    if total_w <= 0:
        raise ValueError("domain weights sum to zero over the measured domains")
    return sum(weights.get(name, 0.0) * (d["tokens"] / d["bytes"]) for name, d in by_domain.items()) / total_w


def arm_rows(fert: dict, serving: ServingCostModel, weights: dict[str, float] | None) -> list[ArmRow]:
    """Price every arm in a fertility report under ``serving``."""
    rows = []
    for arm in fert["arms"]:
        fertility = weighted_fertility(arm["by_domain"], weights)
        cost = arm_cost(arm["name"], arm["vocab_size"], fertility, serving)
        rows.append(ArmRow(name=arm["name"], vocab=arm["vocab_size"], fertility=fertility, cost=cost))
    return rows


def reference_train_flops(bpb_points: dict[str, list], reference: str, ref_budget: float | None) -> float | None:
    """The C_ref feBPB is anchored at: ``ref_budget`` if given, else the reference arm's middle point."""
    if ref_budget is not None:
        return ref_budget
    pts = bpb_points.get(reference)
    if not pts:
        return None
    return sorted(pts)[len(pts) // 2][0]


def score_report(
    fert: dict,
    bpb_points: dict[str, list],
    serving: ServingCostModel,
    *,
    serving_ratio: float = 1.0,
    reference: str = "marin-128k",
    ref_budget: float | None = None,
    weights: dict[str, float] | None = None,
) -> ScoreTable:
    """feBPB ranking of every arm, ordered best-first (trained arms by feBPB).

    Each :class:`ScoredArm` carries the raw serving-cost signature plus ``febpb`` (``None`` for
    arms without the >= 3 BPB points needed to fit ``BPB(C)``). The table echoes the cost-model
    assumptions used.
    """
    rows = arm_rows(fert, serving, weights)
    ref = next((r for r in rows if r.name == reference), rows[0])
    ref_infer = ref.cost.infer_flops_per_byte
    ref_flops = reference_train_flops(bpb_points, reference, ref_budget)

    def febpb_of(r: ArmRow) -> float | None:
        pts = bpb_points.get(r.name, [])
        if ref_flops is None or len(pts) < 3:
            return None
        fit = fit_bpb_curve([tuple(p) for p in pts])
        return febpb(fit, ref_flops, r.cost.infer_flops_per_byte / ref_infer, serving_ratio)

    scored = []
    for r in rows:
        fe = febpb_of(r)
        scored.append(
            ScoredArm(
                name=r.name,
                vocab=r.vocab,
                bytes_per_token=1.0 / r.fertility,
                infer_flops_per_byte=r.cost.infer_flops_per_byte,
                rel_serve=r.cost.infer_flops_per_byte / ref_infer,
                lm_head_fraction=r.cost.lm_head_flop_fraction,
                attention_fraction=r.cost.attention_flop_fraction,
                febpb=None if fe is None or fe == math.inf else fe,
                febpb_infeasible=fe == math.inf,
            )
        )
    # Trained arms (feBPB present) rank first by feBPB; fertility-only arms follow by serving cost.
    scored.sort(key=lambda s: (0, s.febpb) if s.febpb is not None else (1, s.infer_flops_per_byte))
    return ScoreTable(
        reference=reference,
        reference_train_flops=ref_flops,
        serving_ratio=serving_ratio,
        serving=ServingSummary(
            context_len=serving.context_len,
            attention_window=serving.attention_window,
            global_layer_period=serving.global_layer_period,
            speed_factor=serving.speed_factor,
            hidden_dim=serving.model.hidden_dim,
            num_layers=serving.model.num_layers,
        ),
        arms=scored,
    )


def print_report(report: ScoreTable) -> None:
    """Print a feBPB ranking table from a :class:`ScoreTable`."""
    s = report.serving
    print(
        f"=== scored @ ctx={s.context_len}, window={s.attention_window}, "
        f"1:{s.global_layer_period - 1} global:local, speed={s.speed_factor}, "
        f"hidden={s.hidden_dim}, layers={s.num_layers} (reference={report.reference}) ==="
    )
    has_febpb = any(a.febpb is not None for a in report.arms)
    header = f"{'arm':16s} {'vocab':>7s} {'B/tok':>6s} {'infFLOP/B':>10s} {'rel_serve':>9s} {'head%':>5s} {'attn%':>5s}"
    if has_febpb:
        header += f" {'feBPB':>8s}"
    print(header)
    for a in report.arms:
        line = (
            f"{a.name:16s} {a.vocab:7d} {a.bytes_per_token:6.2f} {a.infer_flops_per_byte:10.3e} "
            f"{a.rel_serve:9.3f} {a.lm_head_fraction * 100:4.1f}% {a.attention_fraction * 100:4.1f}%"
        )
        if has_febpb:
            cell = "inf" if a.febpb_infeasible else ("n/a" if a.febpb is None else f"{a.febpb:.4f}")
            line += f" {cell:>8s}"
        print(line)


# --- step ---------------------------------------------------------------------------


def build_bakeoff_report(
    ladder_dir: str,
    fertility_dir: str,
    *,
    serving_ratio: float = 1.0,
    reference: str = "marin-128k",
    ref_budget: float | None = None,
) -> BakeoffReport:
    """Load the typed ladder + fertility artifacts and score at both serving-ratios."""
    ladder = SoakLadder.raw_load(ladder_dir)
    fert = FertilityReport.raw_load(fertility_dir).model_dump()
    bpb_points = {arm: r.fair_macro_ladder for arm, r in ladder.arms.items()}
    serving = ServingCostModel()
    return BakeoffReport(
        febpb=score_report(
            fert, bpb_points, serving, serving_ratio=serving_ratio, reference=reference, ref_budget=ref_budget
        ),
        raw_bpb=score_report(fert, bpb_points, serving, serving_ratio=0.0, reference=reference, ref_budget=ref_budget),
    )


def bakeoff_report_step(
    ladder: ArtifactStep[SoakLadder],
    fertility: ArtifactStep[FertilityReport],
    *,
    serving_ratio: float = 1.0,
    reference: str = "marin-128k",
    ref_budget: float | None = 6e19,
    version: str = "dev",
) -> ArtifactStep[BakeoffReport]:
    """The "compute the final analysis" stage: score the ``ladder`` against ``fertility``.

    Deps on the two upstream artifacts; scores at the given ``serving_ratio`` (feBPB) and at 0 (raw
    BPB) at ``ref_budget`` (default the 6e19 common floor), returning a typed :class:`BakeoffReport`.
    """
    return apply(
        "tokenizer-soak/report",
        build_bakeoff_report,
        version=version,
        artifact_type=BakeoffReport,
        ladder_dir=ladder,
        fertility_dir=fertility,
        serving_ratio=serving_ratio,
        reference=reference,
        ref_budget=ref_budget,
    )


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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fertility", required=True, help="raw fertility JSON from experiments.tokenization.fertility")
    ap.add_argument("--bpb", default=None, help="ladder JSON: {arms: {name: [[flops, bpb]]}} (e.g. bpb_macro7.json)")
    ap.add_argument("--context-len", type=int, default=16_384)
    ap.add_argument("--attention-window", type=int, default=4_096)
    ap.add_argument("--global-period", type=int, default=6, help="1 global layer per N (6 => 5:1 local:global)")
    ap.add_argument("--speed-factor", type=float, default=1.0)
    ap.add_argument("--target-hidden", type=int, default=None, help="override deployment hidden_dim")
    ap.add_argument("--target-layers", type=int, default=None, help="override deployment num_layers")
    ap.add_argument("--serving-ratio", type=float, default=1.0, help="lifetime serving/training weight for feBPB")
    ap.add_argument("--domain-weights", type=str, default=None, help="k=v,k=v domain mix (default: natural bytes)")
    ap.add_argument("--reference", type=str, default="marin-128k", help="reference arm for relative cost / feBPB")
    ap.add_argument("--ref-budget", type=float, default=None, help="override C_ref (train FLOPs) for feBPB")
    args = ap.parse_args()

    with open(args.fertility) as f:
        fert = json.load(f)
    bpb_points: dict[str, list] = {}
    if args.bpb:
        with open(args.bpb) as f:
            bpb_points = json.load(f).get("arms", {})

    report = score_report(
        fert,
        bpb_points,
        _serving_from_args(args),
        serving_ratio=args.serving_ratio,
        reference=args.reference,
        ref_budget=args.ref_budget,
        weights=_parse_weights(args.domain_weights),
    )
    print_report(report)
    if not any(a.febpb is not None for a in report.arms):
        print("\n(no --bpb results: ranking by serving cost only. Add a ladder for feBPB.)")


if __name__ == "__main__":
    main()
