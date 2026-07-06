# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fetch the tokenizer-soak results from W&B into a typed :class:`SoakLadder`.

Each soak arm is one grug-moe run in W&B project ``marin_moe``, group ``tokenizer-soak``,
tagged with its tokenizer arm name. This is the "fetch model logs/results" stage of the
bake-off: it reads each arm's run history and assembles the inputs the scoring stage
(:mod:`experiments.tokenization.analysis`) consumes. Every arm becomes an :class:`ArmResults`:

* ``macro_ladder`` — ``(train_flops, macro_bpb)`` points, the macro-BPB compute-scaling curve
  (``eval/bakeoff-val/macro_bpb`` vs cumulative ``throughput/total_gflops``).
  :func:`experiments.tokenization.cost_model.fit_bpb_curve` fits ``BPB(C)=a*C^-b+c`` from it.
* ``fair_macro_ladder`` — the domain-FAIR macro: BPB collapsed to the mean over only the shared
  :data:`DOMAINS`, so a baseline that eval'd those 7 domains and arms that eval'd 11 are compared
  on identical domains. This is the ladder the scoring stage reads.
* ``domain_finals`` / ``domain_curves`` — final and per-FLOP-point per-domain held-out BPB, so
  per-domain BPB compares at a common budget rather than at each arm's (differing) latest checkpoint.
* ``loss_curve`` — per-step training loss, cross-entropy, and perplexity aligned with FLOPs and
  macro-BPB. NOTE: per-token loss/perplexity is NOT comparable across tokenizers with different
  fertility (a denser tokenizer packs more bytes per token, so its per-token loss is higher for
  the same quality); macro-BPB is the tokenizer-agnostic cross-check.
"""

from __future__ import annotations

import json
import math

import click
import wandb
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, apply
from pydantic import BaseModel, Field

PROJECT = "marin-community/marin_moe"
GROUP = "tokenizer-soak"
FLOP_KEY = "throughput/total_gflops"
BPB_KEY = "eval/bakeoff-val/macro_bpb"
STEP_KEY = "global_step"
LOSS_KEY = "train/loss"  # includes router aux losses
CE_KEY = "train/cross_entropy_loss"  # pure LM cross-entropy; perplexity = exp(ce)
DOMAINS = (
    "ao3_english",
    "arxiv_computer_science",
    "arxiv_physics",
    "bbc_news",
    "github_cpp",
    "github_python",
    "wikipedia_english",
)
_INFRA_TAGS = {"grug", "moe", "cw", "h100", "tokenizer-soak"}
# Loss histories run to ~100k steps; keep the artifact record light by storing an evenly-spaced subset.
_MAX_LOSS_POINTS = 250

Curve = list[tuple[float, float]]  # (total_train_flops, value) points


class LossPoint(BaseModel):
    """One training step's loss signature, aligned with FLOPs and (where logged) macro-BPB.

    ``loss``/``cross_entropy``/``perplexity`` are per-token, so they compare only within a fixed
    tokenizer; ``macro_bpb`` is the tokenizer-agnostic cross-check.
    """

    step: int
    loss: float | None
    cross_entropy: float
    perplexity: float
    train_flops: float | None
    macro_bpb: float | None


class ArmResults(BaseModel):
    """One arm's soak results: the macro-BPB scaling curve, per-domain BPB, and the loss history."""

    macro_ladder: Curve  # (train_flops, macro_bpb) — the run-logged macro-BPB curve
    fair_macro_ladder: Curve  # macro over the shared DOMAINS only — the scoring input
    domain_finals: dict[str, float]  # final held-out BPB per domain
    domain_curves: dict[str, Curve]  # per-domain (train_flops, bpb) curves
    loss_curve: list[LossPoint]  # downsampled to at most _MAX_LOSS_POINTS


class SoakLadder(Artifact):
    """The typed output of the "fetch model logs/results" stage: every soak arm's BPB + loss."""

    project: str = ""
    group: str = ""
    domains: list[str] = Field(default_factory=list)
    arms: dict[str, ArmResults] = Field(default_factory=dict)


def _arm_of(run) -> str | None:
    """The scoring arm name for a run: its first tag that is not an infra tag."""
    return next((t for t in run.tags if t not in _INFRA_TAGS), None)


def _curve(run) -> list[list[float]]:
    """(total_train_flops, macro_bpb) points from a run's eval history (flops = total_gflops * 1e9)."""
    pts: list[list[float]] = []
    for row in run.scan_history(keys=[FLOP_KEY, BPB_KEY]):
        flops, bpb = row.get(FLOP_KEY), row.get(BPB_KEY)
        if flops and bpb is not None:
            pts.append([float(flops) * 1e9, float(bpb)])
    pts.sort()
    return pts


def _domain_finals(run, tokenizer_tag: str) -> dict[str, float]:
    """Final per-domain held-out BPB, keyed by domain (``eval/bakeoff-val/<domain>-<tok>/bpb``)."""
    out: dict[str, float] = {}
    for d in DOMAINS:
        v = run.summary.get(f"eval/bakeoff-val/{d}-{tokenizer_tag}/bpb")
        if isinstance(v, (int, float)):
            out[d] = float(v)
    return out


def _domain_curves(run, tokenizer_tag: str) -> dict[str, list[list[float]]]:
    """Per-domain (total_train_flops, bpb) curves from one history scan (same rows as ``_curve``).

    Reads FLOP_KEY alongside every domain's ``eval/bakeoff-val/<domain>-<tok>/bpb`` key in a
    single ``scan_history`` call (they are logged together at each eval step), so each domain
    gets the same eval-step-indexed curve ``_curve`` gets for macro_bpb.
    """
    domain_keys = {d: f"eval/bakeoff-val/{d}-{tokenizer_tag}/bpb" for d in DOMAINS}
    pts: dict[str, list[list[float]]] = {d: [] for d in DOMAINS}
    for row in run.scan_history(keys=[FLOP_KEY, *domain_keys.values()]):
        flops = row.get(FLOP_KEY)
        if not flops:
            continue
        for d, key in domain_keys.items():
            bpb = row.get(key)
            if bpb is not None:
                pts[d].append([float(flops) * 1e9, float(bpb)])
    for d in pts:
        pts[d].sort()
    return pts


def _downsample(items: list, max_points: int) -> list:
    """An evenly-spaced subset of at most ``max_points``, always keeping the first and last."""
    if len(items) <= max_points:
        return items
    stride = (len(items) - 1) / (max_points - 1)
    idx = sorted({round(i * stride) for i in range(max_points)} | {len(items) - 1})
    return [items[i] for i in idx]


def _loss_curve(run) -> list[LossPoint]:
    """Per-step training loss / cross-entropy / perplexity, aligned with FLOPs and macro-BPB.

    Perplexity is ``exp(cross_entropy)``. Loss/perplexity are per-token, so they compare only
    within a fixed tokenizer (denser tokenizers carry more bytes/token and thus higher per-token
    loss at equal quality); the aligned ``macro_bpb`` is the tokenizer-agnostic cross-check.
    Downsampled to :data:`_MAX_LOSS_POINTS` so the artifact record stays light.
    """
    rows: dict[int, dict[str, float | None]] = {}
    for row in run.scan_history(keys=[STEP_KEY, LOSS_KEY, CE_KEY, FLOP_KEY]):
        step = row.get(STEP_KEY)
        if step is None:
            continue
        rows[int(step)] = {"loss": row.get(LOSS_KEY), "ce": row.get(CE_KEY), "flops": row.get(FLOP_KEY)}
    for row in run.scan_history(keys=[STEP_KEY, BPB_KEY]):
        step, bpb = row.get(STEP_KEY), row.get(BPB_KEY)
        if step is not None and bpb is not None and int(step) in rows:
            rows[int(step)]["bpb"] = bpb
    out: list[LossPoint] = []
    for step in sorted(rows):
        v = rows[step]
        if v.get("ce") is None:
            continue
        ce = float(v["ce"])
        out.append(
            LossPoint(
                step=step,
                loss=None if v.get("loss") is None else float(v["loss"]),
                cross_entropy=ce,
                perplexity=math.exp(ce),
                train_flops=None if not v.get("flops") else float(v["flops"]) * 1e9,
                macro_bpb=v.get("bpb"),
            )
        )
    return _downsample(out, _MAX_LOSS_POINTS)


def fair_macro_ladder(domain_curves: dict[str, dict[str, list[list[float]]]]) -> dict[str, list[list[float]]]:
    """Mean BPB over the shared :data:`DOMAINS` at each FLOP point where every domain logged a value.

    Collapses per-domain BPB curves to a domain-fair macro so arms evaluated on different domain
    counts are compared on identical domains.
    """
    out: dict[str, list[list[float]]] = {}
    for arm, doms in domain_curves.items():
        by_flop: dict[float, dict[str, float]] = {}
        for domain in DOMAINS:
            for flop, bpb in doms.get(domain, []):
                by_flop.setdefault(flop, {})[domain] = bpb
        curve = [
            [flop, sum(vals[d] for d in DOMAINS) / len(DOMAINS)]
            for flop, vals in sorted(by_flop.items())
            if len(vals) == len(DOMAINS)
        ]
        if curve:
            out[arm] = curve
    return out


def collect(project: str = PROJECT, group: str = GROUP) -> SoakLadder:
    """Assemble the typed :class:`SoakLadder` from W&B, keeping per arm the run with the most eval points."""
    api = wandb.Api()
    runs = api.runs(project, filters={"group": group})
    best: dict[str, tuple] = {}  # arm -> (run, curve, tokenizer_tag)
    for run in runs:
        arm = _arm_of(run)
        if arm is None:
            continue
        curve = _curve(run)
        if not curve:
            continue
        if arm not in best or len(curve) > len(best[arm][1]):
            tok = next((t for t in run.tags if t not in _INFRA_TAGS), arm)
            best[arm] = (run, curve, tok)

    domain_curves = {arm: _domain_curves(run, tok) for arm, (run, _, tok) in best.items()}
    fair = fair_macro_ladder(domain_curves)
    arms = {
        arm: ArmResults(
            macro_ladder=curve,
            fair_macro_ladder=fair.get(arm, []),
            domain_finals=_domain_finals(run, tok),
            domain_curves=domain_curves[arm],
            loss_curve=_loss_curve(run),
        )
        for arm, (run, curve, tok) in best.items()
    }
    return SoakLadder(project=project, group=group, domains=list(DOMAINS), arms=arms)


def soak_results_step(*, project: str = PROJECT, group: str = GROUP, version: str = "dev") -> ArtifactStep[SoakLadder]:
    """The "fetch model logs/results" stage as an :class:`ArtifactStep`.

    A leaf step: it reads the W&B ``group`` (populated by the training runs' loggers, which are
    decoupled from the checkpoint artifacts) and returns the typed :class:`SoakLadder` that
    :func:`experiments.tokenization.analysis.bakeoff_report_step` scores.
    """
    return apply(
        f"tokenizer-soak/results/{group}",
        collect,
        version=version,
        artifact_type=SoakLadder,
        project=project,
        group=group,
    )


@click.command()
@click.option("--out", required=True, help="write the SoakLadder JSON here")
@click.option("--project", default=PROJECT, help="W&B project the soak runs live in")
@click.option("--group", default=GROUP, help="W&B group tagging the soak runs")
def main(out: str, project: str, group: str) -> None:
    """Pull the soak results into a SoakLadder JSON and print each arm's ladder length."""
    ladder = collect(project, group)
    with open(out, "w") as f:
        json.dump(ladder.model_dump(exclude={"path"}), f, indent=2)
    print(f"wrote {out}")
    order = sorted(ladder.arms, key=lambda a: len(ladder.arms[a].macro_ladder), reverse=True)
    for arm in order:
        pts = ladder.arms[arm].macro_ladder
        tail = f" latest bpb={pts[-1][1]:.4f} @ {pts[-1][0]:.2e} FLOPs" if pts else ""
        print(f"  {arm:30s} {len(pts):3d} bpb pts{tail}")


if __name__ == "__main__":
    main()
