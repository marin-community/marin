# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Assemble the tokenizer-soak feBPB ladder from W&B run histories.

Each soak arm is one grug-moe run in W&B project ``marin_moe``, group ``tokenizer-soak``,
tagged with its tokenizer arm name. A run's eval trajectory — ``eval/bakeoff-val/macro_bpb``
sampled against cumulative ``throughput/total_gflops`` — is that arm's compute-scaling curve;
``experiments.tokenize.flop_equivalent.fit_bpb_curve`` fits ``BPB(C)=a*C^-b+c`` from it.

Writes the ``{arms: {name: [[train_flops, bpb], ...]}}`` ladder that
``experiments.tokenize.bakeoff_analysis`` consumes, plus a per-domain final-BPB table and,
optionally, per-domain (train_flops, bpb) curves (``--domain-curves-out``) — the same shape
as the macro ladder but one curve per domain per arm, so per-domain BPB can be fit and
compared at a common training-FLOPs budget instead of only each arm's latest checkpoint
(arms sit at different max budgets, so raw final-point comparison is budget-confounded).
The n-gram arm shares arm 2's tokenizer but is a distinct model, so it is emitted under its
own name (``soak-superbpe-64k-ngram``); reuse arm 2's fertility when scoring it.

Run: uv run python -m experiments.tokenize.soak_wandb_ladder --out ladder.json --domains-out domains.json \
    [--domain-curves-out domain_curves.json]
"""

from __future__ import annotations

import argparse
import json

import wandb

PROJECT = "marin-community/marin_moe"
GROUP = "tokenizer-soak"
FLOP_KEY = "throughput/total_gflops"
BPB_KEY = "eval/bakeoff-val/macro_bpb"
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


def _arm_of(run) -> str | None:
    """The scoring arm name for a run: its tokenizer tag, suffixed ``-ngram`` for the n-gram run."""
    arm = next((t for t in getattr(run, "tags", []) if t not in _INFRA_TAGS), None)
    if arm is None:
        return None
    if "ngram" in (run.name or "").lower():
        return f"{arm}-ngram"
    return arm


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


def collect(*, with_domain_curves: bool = False) -> tuple[dict, dict, dict | None]:
    """Return (ladder, domain_finals, domain_curves). Keeps, per arm, the run with the most eval points.

    ``domain_curves`` is ``None`` unless ``with_domain_curves`` (it costs one extra, wider
    history scan per arm, so it is opt-in).
    """
    api = wandb.Api()
    runs = api.runs(PROJECT, filters={"group": GROUP})
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

    ladder = {arm: curve for arm, (_, curve, _) in best.items()}
    domains = {arm: _domain_finals(run, tok) for arm, (run, _, tok) in best.items()}
    domain_curves = None
    if with_domain_curves:
        domain_curves = {arm: _domain_curves(run, tok) for arm, (run, _, tok) in best.items()}
    return {"arms": ladder}, {"domains": list(DOMAINS), "arms": domains}, domain_curves


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="ladder JSON for bakeoff_analysis --bpb")
    ap.add_argument("--domains-out", default=None, help="optional per-domain final-BPB JSON")
    ap.add_argument("--domain-curves-out", default=None, help="optional per-domain (flops, bpb) curves JSON")
    args = ap.parse_args()

    ladder, domains, domain_curves = collect(with_domain_curves=args.domain_curves_out is not None)
    with open(args.out, "w") as f:
        json.dump(ladder, f, indent=2)
    order = sorted(ladder["arms"], key=lambda a: len(ladder["arms"][a]), reverse=True)
    print(f"wrote {args.out}:")
    for arm in order:
        pts = ladder["arms"][arm]
        tail = f" latest bpb={pts[-1][1]:.4f} @ {pts[-1][0]:.2e} FLOPs" if pts else ""
        print(f"  {arm:28s} {len(pts):3d} pts{tail}")
    if args.domains_out:
        with open(args.domains_out, "w") as f:
            json.dump(domains, f, indent=2)
        print(f"wrote {args.domains_out}")
    if args.domain_curves_out:
        with open(args.domain_curves_out, "w") as f:
            json.dump({"domains": list(DOMAINS), "arms": domain_curves}, f, indent=2)
        print(f"wrote {args.domain_curves_out}")


if __name__ == "__main__":
    main()
