# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Extract the (training FLOPs, BPB) point a finished grug-moe run contributes to the ladder.

A grug run logs metrics through Levanter's ``json_logger`` tracker as one JSON object per
event, either persisted to ``tracker_metrics.jsonl`` or emitted on the run's stdout (which
``iris job logs`` echoes, one metrics object per line behind a log prefix). This reads
either form, pulls the final value of each metric of interest, and reduces the run to the
single ``(total_train_flops, bpb)`` point it represents at its budget.

The isoFLOP ladder is several such runs per arm; ``build_ladder`` assembles their points
into the ``{arms: {name: [[flops, bpb], ...]}}`` file that
``experiments.tokenize.bakeoff_analysis`` consumes.

Metric keys (confirmed from a live run's json_logger stream):
  eval/bpb                              held-out bits-per-byte  (REQUIRES compute_bpb=True)
  eval/macro_bpb                        macro-averaged BPB across domains, when present
  throughput/total_gflops               cumulative achieved GFLOPs (fwd+bwd) — the train-FLOP axis
  throughput/total_tokens               cumulative tokens seen
  throughput/flops_per_token_analytic   analytic fwd FLOPs/token (for a model-based cross-check)
  train/loss, train/cross_entropy_loss  final training loss

Run:  uv run python -m experiments.tokenize.collect_metrics run --metrics tracker_metrics.jsonl --arm marin-128k
      iris ... job logs <id> | uv run python -m experiments.tokenize.collect_metrics run --arm marin-128k
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass

TRAIN_FLOP_MULTIPLIER = 3.0

# Metrics we track the last-seen value of. Attention: total_gflops/total_tokens are
# cumulative; eval/* and train/* are point-in-time and we keep the final report.
_TRACKED = (
    "eval/bpb",
    "eval/macro_bpb",
    "eval/loss",
    "throughput/total_gflops",
    "throughput/total_tokens",
    "throughput/flops_per_token_analytic",
    "train/loss",
    "train/cross_entropy_loss",
)


def _iter_metric_events(lines: Iterable[str]) -> Iterator[dict]:
    """Yield the metrics dict of each json_logger event on a stream of raw or log-prefixed lines.

    Tolerates the ``iris job logs`` prefix by parsing from the first ``{"tracker"`` on the line.
    """
    for line in lines:
        start = line.find('{"tracker"')
        if start == -1:
            continue
        try:
            event = json.loads(line[start:])
        except json.JSONDecodeError:
            continue
        metrics = event.get("metrics")
        if isinstance(metrics, dict):
            yield metrics


@dataclass(frozen=True)
class RunPoint:
    """One ladder point: a run's cumulative training FLOPs and its final held-out BPB."""

    arm: str
    total_train_flops: float
    bpb: float | None
    macro_bpb: float | None
    total_tokens: float | None
    flops_per_token_analytic: float | None
    train_loss: float | None

    def as_ladder_pair(self) -> tuple[float, float]:
        if self.bpb is None:
            raise ValueError(f"run for arm {self.arm!r} logged no eval/bpb — was compute_bpb enabled?")
        return (self.total_train_flops, self.bpb)


def collect_run(lines: Iterable[str], arm: str) -> RunPoint:
    """Reduce one run's metric stream to its final ladder point.

    ``total_train_flops`` prefers the logged cumulative ``throughput/total_gflops`` (achieved
    fwd+bwd FLOPs); if absent it reconstructs ``3 * flops_per_token_analytic * total_tokens``.
    """
    last: dict[str, float] = {}
    for metrics in _iter_metric_events(lines):
        for key in _TRACKED:
            if key in metrics and metrics[key] is not None:
                last[key] = float(metrics[key])

    total_tokens = last.get("throughput/total_tokens")
    fpt = last.get("throughput/flops_per_token_analytic")
    if "throughput/total_gflops" in last:
        total_train_flops = last["throughput/total_gflops"] * 1e9
    elif fpt is not None and total_tokens is not None:
        total_train_flops = TRAIN_FLOP_MULTIPLIER * fpt * total_tokens
    else:
        raise ValueError(f"run for arm {arm!r}: no FLOP metric (total_gflops or flops_per_token_analytic+tokens)")

    return RunPoint(
        arm=arm,
        total_train_flops=total_train_flops,
        bpb=last.get("eval/bpb"),
        macro_bpb=last.get("eval/macro_bpb"),
        total_tokens=total_tokens,
        flops_per_token_analytic=fpt,
        train_loss=last.get("train/loss") or last.get("train/cross_entropy_loss"),
    )


def build_ladder(points: Iterable[RunPoint]) -> dict:
    """Assemble run points into the {arms: {name: [[flops, bpb], ...]}} bakeoff_analysis format."""
    arms: dict[str, list[list[float]]] = {}
    for p in points:
        arms.setdefault(p.arm, []).append(list(p.as_ladder_pair()))
    for pairs in arms.values():
        pairs.sort()
    return {"arms": arms}


def _read_lines(path: str | None) -> list[str]:
    if path is None or path == "-":
        return sys.stdin.readlines()
    with open(path) as f:
        return f.readlines()


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="extract one run's ladder point from a metrics file or stdin")
    p_run.add_argument("--arm", required=True)
    p_run.add_argument("--metrics", default=None, help="tracker_metrics.jsonl or iris-log file; default stdin")

    p_lad = sub.add_parser("ladder", help="assemble many run points (arm=path pairs) into a ladder JSON")
    p_lad.add_argument("--point", action="append", required=True, metavar="ARM=PATH", help="repeatable")
    p_lad.add_argument("--out", required=True)

    args = ap.parse_args()

    if args.cmd == "run":
        point = collect_run(_read_lines(args.metrics), args.arm)
        print(json.dumps(asdict(point), indent=2))
        return

    points = []
    for spec in args.point:
        arm, path = spec.split("=", 1)
        points.append(collect_run(_read_lines(path), arm))
    ladder = build_ladder(points)
    with open(args.out, "w") as f:
        json.dump(ladder, f, indent=2)
    print(f"wrote {args.out}: " + ", ".join(f"{a} ({len(p)} pts)" for a, p in ladder["arms"].items()))


if __name__ == "__main__":
    main()
