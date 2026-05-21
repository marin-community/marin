# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Throughput anchors for Delphi training runs — pure reference data.

**Read me before adding code that imports this module.** This file is purely
advisory: it stores measured step-times from past or in-flight W&B runs so
that future agents and humans can answer "how long will this run take on
that TPU?" without re-deriving it from logs every time. **Nothing in the
training stack imports from this module**, and nothing here writes to GCS,
W&B, or any shared service. It must stay that way.

If you find yourself wanting to import these anchors into ``launch.py`` or
``preflight.py``, stop and route the requirement through an explicit code
review — that would couple training behavior to a registry that was never
designed to be load-bearing.

## How to use

Inspect what we have::

    uv run python experiments/throughput_stats.py --list

Estimate a future run::

    uv run python experiments/throughput_stats.py \\
        --model 3e18 --tpu v5p-8 --steps 7400 \\
        --train-batch-size 8 --seq-len 4096

In a notebook / sibling script::

    from experiments.throughput_stats import estimate_wall_time_s
    est = estimate_wall_time_s(
        model_flops_key="3e18",
        tpu_type="v5p-8",
        num_train_steps=7400,
        train_batch_size=8,
        seq_len=4096,
    )
    print(f"~{est.wall_time_s / 3600:.2f} h  (via {est.anchor.wandb_run_path})")

## How to add an anchor

Append a :class:`ThroughputAnchor` to ``ANCHORS`` below.

Rules:

1. ``step_time_s`` is the median wall-clock per **training** step over a
   stable window — i.e. after JIT warmup and dataloader prefetch fill, and
   excluding eval interleaving where possible. Prefer W&B
   ``throughput/duration`` when present; older runs may require reading
   ``train/time_per_step`` or computing (elapsed - sum_of_eval_times) /
   num_train_steps_so_far. Eval wall-clock is intentionally NOT included;
   estimators that want it should add it separately based on the configured
   eval cadence.
2. ``wandb_run_path`` is the canonical ``entity/project/run_id`` triple. It
   must be a real, retrievable run so a future reader can re-verify the
   number without trusting our prose.
3. ``measured_at`` is the date the throughput was read off W&B, ISO-8601.
   Anchors don't expire automatically, but if Levanter, JAX, or libtpu have
   moved materially you should refresh.
4. ``notes`` is the place to record caveats — measurement window,
   preemption count during the source run, batch-axis sharding choices, any
   reason the number might not generalize.

Anchors live in code (not GCS/yaml) so they are reviewable in PRs and travel
with the codebase.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class ThroughputAnchor:
    """One measured (model, hardware, config) → step-time data point.

    Anchors are points, not curves. Linear extrapolation across step count
    is fine; across batch_size / seq_len it is approximate; across TPU
    family / topology it is unreliable and we refuse to do it.
    """

    model_flops_key: str
    """Canonical Delphi base key or historical experiment key.

    Modern keys match :mod:`experiments.delphi_models` (e.g. ``"3e18"``,
    ``"1e21"``). Legacy executor runs used keys like ``"1e21-v5"`` and
    ``"1e22-v5"``; keep those exact keys when the source W&B run used them.
    """

    tpu_type: str
    """Iris TPU type string (``"v5p-8"``, ``"v6e-8"``, ...). Matches the
    string passed to ``iris job run --tpu`` and to ``ComputeProfile.tpu_type``."""

    train_batch_size: int
    """Global train batch size (sequences per step)."""

    seq_len: int
    """Per-sequence token length."""

    per_device_parallelism: int
    """``trainer.per_device_parallelism`` value. ``-1`` means levanter
    auto-resolved it from ``train_batch_size`` and the mesh."""

    step_time_s: float
    """Median wall-clock per training step over a stable window. Excludes
    eval and JIT warmup."""

    wandb_run_path: str
    """Canonical ``entity/project/run_id`` of the source run. Required so
    the number is re-verifiable."""

    measured_at: date
    """When this anchor was recorded (not when the run was started)."""

    notes: str = ""
    """Free-form caveats. Mention sample window, preemption count, anything
    a future reader should know before trusting the number."""

    @property
    def tokens_per_step(self) -> int:
        return self.train_batch_size * self.seq_len

    @property
    def tokens_per_s(self) -> float:
        return self.tokens_per_step / self.step_time_s


@dataclass(frozen=True)
class WallTimeEstimate:
    """Result of an estimation, with provenance and the scaling rule used."""

    wall_time_s: float
    anchor: ThroughputAnchor
    scaling_rule: str
    """Human-readable description of how the anchor was applied, e.g.
    ``"exact (model, tpu, batch, seq)"`` or
    ``"same (model, tpu); tokens-per-step scaled by 0.50"``."""


class NoAnchorError(LookupError):
    """No anchor matches the requested (model, tpu) pair."""


# ---------------------------------------------------------------------------
# Anchors
# ---------------------------------------------------------------------------

ANCHORS: tuple[ThroughputAnchor, ...] = (
    # Hardware naming note: Iris/GCP v5p slice names are twice W&B's reported
    # accelerator count for these runs. For example, Iris ``v5p-512`` appears
    # in W&B throughput summaries as 256 accelerators. Store the Iris launch
    # string in ``tpu_type`` because that is what operators request.
    ThroughputAnchor(
        model_flops_key="3e18",
        tpu_type="v5p-8",
        train_batch_size=8,
        seq_len=4096,
        per_device_parallelism=-1,
        step_time_s=0.151,
        wandb_run_path="marin-community/delphi-midtraining/delphi-3e18-p33m67-k0p20-lr33-a003",
        measured_at=date(2026, 5, 16),
        notes=(
            "Mid-flight K=0.20 CPT run. Median from W&B "
            "``throughput/duration`` over the stable tail; W&B reports 4 "
            "accelerators for this Iris v5p-8 slice. Replaces the stale a001 "
            "anchor, whose W&B run was not retrievable."
        ),
    ),
    ThroughputAnchor(
        model_flops_key="1e20",
        tpu_type="v5p-32",
        train_batch_size=128,
        seq_len=4096,
        per_device_parallelism=-1,
        step_time_s=1.897,
        wandb_run_path="marin-community/delphi-midtraining/true-midtrain-1e20-p33m67-step40000",
        measured_at=date(2026, 5, 16),
        notes=(
            "Canonical true-midtrain 1e20 anchor, not the contaminated "
            "``1e20-iso-d2048-L21`` legacy base. Median from W&B "
            "``throughput/duration``; W&B reports 16 accelerators for this "
            "Iris v5p-32 slice."
        ),
    ),
    ThroughputAnchor(
        model_flops_key="1e21",
        tpu_type="v5p-64",
        train_batch_size=512,
        seq_len=4096,
        per_device_parallelism=-1,
        step_time_s=6.847,
        wandb_run_path="marin-community/delphi-midtraining/true-midtrain-1e21-p33m67-step20000-v5p64",
        measured_at=date(2026, 5, 16),
        notes=(
            "Canonical true-midtrain 1e21 anchor. Median from W&B "
            "``throughput/duration``; W&B reports 32 accelerators for this "
            "Iris v5p-64 slice."
        ),
    ),
    ThroughputAnchor(
        model_flops_key="1e21-v5",
        tpu_type="v5p-64",
        train_batch_size=512,
        seq_len=4096,
        per_device_parallelism=-1,
        step_time_s=7.283,
        wandb_run_path="marin-community/delphi-midtraining/delphi-1e21-p33m67-9p25b-lr0.5-efbc63",
        measured_at=date(2026, 5, 16),
        notes=(
            "Legacy executor K=0.20 anchor for base tag ``1e21-v5``. Median "
            "from W&B ``throughput/duration``; W&B reports 32 accelerators "
            "for this Iris v5p-64 slice."
        ),
    ),
    ThroughputAnchor(
        model_flops_key="1e21-v5",
        tpu_type="v5p-256",
        train_batch_size=512,
        seq_len=4096,
        per_device_parallelism=-1,
        step_time_s=1.532,
        wandb_run_path="marin-community/delphi-midtraining/delphi-1e21-v5-math-10b-lr0.5-v5-v5p256-3f4490",
        measured_at=date(2026, 5, 16),
        notes=(
            "Legacy 10B math-midtraining anchor for base tag ``1e21-v5``. "
            "Hardware comes from the run name; W&B reports 128 accelerators "
            "for this Iris v5p-256 slice. Median from W&B "
            "``throughput/duration``."
        ),
    ),
    ThroughputAnchor(
        model_flops_key="1e22-v5",
        tpu_type="v5p-64",
        train_batch_size=1024,
        seq_len=4096,
        per_device_parallelism=4,
        step_time_s=34.42,
        wandb_run_path="marin-community/delphi-midtraining/delphi-1e22-p50m50-32p07b-lr0.5-ecfa99",
        measured_at=date(2026, 5, 16),
        notes=(
            "Legacy K=0.20 1e22 anchor on v5p-64. Finished after resume work; "
            "history may mix attempts, so prefer v5p-256/v5p-512 anchors when "
            "planning those hardware classes. Median from stable W&B "
            "``throughput/duration`` tail; W&B reports 32 accelerators for "
            "this Iris v5p-64 slice."
        ),
    ),
    ThroughputAnchor(
        model_flops_key="1e22-v5",
        tpu_type="v5p-256",
        train_batch_size=1024,
        seq_len=4096,
        per_device_parallelism=4,
        step_time_s=7.899,
        wandb_run_path="marin-community/delphi-midtraining/delphi-1e22-p67m33-32p07b-lr0.5-f60cb12a",
        measured_at=date(2026, 5, 16),
        notes=(
            "Legacy K=0.20 1e22 anchor. Median from W&B "
            "``throughput/duration``; W&B reports 128 accelerators for this "
            "Iris v5p-256 slice."
        ),
    ),
    ThroughputAnchor(
        model_flops_key="1e22-v5",
        tpu_type="v5p-512",
        train_batch_size=1024,
        seq_len=4096,
        per_device_parallelism=4,
        step_time_s=4.080,
        wandb_run_path="marin-community/delphi-midtraining/delphi-1e22-v5-math-10b-lr0.5-v5-v5p512-B1024-3b631c",
        measured_at=date(2026, 5, 16),
        notes=(
            "Legacy 10B math-midtraining anchor. Run name and logbook identify "
            "Iris v5p-512 with batch 1024; W&B reports 256 accelerators for "
            "this slice. Median from W&B ``throughput/duration``."
        ),
    ),
    ThroughputAnchor(
        model_flops_key="2e19",
        tpu_type="v5p-8",
        train_batch_size=16,
        seq_len=4096,
        per_device_parallelism=-1,
        step_time_s=0.511,
        wandb_run_path="marin-community/delphi-midtraining/delphi-2e19-p67m33-k0p20-lr50-probe-v5p8-20s-a005",
        measured_at=date(2026, 5, 16),
        notes=(
            "20-step p67m33/lr50 throughput probe. Median from W&B "
            "``throughput/duration`` for global_step >= 5 (14 samples); "
            "Iris run was later preempted, but the stable train-step window "
            "was logged before preemption."
        ),
    ),
    ThroughputAnchor(
        model_flops_key="2e19",
        tpu_type="v6e-8",
        train_batch_size=16,
        seq_len=4096,
        per_device_parallelism=-1,
        step_time_s=0.345,
        wandb_run_path="marin-community/delphi-midtraining/delphi-2e19-p67m33-k0p20-lr50-probe-v6e8-20s-a005",
        measured_at=date(2026, 5, 16),
        notes=(
            "20-step p67m33/lr50 throughput probe. Job succeeded after one "
            "preemption. Median from W&B ``throughput/duration`` for "
            "global_step >= 5 (15 samples)."
        ),
    ),
    ThroughputAnchor(
        model_flops_key="2e19",
        tpu_type="v6e-4",
        train_batch_size=16,
        seq_len=4096,
        per_device_parallelism=-1,
        step_time_s=0.573,
        wandb_run_path="marin-community/delphi-midtraining/delphi-2e19-p67m33-k0p20-lr50-probe-v6e4-20s-a005",
        measured_at=date(2026, 5, 16),
        notes=(
            "20-step p67m33/lr50 throughput probe. Median from W&B "
            "``throughput/duration`` for global_step >= 5 (14 samples). "
            "Run had one preemption before this stable training window."
        ),
    ),
    ThroughputAnchor(
        model_flops_key="3e19",
        tpu_type="v6e-4",
        train_batch_size=32,
        seq_len=4096,
        per_device_parallelism=-1,
        step_time_s=1.300,
        wandb_run_path="marin-community/delphi-midtraining/delphi-3e19-p67m33-k0p20-lr50-probe-v6e4-20s-a005",
        measured_at=date(2026, 5, 16),
        notes=(
            "20-step p67m33/lr50 throughput probe. Job succeeded. Median "
            "from W&B ``throughput/duration`` for global_step >= 5 "
            "(15 samples)."
        ),
    ),
    ThroughputAnchor(
        model_flops_key="3e19",
        tpu_type="v6e-8",
        train_batch_size=32,
        seq_len=4096,
        per_device_parallelism=-1,
        step_time_s=0.770,
        wandb_run_path="marin-community/delphi-midtraining/delphi-3e19-p67m33-k0p20-lr50-probe-v6e8-20s-a005",
        measured_at=date(2026, 5, 16),
        notes=(
            "20-step p67m33/lr50 throughput probe. Median from W&B "
            "``throughput/duration`` for global_step >= 5 (14 samples). "
            "Iris job was still marked running after the train window, so "
            "treat this as a measured train-throughput anchor, not an "
            "end-to-end completion anchor."
        ),
    ),
    ThroughputAnchor(
        model_flops_key="9e19",
        tpu_type="v6e-8",
        train_batch_size=64,
        seq_len=4096,
        per_device_parallelism=-1,
        step_time_s=1.988,
        wandb_run_path="marin-community/delphi-midtraining/delphi-9e19-p67m33-k0p20-lr50-probe-v6e8-20s-a005",
        measured_at=date(2026, 5, 16),
        notes=(
            "20-step p67m33/lr50 throughput probe. Median from W&B "
            "``throughput/duration`` for global_step >= 5 (14 samples). "
            "Training reached the step-19 checkpoint; the Iris job failed "
            "afterward during HF export with container-RAM exit 137, not "
            "TPU HBM."
        ),
    ),
    ThroughputAnchor(
        model_flops_key="2e20",
        tpu_type="v5p-16",
        train_batch_size=64,
        seq_len=4096,
        per_device_parallelism=-1,
        step_time_s=2.079,
        wandb_run_path="marin-community/delphi-midtraining/delphi-2e20-p67m33-k0p20-lr50-probe-v5p16-30s-a005",
        measured_at=date(2026, 5, 19),
        notes=(
            "30-step p67m33/lr50 throughput probe. Median from W&B "
            "``throughput/duration`` over stable global_step 2-9 "
            "(8 samples): about 126k tok/s and 41.8% MFU. Iris killed the "
            "job after 5 preemptions before it completed all 30 requested "
            "steps, so treat this as a provisional train-throughput-only "
            "anchor. No HBM utilization metric was present in W&B history "
            "or Iris logs; the run compiled and completed steady train "
            "steps with no HBM/RESOURCE_EXHAUSTED errors."
        ),
    ),
    ThroughputAnchor(
        model_flops_key="2e20",
        tpu_type="v5p-8",
        train_batch_size=64,
        seq_len=4096,
        per_device_parallelism=-1,
        step_time_s=3.471,
        wandb_run_path="marin-community/delphi-midtraining/delphi-2e20-p67m33-k0p20-lr50-probe-v5p8-30s-a001",
        measured_at=date(2026, 5, 20),
        notes=(
            "30-step p67m33/lr50 throughput probe; run finished cleanly "
            "(state=finished, last step 29) on Iris v5p-8 us-east5-a with "
            "no preemptions. Median from W&B ``throughput/duration`` for "
            "global_step >= 5 (25 samples); min/max 3.466/3.476 s, so the "
            "measurement is very stable. Tokens/step = 262,144; about "
            "75.5k tok/s. No HBM/RESOURCE_EXHAUSTED errors. v5p-8 doubles "
            "per-device load relative to the v5p-16 probe (8 vs 4 "
            "seq/chip) and is ~1.67x slower per step but burns ~16% fewer "
            "chip-hours for a full K=0.20 cell (87 vs 104 chip-h)."
        ),
    ),
    # TODO: Add pretrain anchors for 9e18, 2e19, 3e19, 9e19, 2e20 from each
    # base's original W&B run (entity ``marin-community``, project varies —
    # look in ``experiments/delphi_models.py`` for URLs). Each should be a
    # (base, TPU) pair with step_time read off ``train/time_per_step`` on a
    # stable window.
    #
    # TODO: Add more v6e probe/full-run anchors. We have only two v6e probe
    # points so far, and only one terminally succeeded, so cross-TPU estimates
    # remain thin.
)


# ---------------------------------------------------------------------------
# Lookup + estimation
# ---------------------------------------------------------------------------


def find_anchors(
    *,
    model_flops_key: str | None = None,
    tpu_type: str | None = None,
) -> list[ThroughputAnchor]:
    """Filter anchors by exact match on the provided fields.

    ``None`` means "any". Ordered newest-measurement first so callers that
    take the first match get the freshest data.
    """
    matches = [
        a
        for a in ANCHORS
        if (model_flops_key is None or a.model_flops_key == model_flops_key)
        and (tpu_type is None or a.tpu_type == tpu_type)
    ]
    return sorted(matches, key=lambda a: a.measured_at, reverse=True)


def estimate_wall_time_s(
    *,
    model_flops_key: str,
    tpu_type: str,
    num_train_steps: int,
    train_batch_size: int,
    seq_len: int,
) -> WallTimeEstimate:
    """Estimate train-only wall time for a planned run.

    The strategy is intentionally conservative:

    - Require an anchor for the same ``(model_flops_key, tpu_type)``. If
      none exists, raise :class:`NoAnchorError` — we will not extrapolate
      across TPU families (e.g. v5p → v6e), because per-chip throughput
      varies more than linearly with chip count when topology changes.
    - If an anchor exists with the same ``train_batch_size`` and
      ``seq_len``, use its ``step_time_s`` directly.
    - Otherwise, scale ``step_time_s`` linearly by the tokens-per-step
      ratio. This is approximate; the scaling rule is reported on the
      returned :class:`WallTimeEstimate` so the caller can decide whether
      to trust it.
    """
    candidates = find_anchors(model_flops_key=model_flops_key, tpu_type=tpu_type)
    if not candidates:
        raise NoAnchorError(
            f"No throughput anchor for ({model_flops_key!r}, {tpu_type!r}). "
            f"Add one to experiments/throughput_stats.py once a run on this "
            f"hardware completes."
        )

    exact = [a for a in candidates if a.train_batch_size == train_batch_size and a.seq_len == seq_len]
    if exact:
        anchor = exact[0]
        return WallTimeEstimate(
            wall_time_s=num_train_steps * anchor.step_time_s,
            anchor=anchor,
            scaling_rule="exact (model, tpu, batch, seq)",
        )

    anchor = candidates[0]
    anchor_tps = anchor.tokens_per_step
    requested_tps = train_batch_size * seq_len
    scale = requested_tps / anchor_tps
    return WallTimeEstimate(
        wall_time_s=num_train_steps * anchor.step_time_s * scale,
        anchor=anchor,
        scaling_rule=f"same (model, tpu); tokens-per-step scaled by {scale:.3f}",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}min"
    if seconds < 86400:
        return f"{seconds / 3600:.2f}h"
    return f"{seconds / 86400:.2f}d"


def _cmd_list() -> int:
    if not ANCHORS:
        print("No throughput anchors recorded. See module docstring for how to add one.")
        return 0
    print(f"{len(ANCHORS)} anchor(s):\n")
    for a in sorted(ANCHORS, key=lambda a: (a.model_flops_key, a.tpu_type)):
        print(
            f"  {a.model_flops_key:10s} {a.tpu_type:8s} "
            f"bs={a.train_batch_size}  seq={a.seq_len}  "
            f"step={a.step_time_s:.3f}s  ({a.tokens_per_s:,.0f} tok/s)"
        )
        print(f"    wandb: {a.wandb_run_path}")
        print(f"    measured_at: {a.measured_at.isoformat()}")
        if a.notes:
            print(f"    notes: {a.notes}")
        print()
    return 0


def _cmd_estimate(args: argparse.Namespace) -> int:
    est = estimate_wall_time_s(
        model_flops_key=args.model,
        tpu_type=args.tpu,
        num_train_steps=args.steps,
        train_batch_size=args.train_batch_size,
        seq_len=args.seq_len,
    )
    print(f"Estimated train-only wall time: {_format_duration(est.wall_time_s)} " f"({est.wall_time_s:.0f}s)")
    print(f"  anchor:         {est.anchor.wandb_run_path}")
    print(f"  anchor measured: {est.anchor.measured_at.isoformat()}")
    print(f"  scaling:        {est.scaling_rule}")
    print(f"  anchor step:    {est.anchor.step_time_s:.3f}s/step " f"({est.anchor.tokens_per_s:,.0f} tok/s)")
    print("  NOTE:           excludes eval overhead and TPU acquisition / " "preemption restart time.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    sub = parser.add_subparsers(dest="cmd", required=False)

    sub.add_parser("list", help="List all anchors.")

    est = sub.add_parser("estimate", help="Estimate wall time for a planned run.")
    est.add_argument("--model", required=True, help="Model flops key (e.g. '3e18').")
    est.add_argument("--tpu", required=True, help="TPU type (e.g. 'v5p-8').")
    est.add_argument("--steps", required=True, type=int, help="num_train_steps.")
    est.add_argument("--train-batch-size", required=True, type=int)
    est.add_argument("--seq-len", required=True, type=int)

    # Convenience: support `--list` and `--estimate` as flags too, mirroring
    # the docstring examples that don't use a subcommand.
    parser.add_argument("--list", action="store_true", help="Shortcut for the 'list' subcommand.")
    parser.add_argument("--model", help=argparse.SUPPRESS)
    parser.add_argument("--tpu", help=argparse.SUPPRESS)
    parser.add_argument("--steps", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--train-batch-size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--seq-len", type=int, help=argparse.SUPPRESS)

    args = parser.parse_args(argv)

    if args.cmd == "list" or args.list:
        return _cmd_list()
    if args.cmd == "estimate":
        return _cmd_estimate(args)
    # Top-level flags as a poor man's subcommand: `--model X --tpu Y --steps N ...`
    if args.model and args.tpu and args.steps and args.train_batch_size and args.seq_len:
        return _cmd_estimate(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
