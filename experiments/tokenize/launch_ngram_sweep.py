# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sweep hashed n-gram INPUT-embedding hyperparameters on a fixed base tokenizer.

Isolates the n-gram lever (Over-Encoding / LongCat, arXiv 2501.16975 & 2601.21204) from the
tokenizer: the base arm and proxy shape are held fixed, only the n-gram config varies. Each cell
runs :mod:`experiments.grug.moe.launch_tokenizer_bakeoff` with ``BAKEOFF_NGRAM=1`` and a named
config of (hash buckets, orders, hashes/order, low-rank width, combine, base-matched init ratio).

The screening sweep walks hash buckets from the collision-noise regime (65k) up to the paper's
~30x-base scale (≈4M), which is the axis that most plausibly explains an absent gain: too-few
buckets turn the n-gram vocabulary into hash noise. Config variants (orders, ratio) branch off the
paper-scale bucket count.

Building is opt-in: without ``--run`` it prints the exact iris commands and submits nothing.

    # print the plan (marin base, screen at 3500 steps)
    uv run python -m experiments.tokenize.launch_ngram_sweep --base marin-128k --steps 3500
    # submit
    uv run python -m experiments.tokenize.launch_ngram_sweep --base marin-128k --steps 3500 --run

Collect with ``collect_metrics``/``collect_ladder`` (job-name prefix ``grug-ngram-``) and compare
BPB against the same base arm's no-n-gram point from the tokenizer ladder.
"""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass

from experiments.tokenize.bakeoff_tokenizers import arm_by_name
from experiments.tokenize.launch_bakeoff_ladder import PROXY_SHAPE


@dataclass(frozen=True)
class NgramConfig:
    """One n-gram hyperparameter point. Defaults are the paper-config screen center."""

    label: str
    buckets: int
    orders: str = "2,3,4"
    num_hashes: int = 2
    rank: int = 128
    combine: str = "mean"
    ratio: float = 1.0

    def env(self) -> dict[str, str]:
        return {
            "BAKEOFF_NGRAM": "1",
            "BAKEOFF_NGRAM_BUCKETS": str(self.buckets),
            "BAKEOFF_NGRAM_ORDERS": self.orders,
            "BAKEOFF_NGRAM_HASHES": str(self.num_hashes),
            "BAKEOFF_NGRAM_RANK": str(self.rank),
            "BAKEOFF_NGRAM_COMBINE": self.combine,
            "BAKEOFF_NGRAM_RATIO": str(self.ratio),
        }


# Bucket counts are primes chosen to sit well off integer multiples of the 128,256 base vocab —
# the paper reports collision spikes when the n-gram vocab lands near a multiple of the base vocab.
_B_NOISE = 65_537
_B_MID = 786_433
_B_3M = 3_145_739
_B_PAPER = 4_000_037

# Screening sweep: the bucket ladder (isolating collisions) plus orders/ratio variants at the
# paper-scale bucket count.
SCREEN_SWEEP: tuple[NgramConfig, ...] = (
    NgramConfig("b65k", _B_NOISE),
    NgramConfig("b786k", _B_MID),
    NgramConfig("b3M", _B_3M),
    NgramConfig("b4M", _B_PAPER),
    NgramConfig("b4M-o345", _B_PAPER, orders="3,4,5"),
    NgramConfig("b4M-r0p5", _B_PAPER, ratio=0.5),
    NgramConfig("b4M-r2", _B_PAPER, ratio=2.0),
)


def build_command(base_arm: str, steps: int, cluster: str, config: NgramConfig) -> list[str]:
    """The iris command that launches one (base arm, n-gram config, step budget) cell."""
    arm = arm_by_name(base_arm)  # validates the base tokenizer exists
    run_id = f"ngram-{arm.name}-{config.label}-s{steps}"
    env = {
        **PROXY_SHAPE,
        **config.env(),
        "BAKEOFF_ARM": arm.name,
        "SCALE_STEPS": str(steps),
        "RUN_ID": run_id,
    }
    cmd = [
        "iris",
        f"--cluster={cluster}",
        "job",
        "run",
        "--no-wait",
        "--cpu",
        "2",
        "--memory",
        "3GB",
        "--extra",
        "cpu",
        "--job-name",
        f"grug-ngram-{arm.name}-{config.label}-s{steps}",
    ]
    for key, value in env.items():
        cmd += ["-e", key, value]
    cmd += ["--", "python", "-m", "experiments.grug.moe.launch_tokenizer_bakeoff"]
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="marin-128k", help="base tokenizer arm the n-gram sits on")
    ap.add_argument("--steps", type=int, default=3500, help="training steps (isoFLOP point)")
    ap.add_argument("--cluster", default="cw-rno2a")
    ap.add_argument("--run", action="store_true", help="submit the jobs (default: print the plan only)")
    args = ap.parse_args()

    print(f"n-gram sweep on {args.base}: {len(SCREEN_SWEEP)} configs @ {args.steps} steps on {args.cluster}")
    for config in SCREEN_SWEEP:
        cmd = build_command(args.base, args.steps, args.cluster, config)
        print(f"\n# {config.label}: buckets={config.buckets} orders={config.orders} ratio={config.ratio}")
        print(" ".join(cmd))
        if args.run:
            subprocess.run(cmd, check=True)

    if not args.run:
        print(f"\n(plan only — rerun with --run to submit all {len(SCREEN_SWEEP)} jobs)")


if __name__ == "__main__":
    main()
