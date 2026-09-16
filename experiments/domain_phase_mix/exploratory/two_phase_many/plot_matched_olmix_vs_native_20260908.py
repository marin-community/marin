# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Bucket weights of the matched (Qwen-swarm) Olmix policies against the native (Llama-fitted) trained policies.

Reads `delphi_matched_olmix_3e18_20260908/solutions.csv` and draws one panel per candidate: paired horizontal bars of
runtime weight for the native trained mixture and the matched policy, buckets ordered by pool size, with materialized
epochs printed at the bar ends.

usage: uv run --offline --no-sync python plot_matched_olmix_vs_native_20260908.py [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.dolma3_dolmino_top_level_domains import TOP_LEVEL_DOMAIN_TOKEN_COUNTS  # noqa: E402

SOURCE = SCRIPT_DIR / "reference_outputs" / "delphi_matched_olmix_3e18_20260908"
TARGET_BUDGET = 6_325_183_647_689
PANELS = (
    ("olmixq_u_kl0p05_cap04", "Uncheatable, KL 0.05 (ladder coefficient)"),
    ("olmixq_u_kl0p1_cap04", "Uncheatable, KL 0.1 (sweep winner)"),
    ("olmixq_u_kl0_cap04", "Uncheatable, KL 0"),
    ("olmixq_t9_kl0p005_cap04", "OlmoBaseEval Easy, KL 0.005 (sweep winner)"),
    ("olmixq_t9_kl0_cap04", "OlmoBaseEval Easy, KL 0"),
)
NATIVE_COLOR = "#9e9e9e"
MATCHED_COLOR = "#1b7f79"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=SOURCE)
    args = parser.parse_args()
    solutions = pd.read_csv(SOURCE / "solutions.csv")
    buckets = sorted(set(solutions.bucket), key=lambda b: -TOP_LEVEL_DOMAIN_TOKEN_COUNTS[b])
    tokens = np.asarray([TOP_LEVEL_DOMAIN_TOKEN_COUNTS[b] for b in buckets], float)
    y = np.arange(len(buckets))
    fig, axes = plt.subplots(1, len(PANELS), figsize=(4.2 * len(PANELS), 11), sharey=True)
    for ax, (candidate_id, title) in zip(axes, PANELS, strict=True):
        frame = solutions[solutions.candidate_id == candidate_id].set_index("bucket").loc[buckets]
        native = frame["native"].to_numpy(float)
        matched = frame["runtime"].to_numpy(float)
        ax.barh(y + 0.2, native, height=0.4, color=NATIVE_COLOR, label="native (trained, Llama-fitted)")
        ax.barh(y - 0.2, matched, height=0.4, color=MATCHED_COLOR, label="matched (Qwen-swarm fit)")
        for yi, (wn, wm) in enumerate(zip(native, matched, strict=True)):
            en = TARGET_BUDGET * wn / tokens[yi]
            em = TARGET_BUDGET * wm / tokens[yi]
            ax.text(max(wn, 0) + 0.002, yi + 0.2, f"{en:.1f}", va="center", fontsize=5.5, color="#555555")
            ax.text(max(wm, 0) + 0.002, yi - 0.2, f"{em:.1f}", va="center", fontsize=5.5, color=MATCHED_COLOR)
        tv = float(np.abs(native - matched).sum() / 2)
        ax.set_title(f"{title}\nTV native vs matched {tv:.2f}", fontsize=9)
        ax.set_xlabel("mixture weight")
        ax.grid(axis="x", alpha=0.3)
        ax.set_xlim(0, max(native.max(), matched.max()) * 1.25)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(buckets, fontsize=7)
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(
        "Olmix policies at Qwen3 360M/1.6B: native trained mixtures vs matched refits on the 3e18 swarm "
        "(cap 4; labels are materialized epochs; buckets ordered by pool size)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / f"matched_vs_native_weights.{suffix}", dpi=170)
    print(args.output_dir / "matched_vs_native_weights.png")


if __name__ == "__main__":
    main()
