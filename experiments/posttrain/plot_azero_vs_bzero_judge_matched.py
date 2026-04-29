#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bloom SpecEval v2 GPT-4.1 LM-as-judge: A=0 vs B=0 LoRA init at matched LR.

Produces a two-panel figure comparing the rescue A=0 init against the standard
B=0 init on the Bloom-compatible GPT-4.1 judge metric. All values are
prompt-collapsed legacy (parsefail-as-5) means with 95% CIs from the stored
summaries (B=0) and the corrected (Bloom-exact) Batch API runs (A=0).

Inputs are inlined as constants because they come from a small fixed set of
already-archived summary JSON files; rerunning the script is cheap and the
data does not change between runs.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "plot" / "output"

# (lr_label, mean, ci95)
B0 = {
    "5e-6": (8.5040, 0.0811),
    "1e-5": (8.5531, 0.0799),
}
A0 = {
    "1e-6": (8.4163, 0.0795),
    "5e-6": (8.5430, 0.0795),
    "8.75e-6": (8.5690, 0.0794),
    "1e-5": (8.5747, 0.0790),
}
FULL_DPO = (8.4443, 0.0798)  # full DPO β=0.1 lr=5e-7 reference baseline


def main() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 1.4]})

    # ---- Panel 1: matched-LR bars ----
    matched = ["5e-6", "1e-5"]
    b0_m = np.array([B0[lr][0] for lr in matched])
    b0_e = np.array([B0[lr][1] for lr in matched])
    a0_m = np.array([A0[lr][0] for lr in matched])
    a0_e = np.array([A0[lr][1] for lr in matched])
    x = np.arange(len(matched))
    w = 0.36
    ax1.bar(
        x - w / 2,
        b0_m,
        w,
        yerr=b0_e,
        label="B=0 (standard)",
        color="#1f77b4",
        capsize=4,
        edgecolor="black",
        linewidth=0.6,
    )
    ax1.bar(
        x + w / 2, a0_m, w, yerr=a0_e, label="A=0 (rescue)", color="#d62728", capsize=4, edgecolor="black", linewidth=0.6
    )
    ax1.axhline(FULL_DPO[0], ls="--", color="gray", alpha=0.7, label=f"Full DPO β=0.1 ({FULL_DPO[0]:.3f})")
    for i, (bv, av) in enumerate(zip(b0_m, a0_m, strict=True)):
        delta = av - bv
        sign = "+" if delta >= 0 else ""
        ax1.annotate(
            f"Δ={sign}{delta:.3f}",
            xy=(x[i], max(bv, av) + max(b0_e[i], a0_e[i]) + 0.015),
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels(matched)
    ax1.set_xlabel("learning rate (matched)")
    ax1.set_ylabel("GPT-4.1 LM-as-judge prompt-collapsed mean")
    ax1.set_title("Same-LR A=0 vs B=0 contrast")
    ax1.set_ylim(8.30, 8.78)
    ax1.legend(loc="lower right", fontsize=8, framealpha=0.95)
    ax1.grid(axis="y", alpha=0.25)

    # ---- Panel 2: full A=0 LR sweep + matched B=0 overlay ----
    a0_lrs = ["1e-6", "5e-6", "8.75e-6", "1e-5"]
    a0_vals = np.array([A0[lr][0] for lr in a0_lrs])
    a0_errs = np.array([A0[lr][1] for lr in a0_lrs])
    xa = np.arange(len(a0_lrs))
    ax2.errorbar(xa, a0_vals, yerr=a0_errs, fmt="s-", color="#d62728", label="A=0 (rescue)", capsize=4, ms=9, lw=2)
    b0_idx = [i for i, lr in enumerate(a0_lrs) if lr in B0]
    b0_y = np.array([B0[a0_lrs[i]][0] for i in b0_idx])
    b0_e_overlay = np.array([B0[a0_lrs[i]][1] for i in b0_idx])
    ax2.errorbar(
        b0_idx,
        b0_y,
        yerr=b0_e_overlay,
        fmt="o",
        color="#1f77b4",
        label="B=0 (standard, matched LR)",
        capsize=4,
        ms=9,
        lw=0,
    )
    ax2.axhline(FULL_DPO[0], ls="--", color="gray", alpha=0.7, label=f"Full DPO β=0.1 ({FULL_DPO[0]:.3f})")
    ax2.set_xticks(xa)
    ax2.set_xticklabels(a0_lrs)
    ax2.set_xlabel("learning rate")
    ax2.set_ylabel("GPT-4.1 LM-as-judge prompt-collapsed mean")
    ax2.set_title("A=0 LR sweep with B=0 overlay where available")
    ax2.set_ylim(8.30, 8.78)
    ax2.legend(loc="lower right", fontsize=8, framealpha=0.95)
    ax2.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "Bloom SpecEval v2 GPT-4.1 LM-as-judge: A=0 vs B=0 LoRA init "
        "(marin-8b-instruct, β=0.1, b=64, r=64, step-1699, seed=0)",
        y=1.02,
        fontsize=11,
    )
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "azero_vs_bzero_judge_matched.png", dpi=160, bbox_inches="tight")
    fig.savefig(OUT_DIR / "azero_vs_bzero_judge_matched.pdf", bbox_inches="tight")
    print(f"wrote {OUT_DIR}/azero_vs_bzero_judge_matched.png")


if __name__ == "__main__":
    main()
