# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot per-eval BPB for the 67B-A2B 65k and 262k checkpoints against Qwen3.5-35B-A3B-Base.

Marin numbers come from the W&B runs written by ``eval_67b_a2b_step156k_paloma_lm_eval_rolling.py``.
Qwen numbers come from the lm-eval results published with issue #8824. Both sides use lm-eval 0.4.12
``loglikelihood_rolling`` semantics on the same documents (Paloma test split, 7 legacy UncheatableEval subsets).

Run::

    uv run --with matplotlib python experiments/grug/moe/plot_67b_a2b_vs_qwen_bpb.py out.png
"""

import json
import statistics
import sys
import urllib.request

import matplotlib as mpl
import matplotlib.pyplot as plt
import wandb
from matplotlib.patches import Patch

mpl.use("Agg")

WANDB_PROJECT = "marin-community/marin_moe"
QWEN_RESULTS_URL = "https://storage.googleapis.com/marin-public/held/hero-open-weight-ppl/2026.09.05.1/qwen-results.json"
CHECKPOINTS = (
    ("Marin 67B-A2B 65k ckpt (step 156k)", "step156k"),
    ("Marin 67B-A2B 262k ckpt (qk 1.75, skew8, step 157k)", "ctxext_qk175_skew8_step157k"),
)
QWEN_NAME = "Qwen3.5-35B-A3B-Base"
# (panel label, our subset key, lm-eval task suffix)
PALOMA = (
    ("4chan", "4chan", "4chan_meta_sep"),
    ("C4 100 domains", "c4_100_domains", "c4_100_domains"),
    ("C4 English", "c4_en", "c4_en"),
    ("Dolma v1.5", "dolma-v1_5", "dolma-v1_5"),
    ("100 prog. languages", "dolma_100_programing_languages", "dolma_100_programing_languages"),
    ("100 subreddits", "dolma_100_subreddits", "dolma_100_subreddits"),
    ("Falcon RefinedWeb", "falcon-refinedweb", "falcon-refinedweb"),
    ("Gab", "gab", "gab"),
    ("M2D2 S2ORC", "m2d2_s2orc_unsplit", "m2d2_s2orc_unsplit"),
    ("M2D2 Wikipedia", "m2d2_wikipedia_unsplit", "m2d2_wikipedia_unsplit"),
    ("Manosphere", "manosphere_meta_sep", "manosphere_meta_sep"),
    ("mC4", "mc4", "mc4"),
    ("Penn Treebank", "ptb", "ptb"),
    ("RedPajama", "redpajama", "redpajama"),
    ("Twitter AAE", "twitterAAE_HELM_fixed", "twitterAAE_HELM_fixed"),
    ("WikiText-103", "wikitext_103", "wikitext_103"),
)
UNCHEATABLE = (
    ("Wikipedia English", "wikipedia_english"),
    ("GitHub Python", "github_python"),
    ("GitHub C++", "github_cpp"),
    ("BBC News", "bbc_news"),
    ("arXiv Physics", "arxiv_physics"),
    ("arXiv CS", "arxiv_computer_science"),
    ("AO3 English", "ao3_english"),
)
COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
INK, MUTED, GRID, AXIS, MACRO_BG = "#0b0b0b", "#52514e", "#e6e5e0", "#b8b7b0", "#f3f6fb"


def _marin_summaries() -> dict[str, dict[str, float]]:
    api = wandb.Api(timeout=60)
    summaries = {}
    for _, tag in CHECKPOINTS:
        for suite in ("paloma", "uncheatable_eval"):
            run = api.run(f"{WANDB_PROJECT}/moe_67b_a2b_d2560_{tag}_{suite}_lm_eval_rolling4096")
            summaries[f"{tag}/{suite}"] = dict(run.summary)
    return summaries


def _qwen_bpb() -> dict[str, float]:
    with urllib.request.urlopen(QWEN_RESULTS_URL) as f:
        results = json.load(f)["results"]
    return {task: metrics["bits_per_byte,none"] for task, metrics in results.items()}


def panels(marin: dict[str, dict[str, float]], qwen: dict[str, float]) -> list[tuple[str, list[float]]]:
    """One (title, [ckpt values..., qwen]) entry per eval, plus a macro entry per suite."""

    def ours(suite: str, key: str) -> list[float]:
        return [marin[f"{tag}/{suite}"][f"lm_eval_rolling/{suite}/{key}"] for _, tag in CHECKPOINTS]

    out = [(f"Paloma · {label}", [*ours("paloma", f"{key}/bpb"), qwen[f"paloma_{task}"]]) for label, key, task in PALOMA]
    out.append(
        (
            "Paloma · MACRO (16)",
            [*ours("paloma", "macro_bpb"), statistics.mean(qwen[f"paloma_{task}"] for _, _, task in PALOMA)],
        )
    )
    out += [
        (f"Uncheatable · {label}", [*ours("uncheatable_eval", f"{key}/bpb"), qwen[f"uncheatable_eval_hero_{key}"]])
        for label, key in UNCHEATABLE
    ]
    out.append(
        (
            "Uncheatable · MACRO (7)",
            [
                *ours("uncheatable_eval", "macro_bpb"),
                statistics.mean(qwen[f"uncheatable_eval_hero_{key}"] for _, key in UNCHEATABLE),
            ],
        )
    )
    return out


def plot(rows: list[tuple[str, list[float]]], path: str) -> None:
    fig, axes = plt.subplots(5, 5, figsize=(17, 15.5))
    for ax, (title, vals) in zip(axes.flat, rows, strict=True):
        is_macro = "MACRO" in title
        top = max(vals)
        ax.bar(range(len(vals)), vals, color=COLORS, width=0.72, edgecolor="white", linewidth=2)
        ax.set_ylim(0, top * 1.22)
        for x, v in enumerate(vals):
            ax.text(x, v + top * 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9, color=INK)
        delta = (vals[0] / vals[-1] - 1) * 100
        ax.text(
            0.98,
            0.97,
            f"65k vs Qwen {delta:+.1f}%",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            color=MUTED,
        )
        ax.set_title(title, fontsize=10.5, color=INK, fontweight="bold" if is_macro else "normal", loc="left")
        ax.set_xticks([])
        ax.tick_params(axis="y", labelsize=8, colors=MUTED, length=0)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        if is_macro:
            ax.set_facecolor(MACRO_BG)
    names = [name for name, _ in CHECKPOINTS] + [QWEN_NAME]
    fig.legend(
        [Patch(color=c) for c in COLORS],
        names,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=11.5,
        bbox_to_anchor=(0.5, 0.985),
    )
    fig.suptitle(
        "Bits per byte (lower is better), lm-eval 0.4.12 loglikelihood_rolling, max_length=4096; "
        "Paloma test split (154,440 docs), UncheatableEval 7 subsets (7,052 docs)",
        fontsize=11,
        color=MUTED,
        y=0.955,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=110, facecolor="white")


if __name__ == "__main__":
    plot(panels(_marin_summaries(), _qwen_bpb()), sys.argv[1])
