# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render the SFT midtraining probe report from the analysis outputs (rerun after each collect + analyze)."""

from __future__ import annotations

import datetime as dt
import zoneinfo
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "sft_midtraining_probe_20261014"
ANALYSIS = OUTPUT_DIR / "analysis"
PT = zoneinfo.ZoneInfo("America/Los_Angeles")


def _md(frame: pd.DataFrame, **kwargs) -> str:
    return frame.to_markdown(**kwargs) if not frame.empty else "_(nothing landed yet)_"


def contrasts_table(contrasts: pd.DataFrame, metrics: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for contrast, group in contrasts.groupby("contrast", sort=False):
        row = {"contrast": contrast + (" (primary)" if group["primary"].any() else "")}
        for metric in metrics:
            hit = group[group["metric"] == metric]
            row[metric] = (
                f"{hit['delta_mean'].iloc[0]:+.4f} ± {hit['delta_se'].iloc[0]:.4f} "
                f"(n={int(hit['n_trainer_seeds'].iloc[0])})"
                if not hit.empty
                else ""
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    stamp = dt.datetime.now(PT).strftime("%Y-%m-%d %H:%M PT")
    parts = [
        f"# SFT midtraining probe at Delphi 3e18: results (rendered {stamp})",
        "",
        "Design, submissions and decisions: README.md. Question: do two-phase schedules whose late phase stands in "
        "for midtraining lower post-SFT validation loss (Liu, Neubig and Xiong 2025), although they did not beat "
        "single-phase mixtures on base BPB? Losses are final-step validation losses on the supervised tokens; "
        "forgetting is Uncheatable (fixed weights) and Paloma C4 BPB after SFT, per SFT set. SciQ is descriptive "
        "(pretraining-seed SD 0.0115, LR at the grid edge). B = B1 by content (see README).",
        "",
    ]
    grid = OUTPUT_DIR / "grid_results.csv"
    if grid.exists():
        g = pd.read_csv(grid)
        g = g[g["panel"] == "grid_v5p8"]
        parts += [
            "## LR grid (proportional checkpoint, SFT seed 0)",
            "",
            _md(g.pivot_table(index="learning_rate", columns="dataset", values="sft_val_loss").round(4)),
            "",
            "Chosen: GSM8K 1e-4, Pycode 3e-5, no_robots 3e-5, SciQ 1e-5.",
            "",
        ]
    per_run = pd.read_csv(ANALYSIS / "per_run.csv") if (ANALYSIS / "per_run.csv").exists() else pd.DataFrame()
    if (ANALYSIS / "arms_contrasts.csv").exists():
        arms = per_run[per_run["panel"] == "arms"]
        summary = (
            arms.groupby("arm")[["gsm8k", "pycode", "sciq", "norobots", "base_table9", "base_uncheatable"]]
            .agg(["mean", "std", "count"])
            .round(4)
        )
        summary.columns = [f"{a} {b}" for a, b in summary.columns]
        contrasts = pd.read_csv(ANALYSIS / "arms_contrasts.csv")
        parts += [
            "## Arms (pairs by trainer seed at data seed 662009; two SFT seeds per base)",
            "",
            "Per-arm means (std) over trainer seeds:",
            "",
            _md(summary),
            "",
            "Paired contrasts, mean ± SE over trainer seeds:",
            "",
            _md(contrasts_table(contrasts, ("gsm8k", "pycode", "sciq", "norobots")), index=False),
            "",
            "Same contrasts on the base model and on forgetting (Uncheatable BPB after the GSM8K / Pycode fine-tune):",
            "",
            _md(
                contrasts_table(
                    contrasts, ("base_table9", "base_uncheatable", "unch_gsm8k", "unch_pycode", "c4_gsm8k", "c4_pycode")
                ),
                index=False,
            ),
            "",
        ]
    if (ANALYSIS / "july_by_direction.csv").exists():
        july = pd.read_csv(ANALYSIS / "july_by_direction.csv", header=[0, 1], index_col=[0, 1, 2])
        keep = [
            (c, s)
            for c in (
                "pooled_gsm8k",
                "pooled_pycode",
                "pooled_sciq",
                "pooled_norobots",
                "pooled_base_table9",
                "pooled_base_uncheatable",
            )
            for s in ("mean", "sem")
            if (c, s) in july.columns
        ]
        sub = july[keep].copy()
        sub.columns = [f"{c.removeprefix('pooled_')} {s}" for c, s in sub.columns]
        counts = july[("pooled_gsm8k", "count")] if ("pooled_gsm8k", "count") in july.columns else None
        if counts is not None:
            sub.insert(0, "n", counts.astype(int))
        parts += [
            "## July phase-asymmetry panel: treatment minus the mean of the anchor's 16 tied controls",
            "",
            "(The block-matched version, minus the control at the same seed block, is in "
            "analysis/july_by_direction.csv.)",
            "",
            "```",
            sub.to_string(),
            "```",
            "",
        ]
    if (ANALYSIS / "july_antithetic.csv").exists():
        anti = pd.read_csv(ANALYSIS / "july_antithetic.csv")
        if not anti.empty:
            cols = [
                c
                for c in anti.columns
                if c.endswith(("_gsm8k", "_pycode", "_base_table9", "_base_uncheatable"))
                and c.startswith(("linear_", "curvature_"))
            ]
            agg = anti.groupby(["anchor", "tv_level"])[cols].agg(["mean", "sem"]).round(4)
            agg.columns = [f"{a} {b}" for a, b in agg.columns]
            parts += [
                "## Antithetic partition pairs: linear (plus - minus)/2 and curvature (plus + minus)/2 terms",
                "",
                "```",
                agg.to_string(),
                "```",
                "",
            ]
    if (ANALYSIS / "regressions.csv").exists():
        fits = pd.read_csv(ANALYSIS / "regressions.csv")
        parts += ["## SFT deltas regressed on base deltas and late shifts", "", _md(fits.round(4), index=False), ""]
    if (ANALYSIS / "swarm_joint_regression.csv").exists():
        joint = pd.read_csv(ANALYSIS / "swarm_joint_regression.csv")
        parts += [
            "## Swarm pairs: joint OLS of the SFT delta on the base delta and the three late shifts",
            "",
            _md(joint.round(4), index=False),
            "",
        ]
    if (ANALYSIS / "matched_residuals.csv").exists():
        res = pd.read_csv(ANALYSIS / "matched_residuals.csv")
        if not res.empty:
            agg = (
                res.groupby(["population", "x", "direction", "set"])[["sft_delta", "base_delta", "residual"]]
                .agg(["mean", "count"])
                .round(4)
            )
            agg.columns = [f"{a} {b}" for a, b in agg.columns]
            parts += [
                "## Matched-set residuals (negative = midtraining benefit beyond the base-loss change)",
                "",
                "```",
                agg.to_string(),
                "```",
                "",
            ]
    if (ANALYSIS / "swarm_pairs.csv").exists():
        sw = pd.read_csv(ANALYSIS / "swarm_pairs.csv")
        cols = [
            c
            for c in (
                "gsm8k",
                "pycode",
                "sciq",
                "norobots",
                "base_table9",
                "base_uncheatable",
                "code_shift",
                "math_shift",
            )
            if c in sw.columns
        ]
        parts += [
            f"## Swarm pairs: two-phase minus its single-phase average ({len(sw)} designs landed)",
            "",
            _md(sw[cols].describe().round(4)),
            "",
        ]
    if (ANALYSIS / "frontier_pair.csv").exists():
        parts += [
            "## Frontier pair (original B1 run minus its same-seed tied control, seed 690300)",
            "",
            _md(pd.read_csv(ANALYSIS / "frontier_pair.csv").round(4), index=False),
            "",
        ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(parts))
    print(f"report rendered ({len(parts)} blocks)")


if __name__ == "__main__":
    main()
