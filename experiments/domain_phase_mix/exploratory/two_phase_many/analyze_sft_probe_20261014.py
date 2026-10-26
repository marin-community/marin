# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Analyze the SFT midtraining probe (PLAN.md analysis, amended by Calvin on 2026-10-14 10:40 PT).

Inputs: ``sft_results.csv`` and ``arms_table9.csv`` from ``collect_sft_probe_20261014.py``, the inventory's base
metrics, and ``swarm_phase_shifts.csv``. Outputs under ``reference_outputs/sft_midtraining_probe_20261014/analysis/``:

- ``arms_contrasts.csv`` / ``arms_summary.md``: B1-C (primary), B2-C, B3-C, B1-A, C-A, B'-A, CL-C, ML-C, each paired
  by trainer seed (mean over the SFT seeds within a seed, mean and SE over the three seeds) for the four SFT losses,
  the per-set forgetting BPBs, and the base deltas (Table-9, Uncheatable, C4) beside them.
- ``july_treatments.csv``: every July treatment with its block-matched delta (same anchor and seed block) and its
  pooled delta (minus the mean of the anchor's 16 tied controls), for SFT and base metrics; ``july_by_direction.csv``
  aggregates both.
- ``july_antithetic.csv``: each antithetic partition pair split into a linear term (plus - minus)/2 and a curvature
  term (plus + minus)/2, by TV level.
- ``swarm_pairs.csv`` / ``regressions.csv``: per-design (two-phase - single-phase) deltas, the regressions of the SFT
  deltas on the base deltas across July directions and swarm pairs, and on the late code/math shifts; the matched-set
  residuals (late code -> Pycode, late math -> GSM8K, late QA -> SciQ) are listed in ``matched_residuals.csv``.

SciQ is descriptive (pretraining-seed SD 0.0115, LR at the grid edge) and is excluded from the primary statements.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "sft_midtraining_probe_20261014"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"
INVENTORY = SCRIPT_DIR.parents[3] / ".agents/projects/sft_midtraining_probe/checkpoint_inventory.csv"
SETS = ("gsm8k", "pycode", "sciq", "norobots")
SFT_METRICS = (*SETS, *(f"unch_{s}" for s in SETS), *(f"c4_{s}" for s in SETS))
BASE_METRICS = ("base_table9", "base_uncheatable", "base_c4")
PRIMARY_SETS = ("gsm8k", "pycode")
B_CHOICE = "B1"
ARM_CONTRASTS = (("B1", "C"), ("B2", "C"), ("B3", "C"), ("B1", "A"), ("C", "A"), ("BP", "A"), ("CL", "C"), ("ML", "C"))
CODE_BUCKETS = (
    "dolma3_stack_edu",
    "dolmino_stack_edu_fim",
    "dolmino_synth_code",
    "dolma3_cc/electronics_and_hardware_high",
    "dolma3_cc/electronics_and_hardware_low",
)
MATH_BUCKETS = (
    "dolma3_finemath_3plus",
    "dolmino_synth_math",
    "dolma3_cc/science_math_and_technology_high",
    "dolma3_cc/science_math_and_technology_low",
    "dolmino_stem_heavy_crawl",
)
QA_BUCKETS = ("dolmino_synth_qa", "dolmino_synth_instruction", "dolmino_synth_thinking")
# Which SFT set a late shift should help, for the matched-set residuals.
MATCHED = {"code": "pycode", "math": "gsm8k", "qa": "sciq"}
DIRECTION_CONTENT = {
    "recipe_code": "code",
    "recipe_math_reasoning": "math",
    "recipe_instruction_qa": "qa",
    "recipe_synthetic_all": "qa",
    "CL": "code",
    "ML": "math",
}

logger = logging.getLogger(__name__)
KEYS = ["panel", "base_run_name", "arm", "design_id", "policy_class", "data_seed", "trainer_seed"]


def per_run_table(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per base run: SFT losses and per-set forgetting averaged over the SFT seeds that landed.

    Base metrics are added as columns.
    """
    frame = frame.copy()
    for column in ("arm", "design_id"):
        frame[column] = frame[column].fillna("").astype(str)
    keys = [*KEYS, "sft_seed"]
    losses = frame.pivot_table(index=keys, columns="dataset", values="sft_val_loss", aggfunc="mean")
    unch = frame.pivot_table(index=keys, columns="dataset", values="uncheatable_bpb", aggfunc="mean").add_prefix("unch_")
    c4 = frame.pivot_table(index=keys, columns="dataset", values="c4_bpb", aggfunc="mean").add_prefix("c4_")
    wide = losses.join(unch).join(c4).reset_index()
    present = [m for m in SFT_METRICS if m in wide.columns]
    runs = wide.groupby(KEYS)[present].mean().reset_index()
    runs["n_sft_seeds"] = wide.groupby(KEYS)["sft_seed"].nunique().to_numpy()
    base = frame.drop_duplicates("base_run_name").set_index("base_run_name")
    runs["base_table9"] = runs["base_run_name"].map(base["base_table9_bpb"])
    runs["base_uncheatable"] = runs["base_run_name"].map(base["base_uncheatable_bpb"])
    runs["base_c4"] = float("nan")
    for column in ("contrast_family", "anchor_or_design_id", "seed_block", "phase_tv"):
        runs[column] = runs["base_run_name"].map(base[column])
    return runs


def attach_arm_base_metrics(runs: pd.DataFrame, arms_table9: pd.DataFrame) -> pd.DataFrame:
    table = arms_table9.set_index("run_name")
    is_arm = runs["panel"] == "arms"
    for column, source in (
        ("base_table9", "table9_macro_bpb"),
        ("base_uncheatable", "base_uncheatable_bpb"),
        ("base_c4", "base_c4_bpb"),
    ):
        mapped = runs.loc[is_arm, "base_run_name"].map(table[source])
        runs.loc[is_arm, column] = mapped.where(mapped.notna(), runs.loc[is_arm, column])
    return runs


def _paired(left: pd.DataFrame, right: pd.DataFrame, metrics: list[str]) -> list[dict]:
    pair = left.merge(right, on="trainer_seed", suffixes=("_l", "_r"))
    rows = []
    for metric in metrics:
        delta = (pair[f"{metric}_l"] - pair[f"{metric}_r"]).dropna()
        if delta.empty:
            continue
        rows.append(
            {
                "metric": metric,
                "n_trainer_seeds": len(delta),
                "delta_mean": delta.mean(),
                "delta_se": delta.std(ddof=1) / math.sqrt(len(delta)) if len(delta) > 1 else float("nan"),
                "per_seed": json.dumps([round(float(v), 5) for v in delta]),
            }
        )
    return rows


def arm_contrasts(runs: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    arms = runs[runs["panel"] == "arms"]
    metrics = [m for m in (*SFT_METRICS, *BASE_METRICS) if m in arms.columns]
    rows = []
    for left, right in ARM_CONTRASTS:
        for row in _paired(arms[arms["arm"] == left], arms[arms["arm"] == right], metrics):
            rows.append({"contrast": f"{left}-{right}", "primary": left == B_CHOICE and right == "C", **row})
    contrasts = pd.DataFrame(
        rows, columns=["contrast", "primary", "metric", "n_trainer_seeds", "delta_mean", "delta_se", "per_seed"]
    )
    summary = (
        arms.groupby("arm")[[m for m in (*SETS, "base_table9", "base_uncheatable") if m in arms.columns]]
        .agg(["mean", "std", "count"])
        .round(4)
    )
    lines = [
        f"# Arms (B = {B_CHOICE} by content; SciQ descriptive)",
        "",
        "Per-arm means over trainer seeds (each a mean over its SFT seeds); std over trainer seeds.",
        "",
        summary.to_string(),
        "",
        "## Paired contrasts by trainer seed: mean ± SE (n trainer seeds)",
        "",
    ]
    for contrast, group in contrasts.groupby("contrast", sort=False):
        parts = [
            f"{r.metric} {r.delta_mean:+.4f}±{r.delta_se:.4f} (n={r.n_trainer_seeds})"
            for r in group.itertuples()
            if not r.metric.startswith(("c4_", "unch_"))
        ]
        forget = [f"{r.metric} {r.delta_mean:+.4f}" for r in group.itertuples() if r.metric.startswith(("unch_", "c4_"))]
        lines.append(f"- **{contrast}**{' (primary)' if contrast == f'{B_CHOICE}-C' else ''}: " + "; ".join(parts))
        if forget:
            lines.append("  forgetting: " + "; ".join(forget))
    return contrasts, "\n".join(lines) + "\n"


def july_treatments(runs: pd.DataFrame) -> pd.DataFrame:
    july = runs[runs["panel"] == "july"].copy()
    metrics = [m for m in (*SFT_METRICS, *BASE_METRICS) if m in july.columns]
    controls = july[july["contrast_family"] == "center_control"]
    pooled = controls.groupby("anchor_or_design_id")[metrics].mean()
    pooled_n = controls.groupby("anchor_or_design_id")["base_run_name"].count()
    matched = controls.set_index(["anchor_or_design_id", "seed_block"])[metrics]
    treatments = july[july["contrast_family"] != "center_control"].copy()
    treatments["direction"] = (
        treatments["base_run_name"]
        .str.replace(r"^agphase_a\d_", "", regex=True)
        .str.replace(r"_(plus|minus)_tv\d+(_rep\d)?$", "", regex=True)
    )
    treatments["sign"] = treatments["base_run_name"].str.extract(r"_(plus|minus)_tv")[0]
    treatments["tv_level"] = treatments["base_run_name"].str.extract(r"_tv(\d+)")[0].astype(float) / 100
    treatments["n_pooled_controls"] = treatments["anchor_or_design_id"].map(pooled_n)
    for metric in metrics:
        keys = list(zip(treatments["anchor_or_design_id"], treatments["seed_block"], strict=True))
        treatments[f"matched_{metric}"] = [
            treatments[metric].iloc[i] - matched.loc[k, metric] if k in matched.index else float("nan")
            for i, k in enumerate(keys)
        ]
        treatments[f"pooled_{metric}"] = treatments[metric] - treatments["anchor_or_design_id"].map(pooled[metric])
    return treatments


def july_by_direction(treatments: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in treatments.columns if c.startswith(("matched_", "pooled_"))]
    return (
        treatments.groupby(["anchor_or_design_id", "contrast_family", "direction"])[cols]
        .agg(["mean", "sem", "count"])
        .round(4)
    )


def july_antithetic(treatments: pd.DataFrame) -> pd.DataFrame:
    """Linear and curvature terms of each antithetic partition pair, from the pooled deltas.

    Both signs share a block.
    """
    parts = treatments[treatments["contrast_family"] == "balanced_partition"]
    metrics = [c for c in parts.columns if c.startswith("pooled_")]
    rows = []
    for (anchor, direction, tv), group in parts.groupby(["anchor_or_design_id", "direction", "tv_level"]):
        plus = group[group["sign"] == "plus"]
        minus = group[group["sign"] == "minus"]
        if plus.empty or minus.empty:
            continue
        row = {"anchor": anchor, "direction": direction, "tv_level": tv}
        for metric in metrics:
            name = metric.removeprefix("pooled_")
            p, m = float(plus[metric].iloc[0]), float(minus[metric].iloc[0])
            row[f"linear_{name}"] = (p - m) / 2
            row[f"curvature_{name}"] = (p + m) / 2
        rows.append(row)
    return pd.DataFrame(rows)


def swarm_pairs(runs: pd.DataFrame, shifts: pd.DataFrame | None) -> pd.DataFrame:
    swarm = runs[runs["panel"] == "swarm"]
    metrics = [m for m in (*SFT_METRICS, *BASE_METRICS) if m in swarm.columns]
    two = swarm[swarm["policy_class"] == "two_phase"].set_index("design_id")
    one = swarm[swarm["policy_class"] == "one_phase"].set_index("design_id")
    common = two.index.intersection(one.index)
    deltas = (two.loc[common, metrics] - one.loc[common, metrics]).reset_index()
    deltas["phase_tv"] = deltas["design_id"].map(two["phase_tv"])
    if shifts is not None:
        shift = shifts.pivot_table(index="design_id", columns="domain", values="shift")
        for name, buckets in (("code_shift", CODE_BUCKETS), ("math_shift", MATH_BUCKETS), ("qa_shift", QA_BUCKETS)):
            deltas[name] = deltas["design_id"].map(shift[[c for c in buckets if c in shift.columns]].sum(axis=1))
    return deltas


def _fit(x: np.ndarray, y: np.ndarray) -> dict | None:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 4 or np.ptp(x) == 0:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    se = math.sqrt(resid.var(ddof=2) / ((x - x.mean()) ** 2).sum()) if len(x) > 2 else float("nan")
    return {
        "n": len(x),
        "slope": float(slope),
        "slope_se": float(se),
        "intercept": float(intercept),
        "pearson_r": float(np.corrcoef(x, y)[0, 1]),
        "resid_sd": float(resid.std(ddof=2)),
    }


def swarm_joint_regression(deltas: pd.DataFrame) -> pd.DataFrame:
    """OLS of each pair's SFT delta on its base delta and all three late shifts at once (shifts are correlated)."""
    rows = []
    predictors = [c for c in ("base_uncheatable", "code_shift", "math_shift", "qa_shift") if c in deltas.columns]
    for sft_set in SETS:
        if sft_set not in deltas.columns or len(predictors) < 2:
            continue
        sub = deltas[[sft_set, *predictors]].dropna()
        x = np.column_stack([np.ones(len(sub)), sub[predictors].to_numpy()])
        y = sub[sft_set].to_numpy()
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        resid = y - x @ beta
        dof = len(y) - x.shape[1]
        cov = resid @ resid / dof * np.linalg.inv(x.T @ x)
        se = np.sqrt(np.diag(cov))
        for name, b, s_ in zip(["intercept", *predictors], beta, se, strict=True):
            rows.append(
                {
                    "y": sft_set,
                    "term": name,
                    "coef": float(b),
                    "se": float(s_),
                    "n": len(y),
                    "resid_sd": float(resid.std(ddof=x.shape[1])),
                }
            )
    return pd.DataFrame(rows)


def regressions(treatments: pd.DataFrame, deltas: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SFT deltas on base deltas (July pooled treatments; swarm pairs) and on late shifts (swarm); matched residuals."""
    rows, residual_rows = [], []
    for sft_set in SETS:
        for base in ("base_table9", "base_uncheatable"):
            if f"pooled_{sft_set}" in treatments.columns and f"pooled_{base}" in treatments.columns:
                sub = treatments[
                    [
                        f"pooled_{base}",
                        f"pooled_{sft_set}",
                        "direction",
                        "anchor_or_design_id",
                        "base_run_name",
                        "contrast_family",
                    ]
                ].dropna()
                fit = _fit(sub[f"pooled_{base}"].to_numpy(), sub[f"pooled_{sft_set}"].to_numpy())
                if fit:
                    rows.append({"population": "july_pooled", "y": sft_set, "x": base, **fit})
                    resid = sub[f"pooled_{sft_set}"] - (fit["slope"] * sub[f"pooled_{base}"] + fit["intercept"])
                    for i, r in sub.iterrows():
                        content = DIRECTION_CONTENT.get(r["direction"])
                        if content and MATCHED[content] == sft_set:
                            residual_rows.append(
                                {
                                    "population": "july_pooled",
                                    "x": base,
                                    "base_run_name": r["base_run_name"],
                                    "direction": r["direction"],
                                    "anchor": r["anchor_or_design_id"],
                                    "set": sft_set,
                                    "sft_delta": r[f"pooled_{sft_set}"],
                                    "base_delta": r[f"pooled_{base}"],
                                    "residual": float(resid.loc[i]),
                                }
                            )
            if sft_set in deltas.columns and base in deltas.columns:
                fit = _fit(deltas[base].to_numpy(), deltas[sft_set].to_numpy())
                if fit:
                    rows.append({"population": "swarm_pairs", "y": sft_set, "x": base, **fit})
                    # Matched-set residual for the swarm: the part of the pair's SFT delta that the base delta does
                    # not explain, regressed on the late shift that should help this set (negative slope = benefit).
                    matched_shift = next((f"{c}_shift" for c, s in MATCHED.items() if s == sft_set), None)
                    if matched_shift in deltas.columns:
                        resid = deltas[sft_set] - (fit["slope"] * deltas[base] + fit["intercept"])
                        resid_fit = _fit(deltas[matched_shift].to_numpy(), resid.to_numpy())
                        if resid_fit:
                            rows.append(
                                {
                                    "population": "swarm_pairs_residual",
                                    "y": f"{sft_set}|{base}",
                                    "x": matched_shift,
                                    **resid_fit,
                                }
                            )
        for shift in ("code_shift", "math_shift", "qa_shift"):
            if sft_set in deltas.columns and shift in deltas.columns:
                fit = _fit(deltas[shift].to_numpy(), deltas[sft_set].to_numpy())
                if fit:
                    rows.append({"population": "swarm_pairs", "y": sft_set, "x": shift, **fit})
    return pd.DataFrame(rows), pd.DataFrame(residual_rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=OUTPUT_DIR / "sft_results.csv")
    parser.add_argument("--arms-table9", type=Path, default=OUTPUT_DIR / "arms_table9.csv")
    parser.add_argument("--phase-shifts", type=Path, default=OUTPUT_DIR / "swarm_phase_shifts.csv")
    args = parser.parse_args()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.results)
    runs = per_run_table(frame)
    if args.arms_table9.exists():
        runs = attach_arm_base_metrics(runs, pd.read_csv(args.arms_table9))
    runs.to_csv(ANALYSIS_DIR / "per_run.csv", index=False)

    if (runs["panel"] == "arms").any():
        contrasts, text = arm_contrasts(runs)
        contrasts.to_csv(ANALYSIS_DIR / "arms_contrasts.csv", index=False)
        (ANALYSIS_DIR / "arms_summary.md").write_text(text)
        print(text)

    treatments = july_treatments(runs) if (runs["panel"] == "july").any() else pd.DataFrame()
    if not treatments.empty:
        treatments.to_csv(ANALYSIS_DIR / "july_treatments.csv", index=False)
        by_direction = july_by_direction(treatments)
        by_direction.to_csv(ANALYSIS_DIR / "july_by_direction.csv")
        show = [
            c
            for c in by_direction.columns
            if c[0] in {f"{kind}_{s}" for kind in ("matched", "pooled") for s in ("gsm8k", "pycode", "base_table9")}
            and c[1] != "count"
        ]
        print(
            "\nJuly: treatment minus control by direction (matched = same seed block; pooled = mean of 16 controls)\n"
            + by_direction[show].to_string()
        )
        antithetic = july_antithetic(treatments)
        antithetic.to_csv(ANALYSIS_DIR / "july_antithetic.csv", index=False)
        if not antithetic.empty:
            cols = [c for c in antithetic.columns if c.endswith(("_gsm8k", "_pycode", "_base_table9"))]
            print(
                "\nAntithetic pairs by TV level (mean over partitions):\n"
                + antithetic.groupby(["anchor", "tv_level"])[cols].agg(["mean", "sem", "count"]).round(4).to_string()
            )

    shifts = pd.read_csv(args.phase_shifts) if args.phase_shifts.exists() else None
    deltas = swarm_pairs(runs, shifts) if (runs["panel"] == "swarm").any() else pd.DataFrame()
    if not deltas.empty:
        deltas.to_csv(ANALYSIS_DIR / "swarm_pairs.csv", index=False)
        print(
            "\nSwarm pairs (two-phase minus single-phase):\n"
            + deltas[[c for c in SETS if c in deltas] + [c for c in ("base_table9", "base_uncheatable") if c in deltas]]
            .describe()
            .round(4)
            .to_string()
        )
    if not treatments.empty or not deltas.empty:
        joint = swarm_joint_regression(deltas)
        joint.to_csv(ANALYSIS_DIR / "swarm_joint_regression.csv", index=False)
        print(
            "\nSwarm pairs, joint OLS (SFT delta ~ base Uncheatable delta + code/math/qa late shifts):\n"
            + joint.round(4).to_string()
        )
        fits, residuals = regressions(treatments, deltas)
        fits.to_csv(ANALYSIS_DIR / "regressions.csv", index=False)
        residuals.to_csv(ANALYSIS_DIR / "matched_residuals.csv", index=False)
        print(
            "\nRegressions of SFT deltas on base deltas / late shifts:\n"
            + (fits.round(4).to_string() if not fits.empty else "(none yet)")
        )
        if not residuals.empty:
            print(
                "\nMatched-set residuals (negative = midtraining benefit beyond the base-loss change):\n"
                + residuals.groupby(["population", "x", "direction", "set"])[["sft_delta", "base_delta", "residual"]]
                .agg(["mean", "count"])
                .round(4)
                .to_string()
            )


if __name__ == "__main__":
    main()
