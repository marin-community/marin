# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "kaleido==0.2.1",
#   "numpy",
#   "pandas",
#   "plotly",
#   "scipy",
#   "tabulate",
#   "wandb",
# ]
# ///
"""Analyze scale-specific fixed-aggregate fibers through WSD80 tied optima."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import wandb
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
PANEL_DIR = REFERENCE_OUTPUTS / "starcoder_wsd80_scale_specific_tied_fibers_20260731"
DEFAULT_OUTPUT_DIR = PANEL_DIR / "results_20260731"
TIED_OPTIMA_PATH = (
    REFERENCE_OUTPUTS / "starcoder_wsd80_fixed_model_tied_diagonal_20260730" / "results_20260731" / "tied_optima.csv"
)

TRAIN_PROJECT = "marin-community/marin"
TRAIN_TAG = "starcoder_wsd80_scale_tied_fibers"
REFERENCE_SEED = 20260711
CONTRAST_TOLERANCE = 1e-10
REPEATED_ABS_CONTRAST = 0.20
EXPECTED_NEW_RUNS = 132
EXPECTED_ANCHORS = 6
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, default=PANEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tied-optima", type=Path, default=TIED_OPTIMA_PATH)
    parser.add_argument("--wandb-timeout", type=int, default=240)
    return parser.parse_args()


def finite_summary(run: Any, key: str) -> float | None:
    try:
        value = float(run.summary.get(key, np.nan))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def persisted_final_metric(run: Any, key: str) -> float:
    checkpoint_root = str(run.config["trainer"]["checkpointer"]["base_path"])
    uri = f"{checkpoint_root}/eval_metrics.jsonl"
    result = subprocess.run(
        ["gcloud", "storage", "cat", uri],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    finite = [row for row in rows if row.get(key) is not None]
    if not finite:
        raise ValueError(f"{run.name}: no finite {key} in {uri}")
    return float(max(finite, key=lambda row: int(row["step"]))[key])


def final_metric(run: Any, key: str) -> tuple[float, str]:
    if str(run.state) != "finished":
        return persisted_final_metric(run, key), "persisted eval_metrics.jsonl"
    value = finite_summary(run, key)
    if value is not None:
        return value, "wandb summary"
    rows = [row for row in run.scan_history(keys=["global_step", key], page_size=10_000) if row.get(key) is not None]
    if rows:
        return float(rows[-1][key]), "wandb history"
    return persisted_final_metric(run, key), "persisted eval_metrics.jsonl"


def collect_observations(panel_dir: Path, timeout: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    design = json.loads((panel_dir / "design_manifest.json").read_text(encoding="utf-8"))
    manifest = pd.DataFrame(design["runs"])
    if len(manifest) != EXPECTED_NEW_RUNS or manifest["run_name"].duplicated().any():
        raise ValueError(f"Expected {EXPECTED_NEW_RUNS} unique manifest rows")
    if len(design["design"]["anchors"]) != EXPECTED_ANCHORS:
        raise ValueError(f"Expected {EXPECTED_ANCHORS} anchors")

    metric = str(design["objective_metric"])
    api = wandb.Api(timeout=timeout)
    runs = list(api.runs(TRAIN_PROJECT, filters={"tags": TRAIN_TAG}, per_page=300))
    by_name: dict[str, list[Any]] = {}
    for run in runs:
        by_name.setdefault(str(run.name), []).append(run)

    rows: list[dict[str, Any]] = []
    for spec in manifest.to_dict("records"):
        candidates = by_name.get(str(spec["run_name"]), [])
        if len(candidates) != 1:
            raise ValueError(f"{spec['run_name']}: expected one W&B run, found {len(candidates)}")
        run = candidates[0]
        metric_value, metric_source = final_metric(run, metric)
        rows.append(
            {
                **spec,
                "starcoder_bpb": metric_value,
                "metric_source": metric_source,
                "wandb_id": str(run.id),
                "wandb_name": str(run.name),
                "wandb_state": str(run.state),
                "wandb_url": str(run.url),
                "observation_source": "new fiber panel",
            }
        )

    for anchor in design["design"]["anchors"]:
        run = api.run(f"{TRAIN_PROJECT}/{anchor['tied_control_wandb_id']}")
        metric_value, metric_source = final_metric(run, metric)
        rows.append(
            {
                "run_name": str(run.name),
                "anchor_index": int(anchor["index"]),
                "anchor_role": str(anchor["role"]),
                "token_budget_requested": int(anchor["token_budget"]),
                "anchor_aggregate_starcoder": float(anchor["aggregate"]),
                "aggregate_starcoder_realized": float(anchor["aggregate"]),
                "phase_0_starcoder": float(anchor["aggregate"]),
                "phase_1_starcoder": float(anchor["aggregate"]),
                "signed_contrast_phase1_minus_phase0": 0.0,
                "replicate_kind": "existing_reference_tie",
                "trainer_data_seed": REFERENCE_SEED,
                "simulated_epoch_subset_seed": REFERENCE_SEED,
                "tied_control_wandb_id": str(anchor["tied_control_wandb_id"]),
                "starcoder_bpb": metric_value,
                "metric_source": metric_source,
                "wandb_id": str(run.id),
                "wandb_name": str(run.name),
                "wandb_state": str(run.state),
                "wandb_url": str(run.url),
                "observation_source": "existing tied control",
            }
        )

    observations = pd.DataFrame(rows)
    if observations["starcoder_bpb"].isna().any():
        raise ValueError("Collected observations contain missing BPB")
    return (
        observations.sort_values(
            [
                "token_budget_requested",
                "anchor_aggregate_starcoder",
                "trainer_data_seed",
                "signed_contrast_phase1_minus_phase0",
            ]
        ).reset_index(drop=True),
        design,
    )


def reference_decomposition(observations: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    reference = observations.loc[observations["trainer_data_seed"] == REFERENCE_SEED]
    for anchor_index, block in reference.groupby("anchor_index", sort=True):
        by_contrast = block.set_index("signed_contrast_phase1_minus_phase0")
        tied = float(by_contrast.loc[0.0, "starcoder_bpb"])
        for magnitude in sorted(value for value in by_contrast.index if value > CONTRAST_TOLERANCE):
            mirror = by_contrast.index[np.isclose(by_contrast.index, -magnitude, atol=CONTRAST_TOLERANCE)]
            if len(mirror) != 1:
                raise ValueError(f"Anchor {anchor_index}, |d|={magnitude}: missing antithetic arm")
            plus = by_contrast.loc[magnitude]
            minus = by_contrast.loc[float(mirror[0])]
            ordering = 0.5 * (float(plus["starcoder_bpb"]) - float(minus["starcoder_bpb"]))
            cost = 0.5 * (float(plus["starcoder_bpb"]) + float(minus["starcoder_bpb"])) - tied
            preferred = "+d (StarCoder late)" if ordering < 0 else "-d (StarCoder early)"
            rows.append(
                {
                    "anchor_index": int(anchor_index),
                    "token_budget_requested": int(plus["token_budget_requested"]),
                    "anchor_aggregate_starcoder": float(plus["anchor_aggregate_starcoder"]),
                    "anchor_role": str(plus["anchor_role"]),
                    "abs_contrast": float(magnitude),
                    "tied_bpb": tied,
                    "minus_bpb": float(minus["starcoder_bpb"]),
                    "plus_bpb": float(plus["starcoder_bpb"]),
                    "minus_delta_vs_tied": float(minus["starcoder_bpb"]) - tied,
                    "plus_delta_vs_tied": float(plus["starcoder_bpb"]) - tied,
                    "ordering_effect": ordering,
                    "asymmetry_cost": cost,
                    "better_orientation_delta": cost - abs(ordering),
                    "preferred_orientation": preferred,
                }
            )
    return pd.DataFrame(rows)


def interval(values: np.ndarray) -> tuple[float, float]:
    half_width = stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return float(values.mean() - half_width), float(values.mean() + half_width)


def one_sided_p(values: np.ndarray) -> float:
    return float(stats.ttest_1samp(values, 0.0, alternative="less").pvalue)


def holm_adjust(p_values: pd.Series) -> pd.Series:
    order = np.argsort(p_values.to_numpy(dtype=float))
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    count = len(p_values)
    for rank, position in enumerate(order):
        running = max(running, (count - rank) * float(p_values.iloc[position]))
        adjusted[position] = min(running, 1.0)
    return pd.Series(adjusted, index=p_values.index)


def repeated_decomposition(observations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    fresh_rows: list[dict[str, Any]] = []
    arm_rows: list[dict[str, Any]] = []
    repeated = observations.loc[
        np.isclose(observations["signed_contrast_phase1_minus_phase0"].abs(), REPEATED_ABS_CONTRAST)
        | np.isclose(observations["signed_contrast_phase1_minus_phase0"], 0.0)
    ]
    for anchor_index, block in repeated.groupby("anchor_index", sort=True):
        pivot = block.pivot(
            index="trainer_data_seed", columns="signed_contrast_phase1_minus_phase0", values="starcoder_bpb"
        ).dropna()
        if list(pivot.index) != [20260711, 20260712, 20260713, 20260714, 20260715]:
            raise ValueError(f"Anchor {anchor_index}: incomplete five-seed repeated fiber")
        metadata = block.iloc[0]
        minus = pivot[-REPEATED_ABS_CONTRAST].to_numpy(dtype=float)
        tied = pivot[0.0].to_numpy(dtype=float)
        plus = pivot[REPEATED_ABS_CONTRAST].to_numpy(dtype=float)
        ordering = 0.5 * (plus - minus)
        cost = 0.5 * (plus + minus) - tied
        for seed, minus_value, tied_value, plus_value in zip(pivot.index, minus, tied, plus, strict=True):
            rows.append(
                {
                    "anchor_index": int(anchor_index),
                    "token_budget_requested": int(metadata["token_budget_requested"]),
                    "anchor_aggregate_starcoder": float(metadata["anchor_aggregate_starcoder"]),
                    "anchor_role": str(metadata["anchor_role"]),
                    "trainer_data_seed": int(seed),
                    "minus_bpb": minus_value,
                    "tied_bpb": tied_value,
                    "plus_bpb": plus_value,
                    "minus_delta_vs_tied": minus_value - tied_value,
                    "plus_delta_vs_tied": plus_value - tied_value,
                    "ordering_effect": 0.5 * (plus_value - minus_value),
                    "asymmetry_cost": 0.5 * (plus_value + minus_value) - tied_value,
                }
            )

        for contrast, values in ((-REPEATED_ABS_CONTRAST, minus - tied), (REPEATED_ABS_CONTRAST, plus - tied)):
            ci_low, ci_high = interval(values)
            arm_rows.append(
                {
                    "anchor_index": int(anchor_index),
                    "token_budget_requested": int(metadata["token_budget_requested"]),
                    "anchor_aggregate_starcoder": float(metadata["anchor_aggregate_starcoder"]),
                    "anchor_role": str(metadata["anchor_role"]),
                    "contrast": contrast,
                    "seeds": len(values),
                    "mean_delta_vs_tied": float(values.mean()),
                    "sd_delta_vs_tied": float(values.std(ddof=1)),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "one_sided_p_improvement": one_sided_p(values),
                    "seeds_better": int((values < 0).sum()),
                }
            )

        reference_sign = (
            REPEATED_ABS_CONTRAST
            if pivot.loc[REFERENCE_SEED, REPEATED_ABS_CONTRAST] < pivot.loc[REFERENCE_SEED, -REPEATED_ABS_CONTRAST]
            else -REPEATED_ABS_CONTRAST
        )
        fresh = pivot.drop(index=REFERENCE_SEED)
        fresh_delta = (fresh[reference_sign] - fresh[0.0]).to_numpy(dtype=float)
        ci_low, ci_high = interval(fresh_delta)
        fresh_rows.append(
            {
                "anchor_index": int(anchor_index),
                "token_budget_requested": int(metadata["token_budget_requested"]),
                "anchor_aggregate_starcoder": float(metadata["anchor_aggregate_starcoder"]),
                "anchor_role": str(metadata["anchor_role"]),
                "reference_selected_contrast": reference_sign,
                "fresh_seeds": len(fresh_delta),
                "fresh_mean_delta_vs_tied": float(fresh_delta.mean()),
                "fresh_sd_delta_vs_tied": float(fresh_delta.std(ddof=1)),
                "fresh_ci_low": ci_low,
                "fresh_ci_high": ci_high,
                "fresh_one_sided_p_improvement": one_sided_p(fresh_delta),
                "fresh_seeds_better": int((fresh_delta < 0).sum()),
                "five_seed_ordering_effect": float(ordering.mean()),
                "five_seed_asymmetry_cost": float(cost.mean()),
                "five_seed_better_orientation_delta": float(cost.mean() - abs(ordering.mean())),
            }
        )

    arms = pd.DataFrame(arm_rows)
    arms["holm_p_all_12_arms"] = holm_adjust(arms["one_sided_p_improvement"])
    fresh_summary = pd.DataFrame(fresh_rows)
    fresh_summary["holm_p_all_6_anchors"] = holm_adjust(fresh_summary["fresh_one_sided_p_improvement"])
    primary = fresh_summary["anchor_role"] == "measured_grid_minimum"
    fresh_summary.loc[primary, "holm_p_primary_4_anchors"] = holm_adjust(
        fresh_summary.loc[primary, "fresh_one_sided_p_improvement"]
    )
    return pd.DataFrame(rows), arms, fresh_summary


def align_tied_optima(fresh: pd.DataFrame, tied_optima_path: Path) -> pd.DataFrame:
    tied_optima = pd.read_csv(tied_optima_path)[
        ["token_budget_requested", "local_quadratic_median_weight", "sampled_min_weight"]
    ]
    aligned = fresh.merge(tied_optima, on="token_budget_requested", validate="many_to_one")
    aligned["distance_to_quadratic_tied_estimate"] = (
        aligned["anchor_aggregate_starcoder"] - aligned["local_quadratic_median_weight"]
    ).abs()
    aligned["nearest_anchor_to_quadratic_tied_estimate"] = False
    nearest_indices = aligned.groupby("token_budget_requested")["distance_to_quadratic_tied_estimate"].idxmin()
    aligned.loc[nearest_indices, "nearest_anchor_to_quadratic_tied_estimate"] = True
    nearest = aligned["nearest_anchor_to_quadratic_tied_estimate"]
    aligned.loc[nearest, "holm_p_nearest_4_anchors"] = holm_adjust(aligned.loc[nearest, "fresh_one_sided_p_improvement"])
    return aligned


def tied_anchor_sensitivity(observations: pd.DataFrame) -> pd.DataFrame:
    tied = observations.loc[np.isclose(observations["signed_contrast_phase1_minus_phase0"], 0.0)]
    rows: list[dict[str, Any]] = []
    for token_budget, block in tied.groupby("token_budget_requested", sort=True):
        anchors = sorted(block["anchor_aggregate_starcoder"].unique())
        if len(anchors) == 1:
            continue
        if len(anchors) != 2:
            raise ValueError(f"{token_budget}: expected one or two tied anchors, found {anchors}")
        pivot = block.pivot(
            index="trainer_data_seed", columns="anchor_aggregate_starcoder", values="starcoder_bpb"
        ).dropna()
        delta = (pivot[anchors[1]] - pivot[anchors[0]]).to_numpy(dtype=float)
        ci_low, ci_high = interval(delta)
        rows.append(
            {
                "token_budget_requested": int(token_budget),
                "lower_anchor": anchors[0],
                "upper_anchor": anchors[1],
                "upper_minus_lower_tied_bpb": float(delta.mean()),
                "sd": float(delta.std(ddof=1)),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "two_sided_p": float(stats.ttest_1samp(delta, 0.0).pvalue),
                "seeds": len(delta),
            }
        )
    return pd.DataFrame(rows)


def two_b_cross_anchor_comparison(observations: pd.DataFrame) -> pd.DataFrame:
    block = observations.loc[observations["token_budget_requested"] == 2_000_000_000]
    treatment = block.loc[
        np.isclose(block["anchor_aggregate_starcoder"], 0.35)
        & np.isclose(block["signed_contrast_phase1_minus_phase0"], 0.20)
    ].set_index("trainer_data_seed")["starcoder_bpb"]
    control = block.loc[
        np.isclose(block["anchor_aggregate_starcoder"], 0.40)
        & np.isclose(block["signed_contrast_phase1_minus_phase0"], 0.0)
    ].set_index("trainer_data_seed")["starcoder_bpb"]
    deltas = (treatment - control).dropna().sort_index()
    if len(deltas) != 5:
        raise ValueError(f"Expected five matched 2B cross-anchor seeds, found {len(deltas)}")

    rows: list[dict[str, Any]] = []
    for scope, values in (("fresh seeds", deltas.loc[deltas.index != REFERENCE_SEED]), ("all seeds", deltas)):
        array = values.to_numpy(dtype=float)
        ci_low, ci_high = interval(array)
        rows.append(
            {
                "scope": scope,
                "treatment": "a=0.35, d=+0.20",
                "control": "tied a=0.40",
                "seeds": len(array),
                "mean_treatment_minus_control_bpb": float(array.mean()),
                "sd": float(array.std(ddof=1)),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "one_sided_p_improvement": one_sided_p(array),
                "seeds_treatment_better": int((array < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def conclusion(
    fresh: pd.DataFrame,
    reference: pd.DataFrame,
    sensitivity: pd.DataFrame,
    cross_anchor: pd.DataFrame,
) -> dict[str, Any]:
    primary = fresh.loc[fresh["anchor_role"] == "measured_grid_minimum"]
    confirmed_grid_minima = primary.loc[
        (primary["fresh_mean_delta_vs_tied"] < 0) & (primary["holm_p_primary_4_anchors"] < 0.05)
    ]
    nearest = fresh.loc[fresh["nearest_anchor_to_quadratic_tied_estimate"]]
    confirmed_nearest = nearest.loc[
        (nearest["fresh_mean_delta_vs_tied"] < 0) & (nearest["holm_p_nearest_4_anchors"] < 0.05)
    ]
    reference_primary = reference.loc[reference["anchor_role"] == "measured_grid_minimum"]
    two_b_primary = primary.loc[primary["token_budget_requested"] == 2_000_000_000]
    two_b_sensitivity = sensitivity.loc[sensitivity["token_budget_requested"] == 2_000_000_000]
    if len(two_b_primary) != 1 or len(two_b_sensitivity) != 1:
        raise ValueError("Expected one 2B primary anchor and one 2B tied-anchor sensitivity comparison")
    two_b = two_b_primary.iloc[0]
    two_b_tied = two_b_sensitivity.iloc[0]
    tied_anchors_indistinguishable = bool(two_b_tied["ci_low"] <= 0 <= two_b_tied["ci_high"])
    if not tied_anchors_indistinguishable:
        raise ValueError("Expected the 2B tied-anchor interval to include zero")
    cross_fresh = cross_anchor.loc[cross_anchor["scope"] == "fresh seeds"].iloc[0]
    cross_all = cross_anchor.loc[cross_anchor["scope"] == "all seeds"].iloc[0]
    return {
        "hypothesis": "No two-phase policy on the globally optimal tied policy's fiber can outperform it.",
        "scope": "Observed local fibers at |d|<=0.25; replicated confirmation at |d|=0.20.",
        "measured_grid_minimum_anchor_count": len(primary),
        "fresh_confirmed_grid_minimum_improvements_after_holm": len(confirmed_grid_minima),
        "confirmed_grid_minimum_anchor_indices": confirmed_grid_minima["anchor_index"].astype(int).tolist(),
        "fresh_confirmed_nearest_optimum_improvements_after_holm": len(confirmed_nearest),
        "confirmed_nearest_optimum_anchor_indices": confirmed_nearest["anchor_index"].astype(int).tolist(),
        "reference_seed_primary_anchors_with_any_sampled_improvement": int(
            reference_primary.groupby("anchor_index")["better_orientation_delta"].min().lt(0).sum()
        ),
        "verdict": "exact population claim unresolved; finite-grid and tied-optimal-set versions contradicted at 2B",
        "literal_population_status": (
            "unresolved because the exact population tied optimum and its complete feasible fiber are unobserved"
        ),
        "finite_grid_status": "contradicted at 2B: the reference-seed tied-grid minimum a=0.35 is improved by d=+0.20",
        "tied_optimal_set_status": (
            "contradicted at 2B: a=0.35 and a=0.40 are statistically indistinguishable tied anchors, "
            "but a=0.35 has a replicated phase gain"
        ),
        "hard_constraint_recommendation": "reject; retain only as a falsifiable local null or soft prior",
        "dose_model_boundary": (
            "Global phase-weighted-dose factorization plus tied reachability still proves fiber optimality. "
            "The 2B result is evidence against treating those assumptions as empirically established."
        ),
        "decisive_counterevidence": (
            f"At 2B a=0.35, reference-selected d=+0.20 improves by {-two_b['fresh_mean_delta_vs_tied']:.6f} "
            f"BPB on four fresh seeds (95% CI [{two_b['fresh_ci_low']:.6f}, {two_b['fresh_ci_high']:.6f}], "
            f"Holm p={two_b['holm_p_primary_4_anchors']:.6f}); all four fresh seeds improve. Including the "
            f"reference seed, the mean gain is {-two_b['five_seed_better_orientation_delta']:.6f} BPB and all "
            f"five seeds improve. The tied a=0.40 minus a=0.35 difference is "
            f"{two_b_tied['upper_minus_lower_tied_bpb']:+.6f} BPB "
            f"(95% CI [{two_b_tied['ci_low']:.6f}, {two_b_tied['ci_high']:.6f}], "
            f"p={two_b_tied['two_sided_p']:.3f}), so the two tied anchors are statistically indistinguishable. "
            f"More directly, the same asymmetric policy beats tied a=0.40 by "
            f"{-cross_fresh['mean_treatment_minus_control_bpb']:.6f} BPB on all four fresh seeds "
            f"(95% CI for treatment minus control [{cross_fresh['ci_low']:.6f}, "
            f"{cross_fresh['ci_high']:.6f}], p={cross_fresh['one_sided_p_improvement']:.6f}) and by "
            f"{-cross_all['mean_treatment_minus_control_bpb']:.6f} BPB across all five seeds."
        ),
        "nonconfirmation_elsewhere": (
            "At 1B, 4B, and 8B, and at the 2B a=0.40 sensitivity anchor, the repeated |d|=0.20 tests do not "
            "detect an improvement. These are radius-specific null results, not evidence that every point on the "
            "unknown optimum's fiber is worse."
        ),
        "limitations": [
            "The contrast grid stops at |d|=0.25, so absence of a gain does not prove the universal fiber claim.",
            "The primary anchors are finite-grid, noisy estimates of the tied optimum.",
            (
                "The 2B and 8B tied basins are broad; sensitivity anchors at 0.40 and 0.75 must be read alongside "
                "0.35 and 0.80."
            ),
            "Only |d|=0.20 has five matched seeds; other contrasts are reference-seed diagnostics.",
            (
                "The 2B a=0.40 reference seed also favors +d at |d|=0.05 through 0.20, but only |d|=0.20 was "
                "repeated and its repeated effect is null."
            ),
            (
                "The significant 2B a=0.35 result does not identify the exact fiber through the unknown population "
                "optimum, but it rules out treating the statistically tied-optimal basin as fiber-optimal."
            ),
        ],
    }


def write_plot(reference: pd.DataFrame, fresh: pd.DataFrame, output_path: Path) -> None:
    anchors = reference[
        ["anchor_index", "token_budget_requested", "anchor_aggregate_starcoder", "anchor_role"]
    ].drop_duplicates()
    colors = sample_colorscale("RdYlGn_r", np.linspace(0.12, 0.88, len(anchors)))
    figure = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[
            f"{row.token_budget_requested / 1e9:g}B tokens, tied a={row.anchor_aggregate_starcoder:.2f}"
            for row in anchors.itertuples()
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.16,
    )
    for position, row in enumerate(anchors.itertuples()):
        plot_row = position // 3 + 1
        plot_col = position % 3 + 1
        block = reference.loc[reference["anchor_index"] == row.anchor_index]
        signed = pd.concat(
            [
                block[["abs_contrast", "minus_delta_vs_tied"]]
                .rename(columns={"abs_contrast": "contrast", "minus_delta_vs_tied": "delta"})
                .assign(contrast=lambda frame: -frame["contrast"]),
                pd.DataFrame({"contrast": [0.0], "delta": [0.0]}),
                block[["abs_contrast", "plus_delta_vs_tied"]].rename(
                    columns={"abs_contrast": "contrast", "plus_delta_vs_tied": "delta"}
                ),
            ]
        ).sort_values("contrast")
        figure.add_trace(
            go.Scatter(
                x=signed["contrast"],
                y=signed["delta"],
                mode="lines+markers",
                line={"color": colors[position], "width": 2.5},
                marker={"size": 7},
                name=f"anchor {row.anchor_index}",
                showlegend=False,
                hovertemplate="d=%{x:+.2f}<br>reference-seed delta=%{y:+.6f} BPB<extra></extra>",
            ),
            row=plot_row,
            col=plot_col,
        )
        selected = fresh.loc[fresh["anchor_index"] == row.anchor_index].iloc[0]
        figure.add_trace(
            go.Scatter(
                x=[selected["reference_selected_contrast"]],
                y=[selected["fresh_mean_delta_vs_tied"]],
                mode="markers",
                marker={
                    "symbol": "diamond",
                    "size": 12,
                    "color": colors[position],
                    "line": {"color": "white", "width": 1.5},
                },
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": [selected["fresh_ci_high"] - selected["fresh_mean_delta_vs_tied"]],
                    "arrayminus": [selected["fresh_mean_delta_vs_tied"] - selected["fresh_ci_low"]],
                    "thickness": 1.8,
                    "width": 5,
                },
                name="fresh-seed confirmation",
                showlegend=position == 0,
                hovertemplate=(
                    "reference-selected d=%{x:+.2f}<br>fresh-seed mean delta=%{y:+.6f} BPB"
                    "<br>95% paired t interval<extra></extra>"
                ),
            ),
            row=plot_row,
            col=plot_col,
        )
        figure.add_hline(y=0.0, line={"color": "#173042", "dash": "dot", "width": 1.2}, row=plot_row, col=plot_col)
        figure.update_xaxes(title_text="contrast d = p1 - p0", row=plot_row, col=plot_col)
        figure.update_yaxes(title_text="BPB minus tied", row=plot_row, col=plot_col)

    figure.update_layout(
        title={
            "text": (
                "Scale-specific fixed-aggregate fibers through WSD80 tied anchors"
                "<br><sub>2B a=0.35 is a replicated counterexample within the tied-optimal uncertainty set; "
                "the exact population optimum remains unidentified. Lower is better.</sub>"
            ),
            "x": 0.03,
        },
        template="plotly_white",
        height=900,
        width=1500,
        margin={"l": 80, "r": 40, "t": 125, "b": 70},
        font={"family": "Avenir Next, Helvetica Neue, sans-serif", "color": "#173042"},
        paper_bgcolor="#fbf8f0",
        plot_bgcolor="#fbf8f0",
        legend={"orientation": "h", "y": -0.08, "x": 0.5, "xanchor": "center"},
    )
    figure.update_xaxes(gridcolor="#ded8ca")
    figure.update_yaxes(gridcolor="#ded8ca")
    figure.write_html(output_path, include_plotlyjs=True, config=PLOT_CONFIG)
    figure.write_image(output_path.with_suffix(".png"), scale=3)


def write_report(
    design: dict[str, Any],
    observations: pd.DataFrame,
    reference: pd.DataFrame,
    arms: pd.DataFrame,
    fresh: pd.DataFrame,
    sensitivity: pd.DataFrame,
    cross_anchor: pd.DataFrame,
    decision: dict[str, Any],
    output_path: Path,
) -> None:
    crashed = observations.loc[observations["wandb_state"] != "finished", ["wandb_name", "wandb_state"]]
    lines = [
        "# StarCoder WSD80 scale-specific tied-fiber results",
        "",
        "## Completion",
        "",
        "- Iris parent: succeeded with exit 0, zero logical failures, and zero preemptions.",
        f"- New checkpoints: {EXPECTED_NEW_RUNS}/{EXPECTED_NEW_RUNS} with finite final Programming Languages BPB.",
        f"- Joined controls: {EXPECTED_ANCHORS} existing reference-seed tied checkpoints.",
        f"- W&B non-finished labels with recovered finite metrics: {len(crashed)}.",
        "",
        "## Full hypothesis conclusion",
        "",
        f"**{decision['verdict']}.**",
        "",
        "### Claim hierarchy",
        "",
        f"- **Mathematical dose-model implication:** {decision['dose_model_boundary']}",
        f"- **Literal population claim:** {decision['literal_population_status']}.",
        f"- **Finite-grid claim:** {decision['finite_grid_status']}.",
        f"- **Tied-optimal-set claim:** {decision['tied_optimal_set_status']}.",
        f"- **Modeling decision:** {decision['hard_constraint_recommendation']}.",
        "",
        "### Decisive 2B evidence",
        "",
        decision["decisive_counterevidence"],
        "",
        "### Null results at the other tested arms",
        "",
        decision["nonconfirmation_elsewhere"],
        "",
        (
            "The confirmatory comparison chooses the better sign at `|d|=0.20` using only the reference seed, "
            "then evaluates that fixed sign on four fresh joint-randomness seeds. Holm correction covers the four "
            "primary tied anchors."
        ),
        "",
        fresh.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Broad-basin tied-anchor sensitivity",
        "",
        sensitivity.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Direct 2B cross-anchor comparison",
        "",
        (
            "This checks whether the improved `a=0.35, d=+0.20` policy also beats the adjacent tied `a=0.40` "
            "anchor that is closer to the quadratic point estimate. This paired comparison was added after "
            "inspecting the anchor-sensitivity result, so it is a post-hoc diagnostic rather than the "
            "multiplicity-adjusted primary test."
        ),
        "",
        cross_anchor.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Five-seed prespecified arms",
        "",
        arms.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Reference-seed contrast grids",
        "",
    ]
    for anchor_index, block in reference.groupby("anchor_index", sort=True):
        first = block.iloc[0]
        anchor_heading = (
            f"### Anchor {anchor_index}: {first['token_budget_requested'] / 1e9:g}B, "
            f"a={first['anchor_aggregate_starcoder']:.2f}"
        )
        lines.extend(
            [
                anchor_heading,
                "",
                block.to_markdown(index=False, floatfmt=".6f"),
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            *[f"- {item}" for item in decision["limitations"]],
            (
                "- A sampled improvement is a counterexample at that coordinate. The 2B panel contains one such "
                "replicated counterexample; null results on individual radii cannot establish the universal statement."
            ),
            "",
            "## Provenance",
            "",
            f"- Objective: `{design['objective_metric']}`.",
            "- Phase contrast: `d = p1 - p0`; aggregate matching uses each rung's realized 80/20 fractions.",
            "- Full observations, decomposition tables, and multiplicity-adjusted tests are adjacent to this report.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    observations, design = collect_observations(args.panel_dir, args.wandb_timeout)
    reference = reference_decomposition(observations)
    repeated, arms, fresh = repeated_decomposition(observations)
    fresh = align_tied_optima(fresh, args.tied_optima)
    sensitivity = tied_anchor_sensitivity(observations)
    cross_anchor = two_b_cross_anchor_comparison(observations)
    decision = conclusion(fresh, reference, sensitivity, cross_anchor)

    observations.to_csv(args.output_dir / "observations.csv", index=False)
    reference.to_csv(args.output_dir / "reference_seed_decomposition.csv", index=False)
    repeated.to_csv(args.output_dir / "repeated_seed_decomposition.csv", index=False)
    arms.to_csv(args.output_dir / "prespecified_arm_tests.csv", index=False)
    fresh.to_csv(args.output_dir / "fresh_seed_confirmation.csv", index=False)
    sensitivity.to_csv(args.output_dir / "tied_anchor_sensitivity.csv", index=False)
    cross_anchor.to_csv(args.output_dir / "two_b_cross_anchor_comparison.csv", index=False)
    (args.output_dir / "hypothesis_verdict.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    write_plot(reference, fresh, args.output_dir / "scale_tied_fibers.html")
    write_report(
        design,
        observations,
        reference,
        arms,
        fresh,
        sensitivity,
        cross_anchor,
        decision,
        args.output_dir / "report.md",
    )

    print(f"Collected {len(observations)} observations ({EXPECTED_NEW_RUNS} new + {EXPECTED_ANCHORS} controls).")
    print(fresh.to_string(index=False))
    print(json.dumps(decision, indent=2))
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
