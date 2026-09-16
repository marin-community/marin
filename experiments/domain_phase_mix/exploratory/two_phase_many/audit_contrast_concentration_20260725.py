# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "plotly",
#   "scipy",
# ]
# ///
"""Does concentrating the phase contrast improve the two-phase response?

Motivation
----------
The StarCoder probe found that the odd (ordering) channel there is *sublinear* in
contrast radius, ``p = 0.64-0.81``, while the 39-bucket Delphi panel gives
``p = 1.66-1.85``. A small ``p`` means the ordering benefit arrives at tiny
contrast and saturates; a large ``p`` means nothing happens until the contrast is
large, by which point the symmetric cost has caught up.

StarCoder's contrast is concentrated on one specialist bucket. Delphi's balanced
partitions spread mass across roughly twenty buckets in each direction, where
per-bucket recency benefits can partly cancel. That suggests concentration, not
the phase split, is the lever.

Estimand
--------
Concentration is measured by the participation ratio

    PR(d) = (sum_i |d_i|)^2 / (sum_i d_i^2),

the effective number of participating buckets. At a fixed L1 radius
``rho = 0.5 sum_i |d_i|``, a smaller PR means a larger L2 norm, so this asks:
at fixed phase TV, does concentrating the contrast help or hurt?

In the paired panels PR and rho are close to orthogonal (correlation of logs
0.025), so concentration can be varied without moving the radius.

Decomposition
-------------
The paired panels observe only ``Delta = O + C``. Because the contrast cloud is
nearly sign symmetric, within a concentration bin the odd part largely averages
out while the even part does not, so

    mean(Delta) tracks the symmetric cost C,
    spread(Delta) tracks the ordering signal |O|.

Cloud symmetry is verified per bin and reported, since the decomposition rests on
it. The operationally decisive column is the best observed Delta in each bin:
negative means some policy in that bin beat its exact tied counterpart.

The sealed targeted-pairwise panel is never read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from phase_order_spine_20260725 import (
    AGGRESSIVE,
    REFERENCE_OUTPUTS,
    TARGETS,
    build_spine,
    provenance,
)
from plotly.subplots import make_subplots
from scipy.stats import pearsonr

DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "contrast_concentration_20260725"
SIXTY_M_DIR = REFERENCE_OUTPUTS / "60m_39bucket_checkpoint_audit_20260724"

TARGET_LABEL = {"uncheatable_bpb": "uncheatable", "table9_macro_bpb": "table9"}
RUN_SIGMA = {"uncheatable_bpb": 0.000963, "table9_macro_bpb": 0.003121}
CONCENTRATION_BINS = 3
RADIUS_BINS = 2
BOOTSTRAP_DRAWS = 4000
BOOTSTRAP_SEED = 20260725
# StarCoder's two-bucket contrast has PR = 2 by construction; the most
# concentrated 39-bucket contrast available locally is PR = 7.6.
STARCODER_PARTICIPATION_RATIO = 2.0


def participation_ratio(contrast: np.ndarray) -> np.ndarray:
    """Effective number of participating buckets, (L1)^2 / (L2)^2."""
    l1 = np.abs(contrast).sum(axis=1)
    l2 = (contrast**2).sum(axis=1)
    return l1**2 / np.clip(l2, 1e-300, None)


def quantile_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    labels = np.empty(len(values), dtype=int)
    labels[order] = np.floor(np.arange(len(values)) * n_bins / len(values)).astype(int)
    return labels


def cloud_symmetry(contrast: np.ndarray) -> float:
    """Norm of the mean unit contrast direction; 0 is sign symmetric, 1 is a cone."""
    norms = np.linalg.norm(contrast, axis=1, keepdims=True)
    return float(np.linalg.norm((contrast / np.clip(norms, 1e-300, None)).mean(axis=0)))


def concentration_table(
    panel_name: str,
    contrast: np.ndarray,
    deltas: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Location, spread, and best-case Delta by concentration, holding radius fixed."""
    ratio = participation_ratio(contrast)
    radius = 0.5 * np.abs(contrast).sum(axis=1)
    radius_bin = quantile_bins(radius, RADIUS_BINS)
    rows = []
    for target, delta in deltas.items():
        for r_bin in range(RADIUS_BINS):
            in_radius = radius_bin == r_bin
            # Bin concentration *within* a radius block so the comparison is at
            # matched radius rather than across it.
            local_ratio = ratio[in_radius]
            concentration_bin = quantile_bins(local_ratio, CONCENTRATION_BINS)
            for c_bin in range(CONCENTRATION_BINS):
                mask = np.flatnonzero(in_radius)[concentration_bin == c_bin]
                sample = delta[mask]
                if len(sample) < 8:
                    continue
                boot_sd = [
                    sample[rng.integers(0, len(sample), len(sample))].std(ddof=1) for _ in range(BOOTSTRAP_DRAWS // 20)
                ]
                rows.append(
                    {
                        "panel": panel_name,
                        "target": TARGET_LABEL[target],
                        "radius_bin": r_bin,
                        "concentration_bin": c_bin,
                        "concentration_label": (
                            ["most_concentrated", "middle", "most_diffuse"][c_bin]
                            if CONCENTRATION_BINS == 3
                            else str(c_bin)
                        ),
                        "n": len(sample),
                        "median_participation_ratio": float(np.median(ratio[mask])),
                        "median_phase_tv": float(np.median(radius[mask])),
                        "median_l2_norm": float(np.median(np.linalg.norm(contrast[mask], axis=1))),
                        "cloud_symmetry": cloud_symmetry(contrast[mask]),
                        "mean_delta_bpb": float(sample.mean()),
                        "sd_delta_bpb": float(sample.std(ddof=1)),
                        "sd_delta_ci95_low": float(np.quantile(boot_sd, 0.025)),
                        "sd_delta_ci95_high": float(np.quantile(boot_sd, 0.975)),
                        "best_delta_bpb": float(sample.min()),
                        "fraction_better_than_tied": float((sample < 0).mean()),
                    }
                )
    frame = pd.DataFrame(rows)
    # Concentration bin 0 holds the smallest participation ratio, i.e. the most
    # concentrated contrasts.
    return frame


def concentration_regression(
    panel_name: str, contrast: np.ndarray, deltas: dict[str, np.ndarray], rng: np.random.Generator
) -> pd.DataFrame:
    """Regress Delta location and spread on log concentration, controlling for log radius."""
    ratio = np.log(participation_ratio(contrast))
    radius = np.log(0.5 * np.abs(contrast).sum(axis=1))
    design = np.column_stack([np.ones_like(ratio), radius, ratio])
    rows = []
    for target, delta in deltas.items():
        coefficients, *_ = np.linalg.lstsq(design, delta, rcond=None)
        residual = delta - design @ coefficients
        # Scale model: |residual| on the same predictors. A negative concentration
        # coefficient means spread grows as the contrast becomes concentrated.
        scale_coefficients, *_ = np.linalg.lstsq(design, np.abs(residual), rcond=None)
        boot_location, boot_scale = [], []
        for _ in range(BOOTSTRAP_DRAWS // 4):
            pick = rng.integers(0, len(delta), len(delta))
            loc, *_ = np.linalg.lstsq(design[pick], delta[pick], rcond=None)
            res = delta[pick] - design[pick] @ loc
            sca, *_ = np.linalg.lstsq(design[pick], np.abs(res), rcond=None)
            boot_location.append(loc[2])
            boot_scale.append(sca[2])
        rows.append(
            {
                "panel": panel_name,
                "target": TARGET_LABEL[target],
                "n": len(delta),
                "location_log_radius": float(coefficients[1]),
                "location_log_participation_ratio": float(coefficients[2]),
                "location_ci95_low": float(np.quantile(boot_location, 0.025)),
                "location_ci95_high": float(np.quantile(boot_location, 0.975)),
                "scale_log_participation_ratio": float(scale_coefficients[2]),
                "scale_ci95_low": float(np.quantile(boot_scale, 0.025)),
                "scale_ci95_high": float(np.quantile(boot_scale, 0.975)),
                "probability_concentration_raises_spread": float(np.mean(np.asarray(boot_scale) < 0)),
                "probability_concentration_raises_cost": float(np.mean(np.asarray(boot_location) < 0)),
            }
        )
    return pd.DataFrame(rows)


def radius_metric_comparison(panel_name: str, contrast: np.ndarray, deltas: dict[str, np.ndarray]) -> pd.DataFrame:
    """Is the symmetric cost governed by the L1 radius or the L2 norm?

    At fixed L1 the two disagree exactly when the contrast is concentrated, so
    this identifies which norm the cost actually tracks.
    """
    l1 = 0.5 * np.abs(contrast).sum(axis=1)
    l2 = np.linalg.norm(contrast, axis=1)
    rows = []
    for target, delta in deltas.items():
        record = {"panel": panel_name, "target": TARGET_LABEL[target], "n": len(delta)}
        for name, predictor in (("l1_phase_tv", l1), ("l2_norm", l2)):
            design = np.column_stack([np.ones_like(predictor), np.log(predictor)])
            coefficients, *_ = np.linalg.lstsq(design, delta, rcond=None)
            fitted = design @ coefficients
            record[f"{name}_r2"] = 1.0 - float(((delta - fitted) ** 2).sum() / ((delta - delta.mean()) ** 2).sum())
            record[f"{name}_pearson"] = float(pearsonr(np.log(predictor), delta)[0])
        record["cost_tracks"] = "l2_norm" if record["l2_norm_r2"] > record["l1_phase_tv_r2"] else "l1_phase_tv"
        rows.append(record)
    return pd.DataFrame(rows)


def aggressive_family_comparison(spine, rng: np.random.Generator) -> pd.DataFrame:
    """Compare design families at matched target TV using same-seed control deltas.

    The handcrafted late-quality family is materially more concentrated than the
    balanced partitions, so this is a direct family-level test on the 39-bucket
    geometry with the deployment targets.
    """
    runs = pd.read_csv(AGGRESSIVE / "observed_results_with_control_deltas.csv")
    buckets = spine.delphi_3e18.buckets
    phase0 = runs[[f"phase_0_{b}" for b in buckets]].to_numpy(float)
    phase1 = runs[[f"phase_1_{b}" for b in buckets]].to_numpy(float)
    contrast = phase1 - phase0
    runs = runs.assign(
        participation_ratio=participation_ratio(contrast),
        realized_tv=0.5 * np.abs(contrast).sum(axis=1),
    )
    treated = runs[runs["contrast_family"] != "center_control"]
    rows = []
    for target, column in (
        ("uncheatable_bpb", "uncheatable_delta_vs_control"),
        ("table9_macro_bpb", "table9_delta_vs_control"),
    ):
        for tv in sorted(treated["target_phase_tv"].unique()):
            block = treated[(treated["target_phase_tv"] == tv) & treated[column].notna()]
            for family, group in block.groupby("contrast_family"):
                if len(group) < 4:
                    continue
                values = group[column].to_numpy(float)
                boot = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(BOOTSTRAP_DRAWS // 20)]
                rows.append(
                    {
                        "target": TARGET_LABEL[target],
                        "target_phase_tv": float(tv),
                        "contrast_family": family,
                        "n": len(values),
                        "median_participation_ratio": float(group["participation_ratio"].median()),
                        "mean_delta_vs_control_bpb": float(values.mean()),
                        "mean_ci95_low": float(np.quantile(boot, 0.025)),
                        "mean_ci95_high": float(np.quantile(boot, 0.975)),
                        "sd_delta_bpb": float(values.std(ddof=1)),
                        "best_delta_bpb": float(values.min()),
                        "fraction_better_than_control": float((values < 0).mean()),
                    }
                )
    return pd.DataFrame(rows)


def within_family_concentration(spine) -> pd.DataFrame:
    """Regress the control delta on concentration inside the handcrafted family.

    Within one family the direction-construction recipe is held fixed, so this is
    the cleanest available concentration contrast on the deployment targets.
    """
    runs = pd.read_csv(AGGRESSIVE / "observed_results_with_control_deltas.csv")
    buckets = spine.delphi_3e18.buckets
    contrast = runs[[f"phase_1_{b}" for b in buckets]].to_numpy(float) - runs[
        [f"phase_0_{b}" for b in buckets]
    ].to_numpy(float)
    runs = runs.assign(participation_ratio=participation_ratio(contrast), realized_tv=0.5 * np.abs(contrast).sum(axis=1))
    rows = []
    for family in ("handcrafted_late_quality", "balanced_partition"):
        block = runs[runs["contrast_family"] == family]
        for target, column in (
            ("uncheatable_bpb", "uncheatable_delta_vs_control"),
            ("table9_macro_bpb", "table9_delta_vs_control"),
        ):
            usable = block[block[column].notna()]
            if len(usable) < 8:
                continue
            ratio = np.log(usable["participation_ratio"].to_numpy(float))
            radius = np.log(usable["realized_tv"].to_numpy(float))
            delta = usable[column].to_numpy(float)
            design = np.column_stack([np.ones_like(ratio), radius, ratio])
            coefficients, *_ = np.linalg.lstsq(design, delta, rcond=None)
            fitted = design @ coefficients
            rows.append(
                {
                    "contrast_family": family,
                    "target": TARGET_LABEL[target],
                    "n": len(delta),
                    "participation_ratio_range": (
                        f"{usable['participation_ratio'].min():.2f}-{usable['participation_ratio'].max():.2f}"
                    ),
                    "location_log_participation_ratio": float(coefficients[2]),
                    "r2": 1.0 - float(((delta - fitted) ** 2).sum() / ((delta - delta.mean()) ** 2).sum()),
                    "pearson_delta_vs_log_pr": float(pearsonr(ratio, delta)[0]),
                    "pearson_pvalue": float(pearsonr(ratio, delta)[1]),
                }
            )
    return pd.DataFrame(rows)


def load_sixty_m() -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
    """60M paired panel: a third scale, Uncheatable only (tied Table-9 coverage is zero)."""
    two_phase = pd.read_csv(SIXTY_M_DIR / "fit_two_phase.csv")
    tied = pd.read_csv(SIXTY_M_DIR / "fit_single_phase.csv")
    p0_cols = [c for c in two_phase.columns if c.startswith("phase_0_")]
    p1_cols = [c for c in two_phase.columns if c.startswith("phase_1_")]
    matched = tied.set_index("run_name").reindex(two_phase["paired_run_name"])
    alpha = 0.80
    phase0 = two_phase[p0_cols].to_numpy(float)
    phase1 = two_phase[p1_cols].to_numpy(float)
    aggregate = alpha * phase0 + (1 - alpha) * phase1
    error = float(np.abs(aggregate - matched[p0_cols].to_numpy(float)).max())
    assert error < 1e-8, f"60M aggregate match error {error:.3e}"
    contrast = phase1 - phase0
    keep = (0.5 * np.abs(contrast).sum(axis=1) > 1e-9) & two_phase["uncheatable_bpb"].notna().to_numpy()
    keep &= matched["uncheatable_bpb"].notna().to_numpy()
    delta = two_phase["uncheatable_bpb"].to_numpy(float)[keep] - matched["uncheatable_bpb"].to_numpy(float)[keep]
    return contrast[keep], {"uncheatable_bpb": delta}


def plot_concentration(table: pd.DataFrame, path: Path) -> None:
    panels = sorted(table["panel"].unique())
    figure = make_subplots(
        rows=1,
        cols=len(panels),
        subplot_titles=[p for p in panels],
        shared_yaxes=False,
    )
    for column, panel in enumerate(panels, start=1):
        block = table[table["panel"] == panel]
        for (target, radius_bin), group in block.groupby(["target", "radius_bin"]):
            ordered = group.sort_values("median_participation_ratio")
            figure.add_trace(
                go.Scatter(
                    x=ordered["median_participation_ratio"],
                    y=ordered["sd_delta_bpb"],
                    error_y={
                        "type": "data",
                        "symmetric": False,
                        "array": ordered["sd_delta_ci95_high"] - ordered["sd_delta_bpb"],
                        "arrayminus": ordered["sd_delta_bpb"] - ordered["sd_delta_ci95_low"],
                    },
                    mode="lines+markers",
                    name=f"{target} radius bin {radius_bin}",
                    showlegend=column == 1,
                ),
                row=1,
                col=column,
            )
        figure.update_xaxes(title_text="participation ratio (lower = concentrated)", row=1, col=column)
        figure.update_yaxes(title_text="spread of paired Delta (BPB)", row=1, col=column)
    figure.update_layout(
        title="Ordering signal versus contrast concentration at matched radius",
        template="plotly_white",
        height=460,
    )
    figure.write_html(path, include_plotlyjs="cdn")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    spine = build_spine()
    panels: list[tuple[str, np.ndarray, dict[str, np.ndarray]]] = [
        ("delphi_3e18", spine.delphi_3e18.contrast, {t: spine.delphi_3e18.delta[t] for t in TARGETS}),
        ("300m", spine.m300.contrast, {t: spine.m300.delta[t] for t in TARGETS}),
    ]
    sixty = load_sixty_m()
    if sixty is not None:
        panels.append(("60m", sixty[0], sixty[1]))

    tables = pd.concat([concentration_table(n, c, d, rng) for n, c, d in panels], ignore_index=True)
    regressions = pd.concat([concentration_regression(n, c, d, rng) for n, c, d in panels], ignore_index=True)
    metrics = pd.concat([radius_metric_comparison(n, c, d) for n, c, d in panels], ignore_index=True)
    families = aggressive_family_comparison(spine, rng)
    within = within_family_concentration(spine)

    plot_concentration(tables, output / "concentration_signal.html")

    tables.to_csv(output / "concentration_bins.csv", index=False)
    regressions.to_csv(output / "concentration_regression.csv", index=False)
    metrics.to_csv(output / "radius_metric_comparison.csv", index=False)
    families.to_csv(output / "aggressive_family_comparison.csv", index=False)
    within.to_csv(output / "within_family_concentration.csv", index=False)

    reach = pd.DataFrame(
        [
            {
                "geometry": "starcoder two bucket",
                "participation_ratio": STARCODER_PARTICIPATION_RATIO,
                "note": "one specialist bucket; PR = 2 by construction",
            },
            {
                "geometry": "delphi handcrafted_late_quality",
                "participation_ratio": 7.64,
                "note": "most concentrated 39-bucket contrast available locally",
            },
            {
                "geometry": "delphi balanced_partition",
                "participation_ratio": 16.02,
                "note": "median of the antithetic design",
            },
            {
                "geometry": "delphi qsplit paired panel",
                "participation_ratio": 19.56,
                "note": "median of the 238-row fit panel",
            },
        ]
    )
    reach.to_csv(output / "concentration_reach.csv", index=False)

    protocol = {
        "estimand": "paired Delta = L(a,d) - L(a,0) versus participation ratio at matched radius",
        "participation_ratio": "(sum |d_i|)^2 / (sum d_i^2), the effective number of participating buckets",
        "concentration_bins": CONCENTRATION_BINS,
        "radius_bins": RADIUS_BINS,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "run_sigma_bpb": RUN_SIGMA,
        "decomposition_assumption": (
            "within a concentration bin the odd part largely averages out while the even part does not, "
            "so mean(Delta) tracks the cost and spread(Delta) tracks the ordering signal; "
            "cloud_symmetry reports the norm of the mean unit direction per bin"
        ),
        "sealed_targeted_pairwise_panel_accessed": False,
        "provenance_sha256": provenance(),
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")

    print("=== concentration bins at matched radius ===")
    print(
        tables[
            [
                "panel",
                "target",
                "radius_bin",
                "concentration_label",
                "n",
                "median_participation_ratio",
                "median_phase_tv",
                "cloud_symmetry",
                "mean_delta_bpb",
                "sd_delta_bpb",
                "best_delta_bpb",
                "fraction_better_than_tied",
            ]
        ].to_string(index=False)
    )
    print("\n=== regression of Delta on log concentration, controlling for log radius ===")
    print(regressions.to_string(index=False))
    print("\n=== does the cost track L1 or L2? ===")
    print(metrics.to_string(index=False))
    print("\n=== aggressive design families at matched target TV ===")
    print(families.to_string(index=False))
    print("\n=== within-family concentration slope ===")
    print(within.to_string(index=False))
    print("\n=== concentration reach ===")
    print(reach.to_string(index=False))


if __name__ == "__main__":
    main()
