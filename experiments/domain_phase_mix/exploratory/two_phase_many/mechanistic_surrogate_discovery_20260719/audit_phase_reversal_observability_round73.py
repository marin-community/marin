# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy>=1.7",
#   "fsspec>=2025.7",
#   "numpy>=2.0",
#   "pandas>=2.2",
#   "plotly>=6.0",
#   "scikit-learn>=1.6",
#   "scipy>=1.15",
#   "tabulate>=0.9",
# ]
# ///
"""Audit whether the observed designs identify odd and even phase effects."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    export_mixture_fit_observatory as observatory,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    starcoder_refined_data,
)

SCRIPT_DIR = Path(__file__).resolve().parent
TWO_PHASE_ROOT = SCRIPT_DIR.parent
OUTPUT_ROOT = TWO_PHASE_ROOT / "reference_outputs/mechanistic_surrogate_discovery_20260719"
ROUND_DIR = OUTPUT_ROOT / "round73_phase_reversal_observability"
DELPHI_SWARM = TWO_PHASE_ROOT / (
    "reference_outputs/delphi_augmented_swarm_3e18_20260714/delphi_augmented_swarm_3e18_wide.csv"
)
KEY_DECIMALS = 10
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}, "responsive": True}


def coordinate_key(p0: float, p1: float) -> tuple[float, float]:
    return round(float(p0), KEY_DECIMALS), round(float(p1), KEY_DECIMALS)


def starcoder_reversal_rows(dataset: Any, phase0_fraction: float) -> tuple[pd.DataFrame, dict[str, object]]:
    p0 = np.asarray(dataset.weights[:, 0, 1], dtype=float)
    p1 = np.asarray(dataset.weights[:, 1, 1], dtype=float)
    loss = np.asarray(dataset.y, dtype=float)
    phase1_fraction = 1.0 - phase0_fraction
    lookup = {coordinate_key(x, y): index for index, (x, y) in enumerate(zip(p0, p1, strict=True))}
    rows: list[dict[str, object]] = []
    reflected_matches = 0
    for index, (early, late) in enumerate(zip(p0, p1, strict=True)):
        contrast = late - early
        if abs(contrast) < 1e-10:
            continue
        aggregate = phase0_fraction * early + phase1_fraction * late
        reflected_early = aggregate + phase1_fraction * contrast
        reflected_late = aggregate - phase0_fraction * contrast
        reflected_index = lookup.get(coordinate_key(reflected_early, reflected_late))
        if reflected_index is None:
            continue
        reflected_matches += 1
        if index >= reflected_index:
            continue
        tied_index = lookup.get(coordinate_key(aggregate, aggregate))
        odd_effect = 0.5 * (loss[index] - loss[reflected_index])
        even_effect = (
            0.5 * (loss[index] + loss[reflected_index]) - loss[tied_index] if tied_index is not None else np.nan
        )
        rows.append(
            {
                "surface": dataset.name,
                "index": index,
                "reflected_index": reflected_index,
                "tied_index": tied_index if tied_index is not None else -1,
                "aggregate_rare_share": aggregate,
                "absolute_contrast": abs(contrast),
                "early_rare_share": early,
                "late_rare_share": late,
                "reflected_early_rare_share": reflected_early,
                "reflected_late_rare_share": reflected_late,
                "loss": loss[index],
                "reflected_loss": loss[reflected_index],
                "tied_loss": loss[tied_index] if tied_index is not None else np.nan,
                "odd_order_effect_bpb": odd_effect,
                "even_variation_effect_bpb": even_effect,
            }
        )
    frame = pd.DataFrame(rows)
    finite_even = frame["even_variation_effect_bpb"].notna() if len(frame) else pd.Series(dtype=bool)
    summary = {
        "surface": dataset.name,
        "phase0_fraction": phase0_fraction,
        "coordinate_count": len(loss),
        "directed_reflected_coordinates": reflected_matches,
        "unique_reflection_pairs": len(frame),
        "pairs_with_exact_tied_anchor": int(finite_even.sum()) if len(frame) else 0,
        "mean_absolute_odd_effect_bpb": float(frame["odd_order_effect_bpb"].abs().mean()) if len(frame) else np.nan,
        "mean_absolute_even_effect_bpb": (
            float(frame.loc[finite_even, "even_variation_effect_bpb"].abs().mean()) if finite_even.any() else np.nan
        ),
        "median_absolute_odd_effect_bpb": (
            float(frame["odd_order_effect_bpb"].abs().median()) if len(frame) else np.nan
        ),
        "median_absolute_even_effect_bpb": (
            float(frame.loc[finite_even, "even_variation_effect_bpb"].abs().median()) if finite_even.any() else np.nan
        ),
    }
    return frame, summary


def delphi_domains(frame: pd.DataFrame) -> list[str]:
    return [
        column.removeprefix("phase_0_")
        for column in frame.columns
        if column.startswith("phase_0_") and f"phase_1_{column.removeprefix('phase_0_')}" in frame
    ]


def delphi_reversal_support(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    domain_names = delphi_domains(frame)
    phase0 = frame[[f"phase_0_{domain}" for domain in domain_names]].to_numpy(dtype=float)
    phase1 = frame[[f"phase_1_{domain}" for domain in domain_names]].to_numpy(dtype=float)
    gamma0 = float(frame["phase_0_fraction"].median())
    if not np.allclose(frame["phase_0_fraction"], gamma0, atol=1e-12):
        raise ValueError("Delphi fit swarm does not have a constant phase fraction")
    gamma1 = 1.0 - gamma0
    aggregate = gamma0 * phase0 + gamma1 * phase1
    contrast = phase1 - phase0
    tied = np.max(np.abs(contrast), axis=1) < 1e-10
    records: list[dict[str, object]] = []
    for index in range(len(frame)):
        if tied[index]:
            continue
        reflected_phase0 = aggregate[index] + gamma1 * contrast[index]
        reflected_phase1 = aggregate[index] - gamma0 * contrast[index]
        feasible = bool(
            np.all(reflected_phase0 >= -1e-12)
            and np.all(reflected_phase1 >= -1e-12)
            and np.all(reflected_phase0 <= 1.0 + 1e-12)
            and np.all(reflected_phase1 <= 1.0 + 1e-12)
        )
        negativity_mass = float(np.maximum(-reflected_phase0, 0.0).sum() + np.maximum(-reflected_phase1, 0.0).sum())
        policy_distance = 0.25 * (
            np.abs(phase0 - reflected_phase0).sum(axis=1) + np.abs(phase1 - reflected_phase1).sum(axis=1)
        )
        policy_distance[index] = np.inf
        nearest_index = int(np.argmin(policy_distance))
        aggregate_tv = 0.5 * np.abs(aggregate[nearest_index] - aggregate[index]).sum()
        contrast_reversal_mismatch = 0.25 * np.abs(contrast[nearest_index] + contrast[index]).sum()
        records.append(
            {
                "index": index,
                "run_name": frame.iloc[index]["run_name"],
                "reflected_policy_feasible": feasible,
                "reflected_policy_negativity_mass": negativity_mass,
                "phase_tv": 0.5 * np.abs(contrast[index]).sum(),
                "nearest_reflected_index": nearest_index,
                "nearest_reflected_run_name": frame.iloc[nearest_index]["run_name"],
                "nearest_reflected_policy_distance": float(policy_distance[nearest_index]),
                "nearest_aggregate_tv": float(aggregate_tv),
                "nearest_contrast_reversal_mismatch": float(contrast_reversal_mismatch),
            }
        )
    result = pd.DataFrame(records)
    feasible = result["reflected_policy_feasible"]
    distances = result["nearest_reflected_policy_distance"]
    summary = {
        "surface": "delphi_3e18_39_bucket_fit_swarm",
        "phase0_fraction": gamma0,
        "coordinate_count": len(frame),
        "non_tied_coordinate_count": len(result),
        "feasible_reflection_count": int(feasible.sum()),
        "exact_reflection_count_at_1e_8": int(((distances < 1e-8) & feasible).sum()),
        "median_reflected_policy_negativity_mass": float(result["reflected_policy_negativity_mass"].median()),
        "nearest_reflection_distance_p10": float(distances.quantile(0.10)),
        "nearest_reflection_distance_median": float(distances.median()),
        "nearest_reflection_distance_p90": float(distances.quantile(0.90)),
        "nearest_reflection_distance_max": float(distances.max()),
    }
    return result, summary


def main() -> None:
    ROUND_DIR.mkdir(parents=True, exist_ok=True)
    cosine = observatory.load_cosine_starcoder()
    wsd = starcoder_refined_data.load_refined_wsd80_starcoder(cosine)
    starcoder_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for dataset, phase0_fraction in ((cosine, 0.5), (wsd, 0.8)):
        rows, summary = starcoder_reversal_rows(dataset, phase0_fraction)
        starcoder_frames.append(rows)
        summaries.append(summary)
    starcoder = pd.concat(starcoder_frames, ignore_index=True)
    delphi, delphi_summary = delphi_reversal_support(pd.read_csv(DELPHI_SWARM))
    summaries.append(delphi_summary)
    summary_frame = pd.DataFrame(summaries)
    starcoder.to_csv(ROUND_DIR / "starcoder_exact_phase_reversals.csv", index=False)
    delphi.to_csv(ROUND_DIR / "delphi_nearest_phase_reversals.csv", index=False)
    summary_frame.to_csv(ROUND_DIR / "phase_reversal_observability_summary.csv", index=False)

    finite = starcoder.dropna(subset=["even_variation_effect_bpb"])
    figure = px.scatter(
        finite,
        x="odd_order_effect_bpb",
        y="even_variation_effect_bpb",
        color="surface",
        size="absolute_contrast",
        hover_data=["aggregate_rare_share", "early_rare_share", "late_rare_share"],
        title="Exact phase reversals separate odd order effects from even variation effects",
        labels={
            "odd_order_effect_bpb": "Odd order component (BPB)",
            "even_variation_effect_bpb": "Even phase-variation component (BPB)",
        },
    )
    figure.add_hline(y=0.0, line_dash="dash", line_color="#4d5963")
    figure.add_vline(x=0.0, line_dash="dash", line_color="#4d5963")
    figure.update_layout(template="plotly_white", height=560, width=980)
    figure.write_html(ROUND_DIR / "starcoder_odd_even_phase_effects.html", include_plotlyjs="cdn", config=PLOT_CONFIG)

    histogram = px.histogram(
        delphi,
        x="nearest_reflected_policy_distance",
        color="reflected_policy_feasible",
        nbins=30,
        title="The random 39-bucket swarm does not contain reflected phase contrasts",
        labels={"nearest_reflected_policy_distance": "Nearest distance to ideal reflected policy"},
    )
    histogram.update_traces(marker_color="#d95f02")
    histogram.update_layout(template="plotly_white", height=500, width=900)
    histogram.write_html(ROUND_DIR / "delphi_reflection_support.html", include_plotlyjs="cdn", config=PLOT_CONFIG)

    report = "\n".join(
        [
            "# Round 73: phase-reversal observability audit",
            "",
            "This is an invariant and design audit, not a candidate surrogate. It reads the two observed StarCoder surfaces and the 280 Delphi fit-swarm coordinates. It reads no historical, exposed adversarial, or sealed-confirmation target.",
            "",
            "For phase fractions gamma0 and gamma1, define aggregate a = gamma0 w0 + gamma1 w1 and contrast d = w1 - w0. The unique contrast-reversed policy at fixed aggregate is w0' = a + gamma1 d and w1' = a - gamma0 d. Given both policies and a tied observation at a, the loss decomposes into an odd order component [L(a,d)-L(a,-d)]/2 and an even phase-variation component [L(a,d)+L(a,-d)]/2-L(a,0).",
            "",
            "## Observability",
            "",
            summary_frame.to_markdown(index=False, floatfmt=".6f"),
            "",
            "## Interpretation",
            "",
            "An exact StarCoder reversal triple can identify two physically distinct effects without fitting a response law: an odd component that changes sign under schedule reversal and an even component that prices phase variation irrespective of order. The sole cosine triple has material values of both signs, but one triple is not evidence for a transferable response law. The WSD surface contains no exact triple.",
            "",
            "The 39-bucket random two-phase design does not supply any feasible exact reflected contrast at fixed aggregate: reversing its large contrasts under an unequal-duration schedule drives at least one bucket weight below zero. Distances to the algebraic, possibly infeasible reflection quantify the gap. Consequently, the existing swarm cannot identify an odd/even phase decomposition without imposing a functional form that extrapolates across aggregate and contrast simultaneously. This is a design limitation, not evidence for a new model. The preregistered signed phase-fiber confirmation design uses small simplex-feasible +/-d rays and tied anchors to supply the missing intervention.",
        ]
    )
    (ROUND_DIR / "report.md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
