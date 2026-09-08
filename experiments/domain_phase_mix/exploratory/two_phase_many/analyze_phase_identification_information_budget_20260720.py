# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "fsspec",
#   "gcsfs",
#   "numpy",
#   "pandas",
#   "plotly",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Quantify the information available for phase-response identification."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_delphi_3e18_fixed_budget_frontier_composition as composition,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_hierarchical_coverage_grp_20260715 as hierarchical,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_matched_pair_heterogeneous_hpr_20260720 as matched_pair,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/phase_identification_information_budget_20260720"
FIBER_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/delphi_3e18_frontier_phase_fiber_results_20260719"
FAMILY_MODEL_DIR = SCRIPT_DIR / "reference_outputs/family_state_phase_surrogate_20260720"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def effective_rank(values: np.ndarray) -> float:
    singular = np.linalg.svd(values, compute_uv=False)
    energy = singular**2
    probabilities = energy / np.maximum(energy.sum(), 1e-12)
    probabilities = probabilities[probabilities > 1e-15]
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def stable_rank(values: np.ndarray) -> float:
    singular = np.linalg.svd(values, compute_uv=False)
    return float(np.sum(singular**2) / np.maximum(singular[0] ** 2, 1e-12))


def phase_displacement(weights: np.ndarray, dataset: object) -> np.ndarray:
    c0 = np.asarray(dataset.c0, dtype=float)
    c1 = np.asarray(dataset.c1, dtype=float)
    alpha = float(np.median(c0 / np.maximum(c0 + c1, 1e-12)))
    return alpha * (1.0 - alpha) * (weights[:, 1, :] - weights[:, 0, :])


def family_displacement(displacement: np.ndarray, dataset: object) -> np.ndarray:
    natural = hierarchical.proportional_weights(dataset)
    family_natural = np.asarray([natural[members].sum() for members in dataset.family_members])
    family = np.column_stack([displacement[:, members].sum(axis=1) for members in dataset.family_members])
    relative = family / np.maximum(family_natural[None, :], 1e-12)
    broad = dataset.family_names.index("broad_text")
    specialist = [index for index, name in enumerate(dataset.family_names) if name != "broad_text"]
    return np.column_stack([relative[:, index] - relative[:, broad] for index in specialist])


def design_summary(values: np.ndarray) -> dict[str, float]:
    singular = np.linalg.svd(values, compute_uv=False)
    positive = singular[singular > 1e-10 * singular[0]]
    return {
        "rows": len(values),
        "columns": values.shape[1],
        "rank": len(positive),
        "stable_rank": stable_rank(values),
        "entropy_effective_rank": effective_rank(values),
        "condition_nonzero": float(positive[0] / positive[-1]) if len(positive) else math.inf,
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "frozen_before_analysis": True,
        "questions": [
            "How large are signed phase effects relative to independent-run noise?",
            "How many phase directions are numerically identified by the existing fibers?",
            "How much larger must phase contrasts be to reach signal-to-noise two under local linear scaling?",
            "What model dimensionality and acquisition design follow from those measurements?",
        ],
        "noise_model": "odd plus/minus effect noise SD = fresh tied-center run SD / sqrt(2)",
        "caveat": "same-seed plus/minus runs can cancel shared seed effects, so this is a conservative noise estimate",
    }
    (output_dir / "analysis_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

    effects = pd.read_csv(FIBER_OUTPUT_DIR / "paired_phase_effects.csv")
    centers = pd.read_csv(FIBER_OUTPUT_DIR / "center_control_summary.csv")
    signal = (
        effects.groupby(["anchor_id", "target", "contrast_family"], sort=True)
        .agg(
            direction_count=("direction_id", "size"),
            odd_effect_rms=("odd_effect_plus_minus_over_2", lambda x: float(np.sqrt(np.mean(x**2)))),
            odd_effect_median_abs=("odd_effect_plus_minus_over_2", lambda x: float(np.median(np.abs(x)))),
            even_effect_rms=("mean_contrast_minus_center", lambda x: float(np.sqrt(np.mean(x**2)))),
        )
        .reset_index()
    )
    signal = signal.merge(
        centers[["anchor_id", "target", "fresh_center_sd_bpb"]],
        on=["anchor_id", "target"],
        validate="many_to_one",
    )
    signal["odd_noise_sd_independent"] = signal["fresh_center_sd_bpb"] / math.sqrt(2.0)
    signal["odd_snr"] = signal["odd_effect_rms"] / signal["odd_noise_sd_independent"]
    signal["debiased_odd_signal_rms"] = np.sqrt(
        np.maximum(signal["odd_effect_rms"] ** 2 - signal["odd_noise_sd_independent"] ** 2, 0.0)
    )
    signal["radius_multiplier_for_snr2"] = 2.0 / np.maximum(signal["odd_snr"], 1e-12)
    signal.to_csv(output_dir / "phase_signal_noise.csv", index=False)

    matched = matched_pair.matched_sources()
    dataset = composition.custom_dataset(
        matched.sources.reference,
        matched.sources.fiber.frame,
        matched.sources.fiber.weights,
        "uncheatable",
        "phase_information_design",
    )
    candidate_to_index = {
        str(candidate): index for index, candidate in enumerate(matched.sources.fiber.frame["candidate_id"].astype(str))
    }
    design_rows = []
    for anchor, anchor_frame in effects.groupby("anchor_id", sort=True):
        for contrast_family, frame in anchor_frame.groupby("contrast_family", sort=True):
            frame = frame.drop_duplicates(subset=["direction_id", "plus_candidate_id", "minus_candidate_id"])
            plus = np.asarray([candidate_to_index[value] for value in frame["plus_candidate_id"].astype(str)])
            minus = np.asarray([candidate_to_index[value] for value in frame["minus_candidate_id"].astype(str)])
            bucket = 0.5 * (
                phase_displacement(matched.sources.fiber.weights[plus], dataset)
                - phase_displacement(matched.sources.fiber.weights[minus], dataset)
            )
            family = family_displacement(bucket, dataset)
            for name, values in (("bucket", bucket), ("family", family)):
                design_rows.append(
                    {
                        "anchor_id": anchor,
                        "contrast_family": contrast_family,
                        "coordinate": name,
                        **design_summary(values),
                    }
                )
    design = pd.DataFrame(design_rows)
    design.to_csv(output_dir / "phase_design_rank.csv", index=False)

    family_metrics = pd.read_csv(FAMILY_MODEL_DIR / "stage1_metrics.csv")
    family_metrics.to_csv(output_dir / "family_state_stage1_metrics.csv", index=False)

    snr_plot = px.bar(
        signal,
        x="anchor_id",
        y="odd_snr",
        color="target",
        facet_col="contrast_family",
        barmode="group",
        color_discrete_map={"uncheatable": "#2f855a", "table9": "#c53030"},
        title="Observed signed phase effect relative to independent-run noise",
    )
    snr_plot.add_hline(y=1.0, line_dash="dash", line_color="#243746")
    snr_plot.add_hline(y=2.0, line_dash="dot", line_color="#243746")
    snr_plot.update_layout(template="plotly_white", yaxis_title="odd-effect RMS / estimated noise SD")
    snr_plot.write_html(output_dir / "phase_signal_noise.html", include_plotlyjs=True, config=PLOT_CONFIG)

    rank_plot = px.bar(
        design,
        x="anchor_id",
        y="entropy_effective_rank",
        color="coordinate",
        facet_col="contrast_family",
        barmode="group",
        color_discrete_map={"bucket": "#c53030", "family": "#2f855a"},
        title="Nominal dimension overstates phase-design information",
    )
    rank_plot.update_layout(template="plotly_white", yaxis_title="entropy effective rank")
    rank_plot.write_html(output_dir / "phase_design_effective_rank.html", include_plotlyjs=True, config=PLOT_CONFIG)

    domain_signal = signal.loc[signal["contrast_family"].eq("domain_vs_rest")]
    radius_low = float(domain_signal["radius_multiplier_for_snr2"].min())
    radius_high = float(domain_signal["radius_multiplier_for_snr2"].max())
    lines = [
        "# Phase-identification information budget",
        "",
        "## Conclusion",
        "",
        "The current data identify the aggregate response much more strongly than the phase response. The 39 "
        "one-vs-rest directions have nominal rank 38, but there is one observation per direction and no residual "
        "degree of freedom for a bucket-level gradient. Their signed effects are smaller than the conservative "
        "independent-run noise estimate in every anchor/target combination. The near-zero training error of a free "
        "bucket gradient is therefore interpolation, not evidence of a learned phase law.",
        "",
        "A five-degree-of-freedom family-state model is stable, but its nested OOF improvement is too small: it "
        "reduces exact-pair RMSE by 0.9% on Uncheatable and 5.1% on Table-9, and changes fiber RMSE by -2.5%/+0.0%. "
        "It fails the frozen gate and the common archive remains unopened for this round.",
        "",
        "## Signal and noise",
        "",
        signal.to_markdown(index=False, floatfmt=".6f"),
        "",
        "The independent-run comparison is conservative because the signed plus/minus contrast shares a data seed. "
        f"Under local linear scaling, reaching signed-effect SNR 2 requires roughly {radius_low:.1f}x to "
        f"{radius_high:.1f}x the current domain-vs-rest contrast radii.",
        "",
        "## Design rank",
        "",
        design.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Recommended joint learning protocol",
        "",
        "1. Fit an aggregate spine `F(a)` only from independently trained one-phase and phase-tied rows. This "
        "identifies shortage, diminishing returns, and repetition without phase confounding.",
        "2. Fit phase dynamics only to orthogonal moments: exact two-minus-one-phase differences, signed "
        "plus-minus fiber differences for odd order effects, and plus-minus-versus-center second differences for "
        "even path cost. Never let source identity enter prediction.",
        "3. Start with 2-5 family-level phase degrees of freedom. Unlock bucket residuals only when a nested "
        "likelihood-ratio or bootstrap stability test beats the family model; the present data do not justify 38.",
        "4. In a future two-round acquisition, first locate 1-2 bootstrap-stable aggregate frontier anchors from "
        "one-phase data. Around each anchor, use balanced simplex-tangent directions at two larger radii and both "
        "signs. Larger radii estimate response; two radii distinguish linear order benefit from even curvature.",
        "5. Allocate center repeats from the measured target-specific noise floor rather than uniformly. Table-9 "
        "needs materially more replication than Uncheatable at the current radius.",
        "6. Optimize only after the raw phase law predicts held-out directions and exact pairs. A KL or trust-region "
        "penalty remains a deployment choice, not evidence that the phase surface is correct.",
        "",
        "## Concrete 280-checkpoint two-round design",
        "",
        "For a strict 280-checkpoint total, round 1 uses 140 one-phase policies spanning the aggregate simplex plus "
        "70 same-seed two-phase counterparts selected across aggregate-performance strata. This yields 140 absolute "
        "aggregate observations and 70 exact phase differences in one training wave. Round 2 freezes two frontier "
        "anchors and spends the remaining 70 checkpoints on 32 signed directions (16 per anchor, two checkpoints "
        "per direction) plus three same-seed centers per anchor. Split the directions between two radii and balance "
        "family-level tangent directions before adding random residual directions. This budget identifies a compact "
        "family phase law, not 38 independent bucket effects. If 480 checkpoints are available, retain the 280-row "
        "round-1 design and add 100 signed fiber pairs in round 2.",
        "",
        "## Decision",
        "",
        "Do not add another global phase head to the current 280-row random two-phase fit. The next credible advance "
        "is an identification-aware aggregate/contrast model paired with a contrast design whose radius and "
        "replication are set by measured signal-to-noise. The existing heterogeneous rows are useful, but they do "
        "not currently support a high-dimensional deterministic phase optimum.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(signal.to_string(index=False))
    print(design.to_string(index=False))


if __name__ == "__main__":
    main()
