# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2.0",
#   "pandas>=2.2",
#   "plotly>=6.0",
#   "scipy>=1.14",
#   "tabulate>=0.9",
# ]
# ///
"""Compare random two-phase sampling with a same-budget aggregate/fiber design.

This is a coordinate-only identification diagnostic. It never reads BPB values,
historical heldout outcomes, adversarial outcomes, or sealed confirmation data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.linalg import helmert

SCRIPT_DIR = Path(__file__).resolve().parent
TWO_PHASE_ROOT = SCRIPT_DIR.parent
INPUT = TWO_PHASE_ROOT / ("reference_outputs/delphi_augmented_swarm_3e18_20260714/delphi_augmented_swarm_3e18_wide.csv")
OUTPUT_DIR = TWO_PHASE_ROOT / (
    "reference_outputs/mechanistic_surrogate_discovery_20260719/round56_two_stage_design_identifiability"
)
RANDOM_SEED = 20260719
TIED_ROWS = 140
FIBER_PAIRS = 70
ANCHOR_COUNT = 3
PHASE0_FRACTION = 0.8
PHASE1_FRACTION = 0.2
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}, "responsive": True}


def domains(frame: pd.DataFrame) -> list[str]:
    return [
        column.removeprefix("phase_0_")
        for column in frame.columns
        if column.startswith("phase_0_") and f"phase_1_{column.removeprefix('phase_0_')}" in frame.columns
    ]


def stable_rank(singular_values: np.ndarray) -> float:
    squared = singular_values**2
    return float(squared.sum() ** 2 / np.square(squared).sum())


def standardized(block: np.ndarray) -> np.ndarray:
    centered = block - block.mean(axis=0, keepdims=True)
    scale = centered.std(axis=0, keepdims=True)
    if np.any(scale <= 1e-12):
        raise ValueError("Design block has a constant simplex coordinate")
    return centered / scale


def block_metrics(name: str, block: np.ndarray) -> tuple[dict[str, object], np.ndarray]:
    matrix = standardized(block[:, :-1])
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return (
        {
            "design": name,
            "block": "aggregate" if name.endswith("aggregate") else "contrast",
            "rows": len(matrix),
            "columns": matrix.shape[1],
            "numerical_rank": int(np.linalg.matrix_rank(matrix)),
            "stable_rank": stable_rank(singular_values),
            "condition_number": float(singular_values[0] / singular_values[-1]),
            "smallest_singular_value": float(singular_values[-1]),
            "largest_singular_value": float(singular_values[0]),
        },
        singular_values,
    )


def joint_metrics(design: str, aggregate: np.ndarray, contrast: np.ndarray) -> tuple[dict[str, object], np.ndarray]:
    aggregate_z = standardized(aggregate[:, :-1])
    contrast_z = standardized(contrast[:, :-1])
    qa = np.linalg.qr(aggregate_z, mode="reduced")[0]
    qd = np.linalg.qr(contrast_z, mode="reduced")[0]
    canonical = np.linalg.svd(qa.T @ qd, compute_uv=False)
    residual = contrast_z - aggregate_z @ np.linalg.lstsq(aggregate_z, contrast_z, rcond=None)[0]
    joint = np.concatenate([aggregate_z, contrast_z], axis=1)
    singular_values = np.linalg.svd(joint, compute_uv=False)
    return (
        {
            "design": design,
            "rows": len(joint),
            "columns": joint.shape[1],
            "numerical_rank": int(np.linalg.matrix_rank(joint)),
            "stable_rank": stable_rank(singular_values),
            "condition_number": float(singular_values[0] / singular_values[-1]),
            "mean_canonical_correlation": float(canonical.mean()),
            "max_canonical_correlation": float(canonical.max()),
            "canonical_correlations_above_0p75": int(np.sum(canonical > 0.75)),
            "contrast_energy_after_aggregate_residualization": float(
                np.square(residual).sum() / np.square(contrast_z).sum()
            ),
        },
        canonical,
    )


def farthest_point_indices(points: np.ndarray, count: int) -> list[int]:
    if count > len(points):
        raise ValueError("Cannot select more points than are available")
    clr = np.log(np.maximum(points, 1e-8))
    clr -= clr.mean(axis=1, keepdims=True)
    selected = [int(np.argmin(np.linalg.norm(clr - clr.mean(axis=0), axis=1)))]
    distance = np.linalg.norm(clr - clr[selected[0]], axis=1)
    while len(selected) < count:
        index = int(np.argmax(distance))
        selected.append(index)
        distance = np.minimum(distance, np.linalg.norm(clr - clr[index], axis=1))
    return selected


def anchor_indices(aggregate: np.ndarray) -> list[int]:
    clr = np.log(np.maximum(aggregate, 1e-8))
    clr -= clr.mean(axis=1, keepdims=True)
    center = int(np.argmin(np.linalg.norm(clr - clr.mean(axis=0), axis=1)))
    first = int(np.argmax(np.linalg.norm(clr - clr[center], axis=1)))
    distance_to_pair = np.minimum(
        np.linalg.norm(clr - clr[center], axis=1),
        np.linalg.norm(clr - clr[first], axis=1),
    )
    second = int(np.argmax(distance_to_pair))
    return [center, first, second]


def fiber_direction_matrix(domain_count: int, count: int) -> np.ndarray:
    rng = np.random.default_rng(RANDOM_SEED)
    tangent_basis = helmert(domain_count, full=False).T
    rotation, _ = np.linalg.qr(rng.normal(size=(domain_count - 1, domain_count - 1)))
    rotated = tangent_basis @ rotation
    directions = [rotated[:, index] for index in range(domain_count - 1)]
    while len(directions) < count:
        coefficients = rng.normal(size=domain_count - 1)
        vector = tangent_basis @ coefficients
        directions.append(vector / np.linalg.norm(vector))
    return np.stack(directions[:count])


def build_same_budget_design(
    aggregate: np.ndarray,
    alpha0: float,
    alpha1: float,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    tied_indices = farthest_point_indices(aggregate, TIED_ROWS)
    raw_anchors = aggregate[anchor_indices(aggregate)]
    interior = aggregate.mean(axis=0)
    anchors = 0.8 * raw_anchors + 0.2 * interior
    directions = fiber_direction_matrix(aggregate.shape[1], FIBER_PAIRS)

    aggregate_rows = [aggregate[index] for index in tied_indices]
    contrast_rows = [np.zeros(aggregate.shape[1]) for _ in tied_indices]
    metadata = [
        {"row_type": "tied", "source_index": index, "anchor": -1, "direction": -1, "sign": 0} for index in tied_indices
    ]
    for direction_index, direction in enumerate(directions):
        anchor_index = direction_index % ANCHOR_COUNT
        anchor = anchors[anchor_index]
        nonzero = np.abs(direction) > 1e-12
        max_delta_scale = float(np.min(anchor[nonzero] / (max(alpha0, alpha1) * np.abs(direction[nonzero]))))
        delta_scale = 0.8 * max_delta_scale
        for sign in (-1, 1):
            delta = sign * delta_scale * direction
            phase0 = anchor - alpha1 * delta
            phase1 = anchor + alpha0 * delta
            if np.min(phase0) < -1e-12 or np.min(phase1) < -1e-12:
                raise ValueError("Generated fiber policy is outside the simplex")
            aggregate_rows.append(alpha0 * phase0 + alpha1 * phase1)
            contrast_rows.append(alpha0 * alpha1 * (phase1 - phase0))
            metadata.append(
                {
                    "row_type": "fiber",
                    "source_index": -1,
                    "anchor": anchor_index,
                    "direction": direction_index,
                    "sign": sign,
                }
            )
    return np.stack(aggregate_rows), np.stack(contrast_rows), pd.DataFrame(metadata)


def render(blocks: pd.DataFrame, canonical_rows: list[dict[str, object]]) -> None:
    canonical = pd.DataFrame(canonical_rows)
    figure = make_subplots(
        rows=1, cols=2, subplot_titles=("Block singular spectra", "Aggregate/contrast canonical correlation")
    )
    for (design, block), group in blocks.groupby(["design", "block"]):
        figure.add_trace(
            go.Scatter(
                x=group["index"],
                y=group["singular_value"],
                mode="lines+markers",
                name=f"{design}: {block}",
            ),
            row=1,
            col=1,
        )
    for design, group in canonical.groupby("design"):
        figure.add_trace(
            go.Scatter(
                x=group["index"],
                y=group["canonical_correlation"],
                mode="lines+markers",
                name=f"{design}: canonical corr",
            ),
            row=1,
            col=2,
        )
    figure.update_yaxes(type="log", title_text="Singular value", row=1, col=1)
    figure.update_yaxes(title_text="Canonical correlation", range=[-0.02, 1.02], row=1, col=2)
    figure.update_xaxes(title_text="Ordered direction", row=1, col=1)
    figure.update_xaxes(title_text="Canonical direction", row=1, col=2)
    figure.update_layout(
        template="plotly_white",
        width=1200,
        height=560,
        title="Same-budget two-stage sampling orthogonalizes aggregate and phase-contrast evidence",
        legend={"orientation": "h", "y": -0.18},
    )
    figure.write_html(
        OUTPUT_DIR / "design_identifiability.html",
        include_plotlyjs="cdn",
        config=PLOT_CONFIG,
    )


def main() -> None:
    frame = pd.read_csv(INPUT)
    domain_names = domains(frame)
    phase0 = frame[[f"phase_0_{domain}" for domain in domain_names]].to_numpy(float)
    phase1 = frame[[f"phase_1_{domain}" for domain in domain_names]].to_numpy(float)
    phase0_fraction = frame["phase_0_fraction"].to_numpy(float)
    if len(frame) != 280 or np.ptp(phase0_fraction) > 1e-12:
        raise ValueError("Expected one 280-row, fixed-phase-fraction design")
    alpha0 = float(phase0_fraction[0])
    alpha1 = 1.0 - alpha0
    if not np.isclose(alpha0, PHASE0_FRACTION, atol=0.01):
        raise ValueError(f"Expected an approximately 80/20 phase split, found {alpha0}")
    aggregate = alpha0 * phase0 + alpha1 * phase1
    contrast = alpha0 * alpha1 * (phase1 - phase0)
    staged_aggregate, staged_contrast, metadata = build_same_budget_design(aggregate, alpha0, alpha1)

    metrics = []
    spectrum_rows: list[dict[str, object]] = []
    canonical_rows: list[dict[str, object]] = []
    for design, a, d in (
        ("random_two_phase_280", aggregate, contrast),
        ("two_stage_140_tied_70_pairs", staged_aggregate, staged_contrast),
    ):
        for block_name, block in (("aggregate", a), ("contrast", d)):
            row, singular_values = block_metrics(f"{design}_{block_name}", block)
            row["design"] = design
            metrics.append(row)
            spectrum_rows.extend(
                {
                    "design": design,
                    "block": block_name,
                    "index": index + 1,
                    "singular_value": value,
                }
                for index, value in enumerate(singular_values)
            )
        joint, canonical = joint_metrics(design, a, d)
        metrics.append({**joint, "block": "joint"})
        canonical_rows.extend(
            {
                "design": design,
                "index": index + 1,
                "canonical_correlation": value,
            }
            for index, value in enumerate(canonical)
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_frame = pd.DataFrame(metrics)
    spectrum = pd.DataFrame(spectrum_rows)
    metrics_frame.to_csv(OUTPUT_DIR / "design_metrics.csv", index=False)
    spectrum.to_csv(OUTPUT_DIR / "singular_spectra.csv", index=False)
    pd.DataFrame(canonical_rows).to_csv(OUTPUT_DIR / "canonical_correlations.csv", index=False)
    metadata.to_csv(OUTPUT_DIR / "same_budget_design_manifest.csv", index=False)
    render(spectrum, canonical_rows)

    current = metrics_frame[
        metrics_frame["design"].eq("random_two_phase_280") & metrics_frame["block"].eq("joint")
    ].iloc[0]
    staged = metrics_frame[
        metrics_frame["design"].eq("two_stage_140_tied_70_pairs") & metrics_frame["block"].eq("joint")
    ].iloc[0]
    report = "\n".join(
        [
            "# Round 56: same-budget two-stage design identifiability",
            "",
            "This coordinate-only diagnostic compares the observed 280-row random two-phase design with a "
            "same-row-budget design containing 140 phase-tied aggregate policies and 70 signed phase-fiber pairs. "
            "No BPB outcome is read. The synthetic design is diagnostic only and is not a submitted panel.",
            "",
            metrics_frame.to_markdown(index=False, floatfmt=".5f"),
            "",
            "## Result",
            "",
            f"The random design has mean aggregate/contrast canonical correlation {float(current['mean_canonical_correlation']):.3f} "
            f"and preserves {float(current['contrast_energy_after_aggregate_residualization']):.3f} of standardized contrast "
            "energy after aggregate residualization. The signed-pair design makes aggregate and contrast exactly orthogonal "
            f"to numerical precision (mean canonical correlation {float(staged['mean_canonical_correlation']):.3g}) while preserving "
            f"full {int(staged['numerical_rank'])}-dimensional joint rank.",
            "",
            "This does not prove that the proposed two-stage methodology will yield a good surrogate. It does show that the "
            "current design spends each response on jointly identifying aggregate utility and phase transport, whereas signed "
            "fiber pairs can identify phase transport without aggregate confounding at the same checkpoint budget. That is a "
            "design argument for the preregistered future intervention, not a repair to any current response model.",
        ]
    )
    (OUTPUT_DIR / "report.md").write_text(report + "\n")
    manifest = {
        "input": str(INPUT.relative_to(TWO_PHASE_ROOT)),
        "reads_target_outcomes": False,
        "reads_adversarial_outcomes": False,
        "reads_sealed_confirmation_outcomes": False,
        "random_seed": RANDOM_SEED,
        "row_budget": len(frame),
        "tied_rows": TIED_ROWS,
        "fiber_pairs": FIBER_PAIRS,
        "anchors": ANCHOR_COUNT,
        "realized_phase_fractions": [alpha0, alpha1],
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(report)


if __name__ == "__main__":
    main()
