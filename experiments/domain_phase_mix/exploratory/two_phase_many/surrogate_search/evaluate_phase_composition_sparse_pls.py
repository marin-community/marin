# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reproduce and optimize the Yixin phase-composition sparse PLS surrogate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.phase_composition_sparse_pls import (
    SELECTED_TERM_NAMES,
    load_phase_composition_packet,
    optimize_phase_composition_sparse_pls_model,
    reproduction_cv_summary,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_JSON = SCRIPT_DIR / "phase_composition_sparse_pls_summary.json"
OUTPUT_MD = SCRIPT_DIR / "phase_composition_sparse_pls_report.md"
OUTPUT_CSV = SCRIPT_DIR / "phase_composition_sparse_pls_optimum_weights.csv"


def _phase_entropy(weights: np.ndarray) -> float:
    clipped = np.clip(np.asarray(weights, dtype=float), 1e-12, 1.0)
    return float(-np.sum(clipped * np.log(clipped)))


def _top_domains(
    domain_names: list[str], weights: np.ndarray, epochs: np.ndarray, top_k: int = 10
) -> list[dict[str, float | str]]:
    frame = pd.DataFrame({"domain": domain_names, "weight": weights, "epochs": epochs})
    return frame.sort_values(["weight", "epochs"], ascending=False).head(top_k).to_dict(orient="records")


def _optimum_weight_frame(
    domain_names: list[str], phase0: np.ndarray, phase1: np.ndarray, c0: np.ndarray, c1: np.ndarray
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "domain": domain_names,
            "phase_0_weight": phase0,
            "phase_0_epochs": phase0 * c0,
            "phase_1_weight": phase1,
            "phase_1_epochs": phase1 * c1,
        }
    )


def evaluate() -> dict[str, Any]:
    """Run the full reproduction and optimum analysis."""
    data = load_phase_composition_packet()
    reproduction, model = reproduction_cv_summary(data)
    result, phase0, phase1 = optimize_phase_composition_sparse_pls_model(data, model, seed=0)
    optimum = np.stack([phase0, phase1], axis=0)

    distances = 0.5 * np.abs(data.w - optimum[None, :, :]).sum(axis=2).mean(axis=1)
    nearest_idx = int(np.argmin(distances))
    best_idx = int(np.argmin(data.y))
    counts = model.parameter_count()

    optimum_summary = {
        "predicted_optimum_value": float(result.fun),
        "observed_best_run_name": str(data.frame.iloc[best_idx][data.name_col]),
        "observed_best_value": float(data.y[best_idx]),
        "gap_below_observed_best": float(result.fun - data.y[best_idx]),
        "nearest_observed_run_name": str(data.frame.iloc[nearest_idx][data.name_col]),
        "nearest_observed_value": float(data.y[nearest_idx]),
        "nearest_observed_tv_distance": float(distances[nearest_idx]),
        "phase0_max_weight": float(phase0.max()),
        "phase1_max_weight": float(phase1.max()),
        "phase0_support_below_1e4": int(np.sum(phase0 < 1e-4)),
        "phase1_support_below_1e4": int(np.sum(phase1 < 1e-4)),
        "phase0_support_below_1e6": int(np.sum(phase0 < 1e-6)),
        "phase1_support_below_1e6": int(np.sum(phase1 < 1e-6)),
        "phase0_entropy": _phase_entropy(phase0),
        "phase1_entropy": _phase_entropy(phase1),
        "phase0_top_domains": _top_domains(data.domain_names, phase0, phase0 * data.c0),
        "phase1_top_domains": _top_domains(data.domain_names, phase1, phase1 * data.c1),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
    }

    summary = {
        "model": "Phase Composition Sparse PLS",
        "target": reproduction["target"],
        "selected_terms": list(SELECTED_TERM_NAMES),
        "parameter_count": {
            "linear_coefficients": counts.linear_coefficients,
            "reported_total": counts.reported_total,
            "total_with_shapes": counts.total_with_shapes,
        },
        "reproduction": reproduction,
        "optimum": optimum_summary,
    }

    _optimum_weight_frame(data.domain_names, phase0, phase1, data.c0, data.c1).to_csv(OUTPUT_CSV, index=False)
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True))

    lines = [
        "# Phase Composition Sparse PLS",
        "",
        "## Reproduction",
        "",
        f"- target: `{reproduction['target']}`",
        f"- rows: `{reproduction['rows']}`",
        f"- selected terms: `{len(SELECTED_TERM_NAMES)}`",
        f"- parameters (reported): `{counts.reported_total}`",
        f"- CV Spearman: `{reproduction['cv_spearman_mean']:.4f}`",
        f"- CV R^2: `{reproduction['cv_r2_mean']:.4f}`",
        "",
        "## Full-Fit Coefficients",
        "",
    ]
    for name in SELECTED_TERM_NAMES:
        lines.append(f"- `{name}`: `{reproduction['full_fit_coefficients'][name]:.6f}`")
    lines.extend(
        [
            "",
            "## Predicted Optimum",
            "",
            f"- predicted `eval/uncheatable_eval/bpb`: `{optimum_summary['predicted_optimum_value']:.6f}`",
            (
                f"- best observed run: `{optimum_summary['observed_best_run_name']}` "
                f"at `{optimum_summary['observed_best_value']:.6f}`"
            ),
            f"- nearest observed run: `{optimum_summary['nearest_observed_run_name']}`",
            f"- nearest observed TV distance: `{optimum_summary['nearest_observed_tv_distance']:.6f}`",
            f"- phase 0 max weight: `{optimum_summary['phase0_max_weight']:.6f}`",
            f"- phase 1 max weight: `{optimum_summary['phase1_max_weight']:.6f}`",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines))
    return summary


def main() -> None:
    summary = evaluate()
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")
    print(f"Wrote {OUTPUT_CSV}")
    print(f"CV Spearman={summary['reproduction']['cv_spearman_mean']:.4f}")
    print(f"CV R2={summary['reproduction']['cv_r2_mean']:.4f}")
    print(f"Predicted optimum={summary['optimum']['predicted_optimum_value']:.6f}")


if __name__ == "__main__":
    main()
