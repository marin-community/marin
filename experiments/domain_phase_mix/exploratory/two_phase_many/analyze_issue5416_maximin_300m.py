# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scipy"]
# ///
"""Analyze maximin/Pareto improvement over proportional for issue #5416 items.

This is a local diagnostic, not a training launcher. It asks whether any
already-observed 300M/6B signal-row mixture improves over
``baseline_proportional`` on every selected issue #5416 item, after orienting
items so higher is better.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from experiments.domain_phase_mix.exploratory.two_phase_many.metric_registry.issue5416_aggregate import (
    fit_issue5416_projection,
    score_issue5416_aggregate,
)

SCRIPT_DIR = Path(__file__).resolve().parent
MATRIX_DIR = SCRIPT_DIR / "metric_registry" / "raw_metric_matrix_300m"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "issue5416_maximin_300m_20260511"
SIGNAL_CSV = MATRIX_DIR / "raw_metric_matrix_300m.csv"
VARIABLE_NOISE_CSV = MATRIX_DIR / "noise_baseline_run00097_variable_subset_300m.csv"
PROPORTIONAL_RUN_NAME = "baseline_proportional"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal-csv", type=Path, default=SIGNAL_CSV)
    parser.add_argument("--variable-noise-csv", type=Path, default=VARIABLE_NOISE_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _solve_convex_hull(
    *,
    frame: pd.DataFrame,
    scores: pd.Series,
    scaled_deltas: np.ndarray,
    output_dir: Path,
    name: str,
) -> dict[str, Any]:
    """Solve a convex-hull maximin diagnostic over non-proportional rows.

    The convex combination is not itself a guaranteed physically realizable
    mixture response. It is an upper-bound diagnostic for whether the observed
    metric vectors jointly contain a Pareto-improving direction.
    """
    candidate_mask = frame["run_name"].ne(PROPORTIONAL_RUN_NAME).to_numpy()
    matrix = scaled_deltas[candidate_mask]
    n_rows, n_items = matrix.shape
    objective = np.zeros(n_rows + 1)
    objective[-1] = -1.0
    a_ub = np.hstack([-matrix.T, np.ones((n_items, 1))])
    a_eq = np.zeros((1, n_rows + 1))
    a_eq[0, :n_rows] = 1.0
    result = linprog(
        objective,
        A_ub=a_ub,
        b_ub=np.zeros(n_items),
        A_eq=a_eq,
        b_eq=np.array([1.0]),
        bounds=[(0, None)] * n_rows + [(None, None)],
        method="highs",
    )
    summary: dict[str, Any] = {
        "success": bool(result.success),
        "message": result.message,
        "margin": None,
        "support_size": None,
    }
    if not result.success:
        return summary
    weights = result.x[:n_rows]
    support = np.flatnonzero(weights > 1e-6)
    nonprop_frame = frame.loc[candidate_mask].reset_index(drop=True)
    nonprop_scores = scores.loc[candidate_mask].reset_index(drop=True)
    support_frame = pd.DataFrame(
        {
            "run_name": nonprop_frame.iloc[support]["run_name"].to_numpy(),
            "weight": weights[support],
            "issue5416_aggregate": nonprop_scores.iloc[support].to_numpy(),
        }
    ).sort_values("weight", ascending=False)
    support_frame.to_csv(output_dir / f"convex_hull_{name}_support.csv", index=False)
    summary.update({"margin": float(result.x[-1]), "support_size": len(support)})
    return summary


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    signal = pd.read_csv(args.signal_csv, low_memory=False)
    noise = pd.read_csv(args.variable_noise_csv, low_memory=False)
    projection = fit_issue5416_projection(signal_frame=signal, noise_frame=noise)
    task_columns = list(projection.task_columns)
    task_signs = np.asarray(projection.task_signs, dtype=np.float64)

    proportional_mask = signal["run_name"].eq(PROPORTIONAL_RUN_NAME)
    if proportional_mask.sum() != 1:
        raise ValueError(f"Expected one {PROPORTIONAL_RUN_NAME} row, found {proportional_mask.sum()}")
    proportional = signal.loc[proportional_mask].iloc[0]
    complete_mask = signal.loc[:, task_columns].notna().all(axis=1)
    frame = signal.loc[complete_mask].copy().reset_index(drop=True)

    oriented_values = frame.loc[:, task_columns].to_numpy(dtype=np.float64) * task_signs[None, :]
    proportional_values = proportional.loc[task_columns].to_numpy(dtype=np.float64) * task_signs
    raw_deltas = oriented_values - proportional_values[None, :]
    signal_std = np.nanstd(signal.loc[:, task_columns].to_numpy(dtype=np.float64) * task_signs[None, :], axis=0, ddof=1)
    noise_std = np.nanstd(noise.loc[:, task_columns].to_numpy(dtype=np.float64) * task_signs[None, :], axis=0, ddof=1)
    noise_std = np.where((noise_std > 1e-12) & np.isfinite(noise_std), noise_std, np.nan)
    noise_z = raw_deltas / noise_std[None, :]
    signal_z = raw_deltas / signal_std[None, :]

    scores = score_issue5416_aggregate(frame, projection, fail_missing=True).reset_index(drop=True)
    proportional_score = float(
        score_issue5416_aggregate(signal.loc[proportional_mask], projection, fail_missing=True).iloc[0]
    )

    rows: list[dict[str, Any]] = []
    for index, row in frame.iterrows():
        deltas = raw_deltas[index]
        row_noise_z = noise_z[index]
        row_signal_z = signal_z[index]
        worst_noise_index = int(np.nanargmin(row_noise_z))
        worst_signal_index = int(np.argmin(row_signal_z))
        rows.append(
            {
                "run_name": row["run_name"],
                "row_kind": row.get("row_kind", ""),
                "issue5416_aggregate": float(scores.iloc[index]),
                "issue5416_delta_vs_prop": float(scores.iloc[index] - proportional_score),
                "improved_count": int((deltas > 0).sum()),
                "worse_count": int((deltas < 0).sum()),
                "tied_count": int((deltas == 0).sum()),
                "strictly_improves_all": bool((deltas > 0).all()),
                "weakly_improves_all": bool((deltas >= 0).all()),
                "min_raw_delta": float(deltas.min()),
                "min_noise_z_delta": float(np.nanmin(row_noise_z)),
                "min_signal_z_delta": float(row_signal_z.min()),
                "worst_noise_z_metric": task_columns[worst_noise_index],
                "worst_noise_z_delta": float(row_noise_z[worst_noise_index]),
                "worst_signal_z_metric": task_columns[worst_signal_index],
                "worst_signal_z_delta": float(row_signal_z[worst_signal_index]),
            }
        )
    summary = pd.DataFrame(rows)
    summary.sort_values(["min_signal_z_delta", "issue5416_delta_vs_prop"], ascending=False).to_csv(
        args.output_dir / "observed_maximin_summary.csv", index=False
    )
    nonprop = summary[summary["run_name"].ne(PROPORTIONAL_RUN_NAME)]
    nonprop.sort_values(["improved_count", "issue5416_delta_vs_prop"], ascending=False).to_csv(
        args.output_dir / "observed_nonprop_by_improved_count.csv", index=False
    )
    nonprop.sort_values(["min_signal_z_delta", "issue5416_delta_vs_prop"], ascending=False).to_csv(
        args.output_dir / "observed_nonprop_by_min_signal_z.csv", index=False
    )
    pd.DataFrame(
        {
            "metric": task_columns,
            "sign": task_signs,
            "proportional_oriented_value": proportional_values,
            "signal_std_oriented": signal_std,
            "variable_noise_std_oriented": noise_std,
            "best_observed_delta": raw_deltas.max(axis=0),
            "worst_observed_delta": raw_deltas.min(axis=0),
            "rows_improved": (raw_deltas > 0).sum(axis=0),
            "rows_worse": (raw_deltas < 0).sum(axis=0),
        }
    ).to_csv(args.output_dir / "selected_metric_item_summary.csv", index=False)

    finite_noise = np.isfinite(noise_std)
    hull_noise = _solve_convex_hull(
        frame=frame,
        scores=scores,
        scaled_deltas=raw_deltas[:, finite_noise] / noise_std[finite_noise][None, :],
        output_dir=args.output_dir,
        name="noise_z_exclude_prop",
    )
    hull_signal = _solve_convex_hull(
        frame=frame,
        scores=scores,
        scaled_deltas=raw_deltas / signal_std[None, :],
        output_dir=args.output_dir,
        name="signal_z_exclude_prop",
    )
    report = {
        "signal_rows": len(signal),
        "complete_rows_used": len(frame),
        "selected_items": len(task_columns),
        "factor_count": int(projection.factor_count),
        "proportional_issue5416_aggregate": proportional_score,
        "observed_nonprop_strictly_improves_all_items": int(nonprop["strictly_improves_all"].sum()),
        "observed_nonprop_weakly_improves_all_items": int(nonprop["weakly_improves_all"].sum()),
        "best_nonprop_by_improved_count": (
            nonprop.sort_values(["improved_count", "issue5416_delta_vs_prop"], ascending=False)
            .head(1)
            .to_dict("records")[0]
        ),
        "best_nonprop_by_min_signal_z": (
            nonprop.sort_values(["min_signal_z_delta", "issue5416_delta_vs_prop"], ascending=False)
            .head(1)
            .to_dict("records")[0]
        ),
        "convex_hull_noise_z_exclude_prop": hull_noise,
        "convex_hull_signal_z_exclude_prop": hull_signal,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n")
    print(json.dumps(_json_safe(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
