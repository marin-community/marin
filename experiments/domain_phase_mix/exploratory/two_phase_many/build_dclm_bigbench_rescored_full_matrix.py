# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas"]
# ///
"""Overlay DCLM BigBench generation rescores onto the full 300M DCLM matrix."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import dclm_matrix_guard
from experiments.domain_phase_mix.exploratory.two_phase_many import recompute_dclm_bigbench_generation_scores as rescore


TWO_PHASE_MANY_DIR = Path("experiments/domain_phase_mix/exploratory/two_phase_many")
METRIC_REGISTRY_DIR = TWO_PHASE_MANY_DIR / "metric_registry" / "300m_dclm_core_completion"
JOIN_COLUMN = "run_name"
assert_dclm_macro_consistent = dclm_matrix_guard.assert_dclm_macro_consistent


def overlay_columns() -> list[str]:
    """Return exact rescore columns to overlay from the BigBench generation matrix."""
    columns: list[str] = []
    for task in rescore.RESCORABLE_TASKS:
        columns.extend([task.metric_column, task.raw_score_column, task.centered_column])
    return columns


def overlay_bigbench_rescores(base: pd.DataFrame, rescored: pd.DataFrame, output_csv: Path) -> pd.DataFrame:
    """Return the full matrix with BigBench generation rescore columns overlaid."""
    if JOIN_COLUMN not in base.columns:
        raise ValueError(f"Base matrix missing join column {JOIN_COLUMN!r}")
    if JOIN_COLUMN not in rescored.columns:
        raise ValueError(f"Rescore matrix missing join column {JOIN_COLUMN!r}")
    if base[JOIN_COLUMN].duplicated().any():
        raise ValueError(f"Base matrix has duplicate {JOIN_COLUMN} values")
    if rescored[JOIN_COLUMN].duplicated().any():
        raise ValueError(f"Rescore matrix has duplicate {JOIN_COLUMN} values")

    base_by_run = base.set_index(JOIN_COLUMN)
    rescored_by_run = rescored.set_index(JOIN_COLUMN)
    missing = sorted(set(base_by_run.index) - set(rescored_by_run.index))
    if missing:
        raise ValueError(f"Rescore matrix missing {len(missing)} run names; first missing={missing[:5]}")

    selected_columns = overlay_columns()
    missing_columns = [column for column in selected_columns if column not in base_by_run.columns]
    missing_rescore_columns = [column for column in selected_columns if column not in rescored_by_run.columns]
    if missing_columns:
        raise ValueError(f"Base matrix missing overlay columns: {missing_columns}")
    if missing_rescore_columns:
        raise ValueError(f"Rescore matrix missing overlay columns: {missing_rescore_columns}")

    corrected = base_by_run.copy()
    corrected.loc[:, selected_columns] = rescored_by_run.loc[corrected.index, selected_columns]
    rescore.recompute_dclm_macro(corrected)
    corrected = corrected.reset_index()[base.columns]
    dclm_matrix_guard.validate_corrected_dclm_matrix(
        corrected,
        output_csv,
        allow_intermediate_repeat_copy=True,
    )
    return corrected


def task_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return compact summaries for DCLM BigBench generation tasks."""
    rows = []
    for task in (*rescore.RESCORABLE_TASKS, "bb_repeat_copy_logic_10shot"):
        alias = task.alias if hasattr(task, "alias") else str(task)
        column = f"lm_eval/dclm_core/{alias}/raw_score"
        values = pd.to_numeric(frame[column], errors="coerce")
        rows.append(
            {
                "alias": alias,
                "nonnull": int(values.notna().sum()),
                "unique": int(values.nunique(dropna=True)),
                "mean": float(values.mean()) if values.notna().any() else math.nan,
                "min": float(values.min()) if values.notna().any() else math.nan,
                "max": float(values.max()) if values.notna().any() else math.nan,
            }
        )
    return pd.DataFrame.from_records(rows)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-csv",
        type=Path,
        required=True,
        help="Historical full DCLM matrix to overlay. Required to avoid accidentally using stale local data.",
    )
    parser.add_argument(
        "--rescore-csv",
        type=Path,
        required=True,
        help="Intermediate BigBench-only rescore matrix.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Output path for the intermediate BigBench-rescored full matrix.",
    )
    return parser.parse_args()


def main() -> None:
    """Build the corrected full DCLM matrix."""
    args = parse_args()
    base = pd.read_csv(args.base_csv)
    rescored = pd.read_csv(args.rescore_csv)
    corrected = overlay_bigbench_rescores(base, rescored, args.output_csv)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    corrected.to_csv(args.output_csv, index=False)
    print(args.output_csv)
    print(task_summary(corrected).to_string(index=False))


if __name__ == "__main__":
    main()
