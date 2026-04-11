# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare the representative 520M qsplit240 pilot against 60M and completed 300M runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from marin.speedrun.speedrun import get_step_times_from_wandb

from experiments.domain_phase_mix.launch_two_phase_many_run_00097_fixed_subset_study import (
    WANDB_ENTITY,
    WANDB_PROJECT,
)
from experiments.domain_phase_mix.two_phase_many_observed_runs import REPRESENTATIVE12_PANEL_RUN_NAMES

SCRIPT_DIR = Path(__file__).resolve().parent
OBJECTIVE_METRIC = "eval/uncheatable_eval/bpb"
TWO_PHASE_MANY_CSV = SCRIPT_DIR / "two_phase_many.csv"
QSPLIT240_300M_COMPLETED_CSV = SCRIPT_DIR / "qsplit240_300m_6b_completed_vs_60m.csv"
OUTPUT_CSV = SCRIPT_DIR / "qsplit240_520m_pilot_comparison.csv"
OUTPUT_JSON = SCRIPT_DIR / "qsplit240_520m_pilot_summary.json"
PROJECTED_FULL_SWARM_RUNS = 240


def active_compute_hours_for_run(
    run_id: str,
    *,
    wandb_entity: str = WANDB_ENTITY,
    wandb_project: str = WANDB_PROJECT,
) -> float:
    """Return active compute hours for one W&B run."""
    return float(sum(get_step_times_from_wandb(run_id=run_id, entity=wandb_entity, project=wandb_project)) / 3600.0)


def _rank_series(values: pd.Series) -> pd.Series:
    return values.rank(method="min", ascending=True)


def _load_panel_template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "panel_index": np.arange(len(REPRESENTATIVE12_PANEL_RUN_NAMES), dtype=int),
            "run_name": list(REPRESENTATIVE12_PANEL_RUN_NAMES),
        }
    )


def build_pilot_comparison_frame(
    *,
    pilot_results_csv: Path,
    source_60m_csv: Path = TWO_PHASE_MANY_CSV,
    source_300m_csv: Path = QSPLIT240_300M_COMPLETED_CSV,
    wandb_entity: str = WANDB_ENTITY,
    wandb_project: str = WANDB_PROJECT,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Build the joined 60M/300M/520M comparison table for the representative panel."""
    pilot_frame = pd.read_csv(pilot_results_csv)
    if pilot_frame["run_name"].duplicated().any():
        duplicate_names = pilot_frame.loc[pilot_frame["run_name"].duplicated(), "run_name"].tolist()
        raise ValueError(f"Pilot results contain duplicate run names: {duplicate_names}")

    panel_frame = _load_panel_template()
    sixty_frame = (
        pd.read_csv(source_60m_csv, usecols=["run_name", OBJECTIVE_METRIC])
        .rename(columns={OBJECTIVE_METRIC: "bpb_60m"})
        .drop_duplicates("run_name")
    )
    three_hundred_frame = pd.read_csv(source_300m_csv, usecols=["run_name", "bpb_300m_6b"]).drop_duplicates("run_name")
    pilot_keep_columns = ["run_name", OBJECTIVE_METRIC]
    optional_columns = [column for column in ("wandb_run_id", "wandb_url", "status") if column in pilot_frame.columns]
    pilot_keep_columns.extend(optional_columns)
    pilot_frame = pilot_frame[pilot_keep_columns].rename(columns={OBJECTIVE_METRIC: "bpb_520m"})

    joined = (
        panel_frame.merge(sixty_frame, on="run_name", how="left")
        .merge(three_hundred_frame, on="run_name", how="left")
        .merge(pilot_frame, on="run_name", how="left")
    )
    joined["rank_60m_in_panel"] = _rank_series(joined["bpb_60m"])
    joined["rank_300m_6b_in_panel"] = _rank_series(joined["bpb_300m_6b"])
    joined["rank_520m_in_panel"] = _rank_series(joined["bpb_520m"])
    joined["rank_shift_520m_vs_60m"] = joined["rank_60m_in_panel"] - joined["rank_520m_in_panel"]
    joined["rank_shift_520m_vs_300m_6b"] = joined["rank_300m_6b_in_panel"] - joined["rank_520m_in_panel"]

    active_hours: list[float] = []
    for _, row in joined.iterrows():
        wandb_run_id = row.get("wandb_run_id")
        if not isinstance(wandb_run_id, str) or not wandb_run_id:
            active_hours.append(np.nan)
            continue
        active_hours.append(
            active_compute_hours_for_run(
                wandb_run_id,
                wandb_entity=wandb_entity,
                wandb_project=wandb_project,
            )
        )
    joined["active_compute_hours_520m"] = active_hours
    joined = joined.sort_values(["rank_520m_in_panel", "panel_index"], na_position="last").reset_index(drop=True)

    completed_hours = joined["active_compute_hours_520m"].dropna().to_numpy(dtype=np.float64)
    mean_active_hours = float(completed_hours.mean()) if len(completed_hours) else np.nan
    summary = {
        "panel_size": len(joined),
        "completed_520m_runs": int(joined["bpb_520m"].notna().sum()),
        "completed_300m_shared_runs": int(joined["bpb_300m_6b"].notna().sum()),
        "best_60m_run_name": str(joined.loc[joined["bpb_60m"].idxmin(), "run_name"]),
        "best_300m_6b_run_name": (
            None
            if not joined["bpb_300m_6b"].notna().any()
            else str(joined.loc[joined["bpb_300m_6b"].idxmin(), "run_name"])
        ),
        "best_520m_run_name": (
            None if not joined["bpb_520m"].notna().any() else str(joined.loc[joined["bpb_520m"].idxmin(), "run_name"])
        ),
        "active_compute_hours_mean_520m": mean_active_hours,
        "active_compute_hours_median_520m": float(np.median(completed_hours)) if len(completed_hours) else np.nan,
        "projected_serialized_full_240_hours": mean_active_hours * PROJECTED_FULL_SWARM_RUNS,
        "projected_serialized_full_240_days": mean_active_hours * PROJECTED_FULL_SWARM_RUNS / 24.0,
    }
    return joined, summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the representative qsplit240 520M pilot.")
    parser.add_argument("--results-csv", required=True, type=Path)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--output-json", type=Path, default=OUTPUT_JSON)
    parser.add_argument("--source-60m-csv", type=Path, default=TWO_PHASE_MANY_CSV)
    parser.add_argument("--source-300m-csv", type=Path, default=QSPLIT240_300M_COMPLETED_CSV)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    comparison_frame, summary = build_pilot_comparison_frame(
        pilot_results_csv=args.results_csv,
        source_60m_csv=args.source_60m_csv,
        source_300m_csv=args.source_300m_csv,
    )
    comparison_frame.to_csv(args.output_csv, index=False)
    with args.output_json.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
