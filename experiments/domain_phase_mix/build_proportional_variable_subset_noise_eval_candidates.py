# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build eval-candidate CSVs for proportional variable-subset noise baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiments.domain_phase_mix.launch_proportional_variable_subset_noise_baseline import family_for_scale
from experiments.domain_phase_mix.scaling_study_recipes import ScalingStudyScale

SCRIPT_DIR = Path(__file__).resolve().parent
TWO_PHASE_MANY_DIR = SCRIPT_DIR / "exploratory" / "two_phase_many"
RUN_REGISTRY_CSV = TWO_PHASE_MANY_DIR / "run_registry" / "logical_runs.csv"
OUTPUT_DIR = TWO_PHASE_MANY_DIR / "metric_registry" / "proportional_variable_subset_noise"
DEFAULT_OUTPUT_CSV = OUTPUT_DIR / "proportional_variable_subset_eval_candidates.csv"
DEFAULT_SCALES = (ScalingStudyScale.REGMIX_60M_1P2B, ScalingStudyScale.REGMIX_300M_6B)
EXPECTED_ROWS_PER_SCALE = 10


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-registry-csv", type=Path, default=RUN_REGISTRY_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--scales",
        default=",".join(scale.value for scale in DEFAULT_SCALES),
        help="Comma-separated historical scale keys.",
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def _scale_values(raw: str) -> tuple[str, ...]:
    values = tuple(dict.fromkeys(part.strip() for part in raw.split(",") if part.strip()))
    known = {scale.value for scale in ScalingStudyScale}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"Unknown scale keys: {unknown}")
    return values


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    if value is None or pd.isna(value):
        return False
    return bool(value)


def build_candidate_frame(
    *,
    run_registry_csv: Path,
    scales: tuple[str, ...],
    allow_incomplete: bool,
) -> pd.DataFrame:
    """Return eval candidates for completed proportional variable-subset rows."""
    if not run_registry_csv.exists():
        raise FileNotFoundError(f"Missing run registry: {run_registry_csv}")
    registry = pd.read_csv(run_registry_csv, low_memory=False)
    families = {family_for_scale(scale) for scale in scales}
    frame = registry[registry["family"].isin(families)].copy()
    expected_rows = len(scales) * EXPECTED_ROWS_PER_SCALE
    if len(frame) != expected_rows:
        raise ValueError(f"Expected {expected_rows} proportional noise registry rows, found {len(frame)}")

    if "is_perplexity_ready" in frame.columns:
        ready = frame["is_perplexity_ready"].map(_bool_value)
    else:
        ready = frame["checkpoint_root"].notna() & frame["objective_metric_value"].notna()
    if not allow_incomplete and int(ready.sum()) != expected_rows:
        missing = frame.loc[~ready, ["family", "run_name", "logical_status", "checkpoint_root"]]
        raise ValueError(f"Proportional noise rows are not target-ready:\n{missing.to_string(index=False)}")

    frame = frame.loc[ready].copy()
    required_columns = {
        "registry_id",
        "family",
        "scale",
        "run_name",
        "source_experiment",
        "checkpoint_root",
        "target_final_checkpoint_step",
    }
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Registry missing required columns: {missing_columns}")
    out = pd.DataFrame(
        {
            "panel": frame["family"],
            "scale": frame["scale"],
            "run_name": frame["run_name"],
            "registry_key": frame["registry_id"],
            "source_experiment": frame["source_experiment"],
            "cohort": frame.get("cohort", frame["family"]),
            "checkpoint_root": frame["checkpoint_root"].astype(str).str.rstrip("/"),
            "expected_checkpoint_step": pd.to_numeric(frame["target_final_checkpoint_step"], errors="raise").astype(int),
            "noise_anchor_run_name": frame.get("noise_anchor_run_name", "baseline_proportional"),
            "noise_trainer_seed": pd.to_numeric(frame.get("noise_trainer_seed"), errors="coerce").astype("Int64"),
            "noise_subset_mode": "variable",
        }
    ).sort_values(["scale", "run_name"])
    if out["checkpoint_root"].duplicated().any():
        dupes = out.loc[out["checkpoint_root"].duplicated(keep=False), ["scale", "run_name", "checkpoint_root"]]
        raise ValueError(f"Duplicate checkpoint roots:\n{dupes.to_string(index=False)}")
    return out.reset_index(drop=True)


def main() -> None:
    args = _parse_args()
    frame = build_candidate_frame(
        run_registry_csv=args.run_registry_csv,
        scales=_scale_values(args.scales),
        allow_incomplete=args.allow_incomplete,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(frame)} eval candidates to {args.output_csv}")


if __name__ == "__main__":
    main()
