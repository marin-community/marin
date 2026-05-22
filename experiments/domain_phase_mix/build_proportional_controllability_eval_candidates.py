# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build downstream-eval candidates for the 300M proportional controllability panel."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiments.domain_phase_mix.launch_proportional_controllability_300m import FAMILY

SCRIPT_DIR = Path(__file__).resolve().parent
TWO_PHASE_MANY_DIR = SCRIPT_DIR / "exploratory" / "two_phase_many"
RUN_REGISTRY_CSV = TWO_PHASE_MANY_DIR / "run_registry" / "logical_runs.csv"
OUTPUT_DIR = TWO_PHASE_MANY_DIR / "metric_registry" / "proportional_controllability_300m"
DEFAULT_OUTPUT_CSV = OUTPUT_DIR / "proportional_controllability_eval_candidates.csv"
EXPECTED_ROWS = 117


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-registry-csv", type=Path, default=RUN_REGISTRY_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    if value is None or pd.isna(value):
        return False
    return bool(value)


def build_candidate_frame(*, run_registry_csv: Path, allow_incomplete: bool) -> pd.DataFrame:
    """Return target-ready controllability checkpoints for downstream evaluation."""
    if not run_registry_csv.exists():
        raise FileNotFoundError(f"Missing run registry: {run_registry_csv}")
    registry = pd.read_csv(run_registry_csv, low_memory=False)
    frame = registry.loc[registry["family"].eq(FAMILY)].copy()
    if len(frame) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} controllability registry rows, found {len(frame)}")
    if frame["run_name"].duplicated().any():
        dupes = frame.loc[frame["run_name"].duplicated(keep=False), ["run_name", "checkpoint_root"]]
        raise ValueError(f"Duplicate controllability run names:\n{dupes.to_string(index=False)}")

    if "is_perplexity_ready" in frame.columns:
        ready = frame["is_perplexity_ready"].map(_bool_value)
    else:
        ready = frame["checkpoint_root"].notna() & frame["objective_metric_value"].notna()
    if not allow_incomplete and int(ready.sum()) != EXPECTED_ROWS:
        missing = frame.loc[~ready, ["family", "run_name", "logical_status", "checkpoint_root"]]
        raise ValueError(f"Controllability rows are not target-ready:\n{missing.to_string(index=False)}")
    frame = frame.loc[ready].copy()

    required_columns = {
        "registry_id",
        "family",
        "scale",
        "run_name",
        "source_experiment",
        "cohort",
        "checkpoint_root",
        "target_final_checkpoint_step",
        "intervention_id",
        "intervention_type",
        "tv_distance",
    }
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Registry missing required columns: {missing_columns}")

    out = pd.DataFrame(
        {
            "panel": frame["family"],
            "family": frame["family"],
            "scale": frame["scale"],
            "run_name": frame["run_name"],
            "registry_key": frame["registry_id"],
            "source_experiment": frame["source_experiment"],
            "cohort": frame["cohort"],
            "checkpoint_root": frame["checkpoint_root"].astype(str).str.rstrip("/"),
            "expected_checkpoint_step": pd.to_numeric(frame["target_final_checkpoint_step"], errors="raise").astype(int),
            "intervention_id": frame["intervention_id"],
            "intervention_type": frame["intervention_type"],
            "target_domain": frame.get("target_domain"),
            "direction_id": frame.get("direction_id"),
            "direction_type": frame.get("direction_type"),
            "tilt_sign": frame.get("tilt_sign"),
            "alpha": pd.to_numeric(frame.get("alpha"), errors="coerce"),
            "base_mass": pd.to_numeric(frame["base_mass"], errors="coerce"),
            "tv_distance": pd.to_numeric(frame["tv_distance"], errors="coerce"),
            "renormalizer": frame["renormalizer"],
        }
    )
    if out["checkpoint_root"].duplicated().any():
        dupes = out.loc[out["checkpoint_root"].duplicated(keep=False), ["run_name", "checkpoint_root"]]
        raise ValueError(f"Duplicate checkpoint roots:\n{dupes.to_string(index=False)}")
    return out.sort_values(["scale", "intervention_type", "run_name"]).reset_index(drop=True)


def main() -> None:
    args = _parse_args()
    frame = build_candidate_frame(run_registry_csv=args.run_registry_csv, allow_incomplete=args.allow_incomplete)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(frame)} eval candidates to {args.output_csv}")


if __name__ == "__main__":
    main()
