# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build downstream-eval candidates for the proportional perturbation experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
TWO_PHASE_MANY_DIR = SCRIPT_DIR / "exploratory" / "two_phase_many"
RUN_REGISTRY_CSV = TWO_PHASE_MANY_DIR / "run_registry" / "logical_runs.csv"
OUTPUT_DIR = TWO_PHASE_MANY_DIR / "metric_registry" / "proportional_perturbation_scale_transfer"
DEFAULT_OUTPUT_CSV = OUTPUT_DIR / "proportional_perturbation_eval_candidates.csv"
PERTURBATION_FAMILIES = {
    "proportional_perturbation_60m_1p2b",
    "proportional_perturbation_300m_6b",
}
EXPECTED_INTERVENTIONS_PER_SCALE = 55
BASELINE_ANCHORS = (
    {
        "panel": "proportional_baseline_anchor_60m_1p2b",
        "scale": "60m_1p2b",
        "run_name": "baseline_proportional",
        "registry_key": "proportional_baseline_anchor_60m_1p2b:baseline_proportional",
        "source_experiment": "pinlin_calvin_xu/data_mixture/ngd3dm2_hybrid_canary",
        "cohort": "proportional_baseline_anchor",
        "checkpoint_root": (
            "gs://marin-us-east5/checkpoints/pinlin_calvin_xu/data_mixture/"
            "ngd3dm2_hybrid_canary/baseline_proportional-bac80b"
        ),
        "expected_checkpoint_step": 4576,
        "intervention_id": "baseline_proportional",
        "intervention_type": "baseline",
    },
    {
        "panel": "proportional_baseline_anchor_300m_6b",
        "scale": "300m_6b",
        "run_name": "baseline_proportional",
        "registry_key": "proportional_baseline_anchor_300m_6b:baseline_proportional",
        "source_experiment": "pinlin_calvin_xu/data_mixture/ngd3dm2_qsplit240_300m_6b",
        "cohort": "proportional_baseline_anchor",
        "checkpoint_root": (
            "gs://marin-us-east5/checkpoints/pinlin_calvin_xu/data_mixture/"
            "ngd3dm2_qsplit240_300m_6b/baseline_proportional-982696"
        ),
        "expected_checkpoint_step": 22887,
        "intervention_id": "baseline_proportional",
        "intervention_type": "baseline",
    },
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-registry-csv", type=Path, default=RUN_REGISTRY_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--no-baseline-anchors",
        action="store_true",
        help="Only emit the 110 perturbation rows, omitting proportional baseline checkpoints.",
    )
    return parser.parse_args()


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
    allow_incomplete: bool,
    include_baseline_anchors: bool,
) -> pd.DataFrame:
    """Return perturbation checkpoints ready for downstream evaluation."""
    if not run_registry_csv.exists():
        raise FileNotFoundError(f"Missing run registry: {run_registry_csv}")
    registry = pd.read_csv(run_registry_csv, low_memory=False)
    frame = registry.loc[registry["family"].isin(PERTURBATION_FAMILIES)].copy()
    expected_rows = len(PERTURBATION_FAMILIES) * EXPECTED_INTERVENTIONS_PER_SCALE
    if len(frame) != expected_rows:
        raise ValueError(f"Expected {expected_rows} perturbation registry rows, found {len(frame)}")
    if frame.duplicated(["scale", "intervention_id"]).any():
        dupes = frame.loc[frame.duplicated(["scale", "intervention_id"], keep=False)]
        raise ValueError(f"Duplicate perturbation interventions:\n{dupes[['scale', 'intervention_id']].to_string()}")

    if "is_perplexity_ready" in frame.columns:
        ready = frame["is_perplexity_ready"].map(_bool_value)
    else:
        ready = frame["checkpoint_root"].notna() & frame["objective_metric_value"].notna()
    if not allow_incomplete and int(ready.sum()) != expected_rows:
        missing = frame.loc[~ready, ["family", "run_name", "logical_status", "checkpoint_root"]]
        raise ValueError(f"Perturbation rows are not target-ready:\n{missing.to_string(index=False)}")
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
        "target_unit",
        "tv_distance",
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
            "cohort": frame["cohort"],
            "checkpoint_root": frame["checkpoint_root"].astype(str).str.rstrip("/"),
            "expected_checkpoint_step": pd.to_numeric(frame["target_final_checkpoint_step"], errors="raise").astype(int),
            "intervention_id": frame["intervention_id"],
            "intervention_type": frame["intervention_type"],
            "target_unit": frame["target_unit"],
            "target_domain": frame.get("target_domain"),
            "target_family": frame.get("target_family"),
            "tv_distance": pd.to_numeric(frame["tv_distance"], errors="coerce"),
        }
    )
    if include_baseline_anchors:
        out = pd.concat([out, pd.DataFrame.from_records(BASELINE_ANCHORS)], ignore_index=True, sort=False)
    if out["checkpoint_root"].duplicated().any():
        dupes = out.loc[out["checkpoint_root"].duplicated(keep=False), ["panel", "run_name", "checkpoint_root"]]
        raise ValueError(f"Duplicate checkpoint roots:\n{dupes.to_string(index=False)}")
    return out.sort_values(["scale", "panel", "intervention_id"]).reset_index(drop=True)


def main() -> None:
    args = _parse_args()
    frame = build_candidate_frame(
        run_registry_csv=args.run_registry_csv,
        allow_incomplete=args.allow_incomplete,
        include_baseline_anchors=not args.no_baseline_anchors,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(frame)} eval candidates to {args.output_csv}")


if __name__ == "__main__":
    main()
