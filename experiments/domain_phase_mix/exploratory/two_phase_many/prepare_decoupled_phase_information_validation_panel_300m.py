# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "fsspec",
#   "gcsfs",
#   "numpy",
#   "pandas",
#   "tabulate",
# ]
# ///
"""Select and upload the 3e18 decoupled phase-information validation panel."""

from __future__ import annotations

import argparse
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = SCRIPT_DIR / "reference_outputs" / "decoupled_phase_information_model_family_panel_20260712"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "decoupled_phase_information_validation_panel_20260712"
DEFAULT_GCS_OUTPUT_DIR = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "delphi_decoupled_phase_information_validation_20260712/mixtures"
)
CANONICAL_BUDGETS = frozenset({0.005, 0.025, 0.075, 0.125, 0.2})
WEIGHT_COLUMNS = ("phase_0_weight", "phase_1_weight")
WEIGHT_TOLERANCE = 1e-6


def selection_reason(row: pd.Series) -> str | None:
    """Return the scientific inclusion reason, or None when the row is excluded."""
    family = str(row["family"])
    objective = str(row["objective"])
    budget = float(row["phase_information_budget"])
    if family == "control":
        return "fixed-aggregate tied control"
    if family == "separate_heads":
        return "primary model; full phase-information sweep"
    if family == "effective_exposure":
        return "new constraint directly tests the prior aggregate-underexposure failure"
    if family == "canonical" and budget in CANONICAL_BUDGETS:
        return "sparse diagnostic sweep; prior transfer and calibration are weak"
    if family == "effective_exposure_geometry" and objective == "uncheatable":
        return "prior exact tied contrasts support an Uncheatable phase effect"
    return None


def weight_array(path: Path) -> np.ndarray:
    frame = pd.read_csv(path).sort_values("domain")
    missing = set(WEIGHT_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing weight columns: {sorted(missing)}")
    return frame.loc[:, WEIGHT_COLUMNS].to_numpy(dtype=float)


def remove_path_duplicates(selected: pd.DataFrame, source_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove saturated duplicate points within one model-family path."""
    kept_indices: list[int] = []
    duplicate_rows: list[dict[str, object]] = []
    for (_anchor, _family), group in selected.groupby(["anchor_tag", "family"], sort=False):
        seen: list[tuple[str, np.ndarray]] = []
        for index, row in group.sort_values("phase_information_budget").iterrows():
            weights = weight_array(source_dir / "mixtures" / f"{row['candidate']}.csv")
            duplicate_of = next(
                (
                    candidate
                    for candidate, prior_weights in seen
                    if np.allclose(weights, prior_weights, rtol=0.0, atol=WEIGHT_TOLERANCE)
                ),
                None,
            )
            if duplicate_of is None:
                kept_indices.append(index)
                seen.append((str(row["candidate"]), weights))
                continue
            duplicate = row.to_dict()
            duplicate["exclusion_reason"] = f"same phase weights as {duplicate_of}"
            duplicate_rows.append(duplicate)
    kept = selected.loc[kept_indices].sort_values(["objective", "anchor_tag", "family", "phase_information_budget"])
    return kept, pd.DataFrame(duplicate_rows)


def copy_candidate(source: Path, destination: Path, gcs_destination: str | None) -> None:
    frame = pd.read_csv(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    if gcs_destination is not None:
        with fsspec.open(gcs_destination, "wt") as handle:
            frame.to_csv(handle, index=False)


def write_report(selected: pd.DataFrame, excluded: pd.DataFrame, output_dir: Path) -> None:
    counts = (
        selected.groupby(["objective", "anchor_tag", "family"], as_index=False)
        .agg(
            candidates=("candidate", "size"),
            min_epsilon=("phase_information_budget", "min"),
            max_epsilon=("phase_information_budget", "max"),
            max_predicted_gain=("predicted_gain_vs_tied", "max"),
            max_phase_tv=("phase_tv", "max"),
            max_simulated_epoch=("max_simulated_epoch", "max"),
        )
        .sort_values(["objective", "anchor_tag", "family"])
    )
    lines = [
        "# Decoupled phase-information 3e18 validation panel",
        "",
        "The aggregate mixture is fixed to a separately fitted one-phase anchor. The sweep only changes "
        "phase ordering under an explicit phase-information budget. There are no repeat seeds.",
        "",
        "## Inclusion gate",
        "",
        "- Separate-heads: full nonzero epsilon sweep because prior 3e18 deployment evidence made it the "
        "primary candidate generator.",
        "- Effective-exposure: full nonzero epsilon sweep to test whether decoupled regularization fixes "
        "its aggregate-underexposure failure.",
        "- Canonical DSP: sparse epsilon sweep because prior validation and controlled-pair calibration are weak.",
        "- Effective-exposure plus geometry: Uncheatable only; prior Table-9 phase transfer reversed and is excluded.",
        "- Tied controls: one per aggregate anchor.",
        "- Saturated points with identical phase weights within one path are removed.",
        "- Predicted gains are uncalibrated surrogate values: every family overpredicted the controlled phase gain.",
        "- Epsilon 0.2 is the end of the deployment sweep, not necessarily the unconstrained surrogate optimum.",
        "",
        "## Selected panel",
        "",
        counts.to_markdown(index=False, floatfmt=".6f"),
        "",
        f"Selected candidates: {len(selected)}",
        f"Excluded candidates: {len(excluded)}",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gcs-output-dir", default=DEFAULT_GCS_OUTPUT_DIR)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    manifest = pd.read_csv(args.source_dir / "candidate_manifest.csv")
    reasons = manifest.apply(selection_reason, axis=1)
    selected = manifest.loc[reasons.notna()].copy()
    selected["selection_reason"] = reasons.loc[reasons.notna()]
    excluded = manifest.loc[reasons.isna()].copy()
    excluded["exclusion_reason"] = "failed prior-evidence inclusion gate"
    selected, duplicate_rows = remove_path_duplicates(selected, args.source_dir)
    if not duplicate_rows.empty:
        excluded = pd.concat([excluded, duplicate_rows], ignore_index=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gcs_dir = args.gcs_output_dir.rstrip("/")
    source_paths: list[str] = []
    for candidate in selected["candidate"]:
        source = args.source_dir / "mixtures" / f"{candidate}.csv"
        destination = args.output_dir / "mixtures" / f"{candidate}.csv"
        gcs_destination = f"{gcs_dir}/{candidate}.csv"
        copy_candidate(source, destination, gcs_destination if args.upload else None)
        source_paths.append(gcs_destination)

    selected["source_csv"] = source_paths
    selected.to_csv(args.output_dir / "selected_candidate_manifest.csv", index=False)
    excluded.to_csv(args.output_dir / "excluded_candidate_manifest.csv", index=False)
    (args.output_dir / "launch_mixtures.txt").write_text(" ".join(selected["candidate"]) + "\n")
    write_report(selected, excluded, args.output_dir)
    if args.upload:
        gcs_panel_dir = gcs_dir.removesuffix("/mixtures")
        with fsspec.open(f"{gcs_panel_dir}/selected_candidate_manifest.csv", "wt") as handle:
            selected.to_csv(handle, index=False)
        with fsspec.open(f"{gcs_panel_dir}/excluded_candidate_manifest.csv", "wt") as handle:
            excluded.to_csv(handle, index=False)
        with fsspec.open(f"{gcs_panel_dir}/report.md", "wt") as handle:
            handle.write((args.output_dir / "report.md").read_text())

    if selected["candidate"].duplicated().any():
        raise AssertionError("Selected candidate names must be unique")
    if not selected["max_aggregate_error"].le(1e-9).all():
        raise AssertionError("A selected candidate changed its fixed aggregate")
    print(selected.groupby(["objective", "anchor_tag", "family"]).size().to_string())
    print(f"Selected {len(selected)} candidates; excluded {len(excluded)}")
    print(f"Artifacts: {args.output_dir}")
    if args.upload:
        print(f"Uploaded mixtures: {gcs_dir}")
        print(f"Uploaded manifest: {gcs_dir.removesuffix('/mixtures')}/selected_candidate_manifest.csv")


if __name__ == "__main__":
    main()
