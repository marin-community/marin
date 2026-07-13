# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "fsspec",
#   "gcsfs",
#   "pandas",
#   "tabulate",
# ]
# ///
"""Select and upload the new low-epsilon decoupled phase-information panel."""

from __future__ import annotations

import argparse
from pathlib import Path

import fsspec
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = SCRIPT_DIR / "reference_outputs" / "decoupled_phase_information_low_epsilon_paths_20260712"
DEFAULT_PRIOR_MANIFEST = (
    SCRIPT_DIR
    / "reference_outputs"
    / "decoupled_phase_information_validation_panel_20260712"
    / "selected_candidate_manifest.csv"
)
DEFAULT_OUTPUT_DIR = (
    SCRIPT_DIR / "reference_outputs" / "decoupled_phase_information_low_epsilon_validation_panel_20260712"
)
DEFAULT_GCS_OUTPUT_DIR = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "delphi_decoupled_phase_information_low_epsilon_validation_20260712/mixtures"
)
SELECTED_BUDGETS = frozenset({0.001, 0.0025, 0.0075})
SELECTED_FAMILIES = frozenset({"effective_exposure", "separate_heads"})


def copy_candidate(source: Path, destination: Path, gcs_destination: str | None) -> None:
    frame = pd.read_csv(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    if gcs_destination is not None:
        with fsspec.open(gcs_destination, "wt") as handle:
            frame.to_csv(handle, index=False)


def select_candidates(source: pd.DataFrame, prior: pd.DataFrame) -> pd.DataFrame:
    selected = source.loc[
        source["family"].isin(SELECTED_FAMILIES) & source["phase_information_budget"].isin(SELECTED_BUDGETS)
    ].copy()
    overlap = set(selected["candidate"]).intersection(prior["candidate"])
    if overlap:
        raise ValueError(f"Low-epsilon panel overlaps prior validation: {sorted(overlap)}")
    selected["selection_reason"] = "new low-epsilon point between tied and previously validated path points"
    return selected.sort_values(["objective", "anchor_tag", "family", "phase_information_budget"])


def write_report(selected: pd.DataFrame, output_dir: Path) -> None:
    summary = (
        selected.groupby(["objective", "anchor_tag", "family"], as_index=False)
        .agg(
            candidates=("candidate", "size"),
            min_epsilon=("phase_information_budget", "min"),
            max_epsilon=("phase_information_budget", "max"),
            min_phase_tv=("phase_tv", "min"),
            max_phase_tv=("phase_tv", "max"),
            max_simulated_epoch=("max_simulated_epoch", "max"),
        )
        .sort_values(["objective", "anchor_tag", "family"])
    )
    lines = [
        "# Low-epsilon decoupled phase-information validation panel",
        "",
        "This panel fills the unvalidated part of the fixed-aggregate phase-information paths. "
        "Epsilon 0.005 and 0.01 were already validated and are intentionally not repeated.",
        "",
        "- Selected epsilon values: 0.001, 0.0025, 0.0075.",
        "- Families: effective exposure and separate heads.",
        "- Aggregate anchors: Uncheatable KL=0.05, Table-9 KL=0.05, and Table-9 KL=0.075.",
        "- One training seed per candidate; native Table-9 evaluation is required for every checkpoint.",
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        f"Selected candidates: {len(selected)}",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--prior-manifest", type=Path, default=DEFAULT_PRIOR_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gcs-output-dir", default=DEFAULT_GCS_OUTPUT_DIR)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    source = pd.read_csv(args.source_dir / "candidate_manifest.csv")
    prior = pd.read_csv(args.prior_manifest)
    selected = select_candidates(source, prior)
    if len(selected) != 18:
        raise ValueError(f"Expected 18 low-epsilon candidates, found {len(selected)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gcs_dir = args.gcs_output_dir.rstrip("/")
    source_paths = []
    for candidate in selected["candidate"]:
        source_path = args.source_dir / "mixtures" / f"{candidate}.csv"
        local_destination = args.output_dir / "mixtures" / f"{candidate}.csv"
        gcs_destination = f"{gcs_dir}/{candidate}.csv"
        copy_candidate(source_path, local_destination, gcs_destination if args.upload else None)
        source_paths.append(gcs_destination)

    selected["source_csv"] = source_paths
    selected.to_csv(args.output_dir / "selected_candidate_manifest.csv", index=False)
    write_report(selected, args.output_dir)
    if args.upload:
        gcs_panel_dir = gcs_dir.removesuffix("/mixtures")
        with fsspec.open(f"{gcs_panel_dir}/selected_candidate_manifest.csv", "wt") as handle:
            selected.to_csv(handle, index=False)
        with fsspec.open(f"{gcs_panel_dir}/report.md", "wt") as handle:
            handle.write((args.output_dir / "report.md").read_text())

    print(selected.groupby(["objective", "anchor_tag", "family"]).size().to_string())
    print(f"Selected {len(selected)} candidates")
    print(f"Artifacts: {args.output_dir}")
    if args.upload:
        print(f"Uploaded mixtures: {gcs_dir}")


if __name__ == "__main__":
    main()
