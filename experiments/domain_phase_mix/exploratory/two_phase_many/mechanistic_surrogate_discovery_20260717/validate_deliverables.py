# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pandas>=2.2",
# ]
# ///

"""Validate the frozen gate and final mechanistic-surrogate deliverables."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
TWO_PHASE_MANY = SCRIPT_DIR.parent
OUTPUT_ROOT = TWO_PHASE_MANY / "reference_outputs" / "mechanistic_surrogate_discovery_20260717"
FROZEN_DIR = OUTPUT_ROOT / "frozen_gate"
FINAL_DIR = OUTPUT_ROOT / "final_synthesis"

REQUIRED_REGISTRY_COLUMNS = (
    "family",
    "premise",
    "latent_state",
    "state_transition",
    "response",
    "additional_degrees_of_freedom",
    "units_and_symmetries",
    "single_phase_restriction",
    "starcoder_signature",
    "optimism_resolution",
    "cheapest_falsification",
    "status",
    "status_evidence",
)
REQUIRED_NON_STATE_REGISTRY_COLUMNS = tuple(
    column for column in REQUIRED_REGISTRY_COLUMNS if column not in {"latent_state", "state_transition"}
)
REQUIRED_FINAL_FILES = (
    "final_report.md",
    "approach_registry.csv",
    "baseline_metrics.csv",
    "candidate_metrics.csv",
    "all_screen_metrics.csv",
    "acceptance_gate_evaluation.csv",
    "all_3e18_heldout_residuals.csv",
    "heldout_calibration_and_residuals.html",
    "starcoder_leave_region_out.html",
    "raw_optimum_support_paths.html",
    "trimmed_calibration.html",
    "trimmed_calibration_metrics.csv",
    "swarm_provenance_summary.csv",
    "frozen_metric_recomputation.csv",
    "delphi_3e18_uncheatable_worst_policy_exposures.html",
    "delphi_3e18_table9_worst_policy_exposures.html",
    "manifest.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_nonempty_table(path: Path, expected_rows: int | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        raise AssertionError(f"Expected a nonempty table: {path}")
    if expected_rows is not None and len(frame) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} rows in {path}, found {len(frame)}")
    return frame


def validate_frozen_gate(manifest: dict[str, object]) -> None:
    frozen_manifest_path = FROZEN_DIR / "frozen_manifest.json"
    frozen_manifest = json.loads(frozen_manifest_path.read_text())
    if sha256(frozen_manifest_path) != manifest["frozen_manifest_digest"]:
        raise AssertionError("Frozen-manifest digest changed after model search")
    for field, filename in (
        ("acceptance_gate_sha256", "acceptance_gate.json"),
        ("baseline_metrics_sha256", "baseline_metrics.csv"),
        ("calibration_bins_sha256", "calibration_bins.csv"),
    ):
        if sha256(FROZEN_DIR / filename) != frozen_manifest[field]:
            raise AssertionError(f"Frozen artifact changed: {filename}")


def validate_manifest_inputs(manifest: dict[str, object]) -> None:
    gate = json.loads((FROZEN_DIR / "acceptance_gate.json").read_text())
    sealed_tokens = tuple(gate["sealed_tokens_checked_absent"])
    inputs = manifest["inputs"]
    if not isinstance(inputs, dict):
        raise AssertionError("Manifest inputs must be a path-to-digest mapping")
    for relative, expected_digest in inputs.items():
        if any(token in relative for token in sealed_tokens):
            raise AssertionError(f"Sealed confirmatory input entered synthesis: {relative}")
        path = TWO_PHASE_MANY / relative
        if sha256(path) != expected_digest:
            raise AssertionError(f"Synthesis input digest mismatch: {relative}")


def validate_manifest_source(manifest: dict[str, object]) -> int:
    source = manifest["source_code"]
    if not isinstance(source, dict):
        raise AssertionError("Manifest source_code must be a path-to-digest mapping")
    expected_paths = {
        str(path.relative_to(TWO_PHASE_MANY)) for path in (*SCRIPT_DIR.glob("*.py"), *SCRIPT_DIR.glob("*.md"))
    }
    if set(source) != expected_paths:
        missing = sorted(expected_paths - set(source))
        extra = sorted(set(source) - expected_paths)
        raise AssertionError(f"Source manifest coverage mismatch: missing={missing}, extra={extra}")
    for relative, expected_digest in source.items():
        path = TWO_PHASE_MANY / relative
        if sha256(path) != expected_digest:
            raise AssertionError(f"Source-code digest mismatch: {relative}")
    return len(source)


def validate_manifest_outputs(manifest: dict[str, object]) -> int:
    outputs = manifest["outputs"]
    if not isinstance(outputs, dict):
        raise AssertionError("Manifest outputs must be a filename-to-digest mapping")
    actual_outputs = {path.name for path in FINAL_DIR.iterdir() if path.is_file() and path.name != "manifest.json"}
    if set(outputs) != actual_outputs:
        missing = sorted(actual_outputs - set(outputs))
        extra = sorted(set(outputs) - actual_outputs)
        raise AssertionError(f"Final output manifest coverage mismatch: missing={missing}, extra={extra}")
    for filename, expected_digest in outputs.items():
        path = FINAL_DIR / filename
        if sha256(path) != expected_digest:
            raise AssertionError(f"Final output digest mismatch: {filename}")
    return len(outputs)


def validate_registry(manifest: dict[str, object]) -> dict[str, int]:
    registry = assert_nonempty_table(FINAL_DIR / "approach_registry.csv", int(manifest["registry_rows"]))
    missing_columns = sorted(set(REQUIRED_REGISTRY_COLUMNS) - set(registry.columns))
    if missing_columns:
        raise AssertionError(f"Registry is missing required columns: {missing_columns}")
    for column in REQUIRED_NON_STATE_REGISTRY_COLUMNS:
        missing = registry[column].isna() | registry[column].astype(str).str.strip().eq("")
        if missing.any():
            raise AssertionError(f"Registry has {int(missing.sum())} undocumented rows in {column}")
    latent_missing = registry["latent_state"].isna() | registry["latent_state"].astype(str).str.strip().eq("")
    transition_missing = registry["state_transition"].isna() | registry["state_transition"].astype(str).str.strip().eq(
        ""
    )
    if (latent_missing & transition_missing).any():
        missing_ids = registry.loc[latent_missing & transition_missing, "id"].astype(str).tolist()
        raise AssertionError(f"Registry families lack both a dynamic transition and a static state: {missing_ids}")
    status = registry["status"].astype(str).str.lower()
    promoted = status.str.contains("promoted") & ~status.str.contains("rejected")
    active = status.eq("active")
    if promoted.any() or active.any():
        raise AssertionError("Final registry contains an unresolved active or promoted family")
    if int(manifest["promoted_candidates"]) != 0:
        raise AssertionError("Manifest unexpectedly reports a promoted candidate")
    status_counts = status.value_counts().to_dict()
    status_counts["dynamic_state_entries"] = int((~transition_missing).sum())
    status_counts["static_state_entries"] = int(transition_missing.sum())
    return status_counts


def validate_tables(manifest: dict[str, object]) -> dict[str, int]:
    for filename in REQUIRED_FINAL_FILES:
        path = FINAL_DIR / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise AssertionError(f"Required final deliverable is absent or empty: {path}")
    baseline = assert_nonempty_table(FINAL_DIR / "baseline_metrics.csv", int(manifest["baseline_metric_rows"]))
    candidates = assert_nonempty_table(FINAL_DIR / "candidate_metrics.csv", int(manifest["candidate_metric_rows"]))
    screens = assert_nonempty_table(FINAL_DIR / "all_screen_metrics.csv", int(manifest["all_screen_metric_rows"]))
    heldouts = assert_nonempty_table(FINAL_DIR / "all_3e18_heldout_residuals.csv")
    scorecard = assert_nonempty_table(FINAL_DIR / "acceptance_gate_evaluation.csv")
    provenance = assert_nonempty_table(FINAL_DIR / "swarm_provenance_summary.csv")
    if scorecard["all_required_gates_pass"].astype(bool).any():
        raise AssertionError("A candidate passes the frozen gate but the final verdict reports no promotion")
    if provenance["unexpected_overlap_rows"].astype(int).sum() != 0:
        raise AssertionError("Final provenance audit contains unexpected fit/evaluation coordinate overlap")
    datasets = set(heldouts["dataset"].astype(str))
    expected_datasets = {"delphi_3e18_table9", "delphi_3e18_uncheatable"}
    if datasets != expected_datasets:
        raise AssertionError(f"Unexpected frozen-heldout target coverage: {sorted(datasets)}")
    return {
        "baseline_metrics": len(baseline),
        "candidate_metrics": len(candidates),
        "screen_metrics": len(screens),
        "heldout_residuals": len(heldouts),
        "gate_rows": len(scorecard),
        "unexpected_overlap_rows": int(provenance["unexpected_overlap_rows"].astype(int).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-dir", type=Path, default=FINAL_DIR)
    args = parser.parse_args()
    if args.final_dir.resolve() != FINAL_DIR.resolve():
        raise ValueError("This validator is intentionally pinned to the frozen final-synthesis directory")

    manifest = json.loads((FINAL_DIR / "manifest.json").read_text())
    validate_frozen_gate(manifest)
    validate_manifest_inputs(manifest)
    source_files = validate_manifest_source(manifest)
    output_files = validate_manifest_outputs(manifest)
    statuses = validate_registry(manifest)
    table_counts = validate_tables(manifest)
    print(
        json.dumps(
            {
                "status": "valid",
                "frozen_manifest_digest": manifest["frozen_manifest_digest"],
                "registry_statuses": statuses,
                "source_files": source_files,
                "output_files": output_files,
                **table_counts,
                "promoted_candidates": manifest["promoted_candidates"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
