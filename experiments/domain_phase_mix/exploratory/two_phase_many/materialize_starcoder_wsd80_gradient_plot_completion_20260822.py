# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "gcsfs>=2025.5.1",
#   "pandas>=2.2",
# ]
# ///
"""Merge plot-completion probes with the immutable v10 descriptive tables."""

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import gcsfs
import pandas as pd

from experiments.domain_phase_mix import starcoder_wsd80_gradient_plot_completion as runtime
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    analyze_starcoder_wsd80_gradient_mechanism_repair_20260818 as analysis,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    freeze_starcoder_wsd80_gradient_plot_completion_20260822 as freeze,
)

BASE_RESULTS_DIR = freeze.BASE_RESULTS_DIR
BASE_SOURCE_GEOMETRY_PATH = freeze.BASE_SOURCE_GEOMETRY_PATH
DEFAULT_OUTPUT_DIR = freeze.COMPLETE_TABLES_DIR
TARGET_UTILITIES_OUTPUT = "target_source_utilities_visualization_only.csv"
TARGET_ALIGNMENT_OUTPUT = "target_source_choice_alignment_visualization_only.csv"
MAX_WORKERS = 64
SUMMARY_FILES = (
    "source_source_geometry.csv",
    "h2_h3_summary.csv",
    "h3_repetition_mechanism_summary.csv",
    "h5_profile_summary.csv",
)


def _read_manifest() -> list[dict[str, str]]:
    with freeze.FULL_MANIFEST_PATH.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _read_document(
    fs: gcsfs.GCSFileSystem,
    row: Mapping[str, str],
    release_sha256: str,
) -> dict[str, Any]:
    uri = "/".join(
        (
            freeze.RESULT_ROOT,
            "full",
            row["group_id"],
            runtime.ARTIFACT_VERSION,
            "rows",
            f"{row['row_id']}.json",
        )
    )
    with fs.open(uri.removeprefix("gs://"), "rb") as handle:
        document = json.load(handle)
    if document.get("schema_version") != runtime.SCHEMA_VERSION:
        raise RuntimeError(f"Plot-completion schema drifted: {uri}")
    if document.get("payload_sha256") != freeze.canonical_sha256({**document, "payload_sha256": ""}):
        raise RuntimeError(f"Plot-completion payload hash drifted: {uri}")
    if (
        document.get("release_sha256") != release_sha256
        or document.get("row") != row
        or document.get("identity_sha256") != runtime.mechanism._row_identity(row, release_sha256)
        or document.get("endpoint_metrics_read") is not False
    ):
        raise RuntimeError(f"Plot-completion identity or endpoint-blindness drifted: {uri}")
    document["source_uri"] = uri
    return document


def _load_documents(release_sha256: str) -> list[dict[str, Any]]:
    rows = _read_manifest()
    runtime._configure_mechanism_runtime(release_sha256)
    fs = gcsfs.GCSFileSystem()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        return list(executor.map(lambda row: _read_document(fs, row, release_sha256), rows))


def _apply_completion_provenance(frame: pd.DataFrame, documents: list[dict[str, Any]]) -> pd.DataFrame:
    display_roles = {document["row"]["row_id"]: document["row"]["display_analysis_role"] for document in documents}
    completion_roles = {document["row"]["row_id"]: document["row"]["completion_role"] for document in documents}
    result = frame.copy()
    if not result.empty:
        result["original_analysis_role"] = result["analysis_role"]
        result["display_analysis_role"] = result["row_id"].map(display_roles)
        result["completion_role"] = result["row_id"].map(completion_roles)
        if result[["display_analysis_role", "completion_role"]].isna().any().any():
            raise RuntimeError("Completion table contains a row without a display analysis role")
        result["analysis_role"] = result["display_analysis_role"]
        result["evidence_role"] = "Post-outcome plot completion"
        result["visualization_only"] = True
    return result


def _apply_base_provenance(frame: pd.DataFrame, *, evidence_role: str) -> pd.DataFrame:
    result = frame.copy()
    result["original_analysis_role"] = result["analysis_role"]
    result["display_analysis_role"] = result["analysis_role"]
    result["completion_role"] = ""
    result["evidence_role"] = evidence_role
    result["visualization_only"] = True
    return result


def _assert_unique(frame: pd.DataFrame, keys: list[str], *, label: str) -> None:
    duplicates = frame[frame.duplicated(keys, keep=False)]
    if not duplicates.empty:
        raise RuntimeError(f"{label} contains duplicate logical rows: {duplicates[keys].head().to_dict('records')}")


def _historical_runtime_overlap_audit(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare newly computed common source geometry with overlapping v10 rows."""
    completion = analysis.flatten_h1(documents)
    completion = completion[
        completion["policy_role"].eq(freeze.COMMON_POLICY)
        & completion["checkpoint_label"].isin(freeze.COMMON_TARGET_STATES)
    ].copy()
    base = pd.read_csv(BASE_SOURCE_GEOMETRY_PATH)
    base = base[
        base["policy_role"].eq(freeze.COMMON_POLICY) & base["checkpoint_label"].isin(freeze.COMMON_TARGET_STATES)
    ].copy()
    keys = ["trajectory_id", "checkpoint_label", "statistic", "geometry", "component"]
    _assert_unique(completion, keys, label="Completion overlap geometry")
    _assert_unique(base, keys, label="v10 overlap geometry")
    paired = completion.merge(base, on=keys, suffixes=("_completion", "_v10"), how="outer", indicator=True)
    if not paired["_merge"].eq("both").all():
        raise RuntimeError(f"Historical-runtime overlap keys drifted: {paired['_merge'].value_counts().to_dict()}")
    comparisons = 0
    max_absolute_difference: dict[str, float] = {}
    for metric in ("cosine", "dot", "left_norm", "right_norm"):
        observed = pd.to_numeric(paired[f"{metric}_completion"], errors="coerce")
        expected = pd.to_numeric(paired[f"{metric}_v10"], errors="coerce")
        mismatched_missingness = observed.isna() ^ expected.isna()
        if mismatched_missingness.any():
            raise RuntimeError(f"Historical-runtime overlap {metric} missingness drifted")
        present = ~(observed.isna() | expected.isna())
        differences = (observed[present] - expected[present]).abs()
        tolerances = 5e-6 * pd.concat(
            [observed[present].abs(), expected[present].abs(), pd.Series(1.0, index=observed[present].index)],
            axis=1,
        ).max(axis=1)
        if (differences > tolerances).any():
            worst = int((differences - tolerances).idxmax())
            raise RuntimeError(
                f"Historical-runtime overlap {metric} failed at {paired.loc[worst, keys].to_dict()}: "
                f"{observed.loc[worst]} != {expected.loc[worst]}"
            )
        max_absolute_difference[metric] = float(differences.max()) if len(differences) else 0.0
        comparisons += int(present.sum())
    observed_defined = paired["cosine_defined_completion"].astype("boolean")
    expected_defined = paired["cosine_defined_v10"].astype("boolean")
    if not observed_defined.equals(expected_defined):
        raise RuntimeError("Historical-runtime overlap cosine-defined flags drifted")
    comparisons += len(paired)
    return {
        "comparison_rows": len(paired),
        "scalar_comparisons": comparisons,
        "max_absolute_difference": max_absolute_difference,
        "relative_tolerance": 5e-6,
        "passed": True,
    }


def _merge_source_geometry(documents: list[dict[str, Any]]) -> pd.DataFrame:
    base = pd.read_csv(BASE_SOURCE_GEOMETRY_PATH)
    frozen_h1_ids = set(pd.read_csv(BASE_RESULTS_DIR / "source_source_geometry.csv", usecols=["row_id"])["row_id"])
    base = _apply_base_provenance(base, evidence_role="Post-outcome v10 repair state")
    base.loc[base["row_id"].isin(frozen_h1_ids), "evidence_role"] = "v10 H1 contract state"
    completion = analysis.flatten_h1(documents)
    completion = completion[
        completion["policy_role"].isin(freeze.H5_POLICIES)
        & completion["checkpoint_label"].isin(freeze.H5_TARGET_STATES | freeze.H5_SOURCE_ONLY_STATES)
    ].copy()
    completion = _apply_completion_provenance(completion, documents)
    merged = pd.concat([base, completion], ignore_index=True, sort=False)
    keys = ["trajectory_id", "checkpoint_label", "statistic", "geometry", "component"]
    _assert_unique(merged, keys, label="Merged source geometry")
    expected = {
        freeze.COMMON_POLICY: 12,
        "boundary_beta_0p60": 11,
        "boundary_beta_0p85": 11,
    }
    observed = merged.groupby("policy_role")["checkpoint_label"].nunique().to_dict()
    if set(observed) != set(expected):
        raise RuntimeError(
            f"Merged source geometry contains unexpected policy roles: {sorted(set(observed) - set(expected))}"
        )
    if any(observed.get(policy) != count for policy, count in expected.items()):
        raise RuntimeError(f"Merged source-state coverage drifted: {observed} != {expected}")
    return merged


def _merge_target_tables(documents: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_utilities = _apply_base_provenance(
        pd.read_csv(BASE_RESULTS_DIR / "target_source_utilities.csv"),
        evidence_role="Post-outcome v10 repair state",
    )
    base_alignment = _apply_base_provenance(
        pd.read_csv(BASE_RESULTS_DIR / "target_source_choice_alignment.csv"),
        evidence_role="Post-outcome v10 repair state",
    )
    completion_utilities = _apply_completion_provenance(analysis.flatten_utilities(documents), documents)
    completion_alignment = _apply_completion_provenance(analysis.flatten_alignment(documents), documents)
    utilities = pd.concat([base_utilities, completion_utilities], ignore_index=True, sort=False)
    alignment = pd.concat([base_alignment, completion_alignment], ignore_index=True, sort=False)
    _assert_unique(
        utilities,
        ["trajectory_id", "checkpoint_label", "target", "source", "geometry", "component"],
        label="Merged target-source utilities",
    )
    _assert_unique(
        alignment,
        ["trajectory_id", "checkpoint_label", "target", "contrast", "geometry", "component"],
        label="Merged target-source choice alignment",
    )
    expected = {
        freeze.COMMON_POLICY: 11,
        "boundary_beta_0p60": 10,
        "boundary_beta_0p85": 10,
    }
    utility_states = utilities.groupby("policy_role")["checkpoint_label"].nunique().to_dict()
    alignment_states = alignment.groupby("policy_role")["checkpoint_label"].nunique().to_dict()
    if set(utility_states) != set(expected) or set(alignment_states) != set(expected):
        raise RuntimeError("Merged target-source tables contain unexpected policy roles")
    if any(utility_states.get(policy) != count for policy, count in expected.items()):
        raise RuntimeError(f"Merged utility-state coverage drifted: {utility_states} != {expected}")
    if any(alignment_states.get(policy) != count for policy, count in expected.items()):
        raise RuntimeError(f"Merged alignment-state coverage drifted: {alignment_states} != {expected}")
    return utilities, alignment


def materialize(output_dir: Path, *, expected_release_sha256: str) -> None:
    release = json.loads(freeze.RELEASE_PATH.read_text())
    if release["release_sha256"] != expected_release_sha256:
        raise RuntimeError("Materialization release identity does not match the operator-supplied hash")
    runtime._load_release(expected_release_sha256)
    runtime.audit(release)
    documents = _load_documents(release["release_sha256"])
    if len(documents) != release["manifests"]["full"]["row_count"]:
        raise RuntimeError("Plot-completion document inventory is incomplete")
    overlap_audit = _historical_runtime_overlap_audit(documents)
    source_geometry = _merge_source_geometry(documents)
    utilities, alignment = _merge_target_tables(documents)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    for name in SUMMARY_FILES:
        shutil.copy2(BASE_RESULTS_DIR / name, staging / name)
        expected_sha256 = release["plot_inputs"][f"v10_results/{name}"]["sha256"]
        if hashlib.sha256((staging / name).read_bytes()).hexdigest() != expected_sha256:
            raise RuntimeError(f"Frozen v10 summary changed during copy: {name}")
    source_geometry.to_csv(staging / "source_source_geometry_all_states.csv", index=False)
    utilities.to_csv(staging / TARGET_UTILITIES_OUTPUT, index=False)
    alignment.to_csv(staging / TARGET_ALIGNMENT_OUTPUT, index=False)
    audit = {
        "completion_documents": len(documents),
        "endpoint_metrics_read": False,
        "frozen_inferential_tables_copied_unchanged": list(SUMMARY_FILES),
        "historical_runtime_overlap_audit": overlap_audit,
        "release_sha256": release["release_sha256"],
        "scientific_status": freeze.SCIENTIFIC_STATUS,
        "tables_are_visualization_only": True,
        "evidence_roles": {
            "frozen_h1": "v10 H1 contract state",
            "v10": "Post-outcome v10 repair state",
            "completion": "Post-outcome plot completion",
        },
        "source_geometry_rows": len(source_geometry),
        "target_source_alignment_rows": len(alignment),
        "target_source_utility_rows": len(utilities),
        "structural_missingness": freeze.ANALYSIS_CONTRACT["structural_missingness"],
    }
    (staging / "materialization_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    (staging / "README.md").write_text(
        "# Gradient plot completion tables\n\n"
        "These merged tables are visualization-only. They combine immutable v10 descriptive rows with "
        "post-outcome saved-checkpoint plot completion. The `original_analysis_role`, `display_analysis_role`, "
        "`completion_role`, and `evidence_role` columns preserve provenance. Do not pass the merged tables to "
        "the frozen v10 inferential analyzer or use them to revise H1-H5 inference. Their `_visualization_only.csv` "
        "filenames make this boundary explicit.\n\n"
        f"Release: `{release['release_sha256']}`\n"
    )
    analysis._publish_create_only(staging, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--release-sha256", required=True)
    args = parser.parse_args()
    materialize(args.output_dir, expected_release_sha256=args.release_sha256)


if __name__ == "__main__":
    main()
