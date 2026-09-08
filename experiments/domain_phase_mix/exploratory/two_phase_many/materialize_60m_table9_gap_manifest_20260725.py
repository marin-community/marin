# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "gcsfs",
#   "pandas",
# ]
# ///
"""Materialize native Table-9 evaluation gaps for the audited 60M archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fsspec
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
AUDIT_DIR = SCRIPT_DIR / "reference_outputs/60m_39bucket_checkpoint_audit_20260724"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/60m_table9_gap_completion_20260725"

EXPECTED_CHECKPOINT_STEP = 4576
EXPECTED_TARGETS = 309
EXPECTED_ROLE_COUNTS = {
    "fit_single_phase": 241,
    "fit_two_phase": 1,
    "heldout": 56,
    "heldout_single_phase_olmix": 1,
    "repeat_noise": 10,
}
GCS_PREFIX = "gs://marin-us-east5/"
CHECKPOINT_SUFFIX = f"hf/step-{EXPECTED_CHECKPOINT_STEP}"
OLMIX_RUN_NAME = "singleavg_baseline_olmix_loglinear"
SAFE_NAME = re.compile(r"[^a-zA-Z0-9_.-]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--verify-checkpoints",
        action="store_true",
        help="Check that every HF config.json exists in the east5 bucket.",
    )
    return parser.parse_args()


def _load(name: str, role: str) -> pd.DataFrame:
    frame = pd.read_csv(AUDIT_DIR / name, low_memory=False)
    frame = frame.loc[frame["table9_macro_bpb"].isna() & frame["checkpoint_root"].notna()].copy()
    frame["coverage_role"] = role
    return frame


def _eval_name(index: int, run_name: str, checkpoint_root: str) -> str:
    slug = SAFE_NAME.sub("_", run_name).strip("_")[:44]
    digest = hashlib.sha256(checkpoint_root.encode()).hexdigest()[:8]
    return f"t9_60m_gap_{index:03d}_{slug}_{digest}"


def build_manifest() -> pd.DataFrame:
    fit_two = _load("fit_two_phase.csv", "fit_two_phase")
    fit_one = _load("fit_single_phase.csv", "fit_single_phase")
    fit_one.loc[fit_one["run_name"].eq(OLMIX_RUN_NAME), "coverage_role"] = "heldout_single_phase_olmix"
    heldout = _load("heldout_observations.csv", "heldout")
    repeats = _load("repeat_observations.csv", "repeat_noise")

    combined = pd.concat([fit_two, fit_one, heldout, repeats], ignore_index=True)
    combined = combined.drop_duplicates("checkpoint_root", keep="first").reset_index(drop=True)
    if len(combined) != EXPECTED_TARGETS:
        raise ValueError(f"Expected {EXPECTED_TARGETS} unique checkpoint roots, found {len(combined)}")
    role_counts = combined["coverage_role"].value_counts().sort_index().to_dict()
    if role_counts != EXPECTED_ROLE_COUNTS:
        raise ValueError(f"Unexpected coverage-role counts: {role_counts}")

    rows: list[dict[str, object]] = []
    for index, row in combined.iterrows():
        checkpoint_root = str(row["checkpoint_root"]).rstrip("/")
        if not checkpoint_root.startswith(f"{GCS_PREFIX}checkpoints/"):
            raise ValueError(f"Checkpoint is not in the east5 checkpoint root: {checkpoint_root}")
        checkpoint = f"{checkpoint_root.removeprefix(GCS_PREFIX)}/{CHECKPOINT_SUFFIX}"
        role = str(row["coverage_role"])
        rows.append(
            {
                "panel": f"60m_39bucket_{role}",
                "scale": "60m_1p2b",
                "run_name": str(row["run_name"]),
                "eval_name": _eval_name(index, str(row["run_name"]), checkpoint_root),
                "source_manifest": str(AUDIT_DIR),
                "source_experiment": str(row["source_experiment"]),
                "checkpoint_root": checkpoint_root,
                "checkpoint": checkpoint,
                "expected_checkpoint_step": EXPECTED_CHECKPOINT_STEP,
                "method": role,
            }
        )
    result = pd.DataFrame(rows)
    if result["eval_name"].duplicated().any():
        raise ValueError("Generated duplicate Table-9 evaluation names")
    if result["checkpoint"].duplicated().any():
        raise ValueError("Generated duplicate Table-9 checkpoints")
    return result


def verify_checkpoints(frame: pd.DataFrame) -> None:
    config_paths = [f"{GCS_PREFIX}{path}/config.json" for path in frame["checkpoint"]]

    def exists(path: str) -> tuple[str, bool]:
        fs, _, paths = fsspec.get_fs_token_paths(path)
        return path, fs.exists(paths[0])

    with ThreadPoolExecutor(max_workers=32) as executor:
        checks = list(executor.map(exists, config_paths))
    missing = [path for path, present in checks if not present]
    if missing:
        raise ValueError(f"{len(missing)} HF checkpoints are missing config.json; first five: {missing[:5]}")


def main() -> None:
    args = parse_args()
    manifest = build_manifest()
    if args.verify_checkpoints:
        verify_checkpoints(manifest)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "table9_gap_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    summary = {
        "audit_dir": str(AUDIT_DIR),
        "checkpoint_step": EXPECTED_CHECKPOINT_STEP,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "role_counts": manifest["method"].value_counts().sort_index().to_dict(),
        "target_count": len(manifest),
        "verified_hf_checkpoints": bool(args.verify_checkpoints),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
