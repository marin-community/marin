# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "gcsfs",
#   "pandas",
#   "tabulate",
# ]
# ///
"""Recover native Table-9 evaluation targets from historical 60M checkpoints."""

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
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/60m_table9_historical_recovery_20260725"

EXPECTED_GAP_OBSERVATIONS = 52
EXPECTED_CHECKPOINT_STEP = 4576
GCS_PREFIX = "gs://marin-us-east5/"
HF_SUFFIX = f"hf/step-{EXPECTED_CHECKPOINT_STEP}"
SAFE_NAME = re.compile(r"[^a-zA-Z0-9_.-]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_gaps() -> pd.DataFrame:
    rows = []
    for filename, role in (
        ("heldout_observations.csv", "heldout"),
        ("repeat_observations.csv", "repeat_or_alias"),
    ):
        frame = pd.read_csv(AUDIT_DIR / filename, low_memory=False)
        frame = frame[frame["table9_macro_bpb"].isna()].copy()
        frame["coverage_role"] = role
        rows.append(frame)

    combined = pd.concat(rows, ignore_index=True)
    if len(combined) != EXPECTED_GAP_OBSERVATIONS:
        raise ValueError(f"Expected {EXPECTED_GAP_OBSERVATIONS} Table-9 gaps, found {len(combined)}")
    if combined["observation_id"].duplicated().any():
        raise ValueError("Table-9 recovery input contains duplicate observation IDs")
    if combined["checkpoint_root"].isna().any():
        missing = combined.loc[combined["checkpoint_root"].isna(), "run_name"].tolist()
        raise ValueError(f"Table-9 recovery input is missing checkpoint roots: {missing[:10]}")
    if combined["checkpoint_root"].duplicated().any():
        duplicates = combined.loc[combined["checkpoint_root"].duplicated(keep=False), "run_name"].tolist()
        raise ValueError(f"Table-9 recovery input contains duplicate checkpoint roots: {duplicates[:10]}")
    return combined


def inspect_checkpoint(row: pd.Series) -> dict[str, object]:
    checkpoint_root = str(row["checkpoint_root"]).rstrip("/")
    if not checkpoint_root.startswith(f"{GCS_PREFIX}checkpoints/"):
        raise ValueError(f"Checkpoint is not in the east5 checkpoint root: {checkpoint_root}")

    fs_root = checkpoint_root.removeprefix("gs://")
    prefix_relative_root = checkpoint_root.removeprefix(GCS_PREFIX)
    fs = fsspec.filesystem("gcs")
    hf_config = f"{fs_root}/{HF_SUFFIX}/config.json"
    eval_metrics = f"{fs_root}/checkpoints/eval_metrics.jsonl"
    checkpoint_metadata = fs.glob(f"{fs_root}/checkpoints/step-*/metadata.json")
    checkpoint_steps = sorted(
        int(match.group(1))
        for path in checkpoint_metadata
        if (match := re.search(r"/step-(\d+)/metadata\.json$", path)) is not None
    )
    hf_ready = fs.exists(hf_config)
    return {
        "observation_id": str(row["observation_id"]),
        "run_name": str(row["run_name"]),
        "source_experiment": str(row["source_experiment"]),
        "wandb_run_id": str(row["wandb_run_id"]),
        "coverage_role": str(row["coverage_role"]),
        "checkpoint_root": checkpoint_root,
        "hf_checkpoint": f"{prefix_relative_root}/{HF_SUFFIX}",
        "hf_config_exists": hf_ready,
        "eval_metrics_exists": fs.exists(eval_metrics),
        "available_training_steps": " ".join(map(str, checkpoint_steps)),
        "latest_training_step": max(checkpoint_steps, default=None),
        "recovery_status": "ready" if hf_ready else "blocked_missing_final_hf_export",
    }


def eval_name(index: int, run_name: str, checkpoint_root: str) -> str:
    slug = SAFE_NAME.sub("_", run_name).strip("_")[:42]
    digest = hashlib.sha256(checkpoint_root.encode()).hexdigest()[:8]
    return f"t9_60m_recovery_{index:03d}_{slug}_{digest}"


def build_manifest(ready: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for index, row in ready.reset_index(drop=True).iterrows():
        rows.append(
            {
                "panel": f"60m_39bucket_historical_{row['coverage_role']}",
                "scale": "60m_1p2b",
                "run_name": str(row["run_name"]),
                "eval_name": eval_name(index, str(row["run_name"]), str(row["checkpoint_root"])),
                "source_manifest": str(AUDIT_DIR),
                "source_experiment": str(row["source_experiment"]),
                "checkpoint_root": str(row["checkpoint_root"]),
                "checkpoint": str(row["hf_checkpoint"]),
                "expected_checkpoint_step": EXPECTED_CHECKPOINT_STEP,
                "method": f"historical_{row['coverage_role']}_recovery",
            }
        )
    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise ValueError("No recoverable Table-9 checkpoints were found")
    if manifest["eval_name"].duplicated().any() or manifest["checkpoint"].duplicated().any():
        raise ValueError("Generated duplicate Table-9 evaluation targets")
    if not manifest["checkpoint"].str.startswith("checkpoints/").all():
        raise ValueError("Generated checkpoints must be prefix-relative paths under checkpoints/")
    return manifest


def main() -> None:
    args = parse_args()
    gaps = load_gaps()
    with ThreadPoolExecutor(max_workers=24) as executor:
        inventory = pd.DataFrame(executor.map(inspect_checkpoint, (row for _, row in gaps.iterrows())))
    inventory = inventory.sort_values(["recovery_status", "run_name"]).reset_index(drop=True)
    ready = inventory[inventory["recovery_status"].eq("ready")].copy()
    blocked = inventory[~inventory["recovery_status"].eq("ready")].copy()
    manifest = build_manifest(ready)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = args.output_dir / "recovery_inventory.csv"
    blocked_path = args.output_dir / "blocked_checkpoints.csv"
    manifest_path = args.output_dir / "table9_recovery_manifest.csv"
    inventory.to_csv(inventory_path, index=False)
    blocked.to_csv(blocked_path, index=False)
    manifest.to_csv(manifest_path, index=False)
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    summary = {
        "audit_dir": str(AUDIT_DIR),
        "blocked_count": len(blocked),
        "expected_checkpoint_step": EXPECTED_CHECKPOINT_STEP,
        "gap_observation_count": len(gaps),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "ready_count": len(ready),
        "role_counts": manifest["method"].value_counts().sort_index().to_dict(),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    blocked_summary = (
        blocked[["run_name", "latest_training_step", "recovery_status"]].to_markdown(index=False)
        if not blocked.empty
        else "None."
    )
    report = f"""# 60M historical Table-9 checkpoint recovery

- Audited missing observations: **{len(gaps)}**.
- Verified final HF exports at step {EXPECTED_CHECKPOINT_STEP}: **{len(ready)}**.
- Blocked observations: **{len(blocked)}**.
- Materialized manifest SHA-256: `{manifest_sha256}`.

The manifest contains only east5 checkpoints with an object-level verified
`hf/step-{EXPECTED_CHECKPOINT_STEP}/config.json`. Blocked rows are not silently evaluated at a
different training step because that would confound the same-compute 60M comparison.

## Blocked checkpoints

{blocked_summary}
"""
    (args.output_dir / "report.md").write_text(report)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
