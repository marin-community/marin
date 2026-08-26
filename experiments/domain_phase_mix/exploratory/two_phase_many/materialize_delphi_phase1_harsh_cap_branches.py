# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["fsspec", "gcsfs", "pandas"]
# ///
"""Materialize a harsh-cap branch panel without opening sealed referee outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import cast

import fsspec
import pandas as pd

from experiments.domain_phase_mix import launch_delphi_3e18_phase0_prefix_replay as replay
from experiments.domain_phase_mix import launch_delphi_3e18_phase1_harsh_cap_branches as launch

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "delphi_phase1_harsh_cap_branch_results_20260825"
TARGET_PREFIX = "eval/uncheatable_eval/"
TARGET = "eval/uncheatable_eval/bpb"
TERMINAL_STEP = replay.EXPECTED_FULL_TRAIN_STEPS - 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--open-referee", action="store_true")
    args = parser.parse_args()
    if args.output_dir is None:
        if args.open_referee:
            parser.error("--output-dir is required with --open-referee to preserve the sealed materialization")
        args.output_dir = DEFAULT_OUTPUT_DIR
    return args


def read_json_lines(fs: fsspec.AbstractFileSystem, path: str) -> list[dict[str, object]]:
    with fs.open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def manifest_payload(experiment_root: str, expected_sha256: str) -> tuple[dict[str, object], bytes]:
    fs, root = fsspec.core.url_to_fs(experiment_root)
    candidates = set(fs.glob(os.path.join(root, "manifest-*", "manifest.json")))
    direct_path = os.path.join(root, "manifest.json")
    if fs.exists(direct_path):
        candidates.add(direct_path)
    matches = []
    for path in sorted(candidates):
        with fs.open(path, "rb") as handle:
            payload_bytes = handle.read()
        if hashlib.sha256(payload_bytes).hexdigest() == expected_sha256:
            matches.append((path, payload_bytes))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one branch manifest with SHA-256 {expected_sha256}; found {[path for path, _ in matches]}"
        )
    _, payload_bytes = matches[0]
    return json.loads(payload_bytes), payload_bytes


def expected_provenance(row: dict[str, object], manifest: dict[str, object]) -> dict[str, object]:
    prefix = cast(dict[str, object], row["prefix"])
    phase_weights = cast(dict[str, dict[str, float]], row["phase_weights"])
    prefix_hardware = launch.TpuHardware(**cast(dict[str, str], manifest["prefix_hardware"]))
    continuation_hardware = launch.TpuHardware(**cast(dict[str, str], manifest["continuation_hardware"]))
    return {
        "experiment_name": manifest["experiment_name"],
        "run_name": row["run_name"],
        "run_order": row["run_order"],
        "run_id": row["run_id"],
        "data_seed": row["data_seed"],
        "trainer_seed": row["trainer_seed"],
        "prefix_candidate_id": prefix["candidate_id"],
        "prefix_repeat_seed": prefix["repeat_seed"],
        "prefix_checkpoint_uri": prefix["checkpoint_uri"],
        "prefix_provenance_sha256": prefix["provenance_sha256"],
        "prefix_replay_code_commit": manifest["prefix_replay_code_commit"],
        "candidate_weights_sha256": manifest["candidate_weights_sha256"],
        "candidate_aliases_sha256": manifest["candidate_aliases_sha256"],
        "continuation_weights_sha256": manifest["continuation_weights_sha256"],
        "design_manifest_sha256": manifest["design_manifest_sha256"],
        "continuation_id": row["continuation_id"],
        "phase_weights_sha256": launch.phase_weights_sha256(phase_weights),
        "branch_code_commit": manifest["code_commit"],
        "prefix_hardware": manifest["prefix_hardware"],
        "continuation_hardware": manifest["continuation_hardware"],
        "minimum_initial_step": replay.EXPECTED_PREFIX_TRAIN_STEPS,
        "panel_hardware_status": launch.panel_hardware_status(prefix_hardware, continuation_hardware),
        "terminal_checkpoint_step": TERMINAL_STEP,
    }


def validate_provenance(
    provenance: dict[str, object],
    row: dict[str, object],
    manifest: dict[str, object],
    output_root: str,
) -> None:
    expected = expected_provenance(row, manifest)
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError(f"Branch provenance mismatch for {row['run_name']}: {key}={provenance.get(key)!r}")
    terminal_uri = f"{output_root}/checkpoints/step-{TERMINAL_STEP}"
    if provenance.get("terminal_checkpoint_uri") != terminal_uri:
        raise ValueError(f"Terminal checkpoint URI changed for {row['run_name']}")
    observed = provenance.get("observed_continuation_hardware")
    if (
        not isinstance(observed, dict)
        or "v6" not in str(observed.get("device_kind", "")).lower()
        or observed.get("global_device_count") != launch.EXPECTED_TPU_DEVICE_COUNT
        or observed.get("local_device_count") != launch.EXPECTED_TPU_DEVICE_COUNT
    ):
        raise ValueError(f"Branch did not report v6 hardware for {row['run_name']}: {observed}")


def terminal_metrics(fs: fsspec.AbstractFileSystem, output_path: str) -> dict[str, float]:
    path = os.path.join(output_path, "checkpoints", "eval_metrics.jsonl")
    if not fs.exists(path):
        raise FileNotFoundError(f"Terminal metrics are missing under {output_path}")
    records = [row for row in read_json_lines(fs, path) if int(row.get("step", -1)) == TERMINAL_STEP]
    if not records:
        raise ValueError(f"Expected a step-{TERMINAL_STEP} metric row under {output_path}")
    if any(record != records[0] for record in records[1:]):
        raise ValueError(f"Conflicting step-{TERMINAL_STEP} metric rows under {output_path}")
    record = records[0]
    if TARGET not in record or isinstance(record[TARGET], bool) or not isinstance(record[TARGET], (float, int)):
        raise ValueError(f"Terminal Uncheatable BPB is missing under {output_path}")
    return {
        key.removeprefix(TARGET_PREFIX).replace("/", "::"): float(value)
        for key, value in record.items()
        if key.startswith(TARGET_PREFIX) and not isinstance(value, bool) and isinstance(value, (float, int))
    }


def branch_rows(manifest: dict[str, object]) -> list[dict[str, object]]:
    rows = manifest.get("branch_rows")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("Branch manifest rows are malformed")
    return cast(list[dict[str, object]], rows)


def materialize(
    experiment_root: str,
    manifest: dict[str, object],
    *,
    open_referee: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    fs, root = fsspec.core.url_to_fs(experiment_root)
    scheme = experiment_root.split("://", maxsplit=1)[0]
    manifest_rows = branch_rows(manifest)
    rows_by_name = {str(row["run_name"]): row for row in manifest_rows}
    if len(rows_by_name) != len(manifest_rows):
        raise ValueError("Branch manifest repeats a run name")
    observed: set[str] = set()
    result_rows = []
    metric_rows = []
    referee_rows = []
    for provenance_path in sorted(fs.glob(os.path.join(root, "*", launch.BRANCH_PROVENANCE_FILENAME))):
        with fs.open(provenance_path, "rb") as handle:
            provenance_bytes = handle.read()
        provenance = json.loads(provenance_bytes)
        run_name = str(provenance.get("run_name"))
        if run_name not in rows_by_name:
            raise ValueError(f"Unexpected branch provenance row: {run_name}")
        if run_name in observed:
            raise ValueError(f"Duplicate branch provenance row: {run_name}")
        row = rows_by_name[run_name]
        output_path = str(PurePosixPath(provenance_path).parent)
        output_root = f"{scheme}://{output_path}"
        validate_provenance(provenance, row, manifest, output_root)
        identity = {
            "run_order": int(row["run_order"]),
            "run_id": int(row["run_id"]),
            "run_name": run_name,
            "prefix_candidate_id": str(row["prefix_candidate_id"]),
            "prefix_repeat_seed": int(row["prefix_repeat_seed"]),
            "continuation_id": str(row["continuation_id"]),
            "role": str(row["role"]),
            "fit_budget": bool(row["fit_budget"]),
            "data_seed": int(row["data_seed"]),
            "trainer_seed": int(row["trainer_seed"]),
            "source": str(row["source"]),
            "output_root": output_root,
            "provenance_sha256": hashlib.sha256(provenance_bytes).hexdigest(),
        }
        metrics = terminal_metrics(fs, output_path)
        if row["role"] == "sealed_geometry_referee" and not open_referee:
            referee_rows.append({**identity, "outcome_opened": False, "terminal_metrics_verified": True})
            observed.add(run_name)
            continue
        result_rows.append({**identity, **metrics})
        for metric, value in metrics.items():
            metric_rows.append({**identity, "metric": metric, "value": value})
        if row["role"] == "sealed_geometry_referee":
            referee_rows.append({**identity, "outcome_opened": True, "terminal_metrics_verified": True})
        observed.add(run_name)
    missing = sorted(set(rows_by_name) - observed)
    return pd.DataFrame(result_rows), pd.DataFrame(metric_rows), pd.DataFrame(referee_rows), missing


def main() -> None:
    args = parse_args()
    manifest, manifest_bytes = manifest_payload(args.experiment_root, args.expected_manifest_sha256)
    results, metrics, referees, missing = materialize(
        args.experiment_root,
        manifest,
        open_referee=args.open_referee,
    )
    if missing and not args.allow_incomplete:
        raise ValueError(f"Branch panel is incomplete: {len(missing)} missing rows")
    manifest_rows = branch_rows(manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (results.sort_values("run_order") if not results.empty else results).to_csv(
        args.output_dir / "branch_results.csv", index=False
    )
    (metrics.sort_values(["run_order", "metric"]) if not metrics.empty else metrics).to_csv(
        args.output_dir / "uncheatable_metrics_long.csv", index=False
    )
    (referees.sort_values("run_order") if not referees.empty else referees).to_csv(
        args.output_dir / "sealed_referee_inventory.csv", index=False
    )
    (args.output_dir / "missing_rows.json").write_text(json.dumps(missing, indent=2) + "\n")
    coverage = {
        "contract_version": "delphi_phase1_harsh_cap_branch_results_20260825_v1",
        "experiment_root": args.experiment_root,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "expected_rows": len(manifest_rows),
        "observed_rows": len(manifest_rows) - len(missing),
        "visible_result_rows": len(results),
        "sealed_referee_rows": int(sum(row["role"] == "sealed_geometry_referee" for row in manifest_rows)),
        "referee_outcomes_opened": args.open_referee,
        "missing_rows": len(missing),
        "status": "complete" if not missing else "provisional_incomplete",
    }
    (args.output_dir / "coverage.json").write_text(json.dumps(coverage, indent=2, sort_keys=True) + "\n")
    print(json.dumps(coverage, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
