# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["fsspec[gcs]"]
# ///

"""Prove that the frozen East TPP40 graph resolves to its existing lineages."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import fsspec
from rigging.filesystem import marin_prefix_for_region

from experiments.domain_phase_mix import launch_delphi_augmented_swarm_3e18 as base
from experiments.domain_phase_mix import launch_delphi_augmented_swarm_tpp40 as tpp40
from experiments.domain_phase_mix import launch_delphi_tpp40_bridge_uncheatable_eval as bridge_eval
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import executor_status_succeeded

RUN_ORDER_PATTERN = re.compile(r"/fit_(?P<order>\d{3})_")


def _read_json(path: str) -> dict[str, Any]:
    with fsspec.open(path, "rt") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return payload


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with fsspec.open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_exists(path: str) -> bool:
    fs, _, paths = fsspec.get_fs_token_paths(path)
    return fs.exists(paths[0])


def _read_executor_status(output_path: str) -> tuple[str | None, str | None]:
    status_path = f"{output_path.rstrip('/')}/.executor_status"
    if not _path_exists(status_path):
        return None, None
    with fsspec.open(status_path, "rt") as handle:
        status = handle.read()
    return status, hashlib.sha256(status.encode()).hexdigest()


def audit_resolved_training_paths(
    *,
    assignment: dict[str, Any],
    paths_by_order: dict[int, str],
    expected_root: str,
) -> dict[str, Any]:
    assignments = assignment.get("assignments")
    if not isinstance(assignments, dict):
        raise ValueError("Assignment lacks an assignments object")
    completed = set(assignments.get("completed", []))
    east5 = set(assignments.get("east5", []))
    resumable = set(assignments.get("resumable_east5", []))
    expected_orders = completed | east5
    if set(paths_by_order) != expected_orders:
        missing = sorted(expected_orders - set(paths_by_order))
        unexpected = sorted(set(paths_by_order) - expected_orders)
        raise ValueError(f"Resolved East orders changed: missing={missing}, unexpected={unexpected}")
    if len(set(paths_by_order.values())) != len(paths_by_order):
        raise ValueError("Resolved East training paths are not unique")

    records: list[dict[str, Any]] = []
    for run_order in sorted(paths_by_order):
        output_path = paths_by_order[run_order]
        if not output_path.startswith(f"{expected_root.rstrip('/')}/"):
            raise ValueError(f"Run {run_order} resolved outside the East production root: {output_path}")
        match = RUN_ORDER_PATTERN.search(f"/{output_path}")
        if match is None or int(match.group("order")) != run_order:
            raise ValueError(f"Run-order/path mismatch for {run_order}: {output_path}")

        final_marker = f"{output_path.rstrip('/')}/hf/step-{tpp40.EXPECTED_FINAL_CHECKPOINT_STEP}/model.safetensors"
        phase0_checkpoint = f"{output_path.rstrip('/')}/checkpoints/step-{tpp40.EXPECTED_PHASE0_CHECKPOINT_STEP}"
        final_exists = _path_exists(final_marker)
        status, status_sha256 = _read_executor_status(output_path)
        phase0_metadata = bridge_eval._checkpoint_metadata(
            phase0_checkpoint,
            expected_step=tpp40.EXPECTED_PHASE0_CHECKPOINT_STEP,
        )

        classification = "fresh"
        if run_order in completed:
            classification = "completed"
            if not final_exists:
                raise ValueError(f"Completed run {run_order} lacks its final marker at resolved path {output_path}")
            if status is None or not executor_status_succeeded(status):
                raise ValueError(f"Completed run {run_order} lacks SUCCESS status at resolved path {output_path}")
        elif run_order in resumable:
            classification = "resumable"
            if final_exists:
                raise ValueError(f"Resumable run {run_order} unexpectedly has a final marker at {output_path}")
            if phase0_metadata is None:
                raise ValueError(f"Resumable run {run_order} lacks phase-0 checkpoint at {output_path}")
        elif final_exists or phase0_metadata is not None or (status is not None and executor_status_succeeded(status)):
            raise ValueError(f"Fresh run {run_order} has resumable or completed artifacts at {output_path}")

        records.append(
            {
                "run_order": run_order,
                "classification": classification,
                "output_path": output_path,
                "output_path_sha256": hashlib.sha256(output_path.encode()).hexdigest(),
                "executor_status_succeeded": status is not None and executor_status_succeeded(status),
                "executor_status_sha256": status_sha256,
                "phase0_checkpoint_present": phase0_metadata is not None,
                "phase0_metadata_sha256": None if phase0_metadata is None else phase0_metadata[1],
                "final_marker_present": final_exists,
            }
        )

    path_payload_sha256 = hashlib.sha256(
        json.dumps(
            [(record["run_order"], record["output_path"]) for record in records],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return {
        "passed": True,
        "expected_root": expected_root,
        "resolved_count": len(records),
        "completed_count": sum(record["classification"] == "completed" for record in records),
        "resumable_count": sum(record["classification"] == "resumable" for record in records),
        "fresh_count": sum(record["classification"] == "fresh" for record in records),
        "resolved_path_payload_sha256": path_payload_sha256,
        "records": records,
    }


def _resolve_east_training_paths(
    *,
    assignment_file: str,
    expected_assignment_sha256: str,
) -> tuple[dict[str, Any], dict[int, str]]:
    side = bridge_eval.BRIDGE_SIDES["east5"]
    prefix = marin_prefix_for_region(side.region)
    os.environ["MARIN_PREFIX"] = prefix
    source_panel = tpp40._regional_input_path(base.DEFAULT_SOURCE_PANEL, region=side.region)
    analysis_output_path = tpp40._regional_input_path(base.DEFAULT_ANALYSIS_OUTPUT_PATH, region=side.region)
    all_specs, _ = tpp40.build_run_specs(
        source_panel=source_panel,
        analysis_output_path=analysis_output_path,
        tpu_type=side.training_tpu_type,
        tpu_region=side.region,
        tpu_zone=side.training_zone,
    )
    selected_orders, _ = tpp40._assignment_orders(
        assignment_file,
        "east5",
        tpu_region=side.region,
        experiment_name=tpp40.EXPERIMENT_NAME,
        expected_assignment_sha256=expected_assignment_sha256,
    )
    run_specs = [all_specs[run_order] for run_order in selected_orders]
    full_validation_configs, _ = bridge_eval._validation_configs()
    training_paths = bridge_eval._original_training_paths(
        side=side,
        run_specs=run_specs,
        source_panel=source_panel,
        analysis_output_path=analysis_output_path,
        full_validation_configs=full_validation_configs,
        prefix=prefix,
    )
    paths_by_order = {
        run_spec.run_order: output_path for run_spec, output_path in zip(run_specs, training_paths, strict=True)
    }
    return _read_json(assignment_file), paths_by_order


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment-file", required=True)
    parser.add_argument("--expect-assignment-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    assignment, paths_by_order = _resolve_east_training_paths(
        assignment_file=args.assignment_file,
        expected_assignment_sha256=args.expect_assignment_sha256,
    )
    payload = {
        "schema_version": 1,
        "assignment_file": args.assignment_file,
        "assignment_file_sha256": _sha256_file(args.assignment_file),
        "assignment_sha256": args.expect_assignment_sha256,
        "audit": audit_resolved_training_paths(
            assignment=assignment,
            paths_by_order=paths_by_order,
            expected_root=(f"{marin_prefix_for_region('us-east5').rstrip('/')}/{tpp40.EXPERIMENT_NAME}"),
        ),
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    payload_sha256 = hashlib.sha256(encoded.encode()).hexdigest()
    final_payload = {**payload, "payload_sha256": payload_sha256}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(final_payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
