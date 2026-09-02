# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Endpoint-blind acceptance audit for the detached WSD80 gradient-probe canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import fsspec

from experiments.domain_phase_mix import starcoder_wsd80_gradient_probe as probe
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    freeze_starcoder_wsd80_gradient_probe_20260816 as freeze,
)

CANARY_EXPECTED = {
    "probe": {"rows": 112, "groups": 14},
    "optimizer": {"rows": 42, "groups": 6},
    "rollout": {"rows": 14, "groups": 2},
}
NUMERICAL_TOLERANCE = 1e-6
SHA256_LENGTH = 64


def _trajectory_id(row: Mapping[str, Any]) -> str:
    return str(row.get("trajectory_id", row.get("parent_trajectory_id")))


def _checkpoint_step(row: Mapping[str, Any]) -> int:
    return int(row.get("checkpoint_step", row.get("parent_checkpoint_step")))


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == SHA256_LENGTH and all(character in "0123456789abcdef" for character in text)


def _assert_equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise RuntimeError(f"{label}: observed {observed!r}, expected {expected!r}")


def _assert_runtime_summary(
    document: Mapping[str, Any],
    row: Mapping[str, Any],
    pod_config: Any,
) -> None:
    checkpoint_step = _checkpoint_step(row)
    expected_state_step = int(row["expected_restored_state_step"])
    _assert_equal(expected_state_step, checkpoint_step + 1, "checkpoint next-step convention")
    _assert_equal(
        str(row["checkpoint_uri"]).rstrip("/").rsplit("/", 1)[-1],
        f"step-{checkpoint_step}",
        "checkpoint URI label",
    )
    checkpoint_metadata = document["checkpoint_metadata"]
    _assert_equal(int(checkpoint_metadata["step"]), checkpoint_step, "checkpoint metadata step")
    _assert_equal(checkpoint_metadata["is_temporary"], False, "checkpoint permanence")
    _assert_equal(int(document["restored_state_step"]), expected_state_step, "document restored state step")

    runtime = document["runtime_summary"]
    restoration = runtime["restoration"]
    _assert_equal(int(restoration["checkpoint_label_step"]), checkpoint_step, "restoration label step")
    _assert_equal(
        int(restoration["expected_restored_state_step"]), expected_state_step, "restoration expected state step"
    )
    _assert_equal(int(restoration["trainer_state_step"]), expected_state_step, "trainer state step")
    _assert_equal(restoration["trainer_state_step_matches_expected"], True, "trainer state match")
    _assert_equal(restoration["optimizer_counter_matches_expected"], True, "optimizer counter match")
    _assert_equal(restoration["allow_partial_checkpoint"], False, "partial checkpoint restore")
    optimizer_counters = {int(value) for value in restoration["optimizer_step_counters"].values()}
    if expected_state_step not in optimizer_counters:
        raise RuntimeError("Expected restored step is absent from optimizer counters")

    train_config = pod_config.train_config
    expected_sequence_offset = train_config.trainer.batch_schedule.global_data_offset_by_step(expected_state_step)
    source_stream = runtime["source_stream"]
    _assert_equal(int(source_stream["restored_state_step"]), expected_state_step, "source continuation state step")
    _assert_equal(
        int(source_stream["global_sequence_offset"]), expected_sequence_offset, "source continuation sequence offset"
    )
    _assert_equal(
        source_stream["on_policy_stream_rule"],
        "continue_exact_per_source_logical_offset",
        "on-policy source continuation rule",
    )
    _assert_equal(source_stream["step_schedule_rescaled_to_sequences"], True, "sequence-rescaled schedule")
    _assert_equal(set(source_stream["logical_component_offsets"]), set(freeze.TRAINING_COMPONENTS), "source offsets")
    _assert_equal(set(source_stream["source_sequence_counts"]), set(freeze.TRAINING_COMPONENTS), "source counts")
    if any(int(value) < 0 for value in source_stream["logical_component_offsets"].values()):
        raise RuntimeError("Source continuation contains a negative component offset")

    schedule = runtime["optimizer_schedule"]
    configured = train_config.optimizer_schedule_num_train_steps
    effective = train_config.trainer.num_train_steps if configured is None else configured
    _assert_equal(schedule["configured_num_train_steps"], configured, "configured optimizer horizon")
    _assert_equal(int(schedule["effective_num_train_steps"]), effective, "effective optimizer horizon")
    _assert_equal(int(schedule["trainer_num_train_steps"]), train_config.trainer.num_train_steps, "trainer horizon")
    _assert_equal(schedule["matches_frozen_training_horizon"], True, "frozen optimizer horizon")

    projection = runtime["muon_projection"]
    _assert_equal(projection["muon_projection_active"], True, "Muon projection activation")
    if int(projection["muon_parameter_leaf_count"]) <= 0 or int(projection["muon_layer_count"]) <= 0:
        raise RuntimeError("Muon projection did not cover parameter leaves and transformer layers")
    _assert_equal(projection["muon_matrix_axis_counts"], [2], "Muon matrix geometry")


def _assert_numerical_summary(document: Mapping[str, Any], row: Mapping[str, Any]) -> None:
    summary = document["numerical_summary"]
    if float(summary["repeat_gradient_max_abs_difference"]) > NUMERICAL_TOLERANCE:
        raise RuntimeError("Repeated gradient calculation exceeded the numerical tolerance")
    if float(summary["repeat_loss_absolute_difference"]) > NUMERICAL_TOLERANCE:
        raise RuntimeError("Repeated loss calculation exceeded the numerical tolerance")
    if not _is_sha256(summary["first_batch_sha256"]):
        raise RuntimeError("First-batch identity is not a SHA-256 digest")
    if document["kind"] == "gradient_probe":
        expected_blocks = int(row["replicate_blocks"])
        expected_draws = min(int(row["optimizer_update_draw_count"]), expected_blocks // 2)
    else:
        expected_draws = int(row["optimizer_update_draw_count"])
        expected_blocks = expected_draws * 2
    _assert_equal(int(summary["replicate_block_count"]), expected_blocks, "probe block count")
    _assert_equal(int(summary["optimizer_update_draw_count"]), expected_draws, "optimizer update draw count")
    _assert_equal(
        int(summary["data_supply"]["required_sequence_count"]),
        expected_blocks * probe.PROBE_BATCH_SIZE,
        "probe sequence count",
    )


def _assert_rollout(document: Mapping[str, Any], row: Mapping[str, Any]) -> None:
    expected_state_step = int(row["expected_restored_state_step"])
    updates = int(row["updates"])
    _assert_equal(int(document["final_state_step"]), expected_state_step + updates, "rollout final state step")
    expected_readouts = sorted({int(value) for value in str(row["readout_steps"]).split("|")})
    observed_readouts = [int(readout["updates"]) for readout in document["readouts"]]
    _assert_equal(observed_readouts, expected_readouts, "rollout readout schedule")


def _projection_difference_count(document: Mapping[str, Any]) -> int:
    containers: list[Mapping[str, Any]] = []
    if document["kind"] == "gradient_probe":
        containers.extend(document["pairwise_statistics"].values())
    elif document["kind"] == "optimizer_transform":
        containers.extend(document["target_utility_statistics"].values())
    count = 0
    for container in containers:
        for raw_name, projected_name in (
            ("raw_gradient", "projected_gradient"),
            ("raw_optimizer_update", "projected_optimizer_update"),
        ):
            if container[raw_name] != container[projected_name]:
                count += 1
    return count


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _inventory_entry(fs: Any, plain_path: str, payload: bytes) -> dict[str, Any]:
    generation = fs.info(plain_path).get("generation")
    if generation is None:
        raise RuntimeError(f"GCS object exposes no immutable generation: {plain_path}")
    return {
        "path": plain_path,
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "generation": str(generation),
    }


def _write_snapshot(path: Path, entries: list[dict[str, Any]], report: Mapping[str, Any]) -> dict[str, Any]:
    entries = sorted(entries, key=lambda item: item["path"])
    snapshot = {
        "schema_version": "2026-08-16-gradient-probe-canary-inventory-v1",
        "science.endpoint_metrics_read": False,
        "entries": entries,
        "inventory_sha256": hashlib.sha256(_canonical_json(entries)).hexdigest(),
        "acceptance_report": report,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json(snapshot) + b"\n")
    return snapshot


def _assert_idempotent_snapshot(snapshot: Mapping[str, Any], baseline: Mapping[str, Any]) -> None:
    _assert_equal(snapshot["entries"], baseline["entries"], "idempotent replay output inventory")
    _assert_equal(snapshot["inventory_sha256"], baseline["inventory_sha256"], "idempotent replay inventory hash")


def audit_canary(snapshot_path: Path, baseline_path: Path | None, expected_release_sha256: str) -> dict[str, Any]:
    built_in = probe.audit_outputs("canary")
    pod_configs = probe._pod_configs("canary")
    release = probe._load_release(expected_release_sha256)
    entries: list[dict[str, Any]] = []
    maximum_repeat_gradient_difference = 0.0
    maximum_repeat_loss_difference = 0.0
    projection_difference_count = 0
    optimizer_aware_rows = 0

    for kind, expected in CANARY_EXPECTED.items():
        inventory = built_in[kind]
        _assert_equal(inventory["expected_rows"], expected["rows"], f"{kind} expected rows")
        _assert_equal(inventory["found_rows"], expected["rows"], f"{kind} found rows")
        _assert_equal(inventory["complete_groups"], expected["groups"], f"{kind} complete groups")
        for key in (
            "missing_rows",
            "unexpected_row_objects",
            "duplicate_manifest_rows",
            "identity_mismatches",
            "invalid_documents",
            "nonfinite_documents",
            "missing_group_markers",
            "unexpected_group_markers",
            "invalid_group_markers",
        ):
            _assert_equal(inventory[key], 0, f"{kind} {key}")

        rows = probe._read_manifest(f"canary_{kind}_manifest.csv")
        base_uri = probe._path_join(freeze.RESULT_ROOT, "canary", kind)
        fs, base_path = fsspec.core.url_to_fs(base_uri)
        for row in rows:
            plain_path = f"{base_path}/{row['group_id']}/{probe.ARTIFACT_VERSION}/rows/{row['row_id']}.json"
            with fs.open(plain_path, "rb") as handle:
                payload = handle.read()
            document = json.loads(payload)
            _assert_equal(document["release_sha256"], release["release_sha256"], "row release")
            _assert_equal(document["endpoint_metrics_read"], False, "row endpoint access")
            _assert_equal(row["scientific_inference_allowed"], "False", "canary inference prohibition")
            _assert_equal(row["endpoint_metrics_read"], "False", "manifest endpoint access")
            _assert_runtime_summary(document, row, pod_configs[_trajectory_id(row)])
            if kind in {"probe", "optimizer"}:
                _assert_numerical_summary(document, row)
                summary = document["numerical_summary"]
                maximum_repeat_gradient_difference = max(
                    maximum_repeat_gradient_difference, float(summary["repeat_gradient_max_abs_difference"])
                )
                maximum_repeat_loss_difference = max(
                    maximum_repeat_loss_difference, float(summary["repeat_loss_absolute_difference"])
                )
                optimizer_aware_rows += int(summary["optimizer_update_draw_count"] > 0)
                projection_difference_count += _projection_difference_count(document)
            else:
                _assert_rollout(document, row)
            entries.append(_inventory_entry(fs, plain_path, payload))

        group_ids = sorted({row["group_id"] for row in rows})
        for group_id in group_ids:
            plain_path = f"{base_path}/{group_id}/{probe.ARTIFACT_VERSION}/group_complete.json"
            with fs.open(plain_path, "rb") as handle:
                payload = handle.read()
            marker = json.loads(payload)
            _assert_equal(marker["endpoint_metrics_read"], False, "group endpoint access")
            entries.append(_inventory_entry(fs, plain_path, payload))

    if optimizer_aware_rows <= 0:
        raise RuntimeError("Canary produced no optimizer-aware counterfactual rows")
    if projection_difference_count <= 0:
        raise RuntimeError("Muon projection was active but never changed a recorded statistic")
    if not math.isfinite(maximum_repeat_gradient_difference) or not math.isfinite(maximum_repeat_loss_difference):
        raise RuntimeError("Determinism checks produced a non-finite value")

    report = {
        "scope": "canary",
        "release_sha256": release["release_sha256"],
        "endpoint_metrics_read": False,
        "scientific_inference_allowed": False,
        "counts": CANARY_EXPECTED,
        "optimizer_aware_rows": optimizer_aware_rows,
        "projection_difference_count": projection_difference_count,
        "maximum_repeat_gradient_difference": maximum_repeat_gradient_difference,
        "maximum_repeat_loss_difference": maximum_repeat_loss_difference,
        "numerical_tolerance": NUMERICAL_TOLERANCE,
        "gcs_object_generations_pinned": True,
        "acceptance": "pass",
    }
    snapshot = _write_snapshot(snapshot_path, entries, report)
    if baseline_path is not None:
        baseline = json.loads(baseline_path.read_text())
        _assert_idempotent_snapshot(snapshot, baseline)
        report["idempotent_replay"] = "pass"
    return {**report, "inventory_sha256": snapshot["inventory_sha256"], "inventory_entries": len(entries)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--release-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(
        json.dumps(
            audit_canary(args.snapshot, args.baseline, args.release_sha256),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
