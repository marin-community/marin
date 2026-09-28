# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect mechanical before/after evidence for TPP40 bridge idempotence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

from marin.execution.step_status import STATUS_SUCCESS, StatusFile

from experiments.domain_phase_mix import analyze_delphi_tpp40_bridge_acceptance as acceptance
from experiments.domain_phase_mix import launch_delphi_tpp40_bridge_same_region_east5_eval as same_region_eval
from experiments.domain_phase_mix import launch_delphi_tpp40_bridge_uncheatable_eval as bridge_eval

DEFAULT_BEFORE_PATH = bridge_eval.REFERENCE_DIR / "bridge_idempotence_before_v3.json"
DEFAULT_EVIDENCE_PATH = bridge_eval.REFERENCE_DIR / "bridge_idempotence_evidence_v3.json"
JOB_STATE_SUCCEEDED = 4
EXPECTED_OUTPUT_UNITS = {
    "training": len(bridge_eval.BRIDGE_RUN_ORDERS),
    "uncheatable": len(bridge_eval.BRIDGE_RUN_ORDERS) * len(bridge_eval.CHECKPOINT_STEPS),
    "table9": len(bridge_eval.BRIDGE_RUN_ORDERS),
}
EXPECTED_INVENTORY_UNITS = {
    "east5": {**EXPECTED_OUTPUT_UNITS, "mirror": 3},
    "europe": {**EXPECTED_OUTPUT_UNITS, "mirror": 0},
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _successful_output_inventory(path_manifest: dict[str, Any]) -> dict[str, Any]:
    same_region_eval.audit_east5_reference_mirror()
    for side_name, side in path_manifest["sides"].items():
        output_paths = list(side["training_output_paths"])
        output_paths.extend(cell["output_path"] for cell in side["uncheatable_cells"])
        output_paths.extend(cell["output_path"] for cell in side["table9_cells"])
        expected_count = sum(EXPECTED_OUTPUT_UNITS.values())
        if len(output_paths) != expected_count:
            raise ValueError(f"Expected {expected_count} frozen {side_name} outputs, got {len(output_paths)}")
        for output_path in output_paths:
            status = StatusFile(output_path, worker_id="tpp40-bridge-idempotence-audit").status
            if status != STATUS_SUCCESS:
                raise ValueError(f"Frozen bridge output is not successful: {output_path} ({status})")
    return acceptance.result_inventory(path_manifest)


def before_snapshot(path_manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "captured_at_ms": time.time_ns() // 1_000_000,
        "acceptance_contract_sha256": bridge_eval.EXPECTED_CONTRACT_SHA256,
        "path_manifest_sha256": acceptance.EXPECTED_PATH_MANIFEST_SHA256,
        "evaluation_audit_sha256": acceptance.EXPECTED_EVALUATION_AUDIT_SHA256,
        "east5_reference_mirror_manifest_sha256": same_region_eval.EXPECTED_MIRROR_MANIFEST_SHA256,
        "result_inventory": _successful_output_inventory(path_manifest),
    }


def _sql_literal(value: str) -> str:
    if not value.startswith("/") or "\n" in value or "\r" in value:
        raise ValueError(f"Invalid Iris job ID {value!r}")
    return "'" + value.replace("'", "''") + "'"


def query_job_tree(job_id: str) -> list[dict[str, str]]:
    literal = _sql_literal(job_id)
    sql = (
        "SELECT j.job_id,j.parent_job_id,j.state,j.submitted_at_ms,j.finished_at_ms,j.exit_code,"
        "j.num_tasks,"
        f"(SELECT COUNT(*) FROM tasks t WHERE t.job_id=j.job_id AND t.state={JOB_STATE_SUCCEEDED}) "
        "AS succeeded_task_count,"
        f"(SELECT COUNT(*) FROM tasks t WHERE t.job_id=j.job_id AND t.state={JOB_STATE_SUCCEEDED} "
        "AND t.exit_code=0) AS zero_exit_succeeded_task_count,"
        "c.name,c.entrypoint_json,c.bundle_id,c.submit_argv_json "
        "FROM jobs j JOIN job_config c USING(job_id) "
        f"WHERE j.job_id={literal} OR j.parent_job_id={literal} ORDER BY j.job_id"
    )
    completed = subprocess.run(
        ["iris", "--cluster=marin", "query", "-f", "csv", sql],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = list(csv.DictReader(io.StringIO(completed.stdout)))
    if not rows:
        raise ValueError(f"Iris has no record for rerun parent {job_id}")
    return rows


def _split_at_marker(tokens: list[str]) -> tuple[list[str], list[str]]:
    try:
        marker = tokens.index("--")
    except ValueError as error:
        raise ValueError("Frozen launch command lacks the Iris command marker") from error
    return tokens[:marker], tokens[marker + 1 :]


def _expected_rerun_command(command_path: Path, *, rerun_job_id: str) -> tuple[list[str], list[str], str]:
    encoded = command_path.read_bytes()
    tokens = shlex.split(encoded.decode())
    _, iris_arguments = _split_at_marker(tokens)
    outer_arguments, inner_arguments = _split_at_marker(iris_arguments)
    expected_outer = list(outer_arguments)
    job_name_index = expected_outer.index("--job-name") + 1
    expected_outer[job_name_index] = rerun_job_id.rsplit("/", 1)[-1]
    return expected_outer, inner_arguments, _sha256_bytes(encoded)


def _submitted_arguments(row: dict[str, str]) -> tuple[list[str], list[str]]:
    submit_argv = json.loads(row["submit_argv_json"])
    if not isinstance(submit_argv, list) or not all(isinstance(item, str) for item in submit_argv):
        raise ValueError("Iris submit_argv_json is malformed")
    try:
        job_index = next(
            index for index in range(len(submit_argv) - 1) if submit_argv[index : index + 2] == ["job", "run"]
        )
    except StopIteration as error:
        raise ValueError("Iris submission lacks `job run`") from error
    return _split_at_marker(submit_argv[job_index + 2 :])


def validate_rerun_job(
    rows: list[dict[str, str]],
    *,
    job_id: str,
    command_path: Path,
    captured_at_ms: int,
) -> dict[str, Any]:
    parent_rows = [row for row in rows if row["job_id"] == job_id]
    if len(parent_rows) != 1:
        raise ValueError(f"Expected one Iris parent row for {job_id}, got {len(parent_rows)}")
    children = [row for row in rows if row["parent_job_id"] == job_id]
    if children:
        raise ValueError(f"Idempotence rerun {job_id} submitted {len(children)} child jobs")
    parent = parent_rows[0]
    if int(parent["state"]) != JOB_STATE_SUCCEEDED:
        raise ValueError(f"Idempotence rerun {job_id} did not succeed")
    # Iris leaves jobs.exit_code null for an ordinary successful exit. Verify
    # the concrete parent task instead of treating that nullable summary as an
    # integer; the job state alone would otherwise hide a malformed task row.
    if parent["exit_code"] not in {"", "0"}:
        raise ValueError(f"Idempotence rerun {job_id} has a nonzero job exit code")
    if int(parent["num_tasks"]) != 1:
        raise ValueError(f"Idempotence rerun {job_id} did not have exactly one parent task")
    if int(parent["succeeded_task_count"]) != 1 or int(parent["zero_exit_succeeded_task_count"]) != 1:
        raise ValueError(f"Idempotence rerun {job_id} lacks one successful zero-exit parent task")
    submitted_at_ms = int(parent["submitted_at_ms"])
    if submitted_at_ms <= captured_at_ms:
        raise ValueError(f"Idempotence rerun {job_id} predates the before snapshot")
    if not parent["finished_at_ms"]:
        raise ValueError(f"Idempotence rerun {job_id} lacks a completion time")

    expected_outer, expected_inner, command_sha256 = _expected_rerun_command(
        command_path,
        rerun_job_id=job_id,
    )
    observed_outer, observed_inner = _submitted_arguments(parent)
    if observed_outer != expected_outer:
        raise ValueError(f"Idempotence rerun {job_id} changed its Iris launch envelope")
    if observed_inner != expected_inner:
        raise ValueError(f"Idempotence rerun {job_id} changed its launcher arguments")
    entrypoint = json.loads(parent["entrypoint_json"])
    if entrypoint.get("run_command", {}).get("argv") != expected_inner:
        raise ValueError(f"Idempotence rerun {job_id} stored a different runtime entrypoint")
    bundle_id = parent["bundle_id"]
    if len(bundle_id) != 64:
        raise ValueError(f"Idempotence rerun {job_id} lacks a content-addressed bundle")
    return {
        "job_id": job_id,
        "state": "succeeded",
        "exit_code": 0,
        "raw_job_exit_code": int(parent["exit_code"]) if parent["exit_code"] else None,
        "parent_task_count": 1,
        "successful_zero_exit_parent_task_count": 1,
        "submitted_at_ms": submitted_at_ms,
        "finished_at_ms": int(parent["finished_at_ms"]),
        "child_job_count": 0,
        "bundle_id": bundle_id,
        "frozen_command_sha256": command_sha256,
        "entrypoint_sha256": _sha256_bytes(parent["entrypoint_json"].encode()),
        "submit_argv_sha256": _sha256_bytes(parent["submit_argv_json"].encode()),
    }


def after_evidence(
    *,
    path_manifest: dict[str, Any],
    before: dict[str, Any],
    before_sha256: str,
    east5_reference_eval_rerun_job_id: str,
    europe_training_rerun_job_id: str,
    europe_uncheatable_rerun_job_id: str,
) -> dict[str, Any]:
    if before.get("schema_version") != 2:
        raise ValueError("Idempotence before snapshot schema changed")
    if before.get("acceptance_contract_sha256") != bridge_eval.EXPECTED_CONTRACT_SHA256:
        raise ValueError("Idempotence before snapshot refers to the wrong contract")
    if before.get("path_manifest_sha256") != acceptance.EXPECTED_PATH_MANIFEST_SHA256:
        raise ValueError("Idempotence before snapshot refers to the wrong path manifest")
    if before.get("evaluation_audit_sha256") != acceptance.EXPECTED_EVALUATION_AUDIT_SHA256:
        raise ValueError("Idempotence before snapshot refers to the wrong evaluation audit")
    if before.get("east5_reference_mirror_manifest_sha256") != same_region_eval.EXPECTED_MIRROR_MANIFEST_SHA256:
        raise ValueError("Idempotence before snapshot refers to the wrong East5 reference mirror")
    captured_at_ms = before.get("captured_at_ms")
    if isinstance(captured_at_ms, bool) or not isinstance(captured_at_ms, int):
        raise ValueError("Idempotence before snapshot lacks a valid capture time")
    before_inventory = before.get("result_inventory")
    if not isinstance(before_inventory, dict):
        raise ValueError("Idempotence before snapshot lacks a result inventory")
    east5_reference_eval_rerun = validate_rerun_job(
        query_job_tree(east5_reference_eval_rerun_job_id),
        job_id=east5_reference_eval_rerun_job_id,
        command_path=acceptance.COMMAND_FILES["east5"]["reference_eval"],
        captured_at_ms=captured_at_ms,
    )
    europe_training_rerun = validate_rerun_job(
        query_job_tree(europe_training_rerun_job_id),
        job_id=europe_training_rerun_job_id,
        command_path=acceptance.COMMAND_FILES["europe"]["training"],
        captured_at_ms=captured_at_ms,
    )
    europe_uncheatable_rerun = validate_rerun_job(
        query_job_tree(europe_uncheatable_rerun_job_id),
        job_id=europe_uncheatable_rerun_job_id,
        command_path=acceptance.COMMAND_FILES["europe"]["uncheatable"],
        captured_at_ms=captured_at_ms,
    )
    reruns = (
        east5_reference_eval_rerun,
        europe_training_rerun,
        europe_uncheatable_rerun,
    )
    after_inventory_started_at_ms = time.time_ns() // 1_000_000
    for rerun in reruns:
        if rerun["finished_at_ms"] > after_inventory_started_at_ms:
            raise ValueError(f"Idempotence rerun {rerun['job_id']} finished after the inventory audit began")
    after_inventory = _successful_output_inventory(path_manifest)
    after_inventory_captured_at_ms = time.time_ns() // 1_000_000

    sides: dict[str, Any] = {}
    for side_name in bridge_eval.BRIDGE_SIDES:
        before_side = before_inventory["sides"][side_name]
        after_side = after_inventory["sides"][side_name]
        if before_side["inventory_sha256"] != after_side["inventory_sha256"]:
            raise ValueError(f"{side_name} bridge result inventory changed across the idempotence rerun")
        common = {
            "result_inventory_sha256_before": before_side["inventory_sha256"],
            "result_inventory_sha256_after": after_side["inventory_sha256"],
            "completed_output_unit_counts": after_side["unit_counts"],
        }
        if side_name == "east5":
            sides[side_name] = {
                **common,
                "reference_eval_command_sha256": east5_reference_eval_rerun["frozen_command_sha256"],
                "reference_eval_rerun": east5_reference_eval_rerun,
                "mirror_manifest_sha256": same_region_eval.EXPECTED_MIRROR_MANIFEST_SHA256,
            }
        else:
            sides[side_name] = {
                **common,
                "training_command_sha256": europe_training_rerun["frozen_command_sha256"],
                "uncheatable_command_sha256": europe_uncheatable_rerun["frozen_command_sha256"],
                "training_rerun": europe_training_rerun,
                "uncheatable_rerun": europe_uncheatable_rerun,
            }
    return {
        "schema_version": 3,
        "captured_at_ms": time.time_ns() // 1_000_000,
        "before_snapshot_sha256": before_sha256,
        "acceptance_contract_sha256": bridge_eval.EXPECTED_CONTRACT_SHA256,
        "path_manifest_sha256": acceptance.EXPECTED_PATH_MANIFEST_SHA256,
        "evaluation_audit_sha256": acceptance.EXPECTED_EVALUATION_AUDIT_SHA256,
        "east5_reference_mirror_manifest_sha256": same_region_eval.EXPECTED_MIRROR_MANIFEST_SHA256,
        "after_inventory_started_at_ms": after_inventory_started_at_ms,
        "after_inventory_captured_at_ms": after_inventory_captured_at_ms,
        "sides": sides,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("before", "after"))
    parser.add_argument("--path-manifest", type=Path, default=acceptance.PATH_MANIFEST_PATH)
    parser.add_argument("--before", type=Path, default=DEFAULT_BEFORE_PATH)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--east5-reference-eval-rerun")
    parser.add_argument("--europe-training-rerun")
    parser.add_argument("--europe-uncheatable-rerun")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    path_manifest = acceptance._load_frozen_path_manifest(args.path_manifest)
    if args.mode == "before":
        output = args.output or args.before
        payload = before_snapshot(path_manifest)
    else:
        if not all(
            (
                args.east5_reference_eval_rerun,
                args.europe_training_rerun,
                args.europe_uncheatable_rerun,
            )
        ):
            raise ValueError("After mode requires the East5 reference and both Europe rerun parent job IDs")
        before_encoded = args.before.read_bytes()
        payload = after_evidence(
            path_manifest=path_manifest,
            before=json.loads(before_encoded),
            before_sha256=_sha256_bytes(before_encoded),
            east5_reference_eval_rerun_job_id=args.east5_reference_eval_rerun,
            europe_training_rerun_job_id=args.europe_training_rerun,
            europe_uncheatable_rerun_job_id=args.europe_uncheatable_rerun,
        )
        output = args.output or DEFAULT_EVIDENCE_PATH
    _write_json(output, payload)
    print(_sha256_bytes(output.read_bytes()))


if __name__ == "__main__":
    main()
