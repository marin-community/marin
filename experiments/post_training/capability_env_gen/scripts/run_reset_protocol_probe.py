#!/usr/bin/env python3
"""Remote-only five-trial no-tool reset protocol probe with retained evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

from capability_pipeline.non_docker_reset import run_frozen_non_docker_reset
from capability_pipeline.synthesis import OfficialToolchain

DOCKER_IMAGE = "python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea"
RESOURCE_REQUEST = {"cpu": 2, "memory_gb": 2, "disk_gb": 10}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _calibrate(item: Path, output: Path) -> dict:
    """One separate task-bound Daytona start for fixture policy calibration."""
    from capability_pipeline.reset_diagnostics import validate_policy_and_build_plan
    from scripts import run_reset_conformance as reset

    plan_payload, _ = validate_policy_and_build_plan(
        item / "workspace/task/reset-policy.json", item / "harbor",
        item / "workspace/task/resource-receipt.json",
    )
    plan = reset.validate_plan_payload(plan_payload)
    adapter = reset.DaytonaResetAdapter()
    evidence = {
        "schema_version": "capability-reset-protocol-calibration-v1",
        "task_binding": plan_payload["task_binding"],
        "resource_profile": plan_payload["resource_profile"],
        "status": "running",
    }
    sandbox = None
    try:
        snapshot = adapter.ensure_snapshot(plan)
        evidence["snapshot"] = snapshot
        sandbox, attempts = adapter.create_sandbox(snapshot)
        evidence["sandbox_id"] = sandbox.id
        evidence["provisioning_attempts"] = attempts
        adapter.start_task(sandbox, plan)
        evidence["task_startup"] = "completed"
        readiness = adapter.run(sandbox, plan.readiness_command, plan.readiness_timeout_seconds)
        evidence["readiness"] = {"exit": readiness.get("exit"), "timed_out": readiness.get("timed_out") is True}
        if evidence["readiness"] != {"exit": 0, "timed_out": False}:
            raise RuntimeError("calibration readiness failed")
        probe = adapter.run(sandbox, reset._probe_command(plan.public_root), 60)
        evidence["probe_exit"] = probe.get("exit")
        evidence["probe_timed_out"] = probe.get("timed_out") is True
        if evidence["probe_exit"] != 0 or evidence["probe_timed_out"]:
            raise RuntimeError("calibration public-state probe failed")
        observed = json.loads(probe.get("stdout", ""))
        if not isinstance(observed, dict) or set(observed) != {
            "files", "symlinks", "process_comm_counts", "environment_names"
        } or observed["symlinks"]:
            raise ValueError("calibration observation is incomplete")
        evidence["observation"] = observed
        evidence["status"] = "observed"
    except Exception as error:  # noqa: BLE001 - preserve provider failure taxonomy.
        evidence["status"] = "pending_infrastructure"
        evidence["error_type"] = type(error).__name__
    finally:
        if sandbox is None:
            evidence["cleanup"] = {"attempted": False, "verified_absent": False}
        else:
            try:
                evidence["cleanup"] = adapter.delete(sandbox)
            except Exception as error:  # noqa: BLE001 - retain deletion failure.
                evidence["cleanup"] = {"attempted": True, "verified_absent": False, "error_type": type(error).__name__}
        if evidence["cleanup"].get("verified_absent") is not True:
            evidence["status"] = "pending_infrastructure"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    return evidence


def _docker_policy(observed: dict) -> dict:
    counts = observed["process_comm_counts"]
    names = observed["environment_names"]
    if (not isinstance(counts, dict) or not counts or
            any(type(count) is not int or count <= 0 for count in counts.values()) or
            not isinstance(names, list) or not names or any(not isinstance(name, str) for name in names)):
        raise ValueError("calibration lacks a complete process/environment profile")
    return {
        "schema_version": "capability-reset-policy-v1",
        "public_root": "/app",
        "readiness": {"command": "test -f /app/challenge.txt", "timeout_seconds": 30},
        "process_policy": {"allowed_comm": sorted(counts), "max_count": sum(counts.values()) + 4},
        "environment_name_policy": {
            "allowed_names": sorted(names), "required_names": sorted(names), "forbidden_names": [],
        },
        "mutation": {"command": "printf reset-mutated > /app/reset-probe-marker.txt", "timeout_seconds": 30},
    }


def run(out: Path, builder: Path, environment: str = "none") -> dict:
    if os.environ.get("CAPABILITY_REMOTE_RESET_PROBE") != "1":
        raise RuntimeError("reset protocol probe is remote-only")
    if out.exists() or out.is_symlink():
        raise FileExistsError("reset probe output must be fresh")
    out.mkdir(parents=True)
    toolchain = OfficialToolchain.resolve(out)
    if environment not in {"none", "shellsim", "docker"}:
        raise ValueError("reset probe environment must be none, shellsim, or docker")
    item = out / f"{environment}-fixture"
    command = toolchain.runtime_command()
    index = command.index("python")
    command[index + 1:] = [str(builder), "--out", str(item), "--environment", environment]
    if environment == "docker":
        command += ["--image", DOCKER_IMAGE]
    with (out / "builder.log").open("x") as log:
        built = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=600, check=False)
    if built.returncode != 0:
        raise RuntimeError(f"pinned fixture builder failed: {built.returncode}")
    if environment == "none":
        result = run_frozen_non_docker_reset(item, toolchain, 600)
    elif environment == "shellsim":
        from capability_pipeline.shellsim_snapshot_extension import apply_overlay

        with tempfile.TemporaryDirectory(prefix="shellsim-reset-probe-") as temporary:
            temporary_root = Path(temporary)
            overlay = apply_overlay(Path(os.environ["TASKCOMPENDIUM_SOURCE"]), temporary_root / "overlay")
            target = temporary_root / "target"
            with (out / "cargo-build.log").open("x") as log:
                built = subprocess.run([
                    "cargo", "build", "--locked", "--release", "--manifest-path",
                    str(overlay / "shellsim-bridge/Cargo.toml"), "--target-dir", str(target),
                ], stdout=log, stderr=subprocess.STDOUT, timeout=900, check=False)
            if built.returncode != 0:
                raise RuntimeError(f"pinned ShellSim bridge build failed: {built.returncode}")
            result = run_frozen_non_docker_reset(
                item, toolchain, 900, target / "release/taskcompendium-shellsim",
            )
    else:
        from capability_pipeline.daytona_resources import DaytonaResourceProfile
        from capability_pipeline.reset_runner import run_frozen_reset

        if not os.environ.get("DAYTONA_API_KEY"):
            raise RuntimeError("Docker reset protocol requires Daytona credentials")
        helper_root = Path(os.environ["CAPABILITY_DAYTONA_TOOLS"])
        helper = helper_root / "dt.py"
        if not helper.is_file():
            raise FileNotFoundError("staged Daytona helper is missing")
        task = item / "workspace/task"
        task.mkdir(parents=True)
        (task / "candidate-resources.json").write_text(json.dumps(RESOURCE_REQUEST, sort_keys=True) + "\n")
        profile = DaytonaResourceProfile(**RESOURCE_REQUEST, source="adapter_kwargs")
        (task / "resource-receipt.json").write_text(json.dumps(profile.receipt(), sort_keys=True) + "\n")
        policy_path = task / "reset-policy.json"
        provisional = _docker_policy({"process_comm_counts": {"init": 1}, "environment_names": ["PATH"]})
        policy_path.write_text(json.dumps(provisional, indent=2, sort_keys=True) + "\n")
        calib_cmd = toolchain.runtime_command()
        calib_cmd[calib_cmd.index("python") + 1:] = [
            str(Path(__file__).resolve()), "--calibrate", "--item", str(item),
            "--output", str(out / "calibration.json"),
        ]
        with (out / "calibration.log").open("x") as log:
            calibrated = subprocess.run(calib_cmd, stdout=log, stderr=subprocess.STDOUT, timeout=1800, check=False)
        calibration = json.loads((out / "calibration.json").read_text()) if (out / "calibration.json").is_file() else {}
        if calibrated.returncode != 0 or calibration.get("status") != "observed" or calibration.get("cleanup", {}).get("verified_absent") is not True:
            raise RuntimeError("Docker calibration did not complete with verified cleanup")
        policy_path.write_text(json.dumps(_docker_policy(calibration["observation"]), indent=2, sort_keys=True) + "\n")
        result = run_frozen_reset(item, toolchain, 2400, helper_root)
    report = {key: value for key, value in result.items() if key != "extra_files"}
    (out / "reset-result.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    files = {
        path.relative_to(out).as_posix(): sha256(path)
        for path in sorted(out.rglob("*")) if path.is_file() and path.name != "probe-inventory.json"
    }
    (out / "probe-inventory.json").write_text(json.dumps({
        "schema_version": "capability-reset-protocol-probe-v1",
        "state": result["state"], "files": files,
    }, indent=2, sort_keys=True) + "\n")
    if result["state"] != "ready":
        raise RuntimeError(f"{environment} reset protocol did not pass; raw evidence retained")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--builder", type=Path)
    parser.add_argument("--environment", choices=("none", "shellsim", "docker"), default="none")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--item", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.calibrate:
        if args.item is None or args.output is None:
            parser.error("calibration requires --item and --output")
        result = _calibrate(args.item, args.output)
        return 0 if result["status"] == "observed" and result["cleanup"].get("verified_absent") is True else 2
    if args.out is None or args.builder is None:
        parser.error("probe requires --out and --builder")
    run(args.out, args.builder, args.environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
