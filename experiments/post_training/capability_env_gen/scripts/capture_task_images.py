#!/usr/bin/env python3
"""Trusted c32 rootfs capture worker; publication remains a separate step.

The default mode is a credential-free review. ``--execute`` requires an
approved, hash-pinned plan plus Daytona and CW object-store credentials. Raw
legacy capture output is held in memory and only a filtered receipt is written.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from capability_pipeline.image_publication_contract import validate_publication_contract

ENVGEN_PROBES = Path("/Users/k3sc0re/openathena/build_envs/envgen/probes")
CAPTURE = ENVGEN_PROBES / "capture_rootfs.py"
CW_HOST = "marin-us-east-02a.cwobject.com"
FILTERED_ENV = {
    "HOME", "HOSTNAME", "PWD", "SHLVL", "_", "OLDPWD", "SHELL", "TERM",
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy",
    "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "GIT_SSL_CAINFO",
    "NODE_EXTRA_CA_CERTS", "DENO_CERT",
}
CAPTURE_RECEIPT_FIELDS = {
    "capture",
    "excludes",
    "finished_utc",
    "gzip_level",
    "object",
    "object_bytes",
    "object_key",
    "ok",
    "push_wall_s",
    "sandbox",
    "sandbox_create_s",
    "sha256",
    "snapshot",
    "started_utc",
    "tag",
    "unpacked_kb",
}
CAPTURE_RESULT_FIELDS = {
    "bad_parts",
    "compressed_bytes",
    "gzip_exit",
    "mb_per_s",
    "parts",
    "seconds",
    "sha256",
    "tar_exit",
    "tar_stderr_tail",
}
PROVIDER_BOUND_PATHS = {
    "/etc/hostname",
    "/etc/hosts",
    "/etc/resolv.conf",
    "/etc/daytona/netleash/ca.crt",
    "/usr/local/bin/daytona",
    "/usr/local/lib/daytona-computer-use",
}
REQUIRED_CAPTURE_EXCLUDES = {
    "./etc/hostname",
    "./etc/hosts",
    "./etc/resolv.conf",
    "./etc/daytona/*",
    "./usr/local/bin/daytona",
    "./usr/local/lib/daytona-computer-use",
    "./usr/local/lib/daytona-computer-use/*",
    "./var/lib/postgresql/data/*",
    "./var/lib/docker/*",
    "./var/lib/kubelet/*",
    "./var/lib/k0s/*",
    "./var/lib/rancher/*",
    "./var/lib/buildkit/*",
    "./var/lib/containerd/*",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_env(source: dict[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in sorted(source.items())
        if key not in FILTERED_ENV and not key.startswith("DAYTONA_")
    }


def _expected_env(entries: list[str]) -> dict[str, str]:
    result = {}
    for entry in entries:
        key, separator, value = entry.partition("=")
        if not separator or not key or key in result:
            raise RuntimeError("reviewed image Env is malformed")
        result[key] = value
    return result


def _validate_capture_implementation(plan: dict) -> None:
    implementation = plan.get("capture_implementation")
    if not isinstance(implementation, dict):
        raise TypeError("capture implementation binding is absent")
    bindings = {
        "capture_rootfs_path": "capture_rootfs_sha256",
        "in_sandbox_capture_path": "in_sandbox_capture_sha256",
        "capture_worker_path": "capture_worker_sha256",
        "publication_contract_path": "publication_contract_sha256",
        "rootfs_review_path": "rootfs_review_sha256",
        "image_publication_path": "image_publication_sha256",
        "publication_cli_path": "publication_cli_sha256",
    }
    for path_key, hash_key in bindings.items():
        path = Path(str(implementation.get(path_key, "")))
        expected = implementation.get(hash_key)
        if not path.is_file() or not isinstance(expected, str) or sha256(path) != expected:
            raise RuntimeError(f"capture implementation drift: {path_key}")
    local_bindings = {
        Path(__file__).resolve().parents[1] / "capability_pipeline/oci_artifact.py": implementation.get("oci_artifact_sha256"),
        Path(__file__).resolve().parents[1] / "capability_pipeline/oci_registry.py": implementation.get("oci_registry_sha256"),
    }
    for path, expected in local_bindings.items():
        if not path.is_file() or not isinstance(expected, str) or sha256(path) != expected:
            raise RuntimeError(f"capture implementation drift: {path.name}")
    capture_source = Path(implementation["capture_rootfs_path"]).read_text()
    module = ast.parse(capture_source)
    assignments = [
        node
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "EXCLUDES" for target in node.targets)
    ]
    if len(assignments) != 1:
        raise RuntimeError("capture implementation EXCLUDES assignment is ambiguous")
    try:
        excludes = ast.literal_eval(assignments[0].value)
    except (TypeError, ValueError, SyntaxError) as error:
        raise RuntimeError("capture implementation EXCLUDES is not a literal") from error
    if not isinstance(excludes, list) or any(not isinstance(item, str) for item in excludes):
        raise RuntimeError("capture implementation EXCLUDES is not a string list")
    missing_exclusions = sorted(REQUIRED_CAPTURE_EXCLUDES - set(excludes))
    if missing_exclusions:
        raise RuntimeError("capture implementation lacks explicit provider-bound exclusions")


def _sanitize_legacy_receipt(raw: dict) -> dict:
    sanitized = {key: raw[key] for key in sorted(CAPTURE_RECEIPT_FIELDS & raw.keys())}
    capture = raw.get("capture")
    if not isinstance(capture, dict):
        raise TypeError("legacy capture receipt has no capture result")
    sanitized["capture"] = {
        key: capture[key] for key in sorted(CAPTURE_RESULT_FIELDS & capture.keys())
    }
    return sanitized


def _validate_archive_receipt(raw: dict) -> dict:
    capture = raw.get("capture")
    if not isinstance(capture, dict):
        raise TypeError("legacy capture result is absent")
    tar_exit = capture.get("tar_exit")
    gzip_exit = capture.get("gzip_exit")
    diagnostic = str(capture.get("tar_stderr_tail") or "")
    if gzip_exit != 0:
        raise RuntimeError("rootfs gzip process failed")
    if tar_exit != 0:
        raise RuntimeError(f"rootfs tar process failed with exit {tar_exit!r}")
    tar_class = "clean"
    compressed = capture.get("compressed_bytes")
    object_bytes = raw.get("object_bytes")
    capture_hash = capture.get("sha256")
    if (
        not isinstance(compressed, int)
        or compressed <= 0
        or object_bytes != compressed
        or not isinstance(capture_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", capture_hash) is None
        or raw.get("sha256") != capture_hash
        or raw.get("ok") is not True
    ):
        raise RuntimeError("captured object byte/hash receipt is inconsistent")
    return {"tar_exit_class": tar_class, "tar_stderr_tail": diagnostic}


def _create(dtx, client, snapshot: str):
    failures = []
    for attempt in range(1, 5):
        try:
            sandbox, seconds = dtx.create(
                client, snapshot, purpose="capability-c32-capture", ttl_min=120, domains=CW_HOST
            )
            return sandbox, seconds, failures
        except Exception as error:  # noqa: BLE001 -- provider SDK exception set is unstable
            failures.append({"attempt": attempt, "error_type": type(error).__name__})
            if attempt < 4:
                time.sleep(10 * attempt)
    return None, None, failures


def _wait_ready(dtx, sandbox) -> dict:
    command = (
        "test -f /tmp/task-ready && "
        "PGPASSWORD=storm psql -h 127.0.0.1 -U storm -d storm -X -tAc 'SELECT 1' | grep -qx 1"
    )
    for attempt in range(1, 61):
        result = dtx.sh(sandbox, command, timeout=30)
        if result["exit"] == 0:
            return {"attempts": attempt, "ready": True}
        time.sleep(2)
    return {"attempts": 60, "ready": False}


def _capture_role(plan_path: Path, plan: dict, image: dict, output: Path) -> dict:
    sys.path.insert(0, str(ENVGEN_PROBES))
    import dtx  # type: ignore[import-not-found]

    _validate_capture_implementation(plan)
    client = dtx.client()
    expected_snapshot = image["source_snapshot"]
    snapshot = client.snapshot.get(expected_snapshot["name"])
    recipe_path = Path(image["source_recipe"]["path"])
    if not recipe_path.is_absolute():
        recipe_path = Path(plan["frozen_workspace"]) / recipe_path
    recipe_text = recipe_path.read_text()
    provider_recipe = getattr(getattr(snapshot, "build_info", None), "dockerfile_content", None)
    observed = {
        "name": snapshot.name,
        "id": snapshot.id,
        "ref": getattr(snapshot, "ref", None),
        "state": str(snapshot.state),
        "cpu": snapshot.cpu,
        "mem": snapshot.mem,
        "disk": snapshot.disk,
    }
    for key in ("name", "id", "ref"):
        if observed[key] != expected_snapshot[key]:
            raise RuntimeError(f"source snapshot {key} drift")
    if "ACTIVE" not in observed["state"]:
        raise RuntimeError("source snapshot is not active")
    if sha256(recipe_path) != image["source_recipe"]["sha256"] or provider_recipe != recipe_text:
        raise RuntimeError("source snapshot build_info recipe drift")

    sandbox, create_seconds, failures = _create(dtx, client, expected_snapshot["name"])
    receipt = {
        "schema_version": "capability-rootfs-capture-v1",
        "role": image["role"],
        "plan_sha256": sha256(plan_path),
        "source_snapshot": observed,
        "provisioning_failures": failures,
        "sandbox_id": getattr(sandbox, "id", None),
        "sandbox_create_seconds": create_seconds,
        "capture": None,
        "cleanup": None,
    }
    if sandbox is None:
        receipt["state"] = "infrastructure_error"
        output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        return receipt
    try:
        ready = _wait_ready(dtx, sandbox)
        receipt["ready"] = ready
        if not ready["ready"]:
            raise RuntimeError("owned capture sandbox did not become ready")
        for path, expected in image.get("required_ready_hashes", {}).items():
            measured = dtx.sh(sandbox, "sha256sum -- " + shlex.quote(path), timeout=60)
            words = (measured.get("stdout") or "").split(maxsplit=1)
            actual = words[0] if words else None
            if measured["exit"] != 0 or actual != expected:
                raise RuntimeError(f"ready-state content mismatch: {path}")
        mount = dtx.sh(
            sandbox,
            "test \"$(stat -c %d /)\" != \"$(stat -c %d /var/lib/postgresql/data)\" "
            "&& findmnt -T /var/lib/postgresql/data -n >/dev/null",
            timeout=60,
        )
        receipt["pgdata_distinct_mount"] = mount["exit"] == 0
        if mount["exit"] != 0:
            raise RuntimeError("PGDATA is not a distinct mount; live capture refused")
        mounts = dtx.sh(
            sandbox,
            "for p in " + " ".join(shlex.quote(path) for path in sorted(PROVIDER_BOUND_PATHS))
            + "; do findmnt -T \"$p\" -n -o TARGET,SOURCE,FSTYPE,OPTIONS || exit 1; done",
            timeout=60,
        )
        receipt["provider_bound_mounts_checked"] = mounts["exit"] == 0
        if mounts["exit"] != 0:
            raise RuntimeError("provider-bound file mount inventory failed")
        if image["role"] == "candidate":
            privacy = dtx.sh(
                sandbox,
                "set -o pipefail; test ! -e /opt/evaluator && test ! -e /opt/verifier "
                "&& test ! -e /private && "
                "found=$(find / -xdev -type f \\( -iname '*ground_truth*' "
                "-o -iname '*reference_solution*' -o -iname '*negative_control*' "
                "-o -iname 'taskcompendium-result.json' -o -iname 'evaluate.py' "
                "-o -iname 'run_eval.py' -o -iname '*capability-registry-publisher*' \\) -print) "
                "&& test -z \"$found\"",
                timeout=120,
            )
            receipt["candidate_path_scan_passed"] = privacy["exit"] == 0
            if privacy["exit"] != 0:
                raise RuntimeError("candidate private-path scan failed")
        with tempfile.TemporaryDirectory(prefix="cap-c32-capture-") as temporary:
            raw_path = Path(temporary) / "legacy.json"
            command = [
                sys.executable,
                str(CAPTURE),
                "--snapshot", expected_snapshot["name"],
                "--sandbox", sandbox.id,
                "--tag", "c32-" + image["role"],
                "--out", str(raw_path),
            ]
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=4200, check=False
            )
            if completed.returncode != 0 or not raw_path.is_file():
                raise RuntimeError("legacy rootfs capture failed")
            raw = json.loads(raw_path.read_text())
        observed_env = image_env(raw.pop("source_env", {}))
        expected_env = _expected_env(image["image_config"]["Env"])
        mismatches = sorted(
            key for key, value in expected_env.items() if observed_env.get(key) != value
        )
        if mismatches:
            raise RuntimeError("live image environment differs from reviewed config")
        receipt["environment"] = {
            "reviewed_values_matched": True,
            "reviewed_names": sorted(expected_env),
            "additional_filtered_names": sorted(set(observed_env) - set(expected_env)),
            "raw_source_environment_retained": False,
        }
        receipt["archive_process"] = _validate_archive_receipt(raw)
        receipt["capture"] = _sanitize_legacy_receipt(raw)
        receipt["state"] = "captured_pending_privacy_and_publication"
    finally:
        try:
            sandbox.delete()
            absent = False
            checks = []
            for attempt in range(5):
                if attempt:
                    time.sleep(3 * attempt)
                try:
                    client.get(sandbox.id)
                    checks.append("present")
                except Exception as error:  # noqa: BLE001 -- SDK types drift
                    if getattr(error, "status_code", None) == 404 or "not found" in str(error).lower():
                        checks.append("not_found")
                        absent = True
                        break
                    checks.append(type(error).__name__)
            receipt["cleanup"] = {
                "delete_requested": True,
                "absence_verified": absent,
                "observations": checks,
            }
            if not absent:
                receipt["state"] = "cleanup_unverified"
        except Exception as error:  # noqa: BLE001 -- cleanup must retain a receipt
            receipt["cleanup"] = {"delete_requested": False, "error_type": type(error).__name__}
        output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role", choices=("candidate", "private_verifier"), required=True)
    parser.add_argument("--approved-plan-sha256")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    validate_publication_contract(plan, workspace=args.workspace)
    current_hash = sha256(args.plan)
    image = next(item for item in plan["images"] if item["role"] == args.role)
    if not args.execute:
        print(json.dumps({"state": "reviewed_no_execution", "plan_sha256": current_hash, "role": args.role}, sort_keys=True))
        return 0
    if plan["state"] != "approved_for_capture" or args.approved_plan_sha256 != current_hash:
        raise SystemExit("capture requires an approved plan and its exact SHA-256")
    if not os.environ.get("DAYTONA_API_KEY") or not (
        os.environ.get("CW_KEY_ID") and os.environ.get("CW_KEY_SECRET")
    ):
        raise SystemExit("trusted capture credentials are absent")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = _capture_role(args.plan, plan, image, args.output)
    print(json.dumps({"state": result["state"], "role": args.role, "receipt": str(args.output)}, sort_keys=True))
    return 0 if result["state"].startswith("captured") else 1


if __name__ == "__main__":
    raise SystemExit(main())
