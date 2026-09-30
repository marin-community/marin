#!/usr/bin/env python3
"""Compare retained fenced C05 solver responses with fence-only variants remotely.

The private deterministic grader and all candidate variants execute only in one
fresh, network-blocked Daytona sandbox.  Raw historic responses remain intact;
the fence-only variants are diagnostic inputs and are never admitted as trials.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

SNAPSHOT = "daytona-small"
RAW_ROOT = Path(
    "runs/synthesis-c05-image-migration-revalidation-007/terminal-pull-0829/"
    "revalidation/items/c05.analysis.protocol_trace-10-dda40ab415c3"
)
RAW_CASES = (
    "control-c05s4-pos-formatting-variant-attempt-1",
    "control-c05s4-pos-formatting-variant-attempt-2",
    "control-c05s4-pos-independent-solver-path-attempt-1",
    "control-c05s4-pos-independent-solver-path-attempt-2",
)
PRIVATE_FILES = ("grader.py", "reference_model.py", "trace_public.txt")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def remove_one_json_fence(raw: bytes) -> bytes:
    text = raw.decode("utf-8")
    if not text.startswith("```json\n") or not text.rstrip().endswith("```"):
        raise ValueError("response does not have exactly one enclosing json fence")
    body = text[len("```json\n") :].rstrip()
    if not body.endswith("```"):
        raise ValueError("response fence terminator is malformed")
    unwrapped = body[:-3]
    # The only allowed transformation is removing the outer marker/newline.
    unwrapped = unwrapped.removesuffix("\n")
    json.loads(unwrapped)
    return unwrapped.encode("utf-8")


def prepare(inputs: Path) -> dict[str, Any]:
    if inputs.exists():
        shutil.rmtree(inputs)
    inputs.mkdir(parents=True)
    workspace = RAW_ROOT / "workspace"
    for name in PRIVATE_FILES:
        source = workspace / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copyfile(source, inputs / name)
    cases: list[dict[str, str]] = []
    for index, case in enumerate(RAW_CASES, 1):
        raw_path = RAW_ROOT / "runtime-trials" / case / "agent" / "response.txt"
        raw = raw_path.read_bytes()
        raw_name = f"case-{index}-raw.txt"
        variant_name = f"case-{index}-unfenced.json"
        (inputs / raw_name).write_bytes(raw)
        (inputs / variant_name).write_bytes(remove_one_json_fence(raw))
        cases.append(
            {
                "case_id": case,
                "raw": raw_name,
                "unfenced": variant_name,
                "raw_sha256": sha256(inputs / raw_name),
                "unfenced_sha256": sha256(inputs / variant_name),
            }
        )
    manifest = {
        "schema_version": "c05-fence-diagnostic-input-v1",
        "snapshot": SNAPSHOT,
        "private_hashes": {name: sha256(inputs / name) for name in PRIVATE_FILES},
        "cases": cases,
    }
    (inputs / "input-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


REMOTE = r'''import hashlib, json, pathlib, sys
root = pathlib.Path("/tmp/c05-fence-diagnostic/input")
sys.path.insert(0, str(root))
import grader
manifest = json.loads((root / "input-manifest.json").read_text())
trace = (root / "trace_public.txt").read_text()
outcomes = []
for case in manifest["cases"]:
    for variant in ("raw", "unfenced"):
        path = root / case[variant]
        text = path.read_text()
        report = grader.grade(trace, text, self_check=True)
        outcomes.append({
            "case_id": case["case_id"],
            "variant": variant,
            "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "status": report.get("status"),
            "score": report.get("score"),
            "passed": report.get("passed"),
            "pass_rule": report.get("pass_rule"),
            "failed": report.get("failed"),
            "criteria": report.get("criteria"),
            "error": report.get("error"),
            "schema_errors": report.get("schema_errors"),
        })
(pathlib.Path("/tmp/c05-fence-diagnostic/outcomes.json")).write_text(json.dumps({
    "schema_version": "c05-fence-diagnostic-outcome-v1",
    "input_manifest_sha256": hashlib.sha256((root / "input-manifest.json").read_bytes()).hexdigest(),
    "outcomes": outcomes,
}, indent=2) + "\n")
'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("DAYTONA_API_KEY"):
        raise SystemExit("DAYTONA_API_KEY is required")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    inputs = out / "inputs"
    manifest = prepare(inputs)
    (out / "remote.py").write_text(REMOTE)

    probes = Path("/Users/k3sc0re/openathena/build_envs/envgen/probes")
    sys.path.insert(0, str(probes))
    import dtx  # type: ignore[import-not-found]
    sys.path.insert(0, str(probes.parent))
    from dt import upload_path

    client = dtx.client()
    receipt: dict[str, Any] = {
        "schema_version": "c05-fence-diagnostic-v1",
        "snapshot": SNAPSHOT,
        "input_manifest_sha256": sha256(inputs / "input-manifest.json"),
        "input_manifest": manifest,
        "sandbox_id": None,
        "network_block_all_requested": True,
        "network_block_all_observed": None,
        "provider_calls": [],
        "cleanup": None,
    }
    sandbox = None
    try:
        started = time.monotonic()
        sandbox, seconds = dtx.create(
            client, SNAPSHOT, purpose="c05-fence-diagnostic", ttl_min=60, block_all=True
        )
        receipt["sandbox_id"] = sandbox.id
        receipt["create_seconds"] = seconds
        observed = client.get(sandbox.id)
        receipt["network_block_all_observed"] = getattr(
            observed, "network_block_all", None
        )
        if receipt["network_block_all_observed"] is not True:
            raise RuntimeError("provider did not confirm network_block_all")
        receipt["provider_calls"].append({"operation": "create_and_get", "seconds": round(time.monotonic()-started, 3)})
        input_upload = upload_path(sandbox, inputs, "/tmp/c05-fence-diagnostic/input")
        runner_upload = upload_path(sandbox, out / "remote.py", "/tmp/c05-fence-diagnostic/remote.py")
        receipt["provider_calls"].extend(
            [
                {"operation": "upload_inputs", **input_upload},
                {"operation": "upload_runner", **runner_upload},
            ]
        )
        if input_upload["exit"] != 0 or runner_upload["exit"] != 0:
            raise RuntimeError("remote diagnostic input upload failed")
        result = sandbox.process.exec("mkdir -p /tmp/c05-fence-diagnostic && python3 /tmp/c05-fence-diagnostic/remote.py", timeout=120)
        receipt["remote_exit_code"] = result.exit_code
        receipt["remote_stdout_tail"] = (result.result or "")[-1000:]
        if result.exit_code != 0:
            raise RuntimeError("remote grader diagnostic failed")
        outcome_bytes = sandbox.fs.download_file("/tmp/c05-fence-diagnostic/outcomes.json") or b""
        (out / "outcomes.json").write_bytes(outcome_bytes)
        receipt["outcomes_sha256"] = sha256(out / "outcomes.json")
        receipt["state"] = "passed"
    except Exception as error:  # noqa: BLE001
        receipt["state"] = "infrastructure_error"
        receipt["error_type"] = type(error).__name__
        receipt["error"] = str(error)[:400]
    finally:
        if sandbox is not None:
            try:
                sandbox.delete()
                receipt["cleanup"] = {"delete_requested": True}
            except Exception as error:  # noqa: BLE001
                receipt["cleanup"] = {
                    "delete_requested": False,
                    "error_type": type(error).__name__,
                }
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return 0 if receipt["state"] == "passed" and receipt["cleanup"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
