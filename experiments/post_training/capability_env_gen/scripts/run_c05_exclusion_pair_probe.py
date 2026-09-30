#!/usr/bin/env python3
"""Run a paired C05 exclusion-reason probe in network-blocked Daytona."""
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
WORKSPACE = Path(
    "runs/synthesis-c05-image-migration-revalidation-007/terminal-pull-0829/"
    "revalidation/items/c05.analysis.protocol_trace-10-dda40ab415c3/workspace"
)
SOURCE = Path("runs/c05-fence-diagnostic-002/inputs/case-2-unfenced.json")
PRIVATE_FILES = ("grader.py", "reference_model.py", "trace_public.txt")
REASONS = {
    "simple_duplicate": (
        "duplicate ACK: ack=8001 repeats the current cumulative ACK point "
        "and acknowledges no new data"
    ),
    "contextual_duplicate": (
        "duplicate ACK: ack=8001 repeats the current SND.UNA and acknowledges "
        "no new data; it is triggered by the retransmitted S7R arriving at H2 "
        "as duplicate data"
    ),
    "foreign_karn": (
        "Karn exclusion: the retransmitted segment makes an RTT sample ambiguous"
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(inputs: Path) -> dict[str, Any]:
    inputs.mkdir(parents=True)
    for name in PRIVATE_FILES:
        shutil.copyfile(WORKSPACE / name, inputs / name)
    baseline = json.loads(SOURCE.read_text())
    cases = []
    for case_id, reason in REASONS.items():
        candidate = json.loads(json.dumps(baseline))
        row = next(r for r in candidate["estimator_table"] if r["ack_time_ms"] == 263)
        row["exclusion_reason"] = reason
        path = inputs / f"{case_id}.json"
        path.write_text(json.dumps(candidate, separators=(",", ":")) + "\n")
        cases.append({"case_id": case_id, "path": path.name, "reason": reason, "sha256": sha256(path)})
    manifest = {
        "schema_version": "c05-exclusion-pair-input-v1",
        "snapshot": SNAPSHOT,
        "unchanged_grader_sha256": sha256(inputs / "grader.py"),
        "reference_model_sha256": sha256(inputs / "reference_model.py"),
        "trace_sha256": sha256(inputs / "trace_public.txt"),
        "source_candidate_sha256": sha256(SOURCE),
        "cases": cases,
    }
    (inputs / "input-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


REMOTE = r'''import hashlib, json, pathlib, sys
root = pathlib.Path("/tmp/c05-exclusion-pair/input")
sys.path.insert(0, str(root))
import grader
manifest = json.loads((root / "input-manifest.json").read_text())
trace = (root / "trace_public.txt").read_text()
outcomes = []
for case in manifest["cases"]:
    path = root / case["path"]
    report = grader.grade(trace, path.read_text(), self_check=True)
    c1 = (report.get("criteria") or {}).get("C1") or {}
    row = next((r for r in c1.get("rows", []) if r.get("ack_time_ms") == 263), None)
    outcomes.append({
        "case_id": case["case_id"], "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "status": report.get("status"), "score": report.get("score"), "passed": report.get("passed"),
        "failed": report.get("failed"), "c1_detail": c1.get("detail"), "row_263": row,
    })
pathlib.Path("/tmp/c05-exclusion-pair/outcomes.json").write_text(json.dumps({
    "schema_version": "c05-exclusion-pair-outcome-v1",
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
    manifest = prepare(out / "inputs")
    (out / "remote.py").write_text(REMOTE)
    probes = Path("/Users/k3sc0re/openathena/build_envs/envgen/probes")
    sys.path.insert(0, str(probes))
    import dtx  # type: ignore[import-not-found]
    sys.path.insert(0, str(probes.parent))
    from dt import upload_path

    receipt: dict[str, Any] = {
        "schema_version": "c05-exclusion-pair-receipt-v1",
        "input_manifest": manifest,
        "input_manifest_sha256": sha256(out / "inputs/input-manifest.json"),
        "network_block_all_requested": True,
    }
    sandbox = None
    try:
        started = time.monotonic()
        sandbox, seconds = dtx.create(client := dtx.client(), SNAPSHOT, purpose="c05-exclusion-pair", ttl_min=60, block_all=True)
        receipt.update(sandbox_id=sandbox.id, create_seconds=seconds)
        receipt["network_block_all_observed"] = getattr(client.get(sandbox.id), "network_block_all", None)
        if receipt["network_block_all_observed"] is not True:
            raise RuntimeError("provider did not confirm network_block_all")
        receipt["create_and_get_seconds"] = round(time.monotonic() - started, 3)
        uploads = [
            upload_path(sandbox, out / "inputs", "/tmp/c05-exclusion-pair/input"),
            upload_path(sandbox, out / "remote.py", "/tmp/c05-exclusion-pair/remote.py"),
        ]
        receipt["uploads"] = uploads
        if any(upload["exit"] != 0 for upload in uploads):
            raise RuntimeError("probe upload failed")
        result = sandbox.process.exec("mkdir -p /tmp/c05-exclusion-pair && python3 /tmp/c05-exclusion-pair/remote.py", timeout=120)
        receipt["remote_exit_code"] = result.exit_code
        if result.exit_code != 0:
            raise RuntimeError("remote probe failed")
        (out / "outcomes.json").write_bytes(sandbox.fs.download_file("/tmp/c05-exclusion-pair/outcomes.json") or b"")
        receipt["outcomes_sha256"] = sha256(out / "outcomes.json")
        receipt["state"] = "passed"
    except Exception as error:  # noqa: BLE001
        receipt.update(state="infrastructure_error", error_type=type(error).__name__, error=str(error)[:400])
    finally:
        if sandbox is not None:
            try:
                sandbox.delete()
                receipt["cleanup"] = {"delete_requested": True}
            except Exception as error:  # noqa: BLE001
                receipt["cleanup"] = {"delete_requested": False, "error_type": type(error).__name__}
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return 0 if receipt["state"] == "passed" and receipt.get("cleanup", {}).get("delete_requested") else 1


if __name__ == "__main__":
    raise SystemExit(main())
