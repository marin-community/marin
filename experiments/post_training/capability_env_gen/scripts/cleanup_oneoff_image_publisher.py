#!/usr/bin/env python3
"""Remove only the exact temporary one-off publisher Job and Secret."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def cleanup(preparation_path: Path, output: Path, *, runner=subprocess.run) -> dict:
    preparation = json.loads(preparation_path.read_text())
    if preparation.get("schema_version") != "capability-oneoff-image-publisher-preparation-v1":
        raise ValueError("publisher preparation is invalid")
    if output.exists() or output.is_symlink():
        raise ValueError("publisher cleanup receipt already exists")
    name = preparation["job_name"]
    job = json.loads((preparation_path.parent / "job.json").read_text())
    if job["metadata"]["name"] != name or preparation["job_sha256"] != hashlib.sha256(
        (preparation_path.parent / "job.json").read_bytes()
    ).hexdigest():
        raise ValueError("publisher Job differs from preparation")
    secret = job["spec"]["template"]["spec"]["volumes"][1]["secret"]["secretName"]
    base = ["kubectl", "--context", "marin-gpu_US-EAST-02A", "-n", "envreg"]
    observations = []
    for kind in ("job", "secret"):
        resource = name if kind == "job" else secret
        runner([*base, "delete", kind, resource, "--ignore-not-found=true", "--wait=true"],
               check=True, capture_output=True, text=True)
        probe = runner([*base, "get", kind, resource, "--ignore-not-found=true", "-o", "name"],
                       check=True, capture_output=True, text=True)
        if probe.stdout.strip():
            raise ValueError("publisher temporary resource remains after cleanup")
        observations.append({"kind": kind, "name": resource, "absence_verified": True})
    receipt = {"schema_version": "capability-oneoff-image-publisher-cleanup-v1",
               "job_sha256": preparation["job_sha256"],
               "completed_utc": datetime.now(UTC).isoformat(), "resources": observations}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = cleanup(args.preparation, args.output)
    print(json.dumps({"state": "absence_verified", "resources": result["resources"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
