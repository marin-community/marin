# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build immutable native references for one qualified CalibForge base family."""

import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BUILD_NETWORK = "none"


def reference_records(records: list[dict], base: str) -> list[dict]:
    """Select the exact pinned base, not other images in the same repository."""
    return [r for r in records if r.get("base_image") == base and r["status"].startswith("compiled")]


def build_reference(record: dict, source: Path, output: Path) -> dict:
    """Build the unmodified public source and retain failures without admitting it."""
    context = source / record["task_id"] / "environment"
    if hashlib.sha256((context / "Dockerfile").read_bytes()).hexdigest() != record["source_dockerfile_sha256"]:
        raise ValueError(f"Native source changed: {record['task_id']}")
    try:
        result = subprocess.run(
            ["docker", "build", "--platform", "linux/amd64", "--network", BUILD_NETWORK, "-q", str(context)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        status = "built" if result.returncode == 0 else "build_failed"
        image = result.stdout.strip() if result.returncode == 0 else None
        output_text = result.stdout + result.stderr
    except subprocess.TimeoutExpired as error:
        status = "build_timeout"
        image = None
        streams = [error.stdout or b"", error.stderr or b""]
        output_text = "".join(part.decode(errors="replace") if isinstance(part, bytes) else part for part in streams)
        output_text += f"\nNative build exceeded {error.timeout} seconds\n"
    log = output / "logs" / (record["task_id"].replace("/", "--") + ".txt")
    log.write_text(output_text)
    return {
        "task_id": record["task_id"],
        "source_dockerfile_sha256": record["source_dockerfile_sha256"],
        "base_image": record["base_image"],
        "status": status,
        "reference_image": image,
        "build_log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundles", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    if "@sha256:" not in args.base:
        raise ValueError("Native references require an immutable base")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "logs").mkdir()
    manifest = json.loads((args.bundles / "manifest.json").read_text())
    records = reference_records(manifest["tasks"], args.base)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(lambda r: build_reference(r, args.source, args.output), records))
    evidence = {"build_network": BUILD_NETWORK, "admission": "not_run", "tasks": results}
    (args.output / "manifest.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({"candidates": len(results), "built": sum(r["status"] == "built" for r in results)}))


if __name__ == "__main__":
    main()
