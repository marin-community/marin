# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit complete frozen Atlas populations in their storage region."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow.parquet as pq
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join

RUNTIME_PATHS = (
    "lib/taskcompendium/src",
    "lib/rolloutengine/src",
    "lib/verifyit/src",
    "lib/shellbox/src",
    "lib/rigging/src",
    "vendor/skyrl/skyrl-gym",
    "vendor/skyrl",
    "vendor/skyrl/skyrl-train",
)


def read_result(url: str) -> dict:
    with StoragePath(url if url.endswith(".json") else prefix_join(url, ".artifact.json")).open("r") as stream:
        return json.load(stream)["result"]


def audit_source(row: dict, runtime: Path, config: dict, output: str) -> dict:
    binding = row["input"]
    source_key = hashlib.sha256(row["source_id"].encode()).hexdigest()[:20]
    destination = prefix_join(output, source_key + ".json")
    census = read_result(binding["evidence_url"])
    population = read_result(census["input_artifact"])
    if not population["specs_output_dir"].startswith(config["regional_bucket"].rstrip("/") + "/"):
        raise ValueError("Population is outside this audit job region")
    digest = hashlib.sha256()
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(str(runtime / path) for path in RUNTIME_PATHS)}
    with TemporaryDirectory(prefix="consumer-") as temporary:
        result_path = Path(temporary) / "result.json"
        with result_path.open("w") as result_stream:
            with subprocess.Popen(
                [sys.executable, str(Path(__file__).with_name("consumer_probe.py").resolve())],
                stdin=subprocess.PIPE,
                stdout=result_stream,
                env=environment,
                cwd=runtime,
                text=True,
            ) as process:
                assert process.stdin is not None
                process.stdin.write(json.dumps(config) + "\n")
                for path in sorted(
                    StoragePath(prefix_join(population["specs_output_dir"], "*.parquet")).glob(), key=str
                ):
                    with path.open("rb") as stream:
                        for batch in pq.ParquetFile(stream).iter_batches(batch_size=32, columns=["task_spec"]):
                            for record in batch.to_pylist():
                                serialized = record["task_spec"]
                                digest.update(serialized.encode())
                                process.stdin.write(json.dumps(serialized) + "\n")
                process.stdin.close()
                if process.wait() != 0:
                    raise RuntimeError(f"Runtime probe failed for {row['source_id']}")
        result = json.loads(result_path.read_text())
    if digest.hexdigest() != binding["task_specs_sha256"] or result["checked_tasks"] != binding["measured_tasks"]:
        result["failures"]["CONSUMER_POPULATION_MISMATCH"] = result["checked_tasks"]
        result["failed_tasks"] = result["checked_tasks"]
    result.update(
        source_id=row["source_id"],
        dataset_id=binding["dataset_id"],
        dataset_revision=binding["dataset_revision"],
        task_specs_sha256=digest.hexdigest(),
        runtime_commit=config["runtime_commit"],
        evidence_url=destination,
    )
    with StoragePath(destination).open("w", auto_mkdir=True) as stream:
        json.dump(result, stream, indent=2)
    print(
        json.dumps({key: result[key] for key in ("source_id", "checked_tasks", "failed_tasks", "failures")}), flush=True
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--runtime-archive", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    configure_coreweave_s3()
    config = json.loads(args.config.read_text())
    if hashlib.sha256(args.runtime_archive.read_bytes()).hexdigest() != config["runtime_archive_sha256"]:
        raise ValueError("Runtime archive differs from its frozen manifest")
    with TemporaryDirectory(prefix="consumer-runtime-") as temporary:
        runtime = Path(temporary)
        with tarfile.open(args.runtime_archive) as archive:
            archive.extractall(runtime, filter="data")
        rows = json.loads(args.inputs.read_text())
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(lambda row: audit_source(row, runtime, config, args.output), rows))
    with StoragePath(prefix_join(args.output, "audit.json")).open("w", auto_mkdir=True) as stream:
        json.dump({"config": config, "results": results}, stream, indent=2)


if __name__ == "__main__":
    main()
