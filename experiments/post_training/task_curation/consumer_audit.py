# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit complete frozen Atlas populations in their storage region."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow.parquet as pq
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join
from taskcompendium.models import TaskSpec, judge_spec

from experiments.post_training.task_curation.consumer_scope import ConsumerCapacity, consumer_needs
from experiments.post_training.task_curation.lexical_contamination import PARTITION_ROWS

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


def source_row_key(locator: str) -> tuple[tuple[str, int], ...]:
    """Order the source file and numeric row locator used before lexical partition sorting."""
    parts = re.split(r"(\d+)", locator)
    return tuple(
        (parts[index], int(parts[index + 1]) if index + 1 < len(parts) else -1) for index in range(0, len(parts), 2)
    )


def audit_source(row: dict, runtime: Path, config: dict, output: str) -> dict:
    binding = row["input"]
    source_key = hashlib.sha256(row["source_id"].encode()).hexdigest()[:20]
    destination = prefix_join(output, source_key + ".json")
    census = read_result(binding["evidence_url"])
    population = read_result(census["input_artifact"])
    StoragePath(population["specs_output_dir"]).relative_to(StoragePath(config["regional_bucket"]))
    digest, source_digest = hashlib.sha256(), hashlib.sha256()
    capacity = ConsumerCapacity.model_validate(config["capacity"])
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
                    source_rows = []
                    with path.open("rb") as stream:
                        for batch in pq.ParquetFile(stream).iter_batches(batch_size=32, columns=["task_spec"]):
                            for record in batch.to_pylist():
                                serialized = record["task_spec"]
                                digest.update(serialized.encode())
                                task = TaskSpec.model_validate_json(serialized)
                                source_rows.append((source_row_key(task.source.row), serialized))
                                if len(source_rows) == PARTITION_ROWS:
                                    for _, encoded in sorted(source_rows):
                                        source_digest.update(encoded.encode())
                                    source_rows.clear()
                                judge = judge_spec(task.grader)
                                process.stdin.write(
                                    json.dumps(
                                        {
                                            "task": serialized,
                                            "needs": sorted(consumer_needs(task, capacity)),
                                            "judge_model": judge.model if judge is not None else "",
                                        }
                                    )
                                    + "\n"
                                )
                    for _, encoded in sorted(source_rows):
                        source_digest.update(encoded.encode())
                process.stdin.close()
                if process.wait() != 0:
                    raise RuntimeError(f"Runtime probe failed for {row['source_id']}")
        result = json.loads(result_path.read_text())
    if result["checked_tasks"] != binding["measured_tasks"]:
        raise ValueError("Consumer scan count differs from the frozen population")
    result["scanned_task_specs_sha256"] = digest.hexdigest()
    result["source_order_task_specs_sha256"] = source_digest.hexdigest()
    if binding["task_specs_sha256"] not in {digest.hexdigest(), source_digest.hexdigest()}:
        raise ValueError("Frozen population fingerprint differs in stored and original source order")
    result["storage_order_differs"] = digest.hexdigest() != binding["task_specs_sha256"]
    result.update(
        source_id=row["source_id"],
        dataset_id=binding["dataset_id"],
        dataset_revision=binding["dataset_revision"],
        task_specs_sha256=binding["task_specs_sha256"],
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
