# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect one terminal Iris point using its uploader receipt and upstream formats."""

import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from collections.abc import Mapping
from datetime import datetime
from enum import Enum
from pathlib import Path

import zstandard
from fsspec.implementations.local import LocalFileSystem
from iris.client.client import IrisClient
from iris.client.workload import JobStatus, TaskOutputArchive, TaskOutputArchiveState
from iris.cluster.types import JobName
from iris.resources.state import JobState, TaskState
from rigging.filesystem import atomic
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import prefix_join
from rigging.timing import Timestamp

from experiments.evaluation.agentx.recipe import SERIES_PATH

STREAM_LOG_LINE_LIMIT = 20_000


def json_value(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: getattr(value, field.name) for field in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (JobName, Path)):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Timestamp):
        return {"epoch_ms": value.epoch_ms()}
    raise TypeError(f"Cannot serialize {type(value)}")


def read_record(uri: str) -> dict:
    filesystem, path = filesystem_for(uri)
    with filesystem.open(path, "rt") as stream:
        return json.load(stream)


def write_record(uri: str, record: dict) -> None:
    filesystem, path = filesystem_for(uri)
    if isinstance(filesystem, LocalFileSystem):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    with atomic.atomic_rename(path, filesystem=filesystem) as temporary:
        # fsspec.open's automatic mkdir issues CreateBucket on S3. The existing
        # bucket is infrastructure; writing an object must not try to create it.
        with filesystem.open(temporary, "wt") as stream:
            json.dump(record, stream, indent=2, default=json_value)


def copy_file(source: str, destination: str) -> None:
    """Commit a complete file before publishing a receipt that points to it."""
    source_fs, source_path = filesystem_for(source)
    destination_fs, destination_path = filesystem_for(destination)
    if isinstance(destination_fs, LocalFileSystem):
        Path(destination_path).parent.mkdir(parents=True, exist_ok=True)
    with atomic.atomic_rename(destination_path, filesystem=destination_fs) as temporary:
        with source_fs.open(source_path, "rb") as reader, destination_fs.open(temporary, "wb") as writer:
            shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)


def unpack(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with archive.open("rb") as encoded, zstandard.ZstdDecompressor().stream_reader(encoded) as decoded:
        with tarfile.open(fileobj=decoded, mode="r|") as contents:
            contents.extractall(destination, filter="data")


def retain_processed_artifacts(artifacts: Path, bundle: Path, prefix: str) -> tuple[str, str]:
    """Archive output under its checksum so earlier result descriptors remain readable."""
    with bundle.open("wb") as encoded, zstandard.ZstdCompressor().stream_writer(encoded) as compressed:
        with tarfile.open(fileobj=compressed, mode="w|") as contents:
            contents.add(artifacts, arcname=".")
    with bundle.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    uri = prefix_join(prefix, f"inferencex-{digest}.tar.zst")
    copy_file(str(bundle), uri)
    return uri, digest


def retain_archive(archive: TaskOutputArchive, prefix: str, directory: Path) -> tuple[Path, str]:
    directory.mkdir(parents=True, exist_ok=True)
    local = directory / "outputs.tar.zst"
    copy_file(archive.uri, str(local))
    with local.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != archive.sha256:
        raise ValueError("Downloaded output differs from Iris's uploader SHA256")
    copy_file(str(local), prefix_join(prefix, "outputs.tar.zst"))
    return local, digest


def validate_window(point: Path, manifest: dict, attempt_uid: str) -> dict:
    """Keep incomplete, replaced and scenario-invalid windows out of the importer."""
    record = json.loads((point / "attempt.json").read_text())
    if record["status"] != "succeeded" or record["iris_attempt_uid"] != attempt_uid:
        raise ValueError("Output does not belong to this successful Iris attempt")
    for key in ("series_sha256", "recipe_fingerprint", "profile", "concurrency", "vllm_release", "source", "workload"):
        if record[key] != manifest[key]:
            raise ValueError(f"Output changed a pinned input: {key}")
    export = json.loads((point / "aiperf_artifacts" / "profile_export_aiperf.json").read_text())
    metadata = export["metadata"]
    if metadata.get("submission_valid") is not True or export["was_cancelled"]:
        raise ValueError("AgentX did not validate this complete window")
    series = json.loads(SERIES_PATH.read_text())
    coverage = [row for row in metadata["metric_duration_coverage"] if row["phase_name"] == "profiling"]
    if len(coverage) != 1 or coverage[0]["expected_duration_seconds"] != series["duration"]:
        raise ValueError("Measurement duration differs from the pinned normal-duration series")
    if any(coverage[0][key] < coverage[0]["required_ratio"] for key in ("ttft_ratio", "inter_token_latency_ratio")):
        raise ValueError("Profiling coverage is incomplete")
    successes = 0
    errors = 0
    with (point / "aiperf_artifacts" / "profile_export.jsonl").open() as records:
        for line in records:
            request = json.loads(line)
            if request["metadata"]["benchmark_phase"] != "profiling":
                continue
            if request.get("error"):
                errors += 1
            else:
                successes += 1
    if successes == 0:
        raise ValueError("Profiling produced no successful requests")
    error_ratio = errors / (successes + errors)
    if error_ratio > series["max_profiling_error_ratio"]:
        raise ValueError("Profiling request error ratio exceeds the fixed series limit")
    return {
        "submission_valid": True,
        "profiling_requests": successes,
        "profiling_errors": errors,
        "profiling_error_ratio": error_ratio,
        "coverage": coverage[0],
        "profile_start": export["start_time"],
        "profile_end": export["end_time"],
    }


def process_point(point: Path, manifest: dict, processor: Path, artifacts: Path) -> None:
    """Invoke the pinned upstream processor in a separate Python environment."""
    series = json.loads(SERIES_PATH.read_text())
    actual = subprocess.check_output(["git", "-C", str(processor), "rev-parse", "HEAD"], text=True).strip()
    if actual != series["processor_source"]:
        raise ValueError("Processor checkout differs from the tested source pin")
    profile = manifest["profile"]
    count = series["profiles"][profile]["gpus"]
    suffix = f"snowball011_{profile}_vllm_{series['precision']}_agentx_c{manifest['concurrency']}"
    aggregate = artifacts / f"bmk_agentic_{suffix}"
    raw = artifacts / f"agentic_{suffix}"
    raw.mkdir(parents=True, exist_ok=True)
    shutil.copytree(point / "aiperf_artifacts", raw / "aiperf_artifacts", dirs_exist_ok=True)
    shutil.copy2(point / "attempt.json", raw / "attempt.json")
    shutil.copy2(point / "hardware.txt", raw / "hardware.txt")
    release = manifest["vllm_release"]
    environment = os.environ | {
        "RESULT_DIR": str(point.resolve()),
        "AGENTIC_OUTPUT_DIR": str(aggregate.resolve()),
        "RESULT_FILENAME": "agg",
        "MODEL": series["model"],
        "MODEL_PREFIX": "snowball011",
        "FRAMEWORK": "vllm",
        "PRECISION": series["precision"],
        "RUNNER_TYPE": profile,
        "CONC": str(manifest["concurrency"]),
        "TP": str(series["tensor_parallel"]),
        "PP_SIZE": "1",
        "EP_SIZE": str(count),
        "DP_SIZE": str(count),
        "DP_ATTENTION": "true",
        "KV_OFFLOADING": "none",
        "SPEC_DECODING": "none",
        "DISAGG": "false",
        "IS_MULTINODE": "false",
        "IMAGE": f"marin-vllm:{release['version']}@{release['source_commit']}",
        "RECIPE_FINGERPRINT": manifest["recipe_fingerprint"],
    }
    source = manifest["source"]
    if source["kind"] == "github-actions":
        environment["GITHUB_RUN_ATTEMPT"] = str(source["run_attempt"])
    subprocess.run(
        [
            "uv",
            "run",
            "--no-project",
            "--python",
            "3.12",
            "--with",
            "pydantic==2.12.5",
            "--with",
            "PyYAML==6.0.3",
            "python",
            "-m",
            "infx.results.agentic.process_agentic_result",
        ],
        cwd=processor / "inferencex-e2e",
        env=environment,
        check=True,
        timeout=300,
    )
    result = json.loads((aggregate / "agg.json").read_text())
    if result["num_gpus"] != count:
        raise ValueError("Processor GPU normalization differs from the actual allocated device count")


def collect_terminal(
    status: JobStatus,
    manifest: dict,
    retention: str,
    work: Path,
    processor: Path,
    now: float,
    *,
    client: IrisClient | None = None,
) -> dict | None:
    """Retain every final outcome, and process only its complete successful attempt.

    Returns:
        The retained receipt, or None while Iris output finalization is pending.
    """
    if len(status.tasks) > 1:
        raise ValueError("A benchmark point must be one single-node Iris task")
    if not status.tasks or not status.tasks[0].attempts:
        # A scheduling rejection or cancellation can finish before a pod exists.
        prefix = prefix_join(retention, f"{manifest['profile']}/c{manifest['concurrency']}/no-attempt")
        write_record(prefix_join(prefix, "iris-status.json"), json.loads(json.dumps(status, default=json_value)))
        receipt = {
            "job_id": str(status.job_id),
            "attempt_uid": None,
            "attempt_number": None,
            "native_state": status.state.value,
            "status": "failed",
            "source": manifest["source"],
            "iris_status_uri": prefix_join(prefix, "iris-status.json"),
            "reason": "Job ended before creating an Iris attempt",
        }
        write_record(prefix_join(prefix, "receipt.json"), receipt)
        return receipt
    task = status.tasks[0]
    attempt = next(item for item in task.attempts if item.attempt_number == task.current_attempt_number)
    prefix = prefix_join(retention, f"{manifest['profile']}/c{manifest['concurrency']}/{attempt.attempt_uid}")
    directory = work / manifest["profile"] / f"c{manifest['concurrency']}" / attempt.attempt_uid
    directory.mkdir(parents=True, exist_ok=True)
    receipt = {
        "job_id": str(status.job_id),
        "attempt_uid": attempt.attempt_uid,
        "attempt_number": attempt.attempt_number,
        "native_state": status.state.value,
        "status": "failed",
        "source": manifest["source"],
        "iris_status_uri": prefix_join(prefix, "iris-status.json"),
    }
    snapshot = json.loads(json.dumps(status, default=json_value))
    write_record(prefix_join(prefix, "iris-status.json"), snapshot)
    (directory / "iris-status.json").write_text(json.dumps(snapshot, indent=2))
    receipt["previous_attempts"] = []
    receipt["stream_logs"] = []
    for previous in task.attempts:
        previous_prefix = prefix_join(
            retention, f"{manifest['profile']}/c{manifest['concurrency']}/{previous.attempt_uid}"
        )
        previous_archive = previous.output_archive
        if client is not None and (
            previous_archive is None or previous_archive.state != TaskOutputArchiveState.UPLOADED
        ):
            entries = (
                client.task(task.task_id)
                .attempt(previous.attempt_number)
                .logs(max_lines=STREAM_LOG_LINE_LIMIT, tail=True)
            )
            logs_uri = prefix_join(previous_prefix, "stream-logs.json")
            write_record(
                logs_uri,
                {
                    "attempt_uid": previous.attempt_uid,
                    "tail": True,
                    "max_lines": STREAM_LOG_LINE_LIMIT,
                    "entries": entries,
                },
            )
            receipt["stream_logs"].append(
                {
                    "attempt_uid": previous.attempt_uid,
                    "uri": logs_uri,
                    "returned_lines": len(entries),
                    "max_lines": STREAM_LOG_LINE_LIMIT,
                }
            )
        if previous.attempt_uid == attempt.attempt_uid:
            continue
        retained = {"attempt_uid": previous.attempt_uid, "state": previous.state.value}
        if previous_archive is not None and previous_archive.state == TaskOutputArchiveState.UPLOADED:
            _, previous_digest = retain_archive(
                previous_archive, previous_prefix, directory.parent / previous.attempt_uid
            )
            retained.update({"archive_uri": prefix_join(previous_prefix, "outputs.tar.zst"), "sha256": previous_digest})
        else:
            retained["reason"] = "No uploaded archive; native attempt metadata remains in iris-status.json"
        receipt["previous_attempts"].append(retained)
    archive = attempt.output_archive
    if archive is None:
        finished = attempt.finished_at or status.finished_at
        if finished is not None and now - finished.epoch_ms() / 1000 < 600:
            return None
        receipt["reason"] = "No output receipt after the bounded Iris finalization period"
        write_record(prefix_join(prefix, "receipt.json"), receipt)
        return receipt
    if archive.state == TaskOutputArchiveState.UPLOADED:
        local, digest = retain_archive(archive, prefix, directory)
        receipt.update({"archive_uri": prefix_join(prefix, "outputs.tar.zst"), "sha256": digest})
        unpack(local, directory / "raw")
        if status.state == JobState.SUCCEEDED and attempt.state == TaskState.SUCCEEDED and attempt.exit_code == 0:
            point = directory / "raw" / "point"
            try:
                receipt["validation"] = validate_window(point, manifest, attempt.attempt_uid)
            except (ValueError, KeyError, FileNotFoundError) as error:
                receipt["reason"] = str(error)
            else:
                artifacts = directory / "artifacts"
                process_point(point, manifest, processor, artifacts)
                bundle_uri, bundle_digest = retain_processed_artifacts(
                    artifacts, directory / "inferencex.tar.zst", prefix
                )
                receipt.update(
                    {
                        "status": "valid",
                        "artifacts_uri": bundle_uri,
                        "artifacts_sha256": bundle_digest,
                    }
                )
    else:
        receipt["reason"] = f"Iris output capture: {archive.state.value}: {archive.error_message}"
    write_record(prefix_join(prefix, "receipt.json"), receipt)
    (directory / "receipt.json").write_text(json.dumps(receipt, indent=2))
    return receipt
