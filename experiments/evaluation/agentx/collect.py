# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retain valid Iris output and compact final failures in upstream result formats."""

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

from iris.client.client import IrisClient
from iris.client.workload import AttemptStatus, JobStatus, TaskOutputArchiveState, TaskStatus
from iris.resources.state import JobState, TaskState
from rigging.filesystem.storage_path import prefix_join

from experiments.evaluation.agentx.artifacts import copy_file, file_digest, retain_artifacts, unpack, write_record
from experiments.evaluation.agentx.recipe import SERIES_PATH, PointManifest, load_series, processor_package, series_path

logger = logging.getLogger(__name__)
FAILURE_LOG_LINES = 100


def validate_attempt(point: Path, manifest: PointManifest, attempt_uid: str) -> dict:
    """Validate a host's native attempt and pinned inputs before retaining its output."""
    record = json.loads((point / "attempt.json").read_text())
    if record["status"] != "succeeded" or record["iris_attempt_uid"] != attempt_uid:
        raise ValueError("Output does not belong to this successful Iris attempt")
    for key in (
        "series_sha256",
        "recipe_fingerprint",
        "profile",
        "concurrency",
        "vllm_release",
        "source",
        "workload",
        "placement",
    ):
        if record[key] != manifest[key]:
            raise ValueError(f"Output changed a pinned input: {key}")
    return record


def validate_window(point: Path, manifest: PointManifest, attempt_uid: str) -> dict:
    """Keep incomplete, replaced and scenario-invalid windows out of the importer."""
    validate_attempt(point, manifest, attempt_uid)
    export = json.loads((point / "aiperf_artifacts" / "profile_export_aiperf.json").read_text())
    metadata = export["metadata"]
    if metadata.get("submission_valid") is not True or export["was_cancelled"]:
        raise ValueError("AgentX did not validate this complete window")
    series = load_series(series_path(manifest["profile"]))
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


def process_point(point: Path, manifest: PointManifest, artifacts: Path) -> None:
    """Invoke the pinned upstream processor in a separate Python environment."""
    series = load_series(series_path(manifest["profile"]))
    profile = manifest["profile"]
    hardware = series["profiles"][profile]
    count = hardware["gpus"] * hardware["hosts"]
    suffix = f"{series['model_prefix']}_{profile}_vllm_{series['precision']}_agentx_c{manifest['concurrency']}"
    aggregate = artifacts / f"bmk_agentic_{suffix}"
    raw = artifacts / f"agentic_{suffix}"
    raw.mkdir(parents=True, exist_ok=True)
    shutil.copytree(point / "aiperf_artifacts", raw / "aiperf_artifacts", dirs_exist_ok=True)
    shutil.copy2(point / "attempt.json", raw / "attempt.json")
    shutil.copy2(point / "hardware.txt", raw / "hardware.txt")
    for name in ("gang.json", "client-load.jsonl"):
        if (point / name).exists():
            shutil.copy2(point / name, raw / name)
    release = manifest["vllm_release"]
    environment = os.environ | {
        "RESULT_DIR": str(point.resolve()),
        "AGENTIC_OUTPUT_DIR": str(aggregate.resolve()),
        "RESULT_FILENAME": "agg",
        "MODEL": series["model"],
        "MODEL_PREFIX": series["model_prefix"],
        "FRAMEWORK": "vllm",
        "PRECISION": series["precision"],
        "RUNNER_TYPE": profile,
        "CONC": str(manifest["concurrency"]),
        # Upstream uses TP for the full attention worker group (native TP * DP).
        "TP": str(series["tensor_parallel"] * count),
        "PP_SIZE": "1",
        "EP_SIZE": str(count),
        "DP_SIZE": str(count),
        "DP_ATTENTION": "true",
        "KV_OFFLOADING": "none",
        "SPEC_DECODING": "none",
        "DISAGG": "false",
        "IS_MULTINODE": str(hardware["hosts"] > 1).lower(),
        "IMAGE": f"marin-vllm:{release['version']}@{release['source_commit']}",
        "RECIPE_FINGERPRINT": manifest["recipe_fingerprint"],
    }
    source = manifest["source"]
    if source["kind"] == "github-actions":
        environment["GITHUB_RUN_ATTEMPT"] = str(source["run_attempt"])
    subprocess.run(
        [
            "uvx",
            "--python",
            "3.12",
            "--from",
            processor_package(series),
            "--with-requirements",
            str(SERIES_PATH.with_name("requirements-processor.txt")),
            "python",
            "-m",
            "infx.results.agentic.process_agentic_result",
        ],
        env=environment,
        check=True,
        timeout=300,
    )
    result = json.loads((aggregate / "agg.json").read_text())
    if result["num_gpus"] != count:
        raise ValueError("Processor GPU normalization differs from the actual allocated device count")


def current_attempt(task: TaskStatus) -> AttemptStatus | None:
    return next((item for item in task.attempts if item.attempt_number == task.current_attempt_number), None)


def validate_gang(tasks: list[TaskStatus], point: Path, manifest: PointManifest, directory: Path) -> list[dict]:
    """Keep one leader window fenced to all of its unchanged, successful host attempts."""
    series = load_series(series_path(manifest["profile"]))
    profile = series["profiles"][manifest["profile"]]
    if len(tasks) != profile["hosts"]:
        raise ValueError("Serving host count differs from the pinned profile")
    records = []
    archives = []
    for task in tasks:
        attempt = current_attempt(task)
        if attempt is None or attempt.state != TaskState.SUCCEEDED or attempt.exit_code != 0:
            raise ValueError("Every serving host must have a successful current native attempt")
        archive = attempt.output_archive
        if archive is None or archive.state != TaskOutputArchiveState.UPLOADED:
            raise ValueError("A serving host has no uploaded native output archive")
        index = task.task_id.require_task()[1]
        local = directory / f"host-{index}.tar.zst"
        if index == 0:
            host_point = point
            local = directory / "outputs.tar.zst"
        else:
            copy_file(archive.uri, str(local))
            if file_digest(local) != archive.sha256:
                raise ValueError("Serving-host output differs from Iris's uploader SHA256")
            unpack(local, directory / f"host-{index}")
            host_point = directory / f"host-{index}" / "point"
        record = validate_attempt(host_point, manifest, attempt.attempt_uid)
        if record["iris_node_name"] not in manifest["placement"]["nodes"]:
            raise ValueError("A serving host was outside the verified spare-node list")
        if len(record["gpu_uuids"]) != profile["gpus"]:
            raise ValueError("A host's physical GPU count differs from its allocation")
        records.append(record)
        archives.append(
            {
                "task_id": str(task.task_id),
                "attempt_uid": attempt.attempt_uid,
                "local": str(local),
                "sha256": archive.sha256,
            }
        )
    expected = {row["task_id"]: row["attempt_uid"] for row in archives}
    leader = json.loads((point / "attempt.json").read_text())
    if leader["gang_attempts"] != expected:
        raise ValueError("The completed host attempts differ from the measured window's gang")
    if len({row["iris_node_name"] for row in records}) != profile["hosts"]:
        raise ValueError("Serving hosts do not occupy distinct physical nodes")
    if len({uuid for row in records for uuid in row["gpu_uuids"]}) != profile["gpus"] * profile["hosts"]:
        raise ValueError("Serving hosts do not expose the full distinct physical GPU allocation")
    write_record(str(point / "gang.json"), {"tasks": records, "attempts": expected, "placement": manifest["placement"]})
    return archives


def collect_terminal(
    status: JobStatus,
    manifest: PointManifest,
    retention: str,
    work: Path,
    now: float,
    *,
    client: IrisClient | None = None,
) -> dict | None:
    """Retain valid final output; return None while successful capture is pending.

    Failed and preempted archives remain under Iris retention. Only the final
    failure's reason, configuration and bounded log tail become durable.
    """
    tasks = sorted(status.tasks, key=lambda task: task.task_id.require_task()[1])
    task = tasks[0] if tasks else None
    attempt = current_attempt(task) if task is not None else None
    if status.state == JobState.SUCCEEDED:
        for member in tasks:
            native = current_attempt(member)
            if native is not None and native.output_archive is None:
                finished = native.finished_at or status.finished_at
                if finished is not None and now - finished.epoch_ms() / 1000 < 600:
                    return None
    uid = attempt.attempt_uid if attempt is not None else "no-attempt"
    prefix = prefix_join(retention, f"{manifest['profile']}/c{manifest['concurrency']}/{uid}")
    receipt = {
        "job_id": str(status.job_id),
        "attempt_uid": attempt.attempt_uid if attempt is not None else None,
        "attempt_number": attempt.attempt_number if attempt is not None else None,
        "native_state": status.state.value,
        "status": "failed",
        "source": manifest["source"],
        "configuration": {
            "series": load_series(series_path(manifest["profile"])),
            "profile": manifest["profile"],
            "concurrency": manifest["concurrency"],
            "vllm_release": manifest["vllm_release"],
            "recipe_fingerprint": manifest["recipe_fingerprint"],
            "workload_key": manifest["workload"]["key"],
        },
    }
    if attempt is None:
        receipt["reason"] = status.error_message or "Job ended before creating an Iris attempt"
    elif status.state != JobState.SUCCEEDED or attempt.state != TaskState.SUCCEEDED or attempt.exit_code != 0:
        receipt["reason"] = (
            attempt.error_message
            or attempt.terminal_reason
            or status.error_message
            or (f"Iris attempt ended in {attempt.state.value} with exit code {attempt.exit_code}")
        )
    else:
        archive = attempt.output_archive
        if archive is None:
            finished = attempt.finished_at or status.finished_at
            if finished is not None and now - finished.epoch_ms() / 1000 < 600:
                return None
            receipt["reason"] = "No output receipt after the bounded Iris finalization period"
        elif archive.state != TaskOutputArchiveState.UPLOADED:
            receipt["reason"] = f"Iris output capture: {archive.state.value}: {archive.error_message}"
        else:
            directory = work / manifest["profile"] / f"c{manifest['concurrency']}" / uid
            directory.mkdir(parents=True, exist_ok=True)
            local = directory / "outputs.tar.zst"
            copy_file(archive.uri, str(local))
            if file_digest(local) != archive.sha256:
                raise ValueError("Downloaded output differs from Iris's uploader SHA256")
            unpack(local, directory / "raw")
            point = directory / "raw" / "point"
            try:
                receipt["validation"] = validate_window(point, manifest, uid)
                hardware = load_series(series_path(manifest["profile"]))["profiles"][manifest["profile"]]
                peers = validate_gang(tasks, point, manifest, directory) if hardware["hosts"] > 1 else []
            except (ValueError, KeyError, FileNotFoundError) as error:
                receipt["reason"] = str(error)
            else:
                archive_uri = prefix_join(prefix, "outputs.tar.zst")
                copy_file(str(local), archive_uri)
                receipt.update({"archive_uri": archive_uri, "sha256": archive.sha256})
                if peers:
                    receipt["host_archives"] = []
                    for index, peer in enumerate(peers):
                        uri = prefix_join(prefix, f"hosts/{index}/outputs.tar.zst")
                        copy_file(peer.pop("local"), uri)
                        receipt["host_archives"].append(peer | {"uri": uri})
                artifacts = directory / "artifacts"
                process_point(point, manifest, artifacts)
                bundle_uri, bundle_digest = retain_artifacts(
                    artifacts, directory / "inferencex.tar.zst", prefix, name="inferencex"
                )
                receipt.update({"status": "valid", "artifacts_uri": bundle_uri, "artifacts_sha256": bundle_digest})
    if receipt["status"] == "failed" and task is not None and attempt is not None and client is not None:
        try:
            receipt["log_tail"] = [
                entry.data
                for entry in (
                    client.task(task.task_id)
                    .attempt(attempt.attempt_number)
                    .logs(max_lines=FAILURE_LOG_LINES, tail=True)
                )
            ]
        except Exception:
            # A diagnostics read must not discard the terminal failure record.
            logger.warning("Could not fetch failure logs for %s", status.job_id, exc_info=True)
    write_record(prefix_join(prefix, "receipt.json"), receipt)
    return receipt
