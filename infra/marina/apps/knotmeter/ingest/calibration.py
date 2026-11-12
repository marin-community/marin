# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import saved private calibration observations without fabricating GitHub runs."""

import hashlib
import json
import math
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .adapter import ImportRun, read_json
from .replay import prepare_requests, profile_records

CALIBRATION_SECONDS = 900

# Placement, listening ports and output paths do not change the serving recipe.
PLACEMENT_ARGUMENTS = {
    "--host",
    "--port",
    "--master-addr",
    "--master-port",
    "--data-parallel-address",
    "--data-parallel-rpc-port",
    "--node-rank",
    "--hf-config-path",
}
LOCAL_ENVIRONMENT = {
    "VLLM_CACHE_ROOT",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "CUDA_CACHE_PATH",
    "VLLM_ENGINE_READY_TIMEOUT_S",
}


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def curve_controls(attempt: dict[str, Any], duration: int) -> dict[str, Any]:
    """Keep actual serving controls and pins; omit only deployment-local arguments."""
    arguments = iter(attempt["server_command"])
    command = []
    for argument in arguments:
        if argument in PLACEMENT_ARGUMENTS:
            next(arguments)
        else:
            command.append(argument)
    return {
        "source": attempt["source"],
        "server_command": command,
        "server_environment": {k: v for k, v in attempt["server_environment"].items() if k not in LOCAL_ENVIRONMENT},
        "runtime": attempt["private_runtime"],
        "workload": attempt["workload"]["key"],
        "topology": attempt["server_parallelism"],
        "physical_gpus": attempt["all_allocated_gpus"],
        "duration_seconds": duration,
    }


def prepare_calibrations(root: Path, job: str | None = None) -> Iterator[ImportRun]:
    """Validate native receipts and stream valid 15-minute observations one at a time."""
    index = root / "curve-summary.json"
    windows_path = root / "window-effects.json"
    windows = read_json(windows_path)
    by_job = {row["job"]: row for row in windows["rows"]}
    selected = [
        row
        for row in read_json(index)["rows"]
        if row["native_submission_valid"] and not row["screening"] and (job is None or row["job"] == job)
    ]
    if not selected:
        raise ValueError("No valid calibration observations match the requested job")
    for row in selected:
        processed = root / Path(row["summary"]).parent
        raw = processed / "restored-point"
        aggregate_path = processed / "aggregate/agg.json"
        aggregate = read_json(aggregate_path)
        attempt_path = raw / "attempt.json"
        attempt = read_json(attempt_path)
        submission_path = root / row["submission"]
        submission = read_json(submission_path)
        config = submission["series"]
        profile_path = raw / "aiperf_artifacts/profile_export.jsonl"
        native_path = profile_path.with_name("profile_export_aiperf.json")
        native = read_json(native_path)
        validation = native["metadata"]
        phases = [p for p in native["input_config"]["phases"] if p["kind"] == "profiling"]
        coverage = validation["metric_duration_coverage"]
        window = by_job[row["job"]]
        phase_log = root / window["native_phase_log"]
        physical_gpus = attempt["all_allocated_gpus"]
        topology = attempt["server_parallelism"]
        task = attempt["iris_task_id"].rsplit(":", 1)[0]
        source = attempt["source"]
        if (
            source["kind"] != "private-calibration"
            or source != submission["campaign"]["source"]
            or submission["job_id"] != row["job"]
            or not task.startswith(row["job"] + "/")
            or attempt["gang_attempts"].get(task) != attempt["iris_attempt_uid"]
            or attempt["status"] != "succeeded"
            or attempt["screening"]
            or validation["submission_valid"] is not True
            or len(phases) != 1
            or phases[0]["duration"] != CALIBRATION_SECONDS
            or phases[0]["concurrency"] != aggregate["conc"]
            or config["duration"] != CALIBRATION_SECONDS
            or len(coverage) != 1
            or coverage[0]["expected_duration_seconds"] != CALIBRATION_SECONDS
            or any(coverage[0][k] < coverage[0]["required_ratio"] for k in ["ttft_ratio", "inter_token_latency_ratio"])
            or aggregate["recipe_fingerprint"] != attempt["recipe_fingerprint"]
            or aggregate["num_gpus"] != physical_gpus
            or physical_gpus != submission["count"]
            or topology != {k: row[k] for k in ["tp", "dp", "ep"]} | {"pp": 1}
            or aggregate["model"] != config["model"]
            or aggregate["precision"] != config["precision"]
            or aggregate["scenario_type"] != "agentic-coding"
            or attempt["workload"] != submission["campaign"]["manifest"]["workload"]
            or file_digest(native_path) != read_json(processed / "summary.json")["input_export_sha256"]
            or file_digest(phase_log) != window["native_phase_log_sha256"]
        ):
            raise ValueError(f"Calibration receipts, coverage or topology disagree: {row['job']}")
        origin, requests = prepare_requests(lambda path=profile_path: profile_records(path))
        profiling = [r for r in requests if r["phase"] == "profiling" and not r["cancelled"]]
        successful = sum(not r["error"] for r in profiling)
        throughput = aggregate["request_metrics"]["throughput"]
        if (
            len(requests) != aggregate["num_requests_total"]
            or successful != aggregate["num_requests_successful"]
            or successful != native["request_count"]["avg"]
            or successful != window["profiling_requests"]
            or sum(bool(r["error"]) for r in profiling) != window["profiling_errors"]
            or window["profiling_sent"] != successful + window["profiling_errors"] + window["profiling_cancelled"]
            or not math.isclose(
                throughput["per_gpu"]["output_tput_tps"],
                throughput["output"]["tokens_per_second"] / physical_gpus,
                abs_tol=0.00001,
            )
        ):
            raise ValueError(f"Calibration request accounting or GPU normalization disagrees: {row['job']}")
        controls = curve_controls(attempt, CALIBRATION_SECONDS)
        curve_fingerprint = hashlib.sha256(json.dumps(controls, sort_keys=True).encode()).hexdigest()
        key = f"calibration-{attempt['iris_attempt_uid']}"
        created = datetime.fromisoformat(native["start_time"]).replace(tzinfo=UTC).isoformat()
        hardware = config["profiles"][attempt["profile"]]
        lineage = attempt["workload"]["lineage"]
        files = [
            index,
            windows_path,
            root / row["summary"],
            submission_path,
            aggregate_path,
            attempt_path,
            native_path,
            profile_path,
            phase_log,
            processed / "processor-command.json",
        ]
        run = {
            "key": key,
            "date": created[:10],
            "source_created_at": created,
            "source": (
                source
                | {"job_id": row["job"], "attempt_uid": attempt["iris_attempt_uid"], "artifact_uri": row["raw_archive"]}
            ),
            "measurement_seconds": CALIBRATION_SECONDS,
            "topology": topology,
            "calibration": {
                "curve_fingerprint": curve_fingerprint,
                "curve_controls": controls,
                "profiling_sent": window["profiling_sent"],
                "profiling_cancelled": window["profiling_cancelled"],
                "profiling_errors": window["profiling_errors"],
                "caveat": windows["scope"],
                "cancellation_source": window["native_phase_log"],
                "timeline_contains_cancelled_tail": False,
                "raw_archive_sha256": row["raw_archive_sha256"],
            },
            "model": aggregate["model"],
            "hardware": hardware["gpu"],
            "physical_gpus": physical_gpus,
            "hosts": hardware["hosts"],
            "concurrency": aggregate["conc"],
            "precision": aggregate["precision"],
            "scenario": "AgentX",
            "runtime": aggregate["image"],
            "recipe": aggregate["recipe_fingerprint"],
            "model_revision": config["model_revision"],
            "dataset": lineage["dataset"],
            "dataset_revision": lineage["dataset_revision"],
            "dataset_sessions": lineage["sessions"],
            "workload_key": attempt["workload"]["key"],
            "request_count": len(requests),
            "profiling_requests": successful,
            "origin_ns": str(origin),
            "metrics": aggregate["request_metrics"],
            "framework": aggregate["framework"],
            "aggregate": aggregate,
            "diagnostic_aggregates": {
                "isl": aggregate["request_metrics"].get("tokens", {}).get("input"),
                "osl": aggregate["request_metrics"].get("tokens", {}).get("output_actual"),
                "kvCacheUtil": None,
                "prefixCacheHitRate": None,
            },
            "configuration": config,
            "manifest": attempt,
            "validation": validation,
            "source_files": {str(path.relative_to(root)): file_digest(path) for path in files},
            "format_version": 2,
        }
        digest = hashlib.sha256(json.dumps([run, requests], sort_keys=True, allow_nan=False).encode()).hexdigest()
        yield ImportRun(key, digest, run | {"content_sha256": digest}, requests)
