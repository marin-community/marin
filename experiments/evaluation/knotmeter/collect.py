# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recover all native attempts, process upstream exports and account for GPU time."""

import json
import os
import subprocess
from pathlib import Path

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.client.client import IrisClient
from iris.client.workload import TaskOutputArchiveState
from iris.cluster.types import JobName
from iris.resources.state import JobState, is_job_finished
from marin.inference.config import VLLM_METRIC_PREFIX
from marin.profiling.trace_summary import sha256_for_path
from rigging.filesystem.storage_path import prefix_join

from experiments.evaluation.knotmeter.allocation import allocation_cost
from experiments.evaluation.knotmeter.artifacts import (
    PROFILE_EXPORT_RELATIVE_PATH,
    SERVER_METRICS_RELATIVE_PATH,
    copy_file,
    read_record,
    restore_server_metrics,
    unpack,
    write_record,
)
from experiments.evaluation.knotmeter.config import Profile
from experiments.evaluation.knotmeter.recipe import (
    ATTEMPT_RECEIPT_NAME,
    DIRECTORY,
    PointManifest,
    TrialPlan,
    processor_package,
    unmeasured_result,
)


def process_point(point: Path, manifest: PointManifest, artifacts: Path) -> dict:
    """Use unmodified InferenceX with the actual physical device count."""
    restore_server_metrics(point)
    series = manifest["series"]
    profile = Profile.model_validate(series["profile"])
    count = profile.gpus.count
    dp = count // profile.serving.tensor_parallel_size
    # The pinned processor counts a single serving pool through TP, ignoring DP_SIZE.
    # Actual vLLM topology remains in the resolved profile and deployment checks.
    environment = os.environ | {
        "RESULT_DIR": str(point.resolve()),
        "AGENTIC_OUTPUT_DIR": str(artifacts.resolve()),
        "RESULT_FILENAME": "agg",
        "MODEL": series["model"],
        "MODEL_PREFIX": series["model_prefix"],
        "FRAMEWORK": "vllm",
        "PRECISION": series["precision"],
        "RUNNER_TYPE": manifest["profile"],
        "CONC": str(manifest["concurrency"]),
        "TP": str(count),
        "PP_SIZE": "1",
        "EP_SIZE": str(dp),
        "DP_SIZE": str(dp),
        "DP_ATTENTION": "true",
        "KV_OFFLOADING": "none",
        "SPEC_DECODING": "none",
        "DISAGG": "false",
        "IS_MULTINODE": "false",
        "IMAGE": f"marin-vllm:{manifest['vllm_release']['version']}@{manifest['vllm_release']['source_commit']}",
        "RECIPE_FINGERPRINT": manifest["recipe_fingerprint"],
    }
    # GitHub provenance, when present in the calling shell, must not identify a manual trial.
    for key in ("GITHUB_RUN_ATTEMPT", "GITHUB_RUN_ID"):
        environment.pop(key, None)
    subprocess.run(
        [
            "uvx",
            "--python",
            "3.12",
            "--from",
            processor_package(series),
            "--with-requirements",
            str(DIRECTORY / "requirements-processor.txt"),
            "python",
            "-m",
            "infx.results.agentic.process_agentic_result",
        ],
        env=environment,
        check=True,
        timeout=300,
    )
    result = json.loads((artifacts / "agg.json").read_text())
    if result["num_gpus"] != count:
        raise ValueError("Upstream processor normalized to a different GPU allocation")
    return result


def validate_window(point: Path, manifest: PointManifest, record: dict) -> dict:
    """Preserve native validity and independently check retained window admission."""
    export = json.loads((point / PROFILE_EXPORT_RELATIVE_PATH).read_text())
    restore_server_metrics(point)
    metrics = json.loads((point / SERVER_METRICS_RELATIVE_PATH).read_text())
    successful_metrics = record["worker_metrics_url"] in metrics["summary"]["endpoints_successful"] and any(
        name.startswith(VLLM_METRIC_PREFIX) for name in metrics["metrics"]
    )
    coverage = [
        row for row in export["metadata"].get("metric_duration_coverage", []) if row["phase_name"] == "profiling"
    ]
    complete = len(coverage) == 1 and coverage[0]["expected_duration_seconds"] == manifest["series"]["duration"]
    if complete:
        complete = all(
            coverage[0][key] >= coverage[0]["required_ratio"] for key in ("ttft_ratio", "inter_token_latency_ratio")
        )
    phase_counts = {}
    with (point / "aiperf_artifacts/profile_export.jsonl").open() as stream:
        for line in stream:
            request = json.loads(line)
            phase = request["metadata"]["benchmark_phase"]
            counts = phase_counts.setdefault(phase, {"successes": 0, "errors": 0})
            counts["errors" if request.get("error") else "successes"] += 1
    profiling = phase_counts.get("profiling", {"successes": 0, "errors": 0})
    successes, errors = profiling["successes"], profiling["errors"]
    ratio = errors / (successes + errors) if successes + errors else None
    native = export["metadata"].get("submission_valid") is True and not export.get("was_cancelled")
    # The native command's zero exit is recorded before teardown. Export files
    # alone can also survive an interrupted or failed benchmark process.
    completed = isinstance(record.get("measured_unix"), (int, float))
    valid = (
        native
        and completed
        and successful_metrics
        and complete
        and successes > 0
        and ratio <= manifest["series"]["max_profiling_error_ratio"]
    )
    return {
        "submission_valid": native,
        "measurement_completed": completed,
        "worker_metrics": successful_metrics,
        "coverage_complete": complete,
        "profiling_successes": successes,
        "profiling_errors": errors,
        "profiling_error_ratio": ratio,
        "request_counts_by_phase": phase_counts,
        "protocol_valid": bool(valid and not manifest["series"]["unsafe_override"]),
        "diagnostic": manifest["diagnostic"],
    }


def collect_trial(client: IrisClient, plan: TrialPlan, work: Path) -> dict:
    """Read terminal work only. Recovery cannot submit another allocation."""
    try:
        status = client.job_status(JobName.from_string(plan["gpu_job_id"]))
    except ConnectError as error:
        if error.code != Code.NOT_FOUND:
            raise
        return read_record(prefix_join(plan["trial_uri"], "result.json"))
    if not is_job_finished(status.state):
        raise ValueError(f"GPU work is still live: {status.job_id}. Reattach or cancel it before collecting.")
    manifest = plan["manifest"]
    profile = Profile.model_validate(manifest["series"]["profile"])
    tasks = sorted(client.list_tasks(status.job_id), key=lambda task: str(task.task_id))
    attempts = []
    leaders = []
    cost = 0.0
    cost_complete = True
    final_uid = None
    for task_summary in tasks:
        task = client.task_status(task_summary.task_id)
        index = task.task_id.require_task()[1]
        for attempt in task.attempts:
            if index == 0 and attempt.attempt_number == task.current_attempt_number:
                final_uid = attempt.attempt_uid
            prefix = prefix_join(plan["trial_uri"], f"attempts/{attempt.attempt_uid}/hosts/{index}")
            start = attempt.started_at.epoch_ms() / 1000 if attempt.started_at else None
            finish = attempt.finished_at.epoch_ms() / 1000 if attempt.finished_at else None
            gpu_seconds, allocation = allocation_cost(
                plan, attempt.attempt_uid, attempt.pod_uid, finish, profile.gpus.per_host
            )
            if gpu_seconds is None:
                cost_complete = False
            else:
                cost += gpu_seconds
            native = {
                "task_id": str(task.task_id),
                "attempt_uid": attempt.attempt_uid,
                "attempt_number": attempt.attempt_number,
                "state": attempt.state.value,
                "started_unix": start,
                "finished_unix": finish,
                "gpu_seconds": gpu_seconds,
                "node_name": attempt.node_name,
                "pod_uid": attempt.pod_uid,
                "error": attempt.error_message or attempt.terminal_reason,
                "evidence": prefix,
            }
            native["allocation"] = allocation
            local = work / attempt.attempt_uid / str(index)
            local.mkdir(parents=True, exist_ok=True)
            try:
                record = read_record(prefix_join(prefix, ATTEMPT_RECEIPT_NAME))
            except FileNotFoundError as missing:
                archive = attempt.output_archive
                if archive is None or archive.state != TaskOutputArchiveState.UPLOADED:
                    native["output_state"] = archive.state.value if archive else "unavailable"
                    attempts.append(native)
                    continue
                raw = local / "iris.tar.zst"
                copy_file(archive.uri, str(raw))
                if sha256_for_path(raw) != archive.sha256:
                    raise ValueError("Iris output archive checksum changed") from missing
                unpack(raw, local / "iris")
                point = local / "iris/point"
                if not (point / "attempt.json").is_file():
                    native["output_state"] = "no-worker-record"
                    attempts.append(native)
                    continue
                record = read_record(str(point / "attempt.json"))
                native["iris_archive"] = archive.uri
            else:
                raw = local / "native.tar.zst"
                copy_file(record["archive_uri"], str(raw))
                if sha256_for_path(raw) != record["archive_sha256"]:
                    raise ValueError("Retained native archive checksum changed")
                unpack(raw, local / "point")
                point = local / "point"
            if record["iris_attempt_uid"] != attempt.attempt_uid or record["manifest"] != manifest:
                raise ValueError("Attempt output differs from its frozen plan or native identity")
            native["worker"] = record
            attempts.append(native)
            if index == 0 and (point / "aiperf_artifacts/profile_export.jsonl").is_file():
                final_export = point / PROFILE_EXPORT_RELATIVE_PATH
                metrics = point / SERVER_METRICS_RELATIVE_PATH
                if not final_export.is_file() or not (metrics.is_file() or metrics.with_suffix(".json.zst").is_file()):
                    native["measurement_error"] = "Incomplete AgentX exports: final aggregate or server metrics missing"
                    continue
                validity = validate_window(point, manifest, record)
                aggregate = process_point(point, manifest, local / "processed")
                aggregate_uri = prefix_join(prefix, "aggregate.json")
                write_record(aggregate_uri, aggregate)
                leaders.append(
                    {
                        "attempt_uid": attempt.attempt_uid,
                        "validity": validity,
                        "aggregate": aggregate,
                        "checks": record.get("checks", []),
                        "gang_attempts": record.get("gang_attempts", {}),
                        "aggregate_uri": aggregate_uri,
                        "evidence": prefix,
                    }
                )
    final = next((row for row in leaders if row["attempt_uid"] == final_uid), None)
    if final:
        identities = {
            row["task_id"]: row["attempt_uid"]
            for row in attempts
            if row["attempt_uid"] in final["gang_attempts"].values()
        }
        uuids = [
            uuid
            for row in attempts
            if row["attempt_uid"] in final["gang_attempts"].values()
            for uuid in row.get("worker", {}).get("gpu_uuids", [])
        ]
        final["validity"]["gang_valid"] = (
            identities == final["gang_attempts"]
            and len(set(uuids)) == profile.gpus.count
            and len(uuids) == profile.gpus.count
        )
    result = {
        "schema_version": 1,
        "trial_id": plan["trial_id"],
        "trial_uri": plan["trial_uri"],
        "manifest": manifest,
        "job_state": status.state.value,
        "execution_cluster_id": status.execution_cluster_id,
        "job_succeeded": status.state == JobState.SUCCEEDED,
        "attempts": attempts,
        "measurement": final,
        "gpu_seconds": cost,
        "cost_complete": cost_complete,
        "cost_basis": (
            "sum of pod scheduling through container finish, or Iris-observed terminal time "
            "when the pod expired; queueing excluded"
        ),
    }
    write_record(prefix_join(plan["trial_uri"], "result.json"), result)
    return result


def refresh_costs(plan: TrialPlan) -> dict:
    """Apply later allocation snapshots without reprocessing unchanged exports."""
    uri = prefix_join(plan["trial_uri"], "result.json")
    result = read_record(uri)
    if not result["attempts"]:
        return result
    gpus = result["manifest"]["series"]["profile"]["gpus"]["per_host"]
    attempts = []
    for attempt in result["attempts"]:
        seconds, allocation = allocation_cost(
            plan, attempt["attempt_uid"], attempt["pod_uid"], attempt["finished_unix"], gpus
        )
        attempts.append(attempt | {"gpu_seconds": seconds, "allocation": allocation})
    result |= {
        "attempts": attempts,
        "gpu_seconds": sum(row["gpu_seconds"] for row in attempts if row["gpu_seconds"] is not None),
        "cost_complete": all(row["gpu_seconds"] is not None for row in attempts),
    }
    write_record(uri, result)
    return result


def record_preparation_failure(plan: TrialPlan, error: str, *, gpu_job_absent: bool = True) -> None:
    """Preserve preparation/submission failures, including uncertain GPU admission."""
    result = unmeasured_result(plan, "preparation-failed", cost_complete=gpu_job_absent) | {
        "schema_version": 1,
        "error": error,
    }
    write_record(prefix_join(plan["trial_uri"], "result.json"), result)
