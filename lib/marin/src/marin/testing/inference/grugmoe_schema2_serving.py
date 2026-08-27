# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Whole-node qualification for schema-2 GrugMoE serving through Marin.

The public cluster test submits this function as a CPU coordinator. The
coordinator lowers the same ``ModelConfig`` used by the evaluation launcher,
starts the native Iris/vLLM gang, exercises its single leader endpoint, and
prints a receipt only after the complete gang and endpoint have been released.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import logging
import re
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from typing import Any, Protocol

import numpy as np
import requests
from fray.types import JobStatus
from iris.client.client import TaskLogEntry, iris_ctx
from iris.cluster.types import JobName
from iris.resources.state import TaskState
from iris.rpc import job_pb2
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.model_config import ModelConfig
from marin.evaluation.serving_config import inference_config_for_model
from marin.external_dependencies import VLLM_GPU_RELEASE
from marin.inference.iris import RemoteInferenceSession, remote_inference
from marin.inference.vllm_release import vllm_gpu_wheel_for_architecture, vllm_gpu_wheel_provenance

logger = logging.getLogger(__name__)

EXPECTED_VLLM_SOURCE_COMMIT = "3caca1d1237434f427822beabc827ed70defeabe"
REAL_MODEL_KEY = "rav-ladder-d1536"
REAL_ARTIFACT_TREE_SHA256 = "8cc7d8f1f49a387bc51058d0deb3cfe47cb843126465cbc1b22f3fe3f7f7261b"
REAL_RAW_METADATA_SHA256 = "128b0e2779b943fbef139132d4add4961e9e12cda3358b65951169940aa556fe"
REAL_PROMPT_TOKEN_IDS = (128000, 791, 6864, 315, 9822, 374)
REAL_GENERATED_TOKEN_IDS = (12366, 13, 578, 3224, 374, 7559, 304, 11104)

GPU_COUNT = 8
MAX_MODEL_LEN = 4096
MAX_NUM_BATCHED_TOKENS = 4096
MAX_NUM_SEQS = 64
RETURNED_LOGPROBS = 64
SELECTED_LOGPROB_ATOL = 0.25
TOP_LOGPROB_ATOL = 0.50
HTTP_CONNECT_TIMEOUT_SECONDS = 30.0
HTTP_READ_TIMEOUT_SECONDS = 5 * 60.0
RUNTIME_EVIDENCE_TIMEOUT_SECONDS = 30 * 60.0
RELEASE_TIMEOUT_SECONDS = 5 * 60.0

_VERIFIED_WHEEL_MARKER = "MARIN_VLLM_WHEEL_VERIFIED="
_HBM_MARKER = "vLLM leader GPU memory snapshot: "
_RUNTIME_TOPOLOGY = re.compile(
    r"GrugMoE effective config: TP=(\d+)/(\d+) PP=(\d+)/(\d+) "
    r"layers=\[(\d+),(\d+)\) DP=(\d+)/(\d+) EP=(\d+)/(\d+) "
    r"use_ep=(True|False) experts=(\d+) local=(\d+)"
)
_REQUESTED_TOPOLOGY = re.compile(
    r"vLLM requested topology: tasks=(\d+) GPUs/task=(\d+) DP=(\d+) EP=(\d+) " r"PP=(\d+) TP=1 task=(\d+)"
)
_KV_CAPACITY = re.compile(r"GPU KV cache size: ([\d,]+) tokens")
_RUNTIME_EVIDENCE_MARKERS = (
    "vLLM requested topology:",
    "Worker placement:",
    "GrugMoE effective config:",
    _VERIFIED_WHEEL_MARKER,
    "GPU KV cache size:",
)
_RUNTIME_MARKER_MAX_LINES = 256


class _RuntimeLogJob(Protocol):
    def logs(
        self,
        *,
        max_lines: int = 0,
        substring: str = "",
        tail: bool = False,
    ) -> list[TaskLogEntry]: ...


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    return value


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_model_file(model: ModelConfig, filename: str) -> bytes:
    return (StoragePath(model.location) / filename).read_bytes()


def _load_model_config(model: ModelConfig) -> tuple[dict[str, Any], str]:
    payload = _read_model_file(model, "config.json")
    return json.loads(payload), _sha256(payload)


def _assert_exact_model_shape(config: Mapping[str, Any]) -> None:
    expected = {
        "grugmoe_artifact_schema_version": 2,
        "hidden_size": 1536,
        "latent_dim": 768,
        "num_hidden_layers": 16,
        "num_experts": 384,
        "num_shared_experts": 2,
        "num_attention_heads": 12,
        "max_position_embeddings": MAX_MODEL_LEN,
    }
    actual = {name: config.get(name) for name in expected}
    if actual != expected:
        raise AssertionError(f"unexpected d1536 model config: expected={expected}, actual={actual}")


def _expected_placements(pipeline_parallel_size: int) -> Counter[str]:
    return Counter(
        {
            (
                "Worker placement: "
                f"process_rank={dp_rank * pipeline_parallel_size + task_index} "
                f"node_rank={task_index} "
                f"local_rank={dp_rank if pipeline_parallel_size == 1 else 0} "
                f"DP={dp_rank}/{GPU_COUNT} EP={dp_rank}/{GPU_COUNT} "
                f"PP={task_index}/{pipeline_parallel_size} TP=0/1 GPU={dp_rank}"
            ): 1
            for task_index in range(pipeline_parallel_size)
            for dp_rank in range(GPU_COUNT)
        }
    )


def _expected_runtime(
    *,
    pipeline_parallel_size: int,
    num_layers: int,
    num_experts: int,
) -> Counter[tuple[Any, ...]]:
    local_experts = num_experts // GPU_COUNT
    return Counter(
        (
            0,
            1,
            task_index,
            pipeline_parallel_size,
            task_index * num_layers // pipeline_parallel_size,
            (task_index + 1) * num_layers // pipeline_parallel_size,
            dp_rank,
            GPU_COUNT,
            dp_rank,
            GPU_COUNT,
            "True",
            num_experts,
            local_experts,
        )
        for task_index in range(pipeline_parallel_size)
        for dp_rank in range(GPU_COUNT)
    )


def _extract_json_records(logs: str, marker: str) -> list[Any]:
    records: list[Any] = []
    for line in logs.splitlines():
        if marker not in line:
            continue
        records.append(json.loads(line.split(marker, 1)[1]))
    return records


def _task_records(session: RemoteInferenceSession) -> list[dict[str, Any]]:
    job_id = JobName.from_string(str(session.jobs[0].job_id))
    return [
        {
            "task_id": str(task.task_id),
            "state": task.state.value,
            "worker_id": task.worker_id,
            "current_attempt_number": task.current_attempt_number,
            "attempt_count": len(task.attempts),
        }
        for task in iris_ctx().client.list_tasks(job_id)
    ]


def _runtime_evidence_logs(job: _RuntimeLogJob) -> str:
    """Fetch only the bounded startup markers the receipt validates.

    Iris/Finelog treats ``max_lines=0`` as a 1,000-line server-default tail. A
    multi-rank vLLM startup can exceed that before the endpoint is ready, so a
    broad tail can lose the earliest topology lines. Server-side substring
    filters keep every required marker query small and deterministic.
    """
    entries: dict[tuple[Any, ...], TaskLogEntry] = {}
    for marker in (*_RUNTIME_EVIDENCE_MARKERS, _HBM_MARKER):
        for entry in job.logs(
            max_lines=_RUNTIME_MARKER_MAX_LINES,
            substring=marker,
            tail=False,
        ):
            identity = (
                entry.timestamp.epoch_ms(),
                str(entry.task_id),
                entry.source,
                entry.data,
                entry.attempt_id,
                entry.key,
            )
            entries[identity] = entry
    ordered = sorted(
        entries.values(),
        key=lambda entry: (
            entry.timestamp.epoch_ms(),
            str(entry.task_id),
            entry.source,
            entry.attempt_id,
            entry.data,
        ),
    )
    return "\n".join(entry.data.rstrip("\n") for entry in ordered)


def _validate_runtime_logs(
    logs: str,
    *,
    pipeline_parallel_size: int,
    num_layers: int,
    num_experts: int,
    expected_wheel: Mapping[str, Any],
) -> dict[str, Any] | None:
    requested: Counter[tuple[int, int, int, int, int, int]] = Counter(
        (
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
            int(match.group(4)),
            int(match.group(5)),
            int(match.group(6)),
        )
        for match in _REQUESTED_TOPOLOGY.finditer(logs)
    )
    expected_requested = Counter(
        (pipeline_parallel_size, GPU_COUNT, GPU_COUNT, GPU_COUNT, pipeline_parallel_size, task_index)
        for task_index in range(pipeline_parallel_size)
    )
    placements = Counter(
        line[line.index("Worker placement:") :].strip() for line in logs.splitlines() if "Worker placement:" in line
    )
    runtime = Counter(
        tuple(int(value) if value.isdigit() else value for value in match.groups())
        for match in _RUNTIME_TOPOLOGY.finditer(logs)
    )
    verified_wheels = _extract_json_records(logs, _VERIFIED_WHEEL_MARKER)
    kv_capacities = [int(value.replace(",", "")) for value in _KV_CAPACITY.findall(logs)]

    if (
        sum(requested.values()) < pipeline_parallel_size
        or sum(placements.values()) < pipeline_parallel_size * GPU_COUNT
        or sum(runtime.values()) < pipeline_parallel_size * GPU_COUNT
        or len(verified_wheels) < pipeline_parallel_size
        or not kv_capacities
    ):
        return None
    if requested != expected_requested:
        raise AssertionError(f"unexpected requested topology: {requested - expected_requested}")
    expected_placement = _expected_placements(pipeline_parallel_size)
    if placements != expected_placement:
        raise AssertionError(
            "unexpected worker placement: "
            f"extra={placements - expected_placement}, missing={expected_placement - placements}"
        )
    expected_runtime = _expected_runtime(
        pipeline_parallel_size=pipeline_parallel_size,
        num_layers=num_layers,
        num_experts=num_experts,
    )
    if runtime != expected_runtime:
        raise AssertionError(
            f"unexpected runtime topology: extra={runtime - expected_runtime}, missing={expected_runtime - runtime}"
        )
    if len(verified_wheels) != pipeline_parallel_size:
        raise AssertionError(f"expected one verified wheel record per Iris task, got {len(verified_wheels)}")
    for record in verified_wheels:
        for name, value in expected_wheel.items():
            if record.get(name) != value:
                raise AssertionError(
                    f"wheel verifier mismatch for {name}: expected={value!r}, actual={record.get(name)!r}"
                )
        if record.get("compute_capability") != "9.0":
            raise AssertionError(record)
        if not str(record.get("extension_path", "")).endswith((".so", ".py", ".pyd")):
            raise AssertionError(record)
    if min(kv_capacities) < MAX_MODEL_LEN:
        raise AssertionError(f"KV capacity below {MAX_MODEL_LEN}: {kv_capacities}")

    hbm_records = _extract_json_records(logs, _HBM_MARKER)
    if not hbm_records:
        return None
    if len(hbm_records) != 1:
        raise AssertionError(f"expected one leader HBM record, got {len(hbm_records)}")
    devices = sorted(hbm_records[0], key=lambda device: device["gpu"])
    if [device["gpu"] for device in devices] != list(range(GPU_COUNT)):
        raise AssertionError(devices)
    if any(not 0 < device["used_mib"] <= device["total_mib"] for device in devices):
        raise AssertionError(devices)

    return {
        "requested_topology": [list(values) for values in sorted(requested.elements())],
        "worker_placement_count": sum(placements.values()),
        "runtime_topology_count": sum(runtime.values()),
        "verified_wheels": verified_wheels,
        "kv_cache_tokens": kv_capacities,
        "leader_hbm": hbm_records[0],
    }


def _wait_for_runtime_evidence(
    session: RemoteInferenceSession,
    *,
    pipeline_parallel_size: int,
    model_config: Mapping[str, Any],
    expected_wheel: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    deadline = time.monotonic() + RUNTIME_EVIDENCE_TIMEOUT_SECONDS
    last_summary: dict[str, Any] = {}
    while time.monotonic() < deadline:
        session.check_alive()
        tasks = _task_records(session)
        endpoints = iris_ctx().client.list_endpoint_instances(session.endpoint_name)
        job_id = JobName.from_string(str(session.jobs[0].job_id))
        logs = _runtime_evidence_logs(iris_ctx().client.job(job_id))
        last_summary = {
            "tasks": [(task["task_id"], task["state"]) for task in tasks],
            "endpoint_count": len(endpoints),
            "log_lines": len(logs.splitlines()),
        }
        if (
            len(tasks) == pipeline_parallel_size
            and all(task["state"] == TaskState.RUNNING.value for task in tasks)
            and len(endpoints) == 1
        ):
            evidence = _validate_runtime_logs(
                logs,
                pipeline_parallel_size=pipeline_parallel_size,
                num_layers=int(model_config["num_hidden_layers"]),
                num_experts=int(model_config["num_experts"]),
                expected_wheel=expected_wheel,
            )
            if evidence is not None:
                return evidence, tasks, logs
        time.sleep(5)
    raise TimeoutError(f"timed out waiting for complete runtime evidence: {last_summary}")


def _completion_request(
    url: str,
    model_id: str,
    prompts: Sequence[int] | Sequence[Sequence[int]],
    *,
    max_tokens: int,
    request_id: str,
    rank: int | None = None,
    logprobs: int | None = None,
) -> list[dict[str, Any]]:
    headers = {"X-Request-Id": request_id}
    if rank is not None:
        headers["X-data-parallel-rank"] = str(rank)
    payload: dict[str, Any] = {
        "model": model_id,
        "prompt": prompts,
        "add_special_tokens": False,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "ignore_eos": True,
        "seed": 0,
        "return_tokens_as_token_ids": True,
        "return_token_ids": True,
    }
    if logprobs is not None:
        payload["logprobs"] = logprobs
    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=(HTTP_CONNECT_TIMEOUT_SECONDS, HTTP_READ_TIMEOUT_SECONDS),
    )
    try:
        response.raise_for_status()
        body = response.json()
        choices = sorted(body["choices"], key=lambda choice: choice["index"])
        records: list[dict[str, Any]] = []
        for choice in choices:
            record: dict[str, Any] = {
                "prompt_token_ids": [int(token) for token in choice["prompt_token_ids"]],
                "token_ids": [int(token) for token in choice["token_ids"]],
            }
            if logprobs is not None:
                record["top_logprobs"] = [
                    {int(token.removeprefix("token_id:")): float(value) for token, value in step.items()}
                    for step in choice["logprobs"]["top_logprobs"]
                ]
            records.append(record)
        return records
    except Exception as error:
        error.add_note(
            f"completion request failed: request_id={request_id} rank={rank} "
            f"status={response.status_code} body={response.text[:2000]}"
        )
        raise


def _real_oracle(model: ModelConfig) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    oracle_payload = _read_model_file(model, "jax-oracle.json")
    logprobs_payload = _read_model_file(model, "jax-oracle-logprobs.npz")
    manifest_payload = _read_model_file(model, "qualification-manifest.json")
    oracle = json.loads(oracle_payload)
    manifest = json.loads(manifest_payload)
    with np.load(io.BytesIO(logprobs_payload)) as archive:
        logprobs = np.asarray(archive["decode_logprobs"], dtype=np.float64)
    if tuple(int(value) for value in oracle["prompt_token_ids"]) != REAL_PROMPT_TOKEN_IDS:
        raise AssertionError(oracle["prompt_token_ids"])
    if tuple(int(value) for value in oracle["generated_token_ids"]) != REAL_GENERATED_TOKEN_IDS:
        raise AssertionError(oracle["generated_token_ids"])
    if manifest["payload_tree_sha256"] != REAL_ARTIFACT_TREE_SHA256:
        raise AssertionError(manifest)
    if manifest["raw_metadata_sha256"] != REAL_RAW_METADATA_SHA256:
        raise AssertionError(manifest)
    if logprobs.shape[0] != len(REAL_GENERATED_TOKEN_IDS):
        raise AssertionError(logprobs.shape)
    identity = {
        "jax_oracle_sha256": _sha256(oracle_payload),
        "jax_oracle_logprobs_sha256": _sha256(logprobs_payload),
        "qualification_manifest_sha256": _sha256(manifest_payload),
        "payload_tree_sha256": manifest["payload_tree_sha256"],
        "raw_checkpoint": manifest["raw_checkpoint"],
        "raw_metadata_sha256": manifest["raw_metadata_sha256"],
    }
    return oracle, logprobs, identity


def _canonical_observation(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        tuple(record["prompt_token_ids"]),
        tuple(record["token_ids"]),
        tuple(tuple(sorted(step.items())) for step in record["top_logprobs"]),
    )


def _qualify_real_requests(url: str, model_id: str, model: ModelConfig) -> dict[str, Any]:
    oracle, oracle_logprobs, artifact_identity = _real_oracle(model)
    selected_oracle = [float(value) for value in oracle["selected_token_logprobs"]]
    pinned: list[dict[str, Any]] = []
    baseline: tuple[Any, ...] | None = None
    for rank in range(GPU_COUNT):
        started = time.perf_counter()
        (record,) = _completion_request(
            url,
            model_id,
            REAL_PROMPT_TOKEN_IDS,
            max_tokens=len(REAL_GENERATED_TOKEN_IDS),
            request_id=f"oracle-rank-{rank}",
            rank=rank,
            logprobs=RETURNED_LOGPROBS,
        )
        seconds = time.perf_counter() - started
        if tuple(record["token_ids"]) != REAL_GENERATED_TOKEN_IDS:
            raise AssertionError({"rank": rank, "actual": record["token_ids"], "expected": REAL_GENERATED_TOKEN_IDS})
        selected_errors: list[float] = []
        top_errors: list[float] = []
        counts: list[int] = []
        for step, token_id in enumerate(REAL_GENERATED_TOKEN_IDS):
            actual = record["top_logprobs"][step]
            counts.append(len(actual))
            if len(actual) != RETURNED_LOGPROBS:
                raise AssertionError({"rank": rank, "step": step, "count": len(actual)})
            selected_errors.append(abs(actual[token_id] - selected_oracle[step]))
            top_errors.append(max(abs(value - float(oracle_logprobs[step, key])) for key, value in actual.items()))
        rank_receipt = {
            "rank": rank,
            "generated_token_ids": record["token_ids"],
            "selected_logprob_max_abs_error": max(selected_errors),
            "reported_top64_max_abs_error": max(top_errors),
            "reported_logprob_counts": counts,
            "seconds": seconds,
        }
        if rank_receipt["selected_logprob_max_abs_error"] > SELECTED_LOGPROB_ATOL:
            raise AssertionError(rank_receipt)
        if rank_receipt["reported_top64_max_abs_error"] > TOP_LOGPROB_ATOL:
            raise AssertionError(rank_receipt)
        pinned.append(rank_receipt)
        baseline = baseline or _canonical_observation(record)

    assert baseline is not None
    isolation: list[dict[str, Any]] = []

    def assert_matches(name: str, records: Sequence[Mapping[str, Any]], *, seconds: float) -> None:
        for index, record in enumerate(records):
            if _canonical_observation(record) != baseline:
                raise AssertionError(f"request isolation mismatch: {name}[{index}]")
        isolation.append({"name": name, "responses": len(records), "seconds": seconds, "bit_exact": True})

    started = time.perf_counter()
    singleton = _completion_request(
        url,
        model_id,
        REAL_PROMPT_TOKEN_IDS,
        max_tokens=len(REAL_GENERATED_TOKEN_IDS),
        request_id="isolation-singleton",
        logprobs=RETURNED_LOGPROBS,
    )
    assert_matches("singleton", singleton, seconds=time.perf_counter() - started)
    started = time.perf_counter()
    duplicate = _completion_request(
        url,
        model_id,
        (REAL_PROMPT_TOKEN_IDS, REAL_PROMPT_TOKEN_IDS),
        max_tokens=len(REAL_GENERATED_TOKEN_IDS),
        request_id="isolation-duplicate-batch",
        logprobs=RETURNED_LOGPROBS,
    )
    assert_matches("duplicate-batch", duplicate, seconds=time.perf_counter() - started)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                _completion_request,
                url,
                model_id,
                REAL_PROMPT_TOKEN_IDS,
                max_tokens=len(REAL_GENERATED_TOKEN_IDS),
                request_id=f"isolation-concurrent-{index}",
                logprobs=RETURNED_LOGPROBS,
            )
            for index in range(2)
        ]
        concurrent = [record for future in as_completed(futures) for record in future.result()]
    assert_matches("concurrent-pair", concurrent, seconds=time.perf_counter() - started)

    return {
        "artifact": artifact_identity,
        "prompt_token_ids": list(REAL_PROMPT_TOKEN_IDS),
        "expected_generated_token_ids": list(REAL_GENERATED_TOKEN_IDS),
        "selected_token_logprob_atol": SELECTED_LOGPROB_ATOL,
        "reported_top64_logprob_atol": TOP_LOGPROB_ATOL,
        "selected_token_logprob_max_abs_error": max(item["selected_logprob_max_abs_error"] for item in pinned),
        "reported_top64_max_abs_error": max(item["reported_top64_max_abs_error"] for item in pinned),
        "pinned_ranks": pinned,
        "request_isolation": isolation,
    }


def _wait_for_release(session: RemoteInferenceSession, expected_tasks: int) -> dict[str, Any]:
    job = session.jobs[0]
    job_id = JobName.from_string(str(job.job_id))
    deadline = time.monotonic() + RELEASE_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        status = job.status()
        endpoints = iris_ctx().client.list_endpoint_instances(session.endpoint_name)
        tasks = iris_ctx().client.list_tasks(job_id)
        if JobStatus.finished(status) and not endpoints:
            if len(tasks) != expected_tasks:
                raise AssertionError(f"expected {expected_tasks} released tasks, got {len(tasks)}")
            if any(task.state is TaskState.RUNNING for task in tasks):
                raise AssertionError(tasks)
            return {
                "job_status": status.value,
                "endpoint_instances_after_context": 0,
                "tasks": [
                    {
                        "task_id": str(task.task_id),
                        "state": task.state.value,
                        "worker_id": task.worker_id,
                        "current_attempt_number": task.current_attempt_number,
                        "attempt_count": len(task.attempts),
                    }
                    for task in tasks
                ],
                "released": True,
            }
        time.sleep(2)
    raise TimeoutError(
        f"inference gang did not release: status={job.status().value}, "
        f"endpoints={len(iris_ctx().client.list_endpoint_instances(session.endpoint_name))}"
    )


def run_grugmoe_schema2_qualification(
    model: ModelConfig,
    marin_sha: str,
    qualification_command: str,
    ordinary_catalog_command: str,
) -> None:
    """Qualify the real d1536 whole-node topology and emit a post-release JSON receipt."""
    configure_coreweave_s3()
    if VLLM_GPU_RELEASE.source_commit != EXPECTED_VLLM_SOURCE_COMMIT:
        raise AssertionError(
            "Marin is not pinned to the exact vLLM PR head: "
            f"expected={EXPECTED_VLLM_SOURCE_COMMIT}, actual={VLLM_GPU_RELEASE.source_commit}"
        )
    wheel = vllm_gpu_wheel_for_architecture(VLLM_GPU_RELEASE, "x86_64")
    expected_wheel = _jsonable(vllm_gpu_wheel_provenance(VLLM_GPU_RELEASE, wheel))
    model_hf_config, model_config_sha256 = _load_model_config(model)
    _assert_exact_model_shape(model_hf_config)
    pipeline_parallel_size = 1
    if model.serve.pipeline_parallel_size != pipeline_parallel_size:
        raise AssertionError((model.serve.pipeline_parallel_size, pipeline_parallel_size))

    accelerator = AcceleratorChoice(
        platform=Platform.GPU,
        gpu_type="H100",
        gpu_count=GPU_COUNT,
        target_cluster="cw-us-east-02a",
    )
    inference = inference_config_for_model(
        model,
        accelerator,
        env_vars={},
        priority=job_pb2.PRIORITY_BAND_INTERACTIVE,
    )
    receipt: dict[str, Any] = {
        "case": "real_d1536",
        "source": {
            "marin_sha": marin_sha,
            "vllm_sha": EXPECTED_VLLM_SOURCE_COMMIT,
        },
        "commands": {
            "qualification": qualification_command,
            "ordinary_catalog": ordinary_catalog_command,
        },
        "model": {
            "catalog": _jsonable(model),
            "hf_config_sha256": model_config_sha256,
            "hf_shape": {
                name: model_hf_config[name]
                for name in (
                    "grugmoe_artifact_schema_version",
                    "hidden_size",
                    "latent_dim",
                    "num_hidden_layers",
                    "num_experts",
                    "num_shared_experts",
                    "num_attention_heads",
                    "max_position_embeddings",
                )
            },
        },
        "wheel": expected_wheel,
        "lowered": {
            "model": _jsonable(inference.model),
            "engine": _jsonable(inference.engine),
            "worker_resources": _jsonable(inference.iris.worker_resources),
            "priority": inference.iris.priority,
            "instances": inference.instances,
            "broker": inference.broker,
        },
        "jobs": {"coordinator": str(iris_ctx().job_id)},
    }

    startup_started = time.perf_counter()
    session_for_release: RemoteInferenceSession | None = None
    try:
        with remote_inference(inference) as session:
            session_for_release = session
            session.wait_until_ready()
            receipt["startup_seconds"] = time.perf_counter() - startup_started
            receipt["jobs"]["inference_gang"] = str(session.jobs[0].job_id)
            runtime, ready_tasks, _logs = _wait_for_runtime_evidence(
                session,
                pipeline_parallel_size=pipeline_parallel_size,
                model_config=model_hf_config,
                expected_wheel=expected_wheel,
            )
            receipt["runtime"] = runtime
            receipt["ready"] = {
                "task_count": len(ready_tasks),
                "tasks": ready_tasks,
                "endpoint_name": session.endpoint_name,
                "endpoint_instance_count": len(iris_ctx().client.list_endpoint_instances(session.endpoint_name)),
                "single_leader_endpoint": True,
            }
            completions_url = session.model.endpoint.url("completions")
            model_id = session.model.endpoint.model
            receipt["correctness"] = _qualify_real_requests(completions_url, model_id, model)
    finally:
        if session_for_release is not None:
            receipt["release"] = _wait_for_release(session_for_release, pipeline_parallel_size)

    if not receipt.get("release", {}).get("released"):
        raise AssertionError("refusing to emit a receipt before release proof")
    print("GRUGMOE_SCHEMA2_SERVING_QUALIFICATION=" + json.dumps(receipt, sort_keys=True), flush=True)
