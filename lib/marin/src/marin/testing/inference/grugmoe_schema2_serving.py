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
import statistics
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
DUMMY_MODEL_CONFIG_SHA256 = "4cf93824957296e3b36c458ab4ae598bb7616f202a74687a3a8ed64dac9a9ebf"
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
BENCHMARK_TOKENS = 32
BENCHMARK_BATCH1_REPEATS = 5
BENCHMARK_BATCH8_REPEATS = 3

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


def _assert_exact_model_shape(case: str, config: Mapping[str, Any]) -> None:
    expected = {
        "real_d1536": {
            "grugmoe_artifact_schema_version": 2,
            "hidden_size": 1536,
            "latent_dim": 768,
            "num_hidden_layers": 16,
            "num_experts": 384,
            "num_shared_experts": 2,
            "num_attention_heads": 12,
            "max_position_embeddings": MAX_MODEL_LEN,
        },
        "dummy_d6144": {
            "grugmoe_artifact_schema_version": 2,
            "hidden_size": 6144,
            "latent_dim": 3072,
            "num_hidden_layers": 48,
            "num_experts": 384,
            "num_shared_experts": 2,
            "num_attention_heads": 48,
            "max_position_embeddings": MAX_MODEL_LEN,
        },
    }[case]
    actual = {name: config.get(name) for name in expected}
    if actual != expected:
        raise AssertionError(f"unexpected {case} model config: expected={expected}, actual={actual}")


def _d6144_pp2_memory_bound(config: Mapping[str, Any]) -> dict[str, Any]:
    """Compute the BF16-only PP2 floor; the FP32 routers make reality worse."""
    hidden = int(config["hidden_size"])
    latent = int(config["latent_dim"])
    intermediate = int(config["moe_intermediate_size"])
    shared_intermediate = int(config["shared_expert_intermediate_size"])
    num_experts = int(config["num_experts"])
    num_shared_experts = int(config["num_shared_experts"])
    num_heads = int(config["num_attention_heads"])
    num_kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])
    num_layers = int(config["num_hidden_layers"])
    vocab = int(config["vocab_size"])
    kernel = int(config["sconv_kernel"])
    ep_size = GPU_COUNT
    pp_size = 2
    layers_per_stage = num_layers // pp_size
    local_experts = num_experts // ep_size
    gated_norm_rank = 128

    bf16_params_per_layer = {
        "routed_experts": local_experts * 3 * latent * intermediate,
        "latent_projections": 2 * hidden * latent,
        "attention": (
            hidden * (num_heads * head_dim)
            + 2 * hidden * (num_kv_heads * head_dim)
            + (num_heads * head_dim) * hidden
            + hidden * num_heads
        ),
        "gated_norms": 2 * (2 * hidden * gated_norm_rank),
        "shared_experts": num_shared_experts * 3 * hidden * shared_intermediate,
        "rms_norms": 2 * hidden,
        "short_conv": kernel * (num_kv_heads * head_dim + hidden + hidden),
    }
    endpoint_params = vocab * hidden + hidden + 2 * hidden * gated_norm_rank
    stage_bf16_params = layers_per_stage * sum(bf16_params_per_layer.values()) + endpoint_params
    stage_bf16_gib = stage_bf16_params * 2 / 1024**3

    # H100 SXM reports 81,559 MiB; vLLM's 0.9 setting can reserve only 90%.
    reported_h100_mib = 81_559
    utilization_budget_gib = reported_h100_mib / 1024 * 0.9
    if not 73.3 <= stage_bf16_gib < 73.4:
        raise AssertionError(f"unexpected d6144 PP2 BF16 floor: {stage_bf16_gib}")
    if stage_bf16_gib <= utilization_budget_gib:
        raise AssertionError((stage_bf16_gib, utilization_budget_gib))
    return {
        "live_pp2_run": False,
        "reason": "BF16 weights alone exceed the 0.9-utilization H100 budget before FP32 routers, KV, or runtime",
        "pipeline_parallel_size": pp_size,
        "expert_parallel_size": ep_size,
        "layers_per_stage": layers_per_stage,
        "bf16_parameter_components_per_layer": bf16_params_per_layer,
        "endpoint_bf16_parameters_per_stage": endpoint_params,
        "bf16_weight_floor_gib_per_gpu": stage_bf16_gib,
        "h100_reported_total_mib": reported_h100_mib,
        "gpu_memory_utilization": 0.9,
        "utilization_budget_gib": utilization_budget_gib,
    }


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


def _runtime_evidence_logs(job: _RuntimeLogJob, *, case: str) -> str:
    """Fetch only the bounded startup markers the receipt validates.

    Iris/Finelog treats ``max_lines=0`` as a 1,000-line server-default tail. A
    multi-rank vLLM startup can exceed that before the endpoint is ready, so a
    broad tail can lose the earliest topology lines. Server-side substring
    filters keep every required marker query small and deterministic.
    """
    if case == "real_d1536":
        case_marker = _HBM_MARKER
    elif case == "dummy_d6144":
        case_marker = "load_format=dummy"
    else:
        raise ValueError(case)

    entries: dict[tuple[Any, ...], TaskLogEntry] = {}
    for marker in (*_RUNTIME_EVIDENCE_MARKERS, case_marker):
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
    case: str,
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
    if case == "real_d1536":
        if not hbm_records:
            return None
        if len(hbm_records) != 1:
            raise AssertionError(f"expected one leader HBM record, got {len(hbm_records)}")
        devices = sorted(hbm_records[0], key=lambda device: device["gpu"])
        if [device["gpu"] for device in devices] != list(range(GPU_COUNT)):
            raise AssertionError(devices)
        if any(not 0 < device["used_mib"] <= device["total_mib"] for device in devices):
            raise AssertionError(devices)
    else:
        hbm_records = []
        if "load_format=dummy" not in logs:
            return None

    return {
        "requested_topology": [list(values) for values in sorted(requested.elements())],
        "worker_placement_count": sum(placements.values()),
        "runtime_topology_count": sum(runtime.values()),
        "verified_wheels": verified_wheels,
        "kv_cache_tokens": kv_capacities,
        "leader_hbm": hbm_records[0] if hbm_records else None,
        "dummy_load_format_observed": case == "dummy_d6144",
    }


def _wait_for_runtime_evidence(
    session: RemoteInferenceSession,
    *,
    case: str,
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
        logs = _runtime_evidence_logs(iris_ctx().client.job(job_id), case=case)
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
                case=case,
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
        (record,) = _completion_request(
            url,
            model_id,
            REAL_PROMPT_TOKEN_IDS,
            max_tokens=len(REAL_GENERATED_TOKEN_IDS),
            request_id=f"oracle-rank-{rank}",
            rank=rank,
            logprobs=RETURNED_LOGPROBS,
        )
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
        }
        if rank_receipt["selected_logprob_max_abs_error"] > SELECTED_LOGPROB_ATOL:
            raise AssertionError(rank_receipt)
        if rank_receipt["reported_top64_max_abs_error"] > TOP_LOGPROB_ATOL:
            raise AssertionError(rank_receipt)
        pinned.append(rank_receipt)
        baseline = baseline or _canonical_observation(record)

    assert baseline is not None
    isolation: list[dict[str, Any]] = []

    def assert_matches(name: str, records: Sequence[Mapping[str, Any]]) -> None:
        for index, record in enumerate(records):
            if _canonical_observation(record) != baseline:
                raise AssertionError(f"request isolation mismatch: {name}[{index}]")
        isolation.append({"name": name, "responses": len(records), "bit_exact": True})

    singleton = _completion_request(
        url,
        model_id,
        REAL_PROMPT_TOKEN_IDS,
        max_tokens=len(REAL_GENERATED_TOKEN_IDS),
        request_id="isolation-singleton",
        logprobs=RETURNED_LOGPROBS,
    )
    assert_matches("singleton", singleton)
    for repeat in range(2):
        repeated = _completion_request(
            url,
            model_id,
            REAL_PROMPT_TOKEN_IDS,
            max_tokens=len(REAL_GENERATED_TOKEN_IDS),
            request_id=f"isolation-repeat-{repeat}",
            logprobs=RETURNED_LOGPROBS,
        )
        assert_matches(f"repeated-{repeat}", repeated)
    duplicate = _completion_request(
        url,
        model_id,
        (REAL_PROMPT_TOKEN_IDS, REAL_PROMPT_TOKEN_IDS),
        max_tokens=len(REAL_GENERATED_TOKEN_IDS),
        request_id="isolation-duplicate-batch",
        logprobs=RETURNED_LOGPROBS,
    )
    assert_matches("duplicate-batch", duplicate)
    with ThreadPoolExecutor(max_workers=GPU_COUNT) as executor:
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
            for index in range(GPU_COUNT)
        ]
        concurrent = [record for future in as_completed(futures) for record in future.result()]
    assert_matches("concurrent-unpinned", concurrent)

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


def _timed_request(
    url: str,
    model_id: str,
    *,
    rank: int,
    max_tokens: int,
    request_id: str,
) -> tuple[float, dict[str, Any]]:
    started = time.perf_counter()
    (record,) = _completion_request(
        url,
        model_id,
        REAL_PROMPT_TOKEN_IDS,
        max_tokens=max_tokens,
        request_id=request_id,
        rank=rank,
    )
    return time.perf_counter() - started, record


def _summarize_benchmark(calls: Sequence[Mapping[str, Any]], *, active_gpus: int) -> dict[str, Any]:
    latencies = [float(call["seconds"]) for call in calls]
    total_seconds = sum(latencies)
    output_tokens = sum(int(call["output_tokens"]) for call in calls)
    prompt_tokens = sum(int(call["prompt_tokens"]) for call in calls)
    output_rate = output_tokens / total_seconds
    return {
        "calls": list(calls),
        "latency_seconds_median": statistics.median(latencies),
        "latency_seconds_p95": float(np.percentile(np.asarray(latencies), 95)),
        "window_seconds": total_seconds,
        "window_prompt_tokens": prompt_tokens,
        "window_output_tokens": output_tokens,
        "window_output_tokens_per_second": output_rate,
        "window_total_tokens_per_second": (prompt_tokens + output_tokens) / total_seconds,
        "window_output_tokens_per_second_per_active_gpu": output_rate / active_gpus,
        "window_output_tokens_per_second_per_allocated_gpu": output_rate / GPU_COUNT,
    }


def _benchmark_real(url: str, model_id: str) -> dict[str, Any]:
    warmups: list[dict[str, Any]] = []
    for repeat in range(2):
        seconds, record = _timed_request(
            url,
            model_id,
            rank=0,
            max_tokens=16,
            request_id=f"benchmark-warmup-{repeat}",
        )
        if len(record["token_ids"]) != 16:
            raise AssertionError(record)
        warmups.append({"repeat": repeat, "seconds": seconds})

    batch1_calls: list[dict[str, Any]] = []
    for repeat in range(BENCHMARK_BATCH1_REPEATS):
        seconds, record = _timed_request(
            url,
            model_id,
            rank=0,
            max_tokens=BENCHMARK_TOKENS,
            request_id=f"benchmark-b1-{repeat}",
        )
        if len(record["token_ids"]) != BENCHMARK_TOKENS:
            raise AssertionError(record)
        batch1_calls.append(
            {
                "repeat": repeat,
                "seconds": seconds,
                "prompt_tokens": len(record["prompt_token_ids"]),
                "output_tokens": len(record["token_ids"]),
            }
        )

    batch8_calls: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=GPU_COUNT) as executor:
        for repeat in range(BENCHMARK_BATCH8_REPEATS):
            started = time.perf_counter()
            futures = [
                executor.submit(
                    _completion_request,
                    url,
                    model_id,
                    REAL_PROMPT_TOKEN_IDS,
                    max_tokens=BENCHMARK_TOKENS,
                    request_id=f"benchmark-b8-{repeat}-rank-{rank}",
                    rank=rank,
                )
                for rank in range(GPU_COUNT)
            ]
            records = [record for future in as_completed(futures) for record in future.result()]
            seconds = time.perf_counter() - started
            if len(records) != GPU_COUNT or any(len(record["token_ids"]) != BENCHMARK_TOKENS for record in records):
                raise AssertionError(records)
            batch8_calls.append(
                {
                    "repeat": repeat,
                    "seconds": seconds,
                    "prompt_tokens": sum(len(record["prompt_token_ids"]) for record in records),
                    "output_tokens": sum(len(record["token_ids"]) for record in records),
                }
            )
    return {
        "workload": {
            "warmups": "2 x batch-1 x 16 output tokens pinned to DP rank 0",
            "batch_1": f"{BENCHMARK_BATCH1_REPEATS} x 32 output tokens pinned to DP rank 0",
            "batch_8": f"{BENCHMARK_BATCH8_REPEATS} waves x 8 rank-pinned requests x 32 output tokens",
        },
        "warmups": warmups,
        "batch_1": _summarize_benchmark(batch1_calls, active_gpus=1),
        "batch_8": _summarize_benchmark(batch8_calls, active_gpus=GPU_COUNT),
    }


def _qualify_dummy_request(url: str, model_id: str) -> dict[str, Any]:
    started = time.perf_counter()
    (record,) = _completion_request(
        url,
        model_id,
        (128000,),
        max_tokens=1,
        request_id="dummy-d6144-short-request",
        rank=0,
    )
    seconds = time.perf_counter() - started
    if record["prompt_token_ids"] != [128000] or len(record["token_ids"]) != 1:
        raise AssertionError(record)
    return {"seconds": seconds, **record}


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
    case: str,
    model: ModelConfig,
    marin_sha: str,
    qualification_command: str,
    ordinary_catalog_command: str,
) -> None:
    """Qualify one real or dummy topology and emit a post-release JSON receipt."""
    if case not in {"real_d1536", "dummy_d6144"}:
        raise ValueError(case)
    configure_coreweave_s3()
    if VLLM_GPU_RELEASE.source_commit != EXPECTED_VLLM_SOURCE_COMMIT:
        raise AssertionError(
            "Marin is not pinned to the exact vLLM PR head: "
            f"expected={EXPECTED_VLLM_SOURCE_COMMIT}, actual={VLLM_GPU_RELEASE.source_commit}"
        )
    wheel = vllm_gpu_wheel_for_architecture(VLLM_GPU_RELEASE, "x86_64")
    expected_wheel = _jsonable(vllm_gpu_wheel_provenance(VLLM_GPU_RELEASE, wheel))
    model_hf_config, model_config_sha256 = _load_model_config(model)
    _assert_exact_model_shape(case, model_hf_config)
    if case == "dummy_d6144" and model_config_sha256 != DUMMY_MODEL_CONFIG_SHA256:
        raise AssertionError(
            "d6144 dummy config does not match the checked-in qualification shape: "
            f"expected={DUMMY_MODEL_CONFIG_SHA256}, actual={model_config_sha256}"
        )
    pipeline_parallel_size = 1 if case == "real_d1536" else 3
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
        "case": case,
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
    if case == "dummy_d6144":
        receipt["pp2_capacity_bound"] = _d6144_pp2_memory_bound(model_hf_config)
        receipt["dummy_weight_exception"] = {
            "load_format": "dummy",
            "omitted_real_checkpoint_flag": '--model-loader-extra-config {"distributed":true}',
            "reason": "vLLM's dummy loader rejects model-loader extra config; all model and serving settings match",
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
                case=case,
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
            if case == "real_d1536":
                receipt["correctness"] = _qualify_real_requests(completions_url, model_id, model)
                receipt["performance"] = _benchmark_real(completions_url, model_id)
            else:
                receipt["short_request"] = _qualify_dummy_request(completions_url, model_id)
    finally:
        if session_for_release is not None:
            receipt["release"] = _wait_for_release(session_for_release, pipeline_parallel_size)

    if not receipt.get("release", {}).get("released"):
        raise AssertionError("refusing to emit a receipt before release proof")
    print("GRUGMOE_SCHEMA2_SERVING_QUALIFICATION=" + json.dumps(receipt, sort_keys=True), flush=True)
