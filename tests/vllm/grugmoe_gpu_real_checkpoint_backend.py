# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GPU/CoreWeave backend subprocess entrypoints for GrugMoE real-checkpoint e2e.

This mirrors ``grugmoe_real_checkpoint_backend`` while keeping the CoreWeave
S3, CUDA, and JAX GPU guards separate from the TPU/GCS path.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import posixpath
import shutil
import site
import sys
import tempfile
import time
from contextlib import ExitStack
from functools import cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import fsspec
import requests
from rigging.filesystem import StoragePath, marin_temp_bucket, open_url, prefix_join, url_to_fs

from tests.vllm.grugmoe_real_checkpoint_backend import (
    DECODE_SEQ_LEN,
    LEVANTER_PROMPT_ADD_SPECIAL_TOKENS,
    MAX_MODEL_LEN,
    MAX_SHARD_SIZE,
    SERVER_TIMEOUT_SECONDS,
    StagedArtifact,
    _executable_model_from_legacy_split,
    _greedy_decode,
    _legacy_split_expert_inference_state_dict,
    _load_legacy_split_expert_checkpoint,
    _local_filesystem_path,
    _mesh_batch_axis_size,
    _real_checkpoint_model_config,
    _tokenizer_encode,
)
from tests.vllm.grugmoe_real_checkpoint_backend import (
    _stage_artifact_for_vllm as _common_stage_artifact_for_vllm,
)

COREWEAVE_S3_PREFIX = "s3://marin-us-east-02a/"
COREWEAVE_SIGNING_REGION = "US-EAST-02A"
COREWEAVE_ENDPOINT = "http://cwlota.com"
EXPECTED_GPU_COUNT = 8
VISIBLE_CUDA_DEVICES = ",".join(str(index) for index in range(EXPECTED_GPU_COUNT))
VLLM_TENSOR_PARALLEL_SIZE = 1
VLLM_DATA_PARALLEL_SIZE = 8
VLLM_EXPERT_PARALLEL_SIZE = 8
VLLM_MAX_NUM_SEQS = 16
VLLM_GPU_MEMORY_UTILIZATION = "0.15"
VLLM_DTYPE = "bfloat16"
LEVANTER_REFERENCE_MODE = "bf16_compute"
LEVANTER_BF16_POLICY = "params=float32,compute=bfloat16,output=bfloat16"
JAX_GPU_MEMORY_FRACTION = "0.95"
JAX_GPU_ALLOCATOR = "cuda_malloc_async"
LEVANTER_EXPERT_AXIS_SIZE = EXPECTED_GPU_COUNT
LEVANTER_REFERENCE_REPEAT_COUNT = 2
RUN_ID_ENV = "MARIN_GRUGMOE_GPU_E2E_RUN_ID"
OUTPUT_DIR_ENV = "MARIN_GRUGMOE_GPU_E2E_OUTPUT_DIR"
VLLM_ATTENTION_BACKENDS_UNDER_TEST = ("TRITON_ATTN", "FLASH_ATTN")
LEVANTER_MOE_CAPACITY_FACTOR = float(EXPECTED_GPU_COUNT)
LEVANTER_DECODE_USE_ACTIVE_PREFIX = True
CHECKPOINT_SCOPE = "small-real-checkpoint"
CHECKPOINT_PATH = "s3://marin-us-east-02a/marin/grug/moe_may_compute_opt_d512_ep1-05c39b/checkpoints/step-10980"
TOKENIZER_PATH = "s3://marin-us-east-02a/marin/tokenizers/marin-community/marin-tokenizer/hf-hub-0.36.2"
OUTPUT_ROOT = marin_temp_bucket(
    ttl_days=14,
    prefix="grugmoe-gpu-real-checkpoint-e2e",
    source_prefix=COREWEAVE_S3_PREFIX,
)
CACHE_ROOT = "s3://marin-us-east-02a/compilation-cache/grugmoe-gpu-real-checkpoint-e2e"
LOCAL_ARTIFACT_ROOT = os.path.join(tempfile.gettempdir(), "grugmoe-gpu-real-checkpoint-e2e-artifacts")
VLLM_LOCAL_CACHE_ROOT = os.path.join(tempfile.gettempdir(), "grugmoe-gpu-vllm-cache")
PROMPT = "United States of"
PROMPT_BATCH_SIZE = 16
PROMPTS = tuple(PROMPT for _ in range(PROMPT_BATCH_SIZE))
PROMPTS_PER_VLLM_DATA_PARALLEL_RANK = PROMPT_BATCH_SIZE // VLLM_DATA_PARALLEL_SIZE
EXPECTED_CONTINUATION = " America"
MAX_NUM_BATCHED_TOKENS = 1024
MAX_NEW_TOKENS = 1
SERVED_MODEL_NAME = "grugmoe-gpu-real-checkpoint-e2e"


@cache
def _configure_coreweave_s3_env() -> dict[str, Any]:
    """Configure process-local S3 env for CoreWeave AI Object Storage.

    Iris tasks often carry R2 credentials in AWS_* for ``s3://marin-na``. This
    e2e reads ``s3://marin-us-east-02a``, so it always maps CW_* credentials to
    AWS_* and uses the CoreWeave in-cluster endpoint.
    """

    try:
        os.environ["AWS_ACCESS_KEY_ID"] = os.environ["CW_ACCESS_KEY_ID"]
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ["CW_SECRET_ACCESS_KEY"]
    except KeyError as error:
        raise RuntimeError("CoreWeave S3 access requires CW_ACCESS_KEY_ID and CW_SECRET_ACCESS_KEY") from error
    os.environ["AWS_ENDPOINT_URL"] = COREWEAVE_ENDPOINT
    os.environ["AWS_ENDPOINT_URL_S3"] = COREWEAVE_ENDPOINT

    os.environ["AWS_REGION"] = COREWEAVE_SIGNING_REGION
    os.environ["AWS_DEFAULT_REGION"] = COREWEAVE_SIGNING_REGION
    fsspec_conf = {
        "endpoint_url": COREWEAVE_ENDPOINT,
        "client_kwargs": {"region_name": COREWEAVE_SIGNING_REGION},
        "config_kwargs": {"s3": {"addressing_style": "virtual"}},
    }
    os.environ["FSSPEC_S3"] = json.dumps(fsspec_conf, sort_keys=True)

    import s3fs  # noqa: PLC0415

    fsspec.config.set_conf_env(fsspec.config.conf)
    s3fs.S3FileSystem.clear_instance_cache()

    return {
        "endpoint": COREWEAVE_ENDPOINT,
        "signing_region": COREWEAVE_SIGNING_REGION,
        "credential_source": "CW_*",
    }


def _join_path(base: str, *parts: str) -> str:
    joined = base
    for part in parts:
        joined = prefix_join(joined, part)
    return joined


def _coreweave_url_to_fs(path: str):
    if path.startswith("s3://"):
        _configure_coreweave_s3_env()
    return url_to_fs(path)


def _exists(path: str) -> bool:
    fs, plain_path = _coreweave_url_to_fs(path)
    return fs.exists(plain_path)


def _remove_tree(path: str) -> None:
    fs, plain_path = _coreweave_url_to_fs(path)
    if fs.exists(plain_path):
        fs.rm(plain_path, recursive=True)


def _write_json(path: str, payload: dict[str, Any]) -> None:
    fs, plain_path = _coreweave_url_to_fs(path)
    plain_parent = posixpath.dirname(plain_path)
    if plain_parent:
        fs.makedirs(plain_parent, exist_ok=True)
    with fs.open(plain_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _copy_local_file(source_path: str, destination_path: str) -> None:
    _require_coreweave_path("destination_path", destination_path)
    fs, plain_destination_path = _coreweave_url_to_fs(destination_path)
    plain_parent = posixpath.dirname(plain_destination_path)
    if plain_parent:
        fs.makedirs(plain_parent, exist_ok=True)
    with open(source_path, "rb") as src, fs.open(plain_destination_path, "wb") as dst:
        shutil.copyfileobj(src, dst)


def _read_json(path: str) -> dict[str, Any]:
    _require_coreweave_path("json_path", path)
    if path.startswith("s3://"):
        _configure_coreweave_s3_env()

    with open_url(path, "r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object at {path}, got {type(payload).__name__}")
    return payload


def _is_coreweave_s3_path(path: str) -> bool:
    parsed_path = StoragePath.parse(path)
    parsed_prefix = StoragePath.parse(COREWEAVE_S3_PREFIX)
    if (parsed_path.scheme, parsed_path.netloc) != (parsed_prefix.scheme, parsed_prefix.netloc):
        return False
    try:
        parsed_path.relative_to(parsed_prefix)
    except ValueError:
        return False
    return True


def _require_coreweave_path(label: str, path: str) -> None:
    parsed = urlparse(path)
    if parsed.scheme == "s3":
        if _is_coreweave_s3_path(path):
            return
        raise ValueError(f"{label} must be under {COREWEAVE_S3_PREFIX}, got {path!r}")
    if parsed.scheme in {"", "file"}:
        return
    raise ValueError(f"{label} must be a local path or {COREWEAVE_S3_PREFIX} path, got {path!r}")


def _require_file(label: str, path: str) -> None:
    _require_coreweave_path(label, path)
    if not _exists(path):
        raise FileNotFoundError(f"{label} not found at {path}")


def _torch_cuda_snapshot() -> dict[str, Any]:
    import torch  # noqa: PLC0415

    device_count = int(torch.cuda.device_count())
    return {
        "available": bool(torch.cuda.is_available()),
        "device_count": device_count,
        "expected_gpu_count": EXPECTED_GPU_COUNT,
        "devices": [torch.cuda.get_device_name(index) for index in range(device_count)],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _vllm_extra_args(attention_backend: str) -> list[str]:
    return [
        "--runner",
        "generate",
        "--tensor-parallel-size",
        str(VLLM_TENSOR_PARALLEL_SIZE),
        "--data-parallel-size",
        str(VLLM_DATA_PARALLEL_SIZE),
        "--data-parallel-size-local",
        str(VLLM_DATA_PARALLEL_SIZE),
        "--data-parallel-start-rank",
        "0",
        "--data-parallel-backend",
        "mp",
        "--enable-expert-parallel",
        "--expert-placement-strategy",
        "linear",
        "--moe-backend",
        "triton",
        "--attention-backend",
        attention_backend,
        "--dtype",
        VLLM_DTYPE,
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--enforce-eager",
        "--max-num-seqs",
        str(VLLM_MAX_NUM_SEQS),
        "--gpu-memory-utilization",
        VLLM_GPU_MEMORY_UTILIZATION,
    ]


@dataclasses.dataclass(frozen=True)
class VllmCompletionBatch:
    single_payload: dict[str, Any]
    payloads: list[dict[str, Any]]
    rank_request_batches: list[dict[str, Any]]


@dataclasses.dataclass(frozen=True)
class VllmCompletionSummary:
    single_completion: str
    completion: str
    completions: list[str]
    single_prompt_choice_summary: dict[str, Any]
    main_choice_summaries: list[dict[str, Any]]


@dataclasses.dataclass(frozen=True)
class VllmBackendContext:
    args: argparse.Namespace
    attention_backend: str
    s3_env: dict[str, Any]
    vllm_env: dict[str, Any]
    torch_runtime: dict[str, Any]
    staged_artifact: StagedArtifact
    model: Any
    extra_args: list[str]
    started: float


def _collect_vllm_completion_batch(env: Any, *, attention_backend: str) -> VllmCompletionBatch:
    # Separately covers the batch-size-1 serving path before the per-rank
    # batch-size-2 requests below.
    single_payload = _post_completion_request(
        env,
        prompts=[PROMPT],
        data_parallel_rank=0,
        request_id=f"grugmoe-{attention_backend.lower()}-single-rank0",
    )
    payloads: list[dict[str, Any]] = []
    rank_request_batches: list[dict[str, Any]] = []
    for data_parallel_rank in range(VLLM_DATA_PARALLEL_SIZE):
        rank_start = data_parallel_rank * PROMPTS_PER_VLLM_DATA_PARALLEL_RANK
        rank_stop = rank_start + PROMPTS_PER_VLLM_DATA_PARALLEL_RANK
        prompts = list(PROMPTS[rank_start:rank_stop])
        payload = _post_completion_request(
            env,
            prompts=prompts,
            data_parallel_rank=data_parallel_rank,
            request_id=f"grugmoe-{attention_backend.lower()}-main-rank{data_parallel_rank}",
        )
        payloads.append(payload)
        rank_request_batches.append(
            {
                "data_parallel_rank": data_parallel_rank,
                "prompt_indices": list(range(rank_start, rank_stop)),
                "batch_size": len(prompts),
            }
        )
    return VllmCompletionBatch(
        single_payload=single_payload,
        payloads=payloads,
        rank_request_batches=rank_request_batches,
    )


def _jax_compilation_cache_dir(cache_dir: str) -> str:
    parsed = urlparse(cache_dir)
    if parsed.scheme == "s3":
        digest = hashlib.sha256(cache_dir.encode("utf-8")).hexdigest()[:16]
        local_cache_dir = os.path.join(tempfile.gettempdir(), "grugmoe-gpu-jax-cache", digest)
        os.makedirs(local_cache_dir, exist_ok=True)
        return local_cache_dir
    local_cache_dir = _local_filesystem_path(cache_dir)
    os.makedirs(local_cache_dir, exist_ok=True)
    return local_cache_dir


def _configure_jax_gpu_env(cache_dir: str, *, memory_fraction: str = JAX_GPU_MEMORY_FRACTION) -> dict[str, Any]:
    if os.environ.get("PJRT_DEVICE", "").upper() == "TPU":
        raise RuntimeError("GPU GrugMoE e2e cannot run with PJRT_DEVICE=TPU")
    os.environ["JAX_PLATFORMS"] = "cuda,cpu"
    actual_cache_dir = _jax_compilation_cache_dir(cache_dir)
    os.environ["JAX_COMPILATION_CACHE_DIR"] = actual_cache_dir
    os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "1"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = memory_fraction
    os.environ.pop("XLA_PYTHON_CLIENT_PREALLOCATE", None)
    os.environ["TF_GPU_ALLOCATOR"] = JAX_GPU_ALLOCATOR
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    return {
        "requested_cache_dir": cache_dir,
        "actual_cache_dir": os.environ["JAX_COMPILATION_CACHE_DIR"],
        "jax_platforms": os.environ.get("JAX_PLATFORMS"),
        "jax_enable_compilation_cache": os.environ.get("JAX_ENABLE_COMPILATION_CACHE"),
        "xla_python_client_mem_fraction": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
        "xla_python_client_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        "tf_gpu_allocator": os.environ.get("TF_GPU_ALLOCATOR"),
    }


def _require_jax_gpu_runtime() -> dict[str, Any]:
    import jax  # noqa: PLC0415

    devices = jax.devices()
    gpu_devices = [str(device) for device in devices if getattr(device, "platform", "") == "gpu"]
    if not gpu_devices:
        raise RuntimeError(f"Expected JAX GPU devices for GrugMoE e2e; got {devices!r}")
    if len(gpu_devices) < EXPECTED_GPU_COUNT:
        raise RuntimeError(
            f"Expected JAX to see at least {EXPECTED_GPU_COUNT} GPU devices for GrugMoE e2e; "
            f"got {gpu_devices!r} with CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}"
        )
    return {
        "default_backend": jax.default_backend(),
        "local_device_count": jax.local_device_count(),
        "gpu_device_count": len(gpu_devices),
        "expected_gpu_count": EXPECTED_GPU_COUNT,
        "gpu_devices": gpu_devices,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _mesh_snapshot(mesh: Any) -> dict[str, Any]:
    devices = getattr(mesh, "devices", ())
    if hasattr(devices, "flat"):
        flat_devices = [str(device) for device in devices.flat]
    else:
        flat_devices = [str(device) for device in devices]
    shape = getattr(mesh, "shape", {})
    return {
        "axis_names": [str(name) for name in getattr(mesh, "axis_names", ())],
        "shape": {str(name): int(size) for name, size in dict(shape).items()},
        "device_count": len(flat_devices),
        "expected_gpu_count": EXPECTED_GPU_COUNT,
        "devices": flat_devices,
    }


def _python_library_dirs() -> list[str]:
    roots: list[Path] = []
    for path in (*site.getsitepackages(), site.getusersitepackages(), *sys.path):
        if path:
            roots.append(Path(path))

    seen: set[str] = set()
    library_dirs: list[str] = []
    for root in roots:
        resolved_root = str(root)
        if resolved_root in seen:
            continue
        seen.add(resolved_root)

        nvidia_root = root / "nvidia"
        if nvidia_root.is_dir():
            for pattern in ("*/lib", "*/lib64"):
                for lib_dir in sorted(nvidia_root.glob(pattern)):
                    if lib_dir.is_dir() and any(lib_dir.glob("*.so*")):
                        library_dirs.append(str(lib_dir))

        torch_lib = root / "torch" / "lib"
        if torch_lib.is_dir() and any(torch_lib.glob("*.so*")):
            library_dirs.append(str(torch_lib))
    return list(dict.fromkeys(library_dirs))


def _prepend_env_path(name: str, values: list[str]) -> None:
    if not values:
        return
    existing = [value for value in os.environ.get(name, "").split(os.pathsep) if value]
    os.environ[name] = os.pathsep.join(list(dict.fromkeys([*values, *existing])))


def _configure_cuda_library_path() -> dict[str, Any]:
    library_dirs = _python_library_dirs()
    _prepend_env_path("LD_LIBRARY_PATH", library_dirs)
    ld_library_path_entry_count = len(
        [value for value in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if value]
    )
    return {
        "added_library_dirs": library_dirs,
        "ld_library_path_entry_count": ld_library_path_entry_count,
    }


def _configure_vllm_gpu_env() -> dict[str, Any]:
    os.environ["VLLM_TARGET_DEVICE"] = "cuda"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    local_cache_dirs = {
        "TRITON_CACHE_DIR": os.path.join(VLLM_LOCAL_CACHE_ROOT, "triton"),
        "TORCHINDUCTOR_CACHE_DIR": os.path.join(VLLM_LOCAL_CACHE_ROOT, "torchinductor"),
        "CUDA_CACHE_PATH": os.path.join(VLLM_LOCAL_CACHE_ROOT, "cuda"),
        "XDG_CACHE_HOME": os.path.join(VLLM_LOCAL_CACHE_ROOT, "xdg"),
        "VLLM_CACHE_ROOT": os.path.join(VLLM_LOCAL_CACHE_ROOT, "vllm"),
    }
    for name, value in local_cache_dirs.items():
        os.makedirs(value, exist_ok=True)
        if not os.environ.get(name):
            os.environ[name] = value
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "DEBUG")
    os.environ.setdefault("MODEL_IMPL_TYPE", "vllm")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    cuda_library_path = _configure_cuda_library_path()
    return {
        **cuda_library_path,
        "local_cache_dirs": {name: os.environ.get(name) for name in local_cache_dirs},
        "vllm_logging_level": os.environ.get("VLLM_LOGGING_LEVEL"),
    }


def _require_torch_cuda_runtime() -> dict[str, Any]:
    snapshot = _torch_cuda_snapshot()
    if not snapshot.get("available"):
        raise RuntimeError(f"Expected torch CUDA for GrugMoE vLLM e2e; got {snapshot!r}")
    if int(snapshot.get("device_count") or 0) < EXPECTED_GPU_COUNT:
        raise RuntimeError(f"Expected at least {EXPECTED_GPU_COUNT} CUDA devices for GrugMoE vLLM e2e; got {snapshot!r}")
    h100_devices = [device for device in snapshot["devices"] if "H100" in device]
    if len(h100_devices) < EXPECTED_GPU_COUNT:
        raise RuntimeError(f"Expected {EXPECTED_GPU_COUNT} H100 CUDA devices for GrugMoE vLLM e2e; got {snapshot!r}")
    return snapshot


def _export_backend(args: argparse.Namespace) -> None:
    phase_timings: dict[str, float] = {}
    s3_env = _configure_coreweave_s3_env()
    jax_env = _configure_jax_gpu_env(args.cache_dir)
    if _exists(args.artifact_dir):
        _remove_tree(args.artifact_dir)

    import equinox as eqx  # noqa: PLC0415
    import haliax  # noqa: PLC0415
    from haliax import Axis  # noqa: PLC0415
    from haliax.partitioning import set_mesh  # noqa: PLC0415
    from levanter.compat.hf_checkpoints import load_tokenizer  # noqa: PLC0415
    from levanter.grug.sharding import compact_grug_mesh  # noqa: PLC0415

    class LegacySplitExpertExportModel(eqx.Module):
        model: Any
        config: Any = eqx.field(static=True)

        @property
        def Vocab(self) -> Axis:
            return Axis("vocab", self.config.vocab_size)

        def to_state_dict(self, prefix: str | None = None) -> dict[str, Any]:
            return _legacy_split_expert_inference_state_dict(self.model, self.config, prefix=prefix)

    jax_runtime = _require_jax_gpu_runtime()
    model_cfg = _real_checkpoint_model_config()
    mesh = compact_grug_mesh(expert_axis_size=EXPECTED_GPU_COUNT, model_axis_size=1)
    mesh_runtime = _mesh_snapshot(mesh)
    started = time.time()
    with ExitStack() as stack:
        stack.enter_context(set_mesh(mesh))
        stack.enter_context(haliax.axis_mapping({}))
        restore_started = time.time()
        loaded_model = _load_legacy_split_expert_checkpoint(args.checkpoint_path, model_cfg)
        phase_timings["checkpoint_restore_seconds"] = time.time() - restore_started
        tokenizer = load_tokenizer(args.tokenizer_path)
        converter = model_cfg.hf_checkpoint_converter().replaced(tokenizer=tokenizer)
        serialization_started = time.time()
        converter.save_pretrained(
            LegacySplitExpertExportModel(loaded_model, model_cfg),
            args.artifact_dir,
            save_tokenizer=True,
            max_shard_size=MAX_SHARD_SIZE,
        )
        phase_timings["artifact_serialization_seconds"] = time.time() - serialization_started
        reference_started = time.time()
        levanter_result = _build_levanter_reference_result(
            args,
            loaded_model=loaded_model,
            model_cfg=model_cfg,
            tokenizer=tokenizer,
            jax_runtime=jax_runtime,
            mesh=mesh,
            mesh_runtime=mesh_runtime,
        )
        phase_timings["reference_generation_seconds"] = time.time() - reference_started
    _require_file("exported config.json", _join_path(args.artifact_dir, "config.json"))
    _require_file("exported tokenizer.json", _join_path(args.artifact_dir, "tokenizer.json"))
    levanter_timings = {
        "reference_generation_seconds": phase_timings["reference_generation_seconds"],
        "total_seconds": levanter_result["elapsed_seconds"],
    }
    levanter_result["phase_timings"] = levanter_timings
    _write_json(args.levanter_result_path, levanter_result)
    print("grugmoe_gpu_real_checkpoint_levanter_result=" + json.dumps(levanter_result, sort_keys=True), flush=True)
    if levanter_result["passed"] is not True:
        raise AssertionError(
            f"GPU Levanter/JAX reference did not produce stable expected continuation "
            f"{EXPECTED_CONTINUATION!r}: completion={levanter_result['completion']!r}, "
            f"reference_checks={levanter_result['reference_checks']!r}"
        )
    result = {
        "phase": "export",
        "checkpoint_path": args.checkpoint_path,
        "checkpoint_scope": CHECKPOINT_SCOPE,
        "tokenizer_path": args.tokenizer_path,
        "artifact_dir": args.artifact_dir,
        "levanter_result_path": args.levanter_result_path,
        "result_path": args.result_path,
        "coreweave_s3": s3_env,
        "jax_env": jax_env,
        "jax_runtime": jax_runtime,
        "jax_mesh": mesh_runtime,
        "elapsed_seconds": time.time() - started,
    }
    phase_timings["total_seconds"] = result["elapsed_seconds"]
    result["phase_timings"] = phase_timings
    _write_json(args.result_path, result)
    print("grugmoe_gpu_real_checkpoint_export_result=" + json.dumps(result, sort_keys=True), flush=True)


def _stage_artifact_for_vllm(artifact_dir: str) -> StagedArtifact:
    return _common_stage_artifact_for_vllm(
        artifact_dir,
        path_validator=_require_coreweave_path,
        temp_prefix="grugmoe-gpu-real-checkpoint-vllm-artifact-",
    )


def _post_completion_request(
    env: Any,
    *,
    prompts: list[str],
    data_parallel_rank: int | None = None,
    request_id: str | None = None,
    max_tokens: int = MAX_NEW_TOKENS,
) -> dict[str, Any]:
    if env.model_id is None:
        raise RuntimeError("Expected vLLM server to expose a model id.")
    headers = {}
    if data_parallel_rank is not None:
        headers["X-data-parallel-rank"] = str(data_parallel_rank)
    if request_id is not None:
        headers["X-Request-Id"] = request_id
    payload: dict[str, Any] = {
        "model": env.model_id,
        "prompt": prompts,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "add_special_tokens": LEVANTER_PROMPT_ADD_SPECIAL_TOKENS,
        "return_token_ids": True,
    }
    response = requests.post(
        f"{env.server_url}/completions",
        headers=headers,
        json=payload,
        timeout=300,
    )
    print("vllm_gpu_completions_status_code=" + str(response.status_code), flush=True)
    if not response.ok:
        print("vllm_gpu_completions_response_text=" + response.text[:4000], flush=True)
        print("vllm_gpu_server_logs_tail_begin", flush=True)
        print(env.logs_tail(max_lines=400), flush=True)
        print("vllm_gpu_server_logs_tail_end", flush=True)
        response.raise_for_status()
    return response.json()


def _completion_choice_summary(choice: dict[str, Any]) -> dict[str, Any]:
    return {
        "text": str(choice.get("text", "")),
        "finish_reason": choice.get("finish_reason"),
        "token_ids": choice.get("token_ids"),
        "prompt_token_ids": choice.get("prompt_token_ids"),
    }


def _completion_choice_summaries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    choices = payload.get("choices")
    if not isinstance(choices, list):
        raise AssertionError(f"completion payload missing choices list: {payload!r}")
    return [_completion_choice_summary(choice) for choice in choices if isinstance(choice, dict)]


def _summarize_vllm_completion_batch(batch: VllmCompletionBatch) -> VllmCompletionSummary:
    single_choices = batch.single_payload.get("choices")
    if not isinstance(single_choices, list) or len(single_choices) != 1:
        raise AssertionError(f"expected exactly one single-prompt completion choice, got {batch.single_payload!r}")
    single_completion = str(single_choices[0].get("text", ""))

    completions: list[str] = []
    for payload in batch.payloads:
        choices = payload.get("choices")
        if not isinstance(choices, list) or len(choices) != 2:
            raise AssertionError(f"expected exactly two completion choices, got {payload!r}")
        for choice in choices:
            completions.append(str(choice.get("text", "")))
    if len(completions) != PROMPT_BATCH_SIZE:
        raise AssertionError(f"expected {PROMPT_BATCH_SIZE} completions, got {len(completions)}")

    return VllmCompletionSummary(
        single_completion=single_completion,
        completion=completions[0],
        completions=completions,
        single_prompt_choice_summary=_completion_choice_summaries(batch.single_payload)[0],
        main_choice_summaries=[
            choice_summary for payload in batch.payloads for choice_summary in _completion_choice_summaries(payload)
        ],
    )


def _copy_vllm_server_logs(
    log_dir: str | None,
    output_dir: str,
    *,
    attention_backend: str | None = None,
) -> dict[str, Any]:
    if not log_dir:
        return {"copied": False, "reason": "no log directory available"}
    log_root = Path(log_dir)
    if not log_root.is_dir():
        return {"copied": False, "reason": f"log directory does not exist: {log_dir}"}

    artifact_parts = ["vllm-server-logs"]
    if attention_backend is not None:
        artifact_parts.append(attention_backend.lower())
    artifact_dir = _join_path(output_dir, *artifact_parts)
    files: list[dict[str, Any]] = []
    for source_path in sorted(path for path in log_root.iterdir() if path.is_file()):
        destination_path = _join_path(artifact_dir, source_path.name)
        _copy_local_file(str(source_path), destination_path)
        files.append(
            {
                "name": source_path.name,
                "local_path": str(source_path),
                "artifact_path": destination_path,
                "bytes": source_path.stat().st_size,
            }
        )
    return {
        "copied": bool(files),
        "local_log_dir": log_dir,
        "artifact_dir": artifact_dir,
        "files": files,
    }


def _latest_vllm_server_log_dir(*, since: float) -> str | None:
    candidates = [
        path
        for path in Path(tempfile.gettempdir()).glob("vllm_server_*")
        if path.is_dir() and path.stat().st_mtime >= since - 1
    ]
    if not candidates:
        return None
    return str(max(candidates, key=lambda path: path.stat().st_mtime))


def _exception_summary(exc: BaseException) -> dict[str, Any]:
    message = str(exc)
    max_message_length = 8000
    return {
        "type": type(exc).__name__,
        "message": message[:max_message_length],
        "message_truncated": len(message) > max_message_length,
    }


def _prompt_token_ids(tokenizer: Any, prompt: str) -> tuple[list[int], dict[str, Any]]:
    token_ids = _tokenizer_encode(
        tokenizer,
        prompt,
        add_special_tokens=LEVANTER_PROMPT_ADD_SPECIAL_TOKENS,
    )
    return token_ids, {
        "add_special_tokens": LEVANTER_PROMPT_ADD_SPECIAL_TOKENS,
        "prompt_token_count": len(token_ids),
        "prompt_token_ids": token_ids,
    }


def _serving_completion_from_generated_ids(tokenizer: Any, generated_token_ids: list[int]) -> tuple[str, dict[str, Any]]:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    eos_token_id = int(eos_token_id) if eos_token_id is not None else None
    completion_token_ids: list[int] = []
    stop_token_id = None
    for token_id in generated_token_ids:
        token_id = int(token_id)
        if eos_token_id is not None and token_id == eos_token_id:
            stop_token_id = token_id
            break
        completion_token_ids.append(token_id)
    return tokenizer.decode(completion_token_ids, skip_special_tokens=True), {
        "completion_token_ids": completion_token_ids,
        "stop_token_id": stop_token_id,
        "stopped_on_eos": stop_token_id is not None,
        "skip_special_tokens": True,
    }


def _apply_levanter_reference_mode(model: Any) -> tuple[Any, dict[str, Any]]:
    import jmp  # noqa: PLC0415

    policy = jmp.get_policy(LEVANTER_BF16_POLICY)
    return policy.cast_to_compute(model), {
        "mode": LEVANTER_REFERENCE_MODE,
        "applies_mixed_precision_policy": True,
        "policy": LEVANTER_BF16_POLICY,
        "param_dtype": str(policy.param_dtype),
        "compute_dtype": str(policy.compute_dtype),
        "output_dtype": str(policy.output_dtype),
    }


def _vllm_failure_result(
    context: VllmBackendContext,
    exc: BaseException,
    *,
    log_artifacts: dict[str, Any],
    phase_timings: dict[str, float],
) -> dict[str, Any]:
    args = context.args
    return {
        "phase": "vllm",
        "checkpoint_path": args.checkpoint_path,
        "tokenizer_path": args.tokenizer_path,
        "artifact_dir": args.artifact_dir,
        "result_path": args.result_path,
        "prompt": PROMPT,
        "prompt_batch_size": PROMPT_BATCH_SIZE,
        "passed": False,
        "failure": _exception_summary(exc),
        "served_model_name": SERVED_MODEL_NAME,
        "vllm_model_path": context.staged_artifact.vllm_model_path,
        "artifact_staging": context.staged_artifact.staging,
        "vllm_engine_kwargs": context.model.engine_kwargs,
        "vllm_args": context.extra_args,
        "vllm_dtype": VLLM_DTYPE,
        "vllm_tensor_parallel_size": VLLM_TENSOR_PARALLEL_SIZE,
        "vllm_data_parallel_size": VLLM_DATA_PARALLEL_SIZE,
        "vllm_expert_parallel_size": VLLM_EXPERT_PARALLEL_SIZE,
        "vllm_attention_backend": context.attention_backend,
        "vllm_max_num_seqs": VLLM_MAX_NUM_SEQS,
        "vllm_gpu_memory_utilization": VLLM_GPU_MEMORY_UTILIZATION,
        "expected_gpu_count": EXPECTED_GPU_COUNT,
        "coreweave_s3": context.s3_env,
        "vllm_env": context.vllm_env,
        "torch_runtime": context.torch_runtime,
        "vllm_log_artifacts": log_artifacts,
        "phase_timings": phase_timings,
        "elapsed_seconds": time.time() - context.started,
    }


def _vllm_success_result(
    context: VllmBackendContext,
    completion_batch: VllmCompletionBatch,
    completion_summary: VllmCompletionSummary,
    *,
    model_id: str | None,
    log_artifacts: dict[str, Any],
    levanter_reference: dict[str, Any],
    phase_timings: dict[str, float],
) -> dict[str, Any]:
    args = context.args
    levanter_completion = levanter_reference.get("completion")
    levanter_match = completion_summary.completion == levanter_completion
    levanter_single_match = completion_summary.single_completion == levanter_completion
    levanter_all_match = all(completion == levanter_completion for completion in completion_summary.completions)
    expected_all_match = completion_summary.single_completion == EXPECTED_CONTINUATION and all(
        completion == EXPECTED_CONTINUATION for completion in completion_summary.completions
    )
    return {
        "phase": "vllm",
        "checkpoint_path": args.checkpoint_path,
        "checkpoint_scope": CHECKPOINT_SCOPE,
        "tokenizer_path": args.tokenizer_path,
        "artifact_dir": args.artifact_dir,
        "result_path": args.result_path,
        "prompt": PROMPT,
        "prompt_batch_size": PROMPT_BATCH_SIZE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "completion": completion_summary.completion,
        "single_prompt_completion": completion_summary.single_completion,
        "single_prompt_choice_summary": completion_summary.single_prompt_choice_summary,
        "completions": completion_summary.completions,
        "expected_continuation": EXPECTED_CONTINUATION,
        "levanter_reference_result_path": args.levanter_result_path,
        "levanter_reference_completion": levanter_completion,
        "levanter_reference_passed": levanter_reference.get("passed"),
        "levanter_reference_match": levanter_match,
        "levanter_reference_single_match": levanter_single_match,
        "all_completions_match_levanter": levanter_all_match,
        "all_completions_match_expected": expected_all_match,
        "passed": (
            levanter_reference.get("passed") is True
            and levanter_match
            and levanter_single_match
            and levanter_all_match
            and expected_all_match
        ),
        "served_model_name": SERVED_MODEL_NAME,
        "vllm_model_id": model_id,
        "vllm_model_path": context.staged_artifact.vllm_model_path,
        "artifact_staging": context.staged_artifact.staging,
        "vllm_engine_kwargs": context.model.engine_kwargs,
        "vllm_args": context.extra_args,
        "vllm_dtype": VLLM_DTYPE,
        "vllm_tensor_parallel_size": VLLM_TENSOR_PARALLEL_SIZE,
        "vllm_data_parallel_size": VLLM_DATA_PARALLEL_SIZE,
        "vllm_expert_parallel_size": VLLM_EXPERT_PARALLEL_SIZE,
        "vllm_attention_backend": context.attention_backend,
        "vllm_max_num_seqs": VLLM_MAX_NUM_SEQS,
        "vllm_gpu_memory_utilization": VLLM_GPU_MEMORY_UTILIZATION,
        "rank_request_batches": completion_batch.rank_request_batches,
        "requested_data_parallel_ranks": [
            batch["data_parallel_rank"] for batch in completion_batch.rank_request_batches
        ],
        "main_choice_summaries": completion_summary.main_choice_summaries,
        "expected_gpu_count": EXPECTED_GPU_COUNT,
        "torch_runtime": context.torch_runtime,
        "vllm_log_artifacts": log_artifacts,
        "phase_timings": phase_timings,
        "elapsed_seconds": time.time() - context.started,
    }


def _vllm_backend(args: argparse.Namespace) -> None:
    attention_backend = args.attention_backend
    phase_timings: dict[str, float] = {}
    s3_env = _configure_coreweave_s3_env()
    vllm_env = _configure_vllm_gpu_env()
    torch_runtime = _require_torch_cuda_runtime()
    levanter_reference = _read_json(args.levanter_result_path)

    from marin.evaluation.evaluators.evaluator import ModelConfig  # noqa: PLC0415
    from marin.inference.vllm_server import VllmEnvironment  # noqa: PLC0415

    staging_started = time.time()
    staged_artifact = _stage_artifact_for_vllm(args.artifact_dir)
    phase_timings["artifact_staging_seconds"] = time.time() - staging_started
    model = ModelConfig(
        name=SERVED_MODEL_NAME,
        path=staged_artifact.vllm_model_path,
        engine_kwargs={
            "max_model_len": MAX_MODEL_LEN,
            "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        },
    )
    extra_args = _vllm_extra_args(attention_backend)
    started = time.time()
    context = VllmBackendContext(
        args=args,
        attention_backend=attention_backend,
        s3_env=s3_env,
        vllm_env=vllm_env,
        torch_runtime=torch_runtime,
        staged_artifact=staged_artifact,
        model=model,
        extra_args=extra_args,
        started=started,
    )
    log_artifacts: dict[str, Any] = {}
    try:
        startup_started = time.time()
        with VllmEnvironment(model=model, timeout_seconds=SERVER_TIMEOUT_SECONDS, extra_args=extra_args) as env:
            phase_timings["vllm_startup_load_seconds"] = time.time() - startup_started
            print("vllm_gpu_server_initialized=True", flush=True)
            print("vllm_gpu_server_url=" + env.server_url, flush=True)
            print("vllm_gpu_model_path=" + staged_artifact.vllm_model_path, flush=True)
            print("vllm_gpu_artifact_staging=" + json.dumps(staged_artifact.staging, sort_keys=True), flush=True)
            print("vllm_gpu_server_log_dir=" + (env.vllm_server.log_dir if env.vllm_server else ""), flush=True)
            generation_started = time.time()
            completion_batch = _collect_vllm_completion_batch(env, attention_backend=attention_backend)
            phase_timings["vllm_generation_seconds"] = time.time() - generation_started
            log_upload_started = time.time()
            log_artifacts = _copy_vllm_server_logs(
                env.vllm_server.log_dir if env.vllm_server else None,
                args.output_dir,
                attention_backend=attention_backend,
            )
            phase_timings["vllm_log_upload_seconds"] = time.time() - log_upload_started
            model_id = env.model_id
    except Exception as exc:
        phase_timings.setdefault("vllm_startup_load_seconds", time.time() - started)
        log_upload_started = time.time()
        log_artifacts = _copy_vllm_server_logs(
            _latest_vllm_server_log_dir(since=started),
            args.output_dir,
            attention_backend=attention_backend,
        )
        phase_timings["vllm_log_upload_seconds"] = time.time() - log_upload_started
        phase_timings["total_seconds"] = time.time() - started
        failure_result = _vllm_failure_result(
            context,
            exc,
            log_artifacts=log_artifacts,
            phase_timings=phase_timings,
        )
        _write_json(args.result_path, failure_result)
        print("grugmoe_gpu_real_checkpoint_vllm_result=" + json.dumps(failure_result, sort_keys=True), flush=True)
        raise
    completion_summary = _summarize_vllm_completion_batch(completion_batch)
    phase_timings["total_seconds"] = time.time() - started
    result = _vllm_success_result(
        context,
        completion_batch,
        completion_summary,
        model_id=model_id,
        log_artifacts=log_artifacts,
        levanter_reference=levanter_reference,
        phase_timings=phase_timings,
    )
    _write_json(args.result_path, result)
    print("grugmoe_gpu_real_checkpoint_vllm_result=" + json.dumps(result, sort_keys=True), flush=True)
    if result["passed"] is not True:
        raise AssertionError(
            f"GPU vLLM {attention_backend} single={completion_summary.single_completion!r}, "
            f"levanter_reference={levanter_reference.get('completion')!r}, "
            f"expected={EXPECTED_CONTINUATION!r}, completions={completion_summary.completions!r}"
        )


def _build_levanter_reference_result(
    args: argparse.Namespace,
    *,
    loaded_model: Any,
    model_cfg: Any,
    tokenizer: Any,
    jax_runtime: dict[str, Any],
    mesh: Any,
    mesh_runtime: dict[str, Any],
) -> dict[str, Any]:
    prompt_ids, tokenization = _prompt_token_ids(tokenizer, PROMPT)
    decode_batch_size = _mesh_batch_axis_size(mesh)
    pad_token_id = int(getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None) or 0)
    started = time.time()
    executable_model = _executable_model_from_legacy_split(
        loaded_model,
        capacity_factor=LEVANTER_MOE_CAPACITY_FACTOR,
    )
    model, reference_policy = _apply_levanter_reference_mode(executable_model)
    reference_checks: list[dict[str, Any]] = []
    completions: list[str] = []
    for _ in range(LEVANTER_REFERENCE_REPEAT_COUNT):
        decode_result = _greedy_decode(
            model,
            tokenizer,
            prompt_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            batch_size=decode_batch_size,
            decode_seq_len=DECODE_SEQ_LEN,
            pad_token_id=pad_token_id,
            use_active_prefix=LEVANTER_DECODE_USE_ACTIVE_PREFIX,
        )
        serving_completion, serving_decode = _serving_completion_from_generated_ids(
            tokenizer,
            list(decode_result["generated_token_ids"]),
        )
        completions.append(serving_completion)
        reference_checks.append(
            {
                "completion": serving_completion,
                "raw_completion": decode_result["completion"],
                "serving_decode": serving_decode,
                "prompt_token_ids": decode_result["prompt_token_ids"],
                "generated_token_ids": decode_result["generated_token_ids"],
                "generated_token_texts": decode_result["generated_token_texts"],
                "steps": decode_result["steps"],
                "decode_batch_size": decode_result["decode_batch_size"],
                "decode_seq_len": decode_result["decode_seq_len"],
                "use_active_prefix": decode_result["use_active_prefix"],
            }
        )
    completion = completions[0]
    reference_stable = len(completions) == LEVANTER_REFERENCE_REPEAT_COUNT and all(
        item == completion for item in completions
    )
    reference_matches_expected = reference_stable and completion == EXPECTED_CONTINUATION
    result = {
        "phase": "levanter",
        "checkpoint_path": args.checkpoint_path,
        "checkpoint_scope": CHECKPOINT_SCOPE,
        "tokenizer_path": args.tokenizer_path,
        "result_path": args.levanter_result_path,
        "prompt": PROMPT,
        "prompt_batch_size": PROMPT_BATCH_SIZE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "completion": completion,
        "reference_repeat_count": LEVANTER_REFERENCE_REPEAT_COUNT,
        "reference_completions": completions,
        "reference_stable": reference_stable,
        "expected_continuation": EXPECTED_CONTINUATION,
        "reference_contract": "scalar_expected_levanter_reference_and_all_vllm_outputs_must_match",
        "reference_matches_expected": reference_matches_expected,
        "passed": reference_matches_expected,
        "tokenization": tokenization,
        "decode_batch_size": decode_batch_size,
        "levanter_expert_axis_size": LEVANTER_EXPERT_AXIS_SIZE,
        "levanter_moe_capacity_factor": LEVANTER_MOE_CAPACITY_FACTOR,
        "levanter_decode_use_active_prefix": LEVANTER_DECODE_USE_ACTIVE_PREFIX,
        "reference_checks": reference_checks,
        "levanter_reference_mode": LEVANTER_REFERENCE_MODE,
        "levanter_reference_policy": reference_policy,
        "jax_runtime": jax_runtime,
        "jax_mesh": mesh_runtime,
        "elapsed_seconds": time.time() - started,
    }
    return result


def _parse_backend_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Internal GrugMoE GPU real-checkpoint e2e backend")
    parser.add_argument("--backend", choices=("export", "vllm"), required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--levanter-result-path", required=True)
    parser.add_argument("--attention-backend", choices=VLLM_ATTENTION_BACKENDS_UNDER_TEST)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_backend_args(sys.argv[1:] if argv is None else argv)
    match args.backend:
        case "export":
            _export_backend(args)
        case "vllm":
            _vllm_backend(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
