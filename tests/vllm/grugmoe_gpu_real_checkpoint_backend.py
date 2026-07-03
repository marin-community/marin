# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GPU/CoreWeave backend subprocess entrypoints for GrugMoE real-checkpoint e2e.

This mirrors ``grugmoe_real_checkpoint_backend`` while keeping the CoreWeave
S3, CUDA, and JAX GPU guards separate from the TPU/GCS path.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import hashlib
import importlib.metadata as md
import importlib.util
import io
import json
import os
import posixpath
import shlex
import shutil
import site
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

try:
    from tests.vllm import grugmoe_real_checkpoint_backend as common
except ModuleNotFoundError as exc:
    if exc.name not in {"tests", "tests.vllm"}:
        raise
    common_path = Path(__file__).with_name("grugmoe_real_checkpoint_backend.py")
    common_spec = importlib.util.spec_from_file_location("grugmoe_real_checkpoint_backend_common", common_path)
    if common_spec is None or common_spec.loader is None:
        raise
    common = importlib.util.module_from_spec(common_spec)
    sys.modules[common_spec.name] = common
    common_spec.loader.exec_module(common)

COREWEAVE_S3_PREFIX = "s3://marin-us-east-02a/"
REGION = "cw-us-east-02a"
COREWEAVE_SIGNING_REGION = "US-EAST-02A"
COREWEAVE_ENDPOINT_INSIDE = "http://cwlota.com"
COREWEAVE_ENDPOINT_OUTSIDE = "https://cwobject.com"
GPU_NODE_TYPE = "H100x8"
GPU_NODEPOOL = "cw-use02a-h100-8x"
EXPECTED_GPU_COUNT = 8
VISIBLE_CUDA_DEVICES = ",".join(str(index) for index in range(EXPECTED_GPU_COUNT))
VLLM_TENSOR_PARALLEL_SIZE = 1
VLLM_DATA_PARALLEL_SIZE = 8
VLLM_EXPERT_PARALLEL_SIZE = 8
VLLM_MAX_NUM_SEQS = 16
VLLM_ATTENTION_BACKEND_ENV = "MARIN_GRUGMOE_VLLM_ATTENTION_BACKEND"
VLLM_DEFAULT_ATTENTION_BACKEND = "TRITON_ATTN"
VLLM_ATTENTION_BACKENDS_UNDER_TEST = ("TRITON_ATTN", "FLASH_ATTN")
LEVANTER_MOE_CAPACITY_FACTOR = float(EXPECTED_GPU_COUNT)
LEVANTER_DECODE_USE_ACTIVE_PREFIX = True
CHECKPOINT_PATH = "s3://marin-us-east-02a/marin/grug/moe_may_compute_opt_d512_ep1-05c39b/checkpoints/step-10980"
TOKENIZER_PATH = "s3://marin-us-east-02a/marin/tokenizers/marin-community/marin-tokenizer/hf-hub-0.36.2"
OUTPUT_ROOT = "s3://marin-us-east-02a/tmp/ttl=14d/grugmoe-gpu-real-checkpoint-e2e"
CACHE_ROOT = "s3://marin-us-east-02a/compilation-cache/grugmoe-gpu-real-checkpoint-e2e"
PROMPT = common.PROMPT
PROMPT_BATCH_SIZE = 16
PROMPTS = tuple(PROMPT for _ in range(PROMPT_BATCH_SIZE))
EXPECTED_CONTINUATION = common.EXPECTED_CONTINUATION
MAX_MODEL_LEN = common.MAX_MODEL_LEN
MAX_NUM_BATCHED_TOKENS = 1024
MAX_NEW_TOKENS = common.MAX_NEW_TOKENS
LEVANTER_PROMPT_ADD_SPECIAL_TOKENS = common.LEVANTER_PROMPT_ADD_SPECIAL_TOKENS
EXPECTED_PROMPT_TOKEN_COUNT = common.EXPECTED_PROMPT_TOKEN_COUNT
DECODE_SEQ_LEN = common.DECODE_SEQ_LEN
SERVER_TIMEOUT_SECONDS = common.SERVER_TIMEOUT_SECONDS
SERVED_MODEL_NAME = "grugmoe-gpu-real-checkpoint-e2e"
VLLM_DTYPE = common.VLLM_DTYPE
MAX_SHARD_SIZE = common.MAX_SHARD_SIZE

E2EPaths = common.E2EPaths
StagedArtifact = common.StagedArtifact
_join_path = common._join_path
_real_checkpoint_model_config = common._real_checkpoint_model_config
_legacy_split_expert_inference_state_dict = common._legacy_split_expert_inference_state_dict
_load_legacy_split_expert_checkpoint = common._load_legacy_split_expert_checkpoint
_local_filesystem_path = common._local_filesystem_path
_tokenizer_encode = common._tokenizer_encode
_levanter_prompt_token_ids = common._levanter_prompt_token_ids
_decode_one = common._decode_one
_mesh_batch_axis_size = common._mesh_batch_axis_size
_selected_logprob = common._selected_logprob
_executable_model_from_legacy_split = common._executable_model_from_legacy_split
_greedy_decode = common._greedy_decode


def _resolve_vllm_attention_backend() -> str:
    value = os.environ.get(VLLM_ATTENTION_BACKEND_ENV, VLLM_DEFAULT_ATTENTION_BACKEND).strip().upper()
    if value not in VLLM_ATTENTION_BACKENDS_UNDER_TEST:
        raise ValueError(
            f"{VLLM_ATTENTION_BACKEND_ENV}={value!r} is not supported for this validation; "
            f"expected one of {VLLM_ATTENTION_BACKENDS_UNDER_TEST!r}"
        )
    return value


VLLM_ATTENTION_BACKEND = _resolve_vllm_attention_backend()


def _is_coreweave_endpoint(endpoint: str) -> bool:
    hostname = urlparse(endpoint).hostname or ""
    return (
        hostname in {"cwobject.com", "cwlota.com"}
        or hostname.endswith(".cwobject.com")
        or hostname.endswith(".cwlota.com")
    )


def _in_coreweave_runtime() -> bool:
    return any(os.environ.get(key) for key in ("IRIS_POD_NAME", "IRIS_WORKER_NODE_NAME", "KUBERNETES_SERVICE_HOST"))


def _default_coreweave_endpoint() -> str:
    return COREWEAVE_ENDPOINT_INSIDE if _in_coreweave_runtime() else COREWEAVE_ENDPOINT_OUTSIDE


def _source_yoblin_env_if_present() -> dict[str, Any]:
    env_path = Path(os.environ.get("YOBLIN_ENV_PATH", "~/.config/yoblin/env")).expanduser()
    loaded_keys: list[str] = []
    if not env_path.exists():
        return {"path": str(env_path), "exists": False, "loaded_keys": loaded_keys}

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            parts = shlex.split(line, comments=True, posix=True)
        except ValueError:
            continue
        if not parts:
            continue
        if parts[0] == "export":
            parts = parts[1:]
        for part in parts:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            if key in {"CW_ACCESS_KEY_ID", "CW_SECRET_ACCESS_KEY"} and not os.environ.get(key):
                os.environ[key] = value
                loaded_keys.append(key)

    return {"path": str(env_path), "exists": True, "loaded_keys": sorted(loaded_keys)}


def _configure_coreweave_s3_env() -> dict[str, Any]:
    """Configure process-local S3 env for CoreWeave AI Object Storage.

    Iris tasks often carry R2 credentials in AWS_* for ``s3://marin-na``. This
    e2e reads ``s3://marin-us-east-02a``, so CW_* credentials intentionally win
    when they are available.
    """

    yoblin_env = _source_yoblin_env_if_present()
    cw_key = os.environ.get("CW_ACCESS_KEY_ID")
    cw_secret = os.environ.get("CW_SECRET_ACCESS_KEY")
    endpoint: str | None
    using_cw_credentials = bool(cw_key and cw_secret)
    if using_cw_credentials:
        os.environ["AWS_ACCESS_KEY_ID"] = cw_key or ""
        os.environ["AWS_SECRET_ACCESS_KEY"] = cw_secret or ""
        endpoint = os.environ.get("CW_ENDPOINT_URL") or _default_coreweave_endpoint()
        os.environ["AWS_ENDPOINT_URL"] = endpoint
        os.environ["AWS_ENDPOINT_URL_S3"] = endpoint
    else:
        endpoint = os.environ.get("AWS_ENDPOINT_URL_S3") or os.environ.get("AWS_ENDPOINT_URL")
        if not (os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY")):
            raise RuntimeError(
                "CoreWeave S3 access requires CW_ACCESS_KEY_ID/CW_SECRET_ACCESS_KEY, or an already configured "
                "AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY pair."
            )
        if not endpoint or not _is_coreweave_endpoint(endpoint):
            raise RuntimeError(
                "CoreWeave S3 access requires AWS_ENDPOINT_URL or AWS_ENDPOINT_URL_S3 to point at "
                f"{COREWEAVE_ENDPOINT_INSIDE} or {COREWEAVE_ENDPOINT_OUTSIDE} when CW_* credentials are not set."
            )
        os.environ.setdefault("AWS_ENDPOINT_URL", endpoint)
        os.environ.setdefault("AWS_ENDPOINT_URL_S3", endpoint)

    os.environ["AWS_REGION"] = COREWEAVE_SIGNING_REGION
    os.environ["AWS_DEFAULT_REGION"] = COREWEAVE_SIGNING_REGION
    fsspec_conf = {
        "endpoint_url": endpoint,
        "client_kwargs": {"region_name": COREWEAVE_SIGNING_REGION},
        "config_kwargs": {"s3": {"addressing_style": "virtual"}},
    }
    os.environ["FSSPEC_S3"] = json.dumps(fsspec_conf, sort_keys=True)

    try:
        import fsspec  # noqa: PLC0415
        import s3fs  # noqa: PLC0415

        fsspec.config.set_conf_env(fsspec.config.conf)
        s3fs.S3FileSystem.clear_instance_cache()
    except ModuleNotFoundError:
        pass

    return {
        "endpoint": endpoint,
        "signing_region": COREWEAVE_SIGNING_REGION,
        "uses_cw_credentials": using_cw_credentials,
        "uses_aws_credentials": bool(os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY")),
        "fsspec_s3": fsspec_conf,
        "yoblin_env": yoblin_env,
    }


def _fs_path(path: str):
    if path.startswith("s3://"):
        _configure_coreweave_s3_env()

    import fsspec  # noqa: PLC0415

    return fsspec.core.url_to_fs(path)


def _exists(path: str) -> bool:
    fs, plain_path = _fs_path(path)
    return fs.exists(plain_path)


def _remove_tree(path: str) -> None:
    fs, plain_path = _fs_path(path)
    if fs.exists(plain_path):
        fs.rm(plain_path, recursive=True)


def _write_json(path: str, payload: dict[str, Any]) -> None:
    if path.startswith("s3://"):
        _configure_coreweave_s3_env()

    import fsspec  # noqa: PLC0415

    parent = path.rsplit("/", 1)[0]
    fs, plain_parent = _fs_path(parent)
    fs.makedirs(plain_parent, exist_ok=True)
    with fsspec.open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _read_json(path: str) -> dict[str, Any]:
    _require_coreweave_path("json_path", path)
    if path.startswith("s3://"):
        _configure_coreweave_s3_env()

    import fsspec  # noqa: PLC0415

    with fsspec.open(path, "r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object at {path}, got {type(payload).__name__}")
    return payload


def _require_coreweave_path(label: str, path: str) -> None:
    parsed = urlparse(path)
    if parsed.scheme == "s3":
        if path.startswith(COREWEAVE_S3_PREFIX):
            return
        raise ValueError(f"{label} must be under {COREWEAVE_S3_PREFIX}, got {path!r}")
    if parsed.scheme in {"", "file"}:
        return
    raise ValueError(f"{label} must be a local path or {COREWEAVE_S3_PREFIX} path, got {path!r}")


def _require_file(label: str, path: str) -> None:
    _require_coreweave_path(label, path)
    if not _exists(path):
        raise FileNotFoundError(f"{label} not found at {path}")


def _require_constants_are_coreweave(paths: E2EPaths | None = None) -> None:
    if REGION != "cw-us-east-02a":
        raise ValueError(f"GrugMoE GPU e2e region must be cw-us-east-02a, got {REGION!r}")
    if COREWEAVE_SIGNING_REGION != "US-EAST-02A":
        raise ValueError(f"GrugMoE GPU e2e signing region must be US-EAST-02A, got {COREWEAVE_SIGNING_REGION!r}")
    for label, path in {
        "checkpoint_path": CHECKPOINT_PATH,
        "tokenizer_path": TOKENIZER_PATH,
        "output_root": OUTPUT_ROOT,
        "cache_root": CACHE_ROOT,
        **(
            {
                "output_dir": paths.output_dir,
                "cache_dir": paths.cache_dir,
                "artifact_dir": paths.artifact_dir,
                "export_result_path": paths.export_result_path,
                "vllm_result_path": paths.vllm_result_path,
                "levanter_result_path": paths.levanter_result_path,
                "summary_result_path": paths.summary_result_path,
            }
            if paths is not None
            else {}
        ),
    }.items():
        _require_coreweave_path(label, path)


def _normalize_coreweave_region(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip().lower().replace("_", "-")
    if normalized in {"cw-us-east-02a", "us-east-02a"}:
        return REGION
    return normalized or None


def _json_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        strings: list[str] = []
        for item in value:
            strings.extend(_json_strings(item))
        return strings
    if isinstance(value, dict):
        strings: list[str] = []
        for item in value.values():
            strings.extend(_json_strings(item))
        return strings
    return []


def _region_candidates() -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    for env_name in (
        "IRIS_WORKER_REGION",
        "MARIN_GPU_REGION",
        "COREWEAVE_REGION",
        "COREWEAVE_ZONE",
        "CW_REGION",
        "CW_ZONE",
    ):
        value = os.environ.get(env_name)
        if value:
            candidates.append({"source": env_name, "value": value})

    raw_constraints = os.environ.get("IRIS_JOB_CONSTRAINTS")
    if raw_constraints:
        try:
            constraints = json.loads(raw_constraints)
        except json.JSONDecodeError:
            constraints = []
        if isinstance(constraints, list):
            for constraint in constraints:
                if isinstance(constraint, dict) and constraint.get("key") == "region":
                    for value in _json_strings(constraint):
                        candidates.append({"source": "IRIS_JOB_CONSTRAINTS", "value": value})
    return candidates


def _runtime_region() -> str | None:
    for candidate in _region_candidates():
        normalized = _normalize_coreweave_region(candidate["value"])
        if normalized == REGION:
            return normalized
    candidates = _region_candidates()
    if candidates:
        return _normalize_coreweave_region(candidates[0]["value"])
    return None


def _require_runtime_region() -> None:
    region = _runtime_region()
    if region != REGION:
        raise RuntimeError(
            f"GrugMoE GPU real-checkpoint e2e must run in {REGION}; detected {region!r}. "
            "Run it on CoreWeave marin-gpu/cw-us-east-02a, or set IRIS_WORKER_REGION/COREWEAVE_REGION "
            "for explicit validation."
        )


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, OSError) as exc:
        return f"unavailable:{exc!r}"


def _direct_url(package: str) -> str:
    try:
        direct_url = md.distribution(package).read_text("direct_url.json")
    except md.PackageNotFoundError:
        return "not-installed"
    return direct_url.strip() if direct_url else ""


def _version(package: str) -> str:
    try:
        return md.version(package)
    except md.PackageNotFoundError:
        return "not-installed"


def _torch_cuda_snapshot() -> dict[str, Any]:
    try:
        import torch  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    device_count = int(torch.cuda.device_count())
    return {
        "available": bool(torch.cuda.is_available()),
        "device_count": device_count,
        "expected_gpu_count": EXPECTED_GPU_COUNT,
        "devices": [torch.cuda.get_device_name(index) for index in range(device_count)],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _runtime_snapshot(
    *,
    include_jax_devices: bool = False,
    include_grugmoe_spec: bool = False,
    include_torch_cuda: bool = False,
) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "marin_sha": os.environ.get("MARIN_GIT_SHA") or _git_sha(),
        "region": _runtime_region(),
        "region_candidates": _region_candidates(),
        "coreweave_signing_region": COREWEAVE_SIGNING_REGION,
        "gpu_node_type": GPU_NODE_TYPE,
        "gpu_nodepool": GPU_NODEPOOL,
        "expected_gpu_count": EXPECTED_GPU_COUNT,
        "vllm_tensor_parallel_size": VLLM_TENSOR_PARALLEL_SIZE,
        "vllm_data_parallel_size": VLLM_DATA_PARALLEL_SIZE,
        "vllm_expert_parallel_size": VLLM_EXPERT_PARALLEL_SIZE,
        "packages": {
            package: {"version": _version(package), "direct_url": _direct_url(package)}
            for package in ("marin-core", "vllm", "tpu-inference", "jax", "torch")
        },
    }
    if include_grugmoe_spec:
        try:
            snapshot["grugmoe_spec"] = repr(importlib.util.find_spec("tpu_inference.models.jax.grugmoe"))
        except ModuleNotFoundError as exc:
            snapshot["grugmoe_spec"] = f"unavailable:{exc!r}"
    if include_jax_devices:
        import jax  # noqa: PLC0415

        snapshot.update(
            {
                "jax_process_index": jax.process_index(),
                "jax_process_count": jax.process_count(),
                "jax_local_device_count": jax.local_device_count(),
                "jax_default_backend": jax.default_backend(),
                "jax_devices": [str(device) for device in jax.devices()],
            }
        )
    if include_torch_cuda:
        snapshot["torch_cuda"] = _torch_cuda_snapshot()
    return snapshot


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


def _configure_jax_gpu_env(cache_dir: str) -> dict[str, Any]:
    if os.environ.get("PJRT_DEVICE", "").upper() == "TPU":
        raise RuntimeError("GPU GrugMoE e2e cannot run with PJRT_DEVICE=TPU")
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
    actual_cache_dir = _jax_compilation_cache_dir(cache_dir)
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", actual_cache_dir)
    os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "1")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    return {
        "requested_cache_dir": cache_dir,
        "actual_cache_dir": os.environ["JAX_COMPILATION_CACHE_DIR"],
        "jax_platforms": os.environ.get("JAX_PLATFORMS"),
        "jax_enable_compilation_cache": os.environ.get("JAX_ENABLE_COMPILATION_CACHE"),
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
        "uses_expected_gpu_count": len(gpu_devices) >= EXPECTED_GPU_COUNT,
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
        "uses_expected_gpu_count": len(flat_devices) >= EXPECTED_GPU_COUNT,
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
    os.environ.setdefault("MODEL_IMPL_TYPE", "vllm")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    return _configure_cuda_library_path()


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
    _require_constants_are_coreweave(
        E2EPaths(
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            artifact_dir=args.artifact_dir,
            export_result_path=args.result_path,
            vllm_result_path=_join_path(args.output_dir, "vllm-result.json"),
            levanter_result_path=_join_path(args.output_dir, "levanter-result.json"),
            summary_result_path=_join_path(args.output_dir, "result.json"),
        )
    )
    s3_env = _configure_coreweave_s3_env()
    jax_env = _configure_jax_gpu_env(args.cache_dir)
    _require_file("checkpoint metadata", _join_path(args.checkpoint_path, "metadata.json"))
    _require_file("tokenizer.json", _join_path(args.tokenizer_path, "tokenizer.json"))
    if _exists(args.artifact_dir):
        _remove_tree(args.artifact_dir)

    import equinox as eqx  # noqa: PLC0415
    import haliax  # noqa: PLC0415
    import jax  # noqa: PLC0415
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

        def to_state_dict(self, prefix: str | None = None) -> dict[str, jax.Array]:
            return _legacy_split_expert_inference_state_dict(self.model, self.config, prefix=prefix)

    jax_runtime = _require_jax_gpu_runtime()
    model_cfg = _real_checkpoint_model_config()
    mesh = compact_grug_mesh(expert_axis_size=EXPECTED_GPU_COUNT, model_axis_size=1)
    mesh_runtime = _mesh_snapshot(mesh)
    started = time.time()
    with ExitStack() as stack:
        stack.enter_context(set_mesh(mesh))
        stack.enter_context(haliax.axis_mapping({}))
        loaded_model = _load_legacy_split_expert_checkpoint(args.checkpoint_path, model_cfg)
        tokenizer = load_tokenizer(args.tokenizer_path)
        converter = model_cfg.hf_checkpoint_converter().replaced(tokenizer=tokenizer)
        converter.save_pretrained(
            LegacySplitExpertExportModel(loaded_model, model_cfg),
            args.artifact_dir,
            save_tokenizer=True,
            max_shard_size=MAX_SHARD_SIZE,
        )
    _require_file("exported config.json", _join_path(args.artifact_dir, "config.json"))
    _require_file("exported tokenizer.json", _join_path(args.artifact_dir, "tokenizer.json"))
    result = {
        "phase": "export",
        "checkpoint_path": args.checkpoint_path,
        "tokenizer_path": args.tokenizer_path,
        "artifact_dir": args.artifact_dir,
        "result_path": args.result_path,
        "model_config": dataclasses.asdict(model_cfg),
        "coreweave_s3": s3_env,
        "jax_env": jax_env,
        "jax_runtime": jax_runtime,
        "jax_mesh": mesh_runtime,
        "runtime": _runtime_snapshot(include_jax_devices=True, include_grugmoe_spec=True),
        "elapsed_seconds": time.time() - started,
    }
    _write_json(args.result_path, result)
    print("grugmoe_gpu_real_checkpoint_export_result=" + json.dumps(result, sort_keys=True), flush=True)


def _copy_tree_to_local(source_dir: str, local_dir: str) -> int:
    _require_coreweave_path("artifact_dir", source_dir)
    fs, source_path = _fs_path(source_dir)
    if not fs.exists(source_path):
        raise FileNotFoundError(f"artifact_dir not found at {source_dir}")
    if os.path.exists(local_dir):
        shutil.rmtree(local_dir)
    os.makedirs(local_dir, exist_ok=True)

    copied = 0
    for source_file in fs.find(source_path):
        rel_path = posixpath.relpath(source_file, source_path)
        local_file = os.path.join(local_dir, *rel_path.split("/"))
        os.makedirs(os.path.dirname(local_file), exist_ok=True)
        with fs.open(source_file, "rb") as src, open(local_file, "wb") as dst:
            shutil.copyfileobj(src, dst)
        copied += 1
    if copied == 0:
        raise FileNotFoundError(f"artifact_dir contained no files: {source_dir}")
    return copied


def _stage_artifact_for_vllm(artifact_dir: str) -> StagedArtifact:
    _require_coreweave_path("artifact_dir", artifact_dir)
    parsed = urlparse(artifact_dir)
    if parsed.scheme in {"", "file"}:
        local_path = _local_filesystem_path(artifact_dir)
        return StagedArtifact(
            vllm_model_path=local_path,
            staging={
                "staged": False,
                "source_artifact_dir": artifact_dir,
                "vllm_model_path": local_path,
                "copied_files": None,
            },
        )

    local_root = tempfile.mkdtemp(prefix="grugmoe-gpu-real-checkpoint-vllm-artifact-")
    local_path = os.path.join(local_root, "artifact")
    copied_files = _copy_tree_to_local(artifact_dir, local_path)
    _require_file("staged artifact config.json", _join_path(local_path, "config.json"))
    _require_file("staged artifact tokenizer.json", _join_path(local_path, "tokenizer.json"))
    return StagedArtifact(
        vllm_model_path=local_path,
        staging={
            "staged": True,
            "source_artifact_dir": artifact_dir,
            "vllm_model_path": local_path,
            "copied_files": copied_files,
        },
    )


def _format_int_ranges(values: list[int]) -> str:
    if not values:
        return "[]"
    ranges: list[str] = []
    start = prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append(str(start) if start == prev else f"{start}..{prev}")
        start = prev = value
    ranges.append(str(start) if start == prev else f"{start}..{prev}")
    return "[" + ", ".join(ranges) + "]"


def _unwrap_vllm_model(model: Any) -> Any:
    unwrap = getattr(model, "unwrap", None)
    if callable(unwrap):
        return unwrap()
    return model


def _first_grug_moe_mlp(model: Any) -> tuple[str, Any] | None:
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        return None
    for module_name, module in named_modules():
        if module.__class__.__name__ == "GrugMoeMLP":
            return str(module_name), module
    return None


def _grug_moe_ep_state_from_worker(worker: Any) -> dict[str, Any]:
    model_runner = getattr(worker, "model_runner", None)
    model = _unwrap_vllm_model(getattr(model_runner, "model", None))
    found = _first_grug_moe_mlp(model)
    worker_rank = int(getattr(worker, "rank", -1))
    local_rank = int(getattr(worker, "local_rank", -1))
    if found is None:
        return {
            "found": False,
            "worker_rank": worker_rank,
            "local_rank": local_rank,
        }

    module_name, mlp = found
    runner = mlp.experts
    moe_config = runner.moe_config
    moe_parallel_config = moe_config.moe_parallel_config
    expert_map_manager = runner.expert_map_manager
    local_expert_ids = [int(expert_id) for expert_id in expert_map_manager.get_local_expert_ids()]
    vllm_config = getattr(worker, "vllm_config", None)
    model_config = getattr(vllm_config, "model_config", None)
    return {
        "found": True,
        "worker_rank": worker_rank,
        "local_rank": local_rank,
        "module_name": module_name,
        "use_ep": bool(moe_parallel_config.use_ep),
        "tp_size": int(moe_parallel_config.tp_size),
        "tp_rank": int(moe_parallel_config.tp_rank),
        "dp_size": int(moe_parallel_config.dp_size),
        "dp_rank": int(moe_parallel_config.dp_rank),
        "ep_size": int(moe_parallel_config.ep_size),
        "ep_rank": int(moe_parallel_config.ep_rank),
        "global_num_experts": int(moe_config.num_experts),
        "logical_num_experts": int(moe_config.num_logical_experts),
        "local_num_experts": int(moe_config.num_local_experts),
        "local_expert_ids": local_expert_ids,
        "local_expert_ownership": _format_int_ranges(local_expert_ids),
        "top_k": int(moe_config.experts_per_token),
        "expert_placement_strategy": str(runner.expert_placement_strategy),
        "all2all_backend": str(moe_parallel_config.all2all_backend),
        "routed_experts_capture_enabled": bool(getattr(model_config, "enable_return_routed_experts", False)),
    }


class GrugMoeDiagnosticsWorkerExtension:
    """Test-only extension called through vLLM dev-mode collective_rpc."""

    def grugmoe_ep_state(self) -> dict[str, Any]:
        return _grug_moe_ep_state_from_worker(self)


def _server_root_url(env: Any) -> str:
    return str(env.server_url).removesuffix("/v1")


def _collective_rpc_payload(env: Any, method: str) -> dict[str, Any]:
    response = requests.post(
        f"{_server_root_url(env)}/collective_rpc",
        headers={},
        json={"method": method, "timeout": 300},
        timeout=300,
    )
    print(
        "vllm_gpu_collective_rpc_status="
        + json.dumps(
            {
                "method": method,
                "status_code": response.status_code,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not response.ok:
        print("vllm_gpu_collective_rpc_response_text=" + response.text[:4000], flush=True)
        print("vllm_gpu_server_logs_tail_begin", flush=True)
        print(env.logs_tail(max_lines=400), flush=True)
        print("vllm_gpu_server_logs_tail_end", flush=True)
        response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise AssertionError(f"collective_rpc returned non-object payload: {payload!r}")
    return payload


def _collect_grug_moe_worker_ep_states(env: Any) -> list[dict[str, Any]]:
    payload = _collective_rpc_payload(env, "grugmoe_ep_state")
    results = payload.get("results")
    if not isinstance(results, list):
        raise AssertionError(f"collective_rpc missing results list: {payload!r}")
    states: list[dict[str, Any]] = []
    for result in results:
        if not isinstance(result, dict):
            raise AssertionError(f"collective_rpc result is not a dict: {result!r}")
        states.append(result)
    return states


def _assert_grug_moe_worker_ep_states(
    states: list[dict[str, Any]],
    *,
    num_experts: int,
) -> dict[str, Any]:
    if len(states) != EXPECTED_GPU_COUNT:
        raise AssertionError(f"expected {EXPECTED_GPU_COUNT} worker states, got {states!r}")
    dp_ranks = sorted(int(state.get("dp_rank", -1)) for state in states)
    if dp_ranks != list(range(VLLM_DATA_PARALLEL_SIZE)):
        raise AssertionError(f"worker DP ranks did not cover all ranks: {dp_ranks!r}")
    ep_ranks = sorted(int(state.get("ep_rank", -1)) for state in states)
    if ep_ranks != list(range(VLLM_EXPERT_PARALLEL_SIZE)):
        raise AssertionError(f"worker EP ranks did not cover all ranks: {ep_ranks!r}")

    local_expert_ids: set[int] = set()
    for state in states:
        if state.get("found") is not True:
            raise AssertionError(f"worker did not report GrugMoE state: {state!r}")
        if state.get("use_ep") is not True:
            raise AssertionError(f"worker did not enable EP: {state!r}")
        if state.get("tp_size") != VLLM_TENSOR_PARALLEL_SIZE:
            raise AssertionError(f"unexpected worker TP size: {state!r}")
        if state.get("dp_size") != VLLM_DATA_PARALLEL_SIZE:
            raise AssertionError(f"unexpected worker DP size: {state!r}")
        if state.get("ep_size") != VLLM_EXPERT_PARALLEL_SIZE:
            raise AssertionError(f"unexpected worker EP size: {state!r}")
        if state.get("global_num_experts") != num_experts:
            raise AssertionError(f"unexpected worker expert count: {state!r}")
        if state.get("expert_placement_strategy") != "linear":
            raise AssertionError(f"unexpected expert placement: {state!r}")
        if state.get("routed_experts_capture_enabled") is not True:
            raise AssertionError(f"worker did not enable routed-expert capture: {state!r}")
        worker_local_experts = state.get("local_expert_ids")
        if not isinstance(worker_local_experts, list) or not worker_local_experts:
            raise AssertionError(f"worker did not report local experts: {state!r}")
        local_expert_ids.update(int(expert_id) for expert_id in worker_local_experts)

    covered_experts = sorted(local_expert_ids)
    expected_experts = list(range(num_experts))
    if covered_experts != expected_experts:
        raise AssertionError(
            f"worker local experts did not cover global experts: got {covered_experts!r}, expected {expected_experts!r}"
        )
    return {
        "worker_count": len(states),
        "dp_ranks": dp_ranks,
        "dp_rank_coverage": True,
        "ep_ranks": ep_ranks,
        "ep_rank_coverage": True,
        "local_expert_coverage": True,
        "local_expert_ownership": _format_int_ranges(covered_experts),
    }


def _completion_payload(
    env: Any,
    *,
    prompts: list[str],
    data_parallel_rank: int | None = None,
) -> dict[str, Any]:
    if env.model_id is None:
        raise RuntimeError("Expected vLLM server to expose a model id.")
    headers = {}
    if data_parallel_rank is not None:
        headers["X-data-parallel-rank"] = str(data_parallel_rank)
    response = requests.post(
        f"{env.server_url}/completions",
        headers=headers,
        json={
            "model": env.model_id,
            "prompt": prompts,
            "temperature": 0.0,
            "max_tokens": MAX_NEW_TOKENS,
        },
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


def _decode_routed_experts(value: str | None) -> list[Any] | None:
    if value is None:
        return None

    import numpy as np  # noqa: PLC0415

    routed = np.load(io.BytesIO(base64.b64decode(value)))
    return routed.astype("int64").tolist()


def _linear_owner_rank(global_expert_id: int, *, num_experts: int, ep_size: int) -> int:
    base_experts = num_experts // ep_size
    remainder = num_experts % ep_size
    larger_ranks_end = (base_experts + 1) * remainder
    if global_expert_id < larger_ranks_end:
        return global_expert_id // (base_experts + 1)
    if base_experts == 0:
        raise ValueError(f"num_experts={num_experts} must be >= ep_size={ep_size}")
    return remainder + ((global_expert_id - larger_ranks_end) // base_experts)


def _owners_for_linear_expert_placement(
    routed_experts: list[Any] | None,
    *,
    num_experts: int,
    ep_size: int,
) -> list[int]:
    if routed_experts is None:
        return []
    owners: set[int] = set()

    def visit(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        expert_id = int(value)
        if 0 <= expert_id < num_experts:
            owners.add(_linear_owner_rank(expert_id, num_experts=num_experts, ep_size=ep_size))

    visit(routed_experts)
    return sorted(owners)


def _vllm_backend(args: argparse.Namespace) -> None:
    _require_constants_are_coreweave(
        E2EPaths(
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            artifact_dir=args.artifact_dir,
            export_result_path=_join_path(args.output_dir, "export-result.json"),
            vllm_result_path=args.result_path,
            levanter_result_path=_join_path(args.output_dir, "levanter-result.json"),
            summary_result_path=_join_path(args.output_dir, "result.json"),
        )
    )
    s3_env = _configure_coreweave_s3_env()
    cuda_library_path = _configure_vllm_gpu_env()
    torch_runtime = _require_torch_cuda_runtime()
    artifact_config_path = _join_path(args.artifact_dir, "config.json")
    _require_file("artifact config.json", artifact_config_path)
    artifact_config = _read_json(artifact_config_path)
    num_experts = int(artifact_config.get("num_experts", artifact_config.get("num_local_experts", 0)))
    if num_experts <= 0:
        raise AssertionError(f"artifact config did not expose num_experts: {artifact_config!r}")

    from marin.evaluation.evaluators.evaluator import ModelConfig  # noqa: PLC0415
    from marin.inference.vllm_server import VllmEnvironment  # noqa: PLC0415

    staged_artifact = _stage_artifact_for_vllm(args.artifact_dir)
    model = ModelConfig(
        name=SERVED_MODEL_NAME,
        path=staged_artifact.vllm_model_path,
        engine_kwargs={
            "max_model_len": MAX_MODEL_LEN,
            "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        },
    )
    extra_args = [
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
        "--enable-return-routed-experts",
        "--worker-extension-cls",
        "tests.vllm.grugmoe_gpu_real_checkpoint_backend.GrugMoeDiagnosticsWorkerExtension",
        "--moe-backend",
        "triton",
        "--attention-backend",
        VLLM_ATTENTION_BACKEND,
        "--dtype",
        VLLM_DTYPE,
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--enforce-eager",
        "--max-num-seqs",
        str(VLLM_MAX_NUM_SEQS),
    ]
    started = time.time()
    previous_dev_mode = os.environ.get("VLLM_SERVER_DEV_MODE")
    os.environ["VLLM_SERVER_DEV_MODE"] = "1"
    try:
        with VllmEnvironment(model=model, timeout_seconds=SERVER_TIMEOUT_SECONDS, extra_args=extra_args) as env:
            print("vllm_gpu_server_initialized=True", flush=True)
            print("vllm_gpu_server_url=" + env.server_url, flush=True)
            print("vllm_gpu_model_path=" + staged_artifact.vllm_model_path, flush=True)
            print("vllm_gpu_artifact_staging=" + json.dumps(staged_artifact.staging, sort_keys=True), flush=True)
            print("vllm_gpu_server_log_dir=" + (env.vllm_server.log_dir if env.vllm_server else ""), flush=True)
            worker_ep_states = _collect_grug_moe_worker_ep_states(env)
            worker_ep_summary = _assert_grug_moe_worker_ep_states(worker_ep_states, num_experts=num_experts)
            single_payload = _completion_payload(
                env,
                prompts=[PROMPT],
                data_parallel_rank=0,
            )
            payloads: list[dict[str, Any]] = []
            rank_request_batches: list[dict[str, Any]] = []
            for data_parallel_rank in range(VLLM_DATA_PARALLEL_SIZE):
                prompts = list(PROMPTS[data_parallel_rank * 2 : data_parallel_rank * 2 + 2])
                payload = _completion_payload(
                    env,
                    prompts=prompts,
                    data_parallel_rank=data_parallel_rank,
                )
                payloads.append(payload)
                rank_request_batches.append(
                    {
                        "data_parallel_rank": data_parallel_rank,
                        "prompt_indices": [data_parallel_rank * 2, data_parallel_rank * 2 + 1],
                        "batch_size": len(prompts),
                    }
                )
            logs_tail = env.logs_tail(max_lines=160)
            model_id = env.model_id
    finally:
        if previous_dev_mode is None:
            os.environ.pop("VLLM_SERVER_DEV_MODE", None)
        else:
            os.environ["VLLM_SERVER_DEV_MODE"] = previous_dev_mode
    single_choices = single_payload.get("choices")
    if not isinstance(single_choices, list) or len(single_choices) != 1:
        raise AssertionError(f"expected exactly one single-prompt completion choice, got {single_payload!r}")
    single_completion = str(single_choices[0].get("text", ""))
    single_routed_experts = _decode_routed_experts(single_choices[0].get("routed_experts"))
    completions: list[str] = []
    routed_experts_by_completion: list[list[Any] | None] = []
    routed_owner_ranks: set[int] = set()
    for payload in payloads:
        choices = payload.get("choices")
        if not isinstance(choices, list) or len(choices) != 2:
            raise AssertionError(f"expected exactly two completion choices, got {payload!r}")
        for choice in choices:
            completion = str(choice.get("text", ""))
            completions.append(completion)
            routed_experts = _decode_routed_experts(choice.get("routed_experts"))
            routed_experts_by_completion.append(routed_experts)
            routed_owner_ranks.update(
                _owners_for_linear_expert_placement(
                    routed_experts,
                    num_experts=num_experts,
                    ep_size=VLLM_EXPERT_PARALLEL_SIZE,
                )
            )
    completion = completions[0] if completions else ""
    if len(completions) != PROMPT_BATCH_SIZE:
        raise AssertionError(f"expected {PROMPT_BATCH_SIZE} completions, got {len(completions)}")
    if any(item != completion for item in completions):
        raise AssertionError(f"expected identical completions for repeated prompts, got {completions!r}")
    if sorted(routed_owner_ranks) != list(range(VLLM_EXPERT_PARALLEL_SIZE)):
        raise AssertionError(f"routed experts did not cover all EP owner ranks: {sorted(routed_owner_ranks)!r}")
    result = {
        "phase": "vllm",
        "checkpoint_path": args.checkpoint_path,
        "tokenizer_path": args.tokenizer_path,
        "artifact_dir": args.artifact_dir,
        "result_path": args.result_path,
        "prompt": PROMPT,
        "prompt_batch_size": PROMPT_BATCH_SIZE,
        "completion": completion,
        "single_prompt_completion": single_completion,
        "single_prompt_routed_experts": single_routed_experts,
        "completions": completions,
        "expected_continuation": EXPECTED_CONTINUATION,
        "passed": single_completion == EXPECTED_CONTINUATION
        and all(item == EXPECTED_CONTINUATION for item in completions),
        "served_model_name": SERVED_MODEL_NAME,
        "vllm_model_id": model_id,
        "vllm_model_path": staged_artifact.vllm_model_path,
        "artifact_staging": staged_artifact.staging,
        "vllm_engine_kwargs": model.engine_kwargs,
        "vllm_args": extra_args,
        "vllm_attention_backend_env_var": VLLM_ATTENTION_BACKEND_ENV,
        "vllm_tensor_parallel_size": VLLM_TENSOR_PARALLEL_SIZE,
        "vllm_data_parallel_size": VLLM_DATA_PARALLEL_SIZE,
        "vllm_expert_parallel_size": VLLM_EXPERT_PARALLEL_SIZE,
        "vllm_attention_backend": VLLM_ATTENTION_BACKEND,
        "vllm_max_num_seqs": VLLM_MAX_NUM_SEQS,
        "vllm_server_dev_mode_enabled": True,
        "worker_ep_states": worker_ep_states,
        "worker_ep_summary": worker_ep_summary,
        "rank_request_batches": rank_request_batches,
        "observed_worker_data_parallel_ranks": worker_ep_summary["dp_ranks"],
        "requested_data_parallel_ranks": [batch["data_parallel_rank"] for batch in rank_request_batches],
        "routed_experts_by_completion": routed_experts_by_completion,
        "routed_expert_num_experts": num_experts,
        "routed_expert_owner_ranks": sorted(routed_owner_ranks),
        "routed_expert_owner_rank_coverage": sorted(routed_owner_ranks) == list(range(VLLM_EXPERT_PARALLEL_SIZE)),
        "expected_gpu_count": EXPECTED_GPU_COUNT,
        "coreweave_s3": s3_env,
        "cuda_library_path": cuda_library_path,
        "torch_runtime": torch_runtime,
        "raw_responses": payloads,
        "vllm_logs_tail": logs_tail,
        "runtime": _runtime_snapshot(include_grugmoe_spec=True, include_torch_cuda=True),
        "elapsed_seconds": time.time() - started,
    }
    _write_json(args.result_path, result)
    print("grugmoe_gpu_real_checkpoint_vllm_result=" + json.dumps(result, sort_keys=True), flush=True)
    if result["passed"] is not True:
        raise AssertionError(
            f"GPU vLLM single={single_completion!r}, batched={completions!r} != expected {EXPECTED_CONTINUATION!r}"
        )


def _levanter_backend(args: argparse.Namespace) -> None:
    _require_constants_are_coreweave(
        E2EPaths(
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            artifact_dir=args.artifact_dir,
            export_result_path=_join_path(args.output_dir, "export-result.json"),
            vllm_result_path=_join_path(args.output_dir, "vllm-result.json"),
            levanter_result_path=args.result_path,
            summary_result_path=_join_path(args.output_dir, "result.json"),
        )
    )
    s3_env = _configure_coreweave_s3_env()
    jax_env = _configure_jax_gpu_env(args.cache_dir)
    _require_file("checkpoint metadata", _join_path(args.checkpoint_path, "metadata.json"))
    _require_file("tokenizer.json", _join_path(args.tokenizer_path, "tokenizer.json"))

    import haliax  # noqa: PLC0415
    from haliax.partitioning import set_mesh  # noqa: PLC0415
    from levanter.compat.hf_checkpoints import load_tokenizer  # noqa: PLC0415
    from levanter.grug.sharding import compact_grug_mesh  # noqa: PLC0415

    jax_runtime = _require_jax_gpu_runtime()
    tokenizer = load_tokenizer(args.tokenizer_path)
    prompt_ids, tokenization = _levanter_prompt_token_ids(tokenizer)
    model_cfg = _real_checkpoint_model_config()
    mesh = compact_grug_mesh(expert_axis_size=EXPECTED_GPU_COUNT, model_axis_size=1)
    mesh_runtime = _mesh_snapshot(mesh)
    decode_batch_size = _mesh_batch_axis_size(mesh)
    pad_token_id = int(getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None) or 0)
    started = time.time()
    with ExitStack() as stack:
        stack.enter_context(set_mesh(mesh))
        stack.enter_context(haliax.axis_mapping({}))
        loaded_model = _load_legacy_split_expert_checkpoint(args.checkpoint_path, model_cfg)
        model = _executable_model_from_legacy_split(
            loaded_model,
            capacity_factor=LEVANTER_MOE_CAPACITY_FACTOR,
        )
        decode_results: list[dict[str, Any]] = []
        completions: list[str] = []
        remaining = PROMPT_BATCH_SIZE
        while remaining > 0:
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
            decode_results.append(decode_result)
            batch_items = min(decode_batch_size, remaining)
            completions.extend([str(decode_result["completion"])] * batch_items)
            remaining -= batch_items
    completion = completions[0] if completions else ""
    result = {
        "phase": "levanter",
        "checkpoint_path": args.checkpoint_path,
        "tokenizer_path": args.tokenizer_path,
        "result_path": args.result_path,
        "prompt": PROMPT,
        "prompt_batch_size": PROMPT_BATCH_SIZE,
        "completion": completion,
        "completions": completions,
        "expected_continuation": EXPECTED_CONTINUATION,
        "passed": all(item == EXPECTED_CONTINUATION for item in completions),
        "tokenization": tokenization,
        "decode_batch_size": decode_batch_size,
        "levanter_moe_capacity_factor": LEVANTER_MOE_CAPACITY_FACTOR,
        "levanter_decode_use_active_prefix": LEVANTER_DECODE_USE_ACTIVE_PREFIX,
        "decode_results": decode_results,
        "coreweave_s3": s3_env,
        "jax_env": jax_env,
        "jax_runtime": jax_runtime,
        "jax_mesh": mesh_runtime,
        "runtime": _runtime_snapshot(include_jax_devices=True, include_grugmoe_spec=True),
        "elapsed_seconds": time.time() - started,
    }
    _write_json(args.result_path, result)
    print("grugmoe_gpu_real_checkpoint_levanter_result=" + json.dumps(result, sort_keys=True), flush=True)
    if result["passed"] is not True:
        raise AssertionError(f"GPU Levanter/JAX completions {completions!r} != expected {EXPECTED_CONTINUATION!r}")


def _parse_backend_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Internal GrugMoE GPU real-checkpoint e2e backend")
    parser.add_argument("--backend", choices=("export", "vllm", "levanter"), required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--result-path", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_backend_args(sys.argv[1:] if argv is None else argv)
    _require_runtime_region()
    match args.backend:
        case "export":
            _export_backend(args)
        case "vllm":
            _vllm_backend(args)
        case "levanter":
            _levanter_backend(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
