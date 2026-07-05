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
import importlib
import importlib.metadata as md
import importlib.util
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
VLLM_DTYPE = "bfloat16"
VLLM_ATTENTION_BACKEND_ENV = "MARIN_GRUGMOE_VLLM_ATTENTION_BACKEND"
LEVANTER_REFERENCE_MODE = "bf16_compute"
LEVANTER_BF16_POLICY = "params=float32,compute=bfloat16,output=bfloat16"
LEVANTER_EXPERT_AXIS_SIZE = EXPECTED_GPU_COUNT
RUN_ID_ENV = "MARIN_GRUGMOE_GPU_E2E_RUN_ID"
OUTPUT_DIR_ENV = "MARIN_GRUGMOE_GPU_E2E_OUTPUT_DIR"
INSTALL_REPORT_PATH_ENV = "MARIN_GRUGMOE_GPU_E2E_INSTALL_REPORT_PATH"
# TRITON_ATTN is the default pass gate. FLASH_ATTN is retained as an explicit
# debug mode and is not run unless selected through the environment or runner.
VLLM_DEFAULT_ATTENTION_BACKEND = "TRITON_ATTN"
VLLM_ATTENTION_BACKENDS_UNDER_TEST = ("TRITON_ATTN", "FLASH_ATTN")
LEVANTER_MOE_CAPACITY_FACTOR = float(EXPECTED_GPU_COUNT)
LEVANTER_DECODE_USE_ACTIVE_PREFIX = True
CHECKPOINT_PATH = "s3://marin-us-east-02a/marin/grug/moe_may_compute_opt_d512_ep1-05c39b/checkpoints/step-10980"
TOKENIZER_PATH = "s3://marin-us-east-02a/marin/tokenizers/marin-community/marin-tokenizer/hf-hub-0.36.2"
OUTPUT_ROOT = "s3://marin-us-east-02a/tmp/ttl=14d/grugmoe-gpu-real-checkpoint-e2e"
CACHE_ROOT = "s3://marin-us-east-02a/compilation-cache/grugmoe-gpu-real-checkpoint-e2e"
PROMPT = "Answer with one word only. What color is the sky on a clear day?"
PROMPT_BATCH_SIZE = 16
PROMPTS = tuple(PROMPT for _ in range(PROMPT_BATCH_SIZE))
EXPECTED_CONTINUATION = " The sky is the"
MAX_MODEL_LEN = common.MAX_MODEL_LEN
MAX_NUM_BATCHED_TOKENS = 1024
MAX_NEW_TOKENS = 4
LEVANTER_PROMPT_ADD_SPECIAL_TOKENS = common.LEVANTER_PROMPT_ADD_SPECIAL_TOKENS
DECODE_SEQ_LEN = common.DECODE_SEQ_LEN
SERVER_TIMEOUT_SECONDS = common.SERVER_TIMEOUT_SECONDS
SERVED_MODEL_NAME = "grugmoe-gpu-real-checkpoint-e2e"
MAX_SHARD_SIZE = common.MAX_SHARD_SIZE

E2EPaths = common.E2EPaths
StagedArtifact = common.StagedArtifact
_join_path = common._join_path
_real_checkpoint_model_config = common._real_checkpoint_model_config
_legacy_split_expert_inference_state_dict = common._legacy_split_expert_inference_state_dict
_load_legacy_split_expert_checkpoint = common._load_legacy_split_expert_checkpoint
_local_filesystem_path = common._local_filesystem_path
_tokenizer_encode = common._tokenizer_encode
_mesh_batch_axis_size = common._mesh_batch_axis_size
_executable_model_from_legacy_split = common._executable_model_from_legacy_split


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


def _copy_local_file(source_path: str, destination_path: str) -> None:
    _require_coreweave_path("destination_path", destination_path)
    if destination_path.startswith("s3://"):
        _configure_coreweave_s3_env()

    import fsspec  # noqa: PLC0415

    parent = destination_path.rsplit("/", 1)[0]
    fs, plain_parent = _fs_path(parent)
    fs.makedirs(plain_parent, exist_ok=True)
    with open(source_path, "rb") as src, fsspec.open(destination_path, "wb") as dst:
        shutil.copyfileobj(src, dst)


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


def _module_import_check(module_name: str) -> dict[str, Any]:
    started = time.time()
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return {
            "ok": False,
            "module": module_name,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_seconds": time.time() - started,
        }
    return {
        "ok": True,
        "module": module_name,
        "file": getattr(module, "__file__", None),
        "version": getattr(module, "__version__", None),
        "elapsed_seconds": time.time() - started,
    }


def _vllm_import_checks() -> dict[str, Any]:
    return {
        "vllm": _module_import_check("vllm"),
        "vllm._C": _module_import_check("vllm._C"),
        "grugmoe": _module_import_check("vllm.model_executor.models.grugmoe"),
    }


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
        "vllm_dtype": VLLM_DTYPE,
        "levanter_reference_mode": LEVANTER_REFERENCE_MODE,
        "levanter_expert_axis_size": LEVANTER_EXPERT_AXIS_SIZE,
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
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "DEBUG")
    os.environ.setdefault("MODEL_IMPL_TYPE", "vllm")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    cuda_library_path = _configure_cuda_library_path()
    return {
        **cuda_library_path,
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


def _completion_payload(
    env: Any,
    *,
    prompts: list[str],
    data_parallel_rank: int | None = None,
    request_id: str | None = None,
    max_tokens: int = MAX_NEW_TOKENS,
    logprobs: int | None = None,
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
    }
    if logprobs is not None:
        payload["logprobs"] = logprobs
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


def _completion_logprobs_summary(choice: dict[str, Any]) -> dict[str, Any] | None:
    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict):
        return None
    top2_margins: list[float | None] = []
    for top_logprob in logprobs.get("top_logprobs") or []:
        if not isinstance(top_logprob, dict) or len(top_logprob) < 2:
            top2_margins.append(None)
            continue
        values = sorted((float(value) for value in top_logprob.values()), reverse=True)
        top2_margins.append(values[0] - values[1])
    finite_margins = [margin for margin in top2_margins if margin is not None]
    return {
        "tokens": logprobs.get("tokens"),
        "token_logprobs": logprobs.get("token_logprobs"),
        "top_logprobs": logprobs.get("top_logprobs"),
        "top2_logprob_margins": top2_margins,
        "min_top2_logprob_margin": min(finite_margins) if finite_margins else None,
    }


def _completion_choice_summary(choice: dict[str, Any]) -> dict[str, Any]:
    return {
        "text": str(choice.get("text", "")),
        "finish_reason": choice.get("finish_reason"),
        "token_ids": choice.get("token_ids"),
        "logprobs": _completion_logprobs_summary(choice),
    }


def _summarize_completion_payload(
    payload: dict[str, Any],
    *,
    expected_continuation: str | None,
) -> dict[str, Any]:
    choices = payload.get("choices")
    if not isinstance(choices, list):
        raise AssertionError(f"completion payload missing choices list: {payload!r}")
    choice_summaries = [_completion_choice_summary(choice) for choice in choices if isinstance(choice, dict)]
    texts = [choice["text"] for choice in choice_summaries]
    completion_counts = {item: texts.count(item) for item in sorted(set(texts))}
    return {
        "choice_count": len(choice_summaries),
        "texts": texts,
        "completion_counts": completion_counts,
        "all_expected": (
            expected_continuation is not None
            and len(texts) == len(choices)
            and all(text == expected_continuation for text in texts)
        ),
        "choices": choice_summaries,
        "usage": payload.get("usage"),
    }


def _copy_vllm_server_logs(log_dir: str | None, output_dir: str) -> dict[str, Any]:
    if not log_dir:
        return {"copied": False, "reason": "no log directory available"}
    log_root = Path(log_dir)
    if not log_root.is_dir():
        return {"copied": False, "reason": f"log directory does not exist: {log_dir}"}

    artifact_dir = _join_path(output_dir, "vllm-server-logs")
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
    vllm_import_checks = _vllm_import_checks()
    artifact_config_path = _join_path(args.artifact_dir, "config.json")
    _require_file("artifact config.json", artifact_config_path)

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
    log_artifacts: dict[str, Any] = {}
    try:
        with VllmEnvironment(model=model, timeout_seconds=SERVER_TIMEOUT_SECONDS, extra_args=extra_args) as env:
            print("vllm_gpu_server_initialized=True", flush=True)
            print("vllm_gpu_server_url=" + env.server_url, flush=True)
            print("vllm_gpu_model_path=" + staged_artifact.vllm_model_path, flush=True)
            print("vllm_gpu_artifact_staging=" + json.dumps(staged_artifact.staging, sort_keys=True), flush=True)
            print("vllm_gpu_server_log_dir=" + (env.vllm_server.log_dir if env.vllm_server else ""), flush=True)
            # Separately covers the batch-size-1 serving path before the
            # per-rank batch-size-2 requests below.
            single_payload = _completion_payload(
                env,
                prompts=[PROMPT],
                data_parallel_rank=0,
                request_id="grugmoe-single-rank0",
            )
            payloads: list[dict[str, Any]] = []
            rank_request_batches: list[dict[str, Any]] = []
            for data_parallel_rank in range(VLLM_DATA_PARALLEL_SIZE):
                prompts = list(PROMPTS[data_parallel_rank * 2 : data_parallel_rank * 2 + 2])
                payload = _completion_payload(
                    env,
                    prompts=prompts,
                    data_parallel_rank=data_parallel_rank,
                    request_id=f"grugmoe-main-rank{data_parallel_rank}",
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
            log_artifacts = _copy_vllm_server_logs(
                env.vllm_server.log_dir if env.vllm_server else None,
                args.output_dir,
            )
            model_id = env.model_id
    except Exception as exc:
        log_artifacts = _copy_vllm_server_logs(_latest_vllm_server_log_dir(since=started), args.output_dir)
        failure_result = {
            "phase": "vllm",
            "checkpoint_path": args.checkpoint_path,
            "tokenizer_path": args.tokenizer_path,
            "artifact_dir": args.artifact_dir,
            "result_path": args.result_path,
            "passed": False,
            "failure": _exception_summary(exc),
            "served_model_name": SERVED_MODEL_NAME,
            "vllm_model_path": staged_artifact.vllm_model_path,
            "artifact_staging": staged_artifact.staging,
            "vllm_engine_kwargs": model.engine_kwargs,
            "vllm_args": extra_args,
            "vllm_attention_backend_env_var": VLLM_ATTENTION_BACKEND_ENV,
            "vllm_dtype": VLLM_DTYPE,
            "vllm_tensor_parallel_size": VLLM_TENSOR_PARALLEL_SIZE,
            "vllm_data_parallel_size": VLLM_DATA_PARALLEL_SIZE,
            "vllm_expert_parallel_size": VLLM_EXPERT_PARALLEL_SIZE,
            "vllm_attention_backend": VLLM_ATTENTION_BACKEND,
            "vllm_max_num_seqs": VLLM_MAX_NUM_SEQS,
            "expected_gpu_count": EXPECTED_GPU_COUNT,
            "coreweave_s3": s3_env,
            "cuda_library_path": cuda_library_path,
            "torch_runtime": torch_runtime,
            "vllm_import_checks": vllm_import_checks,
            "vllm_log_artifacts": log_artifacts,
            "runtime": _runtime_snapshot(include_grugmoe_spec=True, include_torch_cuda=True),
            "elapsed_seconds": time.time() - started,
        }
        _write_json(args.result_path, failure_result)
        print("grugmoe_gpu_real_checkpoint_vllm_result=" + json.dumps(failure_result, sort_keys=True), flush=True)
        raise
    single_choices = single_payload.get("choices")
    if not isinstance(single_choices, list) or len(single_choices) != 1:
        raise AssertionError(f"expected exactly one single-prompt completion choice, got {single_payload!r}")
    single_completion = str(single_choices[0].get("text", ""))
    completions: list[str] = []
    for payload in payloads:
        choices = payload.get("choices")
        if not isinstance(choices, list) or len(choices) != 2:
            raise AssertionError(f"expected exactly two completion choices, got {payload!r}")
        for choice in choices:
            completion = str(choice.get("text", ""))
            completions.append(completion)
    completion = completions[0] if completions else ""
    if len(completions) != PROMPT_BATCH_SIZE:
        raise AssertionError(f"expected {PROMPT_BATCH_SIZE} completions, got {len(completions)}")
    completion_counts = {item: completions.count(item) for item in sorted(set(completions))}
    repeated_prompt_identical = len(completion_counts) == 1
    single_prompt_choice_summary = _summarize_completion_payload(
        single_payload,
        expected_continuation=EXPECTED_CONTINUATION,
    )["choices"][0]
    main_choice_summaries = [
        choice_summary
        for payload in payloads
        for choice_summary in _summarize_completion_payload(
            payload,
            expected_continuation=EXPECTED_CONTINUATION,
        )["choices"]
    ]
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
        "single_prompt_choice_summary": single_prompt_choice_summary,
        "completions": completions,
        "completion_counts": completion_counts,
        "repeated_prompt_identical": repeated_prompt_identical,
        "expected_continuation": EXPECTED_CONTINUATION,
        "passed": (
            single_completion == EXPECTED_CONTINUATION
            and all(item == EXPECTED_CONTINUATION for item in completions)
            and repeated_prompt_identical
        ),
        "served_model_name": SERVED_MODEL_NAME,
        "vllm_model_id": model_id,
        "vllm_model_path": staged_artifact.vllm_model_path,
        "artifact_staging": staged_artifact.staging,
        "vllm_engine_kwargs": model.engine_kwargs,
        "vllm_args": extra_args,
        "vllm_attention_backend_env_var": VLLM_ATTENTION_BACKEND_ENV,
        "vllm_dtype": VLLM_DTYPE,
        "vllm_tensor_parallel_size": VLLM_TENSOR_PARALLEL_SIZE,
        "vllm_data_parallel_size": VLLM_DATA_PARALLEL_SIZE,
        "vllm_expert_parallel_size": VLLM_EXPERT_PARALLEL_SIZE,
        "vllm_attention_backend": VLLM_ATTENTION_BACKEND,
        "vllm_max_num_seqs": VLLM_MAX_NUM_SEQS,
        "rank_request_batches": rank_request_batches,
        "requested_data_parallel_ranks": [batch["data_parallel_rank"] for batch in rank_request_batches],
        "main_choice_summaries": main_choice_summaries,
        "expected_gpu_count": EXPECTED_GPU_COUNT,
        "coreweave_s3": s3_env,
        "cuda_library_path": cuda_library_path,
        "torch_runtime": torch_runtime,
        "vllm_import_checks": vllm_import_checks,
        "vllm_log_artifacts": log_artifacts,
        "vllm_logs_tail": logs_tail,
        "runtime": _runtime_snapshot(include_grugmoe_spec=True, include_torch_cuda=True),
        "elapsed_seconds": time.time() - started,
    }
    _write_json(args.result_path, result)
    print("grugmoe_gpu_real_checkpoint_vllm_result=" + json.dumps(result, sort_keys=True), flush=True)
    if result["passed"] is not True:
        raise AssertionError(
            f"GPU vLLM single={single_completion!r}, completion_counts={completion_counts!r}, "
            f"expected {EXPECTED_CONTINUATION!r}"
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
    prompt_ids, tokenization = _prompt_token_ids(tokenizer, PROMPT)
    model_cfg = _real_checkpoint_model_config()
    mesh = compact_grug_mesh(expert_axis_size=LEVANTER_EXPERT_AXIS_SIZE, model_axis_size=1)
    mesh_runtime = _mesh_snapshot(mesh)
    decode_batch_size = _mesh_batch_axis_size(mesh)
    pad_token_id = int(getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None) or 0)
    started = time.time()
    with ExitStack() as stack:
        stack.enter_context(set_mesh(mesh))
        stack.enter_context(haliax.axis_mapping({}))
        loaded_model = _load_legacy_split_expert_checkpoint(args.checkpoint_path, model_cfg)
        executable_model = _executable_model_from_legacy_split(
            loaded_model,
            capacity_factor=LEVANTER_MOE_CAPACITY_FACTOR,
        )
        model, reference_policy = _apply_levanter_reference_mode(executable_model)
        decode_results: list[dict[str, Any]] = []
        completions: list[str] = []
        remaining = PROMPT_BATCH_SIZE
        while remaining > 0:
            decode_result = common._greedy_decode(
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
        "levanter_expert_axis_size": LEVANTER_EXPERT_AXIS_SIZE,
        "levanter_moe_capacity_factor": LEVANTER_MOE_CAPACITY_FACTOR,
        "levanter_decode_use_active_prefix": LEVANTER_DECODE_USE_ACTIVE_PREFIX,
        "decode_results": decode_results,
        "levanter_reference_mode": LEVANTER_REFERENCE_MODE,
        "levanter_reference_policy": reference_policy,
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
