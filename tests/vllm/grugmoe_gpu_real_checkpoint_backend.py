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
import importlib
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
from collections import Counter
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
VLLM_DTYPE_ENV = "MARIN_GRUGMOE_VLLM_DTYPE"
VLLM_DEFAULT_DTYPE = "bfloat16"
VLLM_DTYPE_CHOICES = ("bfloat16", "float32")
VLLM_MOE_COMPUTE_ENV = "MARIN_GRUGMOE_VLLM_MOE_COMPUTE"
VLLM_DEFAULT_MOE_COMPUTE = "fp32_accumulation"
VLLM_MOE_COMPUTE_CHOICES = ("model_dtype", "fp32_accumulation")
VLLM_GRUGMOE_MOE_COMPUTE_ENV = "VLLM_GRUGMOE_MOE_COMPUTE"
VLLM_GRUGMOE_ROUTE_DIAGNOSTICS_ENV = "VLLM_GRUGMOE_ROUTE_DIAGNOSTICS"
VLLM_ROUTE_DIAGNOSTICS_ENV = "MARIN_GRUGMOE_VLLM_ROUTE_DIAGNOSTICS"
VLLM_DEFAULT_ROUTE_DIAGNOSTICS = True
WORKER_EXTENSION_MODULE = "grugmoe_gpu_real_checkpoint_backend"
WORKER_EXTENSION_CLASS = "GrugMoeDiagnosticsWorkerExtension"
WORKER_EXTENSION_CLS = f"{WORKER_EXTENSION_MODULE}.{WORKER_EXTENSION_CLASS}"
VLLM_ATTENTION_BACKEND_ENV = "MARIN_GRUGMOE_VLLM_ATTENTION_BACKEND"
LEVANTER_REFERENCE_MODE_ENV = "MARIN_GRUGMOE_LEVANTER_REFERENCE_MODE"
LEVANTER_DEFAULT_REFERENCE_MODE = "current"
LEVANTER_REFERENCE_MODE_CHOICES = ("current", "bf16_compute")
LEVANTER_BF16_POLICY = "params=float32,compute=bfloat16,output=bfloat16"
LEVANTER_EXPERT_AXIS_SIZE_ENV = "MARIN_GRUGMOE_LEVANTER_EXPERT_AXIS_SIZE"
LEVANTER_DEFAULT_EXPERT_AXIS_SIZE = EXPECTED_GPU_COUNT
LEVANTER_PROMPT_SWEEP_ENV = "MARIN_GRUGMOE_LEVANTER_PROMPT_SWEEP"
LEVANTER_DEFAULT_PROMPT_SWEEP = False
LEVANTER_ROUTE_DIAGNOSTICS_ENV = "MARIN_GRUGMOE_LEVANTER_ROUTE_DIAGNOSTICS"
LEVANTER_DEFAULT_ROUTE_DIAGNOSTICS = True
FORCED_PREFIX_DIAGNOSTICS_ENV = "MARIN_GRUGMOE_FORCED_PREFIX_DIAGNOSTICS"
FORCED_PREFIX_DIAGNOSTICS_DEFAULT = False
TRAINING_LOSS_DIAGNOSTICS_ENV = "MARIN_GRUGMOE_TRAINING_LOSS_DIAGNOSTICS"
TRAINING_LOSS_DIAGNOSTICS_DEFAULT = False
PRIMARY_GATE_ENV = "MARIN_GRUGMOE_PRIMARY_GATE"
PRIMARY_GATE_DEFAULT = "stress"
PRIMARY_GATE_CHOICES = ("stress", "stable_prompt")
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
PROMPT = common.PROMPT
PROMPT_BATCH_SIZE = 16
PROMPTS = tuple(PROMPT for _ in range(PROMPT_BATCH_SIZE))
EXPECTED_CONTINUATION = common.EXPECTED_CONTINUATION
DIAGNOSTIC_COMPARE_RANKS = (4, 0)
DIAGNOSTIC_REPEATED_ATTEMPTS = 3
DIAGNOSTIC_LOGPROBS = 8
DIAGNOSTIC_ROUTE_STEPS = 1
DIAGNOSTIC_PROMPT_SWEEP_MAX_NEW_TOKENS = 4
DIAGNOSTIC_STABLE_PROMPT_MIN_TOP2_MARGIN = 0.25
DIAGNOSTIC_PROMPT_SWEEP_PROMPTS = (
    "Answer with digits only. No words. No punctuation. What is two plus two?",
    "Answer with digits only. No words. No punctuation. What is three plus three?",
    "Answer with one word only. What color is the sky on a clear day?",
)
DIAGNOSTIC_NON_REPEATED_PROMPTS = DIAGNOSTIC_PROMPT_SWEEP_PROMPTS[:2]
FORCED_PREFIX_DIAGNOSTIC_PROMPTS = (
    {"name": "original", "prompt": PROMPT},
    {"name": "prefix_the", "prompt": PROMPT + " The"},
    {"name": "prefix_the_ultimate", "prompt": PROMPT + " The Ultimate"},
    {"name": "prefix_the_universe", "prompt": PROMPT + " The Universe"},
)
TRAINING_LOSS_DIAGNOSTIC_EXPERT_AXIS_SIZES = (1, EXPECTED_GPU_COUNT)
TRAINING_LOSS_DIAGNOSTIC_REFERENCE_MODES = ("current", "bf16_compute")
TRAINING_LOSS_DIAGNOSTIC_PROMPTS = (PROMPT + " The Ultimate",)
MAX_MODEL_LEN = common.MAX_MODEL_LEN
MAX_NUM_BATCHED_TOKENS = 1024
MAX_NEW_TOKENS = common.MAX_NEW_TOKENS
LEVANTER_PROMPT_ADD_SPECIAL_TOKENS = common.LEVANTER_PROMPT_ADD_SPECIAL_TOKENS
EXPECTED_PROMPT_TOKEN_COUNT = common.EXPECTED_PROMPT_TOKEN_COUNT
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


def _resolve_vllm_dtype() -> str:
    value = os.environ.get(VLLM_DTYPE_ENV, VLLM_DEFAULT_DTYPE).strip().lower()
    if value not in VLLM_DTYPE_CHOICES:
        raise ValueError(
            f"{VLLM_DTYPE_ENV}={value!r} is not supported for this validation; "
            f"expected one of {VLLM_DTYPE_CHOICES!r}"
        )
    return value


def _resolve_vllm_moe_compute() -> str:
    value = os.environ.get(VLLM_MOE_COMPUTE_ENV, VLLM_DEFAULT_MOE_COMPUTE).strip().lower()
    if value not in VLLM_MOE_COMPUTE_CHOICES:
        raise ValueError(
            f"{VLLM_MOE_COMPUTE_ENV}={value!r} is not supported for this validation; "
            f"expected one of {VLLM_MOE_COMPUTE_CHOICES!r}"
        )
    return value


def _resolve_bool_env(name: str, *, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    value = raw_value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name}={raw_value!r} must be a boolean value")


def _resolve_levanter_reference_mode() -> str:
    value = os.environ.get(LEVANTER_REFERENCE_MODE_ENV, LEVANTER_DEFAULT_REFERENCE_MODE).strip().lower()
    if value not in LEVANTER_REFERENCE_MODE_CHOICES:
        raise ValueError(
            f"{LEVANTER_REFERENCE_MODE_ENV}={value!r} is not supported for this validation; "
            f"expected one of {LEVANTER_REFERENCE_MODE_CHOICES!r}"
        )
    return value


def _resolve_levanter_expert_axis_size() -> int:
    raw_value = os.environ.get(LEVANTER_EXPERT_AXIS_SIZE_ENV, str(LEVANTER_DEFAULT_EXPERT_AXIS_SIZE)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{LEVANTER_EXPERT_AXIS_SIZE_ENV}={raw_value!r} must be an integer") from exc
    if value <= 0 or value > EXPECTED_GPU_COUNT or EXPECTED_GPU_COUNT % value != 0:
        raise ValueError(
            f"{LEVANTER_EXPERT_AXIS_SIZE_ENV}={raw_value!r} must be a positive divisor of {EXPECTED_GPU_COUNT}"
        )
    return value


def _resolve_primary_gate() -> str:
    value = os.environ.get(PRIMARY_GATE_ENV, PRIMARY_GATE_DEFAULT).strip().lower()
    if value not in PRIMARY_GATE_CHOICES:
        raise ValueError(
            f"{PRIMARY_GATE_ENV}={value!r} is not supported for this validation; "
            f"expected one of {PRIMARY_GATE_CHOICES!r}"
        )
    return value


VLLM_ATTENTION_BACKEND = _resolve_vllm_attention_backend()
VLLM_DTYPE = _resolve_vllm_dtype()
VLLM_MOE_COMPUTE = _resolve_vllm_moe_compute()
VLLM_ROUTE_DIAGNOSTICS = _resolve_bool_env(
    VLLM_ROUTE_DIAGNOSTICS_ENV,
    default=VLLM_DEFAULT_ROUTE_DIAGNOSTICS,
)
LEVANTER_REFERENCE_MODE = _resolve_levanter_reference_mode()
LEVANTER_EXPERT_AXIS_SIZE = _resolve_levanter_expert_axis_size()
LEVANTER_PROMPT_SWEEP = _resolve_bool_env(
    LEVANTER_PROMPT_SWEEP_ENV,
    default=LEVANTER_DEFAULT_PROMPT_SWEEP,
)
LEVANTER_ROUTE_DIAGNOSTICS = _resolve_bool_env(
    LEVANTER_ROUTE_DIAGNOSTICS_ENV,
    default=LEVANTER_DEFAULT_ROUTE_DIAGNOSTICS,
)
FORCED_PREFIX_DIAGNOSTICS = _resolve_bool_env(
    FORCED_PREFIX_DIAGNOSTICS_ENV,
    default=FORCED_PREFIX_DIAGNOSTICS_DEFAULT,
)
TRAINING_LOSS_DIAGNOSTICS = _resolve_bool_env(
    TRAINING_LOSS_DIAGNOSTICS_ENV,
    default=TRAINING_LOSS_DIAGNOSTICS_DEFAULT,
)
PRIMARY_GATE = _resolve_primary_gate()


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
        "marin_worker_extension": _module_import_check(WORKER_EXTENSION_MODULE),
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
        "vllm_moe_compute": VLLM_MOE_COMPUTE,
        "vllm_route_diagnostics": VLLM_ROUTE_DIAGNOSTICS,
        "levanter_reference_mode": LEVANTER_REFERENCE_MODE,
        "levanter_expert_axis_size": LEVANTER_EXPERT_AXIS_SIZE,
        "levanter_prompt_sweep": LEVANTER_PROMPT_SWEEP,
        "levanter_route_diagnostics": LEVANTER_ROUTE_DIAGNOSTICS,
        "forced_prefix_diagnostics": FORCED_PREFIX_DIAGNOSTICS,
        "training_loss_diagnostics": TRAINING_LOSS_DIAGNOSTICS,
        "primary_gate": PRIMARY_GATE,
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


def _ensure_worker_extension_import_path() -> str:
    extension_dir = str(Path(__file__).resolve().parent)
    if extension_dir not in sys.path:
        sys.path.insert(0, extension_dir)
    _prepend_env_path("PYTHONPATH", [extension_dir])
    return extension_dir


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
    os.environ[VLLM_GRUGMOE_MOE_COMPUTE_ENV] = VLLM_MOE_COMPUTE
    os.environ[VLLM_GRUGMOE_ROUTE_DIAGNOSTICS_ENV] = "1" if VLLM_ROUTE_DIAGNOSTICS else "0"
    worker_extension_path = _ensure_worker_extension_import_path()
    cuda_library_path = _configure_cuda_library_path()
    return {
        **cuda_library_path,
        "vllm_logging_level": os.environ.get("VLLM_LOGGING_LEVEL"),
        "vllm_moe_compute_env_var": VLLM_MOE_COMPUTE_ENV,
        "vllm_moe_compute": VLLM_MOE_COMPUTE,
        "vllm_grugmoe_moe_compute_env_var": VLLM_GRUGMOE_MOE_COMPUTE_ENV,
        "vllm_grugmoe_moe_compute": os.environ.get(VLLM_GRUGMOE_MOE_COMPUTE_ENV),
        "vllm_route_diagnostics_env_var": VLLM_ROUTE_DIAGNOSTICS_ENV,
        "vllm_route_diagnostics": VLLM_ROUTE_DIAGNOSTICS,
        "worker_extension_module": WORKER_EXTENSION_MODULE,
        "worker_extension_cls": WORKER_EXTENSION_CLS,
        "worker_extension_path": worker_extension_path,
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
    del mlp
    from vllm.model_executor.models.grugmoe import get_grug_moe_runtime_info  # noqa: PLC0415

    runtime_info = get_grug_moe_runtime_info(getattr(worker, "vllm_config", None), model)
    return {
        **runtime_info,
        "found": True,
        "worker_rank": worker_rank,
        "local_rank": local_rank,
        "module_name": module_name,
    }


class GrugMoeDiagnosticsWorkerExtension:
    """Test-only extension called through vLLM dev-mode collective_rpc."""

    def grugmoe_ep_state(self) -> dict[str, Any]:
        return _grug_moe_ep_state_from_worker(self)

    def grugmoe_route_diagnostics(self) -> dict[str, Any]:
        model_runner = getattr(self, "model_runner", None)
        model = _unwrap_vllm_model(getattr(model_runner, "model", None))
        from vllm.model_executor.models.grugmoe import get_grug_moe_route_diagnostics  # noqa: PLC0415

        return {
            **get_grug_moe_route_diagnostics(model),
            "worker_rank": int(getattr(self, "rank", -1)),
            "local_rank": int(getattr(self, "local_rank", -1)),
        }


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


def _collect_grug_moe_route_diagnostics(env: Any) -> list[dict[str, Any]]:
    payload = _collective_rpc_payload(env, "grugmoe_route_diagnostics")
    results = payload.get("results")
    if not isinstance(results, list):
        raise AssertionError(f"collective_rpc missing results list: {payload!r}")
    diagnostics: list[dict[str, Any]] = []
    for result in results:
        if not isinstance(result, dict):
            raise AssertionError(f"collective_rpc result is not a dict: {result!r}")
        diagnostics.append(result)
    return diagnostics


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
        if state.get("num_experts") != num_experts:
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


def _decode_routed_experts(value: str | None) -> list[Any] | None:
    if value is None:
        return None

    import numpy as np  # noqa: PLC0415

    routed = np.load(io.BytesIO(base64.b64decode(value)))
    return routed.astype("int64").tolist()


def _nested_list_shape(value: Any) -> list[int]:
    shape: list[int] = []
    current = value
    while isinstance(current, list):
        shape.append(len(current))
        if not current:
            break
        current = current[0]
    return shape


def _routed_experts_digest(routed_experts: list[Any] | None) -> str | None:
    if routed_experts is None:
        return None
    payload = json.dumps(routed_experts, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


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


def _completion_choice_summary(
    choice: dict[str, Any],
    *,
    worker_ep_states: list[dict[str, Any]],
) -> dict[str, Any]:
    routed_experts = _decode_routed_experts(choice.get("routed_experts"))
    return {
        "text": str(choice.get("text", "")),
        "finish_reason": choice.get("finish_reason"),
        "token_ids": choice.get("token_ids"),
        "logprobs": _completion_logprobs_summary(choice),
        "routed_experts_shape": _nested_list_shape(routed_experts),
        "routed_experts_digest": _routed_experts_digest(routed_experts),
        "routed_experts_tail": routed_experts[-4:] if routed_experts else None,
        "routed_owner_ranks": _owners_for_worker_expert_placement(
            routed_experts,
            worker_ep_states=worker_ep_states,
        ),
    }


def _summarize_completion_payload(
    payload: dict[str, Any],
    *,
    worker_ep_states: list[dict[str, Any]],
    expected_continuation: str | None,
) -> dict[str, Any]:
    choices = payload.get("choices")
    if not isinstance(choices, list):
        raise AssertionError(f"completion payload missing choices list: {payload!r}")
    choice_summaries = [
        _completion_choice_summary(choice, worker_ep_states=worker_ep_states)
        for choice in choices
        if isinstance(choice, dict)
    ]
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


def _diagnostic_prompt_summary(prompts: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "index": index,
            "sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "length": len(prompt),
            "is_main_prompt": prompt == PROMPT,
        }
        for index, prompt in enumerate(prompts)
    ]


def _vllm_completion_diagnostic_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for rank in DIAGNOSTIC_COMPARE_RANKS:
        specs.append(
            {
                "name": f"rank{rank}-first-token-repeated-bs2",
                "data_parallel_rank": rank,
                "prompts": [PROMPT, PROMPT],
                "prompt_kind": "repeated",
                "attempt": 0,
                "expected_continuation": None,
                "max_tokens": 1,
                "logprobs": DIAGNOSTIC_LOGPROBS,
                "collect_route_diagnostics": True,
            }
        )
        specs.append(
            {
                "name": f"rank{rank}-repeated-bs1",
                "data_parallel_rank": rank,
                "prompts": [PROMPT],
                "prompt_kind": "repeated",
                "attempt": 0,
                "expected_continuation": EXPECTED_CONTINUATION,
                "max_tokens": MAX_NEW_TOKENS,
                "logprobs": DIAGNOSTIC_LOGPROBS,
                "collect_route_diagnostics": False,
            }
        )
        for attempt in range(DIAGNOSTIC_REPEATED_ATTEMPTS):
            specs.append(
                {
                    "name": f"rank{rank}-repeated-bs2-attempt{attempt}",
                    "data_parallel_rank": rank,
                    "prompts": [PROMPT, PROMPT],
                    "prompt_kind": "repeated",
                    "attempt": attempt,
                    "expected_continuation": EXPECTED_CONTINUATION,
                    "max_tokens": MAX_NEW_TOKENS,
                    "logprobs": DIAGNOSTIC_LOGPROBS,
                    "collect_route_diagnostics": False,
                }
            )
        specs.append(
            {
                "name": f"rank{rank}-non-repeated-bs2",
                "data_parallel_rank": rank,
                "prompts": list(DIAGNOSTIC_NON_REPEATED_PROMPTS),
                "prompt_kind": "non_repeated",
                "attempt": 0,
                "expected_continuation": None,
                "max_tokens": MAX_NEW_TOKENS,
                "logprobs": DIAGNOSTIC_LOGPROBS,
                "collect_route_diagnostics": False,
            }
        )
    return specs


def _run_vllm_completion_diagnostics(
    env: Any,
    *,
    worker_ep_states: list[dict[str, Any]],
) -> dict[str, Any]:
    requests_summary: list[dict[str, Any]] = []
    for spec in _vllm_completion_diagnostic_specs():
        name = str(spec["name"])
        data_parallel_rank = int(spec["data_parallel_rank"])
        prompts = list(spec["prompts"])
        max_tokens = int(spec["max_tokens"])
        logprobs = int(spec["logprobs"]) if spec["logprobs"] is not None else None
        request_summary: dict[str, Any] = {
            "name": name,
            "data_parallel_rank": data_parallel_rank,
            "batch_size": len(prompts),
            "prompt_kind": spec["prompt_kind"],
            "attempt": spec["attempt"],
            "max_tokens": max_tokens,
            "logprobs": logprobs,
            "prompt_summary": _diagnostic_prompt_summary(prompts),
        }
        try:
            payload = _completion_payload(
                env,
                prompts=prompts,
                data_parallel_rank=data_parallel_rank,
                request_id=f"grugmoe-diagnostic-{name}",
                max_tokens=max_tokens,
                logprobs=logprobs,
            )
            request_summary.update(
                _summarize_completion_payload(
                    payload,
                    worker_ep_states=worker_ep_states,
                    expected_continuation=spec["expected_continuation"],
                )
            )
            if spec["collect_route_diagnostics"]:
                request_summary["route_diagnostics"] = _collect_grug_moe_route_diagnostics(env)
        except Exception as exc:  # Keep the primary vLLM correctness result visible.
            request_summary["failure"] = _exception_summary(exc)
        requests_summary.append(request_summary)

    return {
        "compare_ranks": list(DIAGNOSTIC_COMPARE_RANKS),
        "repeated_attempts": DIAGNOSTIC_REPEATED_ATTEMPTS,
        "requests": requests_summary,
    }


def _vllm_choice_min_top2_margin(choice: dict[str, Any]) -> float | None:
    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict):
        return None
    margin = logprobs.get("min_top2_logprob_margin")
    return float(margin) if margin is not None else None


def _run_vllm_prompt_sweep(
    env: Any,
    *,
    worker_ep_states: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for prompt_index, prompt in enumerate(DIAGNOSTIC_PROMPT_SWEEP_PROMPTS):
        rank_results: list[dict[str, Any]] = []
        completions: list[str] = []
        for data_parallel_rank in range(VLLM_DATA_PARALLEL_SIZE):
            prompts = [prompt, prompt]
            request_summary: dict[str, Any] = {
                "data_parallel_rank": data_parallel_rank,
                "batch_size": len(prompts),
                "max_tokens": DIAGNOSTIC_PROMPT_SWEEP_MAX_NEW_TOKENS,
                "logprobs": DIAGNOSTIC_LOGPROBS,
                "prompt_summary": _diagnostic_prompt_summary(prompts),
            }
            try:
                payload = _completion_payload(
                    env,
                    prompts=prompts,
                    data_parallel_rank=data_parallel_rank,
                    request_id=f"grugmoe-prompt-sweep-{prompt_index}-rank{data_parallel_rank}",
                    max_tokens=DIAGNOSTIC_PROMPT_SWEEP_MAX_NEW_TOKENS,
                    logprobs=DIAGNOSTIC_LOGPROBS,
                )
                request_summary.update(
                    _summarize_completion_payload(
                        payload,
                        worker_ep_states=worker_ep_states,
                        expected_continuation=None,
                    )
                )
                completions.extend(str(text) for text in request_summary.get("texts", []))
            except Exception as exc:  # Keep the primary vLLM correctness result visible.
                request_summary["failure"] = _exception_summary(exc)
            rank_results.append(request_summary)
        margins = [
            margin
            for rank_result in rank_results
            for choice in rank_result.get("choices", [])
            if (margin := _vllm_choice_min_top2_margin(choice)) is not None
        ]
        results.append(
            {
                "prompt_index": prompt_index,
                "prompt": prompt,
                "max_new_tokens": DIAGNOSTIC_PROMPT_SWEEP_MAX_NEW_TOKENS,
                "completion_counts": _value_counts(completions),
                "all_completions_identical": len(set(completions)) == 1 if completions else False,
                "min_top2_logprob_margin": min(margins) if margins else None,
                "rank_results": rank_results,
            }
        )
    return results


def _run_vllm_forced_prefix_diagnostics(
    env: Any,
    *,
    worker_ep_states: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for prompt_spec in FORCED_PREFIX_DIAGNOSTIC_PROMPTS:
        name = str(prompt_spec["name"])
        prompt = str(prompt_spec["prompt"])
        rank_results: list[dict[str, Any]] = []
        for data_parallel_rank in DIAGNOSTIC_COMPARE_RANKS:
            request_summary: dict[str, Any] = {
                "data_parallel_rank": data_parallel_rank,
                "batch_size": 1,
                "max_tokens": 1,
                "logprobs": DIAGNOSTIC_LOGPROBS,
                "prompt_summary": _diagnostic_prompt_summary([prompt]),
            }
            try:
                payload = _completion_payload(
                    env,
                    prompts=[prompt],
                    data_parallel_rank=data_parallel_rank,
                    request_id=f"grugmoe-forced-prefix-{name}-rank{data_parallel_rank}",
                    max_tokens=1,
                    logprobs=DIAGNOSTIC_LOGPROBS,
                )
                request_summary.update(
                    _summarize_completion_payload(
                        payload,
                        worker_ep_states=worker_ep_states,
                        expected_continuation=None,
                    )
                )
                if VLLM_ROUTE_DIAGNOSTICS:
                    request_summary["route_diagnostics"] = _collect_grug_moe_route_diagnostics(env)
            except Exception as exc:  # Keep the primary vLLM correctness result visible.
                request_summary["failure"] = _exception_summary(exc)
            rank_results.append(request_summary)
        results.append(
            {
                "name": name,
                "prompt": prompt,
                "max_new_tokens": 1,
                "rank_results": rank_results,
            }
        )
    return results


def _owners_for_worker_expert_placement(
    routed_experts: list[Any] | None,
    *,
    worker_ep_states: list[dict[str, Any]],
) -> list[int]:
    if routed_experts is None:
        return []
    owner_by_expert: dict[int, int] = {}
    for state in worker_ep_states:
        ep_rank = int(state["ep_rank"])
        for expert_id in state.get("local_expert_ids", []):
            owner_by_expert[int(expert_id)] = ep_rank

    owners: set[int] = set()

    def visit(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        expert_id = int(value)
        owner = owner_by_expert.get(expert_id)
        if owner is not None:
            owners.add(owner)

    visit(routed_experts)
    return sorted(owners)


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


def _value_counts(values: list[Any]) -> dict[str, int]:
    return {str(value): count for value, count in sorted(Counter(values).items(), key=lambda item: str(item[0]))}


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


def _jax_dtype_name(value: Any) -> str:
    dtype = getattr(value, "dtype", None)
    return str(dtype) if dtype is not None else "unavailable"


def _jax_int_rows(value: Any, *, max_rows: int = 4, max_cols: int = 8) -> list[list[int]]:
    import jax  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    array = np.asarray(jax.device_get(value))[:max_rows, :max_cols]
    return [[int(item) for item in row] for row in array.tolist()]


def _jax_float_rows(value: Any, *, max_rows: int = 4, max_cols: int = 8) -> list[list[float]]:
    import jax  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    array = np.asarray(jax.device_get(value))[:max_rows, :max_cols].astype(np.float32)
    return [[float(item) for item in row] for row in array.tolist()]


def _jax_float_vector(value: Any, *, max_items: int = 8) -> list[float]:
    import jax  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    array = np.asarray(jax.device_get(value))[:max_items].astype(np.float32)
    return [float(item) for item in array.tolist()]


def _jax_int_vector(value: Any, *, max_items: int = 8) -> list[int]:
    import jax  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    array = np.asarray(jax.device_get(value))[:max_items]
    return [int(item) for item in array.tolist()]


def _apply_levanter_reference_mode(model: Any, mode: str) -> tuple[Any, dict[str, Any]]:
    if mode == "current":
        return model, {
            "mode": mode,
            "applies_mixed_precision_policy": False,
            "policy": None,
        }
    if mode == "bf16_compute":
        import jmp  # noqa: PLC0415

        policy = jmp.get_policy(LEVANTER_BF16_POLICY)
        return policy.cast_to_compute(model), {
            "mode": mode,
            "applies_mixed_precision_policy": True,
            "policy": LEVANTER_BF16_POLICY,
            "param_dtype": str(policy.param_dtype),
            "compute_dtype": str(policy.compute_dtype),
            "output_dtype": str(policy.output_dtype),
        }
    raise ValueError(f"unknown Levanter reference mode: {mode!r}")


def _jax_model_dtype_summary(model: Any) -> dict[str, Any]:
    first_block = model.blocks[0]
    return {
        "token_embed": _jax_dtype_name(model.token_embed),
        "output_proj": _jax_dtype_name(model.output_proj),
        "router_weight": _jax_dtype_name(first_block.mlp.router),
        "router_bias": _jax_dtype_name(first_block.mlp.router_bias),
        "expert_w_gate_up": _jax_dtype_name(first_block.mlp.expert_mlp.w_gate_up),
        "expert_w_down": _jax_dtype_name(first_block.mlp.expert_mlp.w_down),
        "shared_w_gate": _jax_dtype_name(first_block.shared.w_gate) if first_block.shared is not None else None,
    }


def _jax_mlp_route_summary(
    mlp: Any,
    x: Any,
    *,
    layer_index: int,
    focus_token_indices: Any | None = None,
) -> dict[str, Any]:
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    from jax.sharding import PartitionSpec as P  # noqa: PLC0415
    from jax.sharding import reshard  # noqa: PLC0415

    token_count = int(x.shape[0] * x.shape[1])
    x_flat = x.reshape((token_count, x.shape[-1]))
    router_logits = jnp.einsum("td,de->te", x_flat, reshard(mlp.router, P(None, None))).astype(jnp.float32)
    router_logits = reshard(router_logits, P(None, None))
    biased_logits = router_logits + jax.lax.stop_gradient(reshard(mlp.router_bias, P(None)))
    top_count = min(DIAGNOSTIC_LOGPROBS, int(router_logits.shape[-1]))
    unbiased_top_logits, unbiased_top_ids = jax.lax.top_k(router_logits, top_count)
    biased_top_logits, biased_top_ids = jax.lax.top_k(biased_logits, top_count)
    selected = biased_top_ids[:, : mlp.cfg.num_experts_per_token]
    unbiased_topk = jnp.take_along_axis(router_logits, selected, axis=-1)
    combine_weights = jax.nn.sigmoid(unbiased_topk).astype(x.dtype)
    summary = {
        "layer_index": layer_index,
        "token_count": token_count,
        "hidden_dtype": _jax_dtype_name(x_flat),
        "expert_hidden_dtype": _jax_dtype_name(x_flat),
        "router_weight_dtype": _jax_dtype_name(mlp.router),
        "router_bias_dtype": _jax_dtype_name(mlp.router_bias),
        "router_logits_dtype": _jax_dtype_name(router_logits),
        "combine_weights_dtype": _jax_dtype_name(combine_weights),
        "w_gate_up_dtype": _jax_dtype_name(mlp.expert_mlp.w_gate_up),
        "w_down_dtype": _jax_dtype_name(mlp.expert_mlp.w_down),
        "selected_experts": _jax_int_rows(
            selected,
            max_cols=mlp.cfg.num_experts_per_token,
        ),
        "combine_weights": _jax_float_rows(
            combine_weights,
            max_cols=mlp.cfg.num_experts_per_token,
        ),
        "unbiased_top_expert_ids": _jax_int_rows(unbiased_top_ids, max_cols=top_count),
        "unbiased_top_logits": _jax_float_rows(unbiased_top_logits, max_cols=top_count),
        "biased_top_expert_ids": _jax_int_rows(biased_top_ids, max_cols=top_count),
        "biased_top_logits": _jax_float_rows(biased_top_logits, max_cols=top_count),
        "unbiased_top2_margin": _jax_float_vector(
            unbiased_top_logits[:, 0] - unbiased_top_logits[:, 1],
            max_items=4,
        ),
        "biased_top2_margin": _jax_float_vector(
            biased_top_logits[:, 0] - biased_top_logits[:, 1],
            max_items=4,
        ),
    }
    if focus_token_indices is not None:
        focus_token_indices = jnp.asarray(focus_token_indices, dtype=jnp.int32)
        focus_selected = jnp.take(selected, focus_token_indices, axis=0)
        focus_combine_weights = jnp.take(combine_weights, focus_token_indices, axis=0)
        focus_unbiased_top_ids = jnp.take(unbiased_top_ids, focus_token_indices, axis=0)
        focus_unbiased_top_logits = jnp.take(unbiased_top_logits, focus_token_indices, axis=0)
        focus_biased_top_ids = jnp.take(biased_top_ids, focus_token_indices, axis=0)
        focus_biased_top_logits = jnp.take(biased_top_logits, focus_token_indices, axis=0)
        summary.update(
            {
                "focus_token_flat_indices": _jax_int_vector(
                    focus_token_indices,
                    max_items=int(focus_token_indices.shape[0]),
                ),
                "focus_selected_experts": _jax_int_rows(
                    focus_selected,
                    max_rows=int(focus_selected.shape[0]),
                    max_cols=mlp.cfg.num_experts_per_token,
                ),
                "focus_combine_weights": _jax_float_rows(
                    focus_combine_weights,
                    max_rows=int(focus_combine_weights.shape[0]),
                    max_cols=mlp.cfg.num_experts_per_token,
                ),
                "focus_unbiased_top_expert_ids": _jax_int_rows(
                    focus_unbiased_top_ids,
                    max_rows=int(focus_unbiased_top_ids.shape[0]),
                    max_cols=top_count,
                ),
                "focus_unbiased_top_logits": _jax_float_rows(
                    focus_unbiased_top_logits,
                    max_rows=int(focus_unbiased_top_logits.shape[0]),
                    max_cols=top_count,
                ),
                "focus_biased_top_expert_ids": _jax_int_rows(
                    focus_biased_top_ids,
                    max_rows=int(focus_biased_top_ids.shape[0]),
                    max_cols=top_count,
                ),
                "focus_biased_top_logits": _jax_float_rows(
                    focus_biased_top_logits,
                    max_rows=int(focus_biased_top_logits.shape[0]),
                    max_cols=top_count,
                ),
                "focus_unbiased_top2_margin": _jax_float_vector(
                    focus_unbiased_top_logits[:, 0] - focus_unbiased_top_logits[:, 1],
                    max_items=int(focus_unbiased_top_logits.shape[0]),
                ),
                "focus_biased_top2_margin": _jax_float_vector(
                    focus_biased_top_logits[:, 0] - focus_biased_top_logits[:, 1],
                    max_items=int(focus_biased_top_logits.shape[0]),
                ),
            }
        )
    return summary


def _jax_forward_route_diagnostics(
    model: Any,
    token_ids_array: Any,
    *,
    position: int,
) -> dict[str, Any]:
    import jax.numpy as jnp  # noqa: PLC0415
    from jax.sharding import PartitionSpec as P  # noqa: PLC0415
    from jax.sharding import reshard  # noqa: PLC0415
    from levanter.grug.attention import AttentionMask  # noqa: PLC0415
    from levanter.utils.activation import ActivationFunctionEnum  # noqa: PLC0415

    from experiments.grug.moe import model as grug_model  # noqa: PLC0415

    token_ids = jnp.asarray(token_ids_array, dtype=jnp.int32)
    batch_size = int(token_ids.shape[0])
    seq_len = int(token_ids.shape[1])
    focus_token_indices = jnp.arange(batch_size, dtype=jnp.int32) * seq_len + jnp.asarray(position, dtype=jnp.int32)
    batch_spec = grug_model._batch_spec()
    hidden = model.token_embed.at[token_ids].get(out_sharding=batch_spec)
    hidden = model.embed_gated_norm(model.embed_norm(hidden))
    short_mask, long_mask = grug_model._layer_attention_masks(
        AttentionMask.causal(),
        sliding_window=model.config.sliding_window,
    )
    layer_summaries = []
    for layer_index, block in enumerate(model.blocks):
        layer_mask = long_mask if layer_index % 4 == 3 else short_mask
        attn_in = block.attn_gated_norm(block.rms_attn(hidden))
        hidden = grug_model._batch_reshard(hidden + block.attn(attn_in, layer_mask))
        mlp_in = grug_model._batch_reshard(block.mlp_gated_norm(block.rms_mlp(hidden)))
        layer_summaries.append(
            _jax_mlp_route_summary(
                block.mlp,
                mlp_in,
                layer_index=layer_index,
                focus_token_indices=focus_token_indices,
            )
        )
        mlp_out, _ = block.mlp(mlp_in)
        if block.shared is not None:
            mlp_out = mlp_out + block.shared(mlp_in, activation=ActivationFunctionEnum.silu)
        hidden = hidden + mlp_out

    hidden = model.final_gated_norm(model.final_norm(hidden))
    logits = jnp.einsum("bsh,hd->bsd", hidden, model.output_proj, out_sharding=batch_spec)
    position_logits = reshard(logits[:, position, :].astype(jnp.float32), P(None, None))
    rows = _jax_top_logits_rows(position_logits, max_rows=int(position_logits.shape[0]))
    first_row = rows[0] if rows else {}
    return {
        "position": position,
        "focus_token_flat_indices": _jax_int_vector(focus_token_indices, max_items=batch_size),
        "logits_dtype": _jax_dtype_name(logits),
        "position_logits_dtype": _jax_dtype_name(position_logits),
        "top_token_ids": first_row.get("top_token_ids", []),
        "top_logits": first_row.get("top_logits", []),
        "top2_margin": first_row.get("top2_margin"),
        "rows": rows,
        "layers": layer_summaries,
    }


def _token_top_logits_summary(tokenizer: Any, logits: Any) -> dict[str, Any]:
    import numpy as np  # noqa: PLC0415

    rows = _token_top_logits_rows_summary(tokenizer, np.asarray(logits)[None, :], max_rows=1)
    return {key: value for key, value in rows[0].items() if key != "row_index"}


def _token_top_logits_rows_summary(
    tokenizer: Any, logits_batch: Any, *, max_rows: int | None = None
) -> list[dict[str, Any]]:
    import numpy as np  # noqa: PLC0415

    logits_array = np.asarray(logits_batch)
    if logits_array.ndim != 2:
        raise ValueError(f"expected 2D logits batch, got shape {logits_array.shape!r}")
    row_count = logits_array.shape[0] if max_rows is None else min(int(max_rows), logits_array.shape[0])
    top_count = min(DIAGNOSTIC_LOGPROBS, int(logits_array.shape[-1]))
    rows: list[dict[str, Any]] = []
    for row_index in range(row_count):
        logits = logits_array[row_index]
        top_ids = np.argsort(-logits)[:top_count]
        top_logits = logits[top_ids].astype(np.float32)
        selected_token_id = int(np.argmax(logits, axis=-1))
        rows.append(
            {
                "row_index": row_index,
                "selected_token_id": selected_token_id,
                "selected_token_text": _decode_one(tokenizer, selected_token_id),
                "selected_token_logprob": _selected_logprob(logits, selected_token_id),
                "top_token_ids": [int(token_id) for token_id in top_ids.tolist()],
                "top_token_texts": [_decode_one(tokenizer, int(token_id)) for token_id in top_ids.tolist()],
                "top_logits": [float(value) for value in top_logits.tolist()],
                "top2_margin": float(top_logits[0] - top_logits[1]) if top_count >= 2 else None,
            }
        )
    return rows


def _jax_top_logits_rows(value: Any, *, max_rows: int | None = None) -> list[dict[str, Any]]:
    import jax  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    logits_array = np.asarray(jax.device_get(value))
    if logits_array.ndim != 2:
        raise ValueError(f"expected 2D logits batch, got shape {logits_array.shape!r}")
    row_count = logits_array.shape[0] if max_rows is None else min(int(max_rows), logits_array.shape[0])
    top_count = min(DIAGNOSTIC_LOGPROBS, int(logits_array.shape[-1]))
    rows: list[dict[str, Any]] = []
    for row_index in range(row_count):
        logits = logits_array[row_index]
        top_ids = np.argsort(-logits)[:top_count]
        top_logits = logits[top_ids].astype(np.float32)
        selected_token_id = int(np.argmax(logits, axis=-1))
        rows.append(
            {
                "row_index": row_index,
                "selected_token_id": selected_token_id,
                "top_token_ids": [int(token_id) for token_id in top_ids.tolist()],
                "top_logits": [float(value) for value in top_logits.tolist()],
                "top2_margin": float(top_logits[0] - top_logits[1]) if top_count >= 2 else None,
            }
        )
    return rows


def _greedy_decode_with_diagnostics(
    model: Any,
    tokenizer: Any,
    prompt_ids: list[int],
    *,
    max_new_tokens: int,
    batch_size: int,
    decode_seq_len: int,
    pad_token_id: int,
    use_active_prefix: bool = False,
    collect_route_diagnostics: bool = True,
) -> dict[str, Any]:
    """Run a Marin validation-only greedy decode and expose per-row evidence.

    The row-aware feedback below is a diagnostic harness choice: each repeated
    row feeds back its own selected token so batch sensitivity stays visible.
    It is not a change to Levanter model semantics or a reusable generation API.
    """

    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    if len(prompt_ids) + max_new_tokens > decode_seq_len:
        raise ValueError(
            f"prompt length {len(prompt_ids)} + max_new_tokens {max_new_tokens} exceeds decode_seq_len {decode_seq_len}"
        )

    def position_logits_batch(the_model: Any, token_ids: Any, position: Any) -> Any:
        return the_model.logits(token_ids)[:, position, :].astype(jnp.float32)

    token_ids_array = np.full((batch_size, decode_seq_len), pad_token_id, dtype=np.int32)
    token_ids_array[:, : len(prompt_ids)] = np.asarray(prompt_ids, dtype=np.int32)
    generated_ids: list[int] = []
    generated_token_texts: list[str] = []
    selected_token_logprobs: list[float] = []
    steps: list[dict[str, Any]] = []
    route_diagnostics: list[dict[str, Any]] = []
    active_seq_lengths: list[int] = []
    row_generated_ids: list[list[int]] = [[] for _ in range(batch_size)]
    row_generated_token_texts: list[list[str]] = [[] for _ in range(batch_size)]
    position_logits = jax.jit(position_logits_batch)
    started = time.time()

    for step_index in range(max_new_tokens):
        active_seq_len = len(prompt_ids) + step_index
        position = jnp.asarray(active_seq_len - 1, dtype=jnp.int32)
        step_token_ids_array = token_ids_array[:, :active_seq_len] if use_active_prefix else token_ids_array
        if collect_route_diagnostics and step_index < DIAGNOSTIC_ROUTE_STEPS:
            route_diagnostics.append(
                {
                    "generated_token_index": step_index,
                    **_jax_forward_route_diagnostics(
                        model,
                        step_token_ids_array,
                        position=int(active_seq_len - 1),
                    ),
                }
            )
        step_logits_batch = position_logits(model, jnp.asarray(step_token_ids_array, dtype=jnp.int32), position)
        rows = _token_top_logits_rows_summary(tokenizer, np.asarray(jax.device_get(step_logits_batch)))
        selected_token_ids = [int(row["selected_token_id"]) for row in rows]
        selected_token_texts = [str(row["selected_token_text"]) for row in rows]
        selected_token_logprobs_for_rows = [float(row["selected_token_logprob"]) for row in rows]
        for row_index, (token_id, token_text) in enumerate(zip(selected_token_ids, selected_token_texts, strict=True)):
            row_generated_ids[row_index].append(token_id)
            row_generated_token_texts[row_index].append(token_text)
        selected_token_id = selected_token_ids[0]
        selected_text = selected_token_texts[0]
        selected_logprob = selected_token_logprobs_for_rows[0]
        top_logits = {key: value for key, value in rows[0].items() if key not in {"row_index", "selected_token_id"}}
        active_seq_lengths.append(active_seq_len)
        generated_ids.append(selected_token_id)
        generated_token_texts.append(selected_text)
        selected_token_logprobs.append(selected_logprob)
        steps.append(
            {
                "generated_token_index": step_index,
                "token_id": selected_token_id,
                "token_text": selected_text,
                "selected_token_logprob": selected_logprob,
                **top_logits,
                "row_token_ids": selected_token_ids,
                "row_token_texts": selected_token_texts,
                "row_token_counts": _value_counts(selected_token_ids),
                "row_identical": len(set(selected_token_ids)) == 1,
                "rows": rows,
            }
        )
        token_ids_array[:, len(prompt_ids) + step_index] = np.asarray(selected_token_ids, dtype=np.int32)

    row_completions = [tokenizer.decode(row_ids, skip_special_tokens=False) for row_ids in row_generated_ids]
    return {
        "prompt_token_ids": [int(token_id) for token_id in prompt_ids],
        "prompt_token_count": len(prompt_ids),
        "decode_batch_size": batch_size,
        "decode_seq_len": decode_seq_len,
        "use_active_prefix": use_active_prefix,
        "route_diagnostics_enabled": collect_route_diagnostics,
        "active_seq_lengths": active_seq_lengths,
        "pad_token_id": pad_token_id,
        "generated_token_ids": generated_ids,
        "generated_token_texts": generated_token_texts,
        "completion": tokenizer.decode(generated_ids, skip_special_tokens=False),
        "selected_token_logprobs": selected_token_logprobs,
        "row_generated_token_ids": row_generated_ids,
        "row_generated_token_texts": row_generated_token_texts,
        "row_completions": row_completions,
        "row_completion_counts": _value_counts(row_completions),
        "row_completions_identical": len(set(row_completions)) == 1,
        "steps": steps,
        "route_diagnostics": route_diagnostics,
        "elapsed_seconds": time.time() - started,
    }


def _decode_step_margin_summary(decode_result: dict[str, Any]) -> dict[str, Any]:
    step_min_top2_margins: list[float | None] = []
    step_row_identical: list[bool] = []
    for step in decode_result.get("steps", []):
        rows = step.get("rows", [])
        margins = [float(row["top2_margin"]) for row in rows if row.get("top2_margin") is not None]
        step_min_top2_margins.append(min(margins) if margins else None)
        step_row_identical.append(bool(step.get("row_identical")))
    finite_margins = [margin for margin in step_min_top2_margins if margin is not None]
    return {
        "step_min_top2_margins": step_min_top2_margins,
        "min_step_top2_margin": min(finite_margins) if finite_margins else None,
        "step_row_identical": step_row_identical,
        "all_steps_row_identical": all(step_row_identical) if step_row_identical else False,
    }


def _run_levanter_prompt_sweep(
    model: Any,
    tokenizer: Any,
    *,
    prompts: tuple[str, ...],
    batch_size: int,
    decode_seq_len: int,
    pad_token_id: int,
    use_active_prefix: bool,
    collect_route_diagnostics: bool,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for prompt_index, prompt in enumerate(prompts):
        prompt_ids, tokenization = _prompt_token_ids(tokenizer, prompt)
        decode_result = _greedy_decode_with_diagnostics(
            model,
            tokenizer,
            prompt_ids,
            max_new_tokens=DIAGNOSTIC_PROMPT_SWEEP_MAX_NEW_TOKENS,
            batch_size=batch_size,
            decode_seq_len=decode_seq_len,
            pad_token_id=pad_token_id,
            use_active_prefix=use_active_prefix,
            collect_route_diagnostics=collect_route_diagnostics,
        )
        first_step_rows = decode_result["steps"][0]["rows"] if decode_result["steps"] else []
        top2_margins = [float(row["top2_margin"]) for row in first_step_rows if row.get("top2_margin") is not None]
        step_margin_summary = _decode_step_margin_summary(decode_result)
        results.append(
            {
                "prompt_index": prompt_index,
                "prompt": prompt,
                "tokenization": tokenization,
                "max_new_tokens": DIAGNOSTIC_PROMPT_SWEEP_MAX_NEW_TOKENS,
                "completion": decode_result["completion"],
                "row_completions": decode_result["row_completions"],
                "row_completion_counts": decode_result["row_completion_counts"],
                "row_completions_identical": decode_result["row_completions_identical"],
                "first_step_row_token_counts": (
                    decode_result["steps"][0]["row_token_counts"] if decode_result["steps"] else {}
                ),
                "first_step_min_top2_margin": min(top2_margins) if top2_margins else None,
                **step_margin_summary,
                "decode_result": decode_result,
            }
        )
    return results


def _run_levanter_forced_prefix_diagnostics(
    model: Any,
    tokenizer: Any,
    *,
    batch_size: int,
    decode_seq_len: int,
    pad_token_id: int,
    use_active_prefix: bool,
    collect_route_diagnostics: bool,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for prompt_spec in FORCED_PREFIX_DIAGNOSTIC_PROMPTS:
        name = str(prompt_spec["name"])
        prompt = str(prompt_spec["prompt"])
        prompt_ids, tokenization = _prompt_token_ids(tokenizer, prompt)
        decode_result = _greedy_decode_with_diagnostics(
            model,
            tokenizer,
            prompt_ids,
            max_new_tokens=1,
            batch_size=batch_size,
            decode_seq_len=decode_seq_len,
            pad_token_id=pad_token_id,
            use_active_prefix=use_active_prefix,
            collect_route_diagnostics=collect_route_diagnostics,
        )
        results.append(
            {
                "name": name,
                "prompt": prompt,
                "tokenization": tokenization,
                "max_new_tokens": 1,
                "completion": decode_result["completion"],
                "row_completions": decode_result["row_completions"],
                "row_completion_counts": decode_result["row_completion_counts"],
                "row_completions_identical": decode_result["row_completions_identical"],
                **_decode_step_margin_summary(decode_result),
                "decode_result": decode_result,
            }
        )
    return results


def _fixed_training_loss_batch(tokenizer: Any, *, batch_size: int) -> tuple[Any, Any, dict[str, Any]]:
    import numpy as np  # noqa: PLC0415

    prompt = TRAINING_LOSS_DIAGNOSTIC_PROMPTS[0]
    token_ids, tokenization = _prompt_token_ids(tokenizer, prompt)
    if len(token_ids) < 2:
        raise ValueError(f"training diagnostic prompt produced too few tokens: {tokenization!r}")
    token_ids_array = np.asarray([token_ids for _ in range(batch_size)], dtype=np.int32)
    loss_weight = np.ones_like(token_ids_array, dtype=np.float32)
    loss_weight[:, -1] = 0.0
    return (
        token_ids_array,
        loss_weight,
        {
            "prompt": prompt,
            "tokenization": tokenization,
            "batch_size": batch_size,
            "seq_len": len(token_ids),
            "loss_weight_sum": float(loss_weight.sum()),
        },
    )


def _jax_scalar(value: Any) -> float:
    import jax  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    return float(np.asarray(jax.device_get(value), dtype=np.float32).reshape(()))


def _jax_int_scalar(value: Any) -> int:
    import jax  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    return int(np.asarray(jax.device_get(value), dtype=np.int64).reshape(()))


def _jax_array_finiteness(value: Any) -> dict[str, Any]:
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    array = jnp.asarray(value)
    finite = jnp.isfinite(array)
    finite_count = jnp.sum(finite.astype(jnp.int32))
    total = array.size
    return {
        "shape": [int(dim) for dim in array.shape],
        "finite": bool(jax.device_get(jnp.all(finite))),
        "finite_count": _jax_int_scalar(finite_count),
        "nonfinite_count": int(total) - _jax_int_scalar(finite_count),
    }


def _summarize_grad_tree(grads: Any) -> dict[str, Any]:
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    leaves = [
        leaf
        for leaf in jax.tree_util.tree_leaves(grads)
        if leaf is not None and hasattr(leaf, "dtype") and hasattr(leaf, "shape")
    ]
    if not leaves:
        return {
            "array_leaf_count": 0,
            "parameter_count": 0,
            "global_l2_norm": 0.0,
            "max_abs": 0.0,
            "all_finite": True,
            "nonfinite_leaf_count": 0,
        }

    squared_norms = [jnp.sum(jnp.square(jnp.asarray(leaf, dtype=jnp.float32))) for leaf in leaves]
    max_abs_values = [jnp.max(jnp.abs(jnp.asarray(leaf, dtype=jnp.float32))) for leaf in leaves]
    finite_values = [jnp.all(jnp.isfinite(jnp.asarray(leaf))) for leaf in leaves]
    global_l2_norm = jnp.sqrt(sum(squared_norms, jnp.asarray(0.0, dtype=jnp.float32)))
    max_abs = jnp.max(jnp.stack(max_abs_values))
    finite_flags = jnp.stack(finite_values)
    return {
        "array_leaf_count": len(leaves),
        "parameter_count": int(sum(int(leaf.size) for leaf in leaves)),
        "global_l2_norm": _jax_scalar(global_l2_norm),
        "max_abs": _jax_scalar(max_abs),
        "all_finite": bool(jax.device_get(jnp.all(finite_flags))),
        "nonfinite_leaf_count": _jax_int_scalar(jnp.sum((~finite_flags).astype(jnp.int32))),
    }


def _summarize_routing_counts(value: Any) -> dict[str, Any]:
    import jax  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    counts = np.asarray(jax.device_get(value), dtype=np.float32)
    per_layer: list[dict[str, Any]] = []
    for layer_index, layer_counts in enumerate(counts):
        top_count = min(DIAGNOSTIC_LOGPROBS, int(layer_counts.shape[0]))
        top_ids = np.argsort(-layer_counts)[:top_count]
        nonzero_counts = layer_counts[layer_counts > 0]
        per_layer.append(
            {
                "layer_index": layer_index,
                "total_assignments": float(layer_counts.sum()),
                "nonzero_experts": int(np.count_nonzero(layer_counts)),
                "min_nonzero_load": float(nonzero_counts.min()) if nonzero_counts.size else 0.0,
                "max_load": float(layer_counts.max()) if layer_counts.size else 0.0,
                "top_expert_ids": [int(item) for item in top_ids.tolist()],
                "top_expert_loads": [float(layer_counts[item]) for item in top_ids.tolist()],
                "expert_load_histogram": [float(item) for item in layer_counts.tolist()],
            }
        )
    return {
        "shape": [int(dim) for dim in counts.shape],
        "total_assignments": float(counts.sum()),
        "per_layer": per_layer,
    }


def _summarize_training_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    scalar_names = (
        "train/cross_entropy_loss",
        "train/router/aux_loss_weighted",
        "train/router/routing_entropy_mean",
        "train/router/load_balancing_loss",
        "train/router/router_z_loss",
    )
    summary: dict[str, Any] = {}
    for name in scalar_names:
        if name in metrics:
            summary[name] = _jax_scalar(metrics[name])
            summary[f"{name}:finiteness"] = _jax_array_finiteness(metrics[name])
    if "train/router/dropped_assignments" in metrics:
        summary["train/router/dropped_assignments"] = _jax_int_scalar(metrics["train/router/dropped_assignments"])
    if "train/router/dropped_assignments_per_layer" in metrics:
        summary["train/router/dropped_assignments_per_layer"] = _jax_int_vector(
            metrics["train/router/dropped_assignments_per_layer"],
            max_items=int(metrics["train/router/dropped_assignments_per_layer"].shape[0]),
        )
    if "train/router/routing_counts_per_layer" in metrics:
        summary["expert_load_histograms"] = _summarize_routing_counts(metrics["train/router/routing_counts_per_layer"])
    return summary


def _run_jax_training_loss_case(
    model: Any,
    *,
    token_ids_array: Any,
    loss_weight_array: Any,
) -> dict[str, Any]:
    import equinox as eqx  # noqa: PLC0415
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    token_ids = jnp.asarray(token_ids_array, dtype=jnp.int32)
    loss_weight = jnp.asarray(loss_weight_array, dtype=jnp.float32)

    def loss_fn(the_model: Any) -> tuple[Any, dict[str, Any]]:
        loss, metrics = the_model.next_token_loss(
            token_ids,
            loss_weight,
            reduction="mean",
            return_router_metrics=True,
        )
        return loss, metrics

    grad_fn = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn, has_aux=True))
    started = time.time()
    (loss, metrics), grads = grad_fn(model)
    jax.block_until_ready(loss)
    grad_summary = _summarize_grad_tree(grads)
    metrics_summary = _summarize_training_metrics(metrics)
    try:
        route_diagnostics = _jax_forward_route_diagnostics(
            model,
            token_ids,
            position=int(token_ids.shape[1] - 2),
        )
    except Exception as exc:  # Keep the actual loss/grad path authoritative.
        route_diagnostics = {
            "available": False,
            "failure": _exception_summary(exc),
        }
    focus_biased_margins = [
        margin for layer in route_diagnostics.get("layers", []) for margin in layer.get("focus_biased_top2_margin", [])
    ]
    return {
        "loss": _jax_scalar(loss),
        "loss_finiteness": _jax_array_finiteness(loss),
        "metrics": metrics_summary,
        "grad_norm_summary": grad_summary,
        "router_focus_min_biased_top2_margin": min(focus_biased_margins) if focus_biased_margins else None,
        "router_focus_diagnostics": route_diagnostics,
        "nan_inf_checks": {
            "loss_finite": _jax_array_finiteness(loss)["finite"],
            "grads_all_finite": grad_summary["all_finite"],
        },
        "elapsed_seconds": time.time() - started,
    }


def _run_levanter_training_loss_diagnostics(
    *,
    checkpoint_path: str,
    tokenizer: Any,
    model_cfg: Any,
) -> dict[str, Any]:
    import haliax  # noqa: PLC0415
    from haliax.partitioning import set_mesh  # noqa: PLC0415
    from levanter.grug.sharding import compact_grug_mesh  # noqa: PLC0415

    cases: list[dict[str, Any]] = []
    started = time.time()
    for expert_axis_size in TRAINING_LOSS_DIAGNOSTIC_EXPERT_AXIS_SIZES:
        mesh = compact_grug_mesh(expert_axis_size=expert_axis_size, model_axis_size=1)
        mesh_runtime = _mesh_snapshot(mesh)
        batch_size = _mesh_batch_axis_size(mesh)
        token_ids_array, loss_weight_array, batch_summary = _fixed_training_loss_batch(tokenizer, batch_size=batch_size)
        with ExitStack() as stack:
            stack.enter_context(set_mesh(mesh))
            stack.enter_context(haliax.axis_mapping({}))
            loaded_model = _load_legacy_split_expert_checkpoint(checkpoint_path, model_cfg)
            executable_model = _executable_model_from_legacy_split(
                loaded_model,
                capacity_factor=LEVANTER_MOE_CAPACITY_FACTOR,
            )
            for reference_mode in TRAINING_LOSS_DIAGNOSTIC_REFERENCE_MODES:
                model, reference_policy = _apply_levanter_reference_mode(
                    executable_model,
                    reference_mode,
                )
                cases.append(
                    {
                        "expert_axis_size": expert_axis_size,
                        "reference_mode": reference_mode,
                        "mesh": mesh_runtime,
                        "batch": batch_summary,
                        "reference_policy": reference_policy,
                        "dtype_summary": _jax_model_dtype_summary(model),
                        **_run_jax_training_loss_case(
                            model,
                            token_ids_array=token_ids_array,
                            loss_weight_array=loss_weight_array,
                        ),
                    }
                )

    by_key = {(int(case["expert_axis_size"]), str(case["reference_mode"])): case for case in cases if "loss" in case}
    loss_deltas: dict[str, float] = {}
    for expert_axis_size in TRAINING_LOSS_DIAGNOSTIC_EXPERT_AXIS_SIZES:
        fp32_case = by_key.get((expert_axis_size, "current"))
        bf16_case = by_key.get((expert_axis_size, "bf16_compute"))
        if fp32_case is not None and bf16_case is not None:
            loss_deltas[f"ep{expert_axis_size}:bf16_minus_current"] = float(bf16_case["loss"] - fp32_case["loss"])
    for reference_mode in TRAINING_LOSS_DIAGNOSTIC_REFERENCE_MODES:
        ep1_case = by_key.get((1, reference_mode))
        ep8_case = by_key.get((EXPECTED_GPU_COUNT, reference_mode))
        if ep1_case is not None and ep8_case is not None:
            loss_deltas[f"{reference_mode}:ep8_minus_ep1"] = float(ep8_case["loss"] - ep1_case["loss"])

    return {
        "enabled": True,
        "expert_axis_sizes": list(TRAINING_LOSS_DIAGNOSTIC_EXPERT_AXIS_SIZES),
        "reference_modes": list(TRAINING_LOSS_DIAGNOSTIC_REFERENCE_MODES),
        "cases": cases,
        "loss_deltas": loss_deltas,
        "elapsed_seconds": time.time() - started,
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
        WORKER_EXTENSION_CLS,
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
    log_artifacts: dict[str, Any] = {}
    try:
        try:
            with VllmEnvironment(model=model, timeout_seconds=SERVER_TIMEOUT_SECONDS, extra_args=extra_args) as env:
                print("vllm_gpu_server_initialized=True", flush=True)
                print("vllm_gpu_server_url=" + env.server_url, flush=True)
                print("vllm_gpu_model_path=" + staged_artifact.vllm_model_path, flush=True)
                print("vllm_gpu_artifact_staging=" + json.dumps(staged_artifact.staging, sort_keys=True), flush=True)
                print("vllm_gpu_server_log_dir=" + (env.vllm_server.log_dir if env.vllm_server else ""), flush=True)
                worker_ep_states = _collect_grug_moe_worker_ep_states(env)
                worker_ep_summary = _assert_grug_moe_worker_ep_states(worker_ep_states, num_experts=num_experts)
                # Separately covers the batch-size-1 serving path before the
                # per-rank batch-size-2 requests below.
                single_payload = _completion_payload(
                    env,
                    prompts=[PROMPT],
                    data_parallel_rank=0,
                    request_id="grugmoe-single-rank0",
                    logprobs=DIAGNOSTIC_LOGPROBS,
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
                        logprobs=DIAGNOSTIC_LOGPROBS,
                    )
                    payloads.append(payload)
                    rank_request_batches.append(
                        {
                            "data_parallel_rank": data_parallel_rank,
                            "prompt_indices": [data_parallel_rank * 2, data_parallel_rank * 2 + 1],
                            "batch_size": len(prompts),
                        }
                    )
                completion_diagnostics = _run_vllm_completion_diagnostics(
                    env,
                    worker_ep_states=worker_ep_states,
                )
                prompt_sweep_results = _run_vllm_prompt_sweep(
                    env,
                    worker_ep_states=worker_ep_states,
                )
                forced_prefix_results = (
                    _run_vllm_forced_prefix_diagnostics(
                        env,
                        worker_ep_states=worker_ep_states,
                    )
                    if FORCED_PREFIX_DIAGNOSTICS
                    else []
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
                "vllm_dtype_env_var": VLLM_DTYPE_ENV,
                "vllm_dtype": VLLM_DTYPE,
                "vllm_moe_compute_env_var": VLLM_MOE_COMPUTE_ENV,
                "vllm_moe_compute": VLLM_MOE_COMPUTE,
                "vllm_route_diagnostics_env_var": VLLM_ROUTE_DIAGNOSTICS_ENV,
                "vllm_route_diagnostics": VLLM_ROUTE_DIAGNOSTICS,
                "vllm_tensor_parallel_size": VLLM_TENSOR_PARALLEL_SIZE,
                "vllm_data_parallel_size": VLLM_DATA_PARALLEL_SIZE,
                "vllm_expert_parallel_size": VLLM_EXPERT_PARALLEL_SIZE,
                "vllm_attention_backend": VLLM_ATTENTION_BACKEND,
                "vllm_max_num_seqs": VLLM_MAX_NUM_SEQS,
                "vllm_server_dev_mode_enabled": True,
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
                _owners_for_worker_expert_placement(
                    routed_experts,
                    worker_ep_states=worker_ep_states,
                )
            )
    completion = completions[0] if completions else ""
    if len(completions) != PROMPT_BATCH_SIZE:
        raise AssertionError(f"expected {PROMPT_BATCH_SIZE} completions, got {len(completions)}")
    completion_counts = {item: completions.count(item) for item in sorted(set(completions))}
    repeated_prompt_identical = len(completion_counts) == 1
    routed_owner_rank_coverage = sorted(routed_owner_ranks) == list(range(VLLM_EXPERT_PARALLEL_SIZE))
    single_prompt_choice_summary = _summarize_completion_payload(
        single_payload,
        worker_ep_states=worker_ep_states,
        expected_continuation=EXPECTED_CONTINUATION,
    )["choices"][0]
    main_choice_summaries = [
        choice_summary
        for payload in payloads
        for choice_summary in _summarize_completion_payload(
            payload,
            worker_ep_states=worker_ep_states,
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
        "single_prompt_routed_experts": single_routed_experts,
        "single_prompt_choice_summary": single_prompt_choice_summary,
        "completions": completions,
        "completion_counts": completion_counts,
        "repeated_prompt_identical": repeated_prompt_identical,
        "expected_continuation": EXPECTED_CONTINUATION,
        "passed": (
            single_completion == EXPECTED_CONTINUATION
            and all(item == EXPECTED_CONTINUATION for item in completions)
            and repeated_prompt_identical
            and routed_owner_rank_coverage
        ),
        "served_model_name": SERVED_MODEL_NAME,
        "vllm_model_id": model_id,
        "vllm_model_path": staged_artifact.vllm_model_path,
        "artifact_staging": staged_artifact.staging,
        "vllm_engine_kwargs": model.engine_kwargs,
        "vllm_args": extra_args,
        "vllm_attention_backend_env_var": VLLM_ATTENTION_BACKEND_ENV,
        "vllm_dtype_env_var": VLLM_DTYPE_ENV,
        "vllm_dtype": VLLM_DTYPE,
        "vllm_moe_compute_env_var": VLLM_MOE_COMPUTE_ENV,
        "vllm_moe_compute": VLLM_MOE_COMPUTE,
        "vllm_route_diagnostics_env_var": VLLM_ROUTE_DIAGNOSTICS_ENV,
        "vllm_route_diagnostics": VLLM_ROUTE_DIAGNOSTICS,
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
        "main_choice_summaries": main_choice_summaries,
        "completion_diagnostics": completion_diagnostics,
        "vllm_prompt_sweep_results": prompt_sweep_results,
        "forced_prefix_diagnostics_env_var": FORCED_PREFIX_DIAGNOSTICS_ENV,
        "forced_prefix_diagnostics": FORCED_PREFIX_DIAGNOSTICS,
        "vllm_forced_prefix_results": forced_prefix_results,
        "routed_experts_by_completion": routed_experts_by_completion,
        "routed_expert_num_experts": num_experts,
        "routed_expert_owner_ranks": sorted(routed_owner_ranks),
        "routed_expert_owner_rank_coverage": routed_owner_rank_coverage,
        "expected_gpu_count": EXPECTED_GPU_COUNT,
        "coreweave_s3": s3_env,
        "cuda_library_path": cuda_library_path,
        "torch_runtime": torch_runtime,
        "vllm_import_checks": vllm_import_checks,
        "vllm_log_artifacts": log_artifacts,
        "raw_responses": payloads,
        "vllm_logs_tail": logs_tail,
        "runtime": _runtime_snapshot(include_grugmoe_spec=True, include_torch_cuda=True),
        "elapsed_seconds": time.time() - started,
    }
    _write_json(args.result_path, result)
    print("grugmoe_gpu_real_checkpoint_vllm_result=" + json.dumps(result, sort_keys=True), flush=True)
    if result["passed"] is not True:
        raise AssertionError(
            f"GPU vLLM single={single_completion!r}, completion_counts={completion_counts!r}, "
            f"routed_owner_ranks={sorted(routed_owner_ranks)!r} != expected {EXPECTED_CONTINUATION!r}"
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
        pre_policy_dtype_summary = _jax_model_dtype_summary(executable_model)
        model, reference_policy = _apply_levanter_reference_mode(
            executable_model,
            LEVANTER_REFERENCE_MODE,
        )
        post_policy_dtype_summary = _jax_model_dtype_summary(model)
        decode_results: list[dict[str, Any]] = []
        completions: list[str] = []
        remaining = PROMPT_BATCH_SIZE
        while remaining > 0:
            decode_result = _greedy_decode_with_diagnostics(
                model,
                tokenizer,
                prompt_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                batch_size=decode_batch_size,
                decode_seq_len=DECODE_SEQ_LEN,
                pad_token_id=pad_token_id,
                use_active_prefix=LEVANTER_DECODE_USE_ACTIVE_PREFIX,
                collect_route_diagnostics=LEVANTER_ROUTE_DIAGNOSTICS,
            )
            decode_results.append(decode_result)
            batch_items = min(decode_batch_size, remaining)
            row_completions = [str(item) for item in decode_result["row_completions"]]
            completions.extend(row_completions[:batch_items])
            remaining -= batch_items
        prompt_sweep_results = (
            _run_levanter_prompt_sweep(
                model,
                tokenizer,
                prompts=DIAGNOSTIC_PROMPT_SWEEP_PROMPTS,
                batch_size=decode_batch_size,
                decode_seq_len=DECODE_SEQ_LEN,
                pad_token_id=pad_token_id,
                use_active_prefix=LEVANTER_DECODE_USE_ACTIVE_PREFIX,
                collect_route_diagnostics=LEVANTER_ROUTE_DIAGNOSTICS,
            )
            if LEVANTER_PROMPT_SWEEP
            else []
        )
        forced_prefix_results = (
            _run_levanter_forced_prefix_diagnostics(
                model,
                tokenizer,
                batch_size=decode_batch_size,
                decode_seq_len=DECODE_SEQ_LEN,
                pad_token_id=pad_token_id,
                use_active_prefix=LEVANTER_DECODE_USE_ACTIVE_PREFIX,
                collect_route_diagnostics=LEVANTER_ROUTE_DIAGNOSTICS,
            )
            if FORCED_PREFIX_DIAGNOSTICS
            else []
        )
    training_loss_results = (
        _run_levanter_training_loss_diagnostics(
            checkpoint_path=args.checkpoint_path,
            tokenizer=tokenizer,
            model_cfg=model_cfg,
        )
        if TRAINING_LOSS_DIAGNOSTICS
        else {"enabled": False}
    )
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
        "levanter_expert_axis_size_env_var": LEVANTER_EXPERT_AXIS_SIZE_ENV,
        "levanter_expert_axis_size": LEVANTER_EXPERT_AXIS_SIZE,
        "levanter_moe_capacity_factor": LEVANTER_MOE_CAPACITY_FACTOR,
        "levanter_decode_use_active_prefix": LEVANTER_DECODE_USE_ACTIVE_PREFIX,
        "decode_results": decode_results,
        "levanter_prompt_sweep_env_var": LEVANTER_PROMPT_SWEEP_ENV,
        "levanter_prompt_sweep": LEVANTER_PROMPT_SWEEP,
        "levanter_prompt_sweep_prompts": list(DIAGNOSTIC_PROMPT_SWEEP_PROMPTS),
        "levanter_prompt_sweep_results": prompt_sweep_results,
        "levanter_route_diagnostics_env_var": LEVANTER_ROUTE_DIAGNOSTICS_ENV,
        "levanter_route_diagnostics": LEVANTER_ROUTE_DIAGNOSTICS,
        "forced_prefix_diagnostics_env_var": FORCED_PREFIX_DIAGNOSTICS_ENV,
        "forced_prefix_diagnostics": FORCED_PREFIX_DIAGNOSTICS,
        "levanter_forced_prefix_results": forced_prefix_results,
        "training_loss_diagnostics_env_var": TRAINING_LOSS_DIAGNOSTICS_ENV,
        "training_loss_diagnostics": TRAINING_LOSS_DIAGNOSTICS,
        "levanter_training_loss_results": training_loss_results,
        "levanter_reference_mode_env_var": LEVANTER_REFERENCE_MODE_ENV,
        "levanter_reference_mode": LEVANTER_REFERENCE_MODE,
        "levanter_reference_policy": reference_policy,
        "levanter_pre_policy_dtype_summary": pre_policy_dtype_summary,
        "levanter_post_policy_dtype_summary": post_policy_dtype_summary,
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
