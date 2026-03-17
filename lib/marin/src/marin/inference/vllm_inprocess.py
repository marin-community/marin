# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import jax
import requests
from fsspec import AbstractFileSystem
from iris.marin_fs import url_to_fs

from levanter.compat.fsspec_safetensor import read_safetensors_fsspec

from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.rl.environments.inference_ctx.vllm_utils import MODEL_MAPPINGS, MODEL_TRANSPOSE_KEYS
from marin.rl.weight_utils import levanter_state_dict_to_nnx_state_on_cpu

# Save real stdout — vLLM may wrap sys.stdout with rank-prefixed streams,
# making print() invisible to Iris's container log collector.
_REAL_STDOUT = sys.stdout

# Must be set before LLM() is called — vLLM reads this to decide whether to
# spawn child processes for EngineCore.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

logger = logging.getLogger(__name__)

SAFE_TENSORS_MODEL = "model.safetensors"
SAFE_TENSORS_INDEX_NAME = "model.safetensors.index.json"

# These are already consumed from engine_kwargs by _engine_kwargs_to_cli_args.
_SUPPORTED_ENGINE_CLI_FLAGS_WITH_VALUE = {
    "--load-format",
    "--max-model-len",
    "--gpu-memory-utilization",
    "--model-loader-extra-config",
}
_SUPPORTED_ENGINE_CLI_FLAGS_NO_VALUE = {
    "--trust-remote-code",
}

_DEFAULT_LLAMA_MAPPING_KEY = "meta-llama/Llama-3.1-8B-Instruct"
_DEFAULT_QWEN_MAPPING_KEY = "Qwen/Qwen3-8B"
_INPROCESS_BOOTSTRAP_MODEL_KEY = "inprocess_bootstrap_model"
_BOOTSTRAP_METADATA_FILENAMES: tuple[str, ...] = (
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "added_tokens.json",
    "merges.txt",
    "vocab.json",
    "generation_config.json",
    "chat_template.jinja",
)


def _iris_emit(level_char: str, source: str, message: str) -> None:
    """Emit a log line to the real stdout in Iris-compatible format.

    Uses ``_REAL_STDOUT`` (saved before vLLM import) because vLLM may wrap
    ``sys.stdout`` with rank-prefixed streams that Iris doesn't capture.
    """
    from datetime import datetime, timezone

    ts = datetime.now(timezone.utc).strftime("%Y%m%d %H:%M:%S")
    tid = threading.current_thread().ident or 0
    _REAL_STDOUT.write(f"{level_char}{ts} {tid} {source} {message}\n")
    _REAL_STDOUT.flush()


class InProcessVllmUnsupportedError(RuntimeError):
    """Raised when in-process vLLM startup cannot be used safely."""


@dataclass(frozen=True)
class InProcessEligibility:
    eligible: bool
    reason: str
    mapping_model_name: str | None = None
    bootstrap_model_source: str | None = None


@dataclass
class InProcessVllmRuntime:
    llm: Any
    server: Any
    serve_thread: threading.Thread
    server_url: str
    port: int
    model_id: str | None = None
    bootstrap_local_dir: str | None = None
    events: list[str] = field(default_factory=list)

    def logs_tail(self, *, max_lines: int = 200) -> str:
        if not self.events:
            return "<no in-process startup events captured>"
        return "\n".join(self.events[-max_lines:])


def evaluate_inprocess_eligibility(
    *,
    model: ModelConfig,
    model_name_or_path: str,
    extra_cli_args: list[str] | None,
) -> InProcessEligibility:
    """Return whether in-process vLLM startup can be attempted for this model."""

    if model.path is None:
        return InProcessEligibility(False, "model.path is unset; using hub/local ID path")

    if not _is_object_store_path(model.path):
        return InProcessEligibility(False, f"model path is not object store: {model.path!r}")

    explicit_load_format = model.engine_kwargs.get("load_format")
    if explicit_load_format is not None and explicit_load_format != "dummy":
        return InProcessEligibility(
            False,
            f"explicit load_format={explicit_load_format!r} is not compatible with in-process dummy load",
        )

    unsupported_args = _unsupported_extra_cli_args(extra_cli_args)
    if unsupported_args:
        return InProcessEligibility(
            False,
            f"unsupported CLI args for in-process startup: {unsupported_args}",
        )

    mapping_model_name = _resolve_mapping_model_name(model, model_name_or_path)
    if mapping_model_name is None:
        return InProcessEligibility(
            False,
            "unable to resolve MODEL_MAPPINGS / MODEL_TRANSPOSE_KEYS for this model",
        )

    bootstrap_resolution = _resolve_bootstrap_model_source_for_eligibility(model)
    if not bootstrap_resolution[0]:
        return InProcessEligibility(False, bootstrap_resolution[1])

    return InProcessEligibility(
        True,
        "eligible",
        mapping_model_name=mapping_model_name,
        bootstrap_model_source=bootstrap_resolution[0],
    )


def start_inprocess_vllm_server(
    *,
    model: ModelConfig,
    model_name_or_path: str,
    mapping_model_name: str,
    host: str,
    port: int,
    timeout_seconds: int,
    extra_cli_args: list[str] | None,
) -> InProcessVllmRuntime:
    """Start in-process vLLM with dummy load, inject weights, and serve OpenAI API."""

    unsupported_args = _unsupported_extra_cli_args(extra_cli_args)
    if unsupported_args:
        raise InProcessVllmUnsupportedError(f"unsupported CLI args for in-process vLLM startup: {unsupported_args}")

    if model.path is None:
        raise InProcessVllmUnsupportedError("in-process startup requires model.path for object-store checkpoint loading")

    llm = None
    server = None
    serve_thread: threading.Thread | None = None
    bootstrap_local_dir: str | None = None

    try:
        llm_cls, uvicorn_module = _import_inprocess_vllm_symbols()

        events: list[str] = []

        bootstrap_model_source, bootstrap_local_dir = _resolve_bootstrap_model_source_for_start(model)
        _record_event(
            events,
            f"Using bootstrap model source {bootstrap_model_source!r} for in-process dummy initialization",
        )

        t0 = time.time()
        llm = llm_cls(**_llm_kwargs(bootstrap_model_source=bootstrap_model_source, model=model))
        t_skeleton = time.time() - t0
        _record_event(events, f"Created in-process LLM skeleton for {model_name_or_path}")
        _iris_emit("I", "vllm.inprocess", f"LLM skeleton created in {t_skeleton:.1f}s")

        sync_weights_fn = _resolve_sync_weights_callable(llm)
        _load_and_inject_streaming(
            model_path=model.path,
            sync_weights_fn=sync_weights_fn,
            mapping_model_name=mapping_model_name,
            bootstrap_model_source=bootstrap_model_source,
            events=events,
        )
        _record_event(events, f"Injected weights via sync_weights using mapping key {mapping_model_name!r}")

        # Build a minimal OpenAI-compatible API around LLM.generate().
        # We avoid vLLM's build_app() because it spawns child processes that
        # fail with "TPU already in use by pid 1" on TPU.
        app = _create_inprocess_openai_app(llm, model_name_or_path, bootstrap_model_source)

        server = uvicorn_module.Server(
            uvicorn_module.Config(
                app,
                host=host,
                port=port,
                log_level="warning",
            )
        )
        serve_thread = threading.Thread(target=server.run, daemon=True, name=f"vllm-inprocess-{port}")
        serve_thread.start()

        server_url = f"http://{host}:{port}/v1"
        model_id = _wait_for_models_endpoint(server_url, timeout_seconds, serve_thread)
        _record_event(events, f"OpenAI endpoint ready at {server_url} with model_id={model_id!r}")
        _iris_emit("I", "vllm.inprocess", f"In-process OpenAI server ready at {server_url}")

        return InProcessVllmRuntime(
            llm=llm,
            server=server,
            serve_thread=serve_thread,
            server_url=server_url,
            port=port,
            model_id=model_id,
            bootstrap_local_dir=bootstrap_local_dir,
            events=events,
        )
    except Exception:
        if server is not None:
            server_for_shutdown: Any = server
            server_for_shutdown.should_exit = True
        if serve_thread is not None:
            serve_thread.join(timeout=10)
        if llm is not None:
            _shutdown_llm(llm)
        _cleanup_local_bootstrap_dir(bootstrap_local_dir)
        raise


def stop_inprocess_vllm_server(runtime: InProcessVllmRuntime) -> None:
    """Stop the in-process HTTP server and vLLM engine."""

    server_for_shutdown: Any = runtime.server
    server_for_shutdown.should_exit = True
    runtime.serve_thread.join(timeout=15)
    _shutdown_llm(runtime.llm)
    _cleanup_local_bootstrap_dir(runtime.bootstrap_local_dir)


def load_safetensors_from_remote(model_path: str) -> dict[str, Any]:
    """Load safetensors shards from remote storage through Levanter's fsspec path.

    Uses synchronous fsspec filesystem — Levanter's ``read_safetensors_fsspec``
    handles async internally via its own event loop.  We avoid ``asynchronous=True``
    because vLLM V1 can leave the default asyncio event loop in a broken state.
    """
    import asyncio

    fs, remote_path = url_to_fs(model_path)
    shard_files = _discover_safetensor_shards(fs, remote_path)

    state_dict: dict[str, Any] = {}
    cpu_device = jax.devices("cpu")[0]

    # Levanter's read_safetensors_fsspec is async.  Run it in a fresh event loop
    # to avoid conflicts with vLLM's event loop management.
    async def _load_shard(shard_path: str) -> dict[str, Any]:
        return await read_safetensors_fsspec(shard_path, fs=fs, sharding_fn=None)

    loop = asyncio.new_event_loop()
    try:
        with jax.default_device(cpu_device):
            for shard_file in shard_files:
                shard_path = os.path.join(remote_path, shard_file)
                shard_state = loop.run_until_complete(_load_shard(shard_path))
                state_dict.update(shard_state)
    finally:
        loop.close()

    return state_dict


def _load_and_inject_streaming(
    *,
    model_path: str,
    sync_weights_fn: Any,
    mapping_model_name: str,
    bootstrap_model_source: str,
    events: list[str],
) -> float:
    """Load safetensor shards one at a time and inject each into HBM immediately.

    This keeps peak host RAM at ~5 GiB (one shard) instead of accumulating the
    full model (~131 GiB for 70B).  ``sync_weights`` handles partial state dicts
    by skipping target keys not present in the source.

    Returns total wall-clock time for the weight pipeline (load + reshape + convert + inject).
    """
    import asyncio

    fs, remote_path = url_to_fs(model_path)
    shard_files = _discover_safetensor_shards(fs, remote_path)

    # Read model config for attention reshape constants.
    config_path = os.path.join(bootstrap_model_source, "config.json")
    with open(config_path) as f:
        model_config = json.load(f)
    num_heads = model_config["num_attention_heads"]
    num_kv_heads = model_config.get("num_key_value_heads", num_heads)
    head_dim = model_config["hidden_size"] // num_heads

    mappings = MODEL_MAPPINGS[mapping_model_name]
    transpose_keys = MODEL_TRANSPOSE_KEYS[mapping_model_name]

    async def _load_shard(shard_path: str) -> dict[str, Any]:
        return await read_safetensors_fsspec(shard_path, fs=fs, sharding_fn=None)

    loop = asyncio.new_event_loop()
    cpu_device = jax.devices("cpu")[0]

    t_pipeline_start = time.time()
    total_tensors = 0
    total_bytes = 0

    try:
        for i, shard_file in enumerate(shard_files):
            t_shard_start = time.time()
            shard_path = os.path.join(remote_path, shard_file)

            # 1. Download shard
            with jax.default_device(cpu_device):
                shard_dict = loop.run_until_complete(_load_shard(shard_path))

            shard_tensors = len(shard_dict)
            shard_bytes = sum(v.nbytes for v in shard_dict.values())
            total_tensors += shard_tensors
            total_bytes += shard_bytes

            # 2. Reshape attention projections (HF 2D → Levanter 3D)
            _reshape_attention_tensors(shard_dict, num_heads, num_kv_heads, head_dim)

            # 3. Convert to NNX
            nnx_state = levanter_state_dict_to_nnx_state_on_cpu(shard_dict)

            # 4. Inject into HBM
            sync_weights_fn(
                nnx_state,
                mappings=mappings,
                transpose_keys=transpose_keys,
                reshard_fn=None,
            )

            # 5. Free host RAM
            del shard_dict, nnx_state

            t_shard = time.time() - t_shard_start
            _iris_emit(
                "I",
                "vllm.inprocess",
                f"Shard {i + 1}/{len(shard_files)} injected: "
                f"{shard_tensors} tensors, {shard_bytes / (1024**3):.2f} GiB in {t_shard:.1f}s",
            )
    finally:
        loop.close()

    t_total = time.time() - t_pipeline_start
    total_gib = total_bytes / (1024**3)
    throughput = (total_bytes / (1024**2)) / t_total if t_total > 0 else 0
    _record_event(events, f"Streamed {total_tensors} tensors ({total_gib:.1f} GiB) across {len(shard_files)} shards")
    _iris_emit(
        "I",
        "vllm.inprocess",
        f"WEIGHT PIPELINE COMPLETE (streaming): {t_total:.1f}s, "
        f"{total_tensors} tensors, {total_gib:.1f} GiB, {throughput:.0f} MiB/s aggregate",
    )
    return t_total


def _reshape_attention_tensors(
    state_dict: dict[str, Any],
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> None:
    """Reshape HF 2D attention projections to Levanter 3D format in-place."""
    import numpy as np

    for key in list(state_dict.keys()):
        if "q_proj" in key and "bias" not in key:
            state_dict[key] = np.asarray(state_dict[key]).reshape(num_heads, head_dim, -1)
        elif ("k_proj" in key or "v_proj" in key) and "bias" not in key:
            state_dict[key] = np.asarray(state_dict[key]).reshape(num_kv_heads, head_dim, -1)
        elif "o_proj" in key and "bias" not in key:
            state_dict[key] = np.asarray(state_dict[key]).reshape(-1, num_heads, head_dim)


def _is_object_store_path(path: str) -> bool:
    scheme = urlparse(path).scheme
    return scheme in {"gs", "s3"}


def _unsupported_extra_cli_args(extra_cli_args: list[str] | None) -> list[str]:
    if not extra_cli_args:
        return []

    unsupported: list[str] = []
    i = 0
    while i < len(extra_cli_args):
        arg = extra_cli_args[i]

        if arg in _SUPPORTED_ENGINE_CLI_FLAGS_WITH_VALUE:
            if i + 1 >= len(extra_cli_args):
                unsupported.append(arg)
                break
            i += 2
            continue

        if arg in _SUPPORTED_ENGINE_CLI_FLAGS_NO_VALUE:
            i += 1
            continue

        unsupported.append(arg)
        i += 1

    return unsupported


def _discover_safetensor_shards(fs: AbstractFileSystem, remote_path: str) -> list[str]:
    index_path = os.path.join(remote_path, SAFE_TENSORS_INDEX_NAME)
    if fs.exists(index_path):
        with fs.open(index_path, "r") as f:
            index_payload = json.load(f)
        shard_files = list(dict.fromkeys(index_payload["weight_map"].values()))
        if shard_files:
            return shard_files

    single_file_path = os.path.join(remote_path, SAFE_TENSORS_MODEL)
    if fs.exists(single_file_path):
        return [SAFE_TENSORS_MODEL]

    raise FileNotFoundError(f"No safetensors checkpoint found under {remote_path}")


def _resolve_mapping_model_name(model: ModelConfig, model_name_or_path: str) -> str | None:
    # First preference: explicit model.name, which is how RL currently keys mappings.
    if _mapping_exists(model.name):
        return model.name

    # Second: infer from config.json in the remote checkpoint.
    config_payload = _try_read_model_config(model.path)
    if config_payload is None:
        logger.info("Could not read remote config.json while resolving mapping for %s", model_name_or_path)
        return None

    inferred_name = _infer_mapping_from_model_config(config_payload)
    if inferred_name is None:
        return None

    if _mapping_exists(inferred_name):
        return inferred_name

    return None


def _resolve_bootstrap_model_source_for_eligibility(model: ModelConfig) -> tuple[str | None, str]:
    override = _normalize_bootstrap_override(model.engine_kwargs.get(_INPROCESS_BOOTSTRAP_MODEL_KEY))
    if override is not None:
        if _is_object_store_path(override):
            return None, (
                f"{_INPROCESS_BOOTSTRAP_MODEL_KEY} cannot be an object-store URI when using load_format='dummy': "
                f"{override!r}"
            )
        return override, "eligible"

    # When model.path is GCS, always stage local metadata from remote.
    # Don't trust model.name — it may be an arbitrary label (e.g. "smoke-test-model"),
    # not a valid HF repo ID.
    if model.path is not None and _is_object_store_path(model.path):
        if _can_stage_bootstrap_metadata_from_model_path(model.path):
            return "<staged-local-metadata>", "eligible"
        return None, "could not find config.json in remote model path for bootstrap staging"

    if not _is_object_store_path(model.name):
        return model.name, "eligible"

    return None, (
        "could not resolve a non-object-store bootstrap model source; "
        f"set engine_kwargs[{_INPROCESS_BOOTSTRAP_MODEL_KEY!r}] to a local path or HF model id"
    )


def _resolve_bootstrap_model_source_for_start(model: ModelConfig) -> tuple[str, str | None]:
    override = _normalize_bootstrap_override(model.engine_kwargs.get(_INPROCESS_BOOTSTRAP_MODEL_KEY))
    if override is not None:
        if _is_object_store_path(override):
            raise InProcessVllmUnsupportedError(
                f"{_INPROCESS_BOOTSTRAP_MODEL_KEY} cannot be an object-store URI with load_format='dummy': {override!r}"
            )
        return override, None

    # When model.path is GCS, always stage local metadata from remote.
    # Don't trust model.name — it may be an arbitrary label (e.g. "smoke-test-model"),
    # not a valid HF repo ID.
    if model.path is not None and _is_object_store_path(model.path):
        local_dir = _stage_bootstrap_metadata(model.path)
        return local_dir, local_dir

    if not _is_object_store_path(model.name):
        return model.name, None

    raise InProcessVllmUnsupportedError("model.path is required to stage bootstrap metadata for in-process startup")


def _normalize_bootstrap_override(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise InProcessVllmUnsupportedError(
            f"engine_kwargs[{_INPROCESS_BOOTSTRAP_MODEL_KEY!r}] must be a string when provided"
        )
    stripped = value.strip()
    if not stripped:
        raise InProcessVllmUnsupportedError(
            f"engine_kwargs[{_INPROCESS_BOOTSTRAP_MODEL_KEY!r}] cannot be empty when provided"
        )
    return stripped


def _can_stage_bootstrap_metadata_from_model_path(model_path: str) -> bool:
    try:
        fs, remote_path = url_to_fs(model_path)
        config_path = os.path.join(remote_path, "config.json")
        return bool(fs.exists(config_path))
    except Exception:
        logger.exception("Failed to check bootstrap metadata availability under %s", model_path)
        return False


def _stage_bootstrap_metadata(model_path: str) -> str:
    fs, remote_path = url_to_fs(model_path)
    local_dir = tempfile.mkdtemp(prefix="marin-vllm-bootstrap-")
    copied_any = False

    try:
        for filename in _BOOTSTRAP_METADATA_FILENAMES:
            remote_file = os.path.join(remote_path, filename)
            if not fs.exists(remote_file):
                continue

            local_file = os.path.join(local_dir, filename)
            with fs.open(remote_file, "rb") as src:
                payload = src.read()
            with open(local_file, "wb") as dst:
                dst.write(payload)
            copied_any = True

        if not copied_any:
            raise InProcessVllmUnsupportedError(f"No bootstrap metadata files found under {model_path!r}")
        if not os.path.exists(os.path.join(local_dir, "config.json")):
            raise InProcessVllmUnsupportedError(
                f"Missing config.json under {model_path!r}; cannot initialize vLLM with load_format='dummy'"
            )

        return local_dir
    except Exception:
        _cleanup_local_bootstrap_dir(local_dir)
        raise


def _cleanup_local_bootstrap_dir(local_dir: str | None) -> None:
    if local_dir is None:
        return
    try:
        shutil.rmtree(local_dir, ignore_errors=True)
    except Exception:
        logger.exception("Failed to clean up temporary bootstrap directory %s", local_dir)


def _mapping_exists(model_name: str) -> bool:
    return model_name in MODEL_MAPPINGS and model_name in MODEL_TRANSPOSE_KEYS


def _try_read_model_config(model_path: str | None) -> dict[str, Any] | None:
    if model_path is None:
        return None

    try:
        fs, remote_path = url_to_fs(model_path)
        config_path = os.path.join(remote_path, "config.json")
        if not fs.exists(config_path):
            return None
        with fs.open(config_path, "r") as f:
            return json.load(f)
    except Exception:
        logger.exception("Failed to read config.json for model path %s", model_path)
        return None


def _infer_mapping_from_model_config(config_payload: dict[str, Any]) -> str | None:
    architecture_tokens = config_payload.get("architectures") or []
    model_type = config_payload.get("model_type")

    tokens = [str(token).lower() for token in architecture_tokens]
    if model_type is not None:
        tokens.append(str(model_type).lower())

    joined = " ".join(tokens)

    if "qwen2.5" in joined:
        return "Qwen2.5"

    if "qwen" in joined:
        if _mapping_exists(_DEFAULT_QWEN_MAPPING_KEY):
            return _DEFAULT_QWEN_MAPPING_KEY
        return "Qwen2.5"

    if "llama" in joined and _mapping_exists(_DEFAULT_LLAMA_MAPPING_KEY):
        return _DEFAULT_LLAMA_MAPPING_KEY

    return None


def _llm_kwargs(*, bootstrap_model_source: str, model: ModelConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": bootstrap_model_source,
        "load_format": "dummy",
        "trust_remote_code": True,
    }

    for key in (
        "max_model_len",
        "gpu_memory_utilization",
        "model_loader_extra_config",
    ):
        value = model.engine_kwargs.get(key)
        if value is not None:
            kwargs[key] = value

    return kwargs


def _wait_for_models_endpoint(server_url: str, timeout_seconds: int, serve_thread: threading.Thread) -> str:
    endpoint = f"{server_url}/models"
    start_time = time.time()

    while True:
        if not serve_thread.is_alive():
            raise RuntimeError("In-process vLLM server thread exited before /models became ready")

        try:
            response = requests.get(endpoint, timeout=5)
            if response.status_code == 200:
                payload = response.json()
                return _extract_first_model_id(payload)
        except requests.ConnectionError:
            pass
        except requests.Timeout:
            pass

        if time.time() - start_time > timeout_seconds:
            raise TimeoutError(f"In-process vLLM server did not become ready within {timeout_seconds}s")

        time.sleep(2)


def _extract_first_model_id(payload: dict[str, Any]) -> str:
    data = payload.get("data", [])
    if not data:
        raise RuntimeError(f"No model data returned from /models: {str(payload)[:2000]}")
    model_id = data[0].get("id")
    if not model_id:
        raise RuntimeError(f"Missing model id in /models payload: {str(payload)[:2000]}")
    return str(model_id)


def _import_inprocess_vllm_symbols() -> tuple[Any, Any]:
    try:
        from vllm import LLM
        import uvicorn
    except Exception as exc:
        raise InProcessVllmUnsupportedError("vLLM in-process API imports are unavailable") from exc

    return LLM, uvicorn


def _create_inprocess_openai_app(llm: Any, model_name: str, bootstrap_model_source: str) -> Any:
    """Create a minimal OpenAI-compatible API around LLM.generate().

    This avoids vLLM's build_app() which spawns child processes that fail
    on TPU with "TPU already in use by pid 1". Instead we wrap LLM.generate()
    in a simple FastAPI app running in the same process.

    Exposes /v1/models, /v1/completions, and /v1/chat/completions.
    """
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    app = FastAPI()
    tokenizer = AutoTokenizer.from_pretrained(bootstrap_model_source)

    def _sampling_params_from_request(request: dict) -> SamplingParams:
        return SamplingParams(
            max_tokens=request.get("max_tokens", 128),
            temperature=request.get("temperature", 1.0),
            top_p=request.get("top_p", 1.0),
            n=request.get("n", 1),
            logprobs=request.get("logprobs"),
        )

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [{"id": model_name, "object": "model", "owned_by": "marin"}],
        }

    @app.post("/v1/completions")
    def completions(request: dict):
        prompt = request.get("prompt", "")
        echo = request.get("echo", False)
        params = _sampling_params_from_request(request)

        try:
            outputs = llm.generate([prompt], params)
        except Exception as exc:
            return JSONResponse(
                status_code=500,
                content={"error": {"message": str(exc), "type": "Internal Server Error"}},
            )

        output = outputs[0]
        choices = []
        for i, completion in enumerate(output.outputs):
            text = completion.text
            if echo:
                text = prompt + text
            choice = {
                "text": text,
                "index": i,
                "finish_reason": completion.finish_reason,
            }
            logprobs_req = request.get("logprobs")
            if logprobs_req is not None and completion.logprobs:
                choice["logprobs"] = {
                    "tokens": [lp.decoded_token for lp in completion.logprobs],
                    "token_logprobs": [lp.logprob for lp in completion.logprobs],
                }
            choices.append(choice)

        prompt_tokens = len(output.prompt_token_ids)
        completion_tokens = sum(len(c.token_ids) for c in output.outputs)

        return {
            "id": f"cmpl-inprocess-{id(output)}",
            "object": "text_completion",
            "model": model_name,
            "choices": choices,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    @app.post("/v1/chat/completions")
    def chat_completions(request: dict):
        messages = request.get("messages", [])
        params = _sampling_params_from_request(request)

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        try:
            outputs = llm.generate([prompt], params)
        except Exception as exc:
            return JSONResponse(
                status_code=500,
                content={"error": {"message": str(exc), "type": "Internal Server Error"}},
            )

        output = outputs[0]
        choices = []
        for i, completion in enumerate(output.outputs):
            choices.append(
                {
                    "index": i,
                    "message": {"role": "assistant", "content": completion.text},
                    "finish_reason": completion.finish_reason,
                }
            )

        prompt_tokens = len(output.prompt_token_ids)
        completion_tokens = sum(len(c.token_ids) for c in output.outputs)

        return {
            "id": f"chatcmpl-inprocess-{id(output)}",
            "object": "chat.completion",
            "model": model_name,
            "choices": choices,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    return app


def _resolve_sync_weights_callable(llm: Any) -> Any:
    driver_worker = getattr(getattr(llm.llm_engine, "model_executor", None), "driver_worker", None)
    sync_weights = getattr(driver_worker, "sync_weights", None)
    if callable(sync_weights):
        return sync_weights
    raise InProcessVllmUnsupportedError(
        "driver_worker.sync_weights is unavailable in this vLLM build; in-process weight injection is unsupported."
    )


def _record_event(events: list[str], message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    events.append(f"[{timestamp}] {message}")
    logger.info(message)


def _shutdown_llm(llm: Any) -> None:
    try:
        engine = getattr(llm, "llm_engine", None)
        if engine is None:
            return

        shutdown = getattr(engine, "shutdown", None)
        if callable(shutdown):
            shutdown()
    except Exception:
        logger.exception("Failed to shut down in-process vLLM engine cleanly")
