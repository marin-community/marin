# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import faulthandler
import functools
import json
import logging
import os
import sys
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import jax
import threading
import time
from iris.marin_fs import marin_prefix, url_to_fs

from levanter.compat.fsspec_safetensor import read_safetensors_fsspec

from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.inference.vllm_inprocess import (
    InProcessVllmUnsupportedError,
    _cleanup_local_bootstrap_dir,
    _discover_safetensor_shards,
    _extract_served_model_name,
    _iris_emit,
    _record_event,
    _reshape_attention_tensors,
    _resolve_bootstrap_model_source_for_start,
    _wait_for_models_endpoint,
)
from marin.inference.vllm_tpu_bootstrap_patch import install_marin_fast_tpu_bootstrap_patch
from marin.rl.environments.inference_ctx.startup_debug import install_dummy_init_progress_instrumentation
from marin.rl.environments.inference_ctx.async_vllm import serialize_state_dict_for_rpc

logger = logging.getLogger(__name__)

_WORKER_EXTENSION_CLS = "marin.rl.environments.inference_ctx.inflight.worker.WorkerExtension"
_STARTUP_TIMING_PREFIX = "[marin-vllm-startup]"
_DIRECT_SAMPLING_KEY_ENV = "MARIN_VLLM_DIRECT_SAMPLING_KEY"


@dataclass
class AsyncVllmRuntime:
    serve_thread: threading.Thread
    server_url: str
    port: int
    model_id: str | None = None
    bootstrap_local_dir: str | None = None
    events: list[str] = field(default_factory=list)
    server: Any | None = None
    app: Any | None = None
    startup_error: BaseException | None = None
    startup_traceback: str | None = None

    def logs_tail(self, *, max_lines: int = 200) -> str:
        if not self.events:
            return "<no async vLLM startup events captured>"
        return "\n".join(self.events[-max_lines:])


@dataclass
class _AsyncServerState:
    runtime: AsyncVllmRuntime
    model: ModelConfig
    model_name_or_path: str
    mapping_model_name: str
    host: str
    port: int
    extra_cli_args: list[str] | None


def start_async_vllm_server(
    *,
    model: ModelConfig,
    model_name_or_path: str,
    mapping_model_name: str,
    host: str,
    port: int,
    timeout_seconds: int,
    extra_cli_args: list[str] | None,
) -> AsyncVllmRuntime:
    """Start a native async vLLM server with Marin's fast-loading path."""
    if model.path is None:
        raise InProcessVllmUnsupportedError(
            "async native startup requires model.path for object-store checkpoint loading"
        )

    runtime = AsyncVllmRuntime(
        serve_thread=threading.Thread(target=lambda: None),
        server_url=f"http://{host}:{port}/v1",
        port=port,
    )
    state = _AsyncServerState(
        runtime=runtime,
        model=model,
        model_name_or_path=model_name_or_path,
        mapping_model_name=mapping_model_name,
        host=host,
        port=port,
        extra_cli_args=extra_cli_args,
    )

    runtime.serve_thread = threading.Thread(
        target=_run_async_server_thread,
        args=(state,),
        daemon=True,
        name=f"vllm-async-{port}",
    )
    runtime.serve_thread.start()

    try:
        runtime.model_id = _wait_for_models_endpoint(runtime.server_url, timeout_seconds, runtime.serve_thread)
        _record_event(
            runtime.events,
            f"Async OpenAI endpoint ready at {runtime.server_url} with model_id={runtime.model_id!r}",
        )
        return runtime
    except Exception:
        if runtime.server is not None:
            runtime.server.should_exit = True
        runtime.serve_thread.join(timeout=10)
        _cleanup_local_bootstrap_dir(runtime.bootstrap_local_dir)
        if runtime.startup_error is not None:
            raise RuntimeError(runtime.startup_traceback or str(runtime.startup_error)) from runtime.startup_error
        raise


def stop_async_vllm_server(runtime: AsyncVllmRuntime) -> None:
    """Stop the async vLLM server and clean up temporary bootstrap state."""
    if runtime.server is not None:
        runtime.server.should_exit = True
    runtime.serve_thread.join(timeout=20)
    _cleanup_local_bootstrap_dir(runtime.bootstrap_local_dir)


def _run_async_server_thread(state: _AsyncServerState) -> None:
    try:
        asyncio.run(_run_async_server(state))
    except BaseException as exc:
        state.runtime.startup_error = exc
        state.runtime.startup_traceback = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        _record_event(state.runtime.events, f"Async native vLLM server failed: {type(exc).__name__}: {exc}")
        _iris_emit("E", "vllm.async", f"Async native vLLM server failed: {type(exc).__name__}: {exc}")


async def _run_async_server(state: _AsyncServerState) -> None:
    model = state.model
    runtime = state.runtime
    _configure_async_vllm_environment()
    install_marin_fast_tpu_bootstrap_patch(emit=_emit_startup_trace if _startup_timing_enabled() else None)
    _install_early_async_startup_instrumentation()
    bootstrap_model_source, bootstrap_local_dir = _resolve_bootstrap_model_source_for_start(model)
    runtime.bootstrap_local_dir = bootstrap_local_dir
    _record_event(
        runtime.events,
        f"Using bootstrap model source {bootstrap_model_source!r} for async dummy initialization",
    )

    (
        async_engine_args_cls,
        async_llm_cls,
        build_app,
        flexible_arg_parser_cls,
        init_app_state,
        make_arg_parser,
        uvicorn_module,
        validate_parsed_serve_args,
    ) = _import_async_vllm_symbols()

    served_name = _extract_served_model_name(state.extra_cli_args) or state.model_name_or_path
    cli_args = _build_openai_server_cli_args(
        model=model,
        bootstrap_model_source=bootstrap_model_source,
        host=state.host,
        port=state.port,
        served_name=served_name,
    )

    parser = flexible_arg_parser_cls(description="Marin async native vLLM server")
    parser = make_arg_parser(parser)
    args = parser.parse_args(cli_args)
    validate_parsed_serve_args(args)
    engine_args = async_engine_args_cls.from_cli_args(args)
    _log_async_engine_startup(
        engine_args=engine_args,
        requested_model_name_or_path=state.model_name_or_path,
        events=runtime.events,
    )

    # Create engine in-process (no subprocess). This matches the RL path
    # (inflight/worker.py:492) which works reliably on TPU.
    # build_async_engine_client_from_engine_args() spawns an engine subprocess
    # that hangs during vllm_get_model() on TPU — even with
    # VLLM_ENABLE_V1_MULTIPROCESSING=0, that function still forks.
    t_engine_start = time.time()
    engine = async_llm_cls.from_engine_args(
        engine_args=engine_args,
        start_engine_loop=False,
    )
    t_engine = time.time() - t_engine_start
    _record_event(runtime.events, f"Created AsyncLLM engine (in-process) for {state.model_name_or_path}")
    _iris_emit("I", "vllm.async", f"AsyncLLM engine created in-process in {t_engine:.1f}s")

    try:
        await _load_and_inject_streaming_async(
            model_path=model.path,
            engine=engine,
            mapping_model_name=state.mapping_model_name,
            bootstrap_model_source=bootstrap_model_source,
            events=runtime.events,
        )
        await engine.reset_prefix_cache()

        supported_tasks = await engine.get_supported_tasks()
        app = build_app(args, supported_tasks)
        await init_app_state(engine, app.state, args, supported_tasks)

        runtime.app = app
        server = uvicorn_module.Server(
            uvicorn_module.Config(
                app,
                host=state.host,
                port=state.port,
                log_level="warning",
            )
        )
        app.state.server = server
        runtime.server = server
        _iris_emit("I", "vllm.async", f"Async native OpenAI server starting on {runtime.server_url}")

        watchdog_task = asyncio.create_task(_watchdog_loop(server, engine))
        try:
            await server.serve()
        finally:
            watchdog_task.cancel()
            await _await_task_cancellation(watchdog_task)
    finally:
        engine.shutdown()


def _build_openai_server_cli_args(
    *,
    model: ModelConfig,
    bootstrap_model_source: str,
    host: str,
    port: int,
    served_name: str,
) -> list[str]:
    cli_args = [
        "--model",
        bootstrap_model_source,
        "--host",
        host,
        "--port",
        str(port),
        "--served-model-name",
        served_name,
        "--trust-remote-code",
        "--disable-frontend-multiprocessing",
        "--load-format",
        "dummy",
        "--worker-extension-cls",
        _WORKER_EXTENSION_CLS,
    ]

    max_model_len = model.engine_kwargs.get("max_model_len")
    if max_model_len is not None:
        cli_args.extend(["--max-model-len", str(max_model_len)])

    gpu_memory_utilization = model.engine_kwargs.get("gpu_memory_utilization")
    if gpu_memory_utilization is not None:
        cli_args.extend(["--gpu-memory-utilization", str(gpu_memory_utilization)])

    model_loader_extra_config = model.engine_kwargs.get("model_loader_extra_config")
    if model_loader_extra_config is not None:
        cli_args.extend(["--model-loader-extra-config", json.dumps(model_loader_extra_config)])

    tensor_parallel_size = model.engine_kwargs.get("tensor_parallel_size")
    if tensor_parallel_size is not None:
        cli_args.extend(["--tensor-parallel-size", str(tensor_parallel_size)])

    if model.engine_kwargs.get("enforce_eager"):
        cli_args.append("--enforce-eager")

    return cli_args


async def _load_and_inject_streaming_async(
    *,
    model_path: str,
    engine: Any,
    mapping_model_name: str,
    bootstrap_model_source: str,
    events: list[str],
) -> float:
    """Stream shards from remote storage and inject them into AsyncLLM workers."""
    fs, remote_path = url_to_fs(model_path)
    shard_files = _discover_safetensor_shards(fs, remote_path)

    config_path = os.path.join(bootstrap_model_source, "config.json")
    with open(config_path) as f:
        model_config = json.load(f)
    num_heads = model_config["num_attention_heads"]
    num_kv_heads = model_config.get("num_key_value_heads", num_heads)
    head_dim = model_config["hidden_size"] // num_heads

    cpu_device = jax.devices("cpu")[0]
    t_pipeline_start = time.time()
    total_tensors = 0
    total_bytes = 0

    for i, shard_file in enumerate(shard_files):
        t_shard_start = time.time()
        shard_path = os.path.join(remote_path, shard_file)

        with jax.default_device(cpu_device):
            shard_dict = await read_safetensors_fsspec(shard_path, fs=fs, sharding_fn=None)

        shard_tensors = len(shard_dict)
        shard_bytes = sum(value.nbytes for value in shard_dict.values())
        total_tensors += shard_tensors
        total_bytes += shard_bytes

        _reshape_attention_tensors(shard_dict, num_heads, num_kv_heads, head_dim)
        serialized_shard = serialize_state_dict_for_rpc(shard_dict)
        await engine.engine_core.collective_rpc_async(
            "update_weight",
            args=(serialized_shard, mapping_model_name),
        )

        del shard_dict, serialized_shard

        t_shard = time.time() - t_shard_start
        _iris_emit(
            "I",
            "vllm.async",
            f"Shard {i + 1}/{len(shard_files)} injected via async engine: "
            f"{shard_tensors} tensors, {shard_bytes / (1024**3):.2f} GiB in {t_shard:.1f}s",
        )

    t_total = time.time() - t_pipeline_start
    total_gib = total_bytes / (1024**3)
    throughput = (total_bytes / (1024**2)) / t_total if t_total > 0 else 0
    _record_event(events, f"Streamed {total_tensors} tensors ({total_gib:.1f} GiB) across {len(shard_files)} shards")
    _iris_emit(
        "I",
        "vllm.async",
        f"WEIGHT PIPELINE COMPLETE (async streaming): {t_total:.1f}s, "
        f"{total_tensors} tensors, {total_gib:.1f} GiB, {throughput:.0f} MiB/s aggregate",
    )
    return t_total


def _import_async_vllm_symbols() -> tuple[Any, ...]:
    try:
        from vllm import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM
        import uvicorn
        from vllm.entrypoints.openai.api_server import (
            build_app,
            init_app_state,
        )
        from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
        from vllm.utils.argparse_utils import FlexibleArgumentParser
    except Exception as exc:
        raise InProcessVllmUnsupportedError("vLLM async API imports are unavailable") from exc

    return (
        AsyncEngineArgs,
        AsyncLLM,
        build_app,
        FlexibleArgumentParser,
        init_app_state,
        make_arg_parser,
        uvicorn,
        validate_parsed_serve_args,
    )


def _startup_timing_enabled() -> bool:
    return os.environ.get("MARIN_VLLM_STARTUP_TIMING") == "1"


def _emit_startup_trace(message: str) -> None:
    print(f"{_STARTUP_TIMING_PREFIX} {message}", file=sys.stderr, flush=True)


def _enable_early_startup_faulthandler() -> None:
    if os.environ.get("MARIN_VLLM_STARTUP_FAULTHANDLER") != "1":
        return

    timeout = int(os.environ.get("MARIN_VLLM_STARTUP_FAULTHANDLER_SECS", "300"))
    try:
        faulthandler.cancel_dump_traceback_later()
    except Exception:
        pass
    faulthandler.dump_traceback_later(timeout, repeat=True, file=sys.stderr)
    _emit_startup_trace(f"enabled early startup faulthandler every {timeout}s pid={os.getpid()}")


def _use_direct_sampling_key() -> bool:
    return os.environ.get(_DIRECT_SAMPLING_KEY_ENV) == "1"


def _wrap_upstream_method_with_timing(cls: type[Any], method_name: str, *, label: str | None = None) -> None:
    original = getattr(cls, method_name, None)
    if original is None:
        return
    if getattr(original, "_marin_startup_timed", False):
        return
    original_callable = cast(Callable[..., Any], original)

    effective_label = label or f"{cls.__name__}.{method_name}"

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        _emit_startup_trace(f"START {effective_label} pid={os.getpid()}")
        t_start = time.perf_counter()
        try:
            result = original_callable(*args, **kwargs)
        except Exception:
            _emit_startup_trace(f"FAIL {effective_label} in {time.perf_counter() - t_start:.2f}s pid={os.getpid()}")
            raise
        _emit_startup_trace(f"END {effective_label} in {time.perf_counter() - t_start:.2f}s pid={os.getpid()}")
        return result

    wrapped._marin_startup_timed = True
    setattr(cls, method_name, wrapped)


def _wrap_upstream_module_function_with_timing(module: Any, function_name: str, *, label: str | None = None) -> None:
    original = getattr(module, function_name, None)
    if original is None:
        return
    if getattr(original, "_marin_startup_timed", False):
        return
    original_callable = cast(Callable[..., Any], original)

    effective_label = label or f"{module.__name__}.{function_name}"

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        _emit_startup_trace(f"START {effective_label} pid={os.getpid()}")
        t_start = time.perf_counter()
        try:
            result = original_callable(*args, **kwargs)
        except Exception:
            _emit_startup_trace(f"FAIL {effective_label} in {time.perf_counter() - t_start:.2f}s pid={os.getpid()}")
            raise
        _emit_startup_trace(f"END {effective_label} in {time.perf_counter() - t_start:.2f}s pid={os.getpid()}")
        return result

    wrapped._marin_startup_timed = True
    setattr(module, function_name, wrapped)


def _install_upstream_delegate_method(cls: type[Any], method_name: str, *, label: str | None = None) -> None:
    if method_name in cls.__dict__:
        _wrap_upstream_method_with_timing(cls, method_name, label=label)
        return

    effective_label = label or f"{cls.__name__}.{method_name}"

    def delegated(self, *args, **kwargs):
        _emit_startup_trace(f"START {effective_label} pid={os.getpid()}")
        t_start = time.perf_counter()
        try:
            result = getattr(self.worker, method_name)(*args, **kwargs)
        except Exception:
            _emit_startup_trace(f"FAIL {effective_label} in {time.perf_counter() - t_start:.2f}s pid={os.getpid()}")
            raise
        _emit_startup_trace(f"END {effective_label} in {time.perf_counter() - t_start:.2f}s pid={os.getpid()}")
        return result

    delegated.__name__ = method_name
    delegated.__qualname__ = f"{cls.__qualname__}.{method_name}"
    delegated._marin_startup_timed = True
    setattr(cls, method_name, delegated)


def _install_detailed_upstream_tpu_model_runner_load_model(tpu_runner_module: Any) -> None:
    TPUModelRunner = tpu_runner_module.TPUModelRunner
    original = getattr(TPUModelRunner, "load_model", None)
    if original is None:
        return
    if getattr(original, "_marin_startup_timed", False):
        return

    @functools.wraps(original)
    def wrapped(self):
        _emit_startup_trace(f"START TPUModelRunner.load_model pid={os.getpid()}")
        t_total = time.perf_counter()
        try:
            _emit_startup_trace(f"START TPUModelRunner.load_model.get_model_tuple pid={os.getpid()}")
            t_get_model = time.perf_counter()
            try:
                model_outputs = tpu_runner_module.get_model(
                    self.vllm_config,
                    self.rng_key,
                    self.mesh,
                )
                tuple_len = len(model_outputs)
                _emit_startup_trace(f"TPUModelRunner.load_model.get_model_tuple length={tuple_len} pid={os.getpid()}")
                if tuple_len == 8:
                    (
                        self.model_fn,
                        self.compute_logits_fn,
                        self.pooler_fn,
                        self.combine_hidden_states_fn,
                        multimodal_fns,
                        self.state,
                        self.lora_manager,
                        self.model,
                    ) = model_outputs
                elif tuple_len == 7:
                    (
                        self.model_fn,
                        self.compute_logits_fn,
                        self.pooler_fn,
                        multimodal_fns,
                        self.state,
                        self.lora_manager,
                        self.model,
                    ) = model_outputs
                    self.combine_hidden_states_fn = None
                else:
                    raise ValueError(f"unexpected get_model tuple length: {tuple_len}")
            except Exception:
                _emit_startup_trace(
                    f"FAIL TPUModelRunner.load_model.get_model_tuple "
                    f"in {time.perf_counter() - t_get_model:.2f}s pid={os.getpid()}"
                )
                raise
            _emit_startup_trace(
                f"END TPUModelRunner.load_model.get_model_tuple "
                f"in {time.perf_counter() - t_get_model:.2f}s pid={os.getpid()}"
            )

            _emit_startup_trace(f"START TPUModelRunner.load_model.multimodal_bind pid={os.getpid()}")
            t_multimodal_bind = time.perf_counter()
            try:
                multimodal_fns = multimodal_fns or {}
                self.precompile_vision_encoder_fn = multimodal_fns.get("precompile_vision_encoder_fn", None)
                self.embed_multimodal_fn = multimodal_fns.get("embed_multimodal_fn", None)
                self.embed_input_ids_fn = multimodal_fns.get("embed_input_ids_fn", None)
                self.get_mrope_input_positions_fn = multimodal_fns.get("get_mrope_input_positions_fn", None)
            except Exception:
                _emit_startup_trace(
                    f"FAIL TPUModelRunner.load_model.multimodal_bind "
                    f"in {time.perf_counter() - t_multimodal_bind:.2f}s pid={os.getpid()}"
                )
                raise
            _emit_startup_trace(
                f"END TPUModelRunner.load_model.multimodal_bind "
                f"in {time.perf_counter() - t_multimodal_bind:.2f}s pid={os.getpid()}"
            )

            if self.drafter is not None:
                _emit_startup_trace(f"START TPUModelRunner.load_model.drafter pid={os.getpid()}")
                t_drafter = time.perf_counter()
                try:
                    self.drafter.load_model(self.state)
                except Exception:
                    _emit_startup_trace(
                        f"FAIL TPUModelRunner.load_model.drafter "
                        f"in {time.perf_counter() - t_drafter:.2f}s pid={os.getpid()}"
                    )
                    raise
                _emit_startup_trace(
                    f"END TPUModelRunner.load_model.drafter "
                    f"in {time.perf_counter() - t_drafter:.2f}s pid={os.getpid()}"
                )

            _emit_startup_trace(f"START TPUModelRunner.load_model.rng_params_for_sampling pid={os.getpid()}")
            t_rng = time.perf_counter()
            try:
                if _use_direct_sampling_key():
                    _emit_startup_trace(
                        f"TPUModelRunner.load_model.rng_params_for_sampling mode=direct_key pid={os.getpid()}"
                    )
                    self.rng_params_for_sampling = tpu_runner_module.jax.random.key(self.model_config.seed)
                else:
                    _emit_startup_trace(
                        f"TPUModelRunner.load_model.rng_params_for_sampling mode=nnx_rngs pid={os.getpid()}"
                    )
                    self.rng_params_for_sampling = tpu_runner_module.nnx.Rngs(
                        tpu_runner_module.jax.random.key(self.model_config.seed)
                    ).params()
            except Exception:
                _emit_startup_trace(
                    f"FAIL TPUModelRunner.load_model.rng_params_for_sampling "
                    f"in {time.perf_counter() - t_rng:.2f}s pid={os.getpid()}"
                )
                raise
            _emit_startup_trace(
                f"END TPUModelRunner.load_model.rng_params_for_sampling "
                f"in {time.perf_counter() - t_rng:.2f}s pid={os.getpid()}"
            )

            _emit_startup_trace(f"START TPUModelRunner.load_model.is_multimodal_model pid={os.getpid()}")
            t_is_multimodal = time.perf_counter()
            try:
                self.is_multimodal_model = (
                    self.model_config.is_multimodal_model
                    and self.embed_multimodal_fn is not None
                    and hasattr(self.model_config.hf_config, "architectures")
                )
            except Exception:
                _emit_startup_trace(
                    f"FAIL TPUModelRunner.load_model.is_multimodal_model "
                    f"in {time.perf_counter() - t_is_multimodal:.2f}s pid={os.getpid()}"
                )
                raise
            _emit_startup_trace(
                f"END TPUModelRunner.load_model.is_multimodal_model "
                f"in {time.perf_counter() - t_is_multimodal:.2f}s pid={os.getpid()}"
            )

            _emit_startup_trace(f"START TPUModelRunner.load_model.log_init_model pid={os.getpid()}")
            t_log = time.perf_counter()
            try:
                tpu_runner_module.logger.info(
                    "Init model | hbm=%sGiB",
                    tpu_runner_module.common_utils.hbm_usage_gb(self.devices),
                )
            except Exception:
                _emit_startup_trace(
                    f"FAIL TPUModelRunner.load_model.log_init_model "
                    f"in {time.perf_counter() - t_log:.2f}s pid={os.getpid()}"
                )
                raise
            _emit_startup_trace(
                f"END TPUModelRunner.load_model.log_init_model "
                f"in {time.perf_counter() - t_log:.2f}s pid={os.getpid()}"
            )

        except Exception:
            _emit_startup_trace(
                f"FAIL TPUModelRunner.load_model in {time.perf_counter() - t_total:.2f}s pid={os.getpid()}"
            )
            raise

        _emit_startup_trace(f"END TPUModelRunner.load_model in {time.perf_counter() - t_total:.2f}s pid={os.getpid()}")

    wrapped._marin_startup_timed = True
    TPUModelRunner.load_model = wrapped
    _emit_startup_trace(f"installed detailed early TPUModelRunner.load_model instrumentation pid={os.getpid()}")


def _install_detailed_upstream_kv_cache_manager_initialize_kv_cache(kv_cache_manager_module: Any) -> None:
    KVCacheManager = kv_cache_manager_module.KVCacheManager
    original = getattr(KVCacheManager, "initialize_kv_cache", None)
    if original is None:
        return
    if getattr(original, "_marin_startup_timed", False):
        return

    @functools.wraps(original)
    def wrapped(self, kv_cache_config):
        _emit_startup_trace(f"START KVCacheManager.initialize_kv_cache pid={os.getpid()}")
        t_total = time.perf_counter()
        try:
            self.maybe_reinitialize_input_batch(kv_cache_config)

            if not kv_cache_config.kv_cache_groups:
                _emit_startup_trace(
                    f"END KVCacheManager.initialize_kv_cache in {time.perf_counter() - t_total:.2f}s "
                    f"pid={os.getpid()} empty_kv_cache_groups=1"
                )
                return

            _emit_startup_trace(f"START KVCacheManager.initialize_kv_cache.layer_name_to_spec pid={os.getpid()}")
            t_layer_name_to_spec = time.perf_counter()
            layer_name_to_spec = {}
            for group in kv_cache_config.kv_cache_groups:
                group_spec = group.kv_cache_spec
                if hasattr(group_spec, "kv_cache_specs"):
                    for layer_name in group.layer_names:
                        layer_name_to_spec[layer_name] = group_spec.kv_cache_specs[layer_name]
                else:
                    for layer_name in group.layer_names:
                        layer_name_to_spec[layer_name] = group.kv_cache_spec
            _emit_startup_trace(
                f"END KVCacheManager.initialize_kv_cache.layer_name_to_spec "
                f"in {time.perf_counter() - t_layer_name_to_spec:.2f}s pid={os.getpid()} "
                f"groups={len(kv_cache_config.kv_cache_groups)} layer_specs={len(layer_name_to_spec)}"
            )

            kv_caches = self.runner.kv_caches
            num_blocks_list = []
            num_tensors = len(kv_cache_config.kv_cache_tensors)
            for i, kv_cache_tensor in enumerate(kv_cache_config.kv_cache_tensors):
                layer_name = kv_cache_tensor.shared_by[0]
                layer_spec = layer_name_to_spec[layer_name]

                page_size_bytes = layer_spec.page_size_bytes
                assert kv_cache_tensor.size % page_size_bytes == 0
                num_blocks = kv_cache_tensor.size // page_size_bytes
                dp_size = self.runner.vllm_config.sharding_config.total_dp_size
                num_blocks = (num_blocks // dp_size) * dp_size
                if self.use_mla:
                    head_size = (
                        self.runner.model_config.hf_config.kv_lora_rank
                        + self.runner.model_config.hf_config.qk_rope_head_dim
                    )
                else:
                    head_size = layer_spec.head_size

                _emit_startup_trace(
                    f"KVCacheManager.initialize_kv_cache.tensor index={i + 1}/{num_tensors} "
                    f"layer={layer_name} num_blocks={num_blocks} block_size={layer_spec.block_size} "
                    f"num_kv_heads={layer_spec.num_kv_heads} head_size={head_size} pid={os.getpid()}"
                )
                _emit_startup_trace(
                    f"START KVCacheManager.initialize_kv_cache.tensor_create index={i + 1}/{num_tensors} "
                    f"pid={os.getpid()}"
                )
                t_tensor_create = time.perf_counter()
                kv_cache = kv_cache_manager_module.create_kv_caches(
                    num_blocks=num_blocks,
                    block_size=layer_spec.block_size,
                    num_kv_heads=layer_spec.num_kv_heads,
                    head_size=head_size,
                    mesh=self.runner.mesh,
                    layer_names=[f"kv_cache_tensor.{i}"],
                    cache_dtype=kv_cache_manager_module.t2j_dtype(layer_spec.dtype),
                    use_mla=self.use_mla,
                )[0]
                _emit_startup_trace(
                    f"END KVCacheManager.initialize_kv_cache.tensor_create index={i + 1}/{num_tensors} "
                    f"in {time.perf_counter() - t_tensor_create:.2f}s pid={os.getpid()}"
                )

                kv_caches.append(kv_cache)
                num_blocks_list.append(num_blocks)
                for shared_layer_name in kv_cache_tensor.shared_by:
                    self.runner.layer_name_to_kvcache_index[shared_layer_name] = i

            if self.shared_kv_cache_layers:
                _emit_startup_trace(f"START KVCacheManager.initialize_kv_cache.shared_kv_layers pid={os.getpid()}")
                t_shared = time.perf_counter()
                for layer_name, target_layer_name in self.shared_kv_cache_layers.items():
                    self.runner.layer_name_to_kvcache_index[layer_name] = self.runner.layer_name_to_kvcache_index[
                        target_layer_name
                    ]
                _emit_startup_trace(
                    f"END KVCacheManager.initialize_kv_cache.shared_kv_layers "
                    f"in {time.perf_counter() - t_shared:.2f}s pid={os.getpid()} "
                    f"count={len(self.shared_kv_cache_layers)}"
                )

            kv_cache_manager_module.logger.info(
                f"Init kv-cache | "
                f"num_layers={len(kv_caches)} | "
                f"shape=(num_blocks, {kv_caches[0].shape[1:]}) | "
                f"num_blocks={num_blocks_list} | "
                f"sharding={kv_caches[0].sharding} | "
                f"dtype={kv_caches[0].dtype} | "
                f"hbm={kv_cache_manager_module.utils.hbm_usage_gb(self.runner.mesh.devices.flatten())}Gb"
            )
        except Exception:
            _emit_startup_trace(
                f"FAIL KVCacheManager.initialize_kv_cache in {time.perf_counter() - t_total:.2f}s pid={os.getpid()}"
            )
            raise

        _emit_startup_trace(
            f"END KVCacheManager.initialize_kv_cache in {time.perf_counter() - t_total:.2f}s pid={os.getpid()}"
        )

    wrapped._marin_startup_timed = True
    KVCacheManager.initialize_kv_cache = wrapped
    _emit_startup_trace(f"installed detailed early KVCacheManager.initialize_kv_cache instrumentation pid={os.getpid()}")


def _install_detailed_upstream_create_kv_caches(kv_cache_module: Any) -> None:
    original = getattr(kv_cache_module, "create_kv_caches", None)
    if original is None:
        return
    if getattr(original, "_marin_startup_timed", False):
        return

    @functools.wraps(original)
    def wrapped(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        mesh: Any,
        layer_names: list[str],
        cache_dtype: Any = None,
        use_mla: bool = False,
    ) -> list[Any]:
        if cache_dtype is None:
            cache_dtype = kv_cache_module.DEFAULT_KV_CACHE_DTYPE

        total_layers = len(layer_names)
        _emit_startup_trace(
            "START kv_cache.create_kv_caches "
            f"pid={os.getpid()} num_blocks={num_blocks} block_size={block_size} "
            f"num_kv_heads={num_kv_heads} head_size={head_size} layers={total_layers} use_mla={use_mla}"
        )
        t_total = time.perf_counter()
        try:
            _emit_startup_trace(f"START kv_cache.create_kv_caches.shape pid={os.getpid()}")
            t_shape = time.perf_counter()
            cache_shape = kv_cache_module.get_kv_cache_shape_with_mesh(
                mesh,
                num_blocks,
                block_size,
                num_kv_heads,
                head_size,
                cache_dtype,
                use_mla,
            )
            _emit_startup_trace(
                "END kv_cache.create_kv_caches.shape "
                f"in {time.perf_counter() - t_shape:.2f}s pid={os.getpid()} shape={cache_shape}"
            )

            _emit_startup_trace(f"START kv_cache.create_kv_caches.sharding pid={os.getpid()}")
            t_sharding = time.perf_counter()
            if use_mla:
                sharding = kv_cache_module.NamedSharding(
                    mesh,
                    kv_cache_module.PartitionSpec(kv_cache_module.ShardingAxisName.MLP_TENSOR),
                )
            else:
                sharding = kv_cache_module.NamedSharding(
                    mesh,
                    kv_cache_module.PartitionSpec(
                        kv_cache_module.ShardingAxisName.ATTN_DATA,
                        None,
                        kv_cache_module.ShardingAxisName.ATTN_HEAD,
                    ),
                )
            _emit_startup_trace(
                "END kv_cache.create_kv_caches.sharding "
                f"in {time.perf_counter() - t_sharding:.2f}s pid={os.getpid()} sharding={sharding}"
            )

            def _allocate() -> Any:
                return kv_cache_module.jnp.empty(shape=cache_shape, dtype=cache_dtype)

            _emit_startup_trace(f"START kv_cache.create_kv_caches.make_jit pid={os.getpid()}")
            t_make_jit = time.perf_counter()
            sharded_allocate = kv_cache_module.jax.jit(_allocate, out_shardings=sharding)
            _emit_startup_trace(
                "END kv_cache.create_kv_caches.make_jit " f"in {time.perf_counter() - t_make_jit:.2f}s pid={os.getpid()}"
            )

            kv_caches = []
            for i, _ in enumerate(layer_names):
                _emit_startup_trace(
                    f"START kv_cache.create_kv_caches.allocate index={i + 1}/{total_layers} pid={os.getpid()}"
                )
                t_allocate = time.perf_counter()
                kv_caches.append(sharded_allocate())
                _emit_startup_trace(
                    "END kv_cache.create_kv_caches.allocate "
                    f"index={i + 1}/{total_layers} in {time.perf_counter() - t_allocate:.2f}s pid={os.getpid()}"
                )
            return kv_caches
        except Exception:
            _emit_startup_trace(
                f"FAIL kv_cache.create_kv_caches in {time.perf_counter() - t_total:.2f}s pid={os.getpid()}"
            )
            raise

        finally:
            _emit_startup_trace(
                f"END kv_cache.create_kv_caches in {time.perf_counter() - t_total:.2f}s pid={os.getpid()}"
            )

    wrapped._marin_startup_timed = True
    kv_cache_module.create_kv_caches = wrapped
    _emit_startup_trace(f"installed detailed early kv_cache.create_kv_caches instrumentation pid={os.getpid()}")


def _install_early_async_startup_instrumentation() -> None:
    if not _startup_timing_enabled():
        return
    _enable_early_startup_faulthandler()

    try:
        from vllm.v1.engine.core import EngineCore, EngineCoreProc
        from vllm.v1.executor.uniproc_executor import UniProcExecutor
        from vllm.v1.worker.worker_base import WorkerWrapperBase
    except Exception as exc:
        logger.warning("Could not install early async startup instrumentation: %s", exc)
        return

    _wrap_upstream_method_with_timing(WorkerWrapperBase, "init_device", label="WorkerWrapperBase.init_device")
    _install_upstream_delegate_method(WorkerWrapperBase, "load_model", label="WorkerWrapperBase.load_model")
    _install_upstream_delegate_method(
        WorkerWrapperBase,
        "determine_available_memory",
        label="WorkerWrapperBase.determine_available_memory",
    )
    _wrap_upstream_method_with_timing(
        WorkerWrapperBase,
        "initialize_from_config",
        label="WorkerWrapperBase.initialize_from_config",
    )
    _install_upstream_delegate_method(
        WorkerWrapperBase,
        "compile_or_warm_up_model",
        label="WorkerWrapperBase.compile_or_warm_up_model",
    )
    _wrap_upstream_method_with_timing(EngineCore, "_initialize_kv_caches", label="EngineCore._initialize_kv_caches")
    _wrap_upstream_method_with_timing(UniProcExecutor, "_distributed_args", label="UniProcExecutor._distributed_args")
    _wrap_upstream_method_with_timing(UniProcExecutor, "_init_executor", label="UniProcExecutor._init_executor")
    _wrap_upstream_method_with_timing(EngineCoreProc, "__init__", label="EngineCoreProc.__init__")

    try:
        import tpu_inference.runner.kv_cache as kv_cache_module
        import tpu_inference.runner.kv_cache_manager as kv_cache_manager_module
        import tpu_inference.runner.tpu_runner as tpu_runner_module
        from tpu_inference.models.common import model_loader as model_loader_module
        import tpu_inference.models.vllm.vllm_model_wrapper as vllm_model_wrapper_module
        from tpu_inference.models.vllm.vllm_model_wrapper import VllmModelWrapper
        from tpu_inference.worker.tpu_worker import TPUWorker
    except Exception as exc:
        logger.warning("Could not install core TPU load-model instrumentation: %s", exc)
    else:
        _wrap_upstream_method_with_timing(TPUWorker, "load_model", label="TPUWorker.load_model")
        _wrap_upstream_method_with_timing(
            TPUWorker,
            "initialize_from_config",
            label="TPUWorker.initialize_from_config",
        )
        _install_detailed_upstream_tpu_model_runner_load_model(tpu_runner_module)
        _wrap_upstream_method_with_timing(
            tpu_runner_module.TPUModelRunner,
            "initialize_kv_cache",
            label="TPUModelRunner.initialize_kv_cache",
        )
        _wrap_upstream_method_with_timing(
            kv_cache_manager_module.KVCacheManager,
            "maybe_reinitialize_input_batch",
            label="KVCacheManager.maybe_reinitialize_input_batch",
        )
        _install_detailed_upstream_kv_cache_manager_initialize_kv_cache(kv_cache_manager_module)
        _wrap_upstream_module_function_with_timing(tpu_runner_module, "get_model", label="tpu_runner.get_model")
        _install_detailed_upstream_create_kv_caches(kv_cache_module)
        kv_cache_manager_module.create_kv_caches = kv_cache_module.create_kv_caches
        _emit_startup_trace(f"redirected kv_cache_manager.create_kv_caches to detailed wrapper pid={os.getpid()}")
        _wrap_upstream_module_function_with_timing(
            model_loader_module,
            "get_model",
            label="model_loader.get_model",
        )
        _wrap_upstream_module_function_with_timing(
            model_loader_module,
            "get_vllm_model",
            label="model_loader.get_vllm_model",
        )
        _wrap_upstream_module_function_with_timing(
            model_loader_module,
            "get_flax_model",
            label="model_loader.get_flax_model",
        )
        _wrap_upstream_method_with_timing(
            VllmModelWrapper,
            "load_weights",
            label="VllmModelWrapper.load_weights",
        )
        _wrap_upstream_method_with_timing(
            VllmModelWrapper,
            "jit_step_func",
            label="VllmModelWrapper.jit_step_func",
        )
        _wrap_upstream_method_with_timing(
            VllmModelWrapper,
            "jit_compute_logits_func",
            label="VllmModelWrapper.jit_compute_logits_func",
        )
        _wrap_upstream_module_function_with_timing(
            vllm_model_wrapper_module,
            "vllm_get_model",
            label="vllm_model_wrapper.vllm_get_model",
        )
        _wrap_upstream_module_function_with_timing(
            vllm_model_wrapper_module,
            "shard_model_to_tpu",
            label="vllm_model_wrapper.shard_model_to_tpu",
        )
        _wrap_upstream_module_function_with_timing(
            vllm_model_wrapper_module,
            "load_lora_model",
            label="vllm_model_wrapper.load_lora_model",
        )

    try:
        import vllm.model_executor.model_loader as vllm_model_loader_module
        import vllm.model_executor.model_loader.base_loader as vllm_base_loader_module
        import vllm.model_executor.model_loader.dummy_loader as vllm_dummy_loader_module
        import vllm.model_executor.model_loader.weight_utils as vllm_weight_utils_module
        from vllm.model_executor.model_loader.base_loader import BaseModelLoader
        from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
        from vllm.model_executor.model_loader import utils as vllm_model_loader_utils
    except Exception as exc:
        logger.warning("Could not install upstream vLLM model-loader instrumentation: %s", exc)
    else:
        _wrap_upstream_module_function_with_timing(
            vllm_model_loader_module,
            "get_model",
            label="vllm_model_loader.get_model",
        )
        _wrap_upstream_module_function_with_timing(
            vllm_model_loader_utils,
            "initialize_model",
            label="vllm_model_loader_utils.initialize_model",
        )
        _wrap_upstream_module_function_with_timing(
            vllm_model_loader_utils,
            "process_weights_after_loading",
            label="vllm_model_loader_utils.process_weights_after_loading",
        )
        _wrap_upstream_method_with_timing(
            BaseModelLoader,
            "load_model",
            label="BaseModelLoader.load_model",
        )
        _wrap_upstream_method_with_timing(
            DummyModelLoader,
            "load_weights",
            label="DummyModelLoader.load_weights",
        )
        _wrap_upstream_module_function_with_timing(
            vllm_base_loader_module,
            "initialize_model",
            label="vllm_base_loader.initialize_model",
        )
        _wrap_upstream_module_function_with_timing(
            vllm_base_loader_module,
            "process_weights_after_loading",
            label="vllm_base_loader.process_weights_after_loading",
        )
        install_dummy_init_progress_instrumentation(
            vllm_dummy_loader_module=vllm_dummy_loader_module,
            vllm_weight_utils_module=vllm_weight_utils_module,
            emit=_emit_startup_trace,
        )

    try:
        from tpu_inference.models.vllm.vllm_model_loader import (
            IncrementalModelLoader,
            RunaiIncrementalModelLoader,
        )
    except Exception as exc:
        logger.warning("Could not install TPU incremental-loader instrumentation: %s", exc)
    else:
        _wrap_upstream_method_with_timing(
            IncrementalModelLoader,
            "load_model",
            label="IncrementalModelLoader.load_model",
        )
        _wrap_upstream_method_with_timing(
            RunaiIncrementalModelLoader,
            "load_model",
            label="RunaiIncrementalModelLoader.load_model",
        )

    try:
        from tpu_inference.layers.vllm.process_weights import cleanup_sharding as cleanup_sharding_module
    except Exception as exc:
        logger.warning("Could not install cleanup-sharding instrumentation: %s", exc)
    else:
        _wrap_upstream_module_function_with_timing(
            cleanup_sharding_module,
            "shard_model_to_tpu",
            label="cleanup_sharding.shard_model_to_tpu",
        )
        _wrap_upstream_module_function_with_timing(
            cleanup_sharding_module,
            "_shard_module_to_tpu",
            label="cleanup_sharding._shard_module_to_tpu",
        )

    _emit_startup_trace(
        "installed early async startup instrumentation "
        f"VLLM_WORKER_MULTIPROC_METHOD={os.environ.get('VLLM_WORKER_MULTIPROC_METHOD', 'fork')!r} "
        f"pid={os.getpid()}"
    )


async def _watchdog_loop(server: Any, engine_client: Any) -> None:
    while True:
        await asyncio.sleep(5.0)
        if getattr(engine_client, "errored", False) and not getattr(engine_client, "is_running", True):
            server.should_exit = True
            return


async def _await_task_cancellation(task: asyncio.Task[Any]) -> None:
    try:
        await task
    except asyncio.CancelledError:
        return


def _log_async_engine_startup(
    *,
    engine_args: Any,
    requested_model_name_or_path: str,
    events: list[str],
) -> None:
    message = (
        "Creating AsyncLLM engine with "
        f"bootstrap_model={getattr(engine_args, 'model', None)!r} "
        f"requested_model={requested_model_name_or_path!r} "
        f"MODEL_IMPL_TYPE={os.environ.get('MODEL_IMPL_TYPE')!r} "
        f"MARIN_VLLM_FAST_BOOTSTRAP={os.environ.get('MARIN_VLLM_FAST_BOOTSTRAP')!r} "
        f"{_DIRECT_SAMPLING_KEY_ENV}={os.environ.get(_DIRECT_SAMPLING_KEY_ENV)!r} "
        f"tensor_parallel_size={getattr(engine_args, 'tensor_parallel_size', None)!r} "
        f"enforce_eager={getattr(engine_args, 'enforce_eager', None)!r} "
        f"SKIP_JAX_PRECOMPILE={os.environ.get('SKIP_JAX_PRECOMPILE')!r} "
        f"MARIN_VLLM_STARTUP_TIMING={os.environ.get('MARIN_VLLM_STARTUP_TIMING')!r} "
        f"VLLM_WORKER_MULTIPROC_METHOD={os.environ.get('VLLM_WORKER_MULTIPROC_METHOD', 'fork')!r} "
        f"VLLM_ENABLE_V1_MULTIPROCESSING={os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING')!r}"
    )
    _record_event(events, message)
    _iris_emit("I", "vllm.async", message)


def _configure_async_vllm_environment() -> None:
    """Apply process-level defaults needed for async native TPU serving."""
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    # Let the TPU fork choose the native JAX/flax_nnx path when the model
    # supports it. Forcing MODEL_IMPL_TYPE="vllm" sends async-native startup
    # through the PyTorch wrapper path, which spends many minutes on dummy
    # weight initialization and post-load processing before Marin injects the
    # real streamed weights.
    os.environ.setdefault("MODEL_IMPL_TYPE", "auto")
    os.environ.setdefault("MARIN_VLLM_FAST_BOOTSTRAP", "1")
    os.environ.setdefault("TPU_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TPU_STDERR_LOG_LEVEL", "3")
    os.environ.setdefault("MARIN_VLLM_STARTUP_TIMING", "1")
    os.environ.setdefault("MARIN_VLLM_STARTUP_FAULTHANDLER", "1")
    os.environ.setdefault("MARIN_VLLM_STARTUP_FAULTHANDLER_SECS", "300")
    os.environ.setdefault(_DIRECT_SAMPLING_KEY_ENV, "1")
    os.environ.setdefault("SKIP_JAX_PRECOMPILE", "1")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "fork")
    os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "1")
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", f"{marin_prefix()}/compilation-cache")
    os.environ.setdefault("VLLM_XLA_CACHE_PATH", os.environ["JAX_COMPILATION_CACHE_DIR"])
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "-1")
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "2")
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
