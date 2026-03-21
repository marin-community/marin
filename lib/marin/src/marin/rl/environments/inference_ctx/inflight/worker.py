# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import faulthandler
import functools
import logging
import os
import sys
import time
from collections.abc import Callable
from typing import Any, cast

import numpy as np
from marin.inference.vllm_tpu_bootstrap_patch import install_marin_fast_tpu_bootstrap_patch
from marin.rl.environments.inference_ctx.startup_debug import install_dummy_init_progress_instrumentation
from marin.rl.environments.inference_ctx.inflight.async_bridge import AsyncBridge
from marin.rl.weight_utils import levanter_state_dict_to_nnx_state_on_cpu
from marin.rl.environments.inference_ctx.vllm_utils import MODEL_MAPPINGS, MODEL_TRANSPOSE_KEYS

logger = logging.getLogger(__name__)

_STARTUP_TIMING_ENV = "MARIN_VLLM_STARTUP_TIMING"
_STARTUP_FAULTHANDLER_ENV = "MARIN_VLLM_STARTUP_FAULTHANDLER"
_STARTUP_TIMING_PREFIX = "[marin-vllm-startup]"

try:
    from vllm import AsyncEngineArgs, SamplingParams
    from vllm.v1.engine.async_llm import AsyncLLM
except ImportError:
    AsyncEngineArgs = None
    SamplingParams = None
    AsyncLLM = None
    logger.warning("vLLM async engine is not available. Please install vLLM v1 with: pip install vllm")


def _startup_timing_enabled() -> bool:
    return os.environ.get(_STARTUP_TIMING_ENV) == "1"


def _startup_context(subject: Any | None) -> str:
    if subject is None:
        return f"pid={os.getpid()}"

    fields = [f"pid={os.getpid()}"]
    for attr in ("rank", "local_rank", "rpc_rank", "topology_order_id"):
        value = getattr(subject, attr, None)
        if value is not None:
            fields.append(f"{attr}={value}")
    return " ".join(fields)


def _startup_emit(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _enable_startup_faulthandler() -> None:
    if os.environ.get(_STARTUP_FAULTHANDLER_ENV) != "1":
        return

    timeout = int(os.environ.get("MARIN_VLLM_STARTUP_FAULTHANDLER_SECS", "300"))
    try:
        faulthandler.cancel_dump_traceback_later()
    except RuntimeError:
        pass
    faulthandler.dump_traceback_later(timeout, repeat=True, file=sys.stderr)
    _startup_emit(f"{_STARTUP_TIMING_PREFIX} enabled startup faulthandler every {timeout}s pid={os.getpid()}")


def _startup_log(event: str, label: str, subject: Any | None, elapsed: float | None = None) -> None:
    context = _startup_context(subject)
    if elapsed is None:
        _startup_emit(f"{_STARTUP_TIMING_PREFIX} {event} {label} {context}")
        return

    _startup_emit(f"{_STARTUP_TIMING_PREFIX} {event} {label} in {elapsed:.2f}s {context}")


def _wrap_timed_instance_method(cls: type[Any], method_name: str, *, label: str | None = None) -> None:
    original = getattr(cls, method_name, None)
    if original is None:
        return
    if getattr(original, "_marin_startup_timed", False):
        return
    original_callable = cast(Callable[..., Any], original)

    effective_label = label or f"{cls.__name__}.{method_name}"

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        _startup_log("START", effective_label, self)
        t_start = time.perf_counter()
        try:
            result = original_callable(self, *args, **kwargs)
        except Exception:
            _startup_log("FAIL", effective_label, self, time.perf_counter() - t_start)
            raise
        _startup_log("END", effective_label, self, time.perf_counter() - t_start)
        return result

    wrapped._marin_startup_timed = True
    setattr(cls, method_name, wrapped)


def _wrap_timed_module_function(module: Any, function_name: str, *, label: str | None = None) -> None:
    original = getattr(module, function_name, None)
    if original is None:
        return
    if getattr(original, "_marin_startup_timed", False):
        return
    original_callable = cast(Callable[..., Any], original)

    effective_label = label or f"{module.__name__}.{function_name}"

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        subject = args[0] if args else None
        _startup_log("START", effective_label, subject)
        t_start = time.perf_counter()
        try:
            result = original_callable(*args, **kwargs)
        except Exception:
            _startup_log("FAIL", effective_label, subject, time.perf_counter() - t_start)
            raise
        _startup_log("END", effective_label, subject, time.perf_counter() - t_start)
        return result

    wrapped._marin_startup_timed = True
    setattr(module, function_name, wrapped)


def _install_timed_delegate_method(cls: type[Any], method_name: str, *, label: str | None = None) -> None:
    if method_name in cls.__dict__:
        _wrap_timed_instance_method(cls, method_name, label=label)
        return

    effective_label = label or f"{cls.__name__}.{method_name}"

    def delegated(self, *args, **kwargs):
        _startup_log("START", effective_label, self)
        t_start = time.perf_counter()
        try:
            result = getattr(self.worker, method_name)(*args, **kwargs)
        except Exception:
            _startup_log("FAIL", effective_label, self, time.perf_counter() - t_start)
            raise
        _startup_log("END", effective_label, self, time.perf_counter() - t_start)
        return result

    delegated.__name__ = method_name
    delegated.__qualname__ = f"{cls.__qualname__}.{method_name}"
    delegated._marin_startup_timed = True
    setattr(cls, method_name, delegated)


def _install_detailed_tpu_model_runner_load_model(tpu_runner_module: Any) -> None:
    TPUModelRunner = tpu_runner_module.TPUModelRunner
    original = getattr(TPUModelRunner, "load_model", None)
    if original is None:
        return
    if getattr(original, "_marin_startup_timed", False):
        return

    @functools.wraps(original)
    def wrapped(self):
        _startup_log("START", "TPUModelRunner.load_model", self)
        t_total = time.perf_counter()
        try:
            t_get_model = time.perf_counter()
            try:
                (
                    self.model_fn,
                    self.compute_logits_fn,
                    self.pooler_fn,
                    self.combine_hidden_states_fn,
                    multimodal_fns,
                    self.state,
                    self.lora_manager,
                    self.model,
                ) = tpu_runner_module.get_model(
                    self.vllm_config,
                    self.rng_key,
                    self.mesh,
                )
            except Exception:
                _startup_log(
                    "FAIL",
                    "TPUModelRunner.load_model.get_model_tuple",
                    self,
                    time.perf_counter() - t_get_model,
                )
                raise
            _startup_log(
                "END",
                "TPUModelRunner.load_model.get_model_tuple",
                self,
                time.perf_counter() - t_get_model,
            )

            _startup_log("START", "TPUModelRunner.load_model.multimodal_bind", self)
            t_multimodal_bind = time.perf_counter()
            try:
                multimodal_fns = multimodal_fns or {}
                self.precompile_vision_encoder_fn = multimodal_fns.get("precompile_vision_encoder_fn", None)
                self.embed_multimodal_fn = multimodal_fns.get("embed_multimodal_fn", None)
                self.embed_input_ids_fn = multimodal_fns.get("embed_input_ids_fn", None)
                self.get_mrope_input_positions_fn = multimodal_fns.get("get_mrope_input_positions_fn", None)
            except Exception:
                _startup_log(
                    "FAIL",
                    "TPUModelRunner.load_model.multimodal_bind",
                    self,
                    time.perf_counter() - t_multimodal_bind,
                )
                raise
            _startup_log(
                "END",
                "TPUModelRunner.load_model.multimodal_bind",
                self,
                time.perf_counter() - t_multimodal_bind,
            )

            if self.drafter is not None:
                _startup_log("START", "TPUModelRunner.load_model.drafter", self)
                t_drafter = time.perf_counter()
                try:
                    self.drafter.load_model(self.state)
                except Exception:
                    _startup_log(
                        "FAIL",
                        "TPUModelRunner.load_model.drafter",
                        self,
                        time.perf_counter() - t_drafter,
                    )
                    raise
                _startup_log(
                    "END",
                    "TPUModelRunner.load_model.drafter",
                    self,
                    time.perf_counter() - t_drafter,
                )

            _startup_log("START", "TPUModelRunner.load_model.rng_params_for_sampling", self)
            t_rng = time.perf_counter()
            try:
                self.rng_params_for_sampling = tpu_runner_module.nnx.Rngs(
                    tpu_runner_module.jax.random.key(self.model_config.seed)
                ).params()
            except Exception:
                _startup_log(
                    "FAIL",
                    "TPUModelRunner.load_model.rng_params_for_sampling",
                    self,
                    time.perf_counter() - t_rng,
                )
                raise
            _startup_log(
                "END",
                "TPUModelRunner.load_model.rng_params_for_sampling",
                self,
                time.perf_counter() - t_rng,
            )

            _startup_log("START", "TPUModelRunner.load_model.is_multimodal_model", self)
            t_is_multimodal = time.perf_counter()
            try:
                self.is_multimodal_model = (
                    self.model_config.is_multimodal_model
                    and self.embed_multimodal_fn is not None
                    and hasattr(self.model_config.hf_config, "architectures")
                )
            except Exception:
                _startup_log(
                    "FAIL",
                    "TPUModelRunner.load_model.is_multimodal_model",
                    self,
                    time.perf_counter() - t_is_multimodal,
                )
                raise
            _startup_log(
                "END",
                "TPUModelRunner.load_model.is_multimodal_model",
                self,
                time.perf_counter() - t_is_multimodal,
            )

            _startup_log("START", "TPUModelRunner.load_model.log_init_model", self)
            t_log = time.perf_counter()
            try:
                tpu_runner_module.logger.info(
                    "Init model | hbm=%sGiB",
                    tpu_runner_module.common_utils.hbm_usage_gb(self.devices),
                )
            except Exception:
                _startup_log(
                    "FAIL",
                    "TPUModelRunner.load_model.log_init_model",
                    self,
                    time.perf_counter() - t_log,
                )
                raise
            _startup_log(
                "END",
                "TPUModelRunner.load_model.log_init_model",
                self,
                time.perf_counter() - t_log,
            )

        except Exception:
            _startup_log("FAIL", "TPUModelRunner.load_model", self, time.perf_counter() - t_total)
            raise
        _startup_log("END", "TPUModelRunner.load_model", self, time.perf_counter() - t_total)

    wrapped._marin_startup_timed = True
    TPUModelRunner.load_model = wrapped
    _startup_emit(
        f"{_STARTUP_TIMING_PREFIX} installed detailed TPUModelRunner.load_model instrumentation pid={os.getpid()}"
    )


def _apply_startup_timing_instrumentation() -> None:
    if not _startup_timing_enabled():
        return
    _enable_startup_faulthandler()

    try:
        from vllm.platforms import current_platform
        from vllm.v1.engine.core import EngineCore
        from vllm.v1.worker.worker_base import WorkerWrapperBase
        from tpu_inference.models.common import model_loader as model_loader_module
        import tpu_inference.models.vllm.vllm_model_wrapper as vllm_model_wrapper_module
        from tpu_inference.models.vllm.vllm_model_wrapper import VllmModelWrapper
        import tpu_inference.runner.tpu_runner as tpu_runner_module
        from tpu_inference.runner.tpu_runner import TPUModelRunner
        from tpu_inference.worker.tpu_worker import TPUWorker
    except ImportError as exc:
        logger.warning("Could not import TPU modules for Marin startup timing instrumentation: %s", exc)
        return

    _wrap_timed_instance_method(WorkerWrapperBase, "init_device", label="WorkerWrapperBase.init_device")
    _wrap_timed_instance_method(
        WorkerWrapperBase,
        "initialize_from_config",
        label="WorkerWrapperBase.initialize_from_config",
    )
    _install_timed_delegate_method(WorkerWrapperBase, "load_model", label="WorkerWrapperBase.load_model")
    _install_timed_delegate_method(
        WorkerWrapperBase,
        "get_kv_cache_spec",
        label="WorkerWrapperBase.get_kv_cache_spec",
    )
    _install_timed_delegate_method(
        WorkerWrapperBase,
        "determine_available_memory",
        label="WorkerWrapperBase.determine_available_memory",
    )
    _install_timed_delegate_method(
        WorkerWrapperBase,
        "compile_or_warm_up_model",
        label="WorkerWrapperBase.compile_or_warm_up_model",
    )

    _wrap_timed_instance_method(EngineCore, "_initialize_kv_caches", label="EngineCore._initialize_kv_caches")
    _wrap_timed_instance_method(
        type(current_platform),
        "update_block_size_for_backend",
        label="current_platform.update_block_size_for_backend",
    )

    _wrap_timed_instance_method(TPUWorker, "init_device", label="TPUWorker.init_device")
    _wrap_timed_instance_method(TPUWorker, "load_model", label="TPUWorker.load_model")
    _wrap_timed_instance_method(
        TPUWorker,
        "determine_available_memory",
        label="TPUWorker.determine_available_memory",
    )
    _wrap_timed_instance_method(
        TPUWorker,
        "initialize_from_config",
        label="TPUWorker.initialize_from_config",
    )
    _wrap_timed_instance_method(
        TPUWorker,
        "compile_or_warm_up_model",
        label="TPUWorker.compile_or_warm_up_model",
    )

    _install_detailed_tpu_model_runner_load_model(tpu_runner_module)
    _wrap_timed_instance_method(
        TPUModelRunner,
        "initialize_kv_cache",
        label="TPUModelRunner.initialize_kv_cache",
    )
    _wrap_timed_instance_method(TPUModelRunner, "capture_model", label="TPUModelRunner.capture_model")

    _wrap_timed_module_function(tpu_runner_module, "get_model", label="tpu_runner.get_model")
    _wrap_timed_module_function(model_loader_module, "get_model", label="model_loader.get_model")
    _wrap_timed_module_function(model_loader_module, "get_vllm_model", label="model_loader.get_vllm_model")
    _wrap_timed_module_function(model_loader_module, "get_flax_model", label="model_loader.get_flax_model")
    _wrap_timed_instance_method(VllmModelWrapper, "load_weights", label="VllmModelWrapper.load_weights")
    _wrap_timed_instance_method(VllmModelWrapper, "jit_step_func", label="VllmModelWrapper.jit_step_func")
    _wrap_timed_instance_method(
        VllmModelWrapper,
        "jit_compute_logits_func",
        label="VllmModelWrapper.jit_compute_logits_func",
    )
    _wrap_timed_module_function(
        vllm_model_wrapper_module,
        "vllm_get_model",
        label="vllm_model_wrapper.vllm_get_model",
    )
    _wrap_timed_module_function(
        vllm_model_wrapper_module,
        "shard_model_to_tpu",
        label="vllm_model_wrapper.shard_model_to_tpu",
    )
    _wrap_timed_module_function(
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
    except ImportError as exc:
        logger.warning("Could not import upstream vLLM model-loader timing modules: %s", exc)
    else:
        _wrap_timed_module_function(vllm_model_loader_module, "get_model", label="vllm_model_loader.get_model")
        _wrap_timed_module_function(
            vllm_model_loader_utils,
            "initialize_model",
            label="vllm_model_loader_utils.initialize_model",
        )
        _wrap_timed_module_function(
            vllm_model_loader_utils,
            "process_weights_after_loading",
            label="vllm_model_loader_utils.process_weights_after_loading",
        )
        _wrap_timed_instance_method(
            BaseModelLoader,
            "load_model",
            label="BaseModelLoader.load_model",
        )
        _wrap_timed_instance_method(
            DummyModelLoader,
            "load_weights",
            label="DummyModelLoader.load_weights",
        )
        _wrap_timed_module_function(
            vllm_base_loader_module,
            "initialize_model",
            label="vllm_base_loader.initialize_model",
        )
        _wrap_timed_module_function(
            vllm_base_loader_module,
            "process_weights_after_loading",
            label="vllm_base_loader.process_weights_after_loading",
        )
        install_dummy_init_progress_instrumentation(
            vllm_dummy_loader_module=vllm_dummy_loader_module,
            vllm_weight_utils_module=vllm_weight_utils_module,
            emit=lambda message: _startup_emit(f"{_STARTUP_TIMING_PREFIX} {message}"),
        )

    try:
        from tpu_inference.models.vllm.vllm_model_loader import (
            IncrementalModelLoader,
            RunaiIncrementalModelLoader,
        )
    except ImportError as exc:
        logger.warning("Could not import TPU incremental-loader timing modules: %s", exc)
    else:
        _wrap_timed_instance_method(
            IncrementalModelLoader,
            "load_model",
            label="IncrementalModelLoader.load_model",
        )
        _wrap_timed_instance_method(
            RunaiIncrementalModelLoader,
            "load_model",
            label="RunaiIncrementalModelLoader.load_model",
        )

    try:
        from tpu_inference.layers.vllm.process_weights import cleanup_sharding as cleanup_sharding_module
    except ImportError as exc:
        logger.warning("Could not import cleanup-sharding timing modules: %s", exc)
    else:
        _wrap_timed_module_function(
            cleanup_sharding_module,
            "shard_model_to_tpu",
            label="cleanup_sharding.shard_model_to_tpu",
        )
        _wrap_timed_module_function(
            cleanup_sharding_module,
            "_shard_module_to_tpu",
            label="cleanup_sharding._shard_module_to_tpu",
        )
    _startup_emit(f"{_STARTUP_TIMING_PREFIX} enabled worker startup timing instrumentation pid={os.getpid()}")


def _apply_worker_extension_mro_fix():
    """Monkeypatch vLLM's WorkerWrapperBase.init_worker to fix MRO conflict.

    vLLM V1's init_worker appends worker_extension_cls to worker_class.__bases__:
        worker_class.__bases__ = worker_class.__bases__ + (worker_extension_cls,)

    This causes MRO conflicts when the worker class hierarchy ends with 'object'
    and the extension also inherits from 'object'. The fix is to prepend instead:
        worker_class.__bases__ = (worker_extension_cls,) + worker_class.__bases__

    This is a known issue with dynamic mixin injection in Python.
    See: https://github.com/vllm-project/vllm/issues/XXXX (TODO: file upstream issue)
    """
    try:
        from vllm.v1.worker.worker_base import WorkerWrapperBase
        from vllm.config import set_current_vllm_config
        from vllm.utils.import_utils import resolve_obj_by_qualname
        from vllm.multimodal import MULTIMODAL_REGISTRY
        from vllm.multimodal.cache import worker_receiver_cache_from_config
    except ImportError:
        logger.warning("Could not import vLLM V1 worker modules for MRO fix")
        return

    def _patched_init_worker(self, all_kwargs):
        """Patched init_worker that prepends worker_extension_cls instead of appending."""
        t_start = time.perf_counter()
        timing_enabled = _startup_timing_enabled()
        if timing_enabled:
            _startup_log("START", "WorkerWrapperBase.init_worker", self)

        try:
            kwargs = all_kwargs[self.rpc_rank]
            self.vllm_config = kwargs.get("vllm_config")
            assert self.vllm_config is not None, "vllm_config is required to initialize the worker"
            self.vllm_config.enable_trace_function_call_for_thread()

            from vllm.plugins import load_general_plugins

            load_general_plugins()

            if isinstance(self.vllm_config.parallel_config.worker_cls, str):
                worker_class = resolve_obj_by_qualname(self.vllm_config.parallel_config.worker_cls)
            else:
                raise ValueError(
                    "passing worker_cls is no longer supported. Please keep the class in a "
                    "separate module and pass the qualified name of the class as a string."
                )

            if self.vllm_config.parallel_config.worker_extension_cls:
                worker_extension_cls = resolve_obj_by_qualname(self.vllm_config.parallel_config.worker_extension_cls)
                extended_calls = []
                if worker_extension_cls not in worker_class.__bases__:
                    for attr in dir(worker_extension_cls):
                        if attr.startswith("__"):
                            continue
                        assert not hasattr(worker_class, attr), (
                            f"Worker class {worker_class} already has an attribute"
                            f" {attr}, which conflicts with the worker"
                            f" extension class {worker_extension_cls}."
                        )
                        if callable(getattr(worker_extension_cls, attr)):
                            extended_calls.append(attr)

                    worker_class = type(
                        f"{worker_class.__name__}WithExtension",
                        (worker_extension_cls, worker_class),
                        {},
                    )
                    self.vllm_config.parallel_config.worker_cls = worker_class
                    logger.info(
                        "Created extended worker class %s with %s for collective_rpc calls %s",
                        worker_class.__name__,
                        worker_extension_cls.__name__,
                        extended_calls,
                    )

            shared_worker_lock = kwargs.pop("shared_worker_lock", None)
            if shared_worker_lock is None:
                msg = (
                    "Missing `shared_worker_lock` argument from executor. "
                    "This argument is needed for mm_processor_cache_type='shm'."
                )
                mm_config = self.vllm_config.model_config.multimodal_config
                if mm_config and mm_config.mm_processor_cache_type == "shm":
                    raise ValueError(msg)
                logger.warning(msg)
                self.mm_receiver_cache = None
            else:
                self.mm_receiver_cache = worker_receiver_cache_from_config(
                    self.vllm_config,
                    MULTIMODAL_REGISTRY,
                    shared_worker_lock,
                )

            with set_current_vllm_config(self.vllm_config):
                self.worker = worker_class(**kwargs)
                assert self.worker is not None
                if timing_enabled:
                    worker_class_name = f"{self.worker.__class__.__module__}.{self.worker.__class__.__qualname__}"
                    executor_backend = self.vllm_config.parallel_config.distributed_executor_backend
                    _startup_emit(
                        f"{_STARTUP_TIMING_PREFIX} worker created "
                        f"worker_class={worker_class_name} executor_backend={executor_backend!r} "
                        f"pid={os.getpid()} rpc_rank={self.rpc_rank}"
                    )
        except Exception:
            if timing_enabled:
                _startup_log("FAIL", "WorkerWrapperBase.init_worker", self, time.perf_counter() - t_start)
            raise
        if timing_enabled:
            _startup_log("END", "WorkerWrapperBase.init_worker", self, time.perf_counter() - t_start)

    WorkerWrapperBase.init_worker = _patched_init_worker
    _startup_emit(f"{_STARTUP_TIMING_PREFIX} applied worker extension MRO fix pid={os.getpid()}")


# Apply the monkeypatch when this module is imported
if AsyncEngineArgs is not None:
    _apply_worker_extension_mro_fix()
    install_marin_fast_tpu_bootstrap_patch(emit=lambda message: _startup_emit(f"{_STARTUP_TIMING_PREFIX} {message}"))
    _apply_startup_timing_instrumentation()


def deserialize_state_dict_from_rpc(serialized_state_dict: dict) -> dict:
    """Deserialize (bytes, dtype, shape) tuples/lists back to numpy arrays.

    Inverse of serialize_state_dict_for_rpc in async_vllm.py.
    Note: RPC serialization may convert tuples to lists, so we handle both.
    """
    state_dict = {}
    for key, value in serialized_state_dict.items():
        if isinstance(value, (tuple, list)) and len(value) == 3:
            data_bytes, dtype_str, shape = value
            if isinstance(data_bytes, bytes):
                # Shape may come as list from RPC, convert to tuple for reshape
                if isinstance(shape, list):
                    shape = tuple(shape)
                state_dict[key] = np.frombuffer(data_bytes, dtype=dtype_str).reshape(shape)
            else:
                # Not our serialized format, pass through
                state_dict[key] = value
        else:
            state_dict[key] = value
    return state_dict


class WorkerExtension:
    def update_weight(self, new_state_dict: dict, model_name: str):
        # NOTE(Chris): This step of np -> jax must be on the worker process that
        # has inherited the TPU devices already because vLLM calls os.fork() already
        # this means that we will not be able to create a jax cpu mesh before the fork.

        # Deserialize from (bytes, dtype, shape) tuples back to numpy arrays
        deserialized_state_dict = deserialize_state_dict_from_rpc(new_state_dict)
        new_state = levanter_state_dict_to_nnx_state_on_cpu(deserialized_state_dict)
        self.model_runner._sync_weights(
            new_state,
            MODEL_MAPPINGS[model_name],
            MODEL_TRANSPOSE_KEYS[model_name],
            None,
        )


class SyncVLLMWrapper:
    """
    Synchronous wrapper around AsyncLLM.
    Allows calling async methods from sync code.
    """

    def __init__(
        self,
        model: str,
        max_model_len: int = 1024,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.95,
        load_format: str = "auto",
        enforce_eager: bool = True,
    ):
        if AsyncEngineArgs is None:
            raise RuntimeError("vLLM async engine is not available. Please install vLLM v1 with: pip install vllm")

        self.bridge = AsyncBridge()
        self.bridge.start()

        # Initialize async engine from sync code
        engine_args = AsyncEngineArgs(
            model=model,
            max_model_len=max_model_len,
            worker_extension_cls="marin.rl.environments.inference_ctx.inflight.worker.WorkerExtension",
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            load_format=load_format,
            enforce_eager=enforce_eager,
        )

        self.engine = self.bridge.run(self._init_engine(engine_args))

    async def _init_engine(self, engine_args):
        """Async initialization."""
        engine = AsyncLLM.from_engine_args(engine_args=engine_args, start_engine_loop=False)
        logger.info(f"Engine initialized: {engine}")
        return engine

    def generate(self, prompts: list[str], sampling_params: SamplingParams) -> str:
        """
        Synchronous generate method - runs async code under the hood.

        Args:
            prompt: Input prompt
            max_tokens: Max tokens to generate

        Returns:
            Generated text
        """
        return self.bridge.run(self._generate_batch_async(prompts, sampling_params))

    async def _generate_batch_async(self, prompts: list[str], sampling_params: SamplingParams) -> list[str]:
        """
        Generate for multiple prompts concurrently.
        Each prompt gets its own request_id and runs in parallel.
        """
        import asyncio

        # Create a task for each prompt
        tasks = []
        for i, prompt in enumerate(prompts):
            task = self._generate_single_in_batch(prompt, i, sampling_params)
            tasks.append(task)

        # Run all tasks concurrently
        results = await asyncio.gather(*tasks)
        return results

    async def _generate_single_in_batch(self, prompt: str, idx: int, sampling_params: SamplingParams) -> str:
        """Generate for a single prompt in a batch."""
        request_id = f"batch-{idx}"

        async for output in self.engine.generate(
            request_id=request_id,
            prompt=prompt,
            sampling_params=sampling_params,
        ):
            if output.finished:
                return output

        return ""

    def update_weights(self, new_state_dict: dict, model_name: str):
        """Synchronous weight update."""
        return self.bridge.run(self._update_weights_async(new_state_dict, model_name))

    async def _update_weights_async(self, new_state_dict: dict, model_name: str):
        """Async weight update."""
        await self.engine.engine_core.collective_rpc_async(
            "update_weight",
            args=(new_state_dict, model_name),
        )

    def reset_prefix_cache(self):
        return self.bridge.run(self.engine.reset_prefix_cache())

    def shutdown(self):
        """Shutdown engine and event loop."""
        if self.engine:
            self.engine.shutdown()
        self.bridge.stop()
