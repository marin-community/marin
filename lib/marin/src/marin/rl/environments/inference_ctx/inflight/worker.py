# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import numpy as np
from marin.rl.environments.inference_ctx.inflight.async_bridge import AsyncBridge
from marin.rl.weight_utils import levanter_state_dict_to_nnx_state_on_cpu
from marin.rl.environments.inference_ctx.vllm_utils import MODEL_MAPPINGS, MODEL_TRANSPOSE_KEYS

logger = logging.getLogger(__name__)

try:
    from vllm import AsyncEngineArgs, SamplingParams
    from vllm.v1.engine.async_llm import AsyncLLM
except ImportError:
    AsyncEngineArgs = None
    SamplingParams = None
    AsyncLLM = None
    logger.warning("vLLM async engine is not available. Please install vLLM v1 with: pip install vllm")


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
                # Check any conflicts between worker and worker_extension_cls
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

                # FIX: Create a new class dynamically instead of modifying __bases__
                # Direct __bases__ modification fails with:
                #   TypeError: __bases__ assignment: 'WorkerExtension' deallocator differs from 'object'
                # Creating a new class avoids this CPython restriction.
                worker_class = type(
                    f"{worker_class.__name__}WithExtension",
                    (worker_extension_cls, worker_class),  # Extension first for proper MRO
                    {},  # No additional attributes needed
                )
                # Update the config so the new class is used for future references
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
            else:
                # Use warning_once if available, otherwise regular warning
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

    WorkerWrapperBase.init_worker = _patched_init_worker
    logger.info("Applied MRO fix for vLLM V1 worker extension injection")


# Apply the monkeypatch when this module is imported
if AsyncEngineArgs is not None:
    _apply_worker_extension_mro_fix()


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
