import logging

import numpy as np
from flax import nnx
from marin.rl.environments.inference_ctx.inflight.async_bridge import AsyncBridge
from vllm import AsyncEngineArgs, SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM
from marin.rl.weight_utils import levanter_state_dict_to_nnx_state_on_cpu
from marin.rl.environments.inference_ctx.vllm_utils import MODEL_MAPPINGS, MODEL_TRANSPOSE_KEYS

logger = logging.getLogger(__name__)


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
    
    def __init__(self, model: str, max_model_len: int = 1024, tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.95):
        self.bridge = AsyncBridge()
        self.bridge.start()
        
        # Initialize async engine from sync code
        engine_args = AsyncEngineArgs(
            model=model,
            max_model_len=max_model_len,
            worker_extension_cls="marin.rl.environments.inference_ctx.inflight.worker.WorkerExtension",
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        
        self.engine = self.bridge.run(
            self._init_engine(engine_args)
        )
    
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
        return self.bridge.run(
            self._update_weights_async(new_state_dict, model_name)
        )
    
    async def _update_weights_async(self, new_state_dict: dict, model_name: str):
        """Async weight update."""
        await self.engine.engine_core.collective_rpc_async(
            "update_weight",
            args=(new_state_dict, model_name),
        )

    def reset_prefix_cache(self):
        return self.bridge.run(
            self.engine.reset_prefix_cache()
        )
    
    def shutdown(self):
        """Shutdown engine and event loop."""
        if self.engine:
            self.engine.shutdown()
        self.bridge.stop()