"""SGLang inference engine implementation."""

import torch
import os
from typing import List, Optional, Tuple, Dict, Any, Iterator
import ray
from loguru import logger
import multiprocessing as mp

import sglang.srt.entrypoints.engine
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.utils import (
    assert_pkg_version,
    is_cuda,
    maybe_set_triton_cache_manager,
    set_prometheus_multiproc_dir,
    set_ulimit,
    MultiprocessingSerializer,
)
from sglang.srt.managers.tokenizer_manager import (
    UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromDistributedReqInput,
    InitWeightsUpdateGroupReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
)
from skyrl_train.inference_engines.base import (
    InferenceEngineInterface,
    InferenceEngineInput,
    InferenceEngineOutput,
    NamedWeightsUpdateRequest,
)
from skyrl_train.weight_sync.weight_loader import WeightLoader
from skyrl_train.utils import torch_dtype_to_str
from skyrl_train.inference_engines.sglang.ipc_utils import (
    serialize_ipc_request,
    deserialize_ipc_request,
)


# Patch SGLang's _set_envs_and_config to avoid signal handler issues in Ray actors
# Based on VERL's solution: https://github.com/sgl-project/sglang/issues/6723
# https://github.com/volcengine/verl/blob/v0.4.1/verl/workers/rollout/sglang_rollout/sglang_rollout.py#L85
def _patched_set_envs_and_config(server_args):
    """Patched version of SGLang's _set_envs_and_config that removes signal handler registration."""
    # Set global environments
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = str(int(getattr(server_args, "enable_nccl_nvls", False)))
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "4"
    os.environ["CUDA_MODULE_LOADING"] = "AUTO"

    # Set prometheus env vars
    if server_args.enable_metrics:
        set_prometheus_multiproc_dir()

    # Set ulimit
    set_ulimit()

    # Fix triton bugs
    if server_args.tp_size * server_args.dp_size > 1:
        # FIXME: remove this after https://github.com/triton-lang/triton/pull/4295 is used as a dependency.
        maybe_set_triton_cache_manager()

    # Check flashinfer version
    if server_args.attention_backend == "flashinfer":
        assert_pkg_version(
            "flashinfer_python",
            "0.2.5",
            "Please uninstall the old version and reinstall the latest version by following the instructions at https://docs.flashinfer.ai/installation.html.",
        )
    if is_cuda():
        assert_pkg_version(
            "sgl-kernel",
            "0.1.1",
            "Please reinstall the latest version with `pip install sgl-kernel --force-reinstall`",
        )

    # Set mp start method
    mp.set_start_method("spawn", force=True)

    # We do NOT register signal handlers here to avoid Ray actor issues
    # Original SGLang code had: signal.signal(signal.SIGCHLD, sigchld_handler)
    # But this fails in Ray actors since signal handlers only work in main thread


# Apply the patch
sglang.srt.entrypoints.engine._set_envs_and_config = _patched_set_envs_and_config


# TODO(charlie): duplicate of setup_envvars_for_vllm, is it needed?
def setup_envvars_for_sglang(kwargs, bundle_indices):
    distributed_executor_backend = kwargs.pop("distributed_executor_backend", None)
    noset_visible_devices = kwargs.pop("noset_visible_devices", None)
    if distributed_executor_backend == "ray":
        # a hack to make the script work.
        # stop ray from manipulating *_VISIBLE_DEVICES
        # at the top-level when the distributed_executor_backend is ray.
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ.pop("ROCR_VISIBLE_DEVICES", None)
        os.environ.pop("HIP_VISIBLE_DEVICES", None)
        pass
    elif noset_visible_devices:
        # We need to set CUDA_VISIBLE_DEVICES to the ray assigned GPU
        # when the distributed_executor_backend is not rayargs and
        # RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES is set.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(ray.get_gpu_ids()[0])


class SGLangWeightTransferReceiver:
    """Receives weights via CUDA IPC for SGLang.

    This receiver is used inside the custom_weight_loader function
    which is invoked by SGLang's internal mechanisms.

    Note: this is not used for the broadcast path.
    Because unlike vLLM where we control WorkerWrap and can store the
    process group there, SGLang's custom_weight_loader only receives (model, tensors).
    The process group (_model_update_group) is stored in model_runner, which is not
    accessible from the model object. We also cannot create the group lazily inside
    custom_weight_loader because torch.distributed group creation requires coordination
    (all processes must join at the same time), and by the time custom_weight_loader
    is called, the training side has already completed its init. Therefore, broadcast
    uses SGLang's native update_weights_from_distributed API which has internal access
    to the process group.
    """

    def __init__(self, model_dtype: str, device_id: int) -> None:
        """Initialize the receiver.

        Args:
            model_dtype: Model's dtype as string (e.g., "bfloat16").
            device_id: Target CUDA device index.
        """
        self._model_dtype = model_dtype
        self._device_id = device_id

    def receive_ipc_weights(
        self,
        request: NamedWeightsUpdateRequest,
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        """Receive weights by opening CUDA IPC handles.

        Args:
            request: Weight update request with IPC handles.

        Yields:
            (name, tensor) tuples for each weight.
        """
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        physical_gpu_id = str(props.uuid)

        for i in range(len(request["names"])):
            ipc_handles = request["extras"][i]["ipc_handles"]
            dtype = request["dtypes"][i]
            weight_name = request["names"][i]

            assert dtype == self._model_dtype, f"mismatch dtype: src {dtype}, dst {self._model_dtype}"

            handle = ipc_handles[physical_gpu_id]
            func, args = handle
            list_args = list(args)
            # Change device id to the current device id
            # in case two processes have different CUDA_VISIBLE_DEVICES
            list_args[6] = self._device_id
            weight = func(*list_args)
            yield weight_name, weight


def sglang_custom_weight_loader(model, named_tensors):
    """Custom weight loader for SGLang that handles CUDA IPC.

    This function is called by SGLang's model runner to load weights.
    It reconstructs tensors from SkyRL's NamedWeightsUpdateRequest
    using CUDA IPC handles.
    """
    # Extract tensor name and data
    name, tensor = named_tensors[0]
    if name != "ipc_request":
        raise ValueError(f"Expected tensor name 'ipc_request', got: {name}")

    # Deserialize request from tensor
    request = deserialize_ipc_request(tensor)

    # Get model info and create receiver
    model_dtype = torch_dtype_to_str(next(model.parameters()).dtype)
    device_id = next(model.parameters()).device.index
    receiver = SGLangWeightTransferReceiver(model_dtype, device_id)

    # Receive weights via IPC
    weights_to_load = list(receiver.receive_ipc_weights(request))
    model.load_weights(weights_to_load)


CUSTOM_WEIGHT_LOADER_PATH = "skyrl_train.inference_engines.sglang.sglang_engine.sglang_custom_weight_loader"


class SGLangWeightLoader(WeightLoader):
    """Loads weights into SGLang engine, managing weight transfer coordination.

    This loader encapsulates the SGLang-specific weight loading logic for both
    IPC and broadcast transfer paths:
    - IPC: Uses update_weights_from_tensor with our custom_weight_loader
    - Broadcast: Uses SGLang's native update_weights_from_distributed API
    """

    def __init__(self, engine: Any, tp_size: int) -> None:
        """Initialize the loader.

        Args:
            engine: The SGLang engine.
            tp_size: Tensor parallel size.
        """
        self._engine = engine
        self._tp_size = tp_size

    async def init_communicator(
        self,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str,
    ) -> Tuple[bool, str]:
        """Initialize the process group for broadcast weight sync.

        This is only needed for the broadcast path. IPC path does not require
        a process group since it uses CUDA IPC handles directly.

        Args:
            master_address: Master address for the process group.
            master_port: Master port for the process group.
            rank_offset: Rank offset for this process.
            world_size: Total world size.
            group_name: Name of the process group.
            backend: Backend to use (e.g., "nccl", "gloo").

        Returns:
            Tuple of (success, message).
        """
        obj = InitWeightsUpdateGroupReqInput(
            master_address=master_address,
            master_port=master_port,
            rank_offset=rank_offset,
            world_size=world_size,
            group_name=group_name,
            backend=backend,
        )
        # NOTE(charlie): Call the async method on tokenizer_manager directly to avoid event loop
        # conflicts. Same underlying implementation: https://github.com/sgl-project/sglang/blob/v0.4.8.post1/python/sglang/srt/model_executor/model_runner.py#L689
        return await self._engine.tokenizer_manager.init_weights_update_group(obj, None)

    async def load_weights(self, request: NamedWeightsUpdateRequest) -> None:
        """Load weights by coordinating with SGLang's weight update APIs.

        Args:
            request: Weight update request containing names, dtypes, shapes,
                    and optionally IPC handles.
        """
        extras = request.get("extras")
        is_ipc = extras is not None and len(extras) > 0 and "ipc_handles" in extras[0]

        if is_ipc:
            await self._load_via_ipc(request)
        else:
            await self._load_via_broadcast(request)

    async def _load_via_ipc(self, request: NamedWeightsUpdateRequest) -> None:
        """Load weights via CUDA IPC using custom weight loader.

        Uses SGLangWeightTransferReceiver internally to receive weights
        from IPC handles.
        """
        tensor_array = serialize_ipc_request(request)

        # Use SGLang's API to update weights with our custom loader
        request_tensor = [("ipc_request", tensor_array)]
        obj = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=[
                MultiprocessingSerializer.serialize(request_tensor) for _ in range(self._tp_size)
            ],
            load_format=CUSTOM_WEIGHT_LOADER_PATH,
            flush_cache=False,  # TODO(charlie): flush cache on last weight update?
        )

        success, message = await self._engine.tokenizer_manager.update_weights_from_tensor(obj, None)
        if not success:
            raise RuntimeError(f"IPC weight update failed: {message}")

    async def _load_via_broadcast(self, request: NamedWeightsUpdateRequest) -> None:
        """Load weights via torch.distributed broadcast.

        Uses SGLang's native update_weights_from_distributed API which internally
        uses the process group created during init_weights_update_group.
        """
        assert len(request["names"]) == 1, (
            f"Broadcast only supports a single weight at a time, got {len(request['names'])} entries"
        )

        obj = UpdateWeightsFromDistributedReqInput(
            name=request["names"][0], dtype=request["dtypes"][0], shape=request["shapes"][0]
        )

        success, message = await self._engine.tokenizer_manager.update_weights_from_distributed(obj, None)
        if not success:
            raise RuntimeError(f"Broadcast weight update failed: {message}")


class SGLangInferenceEngine(InferenceEngineInterface):
    """SGLang inference engine that implements InferenceEngineInterface."""

    def __init__(self, *args, bundle_indices: Optional[List[int]] = None, **kwargs):
        setup_envvars_for_sglang(kwargs, bundle_indices)

        # Store common attributes
        self._tp_size = kwargs.get("tp_size", 1)
        if self._tp_size > 1:
            raise ValueError(
                "As of now, we don't support tensor parallel inference engine with SGLang. "
                "Please set `inference_engine_tensor_parallel_size` to 1."
            )
        self.tokenizer = kwargs.pop("tokenizer", None)

        # Unused kwargs
        _ = kwargs.pop("num_gpus", 1)

        # Add custom weight loader
        kwargs["custom_weight_loader"] = CUSTOM_WEIGHT_LOADER_PATH

        # Always use token-in-token-out SGLang engine
        # NOTE(Charlie): unlike vLLM, SGLang cannot do token-in-token-out and
        # token-in-text-out in the same engine config.
        kwargs["skip_tokenizer_init"] = True

        # Create the SGLang engine (signal handler issue is now fixed by patching)
        self.engine = Engine(**kwargs)
        self.max_model_len = self.engine.tokenizer_manager.context_len
        logger.info(f"Created SGLang engine with kwargs: {kwargs}")

        # Create weight loader for coordinating weight updates
        self._weight_loader = SGLangWeightLoader(self.engine, self._tp_size)

    def tp_size(self):
        return self._tp_size

    def pp_size(self):
        # Pipeline parallelism not supported for SGLang
        return 1

    def dp_size(self):
        # TODO(tgriggs): EP/DP not yet supported for SGLang
        return 1

    def _preprocess_prompts(self, input_batch: InferenceEngineInput):
        """Preprocess prompts for SGLang generation."""
        prompts = input_batch.get("prompts")
        prompt_token_ids = input_batch.get("prompt_token_ids")
        request_sampling_params = input_batch.get("sampling_params")

        assert prompts is None and prompt_token_ids is not None, (
            "SGLangInferenceEngine only accepts `prompt_token_ids`, not `prompts`."
        )

        # Use request sampling params if provided.
        sampling_params = request_sampling_params if request_sampling_params is not None else {}

        return prompt_token_ids, sampling_params

    def _postprocess_outputs(self, outputs):
        """Process SGLang outputs to match expected format."""
        responses: List[str] = []
        stop_reasons: List[str] = []
        response_ids: List[List[int]] = []

        for output in outputs:
            response_ids.append(output["output_ids"])
            responses.append(self.tokenizer.decode(output["output_ids"], skip_special_tokens=True))
            stop_reasons.append(output["meta_info"]["finish_reason"]["type"])

        return InferenceEngineOutput(
            responses=responses,
            response_ids=response_ids,
            stop_reasons=stop_reasons,
            response_logprobs=None,
        )

    async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
        """Generate responses using SGLang engine."""
        token_ids_prompts, sampling_params = self._preprocess_prompts(input_batch)
        outputs = await self.engine.async_generate(input_ids=token_ids_prompts, sampling_params=sampling_params)
        return self._postprocess_outputs(outputs)

    async def chat_completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        # TODO(charlie): implement this in the future
        raise NotImplementedError()

    async def completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        # TODO(charlie): implement this in the future
        raise NotImplementedError()

    async def init_weight_update_communicator(
        self, master_addr, master_port, rank_offset, world_size, group_name, backend, override_existing: bool = False
    ):
        """Initialize weight update communicator for SGLang.

        This initializes the process group for broadcast weight sync. Only needed
        when using the broadcast transfer path, not for IPC.
        """
        return await self._weight_loader.init_communicator(
            master_addr, master_port, rank_offset, world_size, group_name, backend
        )

    async def update_named_weights(self, request: NamedWeightsUpdateRequest) -> None:
        """Update named weights in SGLang engine."""
        if "names" not in request:
            raise ValueError(f"Expected update weight request with 'names' entry, got keys: {request.keys()}")

        await self._weight_loader.load_weights(request)

    async def wake_up(self, tags: Optional[List[str]] = None):
        """Wake up the engine. For multi-stage waking up, pass in `"weight"` or `"kv_cache"` to tags."""
        obj = ResumeMemoryOccupationReqInput(tags=tags)
        # Call the underlying async method for the same reason as in `init_weight_update_communicator`
        await self.engine.tokenizer_manager.resume_memory_occupation(obj, None)
        logger.info(
            f"From SGLang engine -- Free GPU memory after wake up with tags {tags if tags is not None else 'None'}: "
            + f"{torch.cuda.mem_get_info()[0] / 1024**2:.1f} MB"
        )

    async def sleep(self, tags: Optional[List[str]] = None):
        """Put engine to sleep."""
        obj = ReleaseMemoryOccupationReqInput(tags=tags)
        # Call the underlying async method for the same reason as in `init_weight_update_communicator`
        await self.engine.tokenizer_manager.release_memory_occupation(obj, None)
        logger.info(
            f"From SGLang engine -- Free GPU memory after sleep with tags {tags if tags is not None else 'None'}: "
            + f"{torch.cuda.mem_get_info()[0] / 1024**2:.1f} MB"
        )

    async def teardown(self):
        """Shutdown the SGLang engine."""
        self.engine.shutdown()

    async def reset_prefix_cache(self):
        """Reset prefix cache in SGLang engine."""
        # Call the underlying async method for the same reason as in `init_weight_update_communicator`
        return await self.engine.tokenizer_manager.flush_cache()

    async def pause_generation(self) -> None:
        raise NotImplementedError("Pausing generation is not supported for SGLang inference engines.")

    async def resume_generation(self, policy_step: int) -> None:
        raise NotImplementedError("Resuming generation is not supported for SGLang inference engines.")


SGLangRayActor = ray.remote(SGLangInferenceEngine)
