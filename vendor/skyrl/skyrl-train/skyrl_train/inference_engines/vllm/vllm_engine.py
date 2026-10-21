import json
import os
from pathlib import Path
import tempfile
import threading
from typing import List, Any, Dict, Optional, Tuple, Iterator, AsyncGenerator
from dataclasses import asdict, dataclass, fields as _dataclass_fields, replace
from loguru import logger
from http import HTTPStatus
import ray
import torch
import numpy as np
import asyncio
import vllm
from zmq.error import ZMQError
from types import SimpleNamespace
from vllm import SamplingParams
from vllm.inputs import TokensPrompt
from vllm.logprobs import FlatLogprobs
from vllm.distributed.parallel_state import get_dp_group, get_ep_group, get_pp_group
from vllm.distributed.weight_transfer.base import WeightTransferUpdateRequest
from vllm.renderers.online_renderer import OnlineRenderer

from marinskyrl.resource_locator import is_cloud_uri, join_resource_path
from skyrl_train.numa_policy import NUMA_AFFINITY_ENV
from skyrl_train.config.behavior_logprobs import (
    ROLLOUT_LOGPROB_VALIDATION_KEY,
    validate_behavior_logprob_sampling,
)
from skyrl_train.config.weight_sync_pause import WeightSyncPausePolicy
from skyrl_train.inference_engines.vllm.online_eagle_trainer import (
    capture_rank_directory,
    capture_rank_name,
    per_worker_capture_token_credit,
    request_id_for_group,
)
from skyrl_train.inference_engines.vllm.route_capture import response_routes
from skyrl_train.io import io

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionLogProb,
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.completion.protocol import CompletionRequest, CompletionResponse
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.engine.protocol import ErrorInfo, ErrorResponse
from vllm.entrypoints.serve.tokenize.protocol import TokenizeChatRequest, TokenizeResponse
from vllm.entrypoints.serve.tokenize.serving import ServingTokenization

from vllm.v1.metrics.loggers import LoggingStatLogger
from vllm.lora.request import LoRARequest
from torch.distributed import DistNetworkError, destroy_process_group
from skyrl_train.distributed.utils import init_custom_process_group
from uuid import uuid4
import warnings
from contextvars import ContextVar


from skyrl_train.inference_engines.base import (
    CHOICE_POLICY_STEPS_KEY,
    InferenceEngineInterface,
    InferenceEngineInput,
    InferenceEngineOutput,
    NamedWeightsUpdateRequest,
    LORA_DISK_LOAD_NAME,
    LORA_DISK_PATH_KEY,
)
from skyrl_train.inference_engines.response_topk import (
    CHAT_LOGPROB_FLOOR,
    CHAT_TOPK_WIDTH_KEY,
    CHOICE_TOPK_KEY,
    select_flat_response_topk,
    select_response_topk,
)
from skyrl_train.inference_engines.vllm.compact_chat import CompactChatEngineClient
from skyrl_train.inference_engines.chat_continuation import EXACT_PROMPT_TOKEN_IDS_KEY
from marinskyrl.inference_placement import InferenceWorkerPlacement
from skyrl_train.inference_engines.placement import inference_worker_placement
from skyrl_train.inference_engines.vllm.numa import set_async_worker_numa_affinity
from skyrl_train.inference_engines.policy_steps import PolicyStepStamps
from skyrl_train.weight_sync.expert_block.receiver import ExpertBlockReceiver
from skyrl_train.weight_sync.weight_loader import WeightLoader
from skyrl_train.weight_sync.vllm_weight_conversion import load_weights_into_vllm
from skyrl_train.weight_sync.weight_extractor import is_weight_sync_dtype_compatible
from skyrl_train.inference_engines.vllm.utils import (
    pop_vllm_wrapper_kwargs,
    apply_openai_sampling,
    ensure_token_ids_in_sse_chunk,
    is_port_collision,
    PrefixCacheHitRateAccumulator,
)
from skyrl_train.inference_engines.utils import VLLM_DATA_PARALLEL_MASTER_PORT_COUNT
from skyrl_train.inference_engines.vllm.stats import (
    IntervalReadMode,
    VLLMCumulativeStats,
    VLLMCurrentStats,
    VLLMEngineStatsSnapshot,
    VLLMIntervalStats,
    snapshot_vllm_prometheus_metrics,
)
from skyrl_train.utils import get_tcp_url, str_to_torch_dtype, torch_dtype_to_str
import time


_exact_chat_prompt_token_ids: ContextVar[list[int] | None] = ContextVar("exact_chat_prompt_token_ids", default=None)


def _create_async_engine(engine_args, stat_loggers, data_parallel_master_ports):
    if data_parallel_master_ports is None:
        return vllm.AsyncLLMEngine.from_engine_args(engine_args, stat_loggers=stat_loggers)
    if len(data_parallel_master_ports) != VLLM_DATA_PARALLEL_MASTER_PORT_COUNT:
        raise ValueError(
            f"vLLM data parallelism requires {VLLM_DATA_PARALLEL_MASTER_PORT_COUNT} master ports, "
            f"got {len(data_parallel_master_ports)}"
        )
    *worker_ports, master_port = data_parallel_master_ports
    vllm_config = engine_args.create_engine_config()
    vllm_config.parallel_config._data_parallel_master_port_list = worker_ports
    vllm_config.parallel_config.data_parallel_master_port = master_port
    return vllm.AsyncLLMEngine.from_vllm_config(
        vllm_config,
        stat_loggers=stat_loggers,
        enable_log_requests=False,
        disable_log_stats=True,
    )


def _create_async_engine_with_port_collision_retries(engine_args, stat_loggers):
    """Start one independent engine, retrying opportunistically selected ports."""
    import random

    max_attempts = 5
    backoff_base_seconds = 15.0
    for attempt in range(max_attempts):
        stagger = random.uniform(1.5, 3.0)
        logger.info(
            f"Engine startup stagger: sleeping {stagger:.2f}s "
            f"(attempt {attempt + 1}/{max_attempts}) to avoid port collisions"
        )
        time.sleep(stagger)
        try:
            return _create_async_engine(engine_args, stat_loggers, data_parallel_master_ports=None)
        except (DistNetworkError, RuntimeError, ZMQError) as error:
            if not is_port_collision(error) or attempt == max_attempts - 1:
                raise
            backoff = backoff_base_seconds * (2**attempt)
            logger.warning(
                f"Engine init hit a port collision (EADDRINUSE) on attempt "
                f"{attempt + 1}/{max_attempts}; retrying in {backoff:.0f}s: {str(error).splitlines()[0]}"
            )
            time.sleep(backoff)

    raise AssertionError("engine startup loop returned without an engine or exception")


class SkyRLOpenAIServingChat(OpenAIServingChat):
    """Serve exact chat tokens and ID-only internal candidate heads."""

    id_only_logprob_requests: set[str]

    async def render_chat_request(self, request):
        rendered = await super().render_chat_request(request)
        exact_ids = _exact_chat_prompt_token_ids.get()
        if exact_ids is None or not isinstance(rendered, tuple):
            return rendered
        conversation, engine_inputs = rendered
        if len(engine_inputs) != 1:
            raise ValueError("Exact OpenCode continuation requires one rendered prompt")
        return conversation, [TokensPrompt(prompt_token_ids=list(exact_ids))]

    async def create_chat_completion_with_exact_prompt(
        self,
        request: ChatCompletionRequest,
        raw_request: "_MinimalRequest",
        prompt_token_ids: list[int],
    ) -> ChatCompletionResponse | ErrorResponse | AsyncGenerator[str, None]:
        id_only = (
            request.logprobs
            and request.top_logprobs is not None
            and request.top_logprobs > 0
            and request.return_tokens_as_token_ids
            and request.n == 1
        )
        capture_id = f"chatcmpl-{self._base_request_id(raw_request, request.request_id)}" if id_only else None
        token = _exact_chat_prompt_token_ids.set(prompt_token_ids)
        try:
            if capture_id is not None:
                self.id_only_logprob_requests.add(capture_id)
            return await super().create_chat_completion(request, raw_request)
        finally:
            if capture_id is not None:
                self.id_only_logprob_requests.discard(capture_id)
            _exact_chat_prompt_token_ids.reset(token)

    def _get_top_logprobs(self, logprobs, top_logprobs, tokenizer, should_return_as_token_id, return_all=False):
        if _exact_chat_prompt_token_ids.get() is None or not should_return_as_token_id:
            return super()._get_top_logprobs(logprobs, top_logprobs, tokenizer, should_return_as_token_id, return_all)
        # Internal captures consume exact IDs and scores. Token-ID placeholders
        # already identify candidates; a byte array for each placeholder is unused.
        return [
            ChatCompletionLogProb(token=f"token_id:{token_id}", logprob=max(value.logprob, CHAT_LOGPROB_FLOOR), bytes=None)
            for index, (token_id, value) in enumerate(logprobs.items())
            if return_all or top_logprobs == -1 or (top_logprobs is not None and index < top_logprobs)
        ]


def _build_error_response(message: str, type_phrase: str, code: int) -> Dict[str, Any]:
    """Build the pinned vLLM fork's nested OpenAI error response."""
    return ErrorResponse(
        error=ErrorInfo(message=message, type=type_phrase, code=code),
    ).model_dump()


# Guard so the fake/meta registration runs at most once per worker process.
_NORM_META_FAKES_REGISTERED = False


def ensure_norm_meta_fakes_registered() -> None:
    """Register Meta/fake kernels for vLLM's RMSNorm custom ops (idempotent).

    WHY: the layerwise weight-reload bracket (``skyrl_begin_weight_reload`` ->
    ``initialize_layerwise_reload``) restores every layer's params/buffers onto
    the **meta** device, then the CP>1 finalize path materializes those meta
    tensors and traces ``process_weights_after_loading`` over them. That trace
    dispatches ``torch.ops._C.rms_norm`` (and its sibling
    ``fused_add_rms_norm``) with Meta tensors, but the vLLM fork registers those
    C++ ops for ``torch::kCUDA`` ONLY (csrc/torch_bindings.cpp) — no Meta/fake
    kernel — so the dispatch dies with:
      ``NotImplementedError: _C::rms_norm: attempted to run this operator with
      Meta tensors, but there was no fake impl or Meta kernel registered``
    which killed all 3 inference engines during ``sync_weights`` on the CP4
    30B-A3B RL cell (agent_logs/2026-07-07_grid30bc_rmsnorm_meta_sync_weights.md).
    The CP1 sibling cell never traces these on meta, so it survived — hence this
    is CP>1-specific.

    Both ops are shape/dtype-preserving in-place normalizations whose C++ schema
    (torch_bindings.cpp) returns nothing:
      ``rms_norm(Tensor! result, Tensor input, Tensor weight, float epsilon) -> ()``
      ``fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor weight, float epsilon) -> ()``
    The correct fake for a mutating op that returns nothing is a no-op returning
    ``None`` (identical shape to vLLM's own ``_C::scaled_fp4_quant.out`` fake).
    This ONLY fires under meta-tensor tracing (the CP>1 sync path); real CUDA
    execution keeps using the registered CUDA kernel, so numerics/MoE routing are
    UNCHANGED. Ships via the /app source sync (no gpu-rl image rebuild).

    Called from the weight-reload bracket rather than at import time because the
    ``_C`` custom-op library is only guaranteed loaded once the vLLM model is
    built — by the first weight sync ``torch.ops._C.rms_norm`` exists.
    """
    global _NORM_META_FAKES_REGISTERED
    if _NORM_META_FAKES_REGISTERED:
        return

    _C = getattr(torch.ops, "_C", None)
    if _C is None:  # vLLM C-extension not loaded yet; try again on the next call.
        return

    from torch.library import register_fake

    # (name, arg-count) pairs. Fake matches the C++ schema exactly: a mutating op
    # that returns nothing => the fake returns None (no output tensors to fake).
    def _rms_norm_fake(result, input, weight, epsilon):  # -> ()
        return None

    def _fused_add_rms_norm_fake(input, residual, weight, epsilon):  # -> ()
        return None

    registrations = (
        ("rms_norm", "_C::rms_norm", _rms_norm_fake),
        ("fused_add_rms_norm", "_C::fused_add_rms_norm", _fused_add_rms_norm_fake),
    )
    for attr, qualname, fake in registrations:
        if not hasattr(_C, attr):
            continue  # op not present in this vLLM build; nothing to register.
        try:
            register_fake(qualname, fake)
            logger.info(f"Registered Meta/fake kernel for {qualname} (CP>1 weight-sync fix)")
        except RuntimeError as e:
            # A fake already exists (e.g. a future vLLM registers one, or a
            # sibling engine in-process already ran this) — that's fine, leave it.
            logger.debug(f"Skipping fake registration for {qualname}: {e}")

    _NORM_META_FAKES_REGISTERED = True


@dataclass
class Logprob:
    logprob: float
    rank: int
    token_id: str


def setup_envvars_for_vllm(kwargs, bundle_indices):
    noset_visible_devices = kwargs.pop("noset_visible_devices")
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"  # TODO(Charlie): may not be needed.

    # When custom all-reduce is disabled (e.g. for TP=2 on H100 where
    # SymmMemCommunicator rendezvous fails), also disable symmetric memory
    # via env var — the engine arg alone doesn't prevent SymmMemCommunicator
    # from being instantiated.
    if kwargs.get("disable_custom_all_reduce"):
        os.environ["VLLM_ALLREDUCE_USE_SYMM_MEM"] = "0"
        logger.info("setup_envvars_for_vllm: set VLLM_ALLREDUCE_USE_SYMM_MEM=0 (disable_custom_all_reduce=True)")
    if kwargs.get("distributed_executor_backend") == "ray":
        # a hack to make the script work.
        # stop ray from manipulating *_VISIBLE_DEVICES
        # at the top-level when the distributed_executor_backend is ray.
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ.pop("ROCR_VISIBLE_DEVICES", None)
        os.environ.pop("HIP_VISIBLE_DEVICES", None)
    elif noset_visible_devices:
        # We need to set CUDA_VISIBLE_DEVICES to the ray assigned GPU
        # when the distributed_executor_backend is not rayargs and
        # RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES is set.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(ray.get_gpu_ids()[0])

    num_gpus = kwargs.pop("num_gpus")
    if bundle_indices is not None:
        os.environ["VLLM_RAY_PER_WORKER_GPUS"] = str(num_gpus)
        os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))
        logger.info(f"creating LLM with bundle_indices={bundle_indices}")

    # Set NUMA CPU affinity for single-GPU (TP=1) inference actors.
    # For TP>1, affinity is set per-worker via WorkerWrap.set_numa_affinity().
    #
    # When NUMA affinity is enabled, we also disable vLLM V1 multiprocessing.
    # vLLM's V1 engine spawns EngineCore as a separate subprocess using
    # multiprocessing with start_method="spawn" (forced when running inside a
    # Ray actor — see vllm.utils._maybe_force_spawn). Spawned processes do NOT
    # inherit the parent's CPU affinity, so NUMA binding set here would be lost.
    # Disabling V1 multiprocessing forces EngineCore to run in the same process,
    # where our affinity settings take effect.
    executor_backend = kwargs.get("distributed_executor_backend")
    logger.info(
        f"setup_envvars_for_vllm: distributed_executor_backend={executor_backend}, "
        f"{NUMA_AFFINITY_ENV}={os.environ.get(NUMA_AFFINITY_ENV, '<unset>')}, "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}, "
        f"VLLM_ENABLE_V1_MULTIPROCESSING={os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING', '<unset>')}"
    )
    # NOTE: the `mp` executor backend (Qwen3-Next R3 capture path) MUST keep v1
    # multiprocessing ENABLED — it spawns its TP worker subprocesses via the v1 mp
    # path, and disabling it cancels the shm message queue at warm-up
    # ("RuntimeError: cancelled"). NUMA single-GPU pinning does not apply to the
    # multi-GPU mp engine anyway, so skip this branch for mp.
    if executor_backend not in ("ray", "mp"):
        try:
            from skyrl_train.utils.numa import is_numa_affinity_enabled, set_numa_affinity_for_gpu

            numa_enabled = is_numa_affinity_enabled()
            logger.info(f"setup_envvars_for_vllm: numa_enabled={numa_enabled}")
            if numa_enabled:
                os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
                logger.info("setup_envvars_for_vllm: set VLLM_ENABLE_V1_MULTIPROCESSING=0 for NUMA affinity")
                cuda_devs = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                if cuda_devs:
                    gpu_ids = [int(x) for x in cuda_devs.split(",")]
                    if len(gpu_ids) == 1:
                        set_numa_affinity_for_gpu(gpu_ids[0])
        except Exception as e:
            logger.warning(f"setup_envvars_for_vllm: NUMA affinity setup failed: {e}")


class WorkerWrap:
    def set_numa_affinity(self):
        """Set CPU affinity to match this worker's GPU NUMA node.

        Called via collective_rpc for TP>1 configurations so each
        vLLM EngineCore process binds to its GPU's local CPUs.
        """
        try:
            from skyrl_train.utils.numa import set_numa_affinity_for_gpu

            gpu_id = self.device.index if self.device is not None else 0
            set_numa_affinity_for_gpu(gpu_id)
        except Exception:
            pass

    def test_rpc(self, *args, **kwargs):
        """Test RPC call to worker"""
        return args, kwargs

    def begin_online_eagle_capture(self, config):
        """Begin bounded verifier-state capture on the resident model runner."""
        resolved = dict(config)
        parallel_config = self.model_runner.parallel_config
        if parallel_config.tensor_parallel_size != 1 or parallel_config.pipeline_parallel_size != 1:
            raise RuntimeError("Online EAGLE training requires vLLM tensor and pipeline parallel size 1")
        worker_rank = parallel_config.data_parallel_rank
        worker_count = parallel_config.data_parallel_size
        global_token_budget = int(resolved["max_tokens"])
        if global_token_budget < worker_count:
            raise ValueError("Online EAGLE capture token budget must cover every data-parallel rank")
        resolved["max_tokens"] = per_worker_capture_token_credit(
            global_max_tokens=global_token_budget,
            max_window_tokens=int(resolved["max_window_tokens"]),
            worker_count=worker_count,
            worker_index=worker_rank,
        )
        resolved["capture_target_snapshot"] = worker_rank == 0
        reserved_gpu_memory_gib = float(resolved.pop("reserved_gpu_memory_gib"))
        total_memory_gib = torch.cuda.get_device_properties(self.device).total_memory / 2**30
        gpu_memory_utilization = float(self.model_runner.cache_config.gpu_memory_utilization)
        unreserved_memory_gib = total_memory_gib * (1 - gpu_memory_utilization)
        if reserved_gpu_memory_gib > unreserved_memory_gib:
            raise RuntimeError(
                "Online EAGLE training reserves "
                f"{reserved_gpu_memory_gib:.2f} GiB, but vLLM gpu_memory_utilization="
                f"{gpu_memory_utilization:.3f} leaves only {unreserved_memory_gib:.2f} GiB "
                f"on this {total_memory_gib:.2f} GiB device"
            )
        result = self.model_runner.begin_online_eagle_capture(resolved)
        return {
            **result,
            "node_id": str(ray.get_runtime_context().get_node_id()),
            "capture_token_credit": resolved["max_tokens"],
        }

    def seal_online_eagle_capture(self, destination):
        """Seal verifier-state capture and publish it to cloud storage."""
        if not is_cloud_uri(destination):
            raise ValueError(f"Online EAGLE capture destination must be cloud-backed: {destination}")
        worker_rank = self.model_runner.parallel_config.data_parallel_rank
        rank_destination = join_resource_path(destination, capture_rank_name(worker_rank))
        try:
            with tempfile.TemporaryDirectory(prefix="marinskyrl-eagle-capture-") as scratch:
                rank_output_dir = capture_rank_directory(Path(scratch), worker_rank)
                result = self.model_runner.seal_online_eagle_capture(str(rank_output_dir))
                io.upload_directory(str(rank_output_dir), rank_destination)
        except Exception as error:
            logger.exception("Online EAGLE capture seal or publication failed for worker rank {}", worker_rank)
            return {
                "active": False,
                "worker_rank": worker_rank,
                "error": f"{type(error).__name__}: {error}",
            }
        return {**result, "active": True, "path": rank_destination}

    def init_weight_update_communicator(
        self,
        master_address,
        master_port,
        rank_offset,
        world_size,
        group_name,
        backend="nccl",
        override_existing: bool = False,
    ):
        """Init torch process group for model weights update"""
        assert torch.distributed.is_initialized(), "default torch process group must be initialized"
        assert group_name != "", "group name must not be empty"

        if getattr(self, "_model_update_group", None):
            if override_existing:
                logger.info("Destroying existing model update group")
                destroy_process_group(self._model_update_group)
                self._model_update_group = None
            else:
                warnings.warn(
                    "Detected an existing weights update group. For overriding, use `generator.override_existing_update_group=True`"
                )

        rank = torch.distributed.get_rank() + rank_offset
        logger.info(
            f"torch.distributed.get_rank(): {torch.distributed.get_rank()}, rank_offset: {rank_offset}, rank: {rank}, world_size: {world_size}, group_name: {group_name}"
        )

        self._model_update_group = init_custom_process_group(
            backend=backend,
            init_method=get_tcp_url(master_address, master_port),
            world_size=world_size,
            rank=rank,
            group_name=group_name,
        )
        logger.info(
            f"init_weight_update_communicator: master_address={master_address}, master_port={master_port}, ",
            f"rank={rank}, world_size={world_size}, group_name={group_name}",
        )

        # Create receiver now that we have all the state
        self._weight_receiver = VLLMWeightTransferReceiver(
            model_update_group=self._model_update_group,
            model_config=self.model_config,
            device=self.device,
        )

    @staticmethod
    def _apply_fp8_weight_loader_patches(*, fuse_weights: bool = False):
        """Patch Fp8LinearMethod.process_weights_after_loading to preserve weight_loader.

        Following verl's approach: after FP8 processing creates new Parameter objects,
        copy custom attributes (weight_loader, output_dim, input_dim, subclass_type)
        from the original specialized parameter so weight sync can reload weights.
        """
        if not fuse_weights:
            return

        try:
            from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
        except ImportError:
            return

        original_process = Fp8LinearMethod.process_weights_after_loading

        def patched_process(self_method, layer, *args, **kwargs):
            # Save original param attributes before processing
            saved_attrs = {}
            for pname, param in layer.named_parameters():
                attrs = {}
                for attr in (
                    "weight_loader",
                    "output_dim",
                    "input_dim",
                    "_output_dim",
                    "_input_dim",
                    "packed_dim",
                    "packed_factor",
                    "tp_rank",
                    "tp_size",
                    "logical_widths",
                    "output_sizes",
                ):
                    if hasattr(param, attr):
                        attrs[attr] = getattr(param, attr)
                attrs["subclass_type"] = type(param)
                saved_attrs[pname] = attrs

            # Call original process_weights_after_loading
            result = original_process(layer, *args, **kwargs)

            # Restore attributes on new parameters
            for pname, param in layer.named_parameters():
                if pname in saved_attrs:
                    for attr, value in saved_attrs[pname].items():
                        try:
                            setattr(param, attr, value)
                        except (AttributeError, TypeError):
                            pass

            return result

        Fp8LinearMethod.process_weights_after_loading = patched_process

    def skyrl_begin_weight_reload(self) -> None:
        """RENAMED from ``start_weight_update`` AND NOW WIRED (the #1685 fix).

        WHY RENAMED: the baked vLLM ``gpu_worker.Worker`` now ALSO defines
        ``start_weight_update``/``finish_weight_update`` (vllm/v1/worker/gpu_worker.py),
        so a WorkerWrap method of the SAME name trips vLLM's ``init_worker_extension``
        shadow-assert ("Worker class already has an attribute finish_weight_update")
        -> EngineCore crash. (The base Worker versions also REQUIRE a configured
        ``weight_transfer_engine``, which SkyRL's NCCL-broadcast path does NOT set up, so
        they cannot be reused directly — hence this SkyRL-native bracket.)

        WHY NOW WIRED (2026-06-27, the disagg-engine diag root cause): on CoreWeave H100
        the unquantized MoE backend auto-selects FlashInfer CUTLASS, whose
        ``process_weights_after_loading`` applies ``swap_w13_to_w31`` ([gate;up]->[up;gate]
        kernel layout) at the INITIAL from-disk load. The RL disaggregated update path
        does immediate per-chunk ``model.load_weights`` with NO subsequent
        ``process_weights_after_loading`` (base_loader.py:80 runs it only on load_model),
        so the update OVERWRITES the [up;gate] kernel buffer with raw checkpoint [gate;up]
        and NEVER re-swaps -> the kernel reads transposed halves -> token-salad. PROVEN by
        the kernel-format discriminator: PRE(from-disk)=[up;gate], POST(RL-update)=[gate;up],
        both ep ranks, layers 0 & 24. The cure is to bracket the whole multi-chunk sync with
        vLLM's layerwise reload so ``finalize`` re-runs ``process_weights_after_loading``
        (the swap) EXACTLY once at the end. Triton clusters select no-swap backends, so
        finalize is swap-inert there -> byte-identical (reconciles Jupiter-OK/CoreWeave-salad).

        Bracket-open one weight sync with a SINGLE vLLM layerwise-reload init.

        Port of SkyRL #1685 + #1737 (the silent-MoE-corruption fix). vLLM's
        ``process_weights_after_loading`` permutes ``FusedMoE.w13_weight`` in place
        at engine init (``[w1;w3] -> [w3;w1]`` kernel layout) while our fork's
        ``replace_parameter`` preserves each param's ``weight_loader``. A naive
        SECOND ``model.load_weights`` (the RL sync) then re-invokes ``_load_w13`` and
        writes raw ``[w1;w3]`` into a ``[w3;w1]`` buffer -> silent value corruption ->
        token-salad. The cure is to run vLLM's layerwise reload ONCE around the whole
        (multi-chunk, streamed) sync: ``initialize_layerwise_reload`` restores layers
        to meta + wraps each weight_loader to DEFER processing; per-chunk raw
        ``model.load_weights`` then just buffers; ``finalize_layerwise_reload``
        (in ``finish_weight_update``) materializes + processes every layer EXACTLY
        ONCE. A per-chunk ``reload_weights`` would instead re-finalize on every call
        and restore layers absent from that chunk (#1737), so we bracket the whole
        sync rather than reload per chunk.

        Idempotent guard: a second ``start`` without a ``finish`` is a protocol error.
        """
        if getattr(self, "_skyrl_weight_update_active", False):
            raise RuntimeError(
                "start_weight_update called while a weight update is already active. Call finish_weight_update first."
            )
        # Register the RMSNorm Meta/fake kernels BEFORE the reload restores layers
        # to the meta device — the CP>1 finalize path traces _C::rms_norm on those
        # meta tensors and would otherwise crash (see ensure_norm_meta_fakes_registered).
        ensure_norm_meta_fakes_registered()
        from vllm.config import set_current_vllm_config
        from vllm.model_executor.model_loader.reload import initialize_layerwise_reload

        model = self.model_runner.model
        with set_current_vllm_config(self.vllm_config), torch.device(self.device):
            initialize_layerwise_reload(model)
        self._skyrl_weight_update_active = True

    def skyrl_finish_weight_reload(self) -> None:
        """RENAMED from ``finish_weight_update`` + NOW WIRED — see
        ``skyrl_begin_weight_reload`` for the collision + root-cause rationale.

        Bracket-close the weight sync with a SINGLE layerwise-reload finalize.

        Materializes + runs ``process_weights_after_loading`` over the WHOLE weight
        set exactly once -> re-applies the FlashInfer-CUTLASS ``swap_w13_to_w31`` the
        per-chunk ``model.load_weights`` skips. Must be called after every chunk's
        ``load_weights`` (and after ``end_weight_update``'s fused flush).
        """
        if not getattr(self, "_skyrl_weight_update_active", False):
            raise RuntimeError("skyrl_begin_weight_reload must be called before skyrl_finish_weight_reload.")
        # Idempotent no-op if begin already registered them; guards the case where
        # finalize is the first meta-materializing call in this worker.
        ensure_norm_meta_fakes_registered()
        from vllm.config import set_current_vllm_config
        from vllm.model_executor.model_loader.reload import finalize_layerwise_reload

        model = self.model_runner.model
        with set_current_vllm_config(self.vllm_config), torch.device(self.device):
            finalize_layerwise_reload(model, self.model_config)
        self._skyrl_weight_update_active = False
        speculative_config = self.vllm_config.speculative_config
        if speculative_config is not None and speculative_config.method == "eagle3":
            self.model_runner.refresh_online_eagle_target_owned_weights()

    def begin_weight_update(self) -> None:
        """Start accumulating weights for batched load_weights call.

        When fused loading is requested, weights are accumulated instead of loaded
        immediately. Call end_weight_update() to flush and apply them all at once
        via model.load_weights(), which handles packed module mapping (qkv_proj, gate_up_proj).
        Weights are stored on CPU to avoid GPU OOM during accumulation.
        """
        self._accumulated_weights = []

    def _is_fp8_model(self):
        """Check if the model uses FP8 quantization."""
        quant_config = getattr(self.model_runner.model, "quant_config", None)
        if quant_config is None:
            return False
        from vllm.model_executor.layers.quantization.fp8 import Fp8Config

        return isinstance(quant_config, Fp8Config)

    def _quantize_weights_for_fp8(self, weights):
        """Quantize BF16 weights to FP8 before loading into FP8 model.

        Follows verl's approach: quantize each weight tensor to FP8 with
        per-tensor scale, then yield (name, fp8_tensor) and (name_scale, scale).
        Non-linear weights (layernorm, embedding) are passed through as-is.
        """
        import torch
        from vllm._custom_ops import scaled_fp8_quant

        model = self.model_runner.model
        # Build set of parameter names that are FP8 quantized
        # These are the linear layer weights (not biases, not layernorms, not embeddings)
        fp8_param_names = set()
        for name, module in model.named_modules():
            from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod

            if hasattr(module, "quant_method") and isinstance(module.quant_method, Fp8LinearMethod):
                for pname, _ in module.named_parameters():
                    if "weight" in pname and "scale" not in pname:
                        full_name = f"{name}.{pname}" if name else pname
                        fp8_param_names.add(full_name)

        for name, tensor in weights:
            # Check if this weight maps to an FP8-quantized parameter
            # The name might be "layers.0.self_attn.q_proj.weight" but the
            # FP8 param is "layers.0.self_attn.qkv_proj.weight"
            # We need to check the ORIGINAL unfused name against the fused params
            is_fp8 = False
            packed_mapping = getattr(model, "packed_modules_mapping", {})
            # Reverse mapping: q_proj -> qkv_proj, gate_proj -> gate_up_proj
            reverse_map = {}
            for fused, originals in packed_mapping.items():
                for orig in originals:
                    reverse_map[orig] = fused

            # Try to find the FP8 param name
            check_name = name
            parts = name.rsplit(".", 2)
            if len(parts) >= 2:
                module_part = parts[-2]  # e.g. "q_proj"
                if module_part in reverse_map:
                    check_name = name.replace(module_part, reverse_map[module_part])

            if check_name in fp8_param_names or name in fp8_param_names:
                is_fp8 = True

            if is_fp8 and tensor.dtype != torch.float8_e4m3fn:
                # Move to GPU, quantize, move back to CPU
                gpu_tensor = tensor.to(device="cuda", dtype=torch.bfloat16)
                fp8_tensor, scale = scaled_fp8_quant(gpu_tensor)
                yield (name, fp8_tensor.cpu())
                # Yield the scale with the FUSED param name
                scale_name = check_name.replace(".weight", ".weight_scale")
                yield (scale_name, scale.cpu())
                del gpu_tensor, fp8_tensor
            else:
                yield (name, tensor)

    def _restore_param_subclasses(self, model):
        """Temporarily restore param __class__ to subclass_type for weight loading.

        After process_weights_after_loading, params are plain Parameter but have
        subclass_type saved. Restoring __class__ makes weight_loader dispatch work.
        Returns list of (param, original_class) for cleanup.
        """
        patched = []
        for name, param in model.named_parameters():
            subclass_type = getattr(param, "subclass_type", None)
            if subclass_type is not None and type(param) is not subclass_type:
                original_class = type(param)
                param.__class__ = subclass_type
                patched.append((param, original_class))
        return patched

    def _undo_param_subclasses(self, patched):
        """Undo the temporary __class__ patching."""
        for param, original_class in patched:
            param.__class__ = original_class

    def end_weight_update(self) -> None:
        """Flush accumulated weights via model.load_weights().

        For FP8 models: quantizes BF16 weights to FP8 before loading,
        following verl's approach. Also temporarily restores param subclass
        types so weight_loader dispatch works correctly with FP8 params.
        """
        import gc

        if hasattr(self, "_accumulated_weights") and self._accumulated_weights:
            model = self.model_runner.model
            if self._is_fp8_model():
                import torch
                import gc
                from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
                from vllm._custom_ops import scaled_fp8_quant

                # Receiver-side FP8 quantization: BF16 weights arrive via NCCL,
                # fuse stacked params, quantize to FP8, write directly to model.
                weight_index = {name: tensor for name, tensor in self._accumulated_weights}
                stacked = [
                    ("qkv_proj", "q_proj", "q"),
                    ("qkv_proj", "k_proj", "k"),
                    ("qkv_proj", "v_proj", "v"),
                    ("gate_up_proj", "gate_proj", 0),
                    ("gate_up_proj", "up_proj", 1),
                ]

                for mname, module in model.named_modules():
                    if not (hasattr(module, "quant_method") and isinstance(module.quant_method, Fp8LinearMethod)):
                        continue
                    param = module.weight
                    device = param.device
                    is_stacked = any(mname.endswith(pn) for pn, _, _ in stacked)

                    if is_stacked:
                        shard_list = []
                        for param_name, weight_name, shard_id in stacked:
                            if not mname.endswith(param_name):
                                continue
                            src_name = mname.replace(param_name, weight_name) + ".weight"
                            if src_name in weight_index:
                                shard_list.append(weight_index[src_name])
                        if shard_list:
                            full_bf16 = torch.cat(shard_list, dim=0).to(
                                device=device, dtype=torch.bfloat16, non_blocking=True
                            )
                            torch.cuda.current_stream().synchronize()
                            fp8_full, scale = scaled_fp8_quant(full_bf16)
                            param.data.copy_(fp8_full)
                            if hasattr(module, "weight_scale"):
                                module.weight_scale.data.copy_(scale.squeeze())
                            del full_bf16, fp8_full, scale, shard_list
                    else:
                        src_name = mname + ".weight"
                        if src_name in weight_index:
                            bf16_w = weight_index[src_name].to(device=device, dtype=torch.bfloat16, non_blocking=True)
                            torch.cuda.current_stream().synchronize()
                            fp8_w, scale = scaled_fp8_quant(bf16_w)
                            param.data.copy_(fp8_w)
                            if hasattr(module, "weight_scale"):
                                module.weight_scale.data.copy_(scale.squeeze())
                            del bf16_w, fp8_w, scale

                # Load non-FP8 params (layernorms, embeddings)
                params_dict = dict(model.named_parameters())
                for name, tensor in self._accumulated_weights:
                    if name in params_dict:
                        param = params_dict[name]
                        if param.dtype != torch.float8_e4m3fn:
                            param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))

                del weight_index

                gc.collect()
                torch.cuda.empty_cache()
            else:
                load_weights_into_vllm(model, self._accumulated_weights)
            self._accumulated_weights.clear()
            del self._accumulated_weights
            gc.collect()
            import torch

            torch.cuda.empty_cache()

    def load_weights(self, request: NamedWeightsUpdateRequest) -> None:
        """Load weights using the receiver.

        This method is called via collective_rpc from VLLMWeightLoader.

        When fused loading is requested and begin_weight_update() was called,
        weights are accumulated on CPU instead of loaded immediately.

        Args:
            request: Weight update request with names, dtypes, shapes, etc.
        """
        weight_list = []
        for name, tensor in self._weight_receiver.receive_weights(request):
            weight_list.append((name, tensor))

        if hasattr(self, "_accumulated_weights"):
            # Batched mode: move to CPU and accumulate for later flush
            for name, tensor in weight_list:
                self._accumulated_weights.append((name, tensor.cpu()))
            del weight_list
        else:
            # Immediate mode (default): load right away
            load_weights_into_vllm(self.model_runner.model, weight_list)
            for weight in weight_list:
                del weight

    # TODO (sumanthrh): Add destroy process group RPC as a atexit handler to Trainer code.
    def destroy_weights_update_group(self):
        if not getattr(self, "_model_update_group", None):
            warnings.warn("No model update group to destroy")
            return
        destroy_process_group(self._model_update_group)

    def read_named_weights(self, hf_names, dump_inventory: bool = False):
        """TEST-ONLY (Stage 6 weight-equality gate): read engine-side weights back
        from the live vLLM model, reconstructed under the HF parameter names the
        trainer broadcasts.

        This is the symmetric inverse of ``load_weights`` (vLLM consumes HF-named
        tensors in ``model.load_weights`` and maps them into its internal
        fused/sharded params; here we read those internal params back and rebuild
        the HF view so the trainer's post-step HF tensors can be compared
        tensor-by-tensor). Returns, per requested HF name, this worker's
        contribution as a CPU fp32 tensor plus the live engine dtype and rank
        coordinates so the caller can assemble across TP/EP shards.

        Supported HF name forms (Qwen1.5-MoE / Qwen2MoE vLLM layout):
          * ``model.embed_tokens.weight``                       -> VocabParallelEmbedding (TP vocab-sharded)
          * ``model.layers.{i}.mlp.gate.weight`` (router)       -> ReplicatedLinear (full copy every rank)
          * ``model.layers.{i}.self_attn.o_proj.weight``        -> RowParallelLinear (TP input-sharded)
          * ``model.layers.{i}.mlp.experts.{j}.gate_proj.weight`` -> RoutedExperts w13_weight[local_e, :I]
          * ``...experts.{j}.up_proj.weight``                   -> RoutedExperts w13_weight[local_e, I:]
          * ``...experts.{j}.down_proj.weight``                 -> RoutedExperts w2_weight[local_e]

        Args:
            hf_names: list of HF parameter names to read back.
            dump_inventory: if True, also returns the full ``named_parameters()``
                name->shape inventory under key ``__inventory__`` (first run aid).
        """
        import re
        import torch as _torch

        model = self.model_runner.model
        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        all_params = {**params, **buffers}

        try:
            from vllm.distributed import parallel_state as _ps

            tp_rank = _ps.get_tensor_model_parallel_rank()
            tp_size = _ps.get_tensor_model_parallel_world_size()
        except Exception:
            tp_rank, tp_size = 0, 1
        try:
            ep_rank = _ps.get_ep_group().rank_in_group
            ep_size = _ps.get_ep_group().world_size
        except Exception:
            ep_rank, ep_size = 0, 1
        try:
            pp_rank, pp_size = _ps.get_pp_group().rank_in_group, _ps.get_pp_group().world_size
        except Exception:
            pp_rank, pp_size = 0, 1

        def _cpu(t):
            return t.detach().to("cpu", dtype=_torch.float32).contiguous()

        out = {}
        if dump_inventory:
            out["__inventory__"] = {n: list(p.shape) for n, p in all_params.items()}
        out["__ranks__"] = {
            "tp_rank": tp_rank,
            "tp_size": tp_size,
            "ep_rank": ep_rank,
            "ep_size": ep_size,
            "pp_rank": pp_rank,
            "pp_size": pp_size,
        }

        expert_re = re.compile(r"^(model\.layers\.\d+\.mlp)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")

        for name in hf_names:
            entry = {"found": False}
            try:
                # 1. Direct (replicated) match: router gate, norms, etc.
                if name in all_params:
                    tensor = all_params[name]
                    entry = {
                        "found": True,
                        "mode": "direct",
                        "dtype": torch_dtype_to_str(tensor.dtype),
                        "tensor": _cpu(tensor),
                    }
                    out[name] = entry
                    continue

                # 2. Routed expert -> FusedMoE fused weights.
                m = expert_re.match(name)
                if m is not None:
                    prefix, gj, proj = m.group(1), int(m.group(2)), m.group(3)
                    # vLLM RoutedExperts stores w13_weight [n_local_experts, 2*I, H] and
                    # w2_weight [n_local_experts, H, I]. Local experts are a contiguous
                    # EP slice: global expert gj lives on ep_rank == gj // n_local.
                    w13 = all_params.get(f"{prefix}.experts.routed_experts.w13_weight")
                    w2 = all_params.get(f"{prefix}.experts.routed_experts.w2_weight")
                    if w13 is None or w2 is None:
                        # Fallback: scan for any experts.*weight tensor under this prefix.
                        cand = {
                            k: v
                            for k, v in all_params.items()
                            if k.startswith(f"{prefix}.experts.") and k.endswith("weight")
                        }
                        entry = {"found": False, "note": f"no w13/w2; candidates={list(cand.keys())}"}
                        out[name] = entry
                        continue
                    n_local = w13.shape[0]
                    owner_ep = gj // n_local
                    if owner_ep != ep_rank:
                        entry = {"found": False, "mode": "expert", "owner_ep": owner_ep, "skip": True}
                        out[name] = entry
                        continue
                    local_e = gj - owner_ep * n_local
                    if proj == "down_proj":
                        t = w2[local_e]
                    else:
                        inter = w13.shape[1] // 2
                        t = w13[local_e, :inter] if proj == "gate_proj" else w13[local_e, inter:]
                    entry = {
                        "found": True,
                        "mode": "expert",
                        "owner_ep": owner_ep,
                        "local_e": local_e,
                        "dtype": torch_dtype_to_str(t.dtype),
                        "tensor": _cpu(t),
                    }
                    out[name] = entry
                    continue

                # 3. Unknown / unsupported name.
                entry = {"found": False, "note": "no mapping"}
                out[name] = entry
            except Exception as e:  # never crash the collective_rpc
                out[name] = {"found": False, "error": repr(e)}
        return out

    def read_expert_slots_raw(self, layer_idx: int):
        """TEST-ONLY (D1/D2 disaggregated-receive diag): return THIS engine worker's
        RAW per-local-slot FusedMoE expert weights + the engine's OWN expert_map for
        ``layer_idx``, with NO assumption about global<->local placement.

        Unlike ``read_named_weights`` (which maps a requested HF expert ``gj`` to a slot
        via ``gj // n_local`` — a CONTIGUOUS-linear assumption that would HIDE a
        receive-side placement bug), this dumps every local slot's bytes AS-IS plus the
        engine's authoritative ``_expert_map`` (global->local, -1 if absent). The driver
        then does a NON-circular cross-expert nearest-match vs the DISK base checkpoint,
        so a slot carrying the wrong global expert (D2 placement) surfaces as m!=j even
        though a per-name readback could not see it.

        Returns dict:
          __ranks__        : {tp_rank, tp_size, ep_rank, ep_size}
          expert_map       : list[int] length global_num_experts (global->local slot, -1 absent)
          slot_to_global   : list[int] length local_num_experts (inverse; -1 if ambiguous)
          local_num_experts, global_num_experts, w13_inter_half (I), placement_strategy
          slots            : {local_slot -> {"w13": cpu_fp32 [2I,H], "w2": cpu_fp32 [H,I]}}
        Heavy (full local expert stack as fp32 CPU); call for ONE layer.
        """
        import torch as _torch

        model = self.model_runner.model
        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        all_params = {**params, **buffers}

        try:
            from vllm.distributed import parallel_state as _ps

            tp_rank = _ps.get_tensor_model_parallel_rank()
            tp_size = _ps.get_tensor_model_parallel_world_size()
        except Exception:
            tp_rank, tp_size = 0, 1
        try:
            ep_rank = _ps.get_ep_group().rank_in_group
            ep_size = _ps.get_ep_group().world_size
        except Exception:
            ep_rank, ep_size = 0, 1

        out = {"__ranks__": {"tp_rank": tp_rank, "tp_size": tp_size, "ep_rank": ep_rank, "ep_size": ep_size}}
        prefix = f"model.layers.{layer_idx}.mlp"
        w13 = all_params.get(f"{prefix}.experts.routed_experts.w13_weight")
        w2 = all_params.get(f"{prefix}.experts.routed_experts.w2_weight")
        if w13 is None or w2 is None:
            cand = [k for k in all_params if k.startswith(f"{prefix}.experts.") and k.endswith("weight")]
            out["error"] = f"no w13/w2 under {prefix}; candidates={cand}"
            return out

        # Find the FusedMoE module to read its authoritative expert_map + counts.
        emap = None
        local_num = int(w13.shape[0])
        global_num = None
        placement = None
        for mod_name, mod in model.named_modules():
            routed_experts_name = f"{prefix}.experts.routed_experts"
            if mod_name == routed_experts_name or mod_name.endswith(routed_experts_name):
                emap = getattr(mod, "_expert_map", None)
                if emap is None:
                    emap = getattr(mod, "expert_map", None)
                local_num = int(getattr(mod, "local_num_experts", local_num))
                global_num = getattr(mod, "global_num_experts", None)
                placement = getattr(mod, "expert_placement_strategy", None)
                break

        # global->local slot map (the engine's OWN authority on placement).
        if emap is not None:
            emap_list = [int(x) for x in emap.detach().cpu().tolist()]
        else:
            emap_list = None
        out["expert_map"] = emap_list
        out["local_num_experts"] = local_num
        out["global_num_experts"] = int(global_num) if global_num is not None else None
        out["placement_strategy"] = str(placement) if placement is not None else None
        out["w13_inter_half"] = int(w13.shape[1] // 2)

        # Inverse: local slot -> global expert (per the engine's expert_map).
        slot_to_global = [-1] * local_num
        if emap_list is not None:
            for g, loc in enumerate(emap_list):
                if 0 <= loc < local_num:
                    slot_to_global[loc] = g
        out["slot_to_global"] = slot_to_global

        slots = {}
        for s in range(local_num):
            slots[s] = {
                "w13": w13[s].detach().to("cpu", dtype=_torch.float32).contiguous(),
                "w2": w2[s].detach().to("cpu", dtype=_torch.float32).contiguous(),
            }
        out["slots"] = slots
        return out

    def report_host(self):
        """TEST-ONLY (disaggregation proof): this engine worker's hostname (one per
        TP/EP worker via collective_rpc) so the driver can prove the engine node is
        DISJOINT from the policy nodes (=> the broadcast is genuinely cross-node)."""
        import socket

        return socket.gethostname()

    def report_device_placement(self) -> dict[str, str | int]:
        """This worker's host, GPU and ranks, as a plain dict."""
        return asdict(self._device_placement())

    def _device_placement(self) -> InferenceWorkerPlacement:
        dp, pp = get_dp_group(), get_pp_group()
        ep = get_ep_group() if self.model_config.is_moe else None
        return inference_worker_placement(
            dp_rank=dp.rank_in_group,
            dp_world_size=dp.world_size,
            ep_rank=ep.rank_in_group if ep is not None else 0,
            ep_world_size=ep.world_size if ep is not None else 1,
            pp_rank=pp.rank_in_group,
            pp_world_size=pp.world_size,
        )

    def expert_block_rpc(self, method: str, *args):
        """Call a method of this worker's expert-block receiver, creating the receiver on first use."""
        receiver = getattr(self, "_expert_block_receiver", None)
        if receiver is None:
            placement = self._device_placement()
            receiver = self._expert_block_receiver = ExpertBlockReceiver(
                self.vllm_config,
                self.device,
                self.model_runner.model,
                ep_rank=placement.ep_rank,
                ep_size=placement.ep_world_size,
                pp_rank=placement.pp_rank,
                pp_size=placement.pp_world_size,
                gpu_uuid=placement.gpu_uuid,
            )
        return getattr(receiver, method)(*args)


class V1LoggingStatLoggerFixed(LoggingStatLogger):
    """
    A fixed version of LoggingStatLogger that actually logs during the record method.
    The log method is otherwise not called in the VLLM codebase.

    Also stores aggregated stats in a class-level registry for programmatic access
    (used by InferenceStatsCallback to bypass Ray log-to-driver unreliability).

    Stats are accumulated throughout a step:
    - Request counts (running, waiting): track peak and median values
    - Throughput metrics: track peak and median values observed during active periods
    - Cache metrics: track peak and median usage
    """

    # Class-level registry mapping engine IDs to their accumulated stats
    _stats_registry: Dict[str, Dict[str, Any]] = {}
    _registry_lock = threading.Lock()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.log_interval = 5
        self._engine_id: Optional[str] = None
        config = args[0] if args else kwargs["vllm_config"]
        engine_index = args[1] if len(args) > 1 else kwargs.get("engine_index", 0)
        self._attributes = {
            "model_name": str(config.model_config.served_model_name),
            "engine_index": str(engine_index),
        }

    def set_engine_id(self, engine_id: str) -> None:
        """Set the engine ID for this stat logger instance."""
        self._engine_id = engine_id

    def record(self, *args: Any, **kwargs: Any) -> None:
        # Call parent with original arguments - important to preserve vLLM's calling convention
        super().record(*args, **kwargs)

        # Accumulate stats in registry if engine ID is set
        if self._engine_id is not None:
            # Extract scheduler_stats from vLLM v1 API:
            # vLLM calls record(scheduler_stats, iteration_stats, ...) with positional args
            # or record(scheduler_stats=..., iteration_stats=...) with keyword args
            scheduler_stats = None
            if args:
                scheduler_stats = args[0]
            elif "scheduler_stats" in kwargs:
                scheduler_stats = kwargs["scheduler_stats"]

            current_running = 0
            current_waiting = 0
            current_cache_usage = 0.0
            prefix_cache_stats = None

            if scheduler_stats is not None:
                current_running = getattr(scheduler_stats, "num_running_reqs", 0)
                current_waiting = getattr(scheduler_stats, "num_waiting_reqs", 0) + getattr(
                    scheduler_stats, "num_skipped_waiting_reqs", 0
                )
                current_cache_usage = getattr(scheduler_stats, "kv_cache_usage", 0.0) * 100.0  # Convert to percentage
                prefix_cache_stats = scheduler_stats.prefix_cache_stats

            # Extract iteration_stats (second positional arg) for per-request latency data
            iteration_stats = None
            if len(args) > 1:
                iteration_stats = args[1]
            elif "iteration_stats" in kwargs:
                iteration_stats = kwargs["iteration_stats"]

            # Collect per-request latency samples from finished requests
            finished_prefill_times: List[float] = []
            finished_decode_times: List[float] = []
            finished_e2e_latencies: List[float] = []
            finished_queued_times: List[float] = []
            finished_ttfts: List[float] = []
            finished_num_preempted = 0
            if iteration_stats is not None:
                # Time-to-first-token samples from this iteration
                ttft_iter = getattr(iteration_stats, "time_to_first_tokens_iter", None)
                if ttft_iter:
                    finished_ttfts.extend(ttft_iter)
                # Preemption count
                finished_num_preempted = getattr(iteration_stats, "num_preempted_reqs", 0)
                # Per-request stats from completed requests
                for req in getattr(iteration_stats, "finished_requests", []):
                    prefill_t = getattr(req, "prefill_time", 0.0)
                    decode_t = getattr(req, "decode_time", 0.0)
                    e2e_t = getattr(req, "e2e_latency", 0.0)
                    queued_t = getattr(req, "queued_time", 0.0)
                    if prefill_t > 0:
                        finished_prefill_times.append(prefill_t)
                    if decode_t > 0:
                        finished_decode_times.append(decode_t)
                    if e2e_t > 0:
                        finished_e2e_latencies.append(e2e_t)
                    if queued_t > 0:
                        finished_queued_times.append(queued_t)

            # Throughput is computed by parent class LoggingStatLogger after super().record()
            # These are stored as instance attributes
            current_prompt_tp = getattr(self, "last_prompt_throughput", 0.0) or 0.0
            current_gen_tp = getattr(self, "last_generation_throughput", 0.0) or 0.0

            is_active = current_running > 0 or current_waiting > 0

            with V1LoggingStatLoggerFixed._registry_lock:
                existing = V1LoggingStatLoggerFixed._stats_registry.get(self._engine_id)

                if existing is None:
                    # Initialize with sample lists for median calculation
                    existing = {
                        # Sample lists for computing median (only active samples)
                        "_samples_prompt_tp": [current_prompt_tp] if is_active else [],
                        "_samples_gen_tp": [current_gen_tp] if is_active else [],
                        "_samples_running": [current_running] if is_active else [],
                        "_samples_waiting": [current_waiting] if is_active else [],
                        "_samples_cache": [current_cache_usage] if is_active else [],
                        "_prefix_hit": PrefixCacheHitRateAccumulator(),
                        # Per-request latency samples (accumulated from finished requests)
                        "_samples_prefill_time": list(finished_prefill_times),
                        "_samples_decode_time": list(finished_decode_times),
                        "_samples_e2e_latency": list(finished_e2e_latencies),
                        "_samples_queued_time": list(finished_queued_times),
                        "_samples_ttft": list(finished_ttfts),
                        "_total_preempted": finished_num_preempted,
                        "_attributes": self._attributes,
                        # Peak values
                        "_peak_prompt_tp": current_prompt_tp,
                        "_peak_gen_tp": current_gen_tp,
                        "_peak_running": current_running,
                        "_peak_waiting": current_waiting,
                        "_peak_cache": current_cache_usage,
                        # Counters
                        "_num_samples": 1,
                        "_num_active_samples": 1 if is_active else 0,
                        "timestamp": time.time(),
                    }
                    V1LoggingStatLoggerFixed._stats_registry[self._engine_id] = existing
                else:
                    # Update peak values
                    existing["_peak_prompt_tp"] = max(existing["_peak_prompt_tp"], current_prompt_tp)
                    existing["_peak_gen_tp"] = max(existing["_peak_gen_tp"], current_gen_tp)
                    existing["_peak_running"] = max(existing["_peak_running"], current_running)
                    existing["_peak_waiting"] = max(existing["_peak_waiting"], current_waiting)
                    existing["_peak_cache"] = max(existing["_peak_cache"], current_cache_usage)

                    # Accumulate per-request latency samples
                    existing["_samples_prefill_time"].extend(finished_prefill_times)
                    existing["_samples_decode_time"].extend(finished_decode_times)
                    existing["_samples_e2e_latency"].extend(finished_e2e_latencies)
                    existing["_samples_queued_time"].extend(finished_queued_times)
                    existing["_samples_ttft"].extend(finished_ttfts)
                    existing["_total_preempted"] += finished_num_preempted

                    # Append to sample lists (only for active samples to get meaningful medians)
                    if is_active:
                        existing["_samples_prompt_tp"].append(current_prompt_tp)
                        existing["_samples_gen_tp"].append(current_gen_tp)
                        existing["_samples_running"].append(current_running)
                        existing["_samples_waiting"].append(current_waiting)
                        existing["_samples_cache"].append(current_cache_usage)
                        existing["_num_active_samples"] += 1

                    existing["_num_samples"] += 1
                    existing["timestamp"] = time.time()

                existing["_prefix_hit"].observe(prefix_cache_stats, is_active=is_active)

        now = time.monotonic()
        if now - self.last_log_time > self.log_interval:
            self.log()
            self.last_log_time = now

    @staticmethod
    def _compute_median(samples: List[float]) -> float:
        """Compute median of a list of samples."""
        if not samples:
            return 0.0
        sorted_samples = sorted(samples)
        n = len(sorted_samples)
        mid = n // 2
        if n % 2 == 0:
            return (sorted_samples[mid - 1] + sorted_samples[mid]) / 2.0
        return sorted_samples[mid]

    @classmethod
    def get_stats_by_engine_id(cls, engine_id: str, reset: bool = True) -> Optional[VLLMEngineStatsSnapshot]:
        """Return the engine's typed metric snapshot, optionally resetting interval fields."""
        with cls._registry_lock:
            stats = cls._stats_registry.get(engine_id)
            if stats is None:
                return None

            # Compute medians from sample lists
            median_prompt_tp = cls._compute_median(stats["_samples_prompt_tp"])
            median_gen_tp = cls._compute_median(stats["_samples_gen_tp"])
            median_running = cls._compute_median(stats["_samples_running"])
            median_waiting = cls._compute_median(stats["_samples_waiting"])
            median_cache = cls._compute_median(stats["_samples_cache"])
            median_prefix_hit = cls._compute_median(stats["_prefix_hit"].samples)

            # Compute means from sample lists
            num_active = stats["_num_active_samples"]
            if num_active > 0:
                mean_prompt_tp = sum(stats["_samples_prompt_tp"]) / num_active
                mean_gen_tp = sum(stats["_samples_gen_tp"]) / num_active
            else:
                mean_prompt_tp = 0.0
                mean_gen_tp = 0.0

            # Compute per-request latency statistics
            prefill_samples = stats["_samples_prefill_time"]
            decode_samples = stats["_samples_decode_time"]
            e2e_samples = stats["_samples_e2e_latency"]
            queued_samples = stats["_samples_queued_time"]
            ttft_samples = stats["_samples_ttft"]

            def _mean(s: List[float]) -> float:
                return sum(s) / len(s) if s else 0.0

            def _p90(s: List[float]) -> float:
                if not s:
                    return 0.0
                sorted_s = sorted(s)
                idx = int(len(sorted_s) * 0.9)
                return sorted_s[min(idx, len(sorted_s) - 1)]

            result = VLLMEngineStatsSnapshot(
                engine_id=str(engine_id),
                timestamp=float(stats["timestamp"]),
                current=VLLMCurrentStats(),
                cumulative=VLLMCumulativeStats(),
                interval=VLLMIntervalStats(
                    peak_prompt_throughput=stats["_peak_prompt_tp"],
                    peak_generation_throughput=stats["_peak_gen_tp"],
                    peak_running_reqs=stats["_peak_running"],
                    peak_waiting_reqs=stats["_peak_waiting"],
                    peak_gpu_cache_usage_perc=stats["_peak_cache"],
                    peak_prefix_cache_hit_rate=stats["_prefix_hit"].peak,
                    median_prompt_throughput=median_prompt_tp,
                    median_generation_throughput=median_gen_tp,
                    median_running_reqs=median_running,
                    median_waiting_reqs=median_waiting,
                    median_gpu_cache_usage_perc=median_cache,
                    median_prefix_cache_hit_rate=median_prefix_hit,
                    mean_prompt_throughput=mean_prompt_tp,
                    mean_generation_throughput=mean_gen_tp,
                    latency_prefill_mean=_mean(prefill_samples),
                    latency_prefill_median=cls._compute_median(prefill_samples),
                    latency_prefill_p90=_p90(prefill_samples),
                    latency_decode_mean=_mean(decode_samples),
                    latency_decode_median=cls._compute_median(decode_samples),
                    latency_decode_p90=_p90(decode_samples),
                    latency_e2e_mean=_mean(e2e_samples),
                    latency_e2e_median=cls._compute_median(e2e_samples),
                    latency_e2e_p90=_p90(e2e_samples),
                    latency_queued_mean=_mean(queued_samples),
                    latency_queued_median=cls._compute_median(queued_samples),
                    latency_queued_p90=_p90(queued_samples),
                    latency_ttft_mean=_mean(ttft_samples),
                    latency_ttft_median=cls._compute_median(ttft_samples),
                    latency_ttft_p90=_p90(ttft_samples),
                    finished_requests=len(e2e_samples),
                    preempted_reqs=stats["_total_preempted"],
                    samples=stats["_num_samples"],
                    active_samples=stats["_num_active_samples"],
                ),
                attributes=stats["_attributes"],
            )

            if reset:
                for key in (
                    "_samples_prompt_tp",
                    "_samples_gen_tp",
                    "_samples_running",
                    "_samples_waiting",
                    "_samples_cache",
                    "_samples_prefill_time",
                    "_samples_decode_time",
                    "_samples_e2e_latency",
                    "_samples_queued_time",
                    "_samples_ttft",
                ):
                    stats[key].clear()
                stats["_prefix_hit"] = PrefixCacheHitRateAccumulator()
                for key in ("_peak_prompt_tp", "_peak_gen_tp", "_peak_running", "_peak_waiting", "_peak_cache"):
                    stats[key] = 0
                stats["_total_preempted"] = 0
                stats["_num_samples"] = 0
                stats["_num_active_samples"] = 0

            return result


class AsyncVLLMInferenceEngine(InferenceEngineInterface):
    """A vLLM ``AsyncLLMEngine`` serving generation, OpenAI-style requests, and weight updates."""

    def __init__(self, *args, bundle_indices: list = None, **kwargs):
        # _create_engine installs stat loggers that report under this engine ID.
        self._stats_engine_id = uuid4().hex
        self._stats_attributes: Dict[str, str] = {}
        self._rendezvous_port_reservation = kwargs.pop("rendezvous_port_reservation", None)
        self._weight_sync_pause_policy: WeightSyncPausePolicy = kwargs.pop("weight_sync_pause_policy")
        setup_envvars_for_vllm(kwargs, bundle_indices)
        vllm_v1_disable_multiproc = kwargs.pop("vllm_v1_disable_multiproc", False)
        logger.info(
            f"AsyncVLLMInferenceEngine: vllm_v1_disable_multiproc={vllm_v1_disable_multiproc}, "
            f"vllm.__version__={vllm.__version__}, "
            f"VLLM_ENABLE_V1_MULTIPROCESSING={os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING', '<unset>')}"
        )
        if vllm_v1_disable_multiproc or vllm.__version__ == "0.8.2":
            # https://github.com/vllm-project/vllm/blob/effc5d24fae10b29996256eb7a88668ff7941aed/examples/offline_inference/reproduciblity.py#L11
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
            logger.info("AsyncVLLMInferenceEngine: set VLLM_ENABLE_V1_MULTIPROCESSING=0")

        self._tp_size = kwargs.get("tensor_parallel_size", 1)
        self._pp_size = kwargs.get("pipeline_parallel_size", 1)
        self._dp_size = kwargs.get("data_parallel_size", 1)
        self._is_lora = kwargs.get("enable_lora", False)

        if "rope_scaling" in kwargs:
            kwargs.pop("rope_scaling")
        self._id_only_logprob_requests: set[str] = set()
        self.llm = self._create_engine(*args, **kwargs)
        self._weight_loader = VLLMWeightLoader(self.llm)
        self._policy_steps = PolicyStepStamps()
        self.openai_serving_chat.id_only_logprob_requests = self._id_only_logprob_requests
        self._stamp_engine_steps()

    def _stamp_engine_steps(self) -> None:
        """Stamp every request's new tokens with the policy step installed before the EngineCore's step timestamp."""
        output_processor = self.llm.output_processor
        process_outputs = output_processor.process_outputs

        def stamped_process_outputs(engine_core_outputs, engine_core_timestamp, iteration_stats):
            for output in engine_core_outputs:
                # vLLM keys its state by an internal id and drops outputs of requests it already aborted.
                state = output_processor.request_states.get(output.request_id)
                if state is not None:
                    if state.external_req_id in self._id_only_logprob_requests:
                        # ID-only captures expose IDs and numeric scores only. The
                        # separate response detokenizer still produces the text.
                        state.logprobs_processor.tokenizer = None
                    self._policy_steps.observe(state.external_req_id, len(output.new_token_ids), engine_core_timestamp)
            return process_outputs(engine_core_outputs, engine_core_timestamp, iteration_stats)

        output_processor.process_outputs = stamped_process_outputs

    def _forget_unclaimed_policy_steps(self) -> None:
        live = {state.external_req_id for state in self.llm.output_processor.request_states.values()}
        self._policy_steps.forget_unclaimed(live)

    def tp_size(self):
        return self._tp_size

    def pp_size(self):
        return self._pp_size

    def dp_size(self):
        return self._dp_size

    def _release_rendezvous_port_reservation(self) -> None:
        if self._rendezvous_port_reservation is None:
            return
        ray.get(self._rendezvous_port_reservation.release.remote())
        self._rendezvous_port_reservation = None

    def _preprocess_prompts(self, input_batch: InferenceEngineInput):
        prompts = input_batch.get("prompts")
        prompt_token_ids = input_batch.get("prompt_token_ids")
        request_sampling_params = input_batch.get("sampling_params")
        per_prompt_sampling_params = input_batch.get("sampling_params_per_prompt")
        if self._validate_rollout_logprob_sampling and request_sampling_params is not None:
            if request_sampling_params.get("logprobs") is not None:
                validate_behavior_logprob_sampling(request_sampling_params)

        assert prompts is None and prompt_token_ids is not None, (
            "The vLLM engine only accepts `prompt_token_ids`, not `prompts`."
        )

        base_params = dict(request_sampling_params or {})
        if base_params.get("logprobs", 0) and base_params["logprobs"] > 0:
            base_params.setdefault("flat_logprobs", True)
        if per_prompt_sampling_params is None:
            sampling_params = SamplingParams(**base_params)
        else:
            if len(per_prompt_sampling_params) != len(prompt_token_ids):
                raise ValueError("per-prompt sampling parameters must align with prompt token rows")
            allowed = ({"prompt_logprob_token_ids"}, {"logprob_token_ids"})
            if any(set(override) not in allowed for override in per_prompt_sampling_params):
                raise ValueError("per-prompt sampling parameters require one supported token-ID override")
            sampling_params = [SamplingParams(**{**base_params, **override}) for override in per_prompt_sampling_params]

        return prompt_token_ids, sampling_params

    @staticmethod
    def _response_top_k(sampling_params: SamplingParams | list[SamplingParams]) -> int | None:
        first = sampling_params[0] if isinstance(sampling_params, list) else sampling_params
        return first.logprobs

    def _postprocess_outputs(
        self,
        outputs,
        response_top_k: int | None = None,
        sampling_params: SamplingParams | list[SamplingParams] | None = None,
    ):
        responses: List[str] = []
        stop_reasons: List[str] = []
        response_ids: List[List[int]] = []
        response_logprobs: Optional[List[List[float]]] = []
        student_topk_indices: List[np.ndarray | List[List[int]]] = []
        behavior_topk_logprobs: List[np.ndarray | List[List[float]]] = []
        routed_experts_rows: list[np.ndarray | None] = []
        all_prompt_logprobs: Optional[List] = None
        params_by_prompt = sampling_params if isinstance(sampling_params, list) else [sampling_params] * len(outputs)
        if len(params_by_prompt) != len(outputs):
            raise ValueError("vLLM sampling parameters do not align with output rows")
        for output, row_params in zip(outputs, params_by_prompt, strict=True):
            # TODO(tgriggs): Support n>1 sampling.
            assert len(output.outputs) == 1, (
                "Each prompt should have only one responses. n>1 sampling is supported by copying prompts."
            )
            resp = output.outputs[0]
            responses.append(resp.text)
            stop_reasons.append(resp.finish_reason)
            response_ids.append(resp.token_ids)
            routed_experts_rows.append(response_routes(resp.routed_experts, len(resp.token_ids)))
            _logprobs = None
            selected_ids = []
            selected_scores = []
            if isinstance(resp.logprobs, FlatLogprobs) and response_top_k is not None and response_top_k > 0:
                _logprobs, selected_ids, selected_scores = select_flat_response_topk(
                    resp.logprobs,
                    resp.token_ids,
                    response_top_k,
                    row_params.logprob_token_ids if row_params is not None else None,
                )
            elif resp.logprobs:
                _logprobs = []
                for i, token_logprobs in enumerate(resp.logprobs):
                    token_logprobs: Dict[str, Logprob]
                    token_id = resp.token_ids[i]
                    logprob = token_logprobs[token_id].logprob
                    _logprobs.append(logprob)
                    if response_top_k is not None and response_top_k > 0:
                        if row_params is not None and row_params.logprob_token_ids:
                            ids = list(row_params.logprob_token_ids)
                            scores = [token_logprobs[candidate].logprob for candidate in ids]
                        else:
                            ids, scores = select_response_topk(
                                {token_id: value.logprob for token_id, value in token_logprobs.items()}, response_top_k
                            )
                        selected_ids.append(ids)
                        selected_scores.append(scores)
                    del token_logprobs
            response_logprobs.append(_logprobs)
            if response_top_k is not None and response_top_k > 0:
                if len(selected_ids) != len(resp.token_ids):
                    raise ValueError("vLLM omitted response top-K logprobs for generated tokens")
                student_topk_indices.append(selected_ids)
                behavior_topk_logprobs.append(selected_scores)

            # Extract prompt_logprobs if available (used for teacher scoring)
            if hasattr(output, "prompt_logprobs") and output.prompt_logprobs is not None:
                if all_prompt_logprobs is None:
                    all_prompt_logprobs = []
                # Convert vLLM's List[Optional[Dict[int, Logprob]]] to
                # List[Optional[Dict[int, float]]] (extract .logprob from Logprob objects)
                prompt_lps = []
                for pos_logprobs in output.prompt_logprobs:
                    if pos_logprobs is None:
                        prompt_lps.append(None)
                    else:
                        prompt_lps.append(
                            {
                                token_id: lp.logprob if hasattr(lp, "logprob") else lp
                                for token_id, lp in pos_logprobs.items()
                            }
                        )
                all_prompt_logprobs.append(prompt_lps)

        if len(response_logprobs) and response_logprobs[0] is None:
            response_logprobs = None  # hack: assume uniform sampling params

        result = InferenceEngineOutput(
            responses=responses,
            stop_reasons=stop_reasons,
            response_ids=response_ids,
            response_logprobs=response_logprobs,
            prompt_logprobs=all_prompt_logprobs,
        )
        if response_top_k is not None and response_top_k > 0:
            result["student_topk_indices"] = student_topk_indices
            result["behavior_topk_logprobs"] = behavior_topk_logprobs
        if any(routes is not None for routes in routed_experts_rows):
            if any(routes is None for routes in routed_experts_rows):
                raise ValueError("vLLM omitted routed experts for part of a batch")
            result["routed_experts"] = [routes for routes in routed_experts_rows if routes is not None]
        return result

    def get_model_max_len(self) -> int:
        """Return the context limit resolved by vLLM's model configuration."""
        return self.llm.model_config.max_model_len

    def _is_lora_disk_loading_request(self, request: NamedWeightsUpdateRequest) -> bool:
        """Check if this is a LoRA disk loading request."""
        is_lora = request["names"][0] == LORA_DISK_LOAD_NAME
        if is_lora:
            assert (
                request.get("extras") and len(request["extras"]) > 0 and LORA_DISK_PATH_KEY in request["extras"][0]
            ), f"vLLM LoRA weight update requests must contain the disk load path under key `{LORA_DISK_PATH_KEY}`"
        return is_lora

    def _create_stat_logger_factory(self):
        """Create a factory that produces stat loggers with the engine ID set."""
        engine_id = self._stats_engine_id

        def factory(*args, **kwargs):
            logger_instance = V1LoggingStatLoggerFixed(*args, **kwargs)
            logger_instance.set_engine_id(engine_id)
            self._stats_attributes = logger_instance._attributes
            return logger_instance

        return factory

    async def initialize_worker_numa_affinity(self):
        """Apply affinity on every asynchronous vLLM worker."""
        return await set_async_worker_numa_affinity(self.llm.collective_rpc)

    def _create_engine(self, *args, **kwargs):
        data_parallel_master_ports = kwargs.pop("data_parallel_master_ports", None)
        wrapper_kwargs = pop_vllm_wrapper_kwargs(kwargs)
        # Store sampling params for OpenAI-style requests (Harbor rollouts)
        self._openai_sampling_params = wrapper_kwargs.pop("openai_sampling_params", {})
        self._validate_rollout_logprob_sampling = wrapper_kwargs.pop(ROLLOUT_LOGPROB_VALIDATION_KEY, False)
        if self._openai_sampling_params:
            logger.warning(
                f"OpenAI API sampling params overridden: "
                f"temperature={self._openai_sampling_params.get('temperature', 1.0)}, "
                f"top_p={self._openai_sampling_params.get('top_p', 1.0)}, "
                f"top_k={self._openai_sampling_params.get('top_k', -1)}"
            )
        # TODO (erictang000): potentially enable log requests for a debugging mode
        # Config resolves the template once so all serving paths render identical prompts.
        custom_chat_template_content = kwargs.pop("chat_template", None)
        chat_template_kwargs = kwargs.pop("default_chat_template_kwargs", None)
        # Use factory to inject engine ID into stat logger
        stat_loggers = [self._create_stat_logger_factory()]

        # Source-built vLLM forks can report misleading version strings, so use
        # the installed constructor fields as the compatibility boundary.
        try:
            _engine_arg_fields = {f.name for f in _dataclass_fields(vllm.AsyncEngineArgs)}
        except TypeError:
            _engine_arg_fields = set()
        # The custom interval logger keeps vLLM stats enabled. This flag only
        # omits vLLM's duplicate console logger; its Prometheus logger remains.
        if "disable_log_stats" in _engine_arg_fields:
            kwargs["disable_log_stats"] = True
        if "enable_log_requests" in _engine_arg_fields:
            engine_args = vllm.AsyncEngineArgs(enable_log_requests=False, **kwargs)
        elif "disable_log_requests" in _engine_arg_fields:
            engine_args = vllm.AsyncEngineArgs(disable_log_requests=True, **kwargs)
        else:
            engine_args = vllm.AsyncEngineArgs(**kwargs)

        if data_parallel_master_ports is not None:
            # Explicit rendezvous ports identify a coordinated DP launch. Rank
            # 0 transfers the reservation immediately before vLLM binds; every
            # rank then joins that generation exactly once. The driver retries
            # the whole actor gang if any rank fails.
            self._release_rendezvous_port_reservation()
            engine = _create_async_engine(engine_args, stat_loggers, data_parallel_master_ports)
        else:
            # Independent engines still use opportunistically selected vLLM
            # ports, so a rank-local retry cannot strand any peers.
            engine = _create_async_engine_with_port_collision_retries(engine_args, stat_loggers)

        # Adapted from https://github.com/volcengine/verl/blob/e90f18c40aa639cd25092b78a5ff7e2d2508c088/verl/workers/rollout/vllm_rollout/vllm_async_server.py#L327
        model_config = engine.model_config
        model_path = kwargs.get("model")
        # Allow overriding the served model name (similar to vLLM's --served-model-name flag).
        # Useful for Harbor/LiteLLM compatibility where model names must have exactly one '/'.
        # See https://github.com/NovaSky-AI/SkyRL/pull/238#discussion_r2326561295
        served_model_name = kwargs.get("served_model_name")
        model_name = served_model_name if served_model_name else model_path

        base_model_paths = [BaseModelPath(name=model_name, model_path=model_path)]

        models = OpenAIServingModels(
            engine_client=engine,
            base_model_paths=base_model_paths,
        )

        # The pinned fork shares one renderer between chat, completion, and
        # tokenization serving.
        online_renderer = OnlineRenderer(
            model_config=model_config,
            renderer=engine.renderer,
            request_logger=None,
            chat_template=custom_chat_template_content,
            chat_template_content_format="auto",
            default_chat_template_kwargs=chat_template_kwargs,
            **wrapper_kwargs,
        )
        online_renderer.warmup()
        self._compact_chat_engine = CompactChatEngineClient(engine, self._id_only_logprob_requests)
        self.openai_serving_chat = SkyRLOpenAIServingChat(
            engine_client=self._compact_chat_engine,
            models=models,
            response_role="assistant",
            online_renderer=online_renderer,
            request_logger=None,
            chat_template=custom_chat_template_content,
            chat_template_content_format="auto",
            default_chat_template_kwargs=chat_template_kwargs,
            **wrapper_kwargs,
        )
        self.openai_serving_completion = OpenAIServingCompletion(
            engine_client=engine,
            models=models,
            online_renderer=online_renderer,
            request_logger=None,
        )
        self.openai_serving_tokenization = ServingTokenization(
            models=models,
            online_renderer=online_renderer,
            request_logger=None,
            chat_template=custom_chat_template_content,
            chat_template_content_format="auto",
            default_chat_template_kwargs=chat_template_kwargs,
        )
        return engine

    async def _load_lora_from_disk(self, lora_path: str):
        """Load LoRA adapters from disk using vLLM's native add_lora method."""
        lora_id = int(time.time_ns() % 0x7FFFFFFF)
        lora_request = LoRARequest(lora_name=f"{lora_id}", lora_int_id=lora_id, lora_path=lora_path)
        result = await self.llm.add_lora(lora_request)
        return result

    async def _collect_outputs(self, prompt_token_ids, request_id: str, sampling_params: SamplingParams):
        """Collect a single prompt's final output and the policy steps of its tokens."""
        # Check if LoRA is enabled and create LoRA request
        final_output = None
        lora_request = None

        if self._is_lora:
            lora_int_ids = list(await self.llm.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                # dummy_lora_path for placeholder (actual loading done in add_lora())
                lora_request = LoRARequest(
                    lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/dummy_lora_path"
                )

        if sampling_params.flat_logprobs and sampling_params.logprobs and sampling_params.logprobs > 0:
            self._id_only_logprob_requests.add(request_id)
        try:
            async for request_output in self.llm.generate(
                prompt=TokensPrompt(prompt_token_ids=prompt_token_ids),
                sampling_params=sampling_params,
                request_id=request_id,
                lora_request=lora_request,
            ):
                final_output = request_output
        finally:
            self._id_only_logprob_requests.discard(request_id)

        return final_output, self._policy_steps.finish(request_id, len(final_output.outputs[0].token_ids))

    async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
        """Generate responses using vLLM's async engine.

        v2 (2026-05-26): wrapped in try/except to explicitly abort all
        sibling in-flight vLLM requests when any task in this batch raises
        (typical case: vLLM serving_chat ValueError on 32k-token validation).
        Without this, the failed task unwinds Python but its sibling tasks'
        Ray ObjectRefs leak into the entrypoint actor's distributed-refcount
        state — that's the `reference_count.cc:1619` SIGABRT pattern that's
        been killing the v3 maxgn09_hint chain links one after another even
        after the harbor rollback_on_exception hook landed (the hook fires
        AFTER the trial, but the ObjectRefs leak DURING the generate batch).
        See agent_logs/2026-05-25_v6a-agrs_507771_moe_combine_stack_pinned.md
        and project_ray_workercrashed_harbor_rollback.md for the chain of
        evidence.
        """
        prompt_token_ids, sampling_params = self._preprocess_prompts(input_batch)

        session_ids = input_batch.get("session_ids")
        if session_ids is not None and len(session_ids) != len(prompt_token_ids):
            raise ValueError("session_ids must align one-for-one with prompt_token_ids")

        tasks = []
        request_ids: list[str] = []
        per_prompt = sampling_params if isinstance(sampling_params, list) else [sampling_params] * len(prompt_token_ids)
        for index, (prompt, row_sampling_params) in enumerate(zip(prompt_token_ids, per_prompt, strict=True)):
            # Schedule the collection of outputs for each prompt.
            # Avoid duplicate request_ids
            if session_ids is None:
                request_id = uuid4().hex
            else:
                request_id = request_id_for_group(session_ids[index])
            request_ids.append(request_id)
            task = asyncio.create_task(self._collect_outputs(prompt, request_id, row_sampling_params))
            tasks.append(task)
        try:
            collected = await asyncio.gather(*tasks)
        except BaseException as e:
            # Cancel any sibling asyncio tasks still in flight.
            for t in tasks:
                if not t.done():
                    t.cancel()
            # Abort their vLLM-side request state so Ray releases the
            # ObjectRefs and the entrypoint's distributed-refcount table
            # doesn't accumulate orphan entries. We tolerate failures of
            # the abort itself — the goal is best-effort cleanup, not a
            # second hard exception.
            try:
                engine = self.llm
                await engine.abort(request_ids)
            except Exception as abort_exc:
                logger.warning(
                    "generate() failed with %r and vllm engine.abort cleanup "
                    "also failed with %r — Ray ObjectRefs may leak",
                    e,
                    abort_exc,
                )
            raise

        outputs, policy_steps = zip(*collected)
        result = self._postprocess_outputs(list(outputs), self._response_top_k(sampling_params), sampling_params)
        result["response_policy_steps"] = list(policy_steps)
        return result

    async def wake_up(self, *args: Any, **kwargs: Any):
        await self.llm.wake_up(tags=kwargs.get("tags", None))

    async def sleep(self, *args: Any, **kwargs: Any):
        engine = self.llm
        output_processor = engine.output_processor
        # make sure that the engine is alive
        engine.engine_core.ensure_alive()
        if output_processor.has_unfinished_requests():
            logger.warning(
                "Calling sleep() with unfinished requests in vLLM engine. This is unexpected since all "
                "generation should be done before sleep() is called. Check for potential failures or "
                "dangling requests in your Generator/Env. Aborting all unfinished requests."
            )
            unfinished_request_ids = list(output_processor.request_states.keys())
            await engine.abort(unfinished_request_ids)
        self._forget_unclaimed_policy_steps()

        # TODO(team): remove once vllm fixes this
        # otherwise waking it up will output gibberish: https://github.com/vllm-project/vllm/issues/17103
        await self.reset_prefix_cache()
        level = 1 if self._is_lora else kwargs.get("level", 2)
        await self.llm.sleep(level=level)

    async def init_weight_update_communicator(
        self, master_addr, master_port, rank_offset, world_size, group_name, backend, override_existing: bool = False
    ):
        engine = self.llm
        return await engine.collective_rpc(
            "init_weight_update_communicator",
            args=(master_addr, master_port, rank_offset, world_size, group_name, backend, override_existing),
        )

    async def begin_online_eagle_capture(self, config: Dict[str, Any]):
        """Begin bounded capture on every worker rank."""
        engine = self.llm
        return await engine.collective_rpc("begin_online_eagle_capture", args=(config,))

    async def seal_online_eagle_capture(self, destination: str):
        """Publish capture from every rank before target synchronization."""
        engine = self.llm
        return await engine.collective_rpc("seal_online_eagle_capture", args=(destination,))

    async def update_draft_weights(self, weights_path: str):
        """Stream one completed checkpoint through vLLM's draft-update API."""
        engine = self.llm
        await engine.start_draft_weight_update()
        await engine.update_weights(WeightTransferUpdateRequest(update_info={"weights_path": weights_path}))
        await engine.finish_weight_update()
        return {"active": True}

    async def update_named_weights(self, request: NamedWeightsUpdateRequest):
        if "names" not in request:
            raise ValueError(f"Expected update weight request with 'names' entry, got keys: {request.keys()}")

        if not len(request["names"]):
            raise ValueError("Update weight request should have atleast one entry in 'names'")

        # Check for LoRA disk loading request
        if self._is_lora_disk_loading_request(request):
            lora_path = request["extras"][0][LORA_DISK_PATH_KEY]
            return await self._load_lora_from_disk(lora_path)

        # Use the weight loader to coordinate weight transfer
        return await self._weight_loader.load_weights(request)

    async def begin_weight_update(self):
        """Signal engines to start accumulating weights for batched loading."""
        engine = self.llm
        return await engine.collective_rpc("begin_weight_update")

    async def end_weight_update(self):
        """Flush accumulated weights via model.load_weights()."""
        engine = self.llm
        return await engine.collective_rpc("end_weight_update")

    async def read_engine_weights(self, hf_names, dump_inventory: bool = False):
        """TEST-ONLY (Stage 6 weight-equality gate): read engine-side weights back
        under the trainer's HF parameter names, gathered across all TP/EP workers.

        Returns ``List[Dict]`` (one dict per worker rank), each as produced by
        ``WorkerWrap.read_named_weights``. The caller assembles the per-rank
        contributions (TP/EP shards) into the full HF tensors to compare against
        the trainer's post-step weights.
        """
        engine = self.llm
        return await engine.collective_rpc("read_named_weights", args=(list(hf_names), dump_inventory))

    async def read_engine_expert_slots_raw(self, layer_idx: int):
        """TEST-ONLY (D1/D2 diag): per-engine-worker RAW FusedMoE local-slot weights +
        the engine's own expert_map for ``layer_idx``. Returns List[Dict] (one per
        engine worker rank). See ``WorkerWrap.read_expert_slots_raw``."""
        engine = self.llm
        return await engine.collective_rpc("read_expert_slots_raw", args=(int(layer_idx),))

    async def report_engine_hosts(self):
        """TEST-ONLY (disaggregation proof): hostname of every engine TP/EP worker."""
        engine = self.llm
        return await engine.collective_rpc("report_host")

    async def report_engine_placement(self):
        """Host, GPU and ranks of every worker of this engine."""
        return await self.llm.collective_rpc("report_device_placement")

    async def expert_block_rpc(self, method: str, *args) -> list:
        """Call one expert-block sync method on every worker of this engine.

        Weights are installed only while the engine is paused. Returns one reply per worker.
        """
        engine = self.llm
        if method == "receive_weights" and not await engine.is_paused():
            raise RuntimeError("Expert-block sync installs weights only while generation is paused")
        return list(await engine.collective_rpc("expert_block_rpc", args=(method, *args)))

    async def begin_weight_reload(self):
        """#1685 fix: open the layerwise-reload bracket on every engine worker so the
        multi-chunk RL sync defers processing; finalize re-runs process_weights_after_loading
        (re-applies the FlashInfer-CUTLASS w13 swap). See WorkerWrap.skyrl_begin_weight_reload."""
        engine = self.llm
        return await engine.collective_rpc("skyrl_begin_weight_reload")

    async def finish_weight_reload(self):
        """#1685 fix: close the layerwise-reload bracket -> finalize_layerwise_reload ->
        process_weights_after_loading (swap_w13_to_w31) re-applied EXACTLY once."""
        engine = self.llm
        return await engine.collective_rpc("skyrl_finish_weight_reload")

    async def teardown(self):
        await self._destroy_weights_update_group()

    async def reset_prefix_cache(self):
        engine = self.llm
        await engine.reset_prefix_cache()

    async def _destroy_weights_update_group(self):
        engine = self.llm
        return await engine.collective_rpc("destroy_weights_update_group")

    # ----------------------------------------
    # Methods for handling OpenAI API requests
    # ----------------------------------------

    async def _handle_openai_request(self, request_payload: Dict[str, Any], endpoint: str) -> Dict[str, Any]:
        """Handle OpenAI API request."""
        assert endpoint in ["/chat/completions", "/completions"]

        body = request_payload.get("json", {})
        headers = request_payload.get("headers", {})
        request_id = headers.get("x-request-id") or uuid4().hex
        exact_prompt_token_ids = body.pop(EXACT_PROMPT_TOKEN_IDS_KEY, None)
        compact_top_k = request_payload.get(CHAT_TOPK_WIDTH_KEY)

        # Apply configured sampling params from generator config.
        # Harbor requests may include their own sampling params; we override
        # with the SkyRL generator config so rollout exploration is consistent.
        apply_openai_sampling(body, self._openai_sampling_params, self._validate_rollout_logprob_sampling)

        # 1. Build request
        try:
            if endpoint == "/chat/completions":
                request = ChatCompletionRequest(**body)
            else:
                request = CompletionRequest(**body)
            assert request.stream is False, "Streaming is not supported in SkyRL yet, please set stream to False."
            if compact_top_k is not None:
                if (
                    endpoint != "/chat/completions"
                    or exact_prompt_token_ids is None
                    or (request.n or 1) != 1
                    or not isinstance(compact_top_k, int)
                    or compact_top_k <= 0
                    or not request.logprobs
                    or request.logprob_token_ids
                    or request.use_beam_search
                ):
                    raise ValueError("Compact heads require an exact-token, single-response native chat request")
                # vLLM still renders and parses the complete chat response. Its
                # public engine client collects the head without display objects.
                request = request.model_copy(update={"logprobs": False, "top_logprobs": 0})
        except Exception as e:
            return _build_error_response(str(e), HTTPStatus.BAD_REQUEST.phrase, HTTPStatus.BAD_REQUEST.value)

        # 2. Call vllm engine
        try:
            # Create a minimal request-like object with attributes used by vLLM
            minimal_request = _MinimalRequest(headers)
            if endpoint == "/chat/completions":
                if exact_prompt_token_ids is None:
                    generator = await self.openai_serving_chat.create_chat_completion(request, minimal_request)
                else:
                    if compact_top_k is None:
                        generator = await self.openai_serving_chat.create_chat_completion_with_exact_prompt(
                            request, minimal_request, exact_prompt_token_ids
                        )
                    else:
                        with self._compact_chat_engine.capture(compact_top_k) as capture:
                            generator = await self.openai_serving_chat.create_chat_completion_with_exact_prompt(
                                request, minimal_request, exact_prompt_token_ids
                            )
                assert isinstance(generator, (ChatCompletionResponse, ErrorResponse))
            else:
                generator = await self.openai_serving_completion.create_completion(request, minimal_request)
                assert isinstance(generator, (CompletionResponse, ErrorResponse))
            response = generator.model_dump()
            if isinstance(generator, ErrorResponse):
                response["request_id"] = request_id
                response["error_category"] = "server_error"
                return response
            choice = response["choices"][0]
            if compact_top_k is not None:
                if capture.response is None or capture.response[0] != choice.get("token_ids"):
                    raise ValueError("Compact chat evidence does not match the served response token IDs")
                choice[CHOICE_TOPK_KEY] = capture.response[1]
            # vLLM generates a chat completion under its response id, which parallel samples share.
            if (
                endpoint == "/chat/completions"
                and len(response["choices"]) == 1
                and choice.get("token_ids") is not None
            ):
                choice[CHOICE_POLICY_STEPS_KEY] = self._policy_steps.finish(response["id"], len(choice["token_ids"]))
            return response

        except Exception as e:
            # Handle it here so we can surface the error from a ray worker.
            #
            # Input-overflow (VLLMValidationError raised at serving.py during
            # input validation, e.g. "You passed 32769 input tokens ... context
            # length is only 32768") is a *deterministic* client error: retrying
            # the identical over-budget prompt can never succeed. Classify it as
            # HTTP 400 (BAD_REQUEST) so that downstream LiteLLM/Harbor map it to
            # a ContextWindowExceededError (non-retryable) instead of treating a
            # generic 500 as a transient server error and retrying. Retrying the
            # doomed request across many concurrent trials is what exhausts the
            # entrypoint actor's file descriptors and aborts its uvloop event
            # loop (uv__epoll_ctl_prep SIGABRT) -> ray.WorkerCrashedError.
            # See project notes: nemotron-junit a3 #11 (chain 521442-448).
            is_input_overflow = False
            try:
                from vllm.exceptions import VLLMValidationError

                if isinstance(e, VLLMValidationError):
                    param = getattr(e, "parameter", None)
                    is_input_overflow = param == "input_tokens" or "input tokens" in str(e)
            except ImportError:
                # Older vLLM without VLLMValidationError: fall back to message match.
                is_input_overflow = "input tokens" in str(e) and "context length" in str(e)

            status = HTTPStatus.BAD_REQUEST if is_input_overflow else HTTPStatus.INTERNAL_SERVER_ERROR
            message = str(e)
            category = (
                "context_overflow"
                if is_input_overflow
                else "constrained_decoding"
                if "Failed to advance FSM" in message or "grammar rejected tokens" in message
                else "internal_error"
            )
            logger.opt(exception=e).error("vLLM {} request_id={} failed: {}", endpoint, request_id, category)
            if is_input_overflow:
                logger.warning("Input-overflow rejected by vLLM serving (returning 400, non-retryable): %s", e)

            response = _build_error_response(message, status.phrase, status.value)
            response["request_id"] = request_id
            response["error_category"] = category
            return response

    async def chat_completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """OpenAI-compatible HTTP endpoint for handling `/chat/completions` in Python vLLM engine.

        Accepts a JSON-serializable payload: {"json": <request-body>, "headers": <headers-dict>}.
        Constructs a minimal request-like object for vLLM's openai_serving_chat.
        Returns a plain dict containing either a chat response or a serving error.
        """
        return await self._handle_openai_request(request_payload, endpoint="/chat/completions")

    async def tokenize(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Return vLLM's native chat-tokenization response."""
        body = request_payload.get("json", {})
        headers = request_payload.get("headers", {})
        try:
            request = TokenizeChatRequest(**body)
        except Exception as e:
            return _build_error_response(str(e), HTTPStatus.BAD_REQUEST.phrase, HTTPStatus.BAD_REQUEST.value)

        response = await self.openai_serving_tokenization.create_tokenize(request, _MinimalRequest(headers))
        assert isinstance(response, (TokenizeResponse, ErrorResponse))
        return response.model_dump()

    async def completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """OpenAI-compatible HTTP endpoint for handling `/completions` in Python vLLM engine.

        Accepts a JSON-serializable payload: {"json": <request-body>, "headers": <headers-dict>}.
        Constructs a minimal request-like object for vLLM's openai_serving_completion.
        Returns a plain dict containing either a completion response or a serving error.
        """
        return await self._handle_openai_request(request_payload, endpoint="/completions")

    @ray.method(num_returns="streaming")
    async def chat_completion_stream(self, request_payload: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Streaming chat completion yielding SSE-formatted ``data: …`` strings.

        Delegates to vLLM's ``openai_serving_chat.create_chat_completion`` with
        ``stream=True`` and surfaces each SSE chunk through Ray's streaming-
        generator transport (``num_returns="streaming"``).  Each ``yield``
        produces one ``ObjectRef`` that the caller resolves to the SSE string.

        ``_ensure_token_ids_in_sse_chunk`` is applied per-chunk so that
        ``provider_specific_fields.token_ids`` is present for harbor's literal
        accumulator.
        """
        body = request_payload.get("json", {})
        headers = request_payload.get("headers", {})
        exact_prompt_token_ids = body.pop(EXACT_PROMPT_TOKEN_IDS_KEY, None)

        apply_openai_sampling(body, self._openai_sampling_params, self._validate_rollout_logprob_sampling)
        body["stream"] = True
        body["return_token_ids"] = True  # force vLLM to emit per-chunk token_ids

        try:
            request = ChatCompletionRequest(**body)
            minimal_request = _MinimalRequest(headers)
            if exact_prompt_token_ids is None:
                result = await self.openai_serving_chat.create_chat_completion(request, minimal_request)
            else:
                result = await self.openai_serving_chat.create_chat_completion_with_exact_prompt(
                    request, minimal_request, exact_prompt_token_ids
                )

            if isinstance(result, ErrorResponse):
                err = result.model_dump()
                yield f"data: {json.dumps(err)}\n\n"
                yield "data: [DONE]\n\n"
                return

            async for chunk in result:
                if isinstance(chunk, bytes):
                    chunk = chunk.decode("utf-8")
                yield ensure_token_ids_in_sse_chunk(chunk)

        except Exception as e:
            is_input_overflow = False
            try:
                from vllm.exceptions import VLLMValidationError

                if isinstance(e, VLLMValidationError):
                    param = getattr(e, "parameter", None)
                    is_input_overflow = param == "input_tokens" or "input tokens" in str(e)
            except ImportError:
                is_input_overflow = "input tokens" in str(e) and "context length" in str(e)

            status = HTTPStatus.BAD_REQUEST if is_input_overflow else HTTPStatus.INTERNAL_SERVER_ERROR
            err = _build_error_response(str(e), status.phrase, status.value)
            yield f"data: {json.dumps(err)}\n\n"
            yield "data: [DONE]\n\n"

    async def get_stats(self, read_mode: IntervalReadMode = IntervalReadMode.RESET) -> VLLMEngineStatsSnapshot:
        """Return the engine's complete typed snapshot without publishing it."""
        from vllm.v1.metrics.reader import get_metrics_snapshot  # noqa: PLC0415

        native = snapshot_vllm_prometheus_metrics(
            get_metrics_snapshot(),
            engine_index=self._stats_attributes.get("engine_index", "0"),
        )
        snapshot = V1LoggingStatLoggerFixed.get_stats_by_engine_id(
            self._stats_engine_id,
            reset=read_mode is IntervalReadMode.RESET,
        )
        if snapshot is None:
            snapshot = VLLMEngineStatsSnapshot(
                engine_id=str(self._stats_engine_id),
                timestamp=time.time(),
                current=VLLMCurrentStats(),
                cumulative=VLLMCumulativeStats(),
                interval=VLLMIntervalStats(),
                attributes=self._stats_attributes,
            )

        return replace(
            snapshot,
            current=native.current,
            cumulative=native.cumulative,
            histograms=native.histograms,
        )

    async def pause_generation(self) -> None:
        """Apply the configured pause policy while the engine reloads weights."""
        engine = self.llm
        outstanding_requests = len(engine.output_processor.request_states)
        policy = self._weight_sync_pause_policy
        await engine.pause_generation(mode=policy.mode.value, clear_cache=policy.clear_cache)
        logger.info(
            "pause_generation() finished, mode={}, clear_cache={}, outstanding_requests={}, EngineCore paused",
            policy.mode.value,
            policy.clear_cache,
            outstanding_requests,
        )

    async def resume_generation(self, policy_step: int) -> None:
        """Release the EngineCore scheduler after the weight reload completes, stamping later tokens with ``policy_step``."""
        self._policy_steps.install(policy_step, time.monotonic())
        self._forget_unclaimed_policy_steps()
        await self.llm.resume_generation()
        logger.info("resume_generation() finished, EngineCore scheduler released, policy_step={}", policy_step)


class _MinimalRequest:
    """
    Minimal request-like object for vLLM's openai_serving_chat and openai_serving_completion.

    We cannot use the original user Request object because it cannot be serialized and hence
    cannot be a ray method argument. Instead we take the original request's headers and
    reconstruct an instance of _MinimalRequest to mimic the FastAPI Request object.

    The fields depend on what vLLM accesses internally.
    """

    def __init__(self, headers):
        self.headers = headers  # Expect a mapping with .get support
        self.state = SimpleNamespace()  # vLLM sets raw_request.state.request_metadata


class VLLMWeightTransferReceiver:
    """Receives weights via broadcast or CUDA IPC for vLLM.

    Handles both transfer strategies based on the request contents.
    Created locally in WorkerWrap with worker-specific state.
    """

    def __init__(self, model_update_group: Any, model_config: Any, device: torch.device) -> None:
        """Initialize the receiver with worker-local state.

        Args:
            model_update_group: Torch process group for weight updates.
            model_config: vLLM model configuration.
            device: CUDA device for this worker.
        """
        self.model_update_group = model_update_group
        self.model_config = model_config
        self.device = device

    def _is_compatible_weight_dtype(self, name: str, dtype: torch.dtype) -> bool:
        hf_config = getattr(self.model_config, "hf_text_config", None)
        if hf_config is None:
            hf_config = getattr(self.model_config, "hf_config", None)
        return is_weight_sync_dtype_compatible(
            getattr(hf_config, "model_type", None),
            name,
            dtype,
            self.model_config.dtype,
        )

    def receive_weights(self, request: NamedWeightsUpdateRequest) -> Iterator[Tuple[str, torch.Tensor]]:
        """Receive weights and yield (name, tensor) tuples.

        Args:
            request: Weight update request with names, dtypes, shapes, and optionally IPC handles.
        """
        extras = request.get("extras")
        is_ipc = extras and len(extras) > 0 and "ipc_handles" in extras[0]

        if is_ipc:
            yield from self._receive_ipc(request)
        else:
            yield from self._receive_broadcast(request)

    def _receive_broadcast(self, request: NamedWeightsUpdateRequest) -> Iterator[Tuple[str, torch.Tensor]]:
        """Receive weights via torch.distributed.broadcast."""
        _fuse = bool(request.get("packed", False))
        for name, dtype_str, shape in zip(request["names"], request["dtypes"], request["shapes"]):
            dtype = str_to_torch_dtype(dtype_str)
            if not _fuse:
                assert self._is_compatible_weight_dtype(name, dtype), (
                    f"mismatch dtype: src {dtype}, dst {self.model_config.dtype}"
                )
            # Always receive in sender's dtype, load_weights handles conversion
            weight = torch.empty(shape, dtype=dtype, device="cuda")
            torch.distributed.broadcast(weight, 0, group=self.model_update_group)
            yield name, weight

    def _receive_ipc(self, request: NamedWeightsUpdateRequest) -> Iterator[Tuple[str, torch.Tensor]]:
        """Receive weights via CUDA IPC handles."""
        names = request["names"]
        dtypes = request["dtypes"]
        shapes = request["shapes"]
        sizes = request.get("sizes", [])
        ipc_handles = [extra["ipc_handles"] for extra in request["extras"]]
        packed = request.get("packed", False)

        if packed:
            assert len(ipc_handles) == 1, "packed weight update should receive one ipc handle for all tensors"
            assert len(set(dtypes)) == 1, "packed weight update should have all tensors with the same dtype"
            dtype = str_to_torch_dtype(dtypes[0])
            assert all(self._is_compatible_weight_dtype(name, dtype) for name in names), (
                f"mismatch dtype: src {dtypes[0]}, dst {self.model_config.dtype}"
            )
            assert len(sizes) == len(names), "sizes must be provided for packed weight update"
            assert all(isinstance(size, int) for size in sizes), "sizes should be a list of integers"

            cuda_device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(cuda_device)
            physical_gpu_id = str(props.uuid)

            handle = ipc_handles[0][physical_gpu_id]
            device_id = self.device.index
            func, args = handle
            list_args = list(args)
            list_args[6] = device_id
            packed_tensor = func(*list_args)

            offset = 0
            for name, shape, size in zip(names, shapes, sizes):
                yield name, packed_tensor[offset : offset + size].view(*shape)
                offset += size
        else:
            cuda_device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(cuda_device)
            physical_gpu_id = str(props.uuid)
            for name, dtype_str, shape, ipc_handle in zip(names, dtypes, shapes, ipc_handles):
                dtype = str_to_torch_dtype(dtype_str)
                assert self._is_compatible_weight_dtype(name, dtype), (
                    f"mismatch dtype: src {dtype}, dst {self.model_config.dtype}"
                )

                handle = ipc_handle[physical_gpu_id]
                device_id = self.device.index
                func, args = handle
                list_args = list(args)
                list_args[6] = device_id
                weight = func(*list_args)
                yield name, weight


class VLLMWeightLoader(WeightLoader):
    """Loads weights into vLLM engine, managing RPC coordination.

    This loader encapsulates the collective_rpc calls to workers.
    Workers create VLLMWeightTransferReceiver locally for the actual weight transfer.
    """

    def __init__(self, engine: vllm.AsyncLLMEngine) -> None:
        self._engine = engine

    async def load_weights(self, request: NamedWeightsUpdateRequest) -> None:
        """Load weights by coordinating RPC to workers.

        Sends the request to workers via collective_rpc. Workers create
        the receiver locally and use it to receive and load weights.

        Args:
            request: Weight update request containing names, dtypes, shapes,
                    and optionally IPC handles.
        """
        await self._engine.collective_rpc(
            "load_weights",
            args=(request,),
        )


AsyncVLLMRayActor = ray.remote(AsyncVLLMInferenceEngine)
