import asyncio
from collections.abc import Callable, Collection
from enum import StrEnum
import os
import time
from typing import Any, Dict, List

import ray
from loguru import logger
from packaging import version
from ray.actor import ActorHandle, ActorMethod
from ray.util.placement_group import PlacementGroupSchedulingStrategy, placement_group, remove_placement_group
from rigging.timing import ExponentialBackoff, retry_with_backoff
from transformers import AutoConfig, PretrainedConfig

from marinskyrl.inference_placement import InferenceReplicaPlacement
from skyrl_train.inference_engines.base import (
    InferenceEngineInterface,
    InferenceEngineInput,
    InferenceEngineOutput,
    NamedWeightsUpdateRequest,
)
from skyrl_train.config.weight_sync_pause import (
    DEFAULT_WEIGHT_SYNC_PAUSE_POLICY,
    WeightSyncPausePolicy,
    validate_weight_sync_pause_backend,
)
from skyrl_train.inference_engines.vllm.stats import IntervalReadMode
from skyrl_train.inference_engines.utils import (
    ReservedRendezvousPorts,
    VLLM_DATA_PARALLEL_MASTER_PORT_COUNT,
    get_pg_bundle_node_ips,
    reserve_rendezvous_ports,
)
from skyrl_train.models.grug_moe import GRUG_MOE_ARCHITECTURE, GRUG_MOE_MODEL_TYPE
from skyrl_train.env_vars import (
    DEFAULT_RUNAI_STREAMER_LOG_TO_STDERR,
    DEFAULT_RUNAI_STREAMER_S3_REQUEST_TIMEOUT_MS,
    EnvVarScope,
    RUNAI_STREAMER_LOG_TO_STDERR_ENV,
    RUNAI_STREAMER_S3_REQUEST_TIMEOUT_MS_ENV,
    NCCL_BUFFER_SIZE_ENV_VAR,
    VLLM_USE_V2_MODEL_RUNNER_ENV,
    managed_environment_names,
)
from skyrl_train.utils import (
    get_all_env_variables,
    get_ray_pg_ready_with_timeout,
    ray_noset_visible_devices,
)
from skyrl_train.utils.constants import DEFAULT_RAY_PLACEMENT_GROUP_TIMEOUT_SECONDS
from skyrl_train.utils.utils import use_per_engine_pg
from skyrl_train.inference_engines.placement import (
    colocated_engine_bundle_layout,
    inference_bundle_nodes,
    verified_inference_replica_placements,
)

MODEL_METADATA_PATH_KEY = "_marinskyrl_metadata_path"

# ---------------------------------------------------------------------------
# #232 FIX B — NCCL flight-recorder observability env -> vLLM engine workers.
#
# The vLLM inference-engine actors (and, under the ray executor backend, the
# per-rank TP worker actors they spawn) do NOT reliably inherit the NCCL
# flight-recorder / watchdog env vars that are set on the host launch shell.
# On Jupiter those vars are exported as APPTAINERENV_TORCH_NCCL_* (so apptainer
# imports them into the `ray start` raylet process), but the Ray-actor-spawned
# vLLM EngineCore / TP workers run with a runtime_env that does NOT carry the
# raylet's TORCH_NCCL_* through to the actual collective-running process
# (job 919724: the execute_model wedge fired but NO nccl_fr_rank* dump was
# written -> NCCL's 600s watchdog never armed inside the worker, so vLLM's 900s
# RPC watchdog always preempted it). Without these vars IN the worker process
# env, a residual TP-rank desync can never write a flight-recorder trace.
#
# Fix: explicitly forward an allowlist of NCCL FR / debug vars from the
# engine-creation process env (which DOES have them, set by the launch shell)
# into the actor's Ray runtime_env env_vars. With
# placement_group_capture_child_tasks=True (already set on the scheduling
# strategy), the ray-backend TP worker actors inherit this runtime_env too, so
# the vars land IN the process that actually runs the NCCL collective. When NONE
# of these vars are set in the launching env (every non-#232 run), the dict is
# empty and runtime_env is None -> byte-identical actor creation as before.
_NCCL_FR_ENV_PASSTHROUGH = (
    # Manual dump triggers are an advanced external input rather than part of
    # the debug preset. All managed watchdog/recorder variables come from the
    # EnvVarManager scope below.
    "TORCH_NCCL_DEBUG_INFO_PIPE_FILE",
)


def validate_grug_vllm_support(hf_config: PretrainedConfig, supported_architectures: Collection[str]) -> None:
    """Fail before actor creation when the running vLLM cannot serve Grug."""

    if getattr(hf_config, "model_type", None) != GRUG_MOE_MODEL_TYPE:
        return
    if GRUG_MOE_ARCHITECTURE not in supported_architectures:
        raise RuntimeError(
            "The running vLLM build does not support GrugMoeForCausalLM. "
            "Install the pinned Marin vLLM wheel from the root lockfile."
        )


def _validate_installed_vllm_for_model(pretrain: str) -> None:
    from vllm.model_executor.models import ModelRegistry  # noqa: PLC0415

    hf_config = AutoConfig.from_pretrained(pretrain, trust_remote_code=True)
    validate_grug_vllm_support(hf_config, ModelRegistry.get_supported_archs())


def resolve_engine_max_model_len(engine_init_kwargs: Dict[str, Any], rope_scaling: Dict[str, Any] | None) -> int | None:
    """Resolve the context limit passed to the inference engine."""
    configured_limit = engine_init_kwargs.get("max_model_len")
    if configured_limit is not None:
        return configured_limit
    if not rope_scaling:
        return None

    rope_factor = rope_scaling.get("factor")
    rope_max_pos = rope_scaling.get("original_max_position_embeddings")
    assert rope_factor is not None, "Please provide rope scaling `factor` to compute model max length"
    assert rope_max_pos is not None, (
        "Please provide rope `original_max_position_embeddings` to compute model max length"
    )
    return int(rope_factor * rope_max_pos)


class EnginePlacementMode(StrEnum):
    HYBRID = "hybrid"
    MULTIPROCESSING = "multiprocessing"
    PER_ENGINE = "per_engine"
    SHARED = "shared"


def _dp_rank_bundle_indices(
    *,
    engine_index: int,
    data_parallel_size: int,
    tensor_pipeline_size: int,
    placement_mode: EnginePlacementMode,
    colocated_engine_bundles: list[list[int]],
) -> list[int]:
    if placement_mode is EnginePlacementMode.HYBRID:
        return [
            colocated_engine_bundles[engine_index * data_parallel_size + rank][0] for rank in range(data_parallel_size)
        ]
    if placement_mode is EnginePlacementMode.MULTIPROCESSING:
        return [engine_index * data_parallel_size + rank for rank in range(data_parallel_size)]
    base_index = (
        0
        if placement_mode is EnginePlacementMode.PER_ENGINE
        else engine_index * tensor_pipeline_size * data_parallel_size
    )
    return [base_index + rank * tensor_pipeline_size for rank in range(data_parallel_size)]


def _validate_node_local_dp_ranks(engine_index: int, placement_group, bundle_indices: list[int]) -> None:
    node_ips = get_pg_bundle_node_ips(placement_group, bundle_indices)
    logger.info(f"inference engine {engine_index}: DP rank -> node {dict(enumerate(node_ips))}")
    if len(set(node_ips)) > 1:
        raise RuntimeError(
            f"inference engine {engine_index}: its DP ranks were placed on "
            f"{len(set(node_ips))} nodes ({dict(enumerate(node_ips))}); a DP/EP group must be node-local"
        )


def _reserve_engine_rendezvous(
    placement_group,
    bundle_index: int,
    data_parallel_size: int,
    allocated_ports: set[int],
) -> ReservedRendezvousPorts | None:
    if data_parallel_size == 1:
        return None
    rendezvous = reserve_rendezvous_ports(
        placement_group,
        bundle_index,
        port_count=1 + VLLM_DATA_PARALLEL_MASTER_PORT_COUNT,
        excluded_ports=allocated_ports,
    )
    allocated_ports.update(rendezvous.ports)
    return rendezvous


def _qwen3_5_vlm_engine_kwargs(pretrain: str, *, revision: str | None = None) -> Dict[str, Any]:
    """vLLM EngineArgs overrides for the Qwen3.5/3.6 VLM-shell rollout (tmax Stage 2).

    The Qwen3.6-35B-A3B (``qwen3_5_moe``) checkpoint's ``architectures`` names the
    multimodal shell ``Qwen3_5MoeForConditionalGeneration``, so vLLM materializes
    the FULL shell (text tower + vision tower + MTP). For RL rollouts we only need
    the text tower; tmax sets ``language_model_only=True`` (its
    ``open_instruct/vllm_utils.py`` ~1556-1559), which in this vLLM fork is a
    top-level ``EngineArgs`` field (``arg_utils.py``: ``language_model_only =
    MultiModalConfig.language_model_only``) that forces every modality's
    ``limit_mm_per_prompt`` to 0. That, via the shell's ``_mark_tower_model``
    guard (``get_limit_per_prompt(m) == 0`` -> ``no_init_weights``), STUBS the
    vision tower so it is neither built nor expected at ``load_weights`` (no
    ``visual.*`` keys required), while the text ``language_model`` tower is built
    normally. The Megatron weight-sync path maps policy weights into the
    language-model tower expected by the engine.

    Returns ``{"language_model_only": True}`` when ``pretrain`` is a qwen3_5 VLM
    shell, else ``{}`` (byte-identical engine construction for every other model).
    Reads only the HF config (cheap, no weights) and never raises — on any failure
    it returns ``{}`` so non-VLM launches are unaffected.
    """
    try:
        from skyrl_train.models.qwen3_5_vlm import is_qwen3_5_vlm_shell

        cfg = AutoConfig.from_pretrained(pretrain, trust_remote_code=True, revision=revision)
        if is_qwen3_5_vlm_shell(cfg):
            logger.info(
                f"[qwen3_5_vlm] {pretrain} is a Qwen3.5/3.6 VLM shell; setting vLLM "
                f"engine language_model_only=True (text tower only, vision stubbed)."
            )
            return {"language_model_only": True}
    except Exception as e:  # pragma: no cover - defensive; never block non-VLM launches
        logger.warning(f"[qwen3_5_vlm] VLM-shell engine-kwargs probe failed for {pretrain!r}: {e}")
    return {}


def _build_inference_engine_runtime_env(
    *,
    require_v1_model_runner: bool = False,
    runai_streamer_enabled: bool = False,
    nccl_buffer_size_bytes: int | None = None,
) -> Dict[str, Any] | None:
    """Forward managed inference settings into each vLLM engine actor.

    This covers NCCL diagnostics, the policy's weight-sync buffer size, and batch
    invariance. Selected-ID scoring also forces the V1 model runner. RunAI-backed engines receive conservative
    S3 stall tolerance and stderr diagnostics. Returns ``None`` when no managed
    or derived settings apply.
    """
    passthrough = set(_NCCL_FR_ENV_PASSTHROUGH) | set(managed_environment_names(EnvVarScope.INFERENCE_WORKER))
    env_vars = {key: os.environ[key] for key in passthrough if key in os.environ}
    if nccl_buffer_size_bytes is not None:
        env_vars[NCCL_BUFFER_SIZE_ENV_VAR] = str(nccl_buffer_size_bytes)
    if require_v1_model_runner:
        # Selected-ID prompt scoring is implemented in vLLM's V1 GPU runner.
        # Set this before actor import so the EngineCore and TP workers agree.
        env_vars[VLLM_USE_V2_MODEL_RUNNER_ENV] = "0"
    if runai_streamer_enabled:
        # RunAI 0.16.x defaults the S3 low-throughput interval to one second;
        # the bundled AWS CRT raises that to only three seconds. A ten-second
        # interval tolerates brief object-store stalls while leaving the whole-
        # gang startup retry as the bounded recovery for a failed chunk.
        env_vars.setdefault(
            RUNAI_STREAMER_S3_REQUEST_TIMEOUT_MS_ENV,
            DEFAULT_RUNAI_STREAMER_S3_REQUEST_TIMEOUT_MS,
        )
        env_vars.setdefault(RUNAI_STREAMER_LOG_TO_STDERR_ENV, DEFAULT_RUNAI_STREAMER_LOG_TO_STDERR)
    if not env_vars:
        return None
    logger.info(f"Forwarding managed environment to vLLM engine actors via runtime_env: {sorted(env_vars)}")
    return {"env_vars": env_vars}


async def _await_actor_task(actor_task: Any) -> Any:
    """Await a Ray actor task and cancel the remote work if the caller leaves."""
    try:
        return await actor_task
    except asyncio.CancelledError:
        # Cancelling an ObjectRef await does not cancel the underlying Ray task.
        ray.cancel(actor_task)
        raise


async def _submit_actor_task(
    remote_method: ActorMethod, *args: Any, **kwargs: Any
) -> ray.ObjectRef | ray.ObjectRefGenerator:
    """Submit without blocking the HTTP event loop, including slow Ray submission."""
    submission = asyncio.create_task(asyncio.to_thread(remote_method.remote, *args, **kwargs))
    try:
        return await asyncio.shield(submission)
    except asyncio.CancelledError:

        async def cancel_late_submission() -> None:
            try:
                actor_task = await submission
            except Exception:
                logger.exception("Ray submission failed after request cancellation")
                return
            await asyncio.to_thread(ray.cancel, actor_task)

        asyncio.create_task(cancel_late_submission())
        raise


class RayWrappedInferenceEngine(InferenceEngineInterface):
    """
    A thin wrapper around a Ray ActorHandle to another InferenceEngineInterface.
    This class implements the InferenceEngineInterface by delegating calls to the remote actor.
    """

    def __init__(
        self,
        inference_engine_actor: ActorHandle,
        *,
        weight_sync_relative_rank_offset: int | None = None,
    ):
        self.inference_engine_actor = inference_engine_actor
        self.weight_sync_relative_rank_offset = weight_sync_relative_rank_offset
        # The verified placement of each worker of a replica, in worker order.
        self.worker_placements: list[InferenceReplicaPlacement] | None = None

    def tp_size(self):
        # Diagnostic: unwrap un-pickleable Ray exceptions into a plain
        # RuntimeError. When a raylet dies (e.g. GPFS SIGBUS/ESTALE during
        # weight-sync-state init) Ray raises a dynamically-generated
        # RayTaskError(ActorDiedError) whose re-serialization across the dying
        # boundary surfaces as a PicklingError / pydantic_compat
        # ModuleNotFoundError red herring. Re-raising as a picklable plain
        # exception preserves the TRUE cause in logs. Happy path unchanged.
        try:
            return ray.get(self.inference_engine_actor.tp_size.remote())
        except ray.exceptions.RayError as e:
            raise RuntimeError(f"tp_size() failed at Ray boundary: {e!r}") from None

    def pp_size(self):
        return ray.get(self.inference_engine_actor.pp_size.remote())

    def dp_size(self):
        return ray.get(self.inference_engine_actor.dp_size.remote())

    async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
        actor_task = await _submit_actor_task(self.inference_engine_actor.generate, input_batch=input_batch)
        return await _await_actor_task(actor_task)

    async def wake_up(self, *args: Any, **kwargs: Any):
        return await self.inference_engine_actor.wake_up.remote(*args, **kwargs)

    async def sleep(self, *args: Any, **kwargs: Any):
        return await self.inference_engine_actor.sleep.remote(*args, **kwargs)

    async def init_weight_update_communicator(
        self, master_addr, master_port, rank_offset, world_size, group_name, backend, override_existing: bool = False
    ):
        return await self.inference_engine_actor.init_weight_update_communicator.remote(
            master_addr, master_port, rank_offset, world_size, group_name, backend, override_existing
        )

    async def update_named_weights(self, request: NamedWeightsUpdateRequest):
        return await self.inference_engine_actor.update_named_weights.remote(request)

    async def expert_block_rpc(self, method: str, *args):
        return await self.inference_engine_actor.expert_block_rpc.remote(method, *args)

    async def begin_weight_reload(self):
        return await self.inference_engine_actor.begin_weight_reload.remote()

    async def finish_weight_reload(self):
        return await self.inference_engine_actor.finish_weight_reload.remote()

    async def teardown(self):
        return await self.inference_engine_actor.teardown.remote()

    async def reset_prefix_cache(self):
        return await self.inference_engine_actor.reset_prefix_cache.remote()

    async def chat_completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        actor_task = await _submit_actor_task(self.inference_engine_actor.chat_completion, request_payload)
        return await _await_actor_task(actor_task)

    async def tokenize(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        actor_task = await _submit_actor_task(self.inference_engine_actor.tokenize, request_payload)
        return await _await_actor_task(actor_task)

    async def chat_completion_stream(self, request_payload: Dict[str, Any]):
        """Stream SSE chunks from the Ray actor via ``num_returns="streaming"``.

        ``ObjectRefGenerator.__anext__`` returns a raw ``ObjectRef`` (NOT the
        resolved value), so each ref must be ``await``-ed to materialise the
        SSE string the actor yielded.  Without this ``await`` the raw
        ``ObjectRef`` objects reach ``StreamingResponse`` and the connection
        dies with ``RemoteProtocolError: incomplete chunked read``.
        """
        gen = await _submit_actor_task(self.inference_engine_actor.chat_completion_stream, request_payload)
        try:
            async for ref in gen:
                yield await ref
        finally:
            ray.cancel(gen)

    async def completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        actor_task = await _submit_actor_task(self.inference_engine_actor.completion, request_payload)
        return await _await_actor_task(actor_task)

    async def pause_generation(self) -> None:
        return await self.inference_engine_actor.pause_generation.remote()

    async def resume_generation(self, policy_step: int) -> None:
        return await self.inference_engine_actor.resume_generation.remote(policy_step)

    async def begin_online_eagle_capture(self, config: Dict[str, Any]):
        return await self.inference_engine_actor.begin_online_eagle_capture.remote(config)

    async def seal_online_eagle_capture(self, destination: str):
        return await self.inference_engine_actor.seal_online_eagle_capture.remote(destination)

    async def update_draft_weights(self, weights_path: str):
        return await self.inference_engine_actor.update_draft_weights.remote(weights_path)

    async def get_stats(self, read_mode: IntervalReadMode = IntervalReadMode.RESET):
        """Return throughput, latency, cache, token, and request statistics."""
        return await self.inference_engine_actor.get_stats.remote(read_mode=read_mode)


def populate_engine_max_model_lens(
    engines: List["RayWrappedInferenceEngine"], *, timeout_seconds: float, kill_on_failure: bool = True
) -> None:
    """Record each serving actor's resolved context limit on its local wrapper."""
    actor_handles = [engine.inference_engine_actor for engine in engines]
    limit_refs = [actor.get_model_max_len.remote() for actor in actor_handles]
    wait_for_inference_engine_startup(
        limit_refs,
        actor_handles,
        timeout_seconds=timeout_seconds,
        kill_on_failure=kill_on_failure,
    )
    for engine, max_model_len in zip(engines, ray.get(limit_refs), strict=True):
        engine.max_model_len = max_model_len


def _release_node_local_gang(actors: list, placement_groups: list) -> None:
    """After a failed node-local startup, kill the started actors and remove the placement groups."""
    for actor in actors:
        ray.kill(actor)
    for pg in placement_groups:
        remove_placement_group(pg)


def create_ray_wrapped_inference_engines_with_retry(
    create_engines: Callable[[float], List[InferenceEngineInterface]],
    *,
    model_path: str,
    engine_init_timeout_seconds: float,
    max_retries: int,
    backoff_base_seconds: float,
    backoff_cap_seconds: float,
) -> List[InferenceEngineInterface]:
    """Retry a failed remote vLLM actor gang within one startup deadline.

    RunAI's S3 streamer reports a transient object-store read failure only after
    the EngineCore processes have exited, so the parent sees an ``ActorDiedError``
    without the underlying status code. A data-parallel pool must restart as a
    gang: retrying one rank would leave it on a different coordinator generation.
    """
    started = time.monotonic()

    def create() -> List[InferenceEngineInterface]:
        remaining = engine_init_timeout_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError(f"inference engine startup timed out after {engine_init_timeout_seconds:g} seconds")
        return create_engines(remaining)

    return retry_with_backoff(
        create,
        retryable=lambda error: isinstance(error, ray.exceptions.RayActorError),
        max_attempts=max_retries + 1,
        max_elapsed=engine_init_timeout_seconds,
        backoff=ExponentialBackoff(
            initial=backoff_base_seconds,
            maximum=backoff_cap_seconds,
            factor=2.0,
            jitter=0.2,
        ),
        operation=f"initialize remote vLLM engine gang for {model_path}",
    )


def _cleanup_failed_engine_startup(inference_engine_actors: list[ActorHandle], owned_placement_groups: list) -> None:
    """Best-effort teardown before retrying a failed engine gang."""
    for actor in inference_engine_actors:
        try:
            ray.kill(actor)
        except Exception as cleanup_error:
            logger.warning("Failed to stop vLLM actor after startup failure: {}", cleanup_error)
    for placement_group_handle in owned_placement_groups:
        try:
            remove_placement_group(placement_group_handle)
        except Exception as cleanup_error:
            logger.warning("Failed to remove vLLM placement group after startup failure: {}", cleanup_error)


def create_ray_wrapped_inference_engines(
    num_inference_engines: int,
    tensor_parallel_size: int,
    model_dtype: str,
    pretrain: str,
    seed: int,
    vllm_v1_disable_multiproc: bool,
    enable_prefix_caching: bool,
    enforce_eager: bool,
    engine_init_timeout_seconds: float,
    expert_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    data_parallel_size: int = 1,
    decode_context_parallel_size: int = 1,
    shared_pg=None,
    gpu_memory_utilization=None,
    inference_engine_enable_sleep=False,
    max_num_batched_tokens=8192,
    max_num_seqs=1024,
    tokenizer=None,
    backend="vllm",
    vllm_attention_backend: str | None = None,
    sleep_level=2,  # we only set to 1 for unit tests that do not explicitly sync weights or for LoRA
    enable_lora=False,
    max_lora_rank=64,
    max_loras=1,
    fully_sharded_loras=False,
    engine_init_kwargs: Dict[str, Any] = {},
    rope_scaling: Dict[str, Any] = {},
    rope_theta: float | None = None,
    max_logprobs: int = 1,
    require_v1_model_runner: bool = False,
    mp_backend: bool = False,
    require_verified_placements: bool = False,
    placement_group_timeout_seconds: int = DEFAULT_RAY_PLACEMENT_GROUP_TIMEOUT_SECONDS,
    weight_sync_pause_policy: WeightSyncPausePolicy = DEFAULT_WEIGHT_SYNC_PAUSE_POLICY,
    nccl_buffer_size_bytes: int | None = None,
) -> List[InferenceEngineInterface]:
    """
    Create a list of RayWrappedInferenceEngine instances wrapping Ray actor handles to InferenceEngineInterface instances.

    mp_backend: opt-in. When True (and TP>1 / PP>1 and NOT colocated), run each vLLM
        inference engine with the `mp` (multiprocessing) executor backend instead of `ray`.
        This is required for the Qwen3-Next-80B-A3B R3 router-capture path
        (`enable_return_routed_experts=true`): the vLLM Ray Compiled-DAG deadlocks on the
        hybrid (GatedDeltaNet + full-attn) arch when capture is on (rank-0 stuck in the DAG
        channel read at 0% GPU; reproduced + root-caused 2026-06-08). The `mp` executor has
        no Ray Compiled-DAG and runs the same config cleanly at full (cudagraph) speed.
        Default False => byte-identical behaviour for every other run. Only valid for
        non-colocated engines (each engine owns its own GPUs); colocated/hybrid engines
        still require the ray backend for shared-GPU resource management.
    """
    engine_init_kwargs = dict(engine_init_kwargs)
    # Direct factory callers bypass generator config validation.
    validate_weight_sync_pause_backend(weight_sync_pause_policy, backend=backend, run_engines_locally=True)
    model_metadata_path = engine_init_kwargs.pop(MODEL_METADATA_PATH_KEY, pretrain)
    if backend == "vllm":
        import vllm
        from skyrl_train.inference_engines.vllm.vllm_engine import AsyncVLLMRayActor

        # if a dev version is being used, skip the version check
        if "dev" not in vllm.__version__:
            assert version.parse(vllm.__version__) >= version.parse("0.8.3"), "SkyRL-Train only supports vLLM >= 0.8.3"
        _validate_installed_vllm_for_model(model_metadata_path)
    elif backend == "sglang":
        # We import SGLang later to avoid importing vllm. See `get_sglang_engine` for more.
        pass
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    inference_engine_actors = []
    weight_sync_relative_rank_offsets = []
    # Qwen3.5/3.6 VLM-shell rollout (tmax Stage 2): materialize the text tower only
    # (language_model_only=True) so vLLM does not build/expect the vision tower.
    # Empty {} for every non-VLM-shell model -> byte-identical engine construction.
    # Do NOT clobber an explicit ++generator.engine_init_kwargs.language_model_only.
    vlm_engine_kwargs = (
        _qwen3_5_vlm_engine_kwargs(model_metadata_path, revision=engine_init_kwargs.get("revision"))
        if backend == "vllm" and "language_model_only" not in engine_init_kwargs
        else {}
    )
    # Forward managed environment and the policy's NCCL buffer size into engine workers.
    inference_engine_runtime_env = _build_inference_engine_runtime_env(
        require_v1_model_runner=require_v1_model_runner,
        runai_streamer_enabled=engine_init_kwargs.get("load_format") == "runai_streamer",
        nccl_buffer_size_bytes=nccl_buffer_size_bytes,
    )
    noset_visible_devices = ray_noset_visible_devices(ray.get(get_all_env_variables.remote()))
    use_hybrid_engine = shared_pg is not None
    tp_pp_size = tensor_parallel_size * pipeline_parallel_size
    colocated_engine_bundles = colocated_engine_bundle_layout(
        shared_pg,
        num_inference_engines=num_inference_engines,
        data_parallel_size=data_parallel_size,
        tensor_pipeline_size=tp_pp_size,
    )
    # NOTE: we use the ray backend for tensor parallel size > 1 or pipeline parallel size > 1 to explicitly manage resource allocation
    # mp_backend (opt-in) lets a NON-colocated multi-GPU engine use vLLM's `mp` executor
    # instead, which avoids the Ray Compiled-DAG deadlock on the Qwen3-Next R3 capture path.
    # In that mode the single SkyRL actor owns the whole TP*PP GPU slice and vLLM spawns its
    # workers as local subprocesses (no per-worker Ray actors).
    use_mp_backend = bool(mp_backend) and tp_pp_size > 1 and not use_hybrid_engine
    if bool(mp_backend) and tp_pp_size > 1 and use_hybrid_engine:
        raise ValueError(
            "generator.inference_engine_mp_backend=true is only supported for NON-colocated "
            "inference engines (trainer.placement.colocate_all=false). Colocated engines need "
            "the ray backend for shared-GPU resource management."
        )
    # Expert-block publication needs checked receivers even for a single serving rank.
    verify_workers = (
        backend == "vllm"
        and tensor_parallel_size == 1
        and (data_parallel_size > 1 or require_verified_placements)
        and not (use_hybrid_engine or use_mp_backend or inference_engine_enable_sleep)
    )
    if require_verified_placements and not verify_workers:
        raise ValueError("Verified expert-block placements require non-colocated TP=1 vLLM workers")
    cross_node_ep = verify_workers and pipeline_parallel_size == 1 and expert_parallel_size == data_parallel_size > 1
    node_hosts: dict[str, str] = {}
    node_gpu_capacities: dict[str, int] = {}
    if verify_workers:
        live_nodes = [node for node in ray.nodes() if node["Alive"]]
        node_hosts = {node["NodeID"]: node["NodeManagerHostname"] for node in live_nodes}
        node_gpu_capacities = {node["NodeID"]: int(node["Resources"].get("GPU", 0)) for node in live_nodes}
    if tensor_parallel_size == 1 and pipeline_parallel_size == 1:
        distributed_executor_backend = "uni"
    elif use_mp_backend:
        distributed_executor_backend = "mp"
    else:
        distributed_executor_backend = "ray"
    data_parallel_backend = "mp"
    # The vLLM `mp` executor REQUIRES v1 multiprocessing to spawn its TP worker
    # subprocesses. The default vllm_v1_disable_multiproc=true sets
    # VLLM_ENABLE_V1_MULTIPROCESSING=0, which kills the mp executor's shm message
    # queue at warm-up ("RuntimeError: cancelled" in shm_broadcast.dequeue ->
    # EngineCore init fail). Force it off for the mp backend so the workers run.
    if use_mp_backend and vllm_v1_disable_multiproc:
        logger.info(
            "mp_backend: overriding vllm_v1_disable_multiproc=True -> False "
            "(the mp executor needs VLLM_ENABLE_V1_MULTIPROCESSING=1 to spawn TP workers)."
        )
        vllm_v1_disable_multiproc = False
    # With the mp executor the single actor must hold ALL tp_pp_size GPUs itself (vLLM forks
    # its workers locally). With ray/uni the actor holds the GPUs per the original logic.
    if use_mp_backend:
        num_gpus_per_actor = tp_pp_size
    else:
        num_gpus_per_actor = int(tensor_parallel_size == 1 and pipeline_parallel_size == 1)

    if use_hybrid_engine and tensor_parallel_size == 1 and pipeline_parallel_size == 1:
        # Every worker will use 0.2 GPU, so that we can schedule
        # inference and training workers on the same GPUs.
        num_gpus_per_actor = 0.2

    per_engine_gpu_count = tensor_parallel_size * pipeline_parallel_size * data_parallel_size
    # Keep each multi-GPU engine node-local to avoid cross-node TP decode hangs (#232).
    # TP=PP=1 expert parallelism can span nodes with every worker's placement checked.
    # Single-GPU engines share a PACK group to leave whole nodes available for training.
    per_engine_pgs: list = []
    owned_placement_groups: list = []
    needs_per_engine_pg = (
        use_per_engine_pg(
            use_hybrid_engine=use_hybrid_engine,
            use_mp_backend=use_mp_backend,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            data_parallel_size=data_parallel_size,
        )
        or require_verified_placements
    )
    if not use_hybrid_engine:
        if use_mp_backend:
            # mp executor: each (engine, DP-rank) is ONE Ray actor that itself
            # reserves the whole TP*PP GPU slice and forks its workers locally.
            # The actor's resource request is {GPU: tp_pp_size}, so it must land in
            # a single bundle that big — one {GPU: tp_pp_size} bundle per DP rank
            # (NOT tp_pp_size separate {GPU:1} bundles, which an actor requesting
            # tp_pp_size GPUs cannot fit into; Ray's _validate_resource_shape
            # requires a single actor to fit one bundle). One bundle per DP rank
            # keeps the bundle count == num actors so each gets a distinct index.
            # The {GPU: tp_pp_size} bundle is already atomic-per-node, so the mp path
            # was never affected by the cross-node split; keep its single PACK PG.
            bundles = [
                {"GPU": tp_pp_size, "CPU": tp_pp_size} for _ in range(num_inference_engines * data_parallel_size)
            ]
            shared_pg = placement_group(bundles, strategy="PACK")
            owned_placement_groups.append(shared_pg)
            get_ray_pg_ready_with_timeout(shared_pg, timeout=placement_group_timeout_seconds)
        elif needs_per_engine_pg:
            # Each engine owns a group. TP/PP and ordinary DP groups remain node-local.
            # PACK prefers fewer nodes but permits spanning them for verified TP=PP=1 EP replicas.
            for _ in range(num_inference_engines):
                pg = placement_group(
                    [{"GPU": 1, "CPU": 1} for _ in range(per_engine_gpu_count)],
                    strategy="PACK" if cross_node_ep else "STRICT_PACK",
                )
                per_engine_pgs.append(pg)
                owned_placement_groups.append(pg)
            for pg in per_engine_pgs:
                get_ray_pg_ready_with_timeout(pg, timeout=placement_group_timeout_seconds)
            # Keep `shared_pg` defined for downstream indexing; per-engine path
            # re-selects the engine's own PG in the loop.
            shared_pg = per_engine_pgs[0]
        else:
            # ray/uni backend, single-GPU engines (TP==PP==DP==1): ONE flat PACK PG over
            # all engine {GPU:1} bundles (the original pre-#232 behavior). PACK packs
            # densely -> fills whole nodes -> leaves whole nodes free for the
            # downstream policy/ref PACK PG.
            bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_inference_engines * per_engine_gpu_count)]
            shared_pg = placement_group(bundles, strategy="PACK")
            owned_placement_groups.append(shared_pg)
            get_ray_pg_ready_with_timeout(shared_pg, timeout=placement_group_timeout_seconds)

    if use_hybrid_engine:
        placement_mode = EnginePlacementMode.HYBRID
    elif use_mp_backend:
        placement_mode = EnginePlacementMode.MULTIPROCESSING
    elif per_engine_pgs:
        placement_mode = EnginePlacementMode.PER_ENGINE
    else:
        placement_mode = EnginePlacementMode.SHARED

    bundle_nodes: list[list[str]] = []
    if verify_workers:
        try:
            bundle_nodes = inference_bundle_nodes(
                per_engine_pgs,
                data_parallel_size=data_parallel_size,
                node_gpu_capacities=node_gpu_capacities,
                pipeline_parallel_size=pipeline_parallel_size,
                expert_parallel_size=expert_parallel_size,
            )
        except Exception:
            _release_node_local_gang([], per_engine_pgs)
            raise

    allocated_rendezvous_ports: set[int] = set()
    for i in range(num_inference_engines):
        if placement_mode is EnginePlacementMode.PER_ENGINE:
            engine_pg = per_engine_pgs[i]
        else:
            engine_pg = shared_pg

        dp_rank_bundle_indices = _dp_rank_bundle_indices(
            engine_index=i,
            data_parallel_size=data_parallel_size,
            tensor_pipeline_size=tp_pp_size,
            placement_mode=placement_mode,
            colocated_engine_bundles=colocated_engine_bundles,
        )
        if data_parallel_size > 1 and not cross_node_ep:
            _validate_node_local_dp_ranks(i, engine_pg, dp_rank_bundle_indices)

        rendezvous = _reserve_engine_rendezvous(
            engine_pg,
            dp_rank_bundle_indices[0],
            data_parallel_size,
            allocated_rendezvous_ports,
        )
        rendezvous_reservation = None
        if rendezvous is not None:
            data_parallel_address = rendezvous.address
            data_parallel_rpc_port, *data_parallel_master_ports = rendezvous.ports
            rendezvous_reservation = rendezvous.reservation

        if backend == "vllm":
            lora_kwargs = {
                "enable_lora": enable_lora,
                "max_lora_rank": max_lora_rank,
                "max_loras": max_loras,
                "fully_sharded_loras": fully_sharded_loras,
            }

            rope_engine_kwargs = {}
            if rope_scaling:
                rope_engine_kwargs["rope_scaling"] = rope_scaling
                if "max_model_len" not in engine_init_kwargs:
                    rope_engine_kwargs["max_model_len"] = resolve_engine_max_model_len(engine_init_kwargs, rope_scaling)
            if rope_theta is not None:
                rope_engine_kwargs["rope_theta"] = rope_theta

            # Launch one actor per DP rank
            for dp_rank in range(data_parallel_size):
                # TP*PP slice reserved for a single DP rank.
                base_dp_pg_index = dp_rank_bundle_indices[dp_rank]
                dp_rank_bundles = (
                    list(range(base_dp_pg_index, base_dp_pg_index + tp_pp_size)) if tp_pp_size > 1 else None
                )
                if use_mp_backend:
                    # The mp executor's single actor reserves the whole TP*PP GPU slice itself
                    # (vLLM forks its workers locally, no per-worker Ray actors). It must land
                    # in ONE bundle holding tp_pp_size GPUs, so the mp PACK PG (built above) is
                    # one {GPU: tp_pp_size} bundle per (engine, DP-rank) and this actor is pinned
                    # to its own dedicated bundle. The
                    # whole-slice bundle keeps all TP workers co-located on one node. bundle_indices
                    # stays None so vLLM does not attempt ray per-worker placement.
                    dp_rank_bundles = None
                    dp_rank_sched = PlacementGroupSchedulingStrategy(
                        placement_group=engine_pg,
                        placement_group_capture_child_tasks=True,
                        placement_group_bundle_index=base_dp_pg_index,
                    )
                else:
                    if use_hybrid_engine:
                        dp_rank_bundles = colocated_engine_bundles[i * data_parallel_size + dp_rank]
                    dp_rank_sched = PlacementGroupSchedulingStrategy(
                        placement_group=engine_pg,
                        placement_group_capture_child_tasks=True,
                        placement_group_bundle_index=base_dp_pg_index,
                    )

                dp_kwargs = (
                    {
                        "data_parallel_backend": data_parallel_backend,
                        "data_parallel_size": data_parallel_size,
                        "data_parallel_rank": dp_rank,
                        "data_parallel_address": data_parallel_address,
                        "data_parallel_rpc_port": data_parallel_rpc_port,
                    }
                    if data_parallel_size > 1
                    else {}
                )
                if data_parallel_size > 1:
                    dp_kwargs["data_parallel_master_ports"] = data_parallel_master_ports
                if dp_rank == 0 and rendezvous_reservation is not None:
                    dp_kwargs["rendezvous_port_reservation"] = rendezvous_reservation

                # The mp executor's TP workers exchange custom-all-reduce IPC handles
                # under the Ray-actor placement + remapped CUDA_VISIBLE_DEVICES; vLLM's
                # custom all-reduce fails there with a CUDA "invalid argument" at
                # custom_all_reduce.cuh (worker dies at warm-up). Disable it for mp so
                # NCCL handles the TP all-reduce (correctness-equal, slightly slower).
                # Do NOT clobber an explicit engine_init_kwargs override: the de-risk /
                # long-ctx configs already pass `disable_custom_all_reduce=true` through
                # `++generator.engine_init_kwargs.disable_custom_all_reduce`, and emitting
                # it here too makes both expand into .remote() -> "got multiple values for
                # keyword argument 'disable_custom_all_reduce'" (TypeError, crashes engine
                # creation before any PG/training; confirmed de-risk 925650-925654).
                mp_extra_kwargs = (
                    {"disable_custom_all_reduce": True}
                    if (use_mp_backend and "disable_custom_all_reduce" not in engine_init_kwargs)
                    else {}
                )

                # vLLM Decode Context Parallel (DCP): only forward the kwarg when enabled
                # (> 1). decode_context_parallel_size is a native vLLM EngineArgs field, so
                # it flows straight through **kwargs into vllm.LLM / AsyncEngineArgs with no
                # new vLLM call site. When == 1 the kwarg is ABSENT, making the engine init
                # byte-identical to today (G1). DCP rides the TP GPUs and does NOT touch
                # per_engine_gpu_count or the PACK PG bundle math above (G4).
                dcp_kwargs = (
                    {"decode_context_parallel_size": decode_context_parallel_size}
                    if decode_context_parallel_size > 1
                    else {}
                )
                attention_backend_kwargs = (
                    {"attention_backend": vllm_attention_backend} if vllm_attention_backend is not None else {}
                )

                # #232 FIX B: attach runtime_env only when FR vars are present, so
                # actor creation is byte-identical for every non-#232 run.
                engine_options = dict(
                    num_cpus=num_gpus_per_actor,
                    num_gpus=num_gpus_per_actor,
                    scheduling_strategy=dp_rank_sched,
                )
                if inference_engine_runtime_env is not None:
                    engine_options["runtime_env"] = inference_engine_runtime_env
                engine = AsyncVLLMRayActor.options(**engine_options).remote(
                    model=pretrain,
                    enforce_eager=enforce_eager,
                    worker_extension_cls="skyrl_train.inference_engines.vllm.vllm_engine.WorkerWrap",
                    tensor_parallel_size=tensor_parallel_size,
                    pipeline_parallel_size=pipeline_parallel_size,
                    enable_expert_parallel=expert_parallel_size > 1,
                    distributed_executor_backend=distributed_executor_backend,
                    seed=seed + i * data_parallel_size + dp_rank,
                    enable_prefix_caching=enable_prefix_caching,
                    dtype=model_dtype,
                    trust_remote_code=True,
                    vllm_v1_disable_multiproc=vllm_v1_disable_multiproc,
                    weight_sync_pause_policy=weight_sync_pause_policy,
                    gpu_memory_utilization=gpu_memory_utilization,
                    bundle_indices=dp_rank_bundles,
                    num_gpus=0.2 if use_hybrid_engine else 1,
                    enable_sleep_mode=inference_engine_enable_sleep,
                    noset_visible_devices=noset_visible_devices,
                    max_num_batched_tokens=max_num_batched_tokens,
                    max_num_seqs=max_num_seqs,
                    max_logprobs=max_logprobs,
                    **dp_kwargs,
                    **mp_extra_kwargs,
                    **dcp_kwargs,
                    **attention_backend_kwargs,
                    **vlm_engine_kwargs,
                    **engine_init_kwargs,
                    **lora_kwargs,
                    **rope_engine_kwargs,
                )
                inference_engine_actors.append(engine)
                weight_sync_relative_rank_offsets.append(i * per_engine_gpu_count)
        elif backend == "sglang":
            # NOTE: there is no async / sync engine distinction in SGLang

            sglang_base_index = dp_rank_bundle_indices[0]
            bundle_indices = None
            if per_engine_gpu_count > 1:
                if use_hybrid_engine:
                    bundle_indices = colocated_engine_bundles[i * data_parallel_size]
                else:
                    bundle_indices = list(range(sglang_base_index, sglang_base_index + per_engine_gpu_count))

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=engine_pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=sglang_base_index,
            )

            # NOTE(Charlie): We need `torch.cuda.is_available()` to be True to import SGLang. Otherwise, it requires
            # importing vllm. See https://github.com/sgl-project/sglang/blob/v0.4.8.post1/python/sglang/srt/layers/quantization/utils.py#L11-L17
            @ray.remote
            def get_sglang_engine():
                # A workaround to avoid importing vllm is to give this task a GPU.
                import os

                before_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                from skyrl_train.inference_engines.sglang.sglang_engine import SGLangRayActor

                os.environ["CUDA_VISIBLE_DEVICES"] = before_cuda_visible_devices

                actor_class = SGLangRayActor
                engine = actor_class.options(
                    num_cpus=num_gpus_per_actor,
                    num_gpus=num_gpus_per_actor,
                    scheduling_strategy=scheduling_strategy,
                ).remote(
                    model_path=pretrain,
                    tp_size=tensor_parallel_size,
                    mem_fraction_static=gpu_memory_utilization,
                    random_seed=seed + i,
                    disable_radix_cache=not enable_prefix_caching,
                    dtype=model_dtype,
                    trust_remote_code=True,
                    max_prefill_tokens=max_num_batched_tokens,
                    max_running_requests=max_num_seqs,
                    # Borrowed from veRL's SGLang rollout
                    mm_attention_backend="fa3",
                    attention_backend="fa3",
                    enable_memory_saver=inference_engine_enable_sleep,
                    # Will be popped before instantiating sgl.Engine
                    distributed_executor_backend=distributed_executor_backend,
                    noset_visible_devices=noset_visible_devices,
                    bundle_indices=bundle_indices,
                    num_gpus=0.2 if use_hybrid_engine else 1,
                    tokenizer=tokenizer,
                    **engine_init_kwargs,
                )
                return engine

            engine = ray.get(get_sglang_engine.remote())

            inference_engine_actors.append(engine)
            weight_sync_relative_rank_offsets.append(i * per_engine_gpu_count)

    try:
        engines = [
            RayWrappedInferenceEngine(
                actor_handle,
                weight_sync_relative_rank_offset=rank_offset,
            )
            for actor_handle, rank_offset in zip(
                inference_engine_actors,
                weight_sync_relative_rank_offsets,
                strict=True,
            )
        ]

        if backend == "vllm" and (tensor_parallel_size > 1 or pipeline_parallel_size > 1):
            numa_refs = [engine.inference_engine_actor.initialize_worker_numa_affinity.remote() for engine in engines]
            wait_for_inference_engine_startup(
                numa_refs,
                [engine.inference_engine_actor for engine in engines],
                timeout_seconds=engine_init_timeout_seconds,
                kill_on_failure=False,
            )

        # Readiness gate (DISAGGREGATED-mode init-deadlock fix): block until every engine
        # actor has finished loading its model (weights + CUDA-graph capture) BEFORE the
        # trainer opens the weight-sync NCCL group in init_weight_sync_state. In COLOCATED
        # mode the bounded sleep barrier below already forces this wait; in
        # DISAGGREGATED mode (inference_engine_enable_sleep=False) nothing otherwise waits,
        # so a slow-loading engine (e.g. Qwen3.6-35B-A3B MoE) is still inside __init__ when
        # the policy ranks post the default-group barrier (worker.py) -> the barrier ALLREDUCE
        # hits the 20-min SKYRL_WORKER_NCCL_TIMEOUT_IN_S watchdog and SIGABRTs the job
        # (init_weight_sync_state failed at Ray boundary). 30B loads fast enough to win the
        # race; 35B loses it deterministically. report_engine_hosts fans a collective_rpc
        # across every engine TP/EP worker, so the ray.get returns only once all workers are
        # up (model fully loaded) -> closes the race for any model size / load time. It is a
        # read-only probe (returns hostnames), no side effects. vLLM-only (collective_rpc);
        # the colocated sleep barrier still covers the sglang/colocated paths unchanged.
        startup_refs = []
        if verify_workers:
            # The placement report is also the readiness barrier.
            startup_refs = [engine.inference_engine_actor.report_engine_placement.remote() for engine in engines]
        elif not inference_engine_enable_sleep and backend == "vllm":
            startup_refs = [engine.inference_engine_actor.report_engine_hosts.remote() for engine in engines]

        if inference_engine_enable_sleep:
            if backend == "vllm":
                # NOTE(shu): set to 1 for LoRA
                sleep_level = 1 if enable_lora else sleep_level
                sleep_refs = [engine.inference_engine_actor.sleep.remote(level=sleep_level) for engine in engines]
            elif backend == "sglang":
                # NOTE(Charlie): we always need to sync weights after waking up: https://github.com/sgl-project/sglang/issues/7939
                assert sleep_level == 2, "SGLang always discards weights, so sleep_level is not applicable."
                sleep_refs = [engine.inference_engine_actor.sleep.remote() for engine in engines]
            startup_refs = sleep_refs

        if startup_refs:
            startup_results = wait_for_inference_engine_startup(
                startup_refs,
                [engine.inference_engine_actor for engine in engines],
                timeout_seconds=engine_init_timeout_seconds,
                kill_on_failure=False,
            )
            if verify_workers:
                placements = verified_inference_replica_placements(
                    startup_results,
                    bundle_nodes=bundle_nodes,
                    node_hosts=node_hosts,
                    relative_rank_offsets=weight_sync_relative_rank_offsets,
                    data_parallel_size=data_parallel_size,
                    expert_parallel_size=expert_parallel_size,
                    pipeline_parallel_size=pipeline_parallel_size,
                )
                for index, engine in enumerate(engines):
                    engine.worker_placements = placements[
                        index * pipeline_parallel_size : (index + 1) * pipeline_parallel_size
                    ]
                    for placement in engine.worker_placements:
                        logger.info("Verified inference replica placement: {}", placement)

        populate_engine_max_model_lens(
            engines,
            timeout_seconds=engine_init_timeout_seconds,
            kill_on_failure=False,
        )
        return engines
    except BaseException:
        _cleanup_failed_engine_startup(inference_engine_actors, owned_placement_groups)
        raise


def wait_for_inference_engine_startup(
    startup_refs: list[ray.ObjectRef],
    actor_handles: list[ActorHandle],
    *,
    timeout_seconds: float,
    kill_on_failure: bool = True,
) -> list[Any]:
    """Wait for every startup reply, killing the actors on failure unless the caller owns teardown."""
    deadline = time.monotonic() + timeout_seconds
    pending = list(startup_refs)
    try:
        while pending:
            ready, pending = ray.wait(
                pending,
                num_returns=1,
                timeout=max(0.0, deadline - time.monotonic()),
                fetch_local=False,
            )
            if not ready:
                pending_set = set(pending)
                pending_indices = [index for index, ref in enumerate(startup_refs) if ref in pending_set]
                raise TimeoutError(
                    f"inference engine startup timed out after {timeout_seconds:g} seconds; "
                    f"pending engine actors: {pending_indices}"
                )
            ray.get(ready)
    except Exception:
        if kill_on_failure:
            for actor in actor_handles:
                ray.kill(actor)
        raise
    return ray.get(startup_refs)
