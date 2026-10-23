import importlib
import ipaddress
import os
import time
import sys
import logging
import math
import socket

import ray
import torch
from loguru import logger
from rigging.secrets import SECRETS_DIR_ENV, require_secret
from omegaconf import DictConfig, OmegaConf
from ray.util.placement_group import (
    placement_group,
    PlacementGroupSchedulingStrategy,
    PlacementGroup,
    placement_group_table,
)

from skyrl_train.batch_invariant import BATCH_INVARIANT_NCCL_ENV
from skyrl_train.config.callbacks import validate_hf_export_config
from skyrl_train.config.utils import generation_context_limit
from skyrl_train.config.query_bias import resolve_grug_query_bias_update
from skyrl_train.config.weight_sync_pause import resolve_weight_sync_pause_policy
from skyrl_train.config.behavior_logprobs import configure_behavior_logprob_sampling
from skyrl_train.config.ftpo import ftpo_config
from skyrl_train.config.mismatch_probe import validate_mismatch_probe_config
from skyrl_train.config.objective_spec import rollout_logprobs_required, validate_objective
from skyrl_train.debug_mode import apply_debug_mode
from skyrl_train.rollouts.training_record import ReasoningPenaltyTarget, ReasoningSpanSettings
from skyrl_train.trajectory_runners.trajectory_reward_shaping import parse_trajectory_reward_shaping_config
from skyrl_train.trajectory_runners.trajectory_retention_config import parse_trajectory_retention_config
from skyrl_train.numa_policy import NUMA_AFFINITY_ENV
from skyrl_train.env_vars import (
    DEBUG_ARTIFACT_DIR_ENV,
    DEBUG_MODE_ENV,
    VLLM_BATCH_INVARIANT_ENV,
    EnvVarManager,
    EnvVarScope,
)
from skyrl_train.group_admission import resolve_group_advantage_invariant
from skyrl_train.trajectory_selection import optimization_samples_per_prompt, trajectory_selector_from_config
from skyrl_train.dynamic_sampling import resolve_dynamic_sampling_criteria
from marinskyrl.runtime_options import reference_model_required
from marinskyrl.rollout_grading import validate_nemotron_ultra_grading
from marinskyrl.process_diagnostics import initialize_process_diagnostics
from marinskyrl.distillation import (
    compile_distillation_plan_from_config,
    validate_distillation_runtime_support,
    validate_generation_logprobs,
)
from marinskyrl.inference_placement import validate_expert_block_transport
from marinskyrl.runtime_options import GDNBackend, R3Transport

from .constants import DEFAULT_RAY_PLACEMENT_GROUP_TIMEOUT_SECONDS
from .algorithm_registry import (
    AdvantageEstimatorRegistry,
    NoGroupAdvantage,
    PolicyLossRegistry,
    sync_registries,
)
from .logging_utils import format_exception_text
from .nccl_environment import worker_nccl_environment
from .placement_geometry import validate_colocated_engine_geometry


def moe_router_replay_requested(cfg: DictConfig, role: str = "policy") -> bool:
    return bool(cfg.trainer[role].megatron_config.get("moe_router_replay", False))


def policy_strict_spread_eligible(cfg: DictConfig) -> bool:
    """Whether to reserve policy nodes before inference engines start.

    Pure (Ray-free) predicate so it is unit-testable. Eligible only when ALL of:
      - `trainer.placement.policy_strict_spread_pg` is enabled (opt-in; default
        false, so every existing run is byte-for-byte unchanged),
      - the run is disaggregated (`colocate_all=false`), and
      - a reference model, when used, shares the policy placement group.
    """
    placement = cfg.trainer.placement
    if not bool(getattr(placement, "policy_strict_spread_pg", False)):
        return False
    if placement.colocate_all:
        return False
    algo = cfg.trainer.algorithm
    use_ref_model = reference_model_required(algo)
    return not use_ref_model or bool(placement.colocate_policy_ref)


def resolve_pinned_local_rank(
    *,
    noset_visible_devices: bool,
    cuda_visible_devices,
    ray_gpu_ids,
    launcher_local_rank: int,
    device_count: int,
    pin_to_ray_gpu_id: bool,
) -> str:
    """Pure decision for the LOCAL_RANK (== torch.cuda.set_device index) of a
    distributed actor. Extracted from DistributedTorchRayActor.__init__ so the
    GH200 device-pinning logic is unit-testable without importing Ray/torch
    worker deps. All inputs are plain values; returns the LOCAL_RANK string.

    Branches (see __init__ docstring for the full GH200 narrative):
      1. NOSET set            -> ray_gpu_ids[0]      (Ray doesn't mask CVD; pin physical id)
      2. CVD masked to 1 dev  -> "0"                 (a3 venv-Ray path; byte-identical)
      3. pin_to_ray_gpu_id    -> ray_gpu_ids[0] if it's a valid device index,
                                 else launcher_local_rank, else ray_gpu_ids[0]/"0"
                                 (per-GPU {GPU:1} bundle: get_gpu_ids() is reliable)
      4. otherwise            -> launcher_local_rank if in range, else ray_gpu_ids[0]
                                 (whole-node bundle: positional rank%num_gpus_per_node)
    """
    if noset_visible_devices:
        return str(ray_gpu_ids[0])

    cvd = cuda_visible_devices
    masked_to_single = cvd is not None and len([d for d in cvd.split(",") if d != ""]) == 1
    if masked_to_single:
        return "0"

    if pin_to_ray_gpu_id:
        if ray_gpu_ids and 0 <= int(ray_gpu_ids[0]) < max(device_count, 1):
            return str(int(ray_gpu_ids[0]))
        if device_count > 0 and 0 <= launcher_local_rank < device_count:
            return str(launcher_local_rank)
        return str(ray_gpu_ids[0]) if ray_gpu_ids else "0"

    if device_count > 0 and 0 <= launcher_local_rank < device_count:
        return str(launcher_local_rank)
    return str(ray_gpu_ids[0]) if ray_gpu_ids else "0"


def resolve_actor_cuda_env(
    *,
    noset_visible_devices: bool,
    cuda_visible_devices,
    ray_gpu_ids,
) -> dict:
    """Pure decision for the per-actor CUDA env that DETERMINISTICALLY pins each
    actor to its single Ray-assigned physical GPU, independent of positional /
    LOCAL_RANK ordering and of how (or whether) Ray masked CUDA_VISIBLE_DEVICES.

    This is the deterministic device-pin used when
    ``trainer.placement.policy_force_cvd_mask`` is enabled (opt-in, default
    off). It is the strongest form of the per-GPU-bundle fix: rather than rely
    on ``set_device(LOCAL_RANK)`` resolving the right physical GPU (which can
    silently collapse onto GPU 0 when CVD is left unmasked on the SIF Ray and
    ``ray.get_gpu_ids()`` collides), it MASKS each actor to exactly one visible
    device and forces a stable PCI ordering BEFORE any CUDA / device-mesh init.
    With one visible device, ``set_device(0)`` and ``init_device_mesh``
    can only land on that physical GPU.

    Returns a dict of env vars to apply (only the keys that should be set):
      - ``CUDA_DEVICE_ORDER`` = "PCI_BUS_ID" (so the logical index agrees with
        the sysfs/NUMA PCI ordering; GH200's default FASTEST_FIRST order ≠ PCI
        order, which is what makes a positional index bind the wrong socket).
      - ``CUDA_VISIBLE_DEVICES`` = the single physical id this actor owns, ONLY
        when Ray left it unmasked / multi-device. If Ray already masked CVD to a
        single device (the a3 venv path / the whole-GPU-request path), we leave
        CVD untouched (re-masking to a *physical* id would be wrong, because the
        physical id is not addressable inside an already-masked view).
      - ``LOCAL_RANK`` = "0" — after masking there is exactly one visible
        device, whose logical index is 0.

    Branches (mirror resolve_pinned_local_rank's input space):
      1. NOSET set            -> CVD unmasked by design; mask to ray_gpu_ids[0].
      2. CVD masked to 1 dev  -> already isolated; leave CVD, LOCAL_RANK "0".
      3. CVD unset / multi-dev-> mask to ray_gpu_ids[0] (the collision case).
    """
    env = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}

    cvd = cuda_visible_devices
    masked_to_single = cvd is not None and len([d for d in cvd.split(",") if d != ""]) == 1

    if masked_to_single and not noset_visible_devices:
        # Ray already isolated this actor to one device; index 0 within the mask.
        env["LOCAL_RANK"] = "0"
        return env

    # Unmasked (NOSET) or multi-device view: mask to this actor's own physical
    # GPU. ray_gpu_ids[0] is the physical id of the actor's {GPU:1} bundle.
    if ray_gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = str(int(ray_gpu_ids[0]))
        env["LOCAL_RANK"] = "0"
    return env


def policy_force_cvd_mask_enabled(cfg: DictConfig) -> bool:
    """Whether to apply the deterministic forced-CVD-mask device pin.

    Pure (Ray-free) predicate. Opt-in sub-flag of the per-GPU-bundle policy PG;
    default false → unchanged set_device(LOCAL_RANK) pinning. Only meaningful
    alongside policy_per_gpu_bundles (per actor owns one {GPU:1} bundle, so
    ray.get_gpu_ids()[0] is its distinct physical id).
    """
    return bool(getattr(cfg.trainer.placement, "policy_force_cvd_mask", False))


def policy_per_gpu_bundles_enabled(cfg: DictConfig) -> bool:
    """Whether the dedicated policy PG should use per-GPU {GPU:1} bundles.

    Pure (Ray-free) predicate. Only meaningful when the dedicated STRICT_SPREAD
    policy PG is itself eligible (see `policy_strict_spread_eligible`); this just
    reads the opt-in sub-flag. Default false → legacy whole-node {GPU:4} bundles.
    """
    return bool(getattr(cfg.trainer.placement, "policy_per_gpu_bundles", False))


def policy_spread_bundles(cfg: DictConfig):
    """The bundle list for the dedicated policy PG.

    Two shapes, selected by `policy_per_gpu_bundles`:

    - whole-node (default): one {GPU:n,CPU:n} bundle per policy node, each
      claiming all of that node's GPUs. len(bundles) == policy_num_nodes.
    - per-GPU (opt-in): one {GPU:1,CPU:1} bundle per policy GPU, i.e.
      num_nodes * num_gpus_per_node bundles. len(bundles) == world_size, which
      engages the reliable get_reordered_bundle_indices() path and gives each
      actor a 1-GPU bundle so ray.get_gpu_ids()[0] resolves a distinct physical
      GPU (the GH200 device-collision fix).

    Pure (Ray-free) so the bundle count / shape is unit-testable.
    """
    num_nodes = cfg.trainer.placement.policy_num_nodes
    num_gpus_per_node = cfg.trainer.placement.policy_num_gpus_per_node
    if policy_per_gpu_bundles_enabled(cfg):
        return [{"GPU": 1, "CPU": 1} for _ in range(num_nodes * num_gpus_per_node)]
    return [{"GPU": num_gpus_per_node, "CPU": num_gpus_per_node} for _ in range(num_nodes)]


def use_per_engine_pg(
    *,
    use_hybrid_engine: bool,
    use_mp_backend: bool,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    data_parallel_size: int,
) -> bool:
    """Return whether placement needs a separate group per multi-GPU engine.

    The engine factory uses PACK for supported cross-node EP. Single-GPU engines
    share a PACK group; hybrid placement supplies its own group, and the mp
    backend reserves each tensor/pipeline slice in one node-atomic bundle.
    """
    if use_hybrid_engine or use_mp_backend:
        return False
    return (tensor_parallel_size * pipeline_parallel_size * data_parallel_size) > 1


class Timer:
    def __init__(self, message, update_dict=None, *, log_events: bool = True):
        self.message = message
        self.update_dict = update_dict
        self.log_events = log_events
        self._duration: float | None = None

    @property
    def duration(self) -> float:
        if self._duration is None:
            raise RuntimeError("Timer duration is only available after its context exits")
        return self._duration

    @property
    def elapsed(self) -> float:
        """Return the elapsed duration without ending an active timer."""
        if self._duration is not None:
            return self._duration
        return time.monotonic() - self.start_time

    def __enter__(self):
        self.start_time = time.monotonic()
        if self.log_events:
            logger.opt(depth=1).info(f"Started: '{self.message}'")
        return self

    def _finish(self, exc_type) -> None:
        duration = time.monotonic() - self.start_time
        self._duration = duration
        if self.log_events:
            log = logger.opt(depth=2).info if exc_type is None else logger.opt(depth=2).error
            outcome = "Finished" if exc_type is None else "Failed"
            log(f"{outcome}: '{self.message}', time cost: {duration:.2f}s")
        if self.update_dict is not None:
            self.update_dict[self.message] = self.update_dict.get(self.message, 0.0) + duration

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._finish(exc_type)

    async def __aenter__(self):
        self.start_time = time.monotonic()
        if self.log_events:
            logger.opt(depth=1).info(f"Started: '{self.message}'")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._finish(exc_type)


def get_system_memory_metrics() -> dict:
    """
    Get system RAM metrics for tracking memory usage over time.

    Returns a dict with memory metrics in GB, suitable for logging to wandb/mlflow/etc.
    Returns empty dict if psutil is not available.
    """
    try:
        import psutil

        # Get system-wide memory info
        mem = psutil.virtual_memory()

        # Get current process memory info
        process = psutil.Process()
        process_mem = process.memory_info()

        return {
            "system/ram_used_gb": mem.used / (1024**3),
            "system/ram_available_gb": mem.available / (1024**3),
            "system/ram_total_gb": mem.total / (1024**3),
            "system/ram_percent": mem.percent,
            "system/process_rss_gb": process_mem.rss / (1024**3),
            "system/process_vms_gb": process_mem.vms / (1024**3),
        }
    except ImportError:
        logger.warning("psutil not installed, skipping system memory metrics")
        return {}
    except Exception as e:
        logger.warning(f"Failed to get system memory metrics: {e}")
        return {}


def validate_batch_sizes(cfg: DictConfig):
    """
    Validate configured batch sizes.

    Explanation of how batching operates:
    1. Each prompt creates `optimization_samples_per_prompt` learner rows after trajectory selection.
    2. During training, these rows are split across data parallel (DP) workers.
    3. Mini batches are normalized using that same post-selection row count.
    4. Per-gpu train batch size must be divisble by per-gpu mini batch size, otherwise the last mini batch will be incomplete.
    5. Per-gpu mini batch size must be divisible by per-gpu micro batch size, otherwise the last micro batch will be incomplete.
    """
    assert cfg.trainer.train_batch_size >= cfg.trainer.policy_mini_batch_size
    assert cfg.trainer.policy_mini_batch_size > 0, "policy_mini_batch_size must be greater than 0"
    assert cfg.trainer.micro_train_batch_size_per_gpu > 0, "micro_train_batch_size_per_gpu must be greater than 0"
    assert cfg.trainer.micro_forward_batch_size_per_gpu > 0, "micro_forward_batch_size_per_gpu must be greater than 0"

    # Validate policy mini batch size
    policy_world_size = cfg.trainer.placement.policy_num_nodes * cfg.trainer.placement.policy_num_gpus_per_node

    pp = cfg.trainer.policy.megatron_config.pipeline_model_parallel_size
    cp = cfg.trainer.policy.megatron_config.context_parallel_size
    tp = cfg.trainer.policy.megatron_config.tensor_model_parallel_size
    assert policy_world_size % (pp * cp * tp) == 0, (
        f"policy_world_size {policy_world_size} should be divisible by (pp * cp * tp) {pp * cp * tp}. "
        "This ensures that the data parallel size is an integer."
    )
    policy_dp_size = policy_world_size // (pp * cp * tp)

    assert cfg.trainer.train_batch_size % cfg.trainer.policy_mini_batch_size == 0, (
        f"train_batch_size {cfg.trainer.train_batch_size} should be divisible by policy_mini_batch_size {cfg.trainer.policy_mini_batch_size}"
    )
    optimization_group_size = optimization_samples_per_prompt(cfg)
    policy_mini_batch_size_per_gpu = cfg.trainer.policy_mini_batch_size * optimization_group_size // policy_dp_size
    assert policy_mini_batch_size_per_gpu > 0, (
        f"Invalid policy_mini_batch_size_per_gpu: {policy_mini_batch_size_per_gpu}. "
        f"mini_batch_size={cfg.trainer.policy_mini_batch_size}, "
        f"optimization_samples_per_prompt={optimization_group_size}, "
        f"dp_size={policy_dp_size}"
    )
    assert policy_mini_batch_size_per_gpu % cfg.trainer.micro_train_batch_size_per_gpu == 0, (
        f"normalized policy_mini_batch_size_per_gpu {policy_mini_batch_size_per_gpu} should be divisible by micro_train_batch_size_per_gpu {cfg.trainer.micro_train_batch_size_per_gpu}"
    )
    assert policy_mini_batch_size_per_gpu // cfg.trainer.micro_train_batch_size_per_gpu > 0, (
        f"normalized policy_mini_batch_size_per_gpu {policy_mini_batch_size_per_gpu} should be larger than micro_train_batch_size_per_gpu {cfg.trainer.micro_train_batch_size_per_gpu}"
    )
    policy_train_batch_size_per_gpu = cfg.trainer.train_batch_size * optimization_group_size // policy_dp_size

    # `train_batch_size_per_gpu` should be divisible by `policy_mini_batch_size_per_gpu`
    assert policy_train_batch_size_per_gpu % policy_mini_batch_size_per_gpu == 0, (
        f"normalized policy_train_batch_size_per_gpu (train_batch_size * optimization_samples_per_prompt // policy_dp_size) {policy_train_batch_size_per_gpu} should be divisible by policy_mini_batch_size_per_gpu (policy_mini_batch_size * optimization_samples_per_prompt // policy_dp_size) {policy_mini_batch_size_per_gpu}"
    )

    # Validate training batch size is larger than the least common multiple of the DP sizes of policy (and ref if used).
    lcm_dp_size = policy_dp_size

    use_ref_model = reference_model_required(cfg.trainer.algorithm)
    if use_ref_model:
        ref_world_size = cfg.trainer.placement.ref_num_nodes * cfg.trainer.placement.ref_num_gpus_per_node
        pp = cfg.trainer.ref.megatron_config.pipeline_model_parallel_size
        cp = cfg.trainer.ref.megatron_config.context_parallel_size
        tp = cfg.trainer.ref.megatron_config.tensor_model_parallel_size
        assert ref_world_size % (pp * cp * tp) == 0, (
            f"ref_world_size {ref_world_size} should be divisible by (pp * cp * tp) {pp * cp * tp}. "
            "This ensures that the data parallel size is an integer."
        )
        ref_dp_size = ref_world_size // (pp * cp * tp)
        lcm_dp_size = math.lcm(lcm_dp_size, ref_dp_size)

    assert cfg.trainer.train_batch_size >= lcm_dp_size, (
        f"train_batch_size ({cfg.trainer.train_batch_size}) should be larger than or equal to the least common multiple of the data parallel sizes of the enabled models: "
        f"policy_dp_size={policy_dp_size}, "
        f"ref_dp_size={ref_dp_size if use_ref_model else 'None'}, "
        f"lcm_dp_size={lcm_dp_size}"
    )


def validate_megatron_cfg(cfg: DictConfig):
    # not yet supported + tested features
    assert cfg.generator.weight_sync_backend == "nccl", "only nccl is supported for megatron weight sync"
    assert cfg.generator.backend == "vllm", "only vllm is supported for with megatron"
    assert cfg.trainer.critic.model.path is None, "only GRPO training is currently supported for megatron"

    if cfg.trainer.flash_attn:
        importlib.import_module("flash_attn")

    worker_configs = [(cfg.trainer.policy, "policy"), (cfg.trainer.ref, "ref")]
    for config, worker_type in worker_configs:
        # context, expert, and expert tensor parallel are not yet supported for megatron
        if config.megatron_config.context_parallel_size > 1:
            assert cfg.trainer.use_sample_packing, "context parallel is only supported with sample packing"
        # check that sequence parallel is not configured outside of megatron
        assert config.sequence_parallel_size == 1, (
            f"found {worker_type}.sequence_parallel_size={config.sequence_parallel_size}, ulysses style sequence parallel is not supported for megatron"
        )
        # The fused router bypasses the replay hook entirely; catch it at config
        # validation on the launcher CPU instead of at model build on the GPUs.
        if config.megatron_config.get("moe_router_replay", False):
            for kwargs_name in ("transformer_config_kwargs", "model_config_kwargs"):
                for key_path, value in _iter_config_kwargs(config.megatron_config.get(kwargs_name, {})):
                    if key_path[-1] == "moe_router_fusion" and value:
                        raise ValueError(
                            f"trainer.{worker_type}.megatron_config.{kwargs_name}.{'.'.join(key_path)} "
                            "must stay false with moe_router_replay: the fused router bypasses the replay hook"
                        )


def _iter_config_kwargs(node):
    """Yield ``(key_path, value)`` for every leaf in a (possibly nested) kwargs mapping."""
    if not node:
        return
    for key, value in node.items():
        if isinstance(value, dict):
            for path, leaf in _iter_config_kwargs(value):
                yield (key, *path), leaf
        else:
            yield (key,), value


def validate_cfg(cfg: DictConfig):
    validate_mismatch_probe_config(cfg)
    if cfg.trainer.strategy != "megatron":
        raise ValueError(f"Unsupported training strategy: {cfg.trainer.strategy}")
    if cfg.trainer.critic.model.path:
        raise ValueError("Megatron does not support a critic worker")
    distillation_plan = compile_distillation_plan_from_config(cfg)
    validate_distillation_runtime_support(distillation_plan)
    validate_nemotron_ultra_grading(cfg, distillation_plan)
    trajectory_selector = trajectory_selector_from_config(cfg)
    if trajectory_selector is not None:
        if cfg.trainer.step_wise_training:
            raise ValueError("best-of-N selection does not support step-wise training")
    resolve_dynamic_sampling_criteria(
        cfg.trainer.algorithm.dynamic_sampling.informative_on,
        float(cfg.trainer.algorithm.dynamic_sampling.min_reward_std),
        cfg.trainer.algorithm.dynamic_sampling.max_mean_reward,
    )
    runtime_values = {
        "trainer.distributed.placement_group_timeout_seconds": cfg.trainer.distributed.placement_group_timeout_seconds,
        "trainer.distributed.worker_collective_timeout_seconds": cfg.trainer.distributed.worker_collective_timeout_seconds,
        "trainer.policy.host_memory_monitor.interval_seconds": cfg.trainer.policy.host_memory_monitor.interval_seconds,
        "trainer.model_load_retry.backoff_base_seconds": cfg.trainer.model_load_retry.backoff_base_seconds,
        "trainer.model_load_retry.backoff_cap_seconds": cfg.trainer.model_load_retry.backoff_cap_seconds,
        "trainer.progress.min_interval_seconds": cfg.trainer.progress.min_interval_seconds,
        "trainer.progress.heartbeat_seconds": cfg.trainer.progress.heartbeat_seconds,
        "trainer.progress.percent_step": cfg.trainer.progress.percent_step,
        "trainer.progress.count_step": cfg.trainer.progress.count_step,
        "generator.r3_dispatch_put_timeout_seconds": cfg.generator.r3_dispatch_put_timeout_seconds,
        "trajectory_runner.rollout_workers.num_workers": cfg.trajectory_runner.rollout_workers.num_workers,
        "trajectory_runner.rollout_workers.cpus_per_worker": cfg.trajectory_runner.rollout_workers.cpus_per_worker,
        "trajectory_runner.rollout_workers.executor_threads": cfg.trajectory_runner.rollout_workers.executor_threads,
        "trajectory_runner.rollout_workers.progress_timeout_seconds": (
            cfg.trajectory_runner.rollout_workers.progress_timeout_seconds
        ),
    }
    for path, value in runtime_values.items():
        if value <= 0:
            raise ValueError(f"{path} must be positive; got {value}")
    start_interval = cfg.trajectory_runner.rollout_workers.start_interval_seconds
    if start_interval < 0:
        raise ValueError(
            f"trajectory_runner.rollout_workers.start_interval_seconds must be non-negative; got {start_interval}"
        )
    if cfg.trainer.model_load_retry.max_retries < 0:
        raise ValueError(
            f"trainer.model_load_retry.max_retries must be non-negative; got {cfg.trainer.model_load_retry.max_retries}"
        )
    group_admission_stall_timeout = cfg.trainer.algorithm.group_admission.stall_timeout
    if group_admission_stall_timeout is not None and group_admission_stall_timeout <= 0:
        raise ValueError(
            "trainer.algorithm.group_admission.stall_timeout must be positive when set; got "
            f"{group_admission_stall_timeout}"
        )
    if cfg.trainer.algorithm.tis_lcs_alert_threshold < 0:
        raise ValueError(
            "trainer.algorithm.tis_lcs_alert_threshold must be non-negative; "
            f"got {cfg.trainer.algorithm.tis_lcs_alert_threshold}"
        )
    if cfg.trainer.progress.mode not in {"auto", "tqdm", "logging"}:
        raise ValueError(f"trainer.progress.mode must be one of auto, tqdm, logging; got {cfg.trainer.progress.mode!r}")
    if cfg.generator.r3_transport not in set(R3Transport):
        raise ValueError(
            f"generator.r3_transport must be one of by_value, resident, decentral; got {cfg.generator.r3_transport!r}"
        )
    if cfg.generator.gdn_backend not in set(GDNBackend):
        raise ValueError(f"generator.gdn_backend must be one of torch, flashqla; got {cfg.generator.gdn_backend!r}")
    available_policy_losses = PolicyLossRegistry.list_available()
    assert available_policy_losses != [], "Policy loss registry is not populated."
    assert cfg.trainer.algorithm.policy_loss_type in available_policy_losses, (
        f"invalid policy_loss_type: {cfg.trainer.algorithm.policy_loss_type}. Must be one of {available_policy_losses}"
    )
    spec = PolicyLossRegistry.spec(cfg.trainer.algorithm.policy_loss_type)
    validate_objective(cfg, loss_spec=spec)
    resolve_weight_sync_pause_policy(cfg.generator)
    validate_generator_cfg(cfg)
    validate_batch_invariant_config(cfg)
    validate_hf_export_config(cfg)
    try:
        resolve_grug_query_bias_update(cfg.trainer.policy)
    except ValueError as error:
        raise AssertionError(str(error)) from error

    # if advantage estimator is GAE, then critic path should be provided
    if cfg.trainer.algorithm.advantage_estimator == "gae":
        assert cfg.trainer.critic.model.path, (
            "`trainer.critic.model.path` should be provided for PPO training, got `None`"
        )

    assert not (cfg.trainer.algorithm.use_kl_in_reward and cfg.trainer.algorithm.use_kl_loss), (
        "use_kl_in_reward and use_kl_loss should be mutually exclusive"
    )

    use_ref_model = reference_model_required(cfg.trainer.algorithm)

    validate_batch_sizes(cfg)

    if cfg.trainer.max_ckpts_to_keep == 0:
        raise ValueError(
            "`max_ckpts_to_keep` must be greater than 0 to keep the last N checkpoints or negative to keep all checkpoints"
        )

    available_advantage_estimators = AdvantageEstimatorRegistry.list_available()
    assert cfg.trainer.algorithm.advantage_estimator in available_advantage_estimators, (
        f"invalid advantage_estimator: {cfg.trainer.algorithm.advantage_estimator}. Must be one of {available_advantage_estimators}"
    )
    if trajectory_selector is not None and not isinstance(
        AdvantageEstimatorRegistry.group_contract(cfg.trainer.algorithm.advantage_estimator),
        NoGroupAdvantage,
    ):
        raise ValueError("best-of-N selection requires a no-group advantage estimator")

    # add field to algorithm config needed for loss functions
    # create a new config to make it modifiable
    algorithm_config = OmegaConf.create(cfg.trainer.algorithm)
    group_advantage = resolve_group_advantage_invariant(
        advantage_estimator=str(algorithm_config.advantage_estimator),
        physical_group_size=optimization_samples_per_prompt(cfg),
        minimum_group_size=algorithm_config.group_advantage_min_size,
    )
    algorithm_config.resolved_group_advantage = group_advantage.to_config()
    # Sequence-normalized losses use the same bound as model generation.
    algorithm_config.max_seq_len = generation_context_limit(cfg.generator)

    cfg.trainer.algorithm = algorithm_config

    behavior_logprobs_required = rollout_logprobs_required(cfg.trainer.algorithm, loss_spec=spec) or bool(
        cfg.trainer.mismatch_probe.get("enabled", False)
    )
    if behavior_logprobs_required:
        if cfg.generator.sampling_params.logprobs is None:
            logger.warning(
                "The selected objective requires rollout logprobs; setting generator.sampling_params.logprobs=0."
            )
            cfg.generator.sampling_params.logprobs = 0
        if cfg.generator.backend == "sglang":
            raise NotImplementedError("Behavior-logprob objectives require the vLLM generator backend")
        configure_behavior_logprob_sampling(cfg.generator)

    if cfg.trainer.policy.model.lora.rank > 0:
        raise ValueError("Megatron training does not support LoRA")

    if cfg.trainer.micro_forward_batch_size_per_gpu != cfg.trainer.micro_train_batch_size_per_gpu:
        raise ValueError(
            "Megatron recomputes old log-probs with micro_forward_batch_size_per_gpu="
            f"{cfg.trainer.micro_forward_batch_size_per_gpu} but trains with micro_train_batch_size_per_gpu="
            f"{cfg.trainer.micro_train_batch_size_per_gpu}; cuBLAS selects kernels per shape, so the two "
            "passes disagree and PPO clipping acts on kernel noise. Use equal sizes."
        )

    if cfg.trainer.offload_optimizer_during_rollouts and cfg.trainer.placement.colocate_all:
        raise ValueError(
            "trainer.offload_optimizer_during_rollouts applies to disaggregated runs only; "
            "colocated runs already offload the optimizer between policy updates"
        )

    # Validate placement
    validate_expert_block_transport(cfg)
    if cfg.trainer.placement.colocate_all and cfg.trainer.rollout_buffer.max_staleness_steps != 0:
        # Colocated engines sleep during training, so no rollout may run ahead of the trained policy.
        raise ValueError("colocate_all requires trainer.rollout_buffer.max_staleness_steps=0")
    if cfg.trainer.placement.colocate_all:
        tp_pp_size = (
            cfg.generator.inference_engine_tensor_parallel_size * cfg.generator.inference_engine_pipeline_parallel_size
        )
        gpus_per_node = cfg.trainer.placement.policy_num_gpus_per_node
        validate_colocated_engine_geometry(tensor_pipeline_size=tp_pp_size, gpus_per_node=gpus_per_node)
        num_policy_gpus = cfg.trainer.placement.policy_num_gpus_per_node * cfg.trainer.placement.policy_num_nodes
        num_rollout_gpus = (
            cfg.generator.num_inference_engines
            * cfg.generator.inference_engine_tensor_parallel_size
            * cfg.generator.inference_engine_pipeline_parallel_size
            * cfg.generator.inference_engine_data_parallel_size
        )
        assert num_policy_gpus == num_rollout_gpus, (
            f"num_policy_gpus ({num_policy_gpus}) and num_rollout_gpus ({num_rollout_gpus}) must be the same when colocating all models"
        )

    else:
        if cfg.trainer.placement.colocate_policy_ref and use_ref_model:
            assert cfg.trainer.placement.policy_num_nodes == cfg.trainer.placement.ref_num_nodes, (
                f"policy_num_nodes ({cfg.trainer.placement.policy_num_nodes}) and ref_num_nodes ({cfg.trainer.placement.ref_num_nodes}) must be the same when colocate policy and ref model."
            )
            assert cfg.trainer.placement.policy_num_gpus_per_node == cfg.trainer.placement.ref_num_gpus_per_node, (
                f"policy_num_gpus_per_node ({cfg.trainer.placement.policy_num_gpus_per_node}) and ref_num_gpus_per_node ({cfg.trainer.placement.ref_num_gpus_per_node}) must be the same when colocate policy and ref model."
            )

    if cfg.generator.engine_init_timeout_seconds <= 0:
        raise ValueError("generator.engine_init_timeout_seconds must be greater than zero")


def validate_batch_invariant_config(cfg: DictConfig) -> None:
    """Validate that trainer and rollout kernels are controlled together."""

    if not cfg.trainer.algorithm.batch_invariant:
        return
    if cfg.generator.backend != "vllm":
        raise ValueError("trainer.algorithm.batch_invariant=true requires generator.backend='vllm'")
    if not cfg.generator.run_engines_locally:
        raise ValueError(
            "trainer.algorithm.batch_invariant=true cannot configure a remote inference server; "
            "run the vLLM engines locally so both rollout and trainer activation is guaranteed"
        )


def validate_generator_cfg(cfg: DictConfig):
    """Validates the correctness of generator-related config.

    Args:
        cfg (DictConfig): config to validate

    Raises:
        NotImplementedError: if feature is not supported, such as sglang for multiturn generation
        ValueError: when cfg.generator.sampling_params.logprobs > 0
    """

    parse_trajectory_reward_shaping_config(cfg.generator.get("trajectory_reward_shaping"))
    parse_trajectory_retention_config(cfg.generator.get("trajectory_retention"))
    reasoning_spans = ReasoningSpanSettings.from_config(cfg.generator.reasoning_spans)
    ftpo = ftpo_config(cfg.trainer.algorithm)
    if (
        reasoning_spans.penalizes
        and reasoning_spans.penalty_target == ReasoningPenaltyTarget.MARKER
        and ftpo is not None
        and ftpo.boundary.uses_verified_boundaries
    ):
        raise ValueError(
            "generator.reasoning_spans.penalty_target=marker adds per-token advantages, which closure-repair "
            "training does not support; use penalty_target=rollout"
        )

    if cfg.generator.max_turns == 1:
        assert cfg.generator.max_input_length == cfg.trainer.max_prompt_length, (
            "generator.max_input_length should be set equal to trainer.max_prompt_length for single-turn generation"
        )
    else:
        assert cfg.generator.max_input_length >= cfg.trainer.max_prompt_length, (
            "generator.max_input_length should be set greater than or equal to trainer.max_prompt_length for multi-turn generation"
        )

    if not cfg.generator.run_engines_locally:
        assert cfg.generator.num_inference_engines == len(cfg.generator.remote_inference_engine_urls), (
            "num_inference_engines should be equal to the number of remote_inference_engine_urls"
        )

    # TODO(tgriggs): use a more modular config validation
    if cfg.trainer.logger == "wandb":
        require_secret("WANDB_API_KEY")

    if cfg.generator.override_existing_update_group == "auto":
        if cfg.generator.backend == "vllm" and not cfg.generator.run_engines_locally:
            # remote engines can be launched separately so we `enable` by default
            cfg.generator.override_existing_update_group = "enable"
        else:
            # for local engines or sglang, we disable
            cfg.generator.override_existing_update_group = "disable"

    # TODO: fix once we support these features with SGLang
    if cfg.generator.backend == "sglang" and cfg.generator.run_engines_locally:
        assert cfg.generator.inference_engine_tensor_parallel_size == 1, (
            "As of now, We do not support tensor parallel inference engine with SGLang when running engines locally. "
            "Please set `inference_engine_tensor_parallel_size` to 1."
        )

    validate_generation_logprobs(cfg)

    validate_megatron_cfg(cfg)
    if cfg.generator.backend == "sglang":
        # Some sampling parameters are not supported in SGLang when `skip_tokenizer_init` is True.
        if cfg.generator.sampling_params.stop is not None or cfg.generator.eval_sampling_params.stop is not None:
            raise ValueError(
                "`sampling_params.stop` and `eval_sampling_params.stop` are not supported for SGLang backend "
                "since we always set `skip_tokenizer_init` to True. If you have to use these parameters, you can switch to vLLM. "
                "See this issue for more: https://github.com/sgl-project/sglang/issues/9039#issuecomment-3218331087"
            )
        if "min_new_tokens" in cfg.generator.sampling_params or "min_new_tokens" in cfg.generator.eval_sampling_params:
            raise ValueError(
                "`sampling_params.min_new_tokens` and `eval_sampling_params.min_new_tokens` are not "
                "supported for SGLang backend since we always set `skip_tokenizer_init` to True. "
                "If you have to use these parameters, you can switch to vLLM. "
                "See this issue for more: https://github.com/sgl-project/sglang/issues/9039#issuecomment-3218331087"
            )

    if cfg.generator.enable_http_endpoint:
        if cfg.generator.backend == "sglang":
            # TODO(Charlie): sglang_server.py not supported for /chat/completion yet because we have
            # skip_tokenizer_init=True in engine creation. Fix by getting tokens via return logprobs
            # instead. sglang_engine.py not supported yet because we still need to figure out how
            # to make SGLang Python engine take OAI request.
            raise ValueError(
                'generator.enable_http_endpoint is not supported for SGLang backend yet. Please set generator.backend="vllm".'
            )

    # Validate inference engine parallelism.
    ep_size = cfg.generator.inference_engine_expert_parallel_size
    dp_size = cfg.generator.inference_engine_data_parallel_size
    tp_size = cfg.generator.inference_engine_tensor_parallel_size
    if cfg.generator.backend == "sglang":
        assert dp_size == 1, "Inference data parallelism is not yet supported for SGLang backend."
        assert ep_size == 1, "Inference expert parallelism is not yet supported for SGLang backend."
    if ep_size > 1:
        assert dp_size * tp_size == ep_size, (
            f"If inference expert parallel is enabled, data parallel size * tensor parallel size must equal expert parallel size. "
            f"Got dp_size={dp_size}, tp_size={tp_size}, ep_size={ep_size}"
        )

    # Validate inference engine Decode Context Parallel (DCP).
    _validate_dcp_cfg(cfg)


def _is_mla_arch(model_cfg) -> bool:
    """Whether an HF model config describes a Multi-head Latent Attention (MLA) arch.

    MLA models (DeepSeek-V2/V3, Kimi, etc.) compress the KV into a single shared latent
    rather than exposing `num_key_value_heads` GQA groups, so the per-head DCP bound does
    not apply the same way (DCP shards the latent; the effective kv-head count for the
    bound is 1 -> the bound relaxes to dcp <= tp). We detect MLA conservatively via the
    presence of the MLA-specific latent-rank fields that DeepSeek/Kimi configs carry.
    """
    if model_cfg is None:
        return False
    # DeepSeek/Kimi MLA configs carry the compressed-KV latent rank. Either field present
    # is a reliable MLA marker across the DeepSeek-V2/V3/Kimi family.
    for field in ("kv_lora_rank", "q_lora_rank"):
        val = getattr(model_cfg, field, None)
        if val is not None:
            return True
    # Fall back to architecture-name sniffing for forks that drop the latent-rank fields.
    archs = getattr(model_cfg, "architectures", None) or []
    if any(("deepseek" in a.lower() or "kimi" in a.lower()) for a in archs):
        return True
    return False


def _resolve_num_kv_heads(model_path: str):
    """Resolve (num_kv_heads, is_mla, model_cfg) from an HF model config, offline-safe.

    Returns (None, False, None) if the config cannot be resolved without a network
    download (so the caller degrades gracefully to the cheap dcp<=tp bound + warning).
    For GQA/MHA models `num_kv_heads = num_key_value_heads` (falling back to
    `num_attention_heads` for pure MHA, which has no separate kv-head count). For MLA
    models the effective kv-head count is 1 (single shared latent).
    """
    try:
        from transformers import AutoConfig
    except Exception:  # transformers unavailable -> cannot resolve
        return None, False, None
    try:
        # Respect offline / cached configs: do not force a network download at
        # validate-time. If the config isn't locally resolvable, transformers raises
        # (offline) or we let any error fall through to the graceful-degrade path.
        model_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception:
        return None, False, None

    is_mla = _is_mla_arch(model_cfg)
    if is_mla:
        return 1, True, model_cfg
    # text_config nesting (multimodal / wrapped configs) — prefer the inner text config.
    inner = getattr(model_cfg, "text_config", None) or model_cfg
    num_kv_heads = getattr(inner, "num_key_value_heads", None)
    if num_kv_heads is None:
        # Pure MHA: no separate kv-head count -> kv-heads == attention heads.
        num_kv_heads = getattr(inner, "num_attention_heads", None)
    if num_kv_heads is None:
        return None, False, model_cfg
    return int(num_kv_heads), False, model_cfg


def _assert_dcp_capable_arch(model_cfg, dcp: int):
    """(G3 f) Assert the model arch has a DCP-capable attention backend.

    DCP is supported for GQA models (default FlashAttention backend) and MLA models
    (FlashMLA / FlashAttnMLA). We keep the gate permissive: any standard decoder runs on
    FlashAttention's GQA path (DCP-capable), and MLA is explicitly capable. We reject only
    when the config positively declares a non-DCP-capable attention implementation
    (e.g. an arch forced onto an attn backend with no DCP path).
    """
    if model_cfg is None:
        # Could not resolve the config offline -> can't positively reject; rely on the
        # cheap bound + the vLLM init-time assert. (Handled by the caller's warning.)
        return
    if _is_mla_arch(model_cfg):
        return  # MLA is DCP-capable (FlashMLA / FlashAttnMLA).
    # GQA / dense decoders run on FlashAttention (DCP-capable). Reject only an explicit,
    # known-incapable attention implementation pinned in the config.
    attn_impl = getattr(model_cfg, "_attn_implementation", None) or getattr(model_cfg, "attn_implementation", None)
    incapable = {"eager", "sdpa"}
    if attn_impl is not None and str(attn_impl).lower() in incapable:
        raise AssertionError(
            f"decode context parallel (inference_engine_decode_context_parallel_size={dcp}) "
            f"requires a DCP-capable attention backend (FlashAttention for GQA, FlashMLA for MLA), "
            f"but the model config pins attn_implementation='{attn_impl}', which has no DCP path. "
            f"Use a GQA/MLA model on FlashAttention, or disable DCP."
        )


def _validate_dcp_cfg(cfg: DictConfig):
    """Validate the vLLM Decode Context Parallel (DCP) generator config (Stages 0–2).

    vLLM DCP shards the KV cache along the token dim across `dcp` ranks *inside* the
    TP group during decode (it reuses the TP GPUs, adding no GPUs — G4). It is gated
    entirely off by default (`inference_engine_decode_context_parallel_size == 1`), in
    which case this function is a strict no-op and the run is byte-identical to today
    (G1). When enabled (`dcp > 1`) we fail-closed at SkyRL config-validate (G3) rather
    than deep in vLLM init, with clear messages:

      (a) tp % dcp == 0 (vLLM splits the TP group into dcp subgroups; parallel.py:474-478),
      (b) dcp <= tp // num_kv_heads(model) — the real upper bound; beyond tp/H the KV is
          merely duplicated and there is no non-attention work to shard (doc :27,29).
          Resolved model-aware from the HF config; degrades to the cheap dcp<=tp bound +
          a warning when the config is not offline-resolvable.
      (c) backend == "vllm" (SGLang has no DCP knob),
      (e) NOT R3 router capture (vLLM rejects DCP + enable_return_routed_experts;
          vllm/config/vllm.py:1939-1944),
      (f) the arch is DCP-capable (GQA via FlashAttention, or MLA via FlashMLA).

    Remote engines (`run_engines_locally == False`): SkyRL does not launch the inference
    server, so the `-dcp N` flag must be set on the EXTERNAL `vllm serve` command. We run
    the geometry checks (a)/(b)/(f) where the config is resolvable and emit a warning
    reminding the operator to pass `-dcp` externally; we do not inject dcp into a server
    SkyRL doesn't spawn.

    See notes/RL/skyrl/vllm_dcp_rollout_stages/.
    """
    gen = cfg.generator
    dcp = gen.inference_engine_decode_context_parallel_size
    tp = gen.inference_engine_tensor_parallel_size
    assert dcp >= 1, f"generator.inference_engine_decode_context_parallel_size must be >= 1, got {dcp}"
    if dcp == 1:
        # DCP disabled -> strict no-op, no further constraints (G1).
        return

    # (a) vLLM splits the TP group into `dcp` subgroups -> tp must be divisible by dcp.
    assert tp % dcp == 0, (
        f"vLLM decode context parallel requires tensor_parallel_size % dcp == 0 "
        f"(DCP splits the TP group); got inference_engine_tensor_parallel_size={tp}, "
        f"inference_engine_decode_context_parallel_size={dcp}"
    )
    # (cheap bound) 1 <= dcp <= tp. Always enforced; the tighter kv-head bound below
    # supersedes it when the model config resolves offline.
    assert 1 <= dcp <= tp, (
        f"inference_engine_decode_context_parallel_size must satisfy 1 <= dcp <= "
        f"inference_engine_tensor_parallel_size; got dcp={dcp}, tp={tp}"
    )
    # (c) SGLang has no DCP knob.
    assert gen.backend == "vllm", (
        f"decode context parallel (inference_engine_decode_context_parallel_size={dcp}) "
        f"is only supported with generator.backend='vllm', got '{gen.backend}'"
    )

    # (b)/(f) model-aware bounds: resolve the HF config offline-safely.
    model_path = cfg.trainer.policy.model.path
    num_kv_heads, is_mla, model_cfg = _resolve_num_kv_heads(model_path)
    if num_kv_heads is None:
        # Could not resolve the config without a network download -> degrade gracefully:
        # keep the cheap dcp<=tp bound (already asserted) and warn, leaning on vLLM's own
        # init-time assert for the exact kv-head bound.
        logger.warning(
            f"DCP enabled (dcp={dcp}) but could not resolve the HF model config for "
            f"'{model_path}' offline -> skipping the model-aware kv-head bound "
            f"(dcp <= tp/num_kv_heads) and arch gate. The cheap bound (1 <= dcp <= tp) is "
            f"enforced; vLLM's engine init will enforce the exact dcp <= tp/num_kv_heads bound."
        )
    else:
        # (b) real upper bound: dcp <= tp // num_kv_heads. For MLA num_kv_heads==1 so this
        # relaxes to dcp <= tp (vLLM's own init enforces the exact MLA bound).
        max_dcp = tp // num_kv_heads
        kv_kind = "MLA latent (1 effective kv-head)" if is_mla else f"num_key_value_heads={num_kv_heads}"
        assert dcp <= max_dcp, (
            f"decode context parallel (inference_engine_decode_context_parallel_size={dcp}) "
            f"exceeds the kv-head bound dcp <= tp // num_kv_heads = {tp} // {num_kv_heads} = {max_dcp} "
            f"for model '{model_path}' ({kv_kind}). Beyond tp/num_kv_heads the KV cache is merely "
            f"duplicated and there is no non-attention work to shard; reduce dcp to <= {max_dcp}."
        )
        # (f) arch must be DCP-capable (GQA via FlashAttention, or MLA).
        _assert_dcp_capable_arch(model_cfg, dcp)

    # Remote path: SkyRL doesn't launch the server, so `-dcp` must be set externally.
    if not bool(gen.get("run_engines_locally", True)):
        logger.warning(
            f"DCP enabled (dcp={dcp}) with run_engines_locally=False: SkyRL does NOT launch "
            f"remote inference servers, so the external `vllm serve` command MUST be started "
            f"with `-dcp {dcp}` (and matching `--tensor-parallel-size {tp}`). SkyRL only carries "
            f"dcp as client metadata for geometry/GPU-accounting consistency."
        )
    # (e) vLLM rejects DCP together with R3 router capture (enable_return_routed_experts).
    # R3 capture is configured at the generator level (direct flag or engine_init_kwargs),
    # Training-side replay is configured through the Megatron policy config.
    #
    # Opt-in bypass: VLLM_ALLOW_ROUTED_EXPERTS_DCP=1 lifts this guard, mirroring the
    # identical env-var-gated bypass in the patched vLLM fork
    # (vllm/config/vllm.py:_validate_return_routed_experts). DCP+R3 is NOT a fundamental
    # incompatibility — MoE routing is a per-token function of each token's hidden state at
    # the MoE layers, independent of how DCP shards the attention decode-KV; each DCP rank
    # captures routing for its own tokens and the buffer is gathered like the output tokens.
    # Validated by the dcp=1-vs-dcp=2 routed-experts parity smoke (token mismatch 6.94% =
    # bf16-tie floor; job 905835 / DCP GQA-LSE fp32 fix). Fail-closed by default — without
    # the env var the original hard assertion stands. (Mirrors the vLLM guard so the two
    # never disagree; the SIF's vLLM honors the same flag.)
    r3_capture = (
        bool(gen.get("enable_return_routed_experts", False))
        or bool(gen.get("engine_init_kwargs", {}).get("enable_return_routed_experts", False))
        or moe_router_replay_requested(cfg)
    )
    allow_routed_experts_dcp = os.environ.get("VLLM_ALLOW_ROUTED_EXPERTS_DCP", "0") == "1"
    if r3_capture and allow_routed_experts_dcp:
        logger.warning(
            f"VLLM_ALLOW_ROUTED_EXPERTS_DCP=1: allowing decode context parallel "
            f"(inference_engine_decode_context_parallel_size={dcp}) together with R3 router "
            f"capture (enable_return_routed_experts / moe_router_replay). Out of vLLM "
            f"PR#39917's validated scope; relies on the dcp=1-vs-dcp=2 routed-experts parity "
            f"smoke (bf16-tie floor). Verify routed-experts capture parity before trusting results."
        )
    else:
        assert not r3_capture, (
            f"decode context parallel (inference_engine_decode_context_parallel_size={dcp}) is not "
            f"compatible with R3 router capture (enable_return_routed_experts / moe_router_replay); "
            f"vLLM rejects DCP + return_routed_experts. Disable one of them, or set "
            f"VLLM_ALLOW_ROUTED_EXPERTS_DCP=1 to opt into the parity-validated bypass."
        )

    # (G4) DCP rides the TP GPUs and must NOT enter rollout-GPU accounting. Assert the
    # rollout-GPU count is a function of tp*pp*dp only, independent of dcp.
    pp = gen.inference_engine_pipeline_parallel_size
    dp = gen.inference_engine_data_parallel_size
    per_engine_gpu_count = tp * pp * dp
    num_rollout_gpus = gen.num_inference_engines * per_engine_gpu_count
    assert num_rollout_gpus == gen.num_inference_engines * tp * pp * dp, (
        "DCP must not change rollout-GPU accounting (it reuses the TP GPUs); "
        f"num_rollout_gpus must equal num_inference_engines*tp*pp*dp regardless of dcp={dcp}"
    )


@ray.remote
def get_all_env_variables():
    import os

    return os.environ


def ray_noset_visible_devices(env_vars=os.environ):
    # Refer to
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/nvidia_gpu.py#L95-L96
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/amd_gpu.py#L102-L103
    # https://github.com/ray-project/ray/blob/3b9e729f6a669ffd85190f901f5e262af79771b0/python/ray/_private/accelerators/amd_gpu.py#L114-L115
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/npu.py#L94-L95
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/hpu.py#L116-L117
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/neuron.py#L108-L109
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/tpu.py#L171-L172
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/intel_gpu.py#L97-L98
    NOSET_VISIBLE_DEVICES_ENV_VARS_LIST = [
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES",
        "RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES",
        "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
        "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR",
    ]
    return any(env_vars.get(env_var) for env_var in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST)


def get_physical_gpu_id():
    import torch

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return str(props.uuid)


def prepare_runtime_environment(cfg: DictConfig) -> dict[str, str]:
    """
    Prepare environment variables for Ray runtime environment.

    Args:
        cfg: Training config

    Returns:
        Dict[str, str]: Environment variables to be used in Ray runtime environment
    """
    env_vars = {}

    # Force CPython stock asyncio (epoll SelectorEventLoop), NOT uvloop, in EVERY
    # Ray worker/actor process spawned under this runtime env.
    #
    # Ray installs uvloop globally in every worker by default
    # (RAY_USE_UVLOOP defaults True -> ray default_worker.py try_install_uvloop).
    # libuv's epoll-ctl machinery SIGABRTs under Daytona sandbox-teardown socket
    # churn (uv__epoll_ctl_prep / uv__io_poll asserts; present across libuv
    # 1.45-1.49+). The driver/orchestrator is already protected by
    # asyncio.set_event_loop_policy(DefaultEventLoopPolicy()) in
    # BasePPOExp.run() (entrypoints/main_base.py), but that ONLY covers the
    # skyrl_entrypoint driver process -- it does NOT propagate into the Ray
    # *actor* processes (rollout workers, inference engines, policy/ref
    # workers), which still install uvloop and still abort. Observed on the 80B
    # production run (job 669177): a rollout worker's CoreWorker aborted at
    # Fatal Python error: Aborted -> uv__epoll_ctl_prep inside
    # CoreWorker.initialize_eventloops_for_actor_concurrency_group AFTER a full
    # clean step 1, taking the run down. The EnvVarManager projection below sets
    # RAY_USE_UVLOOP=0 before actor imports, making Ray's uvloop install a no-op.
    # Actors are network-RTT-bound (vLLM/Daytona HTTP) so uvloop's throughput
    # edge is moot. See feedback_uvloop_libuv_019_pin.
    # Ray's job runtime environment is the explicit contract for worker-wide
    # settings. Forward NUMA placement when the launcher opts in so the early
    # worker hook and actor constructors observe the same value as the driver.
    if NUMA_AFFINITY_ENV in os.environ:
        env_vars[NUMA_AFFINITY_ENV] = os.environ[NUMA_AFFINITY_ENV]

    # Disable libuv's io_uring backend in EVERY Ray actor/worker process.
    #
    # WHY (job 930208, the SSL re-abort): RAY_USE_UVLOOP=0 + the policy hook
    # were STILL insufficient -- 930208 SIGABRT'd in a rollout worker at
    # uvloop/sslproto.pyx:517 SSLProtocol._on_handshake_complete (the
    # litellm->Daytona HTTPS handshake), proving a live uvloop.Loop() was
    # running an SSL transport in the actor despite the stock-asyncio POLICY.
    # uvloop.new_event_loop() constructs uvloop.Loop() DIRECTLY (it does NOT
    # consult asyncio's event-loop policy), so a policy reset cannot stop a
    # uvloop loop that Ray's C++ CoreWorker or aiohttp brings up by calling
    # uvloop.new_event_loop() outright.
    #
    # The crash is libuv 1.48.0's io_uring epoll-ctl race (uv__epoll_ctl_prep;
    # fixed in 1.49.0). libuv reads UV_USE_IO_URING via getenv() inside
    # uv__use_io_uring() AT FIRST USE and caches it atomically; "0" -> io_uring
    # is never armed and the buggy path is dead, while uvloop otherwise keeps
    # working on plain epoll. The var was INERT before because it lived only in
    # the driver/launcher (host) env and was NEVER propagated into the Ray
    # ACTOR process env -- Ray actors derive their environment from the managed
    # runtime_env contract, not from arbitrary driver env. EnvVarManager puts it
    # in the actor's process env before the actor imports uvloop/libuv, so
    # libuv's first uv__use_io_uring() reads 0.
    env_vars.update(EnvVarManager.from_config(cfg).environment_for(EnvVarScope.RAY_WORKER))
    # Resolve the actual collective deadline last so the debug preset's heartbeat
    # cannot exceed a shorter explicitly configured process-group timeout.
    env_vars.update(
        worker_nccl_environment(
            base_environment=env_vars,
            collective_timeout_seconds=int(cfg.trainer.distributed.worker_collective_timeout_seconds),
        )
    )

    # NOTE (charlie): See https://github.com/vllm-project/vllm/blob/c6b0a7d3ba03ca414be1174e9bd86a97191b7090/vllm/worker/worker_base.py#L445
    # and https://docs.vllm.ai/en/v0.9.2/usage/troubleshooting.html?h=nccl_cumem_enable#known-issues
    # Same for SGLang as we set `NCCL_CUMEM_ENABLE` to 0 in `sglang_engine.py`'s _patched_set_envs_and_config
    if cfg.generator.weight_sync_backend == "nccl":
        env_vars["NCCL_CUMEM_ENABLE"] = "0"

    # Useful when TP > 1 (and thus Megatron sequence parallel is enabled).
    # See https://github.com/NVIDIA/Megatron-LM/issues/533#issuecomment-1760193239
    env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    if cfg.trainer.flash_attn:
        # TransformerEngine otherwise overrides the FlashAttention choice on Hopper+.
        env_vars["NVTE_FUSED_ATTN"] = "0"

    if cfg.generator.backend == "vllm":
        env_vars["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "true"

        # NOTE (sumanthrh): In vllm >= 0.9.0, we need to explicitly allow for serialization via pickle for collective RPCs.
        # During weight transfer, we use IPC handles, which contains a `function` object and requires pickling.
        env_vars["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

        # NOTE (sumanthrh): In vLLM >= 0.9.0, we've observed compilatiion failures with torch compile. removing the compilation directory and trying
        # again does not fix the issue. Temporarily we disable compilation cache, which seems to fix the issue.
        # This should not have any effect on performance - compilation will still happen, it's just not cached
        # TODO (sumanthrh): remove this once vLLM fixes the issue
        env_vars["VLLM_DISABLE_COMPILE_CACHE"] = "1"

        if not os.environ.get("VLLM_USE_V1", False):
            logger.info(
                "`VLLM_USE_V1` is not specified, setting `VLLM_USE_V1` to 1. To override, set `VLLM_USE_V1` explicitly"
            )
            env_vars["VLLM_USE_V1"] = "1"
            # The `mp` executor backend (Qwen3-Next R3 capture path, opt-in
            # generator.inference_engine_mp_backend) REQUIRES v1 multiprocessing to
            # spawn its TP worker subprocesses; forcing it to 0 here (the default,
            # for scheduling determinism) cancels the mp executor's shm message
            # queue at warm-up. Keep it ON (unset => vLLM default 1) for mp.
            mp_backend = bool(getattr(cfg.generator, "inference_engine_mp_backend", False))
            if not mp_backend:
                env_vars["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
            else:
                logger.info(
                    "inference_engine_mp_backend=true: NOT setting VLLM_ENABLE_V1_MULTIPROCESSING=0 "
                    "(the mp executor needs v1 multiprocessing to spawn TP workers)."
                )

    # Use max of available GPU counts, defaulting to 1 if none found
    gpu_counts = []
    if hasattr(cfg.generator, "inference_engine_tensor_parallel_size"):
        gpu_counts.append(cfg.generator.inference_engine_tensor_parallel_size)
    if hasattr(cfg, "trainer") and hasattr(cfg.trainer, "placement"):
        placement = cfg.trainer.placement
        gpu_counts.extend(
            [
                placement.policy_num_gpus_per_node,
                placement.critic_num_gpus_per_node,
                placement.ref_num_gpus_per_node,
            ]
        )
    max_num_gpus_per_node = max(gpu_counts) if gpu_counts else 1
    if not peer_access_supported(
        max_num_gpus_per_node=max_num_gpus_per_node,
        placement_group_timeout_seconds=int(cfg.trainer.distributed.placement_group_timeout_seconds),
    ):
        logger.info("Peer access is not supported on this node type, disabling NCCL P2P and SHM")
        env_vars["NCCL_P2P_DISABLE"] = "1"
        env_vars["NCCL_SHM_DISABLE"] = "1"

    # Ray workers share the task's read-only credential mount.
    if path := os.environ.get(SECRETS_DIR_ENV):
        env_vars[SECRETS_DIR_ENV] = path
    elif os.environ.get("WANDB_API_KEY"):
        logger.info("Exporting wandb api key to ray runtime env")
        env_vars["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]

    if os.environ.get("MLFLOW_TRACKING_URI"):
        logger.info("Exporting mlflow tracking uri to ray runtime env")
        env_vars["MLFLOW_TRACKING_URI"] = os.environ["MLFLOW_TRACKING_URI"]

    if os.environ.get("MLFLOW_TRACKING_TOKEN"):
        logger.info("Exporting mlflow tracking token to ray runtime env")
        env_vars["MLFLOW_TRACKING_TOKEN"] = os.environ["MLFLOW_TRACKING_TOKEN"]

    if os.environ.get("RAY_ADDRESS"):
        logger.info("Exporting RAY_ADDRESS to ray runtime env")
        env_vars["RAY_ADDRESS"] = os.environ["RAY_ADDRESS"]

    # NCCL network-fabric selection.
    # Ray actors start with a clean environment and only see env vars that we
    # explicitly forward here. On clusters where NCCL's interface auto-detection
    # picks the wrong NIC (e.g. a down `eno*`/loopback instead of the routable
    # InfiniBand `ib0`), the cross-process weight-sync communicator
    # (policy rank 0 + inference-engine ranks) can never complete its first
    # collective and the run hangs at weight sync. Forwarding these lets the
    # operator pin the fabric from the launcher env (or hpc.launch) and have it
    # actually reach the policy/vLLM workers. Pure passthrough: a no-op unless
    # the var is set in the launcher environment, so the production/default path
    # is unchanged.
    #
    # GLOO_SOCKET_IFNAME is deliberately NOT in this list. runtime_env is
    # job-level, so it pushes one value to every node, and gloo reads the value as
    # a literal interface name with no `^exclude` form. NIC names are not uniform
    # across a gang: job 20260729-102429-52af30 broadcast the head's `enp90s0np0`
    # to a node naming its NIC `enp90s0f0np0` and megatron's gloo group creation
    # there died with `Unable to find address for: enp90s0np0`. Set it per node
    # instead (the iris controller derives it from each pod's own IP before that
    # node's `ray start`, which the node's Ray workers inherit).
    for _net_env in (
        "NCCL_SOCKET_IFNAME",
        "NCCL_IB_HCA",
        "NCCL_IB_DISABLE",
        "NCCL_NET",
        "NCCL_SOCKET_FAMILY",
    ):
        if os.environ.get(_net_env):
            logger.info(f"Exporting `{_net_env}` to ray runtime env: {os.environ[_net_env]}")
            env_vars[_net_env] = os.environ[_net_env]

    if env_vars.get(VLLM_BATCH_INVARIANT_ENV) == "1" or os.environ.get(VLLM_BATCH_INVARIANT_ENV) == "1":
        # Megatron and vLLM share a weight-update NCCL group; mismatched settings broke its first broadcast.
        # Remove this override if publication no longer joins their ranks; it also affects learner collectives.
        env_vars.update(BATCH_INVARIANT_NCCL_ENV)

    # EnvVarManager owns NCCL verbosity through trainer.debug_mode. Do not copy
    # ambient NCCL_DEBUG values into Ray workers: stale launcher extra_env once
    # turned an ordinary checkpoint export into a per-collective INFO trace.
    # WARN still leaves the bounded flight recorder armed for timeout dumps.
    env_vars.setdefault("NCCL_DEBUG", "WARN")

    return env_vars


def configure_ray_worker_logging() -> None:
    """
    In Ray workers, stderr/stdout are not TTYs, so Loguru disables color.
    This method forces color and formatting (e.g., bold) and routes stdlib `logging`
    through Loguru so third-party logs match formatting
    """
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()

    # 1) Loguru formatting (force colors)
    logger.remove()
    logger.level("INFO", color="<bold><green>")
    logger.add(
        sys.stderr,
        colorize=True,  # keep ANSI even without a TTY
        level=level_name,  # ensure Loguru filters below this level
        enqueue=True,
        backtrace=False,
        diagnose=False,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>",
    )

    # 2) Route stdlib logging -> Loguru (so vLLM/transformers/etc. are formatted)
    class _InterceptHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno
            message = record.getMessage()
            if record.exc_info is not None and record.exc_info[1] is not None:
                message = f"{message}\n{format_exception_text(record.exc_info[1])}"
            logger.opt(depth=6).log(level, message)

    logging.root.handlers = [_InterceptHandler()]
    level = getattr(logging, level_name, logging.INFO)
    logging.root.setLevel(level)


def initialize_ray(cfg: DictConfig):
    """
    Initialize Ray cluster with prepared runtime environment.

    Args:
        cfg: Training config
    """
    debug_environment = apply_debug_mode(cfg)
    if debug_environment.get(DEBUG_ARTIFACT_DIR_ENV):
        manifest, stack_path = initialize_process_diagnostics("driver", environment=debug_environment)
        logger.info(f"Debug mode {debug_environment[DEBUG_MODE_ENV]} active; driver manifest: {manifest}")
        if stack_path is not None:
            logger.info(f"On-demand all-thread stack capture active (SIGUSR2): {stack_path}")

    env_vars = prepare_runtime_environment(cfg)
    # worker_process_setup_hook runs ONCE at the start of every Ray worker process,
    # BEFORE the C++ CoreWorker builds actor threads. It installs the inherited
    # host-memory policy and forces CPython stock asyncio (no uvloop/libuv) in every
    # worker -- the only reliable place to cover concurrency-group loops that are
    # created before an actor constructor can reset their event-loop policy.
    # Referenced by fully-qualified name string (importable in every worker via the
    # editable skyrl_train install) so Ray does not have to cloudpickle it.
    ray.init(
        runtime_env={
            "env_vars": env_vars,
            "worker_process_setup_hook": "skyrl_train.worker_setup.configure_worker_process",
        }
    )

    # create the named ray actors for the registries to make available to all workers
    sync_registries()


def get_ray_pg_ready_with_timeout(pg: PlacementGroup, timeout: int = 60):
    try:
        ray.get(pg.ready(), timeout=timeout)
    except Exception as e:
        # Extract resource demands from the placement group
        bundles = pg.bundle_specs
        total_gpus = sum(bundle.get("GPU", 0) for bundle in bundles)
        total_cpus = sum(bundle.get("CPU", 0) for bundle in bundles)

        raise RuntimeError(
            f"Failed to create placement group with {len(bundles)} bundles "
            f"(requiring {total_gpus} GPUs, {total_cpus} CPUs total) in {timeout} seconds. "
            f"This might indicate insufficient GPU resources.\n"
            f"Error: {e}"
        )


@ray.remote(num_gpus=1)
class InfoActor:
    def get_gpu_id(self):
        return ray.get_gpu_ids()[0]


def get_reordered_bundle_indices(pg: PlacementGroup):
    pg_data = placement_group_table(pg)
    num_bundles = len(pg_data["bundles"])
    bundle_to_node_ids = pg_data["bundles_to_node_id"]

    # use info actor to get the GPU id
    info_actors = []
    for i in range(num_bundles):
        info_actors.append(
            InfoActor.options(
                num_cpus=0.01,  # set both num_cpus and num_gpus to be small values to enable assignment in colocated case
                num_gpus=0.01,
                resources=None,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=i,
                ),
            ).remote()
        )

    gpu_ids = ray.get([actor.get_gpu_id.remote() for actor in info_actors])
    for actor in info_actors:
        ray.kill(actor)

    # original index, node_id, gpu_id
    bundle_infos = [(i, bundle_to_node_ids[i], gpu_ids[i]) for i in range(num_bundles)]
    pg_reordered_bundle_indices = [
        bundle_info[0] for bundle_info in sorted(bundle_infos, key=lambda x: (x[1], x[2]))
    ]  # sort by node_id, then gpu_id
    return pg_reordered_bundle_indices


# NOTE (sumanthrh): For SGLang, the string representations here should also match those used by (and supported by) SGLang.
# This is because we do not control the update weight implementation with SGLang backend.
# With VLLM, we use a custom Worker extension to have a custom update weight implementation.
def torch_dtype_to_str(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bfloat16"
    elif dtype == torch.float16:
        return "float16"
    elif dtype == torch.float32:
        return "float32"
    else:
        return str(dtype)


def str_to_torch_dtype(dtype: str) -> torch.dtype:
    name = dtype.removeprefix("torch.")
    value = getattr(torch, name, None)
    if not isinstance(value, torch.dtype):
        raise ValueError(f"unsupported torch dtype: {dtype!r}")
    return value


def format_gib(mem_bytes: int) -> str:
    return f"{mem_bytes / (1024**3):.2f} GiB"


def print_mem(tag: str, mem: dict):
    logger.info(
        f"{tag} - Allocated: {format_gib(mem['allocated'])}, "
        f"Reserved: {format_gib(mem['reserved'])}, "
        f"Free: {format_gib(mem['free'])}, "
        f"Total: {format_gib(mem['total'])}"
    )


def run_p2p_access_check():
    device_count = torch.cuda.device_count()
    if device_count < 2:
        return False

    # Check P2P access between all GPU pairs
    for i in range(device_count):
        for j in range(device_count):
            if i != j:
                # This checks if device i can access device j's memory
                can_access = torch.cuda.can_device_access_peer(i, j)
                if not can_access:
                    return False

    return True


def peer_access_supported(
    max_num_gpus_per_node: int,
    placement_group_timeout_seconds: int = DEFAULT_RAY_PLACEMENT_GROUP_TIMEOUT_SECONDS,
):
    # whatever the max num gpus per node is, we can check p2p access if there are at least 2 GPUs
    # if max is 1, p2p access is not supported
    if max_num_gpus_per_node <= 1:
        return False

    if not torch.cuda.is_available():
        # we are on cpu head node, so we need to check P2P access on a node with 2 GPUs
        ray.init()
        pg = placement_group([{"CPU": 1, "GPU": 2}], strategy="PACK")
        get_ray_pg_ready_with_timeout(pg, timeout=placement_group_timeout_seconds)
        result = ray.get(
            ray.remote(num_gpus=2, scheduling_strategy=PlacementGroupSchedulingStrategy(pg))(
                run_p2p_access_check
            ).remote()
        )
        ray.shutdown()
        return result
    else:
        return run_p2p_access_check()


def update_model_config(module_config, override_config_kwargs):
    """Update the module config with the override_config_kwargs.
    Args:
        module_config: The module config from Huggingface Transformers.
        override_config_kwargs: The kwargs to override the module config.
    """
    for key, val in override_config_kwargs.items():
        if isinstance(val, dict):
            update_model_config(getattr(module_config, key), val)
        else:
            setattr(module_config, key, val)


def get_tcp_url(host: str, port: int) -> str:
    """
    Formats the TCP URL for the given host and port,
    handling IPv6 addresses correctly.
    Args:
        host (str): The hostname or IP address.
        port (int): The port number.
    Returns:
        str: The formatted TCP URL.
    """
    try:
        if isinstance(ipaddress.ip_address(host), ipaddress.IPv6Address):
            return f"tcp://[{host}]:{port}"
    except ValueError:
        # not a literal IP, probably a hostname
        pass
    return f"tcp://{host}:{port}"


def get_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port
