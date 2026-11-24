import ipaddress
import os
import time
import sys
import logging

import ray
import torch
from loguru import logger
from rigging.secrets import SECRETS_DIR_ENV
from omegaconf import DictConfig
from ray.util.placement_group import (
    placement_group,
    PlacementGroupSchedulingStrategy,
    PlacementGroup,
    placement_group_table,
)

from skyrl_train.batch_invariant import BATCH_INVARIANT_NCCL_ENV
from skyrl_train.config.worker_environment import configured_ray_worker_environment
from skyrl_train.debug_mode import apply_debug_mode
from skyrl_train.numa_policy import NUMA_AFFINITY_ENV
from skyrl_train.env_vars import (
    DEBUG_ARTIFACT_DIR_ENV,
    DEBUG_MODE_ENV,
    VLLM_BATCH_INVARIANT_ENV,
    VLLM_DISABLE_COMPILE_CACHE_ENV,
)
from marinskyrl.runtime_options import reference_model_required
from marinskyrl.process_diagnostics import initialize_process_diagnostics

from .constants import DEFAULT_RAY_PLACEMENT_GROUP_TIMEOUT_SECONDS
from .algorithm_registry import sync_registries
from .logging_utils import format_exception_text


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
    env_vars.update(configured_ray_worker_environment(cfg))

    # NOTE (charlie): See https://github.com/vllm-project/vllm/blob/c6b0a7d3ba03ca414be1174e9bd86a97191b7090/vllm/worker/worker_base.py#L445
    # and https://docs.vllm.ai/en/v0.9.2/usage/troubleshooting.html?h=nccl_cumem_enable#known-issues
    if cfg.generator.weight_sync_backend == "nccl":
        env_vars["NCCL_CUMEM_ENABLE"] = "0"

    # Useful when TP > 1 (and thus Megatron sequence parallel is enabled).
    # See https://github.com/NVIDIA/Megatron-LM/issues/533#issuecomment-1760193239
    env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    if cfg.trainer.flash_attn:
        env_vars["NVTE_FUSED_ATTN"] = "0"

    if cfg.generator.backend == "vllm":
        env_vars["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "true"

        # NOTE (sumanthrh): In vllm >= 0.9.0, we need to explicitly allow for serialization via pickle for collective RPCs.
        # During weight transfer, we use IPC handles, which contains a `function` object and requires pickling.
        env_vars["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

        # Keep the non-AOT path by default. Local rollout actors may opt in to
        # rank-local disk caching with AOT explicitly held off.
        env_vars[VLLM_DISABLE_COMPILE_CACHE_ENV] = "1"

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
            if record.exc_info and record.exc_info[1] is not None:
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


# Weight updates use the dtype names accepted by the vLLM worker extension.
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
