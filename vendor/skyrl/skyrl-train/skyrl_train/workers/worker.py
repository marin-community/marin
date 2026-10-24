import asyncio
import concurrent.futures
import contextlib
import logging
import os
import socket
from typing import Dict, Optional, Type, List, Any, Callable
from skyrl_train.utils.progress import configure_progress, tqdm
from marinskyrl.runtime_options import R3Transport
from collections import defaultdict

import ray
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer
import torch.distributed
from ray import ObjectRef
from ray.util.placement_group import (
    PlacementGroup,
    PlacementGroupSchedulingStrategy,
    placement_group,
    placement_group_table,
)

from skyrl_train.config.query_bias import (
    GrugQueryBiasUpdate,
    GrugQueryBiasUpdateMode,
    resolve_grug_query_bias_update,
)
from skyrl_train.utils import ray_noset_visible_devices, get_ray_pg_ready_with_timeout, get_reordered_bundle_indices
from skyrl_train.utils.constants import DEFAULT_RAY_PLACEMENT_GROUP_TIMEOUT_SECONDS
from skyrl_train.io import io
from skyrl_train.utils.numa import physical_gpu_id_for_worker, set_numa_affinity_for_gpu
from skyrl_train.tensor_math import masked_mean
from skyrl_train.distributed.dispatch import ActorInfo, Dispatch, DispatchRegistry, DispatchSettings, MeshRank
from skyrl_train.distributed import collective_phase_diagnostics as _phase_diagnostics
from skyrl_train.distributed.strategy import DistributedStrategy
from loguru import logger
from skyrl_train.distributed.utils import init_custom_process_group, init_worker_process_group_with_device
from skyrl_train.utils.algorithm_registry import PolicyLossRegistry
from skyrl_train.utils.policy_math import ppo_critic_loss
from skyrl_train.utils.importance_ratio_diagnostics import (
    LogRatioMonitor,
)
from skyrl_train.learner_memory import LearnerCudaMetrics
from skyrl_train.timing_observability import PhaseBreakdown
from skyrl_train.telemetry import WORKER_ROLE, ProcessTelemetry, TelemetryConfig
from skyrl_train.config.objective_spec import topk_loss_params
from skyrl_train.objective.objective import TopKTeacherBatch, build_objective_micro_batch, compute_policy_objective
from skyrl_train.objective.reduction import StepCounts, policy_data_weights, step_counts
from skyrl_train.distillation import student_topk_logprobs
from skyrl_train.dataset.replay_buffer import Experience
from skyrl_train.training_batch import (
    TrainingBatchIterator,
    TrainingInputBatch,
    TrainingOutputBatch,
    gradient_accumulation_steps,
    per_data_parallel_batch_size,
)
from skyrl_train.trajectory_selection import optimization_samples_per_prompt
from skyrl_train.utils.metrics import mean_metrics, policy_progress_metrics, policy_training_metrics
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.models.grug_query_bias import (
    GrugLossFreeBiasUpdater,
    GrugQuantileBiasUpdater,
    GrugQueryBiasCapturePlan,
    GrugQueryBiasShardLayout,
    GrugQueryBiasUpdater,
    GrugQueryBiasWindow,
)
from skyrl_train.models.grug_moe import GrugMoeForCausalLM
from skyrl_train.batch_invariant import enable_trainer_batch_invariance
from skyrl_train.utils.utils import (
    configure_ray_worker_logging,
    get_tcp_url,
    resolve_actor_cuda_env,
    resolve_pinned_local_rank,
)
from omegaconf import DictConfig
from pathlib import Path


def _grug_query_bias_updater(
    model: GrugMoeForCausalLM,
    valid_tokens: int,
    update: GrugQueryBiasUpdate,
) -> GrugQueryBiasUpdater:
    if update.mode is GrugQueryBiasUpdateMode.LOSS_FREE:
        assert update.update_rate is not None
        return GrugLossFreeBiasUpdater(model, update_rate=update.update_rate)

    target_weight = 1.0 if update.mode is GrugQueryBiasUpdateMode.REPLACE else update.interpolation_weight
    assert target_weight is not None
    return GrugQuantileBiasUpdater(model, valid_tokens, target_weight=target_weight)


# Rigging's own shutdown waits two seconds; the extra covers the round trip to every rank.
TELEMETRY_DRAIN_TIMEOUT_SECONDS = 5.0


# Adapted from OpenRLHF: https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/trainer/ray/launcher.py#L17
class DistributedTorchRayActor:
    def __init__(
        self,
        world_size,
        rank,
        local_rank,
        master_addr,
        master_port,
        sequence_parallel_size,
        record_memory=False,
        pin_to_ray_gpu_id=False,
        force_cvd_mask=False,
    ):
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            level=logging.INFO,
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        self._world_size = world_size
        self._rank = rank
        self._local_rank = local_rank
        self._master_addr = master_addr if master_addr else self._get_current_node_ip()
        self._master_port = master_port if master_port else self._get_free_port()
        os.environ["MASTER_ADDR"] = self._master_addr
        os.environ["MASTER_PORT"] = str(self._master_port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)
        # Device pinning. `LOCAL_RANK` is consumed by every strategy's
        # setup_distributed() as the argument to torch.cuda.set_device().
        #
        # Three cases:
        #  1. ray_noset_visible_devices() True (a RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES
        #     env var is set): Ray does NOT constrain CUDA_VISIBLE_DEVICES, every actor
        #     sees all physical GPUs, so pin to this actor's Ray-assigned physical id.
        #  2. NOSET unset AND Ray constrained CUDA_VISIBLE_DEVICES to a single device for
        #     this actor (the normal/older-Ray path, e.g. the a3 8B venv runtime): device
        #     0 within the masked view IS the assigned GPU, so LOCAL_RANK="0" is correct.
        #  3. NOSET unset BUT CUDA_VISIBLE_DEVICES is unset / empty / lists >1 device
        #     (newer Ray inside the megatron+vllm SIF no longer overrides the visible-device
        #     env var, emitting "Ray will no longer override accelerator visible devices...";
        #     in this mode ray.get_gpu_ids() is ALSO unreliable — it returned [0] for every
        #     one of the 4 per-node policy actors, so even pinning to get_gpu_ids()[0] still
        #     collapses all of them onto physical GPU 0): every actor sees all GPUs, so bind
        #     to the launcher-assigned per-node rank (rank % num_gpus_per_node = 0,1,2,3),
        #     which is deterministically distinct per node and independent of Ray's
        #     accelerator bookkeeping. This is the lever that fixes the reproducible
        #     Qwen3-Next-80B init OOM (all 4 ranks materializing the 80B on GPU 0).
        #
        # Case 2 keeps the historical behavior byte-identical (single-device mask -> "0").
        # pin_to_ray_gpu_id (case 3) engages only on the per-GPU {GPU:1}-bundle policy PG,
        # where ray.get_gpu_ids() is a reliable distinct physical id per actor. The full
        # decision lives in resolve_pinned_local_rank() (pure / unit-tested).
        _noset = ray_noset_visible_devices()

        # Deterministic forced-CVD-mask pin (opt-in via policy_force_cvd_mask).
        # When engaged, mask this actor to its single Ray-assigned PHYSICAL GPU
        # and force PCI_BUS_ID ordering BEFORE any CUDA / EP-device-mesh init,
        # so set_device(0) and init_device_mesh can only resolve that
        # one physical GPU. This is independent of positional/LOCAL_RANK ordering
        # and of whether Ray masked CVD — closing the GH200
        # GPU-0-stacking init OOM that set_device(LOCAL_RANK) alone could not
        # deterministically prevent. IMPORTANT: this must run before
        # torch.cuda.device_count() below touches CUDA (which would latch the
        # unmasked device set). force_cvd_mask only engages with the per-GPU
        # {GPU:1}-bundle PG, where ray.get_gpu_ids()[0] is a distinct physical id.
        if force_cvd_mask and pin_to_ray_gpu_id:
            _cuda_env = resolve_actor_cuda_env(
                noset_visible_devices=_noset,
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                ray_gpu_ids=ray.get_gpu_ids(),
            )
            for _k, _v in _cuda_env.items():
                os.environ[_k] = _v
            logging.info(
                "[device-pin] force_cvd_mask: applied %s (ray_gpu_ids=%s)",
                _cuda_env,
                ray.get_gpu_ids(),
            )

        resolved_local_rank = resolve_pinned_local_rank(
            noset_visible_devices=_noset,
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            ray_gpu_ids=ray.get_gpu_ids(),
            launcher_local_rank=local_rank,
            device_count=torch.cuda.device_count(),
            pin_to_ray_gpu_id=pin_to_ray_gpu_id,
        )
        physical_gpu_id = physical_gpu_id_for_worker(os.environ.get("CUDA_VISIBLE_DEVICES"), int(resolved_local_rank))
        set_numa_affinity_for_gpu(physical_gpu_id)
        os.environ["LOCAL_RANK"] = resolved_local_rank
        self.sequence_parallel_size: int = sequence_parallel_size
        self._checkpoint_upload_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="checkpoint-upload",
        )
        self._checkpoint_upload_future: concurrent.futures.Future[None] | None = None
        self.record_memory = record_memory
        if record_memory:
            torch.cuda.memory._record_memory_history()
        configure_ray_worker_logging()

    def _start_checkpoint_upload(self, upload: io.PendingDirectoryUpload | None) -> None:
        if upload is None:
            return
        if self._checkpoint_upload_future is not None:
            if not self._checkpoint_upload_future.done():
                raise RuntimeError("cannot stage a checkpoint while the previous upload is still running")
            self._checkpoint_upload_future.result()
        self._checkpoint_upload_future = self._checkpoint_upload_executor.submit(upload.publish)

    async def wait_checkpoint_upload(self) -> None:
        """Wait without blocking this actor's event loop for its staged shard upload."""
        future = self._checkpoint_upload_future
        if future is None:
            return
        try:
            await asyncio.wrap_future(future)
        finally:
            if self._checkpoint_upload_future is future:
                self._checkpoint_upload_future = None

    def get_node_local_rank(self):
        return self._local_rank

    def init_worker_process_group(self):
        # Device-pinned NCCL PG init via the shared helper — pins set_device(LOCAL_RANK) and
        # passes device_id so ProcessGroupNCCL never guesses the device (fixes the cw-rno2a
        # unmasked-CVD collective deadlock; see init_worker_process_group_with_device).
        init_worker_process_group_with_device(
            timeout_seconds=int(self.cfg.trainer.distributed.worker_collective_timeout_seconds)
        )

        # setup device mesh
        # Tensor and pipeline parallelism are configured by the Megatron worker.
        # NOTE (sumanthrh): Device mesh and mesh rank are rank specific attributes. For the current way the strategy is defined, it is only meant to interact with worker state; not hold worker state. Thus, this should live outside the strategy object.
        # This device mesh can be common across all the strategies we use
        dp_size = self._world_size // self.sequence_parallel_size
        device_mesh = torch.distributed.device_mesh.init_device_mesh(
            "cuda", mesh_shape=(dp_size, self.sequence_parallel_size), mesh_dim_names=("dp", "sp")
        )
        self.device_mesh = device_mesh

        mesh_dp = self.device_mesh.get_local_rank(mesh_dim="dp")
        mesh_sp = self.device_mesh.get_local_rank(mesh_dim="sp")
        mesh_dp_size = self.device_mesh.size(0)
        self.mesh_rank = MeshRank(
            dp=mesh_dp,
            sp=mesh_sp,
            tp=0,
            pp=0,
            world_size=self._world_size,
            dp_size=mesh_dp_size,
            pp_size=1,
        )

    def get_mesh_rank(self):
        return self.mesh_rank

    def get_gpu_id(self):
        return ray.get_gpu_ids()[0]

    @staticmethod
    def _get_current_node_ip():
        # Debug: understand where the IP comes from
        import socket

        hostname = socket.gethostname()

        # Check if Ray has a global node set
        global_node = ray._private.worker._global_node
        global_node_ip = global_node.node_ip_address if global_node else "None (no global node)"

        # What does get_node_ip_address() return?
        address = ray._private.services.get_node_ip_address()

        logging.info(f"[ipv4-debug] hostname={hostname}")
        logging.info(f"[ipv4-debug] _global_node.node_ip_address={global_node_ip}")
        logging.info(f"[ipv4-debug] get_node_ip_address()={address}")

        # strip ipv6 address brackets if present
        result = address.strip("[]")
        return result

    def get_ray_node_id(self):
        return ray.get_runtime_context().get_node_id()

    @staticmethod
    def _get_free_port():
        with socket.socket() as sock:
            sock.bind(("", 0))
            return sock.getsockname()[1]

    def get_master_addr_port(self):
        return self._master_addr, self._master_port


def log_r3_resident_set(rank: int, data: TrainingInputBatch) -> None:
    """Per-rank marker that the rollout's routed-experts chunk is resident.

    Ungated INFO so a run can be read rank-by-rank: every rank that receives a
    batch carrying ``rollout_routed_experts`` logs its size on arrival. Strict
    no-op signal when the batch carries no routes.
    """
    if (routes := data.routed_experts) is not None:
        logger.info(
            f"R3_RESIDENT_SET rank={rank} nbytes={int(routes.nbytes)} dtype={routes.dtype} shape={tuple(routes.shape)}"
        )


class Worker(DistributedTorchRayActor):
    def __init__(self, cfg: DictConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        configure_progress(cfg.trainer.progress)
        # Rigging drops records from a process that never configured it.
        self._telemetry = contextlib.ExitStack()
        telemetry_config = TelemetryConfig.from_environment()
        if telemetry_config.endpoint is not None:
            self._telemetry.enter_context(ProcessTelemetry(telemetry_config, WORKER_ROLE))
        enable_trainer_batch_invariance(cfg.trainer.algorithm.batch_invariant)

    @property
    def device(self) -> torch.device:
        """Device that holds this worker's model and training tensors."""
        return torch.device("cuda", torch.cuda.current_device())

    def close_telemetry(self) -> None:
        """Record the terminal event and drain queued telemetry; ray.kill would drop both."""
        self._telemetry.close()

    def init_model(self, *args, **kwargs):
        """Initialize worker state (model, and optimizer if applicable) on worker."""
        raise NotImplementedError()

    def empty_cache(self) -> None:
        """Empty GPU memory cache on Worker's CUDA device"""
        torch.cuda.empty_cache()

    def offload_to_cpu(self, pin_memory=True, non_blocking=True):
        """Offload all worker state to CPU.

        After this function runs, only temporary reserved memory and torch's pre-loaded cuda kernels (~ GB) will remain

        Args:
            pin_memory: Whether to use pinned/ paged-locked memory on CPU
            non_blocking: Whether the operation is non-blocking
        """
        raise NotImplementedError()

    def backload_to_gpu(self, non_blocking=True):
        """Backload worker state to GPU

        Args:
            non_blocking: Whether the operation is non-blocking
        """
        raise NotImplementedError()

    def get_cuda_memory(self) -> Dict[str, Any]:
        """Get CUDA memory usage on worker's CUDA device."""
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        return {
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
            "free": free,
            "total": total,
        }

    def save_memory_snapshot(self, global_step=None, local_step=None):
        """Save a snapshot of memory usage on the Worker's CUDA device.

        .. note::
            This function should be called on all the ranks in the worker group simultaneously.
        """
        rank = torch.distributed.get_rank()
        save_path = os.path.join(self.cfg.trainer.ckpt_path, "memory_snapshots")
        if self._local_rank == 0 and not io.exists(save_path):
            io.makedirs(save_path, exist_ok=True)
        torch.distributed.barrier()
        if global_step is None or local_step is None:
            file_name = f"policy_rank_{rank}.pickle"
        else:
            file_name = f"policy_rank_{rank}_training_step_{global_step}_{local_step}.pickle"
        record_memory_path = os.path.join(save_path, file_name)
        if io.exists(record_memory_path):
            # seeing issues if we don't remove the file first
            io.remove(record_memory_path)
        torch.cuda.memory._dump_snapshot(record_memory_path)

    async def init_weight_sync_state(self, inference_engine_client: InferenceEngineClient):
        """Initialize state for weight syncing with Inference Engine Client

        Initializes a custom process group with the rank 0 Worker and all the inference engine ranks
        for weight syncing.

        .. note::
            This function should be called on all the ranks in the worker group simultaneously.
        """
        assert inference_engine_client is not None

        if torch.distributed.get_rank() == 0:
            # Debug: understand where master_addr comes from
            import socket as sock_module

            hostname = sock_module.gethostname()
            global_node = ray._private.worker._global_node
            global_node_ip = global_node.node_ip_address if global_node else "None"

            master_addr = ray._private.services.get_node_ip_address()

            logging.info(f"[weight-sync] hostname={hostname}")
            logging.info(f"[weight-sync] _global_node.node_ip_address={global_node_ip}")
            logging.info(f"[weight-sync] get_node_ip_address()={master_addr}")

            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]
            logging.info(f"[weight-sync] Using master_addr={master_addr}, master_port={master_port}")

            num_inference_engines, tensor_parallel_size, pipeline_parallel_size, data_parallel_size = (
                self.cfg.generator.num_inference_engines,
                self.cfg.generator.inference_engine_tensor_parallel_size,
                self.cfg.generator.inference_engine_pipeline_parallel_size,
                self.cfg.generator.inference_engine_data_parallel_size,
            )
            world_size = num_inference_engines * tensor_parallel_size * pipeline_parallel_size * data_parallel_size + 1

            backend = self.cfg.generator.weight_sync_backend

            override_existing = False if self.cfg.generator.override_existing_update_group == "disable" else True
            group_name = "skyrl"
            self._model_update_group_name = group_name

            tasks = []
            tasks.append(
                inference_engine_client.init_weight_update_communicator(
                    master_addr=master_addr,
                    master_port=master_port,
                    rank_offset=1,
                    world_size=world_size,
                    group_name=group_name,
                    backend=backend,
                    override_existing=override_existing,
                )
            )

            tasks.append(
                asyncio.to_thread(
                    init_custom_process_group,
                    backend=backend,
                    init_method=get_tcp_url(master_addr, master_port),
                    world_size=world_size,
                    rank=0,
                    group_name=group_name,
                )
            )
            results = await asyncio.gather(*tasks)
            self._model_update_group = results[-1]

            # # Register signal handlers for termination only on rank 0
            # NOTE (sumanthrh): This doesn't work yet, and is thus commented out.
            # The better way is to just have this specified in __del__, but there is
            # no guarattee that __del__ will be called in general. Ray also doesn't
            # explictly call __del__ when the actor shuts down.
            # It's commented out so that we can fix this in the future.
            # atexit.register(self._handle_termination)

        torch.distributed.barrier()

    def forward(
        self,
        data: TrainingInputBatch,
    ) -> TrainingOutputBatch:
        """Run forward pass on the input batch in inference mode.

        This is a wrapper around `_forward_micro_batch` that runs in micro batches of `cfg.trainer.micro_forward_batch_size_per_gpu`.
        """
        # Record which ranks enter the synchronous forward after weight sync.
        logger.info(f"WORKER_FORWARD_ENTER rank={self._rank}")
        # R3 RESIDENT-SET per-rank marker: lets us SEE the resident routed-experts
        # chunk land on EVERY rank (ungated; no-op when the batch carries no routes).
        log_r3_resident_set(self._rank, data)
        # run in micro batches of cfg.trainer.micro_forward_batch_size_per_gpu
        # TODO (sumanthrh): this can be in the policy/critic impl if the micro batch size can be specific to policy, critic, etc.
        micro_batches = data.chunk(self.cfg.trainer.micro_forward_batch_size_per_gpu)

        outputs = []
        for micro_batch in micro_batches:
            outputs.append(self._forward_micro_batch(micro_batch))
        output = TrainingOutputBatch.cat(outputs)
        if output.device is not None and output.device != torch.device("cpu"):
            output = output.to("cpu")
        return output

    async def barrier_all(self) -> None:
        """Drain each actor event loop before the next synchronous forward.

        The weight-sync coroutine can still occupy some rank loops when the
        driver dispatches a synchronous forward. Yielding once lets those
        coroutines finish, then the world barrier aligns all ranks before
        model collectives begin.
        """
        logger.info(f"WORKER_DRAIN_BARRIER rank={self._rank}")
        if self._world_size > 1 and torch.distributed.is_initialized():
            # Yield to the event loop so any lingering weight-sync coroutine task is
            # fully retired and the loop is idle before we issue the collective.
            await asyncio.sleep(0)

            def _sync_barrier() -> None:
                torch.cuda.synchronize()
                torch.distributed.barrier()

            # Run the blocking collective off the event-loop thread so it can't itself
            # re-occupy the loop (head-of-line-block a subsequent dispatch).
            await asyncio.to_thread(_sync_barrier)

    def _forward_micro_batch(self, micro_batch: TrainingInputBatch) -> TrainingOutputBatch:
        raise NotImplementedError()


# adapted from OpenReasonerZero: https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/blob/main/orz/ppo/actors.py
class PPORayActorGroup:
    """
    A group of ray actors
    Functions start with 'async' should return list of object refs

    Args:
        cfg: config object for workers
        num_nodes (int): Number of nodes for this actor group.
        num_gpus_per_node (int): Number of gpus for this actor group.
        ray_actor_type (Type[Worker]): PPO model type that this actor group serve on.
        pg (PlacementGroup, optional): Placement group to schedule actor on.
            If none, create new placement group automatically. Defaults to None.
        num_gpus_per_actor (float, optional): Number of gpus allocated for each actor.
            If < 1.0, multiple models can share same gpu. Defaults to 1.
        actor_env_vars: Environment variables applied to each actor before process-group initialization.
    """

    def __init__(
        self,
        cfg,
        num_nodes,
        num_gpus_per_node,
        ray_actor_type: Type[Worker],
        pg: Optional[PlacementGroup] = None,
        num_gpus_per_actor: float = 1.0,
        resources: Optional[Dict[str, float]] = None,
        num_resources_per_node: Optional[int] = None,
        colocate_all: bool = False,
        sequence_parallel_size: int = 1,
        record_memory: bool = False,
        pin_to_ray_gpu_id: bool = False,
        force_cvd_mask: bool = False,
        actor_env_vars: Optional[Dict[str, str]] = None,
    ) -> None:
        self.cfg = cfg
        self._num_nodes = num_nodes
        self._num_gpus_per_node = num_gpus_per_node
        self.ray_actor_type = ray_actor_type
        # When True, each actor pins its CUDA device to ray.get_gpu_ids()[0]
        # (reliable only when each actor owns a dedicated {GPU:1} bundle, i.e.
        # the per-GPU-bundle policy PG). See DistributedTorchRayActor.__init__.
        self._pin_to_ray_gpu_id = pin_to_ray_gpu_id
        # When True (and pin_to_ray_gpu_id), each actor additionally MASKS
        # CUDA_VISIBLE_DEVICES to its single physical GPU + forces PCI_BUS_ID
        # ordering before any CUDA init — the deterministic EP×DP pin.
        self._force_cvd_mask = force_cvd_mask

        # custom resources, see https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
        self._resources = resources
        self._num_resources_per_node = num_resources_per_node

        self.colocate_all = colocate_all
        self.sequence_parallel_size = sequence_parallel_size
        self.record_memory = record_memory
        self._actor_runtime_env = {"env_vars": actor_env_vars} if actor_env_vars else None
        self._initiate_actors(pg, num_gpus_per_actor)

    def _initiate_actors(self, pg: Optional[PlacementGroup], num_gpus_per_actor: float):
        """Initialize Ray actors in the worker group.

        Args:
            pg: The placement group for the worker group
            num_gpus_per_actor: The number of gpus to allocate per actor.
        """
        world_size = self._num_nodes * self._num_gpus_per_node
        if self.colocate_all:
            assert pg is not None, (
                "if colocate_all is True, the shared placement group must be provided to PPORayActorGroup"
            )
            pg_data = placement_group_table(pg)
            assert len(pg_data["bundles"]) == world_size, (
                "if colocate_all is True, the number of bundles in the shared placement group must match the world size"
            )

        reordered_bundle_indices = []
        actor_options = {"runtime_env": self._actor_runtime_env} if self._actor_runtime_env else {}
        if pg is not None:
            pg_data = placement_group_table(pg)
            should_reorder_bundles = len(pg_data["bundles"]) == world_size
            if should_reorder_bundles:
                reordered_bundle_indices = get_reordered_bundle_indices(pg)

        if self._num_gpus_per_node > 1 and pg is None:
            bundles = [{"GPU": self._num_gpus_per_node, "CPU": self._num_gpus_per_node} for _ in range(self._num_nodes)]
            if self._resources:
                resources_name = list(self._resources.keys())[0]
                for i in range(len(bundles)):
                    bundles[i][resources_name] = self._num_resources_per_node

            pg = placement_group(bundles, strategy="PACK")
            get_ray_pg_ready_with_timeout(
                pg,
                timeout=int(
                    self.cfg.trainer.distributed.get(
                        "placement_group_timeout_seconds", DEFAULT_RAY_PLACEMENT_GROUP_TIMEOUT_SECONDS
                    )
                ),
            )
        if pg:
            master_actor = self.ray_actor_type.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                resources=self._resources,
                **actor_options,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=reordered_bundle_indices[0] if reordered_bundle_indices else 0,
                ),
            ).remote(
                cfg=self.cfg,
                world_size=world_size,
                rank=0,
                local_rank=0,
                master_addr=None,
                master_port=None,
                sequence_parallel_size=self.sequence_parallel_size,
                record_memory=self.record_memory,
                pin_to_ray_gpu_id=self._pin_to_ray_gpu_id,
                force_cvd_mask=self._force_cvd_mask,
            )
        else:
            master_actor = self.ray_actor_type.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                resources=self._resources,
                **actor_options,
            ).remote(
                cfg=self.cfg,
                world_size=world_size,
                rank=0,
                local_rank=0,
                master_addr=None,
                master_port=None,
                sequence_parallel_size=self.sequence_parallel_size,
                record_memory=self.record_memory,
                pin_to_ray_gpu_id=self._pin_to_ray_gpu_id,
                force_cvd_mask=self._force_cvd_mask,
            )
        self._actor_handlers = [master_actor]
        # Create worker actors
        if world_size > 1:
            master_addr, master_port = ray.get(master_actor.get_master_addr_port.remote())
            for rank in range(1, world_size):
                local_rank = rank % self._num_gpus_per_node

                if pg:
                    worker_actor = self.ray_actor_type.options(
                        num_cpus=num_gpus_per_actor,
                        num_gpus=num_gpus_per_actor,
                        resources=self._resources,
                        **actor_options,
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=pg,
                            placement_group_bundle_index=(
                                reordered_bundle_indices[rank]
                                if reordered_bundle_indices
                                else rank // self._num_gpus_per_node
                            ),
                        ),
                    ).remote(
                        cfg=self.cfg,
                        world_size=world_size,
                        rank=rank,
                        local_rank=local_rank,
                        master_addr=master_addr,
                        master_port=master_port,
                        sequence_parallel_size=self.sequence_parallel_size,
                        record_memory=self.record_memory,
                        pin_to_ray_gpu_id=self._pin_to_ray_gpu_id,
                        force_cvd_mask=self._force_cvd_mask,
                    )
                else:
                    worker_actor = self.ray_actor_type.options(
                        num_cpus=num_gpus_per_actor,
                        num_gpus=num_gpus_per_actor,
                        resources=self._resources,
                        **actor_options,
                    ).remote(
                        cfg=self.cfg,
                        world_size=world_size,
                        rank=rank,
                        local_rank=local_rank,
                        master_addr=master_addr,
                        master_port=master_port,
                        sequence_parallel_size=self.sequence_parallel_size,
                        record_memory=self.record_memory,
                        pin_to_ray_gpu_id=self._pin_to_ray_gpu_id,
                        force_cvd_mask=self._force_cvd_mask,
                    )
                self._actor_handlers.append(worker_actor)

        # Initialize process group
        logger.info("Initializing process group for RayActorGroup")
        ray.get([actor.init_worker_process_group.remote() for actor in self._actor_handlers])
        logger.info("Initialized process group for RayActorGroup")
        self.actor_infos = [ActorInfo(actor, ray.get(actor.get_mesh_rank.remote())) for actor in self._actor_handlers]
        logger.info(f"Mesh Ranks: {[actor_info.rank for actor_info in self.actor_infos]}")

    def _dispatch(self, dispatch_class, method_name, *args, **kwargs):
        settings = DispatchSettings(
            r3_transport=R3Transport(self.cfg.generator.r3_transport),
            r3_dispatch_put_timeout_seconds=float(self.cfg.generator.r3_dispatch_put_timeout_seconds),
        )
        return dispatch_class.dispatch(self.actor_infos, method_name, *args, settings=settings, **kwargs)

    def async_init_model(
        self,
        *args,
        **kwargs,
    ) -> List[ObjectRef]:
        """Asynchronously initialize worker state (model, and optimizer if applicable) from model path on all the workers.

        Returns:
            A list of ray object refs.
        """
        return [actor.init_model.remote(*args, **kwargs) for actor in self._actor_handlers]

    def offload_to_cpu(self, nonblocking=False, offload_optimizer=True, offload_model=True):
        """Offload all worker state to CPU.

        Args:
            nonblocking: Whether this operation is synchronous or asynchronous.
            If `nonblocking=True`, then the function returns a list of object refs.
        """
        refs = [
            actor.offload_to_cpu.remote(offload_optimizer=offload_optimizer, offload_model=offload_model)
            for actor in self._actor_handlers
        ]
        if nonblocking:
            return refs
        return ray.get(refs)

    def backload_to_gpu(self, nonblocking=False, backload_optimizer=True, backload_model=True):
        """Backload worker state to GPU

        Args:
            nonblocking: Whether this operation is synchronous or asynchronous.
            If `nonblocking=True`, then the function returns a list of ObjectRefs.
        """
        refs = [
            actor.backload_to_gpu.remote(backload_optimizer=backload_optimizer, backload_model=backload_model)
            for actor in self._actor_handlers
        ]
        if nonblocking:
            return refs
        return ray.get(refs)

    def run_method(self, dispatch_type: str, method_name: str, *args, **kwargs) -> Optional[TrainingOutputBatch]:
        """Run a method on all actors using specified dispatch type synchronously.

        The method should either return `None` or a `TrainingOutputBatch` object.

        Args:
            dispatch_type: Type of dispatch to use ("mesh" or "pass_through")
            method_name: Name of the method to call on actors
            *args: Positional arguments to pass to the method
            **kwargs: Keyword arguments to pass to the method

        Returns:
            Collect results from all the actors.
        """
        dispatch_class: Dispatch = DispatchRegistry.get(dispatch_type)
        # validate the dispatch args to be sent to `.dispatch`
        args, kwargs = dispatch_class.validate_dispatch_args(*args, **kwargs)

        # Dispatch the method call
        object_refs = self._dispatch(dispatch_class, method_name, *args, **kwargs)
        # Collect results from all the actors
        ret = dispatch_class.sync_collect(self.actor_infos, object_refs)
        return ret

    def async_run_ray_method(self, dispatch_type: str, method_name: str, *args, **kwargs) -> List[ObjectRef]:
        """Run a method on all actors using specified dispatch type asynchronously.

        Args:
            dispatch_type: Type of dispatch to use ("mesh" or "pass_through")
            method_name: Name of the method to call on actors
            *args: Positional arguments to pass to the method
            **kwargs: Keyword arguments to pass to the method

        Returns:
            List of object references
        """
        dispatch_class: Dispatch = DispatchRegistry.get(dispatch_type)
        # validate the dispatch args to be sent to `.dispatch`
        args, kwargs = dispatch_class.validate_dispatch_args(*args, **kwargs)

        # Dispatch the method call
        object_refs = self._dispatch(dispatch_class, method_name, *args, **kwargs)
        return object_refs

    async def async_run_method(
        self, dispatch_type: str, method_name: str, *args, **kwargs
    ) -> Optional[TrainingOutputBatch]:
        """Run a method on all actors using specified dispatch type in an asyncio-compatible way.

        Args:
            dispatch_type: Type of dispatch to use ("mesh" or "pass_through")
            method_name: Name of the method to call on actors
            *args: Positional arguments to pass to the method
            **kwargs: Keyword arguments to pass to the method

        Returns:
            TrainingOutputBatch: concatenated results from all actors
        """
        dispatch_class: Dispatch = DispatchRegistry.get(dispatch_type)
        # validate the dispatch args to be sent to `.dispatch`
        args, kwargs = dispatch_class.validate_dispatch_args(*args, **kwargs)

        # Dispatch the method call
        object_refs = self._dispatch(dispatch_class, method_name, *args, **kwargs)
        return await dispatch_class.async_collect(self.actor_infos, object_refs)

    def kill_actors(self, no_restart: bool = True) -> None:
        """Kill all Ray actors in this group for proper teardown.

        Args:
            no_restart: If True, prevents Ray from restarting the actors.
        """
        # ray.kill skips the actor's atexit handlers; a dead actor only costs the timeout.
        drains = [actor.close_telemetry.remote() for actor in self._actor_handlers]
        ray.wait(drains, num_returns=len(drains), timeout=TELEMETRY_DRAIN_TIMEOUT_SECONDS)
        for actor in self._actor_handlers:
            try:
                ray.kill(actor, no_restart=no_restart)
            except Exception:
                pass  # Actor may already be dead


class PolicyWorkerBase(Worker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model: nn.Module = None
        self.scheduler: LRScheduler = None
        self.optimizer: Optimizer = None
        self.strategy: DistributedStrategy = None
        self.record_memory: bool = False
        self.mesh_rank: MeshRank = None
        self.policy_loss_fn: Callable = PolicyLossRegistry.get(self.cfg.trainer.algorithm.policy_loss_type)
        self._grug_query_bias_window: GrugQueryBiasWindow | None = None
        self._policy_train_spans: bool = self.cfg.trainer.policy_train_spans
        self._memory = LearnerCudaMetrics(enabled=self._policy_train_spans, rank=self._rank)
        self._model_version_step: int | None = None

    async def _begin_vllm_layerwise_weight_reload(self, inference_engine_client, *, enabled: bool) -> None:
        """Open a rank-synchronized vLLM reload around a streamed weight update."""
        if not enabled:
            return
        if torch.distributed.get_rank() == 0:
            await inference_engine_client.begin_weight_reload()
        torch.distributed.barrier()

    async def _finish_vllm_layerwise_weight_reload(self, inference_engine_client, *, enabled: bool) -> None:
        """Finalize all streamed weights and apply vLLM's kernel-layout transforms once."""
        if not enabled:
            return
        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            await inference_engine_client.finish_weight_reload()

    def _normalize_mini_batch_size(self):
        """
        Normalize mini batch sizes to per-gpu mini batch sizes..
        """
        if not hasattr(self, "mesh_rank") or self.mesh_rank is None:
            raise RuntimeError("mesh_rank must be initialized before calling _normalize_mini_batch_size()")

        self.policy_mini_batch_size_per_gpu = per_data_parallel_batch_size(
            self.cfg.trainer.policy_mini_batch_size,
            optimization_samples_per_prompt(self.cfg),
            self.mesh_rank.dp_size,
        )

    def _grug_causal_lm(self) -> GrugMoeForCausalLM | None:
        """Return the local Grug CausalLM through the HF wrapper, if present."""

        causal_lm = getattr(self.model, "model", self.model)
        return causal_lm if isinstance(causal_lm, GrugMoeForCausalLM) else None

    def _drain_r3_decentral_stagger(self, train_data: TrainingInputBatch) -> None:
        """Co-arrival barrier for the staggered R3-decentral dispatch path.

        Under ``r3_transport=decentral`` the mesh dispatch relocates each
        dp-group's multi-gigabyte routed-experts chunk to its consumer node
        sequentially, so members of a model-shard group begin ``ppo_train``
        minutes apart. The fastest member would otherwise enter compute and
        issue the first shard-wide collective alone, blowing the shard
        sub-process-group timeout. Call at the very top of ``ppo_train``:
        every rank's routes have already materialized there (the argument
        ObjectRef must have resolved for the method to run), so one barrier
        over the DEFAULT/WORLD process group — whose timeout must exceed the
        shard subgroups' — absorbs the arrival spread before any compute.

        Timing-only (no tensor is touched) and gated to the R3-decentral path
        with routes present, so every other configuration is unchanged.
        """
        staggered = self.cfg.generator.r3_transport == R3Transport.DECENTRAL and train_data.routed_experts is not None
        if staggered and self._world_size > 1 and torch.distributed.is_initialized():
            # Ungated per-rank marker: the timestamp cluster at release proves
            # co-arrival; the first shard collective must not time out after it.
            logger.info(f"WORKER_PPO_TRAIN_DRAIN_BARRIER rank={self._rank}")
            torch.cuda.synchronize()
            torch.distributed.barrier()

    def ppo_train(self, train_data: TrainingInputBatch) -> TrainingOutputBatch:
        step = int(train_data.metadata["global_step"])
        with self._memory.span("ppo_train", step=step):
            timing = PhaseBreakdown("ppo_train", enabled=self._policy_train_spans)
            outcome = "failure"
            try:
                output = self._ppo_train_impl(train_data, timing)
                outcome = "success"
            finally:
                timing.publish(
                    clock_domain="cpu_dispatch_wall",
                    attributes={
                        "backend": "megatron",
                        "outcome": outcome,
                        "rank": str(self._rank),
                        "role": WORKER_ROLE,
                        "step": str(step),
                    },
                )
        self._model_version_step = step
        return output

    async def broadcast_to_inference_engines(self, inference_engine_client):
        with self._memory.span("broadcast_to_inference_engines", step=self._model_version_step):
            return await self._broadcast_to_inference_engines(inference_engine_client)

    def _ppo_train_impl(self, train_data: TrainingInputBatch, timing: PhaseBreakdown) -> TrainingOutputBatch:
        self._drain_r3_decentral_stagger(train_data)

        global_step = train_data.metadata["global_step"]

        dataloader = TrainingBatchIterator(train_data, self.cfg.trainer.micro_train_batch_size_per_gpu)

        # Clear fragmented GPU memory before training to avoid OOM at step boundaries
        # (matches CriticWorkerBase.ppo_train behavior)
        torch.cuda.empty_cache()

        micro_batches_per_mini_batch = gradient_accumulation_steps(
            self.policy_mini_batch_size_per_gpu,
            self.cfg.trainer.micro_train_batch_size_per_gpu,
        )
        # The number of steps (over micro batches) to accumulate gradients before taking an optimizer step.
        accumulation_steps = micro_batches_per_mini_batch

        status_list = []
        all_metrics = defaultdict(list)
        policy_update_steps = 0
        grug_causal_lm = self._grug_causal_lm()
        grug_query_bias_update = resolve_grug_query_bias_update(self.cfg.trainer.policy)
        if grug_causal_lm is None and grug_query_bias_update.enabled:
            raise ValueError("Grug query-bias updates require a Grug policy model")
        grug_query_bias_updates_enabled = grug_query_bias_update.enabled
        grug_capture_plan = None
        if grug_query_bias_updates_enabled:
            micro_batch_size = self.cfg.trainer.micro_train_batch_size_per_gpu
            ep_size = self.strategy.ep_size
            ep_rank = int(self.strategy.device_mesh.get_local_rank(mesh_dim="ep")) if ep_size > 1 else 0
            shard_layout = GrugQueryBiasShardLayout(
                micro_batch_size=micro_batch_size,
                accumulation_steps=accumulation_steps,
                ep_size=ep_size,
                ep_rank=ep_rank,
            )
            grug_capture_plan = GrugQueryBiasCapturePlan.build(
                train_data["attention_mask"],
                shard_layout,
            )

        for epoch in range(self.cfg.trainer.update_epochs_per_batch):
            pbar = tqdm(
                dataloader,
                desc=f"Policy Train epoch [{epoch + 1}/{self.cfg.trainer.update_epochs_per_batch}]",
                disable=not self.strategy.is_rank_0(),
            )
            for local_step, experience in enumerate(pbar):
                if local_step % accumulation_steps == 0:
                    chunks = dataloader.chunks(local_step, local_step + accumulation_steps)
                    counts = step_counts(
                        [
                            policy_data_weights(
                                chunk["loss_mask"],
                                chunk.get("response_span_tags"),
                                self.cfg.trainer.algorithm.think_token_weight,
                            )
                            for chunk in chunks
                        ],
                        [chunk["loss_mask"] for chunk in chunks],
                        [
                            chunk["loss_mask"] * chunk["teacher_valid_mask"]
                            for chunk in chunks
                            if chunk.get("teacher_valid_mask") is not None
                        ],
                        [chunk["advantages"] for chunk in chunks],
                        self.cfg.trainer.algorithm.max_seq_len,
                        lambda value: self.strategy.all_reduce(value, op="sum"),
                    )
                if grug_query_bias_updates_enabled and local_step % accumulation_steps == 0:
                    assert grug_capture_plan is not None
                    assert grug_causal_lm is not None
                    window_end = local_step + accumulation_steps
                    valid_tokens = sum(grug_capture_plan.valid_token_counts[local_step:window_end])
                    updater = _grug_query_bias_updater(
                        grug_causal_lm,
                        valid_tokens,
                        grug_query_bias_update,
                    )
                    self._grug_query_bias_window = GrugQueryBiasWindow(
                        grug_causal_lm,
                        grug_capture_plan,
                        updater,
                    )
                diagnostic_mesh = self.strategy.device_mesh if _phase_diagnostics.enabled() else None
                with _phase_diagnostics.region(
                    diagnostic_mesh,
                    kind=_phase_diagnostics.CollectiveRegionKind.POLICY_TRAINING_STEP,
                    rank=self._rank,
                    metadata=_phase_diagnostics.CollectiveRegionMetadata(
                        global_step=global_step,
                        local_step=local_step,
                    ),
                ):
                    status = self.training_step(
                        experience,
                        global_step,
                        local_step,
                        accumulation_steps,
                        counts,
                    )
                policy_update_steps += 1

                # for DP
                # TODO (sumanthrh): this assumes all workers are data parallel.
                # We assume that outputs are replicated within tp or sp group, otherwise this is not correct.
                status = self.strategy.all_reduce(status)

                # weighted mean for kl
                # TODO (sumanthrh): this weighted mean is no longer correct since we use the max response length in the batch.
                # we can log this in the driver
                # if "kl" in status:
                #     status["kl"] *= status["response_length"]
                #     status["kl"] /= status["response_length"]

                status_list.append(status)
                for k, v in status.items():
                    all_metrics[k].append(v)
                pbar.set_postfix(policy_progress_metrics(status))

        torch.distributed.barrier()
        status_mean = policy_training_metrics(all_metrics, policy_update_steps / accumulation_steps)

        # should return an `TrainingOutputBatch`
        output = TrainingOutputBatch()
        output.metadata = {"train_status": status_mean}
        return output

    def _distillation_student_logprobs(
        self,
        experience: Experience,
        output: dict[str, torch.Tensor],
        num_actions: int,
    ) -> Optional[torch.Tensor]:
        if experience.distillation is None:
            return None
        token_ids = experience.distillation.student_token_ids()
        if token_ids is None:
            return None
        if self.cfg.trainer.use_sample_packing or self.sequence_parallel_size != 1:
            raise ValueError(
                "selected-ID distillation does not support sample packing, sequence parallelism, or context parallelism"
            )
        response_logits = output["logits"][:, -num_actions - 1 : -1]
        if response_logits.shape[:2] != experience.distillation.valid_mask.shape:
            raise ValueError(
                "selected-ID distillation requires response-aligned full-vocabulary logits; "
                f"got {tuple(response_logits.shape[:2])} for "
                f"{tuple(experience.distillation.valid_mask.shape)} response coordinates"
            )
        return student_topk_logprobs(response_logits, token_ids)

    def training_step(
        self,
        experience: Experience,
        global_step: int,
        local_step: int,
        accumulation_steps: int,
        counts: StepCounts,
    ) -> Dict[str, float]:
        """
        Perform one micro-batch of training, accumulate gradients, and step the optimizer only after `accumulation_steps` micro-batches.
        """
        _phase_diagnostics.log_phase(_phase_diagnostics.CollectivePhase.TRAINING_STEP_ENTER)
        self.model.train()
        experience.to_device(self.device)

        sequences = experience.sequences
        old_action_log_probs = experience.action_log_probs
        base_action_log_probs = (
            experience.base_action_log_probs if experience.base_action_log_probs is not None else None
        )
        advantages = experience.advantages
        num_actions = experience.num_actions
        attention_mask = experience.attention_mask
        loss_mask = experience.loss_mask
        rollout_action_logprobs = experience.rollout_logprobs
        rollout_routed_experts = experience.rollout_routed_experts
        response_span_tags = experience.response_span_tags

        grug_causal_lm = self._grug_causal_lm()
        grug_query_bias_window = self._grug_query_bias_window
        grug_capture_started = bool(
            grug_query_bias_window
            and grug_query_bias_window.begin_microbatch(
                attention_mask,
                local_step,
            )
        )

        # The model wrapper controls its own internal precision where needed.
        _phase_diagnostics.start_phase(_phase_diagnostics.CollectivePhase.MODEL_FORWARD_ENTER)
        with torch.autocast(dtype=torch.bfloat16, device_type=self.device.type):
            # actor loss
            action_log_probs, output = self.model(
                sequences,
                num_actions,
                attention_mask=attention_mask,
                temperature=self.cfg.generator.sampling_params.temperature,
                return_output=True,
                compute_entropy=True,
                entropy_requires_grad=self.cfg.trainer.algorithm.use_entropy_loss,
                rollout_routed_experts=rollout_routed_experts,
            )
            if grug_capture_started:
                assert grug_query_bias_window is not None
                grug_query_bias_window.observe_microbatch()
            token_entropy = output["entropy"][:, -num_actions - 1 : -1]
            sparse_student_logprobs = self._distillation_student_logprobs(experience, output, num_actions)
            teacher = None
            if experience.distillation is not None:
                assert sparse_student_logprobs is not None
                teacher = TopKTeacherBatch(
                    experience.distillation,
                    sparse_student_logprobs,
                    topk_loss_params(self.cfg.trainer.algorithm),
                    output["logits"].shape[-1],
                )
            batch = build_objective_micro_batch(
                action_log_probs=action_log_probs,
                old_action_log_probs=old_action_log_probs,
                base_action_log_probs=base_action_log_probs,
                advantages=advantages,
                loss_mask=loss_mask,
                rollout_logprobs=rollout_action_logprobs,
                correction_weights=experience.correction_weights,
                response_span_tags=response_span_tags,
                token_entropy=token_entropy,
                think_token_weight=self.cfg.trainer.algorithm.think_token_weight,
                teacher=teacher,
                pair_roles=experience.pair_roles,
            )
            objective = compute_policy_objective(
                batch,
                loss=self.policy_loss_fn,
                counts=counts,
                config=self.cfg.trainer.algorithm,
                loss_scale=self.mesh_rank.dp_size,
                report_scale=accumulation_steps * self.mesh_rank.dp_size,
            )
        _phase_diagnostics.log_phase(_phase_diagnostics.CollectivePhase.MODEL_FORWARD_EXIT)
        loss = objective.optimization_loss
        policy_loss = objective.rows.policy
        entropy = objective.rows.entropy
        kl_loss = objective.rows.kl
        _phase_diagnostics.start_phase(_phase_diagnostics.CollectivePhase.BACKWARD_ENTER)
        self.strategy.backward(loss, self.model, self.optimizer)
        _phase_diagnostics.log_phase(_phase_diagnostics.CollectivePhase.BACKWARD_EXIT)

        # Stage-7 P3 recompute-safety: the training forward DEFERS the router-replay
        # teardown to here (after backward) so gradient-checkpoint recompute still
        # sees the installed controller -> no CheckpointError. Idempotent + a no-op
        # when replay is disabled (flag-off / a3 path byte-identical).
        teardown_replay = getattr(self.model, "teardown_router_replay", None)
        if teardown_replay is not None:
            teardown_replay()

        # Per-token log-ratio diagnostics — v5 accumulates across all micro-batches
        # of the global_step (sum + count + concat-of-topk) and finalizes once at
        # the end. v4 ran only on the LAST micro-batch, which with
        # `update_epochs_per_batch=1` gave a noisy single-sample view (action_log_probs
        # is always computed BEFORE optimizer_step within a step, so the per-batch
        # ratio reflects only vLLM-to-trainer precision noise — averaging across all
        # micro-batches makes that signal more representative). The final scalar
        # dict has the same wandb keys as v4 so the downstream per-key
        # all_reduce(status) stays keyset-compatible.
        if local_step % accumulation_steps == 0 or getattr(self, "_log_ratio_monitor", None) is None:
            self._log_ratio_monitor = LogRatioMonitor(action_log_probs.device)
        self._log_ratio_monitor.add(action_log_probs, old_action_log_probs, loss_mask)

        grad_norm = None
        ratio_diag = {}
        optimizer_step_succeeded = False
        if (local_step + 1) % accumulation_steps == 0:
            grad_norm = self.strategy.optimizer_step(
                self.optimizer,
                self.model,
                self.scheduler,
                name="actor",
            )
            optimizer_step_succeeded = (
                bool(self.strategy.last_optimizer_step_succeeded) if grug_causal_lm is not None else True
            )
            if grad_norm is not None:
                grad_norm = grad_norm.detach().cpu().item()

            if grug_query_bias_window is not None:
                grug_query_bias_window.finish(optimizer_step_succeeded=optimizer_step_succeeded)
                self._grug_query_bias_window = None

            # Finalize the accumulated diagnostics. Every rank must emit identical
            # keys (the full set from _log_ratio_diag_zero_metrics) — the per-key
            # all_reduce(status) deadlocks otherwise (killed v2/v3 of this diag).
            ratio_diag = self._log_ratio_monitor.metrics()
            self._log_ratio_monitor = None

        if self.record_memory:
            self.save_memory_snapshot(global_step, local_step)

        # status
        status = {
            "final_loss": objective.optimization_loss.item() * accumulation_steps,
            "policy_loss": policy_loss.item(),
            "policy_lr": self.scheduler.get_last_lr()[0],
            "policy_entropy": entropy.item(),
        }
        status.update(objective.metrics)
        # Per-token log-ratio diagnostics — visibility into which tokens
        # carry the gradient signal (heaviest-hit token, fraction of tokens
        # with large probability changes, per-position aggregations).
        # Trainer prefixes these with "policy/" before sending to wandb.
        status.update(ratio_diag)
        if self.cfg.trainer.algorithm.use_kl_loss:
            status["policy_kl"] = kl_loss.item()

        if grad_norm is not None:
            status["raw_grad_norm"] = grad_norm
        if grug_causal_lm is not None and (local_step + 1) % accumulation_steps == 0:
            status["optimizer_step_succeeded"] = float(optimizer_step_succeeded)

        for k, v in experience.info.items():
            if k == "kl":
                # just use the same value as loss if available
                status[k] = kl_loss.item() if isinstance(kl_loss, torch.Tensor) else status["policy_kl"]
            else:
                status[k] = v.mean().item() if isinstance(v, torch.Tensor) else v

        status["response_length"] = num_actions
        _phase_diagnostics.log_phase(_phase_diagnostics.CollectivePhase.TRAINING_STEP_EXIT)
        return status

    def save_checkpoint(self, ckpt_dir: Path, tokenizer=None):
        upload = self.strategy.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            ckpt_dir=ckpt_dir,
            node_local_rank=self.get_node_local_rank(),
            tokenizer=tokenizer,
            client_state={},
        )
        self._start_checkpoint_upload(upload)

    def load_checkpoint(
        self,
        ckpt_dir: Path,
        load_training_state: bool = True,
    ):
        _, states = self.strategy.load_checkpoint(
            model=self.model,
            optimizer=self.optimizer if load_training_state else None,
            scheduler=self.scheduler if load_training_state else None,
            ckpt_dir=ckpt_dir,
            load_training_state=load_training_state,
        )
        return states

    def save_hf_model(self, export_dir: str, tokenizer):
        # Save model in HuggingFace safetensors format
        self.strategy.save_hf_model(
            self.model,
            export_dir,
            tokenizer=tokenizer,
        )

    def _forward_micro_batch(self, micro_batch: TrainingInputBatch) -> TrainingOutputBatch:
        device = self.device
        micro_batch.to(device)
        self.model.eval()
        sequences = micro_batch["sequences"]
        response_length = micro_batch.metadata["response_length"]
        attention_mask = micro_batch["attention_mask"]
        # MoE router-replay (R3): replay the SAME captured routing as the training
        # forward so the old-logprob (pi_old) pass and the training pass share the
        # identical forward path. Without this the old-logprob pass falls through to
        # NATIVE top-k routing while training uses REPLAY routing -> different
        # experts -> a pathological step-1 importance ratio. Absent key (8B /
        # router-replay off) -> None -> stock native forward, unchanged.
        rollout_routed_experts = micro_batch.routed_experts_tensor()

        with torch.no_grad(), torch.autocast(dtype=torch.bfloat16, device_type=self.device.type):
            policy_logprob = self.model(
                sequences,
                response_length,
                attention_mask,
                return_output=False,
                temperature=self.cfg.generator.sampling_params.temperature,
                rollout_routed_experts=rollout_routed_experts,
            )
        policy_logprob = policy_logprob.to("cpu")
        output = TrainingOutputBatch(
            {"output": policy_logprob},
        )
        output.metadata = micro_batch.metadata
        return output

    def process_sequences(self, sequences, input_len, eos_token_id, pad_token_id):
        return self.model.process_sequences(sequences, input_len, eos_token_id, pad_token_id)


class CriticWorkerBase(Worker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model: nn.Module = None
        self.scheduler: LRScheduler = None
        self.optimizer: Optimizer = None
        self.strategy: DistributedStrategy = None
        self.record_memory: bool = False
        self.mesh_rank: MeshRank = None
        self.critic_loss_fn: Callable = ppo_critic_loss

    def _normalize_mini_batch_size(self):
        """
        Normalize batch sizes based on device mesh and generation parameters.
        """
        if not hasattr(self, "mesh_rank") or self.mesh_rank is None:
            raise RuntimeError("mesh_rank must be initialized before calling _normalize_mini_batch_size()")

        self.critic_mini_batch_size_per_gpu = per_data_parallel_batch_size(
            self.cfg.trainer.critic_mini_batch_size,
            optimization_samples_per_prompt(self.cfg),
            self.mesh_rank.dp_size,
        )

    def _forward_micro_batch(
        self,
        micro_batch: TrainingInputBatch,
    ) -> TrainingOutputBatch:
        """Generates critic values."""
        device = self.device
        micro_batch.to(device)
        sequences = micro_batch["sequences"]
        response_length = micro_batch.metadata["response_length"]
        attention_mask = micro_batch["attention_mask"]
        self.model.eval()
        with torch.no_grad(), torch.autocast(dtype=torch.bfloat16, device_type=self.device.type):
            value = self.model(
                sequences,
                response_length,
                attention_mask,
            )
        self.model.train()  # reset model state
        value = value.to("cpu")
        output = TrainingOutputBatch(
            {"output": value},
        )
        output.metadata = micro_batch.metadata
        return output

    def save_hf_model(self, export_dir: str, tokenizer):
        # Save model in HuggingFace safetensors format
        self.strategy.save_hf_model(
            self.model,
            export_dir,
            tokenizer=tokenizer,
        )

    def ppo_train(self, train_data: TrainingInputBatch) -> TrainingOutputBatch:
        global_step = train_data.metadata["global_step"]
        dataloader = TrainingBatchIterator(train_data, self.cfg.trainer.micro_train_batch_size_per_gpu)

        torch.cuda.empty_cache()
        self.model.train()

        micro_batches_per_mini_batch = gradient_accumulation_steps(
            self.critic_mini_batch_size_per_gpu,
            self.cfg.trainer.micro_train_batch_size_per_gpu,
        )
        # The number of steps (over micro batches) to accumulate gradients before taking an optimizer step.
        accumulation_steps = micro_batches_per_mini_batch

        all_metrics = defaultdict(list)
        critic_update_steps = 0
        for epoch in range(self.cfg.trainer.update_epochs_per_batch):
            pbar = tqdm(
                dataloader,
                desc=f"Critic Train epoch [{epoch + 1}/{self.cfg.trainer.update_epochs_per_batch}]",
                disable=not self.strategy.is_rank_0(),
            )
            for local_step, experience in enumerate(pbar):
                status = self.training_step(experience, global_step, local_step, accumulation_steps)
                critic_update_steps += 1

                # for DP
                # TODO (sumanthrh): this assumes all workers are data parallel.
                # We should get more accurate metrics with seq parallel or TP.
                # There are metrics like entropy where we get average over local data size
                status = self.strategy.all_reduce(status)

                for k, v in status.items():
                    all_metrics[k].append(v)
                pbar.set_postfix(status)

        torch.distributed.barrier()

        status_mean = mean_metrics(all_metrics)
        status_mean["critic_update_steps"] = critic_update_steps / accumulation_steps

        output = TrainingOutputBatch()
        output.metadata = {"train_status": status_mean}
        return output

    def training_step(self, experience: Experience, global_step, local_step, accumulation_steps) -> Dict[str, float]:
        """
        Perform one micro-batch of training, accumulate gradients, and step the optimizer only after `accumulation_steps` micro-batches.
        """
        experience.to_device(self.device)

        sequences = experience.sequences
        old_values = experience.values
        returns = experience.returns
        num_actions = experience.num_actions
        attention_mask = experience.attention_mask
        loss_mask = experience.loss_mask

        with torch.autocast(dtype=torch.bfloat16, device_type=self.device.type):
            # critic loss
            values, output = self.model(
                sequences,
                num_actions=num_actions,
                attention_mask=attention_mask,
                return_output=True,
            )
            # loss function
            loss, clipfrac = self.critic_loss_fn(
                values,
                old_values,
                returns,
                config=self.cfg.trainer.algorithm,
                loss_mask=loss_mask,
            )
        loss = loss / accumulation_steps
        self.strategy.backward(loss, self.model, self.optimizer)
        grad_norm = None
        if (local_step + 1) % accumulation_steps == 0:
            grad_norm = self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler, name="critic")
            if grad_norm is not None:
                grad_norm = grad_norm.detach().cpu().item()

        # status
        status = {
            "critic_loss": loss.item(),
            "values_mean": masked_mean(values, loss_mask).item(),
            "critic_lr": self.scheduler.get_last_lr()[0],
            "values_clipfrac": clipfrac,
        }
        if grad_norm is not None:
            status["raw_grad_norm"] = grad_norm
        return status

    def save_checkpoint(self, ckpt_dir: str, tokenizer=None):
        upload = self.strategy.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            ckpt_dir=ckpt_dir,
            node_local_rank=self.get_node_local_rank(),
            tokenizer=tokenizer,
        )
        self._start_checkpoint_upload(upload)

    def load_checkpoint(
        self,
        ckpt_dir=None,
        load_training_state=True,
    ):
        _, states = self.strategy.load_checkpoint(
            model=self.model,
            optimizer=self.optimizer if load_training_state else None,
            scheduler=self.scheduler if load_training_state else None,
            ckpt_dir=ckpt_dir,
            load_training_state=load_training_state,
        )
        return states


class RefWorkerBase(Worker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model: nn.Module = None

    def _forward_micro_batch(self, micro_batch: TrainingInputBatch) -> TrainingOutputBatch:
        device = self.device
        micro_batch.to(device)
        sequences = micro_batch["sequences"]
        response_length = micro_batch.metadata["response_length"]
        attention_mask = micro_batch["attention_mask"]
        # MoE router-replay (R3): ref model also replays captured routing (it is
        # constructed with moe_router_replay=true), so its KL-reference logprobs
        # are computed on the same forward path as the policy. Absent key -> None
        # -> stock native forward (8B / flag-off unchanged).
        rollout_routed_experts = micro_batch.routed_experts_tensor()
        with torch.no_grad(), torch.autocast(dtype=torch.bfloat16, device_type=self.device.type):
            log_probs = self.model(
                sequences,
                response_length,
                attention_mask,
                return_output=False,
                rollout_routed_experts=rollout_routed_experts,
            )
        log_probs = log_probs.to("cpu")
        output = TrainingOutputBatch(
            {"output": log_probs},
        )
        output.metadata = micro_batch.metadata
        return output
