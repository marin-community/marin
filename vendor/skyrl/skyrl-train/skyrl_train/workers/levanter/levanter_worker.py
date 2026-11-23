"""SkyRL actor-group boundary over isolated, host-wide native JAX learners."""

import asyncio
import os
import time
from pathlib import Path

import numpy as np
import ray
import torch
from loguru import logger
from omegaconf import OmegaConf
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from marin.rl.levanter_worker_protocol import LearnerProcess
from skyrl_train.distributed.dispatch import ActorInfo, MeshRank, concatenate_outputs_after_mesh_dispatch
from skyrl_train.training_batch import TrainingInputBatch, TrainingOutputBatch
from skyrl_train.trajectory_selection import optimization_samples_per_prompt
from skyrl_train.weight_sync.expert_block.levanter_sender import LevanterExpertBlockSender
from skyrl_train.config.validation import validate_levanter_algorithm
from skyrl_train.workers.levanter_batch import forward_payload, policy_payload


def _child_environment(python_executable: str, memory_fraction: float) -> dict[str, str]:
    environment = dict(os.environ)
    for name in (
        "CUDA_HOME", "NVRTC_HOME", "LIBRARY_PATH", "LD_LIBRARY_PATH", "LD_PRELOAD",
        "VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH",
    ):
        environment.pop(name, None)
    environment["PATH"] = f"{Path(python_executable).parent}{os.pathsep}{os.defpath}"
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    environment["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(memory_fraction)
    return environment


def _runtime_config(cfg, *, model_path: str, num_training_steps: int, process_id: int, coordinator_address: str) -> dict:
    validate_levanter_algorithm(cfg.trainer.algorithm)
    optimizer = cfg.trainer.policy.optimizer_config
    if optimizer.optimizer != "AdamW" or optimizer.scheduler != "constant_with_warmup" or optimizer.num_warmup_steps != 0:
        raise ValueError("The native quick-loop learner supports constant AdamW without warmup")
    extra = OmegaConf.to_container(optimizer.optimizer_kwargs, resolve=True)
    if set(extra) - {"eps"}:
        raise ValueError(f"Unsupported native AdamW options: {sorted(set(extra) - {'eps'})}")
    if float(cfg.generator.sampling_params.temperature) != 1.0:
        raise ValueError("The native quick-loop learner requires policy scoring temperature=1")
    if cfg.trainer.placement.colocate_all or cfg.trainer.offload_optimizer_during_rollouts:
        raise ValueError("The native learner requires disaggregated, GPU-resident policy state")
    model = cfg.trainer.policy.model
    runtime = OmegaConf.to_container(cfg.trainer.policy.levanter_config, resolve=True)
    runtime["checkpoint_dir"] = cfg.trainer.ckpt_path
    return {
        "runtime": runtime,
        "distributed": {
            "coordinator_address": coordinator_address,
            "num_processes": cfg.trainer.placement.policy_num_nodes,
            "process_id": process_id,
            "local_device_count": cfg.trainer.placement.policy_num_gpus_per_node,
        },
        "initialization_timeout_seconds": runtime["initialization_timeout_seconds"],
        "run_id": cfg.trainer.run_name,
        "seed": cfg.trainer.seed,
        "bf16": cfg.trainer.bf16,
        "num_training_steps": int(num_training_steps),
        "mini_batch_size": cfg.trainer.policy_mini_batch_size * optimization_samples_per_prompt(cfg),
        "micro_batch_size_per_device": cfg.trainer.micro_train_batch_size_per_gpu,
        "eps_clip_low": cfg.trainer.algorithm.eps_clip_low,
        "eps_clip_high": cfg.trainer.algorithm.eps_clip_high,
        "require_accelerator": True,
        "model": {
            "path": model_path,
            "revision": model.revision,
            "source_uri": model.source_uri,
            "source_identity": model.source_identity,
        },
        "optimizer": {
            "lr": float(optimizer.lr),
            "adam_betas": list(optimizer.adam_betas),
            "epsilon": float(extra.get("eps", 1e-8)),
            "weight_decay": float(optimizer.weight_decay),
            "max_grad_norm": float(optimizer.max_grad_norm),
        },
    }


class LevanterHostWorker:
    """Reserve a host's GPUs while JAX owns all learner state in a child process."""

    def __init__(self, cfg, process_id: int):
        self.cfg = cfg
        self.process_id = process_id
        self.local_device_count = cfg.trainer.placement.policy_num_gpus_per_node
        self.child = None
        self.sender = None
        self.coordinator_address = None

    def node_address(self) -> str:
        return ray.util.get_node_ip_address()

    def configure(self, coordinator_address: str) -> None:
        self.coordinator_address = coordinator_address

    def init_model(self, model_path: str, num_training_steps: int):
        runtime = self.cfg.trainer.policy.levanter_config
        self.child = LearnerProcess(
            runtime.python_executable,
            environment=_child_environment(runtime.python_executable, runtime.device_memory_fraction),
            timeout_seconds=runtime.command_timeout_seconds,
        )
        config = _runtime_config(
            self.cfg, model_path=model_path, num_training_steps=num_training_steps,
            process_id=self.process_id, coordinator_address=self.coordinator_address,
        )
        receipt = self.child.request("initialize", config, timeout_seconds=runtime.initialization_timeout_seconds)
        logger.info("Native Levanter learner initialized: {}", receipt)
        self.sender = LevanterExpertBlockSender(self.publication)
        return receipt

    def _set_pad_token_id(self, pad_token_id: int) -> None:
        # Coordinator padding and explicit masks already carry token validity.
        self.pad_token_id = pad_token_id

    def publication(self, method: str, payload: dict):
        return self.child.request("publication", {"method": method, "payload": payload})

    def forward(self, data: TrainingInputBatch) -> TrainingOutputBatch:
        result = self.child.request("forward", forward_payload(data).as_dict())
        output = TrainingOutputBatch(
            {} if result["output"] is None else {"output": torch.from_numpy(np.array(result["output"], copy=True))}
        )
        return output

    def ppo_train(self, data: TrainingInputBatch) -> TrainingOutputBatch:
        mini_batch_size = self.cfg.trainer.policy_mini_batch_size * optimization_samples_per_prompt(self.cfg)
        if data.batch_size % mini_batch_size:
            raise ValueError("The native learner requires complete optimizer minibatches; trailing rows cannot be dropped")
        updates = []
        # SkyRL's TrainingBatchIterator preserves row order and repeats that
        # order across epochs. Normalize each complete optimizer batch once.
        for _ in range(self.cfg.trainer.update_epochs_per_batch):
            for start in range(0, data.batch_size - mini_batch_size + 1, mini_batch_size):
                batch = data.slice(start, start + mini_batch_size)
                status = self.child.request("train", policy_payload(batch, self.cfg.trainer.algorithm).as_dict())
                fingerprint = status.pop("frozen_batch_sha256")
                logger.info("Native policy update: {} frozen_batch_sha256={}", status, fingerprint)
                updates.append(status)
        if not updates:
            raise ValueError("Training input contains no complete optimizer minibatch")
        status = {name: float(np.mean([update[name] for update in updates])) for name in updates[0]}
        status["policy_update_steps"] = float(len(updates))
        output = TrainingOutputBatch({})
        output.metadata = {"train_status": status}
        return output

    def init_weight_sync_state(self, inference_engine_client) -> None:
        if self.cfg.generator.weight_sync_transport != "expert_block":
            raise ValueError("The native learner requires expert-block publication")

    def expert_block_rpc(self, method: str, *args):
        if method == "inventory":
            return self.sender.inventory()
        if method == "trainer_init":
            return self.sender.trainer_init(*args)
        if method == "send_weights":
            return self.sender.send_weights(*args)
        if method == "verify":
            return self.sender.verify(*args)
        if method == "shutdown":
            return self.sender.shutdown()
        raise ValueError(f"Unknown expert-block operation {method}")

    def barrier_all(self):
        return self.child.request("barrier", {})

    def save_checkpoint(self, ckpt_dir: str, tokenizer, include_fingerprints: bool = False):
        return self.child.request("save_checkpoint", {"ckpt_dir": ckpt_dir, "include_fingerprints": include_fingerprints})

    def load_checkpoint(self, ckpt_dir: str, load_training_state: bool, include_fingerprints: bool = False):
        if not load_training_state:
            raise ValueError("Native learner checkpoints always restore complete training state")
        return self.child.request("load_checkpoint", {"ckpt_dir": ckpt_dir, "include_fingerprints": include_fingerprints})

    def state_fingerprints(self):
        return self.child.request("state_fingerprints", {})

    def empty_cache(self) -> None:
        # JAX's allocator owns its buffers for the lifetime of this learner.
        return None

    def close_telemetry(self) -> None:
        if self.child is not None:
            try:
                self.child.request("shutdown", {})
            finally:
                self.child.close()
                self.child = None


_HostActor = ray.remote(LevanterHostWorker)


class LevanterRayActorGroup:
    """SkyRL group facade for a single distributed native optimizer mesh.

    A complete finalized batch goes to every JAX process; JAX installs only each
    device's addressable shards. SkyRL sees one collection rank, rather than
    applying its Torch DP slicing before objective normalization.
    """

    def __init__(self, cfg, num_nodes: int, num_gpus_per_node: int, *, pg):
        if pg is None:
            raise ValueError("Native learner hosts require an explicit whole-node placement group")
        self.cfg = cfg
        deadline = time.monotonic() + cfg.trainer.policy.levanter_config.initialization_timeout_seconds
        self.local_device_count = num_gpus_per_node
        self._actor_handlers = [
            _HostActor.options(
                num_cpus=num_gpus_per_node,
                num_gpus=num_gpus_per_node,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=rank,
                    placement_group_capture_child_tasks=False,
                ),
            ).remote(cfg, rank)
            for rank in range(num_nodes)
        ]
        try:
            address = ray.get(self._actor_handlers[0].node_address.remote(), timeout=max(0, deadline - time.monotonic()))
            coordinator = f"{address}:{cfg.trainer.policy.levanter_config.coordinator_port}"
            ray.get(
                [actor.configure.remote(coordinator) for actor in self._actor_handlers],
                timeout=max(0, deadline - time.monotonic()),
            )
        except Exception:
            for actor in self._actor_handlers:
                ray.kill(actor, no_restart=True)
            raise
        self.actor_infos = [
            ActorInfo(actor, MeshRank(dp=0, sp=0, tp=rank, pp=0, world_size=num_nodes, dp_size=1, pp_size=1))
            for rank, actor in enumerate(self._actor_handlers)
        ]

    def async_init_model(self, *args, **kwargs):
        return [actor.init_model.remote(*args, **kwargs) for actor in self._actor_handlers]

    def async_run_ray_method(self, dispatch_type: str, method_name: str, *args, **kwargs):
        if dispatch_type not in ("mesh", "pass_through"):
            raise ValueError(f"Unknown learner dispatch {dispatch_type}")
        if method_name == "expert_block_rpc":
            # Each host owns eight logical GPU publishers. Ray's multiple return
            # values preserve the existing ExpertBlockSync per-rank contract.
            return [
                reference
                for actor in self._actor_handlers
                for reference in actor.expert_block_rpc.options(num_returns=self.local_device_count).remote(*args, **kwargs)
            ]
        if method_name in ("forward", "ppo_train"):
            # Serialize a finalized coordinator batch once. Every host borrows
            # the same object rather than pinning another copy per dispatch.
            if args:
                args = (ray.put(args[0]), *args[1:])
            else:
                kwargs = {**kwargs, "data": ray.put(kwargs["data"])}
        return [getattr(actor, method_name).remote(*args, **kwargs) for actor in self._actor_handlers]

    async def async_run_method(self, dispatch_type: str, method_name: str, *args, **kwargs):
        outputs = await asyncio.gather(*self.async_run_ray_method(dispatch_type, method_name, *args, **kwargs))
        if dispatch_type == "mesh":
            return concatenate_outputs_after_mesh_dispatch(self.actor_infos, list(outputs))
        return outputs

    def offload_to_cpu(self, **kwargs):
        raise ValueError("Native disaggregated learners keep model and optimizer state on their reserved GPUs")

    def backload_to_gpu(self, nonblocking=False, **kwargs):
        return [] if nonblocking else None

    def kill_actors(self, no_restart: bool = True) -> None:
        # Shutdown all child process groups concurrently before releasing GPUs.
        refs = [actor.close_telemetry.remote() for actor in self._actor_handlers]
        ray.wait(refs, num_returns=len(refs), timeout=10)
        for actor in self._actor_handlers:
            ray.kill(actor, no_restart=no_restart)
