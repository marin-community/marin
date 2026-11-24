"""SkyRL actor-group boundary over isolated, host-wide native JAX learners."""

import asyncio
import json
import os
import shlex
import tempfile
import time
from contextlib import suppress
from pathlib import Path

import numpy as np
import ray
import torch
from loguru import logger
from omegaconf import OmegaConf
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from marin.rl.levanter_worker_protocol import LearnerProcess
from marin.rl.levanter_shutdown_observer import ShutdownObserver, observation_completed
from marin.rl.native_shutdown.artifact_handoff import handoff_artifacts
from marinskyrl.resource_locator import join_resource_path
from skyrl_train.distributed.dispatch import ActorInfo, MeshRank, concatenate_outputs_after_mesh_dispatch
from skyrl_train.training_batch import TrainingInputBatch, TrainingOutputBatch
from skyrl_train.trajectory_selection import optimization_samples_per_prompt
from skyrl_train.weight_sync.expert_block.levanter_sender import LevanterExpertBlockSender
from skyrl_train.workers.levanter_batch import forward_payload, policy_payload, validate_algorithm


def _child_environment(
    python_executable: str,
    memory_fraction: float,
    *,
    require_accelerator: bool,
    compiler_diagnostics: bool = False,
    preallocate: bool = False,
    allocator: str = "bfc",
    numpy_madvise_hugepage: bool | None = None,
    deterministic_gpu_ops: bool = False,
) -> dict[str, str]:
    if allocator not in ("bfc", "vmm") or (allocator == "vmm" and preallocate):
        raise ValueError("Native learner requires bfc, or vmm with preallocate=false")
    environment = dict(os.environ)
    if deterministic_gpu_ops:
        deterministic_flag = "--xla_gpu_deterministic_ops=true"
        inherited_flags = shlex.split(environment.get("XLA_FLAGS", ""))
        if inherited_flags not in ([], [deterministic_flag]):
            raise ValueError(
                "Native deterministic GPU profile requires empty XLA_FLAGS or its exact deterministic flag"
            )
        environment["XLA_FLAGS"] = deterministic_flag
    if numpy_madvise_hugepage is not None:
        environment["NUMPY_MADVISE_HUGEPAGE"] = str(int(numpy_madvise_hugepage))
    for name in (
        "CUDA_HOME",
        "NVRTC_HOME",
        "LIBRARY_PATH",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "VIRTUAL_ENV",
        "PYTHONHOME",
        "PYTHONPATH",
        "JAX_PLATFORM_NAME",
        "JAX_CUDA_VISIBLE_DEVICES",
        "JAX_ROCM_VISIBLE_DEVICES",
        "JAX_LOCAL_DEVICE_IDS",
    ):
        environment.pop(name, None)
    environment["PATH"] = f"{Path(python_executable).parent}{os.pathsep}{os.defpath}"
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = str(preallocate).lower()
    environment.pop("XLA_CLIENT_MEM_FRACTION", None)
    environment["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(memory_fraction)
    environment["XLA_PYTHON_CLIENT_ALLOCATOR"] = allocator
    environment["TF_FORCE_UNIFIED_MEMORY"] = "false"
    # CPU coordinator defaults must not select the learner backend. CUDA comes
    # first; the runtime verifies it before loading a model. CPU supports staging.
    environment["JAX_PLATFORMS"] = "cuda,cpu" if require_accelerator else "cpu"
    if compiler_diagnostics:
        dump = tempfile.mkdtemp(prefix="native-xla-")
        flags = shlex.split(environment.get("XLA_FLAGS", ""))
        flags.extend([f"--xla_dump_to={dump}", "--xla_dump_hlo_as_text", "--xla_dump_hlo_module_re=train_step"])
        environment["XLA_FLAGS"] = shlex.join(flags)
    return environment


def _runtime_config(
    cfg, *, model_path: str, num_training_steps: int, process_id: int, coordinator_address: str
) -> dict:
    validate_algorithm(cfg.trainer.algorithm)
    optimizer = cfg.trainer.policy.optimizer_config
    if (
        optimizer.optimizer != "AdamW"
        or optimizer.scheduler != "constant_with_warmup"
        or optimizer.num_warmup_steps != 0
    ):
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
        self._shutdown_observer = None
        self.sender = None
        self.coordinator_address = None
        self.initialization_config = None
        self._checkpoint_validation_complete = False
        self._publication_memory_recorded = False
        self._training_memory_recorded = False
        self._initial_finite = None
        self._finite_state_receipts = {}

    def node_address(self) -> str:
        return ray.util.get_node_ip_address()

    def configure(self, coordinator_address: str) -> None:
        self.coordinator_address = coordinator_address

    def init_model(self, model_path: str, num_training_steps: int):
        if self._shutdown_observer is not None:
            raise RuntimeError("Observer ownership retained; child replacement is blocked")
        self._initial_finite = None
        self._finite_state_receipts = {}
        self.initialization_config = _runtime_config(
            self.cfg,
            model_path=model_path,
            num_training_steps=num_training_steps,
            process_id=self.process_id,
            coordinator_address=self.coordinator_address,
        )
        receipt = self._start_child()
        logger.info("Native Levanter learner initialized: {}", receipt)
        self.sender = LevanterExpertBlockSender(self.publication)
        return receipt

    def _start_child(self):
        if self._shutdown_observer is not None:
            raise RuntimeError("Observer ownership retained; child replacement is blocked")
        self._initial_finite = None
        runtime = self.cfg.trainer.policy.levanter_config
        environment = _child_environment(
            runtime.python_executable,
            runtime.device_memory_fraction,
            require_accelerator=self.initialization_config["require_accelerator"],
            compiler_diagnostics=runtime.validate_checkpoint_roundtrip,
            preallocate=runtime.preallocate,
            allocator=runtime.allocator,
            numpy_madvise_hugepage=runtime.numpy_madvise_hugepage,
            deterministic_gpu_ops=runtime.deterministic_gpu_ops,
        )
        if runtime.validate_checkpoint_roundtrip:
            self.initialization_config["compiler_dump_dir"] = shlex.split(environment["XLA_FLAGS"])[-3].split("=", 1)[1]
        self.child = LearnerProcess(
            runtime.python_executable,
            environment=environment,
            timeout_seconds=runtime.command_timeout_seconds,
        )
        try:
            return self.child.request(
                "initialize", self.initialization_config, timeout_seconds=runtime.initialization_timeout_seconds
            )
        except Exception:
            self.child.close()
            self.child = None
            raise

    def _set_pad_token_id(self, pad_token_id: int) -> None:
        # Coordinator padding and explicit masks already carry token validity.
        self.pad_token_id = pad_token_id
        # SkyRL invokes this concurrently only after ALL model-init refs finish.
        # A new 200s host rendezvous must not race another host's long HF load.
        if self.cfg.trainer.policy.levanter_config.validate_checkpoint_roundtrip:
            self.capture_initial_finiteness()

    def _record_finite_state(self, phase, step, version, fingerprints=None, error=None):
        receipt = {
            "status": "passed" if error is None else "failed",
            "phase": phase,
            "process_id": self.process_id,
            "optimizer_step": step,
            "policy_version": version,
            "model_source": self.initialization_config["model"],
            "state_fingerprints": fingerprints,
            "error_type": type(error).__name__ if error is not None else None,
            "validation": "Explicit all-rank native host verdict; local unique shards may repeat across processes",
            "proof_authority": "Durable native child candidate and coordinated verdict; parent mirror is supplemental",
            "parent_mirror": {"persisted": False, "paths": [], "error_types": []},
        }
        # The child's mandatory persistence was already checked jointly. A
        # parent-only observer error must not strand peers at the next device
        # collective after they received a successful child verdict.
        if error is None:
            self._finite_state_receipts[phase] = receipt
        for directory in dict.fromkeys(
            value for name in ("SKYRL_DEBUG_ARTIFACT_DIR", "IRIS_OUTPUT_DIR") if (value := os.environ.get(name))
        ):
            try:
                path = Path(directory) / f"native-state-finite-{phase}-process-{self.process_id}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                candidate = {
                    **receipt,
                    "parent_mirror": {
                        "persisted": True,
                        "paths": [str(path)],
                        "error_types": list(receipt["parent_mirror"]["error_types"]),
                    },
                }
                path.write_text(json.dumps(candidate, sort_keys=True) + "\n")
                receipt["parent_mirror"]["persisted"] = True
                receipt["parent_mirror"]["paths"].append(str(path))
            except Exception as mirror_error:
                receipt["parent_mirror"]["error_types"].append(type(mirror_error).__name__)
        with suppress(Exception):
            logger.info("NATIVE_STATE_FINITE_PARENT {}", json.dumps(receipt, sort_keys=True))

    def _capture_finite_state(self, phase, step, version):
        try:
            fingerprints = self.child.request(
                "validate_state_finite", {"phase": phase, "step": step, "version": version}
            )
        except Exception as error:
            self._initial_finite = None
            with suppress(Exception):
                self._record_finite_state(phase, step, version, error=error)
            raise
        self._record_finite_state(phase, step, version, fingerprints)
        return fingerprints

    def capture_initial_finiteness(self):
        try:
            self._initial_finite = self._capture_finite_state("initial", 0, 0)
        except Exception:
            self.child.close()
            self.child = None
            raise
        return self._initial_finite

    def publication(self, method: str, payload: dict):
        if method == "next" and not self._publication_memory_recorded:
            self._record_memory("before-first-publication-next")
            self._publication_memory_recorded = True
        return self.child.request("publication", {"method": method, "payload": payload})

    def _record_memory(self, phase: str) -> None:
        try:
            self._capture_memory(phase)
        except Exception as error:
            # Observer I/O must never replace the publication/update outcome.
            with suppress(Exception):
                logger.warning(
                    "NATIVE_MEMORY_SNAPSHOT {}",
                    json.dumps(
                        {"phase": phase, "process_id": self.process_id, "observer_error_type": type(error).__name__}
                    ),
                )

    def _capture_memory(self, phase: str) -> None:
        receipt = {
            "phase": phase,
            "time_ns": time.time_ns(),
            "process_id": self.process_id,
            "parent_pid": os.getpid(),
            "native": self.child.request("memory_statistics", {}),
            "parent_torch": [],
        }
        for rank, device in sorted(self.sender.devices.items()):
            row = {"rank": rank, "device_type": device.type, "local_device": device.index}
            if device.type == "cuda":
                try:
                    free, total = torch.cuda.mem_get_info(device)
                    row.update(
                        torch_allocated_bytes=torch.cuda.memory_allocated(device),
                        torch_reserved_bytes=torch.cuda.memory_reserved(device),
                        driver_global_free_bytes=free,
                        driver_global_total_bytes=total,
                    )
                except Exception as error:
                    row["error_type"] = type(error).__name__
            receipt["parent_torch"].append(row)
        encoded = json.dumps(receipt, sort_keys=True)
        logger.info("NATIVE_MEMORY_SNAPSHOT {}", encoded)
        if output_dir := os.environ.get("IRIS_OUTPUT_DIR"):
            path = Path(output_dir) / f"native-memory-{phase}-process-{self.process_id}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(encoded + "\n")

    def train(self, payload: dict) -> dict:
        validation = self.cfg.trainer.policy.levanter_config.validate_checkpoint_roundtrip
        if validation and "initial" not in self._finite_state_receipts:
            raise RuntimeError("Native diagnostic training requires its initial finite-state proof")
        if self.cfg.trainer.policy.levanter_config.validate_checkpoint_roundtrip and not self._training_memory_recorded:
            self._record_memory("before-first-train")
            self._training_memory_recorded = True
        try:
            status = self.child.request("train", payload)
        finally:
            self._initial_finite = None
        if validation and int(status["optimizer_step"]) >= 2:
            step, version = int(status["optimizer_step"]), int(status["policy_version"])
            self._capture_finite_state(f"step{step}", step, version)
        return status

    def forward(self, data: TrainingInputBatch) -> TrainingOutputBatch:
        result = self.child.request("forward", forward_payload(data).as_dict())
        output = TrainingOutputBatch(
            {} if result["output"] is None else {"output": torch.from_numpy(np.array(result["output"], copy=True))}
        )
        return output

    def ppo_train(self, data: TrainingInputBatch) -> TrainingOutputBatch:
        mini_batch_size = self.cfg.trainer.policy_mini_batch_size * optimization_samples_per_prompt(self.cfg)
        if data.batch_size % mini_batch_size:
            raise ValueError(
                "The native learner requires complete optimizer minibatches; trailing rows cannot be dropped"
            )
        updates = []
        # SkyRL's TrainingBatchIterator preserves row order and repeats that
        # order across epochs. Normalize each complete optimizer batch once.
        for _ in range(self.cfg.trainer.update_epochs_per_batch):
            for start in range(0, data.batch_size - mini_batch_size + 1, mini_batch_size):
                batch = data.slice(start, start + mini_batch_size)
                validate_checkpoint = (
                    self.cfg.trainer.policy.levanter_config.validate_checkpoint_roundtrip
                    and not self._checkpoint_validation_complete
                )
                if validate_checkpoint and self._initial_finite is None:
                    raise RuntimeError("Native checkpoint validation has no current initial finite-state proof")
                before_model = self._initial_finite["components"]["model"] if validate_checkpoint else None
                status = self.train(policy_payload(batch, self.cfg.trainer.algorithm).as_dict())
                if validate_checkpoint:
                    self.validate_checkpoint_roundtrip(before_model, status)
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

    def _prepare_shutdown_observer(self):
        if self.process_id != 0 or not self.cfg.trainer.policy.levanter_config.get(
            "diagnostic_shutdown_observer", False
        ):
            return
        if self._shutdown_observer is not None:
            raise RuntimeError("A shutdown observer already owns this child")
        destination = os.environ.get("SKYRL_DEBUG_ARTIFACT_DIR") or os.environ.get("IRIS_OUTPUT_DIR")
        if not destination:
            raise RuntimeError("Shutdown observation requires a local artifact directory")
        self._shutdown_observer = ShutdownObserver(self.child.process, Path(destination), self.process_id)
        try:
            self._shutdown_observer.wait_ready()
        except BaseException as error:
            self._retire_shutdown_observer(error, cancel=True)
            raise

    def _retire_shutdown_observer(self, primary=None, *, cancel=False):
        observer = self._shutdown_observer
        if observer is None:
            return
        receipt = None
        retired_safe = False
        retirement_error = None
        try:
            receipt = observer.retire(cancel=cancel)
            retired_safe = True
            self._shutdown_observer = None
            if not cancel and not observation_completed(receipt):
                raise RuntimeError("Native shutdown observation did not complete; diagnostic is nonqualifying")
        except Exception as error:
            retirement_error = type(error).__name__
            if primary is None:
                raise
            primary.add_note(f"Shutdown observer failure: {error!r}")
        finally:
            if not observer.artifact_handoff_attempted:
                observer.artifact_handoff_attempted = True
                try:
                    handoff = handoff_artifacts(
                        observer.directory,
                        observer.config,
                        self.process_id,
                        retired_safe=retired_safe,
                        guardian_receipt=receipt,
                        observation_completed=receipt is not None and observation_completed(receipt),
                        cancel=cancel,
                        retirement_error=retirement_error,
                        primary_error_type=type(primary).__name__ if primary is not None else None,
                    )
                    logger.info("NATIVE_OBSERVER_HANDOFF {}", json.dumps(handoff, sort_keys=True))
                    if primary is not None and handoff["status"] != "uploaded":
                        primary.add_note(f"Shutdown observer artifact handoff: {handoff['status']}")
                except Exception as error:
                    if primary is not None:
                        primary.add_note(f"Shutdown observer artifact handoff failed: {type(error).__name__}")
                    with suppress(Exception):
                        logger.warning(
                            "NATIVE_OBSERVER_HANDOFF {}",
                            json.dumps({"status": "failed", "error_type": type(error).__name__}),
                        )

    def _shutdown_child(self):
        observer = self._shutdown_observer
        payload = {} if observer is None else observer.payload()
        try:
            result = self.child.request("shutdown", payload)
        except BaseException as error:
            self._retire_shutdown_observer(error, cancel=True)
            raise
        self._retire_shutdown_observer()
        return result

    def validate_checkpoint_roundtrip(self, before_model: dict, train_status: dict) -> dict:
        """Restore the first updated policy in a fresh JAX child, keeping serving alive."""
        if "initial" not in self._finite_state_receipts:
            raise RuntimeError("Native checkpoint validation requires the initial finite-state proof")
        checkpoint = join_resource_path(
            self.initialization_config["runtime"]["checkpoint_dir"],
            f"native-validation/optimizer-step-{train_status['optimizer_step']}",
        )
        previous = self.child.request("runtime_identity", {})
        step, version = int(train_status["optimizer_step"]), int(train_status["policy_version"])
        validation = {"phase": "step1", "step": step, "version": version}
        try:
            saved = self.child.request(
                "save_checkpoint",
                {
                    "ckpt_dir": checkpoint,
                    "include_fingerprints": True,
                    "finite_validation": validation,
                },
            )
        except Exception as error:
            with suppress(Exception):
                self._record_finite_state("step1", step, version, error=error)
            raise
        self._record_finite_state("step1", step, version, saved["state_fingerprints"])
        model_change = self.child.request(
            "validate_model_change",
            {"before": before_model, "after": saved["state_fingerprints"]["components"]["model"]},
        )
        # Cold attach must finish before peers leave the existing barrier.
        # Otherwise their five-second exit windows could expire before our sample.
        try:
            self._prepare_shutdown_observer()
            self.child.request("barrier", {})
            self._shutdown_child()
        except BaseException as error:
            self._retire_shutdown_observer(error, cancel=True)
            raise
        old_exit_code = self.child.process.returncode
        self.child.close()
        self.child = None
        self._start_child()
        try:
            restored = self.child.request(
                "load_checkpoint",
                {
                    "ckpt_dir": checkpoint,
                    "include_fingerprints": True,
                    "finite_validation": {"phase": "restored", "step": step, "version": version},
                },
            )
        except Exception as error:
            with suppress(Exception):
                self._record_finite_state("restored", step, version, error=error)
            raise
        self._record_finite_state("restored", step, version, restored["state_fingerprints"])
        if saved != restored:
            raise RuntimeError("Fresh native learner did not restore model/Adam/RNG/step/averaging exactly")
        current = self.child.request("runtime_identity", {})
        if current["pid"] == previous["pid"]:
            raise RuntimeError("Native checkpoint validation did not start a fresh learner process")
        self.child.request("barrier", {})
        self._checkpoint_validation_complete = True
        receipt = {
            "status": "passed",
            "process_id": self.process_id,
            "checkpoint": checkpoint,
            "previous_runtime": previous,
            "restored_runtime": current,
            "old_exit_code": old_exit_code,
            "before_update_model": before_model,
            "model_change": model_change,
            "saved": saved,
            "restored": restored,
            "frozen_batch_sha256": train_status["frozen_batch_sha256"],
            "initial_finite_state": self._finite_state_receipts["initial"],
            "finite_state_phases_at_restore": dict(self._finite_state_receipts),
        }
        encoded = json.dumps(receipt, sort_keys=True)
        logger.info("NATIVE_CHECKPOINT_ROUNDTRIP {}", encoded)
        if output_dir := os.environ.get("IRIS_OUTPUT_DIR"):
            path = Path(output_dir) / f"native-checkpoint-roundtrip-process-{self.process_id}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(encoded + "\n")
        return receipt

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
        return self.child.request(
            "save_checkpoint", {"ckpt_dir": ckpt_dir, "include_fingerprints": include_fingerprints}
        )

    def load_checkpoint(self, ckpt_dir: str, load_training_state: bool, include_fingerprints: bool = False):
        if not load_training_state:
            raise ValueError("Native learner checkpoints always restore complete training state")
        return self.child.request(
            "load_checkpoint", {"ckpt_dir": ckpt_dir, "include_fingerprints": include_fingerprints}
        )

    def state_fingerprints(self):
        return self.child.request("state_fingerprints", {})

    def empty_cache(self) -> None:
        # JAX's allocator owns its buffers for the lifetime of this learner.
        return None

    def close_telemetry(self) -> None:
        if self.child is not None:
            try:
                self._shutdown_child()
            finally:
                # Unknown observer teardown retains ownership; never race a new
                # numeric attach or release the child reservation on this path.
                if self._shutdown_observer is None:
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
            address = ray.get(
                self._actor_handlers[0].node_address.remote(), timeout=max(0, deadline - time.monotonic())
            )
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
                for reference in actor.expert_block_rpc.options(num_returns=self.local_device_count).remote(
                    *args, **kwargs
                )
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
