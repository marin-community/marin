# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""RL job orchestration for Fray v2.

Submits one coordinator job that creates all shared actors and child jobs.
The coordinator runs inside the cluster (not client-side), giving Iris
proper job hierarchy, cascading cleanup, and region inheritance.
"""

import dataclasses
import logging

from fray.v2 import (
    Client,
    Entrypoint,
    JobHandle,
    JobRequest,
    JobStatus,
    ResourceConfig,
    create_environment,
    current_client,
    wait_all,
)
from fray.v2.actor import HostedActor
from fray.v2.types import GpuConfig
from marin.rl.curriculum import Curriculum
from marin.rl.environments.inference_ctx import vLLMInferenceContextConfig
from marin.rl.placement import resolve_launcher_region, singleton_region_list
from marin.rl.rl_job import RLJob, RLJobConfig
from marin.rl.run_state import RLRunState
from marin.rl.runtime import RLRuntimeHandles, WeightTransferRuntime
from marin.rl.weight_transfer import WeightTransferMode
from marin.rl.weight_transfer.arrow_flight import ArrowFlightCoordinator
from marin.training.run_environment import add_run_env_variables
from marin.utils import remove_tpu_lockfile_on_exit
from rigging.log_setup import configure_logging

logger = logging.getLogger(__name__)

RL_COORDINATOR_CPU_DISK = "30g"
RL_COORDINATOR_RESOURCES = ResourceConfig.with_cpu(cpu=0.5, preemptible=False, disk=RL_COORDINATOR_CPU_DISK)
JAX_CHECKPOINT_DEBUG_MODULES = ",".join(
    (
        "jax.experimental.array_serialization.serialization",
        "jax.experimental.array_serialization.tensorstore_impl",
        "jax._src.distributed",
    )
)
TF_CPP_CHECKPOINT_DEBUG_VMODULE = ",".join(
    (
        "coordination_service=2",
        "coordination_service_agent=2",
        "tsl=1",
    )
)
_DEVICE_DEPENDENCY_GROUPS = {"gpu", "tpu"}
_VLLM_DEPENDENCY_GROUPS = {"vllm"}
# Keep CUDA vLLM pinned at the worker boundary instead of a Marin extra. vLLM's
# numba pin conflicts with the workspace test environment, but the rollout
# workers still need a reproducible package set.
_GPU_VLLM_PIP_PACKAGES = ("vllm==0.13.0",)


@dataclasses.dataclass(frozen=True)
class _HostedRuntime:
    runtime: RLRuntimeHandles
    hosted_actors: list[HostedActor]


def _coordinator_extras(config: RLJobConfig) -> list[str]:
    """Dependency extras for the CPU coordinator layer.

    Keep this layer minimal: no vLLM and no TPU/GPU runtime package install.
    """
    return sorted(set(config.pip_dependency_groups) - _VLLM_DEPENDENCY_GROUPS - _DEVICE_DEPENDENCY_GROUPS)


def _worker_dependency_groups(
    config: RLJobConfig,
    resources: ResourceConfig,
    *,
    needs_vllm: bool,
) -> tuple[list[str], list[str]]:
    extras = set(config.pip_dependency_groups) - _DEVICE_DEPENDENCY_GROUPS - _VLLM_DEPENDENCY_GROUPS
    pip_packages: list[str] = []
    device_kind = resources.device.kind

    if device_kind != "cpu":
        extras.add(device_kind)

    if needs_vllm:
        if device_kind == "tpu":
            extras.add("vllm")
        elif device_kind == "gpu":
            pip_packages.extend(_GPU_VLLM_PIP_PACKAGES)
        else:
            raise ValueError("vLLM rollout workers require GPU or TPU resources")
    else:
        extras.discard("vllm")

    return sorted(extras), pip_packages


def _worker_environment(
    config: RLJobConfig,
    resources: ResourceConfig,
    *,
    needs_vllm: bool,
    extra_env_vars: dict[str, str] | None = None,
):
    env = {"EQX_ON_ERROR": "nan"}
    if extra_env_vars is not None:
        env.update(extra_env_vars)
    env = add_run_env_variables(env)
    extras, pip_packages = _worker_dependency_groups(config, resources, needs_vllm=needs_vllm)
    return create_environment(
        env_vars=env,
        extras=extras,
        pip_packages=pip_packages,
    )


def _pin_worker_region(resources: ResourceConfig, launcher_region: str) -> ResourceConfig:
    """Return resources pinned to the concrete launcher region."""
    return dataclasses.replace(
        resources,
        regions=singleton_region_list(launcher_region),
    )


def _validate_worker_configuration(config: RLJobConfig) -> None:
    run_config = config.run_config
    if run_config is None:
        raise ValueError("RLJobConfig.run_config must be set for v2/Iris orchestration")

    rollout_device_kind = run_config.rollout_resources.device.kind
    if config.inference_type == "levanter" and rollout_device_kind != "tpu":
        raise ValueError("Levanter rollout inference currently requires TPU rollout_resources")
    if config.inference_type == "vllm" and rollout_device_kind == "cpu":
        raise ValueError("vLLM rollout inference requires GPU or TPU rollout_resources")
    if config.inference_type == "vllm" and rollout_device_kind == "gpu" and config.inflight_weight_updates:
        raise ValueError("GPU vLLM rollout inference does not yet support inflight_weight_updates")

    if config.inference_type != "vllm" or not isinstance(config.inference_config, vLLMInferenceContextConfig):
        return
    if rollout_device_kind != "gpu":
        return
    rollout_device = run_config.rollout_resources.device
    assert isinstance(rollout_device, GpuConfig)
    if config.inference_config.tensor_parallel_size > rollout_device.count:
        raise ValueError("rollout_resources must provide at least tensor_parallel_size GPUs for vLLM rollout inference")


def submit_rl_job(config: RLJobConfig) -> JobHandle:
    """Submit an RL training job as a single coordinator job.

    The coordinator creates all shared actors and child worker jobs.
    Killing the coordinator kills all children (Iris cascading cleanup).
    """
    client = current_client()

    env = {"EQX_ON_ERROR": "nan"}
    env = add_run_env_variables(env)
    coordinator_extras = _coordinator_extras(config)

    return client.submit(
        JobRequest(
            name=f"rl-{config.resolved_instance_id}",
            entrypoint=Entrypoint.from_callable(_run_rl_coordinator, args=(config,)),
            resources=RL_COORDINATOR_RESOURCES,
            environment=create_environment(
                env_vars=env,
                extras=coordinator_extras,
            ),
            max_retries_failure=0,
            max_retries_preemption=0,
        )
    )


def _run_rl_coordinator(config: RLJobConfig) -> None:
    """Run the in-cluster RL coordinator."""
    configure_logging(level=logging.INFO)
    logger.info("RL coordinator starting for run %s", config.run_id)

    client = current_client()
    rl_job = RLJob(config)
    _validate_worker_configuration(config)
    train_config, rollout_config = rl_job.to_worker_configs()
    run_config = config.run_config
    assert run_config is not None
    hosted_runtime: _HostedRuntime | None = None

    try:
        hosted_runtime = _create_runtime_handles(client, config)
        runtime = hosted_runtime.runtime
        logger.info("Runtime handles created: curriculum, run_state, weight_transfer coordinator")

        worker_env_overrides: dict[str, str] | None = None
        if train_config.trainer.checkpointer.debug.enabled or train_config.weight_transfer.debug_weight_transfer:
            worker_env_overrides = {"PYTHONUNBUFFERED": "1"}
        if train_config.trainer.checkpointer.debug.enabled:
            if worker_env_overrides is None:
                worker_env_overrides = {}
            worker_env_overrides.update(
                {
                    "JAX_TRACEBACK_FILTERING": "off",
                    "JAX_LOGGING_LEVEL": "INFO",
                    "JAX_DEBUG_LOG_MODULES": JAX_CHECKPOINT_DEBUG_MODULES,
                    "JAX_INCLUDE_FULL_TRACEBACKS_IN_LOCATIONS": "1",
                    "TF_CPP_MIN_LOG_LEVEL": "0",
                    "TF_CPP_MAX_VLOG_LEVEL": "1",
                    "TF_CPP_VMODULE": TF_CPP_CHECKPOINT_DEBUG_VMODULE,
                }
            )
        if train_config.weight_transfer.debug_weight_transfer:
            if worker_env_overrides is None:
                worker_env_overrides = {}
            worker_env_overrides.update(
                {
                    "JAX_TRACEBACK_FILTERING": "off",
                    "TF_CPP_MIN_LOG_LEVEL": "0",
                }
            )
        train_worker_env = _worker_environment(
            config,
            run_config.train_resources,
            needs_vllm=False,
            extra_env_vars=worker_env_overrides,
        )
        rollout_worker_env = _worker_environment(
            config,
            run_config.rollout_resources,
            needs_vllm=config.inference_type == "vllm",
            extra_env_vars=worker_env_overrides,
        )

        launcher_region = resolve_launcher_region(run_config.train_resources, run_config.rollout_resources)
        train_resources = _pin_worker_region(run_config.train_resources, launcher_region)
        rollout_resources = _pin_worker_region(run_config.rollout_resources, launcher_region)

        train_job = client.submit(
            JobRequest(
                name=f"rl-{config.resolved_instance_id}-train",
                entrypoint=Entrypoint.from_callable(_train_worker_entry, args=(train_config, runtime)),
                resources=train_resources,
                environment=train_worker_env,
                max_retries_failure=run_config.max_retries_failure,
                max_retries_preemption=run_config.max_retries_preemption,
            )
        )
        rollout_jobs: list[JobHandle] = []

        for i in range(run_config.num_rollout_workers):
            worker_run_id = f"{rollout_config.run_id}-rollout-{i}"
            tracker_config = rollout_config.tracker_config
            if tracker_config is not None:
                tracker_config = dataclasses.replace(tracker_config, name=worker_run_id)
            worker_config = dataclasses.replace(
                rollout_config,
                seed=rollout_config.seed + i,
                run_id=worker_run_id,
                worker_index=i,
                tracker_config=tracker_config,
            )
            rollout_jobs.append(
                client.submit(
                    JobRequest(
                        name=f"rl-{config.resolved_instance_id}-rollout-{i}",
                        entrypoint=Entrypoint.from_callable(_rollout_worker_entry, args=(worker_config, runtime)),
                        resources=rollout_resources,
                        environment=rollout_worker_env,
                        max_retries_failure=run_config.max_retries_failure,
                        max_retries_preemption=run_config.max_retries_preemption,
                    )
                )
            )

        logger.info(
            "Submitted %d child jobs (1 trainer + %d rollout workers)",
            1 + len(rollout_jobs),
            run_config.num_rollout_workers,
        )

        train_status = train_job.wait(raise_on_failure=True)
        if train_status != JobStatus.SUCCEEDED:
            raise RuntimeError(f"Trainer finished with unexpected status {train_status}")

        _terminate_rollout_jobs(rollout_jobs)
        wait_all(rollout_jobs, raise_on_failure=False)
        logger.info("RL coordinator finished for run %s", config.run_id)
    finally:
        if hosted_runtime is not None:
            _shutdown_hosted_actors(hosted_runtime.hosted_actors)


def _terminate_rollout_jobs(rollout_jobs: list[JobHandle]) -> None:
    for rollout_job in rollout_jobs:
        status = rollout_job.status()
        if JobStatus.finished(status):
            continue
        logger.info("Stopping rollout job %s after trainer completion", rollout_job.job_id)
        rollout_job.terminate()


def _shutdown_hosted_actors(hosted_actors: list[HostedActor]) -> None:
    for hosted_actor in reversed(hosted_actors):
        try:
            hosted_actor.shutdown()
        except Exception:
            logger.exception("Failed to shut down hosted actor")


def _create_runtime_handles(client: Client, config: RLJobConfig) -> _HostedRuntime:
    """Create all shared actors for the RL run."""
    hosted_actors: list[HostedActor] = []

    try:
        curriculum_hosted = client.host_actor(
            Curriculum,
            config.curriculum,
            name=f"rl-{config.resolved_instance_id}-curriculum",
        )
        hosted_actors.append(curriculum_hosted)

        run_state_hosted = client.host_actor(
            RLRunState,
            name=f"rl-{config.resolved_instance_id}-run-state",
        )
        hosted_actors.append(run_state_hosted)

        arrow_coordinator = None
        if config.weight_transfer.mode == WeightTransferMode.ARROW_FLIGHT:
            arrow_hosted = client.host_actor(
                ArrowFlightCoordinator,
                name=config.weight_transfer.coordinator_name or f"rl-{config.resolved_instance_id}-wt-coord",
            )
            hosted_actors.append(arrow_hosted)
            arrow_coordinator = arrow_hosted.handle

        return _HostedRuntime(
            runtime=RLRuntimeHandles(
                curriculum=curriculum_hosted.handle,
                run_state=run_state_hosted.handle,
                weight_transfer=WeightTransferRuntime(arrow_flight_coordinator=arrow_coordinator),
            ),
            hosted_actors=hosted_actors,
        )
    except Exception:
        _shutdown_hosted_actors(hosted_actors)
        raise


def _train_worker_entry(train_config, runtime: RLRuntimeHandles) -> None:
    """Entrypoint for the training worker child job."""
    configure_logging(level=logging.INFO)
    with remove_tpu_lockfile_on_exit():
        try:
            from marin.rl.train_worker import TrainWorker

            worker = TrainWorker(config=train_config, runtime=runtime)
            worker.train()
            runtime.run_state.mark_completed.remote().result()
        except Exception:
            logger.exception("TRAIN WORKER CRASHED (orchestration entrypoint)")
            # Do not mark the shared run state failed here. This function runs
            # inside a single trainer task attempt, and Iris may retry the same
            # child job under the still-running coordinator. If we flip the
            # shared run_state to FAILED on an attempt-local crash, rollout
            # workers interpret that as whole-run terminal failure and exit
            # cleanly, leaving the retried trainer without rollout jobs.
            raise


def _rollout_worker_entry(rollout_config, runtime: RLRuntimeHandles) -> None:
    """Entrypoint for a rollout worker child job."""
    configure_logging(level=logging.INFO)
    with remove_tpu_lockfile_on_exit():
        try:
            from marin.rl.rollout_worker import RolloutWorker

            worker = RolloutWorker(config=rollout_config, runtime=runtime)
            worker.run()
        except Exception:
            logger.exception("ROLLOUT WORKER CRASHED (orchestration entrypoint)")
            raise
