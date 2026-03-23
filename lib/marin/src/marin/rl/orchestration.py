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
    ResourceConfig,
    create_environment,
    current_client,
    wait_all,
)
from iris.logging import configure_logging
from marin.rl.rl_job import RLJob, RLJobConfig
from marin.rl.run_state import RLRunState
from marin.rl.runtime import RLRuntimeHandles, WeightTransferRuntime
from marin.rl.weight_transfer.arrow_flight import ArrowFlightCoordinator
from marin.training.training import _add_run_env_variables
from marin.utils import remove_tpu_lockfile_on_exit

logger = logging.getLogger(__name__)


def submit_rl_job(config: RLJobConfig) -> JobHandle:
    """Submit an RL training job as a single coordinator job.

    The coordinator creates all shared actors and child worker jobs.
    Killing the coordinator kills all children (Iris cascading cleanup).
    """
    client = current_client()

    env = {"EQX_ON_ERROR": "nan"}
    env = _add_run_env_variables(env)

    # Use pip_packages instead of extras to avoid uv re-resolving the full dependency
    # tree and replacing TPU torch with CUDA torch (vllm-tpu pulls in CUDA deps).
    return client.submit(
        JobRequest(
            name=f"rl-{config.run_id}",
            entrypoint=Entrypoint.from_callable(_run_rl_coordinator, args=(config,)),
            resources=ResourceConfig.with_cpu(preemptible=False),
            environment=create_environment(
                env_vars=env,
                pip_packages=["vllm-tpu==0.13.2.post6", "sympy", "pylatexenc", "math-verify"],
            ),
            max_retries_failure=0,
            max_retries_preemption=0,
        )
    )


def _run_rl_coordinator(config: RLJobConfig) -> None:
    """In-cluster RL coordinator. Creates actors and child jobs, then waits.

    Runs inside the cluster as a real job. Child jobs inherit region,
    namespace, and environment from this coordinator.
    """
    configure_logging(level=logging.INFO)
    logger.info("RL coordinator starting for run %s", config.run_id)

    client = current_client()
    rl_job = RLJob(config)
    train_config, rollout_config = rl_job.to_worker_configs()
    run_config = config.run_config

    # Create shared control-plane actors (non-preemptible, CPU-only)
    runtime = _create_runtime_handles(client, config)
    logger.info("Runtime handles created: curriculum, run_state, weight_transfer coordinator")

    # Create worker environment
    # Use pip_packages instead of extras to avoid uv re-resolving the full dependency
    # tree and replacing TPU torch with CUDA torch (vllm-tpu pulls in CUDA deps).
    env = {"EQX_ON_ERROR": "nan"}
    env = _add_run_env_variables(env)
    worker_env = create_environment(
        env_vars=env,
        pip_packages=["vllm-tpu==0.13.2.post6", "sympy", "pylatexenc", "math-verify"],
    )

    # Resource configs
    inference_tpu_type = run_config.inference_tpu_type or run_config.train_tpu_type
    # All Iris compute is preemptible — never set preemptible=False
    train_resources = ResourceConfig.with_tpu(
        run_config.train_tpu_type,
        slice_count=run_config.num_train_slices,
    )
    rollout_resources = ResourceConfig.with_tpu(inference_tpu_type)

    # Submit child jobs
    jobs: list[JobHandle] = []

    # Training worker
    jobs.append(
        client.submit(
            JobRequest(
                name=f"rl-{config.run_id}-train",
                entrypoint=Entrypoint.from_callable(_train_worker_entry, args=(train_config, runtime)),
                resources=train_resources,
                environment=worker_env,
                max_retries_failure=run_config.max_retries_failure,
                max_retries_preemption=run_config.max_retries_preemption,
            )
        )
    )

    # Rollout workers
    for i in range(run_config.num_rollout_workers):
        worker_config = dataclasses.replace(
            rollout_config,
            seed=rollout_config.seed + i,
            run_id=f"{rollout_config.run_id}-rollout-{i}",
            worker_index=i,
        )
        jobs.append(
            client.submit(
                JobRequest(
                    name=f"rl-{config.run_id}-rollout-{i}",
                    entrypoint=Entrypoint.from_callable(_rollout_worker_entry, args=(worker_config, runtime)),
                    resources=rollout_resources,
                    environment=worker_env,
                    max_retries_failure=run_config.max_retries_failure,
                    max_retries_preemption=run_config.max_retries_preemption,
                )
            )
        )

    logger.info("Submitted %d child jobs (1 trainer + %d rollout workers)", len(jobs), run_config.num_rollout_workers)

    wait_all(jobs, raise_on_failure=True)
    logger.info("RL coordinator finished for run %s", config.run_id)


def _create_runtime_handles(client: Client, config: RLJobConfig) -> RLRuntimeHandles:
    """Create all shared actors for the RL run.

    Uses host_actor() to run lightweight actors in-process on the coordinator.
    This avoids needing separate CPU worker slots for each actor.
    """
    from marin.rl.curriculum import Curriculum

    # Host actors in-process on the coordinator (no separate jobs needed)
    curriculum_hosted = client.host_actor(
        Curriculum,
        config.curriculum,
        name=f"rl-{config.run_id}-curriculum",
    )

    run_state_hosted = client.host_actor(
        RLRunState,
        name=f"rl-{config.run_id}-run-state",
    )

    # Weight transfer coordinator (Arrow Flight)
    from marin.rl.weight_transfer import WeightTransferMode

    arrow_coordinator = None
    if config.weight_transfer.mode == WeightTransferMode.ARROW_FLIGHT:
        arrow_hosted = client.host_actor(
            ArrowFlightCoordinator,
            name=config.weight_transfer.coordinator_name or f"rl-{config.run_id}-wt-coord",
        )
        arrow_coordinator = arrow_hosted.handle

    return RLRuntimeHandles(
        curriculum=curriculum_hosted.handle,
        run_state=run_state_hosted.handle,
        weight_transfer=WeightTransferRuntime(arrow_flight_coordinator=arrow_coordinator),
    )


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
            try:
                runtime.run_state.mark_failed.remote("trainer crashed").result()
            except Exception:
                logger.exception("Failed to signal run_state failure")
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
