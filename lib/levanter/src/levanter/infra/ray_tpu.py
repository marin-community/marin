# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""
Backward compatibility wrapper for TPU orchestration.

TPU orchestration has been migrated to fray.cluster.ray.tpu.
This module re-exports those functions for backward compatibility.
Docker-related functions remain in levanter as levanter-specific infrastructure.
"""

import dataclasses
import logging
import multiprocessing
import os
import subprocess
import tempfile
import time
from asyncio import QueueEmpty
from dataclasses import dataclass
from typing import Optional, Sequence

import draccus
import ray
from ray.dashboard.modules.job.sdk import JobSubmissionClient

from levanter.infra.docker import make_docker_run_command
from levanter.utils.ray_utils import ser_exc_info

# Re-export all TPU orchestration from fray
from fray.cluster.ray.tpu import (
    HEALTH_CHECK_TIMEOUT,
    START_ACTOR_TIMEOUT,
    TPU_CONFIGS,
    MultisliceInfo,
    ResourcePoolManager,
    SliceActor,
    SliceInfo,
    SlicePoolManager,
    TPUConfig,
    TPUHostActor,
    TPUHostInfo,
    TpuCancelled,
    TpuFailed,
    TpuPreempted,
    TpuRunError,
    TpuSuccess,
    get_current_tpu_is_preempted,
    get_tpu_config,
    run_on_pod,
    run_on_pod_multislice,
    run_on_pod_multislice_resumable,
    run_on_pod_ray,
    run_on_pod_resumable,
)

__all__ = [
    "HEALTH_CHECK_TIMEOUT",
    "START_ACTOR_TIMEOUT",
    "TPU_CONFIGS",
    "MultisliceInfo",
    "ResourcePoolManager",
    "SliceActor",
    "SliceInfo",
    "SlicePoolManager",
    "TPUConfig",
    "TPUHostActor",
    "TPUHostInfo",
    "TpuCancelled",
    "TpuFailed",
    "TpuPreempted",
    "TpuRunError",
    "TpuSuccess",
    "get_current_tpu_is_preempted",
    "get_tpu_config",
    "run_on_pod",
    "run_on_pod_multislice",
    "run_on_pod_multislice_resumable",
    "run_on_pod_ray",
    "run_on_pod_resumable",
    # Docker functions below (levanter-specific)
    "run_docker_on_pod",
    "RunDockerOnPodConfig",
    "submit_tpu_job_on_ray",
]

logger = logging.getLogger("ray")


def _run_command(*args, **kwargs):
    return subprocess.check_call(args, **kwargs)


def run_docker_on_pod(
    image_id: str,
    command: Sequence[str],
    *,
    tpu_type: str,
    num_slices: int | Sequence[int],
    env: dict,
    name: str = "levanter",
    retries: int = 10,
):
    env = _massage_env(env)

    docker_cmd = make_docker_run_command(image_id, command, env=env, foreground=True, name=name)

    def run_docker():
        _kill_old_container(name)
        try:
            return _run_command(*docker_cmd)
        except subprocess.CalledProcessError as e:
            logger.exception("Failed to run docker command")
            raise e

    run_on_pod(
        ray.remote(max_calls=1)(run_docker),
        tpu_type=tpu_type,
        num_slices=num_slices,
        max_retries_failure=retries,
        max_retries_preemption=10000,
    )


def _kill_old_container(name):
    try:
        logger.info(f"Killing old container {name}")
        _run_command("sudo", "docker", "rm", "-f", name)
    except subprocess.CalledProcessError:
        pass


def _separate_process_fn(underlying_function, args, kwargs):
    """
    Helper function for _forkify_remote_fn. This runs the function in a separate process.
    """

    def target_fn(queue, args, kwargs):
        try:
            # Call the original function
            result = underlying_function(*args, **kwargs)
            queue.put((True, result))  # Success, put the result
        except Exception as e:
            # Capture and return the full traceback in case of an exception
            info = ser_exc_info(e)
            queue.put((False, info))

    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=target_fn, args=(queue, args, kwargs))
    process.start()
    process.join()

    # Retrieve the result or error from the queue
    logger.info("Process finished")
    try:
        success, value = queue.get(timeout=1)
    except QueueEmpty:
        logger.error("Process timed out")
        process.terminate()
        raise TimeoutError("Process timed out")

    if success:
        return value
    else:
        value.reraise()


def _hacky_remove_tpu_lockfile():
    """
    This is a hack to remove the lockfile that TPU pods create on the host filesystem.

    libtpu only allows one process to access the TPU at a time, and it uses a lockfile to enforce this.
    Ordinarily a lockfile would be removed when the process exits, but in the case of Ray, the process is
    a long-running daemon that doesn't typically exit until the node is shut down. This means that the lockfile
    persists across Ray tasks. This doesn't apply to our docker-based workloads, but it does apply to other
    tasks that use JAX directly.
    """
    if os.path.exists("/tmp/libtpu_lockfile"):
        try:
            os.unlink("/tmp/libtpu_lockfile")
        except FileNotFoundError:
            pass
        except PermissionError:
            logger.warning("Failed to remove lockfile")
            try:
                os.system("sudo rm /tmp/libtpu_lockfile")
            except Exception:  # noqa
                pass


@dataclass
class RunDockerOnPodConfig:
    image_id: str
    command: list[str] | str
    tpu_type: str
    env: dict = dataclasses.field(default_factory=dict)
    name: str = "levanter"
    retries: int = 10
    node_count: int = 1


def submit_tpu_job_on_ray(config: RunDockerOnPodConfig, ray_address: str, run_id: Optional[str] = None):
    """
    Submit a job to run on a TPU pod on a Ray cluster. This programmatically submits a job to the Ray cluster.
    This should be run on your local machine, not on the Ray cluster itself.

    If run_id is not provided, a default run ID will be generated.
    """

    with tempfile.NamedTemporaryFile(suffix=".yaml", prefix=f"launch-{run_id}-", dir=".") as f:
        yaml = draccus.dump(config)
        f.write(yaml.encode("utf-8"))
        f.flush()

        f_name = os.path.relpath(f.name)
        logger.info(f"Submitting job with config path {f_name}")

        client = JobSubmissionClient(ray_address)

        job_id = _make_unique_job_id(client, run_id) if run_id is not None else None

        job_id = client.submit_job(
            entrypoint=f"python -m levanter.infra.ray_tpu --config_path {f_name}",
            runtime_env={"working_dir": ".", "env_vars": {"PYTHONPATH": "src:."}},
            submission_id=job_id,
        )

        return job_id


# try to make the job id be the same as the run id, but if it already exists, just make it unique
def _make_unique_job_id(client, run_id):
    job_id = run_id
    try:
        while client.get_job_status(job_id) is not None:
            job_id = f"{run_id}-{time.time_ns()}"
    except Exception as e:  # noqa
        if "does not exist" in str(e):
            pass
        else:
            raise
    return job_id


@draccus.wrap()
def main(args: RunDockerOnPodConfig):
    """
    *This command is designed to run on a Ray cluster, not on your local machine. You probably want submit_tpu_job_on_ray.*

    Run a command on a TPU pod. This is a wrapper around `run_docker_on_pod` that takes a config object as a CLI.

    We use this via infra/launch_on_ray.py to run docker containers on TPUs.
    """

    import shlex

    if isinstance(args.command, str):
        command = shlex.split(args.command)
    else:
        command = args.command

    run_docker_on_pod(
        args.image_id,
        command,
        tpu_type=args.tpu_type,
        env=args.env,
        name=args.name,
        retries=args.retries,
        num_slices=args.node_count,
    )


def _massage_env(env):
    # Ray pretends it's running in a TTY, which leads to a ton of log spam from tqdm.
    # Levanter uses tqdm_loggable, which tries to sniff out the TTY, but it doesn't work with Ray.
    # So we force it
    env = dict(env)
    if "TERM" not in env:
        env["TERM"] = "dumb"

    if "TF_CPP_MIN_LOG_LEVEL" not in env:
        # Suppress TensorFlow logs, which can be very verbose
        env["TF_CPP_MIN_LOG_LEVEL"] = "3"

    return env


if __name__ == "__main__":
    main()

    # leaving this here for testing purposes
    # ray.init()
    # tpu_type = "v4-8"
    # num_slices = 2
    #
    # @ray.remote(max_calls=1)
    # def fn():
    #     import jax
    #     import jax.random as jrandom
    #     from jax.lax import with_sharding_constraint
    #     from jax.sharding import Mesh
    #     from jax.sharding import PartitionSpec as P
    #
    #     mesh = Mesh(jax.devices("tpu"), ("x",))
    #     print(jax.devices())
    #
    #     @jax.jit
    #     def init():
    #         with mesh:
    #             x = jrandom.normal(jrandom.PRNGKey(0), (32,))
    #             weights = jrandom.normal(jrandom.PRNGKey(1), (32, 4))
    #             bias = jrandom.normal(jrandom.PRNGKey(2), (4,))
    #
    #             x_sharded = with_sharding_constraint(x, P("x"))
    #             weights_sharded = with_sharding_constraint(weights, P("x"))
    #             return x_sharded, weights_sharded, bias
    #
    #     x, weights, bias = init()
    #
    #     @jax.jit
    #     def layer(x, weights, bias):
    #         with mesh:
    #             return with_sharding_constraint(jax.nn.sigmoid(x @ weights + bias), P())
    #
    #     out = layer(x, weights, bias)
    #
    #     import numpy
    #
    #     return numpy.array(out)
    #
    # results = ray.get(run_on_pod_new(fn, tpu_type, num_slices=num_slices))
    #
    # print(f"Results: {results}")
