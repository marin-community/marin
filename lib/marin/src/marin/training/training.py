# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
import logging
import os
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import TypeVar

import draccus
import levanter.infra.cli_helpers
from fray.v2 import (
    CpuConfig,
    Entrypoint,
    GpuConfig,
    JobRequest,
    JobStatus,
    ResourceConfig,
    TpuConfig,
    create_environment,
    current_client,
    get_tpu_topology,
    wait_all,
)
from levanter.main import train_dpo
from levanter.main import train_lm
from levanter.main.train_dpo import TrainDpoConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.tracker.wandb import WandbConfig
from mergedeep import mergedeep

from iris.marin_fs import REGION_TO_TMP_BUCKET, check_gcs_paths_same_region, marin_temp_bucket
from levanter.elastic import (
    MARIN_ELASTIC_GROUP_ID_ENV,
    MARIN_ELASTIC_WORKER_COUNT_ENV,
    MARIN_ELASTIC_WORKER_ID_ENV,
    read_completion,
    resolve_elastic_paths,
)
from levanter.utils import fsspec_utils

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainLmOnPodConfig:
    """Configuration for language model training on a pod."""

    train_config: train_lm.TrainLmConfig
    resources: ResourceConfig
    output_path: str | None = None
    """Base output directory to be used for training, mainly for use with executor framework."""
    impute_run_id_from_output_path: bool = True
    """
    If true and out_path is not None, the run id will be set to the basename of the out_path plus a random string.

    Note that trainer.id and the RUN_ID env variable take precedence, in that order.
    """
    env_vars: dict[str, str] | None = None
    """Environment variables to pass to the training task (e.g., WANDB_MODE, WANDB_API_KEY)."""
    auto_build_caches: bool = False
    """Whether to allow Levanter to build dataset caches on the fly.

    Defaults to False so Marin jobs fail fast when a cache is missing instead of
    spending time (and money) building it during training. Override to True if
    you explicitly want cache construction.
    """


@dataclass(frozen=True)
class TrainDpoOnPodConfig:
    """Configuration for DPO training on a pod."""

    train_config: TrainDpoConfig
    resources: ResourceConfig
    output_path: str | None = None
    """Base output directory to be used for training, mainly for use with executor framework."""
    impute_run_id_from_output_path: bool = True
    """
    If true and out_path is not None, the run id will be set to the basename of the out_path plus a random string.

    Note that trainer.id and the RUN_ID env variable take precedence, in that order.
    """
    env_vars: dict[str, str] | None = None
    """Environment variables to pass to the training task (e.g., WANDB_MODE, WANDB_API_KEY)."""
    auto_build_caches: bool = False
    """Whether to allow Levanter to build dataset caches on the fly.

    Defaults to False so Marin jobs fail fast when a cache is missing instead of
    spending time (and money) building it during training. Override to True if
    you explicitly want cache construction.
    """


TrainConfigT = TypeVar("TrainConfigT", TrainLmConfig, TrainDpoConfig)
TrainOnPodConfigT = TypeVar("TrainOnPodConfigT", TrainLmOnPodConfig, TrainDpoOnPodConfig)

DEFAULT_CHECKPOINTS_PATH = "checkpoints"
DEFAULT_HF_CHECKPOINTS_PATH = "hf"
ELASTIC_DATA_SEED_STRIDE = 10_000
MARIN_FAULT_INJECTION_BY_WORKER_ENV = "MARIN_FAULT_INJECTION_BY_WORKER"
MARIN_FAULT_INJECTION_STEPS_ENV = "MARIN_FAULT_INJECTION_STEPS"


def _update_config_to_use_out_path(pod_config: TrainOnPodConfigT) -> TrainOnPodConfigT:
    """
    Update the config to use the out_path as the base output directory for training.

    This will set the following paths to be subdirectories of the out_path:
    * checkpoints (in $out_path/checkpoints)
    * hf checkpoints (in $out_path/hf)
    * logging (in $out_path/log)

    This is useful when running with the executor framework, where the output path is set by the executor.
    """
    if pod_config.output_path is None:
        return pod_config

    trainer = replace(
        pod_config.train_config.trainer,
        checkpointer=replace(
            pod_config.train_config.trainer.checkpointer,
            base_path=os.path.join(pod_config.output_path, DEFAULT_CHECKPOINTS_PATH),
        ),
    )

    config = replace(
        pod_config.train_config,
        trainer=trainer,
        hf_save_path=os.path.join(pod_config.output_path, DEFAULT_HF_CHECKPOINTS_PATH),
    )
    return replace(pod_config, train_config=config)


def _suppress_ray_config(config: TrainConfigT) -> TrainConfigT:
    """
    Levanter wants to auto-start the Ray cluster, but we're already in a Ray cluster. Disable that.
    """
    if config.trainer.ray.auto_start_cluster:
        logger.info("Ray cluster is set to auto-start, but that's not what we want for Marin. Disabling.")
        return replace(
            config,
            trainer=replace(
                config.trainer,
                ray=replace(config.trainer.ray, auto_start_cluster=False, start_workers=False),
            ),
        )
    elif config.trainer.ray.start_workers:
        logger.info("Ray cluster is set to start workers, but that's not what we want for Marin. Disabling.")
        return replace(
            config,
            trainer=replace(config.trainer, ray=replace(config.trainer.ray, start_workers=False)),
        )
    return config


def _maybe_override_auto_build_caches(config: TrainConfigT, auto_build: bool) -> TrainConfigT:
    data = config.data
    if data.auto_build_caches != auto_build:
        logger.info("Overriding auto_build_caches to %s", auto_build)
        data = dataclasses.replace(data, auto_build_caches=auto_build)
        config = replace(config, data=data)
    return config


def _enforce_run_id(config: TrainOnPodConfigT) -> TrainOnPodConfigT:
    """
    Levanter will auto-generate a run ID if it's not set. We want to enforce that it's set, so that it resumes
    properly after preemption.

    Look for:
        * config.trainer.id
        * environment variable RUN_ID in config.env_vars
        * environment variable RUN_ID
        * default to a random UID
    """
    run_id = config.train_config.trainer.id

    if run_id is None:
        run_id = (config.env_vars or {}).get("RUN_ID", os.environ.get("RUN_ID"))

    if run_id is None and config.impute_run_id_from_output_path and config.output_path is not None:
        path = config.output_path
        path = path.rstrip("/")
        run_id = os.path.basename(path)
        logger.info(f"Imputing run ID from out path: {run_id}")

    if not run_id:
        run_id = levanter.infra.cli_helpers.default_run_id()
        logger.warning(f"Run ID not set. Using default: {run_id}")

    append_id_to_checkpoints = not config.impute_run_id_from_output_path
    checkpointer_config = replace(
        config.train_config.trainer.checkpointer, append_run_id_to_base_path=append_id_to_checkpoints
    )

    inner_config = replace(
        config.train_config, trainer=replace(config.train_config.trainer, id=run_id, checkpointer=checkpointer_config)
    )
    return replace(config, train_config=inner_config)


def _requested_tpu_slice_count(resources: ResourceConfig) -> int:
    if not isinstance(resources.device, TpuConfig):
        return 1
    topology = get_tpu_topology(resources.device.variant)
    return max(1, resources.replicas // topology.vm_count)


def _elastic_worker_count(config: TrainOnPodConfigT) -> int:
    elastic = config.train_config.trainer.elastic
    if not elastic.enabled or not isinstance(config.resources.device, TpuConfig):
        return 1
    if elastic.worker_count is not None:
        return elastic.worker_count
    return _requested_tpu_slice_count(config.resources)


def _single_slice_resources(resources: ResourceConfig) -> ResourceConfig:
    if not isinstance(resources.device, TpuConfig):
        return resources
    topology = get_tpu_topology(resources.device.variant)
    return replace(resources, replicas=topology.vm_count)


def _shared_elastic_state_path(config: TrainOnPodConfigT) -> str:
    assert config.train_config.trainer.id is not None
    return fsspec_utils.join_path(
        config.train_config.trainer.checkpointer.base_path,
        f"_elastic/{config.train_config.trainer.id}",
    )


def _with_elastic_worker_assignment(
    config: TrainOnPodConfigT,
    *,
    worker_index: int,
    worker_count: int,
    fault_injection_by_worker: str | None = None,
) -> tuple[TrainOnPodConfigT, dict[str, str]]:
    logical_run_id = config.train_config.trainer.id
    assert logical_run_id is not None

    worker_id = f"w{worker_index:03d}"
    worker_run_id = f"{logical_run_id}-{worker_id}"
    elastic = replace(
        config.train_config.trainer.elastic,
        group_id=logical_run_id,
        worker_id=worker_id,
        worker_count=worker_count,
        state_path=_shared_elastic_state_path(config),
    )
    trainer = replace(
        config.train_config.trainer,
        id=worker_run_id,
        elastic=elastic,
        tracker=_tracker_for_elastic_worker(config.train_config.trainer.tracker, logical_run_id, worker_id),
    )

    train_config = replace(config.train_config, trainer=trainer)
    if hasattr(train_config, "data_seed"):
        base_data_seed = config.train_config.data_seed
        if base_data_seed is None:
            base_data_seed = config.train_config.trainer.seed
        train_config = replace(train_config, data_seed=base_data_seed + worker_index * ELASTIC_DATA_SEED_STRIDE)

    worker_config = replace(
        config,
        train_config=train_config,
        resources=_single_slice_resources(config.resources),
    )
    worker_env = {
        MARIN_ELASTIC_GROUP_ID_ENV: logical_run_id,
        MARIN_ELASTIC_WORKER_ID_ENV: worker_id,
        MARIN_ELASTIC_WORKER_COUNT_ENV: str(worker_count),
        "RUN_ID": worker_run_id,
    }
    worker_fault_steps = _fault_steps_for_worker(fault_injection_by_worker, worker_id)
    if worker_fault_steps is not None:
        worker_env[MARIN_FAULT_INJECTION_STEPS_ENV] = worker_fault_steps
    return worker_config, worker_env


def _tracker_for_elastic_worker(tracker, logical_run_id: str, worker_id: str):
    def _rewrite_tracker_entry(entry):
        if not isinstance(entry, WandbConfig):
            return entry

        base_name = entry.name or logical_run_id
        return replace(
            entry,
            group=entry.group or logical_run_id,
            name=f"{base_name}-{worker_id}",
        )

    if isinstance(tracker, tuple):
        return tuple(_rewrite_tracker_entry(entry) for entry in tracker)
    if isinstance(tracker, list):
        return [_rewrite_tracker_entry(entry) for entry in tracker]
    return _rewrite_tracker_entry(tracker)


def _fault_steps_for_worker(fault_injection_by_worker: str | None, worker_id: str) -> str | None:
    if fault_injection_by_worker is None:
        return None

    payload = json.loads(fault_injection_by_worker)
    if not isinstance(payload, dict):
        raise ValueError(f"{MARIN_FAULT_INJECTION_BY_WORKER_ENV} must be a JSON object mapping worker ids to step lists")

    worker_steps = payload.get(worker_id)
    if worker_steps is None:
        return None
    if isinstance(worker_steps, int):
        worker_steps = [worker_steps]
    if not isinstance(worker_steps, list):
        raise ValueError(
            f"{MARIN_FAULT_INJECTION_BY_WORKER_ENV}[{worker_id!r}] must be an int or list of ints, "
            f"got {type(worker_steps)}"
        )

    return json.dumps([int(step) for step in worker_steps])


def _elastic_completion_path(config: TrainOnPodConfigT) -> str:
    assert config.train_config.trainer.id is not None
    paths = resolve_elastic_paths(
        config.train_config.trainer.checkpointer.expanded_path(config.train_config.trainer.id),
        replace(config.train_config.trainer.elastic, state_path=_shared_elastic_state_path(config)),
        run_id=config.train_config.trainer.id,
    )
    return paths.completion_path


def _wait_for_elastic_jobs(jobs, *, completion_path: str) -> None:
    statuses = wait_all(jobs, raise_on_failure=False)
    completion = read_completion(completion_path)
    if completion is not None:
        logger.info(
            "Elastic training completed by worker %s at step %s",
            completion.worker_id,
            completion.completed_step,
        )
        return

    if all(status == JobStatus.SUCCEEDED for status in statuses):
        return

    raise RuntimeError(f"Elastic training group did not report completion. Final statuses: {statuses}")


def _normalize_jax_compilation_cache_dir(path: str) -> str:
    """Normalize cache dir to a form accepted by JAX's compilation cache.

    JAX's ``LRUCache`` delegates I/O to ``etils.epath.Path`` which supports
    local paths, ``gs://`` (via gcsfs), and ``s3://`` (via s3fs/fsspec).
    The only scheme that causes problems is ``file://`` which raises during
    initialization.
    """
    if path.startswith("file://"):
        return path.removeprefix("file://")
    return path


def _disable_xla_autotune_subcache(env: dict) -> None:
    """Disable XLA's per-fusion autotune sub-cache for remote compilation caches.

    JAX automatically places XLA sub-caches (autotune, kernel cache) as
    subdirectories of the compilation cache dir.  The autotune cache uses
    XLA's C++ ``tsl::Env`` which only supports local paths — it crashes on
    ``gs://`` and ``s3://``.  Since the autotune cache is ephemeral (skipped
    entirely on a JAX cache hit) and only saves minutes on cold compiles,
    we disable it via the JAX config rather than trying to redirect it.
    """
    cache_dir = env.get("JAX_COMPILATION_CACHE_DIR", "")
    if "://" not in cache_dir:
        return
    if "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES" in env:
        return
    env["JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES"] = "none"
    logger.info("XLA sub-caches disabled (compilation cache is remote: %s)", cache_dir)


def _compilation_cache_dir_for_resources(resources: ResourceConfig) -> str:
    target_regions = sorted(set(resources.regions or []))
    if len(target_regions) == 1:
        if bucket := REGION_TO_TMP_BUCKET.get(target_regions[0]):
            return f"gs://{bucket}/ttl=30d/compilation-cache"
    return _normalize_jax_compilation_cache_dir(marin_temp_bucket(ttl_days=30, prefix="compilation-cache"))


def run_levanter_train_lm(config: TrainLmOnPodConfig):
    """
    Run the Levanter training main function on a Ray cluster.

    This function is designed to be run on your machine or with sufficient variables in the env dict/os env.
    It should also be run with a Ray cluster already running.

    - WANDB_API_KEY: The API key for Weights and Biases.
    - RUN_ID: (Optional) The run ID for this training run. Will default to a random UID if not set.
    - GIT_COMMIT: (Optional) The git commit hash of the current codebase. Will attempt to fetch it if not set.

    This function makes a number of changes to the config and ensures a few things are set:
    - The run ID is set, or sets a default if not.
    - WANDB_API_KEY is set.
    - It disables the auto-ray-start and auto-worker-start options since we're already in a Ray cluster.
    - It checks that configured GCS paths are in the same region as the VM (except train/validation source URLs).
    """
    default_launch_config = levanter.infra.cli_helpers.load_config()

    if config.output_path is not None:
        logger.info(f"Using output path: {config.output_path}")
        config = _update_config_to_use_out_path(config)

    env = _add_default_env_variables(
        config.env_vars or {},
        default_launch_config.env_for_accel(config.resources.device.variant),
    )
    # if we're on tpu, ensure we have wandb
    if isinstance(config.resources.device, TpuConfig):
        _check_for_wandb_key(env)

    env = _add_run_env_variables(env)

    if "JAX_COMPILATION_CACHE_DIR" not in env:
        env["JAX_COMPILATION_CACHE_DIR"] = _compilation_cache_dir_for_resources(config.resources)
        logger.info("JAX compilation cache: %s", env["JAX_COMPILATION_CACHE_DIR"])
    _disable_xla_autotune_subcache(env)

    config = _enforce_run_id(config)
    logger.info(f"Using run ID: {config.train_config.trainer.id}")

    model_config = config.train_config.model
    logger.info(
        "Model config: type=%s seq_len=%d hidden=%d batch=%s device=%s",
        type(model_config).__name__,
        model_config.max_seq_len,
        model_config.Embed.size,
        config.train_config.trainer.train_batch_size,
        config.resources.device,
    )

    train_config = config.train_config
    train_config = _suppress_ray_config(train_config)
    train_config = _maybe_override_auto_build_caches(train_config, config.auto_build_caches)

    # disable accelerator requirement when running without GPU/TPU resources
    if config.resources.device.kind == "cpu":
        trainer = replace(train_config.trainer, require_accelerator=False)
        train_config = replace(train_config, trainer=trainer)

    if not isinstance(config.resources.device, CpuConfig):
        _doublecheck_paths(config)

    config = replace(config, train_config=train_config)
    client = current_client()

    extras = []
    if isinstance(config.resources.device, TpuConfig):
        extras.append("tpu")
    elif isinstance(config.resources.device, GpuConfig):
        extras.append("gpu")

    worker_count = _elastic_worker_count(config)
    fault_injection_by_worker = env.pop(MARIN_FAULT_INJECTION_BY_WORKER_ENV, None)
    if worker_count > 1:
        completion_path = _elastic_completion_path(config)
        jobs = []
        for worker_index in range(worker_count):
            worker_config, worker_env = _with_elastic_worker_assignment(
                config,
                worker_index=worker_index,
                worker_count=worker_count,
                fault_injection_by_worker=fault_injection_by_worker,
            )
            job_request = JobRequest(
                name=f"train_lm-{worker_env[MARIN_ELASTIC_WORKER_ID_ENV]}",
                entrypoint=Entrypoint.from_callable(train_lm.main, args=[worker_config.train_config]),
                resources=worker_config.resources,
                environment=create_environment(env_vars={**env, **worker_env}, extras=extras),
                max_retries_failure=10,
            )
            jobs.append(client.submit(job_request))
        _wait_for_elastic_jobs(jobs, completion_path=completion_path)
        return

    # Note: Using a constant job name allows restarts to adopt the existing job handle
    # instead of raising a duplicate name error (adopt_existing=True is the default).
    job_request = JobRequest(
        name="train_lm",
        entrypoint=Entrypoint.from_callable(train_lm.main, args=[train_config]),
        resources=config.resources,
        environment=create_environment(env_vars=env, extras=extras),
        max_retries_failure=10,
    )
    job = client.submit(job_request)
    job.wait(raise_on_failure=True)


def run_levanter_train_dpo(config: TrainDpoOnPodConfig):
    """
    Run the Levanter DPO training main function on a Ray cluster.

    This function is designed to be run on your machine or with sufficient variables in the env dict/os env.
    It should also be run with a Ray cluster already running.
    """
    default_launch_config = levanter.infra.cli_helpers.load_config()

    if config.output_path is not None:
        logger.info(f"Using output path: {config.output_path}")
        config = _update_config_to_use_out_path(config)

    env = _add_default_env_variables(
        config.env_vars or {},
        default_launch_config.env_for_accel(config.resources.device.variant),
    )
    if isinstance(config.resources.device, TpuConfig):
        _check_for_wandb_key(env)

    env = _add_run_env_variables(env)

    if "JAX_COMPILATION_CACHE_DIR" not in env:
        env["JAX_COMPILATION_CACHE_DIR"] = _compilation_cache_dir_for_resources(config.resources)
        logger.info("JAX compilation cache: %s", env["JAX_COMPILATION_CACHE_DIR"])
    _disable_xla_autotune_subcache(env)

    config = _enforce_run_id(config)
    logger.info(f"Using run ID: {config.train_config.trainer.id}")

    train_config = config.train_config
    train_config = _suppress_ray_config(train_config)
    train_config = _maybe_override_auto_build_caches(train_config, config.auto_build_caches)

    if config.resources.device.kind == "cpu":
        trainer = replace(train_config.trainer, require_accelerator=False)
        train_config = replace(train_config, trainer=trainer)

    if not isinstance(config.resources.device, CpuConfig):
        _doublecheck_paths(config)

    config = replace(config, train_config=train_config)
    client = current_client()

    extras = []
    if isinstance(config.resources.device, TpuConfig):
        extras.append("tpu")
    elif isinstance(config.resources.device, GpuConfig):
        extras.append("gpu")

    worker_count = _elastic_worker_count(config)
    fault_injection_by_worker = env.pop(MARIN_FAULT_INJECTION_BY_WORKER_ENV, None)
    if worker_count > 1:
        completion_path = _elastic_completion_path(config)
        jobs = []
        for worker_index in range(worker_count):
            worker_config, worker_env = _with_elastic_worker_assignment(
                config,
                worker_index=worker_index,
                worker_count=worker_count,
                fault_injection_by_worker=fault_injection_by_worker,
            )
            job_request = JobRequest(
                name=f"train_dpo-{worker_env[MARIN_ELASTIC_WORKER_ID_ENV]}",
                entrypoint=Entrypoint.from_callable(train_dpo.main, args=[worker_config.train_config]),
                resources=worker_config.resources,
                environment=create_environment(env_vars={**env, **worker_env}, extras=extras),
                max_retries_failure=10,
            )
            jobs.append(client.submit(job_request))
        _wait_for_elastic_jobs(jobs, completion_path=completion_path)
        return

    job_request = JobRequest(
        name="train_dpo",
        entrypoint=Entrypoint.from_callable(train_dpo.main, args=[train_config]),
        resources=config.resources,
        environment=create_environment(env_vars=env, extras=extras),
        max_retries_failure=10,
    )
    job = client.submit(job_request)
    job.wait(raise_on_failure=True)


def _doublecheck_paths(config: TrainOnPodConfigT):
    """
    Double-check that we're not using local paths in some of the standard places that Levanter sets defaults.
    Also check that the paths are in the same region as the VM, to avoid performance issues and billing surprises.

    This function recursively examines all strings/paths in the config to identify GCS paths and checks their regions.
    """
    local_ok = not isinstance(config.resources.device, TpuConfig)
    target_regions = sorted(set(config.resources.regions or []))
    target_region = None
    if len(target_regions) == 1:
        target_region = target_regions[0]
    elif len(target_regions) > 1:
        raise ValueError(
            "Path region validation requires a single target region when launching TPU jobs remotely. "
            f"Got regions={target_regions}."
        )

    check_gcs_paths_same_region(
        config.train_config,
        local_ok=local_ok,
        region=target_region,
    )
    return config


def _add_default_env_variables(env: dict, default_env: dict | None):
    if default_env is not None:
        default_env = deepcopy(default_env)
        env = mergedeep.merge(default_env, env)

    # Ray gets mad if the values aren't all strings, but e.g. ints
    env = {str(k): str(v) for k, v in env.items()}
    return env


def _add_run_env_variables(env: dict):
    """
    Add a few environment variables from `os.environ` into `env` that we need for logging as well as for internal evals.
    Specifically:
    - GIT_COMMIT
    - HF_DATASETS_TRUST_REMOTE_CODE
    - HF_ALLOW_CODE_EVAL (for code evaluation tasks like HumanEval)
    """
    env = deepcopy(env)

    git_commit = env.get("GIT_COMMIT") or os.environ.get("GIT_COMMIT")

    if not git_commit:
        try:
            git_commit = levanter.infra.cli_helpers.get_git_commit()
        except:  # noqa
            pass

    if git_commit:
        env["GIT_COMMIT"] = git_commit
    else:
        logger.warning("Failed to find or infer git commit for logging.")

    # required for internal evals to run some tasks
    if "HF_DATASETS_TRUST_REMOTE_CODE" not in env:
        env["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"

    # required for code evaluation tasks like HumanEval
    if "HF_ALLOW_CODE_EVAL" not in env:
        env["HF_ALLOW_CODE_EVAL"] = "1"

    if "TOKENIZERS_PARALLELISM" not in env:
        env["TOKENIZERS_PARALLELISM"] = "false"

    if "TPU_MIN_LOG_LEVEL" not in env:
        env["TPU_MIN_LOG_LEVEL"] = "2"
    if "TPU_STDERR_LOG_LEVEL" not in env:
        env["TPU_STDERR_LOG_LEVEL"] = "2"

    # Allow the caller (or iris -e) to override the compilation cache dir.
    if "JAX_COMPILATION_CACHE_DIR" not in env:
        if val := os.environ.get("JAX_COMPILATION_CACHE_DIR"):
            env["JAX_COMPILATION_CACHE_DIR"] = val

    return env


def _check_for_wandb_key(env):
    if env.get("WANDB_API_KEY") is None:
        key = os.environ.get("WANDB_API_KEY")
        if key is not None:
            env["WANDB_API_KEY"] = key
        else:
            wandb_disabled = env.get("WANDB_MODE", os.environ.get("WANDB_MODE"))
            if wandb_disabled is None or wandb_disabled.lower() not in {"disabled", "offline", "dryrun"}:
                raise ValueError(
                    "WANDB_API_KEY must be set in the environment. Please add it to your .config, export "
                    "WANDB_API_KEY=..., or add it to the env dict."
                )


if __name__ == "__main__":
    draccus.wrap()(run_levanter_train_lm)()
