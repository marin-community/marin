import dataclasses
import logging
import os
from copy import deepcopy
from dataclasses import dataclass, replace

import draccus
import levanter.infra.cli_helpers
import ray
from google.api_core.exceptions import Forbidden as GcpForbiddenException
from levanter.infra.ray_tpu import run_on_pod_multislice_resumable, run_on_pod_resumable
from levanter.main import train_lm
from levanter.main.train_lm import TrainLmConfig
from mergedeep import mergedeep

from marin.resources import CpuOnlyConfig, ResourceConfig, TpuPodConfig
from marin.utilities.gcs_utils import get_bucket_location, get_vm_region

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
    allow_out_of_region: tuple[str, ...] = ()
    """Tuple of JSON paths (e.g., 'data.cache_dir') that are allowed to be read from or written to different regions."""


DEFAULT_CHECKPOINTS_PATH = "checkpoints"
DEFAULT_HF_CHECKPOINTS_PATH = "hf"


def _update_config_to_use_out_path(pod_config: TrainLmOnPodConfig) -> TrainLmOnPodConfig:
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


def _suppress_ray_config(config: TrainLmConfig) -> TrainLmConfig:
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


def _enforce_run_id(config: TrainLmOnPodConfig) -> TrainLmOnPodConfig:
    """
    Levanter will auto-generate a run ID if it's not set. We want to enforce that it's set, so that it resumes
    properly after preemption.

    Look for:
        * config.trainer.id
        * environment variable RUN_ID in the config
        * environment variable RUN_ID
        * default to a random UID
    """
    run_id = config.train_config.trainer.id

    if run_id is None:
        run_id = config.resources.runtime_env.get("env_vars", {}).get("RUN_ID", os.environ.get("RUN_ID"))

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


@ray.remote(num_cpus=0.1)
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
    - if allow_out_of_region is False, it checks that the data cache paths are in the same region as the VM.
    """
    default_launch_config = levanter.infra.cli_helpers.load_config()

    if config.output_path is not None:
        logger.info(f"Using output path: {config.output_path}")
        config = _update_config_to_use_out_path(config)

    env = _add_default_env_variables(
        config.resources.runtime_env.get("env_vars", {}),
        default_launch_config.env_for_accel(config.resources.accelerator_descriptor() or ""),
    )
    # if we're on tpu, ensure we have wandb
    if isinstance(config.resources, TpuPodConfig):
        _check_for_wandb_key(env)

    env = _add_run_env_variables(env)

    if "JAX_COMPILATION_CACHE_DIR" not in env:
        marin_prefix = os.environ.get("MARIN_PREFIX")
        if marin_prefix:
            env["JAX_COMPILATION_CACHE_DIR"] = os.path.join(marin_prefix, "compilation-cache")
            logger.info(f"JAX compilation cache enabled at: {env['JAX_COMPILATION_CACHE_DIR']}")
        else:
            logger.warning("MARIN_PREFIX environment variable not set. JAX compilation cache will not be configured.")
    hw_config = config.resources.with_env_vars(env)

    config = _enforce_run_id(config)
    logger.info(f"Using run ID: {config.train_config.trainer.id}")

    train_config = config.train_config
    train_config = _suppress_ray_config(train_config)

    # disable accelerator requirement when running without GPU/TPU resources
    if hw_config.accelerator_descriptor() is None:
        trainer = replace(train_config.trainer, require_accelerator=False)
        train_config = replace(train_config, trainer=trainer)

    if not config.allow_out_of_region and not isinstance(hw_config, CpuOnlyConfig):
        # run this on the Ray cluster to get the right region
        # doesn't need to be a TPU because ray insists that all VMs are in the same region
        ray.get(ray.remote(_doublecheck_paths).options(runtime_env=hw_config.runtime_env, num_cpus=0.1).remote(config))

    @ray.remote(**hw_config.as_remote_kwargs(), max_calls=1)
    def train_lm_task():
        train_lm.main(train_config)

    # TODO: abstract this?
    if isinstance(hw_config, TpuPodConfig):
        if hw_config.slice_count == 1:
            return run_on_pod_resumable(train_lm_task, config.resources.accelerator_descriptor(), max_retries_failure=10)
        else:
            return run_on_pod_multislice_resumable(
                train_lm_task,
                config.resources.accelerator_descriptor(),
                hw_config.slice_count,
                max_retries_failure=10,
            )
    else:
        return ray.get(train_lm_task.remote())


def _doublecheck_paths(config: TrainLmOnPodConfig):
    """
    Double-check that we're not using local paths in some of the standard places that Levanter sets defaults.
    Also check that the paths are in the same region as the VM, to avoid performance issues and billing surprises.

    This function recursively examines all strings in the config to identify GCS paths and checks their regions.
    """
    # Determine if we're running locally or if path checks should be bypassed
    allow_out_of_region = config.allow_out_of_region

    local_ok = not isinstance(config.resources, TpuPodConfig)

    try:
        region = get_vm_region()
    except ValueError as e:
        if local_ok:
            logger.warning("Could not determine the region of the VM. This is fine if you're running locally.")
            return
        raise ValueError("Could not determine the region of the VM. This is required for path checks.") from e

    # Recursively check all paths in the config
    _check_paths_recursively(config.train_config, "", region, local_ok, allow_out_of_region)

    return config


def _check_paths_recursively(obj, path_prefix, region, local_ok, allow_out_of_region):
    """
    Recursively check all strings in the config object that look like GCS paths.

    Args:
        obj: The object to check (could be a dict, list, or other object)
        path_prefix: The prefix for the current path (e.g., "config.trainer")
        region: The region of the VM
        local_ok: Whether local paths are allowed
        allow_out_of_region: Tuple of paths that are allowed to be read from or written to different regions
        must_save_checkpoints: Whether checkpoints must be saved
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_prefix = f"{path_prefix}.{key}" if path_prefix else key
            _check_paths_recursively(value, new_prefix, region, local_ok, allow_out_of_region)
    elif isinstance(obj, list | tuple):
        for i, item in enumerate(obj):
            new_prefix = f"{path_prefix}[{i}]"
            _check_paths_recursively(item, new_prefix, region, local_ok, allow_out_of_region)
    elif isinstance(obj, str) and obj.startswith("gs://"):
        # This is a GCS path, check if it's in the right region
        is_allow_listed_path = any(path_prefix.startswith(p) for p in allow_out_of_region)
        # whitelist train and validation urls because we are always cached
        if "train_urls" in path_prefix or "validation_urls" in path_prefix:
            is_allow_listed_path = True

        # Determine if this path should be checked
        if not is_allow_listed_path:
            _check_path_in_region(
                path_prefix,
                obj,
                region=region,
                local_ok=local_ok,
            )
    elif dataclasses.is_dataclass(obj):
        for field in dataclasses.fields(obj):
            new_prefix = f"{path_prefix}.{field.name}" if path_prefix else field.name
            value = getattr(obj, field.name)
            _check_paths_recursively(
                value,
                new_prefix,
                region,
                local_ok,
                allow_out_of_region,
            )
    # allow primitives through, warn on other types
    elif not isinstance(obj, str | int | float | bool | type(None)):
        logger.warning(f"Found unexpected type {type(obj)} at {path_prefix}. Skipping.")


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
    """
    env = deepcopy(env)

    if "GIT_COMMIT" not in env:
        try:
            env["GIT_COMMIT"] = levanter.infra.cli_helpers.get_git_commit()
        except:  # noqa
            logger.warning("Could not infer git commit.")

    # required for internal evals to run some tasks
    if "HF_DATASETS_TRUST_REMOTE_CODE" not in env:
        env["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"

    if "TOKENIZERS_PARALLELISM" not in env:
        env["TOKENIZERS_PARALLELISM"] = "false"

    if "TPU_MIN_LOG_LEVEL" not in env:
        env["TPU_MIN_LOG_LEVEL"] = "2"
    if "TPU_STDERR_LOG_LEVEL" not in env:
        env["TPU_STDERR_LOG_LEVEL"] = "2"

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


def _check_path_in_region(key, path, region, local_ok):

    if not path.startswith("gs://"):
        if local_ok:
            logger.warning(f"{key} is not a GCS path: {path}. This is fine if you're running locally.")
            return
        else:
            raise ValueError(f"{key} must be a GCS path, not {path}")
    try:
        bucket_region = get_bucket_location(path)
        if region.lower() != bucket_region.lower():
            raise ValueError(
                f"{key} is not in the same region ({bucket_region}) as the VM ({region}). "
                f"This can cause performance issues and billing surprises."
            )
    except GcpForbiddenException:
        logger.warning(f"Could not check region for {key}. Be sure it's in the same region as the VM.", exc_info=True)


if __name__ == "__main__":
    draccus.wrap()(run_levanter_train_lm)()
