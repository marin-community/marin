import dataclasses
import logging
import os
from copy import deepcopy
from dataclasses import dataclass

import draccus
import levanter.infra.cli_helpers
import ray
from levanter.infra.ray_tpu import run_on_pod_resumable
from levanter.main import train_lm
from mergedeep import mergedeep
from ray.runtime_env import RuntimeEnv

from marin.utilities.gcs_utils import get_vm_region, is_bucket_in_region

logger = logging.getLogger(__name__)

# TODO: create helpers like the old launch.py to create reasonable train configs automatically
# Examples:
# - create a train-from-scratch config given a model config/path, data config/path, and tokenizer
# - create a llama3-style data ablating config given a model config/path, data config/path, and tokenizer


@dataclass
class TrainLmOnPodConfig(train_lm.TrainLmConfig):
    """Inheritance so we can easily use existing TrainLmConfig configs."""

    tpu_type: str = "v4-64"  # have to specify defaults b/c dataclasses
    env: dict = dataclasses.field(default_factory=dict)
    """Environment variables to set in the training pod."""
    bypass_path_checks: bool = False
    """If True, don't check that paths are set and are in the same region as the VM."""


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
    - it disables the auto-ray-start and auto-worker-start options since we're already in a Ray cluster.
    - it checks that the paths are set and in the same region as the VM, unless bypass_path_checks is set.
    """
    default_launch_config = levanter.infra.cli_helpers.load_config()

    env = _add_run_env_variables(config.env, default_launch_config)
    config = dataclasses.replace(config, env=env)

    config = _suppress_ray_config(config)
    config = _enforce_run_id(config)
    logger.info(f"Using run ID: {config.trainer.id}")

    runtime_env = RuntimeEnv(env_vars=config.env)

    if not config.bypass_path_checks:
        # run this on the Ray cluster to get the right region
        # doesn't need to be a TPU because ray insists that all VMs are in the same region
        ray.get(ray.remote(_doublecheck_paths).options(runtime_env=runtime_env, num_cpus=0.1).remote(config))

    dict_config = dataclasses.asdict(config)
    del dict_config["env"]  # this is the important bit: don't want to leak env vars into the config
    del dict_config["tpu_type"]
    train_config = train_lm.TrainLmConfig(**dict_config)

    @ray.remote
    def train_lm_task():
        # upcast TrainLmOnPodConfig to TrainLmConfig by stripping the TPU type and env
        train_lm.main(train_config)

    return run_on_pod_resumable(train_lm_task.options(runtime_env=runtime_env), config.tpu_type)


def _enforce_run_id(config: TrainLmOnPodConfig):
    """
    Levanter will auto-generate a run ID if it's not set. We want to enforce that it's set for preemption.

    Look for:
        * config.trainer.id
        * config.env.RUN_ID
        * environment variable RUN_ID
        * default to a random UID
    """
    run_id = config.trainer.id

    if run_id is None:
        run_id = config.env.get("RUN_ID", os.environ.get("RUN_ID"))

    if run_id is None:
        run_id = levanter.infra.cli_helpers.default_run_id()
        logger.warning(f"Run ID not set. Using default: {run_id}")

    return dataclasses.replace(config, trainer=dataclasses.replace(config.trainer, id=run_id))


def _suppress_ray_config(config: TrainLmOnPodConfig):
    """
    Levanter wants to auto-start the Ray cluster, but we're already in a Ray cluster. Disable that.
    """
    # my other kingdom for lenses
    if config.trainer.ray.auto_start_cluster:
        logger.info("Ray cluster is set to auto-start, but that's not what we want for Marin. Disabling.")
        config = dataclasses.replace(
            config,
            trainer=dataclasses.replace(
                config.trainer,
                ray=dataclasses.replace(config.trainer.ray, auto_start_cluster=False, start_workers=False),
            ),
        )
    elif config.trainer.ray.start_workers:
        logger.info("Ray cluster is set to start workers, but that's not what we want for Marin. Disabling.")
        config = dataclasses.replace(
            config,
            trainer=dataclasses.replace(
                config.trainer, ray=dataclasses.replace(config.trainer.ray, start_workers=False)
            ),
        )
    return config


def _doublecheck_paths(config: TrainLmOnPodConfig):
    """
    Double-check that we're not using local paths in some of the standard places that Levanter sets defaults.
    Also check that the paths are in the same region as the VM, to avoid performance issues and billing surprises.
    """
    try:
        region = get_vm_region()
    except ValueError:
        region = "us-central2"
        logger.warning(f"Could not determine region. Defaulting to {region}")

    def check(key, path):
        if not path.startswith("gs://"):
            raise ValueError(f"{key} must be a GCS path, not {path}")
        if not is_bucket_in_region(path, region):
            raise ValueError(
                f"{key} is not in the same region as the VM. This can cause performance issues and billing surprises."
            )

    check("data.cache_dir", config.data.cache_dir)
    check("trainer.checkpointer.base_path", config.trainer.checkpointer.base_path)

    if config.hf_save_path is not None:
        check("hf_save_path", config.hf_save_path)
    else:
        logger.warning("hf_save_path is not set. This is fine if you don't want HF checkpoints.")

    return config


def _add_run_env_variables(env: dict, default_env: dict):
    env = deepcopy(env)
    if default_env is not None:
        mergedeep.merge(env, default_env)
    if env.get("WANDB_API_KEY") is None:
        key = os.environ.get("WANDB_API_KEY")
        if key is not None:
            env["WANDB_API_KEY"] = key
        else:
            raise ValueError(
                "WANDB_API_KEY must be set in the environment. Please add it to your .config, export "
                "WANDB_API_KEY=..., or add it to the env dict."
            )

    if "GIT_COMMIT" not in env:
        env["GIT_COMMIT"] = levanter.infra.cli_helpers.get_git_commit()

    return env


if __name__ == "__main__":
    draccus.wrap()(run_levanter_train_lm)()
