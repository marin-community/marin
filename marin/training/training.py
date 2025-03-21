import dataclasses
import logging
import os
from copy import deepcopy
from dataclasses import dataclass, replace

import draccus
import levanter.infra.cli_helpers
import ray
from google.api_core.exceptions import Forbidden as GcpForbiddenException
from levanter.data.text import LMMixtureDatasetConfig
from levanter.infra.ray_tpu import run_on_pod_multislice_resumable, run_on_pod_resumable
from levanter.main import sft, sft_mixture, train_lm
from mergedeep import mergedeep
from ray.runtime_env import RuntimeEnv

from marin.utilities.dataclass_utils import shallow_asdict
from marin.utilities.gcs_utils import get_bucket_location, get_vm_region

logger = logging.getLogger(__name__)

# TODO: create helpers like the old launch.py to create reasonable train configs automatically
# Examples:
# - create a train-from-scratch config given a model config/path, data config/path, and tokenizer
# - create a llama3-style data ablating config given a model config/path, data config/path, and tokenizer


@dataclass
class TrainSFTOnPodConfig(sft.SFTConfig):
    output_path: str | None = None
    tpu_type: str | None = None
    env: dict = dataclasses.field(default_factory=dict)
    bypass_path_checks: bool = False
    impute_run_id_from_output_path: bool = True


@dataclass
class TrainSFTMixturePodConfig(sft_mixture.SFTMixtureConfig):
    output_path: str | None = None
    tpu_type: str | None = None  # None means local
    env: dict = dataclasses.field(default_factory=dict)
    bypass_path_checks: bool = False
    impute_run_id_from_output_path: bool = True


@ray.remote(num_cpus=0.1)
def run_levanter_sft(config: TrainSFTOnPodConfig):
    """
    Run the Levanter SFT training function on a Ray cluster.

    Similar to run_levanter_train_lm but for SFT training.
    """
    default_launch_config = levanter.infra.cli_helpers.load_config()

    if config.output_path is not None:
        logger.info(f"Using output path: {config.output_path}")
        config = _update_config_to_use_out_path(config)

    default_env = default_launch_config.env_for_accel(config.tpu_type or "")
    env = _add_default_env_variables(config.env, default_env)
    _check_for_wandb_key(env)
    env = _add_run_env_variables(env)
    config = replace(config, env=env)

    config = _suppress_ray_config(config)
    config = _enforce_run_id(config)
    logger.info(f"Using run ID: {config.trainer.id}")

    if not config.bypass_path_checks and config.tpu_type is not None:
        ray.get(ray.remote(_doublecheck_paths_sft).options(num_cpus=0.1).remote(config, must_save_checkpoints=True))

    sft_config = _upcast_sft_config(config)

    @ray.remote
    def sft_task():
        sft.train(sft_config)

    if config.tpu_type is not None:
        return run_on_pod_resumable(sft_task, config.tpu_type, max_retries_failure=10)
    else:
        return ray.get(sft_task.remote())


def _upcast_sft_config(config):
    """
    upcast TrainSFTOnPodConfig to SFTConfig by stripping the TPU type and env
    """
    dict_config = shallow_asdict(config)
    fields_to_remote = set(dict_config.keys()) - set(sft.SFTConfig.__dataclass_fields__.keys())
    for field in fields_to_remote:
        del dict_config[field]
    sft_config = sft.SFTConfig(**dict_config)
    return sft_config


@ray.remote(num_cpus=0.1)
def run_levanter_sft_mixture(config: TrainSFTMixturePodConfig):
    """
    Run the Levanter SFT mixture training function on a Ray cluster.
    Similar to run_levanter_sft but for mixture training.
    """
    default_launch_config = levanter.infra.cli_helpers.load_config()

    if config.output_path is not None:
        logger.info(f"Using output path: {config.output_path}")
        config = _update_config_to_use_out_path(config)

    default_env = default_launch_config.env_for_accel(config.tpu_type or "")
    env = _add_default_env_variables(config.env, default_env)
    _check_for_wandb_key(env)
    env = _add_run_env_variables(env)
    config = replace(config, env=env)

    config = _suppress_ray_config(config)
    config = _enforce_run_id(config)
    logger.info(f"Using run ID: {config.trainer.id}")

    if not config.bypass_path_checks and config.tpu_type is not None:
        ray.get(ray.remote(_doublecheck_paths_sft).options(num_cpus=0.1).remote(config, must_save_checkpoints=True))

    sft_mixture_config = _upcast_sft_mixture_config(config)

    @ray.remote
    def sft_mixture_task():
        sft_mixture.train(sft_mixture_config)

    if config.tpu_type is not None:
        return run_on_pod_resumable(sft_mixture_task, config.tpu_type, max_retries_failure=10)
    else:
        return ray.get(sft_mixture_task.remote())


def _upcast_sft_mixture_config(config):
    """Upcast TrainSFTMixturePodConfig to SFTMixtureConfig by stripping TPU type and env"""
    dict_config = shallow_asdict(config)
    fields_to_remove = set(dict_config.keys()) - set(sft_mixture.SFTMixtureConfig.__dataclass_fields__.keys())
    for field in fields_to_remove:
        del dict_config[field]
    return sft_mixture.SFTMixtureConfig(**dict_config)


@dataclass
class TrainLmOnPodConfig(train_lm.TrainLmConfig):
    # Inheritance so we can easily use existing TrainLmConfig configs.
    output_path: str | None = None
    """
    Base output directory to be used for training, mainly for use with executor framework.

    If set, this will override all "output" directories:
    * checkpoints (in $output_path/checkpoints
    * hf checkpoints (in $output_path/hf)
    * logging (in $output_path/log
    """

    tpu_type: str | None = None  # None means local

    env: dict = dataclasses.field(default_factory=dict)
    """Environment variables to set in the training pod."""
    allow_out_of_region_reads: bool = False
    """If True, allow reading from GCS buckets in different regions."""
    allow_out_of_region_writes: bool = False
    """If True, allow writing to GCS buckets in different regions."""
    impute_run_id_from_output_path: bool = True
    """
    If true and out_path is not None, the run id will be set to the basename of the out_path plus a random string.

    Note that trainer.id and the RUN_ID env variable take precedence, in that order.
    """
    node_count: int = 1
    """Number of TPU slices for training."""

    initialize_from_checkpoint_path: str | None = None
    """If set, the training will resume from the checkpoint at this path."""


DEFAULT_CHECKPOINTS_PATH = "checkpoints"
DEFAULT_HF_CHECKPOINTS_PATH = "hf"


def _update_config_to_use_out_path(config: TrainLmOnPodConfig):
    """
    Update the config to use the out_path as the base output directory for training.

    This will set the following paths to be subdirectories of the out_path:
    * checkpoints (in $out_path/checkpoints)
    * hf checkpoints (in $out_path/hf)
    * logging (in $out_path/log)

    This is useful when running with the executor framework, where the output path is set by the executor.
    """
    if config.output_path is None:
        return config

    trainer = replace(
        config.trainer,
        checkpointer=replace(
            config.trainer.checkpointer,
            base_path=os.path.join(config.output_path, DEFAULT_CHECKPOINTS_PATH),
        ),
    )

    return replace(config, trainer=trainer, hf_save_path=os.path.join(config.output_path, DEFAULT_HF_CHECKPOINTS_PATH))


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
    - if allow_out_of_region_reads is False, it checks that the data cache paths are in the same region as the VM.
    - if allow_out_of_region_writes is False, it checks that the checkpoint paths are in the same region as the VM.
    """
    default_launch_config = levanter.infra.cli_helpers.load_config()

    if config.output_path is not None:
        logger.info(f"Using output path: {config.output_path}")
        config = _update_config_to_use_out_path(config)

    env = _add_default_env_variables(config.env, default_launch_config.env_for_accel(config.tpu_type or ""))
    _check_for_wandb_key(env)
    env = _add_run_env_variables(env)
    config = replace(config, env=env)

    config = _suppress_ray_config(config)
    config = _enforce_run_id(config)
    logger.info(f"Using run ID: {config.trainer.id}")

    runtime_env = RuntimeEnv(env_vars=config.env)

    if not config.allow_out_of_region_reads and not config.allow_out_of_region_writes and config.tpu_type is not None:
        # run this on the Ray cluster to get the right region
        # doesn't need to be a TPU because ray insists that all VMs are in the same region
        ray.get(
            ray.remote(_doublecheck_paths)
            .options(runtime_env=runtime_env, num_cpus=0.1)
            .remote(config, must_save_checkpoints=True)
        )

    train_config = _upcast_trainlm_config(config)

    @ray.remote(runtime_env=runtime_env)
    def train_lm_task():
        train_lm.main(train_config)

    if config.tpu_type is not None:
        if config.node_count == 1:
            return run_on_pod_resumable(train_lm_task, config.tpu_type, max_retries_failure=10)
        else:
            return run_on_pod_multislice_resumable(
                train_lm_task, config.tpu_type, config.node_count, max_retries_failure=10
            )
    else:
        return ray.get(train_lm_task.remote())


def _upcast_trainlm_config(config):
    """
    upcast TrainLmOnPodConfig to TrainLmConfig by stripping the TPU type and env
    """
    dict_config = shallow_asdict(config)
    fields_to_remote = set(dict_config.keys()) - set(train_lm.TrainLmConfig.__dataclass_fields__.keys())
    for field in fields_to_remote:
        del dict_config[field]
    train_config = train_lm.TrainLmConfig(**dict_config)

    return train_config


def _enforce_run_id(config: TrainLmOnPodConfig):
    """
    Levanter will auto-generate a run ID if it's not set. We want to enforce that it's set, so that it resumes
    properly after preemption.

    Look for:
        * config.trainer.id
        * config.env.RUN_ID
        * environment variable RUN_ID
        * default to a random UID
    """
    run_id = config.trainer.id

    if run_id is None:
        run_id = config.env.get("RUN_ID", os.environ.get("RUN_ID"))

    if run_id is None and config.impute_run_id_from_output_path and config.output_path is not None:
        path = config.output_path
        path = path.rstrip("/")
        run_id = os.path.basename(path)
        logger.info(f"Imputing run ID from out path: {run_id}")

    if not run_id:
        run_id = levanter.infra.cli_helpers.default_run_id()
        logger.warning(f"Run ID not set. Using default: {run_id}")

    append_id_to_checkpoints = not config.impute_run_id_from_output_path
    checkpointer_config = replace(config.trainer.checkpointer, append_run_id_to_base_path=append_id_to_checkpoints)

    return replace(config, trainer=replace(config.trainer, id=run_id, checkpointer=checkpointer_config))


def _suppress_ray_config(config: TrainLmOnPodConfig):
    """
    Levanter wants to auto-start the Ray cluster, but we're already in a Ray cluster. Disable that.
    """
    # my other kingdom for lenses
    if config.trainer.ray.auto_start_cluster:
        logger.info("Ray cluster is set to auto-start, but that's not what we want for Marin. Disabling.")
        config = replace(
            config,
            trainer=replace(
                config.trainer,
                ray=replace(config.trainer.ray, auto_start_cluster=False, start_workers=False),
            ),
        )
    elif config.trainer.ray.start_workers:
        logger.info("Ray cluster is set to start workers, but that's not what we want for Marin. Disabling.")
        config = replace(
            config,
            trainer=replace(config.trainer, ray=replace(config.trainer.ray, start_workers=False)),
        )
    return config


def _check_path_in_region(key, path, none_ok, region, local_ok):
    if path is None:
        if none_ok:
            return
        raise ValueError(f"{key} must be set")

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


def _doublecheck_paths_sft(config: TrainSFTOnPodConfig, must_save_checkpoints):
    """
    Double-check paths specifically for SFT training configs.
    Handles chat_jsonl dataset configurations.
    """
    local_ok = config.bypass_path_checks or config.tpu_type is None
    try:
        region = get_vm_region()
    except ValueError as e:
        if local_ok:
            logger.warning("Could not determine the region of the VM. This is fine if you're running locally.")
            return
        raise ValueError("Could not determine the region of the VM. This is required for path checks.") from e

    # Check chat_train_urls
    if config.chat_train_urls:
        for url in config.chat_train_urls:
            _check_path_in_region("chat_train_urls", url, none_ok=False, region=region, local_ok=local_ok)

    # Check supervised data cache directory
    if config.supervised_data:
        _check_path_in_region(
            "supervised_data.cache_dir", config.supervised_data.cache_dir, none_ok=True, region=region, local_ok=local_ok
        )

    # Common checks for checkpointing
    _check_path_in_region(
        "trainer.checkpointer.base_path",
        config.trainer.checkpointer.base_path,
        none_ok=not must_save_checkpoints,
        region=region,
        local_ok=local_ok,
    )

    if config.hf_save_path is not None:
        _check_path_in_region(
            "hf_save_path", config.hf_save_path, none_ok=not must_save_checkpoints, region=region, local_ok=local_ok
        )
    else:
        logger.warning("hf_save_path is not set. This is fine if you don't want HF checkpoints.")

    return config


def _doublecheck_paths(config: TrainLmOnPodConfig, must_save_checkpoints):
    """
    Double-check that we're not using local paths in some of the standard places that Levanter sets defaults.
    Also check that the paths are in the same region as the VM, to avoid performance issues and billing surprises.
    """
    local_ok = (config.allow_out_of_region_reads and config.allow_out_of_region_writes) or config.tpu_type is None
    try:
        region = get_vm_region()
    except ValueError as e:
        if local_ok:
            logger.warning("Could not determine the region of the VM. This is fine if you're running locally.")
            return
        raise ValueError("Could not determine the region of the VM. This is required for path checks.") from e

    _check_path_in_region("data.cache_dir", config.data.cache_dir, none_ok=True, region=region, local_ok=local_ok)
    # now check all subcaches if applicable
    if isinstance(config.data, LMMixtureDatasetConfig):
        if not config.allow_out_of_region_reads:
            for key, subcache in config.data.configs.items():
                _check_path_in_region(
                    f"data.configs[{key}].cache_dir",
                    subcache.cache_dir,
                    none_ok=True,
                    region=region,
                    local_ok=local_ok,
                )
    if not config.allow_out_of_region_writes:
        _check_path_in_region(
            "trainer.checkpointer.base_path",
            config.trainer.checkpointer.base_path,
            none_ok=not must_save_checkpoints,
            region=region,
            local_ok=local_ok,
        )

        if config.hf_save_path is not None:
            _check_path_in_region(
                "hf_save_path",
                config.hf_save_path,
                none_ok=not must_save_checkpoints,
                region=region,
                local_ok=local_ok and config.allow_out_of_region_writes,
            )
        else:
            logger.warning("hf_save_path is not set. This is fine if you don't want HF checkpoints.")

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
    """

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
