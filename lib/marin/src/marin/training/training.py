# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import importlib
import logging
import math
import os
import urllib.parse
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import TypeVar, cast

import draccus
from fray import (
    CpuConfig,
    Entrypoint,
    GpuConfig,
    JobRequest,
    ResourceConfig,
    TpuConfig,
    create_environment,
    current_client,
)
from mergedeep import mergedeep

from rigging.filesystem import check_gcs_paths_same_region, marin_temp_bucket
from marin.training.run_environment import add_run_env_variables

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainLmOnPodConfig:
    """Configuration for language model training on a pod."""

    train_config: object
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

    train_config: object
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
    auto_num_epochs: float | None = None
    """When set, resolve num_train_steps from the concrete DPO train cache at launch time."""
    auto_validation_runs: int | None = None
    """When set, schedule this many validation passes including the initial and final evaluations."""


TrainConfigT = TypeVar("TrainConfigT")
TrainOnPodConfigT = TypeVar("TrainOnPodConfigT", TrainLmOnPodConfig, TrainDpoOnPodConfig)

DEFAULT_CHECKPOINTS_PATH = "checkpoints"
DEFAULT_HF_CHECKPOINTS_PATH = "hf"
TEMPORARY_CHECKPOINT_TTL_DAYS = 14
TEMPORARY_CHECKPOINTS_PATH = "checkpoints-temp"


def _cli_helpers_module():
    return importlib.import_module("levanter.infra.cli_helpers")


def _output_path_temp_component(output_path: str) -> str:
    parsed = urllib.parse.urlparse(output_path)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.netloc}{parsed.path}".strip("/")
    if parsed.scheme:
        return f"{parsed.scheme}{parsed.path}".strip("/")
    return output_path.strip("/")


def temporary_checkpoint_base_path(output_path: str) -> str:
    """Return the region-local temporary checkpoint base for an executor output path."""
    output_component = _output_path_temp_component(output_path)
    temp_prefix = os.path.join(TEMPORARY_CHECKPOINTS_PATH, output_component, DEFAULT_CHECKPOINTS_PATH)
    return marin_temp_bucket(
        ttl_days=TEMPORARY_CHECKPOINT_TTL_DAYS,
        prefix=temp_prefix,
        source_prefix=output_path,
    )


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
            temporary_base_path=temporary_checkpoint_base_path(pod_config.output_path),
        ),
    )
    hf_output_path = os.path.join(pod_config.output_path, DEFAULT_HF_CHECKPOINTS_PATH)

    from levanter.adaptation import NoAdaptationConfig
    from levanter.main.train_dpo import TrainDpoConfig

    if isinstance(pod_config.train_config, TrainDpoConfig) and not isinstance(
        pod_config.train_config.adapter, NoAdaptationConfig
    ):
        config = replace(
            pod_config.train_config,
            trainer=trainer,
            hf_save_path=None,
            merged_hf_save_path=hf_output_path,
        )
        return replace(pod_config, train_config=config)

    config = replace(
        pod_config.train_config,
        trainer=trainer,
        hf_save_path=hf_output_path,
    )
    return replace(pod_config, train_config=config)


def _num_validation_sequences(total_sequences: int, fraction: float) -> int:
    if total_sequences <= 1:
        return 0
    if fraction <= 0:
        return 0
    num_val = int(total_sequences * fraction)
    if num_val <= 0:
        num_val = 1
    if num_val >= total_sequences:
        num_val = total_sequences - 1
    return num_val


def _dpo_training_components(config: object) -> dict[str, object]:
    weights = config.train_weights
    if weights is None:
        return dict(config.components)
    if isinstance(weights, dict):
        return {name: comp for name, comp in config.components.items() if weights.get(name, 0) > 0}

    has_weight = set()
    for _, stage_weights in weights:
        for name, weight in stage_weights.items():
            if weight > 0:
                has_weight.add(name)
    return {name: comp for name, comp in config.components.items() if name in has_weight}


def _dpo_training_dataset_size(config: object) -> int:
    from marin.processing.tokenize import read_tokenized_cache_stats

    training_components = _dpo_training_components(config.data)
    if len(training_components) != 1:
        raise ValueError(
            "DPO auto step resolution only supports single-component configs. "
            f"Found {len(training_components)} training components: {list(training_components.keys())}"
        )

    name, component = next(iter(training_components.items()))
    cache_dir = getattr(component, "cache_dir", None)
    if not isinstance(cache_dir, str):
        raise ValueError(
            f"DPO auto step resolution requires a concrete cache_dir string for component {name}, got {cache_dir!r}."
        )

    try:
        stats = read_tokenized_cache_stats(cache_dir, "train")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{exc}. Run tokenization first, or set num_train_steps explicitly.") from exc

    total_examples = stats.total_elements

    if config.validation_split_fraction is not None:
        total_examples -= _num_validation_sequences(total_examples, config.validation_split_fraction)

    if total_examples <= 0:
        raise ValueError(f"DPO train set is empty after validation split for cache_dir={cache_dir}.")

    return total_examples


def _num_train_steps_for_examples(batch_size: object, total_examples: int) -> int:
    from levanter.schedule import BatchSchedule

    if total_examples <= 0:
        raise ValueError(f"total_examples must be positive, got {total_examples}")

    schedule = BatchSchedule(batch_size)
    return schedule.find_step_containing_offset(total_examples - 1) + 1


def _scheduled_dpo_eval_steps(num_train_steps: int, total_validation_runs: int) -> list[int]:
    if total_validation_runs < 2:
        raise ValueError(f"total_validation_runs must be at least 2, got {total_validation_runs}")
    if num_train_steps <= 0:
        raise ValueError(f"num_train_steps must be positive, got {num_train_steps}")

    interval = max(1, math.ceil(num_train_steps / (total_validation_runs - 1)))
    return [step for step in range(interval, num_train_steps - 1, interval)]


def _maybe_auto_resolve_dpo_schedule(config: TrainDpoOnPodConfig) -> TrainDpoOnPodConfig:
    if config.auto_num_epochs is None and config.auto_validation_runs is None:
        return config

    train_config = config.train_config
    trainer = train_config.trainer

    dataset_size = _dpo_training_dataset_size(train_config)
    logger.info("Resolved DPO train set size from tokenizer stats: %d examples", dataset_size)

    if config.auto_num_epochs is not None:
        target_examples = math.ceil(config.auto_num_epochs * dataset_size)
        num_train_steps = _num_train_steps_for_examples(trainer.train_batch_size, target_examples)
        logger.info(
            "Resolved DPO steps from %.3g epoch(s): %d target examples at batch schedule %s -> %d steps",
            config.auto_num_epochs,
            target_examples,
            trainer.train_batch_size,
            num_train_steps,
        )
        trainer = replace(trainer, num_train_steps=num_train_steps)
        train_config = replace(train_config, trainer=trainer)

    if config.auto_validation_runs is not None:
        eval_steps = _scheduled_dpo_eval_steps(train_config.trainer.num_train_steps, config.auto_validation_runs)
        logger.info(
            "Resolved DPO validation schedule: initial eval, interior steps %s, and final eval",
            eval_steps,
        )
        train_config = replace(
            train_config,
            run_initial_eval=True,
            scheduled_eval_steps=eval_steps,
        )

    return replace(
        config,
        train_config=train_config,
        auto_num_epochs=None,
        auto_validation_runs=None,
    )


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
        run_id = _cli_helpers_module().default_run_id()
        logger.warning(f"Run ID not set. Using default: {run_id}")

    append_id_to_checkpoints = not config.impute_run_id_from_output_path
    checkpointer_config = replace(
        config.train_config.trainer.checkpointer, append_run_id_to_base_path=append_id_to_checkpoints
    )

    inner_config = replace(
        config.train_config, trainer=replace(config.train_config.trainer, id=run_id, checkpointer=checkpointer_config)
    )
    return replace(config, train_config=inner_config)


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


def _prepare_training_run(
    config: TrainOnPodConfigT,
) -> tuple[TrainOnPodConfigT, object, dict[str, str], list[str]]:
    """Shared setup for LM and DPO training: env vars, run ID, config adjustments.

    Returns the updated pod config, the ready-to-use train config, the
    environment dict, and the Fray extras list.
    """
    default_launch_config = _cli_helpers_module().load_config()

    if config.output_path is not None:
        logger.info(f"Using output path: {config.output_path}")
        config = _update_config_to_use_out_path(config)

    if isinstance(config, TrainDpoOnPodConfig):
        config = cast(TrainOnPodConfigT, _maybe_auto_resolve_dpo_schedule(config))

    env = _add_default_env_variables(
        config.env_vars or {},
        default_launch_config.env_for_accel(config.resources.device.variant),
    )
    if isinstance(config.resources.device, TpuConfig):
        _check_for_wandb_key(env)

    env = add_run_env_variables(env)

    if "JAX_COMPILATION_CACHE_DIR" not in env:
        env["JAX_COMPILATION_CACHE_DIR"] = _normalize_jax_compilation_cache_dir(
            marin_temp_bucket(ttl_days=30, prefix="compilation-cache")
        )
        logger.info("JAX compilation cache: %s", env["JAX_COMPILATION_CACHE_DIR"])
    _disable_xla_autotune_subcache(env)

    config = _enforce_run_id(config)
    logger.info(f"Using run ID: {config.train_config.trainer.id}")

    train_config = config.train_config
    train_config = _maybe_override_auto_build_caches(train_config, config.auto_build_caches)

    # disable accelerator requirement when running without GPU/TPU resources
    if config.resources.device.kind == "cpu":
        trainer = replace(train_config.trainer, require_accelerator=False)
        train_config = replace(train_config, trainer=trainer)

    if not isinstance(config.resources.device, CpuConfig):
        _doublecheck_paths(config)

    extras: list[str] = []
    if isinstance(config.resources.device, TpuConfig):
        extras.append("tpu")
    elif isinstance(config.resources.device, GpuConfig):
        extras.append("gpu")

    return config, train_config, env, extras


def _submit_training_job(
    *,
    job_name: str,
    main_fn: Callable,
    train_config: TrainConfigT,
    resources: ResourceConfig,
    env: dict[str, str],
    extras: list[str],
) -> None:
    """Submit a Levanter training job to Fray and block until completion."""
    client = current_client()
    # Using a constant job name allows restarts to adopt the existing job handle
    # instead of raising a duplicate name error (adopt_existing=True is the default).
    job_request = JobRequest(
        name=job_name,
        entrypoint=Entrypoint.from_callable(main_fn, args=[train_config]),
        resources=resources,
        environment=create_environment(env_vars=env, extras=extras),
        max_retries_failure=0,
    )
    job = client.submit(job_request)
    job.wait(raise_on_failure=True)


def run_levanter_train_lm(config: TrainLmOnPodConfig):
    """Run the Levanter LM training main function by submitting a job to Fray.

    Expects the following env vars (in the process env or ``config.env_vars``):

    - WANDB_API_KEY: The API key for Weights and Biases.
    - RUN_ID: (Optional) The run ID for this training run. Will default to a random UID if not set.
    - GIT_COMMIT: (Optional) The git commit hash of the current codebase. Will attempt to fetch it if not set.

    This function makes a number of changes to the config and ensures a few things are set:
    - The run ID is set, or sets a default if not.
    - WANDB_API_KEY is set.
    - Accelerator-appropriate extras (``tpu``/``gpu``) are selected for the Fray environment.
    - It checks that configured GCS paths are in the same region as the VM (except train/validation source URLs).
    """
    config, train_config, env, extras = _prepare_training_run(config)

    model_config = train_config.model
    logger.info(
        "Model config: type=%s seq_len=%d hidden=%d batch=%s device=%s",
        type(model_config).__name__,
        model_config.max_seq_len,
        model_config.Embed.size,
        train_config.trainer.train_batch_size,
        config.resources.device,
    )

    _submit_training_job(
        job_name="train_lm",
        main_fn=importlib.import_module("levanter.main.train_lm").main,
        train_config=train_config,
        resources=config.resources,
        env=env,
        extras=extras,
    )


def run_levanter_train_dpo(config: TrainDpoOnPodConfig):
    """Run the Levanter DPO training main function through Fray.

    This function is designed to be run on your machine or with sufficient variables in the env dict/os env.
    """
    config, train_config, env, extras = _prepare_training_run(config)

    _submit_training_job(
        job_name="train_dpo",
        main_fn=importlib.import_module("levanter.main.train_dpo").main,
        train_config=train_config,
        resources=config.resources,
        env=env,
        extras=extras,
    )


def _doublecheck_paths(config: TrainOnPodConfigT):
    """
    Double-check that we're not using local paths in some of the standard places that Levanter sets defaults.
    Also check that the paths are in the same region as the VM, to avoid performance issues and billing surprises.

    This function recursively examines all strings/paths in the config to identify GCS paths and checks their regions.
    """
    local_ok = not isinstance(config.resources.device, TpuConfig)

    check_gcs_paths_same_region(
        config.train_config,
        local_ok=local_ok,
    )
    return config


def _add_default_env_variables(env: dict, default_env: dict | None):
    if default_env is not None:
        default_env = deepcopy(default_env)
        env = mergedeep.merge(default_env, env)

    # Task environment values are serialized as strings.
    env = {str(k): str(v) for k, v in env.items()}
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
