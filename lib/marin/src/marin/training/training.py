# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import importlib
import logging
import os
import urllib.parse
import re
from copy import deepcopy
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TypeVar

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

from rigging.filesystem import (
    REGION_TO_TMP_BUCKET,
    check_gcs_paths_same_region,
    check_path_in_region,
    collect_gcs_paths,
    get_bucket_location,
    marin_temp_bucket,
    region_from_prefix,
    split_gcs_path,
)
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


TrainConfigT = TypeVar("TrainConfigT")
TrainOnPodConfigT = TypeVar("TrainOnPodConfigT", TrainLmOnPodConfig, TrainDpoOnPodConfig)

DEFAULT_CHECKPOINTS_PATH = "checkpoints"
DEFAULT_HF_CHECKPOINTS_PATH = "hf"
TEMPORARY_CHECKPOINT_TTL_DAYS = 14
TEMPORARY_CHECKPOINTS_PATH = "checkpoints-temp"
_SOURCE_URL_FIELDS = ("train_urls", "validation_urls")
_NON_REGIONAL_BUCKET_LOCATIONS = {"us", "eu", "asia", "nam4", "eur4", "asia1"}
_GCP_REGION_PATTERN = re.compile(r"^[a-z]+-[a-z0-9]+[0-9]$")


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

    config = replace(
        pod_config.train_config,
        trainer=trainer,
        hf_save_path=os.path.join(pod_config.output_path, DEFAULT_HF_CHECKPOINTS_PATH),
    )
    return replace(pod_config, train_config=config)


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


def _skip_source_url_fields(include_source_urls: bool) -> tuple[str, ...]:
    return () if include_source_urls else _SOURCE_URL_FIELDS


def _normalize_region(region: str, *, path: str) -> str:
    normalized = region.lower()
    if normalized in _NON_REGIONAL_BUCKET_LOCATIONS or "+" in normalized or not _GCP_REGION_PATTERN.match(normalized):
        raise ValueError(
            f"Training config references {path!r} in a non-regional bucket location ({normalized!r}); "
            "cannot infer a single TPU/VM region."
        )
    return normalized


def _is_bucket_location_permission_error(exc: Exception) -> bool:
    return isinstance(exc, PermissionError) or exc.__class__.__name__ in {"Forbidden", "PermissionDenied"}


def _region_for_gcs_path(path: str, *, bucket_region_cache: dict[str, str]) -> str | None:
    bucket, _ = split_gcs_path(path)

    try:
        if bucket not in bucket_region_cache:
            bucket_region_cache[bucket] = get_bucket_location(bucket)
        return _normalize_region(bucket_region_cache[bucket], path=path)
    except Exception as e:
        if not _is_bucket_location_permission_error(e):
            raise

    region = region_from_prefix(path)
    if region is None:
        logger.warning(
            "Could not infer bucket location for %s due to permission error; skipping this path for training region "
            "inference.",
            path,
            exc_info=True,
        )
        return None

    try:
        return _normalize_region(region, path=path)
    except ValueError:
        logger.warning(
            "Could not infer a concrete region for %s because bucket metadata access failed and the bucket name does "
            "not encode a regional location.",
            path,
            exc_info=True,
        )
        return None


def _infer_single_gcs_region(train_config: object, *, include_source_urls: bool) -> str | None:
    """Infer the one concrete region required by resolved GCS paths in a training config."""
    bucket_region_cache: dict[str, str] = {}
    region_to_evidence: dict[str, list[str]] = {}
    skip_fields = _skip_source_url_fields(include_source_urls)

    for key, path in collect_gcs_paths(train_config, skip_if_prefix_contains=skip_fields):
        region = _region_for_gcs_path(path, bucket_region_cache=bucket_region_cache)
        if region is None:
            continue
        region_to_evidence.setdefault(region, []).append(f"{key}={path}")

    if not region_to_evidence:
        return None
    if len(region_to_evidence) > 1:
        detail = "; ".join(
            f"{region}: {', '.join(sorted(evidence)[:3])}" for region, evidence in sorted(region_to_evidence.items())
        )
        raise ValueError(
            "Training config references GCS paths in multiple regions. "
            f"Found regions {{{', '.join(sorted(region_to_evidence))}}}. {detail}"
        )
    return next(iter(region_to_evidence))


def _regional_temp_bucket(region: str, *, ttl_days: int, prefix: str) -> str:
    bucket = REGION_TO_TMP_BUCKET.get(region)
    if bucket is None:
        raise ValueError(f"No Marin temp bucket is configured for region {region!r}.")
    path = f"gs://{bucket}/ttl={ttl_days}d"
    if prefix:
        path = f"{path}/{prefix.strip('/')}"
    return path


def _align_resources_to_gcs_region(config: TrainOnPodConfigT) -> tuple[TrainOnPodConfigT, str | None]:
    """Pin child training resources to the region implied by resolved GCS paths."""
    inferred_region = _infer_single_gcs_region(
        config.train_config,
        include_source_urls=config.auto_build_caches,
    )
    if inferred_region is None:
        return config, None

    resource_regions = config.resources.regions
    if resource_regions:
        explicit_regions = {region.lower() for region in resource_regions}
        if inferred_region not in explicit_regions:
            raise ValueError(
                f"Training job resources allow regions {sorted(explicit_regions)}, but resolved GCS paths require "
                f"{inferred_region!r}. Running this TPU/VM job elsewhere would read or write across regions."
            )

    resources = dataclasses.replace(config.resources, regions=[inferred_region])
    return replace(config, resources=resources), inferred_region


def _check_jax_compilation_cache_region(env: dict[str, str], *, region: str | None, local_ok: bool) -> None:
    cache_dir = env.get("JAX_COMPILATION_CACHE_DIR")
    if cache_dir is None:
        return
    if not cache_dir.startswith("gs://"):
        return
    if region is not None:
        check_path_in_region("JAX_COMPILATION_CACHE_DIR", cache_dir, region, local_ok=local_ok)
        return
    check_gcs_paths_same_region({"JAX_COMPILATION_CACHE_DIR": cache_dir}, local_ok=local_ok)


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

    config = _enforce_run_id(config)
    logger.info(f"Using run ID: {config.train_config.trainer.id}")

    train_config = config.train_config
    train_config = _maybe_override_auto_build_caches(train_config, config.auto_build_caches)
    config = replace(config, train_config=train_config)

    training_region: str | None = None
    if not isinstance(config.resources.device, CpuConfig):
        config, training_region = _align_resources_to_gcs_region(config)
        train_config = config.train_config

    env = _add_default_env_variables(
        config.env_vars or {},
        default_launch_config.env_for_accel(config.resources.device.variant),
    )
    if isinstance(config.resources.device, TpuConfig):
        _check_for_wandb_key(env)

    env = add_run_env_variables(env)

    if "JAX_COMPILATION_CACHE_DIR" not in env:
        cache_dir = (
            _regional_temp_bucket(training_region, ttl_days=30, prefix="compilation-cache")
            if training_region is not None
            else marin_temp_bucket(ttl_days=30, prefix="compilation-cache")
        )
        env["JAX_COMPILATION_CACHE_DIR"] = _normalize_jax_compilation_cache_dir(cache_dir)
        logger.info("JAX compilation cache: %s", env["JAX_COMPILATION_CACHE_DIR"])
    _disable_xla_autotune_subcache(env)
    # disable accelerator requirement when running without GPU/TPU resources
    if config.resources.device.kind == "cpu":
        trainer = replace(train_config.trainer, require_accelerator=False)
        train_config = replace(train_config, trainer=trainer)

    if not isinstance(config.resources.device, CpuConfig):
        _check_jax_compilation_cache_region(env, region=training_region, local_ok=False)
        _doublecheck_paths(
            config,
            region=training_region,
            include_source_urls=config.auto_build_caches,
        )

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


def _doublecheck_paths(
    config: TrainOnPodConfigT,
    *,
    region: str | None = None,
    include_source_urls: bool = False,
) -> TrainOnPodConfigT:
    """
    Double-check that we're not using local paths in some of the standard places that Levanter sets defaults.
    Also check that the paths are in the same region as the VM, to avoid performance issues and billing surprises.

    This function recursively examines all strings/paths in the config to identify GCS paths and checks their regions.
    """
    local_ok = not isinstance(config.resources.device, TpuConfig)

    check_gcs_paths_same_region(
        config.train_config,
        local_ok=local_ok,
        region=region,
        skip_if_prefix_contains=_skip_source_url_fields(include_source_urls),
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
