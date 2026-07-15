# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import importlib
import json
import logging
import math
import os
import urllib.parse
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any, TypeVar, cast

from draccus.utils import DataclassInstance
from fray.types import CpuConfig, ResourceConfig, TpuConfig
from levanter.adaptor import NoAdaptorConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.main.train_dpo import TrainDpoConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.schedule import BatchSchedule
from mergedeep import mergedeep
from pydantic import BaseModel
from rigging.filesystem import StoragePath, check_gcs_paths_same_region, marin_temp_bucket, prefix_join, url_to_fs

from marin.execution.artifact import Artifact
from marin.processing.tokenize import read_tokenized_cache_stats
from marin.training.run_environment import add_run_env_variables

logger = logging.getLogger(__name__)

# The subdirectory Levanter writes rolling checkpoints into, relative to a run's output dir.
_CHECKPOINTS_SUBDIR = "checkpoints"
# The final-metrics file a run mirrors next to its output via the WandB ``replicate_path``.
_TRACKER_METRICS_FILE = "tracker_metrics.jsonl"


class TrainMetrics(BaseModel):
    """A finished run's final metrics, read on demand from its output.

    ``summary`` is the full mirrored WandB summary (keys like ``train/loss``, ``eval/loss``,
    ``eval/<name>/loss``); ``train_loss``/``eval_loss`` are the common scalars pulled out, each
    ``None`` when the run did not log it.
    """

    summary: dict[str, Any]
    train_loss: float | None = None
    eval_loss: float | None = None


def _as_float(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) else None


class LevanterCheckpoint(Artifact):
    """A Levanter training run's output: rolling checkpoints, config, and mirrored metrics.

    The realized artifact for every :func:`~marin.experiment.train.train_lm` handle. ``load`` is
    a path ref; the checkpoint structure and metrics are exposed as accessors so callers never
    hard-code the layout.
    """

    @property
    def checkpoint_dir(self) -> str:
        """The directory holding this run's rolling checkpoints."""
        return prefix_join(self.path, _CHECKPOINTS_SUBDIR)

    def training_metrics(self) -> TrainMetrics:
        """This run's final metrics, parsed from ``tracker_metrics.jsonl`` under its output.

        Raises :class:`FileNotFoundError` if the run wrote no metrics file.
        """
        path = prefix_join(self.path, _TRACKER_METRICS_FILE)
        if not url_to_fs(path, use_listings_cache=False)[0].exists(path):
            raise FileNotFoundError(f"no {_TRACKER_METRICS_FILE} for checkpoint at {self.path}")
        lines = [line for line in StoragePath(path).read_text().splitlines() if line.strip()]
        if not lines:
            raise FileNotFoundError(f"empty {_TRACKER_METRICS_FILE} at {path}")
        summary = json.loads(lines[-1]).get("summary", {})
        return TrainMetrics(
            summary=summary,
            train_loss=_as_float(summary.get("train/loss")),
            eval_loss=_as_float(summary.get("eval/loss")),
        )


@dataclass(frozen=True)
class TrainLmOnPodConfig:
    """Configuration for language model training on a pod."""

    train_config: object
    resources: ResourceConfig
    output_path: str | None = None
    """Base output directory for the run. The checkpointer, HF export, and run id derive from it."""
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
    """Base output directory for the run. The checkpointer, HF export, and run id derive from it."""
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


def resolve_checkpointer_output_path(checkpointer: CheckpointerConfig, output_path: str) -> CheckpointerConfig:
    """Point ``checkpointer`` at ``output_path``: rolling checkpoints under ``<output_path>/checkpoints``
    and time-policy (temporary) checkpoints on region-local storage keyed off ``output_path``.

    ``append_run_id_to_base_path`` is ``False`` because ``output_path`` already encodes the run's
    identity, so a run id suffix would double it up. Every other checkpointer field is preserved.
    """
    return replace(
        checkpointer,
        base_path=prefix_join(output_path, DEFAULT_CHECKPOINTS_PATH),
        temporary_base_path=temporary_checkpoint_base_path(output_path),
        append_run_id_to_base_path=False,
    )


def apply_output_path(train_config: TrainConfigT, output_path: str) -> TrainConfigT:
    """Set every run-scoped path on ``train_config`` from ``output_path``.

    Points the checkpointer at ``output_path`` and sets ``hf_save_path`` to ``<output_path>/hf``.
    Adapter LM/DPO exports PEFT rather than a merged HF model, so for those the merged ``hf_save_path``
    is cleared and ``peft_save_path`` takes the HF location.
    """
    config = replace(  # type: ignore[bad-specialization]
        train_config,
        trainer=replace(
            train_config.trainer,
            checkpointer=resolve_checkpointer_output_path(train_config.trainer.checkpointer, output_path),
        ),
        hf_save_path=prefix_join(output_path, DEFAULT_HF_CHECKPOINTS_PATH),
    )

    if isinstance(config, (TrainDpoConfig, TrainLmConfig)) and not isinstance(config.adapter, NoAdaptorConfig):
        peft_save_path = config.peft_save_path
        if peft_save_path is None and config.hf_save_steps is not None:
            peft_save_path = config.hf_save_path
        config = replace(config, hf_save_path=None, peft_save_path=peft_save_path)

    return config


def _resolve_run_id(
    train_config: TrainConfigT, *, output_path: str | None, env_run_id: str | None
) -> tuple[TrainConfigT, str]:
    """Pick a stable run id and stamp it into ``train_config.trainer.id``.

    A stable id is required so a run resumes into the same W&B run and checkpoint directory after
    preemption. Priority: ``trainer.id`` already set by the caller, then ``env_run_id`` (from
    ``env_vars["RUN_ID"]``), then the ``RUN_ID`` environment variable, then ``basename(output_path)``,
    and finally a random UID as a last resort.
    """
    run_id = train_config.trainer.id or env_run_id or os.environ.get("RUN_ID")
    if run_id is None and output_path is not None:
        run_id = os.path.basename(output_path.rstrip("/"))
        logger.info("Imputing run ID from output path: %s", run_id)
    if not run_id:
        run_id = _cli_helpers_module().default_run_id()
        logger.warning("Run ID not set. Using default: %s", run_id)
    updated = replace(train_config, trainer=replace(train_config.trainer, id=run_id))  # type: ignore[bad-specialization]
    return updated, run_id


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

    if config.auto_num_epochs is not None:
        dataset_size = _dpo_training_dataset_size(train_config)
        logger.info("Resolved DPO train set size from tokenizer stats: %d examples", dataset_size)
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
        train_config = replace(cast(DataclassInstance, train_config), trainer=trainer)

    if config.auto_validation_runs is not None:
        eval_steps = _scheduled_dpo_eval_steps(train_config.trainer.num_train_steps, config.auto_validation_runs)
        logger.info(
            "Resolved DPO validation schedule: initial eval, interior steps %s, and final eval",
            eval_steps,
        )
        train_config = replace(
            cast(DataclassInstance, train_config),
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
        data = dataclasses.replace(cast(DataclassInstance, data), auto_build_caches=auto_build)
        config = cast(TrainConfigT, replace(cast(DataclassInstance, config), data=data))
    return config


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


def resolve_training_env(
    base_env: dict[str, str] | None,
    resources: ResourceConfig,
) -> dict[str, str]:
    """Build the training-side environment dict.

    Combines the base env from the user (typically ``train_config.env_vars``)
    with hardware-specific defaults from ``levanter.infra.cli_helpers``, run
    metadata (GIT_COMMIT, FERRY_DATE, etc. via ``add_run_env_variables``), a
    JAX compilation cache pointing at ``marin_temp_bucket``, and a guard
    against XLA's autotune subcache when the cache lives on remote storage.
    """
    default_launch_config = _cli_helpers_module().load_config()

    env = _add_default_env_variables(
        base_env or {},
        default_launch_config.env_for_accel(resources.device.variant),
    )
    if isinstance(resources.device, TpuConfig):
        _check_for_wandb_key(env)

    env = add_run_env_variables(env)

    if "JAX_COMPILATION_CACHE_DIR" not in env:
        env["JAX_COMPILATION_CACHE_DIR"] = _normalize_jax_compilation_cache_dir(
            marin_temp_bucket(ttl_days=30, prefix="compilation-cache")
        )
        logger.info("JAX compilation cache: %s", env["JAX_COMPILATION_CACHE_DIR"])
    _disable_xla_autotune_subcache(env)

    return env


def _prepare_training_run(
    config: TrainOnPodConfigT,
) -> tuple[TrainOnPodConfigT, object, dict[str, str]]:
    """Shared setup for LM and DPO training: env vars, run ID, config adjustments.

    Returns the updated pod config, the ready-to-use train config, and the
    environment dict that callers should merge into ``os.environ`` before
    invoking the Levanter main.
    """
    train_config = config.train_config
    if config.output_path is not None:
        logger.info(f"Using output path: {config.output_path}")
        train_config = apply_output_path(train_config, config.output_path)

    train_config, run_id = _resolve_run_id(
        train_config,
        output_path=config.output_path,
        env_run_id=(config.env_vars or {}).get("RUN_ID"),
    )
    logger.info(f"Using run ID: {run_id}")
    config = replace(config, train_config=train_config)

    if isinstance(config, TrainDpoOnPodConfig):
        config = cast(TrainOnPodConfigT, _maybe_auto_resolve_dpo_schedule(config))
        train_config = config.train_config

    env = resolve_training_env(config.env_vars, config.resources)

    train_config = _maybe_override_auto_build_caches(train_config, config.auto_build_caches)

    # disable accelerator requirement when running without GPU/TPU resources
    if config.resources.device.kind == "cpu":
        trainer = replace(train_config.trainer, require_accelerator=False)
        train_config = replace(cast(DataclassInstance, train_config), trainer=trainer)

    if not isinstance(config.resources.device, CpuConfig):
        doublecheck_paths(config)

    return config, train_config, env


def _apply_env_to_process(env: dict[str, str]) -> None:
    """Apply training env vars to ``os.environ`` so Levanter's main reads them.

    Uses ``setdefault`` so ambient env (set by Iris from the parent
    JobRequest) wins on conflict; only missing keys are filled in.
    """
    for key, value in env.items():
        os.environ.setdefault(key, value)


def run_levanter_train_lm(config: TrainLmOnPodConfig):
    """Run the Levanter LM training main function in the current process.

    Expects the following env vars (in the process env or ``config.env_vars``):

    - WANDB_API_KEY: The API key for Weights and Biases.
    - RUN_ID: (Optional) The run ID for this training run. Will default to a random UID if not set.
    - GIT_COMMIT: (Optional) The git commit hash of the current codebase. Will attempt to fetch it if not set.

    This function makes a number of changes to the config and ensures a few things are set:
    - The run ID is set, or sets a default if not.
    - WANDB_API_KEY is set.
    - It checks that configured GCS paths are in the same region as the VM (except train/validation source URLs).
    """
    config, train_config, env = _prepare_training_run(config)

    model_config = train_config.model
    logger.info(
        "Model config: type=%s seq_len=%d hidden=%d batch=%s device=%s",
        type(model_config).__name__,
        model_config.max_seq_len,
        model_config.Embed.size,
        train_config.trainer.train_batch_size,
        config.resources.device,
    )

    _apply_env_to_process(env)
    importlib.import_module("levanter.main.train_lm").main(train_config)


def run_levanter_train_dpo(config: TrainDpoOnPodConfig):
    """Run the Levanter DPO training main function in the current process."""
    config, train_config, env = _prepare_training_run(config)
    _apply_env_to_process(env)
    importlib.import_module("levanter.main.train_dpo").main(train_config)


def check_train_config_paths(train_config: object, resources: ResourceConfig) -> None:
    """Check that all GCS paths in ``train_config`` are in the same region as the VM.

    Skips the check if ``resources.device`` is a CPU (local paths are always OK
    on CPU workers, and there is no region to match against).

    Args:
        train_config: The inner Levanter train config (e.g. ``TrainLmConfig``).
        resources: The resource config used for the training job.
    """
    if isinstance(resources.device, CpuConfig):
        return
    local_ok = not isinstance(resources.device, TpuConfig)
    check_gcs_paths_same_region(train_config, local_ok=local_ok)


def doublecheck_paths(config: TrainOnPodConfigT) -> TrainOnPodConfigT:
    """Check GCS path regions for a full ``TrainOnPodConfig``.

    Delegates to ``check_train_config_paths`` after extracting the inner config
    and resource config. Returns the config unchanged (for easy chaining).
    """
    check_train_config_paths(config.train_config, config.resources)
    return config


def _add_default_env_variables(env: dict, default_env: dict | None) -> dict:
    merged: Mapping = env
    if default_env is not None:
        merged = mergedeep.merge(deepcopy(default_env), env)

    # Task environment values are serialized as strings.
    return {str(k): str(v) for k, v in merged.items()}


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
