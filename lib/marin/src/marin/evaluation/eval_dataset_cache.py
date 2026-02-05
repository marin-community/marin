# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utilities for caching evaluation datasets to GCS to avoid HuggingFace rate limiting.

When running multiple concurrent training jobs that all evaluate using the same
evaluation tasks, each job would hit HuggingFace's API to download evaluation datasets.
This can trigger rate limiting (429 errors), especially for less popular datasets.

This module provides functions to:
1. Extract dataset information from lm-eval task configurations
2. Pre-download datasets to GCS (done once before training starts)
3. Sync datasets from GCS to local cache (done on each worker before evaluation)

Usage as ExecutorStep:
    from marin.evaluation.eval_dataset_cache import create_cache_eval_datasets_step
    from experiments.evals.task_configs import CODE_TASKS

    cache_step = create_cache_eval_datasets_step(
        eval_tasks=CODE_TASKS,
        gcs_path="gs://marin-us-central1/raw/eval-datasets/code-tasks",
        name_prefix="my_experiment",
    )
"""

import json
import logging
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import fsspec

from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import ExecutorStep
from marin.utils import call_with_hf_backoff, fsspec_exists

logger = logging.getLogger(__name__)


def extract_datasets_from_tasks(
    eval_tasks: Sequence[EvalTaskConfig],
    *,
    log: logging.Logger | None = None,
) -> set[tuple[str, str | None]]:
    """
    Extract HuggingFace dataset (path, name) pairs from eval task configurations.

    This uses lm-eval's TaskManager to look up the YAML configuration for each task
    and extract the dataset_path and dataset_name fields.

    Args:
        eval_tasks: List of EvalTaskConfig objects specifying the evaluation tasks.
        log: Optional logger instance.

    Returns:
        A set of (dataset_path, dataset_name) tuples that can be passed to
        datasets.load_dataset().
    """
    import lm_eval.tasks as tasks

    log_obj = log or logger

    task_manager = tasks.TaskManager()
    datasets_needed: set[tuple[str, str | None]] = set()

    for task in eval_tasks:
        try:
            # Get the task configuration from lm-eval's YAML files.
            # NOTE: _get_config is a private API. There's no public alternative for
            # extracting dataset info from task configs. This may break if lm-eval
            # changes its internals.
            config = task_manager._get_config(task.name)

            dataset_path = config.get("dataset_path")
            dataset_name = config.get("dataset_name")

            if dataset_path:
                datasets_needed.add((dataset_path, dataset_name))
                log_obj.debug(f"Task '{task.name}' uses dataset: {dataset_path} (config: {dataset_name})")
            else:
                log_obj.debug(f"Task '{task.name}' has no dataset_path configured")

        except Exception as e:
            log_obj.warning(f"Could not extract dataset info for task '{task.name}': {e}")

    log_obj.info(f"Extracted {len(datasets_needed)} unique datasets from {len(eval_tasks)} tasks")
    return datasets_needed


@dataclass
class CacheManifest:
    """Manifest tracking which datasets were successfully cached."""

    task_names: list[str]
    """List of task names that were requested to be cached."""

    cached_datasets: list[tuple[str, str | None]]
    """List of (dataset_path, dataset_name) tuples that were successfully cached."""

    failed_datasets: list[tuple[str, str | None, str]]
    """List of (dataset_path, dataset_name, error_message) tuples that failed to cache."""

    def to_dict(self) -> dict:
        return {
            "task_names": self.task_names,
            "cached_datasets": [[d[0], d[1]] for d in self.cached_datasets],
            "failed_datasets": [[d[0], d[1], d[2]] for d in self.failed_datasets],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CacheManifest":
        return cls(
            task_names=data["task_names"],
            cached_datasets=[(d[0], d[1]) for d in data["cached_datasets"]],
            failed_datasets=[(d[0], d[1], d[2]) for d in data["failed_datasets"]],
        )

    def is_complete(self) -> bool:
        """Return True if all requested datasets were successfully cached."""
        return len(self.failed_datasets) == 0


def save_eval_datasets_to_gcs(
    eval_tasks: Sequence[EvalTaskConfig],
    gcs_path: str,
    *,
    local_cache_dir: str | None = None,
    dataset_kwargs: dict[str, Any] | None = None,
    log: logging.Logger | None = None,
) -> str:
    """
    Download evaluation datasets and upload them to GCS.

    This function downloads all datasets needed by the specified eval tasks
    to a local cache directory, then uploads that cache to GCS. This should
    be called once before training starts to pre-populate the GCS cache.

    Args:
        eval_tasks: List of EvalTaskConfig objects specifying the evaluation tasks.
        gcs_path: GCS path to save the datasets (e.g., "gs://bucket/.eval-datasets/abc123").
        local_cache_dir: Optional local cache directory. If None, a temporary directory is used.
        dataset_kwargs: Optional additional kwargs to pass to datasets.load_dataset().
        log: Optional logger instance.

    Returns:
        The GCS path where datasets were saved.

    Raises:
        RuntimeError: If any datasets failed to download.
    """
    import datasets

    log_obj = log or logger
    kwargs = dataset_kwargs or {}

    # Check if already cached in GCS with a complete manifest
    manifest_path = os.path.join(gcs_path, ".eval_datasets_manifest.json")
    if fsspec_exists(manifest_path):
        try:
            with fsspec.open(manifest_path, "r") as f:
                manifest = CacheManifest.from_dict(json.load(f))
            if manifest.is_complete():
                log_obj.info(f"Eval datasets already cached at {gcs_path} (complete)")
                return gcs_path
            else:
                log_obj.info(
                    f"Found incomplete cache at {gcs_path} "
                    f"({len(manifest.failed_datasets)} failed). Re-downloading..."
                )
        except Exception as e:
            log_obj.warning(f"Could not read manifest at {manifest_path}: {e}. Re-downloading...")

    # Extract datasets needed
    datasets_needed = extract_datasets_from_tasks(eval_tasks, log=log_obj)
    if not datasets_needed:
        log_obj.warning("No datasets found in eval tasks, nothing to cache")
        return gcs_path

    # Create or use local cache directory
    temp_dir_obj = None
    if local_cache_dir is None:
        temp_dir_obj = tempfile.TemporaryDirectory()
        cache_dir = temp_dir_obj.name
    else:
        cache_dir = local_cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    try:
        # Download all datasets to local cache
        log_obj.info(f"Downloading {len(datasets_needed)} datasets to {cache_dir}")
        cached_datasets: list[tuple[str, str | None]] = []
        failed_datasets: list[tuple[str, str | None, str]] = []

        for dataset_path, dataset_name in datasets_needed:
            log_obj.info(f"Downloading dataset: {dataset_path} (config: {dataset_name})")
            try:
                call_with_hf_backoff(
                    lambda dp=dataset_path, dn=dataset_name: datasets.load_dataset(
                        path=dp,
                        name=dn,
                        cache_dir=cache_dir,
                        trust_remote_code=True,
                        **kwargs,
                    ),
                    context=f"download dataset {dataset_path}",
                    logger=log_obj,
                )
                cached_datasets.append((dataset_path, dataset_name))
            except Exception as e:
                error_msg = str(e)
                log_obj.warning(f"Failed to download dataset {dataset_path}: {error_msg}")
                failed_datasets.append((dataset_path, dataset_name, error_msg))

        # Create manifest
        manifest = CacheManifest(
            task_names=sorted(task.name for task in eval_tasks),
            cached_datasets=cached_datasets,
            failed_datasets=failed_datasets,
        )

        # Upload cache directory to GCS
        # trailing slash is needed to upload the contents of the folder to gcs_path
        log_obj.info(f"Uploading datasets to {gcs_path}")
        fs = fsspec.core.url_to_fs(gcs_path)[0]
        fs.put(cache_dir + "/", gcs_path, recursive=True)

        # Write manifest file
        with fsspec.open(manifest_path, "w") as f:
            json.dump(manifest.to_dict(), f, indent=2)

        if failed_datasets:
            failed_names = [f"{d[0]}:{d[1]}" for d in failed_datasets]
            raise RuntimeError(
                f"Failed to download {len(failed_datasets)} datasets: {failed_names}. "
                f"Successfully cached {len(cached_datasets)} datasets to {gcs_path}."
            )

        log_obj.info(f"Successfully cached {len(cached_datasets)} eval datasets to {gcs_path}")
        return gcs_path

    finally:
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()


def load_eval_datasets_from_gcs(
    gcs_path: str,
    local_cache_dir: str | None = None,
    *,
    log: logging.Logger | None = None,
) -> bool:
    """
    Sync evaluation datasets from GCS to local HuggingFace cache directory.

    This function downloads the pre-cached datasets from GCS to the local
    HuggingFace datasets cache directory. After this, lm-eval will find the
    datasets in the local cache and won't need to hit the HuggingFace API.

    Args:
        gcs_path: GCS path where datasets are cached.
        local_cache_dir: Local cache directory to sync to. If None, uses the
            default HuggingFace datasets cache directory.
        log: Optional logger instance.

    Returns:
        True if sync was successful, False otherwise.
    """
    log_obj = log or logger

    # Check if GCS cache exists
    manifest_path = os.path.join(gcs_path, ".eval_datasets_manifest.json")
    if not fsspec_exists(manifest_path):
        log_obj.info(f"No eval datasets cache found at {gcs_path}")
        return False

    # Read manifest to check completeness
    try:
        with fsspec.open(manifest_path, "r") as f:
            manifest = CacheManifest.from_dict(json.load(f))
        if not manifest.is_complete():
            log_obj.warning(
                f"Eval datasets cache at {gcs_path} is incomplete "
                f"({len(manifest.failed_datasets)} failed). Proceeding with available datasets."
            )
    except Exception as e:
        log_obj.warning(f"Could not read manifest: {e}. Proceeding anyway.")

    # Determine local cache directory
    if local_cache_dir is None:
        local_cache_dir = os.environ.get("HF_DATASETS_CACHE")
        if local_cache_dir is None:
            local_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "datasets")

    # Ensure local cache directory exists
    os.makedirs(local_cache_dir, exist_ok=True)

    # Sync from GCS to local
    # The trailing slash is needed to download the contents of the folder to local_cache_dir
    # rather than creating a subdirectory (see marin/evaluation/utils.py)
    log_obj.info(f"Syncing eval datasets from {gcs_path} to {local_cache_dir}")
    try:
        fs = fsspec.core.url_to_fs(gcs_path)[0]
        fs.get(gcs_path + "/", local_cache_dir, recursive=True)
        log_obj.info(f"Successfully synced eval datasets to {local_cache_dir}")
        return True
    except Exception as e:
        log_obj.warning(f"Failed to sync eval datasets from GCS: {e}")
        return False


# ============================================================================
# EXECUTOR STEP
# ============================================================================


@dataclass(frozen=True)
class CacheEvalDatasetsConfig:
    """Configuration for the eval dataset caching executor step."""

    eval_tasks: tuple[EvalTaskConfig, ...]
    gcs_path: str


def _cache_eval_datasets(config: CacheEvalDatasetsConfig) -> str:
    """ExecutorStep function to cache eval datasets to GCS.

    This is idempotent - if datasets are already cached, it returns immediately.
    """
    # HF_ALLOW_CODE_EVAL is required for code evaluation tasks like HumanEval
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    return save_eval_datasets_to_gcs(
        eval_tasks=config.eval_tasks,
        gcs_path=config.gcs_path,
        log=logger,
    )


def create_cache_eval_datasets_step(
    eval_tasks: Sequence[EvalTaskConfig],
    gcs_path: str,
    name_prefix: str,
) -> ExecutorStep:
    """Create an ExecutorStep to pre-cache eval datasets to GCS.

    This step should run before training steps to ensure eval datasets are
    available in GCS. Workers will then sync from GCS instead of hitting
    the HuggingFace API, avoiding rate limiting.

    Args:
        eval_tasks: List of EvalTaskConfig objects specifying the evaluation tasks.
        gcs_path: GCS path to cache datasets (e.g., "gs://bucket/raw/eval-datasets/code-tasks").
        name_prefix: Experiment name prefix for the step name.

    Returns:
        ExecutorStep that caches eval datasets to GCS.

    Example:
        cache_step = create_cache_eval_datasets_step(
            eval_tasks=CORE_TASKS + CODE_TASKS,
            gcs_path="gs://marin-us-central1/raw/eval-datasets/code-tasks",
            name_prefix="my_experiment",
        )
        training_steps = [...]
        all_steps = [cache_step, *training_steps]
    """
    return ExecutorStep(
        name=f"{name_prefix}/cache_eval_datasets",
        description="Pre-cache evaluation datasets to GCS to avoid HuggingFace rate limiting",
        fn=_cache_eval_datasets,
        config=CacheEvalDatasetsConfig(
            eval_tasks=tuple(eval_tasks),
            gcs_path=gcs_path,
        ),
    )
