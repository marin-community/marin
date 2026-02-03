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

import hashlib
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
            # Get the task configuration from lm-eval's YAML files
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


def compute_cache_key(eval_tasks: Sequence[EvalTaskConfig]) -> str:
    """
    Compute a stable hash key for a set of evaluation tasks.

    This is used to create unique GCS paths for different sets of eval tasks.

    Args:
        eval_tasks: List of EvalTaskConfig objects.

    Returns:
        A short hash string identifying this set of tasks.
    """
    # Sort tasks by name for stable ordering
    task_names = sorted(task.name for task in eval_tasks)
    content = "|".join(task_names)
    return hashlib.sha256(content.encode()).hexdigest()[:12]


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
    """
    import datasets

    log_obj = log or logger
    kwargs = dataset_kwargs or {}

    # Check if already cached in GCS
    marker_path = os.path.join(gcs_path, ".eval_datasets_cached")
    if fsspec_exists(marker_path):
        log_obj.info(f"Eval datasets already cached at {gcs_path}")
        return gcs_path

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
            except Exception as e:
                log_obj.warning(f"Failed to download dataset {dataset_path}: {e}")
                # Continue with other datasets

        # Upload cache directory to GCS
        log_obj.info(f"Uploading datasets to {gcs_path}")
        fs = fsspec.core.url_to_fs(gcs_path)[0]
        fs.put(cache_dir, gcs_path, recursive=True)

        # Write marker file
        task_names = sorted(task.name for task in eval_tasks)
        marker_content = "\n".join(task_names)
        with fsspec.open(marker_path, "w") as f:
            f.write(marker_content)  # type: ignore[union-attr]

        log_obj.info(f"Eval datasets cached to {gcs_path}")
        return gcs_path

    finally:
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()


def load_eval_datasets_from_gcs(
    gcs_path: str,
    local_cache_dir: str,
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
        local_cache_dir: Local cache directory to sync to (typically HF_DATASETS_CACHE).
        log: Optional logger instance.

    Returns:
        True if sync was successful, False otherwise.
    """
    log_obj = log or logger

    # Check if GCS cache exists
    marker_path = os.path.join(gcs_path, ".eval_datasets_cached")
    if not fsspec_exists(marker_path):
        log_obj.info(f"No eval datasets cache found at {gcs_path}")
        return False

    # Ensure local cache directory exists
    os.makedirs(local_cache_dir, exist_ok=True)

    # Sync from GCS to local
    log_obj.info(f"Syncing eval datasets from {gcs_path} to {local_cache_dir}")
    try:
        fs = fsspec.core.url_to_fs(gcs_path)[0]
        fs.get(gcs_path, local_cache_dir, recursive=True)
        log_obj.info(f"Successfully synced eval datasets to {local_cache_dir}")
        return True
    except Exception as e:
        log_obj.warning(f"Failed to sync eval datasets from GCS: {e}")
        return False


def ensure_eval_datasets_cached(
    eval_tasks: Sequence[EvalTaskConfig],
    cache_base_path: str,
    *,
    local_cache_dir: str | None = None,
    log: logging.Logger | None = None,
) -> str:
    """
    Ensure evaluation datasets are cached in GCS, downloading if necessary.

    This is the main entry point for the eval dataset caching system.
    It computes a cache key for the set of tasks, checks if they're already
    cached in GCS, and downloads them if not.

    Args:
        eval_tasks: List of EvalTaskConfig objects specifying the evaluation tasks.
        cache_base_path: Base GCS path for caching (e.g., "gs://bucket/.eval-datasets").
        local_cache_dir: Optional local cache directory for intermediate storage.
        log: Optional logger instance.

    Returns:
        The GCS path where datasets are cached.
    """
    log_obj = log or logger

    if not eval_tasks:
        log_obj.warning("No eval tasks provided, nothing to cache")
        return cache_base_path

    # Compute cache key and full path
    cache_key = compute_cache_key(eval_tasks)
    gcs_path = os.path.join(cache_base_path, cache_key)

    log_obj.info(f"Ensuring eval datasets are cached at {gcs_path}")

    return save_eval_datasets_to_gcs(
        eval_tasks,
        gcs_path,
        local_cache_dir=local_cache_dir,
        log=log_obj,
    )


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
