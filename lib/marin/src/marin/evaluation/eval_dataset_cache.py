# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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

import contextlib
import json
import logging
import os
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import fsspec
import rigging.filesystem  # noqa: F401

from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import ExecutorStep
from marin.utils import call_with_hf_backoff, fsspec_exists

logger = logging.getLogger(__name__)
MANIFEST_FILE = ".eval_datasets_manifest.json"
HF_CACHE_LAYOUT_VERSION = 2


def default_hf_cache_root() -> str:
    """Return the default Hugging Face cache root for this process."""
    return os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")


def _hf_cache_subdirs(cache_root: str) -> dict[str, str]:
    return {
        "root": cache_root,
        "datasets": os.path.join(cache_root, "datasets"),
        "hub": os.path.join(cache_root, "hub"),
        "modules": os.path.join(cache_root, "modules"),
    }


@contextlib.contextmanager
def _temporary_hf_cache_root(cache_root: str):
    """Temporarily redirect HF caches into one explicit root."""
    paths = _hf_cache_subdirs(cache_root)
    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    env_updates = {
        "HF_HOME": paths["root"],
        "HF_DATASETS_CACHE": paths["datasets"],
        "HF_HUB_CACHE": paths["hub"],
        "HUGGINGFACE_HUB_CACHE": paths["hub"],
        "HF_MODULES_CACHE": paths["modules"],
    }
    previous_env = {key: os.environ.get(key) for key in env_updates}
    try:
        os.environ.update(env_updates)
        if "datasets" in sys.modules:
            try:
                import datasets.config as datasets_config

                datasets_config.HF_HOME = paths["root"]
                datasets_config.HF_DATASETS_CACHE = paths["datasets"]
                datasets_config.HF_HUB_CACHE = paths["hub"]
            except Exception:
                logger.debug("datasets.config unavailable while rebasing HF cache root", exc_info=True)
        if "huggingface_hub.constants" in sys.modules:
            try:
                import huggingface_hub.constants as hub_constants

                hub_constants.HF_HOME = paths["root"]
                hub_constants.HF_HUB_CACHE = paths["hub"]
                hub_constants.HUGGINGFACE_HUB_CACHE = paths["hub"]
                hub_constants.HF_TOKEN_PATH = os.path.join(paths["root"], "token")
            except Exception:
                logger.debug("huggingface_hub.constants unavailable while rebasing HF cache root", exc_info=True)
        yield paths
    finally:
        for key, previous in previous_env.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


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

    cache_layout_version: int = HF_CACHE_LAYOUT_VERSION
    """Layout version for the uploaded HF cache root."""

    includes_hf_hub_cache: bool = True
    """Whether the uploaded cache includes HF Hub metadata/cache content."""

    includes_hf_modules_cache: bool = True
    """Whether the uploaded cache includes HF modules cache content."""

    def to_dict(self) -> dict:
        return {
            "task_names": self.task_names,
            "cached_datasets": [[d[0], d[1]] for d in self.cached_datasets],
            "failed_datasets": [[d[0], d[1], d[2]] for d in self.failed_datasets],
            "cache_layout_version": self.cache_layout_version,
            "includes_hf_hub_cache": self.includes_hf_hub_cache,
            "includes_hf_modules_cache": self.includes_hf_modules_cache,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CacheManifest":
        return cls(
            task_names=data["task_names"],
            cached_datasets=[(d[0], d[1]) for d in data["cached_datasets"]],
            failed_datasets=[(d[0], d[1], d[2]) for d in data["failed_datasets"]],
            cache_layout_version=int(data.get("cache_layout_version", 1)),
            includes_hf_hub_cache=bool(data.get("includes_hf_hub_cache", False)),
            includes_hf_modules_cache=bool(data.get("includes_hf_modules_cache", False)),
        )

    def is_complete(self) -> bool:
        """Return True if all requested datasets were successfully cached."""
        return len(self.failed_datasets) == 0

    def supports_full_offline_task_loading(self) -> bool:
        """Return True if the cache should support offline lm-eval task loading."""
        return (
            self.is_complete()
            and self.cache_layout_version >= HF_CACHE_LAYOUT_VERSION
            and self.includes_hf_hub_cache
            and self.includes_hf_modules_cache
        )


def load_cache_manifest(gcs_path: str) -> CacheManifest | None:
    """Load the eval cache manifest from a cache root, if it exists."""
    manifest_path = os.path.join(gcs_path, MANIFEST_FILE)
    if not fsspec_exists(manifest_path):
        return None
    with fsspec.open(manifest_path, "r") as f:
        return CacheManifest.from_dict(json.load(f))


def warm_task_metadata_cache(
    eval_tasks: Sequence[EvalTaskConfig],
    *,
    log: logging.Logger | None = None,
) -> None:
    """Warm lm-eval task metadata once so workers can resolve tasks offline later."""
    import lm_eval.tasks as tasks

    log_obj = log or logger
    task_manager = tasks.TaskManager()
    unique_task_names = sorted({task.name for task in eval_tasks})
    for task_name in unique_task_names:
        log_obj.info(f"Warming task metadata cache for task: {task_name}")
        call_with_hf_backoff(
            lambda name=task_name: tasks.get_task_dict([name], task_manager),
            context=f"warm task metadata {task_name}",
        )


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
    log_obj = log or logger
    kwargs = dataset_kwargs or {}

    # Check if already cached in GCS with a complete manifest
    manifest_path = os.path.join(gcs_path, MANIFEST_FILE)
    existing_manifest = load_cache_manifest(gcs_path)
    if existing_manifest is not None:
        try:
            if existing_manifest.supports_full_offline_task_loading():
                log_obj.info(f"Eval datasets already cached at {gcs_path} (complete)")
                return gcs_path
            else:
                log_obj.info(
                    f"Found incomplete cache at {gcs_path} "
                    f"(failed={len(existing_manifest.failed_datasets)}, "
                    f"layout_version={existing_manifest.cache_layout_version}, "
                    f"hub_cache={existing_manifest.includes_hf_hub_cache}). Re-downloading..."
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
        cache_root = temp_dir_obj.name
    else:
        cache_root = local_cache_dir
        os.makedirs(cache_root, exist_ok=True)

    try:
        with _temporary_hf_cache_root(cache_root) as cache_paths:
            import datasets

            datasets_cache_dir = cache_paths["datasets"]
            log_obj.info(f"Downloading {len(datasets_needed)} datasets to {datasets_cache_dir}")
            cached_datasets: list[tuple[str, str | None]] = []
            failed_datasets: list[tuple[str, str | None, str]] = []

            for dataset_path, dataset_name in datasets_needed:
                log_obj.info(f"Downloading dataset: {dataset_path} (config: {dataset_name})")
                try:
                    call_with_hf_backoff(
                        lambda dp=dataset_path, dn=dataset_name: datasets.load_dataset(
                            path=dp,
                            name=dn,
                            cache_dir=datasets_cache_dir,
                            trust_remote_code=True,
                            **kwargs,
                        ),
                        context=f"download dataset {dataset_path}",
                    )
                    cached_datasets.append((dataset_path, dataset_name))
                except Exception as e:
                    error_msg = str(e)
                    log_obj.warning(f"Failed to download dataset {dataset_path}: {error_msg}")
                    failed_datasets.append((dataset_path, dataset_name, error_msg))

            if not failed_datasets:
                warm_task_metadata_cache(eval_tasks, log=log_obj)

            token_path = os.path.join(cache_paths["root"], "token")
            if os.path.exists(token_path):
                os.remove(token_path)

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
        fs.put(cache_root + "/", gcs_path, recursive=True)

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
) -> CacheManifest | None:
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
        The cache manifest if sync was successful, otherwise None.
    """
    log_obj = log or logger

    # Check if GCS cache exists
    manifest = load_cache_manifest(gcs_path)
    if manifest is None:
        log_obj.info(f"No eval datasets cache found at {gcs_path}")
        return None
    if not manifest.is_complete():
        log_obj.warning(
            f"Eval datasets cache at {gcs_path} is incomplete "
            f"({len(manifest.failed_datasets)} failed). Proceeding with available datasets."
        )

    # Determine local cache directory
    if local_cache_dir is None:
        local_cache_dir = default_hf_cache_root()

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
        return manifest
    except Exception as e:
        log_obj.warning(f"Failed to sync eval datasets from GCS: {e}")
        return None


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
