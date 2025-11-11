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

import dataclasses
import logging
import os
from dataclasses import dataclass

import ray

# Import evaluation dataset conversion steps so executor can resolve paths
from experiments.train_test_overlap.eval_datasets_overlap import (
    ai2_arc_convert_dolma,
    bbh_convert_dolma,
    boolq_convert_dolma,
    commonsense_qa_convert_dolma,
    gpqa_convert_dolma,
    gsm8k_convert_dolma,
    hellaswag_convert_dolma,
    humaneval_convert_dolma,
    instruction_following_convert_dolma,
    lambada_openai_convert_dolma,
    math_convert_dolma,
    mmlu_convert_dolma,
    mmlu_pro_convert_dolma,
    musr_convert_dolma,
    openbookqa_convert_dolma,
    piqa_convert_dolma,
    truthful_qa_convert_dolma,
    winograd_wsc_convert_dolma,
)
from marin.core.runtime import simple_backpressure
from marin.execution.executor import ExecutorStep
from marin.processing.classification.dedupe import DedupeConfig, DedupMode, NGramConfig, dedupe_with_config_resources
from marin.utils import fsspec_glob

# File types that the dedupe pipeline knows how to handle.
SUPPORTED_SHARD_EXTENSIONS: tuple[str, ...] = (
    ".parquet",
    ".jsonl.gz",
    ".jsonl.zst",
    ".jsonl",
    ".json.gz",
    ".json.zst",
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# List of evaluation dataset steps - same as in aggregate_total.py
EVAL_DATASET_STEPS: list[ExecutorStep] = [
    gsm8k_convert_dolma,
    math_convert_dolma,
    truthful_qa_convert_dolma,
    bbh_convert_dolma,
    mmlu_convert_dolma,
    humaneval_convert_dolma,
    instruction_following_convert_dolma,
    gpqa_convert_dolma,
    musr_convert_dolma,
    mmlu_pro_convert_dolma,
    hellaswag_convert_dolma,
    ai2_arc_convert_dolma,
    boolq_convert_dolma,
    commonsense_qa_convert_dolma,
    lambada_openai_convert_dolma,
    openbookqa_convert_dolma,
    piqa_convert_dolma,
    winograd_wsc_convert_dolma,
]


ALLOWED_TPU_TYPES: tuple[str, ...] = ("v4-8", "v5p-8", "v6e-4")


@dataclass(frozen=True)
class UnifiedResources:
    """Unified resource configuration for Ray tasks.

    This provides a single place to describe CPU/memory and TPU-related resource tags
    in a consistent way across tasks.

    Fields:
        tpu_type: TPU type string (e.g., "v4-8", "v6e-4"). If provided and
            `tpu_head_fraction` is set, we add a custom resource key
            f"TPU-{tpu_type}-head": tpu_head_fraction.
        tpu_head_fraction: Fractional amount to reserve against the TPU head resource
            (used as a global concurrency limiter for tasks on a particular TPU fleet).
        tpus: Number of TPU chips to require on a worker (adds {"TPU": tpus}).
            Most non-TPU tasks can leave this as None.
        num_cpus: Number of CPUs for the Ray task options.
        memory: Memory in bytes for the Ray task options.
        extra: Additional custom resource requirements to merge into the resources dict.
    """

    tpu_type: str | None = None
    tpu_head_fraction: float | None = None
    tpus: int | None = None
    num_cpus: int | None = None
    memory: int | None = None
    extra: dict[str, float] | None = None

    def to_ray_overrides(self) -> dict:
        overrides: dict = {}
        resources: dict[str, float] = {}

        if self.num_cpus is not None:
            overrides["num_cpus"] = self.num_cpus
        if self.memory is not None:
            overrides["memory"] = self.memory

        if self.tpu_type and self.tpu_head_fraction is not None:
            if self.tpu_type not in ALLOWED_TPU_TYPES:
                raise ValueError(f"Unsupported tpu_type '{self.tpu_type}'. Allowed: {', '.join(ALLOWED_TPU_TYPES)}")
            resources[f"TPU-{self.tpu_type}-head"] = self.tpu_head_fraction
        if self.tpus is not None:
            resources["TPU"] = float(self.tpus)
        if self.extra:
            resources.update(self.extra)

        if resources:
            overrides["resources"] = resources
        return overrides


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for a single dataset to process for train-test overlap detection."""

    name: str
    """Human-readable name for the dataset (used in output paths)."""

    path: str
    """Path to the dataset directory (local, GCS, or S3)."""

    max_in_flight: int
    """Maximum number of parallel tasks to run for this dataset."""

    text_field: str = "text"
    """Name of the text field in the parquet file."""


@dataclass(frozen=True)
class ShardedDedupeConfig:
    """Configuration for running dedupe across multiple shards with backpressure."""

    dataset_dir: str
    output_path: str
    max_in_flight: int = 16
    eval_dataset_steps: list[ExecutorStep] = None  # Evaluation dataset steps for path resolution
    text_field: str = "text"
    # Dedupe parallelism inside Dolma (number of processes in DedupeConfig)
    processes: int | None = None
    # Ray resource overrides - if provided, will override BASE_DEDUPE_CONFIG defaults
    num_cpus: int | None = None
    memory: int | None = None  # in bytes
    resources: dict[str, float] | None = None
    # Unified resources alternative to num_cpus/memory/resources above. If provided,
    # takes precedence over individual fields when constructing Ray options.
    unified_resources: UnifiedResources | None = None
    # Directory for temporary files (defaults to /dev/shm for performance)
    temp_dir: str | None = None
    # Debug flag to control verbose print statements
    debug: bool = False


# Base dedupe configuration - modify this to change n-gram settings, processes, etc.
BASE_DEDUPE_CONFIG = DedupeConfig(
    input_path=[],  # Will be replaced with resolved evaluation dataset paths
    output_path="",  # Will be replaced per shard
    attribute_name="ngram_overlap",
    false_positive_rate=1e-20,
    ngram=NGramConfig(
        ngram_length=[15],  # Multiple n-gram sizes - modify this to change n-grams
        overlap_threshold=1e-6,
        stride=0,
    ),
    processes=15,  # Modify this to change number of processes
    mode=DedupMode.TRAIN_TEST_OVERLAP,
    decontaminate_source="",  # Will be replaced per shard
    # Ray resource configuration - modify these defaults as needed
    num_cpus=15,
    memory=16 * 1024 * 1024 * 1024,  # 16GB
    resources=None,
    # Debug flag - set to True to enable verbose print statements
    debug=False,
)


def make_task(
    shard_path: str,
    base_output_path: str,
    dataset_dir: str,
    eval_dataset_steps: list[ExecutorStep],
    text_field: str,
    base_config: DedupeConfig = BASE_DEDUPE_CONFIG,
    temp_dir: str | None = None,
    debug: bool = False,
) -> DedupeConfig:
    """Create a DedupeConfig for a single shard using the base config.

    Args:
        shard_path: Path to the training shard file
        base_output_path: Base output directory for results
        dataset_dir: Root dataset directory (used for relative path calculation)
        eval_dataset_steps: List of evaluation dataset ExecutorSteps to resolve paths for
        text_field: Name of the text field in the data files
        base_config: Base configuration to modify for this shard
        temp_dir: Directory for temporary files (overrides base_config if provided)

    Returns:
        DedupeConfig configured for this specific shard with resolved evaluation dataset paths
    """

    # Get relative path from dataset_dir to preserve directory structure
    relative_path = get_relative_path_no_extension(shard_path, dataset_dir)
    output_path = os.path.join(base_output_path, relative_path)

    # Use dataclasses.replace to create a new config with the shard-specific values
    replace_args = {
        "input_path": eval_dataset_steps,
        "output_path": output_path,
        "decontaminate_source": shard_path,
        "text_field": text_field,
        "debug": debug,
    }

    # Only include temp_dir if it's provided (don't override with None)
    if temp_dir is not None:
        replace_args["temp_dir"] = temp_dir

    return dataclasses.replace(base_config, **replace_args)


@ray.remote
def run_all_shards(config: ShardedDedupeConfig) -> str:
    """
    Discover all dataset shards and launch dedupe tasks with backpressure.

    Automatically resolves evaluation dataset paths using the executor framework
    and runs train-test overlap detection between each training shard and all
    evaluation datasets.

    Args:
        config: Configuration including dataset directory, output path, and evaluation dataset steps

    Returns:
        Success message with number of shards processed
    """
    logger.info(f"Looking for dataset shards in {config.dataset_dir}")
    # Find all supported dataset shards under root (Parquet or compressed JSONL)

    shard_paths = find_dataset_shards(config.dataset_dir)

    # Apply resource overrides to base config if specified
    base_config = BASE_DEDUPE_CONFIG
    # Collect overrides for the underlying DedupeConfig
    overrides: dict = {}
    # Allow customizing internal Dolma processes
    if config.processes is not None:
        overrides["processes"] = config.processes

    # Prefer unified resources if provided; otherwise use individual fields
    if config.unified_resources is not None:
        ur_overrides = config.unified_resources.to_ray_overrides()
        overrides.update(ur_overrides)
    else:
        # Allow customizing Ray resources for the remote task
        if config.num_cpus is not None:
            overrides["num_cpus"] = config.num_cpus
        if config.memory is not None:
            overrides["memory"] = config.memory
        if config.resources is not None:
            overrides["resources"] = config.resources

    if overrides:
        base_config = dataclasses.replace(BASE_DEDUPE_CONFIG, **overrides)

    # Choose remote function: only need a custom one if Ray resource overrides are present
    if (
        config.unified_resources is not None
        or (config.num_cpus is not None)
        or (config.memory is not None)
        or (config.resources is not None)
    ):
        remote_func = dedupe_with_config_resources(base_config)
    else:
        # Use default remote function
        from marin.processing.classification.dedupe import dedupe

        remote_func = dedupe

    # Generator of arguments for each Ray task
    task_generator = (
        (
            make_task(
                shard_path,
                config.output_path,
                config.dataset_dir,
                config.eval_dataset_steps,
                config.text_field,
                base_config=base_config,
                temp_dir=config.temp_dir,
                debug=config.debug,
            ),
        )
        for shard_path in shard_paths
    )

    # Launch tasks with simple_backpressure
    for ref in simple_backpressure(
        remote_func,
        task_generator,
        max_in_flight=config.max_in_flight,
        fetch_local=True,
    ):
        ray.get(ref)

    return f"Sharded dedupe pipeline completed! Processed {len(shard_paths)} shards."


def find_dataset_shards(root_dir: str) -> list[str]:
    """
    Find all dataset shard files under root_dir with supported extensions.
    Supported extensions: .parquet, .jsonl.gz, .jsonl.zst, .jsonl, .json.gz, .json.zst.

    Uses recursive search to find files at any depth under root_dir.

    Raises:
        FileNotFoundError: if no matching files are found.

    Returns:
        A sorted list of unique file paths.
    """
    root = root_dir.rstrip("/")
    # gather all matching files for each supported extension
    matches = [fp for ext in SUPPORTED_SHARD_EXTENSIONS for fp in fsspec_glob(os.path.join(root, f"**/*{ext}"))]

    if not matches:
        exts = ", ".join(SUPPORTED_SHARD_EXTENSIONS)
        raise FileNotFoundError(f"No shard files with extensions ({exts}) found under {root_dir}")
    return sorted(set(matches))


def get_relative_path_no_extension(shard_path: str, dataset_dir: str) -> str:
    """
    Get the relative path from dataset_dir to shard_path and remove file extensions.

    Args:
        shard_path: Full path to a shard file
        dataset_dir: Root directory path

    Returns:
        Relative path with extensions removed, preserving directory structure

    Example:
        >>> get_relative_path_no_extension(
        ...     "gs://bucket/raw/starcoderdata-720c8c/9fc30b5/ada/train-00000-of-00001.parquet",
        ...     "gs://bucket/raw/starcoderdata-720c8c"
        ... )
        "9fc30b5/ada/train-00000-of-00001"
    """
    # Normalize paths to handle trailing slashes consistently
    dataset_dir = dataset_dir.rstrip("/")

    # Get relative path from dataset_dir
    if shard_path.startswith(dataset_dir + "/"):
        relative_path = shard_path[len(dataset_dir) + 1 :]  # +1 to skip the "/"
    else:
        # Fallback to just the basename if we can't determine relative path
        relative_path = os.path.basename(shard_path)

    # Remove the file extension if it's one we know about
    for ext in SUPPORTED_SHARD_EXTENSIONS:
        if relative_path.endswith(ext):
            relative_path = relative_path[: -len(ext)]
            break

    return relative_path


def clean_shard_basename(shard_path: str) -> str:
    """
    Extract basename from shard path and remove supported file extensions.

    Args:
        shard_path: Full path to a shard file

    Returns:
        Clean basename with extensions removed (e.g., "train-00001-of-00128")

    Example:
        >>> clean_shard_basename("gs://bucket/path/train-00001-of-00128.parquet")
        "train-00001-of-00128"
    """
    basename = os.path.basename(shard_path)
    for ext in SUPPORTED_SHARD_EXTENSIONS:
        if basename.endswith(ext):
            return basename[: -len(ext)]
    return basename
