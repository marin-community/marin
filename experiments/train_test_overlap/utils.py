import dataclasses
import logging
import os
from dataclasses import dataclass

import ray

from marin.core.runtime import simple_backpressure
from marin.processing.classification.dedupe import DedupeConfig, DedupMode, NGramConfig, dedupe
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


@dataclass(frozen=True)
class ShardedDedupeConfig:
    """Configuration for running dedupe across multiple shards with backpressure."""

    dataset_dir: str
    output_path: str
    max_in_flight: int = 16


# Base dedupe configuration - modify this to change n-gram settings, processes, etc.
BASE_DEDUPE_CONFIG = DedupeConfig(
    input_path="gs://marin-us-central2/decontamination/",
    output_path="",  # Will be replaced per shard
    attribute_name="ngram_overlap",
    false_positive_rate=1e-12,
    ngram=NGramConfig(
        ngram_length=[15],  # Multiple n-gram sizes - modify this to change n-grams
        overlap_threshold=1e-6,
        stride=0,
    ),
    processes=16,  # Modify this to change number of processes
    mode=DedupMode.TRAIN_TEST_OVERLAP,
    decontaminate_source="",  # Will be replaced per shard
)


def make_task(
    shard_path: str, base_output_path: str, dataset_dir: str, base_config: DedupeConfig = BASE_DEDUPE_CONFIG
) -> DedupeConfig:
    """Create a DedupeConfig for a single shard using the base config."""

    # Get relative path from dataset_dir to preserve directory structure
    relative_path = get_relative_path_no_extension(shard_path, dataset_dir)
    output_path = os.path.join(base_output_path, relative_path)

    # Use dataclasses.replace to create a new config with the shard-specific values
    return dataclasses.replace(
        base_config,
        output_path=output_path,
        decontaminate_source=shard_path,
    )


@ray.remote
def run_all_shards(config: ShardedDedupeConfig) -> str:
    """
    Discover all dataset shards and launch dedupe tasks with backpressure.
    """
    logger.info(f"Looking for dataset shards in {config.dataset_dir}")
    # Find all supported dataset shards under root (Parquet or compressed JSONL)

    shard_paths = find_dataset_shards(config.dataset_dir)
    # Generator of arguments for each Ray task - now includes dataset_dir
    task_generator = ((make_task(shard_path, config.output_path, config.dataset_dir),) for shard_path in shard_paths)

    # Launch tasks with simple backpressure
    for ref in simple_backpressure(
        dedupe,
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
