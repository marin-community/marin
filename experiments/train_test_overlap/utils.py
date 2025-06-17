import os

from marin.utils import fsspec_glob

SUPPORTED_SHARD_EXTENSIONS = [
    ".parquet",
    ".jsonl.gz",
    ".jsonl.zst",
    ".jsonl",
    ".json.gz",
    ".json.zst",
]


def find_dataset_shards(root_dir: str) -> list[str]:
    """
    Find all dataset shard files under root_dir with supported extensions.
    Supported extensions: .parquet, .jsonl.gz, .jsonl.zst, .jsonl, .json.gz, .json.zst.

    Raises:
        FileNotFoundError: if no matching files are found.

    Returns:
        A sorted list of unique file paths.
    """
    root = root_dir.rstrip("/")
    matches: list[str] = []
    for ext in SUPPORTED_SHARD_EXTENSIONS:
        pattern = os.path.join(root, f"*{ext}")
        matches.extend(fsspec_glob(pattern))
    if not matches:
        exts = ", ".join(SUPPORTED_SHARD_EXTENSIONS)
        raise FileNotFoundError(f"No shard files with extensions ({exts}) found under {root_dir}")
    # dedupe and sort
    return sorted(set(matches))


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
        basename = basename.removesuffix(ext)
    return basename
