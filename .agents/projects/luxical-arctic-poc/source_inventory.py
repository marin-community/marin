# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inventory normalized sources for the Luxical scaling ladder."""

import json
import logging
import posixpath
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import fsspec
import pyarrow.parquet as pq
from marin.datakit.sources import all_sources
from rigging.filesystem import atomic_rename

MIRROR_PREFIX = "s3://marin-us-east-02a/marin"
CANONICAL_BUCKET = "marin-us-west2"
OUTPUT_URL = "s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/source_inventory.json"
MIN_USABLE_SOURCES = 140
REQUIRED_COLUMNS = frozenset(("id", "text"))
RESULT_FILE = Path("/tmp/luxical-arctic-source-inventory")
REGISTRY_REVISION = "656d77bff319a851cb775e5bef33570ccfd9a9f8"
LATEST_SOURCE_PATH_OVERRIDES = {
    # These fixed outputs can be newer than the source registry at REGISTRY_REVISION.
    "biocorpus": "gs://marin-us-west2/normalized/biocorpus_dd02c263",
    "ghalogs/public": "gs://marin-us-west2/normalized/ghalogs/public_55a2fec7",
    "identity-data/content": "gs://marin-us-west2/normalized/identity-data/content_815a5afb",
    "nemotron_code_v1/content": "gs://marin-us-west2/normalized/nemotron_code_v1_content_b6337b6c",
    "stack-v3": "gs://marin-us-west2/normalized/stack-v3_32b6fa6f",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceInventory:
    """Store the inventory result for one normalized source."""

    name: str
    canonical_output_path: str
    artifact_url: str
    main_output_dir: str | None
    parquet_file_count: int
    inspected_rows: int
    columns: tuple[str, ...]
    usable: bool
    error: str | None


def mirror_url(url: str) -> str:
    """Convert a canonical Datakit URL to its private mirror URL."""
    if url.startswith(f"{MIRROR_PREFIX.rstrip('/')}/"):
        return url
    protocol, separator, bucket_and_path = url.partition("://")
    if protocol != "gs" or not separator:
        raise ValueError(f"Path is not a canonical GCS URL: {url}")
    bucket, path_separator, relative_path = bucket_and_path.partition("/")
    if not path_separator or not relative_path:
        raise ValueError(f"Path has no object key: {url}")
    if bucket != CANONICAL_BUCKET:
        raise ValueError(f"Unsupported canonical bucket {bucket}: {url}")
    return f"{MIRROR_PREFIX.rstrip('/')}/{relative_path}"


def read_json(url: str) -> dict[str, Any]:
    """Read one JSON object from a URL."""
    filesystem, path = fsspec.core.url_to_fs(url)
    with filesystem.open(path) as file:
        return json.load(file)


def artifact_main_output_dir(artifact_url: str) -> str:
    """Return the mirrored main output directory from an artifact."""
    artifact = read_json(artifact_url)
    payload = artifact.get("result", artifact)
    return mirror_url(payload["main_output_dir"])


def parquet_paths(main_output_dir: str) -> tuple[Any, list[str]]:
    """Return the filesystem and normalized Parquet file paths."""
    filesystem, root = fsspec.core.url_to_fs(main_output_dir)
    paths = sorted(filesystem.glob(posixpath.join(root, "*.parquet")))
    return filesystem, paths


def inspect_source(name: str, canonical_output_path: str) -> SourceInventory:
    """Inspect one source without reading document text."""
    artifact_url = f"{mirror_url(canonical_output_path).rstrip('/')}/.artifact.json"
    main_output_dir = None
    parquet_file_count = 0
    inspected_rows = 0
    columns: tuple[str, ...] = ()
    try:
        main_output_dir = artifact_main_output_dir(artifact_url)
        filesystem, paths = parquet_paths(main_output_dir)
        parquet_file_count = len(paths)
        if not paths:
            raise FileNotFoundError(f"No Parquet files under {main_output_dir}")

        inspected_paths = paths[:1] if len(paths) == 1 else [paths[0], paths[-1]]
        schemas = []
        for path in inspected_paths:
            with pq.ParquetFile(path, filesystem=filesystem) as parquet_file:
                schemas.append(frozenset(parquet_file.schema_arrow.names))
                inspected_rows += parquet_file.metadata.num_rows
        if any(not REQUIRED_COLUMNS.issubset(schema) for schema in schemas):
            raise ValueError(f"Required columns are missing from inspected schemas: {schemas}")
        if inspected_rows == 0:
            raise ValueError("Inspected Parquet files contain no rows")
        columns = tuple(sorted(set(schemas[0]).intersection(*schemas[1:])))
        error = None
        usable = True
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
        usable = False
        logger.warning("Source %s is not usable: %s", name, error)

    return SourceInventory(
        name=name,
        canonical_output_path=canonical_output_path,
        artifact_url=artifact_url,
        main_output_dir=main_output_dir,
        parquet_file_count=parquet_file_count,
        inspected_rows=inspected_rows,
        columns=columns,
        usable=usable,
        error=error,
    )


def write_report(report: dict[str, Any]) -> None:
    """Write the complete inventory report atomically."""
    filesystem, path = fsspec.core.url_to_fs(OUTPUT_URL)
    with atomic_rename(path, fs=filesystem) as temporary_path:
        with filesystem.open(temporary_path, "w") as file:
            json.dump(report, file, indent=2, sort_keys=True)


def main() -> None:
    """Inventory every active Datakit source."""
    sources = {name: source.normalized.output_path for name, source in all_sources().items()}
    sources.update(LATEST_SOURCE_PATH_OVERRIDES)
    inventory = []
    for index, (name, canonical_output_path) in enumerate(
        sorted(sources.items()),
        start=1,
    ):
        logger.info("Inspecting source %d/%d: %s", index, len(sources), name)
        inventory.append(
            inspect_source(
                name=name,
                canonical_output_path=canonical_output_path,
            )
        )

    usable_sources = [result.name for result in inventory if result.usable]
    failed_sources = {result.name: result.error for result in inventory if not result.usable}
    summary = {
        "registered_source_count": len(inventory),
        "usable_source_count": len(usable_sources),
        "failed_source_count": len(failed_sources),
        "minimum_usable_sources": MIN_USABLE_SOURCES,
        "passes_source_gate": len(usable_sources) >= MIN_USABLE_SOURCES,
        "usable_sources": usable_sources,
        "failed_sources": failed_sources,
    }
    report = {
        "registry_revision": REGISTRY_REVISION,
        "mirror_prefix": MIRROR_PREFIX,
        "output_url": OUTPUT_URL,
        "summary": summary,
        "sources": [asdict(result) for result in inventory],
    }
    write_report(report)
    RESULT_FILE.write_text(json.dumps(summary, sort_keys=True))
    logger.info("LUXICAL_ARCTIC_SOURCE_INVENTORY=%s", json.dumps(summary, sort_keys=True))
    if not summary["passes_source_gate"]:
        raise ValueError(
            f"Only {summary['usable_source_count']} sources are usable; " f"the ladder requires {MIN_USABLE_SOURCES}"
        )


if __name__ == "__main__":
    main()
