# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit all Arctic teacher shards before student training."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import pyarrow.parquet as pq
from ladder_config import MANIFEST_ROOT, TEACHER_ID, TEACHER_REVISION
from rigging.filesystem import atomic_rename

MANIFEST_URL = f"{MANIFEST_ROOT}/manifest.json"
TEACHER_ROOT = f"{MANIFEST_ROOT}/teacher-arctic-v1"
AUDIT_URL = f"{TEACHER_ROOT}/audit.json"
EMBEDDING_DIMENSION = 256
RESULT_FILE = Path("/tmp/luxical-arctic-teacher-audit")
MANIFEST_METADATA_KEY = b"luxical_manifest_sha256"
TEACHER_ID_METADATA_KEY = b"luxical_teacher_id"
TEACHER_REVISION_METADATA_KEY = b"luxical_teacher_revision"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def read_json(url: str) -> dict[str, Any]:
    """Read one JSON object from private storage."""
    filesystem, path = fsspec.core.url_to_fs(url)
    with filesystem.open(path) as file:
        return json.load(file)


def teacher_output_url(manifest_output_url: str) -> str:
    """Return the teacher file paired with one manifest source file."""
    return f"{TEACHER_ROOT}/sources/{Path(manifest_output_url).name}"


def source_metrics(url: str, expected_rows: int, manifest_sha256: str) -> dict[str, Any]:
    """Read and validate one complete teacher source file."""
    filesystem, path = fsspec.core.url_to_fs(url)
    if not filesystem.exists(path):
        raise FileNotFoundError(f"Missing teacher output: {url}")
    with pq.ParquetFile(path, filesystem=filesystem) as parquet_file:
        metadata = parquet_file.schema_arrow.metadata or {}
    expected_metadata = {
        MANIFEST_METADATA_KEY: manifest_sha256.encode(),
        TEACHER_ID_METADATA_KEY: TEACHER_ID.encode(),
        TEACHER_REVISION_METADATA_KEY: TEACHER_REVISION.encode(),
    }
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise ValueError(f"Teacher output has different input metadata: {url}")
    table = pq.read_table(path, filesystem=filesystem, columns=["embedding"])
    if len(table) != expected_rows:
        raise ValueError(f"Teacher output has {len(table)} rows; expected {expected_rows}: {url}")
    embeddings = table["embedding"].combine_chunks()
    if embeddings.type.list_size != EMBEDDING_DIMENSION:
        raise ValueError(f"Teacher output has dimension {embeddings.type.list_size}: {url}")
    quantized = embeddings.values.to_numpy(zero_copy_only=False).reshape(
        expected_rows,
        EMBEDDING_DIMENSION,
    )
    if quantized.dtype != np.uint8:
        raise ValueError(f"Teacher output has dtype {quantized.dtype}: {url}")
    unique_rows = int(np.unique(quantized, axis=0).shape[0])
    varying_dimensions = int(np.count_nonzero(quantized.max(axis=0) > quantized.min(axis=0)))
    if expected_rows > 1 and unique_rows == 1:
        raise ValueError(f"Teacher output is constant: {url}")
    if varying_dimensions == 0:
        raise ValueError(f"Teacher output has no varying dimensions: {url}")
    return {
        "rows": expected_rows,
        "unique_quantized_rows": unique_rows,
        "unique_quantized_fraction": unique_rows / expected_rows,
        "varying_dimensions": varying_dimensions,
        "minimum_quantized_value": int(quantized.min()),
        "maximum_quantized_value": int(quantized.max()),
    }


def write_json(url: str, value: dict[str, Any]) -> None:
    """Write one JSON object atomically."""
    filesystem, path = fsspec.core.url_to_fs(url)
    with atomic_rename(path, fs=filesystem) as temporary_path:
        with filesystem.open(temporary_path, "w") as file:
            json.dump(value, file, indent=2, sort_keys=True)


def parse_args() -> argparse.Namespace:
    """Parse the expected shard count."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-shards", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    """Audit shard reports and every teacher source output."""
    arguments = parse_args()
    manifest = read_json(MANIFEST_URL)
    shard_reports = []
    for shard_index in range(arguments.num_shards):
        report_url = f"{TEACHER_ROOT}/shards/shard-{shard_index:02d}-of-{arguments.num_shards:02d}.json"
        shard_report = read_json(report_url)
        if shard_report["shard_index"] != shard_index:
            raise ValueError(f"Teacher shard report has index {shard_report['shard_index']}; expected {shard_index}")
        if shard_report["num_shards"] != arguments.num_shards:
            raise ValueError(
                f"Teacher shard {shard_index} has {shard_report['num_shards']} shards; "
                f"expected {arguments.num_shards}"
            )
        if shard_report["manifest_sha256"] != manifest["sha256"]:
            raise ValueError(f"Teacher shard {shard_index} has a different manifest digest")
        if shard_report["teacher_id"] != TEACHER_ID:
            raise ValueError(f"Teacher shard {shard_index} has teacher {shard_report['teacher_id']}")
        if shard_report["teacher_revision"] != TEACHER_REVISION:
            raise ValueError(f"Teacher shard {shard_index} has revision {shard_report['teacher_revision']}")
        shard_reports.append(shard_report)
    reported_sources = [source for report in shard_reports for source in report["sources"]]
    if len(reported_sources) != len(set(reported_sources)):
        raise ValueError("Teacher shard reports contain duplicate sources")
    if set(reported_sources) != set(manifest["sources"]):
        missing = sorted(set(manifest["sources"]) - set(reported_sources))
        extra = sorted(set(reported_sources) - set(manifest["sources"]))
        raise ValueError(f"Teacher shard report coverage differs: missing={missing}, extra={extra}")

    sources: dict[str, dict[str, Any]] = {}
    for index, (source, result) in enumerate(sorted(manifest["sources"].items()), start=1):
        logger.info("Auditing teacher source %d/%d: %s", index, len(manifest["sources"]), source)
        expected_rows = result["counts"]["train_3m"] + result["counts"]["eval"]
        output_url = teacher_output_url(result["output_url"])
        sources[source] = {
            "output_url": output_url,
            "metrics": source_metrics(output_url, expected_rows, manifest["sha256"]),
        }
    expected_total = sum(
        result["counts"]["train_3m"] + result["counts"]["eval"] for result in manifest["sources"].values()
    )
    actual_total = sum(result["metrics"]["rows"] for result in sources.values())
    if actual_total != expected_total:
        raise ValueError(f"Teacher total is {actual_total}; expected {expected_total}")
    report = {
        "manifest_url": MANIFEST_URL,
        "manifest_sha256": manifest["sha256"],
        "teacher_id": TEACHER_ID,
        "teacher_revision": TEACHER_REVISION,
        "num_shards": arguments.num_shards,
        "source_count": len(sources),
        "row_count": actual_total,
        "minimum_source_unique_fraction": min(
            result["metrics"]["unique_quantized_fraction"] for result in sources.values()
        ),
        "minimum_source_varying_dimensions": min(result["metrics"]["varying_dimensions"] for result in sources.values()),
        "sources": sources,
    }
    write_json(AUDIT_URL, report)
    summary = {
        "audit_url": AUDIT_URL,
        "source_count": report["source_count"],
        "row_count": report["row_count"],
        "minimum_source_unique_fraction": report["minimum_source_unique_fraction"],
        "minimum_source_varying_dimensions": report["minimum_source_varying_dimensions"],
    }
    RESULT_FILE.write_text(json.dumps(summary, sort_keys=True))
    logger.info("LUXICAL_ARCTIC_TEACHER_AUDIT=%s", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
