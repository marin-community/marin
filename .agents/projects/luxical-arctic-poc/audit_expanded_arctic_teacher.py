# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit one expanded Arctic teacher rung before student preparation."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
from extend_arctic_teacher import (
    EMBEDDING_DIMENSION,
    expanded_root,
    expected_metadata,
    selected_expanded_table,
    teacher_root,
)
from ladder_config import read_json, write_json

MINIMUM_UNIQUE_FRACTION = 0.80
RESULT_FILE = Path("/tmp/luxical-expanded-arctic-audit")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def aligned_columns_equal(left: Any, right: Any, columns: tuple[str, ...]) -> bool:
    """Return true when each requested Arrow column is exactly equal."""
    return all(pc.all(pc.equal(left[column], right[column])).as_py() for column in columns)


def source_metrics(
    input_url: str,
    teacher_url: str,
    rung: str,
    expanded_manifest_sha256: str,
    base_manifest_sha256: str,
    expected_rows: int,
) -> dict[str, Any]:
    """Verify one teacher file and return collapse metrics."""
    source = selected_expanded_table(input_url, rung)
    filesystem, path = fsspec.core.url_to_fs(teacher_url)
    with pq.ParquetFile(path, filesystem=filesystem) as parquet_file:
        rows = parquet_file.metadata.num_rows
        metadata = parquet_file.schema_arrow.metadata or {}
    if rows != expected_rows or len(source) != expected_rows:
        raise ValueError(f"Row count mismatch for {teacher_url}: {len(source)}, {rows}, {expected_rows}")
    required_metadata = expected_metadata(expanded_manifest_sha256, base_manifest_sha256, rung)
    if any(metadata.get(key) != value for key, value in required_metadata.items()):
        raise ValueError(f"Teacher metadata mismatch for {teacher_url}")
    teacher = pq.read_table(path, filesystem=filesystem)
    columns = ("raw_sha256", "split", "eval_rank", "train_rank", f"in_{rung}")
    if not aligned_columns_equal(source, teacher, columns):
        raise ValueError(f"Teacher alignment mismatch for {teacher_url}")
    embedding = teacher["embedding"].combine_chunks()
    values = embedding.values.to_numpy(zero_copy_only=False).reshape(expected_rows, EMBEDDING_DIMENSION)
    if values.dtype != np.uint8:
        raise ValueError(f"Teacher dtype mismatch for {teacher_url}: {values.dtype}")
    unique_fraction = float(np.unique(values, axis=0).shape[0] / len(values))
    varying_dimensions = int(np.count_nonzero(values.max(axis=0) > values.min(axis=0)))
    return {
        "rows": expected_rows,
        "unique_quantized_fraction": unique_fraction,
        "varying_dimensions": varying_dimensions,
        "minimum_quantized_value": int(values.min()),
        "maximum_quantized_value": int(values.max()),
        "passed": unique_fraction >= MINIMUM_UNIQUE_FRACTION and varying_dimensions == EMBEDDING_DIMENSION,
    }


def audit_teacher(rung: str, num_shards: int) -> dict[str, Any]:
    """Audit all shard reports, source files, counts, and vector health."""
    manifest_url = f"{expanded_root(rung)}/manifest.json"
    manifest = read_json(manifest_url)
    reports = [
        read_json(f"{teacher_root(rung)}/shards/shard-{index:02d}-of-{num_shards:02d}.json")
        for index in range(num_shards)
    ]
    report_sources = [source for report in reports for source in report["sources"]]
    if len(report_sources) != len(set(report_sources)) or set(report_sources) != set(manifest["sources"]):
        raise ValueError("Expanded teacher shard source coverage differs from the manifest")
    if any(
        report["expanded_manifest_sha256"] != manifest["sha256"]
        or report["base_manifest_sha256"] != manifest["base_manifest_sha256"]
        or report["rung"] != rung
        or report["num_shards"] != num_shards
        for report in reports
    ):
        raise ValueError("Expanded teacher shard provenance differs")
    report_by_source = {source: report["sources"][source] for report in reports for source in report["sources"]}
    source_reports = {}
    for index, (source, result) in enumerate(sorted(manifest["sources"].items()), start=1):
        logger.info("Auditing source %d/%d: %s", index, len(manifest["sources"]), source)
        expected_rows = int(result["counts"][f"train_{rung}"]) + int(manifest["evaluation_rows_per_source"])
        teacher_url = report_by_source[source]["output_url"]
        source_reports[source] = source_metrics(
            result["output_url"],
            teacher_url,
            rung,
            manifest["sha256"],
            manifest["base_manifest_sha256"],
            expected_rows,
        ) | {"output_url": teacher_url}
    row_count = sum(report["rows"] for report in source_reports.values())
    expected_total = int(manifest["training_targets"][rung]) + len(source_reports) * int(
        manifest["evaluation_rows_per_source"]
    )
    if row_count != expected_total:
        raise ValueError(f"Expanded teacher has {row_count} rows; expected {expected_total}")
    return {
        "rung": rung,
        "manifest_url": manifest_url,
        "manifest_sha256": manifest["sha256"],
        "base_manifest_sha256": manifest["base_manifest_sha256"],
        "num_shards": num_shards,
        "source_count": len(source_reports),
        "row_count": row_count,
        "minimum_unique_quantized_fraction": min(
            report["unique_quantized_fraction"] for report in source_reports.values()
        ),
        "minimum_varying_dimensions": min(report["varying_dimensions"] for report in source_reports.values()),
        "all_sources_passed": all(report["passed"] for report in source_reports.values()),
        "sources": source_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rung", choices=("10m", "30m"), required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    arguments = parser.parse_args()
    if arguments.num_shards < 1:
        parser.error("--num-shards must be positive")
    report = audit_teacher(arguments.rung, arguments.num_shards)
    report_url = f"{teacher_root(arguments.rung)}/audit.json"
    write_json(report_url, report)
    summary = {key: value for key, value in report.items() if key != "sources"}
    summary["report_url"] = report_url
    RESULT_FILE.with_name(f"{RESULT_FILE.name}-{arguments.rung}").write_text(json.dumps(summary, sort_keys=True))
    logger.info("LUXICAL_EXPANDED_ARCTIC_AUDIT=%s", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
