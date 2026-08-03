# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Embed only the new rows in one expanded fast-student rung."""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from iris.cluster.client.job_info import get_job_info
from ladder_config import MANIFEST_ROOT, TEACHER_ID, TEACHER_REVISION, read_json, write_json
from rigging.filesystem import atomic_rename
from teacher_shard import (
    EMBEDDING_DIMENSION,
    MANIFEST_METADATA_KEY,
    TEACHER_ID_METADATA_KEY,
    TEACHER_REVISION_METADATA_KEY,
    new_teacher,
    teacher_batch,
)

BASE_MANIFEST_URL = f"{MANIFEST_ROOT}/manifest.json"
BASE_TEACHER_ROOT = f"{MANIFEST_ROOT}/teacher-arctic-v1"
EXPANDED_MANIFEST_METADATA_KEY = b"luxical_expanded_manifest_sha256"
BASE_MANIFEST_METADATA_KEY = b"luxical_base_manifest_sha256"
RUNG_METADATA_KEY = b"luxical_training_rung"
TABLE_BATCH_SIZE = 512
RESULT_FILE = Path("/tmp/luxical-expanded-arctic-teacher")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def expanded_root(rung: str) -> str:
    return f"{MANIFEST_ROOT}/fast-student/expanded-{rung}"


def teacher_root(rung: str) -> str:
    return f"{expanded_root(rung)}/teacher-arctic-v1"


def assigned_sources(manifest: dict[str, Any], rung: str, shard_index: int, num_shards: int) -> list[str]:
    """Assign sources by expanded row count to independent workers."""
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"Shard index {shard_index} is outside [0, {num_shards})")
    totals = [0] * num_shards
    assignments: dict[int, list[str]] = defaultdict(list)
    count_key = f"train_{rung}"
    for source in sorted(manifest["sources"], key=lambda name: (-manifest["sources"][name]["counts"][count_key], name)):
        target = min(range(num_shards), key=lambda index: (totals[index], index))
        assignments[target].append(source)
        totals[target] += int(manifest["sources"][source]["counts"][count_key])
    logger.info("Expanded teacher shard row totals: %s", totals)
    return sorted(assignments[shard_index])


def selected_expanded_table(url: str, rung: str) -> pa.Table:
    """Read evaluation rows and one expanded training rung."""
    filesystem, path = fsspec.core.url_to_fs(url)
    table = pq.read_table(path, filesystem=filesystem)
    selected = pc.or_(pc.equal(table["split"], "eval"), table[f"in_{rung}"])
    return table.filter(selected)


def parquet_table(url: str) -> pa.Table:
    """Read one Parquet table."""
    filesystem, path = fsspec.core.url_to_fs(url)
    return pq.read_table(path, filesystem=filesystem)


def checked_prefix_embeddings(
    expanded: pa.Table,
    prefix_source: pa.Table,
    prefix_teacher: pa.Table,
) -> np.ndarray:
    """Return aligned quantized embeddings for an unchanged prefix."""
    prefix_rows = len(prefix_source)
    if len(prefix_teacher) != prefix_rows or len(expanded) < prefix_rows:
        raise ValueError("The expanded, prefix, and teacher row counts do not align")
    columns = ("raw_sha256", "eval_rank", "train_rank")
    for column in columns:
        expected = prefix_source[column].to_pylist()
        if expanded[column].slice(0, prefix_rows).to_pylist() != expected:
            raise ValueError(f"The expanded prefix differs in {column}")
        if prefix_teacher[column].to_pylist() != expected:
            raise ValueError(f"The prefix teacher differs in {column}")
    embeddings = prefix_teacher["embedding"].combine_chunks()
    values = embeddings.values.to_numpy(zero_copy_only=False).reshape(prefix_rows, EMBEDDING_DIMENSION)
    if values.dtype != np.uint8:
        raise ValueError(f"The prefix teacher has dtype {values.dtype}; expected uint8")
    return values


def validate_teacher_metadata(table: pa.Table, expected: dict[bytes, bytes]) -> None:
    """Check that reused vectors identify their exact teacher inputs."""
    metadata = table.schema.metadata or {}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("The prefix teacher metadata differs from the fixed teacher inputs")


def teacher_prefix(rung: str, base_manifest: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Return the largest completed teacher rung that is a prefix of the target."""
    if rung == "10m":
        return base_manifest, None
    if rung != "30m":
        raise ValueError(f"The teacher rung is not supported: {rung}")
    prefix = read_json(f"{expanded_root('10m')}/manifest.json")
    if prefix["base_manifest_sha256"] != base_manifest["sha256"]:
        raise ValueError("The 10M prefix has a different base manifest")
    if int(prefix["training_targets"]["10m"]) != 10_000_000:
        raise ValueError("The 10M prefix has an incorrect training target")
    return prefix, "10m"


def prefix_metadata(prefix_manifest: dict[str, Any], prefix_rung: str | None) -> dict[bytes, bytes]:
    """Return the required metadata for reusable prefix vectors."""
    if prefix_rung is None:
        return {
            MANIFEST_METADATA_KEY: prefix_manifest["sha256"].encode(),
            TEACHER_ID_METADATA_KEY: TEACHER_ID.encode(),
            TEACHER_REVISION_METADATA_KEY: TEACHER_REVISION.encode(),
        }
    return expected_metadata(
        prefix_manifest["sha256"],
        prefix_manifest["base_manifest_sha256"],
        prefix_rung,
    )


def expected_metadata(expanded_manifest_sha256: str, base_manifest_sha256: str, rung: str) -> dict[bytes, bytes]:
    """Return metadata that binds an expanded teacher file to its inputs."""
    return {
        EXPANDED_MANIFEST_METADATA_KEY: expanded_manifest_sha256.encode(),
        BASE_MANIFEST_METADATA_KEY: base_manifest_sha256.encode(),
        RUNG_METADATA_KEY: rung.encode(),
        TEACHER_ID_METADATA_KEY: TEACHER_ID.encode(),
        TEACHER_REVISION_METADATA_KEY: TEACHER_REVISION.encode(),
    }


def reusable_output(url: str, rows: int, metadata: dict[bytes, bytes]) -> bool:
    """Return true when one complete expanded teacher file already exists."""
    filesystem, path = fsspec.core.url_to_fs(url)
    if not filesystem.exists(path):
        return False
    with pq.ParquetFile(path, filesystem=filesystem) as parquet_file:
        actual_rows = parquet_file.metadata.num_rows
        actual_metadata = parquet_file.schema_arrow.metadata or {}
    if actual_rows != rows:
        raise ValueError(f"Existing teacher output has {actual_rows} rows; expected {rows}: {url}")
    if any(actual_metadata.get(key) != value for key, value in metadata.items()):
        raise ValueError(f"Existing teacher output has different input metadata: {url}")
    return True


def embed_extension(teacher: Any, table: pa.Table, start_row: int, source: str) -> np.ndarray:
    """Embed and quantize the rows after the reused prefix."""
    texts = table["text"].slice(start_row).to_pylist()
    batches = []
    for start in range(0, len(texts), TABLE_BATCH_SIZE):
        batch = texts[start : start + TABLE_BATCH_SIZE]
        batches.append(teacher_batch(teacher, batch, start_row + start))
        logger.info("Teacher embedded %s: %d/%d new rows", source, start + len(batch), len(texts))
    if not batches:
        return np.empty((0, EMBEDDING_DIMENSION), dtype=np.uint8)
    return np.concatenate(batches)


def write_teacher_table(
    url: str,
    expanded: pa.Table,
    quantized: np.ndarray,
    metadata: dict[bytes, bytes],
) -> None:
    """Write one aligned expanded teacher table."""
    if quantized.shape != (len(expanded), EMBEDDING_DIMENSION) or quantized.dtype != np.uint8:
        raise ValueError(f"The expanded teacher has invalid shape or dtype: {quantized.shape}, {quantized.dtype}")
    embedding = pa.FixedSizeListArray.from_arrays(pa.array(quantized.reshape(-1)), EMBEDDING_DIMENSION)
    columns = ["raw_sha256", "split", "eval_rank", "train_rank"]
    columns.extend(name for name in expanded.column_names if name.startswith("in_"))
    output = expanded.select(columns).append_column("embedding", embedding)
    output = output.replace_schema_metadata(metadata)
    filesystem, path = fsspec.core.url_to_fs(url)
    with atomic_rename(path, fs=filesystem) as temporary_path:
        pq.write_table(output, temporary_path, filesystem=filesystem, compression="zstd")


def embed_source(
    teacher: Any,
    expanded_manifest: dict[str, Any],
    base_manifest: dict[str, Any],
    prefix_manifest: dict[str, Any],
    prefix_rung: str | None,
    rung: str,
    source: str,
) -> dict[str, Any]:
    """Reuse the largest completed prefix and embed one source extension."""
    expanded_result = expanded_manifest["sources"][source]
    expected_rows = int(expanded_result["counts"][f"train_{rung}"]) + int(
        expanded_manifest["evaluation_rows_per_source"]
    )
    output_url = f"{teacher_root(rung)}/sources/{Path(expanded_result['output_url']).name}"
    metadata = expected_metadata(expanded_manifest["sha256"], base_manifest["sha256"], rung)
    if reusable_output(output_url, expected_rows, metadata):
        logger.info("Reusing complete expanded teacher output for %s", source)
        return {"output_url": output_url, "rows": expected_rows, "reused": True}

    expanded = selected_expanded_table(expanded_result["output_url"], rung)
    prefix_result = prefix_manifest["sources"][source]
    if prefix_rung is None:
        prefix_source = parquet_table(prefix_result["output_url"])
        prefix_teacher_url = f"{BASE_TEACHER_ROOT}/sources/{Path(prefix_result['output_url']).name}"
    else:
        prefix_source = selected_expanded_table(prefix_result["output_url"], prefix_rung)
        prefix_teacher_url = f"{teacher_root(prefix_rung)}/sources/{Path(prefix_result['output_url']).name}"
    prefix_teacher = parquet_table(prefix_teacher_url)
    validate_teacher_metadata(prefix_teacher, prefix_metadata(prefix_manifest, prefix_rung))
    prefix_quantized = checked_prefix_embeddings(expanded, prefix_source, prefix_teacher)
    extension_quantized = embed_extension(teacher, expanded, len(prefix_source), source)
    quantized = np.concatenate((prefix_quantized, extension_quantized))
    if len(quantized) != expected_rows:
        raise ValueError(f"Source {source} produced {len(quantized)} rows; expected {expected_rows}")
    write_teacher_table(output_url, expanded, quantized, metadata)
    if len(extension_quantized):
        unique_fraction = float(np.unique(extension_quantized, axis=0).shape[0] / len(extension_quantized))
        varying_dimensions = int(np.count_nonzero(extension_quantized.max(axis=0) > extension_quantized.min(axis=0)))
    else:
        unique_fraction = 1.0
        varying_dimensions = 0
    return {
        "output_url": output_url,
        "rows": expected_rows,
        "reused": False,
        "prefix_rung": prefix_rung or "base",
        "reused_prefix_rows": len(prefix_source),
        "embedded_extension_rows": len(extension_quantized),
        "extension_unique_quantized_fraction": unique_fraction,
        "extension_varying_dimensions": varying_dimensions,
    }


def shard_coordinates(arguments: argparse.Namespace) -> tuple[int, int]:
    """Return explicit or Iris-provided shard coordinates."""
    if arguments.shard_index is not None or arguments.num_shards is not None:
        if arguments.shard_index is None or arguments.num_shards is None:
            raise ValueError("Both explicit shard arguments are required")
        return arguments.shard_index, arguments.num_shards
    job_info = get_job_info()
    if job_info is None:
        raise ValueError("Shard arguments are required outside an Iris job")
    return job_info.task_index, job_info.num_tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rung", choices=("10m", "30m"), required=True)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--num-shards", type=int)
    arguments = parser.parse_args()
    shard_index, num_shards = shard_coordinates(arguments)
    expanded_manifest_url = f"{expanded_root(arguments.rung)}/manifest.json"
    expanded_manifest = read_json(expanded_manifest_url)
    base_manifest = read_json(BASE_MANIFEST_URL)
    if expanded_manifest["base_manifest_sha256"] != base_manifest["sha256"]:
        raise ValueError("The expanded manifest has a different base manifest")
    prefix_manifest, prefix_rung = teacher_prefix(arguments.rung, base_manifest)
    teacher = new_teacher()
    reports = {}
    for source in assigned_sources(expanded_manifest, arguments.rung, shard_index, num_shards):
        reports[source] = embed_source(
            teacher,
            expanded_manifest,
            base_manifest,
            prefix_manifest,
            prefix_rung,
            arguments.rung,
            source,
        )
    report = {
        "rung": arguments.rung,
        "teacher_id": TEACHER_ID,
        "teacher_revision": TEACHER_REVISION,
        "expanded_manifest_url": expanded_manifest_url,
        "expanded_manifest_sha256": expanded_manifest["sha256"],
        "base_manifest_url": BASE_MANIFEST_URL,
        "base_manifest_sha256": base_manifest["sha256"],
        "prefix_manifest_sha256": prefix_manifest["sha256"],
        "prefix_rung": prefix_rung or "base",
        "shard_index": shard_index,
        "num_shards": num_shards,
        "source_count": len(reports),
        "row_count": sum(result["rows"] for result in reports.values()),
        "sources": reports,
    }
    report_url = f"{teacher_root(arguments.rung)}/shards/shard-{shard_index:02d}-of-{num_shards:02d}.json"
    write_json(report_url, report)
    summary = {key: report[key] for key in ("rung", "shard_index", "num_shards", "source_count", "row_count")}
    summary["report_url"] = report_url
    RESULT_FILE.with_name(f"{RESULT_FILE.name}-{arguments.rung}-{shard_index:02d}").write_text(
        json.dumps(summary, sort_keys=True)
    )
    logger.info("LUXICAL_EXPANDED_ARCTIC_TEACHER=%s", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
