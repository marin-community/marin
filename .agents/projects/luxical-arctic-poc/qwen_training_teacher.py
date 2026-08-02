# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Create and audit aligned Qwen teacher labels for the 750K student rung."""

import argparse
import hashlib
import json
import logging
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from evaluate_teacher_candidate import (
    ATTENTION_IMPLEMENTATION,
    CANDIDATES,
    INFERENCE_DTYPE,
    MAX_TEACHER_TOKENS,
    SDPA_BACKEND_NAMES,
    TEACHER_QUANTIZATION_LIMIT,
    WINDOWS_PER_DOCUMENT,
    Candidate,
    CandidateEmbedder,
)
from ladder_config import MANIFEST_ROOT, read_json, write_json
from rigging.filesystem import atomic_rename

MANIFEST_URL = f"{MANIFEST_ROOT}/manifest.json"
TEACHER_NAME = "qwen3-embedding-0.6b"
TEACHER_ROOT = f"{MANIFEST_ROOT}/teacher-{TEACHER_NAME}-1024-train-750k-v1"
AUDIT_URL = f"{TEACHER_ROOT}/audit.json"
TABLE_BATCH_SIZE = 512
EXPECTED_ROWS = 750_000
RESULT_FILE = Path("/tmp/luxical-qwen-training-teacher")

MANIFEST_METADATA_KEY = b"luxical_manifest_sha256"
TEACHER_ID_METADATA_KEY = b"luxical_teacher_id"
TEACHER_REVISION_METADATA_KEY = b"luxical_teacher_revision"
TEACHER_SCOPE_METADATA_KEY = b"luxical_teacher_scope"
TEACHER_DIMENSION_METADATA_KEY = b"luxical_teacher_embedding_dimension"
TEACHER_QUANTIZATION_METADATA_KEY = b"luxical_teacher_quantization_limit"
TEACHER_MAX_TOKENS_METADATA_KEY = b"luxical_teacher_max_tokens"
TEACHER_WINDOWS_METADATA_KEY = b"luxical_teacher_windows_per_document"
TEACHER_ATTENTION_METADATA_KEY = b"luxical_teacher_attention_implementation"
TEACHER_DTYPE_METADATA_KEY = b"luxical_teacher_inference_dtype"
TEACHER_SDPA_BACKENDS_METADATA_KEY = b"luxical_teacher_sdpa_backends"
TEACHER_POOLING_METADATA_KEY = b"luxical_teacher_pooling_implementation"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def teacher_metadata(candidate: Candidate, manifest_sha256: str) -> dict[bytes, bytes]:
    """Return metadata that binds labels to the teacher and manifest."""
    return {
        MANIFEST_METADATA_KEY: manifest_sha256.encode(),
        TEACHER_ID_METADATA_KEY: candidate.model_id.encode(),
        TEACHER_REVISION_METADATA_KEY: candidate.revision.encode(),
        TEACHER_SCOPE_METADATA_KEY: b"training-750k",
        TEACHER_DIMENSION_METADATA_KEY: str(candidate.embedding_dimension).encode(),
        TEACHER_QUANTIZATION_METADATA_KEY: str(TEACHER_QUANTIZATION_LIMIT).encode(),
        TEACHER_MAX_TOKENS_METADATA_KEY: str(MAX_TEACHER_TOKENS).encode(),
        TEACHER_WINDOWS_METADATA_KEY: str(WINDOWS_PER_DOCUMENT).encode(),
        TEACHER_ATTENTION_METADATA_KEY: ATTENTION_IMPLEMENTATION.encode(),
        TEACHER_DTYPE_METADATA_KEY: str(INFERENCE_DTYPE).removeprefix("torch.").encode(),
        TEACHER_SDPA_BACKENDS_METADATA_KEY: SDPA_BACKEND_NAMES.encode(),
        TEACHER_POOLING_METADATA_KEY: candidate.pooling.encode(),
    }


def selected_training_table(url: str) -> pa.Table:
    """Read the fixed 750K rows for one source."""
    filesystem, path = fsspec.core.url_to_fs(url)
    table = pq.read_table(
        path,
        filesystem=filesystem,
        columns=["raw_sha256", "train_rank", "split", "in_750k", "text"],
    )
    selected = table.filter(pc.and_(pc.equal(table["split"], "train"), table["in_750k"]))
    return selected.sort_by("train_rank")


def assigned_sources(manifest: dict[str, Any], shard_index: int, num_shards: int) -> list[str]:
    """Assign complete sources to balanced shards."""
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"Shard index {shard_index} is outside [0, {num_shards})")
    totals = [0] * num_shards
    assignments: dict[int, list[str]] = defaultdict(list)
    source_rows = {source: result["counts"]["train_750k"] for source, result in manifest["sources"].items()}
    for source in sorted(source_rows, key=lambda name: (-source_rows[name], name)):
        target = min(range(num_shards), key=lambda index: (totals[index], index))
        assignments[target].append(source)
        totals[target] += source_rows[source]
    logger.info("Qwen shard row totals: %s", totals)
    return sorted(assignments[shard_index])


def teacher_output_url(manifest_output_url: str) -> str:
    """Return the teacher output for one manifest source."""
    return f"{TEACHER_ROOT}/sources/{Path(manifest_output_url).name}"


def complete_output_metrics(
    output_url: str,
    expected_rows: int,
    expected_metadata: dict[bytes, bytes],
) -> dict[str, Any] | None:
    """Return metrics for a valid complete output, or None when it is absent."""
    filesystem, path = fsspec.core.url_to_fs(output_url)
    if not filesystem.exists(path):
        return None
    with pq.ParquetFile(path, filesystem=filesystem) as parquet_file:
        rows = parquet_file.metadata.num_rows
        metadata = parquet_file.schema_arrow.metadata or {}
    if rows != expected_rows:
        raise ValueError(f"Existing teacher output has {rows} rows; expected {expected_rows}: {output_url}")
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise ValueError(f"Existing teacher output has different metadata: {output_url}")
    return {"rows": rows, "reused": True}


def embed_source(
    candidate: Candidate,
    embedder: CandidateEmbedder,
    input_url: str,
    manifest_sha256: str,
) -> tuple[str, dict[str, Any]]:
    """Embed and save one source."""
    input_table = selected_training_table(input_url)
    output_url = teacher_output_url(input_url)
    metadata = teacher_metadata(candidate, manifest_sha256)
    reused = complete_output_metrics(output_url, len(input_table), metadata)
    if reused is not None:
        logger.info("Reusing complete Qwen teacher output: %s", output_url)
        return output_url, reused

    started = time.perf_counter()
    quantized_batches = []
    texts = input_table["text"].to_pylist()
    for start in range(0, len(texts), TABLE_BATCH_SIZE):
        batch = texts[start : start + TABLE_BATCH_SIZE]
        quantized_batches.append(embedder.quantized_documents(batch))
        logger.info("Embedded %d/%d rows for %s", start + len(batch), len(texts), output_url)
    quantized = np.concatenate(quantized_batches)
    if quantized.shape != (len(input_table), candidate.embedding_dimension):
        raise ValueError(f"Teacher returned shape {quantized.shape} for {output_url}")

    embeddings = pa.FixedSizeListArray.from_arrays(pa.array(quantized.ravel()), candidate.embedding_dimension)
    output_table = pa.table(
        {
            "raw_sha256": input_table["raw_sha256"],
            "train_rank": input_table["train_rank"],
            "embedding": embeddings,
        }
    ).replace_schema_metadata(metadata)
    output_filesystem, output_path = fsspec.core.url_to_fs(output_url)
    with tempfile.TemporaryDirectory() as temporary_directory:
        local_path = Path(temporary_directory) / "teacher.parquet"
        pq.write_table(output_table, local_path, compression="zstd", row_group_size=8_192)
        digest = hashlib.sha256(local_path.read_bytes()).hexdigest()
        with atomic_rename(output_path, fs=output_filesystem) as temporary_path:
            output_filesystem.put(str(local_path), temporary_path)
    return output_url, {
        "rows": len(output_table),
        "reused": False,
        "duration_seconds": time.perf_counter() - started,
        "sha256": digest,
        "unique_quantized_rows": int(np.unique(quantized, axis=0).shape[0]),
        "varying_dimensions": int(np.count_nonzero(quantized.max(axis=0) > quantized.min(axis=0))),
    }


def run_shard(shard_index: int, num_shards: int) -> dict[str, Any]:
    """Create all labels for one source shard."""
    manifest = read_json(MANIFEST_URL)
    candidate = CANDIDATES[TEACHER_NAME]
    sources = assigned_sources(manifest, shard_index, num_shards)
    embedder = CandidateEmbedder(candidate)
    source_reports: dict[str, dict[str, Any]] = {}
    for index, source in enumerate(sources, start=1):
        logger.info("Embedding source %d/%d on shard %d/%d: %s", index, len(sources), shard_index, num_shards, source)
        output_url, metrics = embed_source(
            candidate,
            embedder,
            manifest["sources"][source]["output_url"],
            manifest["sha256"],
        )
        source_reports[source] = {"output_url": output_url, "metrics": metrics}
    report = {
        "teacher_id": candidate.model_id,
        "teacher_revision": candidate.revision,
        "teacher_dimension": candidate.embedding_dimension,
        "manifest_url": MANIFEST_URL,
        "manifest_sha256": manifest["sha256"],
        "shard_index": shard_index,
        "num_shards": num_shards,
        "source_count": len(sources),
        "row_count": sum(result["metrics"]["rows"] for result in source_reports.values()),
        "sources": source_reports,
    }
    report_url = f"{TEACHER_ROOT}/shards/shard-{shard_index:02d}-of-{num_shards:02d}.json"
    write_json(report_url, report)
    return {"mode": "shard", "report_url": report_url, **report}


def source_audit(
    candidate: Candidate,
    manifest_sha256: str,
    input_url: str,
    expected_rows: int,
) -> dict[str, Any]:
    """Validate one complete teacher source."""
    input_table = selected_training_table(input_url)
    if len(input_table) != expected_rows:
        raise ValueError(f"Manifest source has {len(input_table)} rows; expected {expected_rows}: {input_url}")
    output_url = teacher_output_url(input_url)
    filesystem, path = fsspec.core.url_to_fs(output_url)
    if not filesystem.exists(path):
        raise FileNotFoundError(f"Missing teacher output: {output_url}")
    with pq.ParquetFile(path, filesystem=filesystem) as parquet_file:
        metadata = parquet_file.schema_arrow.metadata or {}
    if any(metadata.get(key) != value for key, value in teacher_metadata(candidate, manifest_sha256).items()):
        raise ValueError(f"Teacher output has different metadata: {output_url}")
    output_table = pq.read_table(path, filesystem=filesystem)
    if output_table["raw_sha256"].to_pylist() != input_table["raw_sha256"].to_pylist():
        raise ValueError(f"Teacher hashes are not aligned: {output_url}")
    if output_table["train_rank"].to_pylist() != input_table["train_rank"].to_pylist():
        raise ValueError(f"Teacher ranks are not aligned: {output_url}")
    embeddings = output_table["embedding"].combine_chunks()
    if embeddings.type.list_size != candidate.embedding_dimension:
        raise ValueError(f"Teacher output has dimension {embeddings.type.list_size}: {output_url}")
    quantized = embeddings.values.to_numpy(zero_copy_only=False).reshape(expected_rows, candidate.embedding_dimension)
    if quantized.dtype != np.uint8:
        raise ValueError(f"Teacher output has dtype {quantized.dtype}: {output_url}")
    unique_rows = int(np.unique(quantized, axis=0).shape[0])
    varying_dimensions = int(np.count_nonzero(quantized.max(axis=0) > quantized.min(axis=0)))
    if expected_rows > 1 and unique_rows == 1:
        raise ValueError(f"Teacher output is constant: {output_url}")
    if varying_dimensions == 0:
        raise ValueError(f"Teacher output has no varying dimensions: {output_url}")
    return {
        "output_url": output_url,
        "rows": expected_rows,
        "unique_quantized_rows": unique_rows,
        "unique_quantized_fraction": unique_rows / expected_rows,
        "varying_dimensions": varying_dimensions,
    }


def run_audit(num_shards: int) -> dict[str, Any]:
    """Validate shard coverage and all teacher outputs."""
    manifest = read_json(MANIFEST_URL)
    candidate = CANDIDATES[TEACHER_NAME]
    shard_reports = [
        read_json(f"{TEACHER_ROOT}/shards/shard-{index:02d}-of-{num_shards:02d}.json") for index in range(num_shards)
    ]
    for index, report in enumerate(shard_reports):
        if report["shard_index"] != index or report["num_shards"] != num_shards:
            raise ValueError(f"Shard report {index} has different coordinates")
        if report["manifest_sha256"] != manifest["sha256"]:
            raise ValueError(f"Shard report {index} has a different manifest digest")
        if report["teacher_id"] != candidate.model_id or report["teacher_revision"] != candidate.revision:
            raise ValueError(f"Shard report {index} has a different teacher")
    reported_sources = [source for report in shard_reports for source in report["sources"]]
    if len(reported_sources) != len(set(reported_sources)):
        raise ValueError("Shard reports contain duplicate sources")
    if set(reported_sources) != set(manifest["sources"]):
        raise ValueError("Shard reports do not cover the manifest sources")

    sources = {}
    for index, (source, result) in enumerate(sorted(manifest["sources"].items()), start=1):
        logger.info("Auditing Qwen source %d/%d: %s", index, len(manifest["sources"]), source)
        sources[source] = source_audit(
            candidate,
            manifest["sha256"],
            result["output_url"],
            result["counts"]["train_750k"],
        )
    row_count = sum(result["rows"] for result in sources.values())
    if row_count != EXPECTED_ROWS:
        raise ValueError(f"Teacher audit found {row_count} rows; expected {EXPECTED_ROWS}")
    report = {
        "teacher_id": candidate.model_id,
        "teacher_revision": candidate.revision,
        "teacher_dimension": candidate.embedding_dimension,
        "manifest_url": MANIFEST_URL,
        "manifest_sha256": manifest["sha256"],
        "num_shards": num_shards,
        "source_count": len(sources),
        "row_count": row_count,
        "minimum_source_unique_fraction": min(result["unique_quantized_fraction"] for result in sources.values()),
        "minimum_source_varying_dimensions": min(result["varying_dimensions"] for result in sources.values()),
        "sources": sources,
    }
    write_json(AUDIT_URL, report)
    return {"mode": "audit", "audit_url": AUDIT_URL, **report}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("shard", "audit"), required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--shard-index", type=int)
    arguments = parser.parse_args()
    if arguments.mode == "shard" and arguments.shard_index is None:
        parser.error("--shard-index is required in shard mode")
    if arguments.mode == "audit" and arguments.shard_index is not None:
        parser.error("--shard-index is not valid in audit mode")
    return arguments


def main() -> None:
    """Create one shard or audit all completed shards."""
    arguments = parse_args()
    if arguments.num_shards < 1:
        raise ValueError("The shard count must be positive")
    if arguments.mode == "shard":
        result = run_shard(arguments.shard_index, arguments.num_shards)
    else:
        result = run_audit(arguments.num_shards)
    RESULT_FILE.write_text(json.dumps(result, sort_keys=True))
    logger.info("QWEN_TRAINING_TEACHER=%s", json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
