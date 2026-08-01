# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build one source-balanced vocabulary and aligned fast-student arrays."""

import hashlib
import json
import logging
import tempfile
from itertools import chain
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from fast_student import (
    COMPACT_VOCAB_SIZE,
    TOKENIZER_NAME,
    packed_document_ids,
    raw_document_window_ids,
    tokenizer_vocab_size,
)
from ladder_config import MANIFEST_ROOT, TEACHER_ID, TEACHER_REVISION
from rigging.filesystem import atomic_rename

from experiments.datakit.cluster.quality.fast_transformer.embedding import source_balanced_token_remap

MANIFEST_URL = f"{MANIFEST_ROOT}/manifest.json"
TEACHER_ROOT = f"{MANIFEST_ROOT}/teacher-arctic-v1"
TEACHER_AUDIT_URL = f"{TEACHER_ROOT}/audit.json"
OUTPUT_ROOT = f"{MANIFEST_ROOT}/fast-student/prepared-3m"
OUTPUT_MANIFEST_URL = f"{OUTPUT_ROOT}/manifest.json"
REMAP_URL = f"{OUTPUT_ROOT}/raw-to-compact.npy"
TOKENIZE_BATCH_SIZE = 4_096
RESULT_FILE = Path("/tmp/luxical-fast-student-prepare")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def read_json(url: str) -> dict[str, Any]:
    """Read one JSON object from private storage."""
    filesystem, path = fsspec.core.url_to_fs(url)
    with filesystem.open(path) as file:
        return json.load(file)


def selected_training_table(output_url: str, columns: list[str]) -> pa.Table:
    """Read the fixed 3M rows for one source."""
    filesystem, path = fsspec.core.url_to_fs(output_url)
    table = pq.read_table(path, filesystem=filesystem, columns=[*columns, "split", "in_3m"])
    table = table.filter(pc.and_(pc.equal(table["split"], "train"), table["in_3m"]))
    return table.select(columns)


def teacher_output_url(manifest_output_url: str) -> str:
    """Return the teacher file paired with one manifest source file."""
    return f"{TEACHER_ROOT}/sources/{Path(manifest_output_url).name}"


def token_counts(texts: list[str], raw_vocab_size: int) -> np.ndarray:
    """Count raw tokenizer IDs for one source without retaining all token rows."""
    counts = np.zeros(raw_vocab_size, dtype=np.int64)
    for start in range(0, len(texts), TOKENIZE_BATCH_SIZE):
        grouped = raw_document_window_ids(texts[start : start + TOKENIZE_BATCH_SIZE])
        tokens = np.fromiter(
            chain.from_iterable(chain.from_iterable(grouped)),
            dtype=np.int32,
        )
        if len(tokens) == 0:
            continue
        batch_counts = np.bincount(tokens, minlength=raw_vocab_size)
        if len(batch_counts) != raw_vocab_size:
            raise ValueError(f"Tokenizer returned ID {len(batch_counts) - 1} for vocabulary {raw_vocab_size}")
        counts += batch_counts
    return counts


def write_numpy(url: str, values: np.ndarray) -> str:
    """Write a NumPy array atomically and return its SHA-256 digest."""
    filesystem, path = fsspec.core.url_to_fs(url)
    with tempfile.TemporaryDirectory() as temporary_directory:
        local_path = Path(temporary_directory) / "values.npy"
        np.save(local_path, values)
        digest = hashlib.sha256(local_path.read_bytes()).hexdigest()
        with atomic_rename(path, fs=filesystem) as temporary_path:
            filesystem.put(str(local_path), temporary_path)
    return digest


def write_json(url: str, value: dict[str, Any]) -> None:
    """Write one JSON object atomically."""
    filesystem, path = fsspec.core.url_to_fs(url)
    with atomic_rename(path, fs=filesystem) as temporary_path:
        with filesystem.open(temporary_path, "w") as file:
            json.dump(value, file, indent=2, sort_keys=True)


def prepare_source(
    source: str,
    manifest_output_url: str,
    raw_to_compact: np.ndarray,
) -> dict[str, Any]:
    """Write aligned packed IDs and quantized teacher vectors for one source."""
    source_table = selected_training_table(manifest_output_url, ["raw_sha256", "train_rank", "text"])
    teacher_url = teacher_output_url(manifest_output_url)
    teacher_table = selected_training_table(teacher_url, ["raw_sha256", "train_rank", "embedding"])
    if len(source_table) != len(teacher_table):
        raise ValueError(f"Source and teacher row counts differ for {source}")
    if source_table["raw_sha256"].to_pylist() != teacher_table["raw_sha256"].to_pylist():
        raise ValueError(f"Source and teacher rows are not aligned for {source}")
    if source_table["train_rank"].to_pylist() != teacher_table["train_rank"].to_pylist():
        raise ValueError(f"Source and teacher ranks are not aligned for {source}")

    texts = source_table["text"].to_pylist()
    packed_chunks = []
    for start in range(0, len(texts), TOKENIZE_BATCH_SIZE):
        packed_chunks.append(packed_document_ids(texts[start : start + TOKENIZE_BATCH_SIZE], raw_to_compact))
    packed = np.concatenate(packed_chunks)
    ids = pa.FixedSizeListArray.from_arrays(pa.array(packed.reshape(-1)), packed.shape[1])
    output_table = pa.table(
        {
            "raw_sha256": source_table["raw_sha256"],
            "train_rank": source_table["train_rank"],
            "ids": ids,
            "embedding": teacher_table["embedding"],
        }
    )
    output_url = f"{OUTPUT_ROOT}/sources/{Path(manifest_output_url).name}"
    output_filesystem, output_path = fsspec.core.url_to_fs(output_url)
    with tempfile.TemporaryDirectory() as temporary_directory:
        local_path = Path(temporary_directory) / "source.parquet"
        pq.write_table(output_table, local_path, compression="zstd", row_group_size=8_192)
        digest = hashlib.sha256(local_path.read_bytes()).hexdigest()
        with atomic_rename(output_path, fs=output_filesystem) as temporary_path:
            output_filesystem.put(str(local_path), temporary_path)
    return {"output_url": output_url, "rows": len(output_table), "sha256": digest}


def main() -> None:
    """Build the fixed vocabulary and packed 3M training set."""
    manifest = read_json(MANIFEST_URL)
    audit = read_json(TEACHER_AUDIT_URL)
    if audit["manifest_sha256"] != manifest["sha256"]:
        raise ValueError("The teacher audit has a different manifest digest")
    if audit["teacher_id"] != TEACHER_ID or audit["teacher_revision"] != TEACHER_REVISION:
        raise ValueError("The teacher audit does not identify the fixed Arctic Medium teacher")

    raw_vocab_size = tokenizer_vocab_size(TOKENIZER_NAME)
    source_counts = []
    for index, (source, result) in enumerate(sorted(manifest["sources"].items()), start=1):
        table = selected_training_table(result["output_url"], ["text"])
        logger.info("Vocabulary source %d/%d: %s (%d rows)", index, len(manifest["sources"]), source, len(table))
        source_counts.append(token_counts(table["text"].to_pylist(), raw_vocab_size))
    raw_to_compact = source_balanced_token_remap(source_counts, COMPACT_VOCAB_SIZE)
    remap_sha256 = write_numpy(REMAP_URL, raw_to_compact)

    sources = {}
    for index, (source, result) in enumerate(sorted(manifest["sources"].items()), start=1):
        logger.info("Packing source %d/%d: %s", index, len(manifest["sources"]), source)
        sources[source] = prepare_source(source, result["output_url"], raw_to_compact)
    row_count = sum(result["rows"] for result in sources.values())
    if row_count != 3_000_000:
        raise ValueError(f"Prepared {row_count} rows; expected 3000000")
    output = {
        "manifest_url": MANIFEST_URL,
        "manifest_sha256": manifest["sha256"],
        "teacher_audit_url": TEACHER_AUDIT_URL,
        "teacher_id": TEACHER_ID,
        "teacher_revision": TEACHER_REVISION,
        "tokenizer": TOKENIZER_NAME,
        "compact_vocab_size": COMPACT_VOCAB_SIZE,
        "vocabulary_method": "mean within-source token frequency over fixed 3M rows",
        "raw_to_compact_url": REMAP_URL,
        "raw_to_compact_sha256": remap_sha256,
        "rows": row_count,
        "sources": sources,
    }
    write_json(OUTPUT_MANIFEST_URL, output)
    summary = {"output_url": OUTPUT_MANIFEST_URL, "rows": row_count, "sources": len(sources)}
    RESULT_FILE.write_text(json.dumps(summary, sort_keys=True))
    logger.info("FAST_STUDENT_PREPARE=%s", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
