# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import pytest
import pyarrow as pa
from typing import Any
import dupekit

# Legacy Python Implementations for baseline


def _str_hash_legacy(s: str) -> str:
    return hashlib.blake2b(s.encode(), digest_size=8).hexdigest()


def python_compute_paragraph_hashes(batch: pa.RecordBatch, text_col: str, id_col: str) -> list[dict[str, str]]:
    results = []
    for record in batch.to_pylist():
        text, record_id = record[text_col], record[id_col]
        for para in text.split("\n"):
            results.append({"hash": _str_hash_legacy(para), "id": record_id})
    return results


def python_compute_document_hashes(batch: pa.RecordBatch, text_col: str, id_col: str) -> list[dict[str, str]]:
    results = []
    for record in batch.to_pylist():
        text, record_id = record[text_col], record[id_col]
        results.append({"hash": _str_hash_legacy(text), "id": record_id})
    return results


def rust_compute_paragraph_hashes(batch: pa.RecordBatch, text_col: str, id_col: str) -> pa.RecordBatch:
    pipeline = [
        dupekit.Transformation.SplitParagraphs(text_col=text_col, id_col=id_col),
        dupekit.Transformation.Hash(input_col="paragraph_text", output_col="hash", algo=dupekit.HashAlgorithm.Xxh3_128),
    ]
    return dupekit.transform(batch, pipeline)


def rust_compute_document_hashes(batch: pa.RecordBatch, text_col: str, id_col: str) -> pa.RecordBatch:
    pipeline = [
        dupekit.Transformation.Hash(input_col=text_col, output_col="hash", algo=dupekit.HashAlgorithm.Xxh3_128),
    ]
    return dupekit.transform(batch, pipeline)


# Benchmarks

PROCESS_FUNCS = {
    ("paragraphs", "python"): python_compute_paragraph_hashes,
    ("paragraphs", "rust"): rust_compute_paragraph_hashes,
    ("documents", "python"): python_compute_document_hashes,
    ("documents", "rust"): rust_compute_document_hashes,
}


@pytest.mark.parametrize("granularity", ["paragraphs", "documents"])
@pytest.mark.parametrize("backend", ["python", "rust"])
def test_hashing(benchmark: Any, sample_batch: pa.RecordBatch, granularity: str, backend: str) -> None:
    """Benchmark the hash generation step."""
    func = PROCESS_FUNCS[(granularity, backend)]
    benchmark.group = f"{granularity.title()}: Hash Generation"
    benchmark(func, sample_batch, "text", "id")
