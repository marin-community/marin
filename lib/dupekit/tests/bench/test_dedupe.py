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

import hashlib
import pytest
import pyarrow as pa
from typing import Any
import dupekit


def build_map(hashes: list[str], ids: list[str]) -> dict[str, Any]:
    """
    Builds a duplicate map where ~50% of items are marked as canonical and 50% as duplicates.
    """
    dup_map = {}
    for i, (h, rec_id) in enumerate(zip(hashes, ids, strict=True)):
        # Every other element is treated as a duplicate of "other_canonical_id"
        canon = rec_id if i % 2 != 0 else "other_canonical_id"
        dup_map[h] = {"canonical": canon}
    return dup_map


def extract_results(result: Any) -> tuple[list[str], list[str]]:
    if isinstance(result, (pa.RecordBatch, pa.Table)):
        id_col = "doc_id" if "doc_id" in result.schema.names else "id"
        return result["hash"].to_pylist(), result[id_col].to_pylist()
    return [x["hash"] for x in result], [x["id"] for x in result]


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


def python_mark_exact_dups_paragraphs(
    batch: pa.RecordBatch,
    text_col: str,
    id_col: str,
    dup_map: dict[str, Any],
    attribute_name: str,
) -> list[dict[str, Any]]:
    results = []
    for record in batch.to_pylist():
        text, record_id = record[text_col], record[id_col]
        spans = []
        offset = 0
        for para in text.split("\n"):
            h = _str_hash_legacy(para)
            if h in dup_map and dup_map[h]["canonical"] != record_id:
                spans.append([offset, offset + len(para), 1.0])
            offset += len(para) + 1
        results.append({"id": record_id, "attributes": {attribute_name: spans}})
    return results


def python_compute_document_hashes(batch: pa.RecordBatch, text_col: str, id_col: str) -> list[dict[str, str]]:
    results = []
    for record in batch.to_pylist():
        text, record_id = record[text_col], record[id_col]
        results.append({"hash": _str_hash_legacy(text), "id": record_id})
    return results


def python_mark_exact_dups_documents(
    batch: pa.RecordBatch,
    text_col: str,
    id_col: str,
    dup_map: dict[str, Any],
    attribute_name: str,
) -> list[dict[str, Any]]:
    results = []
    for record in batch.to_pylist():
        text, record_id = record[text_col], record[id_col]
        h = _str_hash_legacy(text)
        is_dup = h in dup_map and dup_map[h]["canonical"] != record_id
        results.append({"id": record_id, "attributes": {attribute_name: is_dup}})
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


def rust_mark_paragraph_duplicates(
    batch: pa.RecordBatch, text_col: str, id_col: str, dup_map: dict, attribute_name: str
) -> pa.RecordBatch:
    # The mark function only needs the batch, map, and attr name.
    # text_col and id_col are ignored but kept for signature compatibility with the benchmark.
    return dupekit.mark_paragraph_duplicates(batch, dup_map, attribute_name)


def rust_mark_document_duplicates(
    batch: pa.RecordBatch, text_col: str, id_col: str, dup_map: dict, attribute_name: str
) -> pa.RecordBatch:
    # The mark function only needs the batch, map, and attr name.
    # text_col and id_col are ignored but kept for signature compatibility with the benchmark.
    return dupekit.mark_document_duplicates(batch, dup_map, attribute_name)


# Benchmarks

PROCESS_FUNCS = {
    ("paragraphs", "python"): python_compute_paragraph_hashes,
    ("paragraphs", "rust"): rust_compute_paragraph_hashes,
    ("documents", "python"): python_compute_document_hashes,
    ("documents", "rust"): rust_compute_document_hashes,
}

MARK_FUNCS = {
    ("paragraphs", "python"): python_mark_exact_dups_paragraphs,
    ("paragraphs", "rust"): rust_mark_paragraph_duplicates,
    ("documents", "python"): python_mark_exact_dups_documents,
    ("documents", "rust"): rust_mark_document_duplicates,
}


@pytest.mark.parametrize("granularity", ["paragraphs", "documents"])
@pytest.mark.parametrize("backend", ["python", "rust"])
def test_hashing(benchmark: Any, sample_batch: pa.RecordBatch, granularity: str, backend: str) -> None:
    """Benchmark the hash generation step."""
    func = PROCESS_FUNCS[(granularity, backend)]
    benchmark.group = f"{granularity.title()}: Hash Generation"
    benchmark(func, sample_batch, "text", "id")


@pytest.mark.parametrize("granularity", ["paragraphs", "documents"])
@pytest.mark.parametrize("backend", ["python", "rust"])
def test_deduplication(benchmark: Any, sample_batch: pa.RecordBatch, granularity: str, backend: str) -> None:
    """Benchmark the duplicate marking step (requires pre-calculated map)."""
    process_func = PROCESS_FUNCS[(granularity, "rust")]  # Always use fast Rust version to build map
    processed = process_func(sample_batch, "text", "id")
    hashes, ids = extract_results(processed)
    dup_map = build_map(hashes, ids)

    mark_func = MARK_FUNCS[(granularity, backend)]
    benchmark.group = f"{granularity.title()}: Exact Deduplication"

    benchmark(mark_func, sample_batch, "text", "id", dup_map, "dups")
