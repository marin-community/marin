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

import pyarrow as pa
from dupekit import (
    HashAlgorithm,
    Transformation,
    hash_blake2,
    hash_xxh3_128,
    mark_document_duplicates,
    mark_paragraph_duplicates,
    transform,
)


def compute_blake2_hex(text: str) -> str:
    return bytes(hash_blake2(text.encode("utf-8"))).hex()


def compute_xxh3_128_hex(text: str) -> str:
    val = hash_xxh3_128(text.encode("utf-8"))
    return f"{val:032x}"


def test_compute_paragraph_hashes_pipeline():
    """Test generating hash/id pairs from a document batch using the transform pipeline."""
    text1 = "Para 1\nPara 2"
    text2 = "Para 3"
    batch = pa.RecordBatch.from_pydict({"doc_id": ["d1", "d2"], "content": [text1, text2]})

    pipeline = [
        Transformation.SplitParagraphs(text_col="content", id_col="doc_id"),
        Transformation.Hash(input_col="paragraph_text", output_col="hash", algo=HashAlgorithm.Xxh3_128),
        Transformation.SelectColumns(columns=["hash", "doc_id"]),
    ]
    result_batch = transform(batch, pipeline)
    res_dict = result_batch.to_pydict()
    hashes, ids = res_dict["hash"], res_dict["doc_id"]

    expected_paras = ["Para 1", "Para 2", "Para 3"]
    assert len(hashes) == 3
    assert ids == ["d1", "d1", "d2"]
    assert hashes == [compute_xxh3_128_hex(p) for p in expected_paras]


def test_resolve_ids_transformation():
    """Test that the ResolveIds transform imputes missing IDs correctly."""
    text_content = "Some text"
    batch_null_id = pa.RecordBatch.from_pydict({"id": [None], "text": [text_content]})
    batch_missing_col = pa.RecordBatch.from_pydict({"text": [text_content]})
    expected_id = compute_xxh3_128_hex(text_content)

    pipeline = [Transformation.ResolveIds(text_col="text", id_col="id", output_col="resolved_id")]

    # Test with null ID
    res_null = transform(batch_null_id, pipeline).to_pydict()
    assert res_null["resolved_id"][0] == expected_id

    # Test with missing ID column
    res_missing = transform(batch_missing_col, pipeline).to_pydict()
    assert res_missing["resolved_id"][0] == expected_id


def test_mark_paragraph_duplicates():
    """Test marking exact duplicate paragraphs."""
    common_text = "Common paragraph"
    unique_text = "Unique paragraph"
    common_hash = compute_xxh3_128_hex(common_text)
    batch = pa.RecordBatch.from_pydict({"id": ["d1", "d2"], "text": [common_text, f"{common_text}\n{unique_text}"]})
    dup_map = {common_hash: {"canonical": "d1"}}

    # The marker function now takes the original batch and the dup_map
    res_dict = mark_paragraph_duplicates(batch, dup_map, "dups").to_pydict()
    attrs = res_dict["attributes"]

    # d1 has no duplicate spans (it's the canonical version)
    assert attrs[0]["dups"] == []

    # d2 has one duplicate span (the first paragraph)
    d2_attrs = attrs[1]["dups"]
    assert len(d2_attrs) == 1
    assert d2_attrs[0] == [0, len(common_text), 1]


def test_compute_document_hashes_pipeline():
    """Test generating document hashes using the transform pipeline."""
    text1 = "Document One\nWith Newline"
    text2 = "Document Two"
    batch = pa.RecordBatch.from_pydict({"id": ["d1", "d2"], "text": [text1, text2]})

    pipeline = [
        Transformation.Hash(input_col="text", output_col="hash", algo=HashAlgorithm.Xxh3_128),
    ]
    result_batch = transform(batch, pipeline)
    res_dict = result_batch.to_pydict()
    hashes = res_dict["hash"]

    assert len(hashes) == 2
    assert hashes[0] == compute_xxh3_128_hex(text1)
    assert hashes[1] == compute_xxh3_128_hex(text2)


def test_mark_document_duplicates():
    """Test marking exact duplicate documents."""
    text = "Duplicate Content"
    text_hash = compute_xxh3_128_hex(text)

    batch = pa.RecordBatch.from_pydict({"id": ["d1", "d2"], "text": [text, text]})
    dup_map = {text_hash: {"canonical": "d1"}}
    res_dict = mark_document_duplicates(batch, dup_map, "is_duplicate").to_pydict()
    attrs = res_dict["attributes"]
    # Row 0 (d1): Matches canonical -> False
    assert attrs[0]["is_duplicate"] is False
    # Row 1 (d2): Does not match canonical -> True
    assert attrs[1]["is_duplicate"] is True
