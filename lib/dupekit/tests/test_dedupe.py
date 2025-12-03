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
    process_batch_paragraphs,
    mark_exact_dups_paragraphs,
    process_batch_documents,
    mark_exact_dups_documents,
    hash_blake2,
    hash_xxh3_128,
    HashAlgorithm,
)


def compute_blake2_hex(text: str) -> str:
    return bytes(hash_blake2(text.encode("utf-8"))).hex()


def compute_xxh3_128_hex(text: str) -> str:
    val = hash_xxh3_128(text.encode("utf-8"))
    return f"{val:032x}"


def test_process_batch_paragraphs():
    """Test generating hash/id pairs from document batch using default algorithm (Xxh3_128)."""
    text1 = "Para 1\nPara 2"
    text2 = "Para 3"

    batch = pa.RecordBatch.from_pydict({"doc_id": ["d1", "d2"], "content": [text1, text2]})
    res_dict = process_batch_paragraphs(batch, "content", "doc_id").to_pydict()
    hashes, ids = res_dict["hash"], res_dict["id"]

    expected_paras = ["Para 1", "Para 2", "Para 3"]
    assert len(hashes) == 3
    assert ids == ["d1", "d1", "d2"]
    assert hashes == [compute_xxh3_128_hex(p) for p in expected_paras]


def test_process_batch_paragraphs_algorithm_selection():
    """Test using a specific algorithm (Blake2b)."""
    text = "Hello world"
    batch = pa.RecordBatch.from_pydict({"doc_id": ["d1"], "content": [text]})

    res_dict = process_batch_paragraphs(batch, "content", "doc_id", algorithm=HashAlgorithm.Blake2b).to_pydict()
    assert res_dict["hash"][0] == compute_blake2_hex(text)


def test_process_batch_missing_id():
    """Test that ID is imputed from text hash when ID column is missing or null."""
    text_content = "Some text"
    batch_null = pa.RecordBatch.from_pydict({"doc_id": [None], "content": [text_content]})

    res_dict = process_batch_paragraphs(batch_null, "content", "doc_id").to_pydict()
    expected_id = compute_xxh3_128_hex(text_content)
    assert res_dict["id"][0] == expected_id

    batch_missing = pa.RecordBatch.from_pydict({"content": [text_content]})
    res_dict = process_batch_paragraphs(batch_missing, "content", "non_existent_id_col").to_pydict()
    assert res_dict["id"][0] == expected_id


def test_mark_exact_dups_paragraphs():
    """Test marking exact duplicates."""
    common_text = "Common paragraph"
    unique_text = "Unique paragraph"

    common_hash = compute_xxh3_128_hex(common_text)
    batch = pa.RecordBatch.from_pydict({"doc_id": ["d1", "d2"], "text": [common_text, f"{common_text}\n{unique_text}"]})
    dup_map = {common_hash: {"canonical": "d1"}}

    res_dict = mark_exact_dups_paragraphs(batch, "text", "doc_id", dup_map, "dups").to_pydict()
    attrs = res_dict["attributes"]

    assert attrs[0]["dups"] == []
    d2_attrs = attrs[1]["dups"]
    assert len(d2_attrs) == 1
    assert d2_attrs[0] == [0, len(common_text), 1]


def test_mark_exact_dups_missing_id():
    """Test duplicate marking handles missing IDs by imputing them correctly."""
    common_text = "Common"
    common_hash = compute_xxh3_128_hex(common_text)

    imputed_id = compute_xxh3_128_hex(common_text)
    batch = pa.RecordBatch.from_pydict({"text": [common_text]})
    dup_map_canon = {common_hash: {"canonical": imputed_id}}

    result = mark_exact_dups_paragraphs(batch, "text", "missing_id", dup_map_canon, "dups")
    attrs = result.to_pydict()["attributes"]
    assert attrs[0]["dups"] == []

    dup_map_other = {common_hash: {"canonical": "other_id"}}
    result_other = mark_exact_dups_paragraphs(batch, "text", "missing_id", dup_map_other, "dups")
    attrs_other = result_other.to_pydict()["attributes"]
    assert len(attrs_other[0]["dups"]) == 1


def test_process_batch_documents():
    """Test generating hash/id pairs for whole documents."""
    text1 = "Document One\nWith Newline"
    text2 = "Document Two"

    batch = pa.RecordBatch.from_pydict({"doc_id": ["d1", "d2"], "content": [text1, text2]})
    res_dict = process_batch_documents(batch, "content", "doc_id").to_pydict()
    hashes, ids = res_dict["hash"], res_dict["id"]

    assert len(hashes) == 2
    assert ids == ["d1", "d2"]

    assert hashes[0] == compute_xxh3_128_hex(text1)
    assert hashes[1] == compute_xxh3_128_hex(text2)


def test_mark_exact_dups_documents():
    """Test marking exact duplicate documents."""
    text = "Duplicate Content"
    text_hash = compute_xxh3_128_hex(text)

    batch = pa.RecordBatch.from_pydict({"doc_id": ["d1", "d2"], "text": [text, text]})
    dup_map = {text_hash: {"canonical": "d1"}}
    res_dict = mark_exact_dups_documents(batch, "text", "doc_id", dup_map, "is_duplicate").to_pydict()
    attrs = res_dict["attributes"]
    # Row 0 (d1): Matches canonical -> False
    assert attrs[0]["is_duplicate"] is False
    # Row 1 (d2): Does not match canonical -> True
    assert attrs[1]["is_duplicate"] is True
