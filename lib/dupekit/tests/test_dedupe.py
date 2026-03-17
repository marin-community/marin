# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
from dupekit import (
    HashAlgorithm,
    Transformation,
    hash_xxh3_128,
    transform,
)


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
