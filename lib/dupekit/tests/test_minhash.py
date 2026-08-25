# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pytest
from dupekit import Transformation, transform


def test_clean_text():
    """Test text cleaning (lowercase, punct removal, whitespace norm)."""
    batch = pa.RecordBatch.from_pydict(
        {"text": ["Hello,   World! This is a test.", None, "   ", "  NAÏVE\u2003CAFÉ...  ", "ΟΣ"]}
    )
    pipeline = [Transformation.CleanText(input_col="text", output_col="clean")]
    clean = transform(batch, pipeline)["clean"]
    assert clean[0].as_py() == "hello world this is a test"
    assert clean[1].as_py() is None
    assert clean[2].as_py() == ""
    assert clean[3].as_py() == "naïve café"
    assert clean[4].as_py() == "ος"


def test_minhash_dimensions():
    """Test that MinHash output has correct dimensions."""
    texts = ["doc one", "doc two"]
    num_perms = 128
    batch = pa.RecordBatch.from_pydict({"text": texts})
    pipeline = [Transformation.MinHash(input_col="text", output_col="sig", num_perms=num_perms, ngram_size=3, seed=42)]
    sigs = transform(batch, pipeline)["sig"]
    for sig in sigs:
        assert len(sig.as_py()) == num_perms
        assert all(isinstance(x, int) for x in sig.as_py())


def test_minhash_preserves_ascii_and_unicode_signatures():
    batch = pa.RecordBatch.from_pydict({"text": ["ascii text for minhash", "naïve café text", "tiny", ""]})
    pipeline = [Transformation.MinHash(input_col="text", output_col="sig", num_perms=4, ngram_size=5, seed=1)]

    assert transform(batch, pipeline)["sig"].to_pylist() == [
        [141685336939587008, 1496994699323061486, 1651319965729644324, 683409266644494111],
        [796668610022495371, 1478408615390012735, 2970420477520494696, 3003677083264146706],
        [16072872466350250770, 14657179075576985902, 12503415097090635938, 8969363072275430507],
        [2476109827587840581, 5300362173287906356, 8828609975635180869, 801036783346226840],
    ]


def test_minhash_rejects_zero_ngram_size():
    batch = pa.RecordBatch.from_pydict({"text": ["some text"]})
    pipeline = [Transformation.MinHash(input_col="text", output_col="sig", num_perms=4, ngram_size=0, seed=1)]

    with pytest.raises(ValueError, match="ngram_size must be positive"):
        transform(batch, pipeline)


def test_minhash_lsh_dimensions():
    """Test LSH banding logic."""
    num_bands = 26
    sig = list(range(286))
    batch = pa.RecordBatch.from_pydict({"sig": [sig]}, schema=pa.schema([("sig", pa.list_(pa.uint64()))]))
    pipeline = [Transformation.MinHashLSH(input_col="sig", output_col="buckets", num_bands=num_bands)]
    res = transform(batch, pipeline)
    buckets = res["buckets"][0].as_py()
    assert len(buckets) == num_bands
    res2 = transform(batch, pipeline)
    assert buckets == res2["buckets"][0].as_py()


def test_full_pipeline_determinism():
    """Test that the full MinHash pipeline produces deterministic results."""
    text = "The quick brown fox jumps over the lazy dog."
    batch = pa.RecordBatch.from_pydict({"text": [text, text]})
    pipeline = [
        Transformation.CleanText(input_col="text", output_col="clean"),
        Transformation.MinHash(input_col="clean", output_col="sig", num_perms=20, ngram_size=5, seed=1),
        Transformation.MinHashLSH(input_col="sig", output_col="buckets", num_bands=4),
    ]
    res = transform(batch, pipeline)
    b0 = res["buckets"][0].as_py()
    b1 = res["buckets"][1].as_py()
    assert b0 == b1
