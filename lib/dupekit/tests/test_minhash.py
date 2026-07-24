# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import random

import pyarrow as pa
from dupekit import NgramKind, Transformation, transform


def _signatures(texts: list[str], ngram_kind: NgramKind) -> list[list[int]]:
    batch = pa.RecordBatch.from_pydict({"text": texts})
    pipeline = [
        Transformation.CleanText(input_col="text", output_col="clean"),
        Transformation.MinHash(
            input_col="clean",
            output_col="signature",
            num_perms=286,
            ngram_size=5,
            ngram_kind=ngram_kind,
            seed=42,
        ),
    ]
    return transform(batch, pipeline)["signature"].to_pylist()


def _estimated_jaccard(a: list[int], b: list[int]) -> float:
    return sum(left == right for left, right in zip(a, b, strict=True)) / len(a)


def test_clean_text():
    """Test text cleaning (lowercase, punct removal, whitespace norm)."""
    text = "Hello,   World! This is a test."
    expected = "hello world this is a test"
    batch = pa.RecordBatch.from_pydict({"text": [text, None, "   "]})
    pipeline = [Transformation.CleanText(input_col="text", output_col="clean")]
    clean = transform(batch, pipeline)["clean"]
    assert clean[0].as_py() == expected
    assert clean[1].as_py() is None
    assert clean[2].as_py() == ""


def test_minhash_dimensions():
    """Test that MinHash output has correct dimensions."""
    texts = ["doc one", "doc two"]
    num_perms = 128
    batch = pa.RecordBatch.from_pydict({"text": texts})
    pipeline = [
        Transformation.MinHash(
            input_col="text",
            output_col="sig",
            num_perms=num_perms,
            ngram_size=3,
            ngram_kind=NgramKind.Char,
            seed=42,
        )
    ]
    sigs = transform(batch, pipeline)["sig"]
    for sig in sigs:
        assert len(sig.as_py()) == num_perms
        assert all(isinstance(x, int) for x in sig.as_py())


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
        Transformation.MinHash(
            input_col="clean",
            output_col="sig",
            num_perms=20,
            ngram_size=5,
            ngram_kind=NgramKind.Char,
            seed=1,
        ),
        Transformation.MinHashLSH(input_col="sig", output_col="buckets", num_bands=4),
    ]
    res = transform(batch, pipeline)
    b0 = res["buckets"][0].as_py()
    b1 = res["buckets"][1].as_py()
    assert b0 == b1


def test_word_ngrams_distinguish_long_documents_with_shared_vocabulary():
    """Word shingles distinguish sequences that saturate character shingles."""
    vocabulary = (
        "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec "
        "romeo sierra tango uniform victor whiskey xray yankee zulu amber birch cedar dune ember frost grove harbor "
        "ivory jasmine knoll lagoon meadow nectar olive prairie quartz ridge solar timber umber valley willow xenon "
        "yarrow zenith"
    ).split()
    texts = []
    for seed in (1, 2):
        rng = random.Random(seed)
        texts.append(" ".join(rng.choice(vocabulary) for _ in range(4_000)))

    char_a, char_b = _signatures(texts, NgramKind.Char)
    word_a, word_b = _signatures(texts, NgramKind.Word)

    assert _estimated_jaccard(char_a, char_b) > 0.75
    assert _estimated_jaccard(word_a, word_b) < 0.05
