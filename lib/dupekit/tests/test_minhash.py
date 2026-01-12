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
from dupekit import Transformation, transform


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
    pipeline = [Transformation.MinHash(input_col="text", output_col="sig", num_perms=num_perms, ngram_size=3, seed=42)]
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
        Transformation.MinHash(input_col="clean", output_col="sig", num_perms=20, ngram_size=5, seed=1),
        Transformation.MinHashLSH(input_col="sig", output_col="buckets", num_bands=4),
    ]
    res = transform(batch, pipeline)
    b0 = res["buckets"][0].as_py()
    b1 = res["buckets"][1].as_py()
    assert b0 == b1
