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

from collections.abc import Iterator
import struct
from typing import Any, TypeVar, TypedDict
from dupekit import hash_xxh3_128
from marin.processing.classification.deduplication.minhash import minhash
from marin.processing.classification.deduplication.text_cleaning import clean_text
from zephyr.dataset import Dataset

T = TypeVar("T")


class MinHashLshInputRecord(TypedDict):
    text: str
    id: Any


class MinHashLshOutputRecord(TypedDict):
    bucket: str
    id: Any


# NOTE: defaults are from http://allenai.org/papers/olmo3
# > We then apply a MinHash localitysensitive hashing scheme with 26 bands of size 11,
# > configured to target a Jaccard similarity threshold of 0.80.
def minhash_lsh(
    ds: Dataset[MinHashLshInputRecord], *, vector_length: int = 286, num_bands: int = 26, shingle_size: int = 5
) -> Dataset[MinHashLshOutputRecord]:
    """
    Vanilla MinHashLSH implementation using zephyr Dataset API

    Args:
        ds: Input dataset with records containing 'text' and 'id' fields
        vector_length: Length of the MinHash signature vector
        num_bands: Number of bands to split the MinHash signature into for LSH
        shingle_size: Size of character shingles to extract from text (default: 5)

    Returns:
        A dataset of MinHash LSH output records containing 'bucket' and 'ids' fields
    """

    assert (
        vector_length % num_bands == 0
    ), f"vector_length must be divisible by num_bands, got {vector_length} and {num_bands}"

    return ds.flat_map(lambda record: _minhash_lsh(record, vector_length, num_bands, shingle_size))


def _extract_char_shingles(text: str, shingle_size: int) -> set[str]:
    """
    Extract character shingles from text.

    Args:
        text: Input text to extract shingles from
        shingle_size: Size of each character shingle

    Returns:
        Set of character shingles
    """
    if len(text) < shingle_size:
        # For very short text, return the text itself as a single shingle
        return {text} if text else set()

    return {text[i : i + shingle_size] for i in range(len(text) - shingle_size + 1)}


def _minhash_lsh(
    record: MinHashLshInputRecord, vector_length: int, num_bands: int, shingle_size: int
) -> Iterator[MinHashLshOutputRecord]:
    text = record["text"]
    record_id = record.get("id")
    if record_id is None:
        raise ValueError(f"Record missing required 'id' field: {record}")

    text = clean_text(text)

    # Extract character shingles instead of word unigrams
    shingles = _extract_char_shingles(text, shingle_size)

    sig = minhash(shingles, vector_length=vector_length)

    rows_per_band = len(sig) // num_bands

    for band in range(num_bands):
        start = band * rows_per_band
        end = start + rows_per_band

        bucket = sig[start:end]
        bucket_bytes = struct.pack(f"{len(bucket)}d", *bucket)
        bucket_hash = str(hash_xxh3_128(bucket_bytes))

        yield {"bucket": bucket_hash, "id": record_id}
