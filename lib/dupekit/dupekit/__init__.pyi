# MIT License
#
# Copyright (c) 2023-2025 Kenan Hanke
# Copyright (c) 2023-2025 Zachary Dremann
# Copyright (c) 2023 Rory McNamara
# Copyright (c) 2024 Dan Lenski
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
from enum import Enum
from typing import Any, BinaryIO, final
from collections.abc import Iterable

import pyarrow as pa

@final
class Bloom:
    # expected_items:  max number of items to be added to the filter
    # false_positive_rate:  max false positive rate of the filter
    # Note: This bloom filter expects pre-hashed integers (i128) for add() and __contains__()
    def __init__(self, expected_items: int, false_positive_rate: float) -> None: ...

    # number of buckets in the filter
    @property
    def size_in_bits(self) -> int: ...

    # estimated number of items in the filter
    @property
    def approx_items(self) -> float: ...

    # load from file path or file-like object, see section "Persistence"
    @classmethod
    def load(cls, source: str | bytes | os.PathLike | BinaryIO) -> Bloom: ...

    # load from bytes(), see section "Persistence"
    @classmethod
    def load_bytes(cls, data: bytes) -> Bloom: ...

    # save to file path or file-like object, see section "Persistence"
    def save(self, dest: str | bytes | os.PathLike | BinaryIO) -> None: ...

    # save to a bytes(), see section "Persistence"
    def save_bytes(self) -> bytes: ...

    #####################################################################
    #                    ALL SUBSEQUENT METHODS ARE                     #
    #              EQUIVALENT TO THE CORRESPONDING METHODS              #
    #                     OF THE BUILT-IN SET TYPE                      #
    #      EXCEPT THEY ACCEPT PRE-HASHED INTEGERS (i128) INSTEAD        #
    #####################################################################

    def add(self, hashed: int, /) -> None: ...
    def __contains__(self, hashed: int) -> bool: ...
    def __bool__(self) -> bool: ...  # False if empty
    def __repr__(self) -> str: ...  # basic info
    def __or__(self, other: Bloom) -> Bloom: ...  # self | other
    def __ior__(self, other: Bloom) -> None: ...  # self |= other
    def __and__(self, other: Bloom) -> Bloom: ...  # self & other
    def __iand__(self, other: Bloom) -> None: ...  # self &= other

    # extension of __or__
    def union(self, *others: Iterable[int] | Bloom) -> Bloom: ...

    # extension of __ior__
    def update(self, *others: Iterable[int] | Bloom) -> None: ...

    # extension of __and__
    def intersection(self, *others: Iterable[int] | Bloom) -> Bloom: ...

    # extension of __iand__
    def intersection_update(self, *others: Iterable[int] | Bloom) -> None: ...

    # these implement <, >, <=, >=, ==, !=
    def __lt__(self, other: Bloom) -> bool: ...
    def __gt__(self, other: Bloom) -> bool: ...
    def __le__(self, other: Bloom) -> bool: ...
    def __ge__(self, other: Bloom) -> bool: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def issubset(self, other: Bloom, /) -> bool: ...  # self <= other
    def issuperset(self, other: Bloom, /) -> bool: ...  # self >= other
    def clear(self) -> None: ...  # remove all items
    def copy(self) -> Bloom: ...  # duplicate self

class HashAlgorithm(Enum):
    Blake2b = ...
    Blake3 = ...
    Xxh3_128 = ...
    Xxh3_64 = ...

DEFAULT_HASH_ALGORITHM: HashAlgorithm

@final
class Transformation:
    """Transformation steps for the deduplication pipeline.

    Use these steps to build a list of operations that `transform` will apply
    sequentially to a PyArrow RecordBatch.
    """

    @staticmethod
    def ResolveIds(text_col: str, id_col: str, output_col: str) -> Transformation:
        """Creates a new column `output_col` containing resolved document IDs.

        If `id_col` exists and is not null, it is used. Otherwise, a stable hash
        of `text_col` is generated (using the default hash algorithm) and used as the ID.
        """
        ...

    @staticmethod
    def SplitParagraphs(text_col: str, id_col: str) -> Transformation:
        """Explodes the batch by splitting `text_col` into paragraphs (by newline).

        The resulting batch will have columns: `doc_id`, `paragraph_text`, and `paragraph_span`.
        """
        ...

    @staticmethod
    def Hash(input_col: str, output_col: str, algo: HashAlgorithm) -> Transformation:
        """Computes the hash of `input_col` using the specified `algo` and stores it in `output_col`."""
        ...

    @staticmethod
    def SelectColumns(columns: list[str]) -> Transformation:
        """Projects the batch to keep only the specified columns."""
        ...

    @staticmethod
    def CleanText(input_col: str, output_col: str) -> Transformation:
        """Normalizes text (lowercase, remove punctuation, normalize whitespace)."""
        ...

    @staticmethod
    def MinHash(input_col: str, output_col: str, num_perms: int, ngram_size: int, seed: int) -> Transformation:
        """Computes MinHash signatures for the input text using fused shingling/hashing.

        Args:
            input_col: Column containing text.
            output_col: Column to store signature (List[uint64]).
            num_perms: Number of permutation functions (length of signature).
            ngram_size: Size of char-ngrams for shingling.
            seed: Random seed for permutation coefficients.
        """
        ...

    @staticmethod
    def MinHashLSH(input_col: str, output_col: str, num_bands: int) -> Transformation:
        """Computes LSH buckets from a MinHash signature.

        Args:
            input_col: Column containing signatures (List[uint64]).
            output_col: Column to store buckets (List[uint64]).
            num_bands: Number of LSH bands.
        """
        ...

def transform(batch: pa.RecordBatch, steps: list[Transformation]) -> pa.RecordBatch: ...
def mark_paragraph_duplicates(
    batch: pa.RecordBatch,
    dup_map: dict[str, dict[str, str]],
    attribute_name: str,
    algorithm: HashAlgorithm | None = None,
) -> pa.RecordBatch: ...
def mark_document_duplicates(
    batch: pa.RecordBatch,
    dup_map: dict[str, dict[str, str]],
    attribute_name: str,
    hash_col: str | None = None,
    algorithm: HashAlgorithm | None = None,
) -> pa.RecordBatch: ...
def hash_blake2(text: bytes) -> list[int]: ...
def hash_blake3(text: bytes) -> list[int]: ...
def hash_xxh3_128(text: bytes) -> int: ...
def hash_xxh3_64(text: bytes) -> int: ...
def hash_xxh3_64_batch(texts: list[bytes]) -> list[int]: ...

# Marshaling benchmarks
@final
class Document:
    id: str
    text: str
    def __init__(self, id: str, text: str) -> None: ...

def process_native(path: str) -> pa.RecordBatch: ...
def process_arrow_batch(batch: pa.RecordBatch) -> pa.RecordBatch: ...
def process_rust_structs(docs: list[Document]) -> list[Document]: ...
def process_dicts_batch(docs: list[dict[str, Any]]) -> list[dict[str, Any]]: ...
def process_dicts_loop(doc: dict[str, Any]) -> dict[str, Any]: ...
