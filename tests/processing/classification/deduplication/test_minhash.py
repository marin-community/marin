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

from marin.processing.classification.deduplication.vendor.datasketch.minhash import MinHash, _max_hash
import pytest


def minhash(shingle_set):
    m = MinHash(num_perm=128)
    for shingle in shingle_set:
        m.update(shingle.encode("utf8"))
    return m.hashvalues.tolist()


def test_minhash_exact_equal():
    assert minhash({"foo"}) == minhash({"foo"})


def test_minhash_different():
    assert minhash({"foo"}) != minhash({"bar"})


def test_minhash_empty():
    assert minhash(set()) == [_max_hash] * 128


def test_minhash_similar():
    set1 = {"the", "quick", "brown", "fox"}
    set2 = {"the", "quick", "brown", "fox", "jumps"}
    hash1, hash2 = minhash(set1), minhash(set2)
    assert hash1 != hash2
    # Since the sets are similar, their minhashes should also be similar
    similarity = sum(1 for a, b in zip(hash1, hash2, strict=True) if a == b) / len(hash1)
    assert similarity > 0.7


def test_jaccard_similarity():
    set1 = {"apple", "banana", "cherry"}
    set2 = {"banana", "cherry", "date"}
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    jaccard_sim = intersection / union
    assert jaccard_sim == 0.5
    hash1, hash2 = minhash(set1), minhash(set2)
    estimated_similarity = sum(1 for a, b in zip(hash1, hash2, strict=True) if a == b) / len(hash1)
    # The estimated similarity should be close to the actual Jaccard similarity
    assert pytest.approx(estimated_similarity, rel=0.3) == jaccard_sim
