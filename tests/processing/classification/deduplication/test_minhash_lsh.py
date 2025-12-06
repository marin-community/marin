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
from marin.processing.classification.deduplication.minhash_lsh import minhash_lsh
from zephyr.dataset import Dataset


def _get_ids_per_bucket(_: str, records: Iterator[dict]) -> set:
    return {record["id"] for record in records}


def test_minhash_lsh_happy_path(sync_backend):
    input_data = [
        {"text": "the quick brown fox", "id": 1},
        {"text": "the quick brown fox jumps", "id": 2},
        {"text": "lorem ipsum dolor sit amet", "id": 3},
        {"text": "the quick brown fox", "id": 4},
    ]

    ds = Dataset.from_list(input_data)

    num_bands = 32
    lsh_result = minhash_lsh(ds, vector_length=128, num_bands=num_bands)

    output = sync_backend.execute(lsh_result)
    assert len(output) == len(input_data) * num_bands

    bucketized = sync_backend.execute(lsh_result.group_by(lambda x: x["bucket"], _get_ids_per_bucket))

    connected_1_and_4 = 0
    connected_1_2 = 0
    connected_1_3 = 0

    for b in bucketized:
        if {1, 4}.issubset(b):
            connected_1_and_4 += 1
        if {1, 2}.issubset(b):
            connected_1_2 += 1
        if {1, 3}.issubset(b):
            connected_1_3 += 1

    assert connected_1_and_4 == num_bands  # docs 1 and 4 are identical, should hash to same bucket in all bands
    assert connected_1_2 > 0  # docs 1 and 2 should hash to the same bucket in some bands
    assert connected_1_3 == 0  # docs 1 and 3 should not collide


def test_minhash_docs(sync_backend, docs):
    input_data = [{"text": text, "id": doc_id} for doc_id, text in docs.items()]

    ds = Dataset.from_list(input_data)

    lsh_result = minhash_lsh(ds, vector_length=128, num_bands=32).group_by(lambda x: x["bucket"], _get_ids_per_bucket)

    output = sync_backend.execute(lsh_result)

    similar_doc_collisions = 0
    different_doc_collisions = 0
    for b in output:
        if {"doc_1", "doc_1_diff_header"} == b:
            similar_doc_collisions += 1
        if {"doc_2"}.issubset(b) and b.intersection({"doc_1", "doc_1_diff_header"}):
            different_doc_collisions += 1

    assert similar_doc_collisions > 0  # doc_1 and doc_1_diff_header should collide in some buckets
    assert different_doc_collisions == 0  # doc_2 should not collide with doc_1 or doc_1_diff_header
