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

import fsspec
import pytest

from marin.classifiers.utils import create_dataset_shard, reservoir_sample
from marin.utils import fsspec_rm

pytestmark = pytest.mark.tpu_ci

TEST_OUTPUT_PATH = "gs://marin-us-east5/documents/test-sampling.jsonl.gz"


def test_sample_document_matches_keeping_all_examples(test_crawl_file_path: str):
    # Remove the success file if it exists
    fsspec_rm(f"{TEST_OUTPUT_PATH}.SUCCESS")

    num_examples = 0
    with fsspec.open(test_crawl_file_path, "r", compression="gzip") as f:
        for _ in f:
            num_examples += 1

    create_dataset_shard(
        test_crawl_file_path,
        TEST_OUTPUT_PATH,
        label_func=None,
        input_attr_file_paths=[],
        sampling_rate=1.0,
        seed=42,
        columns_to_keep=["text"],
    )

    num_examples_sampled = 0
    with fsspec.open(TEST_OUTPUT_PATH, "r", compression="gzip") as f:
        for _ in f:
            num_examples_sampled += 1

    assert num_examples_sampled == num_examples, f"Got {num_examples_sampled} examples, expected {num_examples}"


def test_reservoir_sample_matches_expected_number_of_examples(test_crawl_file_path: str):
    fsspec_rm(f"{TEST_OUTPUT_PATH}.SUCCESS")

    num_examples = 0
    with fsspec.open(test_crawl_file_path, "r", compression="gzip") as f:
        for _ in f:
            num_examples += 1

    expected_num_examples_sampled = int(num_examples * 0.5)
    reservoir_sample(
        [test_crawl_file_path],
        TEST_OUTPUT_PATH,
        sample_size=expected_num_examples_sampled,
        seed=42,
    )

    num_examples_sampled = 0
    with fsspec.open(TEST_OUTPUT_PATH, "r", compression="gzip") as f:
        for _ in f:
            num_examples_sampled += 1

    assert (
        num_examples_sampled == expected_num_examples_sampled
    ), f"Got {num_examples_sampled} examples, expected {expected_num_examples_sampled}"


def test_multiple_file_reservoir_sample_matches_expected_number_of_examples(test_crawl_file_path: str):
    fsspec_rm(f"{TEST_OUTPUT_PATH}.SUCCESS")

    num_examples = 0
    with fsspec.open(test_crawl_file_path, "r", compression="gzip") as f:
        for _ in f:
            num_examples += 1

    expected_num_examples_sampled = 2 * num_examples
    reservoir_sample(
        [test_crawl_file_path] * 3,
        TEST_OUTPUT_PATH,
        sample_size=expected_num_examples_sampled,
        seed=42,
    )

    num_examples_sampled = 0
    with fsspec.open(TEST_OUTPUT_PATH, "r", compression="gzip") as f:
        for _ in f:
            num_examples_sampled += 1

    assert (
        num_examples_sampled == expected_num_examples_sampled
    ), f"Got {num_examples_sampled} examples, expected {expected_num_examples_sampled}"
