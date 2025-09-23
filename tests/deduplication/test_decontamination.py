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

import json
import os
import tempfile

import fsspec
import pytest
import ray

try:
    import dolma  # noqa: F401

    from marin.processing.classification.dedupe import (
        DedupeConfig,
        DedupMode,
        NGramConfig,
        dedupe_with_config_resources,
    )
    from marin.utils import fsspec_exists
except ImportError:
    pytest.skip("dolma not installed", allow_module_level=True)


@pytest.fixture
def sample_documents():
    """Create sample documents for testing decontamination"""
    return [
        {
            "id": "gsm8k",
            "text": (
                "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"  # noqa: E501
            ),
            "source": "test",
        },
        {
            "id": "gsm8k-middle",
            "text": (
                "Some random words in the paragraph and then going to put the sample in the middle of the paragraph. \nJames decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?\nSome more words over here"  # noqa: E501
            ),
            "source": "test",
        },
        {"id": "doc4", "text": "Some unique text.", "source": "test"},
    ]


@pytest.fixture
def test_set_documents():
    """Set of documents to decontaminate against."""
    return [
        {
            "id": "gsm8k",
            "text": (
                "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"  # noqa: E501
            ),
            "source": "test",
        },
        {
            "id": "gsm8k-middle",
            "text": (
                "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?"  # noqa: E501
            ),
            "source": "test",
        },
    ]


def test_exact_decontamination_paragraph(ray_tpu_cluster, sample_documents, test_set_documents):
    with (
        tempfile.TemporaryDirectory() as temp_input_dir,
        tempfile.TemporaryDirectory() as temp_attribute_dir,
        tempfile.TemporaryDirectory() as temp_decontamination_dir,
        tempfile.TemporaryDirectory() as temp_bloom_filter_dir,
        tempfile.TemporaryDirectory() as temp_dedupe_dir,
    ):
        input_file_path = os.path.join(temp_input_dir, "test_docs.jsonl.gz")
        with fsspec.open(input_file_path, "w", compression="gzip") as f:
            for doc in sample_documents:
                f.write(json.dumps(doc) + "\n")

        test_set_file_path = os.path.join(temp_decontamination_dir, "test_set.jsonl.gz")
        with fsspec.open(test_set_file_path, "w", compression="gzip") as f:
            for doc in test_set_documents:
                f.write(json.dumps(doc) + "\n")

        dedupe_config = DedupeConfig(
            input_path=temp_input_dir,
            output_path=temp_attribute_dir,
            attribute_name="duplicate_text",
            min_length=0,
            min_words=0,
            estimated_doc_count=10000,
            false_positive_rate=0.001,
            ngram=NGramConfig(
                ngram_length=8,
                stride=0,
                overlap_threshold=0.0,  # For debugging purposes, this outputs all possible duplicates
            ),
            mode=DedupMode.DECONTAMINATE,
            decontaminate_source=temp_decontamination_dir,
            bloom_filter_path=os.path.join(temp_bloom_filter_dir, "deduper_bloom_filter.bin"),
            temp_dir=temp_dedupe_dir,
            num_cpus=2,
            memory=2 * 1024 * 1024 * 1024,
        )

        remote_func = dedupe_with_config_resources(dedupe_config)
        ray.get(remote_func.remote(dedupe_config))

        attribute_file_path = os.path.join(temp_attribute_dir, "test_docs.jsonl.gz")
        with fsspec.open(attribute_file_path, "r", compression="gzip") as f:
            attributes = [json.loads(line) for line in f]

        # Duplicates
        assert len(attributes[0]["attributes"]["duplicate_text"]) == 1
        # The amount of overlap is 100%
        assert attributes[0]["attributes"]["duplicate_text"][0][2] == 1

        # Only middle paragraph is a duplicate
        assert attributes[1]["attributes"]["duplicate_text"][0][2] == 0
        assert attributes[1]["attributes"]["duplicate_text"][1][2] == 1
        assert attributes[1]["attributes"]["duplicate_text"][2][2] == 0

        # Not a duplicate at all
        assert len(attributes[2]["attributes"]["duplicate_text"]) == 1
        assert attributes[2]["attributes"]["duplicate_text"][0][2] == 0

        assert fsspec_exists(os.path.join(temp_bloom_filter_dir, "deduper_bloom_filter.bin"))
