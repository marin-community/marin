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
except ImportError:
    pytest.skip("dolma not installed", allow_module_level=True)

BASE_INPUT_DIR = "gs://marin-us-east1/documents/test-data/deduplication"
BASE_ATTRIBUTE_OUTPUT_DIR = "gs://marin-us-east1/attributes/test-data/deduplication"


@pytest.fixture
def sample_documents():
    """Create sample documents with duplicates for testing"""
    return [
        {"id": "doc1", "text": "This is a unique document that should not be marked as duplicate.", "source": "test"},
        {
            "id": "doc2",
            "text": "This is a duplicate paragraph.\nThis paragraph appears multiple times.",
            "source": "test",
        },
        {
            "id": "doc3",
            "text": "This is a duplicate paragraph.\nThis paragraph appears multiple times.",
            "source": "test",
        },
        {"id": "doc4", "text": "Another unique document with different content entirely.", "source": "test"},
        {
            "id": "doc5",
            "text": "This is a duplicate paragraph.\nThis paragraph appears multiple times.",
            "source": "test",
        },
        {
            "id": "doc6",
            "text": "This is not a duplicate paragraph.\nThis paragraph appears multiple times.",
            "source": "test",
        },
    ]


# @pytest.mark.skip("Seems broken locally, and I dont' want to copy files all the time.")
def test_exact_deduplication_paragraph(ray_tpu_cluster, sample_documents):
    output_file_path = os.path.join(BASE_INPUT_DIR, "deduplication", "test_docs.jsonl.gz")
    with fsspec.open(output_file_path, "w", compression="gzip") as f:
        for doc in sample_documents:
            f.write(json.dumps(doc) + "\n")

    dedupe_config = DedupeConfig(
        input_path=BASE_INPUT_DIR,
        output_path=BASE_ATTRIBUTE_OUTPUT_DIR,
        attribute_name="duplicate_text",
        min_length=0,
        min_words=0,
        estimated_doc_count=100,
        false_positive_rate=0.001,
        ngram=NGramConfig(
            ngram_length=8,
            stride=0,
            overlap_threshold=0.7,
        ),
        mode=DedupMode.DEDUPLICATE,
        num_cpus=2,
        memory=2 * 1024 * 1024 * 1024,
    )

    remote_func = dedupe_with_config_resources(dedupe_config)
    ray.get(remote_func.remote(dedupe_config))

    attribute_file_path = os.path.join(BASE_ATTRIBUTE_OUTPUT_DIR, "deduplication", "test_docs.jsonl.gz")
    with fsspec.open(attribute_file_path, "r", compression="gzip") as f:
        attributes = [json.loads(line) for line in f]

    # Not duplicate
    assert attributes[0]["attributes"]["duplicate_text"] == []

    # First duplicate will not be marked as duplicate
    assert attributes[1]["attributes"]["duplicate_text"] == []

    # Second duplicate will be marked as duplicate
    # Paragraph level deduplication should mark both paragraphs as duplicate
    assert len(attributes[2]["attributes"]["duplicate_text"]) == 2
