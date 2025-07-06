"""
Major problems with getting dedupe to work currently.

Dolma requires some pretty bad dependencies:
1. tokenizers <=0.19.1 means that no modern transformers can be used hence why we have to use
transformers==4.44.0.
2. s3fs==2023.06 means that a pretty old version of s3fs needs to be used which means
an old fsspec needs to be used. This is a problem because this version will not recognize
the recursive glob pattern **/*.jsonl.gz correctly!
"""

import json
import os

import fsspec
import pytest
import ray

from marin.processing.classification.dedupe import DedupeConfig, NGramConfig, dedupe

BASE_INPUT_DIR = "gs://marin-us-east1/documents/test-data"
BASE_ATTRIBUTE_OUTPUT_DIR = "gs://marin-us-east1/attributes/test-data"


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


def current_runtime_env_with_additional_pip_packages(pip_packages):
    # Get the current runtime environment
    current_runtime_env = ray.get_runtime_context().runtime_env or {}

    # Get the current pip packages
    current_pip_packages = []
    if current_runtime_env.get("pip"):
        if isinstance(current_runtime_env["pip"], dict):
            current_pip_packages = current_runtime_env["pip"].get("packages", [])
        else:
            current_pip_packages = current_runtime_env["pip"]

    all_packages = current_pip_packages + [str(package) for package in pip_packages]

    return {"pip": all_packages}


@ray.remote(
    runtime_env=current_runtime_env_with_additional_pip_packages(
        ["dolma@git+https://github.com/allenai/dolma.git", "transformers==4.44.0"]
    )
)
def _run_dedupe(dedupe_config):
    ray.get(dedupe.remote(dedupe_config))


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
    )

    ray.get(_run_dedupe.remote(dedupe_config))

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
