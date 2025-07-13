import json
import os

import fsspec
import pytest
import ray

from marin.processing.classification.dedupe import DedupeConfig, NGramConfig, dedupe

BASE_INPUT_DIR = "gs://marin-us-east1/documents/test-data/decontamination"
BASE_ATTRIBUTE_OUTPUT_DIR = "gs://marin-us-east1/attributes/test-data/decontamination"
GSM8K_DECONTAMINATION_PATH = "gs://marin-us-east1/decontamination/gsm8k-dolma-9f147d/gsm8k"


@pytest.fixture
def sample_documents():
    """Create sample documents with duplicates for testing"""
    return [
        {
            "id": "gsm8k",
            "text": (
                "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"  # noqa: E501, RUF001
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


@ray.remote
def _run_dedupe(dedupe_config):
    ray.get(dedupe.remote(dedupe_config))


def test_exact_decontamination_paragraph(ray_tpu_cluster, sample_documents):
    output_file_path = os.path.join(BASE_INPUT_DIR, "test_docs.jsonl.gz")
    with fsspec.open(output_file_path, "w", compression="gzip") as f:
        for doc in sample_documents:
            f.write(json.dumps(doc) + "\n")

    dedupe_config = DedupeConfig(
        input_path=BASE_INPUT_DIR,
        output_path=BASE_ATTRIBUTE_OUTPUT_DIR,
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
        decontaminate=True,
        decontaminate_path=GSM8K_DECONTAMINATION_PATH,
    )

    ray.get(_run_dedupe.remote(dedupe_config))

    attribute_file_path = os.path.join(BASE_ATTRIBUTE_OUTPUT_DIR, "test_docs.jsonl.gz")
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
