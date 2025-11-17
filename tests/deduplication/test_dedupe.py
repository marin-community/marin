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
from marin.processing.classification.dedupe import DedupeConfig, DedupMode, NGramConfig, dedupe
from marin.utils import fsspec_exists


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


def test_decontamination():
    """Test basic decontamination workflow"""
    train_docs = [
        {"id": "train1", "text": "Training data example"},
        {"id": "train2", "text": "Another training sample"},
    ]

    test_docs = [
        {"id": "test1", "text": "Training data example"},  # Contaminated - exact match
        {"id": "test2", "text": "Clean test data here"},  # Clean
        {"id": "test3", "text": "Another training sample"},  # Contaminated - exact match
    ]

    with (
        tempfile.TemporaryDirectory() as train_dir,
        tempfile.TemporaryDirectory() as test_dir,
        tempfile.TemporaryDirectory() as output_dir,
    ):
        # Write training data
        with fsspec.open(os.path.join(train_dir, "train.jsonl.gz"), "w", compression="gzip") as f:
            for doc in train_docs:
                f.write(json.dumps(doc) + "\n")

        # Write test data
        with fsspec.open(os.path.join(test_dir, "test.jsonl.gz"), "w", compression="gzip") as f:
            for doc in test_docs:
                f.write(json.dumps(doc) + "\n")

        # Run decontamination
        config = DedupeConfig(
            input_path=test_dir,
            output_path=output_dir,
            decontaminate_source=train_dir,
            attribute_name="contaminated",
            estimated_doc_count=10,
            false_positive_rate=0.01,
            mode=DedupMode.DECONTAMINATE,
            processes=2,
        )

        result = dedupe(config)
        assert result["success"]
        assert result["mode"] == "decontamination"

        # Read output
        output_files = list(fsspec.open("file://" + output_dir).fs.glob(f"{output_dir}/**/*.jsonl.gz"))
        assert len(output_files) > 0

        results = []
        for output_file in output_files:
            with fsspec.open(output_file, "r", compression="gzip") as f:
                results.extend([json.loads(line) for line in f])

        results.sort(key=lambda x: x["id"])

        # Test1 is contaminated (exact match with train1)
        assert len(results[0]["attributes"]["contaminated"]) == 1
        assert results[0]["attributes"]["contaminated"][0][2] == 1.0

        # Test2 is clean
        assert len(results[1]["attributes"]["contaminated"]) == 1
        assert results[1]["attributes"]["contaminated"][0][2] == 0.0

        # Test3 is contaminated (exact match with train2)
        assert len(results[2]["attributes"]["contaminated"]) == 1
        assert results[2]["attributes"]["contaminated"][0][2] == 1.0


def test_ngram_decontamination():
    """Test n-gram based decontamination"""
    train_docs = [
        {"id": "train1", "text": "The quick brown fox jumps over the lazy dog"},
    ]

    test_docs = [
        {"id": "test1", "text": "The quick brown fox jumps over the lazy cat"},  # High overlap
        {"id": "test2", "text": "Something completely different here"},  # No overlap
    ]

    with (
        tempfile.TemporaryDirectory() as train_dir,
        tempfile.TemporaryDirectory() as test_dir,
        tempfile.TemporaryDirectory() as output_dir,
    ):
        # Write training data
        with fsspec.open(os.path.join(train_dir, "train.jsonl.gz"), "w", compression="gzip") as f:
            for doc in train_docs:
                f.write(json.dumps(doc) + "\n")

        # Write test data
        with fsspec.open(os.path.join(test_dir, "test.jsonl.gz"), "w", compression="gzip") as f:
            for doc in test_docs:
                f.write(json.dumps(doc) + "\n")

        # Run decontamination with n-grams
        config = DedupeConfig(
            input_path=test_dir,
            output_path=output_dir,
            decontaminate_source=train_dir,
            attribute_name="overlap",
            estimated_doc_count=10,
            false_positive_rate=0.01,
            ngram=NGramConfig(ngram_length=3, stride=0, overlap_threshold=0.5),
            mode=DedupMode.DECONTAMINATE,
            processes=1,
        )

        result = dedupe(config)
        assert result["success"]
        assert result["mode"] == "decontamination"

        # Read output
        output_files = list(fsspec.open("file://" + output_dir).fs.glob(f"{output_dir}/**/*.jsonl.gz"))
        results = []
        for output_file in output_files:
            with fsspec.open(output_file, "r", compression="gzip") as f:
                results.extend([json.loads(line) for line in f])

        results.sort(key=lambda x: x["id"])

        # Test1 has high overlap (>50% of 3-grams match)
        assert len(results[0]["attributes"]["overlap"]) == 1
        assert results[0]["attributes"]["overlap"][0][2] > 0.5

        # Test2 has no overlap (score is 0.0)
        assert len(results[1]["attributes"]["overlap"]) == 1
        assert results[1]["attributes"]["overlap"][0][2] == 0.0


def test_train_test_overlap():
    """Test train-test overlap with multiple n-gram sizes"""
    train_docs = [
        {"id": "train1", "text": "The quick brown fox jumps over the lazy dog"},
        {"id": "train2", "text": "Pack my box with five dozen liquor jugs"},
    ]

    test_docs = [
        {"id": "test1", "text": "The quick brown fox jumps over the lazy cat"},  # Partial overlap with train1
        {"id": "test2", "text": "Completely unrelated content here now"},  # No overlap
    ]

    with (
        tempfile.TemporaryDirectory() as train_dir,
        tempfile.TemporaryDirectory() as test_dir,
        tempfile.TemporaryDirectory() as output_dir,
    ):
        # Write training data
        with fsspec.open(os.path.join(train_dir, "train.jsonl.gz"), "w", compression="gzip") as f:
            for doc in train_docs:
                f.write(json.dumps(doc) + "\n")

        # Write test data
        with fsspec.open(os.path.join(test_dir, "test.jsonl.gz"), "w", compression="gzip") as f:
            for doc in test_docs:
                f.write(json.dumps(doc) + "\n")

        # Run train-test overlap with multiple n-gram sizes
        config = DedupeConfig(
            input_path=test_dir,
            output_path=output_dir,
            decontaminate_source=train_dir,
            attribute_name="overlap",
            estimated_doc_count=10,
            false_positive_rate=0.01,
            ngram=NGramConfig(ngram_length=[3, 5], stride=0, overlap_threshold=0.0),  # Show all overlaps
            mode=DedupMode.TRAIN_TEST_OVERLAP,
            processes=1,
        )

        result = dedupe(config)
        assert result["success"]
        assert result["mode"] == "train_test_overlap"
        assert result["ngram_lengths_processed"] == [3, 5]

        # Check outputs for each n-gram size
        for ngram_len in [3, 5]:
            ngram_dir = os.path.join(output_dir, str(ngram_len))
            assert fsspec_exists(ngram_dir)

            output_files = list(fsspec.open("file://" + ngram_dir).fs.glob(f"{ngram_dir}/**/*.jsonl.gz"))
            assert len(output_files) > 0

            results = []
            for output_file in output_files:
                with fsspec.open(output_file, "r", compression="gzip") as f:
                    results.extend([json.loads(line) for line in f])

            results.sort(key=lambda x: x["id"])

            # Test1 should have some overlap
            assert len(results[0]["attributes"][f"overlap_{ngram_len}"]) == 1
            assert results[0]["attributes"][f"overlap_{ngram_len}"][0][2] > 0.0

            # Test2 should have no overlap
            assert len(results[1]["attributes"][f"overlap_{ngram_len}"]) == 1
            assert results[1]["attributes"][f"overlap_{ngram_len}"][0][2] == 0.0


def test_multi_paragraph_decontamination():
    """Test decontamination with multi-paragraph documents"""
    train_docs = [
        {"id": "train1", "text": "First paragraph here.\nSecond paragraph content.\nThird paragraph text."},
    ]

    test_docs = [
        {
            "id": "test1",
            "text": "Different first paragraph.\nSecond paragraph content.\nDifferent third paragraph.",
        },  # Middle paragraph is contaminated
        {"id": "test2", "text": "All clean paragraphs.\nNothing matches.\nCompletely original."},  # All clean
    ]

    with (
        tempfile.TemporaryDirectory() as train_dir,
        tempfile.TemporaryDirectory() as test_dir,
        tempfile.TemporaryDirectory() as output_dir,
    ):
        # Write training data
        with fsspec.open(os.path.join(train_dir, "train.jsonl.gz"), "w", compression="gzip") as f:
            for doc in train_docs:
                f.write(json.dumps(doc) + "\n")

        # Write test data
        with fsspec.open(os.path.join(test_dir, "test.jsonl.gz"), "w", compression="gzip") as f:
            for doc in test_docs:
                f.write(json.dumps(doc) + "\n")

        # Run decontamination (exact paragraph matching)
        config = DedupeConfig(
            input_path=test_dir,
            output_path=output_dir,
            decontaminate_source=train_dir,
            attribute_name="contaminated",
            estimated_doc_count=10,
            false_positive_rate=0.01,
            mode=DedupMode.DECONTAMINATE,
            processes=1,
        )

        result = dedupe(config)
        assert result["success"]
        assert result["mode"] == "decontamination"

        # Read output
        output_files = list(fsspec.open("file://" + output_dir).fs.glob(f"{output_dir}/**/*.jsonl.gz"))
        results = []
        for output_file in output_files:
            with fsspec.open(output_file, "r", compression="gzip") as f:
                results.extend([json.loads(line) for line in f])

        results.sort(key=lambda x: x["id"])

        # Test1: first paragraph clean (0.0), second contaminated (1.0), third clean (0.0)
        assert len(results[0]["attributes"]["contaminated"]) == 3
        assert results[0]["attributes"]["contaminated"][0][2] == 0.0  # First paragraph
        assert results[0]["attributes"]["contaminated"][1][2] == 1.0  # Second paragraph (match!)
        assert results[0]["attributes"]["contaminated"][2][2] == 0.0  # Third paragraph

        # Test2: all paragraphs clean
        assert len(results[1]["attributes"]["contaminated"]) == 3
        assert all(span[2] == 0.0 for span in results[1]["attributes"]["contaminated"])


def test_exact_deduplication_paragraph(sample_documents):
    """Test exact deduplication using n-gram matching"""
    with tempfile.TemporaryDirectory() as temp_input_dir, tempfile.TemporaryDirectory() as temp_attribute_dir:
        # Write sample documents
        input_file_path = os.path.join(temp_input_dir, "test_docs.jsonl.gz")
        with fsspec.open(input_file_path, "w", compression="gzip") as f:
            for doc in sample_documents:
                f.write(json.dumps(doc) + "\n")

        # Run deduplication
        dedupe_config = DedupeConfig(
            input_path=temp_input_dir,
            output_path=temp_attribute_dir,
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
        )

        result = dedupe(dedupe_config)
        assert result["success"]
        assert result["mode"] == "deduplication"

        # Check output
        output_files = fsspec.open("file://" + temp_attribute_dir).fs.glob(f"{temp_attribute_dir}/**/*.jsonl.gz")
        assert len(output_files) > 0

        # Read and verify attributes
        attributes = []
        for output_file in output_files:
            with fsspec.open(output_file, "r", compression="gzip") as f:
                attributes.extend([json.loads(line) for line in f])

        # Sort by id to ensure consistent ordering
        attributes.sort(key=lambda x: x["id"])

        # All documents have duplicate_text annotations (even unique ones)
        assert all("duplicate_text" in attr["attributes"] for attr in attributes)

        # Documents with actually duplicated paragraphs should have multiple spans
        # doc3, doc5 all have the same text as doc2 so they should all be marked
        assert len(attributes[2]["attributes"]["duplicate_text"]) == 2  # doc3
        assert len(attributes[4]["attributes"]["duplicate_text"]) == 2  # doc5
        assert len(attributes[5]["attributes"]["duplicate_text"]) == 1  # doc6, one para differs
