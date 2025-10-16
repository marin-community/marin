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

"""
Simple integration test for classification inference pipeline.
This test can be run quickly to validate the basic functionality.
"""

import gzip
import json
import os
import tempfile

from marin.processing.classification.config.inference_config import DatasetSchemaConfig
from marin.processing.classification.inference import (
    process_file_with_quality_classifier_streaming,
    read_dataset_streaming,
)

DEFAULT_DATASET_SCHEMA = DatasetSchemaConfig(input_columns=["id", "text"], output_columns=["id", "attributes"])


def test_single_file_processing():
    """Test processing a single JSONL.GZ file"""

    # Sample test data
    test_data = [
        {"id": "1", "text": "High quality content with detailed information."},
        {"id": "2", "text": "Average content with some useful details."},
        {"id": "3", "text": "Poor quality content with minimal information."},
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create input file
        input_file = os.path.join(temp_dir, "input.jsonl.gz")
        with gzip.open(input_file, "wt", encoding="utf-8") as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")

        # Create output file path
        output_file = os.path.join(temp_dir, "output.jsonl.gz")

        # Process file
        process_file_with_quality_classifier_streaming(
            input_file, output_file, "dummy", "quality", "dummy", {}, DEFAULT_DATASET_SCHEMA, batch_size=2, resume=False
        )

        # Verify output
        assert os.path.exists(output_file)

        # Read and verify results
        results = []
        with gzip.open(output_file, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))

        assert len(results) == len(test_data)

        # Check structure
        for result in results:
            assert "id" in result
            assert "attributes" in result
            assert "dummy-quality" in result["attributes"]
            assert "score" in result["attributes"]["dummy-quality"]


def test_multiple_files_processing():
    """Test processing multiple JSONL.GZ files"""

    # Sample test data split across files
    file_data = [
        [{"id": "1", "text": "Content for file 1 row 1"}, {"id": "2", "text": "Content for file 1 row 2"}],
        [{"id": "3", "text": "Content for file 2 row 1"}],
        [{"id": "4", "text": "Content for file 3 row 1"}, {"id": "5", "text": "Content for file 3 row 2"}],
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
        input_files = []
        output_files = []

        # Create multiple input files
        for i, data in enumerate(file_data):
            input_file = os.path.join(temp_dir, f"input_{i}.jsonl.gz")
            output_file = os.path.join(temp_dir, f"output_{i}.jsonl.gz")

            # Create input file
            with gzip.open(input_file, "wt", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item) + "\n")

            input_files.append(input_file)
            output_files.append(output_file)

        # Process each file
        for input_file, output_file in zip(input_files, output_files, strict=False):
            process_file_with_quality_classifier_streaming(
                input_file,
                output_file,
                "dummy",
                "quality",
                "dummy",
                {},
                DEFAULT_DATASET_SCHEMA,
                batch_size=1,
                resume=False,
            )

        # Verify all outputs
        total_rows = 0
        for _i, (output_file, expected_data) in enumerate(zip(output_files, file_data, strict=False)):
            assert os.path.exists(output_file)

            # Read results
            results = []
            with gzip.open(output_file, "rt", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line))

            assert len(results) == len(expected_data)
            total_rows += len(results)

            # Check structure
            for result in results:
                assert "id" in result
                assert "attributes" in result
                assert "dummy-quality" in result["attributes"]

        # Verify total processed rows
        expected_total = sum(len(data) for data in file_data)
        assert total_rows == expected_total


def test_resumption_functionality():
    """Test that resumption works correctly"""

    test_data = [
        {"id": "1", "text": "First document"},
        {"id": "2", "text": "Second document"},
        {"id": "3", "text": "Third document"},
        {"id": "4", "text": "Fourth document"},
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
        input_file = os.path.join(temp_dir, "input.jsonl.gz")
        output_file = os.path.join(temp_dir, "output.jsonl.gz")

        # Create input file
        with gzip.open(input_file, "wt", encoding="utf-8") as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")

        # Create partial output (first 2 rows)
        partial_results = [
            {"id": "1", "attributes": {"dummy-quality": {"score": 1.0}}},
            {"id": "2", "attributes": {"dummy-quality": {"score": 1.0}}},
        ]
        with gzip.open(output_file, "wt", encoding="utf-8") as f:
            for item in partial_results:
                f.write(json.dumps(item) + "\n")

        # Process with resumption
        process_file_with_quality_classifier_streaming(
            input_file, output_file, "dummy", "quality", "dummy", {}, DEFAULT_DATASET_SCHEMA, batch_size=2, resume=True
        )

        # Verify final output has all rows
        results = []
        with gzip.open(output_file, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))

        assert len(results) == len(test_data)

        # Verify all have attributes
        for result in results:
            assert "id" in result
            assert "attributes" in result


def test_streaming_reading():
    """Test streaming dataset reading"""

    test_data = [
        {"id": "1", "text": "Document 1", "metadata": {"source": "test"}},
        {"id": "2", "text": "Document 2", "metadata": {"source": "test"}},
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
        input_file = os.path.join(temp_dir, "test.jsonl.gz")

        # Create input file
        with gzip.open(input_file, "wt", encoding="utf-8") as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")

        # Test reading all columns
        rows = list(read_dataset_streaming(input_file))
        assert len(rows) == 2
        assert "id" in rows[0]
        assert "text" in rows[0]
        assert "metadata" in rows[0]

        # Test reading specific columns
        rows_filtered = list(read_dataset_streaming(input_file, columns=["id", "text"]))
        assert len(rows_filtered) == 2
        assert "id" in rows_filtered[0]
        assert "text" in rows_filtered[0]
        assert "metadata" not in rows_filtered[0]
