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

from marin.core.runtime import TaskConfig
from marin.processing.classification.config.inference_config import DatasetSchemaConfig
from marin.processing.classification.config.inference_config import InferenceConfig, RuntimeConfig
from marin.processing.classification.inference import (
    process_file_with_quality_classifier_streaming,
    read_dataset_streaming,
    run_inference,
)
from marin.utils import fsspec_exists, fsspec_mkdirs

DEFAULT_DATASET_SCHEMA = DatasetSchemaConfig(input_columns=["id", "text"], output_columns=["id", "attributes"])


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    return [
        {"id": "doc1", "text": "This is a high quality document with excellent content."},
        {"id": "doc2", "text": "This is an average document with some useful information."},
        {"id": "doc3", "text": "This is a poor quality document with minimal content."},
        {"id": "doc4", "text": "Another excellent document with comprehensive details."},
        {"id": "doc5", "text": "A mediocre document that could be improved."},
    ]


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def create_jsonl_gz_file(data: list[dict], filepath: str):
    """Helper function to create a JSONL.GZ file"""
    with fsspec.open(filepath, "wt", encoding="utf-8", compression="infer") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def read_jsonl_gz_file(filepath: str) -> list[dict]:
    """Helper function to read a JSONL.GZ file"""
    results = []
    with fsspec.open(filepath, "rt", encoding="utf-8", compression="infer") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def create_parquet_file(data: list[dict], filepath: str):
    """Helper function to create a Parquet file"""
    import pandas as pd

    df = pd.DataFrame(data)
    df.to_parquet(filepath, index=False)


def read_parquet_file(filepath: str) -> list[dict]:
    """Helper function to read a Parquet file"""
    import pandas as pd

    df = pd.read_parquet(filepath)
    return df.to_dict("records")


class TestStreamingReading:
    """Test streaming dataset reading functionality"""

    def test_read_dataset_streaming_jsonl_gz(self, sample_data, temp_dir):
        """Test streaming reading of JSONL.GZ files"""
        input_file = os.path.join(temp_dir, "test_input.jsonl.gz")
        create_jsonl_gz_file(sample_data, input_file)

        # Read streaming
        rows = list(read_dataset_streaming(input_file))

        assert len(rows) == len(sample_data)
        assert rows[0]["id"] == "doc1"
        assert rows[0]["text"] == sample_data[0]["text"]

    def test_read_dataset_streaming_with_columns(self, sample_data, temp_dir):
        """Test streaming reading with column selection"""
        input_file = os.path.join(temp_dir, "test_input.jsonl.gz")
        create_jsonl_gz_file(sample_data, input_file)

        # Read only specific columns
        rows = list(read_dataset_streaming(input_file, columns=["id"]))

        assert len(rows) == len(sample_data)
        assert "id" in rows[0]
        assert "text" not in rows[0]

    def test_read_dataset_streaming_parquet(self, sample_data, temp_dir):
        """Test streaming reading of Parquet files"""
        input_file = os.path.join(temp_dir, "test_input.parquet")
        create_parquet_file(sample_data, input_file)

        # Read streaming
        rows = list(read_dataset_streaming(input_file))

        assert len(rows) == len(sample_data)
        assert rows[0]["id"] == "doc1"
        assert rows[0]["text"] == sample_data[0]["text"]

    def test_read_dataset_streaming_parquet_with_columns(self, sample_data, temp_dir):
        """Test streaming reading of Parquet files with column selection"""
        input_file = os.path.join(temp_dir, "test_input.parquet")
        create_parquet_file(sample_data, input_file)

        # Read only specific columns
        rows = list(read_dataset_streaming(input_file, columns=["id"]))

        assert len(rows) == len(sample_data)
        assert "id" in rows[0]
        assert "text" not in rows[0]


class TestSingleFileProcessing:
    """Test processing of single files"""

    def test_process_single_file_streaming(self, sample_data, temp_dir):
        """Test streaming processing of a single file"""
        input_file = os.path.join(temp_dir, "test_input.jsonl.gz")
        output_file = os.path.join(temp_dir, "test_output.jsonl.gz")

        # Create input file
        create_jsonl_gz_file(sample_data, input_file)

        # Create dummy classifier
        # classifier = DummyClassifier("dummy", "quality")

        # Process file
        process_file_with_quality_classifier_streaming(
            input_file, output_file, "dummy", "quality", "dummy", {}, DEFAULT_DATASET_SCHEMA, batch_size=2, resume=False
        )

        # Verify output
        results = read_jsonl_gz_file(output_file)
        assert len(results) == len(sample_data)

        # Check that attributes were added
        for result in results:
            assert "id" in result
            assert "attributes" in result
            assert "dummy-quality" in result["attributes"]

    def test_process_single_file_with_resumption(self, sample_data, temp_dir):
        """Test resumption capability"""
        input_file = os.path.join(temp_dir, "test_input.jsonl.gz")
        output_file = os.path.join(temp_dir, "test_output.jsonl.gz")

        # Create input file
        create_jsonl_gz_file(sample_data, input_file)

        # Create partial output file (first 2 rows)
        partial_output = [
            {"id": "doc1", "attributes": {"dummy-quality": {"score": 1.0}}},
            {"id": "doc2", "attributes": {"dummy-quality": {"score": 1.0}}},
        ]
        create_jsonl_gz_file(partial_output, output_file)

        # Create dummy classifier
        # classifier = DummyClassifier("dummy", "quality")

        # Process file with resumption
        process_file_with_quality_classifier_streaming(
            input_file, output_file, "dummy", "quality", "dummy", {}, DEFAULT_DATASET_SCHEMA, batch_size=2, resume=True
        )

        # Verify output contains all rows
        results = read_jsonl_gz_file(output_file)
        assert len(results) == len(sample_data)

        # Check that all rows have attributes
        for result in results:
            assert "id" in result
            assert "attributes" in result

    def test_process_single_file_different_batch_sizes(self, sample_data, temp_dir):
        """Test processing with different batch sizes"""
        input_file = os.path.join(temp_dir, "test_input.jsonl.gz")

        # Create input file
        create_jsonl_gz_file(sample_data, input_file)

        # Create dummy classifier
        # classifier = DummyClassifier("dummy", "quality")

        # Test different batch sizes
        for batch_size in [1, 2, 3, 10]:
            output_file = os.path.join(temp_dir, f"test_output_batch_{batch_size}.jsonl.gz")

            process_file_with_quality_classifier_streaming(
                input_file,
                output_file,
                "dummy",
                "quality",
                "dummy",
                {},
                DEFAULT_DATASET_SCHEMA,
                batch_size=batch_size,
                resume=False,
            )

            results = read_jsonl_gz_file(output_file)
            assert len(results) == len(sample_data)

    def test_process_single_parquet_file_streaming(self, sample_data, temp_dir):
        """Test streaming processing of a single Parquet file"""
        input_file = os.path.join(temp_dir, "test_input.parquet")
        output_file = os.path.join(temp_dir, "test_output.parquet")

        # Create input Parquet file
        create_parquet_file(sample_data, input_file)

        # Create dummy classifier
        # classifier = DummyClassifier("dummy", "quality")

        # Process file
        process_file_with_quality_classifier_streaming(
            input_file, output_file, "dummy", "quality", "dummy", {}, DEFAULT_DATASET_SCHEMA, batch_size=2, resume=False
        )

        # Verify output
        results = read_parquet_file(output_file)
        assert len(results) == len(sample_data)

        # Check that attributes were added
        for result in results:
            assert "id" in result
            assert "attributes" in result
            assert "dummy-quality" in result["attributes"]

    def test_process_parquet_file_with_resumption(self, sample_data, temp_dir):
        """Test resumption capability with Parquet files"""
        input_file = os.path.join(temp_dir, "test_input.parquet")
        output_file = os.path.join(temp_dir, "test_output.parquet")

        # Create input Parquet file
        create_parquet_file(sample_data, input_file)

        # Create partial output file (first 2 rows)
        partial_output = [
            {"id": "doc1", "attributes": {"dummy-quality": {"score": 1.0}}},
            {"id": "doc2", "attributes": {"dummy-quality": {"score": 1.0}}},
        ]
        create_parquet_file(partial_output, output_file)

        # Create dummy classifier
        # classifier = DummyClassifier("dummy", "quality")

        # Process file with resumption
        process_file_with_quality_classifier_streaming(
            input_file, output_file, "dummy", "quality", "dummy", {}, DEFAULT_DATASET_SCHEMA, batch_size=2, resume=True
        )

        # Verify output contains all rows
        results = read_parquet_file(output_file)
        assert len(results) == len(sample_data)

        # Check that all rows have attributes
        for result in results:
            assert "id" in result
            assert "attributes" in result


class TestMultipleFileProcessing:
    """Test processing of multiple files"""

    def test_process_multiple_files(self, ray_tpu_cluster, sample_data, temp_dir):
        """Test processing multiple files with Ray"""
        # Create input directory structure

        input_dir = os.path.join(temp_dir, "tagger-input")
        output_dir = os.path.join(temp_dir, "tagger-output")
        fsspec_mkdirs(input_dir)
        fsspec_mkdirs(output_dir)

        # Create multiple input files
        file_data = [
            sample_data[:2],  # First 2 rows
            sample_data[2:4],  # Next 2 rows
            sample_data[4:],  # Last row
        ]

        input_files = []
        for i, data in enumerate(file_data):
            input_file = os.path.join(input_dir, f"file_{i}.jsonl.gz")
            create_jsonl_gz_file(data, input_file)
            input_files.append(input_file)

        # Create inference config
        config = InferenceConfig(
            input_path=input_dir,
            output_path=output_dir,
            model_name="dummy",
            model_type="dummy",
            attribute_name="quality",
            filetype="jsonl.gz",
            batch_size=2,
            resume=True,
            runtime=RuntimeConfig(memory_limit_gb=1),
            task=TaskConfig(max_in_flight=2),
            classifier_kwargs={},
            dataset_schema=DEFAULT_DATASET_SCHEMA,
        )

        # Run inference
        ray.get(run_inference.remote(config))

        # Verify all output files were created
        for i in range(len(file_data)):
            output_file = os.path.join(output_dir, f"file_{i}.jsonl.gz")
            assert fsspec_exists(output_file)

            results = read_jsonl_gz_file(output_file)
            assert len(results) == len(file_data[i])

            # Check that attributes were added
            for result in results:
                assert "id" in result
                assert "attributes" in result

    def test_process_multiple_parquet_files(self, ray_tpu_cluster, sample_data, temp_dir):
        """Test processing multiple Parquet files with Ray"""
        # Create input directory structure
        input_dir = os.path.join(temp_dir, "tagger-parquet-input")
        output_dir = os.path.join(temp_dir, "tagger-parquet-output")
        fsspec_mkdirs(input_dir)
        fsspec_mkdirs(output_dir)

        # Create multiple input Parquet files
        file_data = [
            sample_data[:2],  # First 2 rows
            sample_data[2:4],  # Next 2 rows
            sample_data[4:],  # Last row
        ]

        input_files = []
        for i, data in enumerate(file_data):
            input_file = os.path.join(input_dir, f"file_{i}.parquet")
            create_parquet_file(data, input_file)
            input_files.append(input_file)

        # Create inference config for Parquet files
        config = InferenceConfig(
            input_path=input_dir,
            output_path=output_dir,
            model_name="dummy",
            model_type="dummy",
            attribute_name="quality",
            filetype="parquet",
            batch_size=2,
            resume=True,
            runtime=RuntimeConfig(memory_limit_gb=1),
            task=TaskConfig(max_in_flight=2),
            classifier_kwargs={},
            dataset_schema=DEFAULT_DATASET_SCHEMA,
        )

        # Run inference
        ray.get(run_inference.remote(config))

        # Verify all output files were created
        for i in range(len(file_data)):
            output_file = os.path.join(output_dir, f"file_{i}.parquet")
            assert fsspec_exists(output_file)

            results = read_parquet_file(output_file)
            assert len(results) == len(file_data[i])

            # Check that attributes were added
            for result in results:
                assert "id" in result
                assert "attributes" in result


class TestInferenceConfig:
    """Test inference configuration"""

    def test_inference_config_defaults(self):
        """Test default values in InferenceConfig"""
        config = InferenceConfig(input_path="/test/input", model_name="test_model", attribute_name="test_attr")

        assert config.batch_size == 512
        assert config.resume is True
        assert config.filetype == "jsonl.gz"
        assert config.model_type is None
        assert config.runtime.memory_limit_gb == 0.1

    def test_inference_config_custom_values(self):
        """Test custom values in InferenceConfig"""
        config = InferenceConfig(
            input_path="/test/input",
            model_name="test_model",
            attribute_name="test_attr",
            batch_size=256,
            resume=False,
            filetype="jsonl.zst",
            model_type="vllm",
        )

        assert config.batch_size == 256
        assert config.resume is False
        assert config.filetype == "jsonl.zst"
        assert config.model_type == "vllm"


class TestErrorHandling:
    """Test error handling scenarios"""

    def test_unsupported_file_format(self, temp_dir):
        """Test handling of unsupported file formats"""
        input_file = os.path.join(temp_dir, "test.txt")

        # Create a text file
        with open(input_file, "w") as f:
            f.write("test content")

        # Should raise ValueError for unsupported format
        with pytest.raises(ValueError, match="Unsupported filetype"):
            list(read_dataset_streaming(input_file))

    def test_empty_input_file(self, temp_dir):
        """Test handling of empty input files"""
        input_file = os.path.join(temp_dir, "empty.jsonl.gz")
        output_file = os.path.join(temp_dir, "empty_output.jsonl.gz")

        # Create empty file
        create_jsonl_gz_file([], input_file)

        # Create dummy classifier
        # classifier = DummyClassifier("dummy", "quality")

        # Process empty file
        process_file_with_quality_classifier_streaming(
            input_file, output_file, "dummy", "quality", "dummy", {}, DEFAULT_DATASET_SCHEMA, batch_size=2, resume=False
        )

        # Should create empty output file
        assert not os.path.exists(output_file)
