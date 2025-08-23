import json
import os

import fsspec
import pytest
import ray

from marin.core.runtime import TaskConfig
from marin.datashop.dataset_processor import FinalScoreZeroToFiveDatasetOutputProcessor
from marin.processing.classification.classifier import vLLMClassifier
from marin.processing.classification.config.inference_config import InferenceConfig, RuntimeConfig
from marin.processing.classification.inference import (
    process_file_ray,
    run_inference,
)
from marin.utils import fsspec_exists, fsspec_mkdirs
from tests.conftest import default_engine_kwargs, default_generation_params


@pytest.fixture
def sample_data():
    """Create sample data for testing vLLM classification"""
    return [
        {
            "id": "doc1",
            "text": (
                "This is a high quality document with excellent content and comprehensive information "
                "that would be very useful for educational purposes."
            ),
        },
        {"id": "doc2", "text": "This is an average document with some useful information but lacks depth and detail."},
        {
            "id": "doc3",
            "text": "This is a poor quality document with minimal content and very little useful information.",
        },
        {
            "id": "doc4",
            "text": (
                "Another excellent document with comprehensive details, well-structured content, "
                "and high educational value."
            ),
        },
        {"id": "doc5", "text": "A mediocre document that could be improved with more detail and better organization."},
    ]


@pytest.fixture
def gs_bucket_prefix():
    return "gs://marin-us-east1/tests"


@pytest.fixture
def quality_classification_template():
    """Template for quality classification"""
    return """Please rate the quality of the following text on a scale from 0 to 5, where:
- 0: Very poor quality, useless content
- 1: Poor quality, minimal useful content
- 2: Below average quality, some issues
- 3: Average quality, acceptable content
- 4: Good quality, well-written content
- 5: Excellent quality, outstanding content

Text: {example}

Please provide your rating as "Final score: X" where X is a number from 0 to 5."""


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


class TestVLLMClassifierDirect:
    """Test vLLM classifier directly"""

    def test_vllm_classifier_initialization(self, gcsfuse_mount_model_path, quality_classification_template):
        """Test that vLLM classifier can be initialized"""
        classifier = vLLMClassifier(
            model_name=gcsfuse_mount_model_path,
            attribute_name="quality",
            template=quality_classification_template,
            score_extractor_fn=FinalScoreZeroToFiveDatasetOutputProcessor.extract_score,
            engine_kwargs=default_engine_kwargs,
            generation_kwargs=default_generation_params,
            save_original_generation=True,
        )

        assert classifier.model_name == gcsfuse_mount_model_path
        assert classifier.attribute_name == "quality"
        assert classifier.save_original_generation is True

    def test_vllm_classifier_batch_processing(
        self, gcsfuse_mount_model_path, quality_classification_template, sample_data
    ):
        """Test vLLM classifier on a batch of documents"""
        classifier = vLLMClassifier(
            model_name=gcsfuse_mount_model_path,
            attribute_name="quality",
            template=quality_classification_template,
            score_extractor_fn=FinalScoreZeroToFiveDatasetOutputProcessor.extract_score,
            engine_kwargs=default_engine_kwargs,
            generation_kwargs=default_generation_params,
            save_original_generation=True,
        )

        # Prepare batch
        batch = {"id": [doc["id"] for doc in sample_data], "text": [doc["text"] for doc in sample_data]}

        # Process batch
        result = classifier(batch)

        # Verify results
        assert "attributes" in result
        assert len(result["attributes"]) == len(sample_data)

        # Check that each document has quality attributes
        for _i, attr in enumerate(result["attributes"]):
            assert "quality" in attr
            assert "score" in attr["quality"]
            score = attr["quality"]["score"]
            assert isinstance(score, int)
            assert 0 <= score <= 5

        # Check that original generations are saved
        assert "generated_text" in result
        assert len(result["generated_text"]) == len(sample_data)

        # Verify that generated text contains score information
        for generated_text in result["generated_text"]:
            assert isinstance(generated_text, str)
            assert len(generated_text) > 0


class TestVLLMSingleFileProcessing:
    """Test vLLM classification on single files using process_file_ray"""

    def test_process_single_file_with_vllm(
        self, ray_tpu_cluster, gcsfuse_mount_model_path, quality_classification_template, sample_data, gs_bucket_prefix
    ):
        """Test processing a single file with vLLM classifier using process_file_ray"""
        # Create input and output paths in cloud storage
        input_file = os.path.join(gs_bucket_prefix, "vllm-single-input", "test_input.jsonl.gz")
        output_file = os.path.join(gs_bucket_prefix, "vllm-single-output", "test_output.jsonl.gz")

        # Ensure directories exist
        fsspec_mkdirs(os.path.dirname(input_file))
        fsspec_mkdirs(os.path.dirname(output_file))

        # Create input file
        create_jsonl_gz_file(sample_data, input_file)

        # Process file using Ray
        ray.get(
            process_file_ray.options(resources={"TPU": 1}).remote(
                input_file,
                output_file,
                gcsfuse_mount_model_path,
                "quality",
                "vllm",
                "jsonl.gz",
                {
                    "template": quality_classification_template,
                    "score_extractor_fn": None,
                    "engine_kwargs": default_engine_kwargs,
                    "generation_kwargs": default_generation_params,
                    "save_original_generation": True,
                },
                batch_size=5,
                resume=False,
            )
        )

        # Verify output file was created
        assert fsspec_exists(output_file)

        # Read and verify results
        results = read_jsonl_gz_file(output_file)
        assert len(results) == len(sample_data)

        # Check that attributes were added correctly
        for _i, result in enumerate(results):
            # Check that original generation is saved
            assert "generated_text" in result

    def test_process_single_file_with_resumption(
        self, ray_tpu_cluster, gcsfuse_mount_model_path, quality_classification_template, sample_data, gs_bucket_prefix
    ):
        """Test resumption capability with vLLM classifier"""
        input_file = os.path.join(gs_bucket_prefix, "vllm-resume-input", "test_input.jsonl.gz")
        output_file = os.path.join(gs_bucket_prefix, "vllm-resume-output", "test_output.jsonl.gz")

        # Ensure directories exist
        fsspec_mkdirs(os.path.dirname(input_file))
        fsspec_mkdirs(os.path.dirname(output_file))

        # Create input file
        create_jsonl_gz_file(sample_data, input_file)

        # Create partial output file (first 2 rows)
        partial_output = [
            {"id": "doc1", "attributes": {"quality": {"score": 4, "generated_text": "Final score: 4"}}},
            {"id": "doc2", "attributes": {"quality": {"score": 3, "generated_text": "Final score: 3"}}},
        ]
        create_jsonl_gz_file(partial_output, output_file)

        # Process file with resumption
        ray.get(
            process_file_ray.options(resources={"TPU": 1}).remote(
                input_file,
                output_file,
                gcsfuse_mount_model_path,
                "quality",
                "vllm",
                "jsonl.gz",
                {
                    "template": quality_classification_template,
                    "score_extractor_fn": None,
                    "engine_kwargs": default_engine_kwargs,
                    "generation_kwargs": default_generation_params,
                    "save_original_generation": True,
                },
                batch_size=2,
                resume=True,
            )
        )

        # Verify output contains all rows
        results = read_jsonl_gz_file(output_file)
        assert len(results) == len(sample_data)

        # Check that all rows have attributes
        for result in results:
            assert "generated_text" in result


class TestVLLMFullInferencePipeline:
    """Test vLLM classification using the full run_inference pipeline"""

    def test_full_inference_pipeline_multiple_files(
        self, ray_tpu_cluster, gcsfuse_mount_model_path, quality_classification_template, sample_data, gs_bucket_prefix
    ):
        """Test the full inference pipeline with multiple files"""
        # Create input and output directories
        input_dir = os.path.join(gs_bucket_prefix, "vllm-pipeline-input")
        output_dir = os.path.join(gs_bucket_prefix, "vllm-pipeline-output")
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

        # Create inference config for vLLM
        config = InferenceConfig(
            input_path=input_dir,
            output_path=output_dir,
            model_name=gcsfuse_mount_model_path,
            model_type="vllm",
            attribute_name="quality",
            filetype="jsonl.gz",
            batch_size=2,
            resume=True,
            runtime=RuntimeConfig(memory_limit_gb=2, resources={"TPU": 1}),
            task=TaskConfig(max_in_flight=1),  # Keep low for test stability
            classifier_kwargs={
                "template": quality_classification_template,
                "score_extractor_fn": FinalScoreZeroToFiveDatasetOutputProcessor.extract_score,
                "engine_kwargs": default_engine_kwargs,
                "generation_kwargs": default_generation_params,
                "save_original_generation": True,
            },
        )

        # Run full inference pipeline
        ray.get(run_inference.remote(config))

        # Verify all output files were created
        for i in range(len(file_data)):
            output_file = os.path.join(output_dir, f"file_{i}.jsonl.gz")
            assert fsspec_exists(output_file)

            results = read_jsonl_gz_file(output_file)
            assert len(results) == len(file_data[i])

            # Check that attributes were added correctly
            for _j, result in enumerate(results):
                assert "id" in result
                assert "attributes" in result
                assert "quality" in result["attributes"]
                assert "score" in result["attributes"]["quality"]

                score = result["attributes"]["quality"]["score"]
                assert isinstance(score, int)
                assert 0 <= score <= 5

                # Check that original generation is saved
                assert "generated_text" in result["attributes"]["quality"]
                assert isinstance(result["attributes"]["quality"]["generated_text"], str)

    def test_full_inference_pipeline_with_different_templates(
        self, ray_tpu_cluster, gcsfuse_mount_model_path, sample_data, gs_bucket_prefix
    ):
        """Test the full inference pipeline with different classification templates"""
        # Create a different template for educational value
        educational_template = """Rate the educational value of this text on a scale from 0 to 5:
- 0: No educational value
- 1: Minimal educational value
- 2: Some educational value
- 3: Moderate educational value
- 4: High educational value
- 5: Exceptional educational value

Text: {examples}

Provide your rating as "Final score: X" where X is 0-5."""

        # Create input and output directories
        input_dir = os.path.join(gs_bucket_prefix, "vllm-edu-input")
        output_dir = os.path.join(gs_bucket_prefix, "vllm-edu-output")
        fsspec_mkdirs(input_dir)
        fsspec_mkdirs(output_dir)

        # Create input file
        input_file = os.path.join(input_dir, "educational_test.jsonl.gz")
        create_jsonl_gz_file(sample_data, input_file)

        # Create inference config for educational value classification
        config = InferenceConfig(
            input_path=input_dir,
            output_path=output_dir,
            model_name=gcsfuse_mount_model_path,
            model_type="vllm",
            attribute_name="educational_value",
            filetype="jsonl.gz",
            batch_size=3,
            resume=False,
            runtime=RuntimeConfig(memory_limit_gb=2, resources={"TPU": 1}),
            task=TaskConfig(max_in_flight=1),
            classifier_kwargs={
                "template": educational_template,
                "score_extractor_fn": None,
                "engine_kwargs": default_engine_kwargs,
                "generation_kwargs": default_generation_params,
                "save_original_generation": True,  # Test without saving original generation
            },
        )

        # Run inference
        ray.get(run_inference.remote(config))

        # Verify output
        output_file = os.path.join(output_dir, "educational_test.jsonl.gz")
        assert fsspec_exists(output_file)

        results = read_jsonl_gz_file(output_file)
        assert len(results) == len(sample_data)

        # Check results
        for result in results:
            assert "id" in result
            assert "attributes" in result
            assert "educational_value" in result["attributes"]
            assert "score" in result["attributes"]["educational_value"]

            score = result["attributes"]["educational_value"]["score"]
            assert isinstance(score, int)
            assert 0 <= score <= 5

            # Check that original generation is NOT saved (since save_original_generation=False)
            assert "generated_text" not in result["attributes"]["educational_value"]


class TestVLLMErrorHandling:
    """Test error handling scenarios for vLLM classification"""

    def test_invalid_model_path(self, quality_classification_template):
        """Test handling of invalid model paths"""
        with pytest.raises((FileNotFoundError, OSError, ValueError, RuntimeError)):
            classifier = vLLMClassifier(
                model_name="invalid/model/path",
                attribute_name="quality",
                template=quality_classification_template,
                engine_kwargs={"enforce_eager": True, "max_model_len": 512},
            )
            # Try to process a small batch to trigger the error
            batch = {"id": ["test"], "text": ["test text"]}
            classifier(batch)

    def test_empty_batch_processing(self, gcsfuse_mount_model_path, quality_classification_template):
        """Test processing of empty batches"""
        classifier = vLLMClassifier(
            model_name=gcsfuse_mount_model_path,
            attribute_name="quality",
            template=quality_classification_template,
            score_extractor_fn=FinalScoreZeroToFiveDatasetOutputProcessor.extract_score,
            engine_kwargs=default_engine_kwargs,
            generation_kwargs=default_generation_params,
        )

        # Process empty batch
        batch = {"id": [], "text": []}
        result = classifier(batch)

        # Should return empty attributes
        assert "attributes" in result
        assert len(result["attributes"]) == 0


if __name__ == "__main__":
    pytest.main([__file__])
