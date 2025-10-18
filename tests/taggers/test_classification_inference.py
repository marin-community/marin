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

# # Copyright 2025 The Marin Authors
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     https://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# import json
# import os

# import fsspec
# import pytest
# import ray

# from marin.core.runtime import TaskConfig
# from marin.processing.classification.config.inference_config import DatasetSchemaConfig
# from marin.processing.classification.config.inference_config import InferenceConfig, RuntimeConfig
# from marin.processing.classification.inference import (
#     read_dataset_streaming,
#     run_inference,
# )
# from marin.utils import fsspec_exists, fsspec_mkdirs
# from marin.processing.classification.autoscaler import DEFAULT_AUTOSCALING_ACTOR_POOL_CONFIG

# DEFAULT_DATASET_SCHEMA = DatasetSchemaConfig(input_columns=["id", "text"], output_columns=["id", "attributes"])


# @pytest.fixture
# def sample_data():
#     """Create sample data for testing"""
#     return [
#         {"id": "doc1", "text": "This is a high quality document with excellent content."},
#         {"id": "doc2", "text": "This is an average document with some useful information."},
#         {"id": "doc3", "text": "This is a poor quality document with minimal content."},
#         {"id": "doc4", "text": "Another excellent document with comprehensive details."},
#         {"id": "doc5", "text": "A mediocre document that could be improved."},
#     ]


# def create_jsonl_gz_file(data: list[dict], filepath: str):
#     """Helper function to create a JSONL.GZ file"""
#     with fsspec.open(filepath, "wt", encoding="utf-8", compression="infer") as f:
#         for item in data:
#             f.write(json.dumps(item) + "\n")


# def read_jsonl_gz_file(filepath: str) -> list[dict]:
#     """Helper function to read a JSONL.GZ file"""
#     results = []
#     with fsspec.open(filepath, "rt", encoding="utf-8", compression="infer") as f:
#         for line in f:
#             if line.strip():
#                 results.append(json.loads(line))
#     return results


# def create_parquet_file(data: list[dict], filepath: str):
#     """Helper function to create a Parquet file"""
#     import pandas as pd

#     df = pd.DataFrame(data)
#     df.to_parquet(filepath, index=False)


# def read_parquet_file(filepath: str) -> list[dict]:
#     """Helper function to read a Parquet file"""
#     import pandas as pd

#     df = pd.read_parquet(filepath)
#     return df.to_dict("records")


# class TestStreamingReading:
#     """Test streaming dataset reading functionality"""

#     @pytest.mark.parametrize(
#         "ext,create_fn",
#         [
#             ("jsonl.gz", create_jsonl_gz_file),
#             ("parquet", create_parquet_file),
#         ],
#     )
#     def test_read_dataset_streaming(self, sample_data, tmpdir, ext, create_fn):
#         """Test streaming reading for both JSONL.GZ and Parquet files"""
#         input_file = os.path.join(tmpdir, f"test_input.{ext}")
#         create_fn(sample_data, input_file)

#         rows = list(read_dataset_streaming(input_file))

#         assert len(rows) == len(sample_data)
#         assert rows[0]["id"] == "doc1"
#         assert rows[0]["text"] == sample_data[0]["text"]

#     @pytest.mark.parametrize(
#         "ext,create_fn",
#         [
#             ("jsonl.gz", create_jsonl_gz_file),
#             ("parquet", create_parquet_file),
#         ],
#     )
#     def test_read_dataset_streaming_with_columns(self, sample_data, tmpdir, ext, create_fn):
#         """Test streaming reading with column selection for both formats"""
#         input_file = os.path.join(tmpdir, f"test_input.{ext}")
#         create_fn(sample_data, input_file)

#         rows = list(read_dataset_streaming(input_file, columns=["id"]))

#         assert len(rows) == len(sample_data)
#         assert "id" in rows[0]
#         assert "text" not in rows[0]


# class TestMultipleFileProcessing:
#     """Test processing of multiple files"""

#     @pytest.mark.parametrize("num_files", [1, 3])
#     @pytest.mark.parametrize(
#         "ext,create_fn,read_fn",
#         [
#             ("jsonl.gz", create_jsonl_gz_file, read_jsonl_gz_file),
#             ("parquet", create_parquet_file, read_parquet_file),
#         ],
#     )
#     def test_process_multiple_files(self, ray_tpu_cluster, sample_data, tmpdir, num_files, ext, create_fn, read_fn):
#         """Test processing multiple files with Ray for both JSONL.GZ and Parquet"""
#         # Create input directory structure
#         input_dir = os.path.join(tmpdir, f"tagger-{ext}-input")
#         output_dir = os.path.join(tmpdir, f"tagger-{ext}-output")
#         fsspec_mkdirs(input_dir)
#         fsspec_mkdirs(output_dir)

#         # Create multiple input files
#         file_data = []
#         for _ in range(num_files):
#             file_data.append(sample_data)

#         input_files = []
#         for i, data in enumerate(file_data):
#             input_file = os.path.join(input_dir, f"file_{i}.{ext}")
#             create_fn(data, input_file)
#             input_files.append(input_file)

#         # Create inference config
#         config = InferenceConfig(
#             input_path=input_dir,
#             output_path=output_dir,
#             model_name="dummy",
#             model_type="dummy",
#             attribute_name="quality",
#             filetype=ext,
#             batch_size=2,
#             resume=True,
#             runtime=RuntimeConfig(memory_limit_gb=1),
#             task=TaskConfig(max_in_flight=2),
#             autoscaling_actor_pool_config=DEFAULT_AUTOSCALING_ACTOR_POOL_CONFIG,
#             dataset_schema=DEFAULT_DATASET_SCHEMA,
#         )

#         # Run inference
#         ray.get(run_inference.remote(config))

#         # Verify all output files were created
#         for i in range(len(file_data)):
#             output_file = os.path.join(output_dir, f"file_{i}.{ext}")
#             assert fsspec_exists(output_file)

#             results = read_fn(output_file)
#             assert len(results) == len(file_data[i])

#             # Check that attributes were added
#             for result in results:
#                 assert "id" in result
#                 assert "attributes" in result
