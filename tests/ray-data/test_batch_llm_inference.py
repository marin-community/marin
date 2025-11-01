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

import os

import pytest
import ray

from marin.generation.inference import OverwriteOutputFiletypeFilenameProvider
from marin.generation.pipeline import vLLMTextGeneration

TEST_OUTPUT_PATH = "gs://marin-us-east5/documents/ray-data-test-llama-200m"


@pytest.mark.tpu_ci
def test_ray_data(gcsfuse_mount_model_path, test_file_path):
    ds = ray.data.read_json(test_file_path, arrow_open_stream_args={"compression": "gzip"})

    ds = ds.map_batches(  # Apply batch inference for all input data.
        vLLMTextGeneration,
        # Set the concurrency to the number of LLM instances.
        concurrency=1,
        # Specify the batch size for inference.
        batch_size=16,
        fn_constructor_kwargs={
            "model_name": gcsfuse_mount_model_path,
            "engine_kwargs": {"enforce_eager": True, "max_model_len": 1024},
            "generation_kwargs": {"max_tokens": 16},
            "template": "What is this text about? {example}",
            "prompt_column": "text",
            "save_templated_prompt": False,
            "apply_chat_template": True,
            "max_doc_tokens": 896,
        },
        resources={"TPU": 1, "TPU-v6e-8-head": 1},
    )

    ds = ds.write_json(TEST_OUTPUT_PATH, filename_provider=OverwriteOutputFiletypeFilenameProvider("jsonl.gz"))
