import os

import pytest
import ray

from marin.generation.inference import OverwriteOutputFiletypeFilenameProvider
from marin.generation.pipeline import vLLMTextGeneration
from tests.conftest import TPU_V6E_8_WITH_HEAD_CONFIG, default_engine_kwargs, default_generation_params

TEST_OUTPUT_PATH = "gs://marin-us-east5/documents/ray-data-test-llama-200m"


@pytest.mark.skipif(os.getenv("TPU_CI") != "true", reason="Skip this test if not running with a TPU in CI.")
def test_ray_data(ray_cluster, gcsfuse_mount_model_path, test_file_path):
    ds = ray.data.read_json(test_file_path, arrow_open_stream_args={"compression": "gzip"})

    ds = ds.map_batches(  # Apply batch inference for all input data.
        vLLMTextGeneration,
        # Set the concurrency to the number of LLM instances.
        concurrency=1,
        # Specify the batch size for inference.
        batch_size=16,
        fn_constructor_kwargs={
            "model_name": gcsfuse_mount_model_path,
            "engine_kwargs": default_engine_kwargs,
            "generation_kwargs": default_generation_params,
            "template": "What is this text about? {example}",
            "prompt_column": "text",
            "save_templated_prompt": False,
            "apply_chat_template": True,
            "max_doc_length": 896,
        },
        resources=TPU_V6E_8_WITH_HEAD_CONFIG.get_ray_resources_dict(),
    )

    ds = ds.write_json(TEST_OUTPUT_PATH, filename_provider=OverwriteOutputFiletypeFilenameProvider("jsonl.gz"))
