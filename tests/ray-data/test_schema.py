import os

import pytest
import ray

from marin.generation.inference import OverwriteOutputFiletypeFilenameProvider, fix_warc_truncated_schema
from marin.generation.pipeline import vLLMTextGeneration
from tests.conftest import default_engine_kwargs, default_generation_params

TEST_INPUT_PATH = "gs://marin-us-east1/documents/datashop-datasets/datashop-dclm-annotation-subset-e035ad/shard_00000005_processed.jsonl.zst"
TEST_OUTPUT_PATH = "gs://marin-us-east5/documents/ray-data-test-schema"


@pytest.mark.skipif(os.getenv("TPU_CI") != "true", reason="Skip this test if not running with a TPU in CI.")
def test_ray_data(gcsfuse_mount_model_path):
    ds = ray.data.read_json(TEST_INPUT_PATH, arrow_open_stream_args={"compression": "zstd"})

    # Apply schema fix BEFORE vLLMTextGeneration to prevent PyArrow concatenation errors
    ds = ds.map_batches(fix_warc_truncated_schema)

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
            "max_doc_tokens": 896,
        },
        resources={"TPU": 1, "TPU-v6e-8-head": 1},
    )

    ds = ds.write_json(TEST_OUTPUT_PATH, filename_provider=OverwriteOutputFiletypeFilenameProvider("jsonl.gz"))
