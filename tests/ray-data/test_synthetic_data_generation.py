import os
import pytest
import ray
from marin.generation.inference import run_inference, TextGenerationInferenceConfig
from tests.conftest import SINGLE_GPU_CONFIG, default_engine_kwargs, default_generation_params

@pytest.mark.skipif(os.getenv("GPU_CI") != "true", reason="Skip this test if not running with a GPU in CI.")
def test_synthetic_data_generation_gpu(ray_cluster, test_file_path):
    ray.get(run_inference.remote(
        config=TextGenerationInferenceConfig(
            input_path=test_file_path,
            output_path="documents/synthetic_data_llama_8b_gpu",
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            engine_kwargs=default_engine_kwargs,
            generation_kwargs=default_generation_params,
            template="Summarize the following text in 20 words:\n{example}",
            prompt_column="text",
            filetype="jsonl.gz",    
            output_filetype_override="jsonl.gz",
            one_to_one_input_output_mapping=False,
            generated_text_column_name="generated_text",
            resource_config=SINGLE_GPU_CONFIG,
            max_doc_length=800,
        ),
    ))