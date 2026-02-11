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
Downloads marin-community/open-thoughts-4-math-qwen3-32b-annotated (the full math dataset,
which uses 7500 max tokens for each generation) and regenerates the responses with 32768 max tokens.
"""
from dataclasses import replace

from marin.download.huggingface.download_hf import DownloadConfig, download_hf
from marin.execution.executor import ExecutorStep, this_output_path, executor_main
from marin.export.hf_upload import upload_dir_to_hf
from marin.generation.inference import TextGenerationInferenceConfig
from marin.generation.inference import run_inference as run_generation_inference
from fray.cluster import ResourceConfig

# Download the full math dataset from HuggingFace
open_thoughts_4_all_math = ExecutorStep(
    name="raw/marin-community/open-thoughts-4-math-qwen3-32b-annotated",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="marin-community/open-thoughts-4-math-qwen3-32b-annotated",
        revision="1bd0352",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path="raw/marin-community/open-thoughts-4-math-qwen3-32b-annotated-1bd0352",
    pip_dependency_groups=["vllm"],
)

# Generate responses with 32768 max tokens using v6e-8 TPUs
# Each step starts from the source dataset independently (no chaining)
generation_steps = []

for i in range(0, 1):
    step = ExecutorStep(
        name=f"documents/open-thoughts-4-math-qwen3-32b-annotated-o32768-n{i}",
        fn=run_generation_inference,
        config=TextGenerationInferenceConfig(
            # IO specific
            input_path=open_thoughts_4_all_math,
            output_path=this_output_path(),

            # Model specific — stream weights directly from GCS to avoid local disk space issues
            model_name="gs://marin-eu-west4/gcsfuse_mount/models/Qwen--Qwen3-32B--9216db5781bf21249d130ec9da846c4624c16137",
            engine_kwargs={
                "tensor_parallel_size": 4,
                "max_model_len": 32768 + 2048,
                "load_format": "runai_streamer",
                "seed": i,
            },
            generation_kwargs={
                "temperature": 0.8,
                "max_tokens": 32768,
            },

            # Prompting specific
            template="{example}",
            prompt_column="instruction_seed",
            apply_chat_template=True,
            save_templated_prompt=False,
            max_doc_tokens=32768,

            # Ray data specific
            num_instances=(10, 200),
            batch_size=64,
            tensor_parallel_size=4,
            preserve_order=False,

            # File specific
            filetype="parquet",
            output_filetype_override="parquet",

            # Hardware specific
            resource_config=ResourceConfig.with_tpu("v6e-8"),

            # Output column
            generated_text_column_name="generated_text" if i == 0 else f"generated_text{i}",

            # Checkpointing for resumption
            checkpoint_id_column="ms_id",
            max_rows_per_file=512,
        ),
        pip_dependency_groups=["vllm"],
    )
    generation_steps.append(step)

final_generation_step = generation_steps[-1]

upload_step = upload_dir_to_hf(
    input_path=final_generation_step,
    repo_id="marin-community/open-thoughts-4-math-qwen3-32b-annotated-32768-tokens",
    repo_type="dataset",
)

upload_step = replace(upload_step, pip_dependency_groups=["vllm"])


if __name__ == "__main__":
    executor_main(generation_steps)
    # executor_main([upload_step])
