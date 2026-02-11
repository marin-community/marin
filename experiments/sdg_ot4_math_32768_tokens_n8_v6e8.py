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
Starts from the original dataset and generates 2 responses with Qwen3-32B (n7 and n8),
storing them in "generated_text7" and "generated_text8". Uses v6e-8 TPUs.
"""
from dataclasses import replace

from marin.execution.executor import ExecutorStep, this_output_path, executor_main
from marin.export.hf_upload import upload_dir_to_hf
from marin.generation.inference import TextGenerationInferenceConfig
from marin.generation.inference import run_inference as run_generation_inference
from fray.cluster import ResourceConfig

# Start from original dataset - only needs instruction_seed column for generation
original_dataset = "gs://marin-eu-west4/raw/marin-community/open-thoughts-4-30k-math-qwen3-32b-annotated-32768-tokens-6a05237/data"

# Generate 2 responses (generated_text7 and generated_text8)
generation_steps = []
prev_step = original_dataset

for i in range(7, 9):
    step = ExecutorStep(
        name=f"documents/open-thoughts-4-30k-math-qwen3-32b-annotated-o32768-n{i}",
        fn=run_generation_inference,
        config=TextGenerationInferenceConfig(
            # IO specific
            input_path=prev_step,
            output_path=this_output_path(),

            # Model specific — stream weights directly from GCS to avoid local disk space issues
            model_name="gs://marin-us-central1/gcsfuse_mount/models/Qwen--Qwen3-32B--9216db5781bf21249d130ec9da846c4624c16137",
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
            num_instances=(45, 200),
            batch_size=64,
            tensor_parallel_size=4,
            preserve_order=False,

            # File specific
            filetype="parquet",
            output_filetype_override="parquet",

            # Hardware specific
            resource_config=ResourceConfig.with_tpu("v6e-8"),

            # Output column
            generated_text_column_name=f"generated_text{i}",

            # Checkpointing for resumption
            checkpoint_id_column="ms_id",
        ),
        pip_dependency_groups=["vllm"],
    )
    generation_steps.append(step)
    prev_step = step

final_generation_step = generation_steps[-1]

upload_step = upload_dir_to_hf(
    input_path=final_generation_step,
    repo_id="marin-community/open-thoughts-4-30k-math-qwen3-32b-annotated-32768-tokens-n8",
    repo_type="dataset",
)

upload_step = replace(upload_step, pip_dependency_groups=["vllm"])


if __name__ == "__main__":
    # Run generation steps for n7 and n8
    executor_main(generation_steps)
    # executor_main([upload_step])
