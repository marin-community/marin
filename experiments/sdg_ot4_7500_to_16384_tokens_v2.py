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
Downloads marin-community/open-thoughts-4-30k-math-qwen3-32b-annotated (which uses 7500 max tokens for each generation)
and regenerates the responses with 16384 max tokens.

This version uses the generation inference API with Fray ResourceConfig instead of the
classification inference API with low-level Ray actor options.
"""
from dataclasses import replace

from marin.download.huggingface.download_hf import DownloadConfig, download_hf
from marin.execution.executor import ExecutorStep, this_output_path, executor_main
from experiments.models import qwen3_32b, get_model_local_path
from marin.export.hf_upload import upload_dir_to_hf
from marin.generation.inference import TextGenerationInferenceConfig
from marin.generation.inference import run_inference as run_generation_inference
from fray.cluster import ResourceConfig

open_thoughts_4_math = ExecutorStep(
    name="raw/marin-community/open-thoughts-4-30k-math-qwen3-32b-annotated",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="marin-community/open-thoughts-4-30k-math-qwen3-32b-annotated",
        revision="6683ed6",  # 100 parquets version
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path="raw/marin-community/open-thoughts-4-30k-math-qwen3-32b-annotated-6683ed6",
    pip_dependency_groups=["vllm"],
).cd("data")

qwen_32B_annotated_open_thoughts_4_math = ExecutorStep(
    name="documents/open-thoughts-4-30k-math-qwen3-32b-annotated-o16384-v2",
    fn=run_generation_inference,
    config=TextGenerationInferenceConfig(
        # IO specific
        input_path=open_thoughts_4_math,
        output_path=this_output_path(),

        # Model specific
        model_name=get_model_local_path(qwen3_32b),
        engine_kwargs={
            "tensor_parallel_size": 4,
            "max_model_len": 16384 + 2048,
        },
        generation_kwargs={
            "temperature": 0.8,
            "max_tokens": 16384,
        },

        # Prompting specific
        template="{example}",  # {example} is the placeholder for the value from prompt_column
        prompt_column="instruction_seed",
        apply_chat_template=True,
        save_templated_prompt=False,
        max_doc_tokens=16384,

        # Ray data specific
        num_instances=(4, 32),  # Use between 4 and 32 parallel workers
        batch_size=48,
        tensor_parallel_size=4,
        preserve_order=False,

        # File specific
        filetype="parquet",
        output_filetype_override="parquet",

        # Hardware specific
        resource_config=ResourceConfig.with_tpu("v5p-8"),

        # Output column
        generated_text_column_name="generated_text",

        # Checkpointing for resumption
        checkpoint_id_column="ms_id",
    ),
    pip_dependency_groups=["vllm"],
)

upload_qwen32b_annotated_open_thoughts_4_math = upload_dir_to_hf(
    input_path=qwen_32B_annotated_open_thoughts_4_math,
    repo_id="marin-community/open-thoughts-4-30k-math-qwen3-32b-annotated-16384-tokens",
    repo_type="dataset",
)

upload_qwen32b_annotated_open_thoughts_4_math = replace(
    upload_qwen32b_annotated_open_thoughts_4_math, pip_dependency_groups=["vllm"]
)


if __name__ == "__main__":
    # It's recommended to just run the first 2 steps (download model and annotate) separately
    # and then run the last step (upload to HF) afterwards
    executor_main([qwen3_32b])
    executor_main([qwen_32B_annotated_open_thoughts_4_math])
    # executor_main([upload_qwen32b_annotated_open_thoughts_4_math])
