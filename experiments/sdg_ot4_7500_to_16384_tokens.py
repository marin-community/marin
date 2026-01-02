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
"""
from dataclasses import replace

from marin.download.huggingface.download_hf import DownloadConfig, download_hf
from marin.execution.executor import ExecutorStep, this_output_path, executor_main
from experiments.models import qwen3_32b, get_model_local_path
from marin.export.hf_upload import upload_dir_to_hf
from marin.processing.classification.inference import run_inference
from marin.processing.classification.config.inference_config import InferenceConfig, RuntimeConfig, DatasetSchemaConfig
from marin.processing.classification.autoscaler import AutoscalingActorPoolConfig

open_thoughts_4_math = ExecutorStep(
    name="raw/marin-community/open-thoughts-4-30k-math-qwen3-32b-annotated",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="marin-community/open-thoughts-4-30k-math-qwen3-32b-annotated",
        # revision="0d6d1f4",  # 4 parquets version
        revision="6683ed6",  # 100 parquets version
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    # override_output_path="raw/marin-community/open-thoughts-4-30k-math-qwen3-32b-annotated-0d6d1f4",  # 4 parquets version
    override_output_path="raw/marin-community/open-thoughts-4-30k-math-qwen3-32b-annotated-6683ed6",  # 100 parquets version
    pip_dependency_groups=["vllm"],
).cd("data")

qwen_32B_annotated_open_thoughts_4_math = ExecutorStep(
    name="documents/open-thoughts-4-30k-math-qwen3-32b-annotated-o16384",
    fn=run_inference,
    config=InferenceConfig(
        input_path=open_thoughts_4_math,
        output_path=this_output_path(),
        model_name=get_model_local_path(qwen3_32b),
        model_type="vllm",
        attribute_name="open_thoughts_4_30k_math_qwen3_32b_annotated",
        filetype="parquet",
        batch_size=8,
        resume=True,
        runtime=RuntimeConfig(memory_limit_gb=16),
        num_batches_per_upload=1,
        dataset_schema=DatasetSchemaConfig(
            input_columns=[
                "instruction_seed",
                "_source",
                "gpt41_mini_response",
                "__original_row_idx",
                "length",
                "ms_id",
                "generated_text",  # Old
                "conversations",
            ],
            output_columns=[
                "instruction_seed",
                "_source",
                "gpt41_mini_response",
                "__original_row_idx",
                "length",
                "ms_id",
                "generated_text",  # New (to be overwritten)
            ],
            id_column=("ms_id",),
        ),
        autoscaling_actor_pool_config=AutoscalingActorPoolConfig(
            min_actors=1,
            max_actors=1,
            scale_up_threshold=0.8,
            scale_down_threshold=0.2,
            scale_check_interval=1.0,
            actor_kwargs={
                "template": "{example}",
                "post_process_fn": None,
                "engine_kwargs": {
                    "tensor_parallel_size": 4,
                    "max_model_len": 16384+2048,
                },
                "generation_kwargs": {
                    "temperature": 0.8,
                    "max_tokens": 16384,
                },
                "save_original_generation": True,
                "prompt_column": "instruction_seed",  # TODO(chris): This should be in the dataset schema instead.
            },
            actor_options={
                "resources": {"TPU": 4},
                "runtime_env": {
                    "env_vars": {"JAX_PLATFORMS": ""}
                }
            },  # 4 chips on v5p-8
        ),
    ),
    pip_dependency_groups=["vllm"],
)

upload_qwen32b_annotated_open_thoughts_4_math = upload_dir_to_hf(
    input_path=qwen_32B_annotated_open_thoughts_4_math,
    repo_id="marin-community/open-thoughts-4-30k-math-qwen3-32b-annotated-16384-tokens",
    repo_type="dataset",
)

#  HACK needed on the vLLM cluster to get things running on vLLM cluster
upload_qwen32b_annotated_open_thoughts_4_math = replace(
    upload_qwen32b_annotated_open_thoughts_4_math, pip_dependency_groups=["vllm"]
)


if __name__ == "__main__":
    executor_main([qwen3_32b])
    executor_main([qwen_32B_annotated_open_thoughts_4_math, upload_qwen32b_annotated_open_thoughts_4_math])
