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

from marin.download.huggingface.download import DownloadConfig
from marin.download.huggingface.download_hf import download_hf
from marin.execution.executor import ExecutorStep, this_output_path, executor_main
from experiments.models import qwen3_32b, get_model_local_path
from marin.processing.classification.inference import run_inference
from marin.processing.classification.config.inference_config import InferenceConfig, RuntimeConfig, DatasetSchemaConfig
from marin.processing.classification.autoscaler import AutoscalingActorPoolConfig

open_thoughts_4 = ExecutorStep(
    name="raw/open-thoughts-4",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="mlfoundations-dev/hero_run_4_math",
        revision="b2c8e95",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path="raw/open-thoughts-4-b2c8e95",
    pip_dependency_groups=["vllm"],
).cd("data")

qwen_32B_annotated_open_thoughts_4 = ExecutorStep(
    name="documents/open-thoughts-4-qwen3-32b-annotated-o7500",
    fn=run_inference,
    config=InferenceConfig(
        input_path=open_thoughts_4,
        output_path=this_output_path(),
        model_name=get_model_local_path(qwen3_32b),
        model_type="vllm",
        attribute_name="open_thoughts_4_qwen3_32b_annotated",
        filetype="parquet",
        batch_size=512,
        resume=True,
        runtime=RuntimeConfig(memory_limit_gb=16),
        num_batches_per_upload=1,
        dataset_schema=DatasetSchemaConfig(
            input_columns=[
                "messages",
                "instruction_seed",
                "response_seed",
                "_source",
                "gpt41_mini_response",
                "__original_row_idx",
                "length",
                "ms_id",
            ],
            output_columns=[
                "messages",
                "instruction_seed",
                "response_seed",
                "_source",
                "gpt41_mini_response",
                "__original_row_idx",
                "length",
                "ms_id",
                "generated_text",
            ],
            id_column=("ms_id",),
        ),
        autoscaling_actor_pool_config=AutoscalingActorPoolConfig(
            min_actors=1,
            max_actors=32,
            scale_up_threshold=0.8,
            scale_down_threshold=0.2,
            scale_check_interval=1.0,
            actor_kwargs={
                "template": "{example}",
                "post_process_fn": None,
                "engine_kwargs": {
                    "tensor_parallel_size": 8,
                    "max_model_len": 8192,
                },
                "generation_kwargs": {
                    "temperature": 0.8,
                    "max_tokens": 7500,
                },
                "save_original_generation": True,
                "prompt_column": "instruction_seed",  # TODO(chris): This should be in the dataset schema instead.
            },
            actor_options={"resources": {"TPU": 8}},
        ),
    ),
    pip_dependency_groups=["vllm"],
)

# qwen_32B_annotated_open_thoughts_4 = default_synthetic_data_generation(
#     input_path=open_thoughts_4,
#     output_path="documents/open-thoughts-4-qwen3-32b-annotated",
#     model_name_or_path=get_model_local_path(qwen3_32b),
#     data_generation_template="{example}",
#     input_filetype="parquet",
#     prompt_column="instruction_seed",
#     resource_config=TPU_V6E_8_STRICT_PACK,
#     engine_kwargs={
#         "tensor_parallel_size": 8,
#         "max_model_len": 8192,
#     },
#     generation_kwargs={
#         "temperature": 0.8,
#         "max_tokens": 7500,
#     },
# )

if __name__ == "__main__":
    executor_main([qwen3_32b])
    executor_main([qwen_32B_annotated_open_thoughts_4])
