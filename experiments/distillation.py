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

"""An example to demonstrate how to generate synthetic data given a seed dataset.

In this example, we use the MATH-500 dataset from HuggingFace and generate synthetic data using
a Llama-3.1-8B-Instruct model. To try a different model or dataset,
you can change the `model_name` or `huggingface_dataset_id` variables, respectively.
"""

from experiments.datashop.defaults import default_synthetic_data_generation
from experiments.models import get_model_local_path, llama_3_1_8b_instruct
from fray.cluster import ResourceConfig
from marin.download.huggingface.download_hf import DownloadConfig, download_hf
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.utils import get_directory_friendly_name

huggingface_dataset_id = "HuggingFaceH4/MATH-500"
tensor_parallel_size = 1

dataset_name = get_directory_friendly_name(huggingface_dataset_id)

math500 = ExecutorStep(
    name=f"raw/{dataset_name}-retry-2",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id=huggingface_dataset_id,
        revision=versioned("ff5b202"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
)

generations = default_synthetic_data_generation(
    input_path=math500,
    model_name_or_path=get_model_local_path(llama_3_1_8b_instruct),
    data_generation_template="You will be given a problem. Please reason step by step, \
        and put your final answer within \boxed{{}}:\n{example}",
    input_filetype="jsonl",
    prompt_column="problem",
    resource_config=ResourceConfig.with_tpu("v6e-8"),
    output_path="documents/synthetic_data_llama_8b",
)

steps = [math500, generations]

if __name__ == "__main__":
    executor_main(steps)
