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

from marin.download.huggingface.download_hf import DownloadConfig, download_hf
from marin.execution.executor import ExecutorStep, executor_main, StepRef, versioned
from marin.transform.huggingface.dataset_to_eval import DatasetConversionConfig, OutputFormatOptions, hf_dataset_to_jsonl

"""
Downloads the following datasets
- mmlu
"""
############################################################
# download mmlu dataset
mmlu_download_step = ExecutorStep(
    name="raw/cais/mmlu",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="cais/mmlu",
        revision=versioned("c30699e"),
        gcs_output_path=StepRef(_step=None),
        wait_for_completion=True,
    ),
    override_output_path="raw/cais/mmlu",
).cd("c30699e/huggingface.co/datasets/cais/mmlu/resolve/c30699e")


"""
Converts raw to JSON for:
- mmlu
"""
############################################################
# Convert mmlu to evaluation format (i.e. JSON with "prompt", "response" fields)
# This is the input for internal evaluation which measures PPL model gives to correct responses to prompts

# This creates a JSON file representing the auxiliary training data subset of MMLU
mmlu_convert_eval_aux = ExecutorStep(
    name="evaluation/mmlu-eval-aux",
    fn=hf_dataset_to_jsonl,
    config=DatasetConversionConfig(
        dataset_name="cais/mmlu",
        subsets=["all"],
        splits=["auxiliary_train"],
        input_path=mmlu_download_step,
        hf_path="cais/mmlu",
        output_path=StepRef(_step=None),
        output_format=OutputFormatOptions("evaluation"),
        prompt_key="question",
        options_key="choices",
        answer_idx_key="answer",
        answer_labels=["A", "B", "C", "D"],
    ),
)

# This creates one file per subject from MMLU, excluding the all and auxiliary training subsets
mmlu_convert_eval_subject = ExecutorStep(
    name="evaluation/mmlu-eval-subject",
    fn=hf_dataset_to_jsonl,
    config=DatasetConversionConfig(
        dataset_name="cais/mmlu",
        subsets=["*"],
        splits=["dev", "validation"],
        input_path=mmlu_download_step,
        hf_path="cais/mmlu",
        output_path=StepRef(_step=None),
        output_format=OutputFormatOptions("evaluation"),
        prompt_key="question",
        options_key="choices",
        answer_idx_key="answer",
        answer_labels=["A", "B", "C", "D"],
        exclude_subsets=["all", "auxiliary_train"],
    ),
)
############################################################
# Convert mmlu to dolma format (i.e. JSON with "text" field)
# This is used as input to the decontamination pipeline so documents with MMLU content are removed
mmlu_convert_dolma = ExecutorStep(
    name="decontamination/mmlu-dolma",
    fn=hf_dataset_to_jsonl,
    config=DatasetConversionConfig(
        dataset_name="cais/mmlu",
        subsets=["all"],
        splits=["dev", "test", "validation"],
        input_path=mmlu_download_step,
        hf_path="cais/mmlu",
        output_path=StepRef(_step=None),
        output_format=OutputFormatOptions("decontamination"),
        prompt_key="question",
        options_key="choices",
        answer_idx_key="answer",
        answer_labels=["A", "B", "C", "D"],
    ),
)

############################################################

if __name__ == "__main__":
    executor_main(
        steps=[
            mmlu_download_step,
            mmlu_convert_eval_aux,
            mmlu_convert_eval_subject,
            mmlu_convert_dolma,
        ]
    )
