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

# nodryrun
from dataclasses import replace

from experiments.datashop.datashop_datasets import datashop_dclm_annotation_subset, datashop_dclm_pretraining_subset
from experiments.datashop.datashop_runner import DatashopRunner, DatashopRunnerConfig
from experiments.datashop.default_configs import (
    default_consolidate_filter_config_kwargs,
    default_quality_filter_train_config_kwargs,
)
from experiments.exp939_finemath import FINEMATH_DATA_FILTER_PROMPT
from experiments.models import get_model_local_path, llama_3_3_70b_instruct
from marin.classifiers.utils import CreateDatasetConfig, create_dataset
from marin.execution.executor import ExecutorStep, output_path_of, this_output_path

# From the SmolLM2 paper: https://arxiv.org/pdf/2502.02737
FINEMATH_3_POINT_DATA_FILTER_PROMPT = """
Evaluate the following text extract for its potential usefulness for studying mathematics up to high school and early undergraduate levels. Use the following 3-point scoring system described below. Points are accumulated based on the satisfaction of
each criterion:
- Add 1 point if the extract contains some mathematical content, even if it’s not very useful for studying or is an academic
paper that is too advanced.
- Add another point if the extract demonstrates logical reasoning in a mathematical context, even if it lacks step-by-step
explanations or is too advanced.
- Award a third point if the extract is at an appropriate level (up to high school and early undergraduate levels) and contains
clear mathematical deductions and step-by-step solutions to mathematical problems.
Question-answer formats (e.g., from educational websites or forums) are acceptable if they meet the criteria. Ignore any
formatting errors or missing equations and make assumptions based on the overall content.
The text extract:
{example}
After examining the extract:
- Briefly justify your total score, up to 100 words.
- Conclude with the score using the format: "Final score: <total points>"
"""  # noqa: E501, RUF001


# ===== STAGE 1: Filter out most of the low-quality data =====
quality_filter_train_config_kwargs = default_quality_filter_train_config_kwargs.copy()
quality_filter_train_config_kwargs["training_config"] = replace(
    quality_filter_train_config_kwargs["training_config"],
    max_label=3,
)

datashop_runner = DatashopRunner(
    DatashopRunnerConfig(
        experiment_name="finemath-cascade",
        annotator_model_name=get_model_local_path(llama_3_3_70b_instruct),
        pretraining_data_path=datashop_dclm_pretraining_subset,
        annotator_data_path=datashop_dclm_annotation_subset,
        data_filter_prompt=FINEMATH_3_POINT_DATA_FILTER_PROMPT,
        dataset_output_processor_config_kwargs={"processor_type": "finalscore0-5"},
        quality_train_config_kwargs=quality_filter_train_config_kwargs,
    )
)

# ===== STAGE 2: From the remaining data, create a annotation pool and train a model on its filtered data =====
new_annotation_data_pool = ExecutorStep(
    name="documents/finemath-cascade-3-point-filter/dclm-baseline",
    fn=create_dataset,
    config=CreateDatasetConfig(
        input_doc_path=output_path_of(datashop_runner.filtered_documents),
        output_dataset_path=this_output_path(),
        max_sample_size=1_000_000,
        filetype="jsonl.zst",
        merge_dataset_shards=False,
    ),
)

phase_2_quality_filter_config_kwargs = default_quality_filter_train_config_kwargs.copy()
phase_2_quality_filter_config_kwargs["training_config"] = replace(
    phase_2_quality_filter_config_kwargs["training_config"],
    learning_rate=1e-4,
    max_label=5,
)

consolidate_filter_kwargs = default_consolidate_filter_config_kwargs.copy()
consolidate_filter_kwargs["keep_fraction"] = 0.25
datashop_runner_phase_2 = DatashopRunner(
    DatashopRunnerConfig(
        experiment_name="finemath-cascade-phase-2",
        annotator_model_name=get_model_local_path(llama_3_3_70b_instruct),
        pretraining_data_path=datashop_runner.filtered_documents,
        annotator_data_path=new_annotation_data_pool,
        data_filter_prompt=FINEMATH_DATA_FILTER_PROMPT,
        dataset_output_processor_config_kwargs={"processor_type": "finalscore0-5"},
        quality_train_config_kwargs=phase_2_quality_filter_config_kwargs,
        filter_config_kwargs=consolidate_filter_kwargs,
    )
)

if __name__ == "__main__":
    datashop_runner_phase_2.run_all_steps()
