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

from experiments.datashop.datashop_datasets import (
    datashop_dclm_tutorial_annotation_subset,
    datashop_dclm_tutorial_pretraining_subset,
)
from experiments.datashop.datashop_runner import DatashopRunner, DatashopRunnerConfig
from experiments.exp939_finemath import FINEMATH_DATA_FILTER_PROMPT

datashop_runner = DatashopRunner(
    DatashopRunnerConfig(
        experiment_name="datashop-tutorial",
        annotator_model_name="meta-llama/Llama-3.1-8B-Instruct",
        pretraining_data_path=datashop_dclm_tutorial_pretraining_subset,
        annotator_data_path=datashop_dclm_tutorial_annotation_subset,
        data_filter_prompt=FINEMATH_DATA_FILTER_PROMPT,
        dataset_output_processor_config_kwargs={"processor_type": "finalscore0-5"},
    )
)

if __name__ == "__main__":
    datashop_runner.run_eval_cluster_steps()
