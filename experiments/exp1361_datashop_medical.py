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

from experiments.datashop.datashop_datasets import datashop_dclm_annotation_subset, datashop_dclm_pretraining_subset
from experiments.datashop.datashop_runner import DatashopRunner, DatashopRunnerConfig
from experiments.evals.evals import default_eval
from experiments.evals.task_configs import MEDICAL_TASKS
from experiments.models import llama_3_1_8b, llama_3_1_8b_instruct
from marin.execution.executor import executor_main

DATASHOP_MEDICAL_DATA_FILTER_PROMPT = """
Evaluate the following text extract for its potential usefulness for studying medical content. Use the following 5-point scoring system described below. Points are accumulated based on the satisfaction of
each criterion:

- Add 1 point if the extract contains some medical or biology content, even if it’s not very useful for studying.
- Add another point if the extract touches on biology, anatomy, or medical topics, even if it’s poorly written if it’s too complex such as an
academic paper that is too advanced.
- Award a third point if the extract demonstrates strong technical content that teaches the reader something about medical or biology.
- Grant a fourth point if the extract is at an appropriate level and contains clear medical or biology terminology and step-by-step solutions to medical or biology problems. It should be similar to a chapter from a
textbook or a tutorial.
- Give a fifth point if the extract is outstanding in its educational value for teaching and studying medical or biology. It should include very detailed and easy to follow explanations.
Question-answer formats (e.g., from educational websites or forums) are acceptable if they meet the criteria.
The text extract:
{example}
After examining the extract:
- Briefly justify your total score, up to 100 words.
- Conclude with the score using the format: Final score: <total points>.
"""  # noqa: E501, RUF001

datashop_runner = DatashopRunner(
    DatashopRunnerConfig(
        experiment_name="datashop-medical",
        annotator_model_name="/opt/gcsfuse_mount/models/meta-llama--Llama-3-3-70B-Instruct",
        pretraining_data_path=datashop_dclm_pretraining_subset,
        annotator_data_path=datashop_dclm_annotation_subset,
        data_filter_prompt=DATASHOP_MEDICAL_DATA_FILTER_PROMPT,
        dataset_output_processor_config_kwargs={"processor_type": "finalscore0-5"},
    )
)

initial_model_evals = default_eval(
    "gs://marin-us-east1/checkpoints/llama-8b-tootsie-0.001-19ad63/hf/step-660000",
    datashop_runner.config.eval_resource_config,
    MEDICAL_TASKS,
)

datashop_model_evals = default_eval(
    datashop_runner.quality_ablation_model, datashop_runner.config.eval_resource_config, MEDICAL_TASKS
)

llama_3_1_8b_instruct_evals = default_eval(
    llama_3_1_8b_instruct, datashop_runner.config.eval_resource_config, MEDICAL_TASKS
)

llama_3_1_8b_evals = default_eval(llama_3_1_8b, datashop_runner.config.eval_resource_config, MEDICAL_TASKS)

if __name__ == "__main__":
    executor_main(
        [
            initial_model_evals,
            datashop_model_evals,
            llama_3_1_8b_instruct_evals,
            llama_3_1_8b_evals,
        ]
    )
