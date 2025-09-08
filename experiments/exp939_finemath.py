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

FINEMATH_DATA_FILTER_PROMPT = """
Evaluate the following text extract for its potential usefulness for studying mathematics up to high school and early undergraduate levels. Use the following 5-point scoring system described below. Points are accumulated based on the satisfaction of
each criterion:

- Add 1 point if the extract contains some mathematical content, even if it’s not very useful for studying, or if it contains
non-academic content such as advertisements and generated pages for converting weight and currencies.
- Add another point if the extract touches on mathematical topics, even if it’s poorly written if it’s too complex such as an
academic paper that is too advanced.
- Award a third point if the extract demonstrates problem solving or logical reasoning in a mathematical context, even if it lacks
step-by-step explanations.
- Grant a fourth point if the extract is at an appropriate level (up to high school and early undergraduate levels) and contains
clear mathematical deductions and step-by-step solutions to mathematical problems. It should be similar to a chapter from a
textbook or a tutorial.
- Give a fifth point if the extract is outstanding in its educational value for teaching and studying mathematics in middle school
and high school. It should include very detailed and easy to follow explanations.
Question-answer formats (e.g., from educational websites or forums) are acceptable if they meet the criteria.
The text extract:
{example}
After examining the extract:
- Briefly justify your total score, up to 100 words.
- Conclude with the score using the format: Final score: <total points>.
"""  # noqa: E501, RUF001

datashop_runner = DatashopRunner(
    DatashopRunnerConfig(
        experiment_name="finemath-replication",
        annotator_model_name="Llama-3.3-70B-Instruct",
        pretraining_data_path=datashop_dclm_pretraining_subset,
        annotator_data_path=datashop_dclm_annotation_subset,
        data_filter_prompt=FINEMATH_DATA_FILTER_PROMPT,
        dataset_output_processor_config_kwargs={"processor_type": "finalscore0-5"},
    )
)

if __name__ == "__main__":
    datashop_runner.run_all_steps()
