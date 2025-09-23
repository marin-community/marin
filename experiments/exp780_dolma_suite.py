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
This script runs a suite of scaling laws on the Dolma mix.

Link to issue: https://github.com/marin-community/marin/issues/780
"""

from experiments.defaults import default_scaling_law_pred
from experiments.dolma.exp442_dolma import dolma_llama3_tokenized
from experiments.evals.task_configs import CORE_TASKS_PLUS_MMLU
from marin.execution.executor import executor_main
from marin.scaling_laws.create_ladder_suite import scaling_law_suite

dolma_suite = scaling_law_suite(
    sweep_name="scaling-law-suite-dolma-v2",
    tokenized=dolma_llama3_tokenized,
    tags=["scaling_laws"],
)

dolma_suite_scaling_laws_pred = default_scaling_law_pred(
    ladder_runs=dolma_suite,
    pred_run=None,  # this will give us readouts at various scales
    task_losses=(
        "eval/paloma/c4_en/bpb",
        "eval/bpb",
        "eval/loss",
        "lm_eval/hellaswag_10shot/bpb",
        "lm_eval/hellaswag_0shot/bpb",
        "lm_eval/boolq/bpb",
        "lm_eval/copa/bpb",
    ),
    task_accuracies=CORE_TASKS_PLUS_MMLU,
)


if __name__ == "__main__":
    executor_main(
        steps=[
            *dolma_suite,
            # dolma_suite_scaling_laws_pred,
        ],
        description="suite for scaling laws on Dolma mix",
    )
