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

"""An experiment to cooldown a 8B model on a 30/70 mixture of MegaMath and Dolmino DCLM.

Evaluates the quality of MegaMath: hope to see boost in GSM8K and MMLU mathematics.

cf exp722_anneal.py for more details on the annealing process and configuration.
"""

from experiments.anneal_config import AnnealConfig
from experiments.defaults import default_anneal
from experiments.dolmino.tokenize_dolmino import get_dolmino_step_llama3
from experiments.evals.evals import default_eval
from experiments.evals.task_configs import MMLU_TASKS
from experiments.midtraining_datasets import megamath_mixture, megamath_real_only
from experiments.tootsie.exp600_tootsie import phoenix_phase4_checkpoint_for_phase5
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import interpolate_mixture_configs, lm_mixture_data_config

dolmino_dclm = get_dolmino_step_llama3("dclm")

pure_dolmino_mixture = lm_mixture_data_config(
    components={"dolmino/dclm": dolmino_dclm},
    weights={"dolmino/dclm": 1.0},
    permutation_type="linear",
)

checkpoint = phoenix_phase4_checkpoint_for_phase5


megamath_real_anneal_config = AnnealConfig(
    dataset_config=interpolate_mixture_configs(
        [pure_dolmino_mixture, megamath_real_only],
        [0.7, 0.3],  # 70% Dolmino DCLM, 30% MegaMath Real
    ),
    initialize_from_checkpoint_path=checkpoint,
)

megamath_anneal_config = AnnealConfig(
    dataset_config=interpolate_mixture_configs(
        [pure_dolmino_mixture, megamath_mixture],
        [0.7, 0.3],  # 70% Dolmino DCLM, 30% MegaMath
    ),
    initialize_from_checkpoint_path=checkpoint,
)

control_dclm_anneal_config = AnnealConfig(
    dataset_config=pure_dolmino_mixture,
    initialize_from_checkpoint_path=checkpoint,
)

control_model = default_anneal(
    name="marin-8b-anneal-megamath-control", anneal_config=control_dclm_anneal_config
).with_output_path("checkpoints/marin-8b-anneal-megamath-control")

annealed_model_all = default_anneal(
    name="marin-8b-anneal-megamath-all", anneal_config=megamath_anneal_config
).with_output_path("checkpoints/marin-8b-anneal-megamath-all")

annealed_model_real_megamath = default_anneal(
    name="marin-8b-anneal-megamath-real-dclm", anneal_config=megamath_real_anneal_config
).with_output_path("checkpoints/marin-8b-anneal-megamath-real-dclm")


# Note: Checkpoint paths will be determined dynamically by the annealing process.
# We define what to evaluate, not a specific checkpoint here.
eval_annealed_model_all = default_eval(
    step=annealed_model_all,  # Evaluate the output of the annealed_model step
    evals=MMLU_TASKS,
)

eval_annealed_model_real = default_eval(
    step=annealed_model_real_megamath,  # Evaluate the output of the annealed_model step
    evals=MMLU_TASKS,
)

eval_control_model = default_eval(
    step=control_model,  # Evaluate the output of the control_model step
    evals=MMLU_TASKS,
)

if __name__ == "__main__":
    executor_main(
        steps=[
            annealed_model_all,
            annealed_model_real_megamath,
            control_model,
            eval_annealed_model_all,
            eval_annealed_model_real,
            eval_control_model,
        ],
        description="Anneal Marin 8B model on MegaMath and Dolmino DCLM, comparing real vs all MegaMath data.",
    )
