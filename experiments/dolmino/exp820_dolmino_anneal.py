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

"""An experiment to evaluate the quality of individual splits of the Dolmino dataset.

We cooldown a 8B model on a 30/70 mixture of some high quality Dolmino split and Dolmino DCLM.
Link to issue: https://github.com/marin-community/marin/issues/820
"""

from experiments.anneal_config import AnnealConfig
from experiments.defaults import default_anneal
from experiments.pretraining_datasets import get_dolmino_step_llama3
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import lm_mixture_data_config

dolmino_dclm = get_dolmino_step_llama3("dclm")

# Control model is 100% dolmino DCLM dataset
control_dataset_config = lm_mixture_data_config(
    components={"dolmino_dclm": dolmino_dclm},
    weights={"dolmino_dclm": 1.0},
    permutation_type="linear",
)
control_anneal_config = AnnealConfig(
    dataset_config=control_dataset_config,
    num_anneal_training_tokens=8_400_000_000,
)

control_model = default_anneal(
    name="llama-8b-anneal-control",
    anneal_config=control_anneal_config,
)

stackexchange_tokenized = get_dolmino_step_llama3("stackexchange")

dataset_config = lm_mixture_data_config(
    components={
        "stackexchange": stackexchange_tokenized,
        "dclm": dolmino_dclm,
    },
    weights={"stackexchange": 0.30, "dclm": 0.70},
    permutation_type="linear",
)
# Dolmino Stack Exchange dataset has 1.26B tokens.
# Our mixed dataset is 30% dolmino and 70% high-quality web data.
# This means we will epoch dolmino dataset 2 times.
stackexchange_anneal_config = AnnealConfig(
    dataset_config=dataset_config,
    num_anneal_training_tokens=8_400_000_000,
)

stackexchange_anneal_model = default_anneal(
    name="llama-8b-anneal-stackexchange-0",
    anneal_config=stackexchange_anneal_config,
)

# eval_stackexchange_anneal_model = default_eval(
#     step="gs://marin-us-central2/checkpoints/llama-8b-anneal-stackexchange-0-db2e46/hf/step-200470",
#     evals=MMLU_TASKS,
# )

wiki_tokenized = get_dolmino_step_llama3("wiki")
wiki_dataset_config = lm_mixture_data_config(
    components={"wiki": wiki_tokenized, "dolmino_dclm": dolmino_dclm},
    weights={"wiki": 0.3, "dolmino_dclm": 0.7},
    permutation_type="linear",
)
wiki_anneal_config = AnnealConfig(
    dataset_config=wiki_dataset_config,
    num_anneal_training_tokens=8_400_000_000,
)

wiki_anneal_model = default_anneal(
    name="llama-8b-anneal-wiki-0",
    anneal_config=wiki_anneal_config,
)

if __name__ == "__main__":
    executor_main(
        steps=[
            stackexchange_anneal_model,
            control_model,
            wiki_anneal_model,
        ],
    )
