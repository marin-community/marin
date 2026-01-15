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

"""Annealing on medical QA datasets.

We found that the pubmedqa and medmcqa datasets are generally pretty good for
increasing scores on MMLU medical tasks. Lavita-allprocessed is highly contaminated
with MMLU so we don't train on it.

Report: https://api.wandb.ai/links/marin-community/qfnxfxc3
"""

from dataclasses import replace

from experiments.anneal_config import AnnealConfig
from experiments.pretraining_datasets.dclm import dclm_components_llama3
from experiments.defaults import default_anneal
from experiments.midtraining_datasets import (
    lavita_medmcqa_tokenized,
    lavita_medmcqa_validation_tokenized,
    lavita_pubmedqa_tokenized,
    lavita_pubmedqa_validation_tokenized,
    pile_pubmed_abstracts_validation_tokenized,
    pile_pubmed_central_validation_tokenized,
)
from experiments.tootsie.exp600_tootsie import phoenix_phase4_checkpoint_for_phase5
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main, step
from marin.processing.tokenize.data_configs import lm_mixture_data_config

pubmed_qa_tokens = 78_993_593

# We found that the lavita-allprocessed dataset is highly contaminated with MMLU (20%) overlap
# with MMLU so we don't train on it.
allprocessed_tokens = 0
medmcqa_tokens = 30_779_801
all_medical_tokens = pubmed_qa_tokens + allprocessed_tokens + medmcqa_tokens
medical_token_proportion = 0.3
dclm_token_proportion = 1 - medical_token_proportion
num_anneal_tokens = int(all_medical_tokens * 4 / medical_token_proportion)
anneal_config = AnnealConfig(
    initialize_from_checkpoint_path=phoenix_phase4_checkpoint_for_phase5,
    dataset_config=lm_mixture_data_config(
        components={
            "dclm": dclm_components_llama3["dclm_baseline"],
            "lavita_pubmedqa": lavita_pubmedqa_tokenized,
            # "lavita_allprocessed": lavita_allprocessed_tokenized,
            "lavita_medmcqa": lavita_medmcqa_tokenized,
            # Validation sets
            "pile_pubmed_abstracts": pile_pubmed_abstracts_validation_tokenized,
            "pile_pubmed_central": pile_pubmed_central_validation_tokenized,
            "lavita_pubmedqa_validation": lavita_pubmedqa_validation_tokenized,
            "lavita_medmcqa_validation": lavita_medmcqa_validation_tokenized,
        },
        weights={
            "dclm": dclm_token_proportion,
            "lavita_pubmedqa": medical_token_proportion * pubmed_qa_tokens / all_medical_tokens,
            # "lavita_allprocessed": medical_token_proportion * allprocessed_tokens / all_medical_tokens,
            "lavita_medmcqa": medical_token_proportion * medmcqa_tokens / all_medical_tokens,
        },
        permutation_type="linear",
    ),
    resources=ResourceConfig.with_tpu("v6e-128", slice_count=2),
    num_anneal_training_tokens=num_anneal_tokens,
)
medical_tootsie_anneal = default_anneal(
    name="checkpoints/medical_tootsie",
    anneal_config=anneal_config,
)


control_anneal_config = replace(
    anneal_config,
    dataset_config=lm_mixture_data_config(
        components={
            "dclm": dclm_components_llama3["dclm_baseline"],
            "pile_pubmed_abstracts": pile_pubmed_abstracts_validation_tokenized,
            "pile_pubmed_central": pile_pubmed_central_validation_tokenized,
            "lavita_pubmedqa_validation": lavita_pubmedqa_validation_tokenized,
            "lavita_medmcqa_validation": lavita_medmcqa_validation_tokenized,
        },
        weights={
            "dclm": 1.0,
        },
        permutation_type="linear",
    ),
)
tootsie_control = default_anneal(
    name="checkpoints/medical_tootsie_control",
    anneal_config=control_anneal_config,
)


@step(name="tootsie/exp1356_medical_tootsie/all")
def run_experiment():
    """Entry point for the experiment."""
    return [
        medical_tootsie_anneal,
        tootsie_control,
    ]


if __name__ == "__main__":
    executor_main(steps=[run_experiment()], description="Annealing on medical QA datasets")
