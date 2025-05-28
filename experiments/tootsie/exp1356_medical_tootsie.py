from dataclasses import replace

from experiments.anneal_config import AnnealConfig
from experiments.dclm.tokenize_dclm import dclm_components_llama3
from experiments.defaults import default_anneal
from experiments.midtraining_datasets import (
    lavita_allprocessed_tokenized,
    lavita_medmcqa_tokenized,
    lavita_medmcqa_validation_tokenized,
    lavita_pubmedqa_tokenized,
    lavita_pubmedqa_validation_tokenized,
    pile_pubmed_abstracts_validation_tokenized,
    pile_pubmed_central_validation_tokenized,
)
from experiments.tootsie.exp600_tootsie import phoenix_phase4_checkpoint_for_phase5
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import lm_mixture_data_config
from marin.resources import TpuPodConfig

pubmed_qa_tokens = 78_993_593
allprocessed_tokens = 58_717_739
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
            "lavita_allprocessed": lavita_allprocessed_tokenized,
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
            "lavita_allprocessed": medical_token_proportion * allprocessed_tokens / all_medical_tokens,
            "lavita_medmcqa": medical_token_proportion * medmcqa_tokens / all_medical_tokens,
        },
    ),
    resources=TpuPodConfig(tpu_type="v6e-128", slice_count=2),
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
    ),
)
tootsie_control = default_anneal(
    name="checkpoints/medical_tootsie_control",
    anneal_config=control_anneal_config,
)

if __name__ == "__main__":
    executor_main(
        [
            medical_tootsie_anneal,
            tootsie_control,
        ]
    )
