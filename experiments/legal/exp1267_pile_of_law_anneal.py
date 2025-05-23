from experiments.anneal_config import AnnealConfig
from experiments.defaults import default_anneal
# Updated import:
from experiments.legal.tokenize_pile_of_law import get_pile_of_law_download_step, get_pile_of_law_all_tokenized_step
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import lm_mixture_data_config

# 1. Create the download step for Pile of Law data
pile_of_law_download_step = get_pile_of_law_download_step()

# 2. Create the tokenization step, which depends on the download step
pile_of_law_tokenize_step = get_pile_of_law_all_tokenized_step(download_step=pile_of_law_download_step)

# Define the dataset mixture for AnnealConfig using the single "pile_of_law_all" component
pile_of_law_mixture_config = lm_mixture_data_config(
    components={
        "pile_of_law_all": pile_of_law_tokenize_step, # Use the new tokenize_step variable
    },
    weights={
        "pile_of_law_all": 1.0,
    },
)

# Define the AnnealConfig
pile_of_law_anneal_config = AnnealConfig(
    dataset_config=pile_of_law_mixture_config,
    # Optional: Adjust other AnnealConfig parameters if needed, e.g.,
    # num_anneal_training_tokens for the larger combined dataset.
)

# Create the annealing experiment step
pile_of_law_anneal_experiment = default_anneal(
    name="llama-8b-anneal-pile-of-law-all", # Updated name
    anneal_config=pile_of_law_anneal_config,
)

# Main execution block
if __name__ == "__main__":
    # The check for PILE_OF_LAW_BASE_PATH is removed.
    # download_step and tokenize_step are defined above and used to construct
    # pile_of_law_anneal_experiment. The executor should handle the dependency chain.
    executor_main(
        steps=[
            pile_of_law_anneal_experiment,
        ],
        description="Annealing experiment with the entire Pile of Law dataset (downloaded and globbed).", # Updated description
    )
