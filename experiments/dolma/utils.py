"""
This module provides helper function to set up and run experiments where a portion of the Dolma
dataset is added with custom processed data. The main workflow includes:

1. Tokenizing the custom dataset
2. Creating a mixture configuration where the custom dataset replaces a specific Dolma component
3. Training models on this modified mixture
4. Evaluating the resulting models
"""

from experiments.defaults import default_tokenize, default_train
from experiments.dolma.tokenize_dolma import DOLMA_OLMO_MIXTURE_WEIGHTS, tokenize_dolma_steps
from experiments.evals.evals import default_eval
from experiments.llama import llama3_tokenizer, llama_1_4b, llama_1_4b_train_config
from marin.execution.executor import ExecutorStep
from marin.processing.tokenize.data_configs import lm_mixture_data_config


def get_default_experiment_steps(
    path_suffix: str,
    dataset: ExecutorStep,
    dolma_dataset: str,
    experiment_tag: list[str],
    substitute_dolma_dataset: bool = True,
) -> list[ExecutorStep]:
    """
    Sets up a complete experiment pipeline for evaluating the impact of custom datasets on
    language model performance. This experiment substitutes or adds specific components of the
    Dolma dataset with custom processed data, then measures how these modifications affect model
    training.

    Args:
        path_suffix: The suffix to add to the output paths.
        dataset: The custom dataset that will either replace or be added to the Dolma mixture.
                 This should be a preprocessed dataset ready for tokenization.
        dolma_dataset: The Dolma dataset to be substituted or added to the mixture.
        substitute_dolma_dataset: Whether to substitute the Dolma dataset with the custom dataset or
                                  add it to the mixture.
        experiment_tag: The tag to add to the wandb experiment.
    Returns:
        A list of ExecutorSteps for the default experiment.
    """
    tokenized_dolma_steps = tokenize_dolma_steps()

    tokenized_dataset = default_tokenize(
        name=f"dolma-{path_suffix}",
        dataset=dataset,
        tokenizer=llama3_tokenizer,
    )

    dolma_tokenization_steps = dict(
        tokenized_dolma_steps,
        {f"dolma-{path_suffix}": tokenized_dataset},
    )

    dolma_weights = dict(
        DOLMA_OLMO_MIXTURE_WEIGHTS,
        {f"{path_suffix}": DOLMA_OLMO_MIXTURE_WEIGHTS[f"dolma/{dolma_dataset}"]},
    )
    if substitute_dolma_dataset:
        dolma_weights.pop(f"dolma/{dolma_dataset}")

    arxiv_llama3_tokenized = lm_mixture_data_config(
        components=dolma_tokenization_steps,
        weights=dolma_weights,
    )

    trained_1_4b_model = default_train(
        name=f"dolma-{path_suffix}-1.4b",
        tokenized=arxiv_llama3_tokenized,
        model_config=llama_1_4b,
        train_config=llama_1_4b_train_config,
        tags=experiment_tag,
    )

    trained_1_4b_evals = default_eval(step=trained_1_4b_model)

    return [
        tokenized_dataset,
        trained_1_4b_model,
        trained_1_4b_evals,
    ]
