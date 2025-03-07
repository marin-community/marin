"""
Test Simulated Epoching on a Data Mixture
https://github.com/stanford-crfm/marin/issues/718
"""

from experiments.defaults import default_train, simulated_epoching_train
from experiments.dolma.tokenize_dolma import DOLMA_OLMO_MIXTURE_WEIGHTS, tokenize_dolma_steps
from experiments.llama import llama_150m, llama_150m_train_config
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import lm_mixture_data_config

EXPERIMENT_TAG = ["718_simulated_epoching"]

dolma_llama3_tokenized = lm_mixture_data_config(
    components=tokenize_dolma_steps(),
    weights=DOLMA_OLMO_MIXTURE_WEIGHTS,
)

# Regular 20B Token Run
regular_llama = default_train(
    name="simulated_epoching_150m_control",
    tokenized=dolma_llama3_tokenized,
    model_config=llama_150m,
    train_config=llama_150m_train_config,
    tags=[*EXPERIMENT_TAG, "no_sim", "150m"],
)

# Simulate Llama 3 Reported Budget which definitely causes epoching for Dolma
sim_llama = simulated_epoching_train(
    name="simulated_epoching_150m_experiment",
    tokenized=dolma_llama3_tokenized,
    model_config=llama_150m,
    train_config=llama_150m_train_config,
    target_budget=15e12,
    tags=[*EXPERIMENT_TAG, "sim", "150m"],
)

if __name__ == "__main__":
    executor_main(steps=[regular_llama, sim_llama])
