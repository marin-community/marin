"""
This script runs a suite of scaling laws on the Dolma mix.

Link to issue: https://github.com/stanford-crfm/marin/issues/780 
"""

from marin.scaling_laws.create_ladder_suite import scaling_law_suite, WS_EMA_DEFAULT_TRAIN_CONFIG
from marin.execution.executor import executor_main
from experiments.defaults import default_tokenize
from experiments.dolma.tokenize_dolma import tokenize_dolma_steps
from experiments.dolma.exp442_dolma import dolma_llama3_tokenized

dolma_suite = scaling_law_suite(
    sweep_name="dolma-suite",
    tokenized=dolma_llama3_tokenized,
)

if __name__ == "__main__":
    executor_main(
        steps=[
            *dolma_suite,
        ],
        description="suite for scaling laws on Dolma mix",
    )
