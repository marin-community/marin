"""
This script runs a suite of scaling laws on the Dolma mix.

Link to issue: https://github.com/stanford-crfm/marin/issues/780
"""

from experiments.dolma.exp442_dolma import dolma_llama3_tokenized
from marin.execution.executor import executor_main
from marin.scaling_laws.create_ladder_suite import scaling_law_suite

dolma_suite = scaling_law_suite(
    sweep_name="scaling-law-suite-dolma",
    tokenized=dolma_llama3_tokenized,
    tags=["scaling_laws"],
)

if __name__ == "__main__":
    executor_main(
        steps=[
            *dolma_suite,
        ],
        description="suite for scaling laws on Dolma mix",
    )
