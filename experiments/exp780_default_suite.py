"""
This script runs a suite of scaling laws on the DCLM-Baseline+StarCoder+ProofPile mix.
This is the default mix that we use for our experiments/scaling laws, and can be used 
as a reference point to compare other mixes/scaling law suites against.

Link to issue for scaling law experiments: https://github.com/stanford-crfm/marin/issues/780
"""

from experiments.exp600_tootsie import dclm_mixture_config_llama3
from marin.execution.executor import executor_main
from marin.scaling_laws.create_ladder_suite import scaling_law_suite

default_suite = scaling_law_suite(
    sweep_name="scaling-law-suite-default",
    tokenized=dclm_mixture_config_llama3,
    tags=["scaling_laws"],
)

if __name__ == "__main__":
    executor_main(
        steps=[
            *default_suite,
        ],
        description="suite for scaling laws on DCLM-Baseline+StarCoder+ProofPile mix",
    )
