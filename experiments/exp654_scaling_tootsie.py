from experiments.exp600_tootsie import dclm_mixture_config_llama3_wrong
from marin.execution.executor import executor_main
from marin.training.scaling_laws import scaling_law_suite

TAG = ["654_scaling_tootsie"]

suite = scaling_law_suite(sweep_name="tootsie-scaling", tokenized=dclm_mixture_config_llama3_wrong, tags=TAG)

if __name__ == "__main__":
    executor_main(
        steps=[
            *suite,
        ],
        description="scaling law suite to predict performance of 8B model on DCLM mix",
    )
