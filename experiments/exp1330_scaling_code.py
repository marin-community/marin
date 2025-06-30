from experiments.dclm.tokenize_dclm import dclm_components_llama3
from marin.execution.executor import executor_main
from marin.scaling_laws.create_ladder_suite import scaling_law_suite

TAG = ["1330_scaling_code"]

suite = scaling_law_suite(
    sweep_name="code-scaling", tokenized=dclm_components_llama3["starcoderdata"], tags=TAG, intermediate_scale=4
)

if __name__ == "__main__":
    executor_main(
        steps=[
            *suite,
        ],
        description="scaling law suite to predict performance of 8B model on Code mix",
    )
