"""
Base SFT evaluations across multiple LLMs.

This experiment evaluates OLMO Base 8B, LLAMA 3.1 8B, Deeper Starling 8B,
MAP-NEO 7B, and Amber Base 7B models on CORE_TASKS (augmented with OLMo Eval Tasks) as well
as dedicated MMLU 0-shot and 5-shot configurations.
"""

from experiments.evals.evals import default_sft_eval
from experiments.models import tulu_3_1_8b_instruct
from marin.execution.executor import executor_main

if __name__ == "__main__":
    # Run all evaluations on all models
    executor_main(
        steps=[
            # *default_sft_eval("gs://marin-us-central2/checkpoints/sft/mixture_sft_starling_1e-4-longer/hf/step-10227")
            # * default_sft_eval(deeper_mixture_experiment)[:1],
            # *default_sft_eval(llama_3_1_8b_instruct)[:1],
            *default_sft_eval(tulu_3_1_8b_instruct)[:-2],
        ]
    )
