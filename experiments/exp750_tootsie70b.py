"""
This is a continuation of 201's 70b but we are using a higher LR and switching to WSD with EMA
instead of WSD-S.

* Schedule: WSD, Decay is 40%
* Peak LR is 2e-4
* ema beta 0.995

Mix is still DCLM+Math+Code
"""
import dataclasses

from experiments.defaults import default_train
from experiments.exp201_tootsie22b import llama_70b, llama_70b_train_config, dclm_mixture_config_llama3

from marin.execution.executor import executor_main

llama_70b_train_config_mk2 = dataclasses.replace(
    llama_70b_train_config,
    train_batch_size=1024,
    tpu_type="v4-512",
    node_count=4,
    learning_rate=2e-4,
    decay=0.4,
    ema_beta=0.995,
    lr_schedule="linear",
    cycle_length=None,
    allow_partial_checkpoint=True,
)


llama_70b_tootsie_mk2 = dataclasses.replace(
    default_train(
        name="llama-70b-tootsie-mk2",
        # not recorded here:
        # warmstart weights from llama_70b_tootsie step 80000
        tokenized=dclm_mixture_config_llama3,
        model_config=llama_70b,
        train_config=llama_70b_train_config_mk2,
        tags=["llama", "70b", "wsd", "exp750", "tootsie", "ema"],
        eval_harness_tasks=[],
        use_default_evaluation=False,
    ),
    override_output_path="checkpoints/llama-70b-tootsie-mk2",
)


if __name__ == "__main__":
    executor_main(
        [llama_70b_tootsie_mk2],
        description="Train 70B model on DCLM using WSD with EMA.",
    )


