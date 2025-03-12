"""
This is a continuation of 201's 70b but we are using a higher LR and switching to WSD with EMA
instead of WSD-S.

* Schedule: WSD, Decay is 40%
* Peak LR is 2e-4
* ema beta 0.995

Mix is still DCLM+Math+Code
"""

import dataclasses

from levanter.schedule import ScheduleStep

from experiments.defaults import default_train
from experiments.exp600_tootsie import dclm_mixture_config_llama3
from experiments.exp859_big_tootsies import dclm_mixture_config_llama3_zoned
from experiments.llama import llama_70b
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main

llama_70b_train_config_mk6 = SimpleTrainConfig(
    num_train_steps=1_000_000,  # we won't actually go this long. will adjust as needed
    # 3072 is slightly too big for us to fit on 6x v6e-128. 2048 would round up to 3 which fits, but 1536 works on 4
    # so it's safer.
    train_batch_size=[ScheduleStep(start=0, value=1024), ScheduleStep(start=96001, value=1536)],
    weight_decay=0.05,
    tpu_type="v6e-128",
    node_count=6,
    # LR doesn't support schedule yet
    # until 93_621, was 2e-4
    # learning_rate=2e-4,
    # until 95_920, was 2.5e-4 (on accident)
    # learning_rate=2.5e-4, # approx 2e-4 * sqrt(1.5)
    # learning_rate=2e-4,
    # bumping to 3.5e-4 at 96_001 because we're increasing the batch size
    learning_rate=3.5e-4,
    decay=0.4,
    ema_beta=0.995,
    lr_schedule="linear",
    warmup=1000,  # initial warmup
    cycle_length=None,
    allow_partial_checkpoint=True,
    allow_out_of_region_reads=True,
    allow_out_of_region_writes=False,
    steps_per_eval=1000,
    steps_per_task_eval=10000,
)

llama_70b_train_config_1536 = dataclasses.replace(
    llama_70b_train_config_mk6,
    train_batch_size=1536,  # 2 * 6 * 128
    tpu_type="v6e-128",
    node_count=6,
    learning_rate=2e-4,
    decay=0.4,
    ema_beta=0.995,
    lr_schedule="linear",
    cycle_length=None,
    allow_partial_checkpoint=True,
    allow_out_of_region_reads=True,
    allow_out_of_region_writes=False,
)


# this was a quick-ish experiment to compare 1536 batch size to 1024
# 1536 is better.
llama_70b_tootsie_bs1536 = dataclasses.replace(
    default_train(
        name="llama-bs1536-70b-tootsie",
        tokenized=dclm_mixture_config_llama3_zoned,
        model_config=llama_70b,
        train_config=llama_70b_train_config_1536,
        tags=["llama", "70b", "wsd", "exp750", "tootsie", "ema"],
        eval_harness_tasks=[],
    ),
    override_output_path="checkpoints/llama-bs1536-70b-tootsie",
)


llama_real_70b_tootsie = dataclasses.replace(
    default_train(
        name="llama-real-70b-tootsie",
        tokenized=dclm_mixture_config_llama3,
        model_config=llama_70b,
        train_config=llama_70b_train_config_mk6,
        tags=["llama", "70b", "wsd", "exp750", "tootsie", "ema"],
        eval_harness_tasks=[],
    ),
    override_output_path="checkpoints/llama-real-70b-tootsie",
)


if __name__ == "__main__":
    executor_main(
        [
            # llama_70b_tootsie_1536,
            llama_real_70b_tootsie,
        ],
        description="Train 70B model on DCLM using WSD with EMA.",
    )
