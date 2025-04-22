"""
These are larger versions of @dlwh's "YOLO"/vibes run described in https://github.com/stanford-crfm/marin/issues/600.

Initially, these were just testing runs, since we didn't know if we'd actually have the capacity for any length of time.
Turns out we did and they seem pretty decent.

The first phase is WSD-S on the same mixture as tootsie 8b.
The second phase is EMA on the same mixture as tootsie 8b, with an increased batch size.

Note: The 22B model is actually a 24B model, but we're going to keep calling it 22B for consistency.

Also buried in here is a 56B model that I thought was a 70B model. Always double-check your config, kids.
"""

import dataclasses

from levanter.models.rotary import DefaultRotaryEmbeddingsConfig
from levanter.schedule import ScheduleStep

from experiments.dclm.tokenize_dclm import DCLM_MIXTURE_WEIGHTS, dclm_components_llama3, dclm_mixture_config_llama3
from experiments.defaults import default_train
from experiments.llama import llama_13b, llama_24b, llama_56b
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.processing.tokenize import lm_mixture_data_config

# data

MARIN_CENTRAL_DCLM_COMPONENTS = {
    "dclm_baseline": "gs://marin-us-central2/tokenized/dclm_baseline-0206f1",
    "starcoderdata": "gs://marin-us-central2/tokenized/starcoderdata-12f018/",
    "proofpile_2": "gs://marin-us-central2/tokenized/proofpile_2-4a35c7",
}
dclm_components_zoned = {
    name: dataclasses.replace(
        step,
        override_output_path=MARIN_CENTRAL_DCLM_COMPONENTS[name],
    )
    for name, step in dclm_components_llama3.items()
}

#####
## Phase 1 for 13B, 22B: WSD-S on same mixture as tootsie 8b
#####

# initially we were using the wrong rotary, just like tootsie 8b. Oh well.
llama_13b_old_rotary = dataclasses.replace(llama_13b, rope=DefaultRotaryEmbeddingsConfig())
llama_24b_old_rotary = dataclasses.replace(llama_24b, rope=DefaultRotaryEmbeddingsConfig())

## Initial 13B config for the first phase
llama_13b_train_config = SimpleTrainConfig(
    tpu_type="v6e-64",
    node_count=4,
    train_batch_size=1024,
    num_train_steps=1_000_000,  # using wsd-s so this doesn't really matter
    learning_rate=3e-4,
    weight_decay=0.05,
    # WSD-S
    cycle_length=10000,
    steps_per_eval=10000,
    steps_per_export=20000,
    warmup=1000,  # initial warmup
    # TODO: do we need rewarmup
    decay=0.1,  # 10% of 10000 = 500 steps
    lr_schedule="inv",
)

## Initial "22B" config for the first phase
llama_22b_train_config = SimpleTrainConfig(
    tpu_type="v6e-256",
    node_count=2,
    train_batch_size=1024,
    num_train_steps=1_000_000,  # using wsd-s so this doesn't really matter
    learning_rate=3e-4,
    weight_decay=0.05,
    # WSD-S
    cycle_length=10000,
    steps_per_eval=10000,
    steps_per_export=20000,
    warmup=1000,  # initial warmup
    # TODO: do we need rewarmup
    decay=0.1,  # 10% of 10000 = 500 steps
    lr_schedule="inv",
)

# We didn't know if we'd actually have this capacity for any length of time,
# so they were initially just "testing" runs.
llama_13b_tootsie_phase1 = default_train(
    name="llama-13b-tootsie-dummy-testing",
    tokenized=dclm_mixture_config_llama3,
    model_config=llama_13b_old_rotary,
    train_config=llama_13b_train_config,
    tags=["llama", "13b", "wsd-s", "exp201", "tootsie"],
    eval_harness_tasks=[],
)

llama_22b_tootsie_phase1 = default_train(
    name="llama-22b-tootsie-dummy-testing",
    tokenized=dclm_mixture_config_llama3,
    model_config=llama_24b_old_rotary,
    train_config=llama_22b_train_config,
    tags=["llama", "22b", "wsd-s", "exp201", "tootsie"],
    eval_harness_tasks=[],
)

#####
# Phase 2 for 13B, 22B: EMA on same mixture as tootsie 8b, increased batch size
#####

llama_13b_train_config_ema = SimpleTrainConfig(
    tpu_type="v6e-64",
    node_count=7,
    train_batch_size=[ScheduleStep(start=0, value=1024), ScheduleStep(start=280_000, value=3072)],
    num_train_steps=1_000_000,
    weight_decay=0.05,
    learning_rate=4.2e-4,  # 3e-4 * 1.4
    decay=0.4,
    ema_beta=0.995,
    lr_schedule="linear",
    cycle_length=None,
    allow_partial_checkpoint=True,
)

# 22b warmstart, switching to EMA
llama_22b_train_config_ema = SimpleTrainConfig(
    tpu_type="v6e-128",
    node_count=4,
    # train_batch_size=1024,
    train_batch_size=[ScheduleStep(start=0, value=1024), ScheduleStep(start=200_000, value=3072)],
    num_train_steps=1_000_000,
    weight_decay=0.05,
    learning_rate=4.2e-4,  # 3e-4 * 1.4
    decay=0.4,
    ema_beta=0.995,
    lr_schedule="linear",
    cycle_length=None,
    allow_partial_checkpoint=True,
    steps_per_eval=1000,
    steps_per_task_eval=10000,
)

dclm_mixture_config_llama3_zoned = lm_mixture_data_config(
    components=dclm_components_zoned, weights=DCLM_MIXTURE_WEIGHTS, include_raw_paths=False
)
llama_13b_tootsie_ema_warmstart = dataclasses.replace(
    default_train(
        name="llama-13b-tootsie-ema-mk2",
        tokenized=dclm_mixture_config_llama3_zoned,
        model_config=llama_13b,
        train_config=llama_13b_train_config_ema,
        tags=["llama", "13b", "ema", "exp201", "tootsie"],
        eval_harness_tasks=[],
    ),
    override_output_path="checkpoints/llama-13b-tootsie-ema-mk2",
)

# warmstarted from llama_22b_tootsie at 200,000
llama_22b_tootsie_ema_warmstart = dataclasses.replace(
    default_train(
        name="llama-22b-tootsie-ema-mk2",
        tokenized=dclm_mixture_config_llama3_zoned,
        model_config=llama_24b,
        train_config=llama_22b_train_config_ema,
        tags=["llama", "22b", "ema", "exp201", "tootsie"],
        eval_harness_tasks=[],
    ),
    override_output_path="checkpoints/llama-22b-tootsie-ema-mk2",
)

#####
# sigh... 56B. you can ignore this.
#####
llama_56b_train_config = SimpleTrainConfig(
    tpu_type="v6e-256",
    node_count=2,
    train_batch_size=1024,
    num_train_steps=1_000_000,  # using wsd-s so this doesn't really matter
    learning_rate=3e-5,
    weight_decay=0.05,
    # WSD-S
    cycle_length=10000,
    steps_per_eval=10000,
    steps_per_export=20000,
    warmup=1000,  # initial warmup
    # TODO: do we need rewarmup
    decay=0.1,  # 10% of 10000 = 500 steps
    lr_schedule="inv",
)


# All of all of these are 56B models but were intended to be 70b. Sigh.
llama_56b_train_config_mk2 = dataclasses.replace(
    llama_56b_train_config,
    train_batch_size=1024,
    tpu_type="v4-2048",
    node_count=1,
    learning_rate=2e-4,
    decay=0.4,
    ema_beta=0.995,
    lr_schedule="linear",
    cycle_length=None,
    allow_partial_checkpoint=True,
)


# actual 56B model
llama_70b_tootsie_mk2_BAD = dataclasses.replace(
    default_train(
        name="llama-70b-tootsie-mk2",
        # not recorded here:
        # warmstart weights from llama_70b_tootsie step 80000
        tokenized=dclm_mixture_config_llama3,
        model_config=llama_56b,
        train_config=llama_56b_train_config_mk2,
        tags=["llama", "70b", "wsd", "exp750", "tootsie", "ema"],
        eval_harness_tasks=[],
    ),
    override_output_path="checkpoints/llama-70b-tootsie-mk2",
)


llama_56b_tootsie = default_train(
    name="llama-70b-tootsie-dummy-testing",
    tokenized=dclm_mixture_config_llama3,
    model_config=llama_56b,
    train_config=llama_56b_train_config,
    tags=["llama", "70b", "wsd-s", "exp201", "tootsie"],
    eval_harness_tasks=[],
)


if __name__ == "__main__":
    executor_main(
        steps=[
            llama_13b_tootsie_phase1,
            llama_22b_tootsie_phase1,
            llama_13b_tootsie_ema_warmstart,
            llama_22b_tootsie_ema_warmstart,
        ],
        description="Train some models on DCLM using WSD-S, switching to EMA.",
    )
