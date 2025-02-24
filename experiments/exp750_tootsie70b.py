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

from experiments.dclm.exp433_dclm_run import DCLM_MIXTURE_WEIGHTS
from experiments.dclm.tokenize_dclm import dclm_components_llama3
from experiments.defaults import default_train
from experiments.exp201_tootsie22b import llama_56b, llama_70b_train_config
from experiments.exp600_tootsie import dclm_mixture_config_llama3
from experiments.llama import llama_13b, llama_22b, llama_70b
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.processing.tokenize import lm_mixture_data_config

# I was an idiot and all of these are 56B models but were intended to be 70b. Sigh.

llama_70b_train_config_mk2 = dataclasses.replace(
    llama_70b_train_config,
    train_batch_size=1024,
    tpu_type="v4-2048",
    node_count=1,
    learning_rate=2e-4,
    decay=0.4,
    ema_beta=0.995,
    lr_schedule="linear",
    cycle_length=None,
    allow_partial_checkpoint=True,
    allow_out_of_region_reads=True,
    allow_out_of_region_writes=False,
)

llama_70b_train_config_mk4 = dataclasses.replace(
    llama_70b_train_config,
    train_batch_size=1024,
    tpu_type="v6e-128",
    node_count=4,
    learning_rate=2e-4,
    decay=0.4,
    ema_beta=0.995,
    lr_schedule="linear",
    cycle_length=None,
    allow_partial_checkpoint=True,
    allow_out_of_region_reads=True,
    allow_out_of_region_writes=False,
)


llama_70b_train_config_mk6 = dataclasses.replace(
    llama_70b_train_config,
    # 3072 is slightly too big for us to fit on 6x v6e-128. 2048 would round up to 3 which fits, but 1536 works on 4
    # so it's safer.
    train_batch_size=[ScheduleStep(start=0, value=1024), ScheduleStep(start=96001, value=1536)],
    tpu_type="v6e-128",
    # tpu_type="v5litepod-256",
    node_count=4,
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
    cycle_length=None,
    allow_partial_checkpoint=True,
    allow_out_of_region_reads=True,
    allow_out_of_region_writes=False,
    steps_per_eval=1000,
    steps_per_task_eval=10000,
)

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

dclm_mixture_config_llama3_zoned = lm_mixture_data_config(
    components=dclm_components_zoned, weights=DCLM_MIXTURE_WEIGHTS, include_raw_paths=False
)


llama_70b_tootsie_mk2 = dataclasses.replace(
    default_train(
        name="llama-70b-tootsie-mk2",
        # not recorded here:
        # warmstart weights from llama_70b_tootsie step 80000
        tokenized=dclm_mixture_config_llama3,
        model_config=llama_56b,
        train_config=llama_70b_train_config_mk2,
        tags=["llama", "70b", "wsd", "exp750", "tootsie", "ema"],
        eval_harness_tasks=[],
    ),
    override_output_path="checkpoints/llama-70b-tootsie-mk2",
)


llama_70b_tootsie_mk4 = dataclasses.replace(
    default_train(
        name="llama-70b-tootsie-mk4",
        # not recorded here:
        # warmstart weights from llama_70b_tootsie step 80000
        tokenized=dclm_mixture_config_llama3_zoned,
        model_config=llama_56b,
        train_config=llama_70b_train_config_mk4,
        tags=["llama", "70b", "wsd", "exp750", "tootsie", "ema"],
        eval_harness_tasks=[],
    ),
    override_output_path="checkpoints/llama-70b-tootsie-mk4",
)


llama_70b_tootsie_mk6 = dataclasses.replace(
    default_train(
        name="llama-70b-tootsie-mk6",
        # not recorded here:
        # warmstart weights from llama_70b_tootsie step 87613
        tokenized=dclm_mixture_config_llama3,
        model_config=llama_56b,
        train_config=llama_70b_train_config_mk6,
        tags=["llama", "70b", "wsd", "exp750", "tootsie", "ema"],
        eval_harness_tasks=[],
    ),
    override_output_path="checkpoints/llama-70b-tootsie-mk6",
)


llama_70b_train_config_1536 = dataclasses.replace(
    llama_70b_train_config,
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


llama_70b_tootsie_1536 = dataclasses.replace(
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


llama_70b_tootsie_warmstart = dataclasses.replace(
    default_train(
        name="llama-warmstart-70b-tootsie",
        tokenized=dclm_mixture_config_llama3,
        model_config=llama_70b,
        train_config=llama_70b_train_config_mk6,
        tags=["llama", "70b", "wsd", "exp750", "tootsie", "ema"],
        eval_harness_tasks=[],
    ),
    override_output_path="checkpoints/llama-warmstart-70b-tootsie",
)

## 22b warmstart with similar config

# warmstarted from llama_22b_train_config at 200,000
llama_22b_train_config_ema = SimpleTrainConfig(
    tpu_type="v6e-128",
    node_count=8,
    # train_batch_size=1024,
    train_batch_size=[ScheduleStep(start=0, value=1024), ScheduleStep(start=200_000, value=3072)],
    num_train_steps=1_000_000,
    weight_decay=0.05,
    learning_rate=4.2e-4,  # 3e-4 * 1.4
    decay=0.4,
    ema_beta=0.995,
    lr_schedule="linear",
    cycle_length=None,
    allow_out_of_region_reads=True,
    allow_out_of_region_writes=False,
    allow_partial_checkpoint=True,
    steps_per_eval=1000,
    steps_per_task_eval=10000,
)


llama_22b_tootsie_ema_warmstart = dataclasses.replace(
    default_train(
        name="llama-22b-tootsie-ema-mk2",
        tokenized=dclm_mixture_config_llama3_zoned,
        model_config=llama_22b,
        train_config=llama_22b_train_config_ema,
        tags=["llama", "22b", "ema", "exp201", "tootsie"],
        eval_harness_tasks=[],
    ),
    override_output_path="checkpoints/llama-22b-tootsie-ema-mk2",
)


llama_13b_train_config_ema = SimpleTrainConfig(
    tpu_type="v6e-128",
    node_count=3,
    train_batch_size=[ScheduleStep(start=0, value=1024), ScheduleStep(start=280_000, value=3072)],
    num_train_steps=1_000_000,
    weight_decay=0.05,
    learning_rate=4.2e-4,  # 3e-4 * 1.4
    decay=0.4,
    ema_beta=0.995,
    lr_schedule="linear",
    cycle_length=None,
    allow_out_of_region_reads=True,
    allow_out_of_region_writes=False,
    allow_partial_checkpoint=True,
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


if __name__ == "__main__":
    executor_main(
        [
            llama_70b_tootsie_1536,
            llama_real_70b_tootsie,
            llama_22b_tootsie_ema_warmstart,
            llama_13b_tootsie_ema_warmstart,
        ],
        description="Train 70B model on DCLM using WSD with EMA.",
    )
