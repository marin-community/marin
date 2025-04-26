"""
This is @dlwh's "YOLO"/vibes run described in https://github.com/stanford-crfm/marin/issues/600.

The idea is/was to train a 8B model continuously updating the mixture, data, and anything else. With WSD-S,
there's no "middle" or "end" of the run, there's just the run. So we'll just train for a long time, updating as we go.

We call it "tootsie" because tootsie rolls are famously made by folding in the previous batch of tootsie roll into the
next batch, so we're folding in the previous mixture into the next mixture.

For now, we're training on DCLM's best mix, but that will change.
"""

# You will see in many, many places in this file that I (dlwh) made many, many mistakes.
# I'm leaving them in for posterity.

import dataclasses

from levanter.schedule import ScheduleStep

from experiments.cooldown_anneal import dolmino_dclm
from experiments.dclm.tokenize_dclm import DCLM_MIXTURE_WEIGHTS, dclm_components_llama3, dclm_mixture_config_llama3
from experiments.defaults import default_train
from experiments.dolma.tokenize_dolma import tokenize_dolma_steps
from experiments.dolmino.tokenize_dolmino import dolmino_math_tokenized_llama3, get_dolmino_step_llama3
from experiments.evals.task_configs import CORE_TASKS_PLUS_MMLU
from experiments.llama import llama3_tokenizer, llama_8b, llama_8b_old_rotary
from experiments.midtraining_datasets import finemath_3_plus_tokenized
from experiments.nemotron_cc.tokenize_nemotron import NEMOTRON_WEIGHTS, tokenize_nemotron_steps
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import lm_varying_mixture_data_config

# Phases/Runs in this file:
# 1. WSD-S on DCLM+Starcode+Proofpile on 2x v5litepod-256 (from scratch)
# 2. Switch to WSD with EMA on v4-2048 (from (1))
# 3. Cooldown v1: Switch to a 30% Dolmino-ish HQ dataset mixture, decay the LR (from (2))
# 4a. Tootsie Dessert (Attempt 1): Sprinkle in a bit of FLAN and Synth Math (from (3))
# 4b. Tootsie Dessert (Attempt 2): Fix the weights and add in all the HQ docs from dolmino (from (3))
# 5. Tootsie Cooldown v2: Another attempt at a final cooldown (from (2))
# 6. Tootsie rewarm: from (3), rewarmup and use mix of nemotron_cc and starcoder to keep moving


################################################################
# PHASE 1: WSD-S on DCLM+Starcode+Proofpile on 2x v5litepod-256
################################################################

# Initially, we start with WSD-S (cyclic stable/decay). The idea was to use WSD-S to train forever,
# but we have learned that WSD with a long cooldown is superior. Once we switch to WSD,
# we use the exponential moving average (EMA) of the model weights to get a better model.

tootise_phase1_config = SimpleTrainConfig(
    tpu_type="v5litepod-256",
    node_count=2,
    train_batch_size=1024,
    num_train_steps=1_000_000,  # using wsd-s so this doesn't really matter
    # these hypers from Table 12 in https://arxiv.org/html/2406.11794v1#A6
    learning_rate=1e-3,  # we get divergence with 2e-3
    weight_decay=0.05,
    # WSD-S
    cycle_length=10000,
    steps_per_eval=10000,
    steps_per_export=20000,
    warmup=1000,  # initial warmup
    decay=0.1,  # 10% of 5000 = 500 steps
    lr_schedule="inv",
)

llama_8b_tootsie_phase1 = dataclasses.replace(
    default_train(
        name="llama-8b-tootsie-0.001",
        tokenized=dclm_mixture_config_llama3,
        # I am a dummy and use old rotary config
        model_config=llama_8b_old_rotary,
        train_config=tootise_phase1_config,
        tags=["llama", "8b", "wsd-s", "exp600"],
    ),
    override_output_path="checkpoints/llama-8b-tootsie-0.001-19ad63",
)


# PHASE 2: We switch to WSD with EMA, moving to v4-2048 and increasing the batch size to better utilize the hardware.
# Because we increased the batch size, we need to increase the LR by \sqrt(ratio), which is ≈1.7x


llama_8b_train_config_phase2 = SimpleTrainConfig(
    tpu_type="v4-2048",
    node_count=1,
    num_train_steps=1_000_000,
    # after 660,600 we changed things up:
    train_batch_size=[ScheduleStep(start=0, value=1024), ScheduleStep(start=660_001, value=3072)],
    # LR doesn't (yet) support the schedule stuff so we just set it to the new value
    # because we're increasing the batch size, we need to increase the LR by \sqrt(ratio), which is ≈1.7x
    learning_rate=1.7e-3,
    # we're also switching to EMA because it's supposed to better than WSD-S
    decay=0.0,
    ema_beta=0.995,
    lr_schedule="linear",
    cycle_length=None,
    allow_partial_checkpoint=True,
    steps_per_eval=1000,
    steps_per_task_eval=10000,
    steps_per_export=20000,
    per_device_eval_parallelism=16,
)


llama_8b_tootsie_phase2 = dataclasses.replace(
    default_train(
        name="llama-8b-tootsie-phase2",
        tokenized=dclm_mixture_config_llama3,
        model_config=llama_8b,
        train_config=llama_8b_train_config_phase2,
        tags=["llama", "8b", "ema", "exp600"],
    ),
    override_output_path="checkpoints/lama-8b-tootsie-phase2",
)

# Note, we originally tried to fold the phase 3 mixture (below) into phase 2, but I messed up the hand off, so
# we made a new config. (Specifically, mixture schedules used to be in terms of samples not steps and I didn't
# account for that in the handoff. WandB doesn't work if you try to overwrite steps so we just made a new run.)

################################################################
# PHASE 3 (v1): We switch to a new mixture, decay the LR
################################################################
# This mixture is basically a subset of dolmino.
# At this time, some of us had prior experience with FLAN
# that suggested it was not great and we were a bit
# leery of the very specific synth math data, so we left those parts out.

# main phase: base mix for 740,500 steps
# more dclm out to ≈3.78e+12 tokens (740,500 total steps)
# dolmino-ish mixture out to ≈4.78e+12 tokens (820,000 steps)

PHASE_3_START = 740_500
PHASE_3_END = 820_000
DECAY_FRACTION = (PHASE_3_END - PHASE_3_START) / PHASE_3_END

# This is basically the same as the phase 2 train config, but we
# add DECAY_FRACTION.
llama_8b_train_config_phase3 = SimpleTrainConfig(
    tpu_type="v4-2048",
    node_count=1,
    num_train_steps=PHASE_3_END,
    # From Phase 2:
    train_batch_size=[ScheduleStep(start=0, value=1024), ScheduleStep(start=660_001, value=3072)],
    learning_rate=1.7e-3,
    decay=DECAY_FRACTION,
    ema_beta=0.995,
    lr_schedule="linear",
    cycle_length=None,
    allow_partial_checkpoint=True,
    steps_per_eval=1000,
    steps_per_task_eval=10000,
    steps_per_export=20000,
    per_device_eval_parallelism=16,
)

# Data for phase 3 consists of three parts:
#
# "Web" which is DCLM HQ + Starcoder + Proofpile
# "HQ" which is a mix of dolmino and dolma datasets
# (DCLM HQ refers to the dolmino subset of DCLM)

phase_3_tokenized = {**dclm_components_llama3}

dolma_splits = [
    "dolma/algebraic-stack",
    "dolma/arxiv",
    "dolma/megawika",
    "dolma/open-web-math",
    "dolma/pes2o",
    "dolma/stackexchange",
    "dolma/wiki",
]
all_dolma_steps = tokenize_dolma_steps(tokenizer=llama3_tokenizer)
phase_3_tokenized.update({dataset: step for dataset, step in all_dolma_steps.items() if dataset in dolma_splits})
phase_3_tokenized["finemath_3_plus"] = finemath_3_plus_tokenized
# phase_3_tokenized["fineweb_edu"] = fineweb_edu_tokenized
phase_3_tokenized["dolmino_dclm"] = dolmino_dclm


# Dolma counts are done with llama 3 tokens (https://docs.google.com/spreadsheets/d/1ykVJ1EGJvA1zwF67FZGFBzlm7P0ZBIMuCpBW9Pqp7cY/edit?gid=0#gid=0)
# This is slightly different from olmo tokenizer token counts
# The first number is the number of tokens in the dataset, the second is the desired mixing portion
high_quality_token_counts = {
    "dolma/algebraic-stack": 11.5 * 1.0,
    "dolma/arxiv": 27.9 * 1.0,
    "dolma/megawika": 4.44 * 1.0,
    "dolma/open-web-math": 5.06 * 1.0,
    "dolma/pes2o": 58.1 * 1.0,
    "dolma/stackexchange": 17.1 * 1.0,
    "dolma/wiki": 3.65 * 1.0,
    "finemath_3_plus": 34.0 * 1.0,  # https://huggingface.co/datasets/HuggingFaceTB/finemath
}

total_high_quality_token_count = sum(high_quality_token_counts.values())
# total HQ token count is ≈ 161.7B
# we're training for 1T tokens or so.
# we'd like to keep the HQ data to ≈2 epochs

HQ_WEIGHT = 30.0
WEB_WEIGHT = 70.0

# dolmino dclm is about 700B tokens (llama 3)
# starcoder we've seen, but I don't want to exclude all coding from the final mix

web_counts = {
    "dolmino_dclm": 700.0 * 1.0,
    "starcoderdata": 230.0 * 0.1,
}

total_web_token_count = sum(web_counts.values())

# reweight data so that 30% are high-quality sources and 70% are dclm+other
cooldown_mixture_weights_v1 = {
    **{
        dataset: HQ_WEIGHT * token_count / total_high_quality_token_count
        for dataset, token_count in high_quality_token_counts.items()
    },
    **{dataset: WEB_WEIGHT * token_count / total_web_token_count for dataset, token_count in web_counts.items()},
}

phase_3_data_mixture = lm_varying_mixture_data_config(
    components=phase_3_tokenized,
    weights_list=[
        (0, DCLM_MIXTURE_WEIGHTS),
        (PHASE_3_START, cooldown_mixture_weights_v1),
    ],
)

llama_8b_tootsie_phase3 = dataclasses.replace(
    default_train(
        name="llama-8b-tootsie-phase3",
        tokenized=phase_3_data_mixture,
        model_config=llama_8b,
        train_config=llama_8b_train_config_phase3,
        tags=["llama", "8b", "ema", "exp600"],
        # eval_harness_tasks=[],
    ),
    override_output_path="checkpoints/llama-8b-tootsie-phase3",
)

################################################################
## Tootsie Dessert (Attempt 1): Sprinkle in a bit of flan and synth math
################################################################
# Motivation: our ablations found that we needed more math and more task-prep data (e.g. FLAN) so we're going to
# add in more of that. We're already cooled down but the LR is actually still pretty high (1.7e-4) so we're going to
# just coast.

## Add in:
## FLAN
## DolminoSynthMath
## TuluMath
## GSM8K
## MathCoder2
## Metamath-owm-filteR
## CodeSearchNet-owmfilter
## TinyGSM-Mind

dessert_dolmino_sets = {
    s: get_dolmino_step_llama3(s)
    for s in [
        "flan",
        "math/dolmino_math_synth",
        "math/tulu_math",
        "math/gsm8k",
        "math/mathcoder2-synthmath",
        "math/metamath-owmfilter",
        "math/codesearchnet-owmfilter",
    ]
}

approx_dessert_sizes = {
    "flan": 17e9,
    "math/dolmino_math_synth": 28.7e6,
    "math/tulu_math": 230e6,
    "math/gsm8k": 2.74e6,
    "math/mathcoder2-synthmath": 3.87e9,
    "math/metamath-owmfilter": 84.2e6,
    "math/codesearchnet-owmfilter": 1.8e6,
}

# about 21.2e9
total_dessert_size = sum(approx_dessert_sizes.values())

DESSERT_WEB = 0.7  # DCLM HQ + Starcoder
DESSERT_HQ = 0.2  # Same as before
DESSERT_DESSERT = 0.1  # FLAN + Math

## Dessert Attempt 1: This has a bug where I swapped the HQ and Dessert weights.
# Also, the math sets are so small they don't get picked up

# I'm such a dummy: I swapped HQ and Dessert weights
bad_dessert_weights_v1 = {
    **{dataset: DESSERT_HQ * size / total_dessert_size for dataset, size in approx_dessert_sizes.items()},
    **{dataset: DESSERT_WEB * size / total_web_token_count for dataset, size in web_counts.items()},
    **{
        dataset: DESSERT_DESSERT * size / total_high_quality_token_count
        for dataset, size in high_quality_token_counts.items()
    },
}

dessert_tokenized = {**phase_3_tokenized, **dessert_dolmino_sets}

bad_dessert_data_mixture_v1 = lm_varying_mixture_data_config(
    components=dessert_tokenized,
    weights_list=[
        (0, DCLM_MIXTURE_WEIGHTS),
        (PHASE_3_START, cooldown_mixture_weights_v1),
        (PHASE_3_END, bad_dessert_weights_v1),
    ],
)

# we're aiming to do 1 pass through the new mixes, which is another ~212e9 tokens
# 3072 * 4096 tokens per step = 12.6e6 tokens per step
# so ~17000 steps is about right

DESSERT_END = PHASE_3_END + 17000

llama_8b_train_config_dessert = SimpleTrainConfig(
    tpu_type="v4-2048",
    node_count=1,
    num_train_steps=DESSERT_END,
    train_batch_size=[ScheduleStep(start=0, value=1024), ScheduleStep(start=660_001, value=3072)],
    # coast along at 1.7e-4
    learning_rate=1.7e-4,
    decay=0.0,  # we're already at the lowest we want to go
    ema_beta=0.995,
    lr_schedule="linear",
    cycle_length=None,
    allow_partial_checkpoint=True,
    steps_per_eval=1000,
    steps_per_task_eval=1000000,
    # only export last step (which is forced)
    steps_per_export=2000000,
    per_device_eval_parallelism=16,
)

# BAD, Don't use this. Here for documentation purposes.
llama_8b_tootsie_dessert_BAD = dataclasses.replace(
    default_train(
        name="llama-8b-tootsie-dessert",
        tokenized=bad_dessert_data_mixture_v1,
        model_config=llama_8b,
        train_config=llama_8b_train_config_dessert,
        tags=["llama", "8b", "ema", "exp600"],
    ),
    override_output_path="checkpoints/llama-8b-tootsie-dessert",
)


# attempt 2: I had swapped HQ and Dessert weights. Also, the math sets are so small they don't get picked up
# with the block size we use. So we concat the math sets into a single set and weight them as a single set.

# I'm such a dummy
dessert_weights_v2 = {
    **{dataset: DESSERT_WEB * size / total_web_token_count for dataset, size in web_counts.items()},
    **{
        dataset: DESSERT_HQ * size / total_high_quality_token_count
        for dataset, size in high_quality_token_counts.items()
    },
    "flan": DESSERT_DESSERT * approx_dessert_sizes["flan"] / total_dessert_size,
    "all_math": (
        DESSERT_DESSERT
        * sum(size for dataset, size in approx_dessert_sizes.items() if "math" in dataset)
        / total_dessert_size
    ),
}

all_math = dolmino_math_tokenized_llama3

dessert_tokenized_v2 = {
    **phase_3_tokenized,
    "flan": dessert_tokenized["flan"],
    "all_math": all_math,
}

dessert_data_mixture_v3 = lm_varying_mixture_data_config(
    components=dessert_tokenized_v2,
    weights_list=[
        (0, DCLM_MIXTURE_WEIGHTS),
        (PHASE_3_START, cooldown_mixture_weights_v1),
        (PHASE_3_END, dessert_weights_v2),
    ],
)

llama_8b_tootsie_dessert_v3 = dataclasses.replace(
    default_train(
        name="llama-8b-tootsie-dessert-v3",
        tokenized=dessert_data_mixture_v3,
        model_config=llama_8b,
        train_config=llama_8b_train_config_dessert,
        tags=["llama", "8b", "ema", "exp600"],
        eval_harness_tasks=[],
    ),
    override_output_path="checkpoints/llama-8b-tootsie-dessert-v3",
)

################################################################
## Tootsie Cooldown v2: Another attempt at a final cooldown
################################################################

# ok the attempts at "dessert" were a failure.
# We're going to try again with a new mixture starting at the same point as the first cooldown.
# This mixture is basically adding in all the HQ docs from dolmino, though still leaving out GSM-MIND

# Our ablations found that 5% FLAN for a 1T run would likely lead to the best results. This epochs
# FLAN about 4 times.

WEB_WEIGHT_V2 = 0.7
HQ_WEIGHT_V2 = 0.25
FLAN_WEIGHT_V2 = 0.05


# HQ docs are the same as before, but adding in some math
high_quality_token_counts_v2 = {
    **high_quality_token_counts,
    "all_math": sum(size for dataset, size in approx_dessert_sizes.items() if "math" in dataset) / 1e9,
}

total_high_quality_token_count_v2 = sum(high_quality_token_counts_v2.values())

cooldown_mixture_weights_v2 = {
    **{
        dataset: HQ_WEIGHT_V2 * token_count / total_high_quality_token_count_v2
        for dataset, token_count in high_quality_token_counts_v2.items()
    },
    **{dataset: WEB_WEIGHT_V2 * token_count / total_web_token_count for dataset, token_count in web_counts.items()},
    "flan": FLAN_WEIGHT_V2,
}

# sanity checks because I've been burned too many times:
# we should add up to about 1.0
assert 0.99 < sum(cooldown_mixture_weights_v2.values()) < 1.01
# none of the HQ docs should be more than ~11% (Pes2o is about 10.3%) or less than 0.5%
assert all(0.005 < w < 0.11 for k, w in cooldown_mixture_weights_v2.items() if k in high_quality_token_counts_v2)

cooldown_components_v2 = {**phase_3_tokenized, "flan": dessert_tokenized["flan"], "all_math": all_math}

cooldown_config_v2 = lm_varying_mixture_data_config(
    components=cooldown_components_v2,
    weights_list=[
        (0, DCLM_MIXTURE_WEIGHTS),
        (PHASE_3_START, cooldown_mixture_weights_v2),
    ],
)

# deliberately using same number of steps as the previous run
# This means we're doing slightly fewer effective passes through the data, but it feels more science-y
# to keep the number of steps the same.

llama_8b_tootsie_cooldown_v2 = dataclasses.replace(
    default_train(
        name="llama-8b-tootsie-cooldown-take-2",
        tokenized=cooldown_config_v2,
        model_config=llama_8b,
        train_config=llama_8b_train_config_phase3,
        tags=["llama", "8b", "ema", "exp600"],
        eval_harness_tasks=[],
    ),
    override_output_path="checkpoints/llama-8b-tootsie-cooldown-take-2",
)


## Tootsie rewarm: from (3), rewarmup and use mix of nemotron_cc and starcoder to keep moving

# We're going to try to keep moving by rewarming the model with a mix of nemotron_cc and starcoder.

PHASE_4_START = PHASE_3_END
PHASE_4_END = 2_000_000
# ramp up to 1.7e-3 over 2k steps
PHASE_4_REWARMUP_DURATION = 2000


nemotron_cc_steps = tokenize_nemotron_steps()

# Nemotron weights are in compressed TiB. We'll use the rule of thumb that compressed bytes ≈ tokens

phase_4_steady_state_weights = {
    **NEMOTRON_WEIGHTS,
    "starcoderdata": 0.25,  # 250B tokens
}

# We bridge the mixture from the end of the cooldown to the steady state mixture. We used a mixture that was
# roughly proportional to token count for each phase.
phase_4_warmup_weights = {
    **{k: v for k, v in DCLM_MIXTURE_WEIGHTS.items()},
    **{k: v for k, v in phase_4_steady_state_weights.items()},
}

llama_8b_train_config_phase4 = dataclasses.replace(
    llama_8b_train_config_phase3,
    num_train_steps=PHASE_4_END,
    learning_rate=1.7e-3,
    lr_schedule="linear",
    decay=DECAY_FRACTION,
    # use the WSD-S api to do the re-warmup
    cycle_length=[PHASE_3_END, (PHASE_4_END - PHASE_3_END)],
    rewarmup=PHASE_4_REWARMUP_DURATION,
)

phase_4_data_mixture = lm_varying_mixture_data_config(
    components={**phase_3_tokenized, **nemotron_cc_steps},
    weights_list=[
        (0, DCLM_MIXTURE_WEIGHTS),
        (PHASE_3_START, cooldown_mixture_weights_v1),
        (PHASE_4_START, phase_4_warmup_weights),
        (PHASE_4_START + PHASE_4_REWARMUP_DURATION, phase_4_steady_state_weights),
    ],
)

llama_8b_tootsie_adept_phoenix = dataclasses.replace(
    default_train(
        name="llama-8b-tootsie-adept-phoenix",
        tokenized=phase_4_data_mixture,
        model_config=llama_8b,
        train_config=llama_8b_train_config_phase4,
        tags=["llama", "8b", "ema", "exp600"],
        eval_harness_tasks=CORE_TASKS_PLUS_MMLU,
    ),
    override_output_path="checkpoints/llama-8b-tootsie-adept-phoenix",
)

# little bit of sanity check code to make sure the LR schedules line up
# levanter_train_config_old = llama_8b_tootsie_phase3.config
# lr_schedule_old = levanter_train_config_old.optimizer.lr_scheduler(PHASE_3_END)
# levanter_train_config = llama_8b_tootsie_adept_phoenix.config
# lr_schedule = levanter_train_config.optimizer.lr_scheduler(PHASE_4_END)
#
# # plot the entire LR schedule for phase 3 and phase 4 and make sure they line up
# import matplotlib.pyplot as plt
#
# # offset old by a little bit so i can see it
# plt.plot(range(0, PHASE_3_END, 10), [lr_schedule_old(x) - 0.5e-4 for x in range(0, PHASE_3_END, 10)])
# plt.plot(range(0, PHASE_4_END, 10), [lr_schedule(x) for x in range(0, PHASE_4_END, 10)])
# plt.show()


if __name__ == "__main__":
    executor_main(
        steps=[
            llama_8b_tootsie_phase1,
            llama_8b_tootsie_phase3,
            llama_8b_tootsie_dessert_BAD,
            llama_8b_tootsie_dessert_v3,
            llama_8b_tootsie_cooldown_v2,
            llama_8b_tootsie_adept_phoenix,
        ],
        description="Train 8B model on DCLM using WSD-S, then switching to EMA with a new mixture.",
    )
