"""
This is @dlwh's "YOLO"/vibes run described in https://github.com/stanford-crfm/marin/issues/600.

The idea is to train a 8B model continuously updating the mixture, data, and anything else. With WSD-S,
there's no "middle" or "end" of the run, there's just the run. So we'll just train for a long time, updating as we go.

We call it "tootsie" because tootsie rolls are famously made by folding in the previous batch of tootsie roll into the
next batch, so we're folding in the previous mixture into the next mixture.

For now, we're training on DCLM's best mix, but that will change.
"""

import dataclasses

from levanter.schedule import ScheduleStep

from experiments.cooldown_anneal import dolmino_dclm
from experiments.dclm.tokenize_dclm import DCLM_MIXTURE_WEIGHTS, dclm_components_llama3, dclm_mixture_config_llama3
from experiments.defaults import default_tokenize, default_train
from experiments.dolma.tokenize_dolma import tokenize_dolma_steps
from experiments.dolmino.tokenize_dolmino import dolmino_math_tokenized_llama3, get_dolmino_step
from experiments.llama import llama3_tokenizer, llama_8b, llama_8b_old_rotary
from experiments.midtraining_datasets import finemath_3_plus_tokenized
from experiments.pretraining_datasets import dclm_baseline_wrong, proofpile_2, starcoderdata
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.processing.tokenize import lm_mixture_data_config
from marin.processing.tokenize.data_configs import lm_varying_mixture_data_config

## NOTE: on 20250211, we discovered that the DCLM baseline data in us-central2 was corrupted/partial.
# These are preserved for reproducibility, but future runs should use the correct data.
dclm_components_llama3_wrong = {
    "dclm_baseline": dataclasses.replace(
        default_tokenize(
            name="dclm_baseline",
            dataset=dclm_baseline_wrong,
            tokenizer=llama3_tokenizer,
        ),
        override_output_path="gs://marin-us-central2/tokenized/dclm_baseline-0206f1_WRONG_20250211/",
    ),
    "starcoderdata": default_tokenize(
        name="starcoderdata", dataset=starcoderdata, tokenizer=llama3_tokenizer, text_key="content"
    ),
    "proofpile_2": default_tokenize(
        name="proofpile_2",
        dataset=proofpile_2,
        tokenizer=llama3_tokenizer,
    ),
}

dclm_mixture_config_llama3_wrong = lm_mixture_data_config(
    components=dclm_components_llama3_wrong, weights=DCLM_MIXTURE_WEIGHTS
)

llama_8b_train_config = SimpleTrainConfig(
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
    # TODO: do we need rewarmup
    decay=0.1,  # 10% of 5000 = 500 steps
    lr_schedule="inv",
)



llama_8b_tootsie = dataclasses.replace(
    default_train(
        name="llama-8b-tootsie-0.001",
        tokenized=dclm_mixture_config_llama3,
        # I am a dummy and use old rotary config
        model_config=llama_8b_old_rotary,
        train_config=llama_8b_train_config,
        tags=["llama", "8b", "wsd-s", "exp600"],
    ),
    override_output_path="checkpoints/llama-8b-tootsie-0.001-19ad63",
)

# phase 3 data is a variant of the dolmino mix

# main phase: raw dclm for 660,000 steps
# phase 2 is divided into two subparts (lolcry):
# more dclm out to ≈3.78e+12 tokens (740,000 total steps)
# dolmino-ish mixture out to ≈4.78e+12 tokens (820,000 steps)

PHASE_3_END = 820_000
PHASE_3 = 740_500
DECAY_FRACTION = (PHASE_3_END - PHASE_3) / PHASE_3_END

llama_8b_train_config_phase3 = SimpleTrainConfig(
    # tpu_type="v5litepod-256",
    # node_count=2,
    tpu_type="v4-2048",
    node_count=1,
    num_train_steps=PHASE_3_END,
    # after 660,600 we changed things up:
    train_batch_size=[ScheduleStep(start=0, value=1024), ScheduleStep(start=660_001, value=3072)],
    # LR doesn't (yet) support the schedule stuff so we just set it to the new value
    # because we're increasing the batch size, we need to increase the LR by \sqrt(ratio), which is ≈1.7x
    learning_rate=1.7e-3,
    # we're also switching to EMA because it's supposed to better than WSD-S
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


# Dolma counts are done with llama tokens (https://docs.google.com/spreadsheets/d/1ykVJ1EGJvA1zwF67FZGFBzlm7P0ZBIMuCpBW9Pqp7cY/edit?gid=0#gid=0)
# This is slightly different from standard olmo tokenizer token counts
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

# dolmino dclm is about 700B tokens (llama 3)
# fineweb edu is ~1.1T tokens (llama 3), but lower quality so skip
# starcoder we've seen, but I don't want to exclude all coding from the final mix

web_counts = {
    "dolmino_dclm": 700.0 * 1.0,
    # "fineweb_edu": 1100 * 0.05,
    "starcoderdata": 230.0 * 0.1,
}

total_web_token_count = sum(web_counts.values())


# reweight data so that 30% are high-quality sources and 70% are dclm+other
cooldown_mixture_weights = {
    **{
        dataset: HQ_WEIGHT * token_count / total_high_quality_token_count
        for dataset, token_count in high_quality_token_counts.items()
    },
    **{
        dataset: (100.0 - HQ_WEIGHT) * token_count / total_web_token_count for dataset, token_count in web_counts.items()
    },
}

phase_3_data_mixture = lm_varying_mixture_data_config(
    components=phase_3_tokenized,
    weights_list=[
        (0, DCLM_MIXTURE_WEIGHTS),
        (PHASE_3, cooldown_mixture_weights),
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


## Tootsie Dessert
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
    s: get_dolmino_step(s)
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

DESSERT_WEB = 0.7
DESSERT_HQ = 0.2
DESSERT_DESSERT = 0.1

# I'm such a dummy: I swapped HQ and Dessert weights
dessert_weights_v1 = {
    **{dataset: DESSERT_HQ * size / total_dessert_size for dataset, size in approx_dessert_sizes.items()},
    **{dataset: DESSERT_WEB * size / total_web_token_count for dataset, size in web_counts.items()},
    **{
        dataset: DESSERT_DESSERT * size / total_high_quality_token_count
        for dataset, size in high_quality_token_counts.items()
    },
}

dessert_tokenized = {**phase_3_tokenized, **dessert_dolmino_sets}

dessert_data_mixture_v1 = lm_varying_mixture_data_config(
    components=dessert_tokenized,
    weights_list=[
        (0, DCLM_MIXTURE_WEIGHTS),
        (PHASE_3, cooldown_mixture_weights),
        (PHASE_3_END, dessert_weights_v1),
    ],
)

# we're aiming to do 1 pass through the new mixes, which is another ~212e9 tokens
# 3072 * 4096 tokens per step = 12.6e6 tokens per step
# so ~17000 steps is about right

DESSERT_END = PHASE_3_END + 17000

# this is a (phase4?) config that we'll use for a final cooldown
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

llama_8b_tootsie_dessert = dataclasses.replace(
    default_train(
        name="llama-8b-tootsie-dessert",
        tokenized=dessert_data_mixture_v1,
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
    "all_math": DESSERT_DESSERT * sum(size for dataset, size in approx_dessert_sizes.items() if "math" in dataset) / total_dessert_size,
}

all_math = dolmino_math_tokenized_llama3

dessert_tokenized = {
    **phase_3_tokenized,
    "flan": dessert_tokenized["flan"],
    "all_math": all_math,
 }

dessert_data_mixture_v3 = lm_varying_mixture_data_config(
    components=dessert_tokenized,
    weights_list=[
        (0, DCLM_MIXTURE_WEIGHTS),
        (PHASE_3, cooldown_mixture_weights),
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


if __name__ == "__main__":
    executor_main(
        steps=[llama_8b_tootsie, llama_8b_tootsie_phase3, llama_8b_tootsie_dessert, llama_8b_tootsie_dessert_v3],
        description="Train 8B model on DCLM using WSD-S, then switching to EMA with a new mixture.",
    )
