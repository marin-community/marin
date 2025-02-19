"""
This is @dlwh's "YOLO"/vibes run described in https://github.com/stanford-crfm/marin/issues/600.

The idea is to train a 8B model continuously updating the mixture, data, and anything else. With WSD-S,
there's no "middle" or "end" of the run, there's just the run. So we'll just train for a long time, updating as we go.

We call it "tootsie" because tootsie rolls are famously made by folding in the previous batch of tootsie roll into the
next batch, so we're folding in the previous mixture into the next mixture.

For now, we're training on DCLM's best mix, but that will change.
"""

import dataclasses

from experiments.dolma.tokenize_dolma import tokenize_dolma_steps
from levanter.schedule import ScheduleStep

from experiments.dclm.tokenize_dclm import DCLM_MIXTURE_WEIGHTS, dclm_mixture_config_llama3
from experiments.defaults import default_tokenize, default_train
from experiments.llama import llama3_tokenizer, llama_8b
from experiments.pretraining_datasets import dclm_baseline_wrong, proofpile_2, starcoderdata
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.processing.tokenize import lm_mixture_data_config

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

# main phase: raw dclm for 660,000 steps
# phase 2 is divided into two subparts (lolcry):
# more dclm out to ≈3.78e+12 tokens (740,000 total steps)
# dolmino-ish mixture out to ≈5.28e+12 tokens (860,000 steps)

TOTAL = 860_000
DECAY_FRACTION = (860_000.0 - 740_000.0)/860_000.0


llama_8b_train_config_phase2 = SimpleTrainConfig(
    # tpu_type="v5litepod-256",
    # node_count=2,
    tpu_type="v4-2048",
    node_count=1,
    num_train_steps=TOTAL,
    # after 660,600 we changed things up:
    train_batch_size=[ScheduleStep(until=660_000, value=1024), ScheduleStep(until=-1, value=3072)],
    # LR doesn't (yet) support the schedule stuff so we just set it to the new value
    # because we're increasing the batch size, we need to increase the LR by \sqrt(ratio), which is ≈1.7x
    learning_rate=1.7e-3,
    # we're also switching to EMA because it's supposed to better than WSD-S
    decay=DECAY_FRACTION,
    ema_beta=0.995,
    lr_schedule="linear",
    cycle_length=None,
    allow_partial_checkpoint=True,
    steps_per_eval=2000,
    steps_per_task_eval=10000,
    steps_per_export=20000,
)

llama_8b_tootsie = dataclasses.replace(
    default_train(
        name="llama-8b-tootsie-0.001",
        tokenized=dclm_mixture_config_llama3,
        model_config=llama_8b,
        train_config=llama_8b_train_config,
        tags=["llama", "8b", "wsd-s", "exp600"],
    ),
    override_output_path="checkpoints/llama-8b-tootsie-0.001-19ad63",
)


# phase 2 data is a variant of the dolmino mix
phase_2_tokenized = {}

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
phase_2_tokenized.update({dataset: step for dataset, step in all_dolma_steps.items() if dataset in dolma_splits})
phase_2_tokenized["finemath_3_plus"] = finemath_3_plus_tokenized
phase_2_tokenized["fineweb_edu"] = fineweb_edu_tokenized
phase_2_tokenized["dclm"] = dolmino_dclm


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
    "fineweb_edu": 0.0 * 1.0,  # 1.3T tokens total (https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
}

total_high_quality_token_count = sum(high_quality_token_counts.values())

# reweight data so that 30% are high-quality sources and 70% are dclm
cooldown_mixture_weights = {
    **{
        dataset: 30 * token_count / total_high_quality_token_count
        for dataset, token_count in high_quality_token_counts.items()
    },
    "dclm": 70,
}


llama_8b_tootsie_phase2 = dataclasses.replace(
    default_train(
        name="llama-8b-tootsie-phase2",
        tokenized=dclm_mixture_config_llama3,
        model_config=llama_8b,
        train_config=llama_8b_train_config_phase2,
        tags=["llama", "8b", "ema", "exp600"],
    ),
    override_output_path="checkpoints/llama-8b-tootsie-phase2",
)


if __name__ == "__main__":
    executor_main(
        steps=[llama_8b_tootsie, llama_8b_tootsie_phase2],
        description="Train 8B model on DCLM using WSD-S.",
    )
