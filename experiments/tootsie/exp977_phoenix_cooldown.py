"""
https://github.com/marin-community/marin/issues/977

Codename: sensible-starling

This experiment is a cooldown run for the tootsie-8b model starting from adept-phoenix. It is trained on the
same mix as exp934_hq_vs_pt's best mix's full mix

We also add z-loss, since in spoonbill we found that to be very helpful
"""

import dataclasses

from levanter.schedule import ScheduleStep

from experiments.dclm.tokenize_dclm import DCLM_MIXTURE_WEIGHTS
from experiments.defaults import default_train
from experiments.evals.task_configs import CORE_TASKS_PLUS_MMLU
from experiments.exp934_hq_vs_pt import full_mix_components as cooldown_components
from experiments.llama import llama_8b
from experiments.midtraining_datasets import finemath_3_plus_tokenized
from experiments.nemotron_cc.tokenize_nemotron import NEMOTRON_WEIGHTS
from experiments.tootsie.exp600_tootsie import (
    PHASE_3_START,
    PHASE_4_REWARMUP_DURATION,
    PHASE_4_START,
    cooldown_mixture_weights_v1,
    llama_8b_train_config_phase4,
    nemotron_cc_steps,
    phase_3_tokenized,
    phase_4_steady_state_weights,
    phase_4_warmup_weights,
)
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import lm_varying_mixture_data_config

PHASE_4_END = 1_320_000

# aiming for 1.3e12 more tokens
# 4096 * 4096 * 80_000 is ~1.34e12
COOLDOWN_LEN = 80_000

checkpoint = "gs://marin-us-central2/checkpoints/llama-8b-tootsie-adept-phoenix/checkpoints/step-1320000"

cooldown_train_config = dataclasses.replace(
    llama_8b_train_config_phase4,
    train_batch_size=[
        ScheduleStep(start=0, value=1024),
        ScheduleStep(start=660_001, value=3072),
        ScheduleStep(start=1320001, value=4096),
    ],
    # from spoonbill: zloss is important for low LR phase
    z_loss_weight=1e-4,
    initialize_from_checkpoint_path=checkpoint,
    decay=COOLDOWN_LEN,
    num_train_steps=PHASE_4_END + COOLDOWN_LEN,
    learning_rate=1.7e-3,  # same peak lr
    lr_schedule="linear",
    # spoonbill went to just 2.75e-5 but with zloss I think we can go lower
    min_lr_ratio=1.7e-5 / 1.7e-3,  # 0.01 of peak lr
    cycle_length=None,
)

starling_hq_cooldown_weights = {
    "starcoderdata": 0.25,
    "proofpile_2": 0.25,
    "dolmino/flan": 0.017 * 10,
    "dolmino/pes2o": 0.0581 * 5,
    "dolmino/stackexchange": 0.0171 * 5,
    "dolmino/wiki": 0.00365 * 5,
    "all_math": 0.00422 * 10,
    "arxiv_markdownified": 0.0581 * 5,
    "stackexchange_custom": 0.0171 * 5,
    "wikipedia_markdown": 0.00365 * 5,
    "medu_science_qa": 0.0012 * 5,
    # about 34B tokens
    "finemath-3-plus": 0.034 * 5,
}

total_hq_weight = sum(v for k, v in starling_hq_cooldown_weights.items())
# we want nemotron to be 0.7 of the total weight
nemotron_total = sum(v for k, v in NEMOTRON_WEIGHTS.items())


starling_cooldown_weights = {
    **{k: v * 0.7 / nemotron_total for k, v in NEMOTRON_WEIGHTS.items()},
    **{k: v * 0.3 / total_hq_weight for k, v in starling_hq_cooldown_weights.items()},
}

starling_components = {"finemath-3-plus": finemath_3_plus_tokenized}

starling_cooldown_mixture = lm_varying_mixture_data_config(
    components={**phase_3_tokenized, **nemotron_cc_steps, **cooldown_components, **starling_components},
    weights_list=[
        (0, DCLM_MIXTURE_WEIGHTS),
        (PHASE_3_START, cooldown_mixture_weights_v1),
        (PHASE_4_START, phase_4_warmup_weights),
        (PHASE_4_START + PHASE_4_REWARMUP_DURATION, phase_4_steady_state_weights),
        (PHASE_4_END, starling_cooldown_weights),
    ],
)

tootsie_8b_sensible_starling = default_train(
    name="tootsie-8b-sensible-starling",
    tokenized=starling_cooldown_mixture,
    model_config=llama_8b,
    train_config=cooldown_train_config,
    tags=["llama", "8b", "ema", "exp977", "tootsie", "cooldown"],
    eval_harness_tasks=CORE_TASKS_PLUS_MMLU,
).with_output_path("checkpoints/tootsie-8b-sensible-starling")

# print normalized weights for final phase
# sanity checks:
normalized = {k: v / sum(starling_cooldown_weights.values()) for k, v in starling_cooldown_weights.items()}

# sum up the nemotron ones:
assert 0.69 < sum(v for k, v in normalized.items() if k.startswith("nemotron")) < 0.71
assert 0.29 < sum(v for k, v in normalized.items() if k not in NEMOTRON_WEIGHTS) < 0.31

if __name__ == "__main__":
    executor_main(
        [tootsie_8b_sensible_starling],
        description="Train Tootsie 8b with cooldown from 1.7e-4 to 1.7e-5 over 125B tokens",
    )
