"""
This is @dlwh's "YOLO"/vibes run described in https://github.com/marin-community/marin/issues/600.

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

from experiments.dclm.tokenize_dclm import DCLM_MIXTURE_WEIGHTS
from experiments.defaults import default_train
from experiments.evals.task_configs import CORE_TASKS_PLUS_MMLU
from experiments.llama import llama_8b
from experiments.multilingual_fineweb2_hq.constants import FINEWEB2_HQ_MIXTURE_BYTES
from experiments.multilingual_fineweb2_hq.download_and_tokenize_fineweb2_hq import tokenize_fineweb2hq_steps
from experiments.tootsie.exp600_tootsie import (
    PHASE_1_END,
    PHASE_3_START,
    PHASE_4_REWARMUP_DURATION,
    PHASE_4_START,
    cooldown_mixture_weights_v1,
    llama_8b_tootsie_adept_phoenix,
    llama_8b_train_config_phase4,
    phase_3_tokenized,
    phase_4_steady_state_weights,
    phase_4_warmup_weights,
)
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import lm_varying_mixture_data_config

##################################################
# Phase 5: Starling second cooldown and CPT Start
##################################################

# This is documented in https://github.com/marin-community/marin/issues/977

PHASE_4_END = 1_320_000

# aiming for 1.3e12 more tokens
# 4096 * 4096 * 80_000 is ~1.34e12
COOLDOWN_LEN = 80_000

# for these long runs we don't usually actually **finish** the run in the Executor's eyes,
# so we use `wait_for_completion`
phoenix_phase4_checkpoint_for_phase5 = llama_8b_tootsie_adept_phoenix.cd("checkpoints/step-1320000").nonblocking()

cooldown_train_config = dataclasses.replace(
    llama_8b_train_config_phase4,
    train_batch_size=[
        ScheduleStep(start=0, value=1024),
        ScheduleStep(start=PHASE_1_END + 1, value=3072),
        ScheduleStep(start=PHASE_4_END + 1, value=4096),
    ],
    # from spoonbill: zloss is important for low LR phase
    z_loss_weight=1e-4,
    initialize_from_checkpoint_path=phoenix_phase4_checkpoint_for_phase5,
    decay=COOLDOWN_LEN,
    num_train_steps=PHASE_4_END + COOLDOWN_LEN,
    learning_rate=1.7e-3,  # same peak lr
    lr_schedule="linear",
    # spoonbill went to just 2.75e-5 but with zloss I think we can go lower
    min_lr_ratio=1.7e-5 / 1.7e-3,  # 0.01 of peak lr
    cycle_length=None,
)

fineweb_weights = FINEWEB2_HQ_MIXTURE_BYTES

total_hq_weight = sum(v for k, v in fineweb_weights.items())
# we want nemotron to be 0.7 of the total weight
fineweb_total = sum(v for k, v in fineweb_weights.items())


starling_cooldown_weights = {
    **{k: v * 0.7 / fineweb_total for k, v in FINEWEB2_HQ_MIXTURE_BYTES.items()},
    **{k: v * 0.3 / total_hq_weight for k, v in FINEWEB2_HQ_MIXTURE_BYTES.items()},
}


fineweb2_hq_mixture = lm_varying_mixture_data_config(
    components={**phase_3_tokenized, **tokenize_fineweb2hq_steps},
    weights_list=[
        (0, DCLM_MIXTURE_WEIGHTS),
        (PHASE_3_START, cooldown_mixture_weights_v1),
        (PHASE_4_START, phase_4_warmup_weights),
        (PHASE_4_START + PHASE_4_REWARMUP_DURATION, phase_4_steady_state_weights),
    ],
)

tootsie_8b_sensible_starling = default_train(
    name="tootsie-8b-sensible-starling",
    tokenized=fineweb2_hq_mixture,
    model_config=llama_8b,
    train_config=cooldown_train_config,
    tags=["llama", "8b", "ema", "exp977", "tootsie", "cooldown"],
    eval_harness_tasks=CORE_TASKS_PLUS_MMLU,
).with_output_path("checkpoints/tootsie-8b-sensible-starling")

# print normalized weights for final phase
# sanity checks:
normalized = {k: v / sum(fineweb_weights.values()) for k, v in fineweb_weights.items()}

# sum up the nemotron ones:
assert 0.69 < sum(v for k, v in normalized.items() if k.startswith("fineweb")) < 0.71
assert 0.29 < sum(v for k, v in normalized.items() if k not in FINEWEB2_HQ_MIXTURE_BYTES) < 0.31

# ############################
# # Phase 6: Deeper Starling (dessert-ish)
# ############################
# # things kept getting better, so we'll do a constant LR run for a bit longer

# # starling_checkpoint = "gs://marin-us-central2/checkpoints/tootsie-8b-sensible-starling/checkpoints/step-1399923/"
# starling_checkpoint = tootsie_8b_sensible_starling.cd("checkpoints/step-1399923").nonblocking()

# EXTRA_COOLDOWN_LEN = 20000

# tootsie_8b_deeper_starling_train_config = dataclasses.replace(
#     cooldown_train_config,
#     learning_rate=1.7e-5,
#     min_lr_ratio=1.0,
#     decay=0,
#     num_train_steps=PHASE_4_END + COOLDOWN_LEN + EXTRA_COOLDOWN_LEN,
#     initialize_from_checkpoint_path=starling_checkpoint,
#     reset_data_loader_on_init=False,
# )

# tootsie_8b_deeper_starling = default_train(
#     name="tootsie-8b-deeper-starling",
#     tokenized=starling_cooldown_mixture,
#     model_config=llama_8b,
#     train_config=tootsie_8b_deeper_starling_train_config,
#     tags=["llama", "8b", "ema", "exp977", "tootsie", "cooldown"],
#     eval_harness_tasks=CORE_TASKS_PLUS_MMLU,
# ).with_output_path("checkpoints/tootsie-8b-deeper-starling")


if __name__ == "__main__":
    executor_main(
        steps=[
            tootsie_8b_sensible_starling,
            # tootsie_8b_deeper_starling,
            # *default_base_eval(tootsie_8b_deeper_starling),
        ],
        description="Continually Pretrain on Fineweb2 HQ from Phoenix Phase",
    )
