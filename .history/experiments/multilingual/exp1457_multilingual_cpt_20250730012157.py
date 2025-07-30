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

from experiments.cooldown_anneal import dolmino_dclm
from experiments.dclm.tokenize_dclm import DCLM_MIXTURE_WEIGHTS, dclm_components_llama3, dclm_mixture_config_llama3
from experiments.defaults import default_train
from experiments.dolma.tokenize_dolma import tokenize_dolma_steps
from experiments.dolmino.tokenize_dolmino import dolmino_math_tokenized_llama3, get_dolmino_step_llama3
from experiments.multilingual_fineweb2_hq.download_and_tokenize_fineweb2_hq import tokenize_fineweb2hq_steps, _get_fineweb2_split_paths
from experiments.evals.evals import default_base_eval
from experiments.evals.task_configs import CORE_TASKS_PLUS_MMLU
from experiments.exp934_hq_vs_pt import pt_vs_hq_components
from experiments.llama import llama3_tokenizer, llama_8b, llama_8b_old_rotary
from experiments.midtraining_datasets import finemath_3_plus_tokenized
from experiments.nemotron_cc.tokenize_nemotron import NEMOTRON_WEIGHTS, tokenize_nemotron_steps
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import lm_varying_mixture_data_config
from marin.resources import TpuPodConfig

# Phases/Runs in this file:
# 1. Kestrel: WSD-S on DCLM+Starcode+Proofpile on 2x v5litepod-256 (from scratch)
# 2. Ocelot: Switch to WSD with EMA on v4-2048 (from (1))
# 3. Jellyfish Cooldown v1: Switch to a 30% Dolmino-ish HQ dataset mixture, decay the LR (from (2))
# 4a. Dessert (Attempt 1): Sprinkle in a bit of FLAN and Synth Math (from (3))
# 4b. Dessert (Attempt 2): Fix the weights and add in all the HQ docs from dolmino (from (3))
# 5. Cooldown v2: Another attempt at a final cooldown (from (2))
# 6. Phoenix: from (3), rewarmup and use mix of nemotron_cc and starcoder to keep moving
# 7. Starling: from (6), cooldown from 1.7e-3 to 1.7e-5 over 1.34T tokens
# 8. Deeper Starling: from (7), coast at 1.7e-5 over ~250B tokens


################################################################
# PHASE 1: "Kestrel" WSD-S on DCLM+Starcode+Proofpile on 2x v5litepod-256
################################################################

# Initially, we start with WSD-S (cyclic stable/decay). The idea was to use WSD-S to train forever,
# but we have learned that WSD with a long cooldown is superior. Once we switch to WSD,
# we use the exponential moving average (EMA) of the model weights to get a better model.

tootise_phase1_config = SimpleTrainConfig(
    resources=TpuPodConfig(tpu_type="v5litepod-256", slice_count=2),
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
        tokenized=tokenize_fineweb2hq_steps,
        # I am a dummy and use old rotary config
        model_config=llama_8b_old_rotary,
        train_config=tootise_phase1_config,
        tags=["llama", "8b", "wsd-s", "exp600"],
    ),
    override_output_path="checkpoints/llama-8b-tootsie-0.001-19ad63",
)


##########################
# PHASE 2: Ocelot We switch to WSD with EMA, moving to v4-2048, increased batch size
###########################
# We increased batch size b/c we have more hardware
# Because we increased the batch size, we need to increase the LR by \sqrt(ratio), which is ≈1.7x

PHASE_1_END = 660_000

kestrel_phase_1_checkpoint_for_phase2 = llama_8b_tootsie_phase1.cd(f"checkpoints/step-{PHASE_1_END}").nonblocking()

llama_8b_train_config_phase2 = SimpleTrainConfig(
    resources=TpuPodConfig(tpu_type="v4-2048", slice_count=1),
    num_train_steps=1_000_000,
    # after PHASE_1_END we changed things up:
    train_batch_size=[ScheduleStep(start=0, value=1024), ScheduleStep(start=PHASE_1_END + 1, value=3072)],
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
    # this is retconning a bit: I actually copied the checkpoint manually. But this is the same thing
    initialize_from_checkpoint_path=kestrel_phase_1_checkpoint_for_phase2,  # from phase 1
    reset_data_loader_on_init=False,
)


llama_8b_tootsie_phase2 = dataclasses.replace(
    default_train(
        name="llama-8b-tootsie-phase2",
        tokenized=tokenize_fineweb2hq_steps,
        model_config=llama_8b,
        train_config=llama_8b_train_config_phase2,
        tags=["llama", "8b", "ema", "exp600"],
    ),
    override_output_path="checkpoints/llama-8b-tootsie-phase2",
)



if __name__ == "__main__":
    executor_main(
        steps=[
            llama_8b_tootsie_phase1,
            llama_8b_tootsie_phase3,
            llama_8b_tootsie_dessert_BAD,
            llama_8b_tootsie_dessert_v3,
            llama_8b_tootsie_cooldown_v2,
            llama_8b_tootsie_adept_phoenix,
            tootsie_8b_sensible_starling,
            tootsie_8b_deeper_starling,
            *default_base_eval(tootsie_8b_deeper_starling),
        ],
        description="Train 8B model on DCLM using WSD-S, then switching to EMA with a new mixture.",
    )