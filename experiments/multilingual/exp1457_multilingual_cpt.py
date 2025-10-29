# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Multilingual continual pretraining experiment starting from Phoenix phase.
This experiment continues training from the Phoenix phase checkpoint (step 1320000) using a mix of
Fineweb2 HQ multilingual data (70%) and high-quality datasets from the starling cooldown (30%).
The training uses a linear LR decay from 1.7e-3 to 1.7e-5 over 80,000 steps (~1.34T tokens).
"""

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
    starling_hq_cooldown_weights,
)
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import lm_varying_mixture_data_config

##############################################################
# Phase 5: Starling second cooldown and Multilingual CPT Start
##############################################################

# This is documented in https://github.com/marin-community/marin/issues/1457

PHASE_4_END = 1_320_000

# aiming for 1.3e12 more tokens
# 4096 * 4096 * 80_000 is ~1.34e12
COOLDOWN_LEN = 80_000

# for these long runs we don't usually actually **finish** the run in the Executor's eyes,
# so we use `nonblocking`
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

fineweb2_hq_weights = FINEWEB2_HQ_MIXTURE_BYTES

# Build the full component set first so we can filter weight dicts to valid keys.
components = {**phase_3_tokenized, **tokenize_fineweb2hq_steps()}

def _filter_weights_to_components(weights: dict[str, float]) -> dict[str, float]:
    """Return a copy of weights keeping only keys present in components.

    This avoids passing weights for datasets that aren't included in the mixture,
    which Levanter correctly rejects.
    """
    allowed = set(components.keys())
    return {k: v for k, v in weights.items() if k in allowed}

# We want fineweb2 hq to be 0.7 of the total weight. Normalize per-language by bytes.
fineweb_total = sum(v for v in fineweb2_hq_weights.values())

# For the remaining 0.3, use the HQ portion from the starling cooldown (no nemotron_cc).
# Then filter to the keys that actually exist in this experiment's components.
hq_base = _filter_weights_to_components(starling_hq_cooldown_weights)
hq_total = sum(hq_base.values()) if hq_base else 1.0

multilingual_transition_weights = {
    # fineweb2 HQ languages
    **{f"fineweb2_hq/{k}": v * 0.7 / fineweb_total for k, v in FINEWEB2_HQ_MIXTURE_BYTES.items()},
    # non-nemotron HQ sets (e.g., starcoder, proofpile, dolma, finemath, etc.)
    **{k: v * 0.3 / hq_total for k, v in hq_base.items()},
}

MULTILINGUAL_CPT_STEPS = 100_000
MULTILINGUAL_CPT_START = PHASE_4_END
MULTILINGUAL_CPT_TRANSITION_END = MULTILINGUAL_CPT_START + 1000
MULTILINGUAL_CPT_END = MULTILINGUAL_CPT_START + MULTILINGUAL_CPT_STEPS


fineweb2_hq = lm_varying_mixture_data_config(
    components=components,
    weights_list=[
        (0, _filter_weights_to_components(DCLM_MIXTURE_WEIGHTS)),
        (PHASE_3_START, _filter_weights_to_components(cooldown_mixture_weights_v1)),
        # Filter out nemotron_cc keys from phase 4 warmup and steady state
        (PHASE_4_START, _filter_weights_to_components(phase_4_warmup_weights)),
        (PHASE_4_START + PHASE_4_REWARMUP_DURATION, _filter_weights_to_components(phase_4_steady_state_weights)),
        (MULTILINGUAL_CPT_START, _filter_weights_to_components(multilingual_transition_weights)),
        (MULTILINGUAL_CPT_TRANSITION_END, _filter_weights_to_components(multilingual_transition_weights)),
    ],
)

multilingual_cpt_8b_fineweb2_hq = default_train(
    name="multilingual-cpt-8b-fineweb2-hq",
    tokenized=fineweb2_hq,
    model_config=llama_8b,
    train_config=cooldown_train_config,
    tags=["llama", "8b", "ema", "exp1457", "multilingual", "cpt"],
    eval_harness_tasks=CORE_TASKS_PLUS_MMLU,
).with_output_path("checkpoints/multilingual-cpt-8b-fineweb2-hq")

# print normalized weights for final phase
# sanity checks:
normalized = {k: v / sum(multilingual_transition_weights.values()) for k, v in multilingual_transition_weights.items()}

# sum up the fineweb2 hq ones:
assert 0.69 < sum(v for k, v in normalized.items() if k.startswith("fineweb")) < 0.71
assert 0.29 < sum(v for k, v in normalized.items() if not k.startswith("fineweb")) < 0.31


if __name__ == "__main__":
    executor_main(
        steps=[
            multilingual_cpt_8b_fineweb2_hq,
        ],
        description="Continually Pretrain on Fineweb2 HQ from Phoenix Phase",
    )
