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
https://github.com/marin-community/marin/issues/1529

Cooldown run for 32B Tootsie Model using the same data mixture as in Starling for 8B
"""

import dataclasses

from levanter.optim import AdamConfig
from levanter.optim.clip_update_norm import ClipUpdateNormConfig
from levanter.schedule import ScheduleStep

from experiments.defaults import default_train
from experiments.tootsie.exp1395_qwen3_32b import (
    marin_32b_qwen,
    qwen3_32b_remat,
    qwen_32b_warmstart_train,
)
from experiments.evals.evals import default_base_eval
from experiments.models import ModelConfig, download_model_step
from experiments.nemotron_cc.tokenize_nemotron import (
    NEMOTRON_WEIGHTS,
    tokenize_nemotron_steps,
)
from experiments.dclm.tokenize_dclm import dclm_components_llama3
from experiments.exp934_hq_vs_pt import pt_vs_hq_components
from experiments.tootsie.exp600_tootsie import phase_3_tokenized, starling_components
from marin.execution import executor_main, output_path_of
from marin.processing.tokenize.data_configs import lm_varying_mixture_data_config

from experiments.evals.resource_configs import SINGLE_TPU_V5p_8

PHASE_3_START = 160_000
PHASE_3_END = 192_000  # 20% of Training for Cooldown

nemotron_steps = tokenize_nemotron_steps()
proofpile_2 = dclm_components_llama3["proofpile_2"]
starcoderdata = dclm_components_llama3["starcoderdata"]

NEMOTRON_PT_MIX_WEIGHTS = {
    **NEMOTRON_WEIGHTS,
    "starcoderdata": 0.25,
    "proofpile_2": 0.055,
}

HQ_COOLDOWN_WEIGHTS = {
    "dolmino/flan": 0.017 * 2,  # Double Weight in Starling
    "dolmino/pes2o": 0.0581,
    "dolmino/stackexchange": 0.0171,
    "dolmino/wiki": 0.00365,
    "all_math": 0.00422 * 2,  # Double Weight in Starling
    "arxiv_markdownified": 0.0581,
    "stackexchange_custom": 0.0171,
    "wikipedia_markdown": 0.00365,
    "medu_science_qa": 0.0012,
    "finemath-3-plus": 0.034,
}

nemotron_total = sum(NEMOTRON_PT_MIX_WEIGHTS.values())
total_hq_weight = sum(HQ_COOLDOWN_WEIGHTS.values())

bison_cooldown_weights = {
    **{k: v * 0.7 / nemotron_total for k, v in NEMOTRON_PT_MIX_WEIGHTS.items()},
    **{k: v * 0.3 / total_hq_weight for k, v in HQ_COOLDOWN_WEIGHTS.items()},
}

bison_cooldown_mixture = lm_varying_mixture_data_config(
    components={
        **nemotron_steps,
        "starcoderdata": starcoderdata,
        "proofpile_2": proofpile_2,
        **phase_3_tokenized,
        **pt_vs_hq_components,
        **starling_components,
    },
    weights_list=[
        (0, NEMOTRON_PT_MIX_WEIGHTS),  # Phase 1 and 2 used the same data mixture and just changed the model arch
        (PHASE_3_START, bison_cooldown_weights),
    ],
)

DECAY_FRACTION = (PHASE_3_END - PHASE_3_START) / PHASE_3_END

qwen_phase2_checkpoint_for_phase3 = marin_32b_qwen.cd(f"checkpoints/step-{PHASE_3_START}").nonblocking()

bison_train_config = dataclasses.replace(
    qwen_32b_warmstart_train,
    initialize_from_checkpoint_path=qwen_phase2_checkpoint_for_phase3,
    decay=DECAY_FRACTION,
    num_train_steps=PHASE_3_END,
    train_batch_size=[
        ScheduleStep(start=0, value=8192),
        ScheduleStep(start=18500, value=7680),
        ScheduleStep(start=21010, value=8192),
        ScheduleStep(
            start=159_999, value=1024
        ),  # Hack to make the block size change a power of 2048, which was otherwise messed up by the 7680 phase.
        ScheduleStep(start=160_000, value=8192),
    ],
    optimizer_config=AdamConfig(
        # Modulate Decay And Warmup to Just Cool This Model Down
        decay=DECAY_FRACTION,
        warmup=0.00,
        adamc_weight_decay=True,
        # From here out, this is a copy of the Optimizer hparams from in exp1395_qwen3_32b
        beta1=0.9,
        beta2=0.95,
        epsilon=1e-8,
        max_grad_norm=0.2,  # we're almost always < .2 except during spikes
        # width is a little smaller than the 24B and we're using a much larger batch size
        # 4.2e-4 * sqrt(8192/3072) ≈ 7e-4
        learning_rate=7e-4,
        weight_decay=0.05,
        skip_bad_steps=True,
        # update_rms_clipping=1.0,  # added at 67522, removed at 72233
        lr_schedule="linear",
        # this was inadvertently off from about 74k to 80k
        clip_update_norm=ClipUpdateNormConfig(rolling_interval_length=128, sigma_factor=2.0),
    ),
)

tootsie_32b_cooldown_bison = default_train(
    name="tootsie-32b-cooldown-bison-adamc",
    tokenized=bison_cooldown_mixture,
    model_config=qwen3_32b_remat,
    train_config=bison_train_config,
    tags=["qwen", "32b", "ema", "exp1529", "tootsie", "cooldown"],
    eval_harness_tasks=[],
).with_output_path("checkpoints/tootsie-32b-cooldown-bison-adamc")


# Loss Spiked, see if flat LR works. (Will: It doesn't but preserving for posterity)

qwen_phase3_checkpoint_for_phase4 = tootsie_32b_cooldown_bison.cd("checkpoints/step-190000").nonblocking()

bison_train_config_flat = dataclasses.replace(
    bison_train_config,
    decay=0,
    num_train_steps=PHASE_3_END,
    initialize_from_checkpoint_path=qwen_phase3_checkpoint_for_phase4,
    optimizer_config=AdamConfig(
        # Modulate Decay And Warmup to Just Cool This Model Down
        decay=0,
        warmup=0.0,
        adamc_weight_decay=True,
        learning_rate=(7e-4 * (1 - (30 / 32))) + (7e-5 * (30 / 32)),
        # From here out, this is a copy of the Optimizer hparams from bison_train_config
        beta1=0.9,
        beta2=0.95,
        epsilon=1e-8,
        max_grad_norm=0.2,  # we're almost always < .2 except during spikes
        # width is a little smaller than the 24B and we're using a much larger batch size
        # 4.2e-4 * sqrt(8192/3072) ≈ 7e-4
        weight_decay=0.05,
        skip_bad_steps=True,
        # update_rms_clipping=1.0,  # added at 67522, removed at 72233
        lr_schedule="linear",
        # this was inadvertently off from about 74k to 80k
        clip_update_norm=ClipUpdateNormConfig(rolling_interval_length=128, sigma_factor=2.0),
    ),
)

tootsie_32b_cooldown_bison_flat = default_train(
    name="tootsie-32b-cooldown-bison-adamc-flat",
    tokenized=bison_cooldown_mixture,
    model_config=qwen3_32b_remat,
    train_config=bison_train_config_flat,
    tags=["qwen", "32b", "ema", "exp1529", "tootsie", "cooldown"],
    eval_harness_tasks=[],
).with_output_path("checkpoints/tootsie-32b-cooldown-bison-adamc-flat")

baselines = [
    ("Qwen/Qwen3-32B", "9216db5781bf21249d130ec9da846c4624c16137"),
    ("Qwen/Qwen2.5-32B", "1818d35814b8319459f4bd55ed1ac8709630f003"),
    # ("allenai/OLMo-2-0325-32B", "stage2-ingredient3-step9000-tokens76B"),
    # OLMo 32B is currently borked on our eval harness so we will evaluate with GPU
]
baseline_evals = []
for model, revision in baselines:
    model_instance = download_model_step(ModelConfig(hf_repo_id=model, hf_revision=revision))
    baseline_evals.extend(
        default_base_eval(
            output_path_of(model_instance),
            resource_config=SINGLE_TPU_V5p_8,
            run_generation_evals=False,
            discover_latest_checkpoint=False,
        )
    )

if __name__ == "__main__":
    executor_main(
        [
            tootsie_32b_cooldown_bison,
            *default_base_eval(tootsie_32b_cooldown_bison, resource_config=SINGLE_TPU_V5p_8, run_generation_evals=False),
            *baseline_evals,
        ],
        description="Cooldown the 32B Qwen model on bison mixture",
    )
