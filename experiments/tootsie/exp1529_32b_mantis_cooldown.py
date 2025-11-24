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

Cooldown run for the 32B Tootsie model using MegaMath in place of the Dolmino math mixture.
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
from experiments.pretraining_datasets import (
    NEMOTRON_WEIGHTS,
    tokenize_nemotron,
)
from experiments.pretraining_datasets.dclm import dclm_components_llama3
from experiments.exp934_hq_vs_pt import pt_vs_hq_components
from experiments.midtraining_datasets import (
    megamath_token_counts,
    megamath_tokenized,
    stackv2_edu_filtered_python_tokenized,
)
from experiments.tootsie.exp600_tootsie import phase_3_tokenized, starling_components
from marin.execution import executor_main, output_path_of
from marin.processing.tokenize.data_configs import lm_varying_mixture_data_config

from experiments.evals.resource_configs import SINGLE_TPU_V5p_8

PHASE_3_START = 160_000
PHASE_3_END = 192_000  # 20% of Training for Cooldown

nemotron_steps = tokenize_nemotron()
proofpile_2 = dclm_components_llama3["proofpile_2"]
starcoderdata = dclm_components_llama3["starcoderdata"]

NEMOTRON_PT_MIX_WEIGHTS = {
    **NEMOTRON_WEIGHTS,
    "starcoderdata": 0.25,
    "proofpile_2": 0.055,
}

# In Trillions, as usual, upweighting a few things
HQ_COOLDOWN_WEIGHTS = {
    "dolmino/flan": 0.017 * 2,  # Double Weight in Starling
    "dolmino/pes2o": 0.0581,
    "dolmino/stackexchange": 0.0171,
    "dolmino/wiki": 0.00365,
    "all_math": 0.371,
    "arxiv_markdownified": 0.0581,
    "stackexchange_custom": 0.0171,
    "wikipedia_markdown": 0.00365,
    "medu_science_qa": 0.0012,
    "finemath-3-plus": 0.034,
}

nemotron_total = sum(NEMOTRON_PT_MIX_WEIGHTS.values())
all_math_weight = HQ_COOLDOWN_WEIGHTS["all_math"]
megamath_total = sum(megamath_token_counts.values())

mantis_hq_cooldown_weights = {
    **{k: v for k, v in HQ_COOLDOWN_WEIGHTS.items() if k != "all_math"},
    **{
        split: (all_math_weight if split != "megamath/web" else all_math_weight / 4) * weight / megamath_total
        for split, weight in megamath_token_counts.items()
    },
}

mantis_total_hq_weight = sum(mantis_hq_cooldown_weights.values())

mantis_cooldown_weights = {
    **{k: v * 0.7 / nemotron_total for k, v in NEMOTRON_PT_MIX_WEIGHTS.items()},
    **{k: v * (0.3 / mantis_total_hq_weight) for k, v in mantis_hq_cooldown_weights.items()},
}

STACKV2_EDU_PYTHON_KEY = "common_pile_stackv2_edu_filtered_python"
STACKV2_EDU_PYTHON_INTRO_STEP = 174_000
# Approximate dataset weight by converting 14.63 GiB
STACKV2_EDU_PYTHON_WEIGHT = 0.01463

mantis_hq_cooldown_weights_with_stackv2_python = {
    **mantis_hq_cooldown_weights,
    STACKV2_EDU_PYTHON_KEY: STACKV2_EDU_PYTHON_WEIGHT,
}

mantis_total_hq_weight_with_stackv2_python = sum(mantis_hq_cooldown_weights_with_stackv2_python.values())

mantis_cooldown_weights_with_stackv2_python = {
    **{k: v * 0.7 / nemotron_total for k, v in NEMOTRON_PT_MIX_WEIGHTS.items()},
    **{
        k: v * (0.3 / mantis_total_hq_weight_with_stackv2_python)
        for k, v in mantis_hq_cooldown_weights_with_stackv2_python.items()
    },
}

mantis_cooldown_mixture = lm_varying_mixture_data_config(
    components={
        **nemotron_steps,
        "starcoderdata": starcoderdata,
        "proofpile_2": proofpile_2,
        **phase_3_tokenized,
        **{k: v for k, v in pt_vs_hq_components.items() if k != "all_math"},
        **megamath_tokenized,
        **starling_components,
        STACKV2_EDU_PYTHON_KEY: stackv2_edu_filtered_python_tokenized,
    },
    weights_list=[
        (0, NEMOTRON_PT_MIX_WEIGHTS),  # Phase 1 and 2 used the same data mixture and just changed the model arch
        (PHASE_3_START, mantis_cooldown_weights),
        (STACKV2_EDU_PYTHON_INTRO_STEP, mantis_cooldown_weights_with_stackv2_python),
    ],
    permutation_type="feistel",  # the first phase was actually linear, but this is better for mixing things up
)

DECAY_FRACTION = (PHASE_3_END - PHASE_3_START) / PHASE_3_END

qwen_phase2_checkpoint_for_phase3 = marin_32b_qwen.cd(f"checkpoints/step-{PHASE_3_START}").nonblocking()

mantis_train_config = dataclasses.replace(
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

tootsie_32b_cooldown_mantis = default_train(
    name="tootsie-32b-cooldown-mantis-adamc-v2",
    tokenized=mantis_cooldown_mixture,
    model_config=qwen3_32b_remat,
    train_config=mantis_train_config,
    tags=["qwen", "32b", "ema", "exp1529", "tootsie", "cooldown", "mantis"],
    eval_harness_tasks=[],
).with_output_path("checkpoints/tootsie-32b-cooldown-mantis-adamc-v2")


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
            tootsie_32b_cooldown_mantis,
            *default_base_eval(
                tootsie_32b_cooldown_mantis,
                resource_config=SINGLE_TPU_V5p_8,
                run_generation_evals=False,
            ),
            *baseline_evals,
        ],
        description="Cooldown the 32B Qwen model on mantis mixture",
    )
