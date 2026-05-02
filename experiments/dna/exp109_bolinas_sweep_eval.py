# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-off VEP eval of a transfer sweep checkpoint via the standalone eval harness."""

import os

import jmp
from levanter.eval_harness import EvalHarnessMainConfig, LmEvalHarnessConfig, run_eval_harness_main
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig

from experiments.dna.exp109_bolinas_scaling_sweep import (
    REFERENCE_HPARAMS,
    TOKENIZER,
    TRANSFER_PRIMARY_HIDDEN_SIZE,
    _build_model_config,
)
from experiments.evals.task_configs import TRAITGYM_MENDELIAN_V2_255, convert_to_levanter_task_config
from rigging.filesystem import REGION_TO_DATA_BUCKET

REGION = os.environ.get("REGION")
if not REGION:
    raise ValueError("REGION environment variable must be set")
if REGION not in REGION_TO_DATA_BUCKET:
    raise ValueError(f"Unknown region: {REGION}. Must be one of {list(REGION_TO_DATA_BUCKET.keys())}")
CHECKPOINT = (
    f"gs://{REGION_TO_DATA_BUCKET[REGION]}/checkpoints/dna-bolinas-transfer-v0.14-positive-control-ba48ae/hf/step-9535"
)
RUN = f"dna-bolinas-transfer-v0.14-vep-{REGION}-v0.2"


def run_sweep_eval():
    model_config = _build_model_config(TRANSFER_PRIMARY_HIDDEN_SIZE, REFERENCE_HPARAMS.initializer_range)

    config = EvalHarnessMainConfig(
        eval_harness=LmEvalHarnessConfig(
            task_spec=convert_to_levanter_task_config([TRAITGYM_MENDELIAN_V2_255]),
            include_path="experiments/evals/custom_tasks",
            max_packed_segments=1,
        ),
        tokenizer=TOKENIZER,
        checkpoint_path=CHECKPOINT,
        checkpoint_is_hf=True,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                project="marin",
                name=RUN,
                tags=["dna", "bolinas", "checkpoint_vep_eval"],
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
        ),
        model=model_config,
    )

    run_eval_harness_main(config)


if __name__ == "__main__":
    run_sweep_eval()
