# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reference: Single-run pretraining → midtraining → SFT pipeline.

Demonstrates that pretrain/midtrain/SFT are all just data mixing phases.
The entire pipeline is one training run with time-varying mixture weights:

  1. Pretrain (steps 0-40k): DCLM baseline
  2. Midtrain (steps 40k-50k): Blend DCLM + Dolmino math
  3. SFT (steps 50k-52k): SmolTalk instruction data
"""

import dataclasses

from fray.cluster import ResourceConfig
from levanter.data.text import ChatLmDatasetFormat
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.execution.remote import remote
from marin.processing.tokenize import add_validation_sets_to_mixture
from marin.processing.tokenize.data_configs import lm_varying_mixture_data_config

from experiments.defaults import default_tokenize, default_validation_sets
from experiments.grug.base.launch import GrugBaseLaunchConfig, run_grug_base_trial
from experiments.grug.base.model import GrugModelConfig
from experiments.grug.base.train import GrugEvalConfig
from experiments.marin_models import marin_tokenizer
from experiments.posttrain.instruction_datasets import get_instruction_dataset
from experiments.pretraining_datasets.dclm import dclm_components_llama3
from experiments.pretraining_datasets.dolmino import tokenize_dolmino

# --- Model: 600M Grug ---
model = GrugModelConfig(
    vocab_size=128_256,
    max_seq_len=4096,
    hidden_dim=1024,
    intermediate_dim=3584,
    num_heads=16,
    num_kv_heads=8,
    num_layers=24,
)

# --- Schedule ---
PRETRAIN_STEPS = 40_000
MIDTRAIN_STEPS = 10_000
SFT_STEPS = 2_000
TOTAL_STEPS = PRETRAIN_STEPS + MIDTRAIN_STEPS + SFT_STEPS

# --- Data components ---
pretrain = {"dclm": dclm_components_llama3["dclm_baseline"]}

dolmino = tokenize_dolmino()
midtrain = {"dolmino_math": dolmino["dolmino/math/metamath-owmfilter"]}

smoltalk = get_instruction_dataset("HuggingFaceTB/smoltalk", splits=["train"])
sft = {
    "smoltalk": default_tokenize(
        name="smoltalk_marin",
        dataset=smoltalk / "**/*.jsonl.gz",
        tokenizer=marin_tokenizer,
        format=ChatLmDatasetFormat(),
    )
}

# --- Time-varying mixture weights ---
data = lm_varying_mixture_data_config(
    components={**pretrain, **midtrain, **sft},
    weights_list=[
        (0, {"dclm": 1.0, "dolmino_math": 0.0, "smoltalk": 0.0}),
        (PRETRAIN_STEPS, {"dclm": 0.7, "dolmino_math": 0.3, "smoltalk": 0.0}),
        (PRETRAIN_STEPS + MIDTRAIN_STEPS, {"dclm": 0.0, "dolmino_math": 0.0, "smoltalk": 1.0}),
    ],
)
# Override tokenizer to use marin_tokenizer (same vocab as llama3 but with chat template for SFT)
data = dataclasses.replace(data, tokenizer=marin_tokenizer)
data = add_validation_sets_to_mixture(data, default_validation_sets(tokenizer=data.tokenizer))

# --- Training ---
training_step = ExecutorStep(
    name="reference-pipeline",
    fn=remote(run_grug_base_trial, resources=ResourceConfig.with_tpu("v4-8")),
    config=GrugBaseLaunchConfig(
        model=model,
        data=data,
        output_path=this_output_path(),
        run_id="reference-pipeline",
        resources=ResourceConfig.with_tpu("v4-8"),
        steps=TOTAL_STEPS,
        batch_size=256,
        seed=0,
        mp="params=float32,compute=bfloat16,output=bfloat16",
        tracker=WandbConfig(
            project="marin",
            tags=["reference", "pipeline"],
            group="reference-pipeline",
            name=None,
        ),
        optimizer=AdamConfig(
            learning_rate=3e-3,
            weight_decay=0.1,
            warmup=0.05,
            decay=0.2,
        ),
        eval=GrugEvalConfig(
            steps_per_eval=500,
        ),
    ),
)

if __name__ == "__main__":
    executor_main(steps=[training_step])
