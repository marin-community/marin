# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tuned-Muon variant of `nano_trial`, mirroring `experiments/grug/muon_tuned.py`.

Same nano architecture and AdamW+Muon optimizer as `launch.py:nano_trial`,
with the tuned hyperparameters from `experiments/grug/muon_tuned.py`:

- `train_steps`:        3600 -> **3350**
- `muon_lr`:            0.02 -> **0.035**
- `muon_weight_decay`:  0.01 -> **0.025**
- Init scheme:          `default` -> **`muon_tuned`**
    - embed: ``N(0, 1)`` (PyTorch ``nn.Embedding`` default)
    - non-proj weights: ``N(0, sqrt(0.33 / fan_in))`` (PyTorch ``nn.Linear`` default)
    - proj weights, biases: zeros (same as the original ref)
    - RMSNorm gains: ones

Everything else (data, schedule shape, NS coefficients, AdamW per-group LRs,
β/ε, eval cadence) is unchanged from `nano_trial`.
"""

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.nano.launch import (
    NanoAdamWMuonConfig,
    NanoLaunchConfig,
    _fineweb_gpt2_data,
    _resolve_run_id,
    run_nano_trial,
)
from experiments.grug.nano.model import NanoModelConfig
from experiments.grug.nano.train import GrugEvalConfig, GrugTrainerConfig

NANO_MUON_TUNED_TRAIN_STEPS = 3350
NANO_MUON_TUNED_LR = 0.035
NANO_MUON_TUNED_WD = 0.025


NANO_124M_MUON_TUNED_MODEL = NanoModelConfig(
    vocab_size=50304,
    hidden_dim=768,
    intermediate_dim=3072,
    num_layers=12,
    num_heads=6,
    head_dim=128,
    max_seq_len=1024,
    init_scheme="muon_tuned",
)


NANO_MUON_TUNED_OPTIMIZER = NanoAdamWMuonConfig(
    muon_lr=NANO_MUON_TUNED_LR,
    muon_weight_decay=NANO_MUON_TUNED_WD,
)


RESOLVED_RUN_ID = _resolve_run_id("may7-nano-muon-tuned-rsqrt_cap-v2")


nano_muon_tuned_trial = ExecutorStep(
    name="grug/nano-muon-tuned-trial",
    fn=run_nano_trial,
    config=NanoLaunchConfig(
        model=versioned(NANO_124M_MUON_TUNED_MODEL),
        data=_fineweb_gpt2_data(),
        output_path=this_output_path(),
        run_id=RESOLVED_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        steps=versioned(NANO_MUON_TUNED_TRAIN_STEPS),
        batch_size=versioned(512),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin",
            tags=["grug", "nano", "muon", "fineweb-gpt2", "tuned"],
            group="nano-trial",
            name=None,
            replicate_path=this_output_path(),
        ),
        optimizer=versioned(NANO_MUON_TUNED_OPTIMIZER),
        grug_trainer=versioned(
            GrugTrainerConfig(
                z_loss_weight=0.0,
                ema_beta=None,
                log_every=1,
            )
        ),
        eval=versioned(
            GrugEvalConfig(
                eval_batch_size=512,
                steps_per_eval=125,
                max_eval_batches=20,
                eval_current=True,
                eval_ema=False,
            )
        ),
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[nano_muon_tuned_trial],
        description="Nano (modded-nanogpt) 124M, tuned Muon (lr=0.035, wd=0.025), 3350 steps on fineweb10B-gpt2.",
    )
