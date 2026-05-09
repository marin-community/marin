# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Muon walk p16 with final-logit z-loss enabled (z_loss_weight = 1e-4).

Identical to ``launch_muon_tuned_walk_p16.py`` except for one line: the
``GrugTrainerConfig`` has ``z_loss_weight=1e-4`` instead of ``0.0``. This
makes the trainer pass ``logsumexp_weight=1e-4`` into ``next_token_loss``,
so the fused kernel adds ``1e-4 * logsumexp(logits)**2`` per position to
the cross-entropy. The router z-loss (``router_z_loss_coef=1e-3`` from p13)
was already on; this run adds the *final-logit* z-loss on top so muon
matches the adamh-side recipe exactly.
"""

from fray.cluster import ResourceConfig
from jax.sharding import PartitionSpec as P
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.nano.launch import (
    NanoAdamWMuonConfig,
    NanoLaunchConfig,
    _resolve_run_id,
    run_nano_trial,
)
from experiments.grug.nano.launch_muon_tuned_walk_p16 import NEMOTRON_MIX_WITH_DEFAULT_VALIDATION, P16_MODEL
from experiments.grug.nano.train import GrugEvalConfig, GrugTrainerConfig

P16_MUONZ_TRAIN_STEPS = 10343
P16_MUONZ_BATCH_SIZE = 64
NANO_MUON_TUNED_LR = 0.035
NANO_MUON_TUNED_WD = 0.025

P16_MUONZ_OPTIMIZER = NanoAdamWMuonConfig(
    muon_lr=NANO_MUON_TUNED_LR,
    muon_weight_decay=NANO_MUON_TUNED_WD,
)


RESOLVED_RUN_ID = _resolve_run_id("may7-nano-muon-tuned-p16-muonz")


nano_muon_tuned_p16_muonz_trial = ExecutorStep(
    name="grug/nano-muon-tuned-p16-muonz-trial",
    fn=run_nano_trial,
    config=NanoLaunchConfig(
        model=versioned(P16_MODEL),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        output_path=this_output_path(),
        run_id=RESOLVED_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        steps=versioned(P16_MUONZ_TRAIN_STEPS),
        batch_size=versioned(P16_MUONZ_BATCH_SIZE),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin",
            tags=["grug", "nano", "muon", "nemotron", "tuned", "p16", "moe", "fused-ce", "muonz"],
            group="nano-trial",
            name=None,
            replicate_path=this_output_path(),
        ),
        optimizer=versioned(P16_MUONZ_OPTIMIZER),
        grug_trainer=versioned(
            GrugTrainerConfig(
                z_loss_weight=1e-4,
                ema_beta=None,
                log_every=1,
                train_batch_pspec=P(("data", "expert")),
            )
        ),
        eval=versioned(
            GrugEvalConfig(
                eval_batch_size=P16_MUONZ_BATCH_SIZE,
                steps_per_eval=250,
                max_eval_batches=40,
                eval_current=True,
                eval_ema=False,
                eval_batch_pspec=P(("data", "expert")),
            )
        ),
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[nano_muon_tuned_p16_muonz_trial],
        description="muon-tuned p16 + final-logit z_loss=1e-4 (matches adamh recipe).",
    )
