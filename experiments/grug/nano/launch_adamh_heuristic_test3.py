# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Round-3 100-step diagnostic for the heuristic-AdamH NaN.

Round 1 (`launch_adamh_heuristic_test.py`): tests A and B both NaN'd,
ruling out the lm-head routing.

Round 2 (`launch_adamh_heuristic_test2.py`): tests C and D both NaN'd,
ruling out ε=6.76e-16 and β2=0.996 individually.

This round tests the **z_loss** hypothesis, which is the only remaining
config-dependent code-path difference inside `next_token_loss` between the
working `nano-adamh` (z_loss_weight=0, branch skipped) and the broken
heuristic (z_loss_weight=1e-4, `jax.scipy.special.logsumexp` branch entered).

  Test E — `may7-nano-heur-test-e-no-zloss`
      Identical to Test B (init_scheme="adamh_ref" → zero lm head,
      heuristic optimizer with lm head routed to AdamW, ε and β2 still at
      heuristic defaults) but with **z_loss_weight = 0**. The lse-squared
      branch in `next_token_loss` is skipped entirely.

If E trains cleanly AND the train/loss = eval/loss / 4 reporting bug
disappears, then `jax.scipy.special.logsumexp` interacting badly with our
explicit-mesh sharding is the proximate cause of both issues.
"""

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.nano.launch_adamh_heuristic import (
    NanoAdamHHeuristicLaunchConfig,
    _fineweb_gpt2_data,
    _resolve_run_id,
    run_nano_adamh_heuristic_trial,
)
from experiments.grug.nano.launch_adamh_heuristic_test import (
    NANO_HEUR_TEST_OPTIMIZER,
    NANO_HEUR_TEST_TRAIN_STEPS,
    NANO_TEST_B_MODEL,
)
from experiments.grug.nano.train import GrugEvalConfig, GrugTrainerConfig

nano_heur_test_e = ExecutorStep(
    name="grug/nano-heur-test-e-no-zloss",
    fn=run_nano_adamh_heuristic_trial,
    config=NanoAdamHHeuristicLaunchConfig(
        model=versioned(NANO_TEST_B_MODEL),  # zero lm head (adamh_ref init)
        data=_fineweb_gpt2_data(),
        output_path=this_output_path(),
        run_id=_resolve_run_id("may7-nano-heur-test-e-no-zloss"),
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        steps=versioned(NANO_HEUR_TEST_TRAIN_STEPS),
        batch_size=versioned(512),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin",
            tags=["grug", "nano", "adamh", "fineweb-gpt2", "heuristic", "diagnostic", "test-e", "no-zloss"],
            group="nano-trial",
            name=None,
            replicate_path=this_output_path(),
        ),
        optimizer=versioned(NANO_HEUR_TEST_OPTIMIZER),
        grug_trainer=versioned(
            GrugTrainerConfig(
                z_loss_weight=0.0,  # <-- the only change vs Test B
                ema_beta=None,
                log_every=1,
            )
        ),
        eval=versioned(
            GrugEvalConfig(
                eval_batch_size=512,
                steps_per_eval=25,
                max_eval_batches=20,
                eval_current=True,
                eval_ema=False,
            )
        ),
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[nano_heur_test_e],
        description="Round-3 diagnostic: heuristic + lm-head-on-AdamW + z_loss=0 (test E); 100 steps.",
    )
