# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Round-4 100-step diagnostic for the heuristic-AdamH NaN.

Rounds 1-3 (`launch_adamh_heuristic_test{,2,3}.py`) ruled out:
- lm-head routing (test A: non-zero lm-head + AdamW; test B: zero + AdamW)
- ε too small (test C: ε = 1e-8 instead of 6.76e-16)
- β2 too high (test D: β2 = 0.95 in addition to ε = 1e-8)
- z_loss (test E: z_loss_weight = 0)

All NaN'd. Test F flips `max_grad_norm` from 1.0 (heuristic default) to None,
on the hypothesis that `optax.clip_by_global_norm` is doing a tree-wide L2
sum that needs an explicit psum across data-sharded grads — under
`use_explicit_mesh_axes=True` that psum may not be inserted automatically,
which would explain both the `train/loss = eval/loss / 4` reporting bug
*and* the per-shard divergent training that ends in NaN.

  Test F — `may7-nano-heur-test-f-no-clip`
      Identical to Test E (heuristic + lm-head-on-AdamW + zero lm-head init
      + z_loss=0) but with **max_grad_norm = None**, so the
      `clip_by_global_norm` step is skipped entirely.

Result: Test F trained cleanly, confirming the diagnosis. Don't enable
`clip_by_global_norm` under our explicit-mesh configuration without an
explicit psum on the global-norm reduction.
"""

import dataclasses

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

# Drop the global-norm clip; everything else stays at the heuristic defaults.
NANO_HEUR_TEST_F_OPTIMIZER = dataclasses.replace(
    NANO_HEUR_TEST_OPTIMIZER,
    max_grad_norm=None,
)


nano_heur_test_f = ExecutorStep(
    name="grug/nano-heur-test-f-no-clip",
    fn=run_nano_adamh_heuristic_trial,
    config=NanoAdamHHeuristicLaunchConfig(
        model=versioned(NANO_TEST_B_MODEL),
        data=_fineweb_gpt2_data(),
        output_path=this_output_path(),
        run_id=_resolve_run_id("may7-nano-heur-test-f-no-clip"),
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        steps=versioned(NANO_HEUR_TEST_TRAIN_STEPS),
        batch_size=versioned(512),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin",
            tags=["grug", "nano", "adamh", "fineweb-gpt2", "heuristic", "diagnostic", "test-f", "no-clip"],
            group="nano-trial",
            name=None,
            replicate_path=this_output_path(),
        ),
        optimizer=versioned(NANO_HEUR_TEST_F_OPTIMIZER),
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
        steps=[nano_heur_test_f],
        description="Round-4 diagnostic: heuristic + max_grad_norm=None (test F); 100 steps.",
    )
