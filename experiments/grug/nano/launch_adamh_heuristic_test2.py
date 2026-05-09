# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Round-2 100-step diagnostics for the heuristic-AdamH NaN.

Round 1 (`launch_adamh_heuristic_test.py`) ruled out *lm-head routing*: both
Test A (lm-head on AdamW, non-zero init) and Test B (lm-head on AdamW, zero
init) NaN'd within 25 steps. Eval at step 0 was correct (10.985 nats with
non-zero lm-head, exactly ln(50304) = 10.825 with zero lm-head), so the
forward pass is fine — the optimizer is the problem.

The two heuristic hyperparameters most out of band vs. the working ref AdamH:

  - **ε = 6.76e-16** (vs ref Adam ε = 1e-8, AdamW ε = 1e-10) — eight orders of
    magnitude smaller. Combined with β2 = 0.996, the `m_hat / (sqrt(v_hat) + ε)`
    update is essentially `m / |g|` for any non-degenerate gradient, but a
    transient where v_hat lags m_hat could blow up.
  - **β2 = 0.996** (vs ref's 0.95) — slow second-moment EMA, long memory; pairs
    poorly with tiny ε.

Two tests, both starting from Test A (lm-head on AdamW, non-zero init,
heuristic optimizer):

  Test C — `may7-nano-heur-test-c-eps-1e8`
      Single change: ε = 1e-8 (standard Adam).
      If C trains cleanly, ε was the sole culprit.

  Test D — `may7-nano-heur-test-d-eps-beta2`
      Two changes: ε = 1e-8 AND β2 = 0.95.
      If C NaNs but D trains, β2 also matters.

Both share the same lm-head-on-AdamW mask from
`launch_adamh_heuristic_test.NanoHeuristicAdamHLmAdamConfig`. Run sequentially
under a single iris job (~10 min total).
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
    NANO_TEST_A_MODEL,
)
from experiments.grug.nano.train import GrugEvalConfig, GrugTrainerConfig

# Test C: ε = 1e-8 (standard), everything else inherited from the round-1 base.
NANO_HEUR_TEST_C_OPTIMIZER = dataclasses.replace(
    NANO_HEUR_TEST_OPTIMIZER,
    epsilon=1e-8,
)

# Test D: ε = 1e-8 AND β2 = 0.95.
NANO_HEUR_TEST_D_OPTIMIZER = dataclasses.replace(
    NANO_HEUR_TEST_OPTIMIZER,
    epsilon=1e-8,
    beta2=0.95,
)


def _make_test_step(name: str, run_id: str, optimizer, extra_tags: list[str]) -> ExecutorStep:
    return ExecutorStep(
        name=name,
        fn=run_nano_adamh_heuristic_trial,
        config=NanoAdamHHeuristicLaunchConfig(
            model=versioned(NANO_TEST_A_MODEL),  # non-zero lm head, default init
            data=_fineweb_gpt2_data(),
            output_path=this_output_path(),
            run_id=run_id,
            resources=versioned(ResourceConfig.with_tpu("v5p-8")),
            steps=versioned(NANO_HEUR_TEST_TRAIN_STEPS),
            batch_size=versioned(512),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                project="marin",
                tags=["grug", "nano", "adamh", "fineweb-gpt2", "heuristic", "diagnostic", *extra_tags],
                group="nano-trial",
                name=None,
                replicate_path=this_output_path(),
            ),
            optimizer=versioned(optimizer),
            grug_trainer=versioned(
                GrugTrainerConfig(
                    z_loss_weight=1e-4,
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


nano_heur_test_c = _make_test_step(
    name="grug/nano-heur-test-c-eps-1e8",
    run_id=_resolve_run_id("may7-nano-heur-test-c-eps-1e8"),
    optimizer=NANO_HEUR_TEST_C_OPTIMIZER,
    extra_tags=["test-c", "eps-1e8"],
)


nano_heur_test_d = _make_test_step(
    name="grug/nano-heur-test-d-eps-beta2",
    run_id=_resolve_run_id("may7-nano-heur-test-d-eps-beta2"),
    optimizer=NANO_HEUR_TEST_D_OPTIMIZER,
    extra_tags=["test-d", "eps-1e8", "beta2-0.95"],
)


if __name__ == "__main__":
    executor_main(
        steps=[nano_heur_test_c, nano_heur_test_d],
        description="Round-2 diagnostics: drop heuristic ε (test C) and ε+β2 (test D); 100 steps each.",
    )
