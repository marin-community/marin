# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""May Recipe compute-optimal at d=512 and d=768 with bf16 NS — V2 / MuonH LR curve.

Companion to ``moe_may_compute_opt_bf16_ns`` (which uses ``MoeHeuristicV1``).
This launcher uses ``MoeHeuristicV2`` — the MuonH ISOFlop refit (issue #5951)
that the live ``moe_may_june_prep_*`` runs and most current "May Recipe"
production training is on. Same NS-precision change (bf16 NS body, fp32
momentum buffer), same model size / batch / steps, same compute-optimal
points; only the LR / beta2 / epsilon schedule differs from the V1 variant.

This sits next to the V1 variant so we get coverage on both schedules:

- V1 copy: matches the ``agent.md`` gate-1 effective-speedup reference.
- V2 copy: matches the LR regime of the live MuonH-heuristic baselines.

Compute budgets (drop-1e18 isoflop fit, issue #6074) same as the V1 launcher:

For d=512  -> C ≈ 3.82e17, tokens ≈ 1.44e9
For d=768  -> C ≈ 2.81e18, tokens ≈ 4.42e9

Submit on us-east5-a, interactive priority, v5p-8 (per agent.md)::

    .venv/bin/iris --cluster=marin job run \\
      --no-wait \\
      --zone us-east5-a \\
      --priority interactive \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.moe_may_compute_opt_bf16_ns_v2
"""


from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned

from experiments.grug.moe.heuristic_v2 import MoeHeuristicV2
from experiments.grug.moe.launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
    run_grug_moe_trial,
)
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_SEQ_LEN: int = 4096
_TPU: str = "v5p-8"
_GROUP_NAME: str = "moe-may-compute-opt-bf16-ns-v2"

# (hidden_dim, batch_size, num_steps) — same compute-optimal points as the
# baseline / V1 launchers so paired comparisons are 1:1.
_POINTS: tuple[tuple[int, int, int], ...] = (
    (512, 32, 10980),
    (768, 64, 16875),
)


def _build_step(hidden_dim: int, batch_size: int, num_steps: int) -> ExecutorStep:
    h = MoeHeuristicV2()
    model = h.build_model_config(hidden_dim, seq_len=_SEQ_LEN)
    tokens = float(num_steps * batch_size * _SEQ_LEN)
    # V2.build_optimizer_config already returns a GrugMoeMuonHConfig with the
    # May Recipe 1pct-noclip schedule (warmup=0.01, max_grad_norm=None) — no
    # rewrap needed (unlike V1 which returns an AdamH config).
    optimizer = h.build_optimizer_config(batch_size, tokens, hidden_dim, seq_len=_SEQ_LEN)

    run_id = f"moe_may_compute_opt_bf16_ns_v2_d{hidden_dim}"
    step_name = f"grug/moe_may_compute_opt_bf16_ns_v2/{run_id}"

    return ExecutorStep(
        name=step_name,
        fn=run_grug_moe_trial,
        config=GrugMoeLaunchConfig(
            model=versioned(model),
            data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
            output_path=this_output_path(),
            run_id=run_id,
            resources=versioned(ResourceConfig.with_tpu(_TPU)),
            steps=versioned(num_steps),
            batch_size=versioned(batch_size),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                entity="marin-community",
                project="marin_moe",
                tags=[
                    "moe",
                    "moe_may_compute_opt",
                    "bf16_ns",
                    "heuristic_v2",
                    f"d{hidden_dim}",
                ],
                group=_GROUP_NAME,
                name=None,
            ),
            optimizer=versioned(optimizer),
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
                    steps_per_eval=1000,
                    max_eval_batches=8,
                    eval_current=True,
                    eval_ema=False,
                )
            ),
        ),
    )


if __name__ == "__main__":
    steps = [_build_step(d, bs, n) for (d, bs, n) in _POINTS]
    executor_main(
        steps=steps,
        description=(
            "May Recipe compute-optimal at d=512/768 with bf16 Newton-Schulz, V2 / MuonH "
            "heuristic (May Recipe refit, issue #5951). NS iteration body in bf16, momentum "
            "buffer fp32. Companion to the V1 launcher; this variant sits on the same LR "
            f"curve as the live moe_may_june_prep_* runs. d/bs/steps in {_POINTS}, TPU={_TPU}."
        ),
    )
