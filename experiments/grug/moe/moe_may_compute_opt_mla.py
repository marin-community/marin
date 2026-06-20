# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""May Recipe compute-optimal at d=512 and d=768 with Multi-head Latent Attention.

Architectural sibling of ``moe_may_compute_opt`` (same heuristic, same
compute-optimal points, same TPU, same data, same LR / beta / epsilon).
The only change is in ``experiments/grug/moe/model.py``:

  - ``CausalSelfAttention`` is now Multi-head Latent Attention (MLA) with a
    single ``kv_compression_dim`` knob; rope-bearing K is shared across heads
    (per DeepSeek-V2), no-rope K and V are up-projected from a compressed
    latent. Q is projected directly (no Q-side compression).
  - Removed: attention gate, XSA (exclusive self-attention), PKO (partial key
    offset), sliding window, ``num_kv_heads`` (GQA), all long/short branching.

This is a clean-comparison ablation against the agent.md gate-1 anchor:
identical training conditions, only the attention block differs.

Compute budgets (drop-1e18 isoflop fit, issue #6074), batch sizes from the
README pattern, LR / beta2 / epsilon from ``MoeHeuristicV1`` (issue #5951):

  d=512  → C ≈ 3.82e17, tokens ≈ 1.44e9, BS=32,  steps=10980
  d=768  → C ≈ 2.81e18, tokens ≈ 4.42e9, BS=64,  steps=16875

Submit on us-east5-a, interactive priority, v5p-8 (per agent.md)::

    .venv/bin/iris --cluster=marin job run \\
      --no-wait \\
      --zone us-east5-a \\
      --priority interactive \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.moe_may_compute_opt_mla
"""


from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned

from experiments.grug.moe.heuristic_v1 import MoeHeuristicV1
from experiments.grug.moe.launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
    run_grug_moe_trial,
)
from experiments.grug.moe.optimizer import GrugMoeMuonHConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_SEQ_LEN: int = 4096
_TPU: str = "v5p-8"
_GROUP_NAME: str = "moe-may-compute-opt-mla"
_WARMUP_FRACTION: float = 0.01

# (hidden_dim, batch_size, num_steps) -- same compute-optimal points as the
# baseline ``moe_may_compute_opt`` so paired comparisons are 1:1.
_POINTS: tuple[tuple[int, int, int], ...] = (
    (512, 32, 10980),
    (768, 64, 16875),
)


def _build_step(hidden_dim: int, batch_size: int, num_steps: int) -> ExecutorStep:
    h = MoeHeuristicV1()
    model = h.build_model_config(hidden_dim, seq_len=_SEQ_LEN)
    tokens = float(num_steps * batch_size * _SEQ_LEN)
    base_optimizer = h.build_optimizer_config(batch_size, tokens, hidden_dim, seq_len=_SEQ_LEN)

    optimizer = GrugMoeMuonHConfig(
        learning_rate=base_optimizer.learning_rate,
        adam_lr=base_optimizer.adam_lr,
        min_lr_ratio=base_optimizer.min_lr_ratio,
        warmup=_WARMUP_FRACTION,
        beta1=base_optimizer.beta1,
        beta2=base_optimizer.beta2,
        epsilon=base_optimizer.epsilon,
        max_grad_norm=None,
        lr_schedule=base_optimizer.lr_schedule,
        decay=base_optimizer.decay,
    )

    run_id = f"moe_may_compute_opt_mla_d{hidden_dim}"
    step_name = f"grug/moe_may_compute_opt_mla/{run_id}"

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
                    "mla",
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
            "May Recipe compute-optimal at d=512/768 with Multi-head Latent Attention. "
            "Drop attention gate / XSA / PKO / sliding window / num_kv_heads; replace "
            f"QKV with MLA (kv_compression_dim = hidden_dim // 4 by default). "
            f"Same heuristic / batch / steps / TPU as baseline. {_POINTS=}, TPU={_TPU}."
        ),
    )
