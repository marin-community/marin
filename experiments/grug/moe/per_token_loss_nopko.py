# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-token eval-loss dumper for the no-PKO d=1024 1e19 run.

Companion to ``per_token_loss.py`` (which dumps the PKO-on baseline).
Targets ``grug-moe-nopko-d1024-1e19-v1``, which lives in us-central1, so the
v5p-8 is pinned to us-central1 to avoid a cross-region checkpoint read.

Submit on us-central1-a, interactive priority:

    .venv/bin/iris --cluster=marin job run \\
      --no-wait \\
      --zone us-central1-a \\
      --priority interactive \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.per_token_loss_nopko
"""

from __future__ import annotations

import dataclasses

from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path

from experiments.grug.moe.heuristic_v1 import build_from_heuristic
from experiments.grug.moe.launch import NEMOTRON_MIX_WITH_DEFAULT_VALIDATION
from experiments.grug.moe.optimizer import GrugMoeMuonHConfig
from experiments.grug.moe.per_token_loss import PerTokenLossConfig, run_per_token_loss

_TARGET_RUN_ID = "grug-moe-nopko-d1024-1e19-v1"
_TARGET_CHECKPOINT_BASE = "gs://marin-us-central1/grug/grug_moe_nopko/grug-moe-nopko-d1024-1e19-v1-dec1d6/checkpoints"


def _build_step() -> ExecutorStep:
    model, base_optimizer, batch_size, num_steps = build_from_heuristic(budget=1e19, hidden_dim=1024)
    # The training run used disable_pko=True; the eval forward must match so
    # the loss reflects the network the trained weights are paired with.
    model = dataclasses.replace(model, disable_pko=True)

    optimizer = GrugMoeMuonHConfig(
        learning_rate=base_optimizer.learning_rate,
        adam_lr=base_optimizer.adam_lr,
        min_lr_ratio=base_optimizer.min_lr_ratio,
        warmup=0.01,
        beta1=base_optimizer.beta1,
        beta2=base_optimizer.beta2,
        epsilon=base_optimizer.epsilon,
        max_grad_norm=None,
        lr_schedule=base_optimizer.lr_schedule,
        decay=base_optimizer.decay,
    )
    return ExecutorStep(
        name=f"grug/per_token_loss/{_TARGET_RUN_ID}",
        fn=run_per_token_loss,
        config=PerTokenLossConfig(
            run_id=f"{_TARGET_RUN_ID}-pertoken",
            checkpoint_base_path=_TARGET_CHECKPOINT_BASE,
            output_parquet=this_output_path(f"{_TARGET_RUN_ID}.parquet"),
            model=model,
            data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
            resources=ResourceConfig.with_tpu("v5p-8", regions=("us-central1",)),
            optimizer=optimizer,
            num_train_steps=num_steps,
            batch_size=batch_size,
            seq_len=4096,
            eval_batch_size=24,
            max_batches_per_set=1,
        ),
    )


if __name__ == "__main__":
    executor_main(
        steps=[_build_step()],
        description=(
            f"Per-token eval-loss dump for {_TARGET_RUN_ID} on paloma + uncheatable. "
            "24-batch x 4096-seq, one batch per tagged set. Forward pass uses "
            "disable_pko=True to match the trained network."
        ),
    )
