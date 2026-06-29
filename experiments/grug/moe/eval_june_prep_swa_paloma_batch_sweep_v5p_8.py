# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Re-eval june_prep d={512,768,1024} 80%-mark checkpoints on v5p-8 us-east5-a.

For each of the three june_prep_moe_may MoE runs (d=512/768/1024, EP=2,
no_long_rope, seq8192, sw=2048), reload the permanent checkpoint saved
near 80% of training and run a single paloma TaggedEvaluator pass with
``eval_batch_size=128`` and ``max_eval_batches`` in {4, 8, 16, 32}. No
training steps execute: ``num_train_steps`` is set to the checkpoint
step so the main loop's ``while state.step < num_train_steps`` exits
immediately, then the forced final ``state_callbacks.run(..., force=True)``
fires the registered ``cb_tagged_evaluate`` hook on the loaded state.

Each (dim, max_eval_batches) pair is a separate ExecutorStep / iris task
on its own v5p-8 slice — 3 dims * 4 batch counts = 12 jobs total.

Submit (us-east5-a, production, preemptible)::

    WANDB_KEY=$(python3 -c "import os; print(os.environ['WANDB_API_KEY'])") && \\
    .venv/bin/iris --cluster=marin job run --no-wait --zone us-east5-a \\
        --priority production -e WANDB_API_KEY "$WANDB_KEY" \\
        -- python -m experiments.grug.moe.eval_june_prep_swa_paloma_batch_sweep_v5p_8
"""

import dataclasses

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned

from experiments.grug.moe.heuristic_muonh import MoeMuonHHeuristic
from experiments.grug.moe.launch import NEMOTRON_MIX_WITH_DEFAULT_VALIDATION, GrugMoeLaunchConfig, run_grug_moe_trial
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_SEQ: int = 8192
_EP: int = 1
_SLICE: str = "v5p-8"
_EVAL_BATCH_SIZE: int = 128
_EVAL_BATCH_COUNTS: tuple[int, ...] = (4, 8, 16, 32)

# (dim, ckpt_step, ckpt_path) — ckpt_path is the EP=2 permanent save at ~80% completion.
_RUNS: tuple[tuple[int, int, str], ...] = (
    (
        512,
        71808,
        "gs://marin-us-east5/grug/june_prep_moe_may_d512_ep2_no_long_rope_seq8192_sw2k-c9de46/checkpoints/step-71808/",
    ),
    (
        768,
        210048,
        "gs://marin-us-east5/grug/june_prep_moe_may_d768_ep2_no_long_rope_seq8192_sw2k-2ffc9a/checkpoints/step-210048/",
    ),
    (
        1024,
        129312,
        "gs://marin-us-east5/grug/june_prep_moe_may_d1024_ep2_bs512_no_long_rope_seq8192_sw2k-17c621/checkpoints/step-129312/",
    ),
)

# v5p-8 has 4 chips. With EP=1, model=1, replica=1, the mesh is (1, 4, 1, 1)
# → batch_shards = 4. Set train_batch_size to the minimum divisible value (4),
# since num_train_steps == ckpt_step means no training-step batches are ever
# pulled; eval_batch_size below drives the only forward passes that run.
_TRAIN_BATCH_SIZE: int = 4

_heuristic = MoeMuonHHeuristic(min_lr_ratio=0.05)

steps: list[ExecutorStep] = []
for _dim, _ckpt_step, _ckpt_path in _RUNS:
    _model_base = _heuristic.build_model_config(_dim, seq_len=_SEQ)
    _model = dataclasses.replace(
        _model_base,
        disable_pko=True,
        disable_long_rope=True,
        sliding_window=2048,
    )
    # LR schedule is built off num_train_steps; tokens here is just for the
    # MuonH heuristic's LR formula and has no effect on the loaded weights.
    _tokens = float(_ckpt_step * _TRAIN_BATCH_SIZE * _SEQ)
    _optimizer = _heuristic.build_muonh_config(_TRAIN_BATCH_SIZE, _tokens, _dim, seq_len=_SEQ)

    for _max_eval in _EVAL_BATCH_COUNTS:
        _run_id = f"eval_june_prep_d{_dim}_ep{_EP}_step{_ckpt_step}_xb{_EVAL_BATCH_SIZE}_nb{_max_eval}_v5p_8"
        steps.append(
            ExecutorStep(
                name=f"grug/{_run_id}",
                fn=run_grug_moe_trial,
                config=GrugMoeLaunchConfig(
                    model=versioned(_model),
                    data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
                    output_path=this_output_path(),
                    run_id=_run_id,
                    resources=versioned(ResourceConfig.with_tpu(_SLICE)),
                    steps=versioned(_ckpt_step),
                    batch_size=versioned(_TRAIN_BATCH_SIZE),
                    seed=versioned(0),
                    mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
                    tracker=WandbConfig(
                        project="marin_moe",
                        tags=[
                            "moe",
                            "eval_only",
                            "june_prep_resume",
                            f"d{_dim}",
                            f"ep{_EP}",
                            "swa",
                            f"step{_ckpt_step}",
                            f"nb{_max_eval}",
                            "v5p_8",
                            "us_east5_a",
                        ],
                        group="june-prep-eval-batch-sweep",
                        name=None,
                    ),
                    optimizer=versioned(_optimizer),
                    expert_parallel=_EP,
                    load_checkpoint_path=_ckpt_path,
                    grug_trainer=versioned(
                        GrugTrainerConfig(
                            z_loss_weight=0.0,
                            ema_beta=None,
                            log_every=1,
                        )
                    ),
                    eval=versioned(
                        GrugEvalConfig(
                            eval_batch_size=_EVAL_BATCH_SIZE,
                            steps_per_eval=1000,
                            max_eval_batches=_max_eval,
                            eval_current=True,
                            eval_ema=False,
                        )
                    ),
                ),
            )
        )


if __name__ == "__main__":
    executor_main(
        steps=steps,
        description=(
            f"Eval-only reload of june_prep_moe_may d=512/768/1024 80%-mark checkpoints "
            f"on {_SLICE} us-east5-a. EP={_EP}, eval_batch_size={_EVAL_BATCH_SIZE}, "
            f"max_eval_batches in {_EVAL_BATCH_COUNTS}. {len(steps)} ExecutorSteps."
        ),
    )
