# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""20-step v4-2048 smoke test at d=2560, BS=8192, replica=4, EP=1.

Same recipe and data as ``moe_67b_a2b_d2560_10T.py`` with BS doubled to
8192 (~67M tokens/step) and ``replica_axis_size=4`` instead of EP. The
mesh is (4, 256, 1, 1): 4 data-parallel replicas of an FSDP-256 model.
Communication cost is one extra cross-replica grad all-reduce per step,
vs EP=16's 26-layer x 2 token all-to-alls — meant to test whether the
cheaper collective wins despite the 4x param/state duplication and 12.8x
MXU pad bloat (per-shard hidden = 2560/256 = 10). ``num_train_steps=20``
is just a boot + step-time probe.

Submit (us-central2)::

    WANDB_KEY=$(python3 -c "import os; print(os.environ['WANDB_API_KEY'])") && \\
    .venv/bin/iris --cluster=marin job run --no-wait --region us-central2 \\
        --priority interactive -e WANDB_API_KEY "$WANDB_KEY" \\
        -- python -m experiments.grug.moe.moe_67b_a2b_d2560_v4_2048_test
"""

import dataclasses

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned

from experiments.grug.moe.heuristic_muonh import MoeMuonHHeuristic
from experiments.grug.moe.launch import GrugMoeLaunchConfig, run_grug_moe_trial
from experiments.grug.moe.launch_datakit_moe_mix import _datakit_data_config
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_DIM: int = 2560
_BS: int = 8192  # doubled from the 10T's 4096 (~67M tokens/step)
_SEQ: int = 8192
_STEPS: int = 20
# v4-2048 (1024 chips), replica=2, EP=1 → data=512, per-shard hidden 2560/512=5
# (25.6x MXU pad bloat). 2 replicas of the FSDP-512 model, one grad all-reduce
# across replicas per step.
_EP: int = 1
_REPLICA_AXIS: int = 2
_SLICE: str = "v4-2048"
_LOGIT_Z_LOSS_WEIGHT: float = 1e-4

_heuristic = MoeMuonHHeuristic(min_lr_ratio=0.05)
_model_base = _heuristic.build_model_config(_DIM, seq_len=_SEQ)
_model = dataclasses.replace(
    _model_base,
    disable_pko=True,
    disable_long_rope=True,
    sliding_window=2048,
    use_array_stacked_blocks=True,
)
_tokens = float(_STEPS * _BS * _SEQ)
_optimizer = _heuristic.build_muonh_config(_BS, _tokens, _DIM, seq_len=_SEQ)

_run_id = f"moe_67b_a2b_d{_DIM}_ep{_EP}_rep{_REPLICA_AXIS}_bs{_BS}_seq{_SEQ}_sw2k_v4_2048_test"
step = ExecutorStep(
    name=f"grug/{_run_id}",
    fn=run_grug_moe_trial,
    config=GrugMoeLaunchConfig(
        model=versioned(_model),
        data=_datakit_data_config(
            total_steps=_STEPS,
            batch_size=_BS,
            max_seq_len=_SEQ,
            enable_simulated_epoching=False,
        ),
        output_path=this_output_path(),
        run_id=_run_id,
        resources=versioned(ResourceConfig.with_tpu(_SLICE, preemptible=False)),
        steps=versioned(_STEPS),
        batch_size=versioned(_BS),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin_moe",
            tags=[
                "moe",
                "june_tpu",
                "67b_a2b",
                f"d{_DIM}",
                f"ep{_EP}",
                "datakit_mix",
                "disable_pko",
                "no_long_rope",
                "seq8k",
                "stacked",
                "logit_z_loss",
                "v4_2048",
                "smoke",
            ],
            group="june-tpu-67b-a2b-v4-2048-smoke",
            name=None,
        ),
        optimizer=versioned(_optimizer),
        expert_parallel=_EP,
        grug_trainer=versioned(
            GrugTrainerConfig(
                z_loss_weight=_LOGIT_Z_LOSS_WEIGHT,
                ema_beta=None,
                log_every=1,
                replica_axis_size=_REPLICA_AXIS,
            )
        ),
        eval=versioned(
            GrugEvalConfig(
                # v4-2048 = 1024 devices, replica=2, EP=1 → data=512.
                # Batch is sharded across (replica_dcn, data, expert) =
                # 2 * 512 * 1 = 1024 shards, so eval_batch_size must be
                # divisible by 1024.
                eval_batch_size=1024,
                steps_per_eval=_STEPS,
                max_eval_batches=1,
                eval_current=True,
                eval_ema=False,
            )
        ),
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[step],
        description=(
            f"d={_DIM} EP={_EP} BS={_BS} seq={_SEQ} {_STEPS}-step smoke test on {_SLICE}. "
            f"TPB={_BS*_SEQ:,} (~67M tokens/step) — doubled from the 10T config to "
            f"check boot + step time before the production move."
        ),
    )
