# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""10T-token long-run launcher: d=2560 / 67B total, 2B active per token.

Production-style run after the d=2560 MFU probes converged at ~21.4% MFU.
Branch: ``june_tpu_67b_a2b``. Inherits the grugmuon 4D NS fix so MoE expert
weights actually go through Newton-Schulz.

Config:
- d=2560 (V2 / MuonH heuristic; num_layers=26, NH=20, KV=5, HD=128,
  num_experts=256, num_experts_per_token=4, intermediate=1280,
  shared_expert_intermediate_dim=2560) → ~67B total params, ~2B active/token
- BS=4096 sequences, seq=8192 → 33,554,432 tokens/step (~33.5M)
- num_train_steps=300,000 → 10.07T total tokens
- LR: MuonH default schedule, ``min_lr_ratio=0.05`` (decay to 5% of peak)
- z-loss: logit-only ``z_loss_weight=1e-4`` (router z-loss stays at the model
  default of 0.0)
- Sliding window 2048, ``disable_pko=True``, ``disable_long_rope=True``
- Stacked blocks + bf16 NS + distributed 4D MoE NS
- EP=1, ``replica_axis_size=1`` on v4-1024: mesh (1, 512, 1, 1) — pure
  FSDP across all 512 chips, no per-MoE-layer all-to-all, no cross-replica
  grad reduction. The sharding.py change that drops ``replica_dcn`` from
  Pembed_vocab / Plm_head is a no-op at replica=1 (the combined
  ``(replica_dcn, data)`` axis was already just the data axis with size 1
  on the leading slot).
- Checkpoints: permanent saves every 3,000 steps; temp save every 10 min
  (default); ``load_checkpoint=None`` auto-resumes on preemption (via the
  launch.py fix landed in june_tpu_67b_a2b's parent)
- Eval every 3,000 steps
- ``enable_simulated_epoching=False``

Submit (us-central2, production, --no-preemptible)::

    WANDB_KEY=$(python3 -c "import os; print(os.environ['WANDB_API_KEY'])") && \\
    .venv/bin/iris --cluster=marin job run --no-wait --region us-central2 \\
        --priority production --no-preemptible -e WANDB_API_KEY "$WANDB_KEY" \\
        -- python -m experiments.grug.moe.moe_67b_a2b_d2560_10T
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
_BS: int = 4096  # 4096 * 8192 = 33,554,432 tokens/step (~33.5M)
_SEQ: int = 8192
_STEPS: int = 300_000  # ~10.07T total tokens
_EP: int = 1
_REPLICA_AXIS: int = 16
_SLICE: str = "v4-2048"
_LOGIT_Z_LOSS_WEIGHT: float = 1e-4
_CHECKPOINT_EVERY: int = 3_000
_STEPS_PER_EVAL: int = 3_000

_heuristic = MoeMuonHHeuristic(min_lr_ratio=0.05)
_model_base = _heuristic.build_model_config(_DIM, seq_len=_SEQ)
_model = dataclasses.replace(
    _model_base,
    disable_pko=True,  # no PKO on long layers
    disable_long_rope=True,  # no RoPE on long layers
    sliding_window=2048,
    use_array_stacked_blocks=True,
)
_tokens = float(_STEPS * _BS * _SEQ)
_optimizer = _heuristic.build_muonh_config(_BS, _tokens, _DIM, seq_len=_SEQ)
# Route stacked RMSNorm scales to plain Adam instead of muonh's NS+Frobenius
# hyperball — see the d=512 rmsadam ablation.
_optimizer = dataclasses.replace(_optimizer, rmsnorm_to_adam=True)

_run_id = f"moe_67b_a2b_d{_DIM}_ep{_EP}_rep{_REPLICA_AXIS}_bs{_BS}_seq{_SEQ}_sw2k_v4_2048_muon_10T"
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
                f"rep{_REPLICA_AXIS}",
                "datakit_mix",
                "disable_pko",
                "no_long_rope",
                "seq8k",
                "stacked",
                "logit_z_loss",
                "rmsadam",
                "muon",
                "v4_2048",
                "10T",
            ],
            group="june-tpu-67b-a2b-10T",
            name=None,
        ),
        optimizer=versioned(_optimizer),
        expert_parallel=_EP,
        checkpoint_keep=[{"every": _CHECKPOINT_EVERY}],
        save_interval_minutes=60,
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
                # v4-2048 = 1024 devices, replica=16, EP=1 → data=64.
                # Batch is sharded across (replica_dcn, data, expert) =
                # 16 * 64 * 1 = 1024 shards, so eval_batch_size must be
                # divisible by 1024.
                eval_batch_size=1024,
                steps_per_eval=_STEPS_PER_EVAL,
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
            f"d={_DIM} EP={_EP} BS={_BS} seq={_SEQ} 10T-token long run on {_SLICE}. "
            f"TPB={_BS*_SEQ:,} (~33.5M tokens/step), {_STEPS:,} steps → "
            f"{_STEPS*_BS*_SEQ/1e12:.2f}T tokens total. "
            f"MuonH min_lr_ratio=0.05, logit z_loss={_LOGIT_Z_LOSS_WEIGHT}, "
            f"sliding_window=2048, disable_pko, disable_long_rope, "
            f"ckpt every {_CHECKPOINT_EVERY} steps, eval every {_STEPS_PER_EVAL}."
        ),
    )
