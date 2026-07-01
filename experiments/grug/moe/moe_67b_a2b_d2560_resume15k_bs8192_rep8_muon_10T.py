# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resume the 67B/2B-active MuonH 10T pretrain at step 15,000 with BS doubled to 8,192.

Source run::

    run_id   moe_67b_a2b_d2560_ep1_rep16_bs4096_seq8192_sw2k_v4_2048_muon_10T
    output   gs://marin-us-central2/grug/moe_67b_a2b_d2560_ep1_rep16_bs4096_seq8192_sw2k_v4_2048_muon_10T-93542a
    mesh     v4-2048, replica=16, EP=1, BS=4096

This run::

- Loads ``step-15000`` from the source on the first launch only; iris
  restarts auto-resume from this run's own output dir (see
  ``GrugMoeLaunchConfig2xBS.initialize_from_path``).
- Doubles BS: 4096 -> 8192  (tokens/step 33.5M -> 67.1M)
- Reduces total steps to 157,500 so total tokens stay near 10T
  (15,000 * 33.5M + 142,500 * 67.1M = ~10.07T)
- Switches to replica=8 (mesh = (8, 128, 1, 1), batch_shards = 1024)
- LR schedule (see ``GrugMoeMuonHResumeConfig``):

  ===========  ==============================================================
  step         muonh_lr
  ===========  ==============================================================
  15,000       0.003590  (matches source LR at this step:
                          pre-ramp peak 0.003734 * decay-multiplier 0.9616
                          where multiplier = 1 - (12,000/297,000) * 0.95)
  15,000-100   linear ramp to 0.005078
  15,100       0.005078  (= new peak 0.005281 * same 0.9616 multiplier,
                          i.e. 5% into the new decay schedule)
  15,100-end   linear decay
  157,500      0.000264  (= 0.05 * new peak 0.005281, which is
                          sqrt(2) * source floor 0.0001867)
  ===========  ==============================================================

  ``adam_lr`` follows the same shape with peak = muonh_lr / (13/3).
  ``epsilon`` and ``beta2`` come from the heuristic at the new B*S (eps ÷ sqrt(2);
  beta2 still clamps to 0.95). Schedule shape preserved across resume so there
  is no LR discontinuity at step 15,000.

Submit (us-central2, production, --no-preemptible)::

    WANDB_KEY=$(python3 -c "import os; print(os.environ['WANDB_API_KEY'])") && \\
    .venv/bin/iris --cluster=marin job run --no-wait --region us-central2 \\
        --priority production --no-preemptible -e WANDB_API_KEY "$WANDB_KEY" \\
        -- python -m experiments.grug.moe.moe_67b_a2b_d2560_resume15k_bs8192_rep8_muon_10T
"""

import dataclasses

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned

from experiments.grug.moe.heuristic_muonh import MoeMuonHHeuristic
from experiments.grug.moe.launch_2x_bs import (
    GrugMoeLaunchConfig2xBS,
    GrugMoeMuonHResumeConfig,
    run_grug_moe_trial_2x_bs,
)
from experiments.grug.moe.launch_datakit_moe_mix import _datakit_data_config
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_DIM: int = 2560
_BS_SOURCE: int = 4096  # source-run BS, used for the piecewise BS schedule's pre-resume segment
_BS_NEW: int = 8192  # 8192 * 8192 = 67,108,864 tokens/step (~67.1M)
_SEQ: int = 8192
_TOTAL_STEPS: int = 157_500  # 15,000 (BS=4096 phase) + 142,500 (BS=8192 phase) → ~10.07T
_EP: int = 1
_REPLICA_AXIS: int = 8
_SLICE: str = "v4-2048"
_LOGIT_Z_LOSS_WEIGHT: float = 1e-4
_CHECKPOINT_EVERY: int = 3_000
_STEPS_PER_EVAL: int = 3_000

# Mesh (replica=8, data=128, expert=1, model=1) on v4-2048 (1,024 chips):
# batch_shards = 8 * 128 * 1 = 1024. BS=8192 → 8 sequences per chip.
_BATCH_SHARDS: int = _REPLICA_AXIS * (1024 // _REPLICA_AXIS // _EP) * _EP  # = 1024
_PER_DEVICE_PARALLELISM: int = _BS_NEW // _BATCH_SHARDS  # = 8

# Phase-0 → phase-1 mixture transition, pinned to 80% of tokens (the source's
# design intent) rather than 80% of steps. With the BS=4096 head consuming
# only 5% of tokens despite being 10% of steps, the default
# `_phase_1_start_step` would land at 79.0% of tokens; bumping by 1,500
# steps puts it at exactly 80.00%. Multiple of step_multiple=6 for BS=8192.
_PHASE_1_START_STEP: int = 127_500

# LR transition at the BS-ramp boundary (pre-computed; see job summary doc).
_RESUME_STEP: int = 15_000
_RAMP_END_STEP: int = 15_100  # 100-step linear ramp
_LR_AT_RESUME: float = 0.003590
"""Source muonh_lr at step 15k. Pre-ramp peak 0.003734 * decay-multiplier
0.9616 (= 1 - (12,000 / 297,000) * 0.95)."""
_LR_PEAK_NEW: float = 0.005281
"""BS-doubled heuristic peak (=pre-ramp peak * sqrt(2)). The decay floor
of the new schedule = ``_LR_PEAK_NEW * 0.05 = 0.000264`` (= sqrt(2) * the
source floor 0.0001867, since the whole MuonH schedule scales with
sqrt(B*S))."""
_LR_AT_RAMP_END: float = 0.005078
"""LR target at step 15,100 (end of 100-step ramp). Equals
``_LR_PEAK_NEW * 0.9616`` -- the same 5%-into-decay multiplier applied to
the doubled-BS peak. Joining the doubled schedule here keeps the post-ramp
LR trajectory identical to a fresh BS=8192 run that had been training
since step 0."""
_ADAMH_RATIO: float = 13.0 / 3.0

# Source checkpoint path. Loaded once on first launch; auto-resume from this
# run's own output dir afterward (initialize_from_path semantics).
_RESUME_CKPT_PATH: str = (
    "gs://marin-us-central2/grug/"
    "moe_67b_a2b_d2560_ep1_rep16_bs4096_seq8192_sw2k_v4_2048_muon_10T-93542a/"
    "checkpoints/step-15000/"
)

_heuristic = MoeMuonHHeuristic(min_lr_ratio=0.05)
_model_base = _heuristic.build_model_config(_DIM, seq_len=_SEQ)
_model = dataclasses.replace(
    _model_base,
    disable_pko=True,
    disable_long_rope=True,
    sliding_window=2048,
    use_array_stacked_blocks=True,
)

# Build the standard heuristic optimizer at the NEW batch size (for beta2,
# epsilon, momentum, etc.), then override learning_rate to our pre-computed
# post-ramp peak (0.005078, not the fresh 0.005281 -- we are already 5% into
# training) and promote to the resume subclass that owns the piecewise
# lr_scheduler.
_tokens = float(_TOTAL_STEPS * _BS_NEW * _SEQ)
_optimizer_base = _heuristic.build_muonh_config(_BS_NEW, _tokens, _DIM, seq_len=_SEQ)
_optimizer_base = dataclasses.replace(_optimizer_base, rmsnorm_to_adam=True)
_optimizer_replaced = dataclasses.replace(
    _optimizer_base,
    learning_rate=_LR_PEAK_NEW,
    adam_lr=_LR_PEAK_NEW / _ADAMH_RATIO,
)
_optimizer = GrugMoeMuonHResumeConfig(
    **dataclasses.asdict(_optimizer_replaced),
    resume_step=_RESUME_STEP,
    ramp_end_step=_RAMP_END_STEP,
    end_step=_TOTAL_STEPS,
    lr_at_resume=_LR_AT_RESUME,
    lr_at_ramp_end=_LR_AT_RAMP_END,
)

_run_id = f"moe_67b_a2b_d{_DIM}_ep{_EP}_rep{_REPLICA_AXIS}_bs{_BS_NEW}_" f"seq{_SEQ}_sw2k_v4_2048_muon_resume15k_v2_10T"
step = ExecutorStep(
    name=f"grug/{_run_id}",
    fn=run_grug_moe_trial_2x_bs,
    config=GrugMoeLaunchConfig2xBS(
        model=versioned(_model),
        data=_datakit_data_config(
            total_steps=_TOTAL_STEPS,
            batch_size=_BS_NEW,
            max_seq_len=_SEQ,
            enable_simulated_epoching=False,
            phase_1_start_step=_PHASE_1_START_STEP,
        ),
        output_path=this_output_path(),
        run_id=_run_id,
        resources=versioned(ResourceConfig.with_tpu(_SLICE, preemptible=False)),
        steps=versioned(_TOTAL_STEPS),
        batch_size=versioned(_BS_NEW),
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
                f"bs{_BS_NEW}",
                "datakit_mix",
                "disable_pko",
                "no_long_rope",
                "seq8k",
                "stacked",
                "logit_z_loss",
                "rmsadam",
                "muon",
                "resume15k",
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
        initialize_from_path=_RESUME_CKPT_PATH,
        source_batch_size=_BS_SOURCE,
        resume_step=_RESUME_STEP,
        per_device_parallelism=_PER_DEVICE_PARALLELISM,
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
                # v4-2048 = 1024 chips, replica=8, EP=1 → data=128.
                # Batch shards = 8*128*1 = 1024 → eval_batch_size must be
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
            f"Resume d={_DIM} _muon_10T from step {_RESUME_STEP} of the source "
            f"rep=16 BS=4096 run; switch to BS={_BS_NEW}, rep={_REPLICA_AXIS}, total "
            f"{_TOTAL_STEPS} steps. LR ramps {_LR_AT_RESUME:.6f} → {_LR_PEAK_NEW:.6f} over "
            f"{_RAMP_END_STEP - _RESUME_STEP} steps, then linear decay to "
            f"{_LR_PEAK_NEW * 0.05:.6f}. initialize_from_path = source step-15000; "
            f"auto-resume from this run's own output dir on iris restarts."
        ),
    )
