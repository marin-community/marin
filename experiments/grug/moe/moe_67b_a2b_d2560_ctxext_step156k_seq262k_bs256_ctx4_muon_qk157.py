# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""qk_mult=1.57 twin of the 262k context extension probe.

Identical to ``moe_67b_a2b_d2560_ctxext_step156k_seq262k_bs256_ctx4_muon``
in every dimension (mesh, data continuation, LR schedule, checkpoint /
eval cadence, everything) EXCEPT for the qk_mult value: this run keeps
qk_mult at the prior step-141k cooldown's 1.57 rather than scaling it
up to 1.75 via the ``1.3 * (0.1 * ln(s) + 1)`` formula at s=32.

Purpose: A/B on whether the theoretical YaRN mscale for the 4x stretch
(1.75) actually beats "hold qk_mult flat at the prior value" (1.57).
Since qk_mult is applied Q-only as a single logit multiplier (not the
paper's two-sided sqrt(1/t) on both RoPE cos and sin), the "1.3 * mscale"
recipe is already an undercorrection relative to canonical YaRN; keeping
it at 1.57 vs bumping to 1.75 is a small delta on top of that. The
comparison tells us whether the fine-tune actually adapts to the new
temperature or prefers the old one.

Same run_id family, but with a ``_qk157`` suffix so the wandb / output
paths are distinct from the qk_mult=1.75 twin.

Submit (us-central2, production, --no-preemptible)::

    WANDB_KEY=$(python3 -c "import os; print(os.environ['WANDB_API_KEY'])") && \\
    .venv/bin/iris --cluster=marin job run --no-wait --region us-central2 \\
        --priority production --no-preemptible --max-retries 100 -e WANDB_API_KEY "$WANDB_KEY" \\
        -- python -m experiments.grug.moe.moe_67b_a2b_d2560_ctxext_step156k_seq262k_bs256_ctx4_muon_qk157
"""

import dataclasses

from fray.cluster import ResourceConfig
from levanter.data.text import LmDataConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned
from marin.processing.tokenize import add_validation_sets_to_mixture

from experiments.defaults import default_validation_sets
from experiments.grug.moe.heuristic_muonh import MoeMuonHHeuristic
from experiments.grug.moe.launch_2x_bs import (
    GrugMoeLaunchConfig2xBS,
    GrugMoeMuonHResumeConfig,
    run_grug_moe_trial_2x_bs,
)
from experiments.grug.moe.launch_datakit_moe_mix import (
    _MIXTURE_BLOCK_SIZE,
    _datakit_components,
    _phase_weights,
)
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig
from experiments.marin_models import marin_tokenizer

_DIM: int = 2560
_BS: int = 256
_SEQ: int = 262_144
_EP: int = 1
_REPLICA_AXIS: int = 1
_CONTEXT_AXIS: int = 4
_SLICE: str = "v4-2048"
_LOGIT_Z_LOSS_WEIGHT: float = 1e-4

_RESUME_STEP: int = 156_000
_STAGE_STEPS: int = 1_000
_TOTAL_STEPS: int = _RESUME_STEP + _STAGE_STEPS  # 157,000

_CHECKPOINT_EVERY: int = 250
_STEPS_PER_EVAL: int = 250

# LR: 5% -> 1% of source peak, no ramp. Same schedule as the qk_mult=1.75
# twin (see moe_67b_a2b_d2560_ctxext_step156k_seq262k_bs256_ctx4_muon.py
# for the derivation).
_LR_AT_RESUME: float = 0.000264
_LR_END: float = 0.0000528
_MIN_LR_RATIO: float = 0.2
_ADAMH_RATIO: float = 13.0 / 3.0

_BATCH_SHARDS: int = _REPLICA_AXIS * (1024 // _REPLICA_AXIS // _CONTEXT_AXIS // _EP) * _EP  # = 256
_PER_DEVICE_PARALLELISM: int = _BS // _BATCH_SHARDS  # = 1

_RESUME_CKPT_PATH: str = (
    "gs://marin-us-central2/grug/"
    "moe_67b_a2b_d2560_ep1_rep8_bs1024_seq65536_sw2k_v4_2048_muon_cooldown_step141k-a30ef8/"
    "checkpoints/step-156000/"
)

_data_train = LmDataConfig(
    tokenizer=marin_tokenizer,
    cache_dir=None,
    components=_datakit_components(),
    train_weights=[(0, _phase_weights(1))],
    auto_build_caches=False,
    mixture_block_size=_MIXTURE_BLOCK_SIZE,
)
_data = add_validation_sets_to_mixture(_data_train, default_validation_sets(tokenizer=marin_tokenizer))

_MAIN_RUN_MIX2_TOKENS: int = 13_500 * 8_192 * 8_192
_COOLDOWN_MIX2_TOKENS: int = 15_000 * 1_024 * 65_536
_PRIOR_MIX2_TOKENS: int = _MAIN_RUN_MIX2_TOKENS + _COOLDOWN_MIX2_TOKENS
_RESUME_DATA_OFFSET: int = _PRIOR_MIX2_TOKENS // _SEQ  # 7,296,000

# qk_mult = 1.57: HELD at the step-141k cooldown's value (derived at s=8
# from 1.3 * (0.1 * ln(8) + 1)) rather than scaled to s=32 (=1.75). This
# tests whether the fine-tune prefers the prior temperature or the
# formula-scaled one for the 4x context stretch.
_QK_MULT: float = 1.57

_heuristic = MoeMuonHHeuristic(min_lr_ratio=_MIN_LR_RATIO)
_model_base = _heuristic.build_model_config(_DIM, seq_len=_SEQ)
_model = dataclasses.replace(
    _model_base,
    disable_pko=True,
    disable_long_rope=True,
    sliding_window=2048,
    use_array_stacked_blocks=True,
    qk_mult=_QK_MULT,
    hybrid_attention_flops_accounting=True,
)

_tokens = float(_TOTAL_STEPS * _BS * _SEQ)
_optimizer_base = _heuristic.build_muonh_config(_BS, _tokens, _DIM, seq_len=_SEQ)
_optimizer_base = dataclasses.replace(_optimizer_base, rmsnorm_to_adam=True)
_optimizer_replaced = dataclasses.replace(
    _optimizer_base,
    learning_rate=_LR_AT_RESUME,
    adam_lr=_LR_AT_RESUME / _ADAMH_RATIO,
    min_lr_ratio=_MIN_LR_RATIO,
)
_optimizer = GrugMoeMuonHResumeConfig(
    **dataclasses.asdict(_optimizer_replaced),
    resume_step=_RESUME_STEP,
    ramp_end_step=_RESUME_STEP + 1,
    end_step=_TOTAL_STEPS,
    lr_at_resume=_LR_AT_RESUME,
    lr_at_ramp_end=_LR_AT_RESUME,
)

_run_id = f"moe_67b_a2b_d{_DIM}_ep{_EP}_rep{_REPLICA_AXIS}_ctx{_CONTEXT_AXIS}_bs{_BS}_seq{_SEQ}_ctxext_step156k_qk157"
step = ExecutorStep(
    name=f"grug/{_run_id}",
    fn=run_grug_moe_trial_2x_bs,
    config=GrugMoeLaunchConfig2xBS(
        model=versioned(_model),
        data=_data,
        output_path=this_output_path(),
        run_id=_run_id,
        resources=versioned(ResourceConfig.with_tpu(_SLICE, preemptible=False)),
        steps=versioned(_TOTAL_STEPS),
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
                f"ctx{_CONTEXT_AXIS}",
                f"bs{_BS}",
                f"seq{_SEQ}",
                "context_extension",
                "step156k",
                "phase1_only",
                "qk157",
                "context_parallel",
                "disable_pko",
                "no_long_rope",
                "stacked",
                "logit_z_loss",
                "rmsadam",
                "muon",
                "v4_2048",
            ],
            group="june-tpu-67b-a2b-context-ext",
            name=None,
        ),
        optimizer=versioned(_optimizer),
        expert_parallel=_EP,
        checkpoint_keep=[{"every": _CHECKPOINT_EVERY}],
        save_interval_minutes=60,
        initialize_from_path=_RESUME_CKPT_PATH,
        resume_data_offset=_RESUME_DATA_OFFSET,
        resume_step=_RESUME_STEP,
        per_device_parallelism=_PER_DEVICE_PARALLELISM,
        grug_trainer=versioned(
            GrugTrainerConfig(
                z_loss_weight=_LOGIT_Z_LOSS_WEIGHT,
                ema_beta=None,
                log_every=1,
                replica_axis_size=_REPLICA_AXIS,
                context_axis_size=_CONTEXT_AXIS,
            )
        ),
        eval=versioned(
            GrugEvalConfig(
                eval_batch_size=256,
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
            f"qk_mult=1.57 twin of the 262k context extension probe. Identical to the "
            f"qk_mult=1.75 run in every other dimension. Resume step {_RESUME_STEP}, "
            f"seq={_SEQ}, BS={_BS}, context={_CONTEXT_AXIS}, {_STAGE_STEPS} steps, "
            f"LR {_LR_AT_RESUME:.6f} -> {_LR_END:.7f}."
        ),
    )
