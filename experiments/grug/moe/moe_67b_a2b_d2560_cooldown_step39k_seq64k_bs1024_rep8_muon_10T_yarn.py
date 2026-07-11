# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""YaRN mscale sweep for the early cooldown from step 39,000.

Twin of ``moe_67b_a2b_d2560_cooldown_step39k_seq64k_bs1024_rep8_muon_10T`` --
same seq=64k / BS=1024 / 3,150-step / phase-1-only / linear-LR-decay recipe --
but sweeps the YaRN attention-temperature scale (``mscale``) instead of the
plain baseline.

Rationale: this model has ``disable_long_rope=True`` (long full-attention
layers don't rotate) and ``sliding_window=2048`` on short layers (positional
range bounded by the window regardless of seq_len). Neither branch of the
RoPE inv_freq rescale that "real" YaRN adds would fire meaningfully on this
architecture. What DOES still matter at long context is the attention
temperature -- softmax logits sharpen as context grows, so YaRN's
``mscale`` factor pumps ``qk_mult`` up to compensate::

    mscale     = mscale_coef * log(new_seq / old_seq) + 1.0
    qk_mult    = qk_mult_baseline * mscale

For new_seq=65,536 and old_seq=8,192, ``log(8) = 2.079``, giving arm values::

    coef=0.0  → mscale=1.000  → qk_mult=1.300  (baseline, no attn scaling)
    coef=0.1  → mscale=1.208  → qk_mult=1.570  (paper-suggested value)
    coef=0.2  → mscale=1.416  → qk_mult=1.841  (Peng et al. empirical)

All three arms share everything else with the baseline cooldown launcher.

**Short-run TEST mode**: this launcher trains only ``_TEST_TRAIN_STEPS`` (=20)
steps past the resume checkpoint -- just enough to confirm each YaRN-mscale
arm loads, takes forward/backward steps, and produces sane loss/eval at long
context. The LR schedule is still constructed with the full-length
``end_step`` (42,150), so the LR values over those 20 steps exactly match
what a full 3,150-step cooldown would use for its first 20 steps. The
run_id / output_path / wandb group are all suffixed with ``_test20`` to
keep these outputs separate from a later full-length submission.

Submit (us-central2, production, --no-preemptible)::

    WANDB_KEY=$(python3 -c "import os; print(os.environ['WANDB_API_KEY'])") && \\
    .venv/bin/iris --cluster=marin job run --no-wait --region us-central2 \\
        --priority production --no-preemptible -e WANDB_API_KEY "$WANDB_KEY" \\
        -- python -m experiments.grug.moe.moe_67b_a2b_d2560_cooldown_step39k_seq64k_bs1024_rep8_muon_10T_yarn

Each ExecutorStep dispatches a separate iris task on its own v4-2048 slice
(3 arms x v4-2048 each). Adjust ``_ARMS`` to run fewer if slice availability
is tight.
"""

import dataclasses
import math
from dataclasses import dataclass

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
_BS: int = 1024
_SEQ: int = 65_536  # 8x source seq_len
_ORIG_SEQ: int = 8_192  # baseline seq_len the source run trained at
_EP: int = 1
_REPLICA_AXIS: int = 8
_SLICE: str = "v4-2048"
_LOGIT_Z_LOSS_WEIGHT: float = 1e-4

_RESUME_STEP: int = 39_000
_COOLDOWN_STEPS: int = 3_150
_LR_END_STEP: int = _RESUME_STEP + _COOLDOWN_STEPS  # 42,150 — LR schedule endpoint
# Short-run test cutoff: train only 20 steps past resume, but the LR schedule
# (built with end_step=_LR_END_STEP) still walks its full 3,150-step decay —
# the trainer just stops at step 39,020 before the schedule finishes.
_TEST_TRAIN_STEPS: int = 20
_STOP_STEP: int = _RESUME_STEP + _TEST_TRAIN_STEPS  # 39,020

# LR schedule -- match the source's absolute floor.
# Source config (moe_67b_a2b_d2560_resume15k_bs8192_rep8_muon_10T.py):
#   learning_rate (peak) = 0.005281,  min_lr_ratio = 0.05
#   → floor at step 157,500 = 0.005281 * 0.05 = 0.000264
# The cooldown decays to that same absolute floor (0.000264), NOT to 5% of
# 0.004270 (which would be 0.0002135 -- too low by ~19%). So the schedule is
# built with the source's peak values, and only `lr_at_resume` /
# `lr_at_ramp_end` (both = 0.004270) determine where we JOIN the schedule.
_LR_AT_RESUME: float = 0.004270  # Muon LR at source step 39,000 (interpolated)
_LR_SOURCE_PEAK: float = 0.005281  # source `learning_rate` (never reached in cooldown)
_MIN_LR_RATIO: float = 0.05
_LR_FLOOR: float = _LR_SOURCE_PEAK * _MIN_LR_RATIO  # 0.0002641 -- same as source's floor
_ADAMH_RATIO: float = 13.0 / 3.0

_BATCH_SHARDS: int = _REPLICA_AXIS * (1024 // _REPLICA_AXIS // _EP) * _EP  # = 1024
_PER_DEVICE_PARALLELISM: int = _BS // _BATCH_SHARDS  # = 1

_RESUME_CKPT_PATH: str = (
    "gs://marin-us-central2/grug/"
    "moe_67b_a2b_d2560_ep1_rep8_bs8192_seq8192_sw2k_v4_2048_muon_resume15k_v2_10T-9fcc1f/"
    "checkpoints/step-39000/"
)


@dataclass(frozen=True)
class _Arm:
    tag: str
    mscale_coef: float


# Baseline (coef=0) is included as a control so all three arms share the
# same output_path hashing pattern; omit it if it would just re-run the
# non-yarn cooldown baseline that already exists.
_ARMS: tuple[_Arm, ...] = (
    _Arm("yarn_mscale00", 0.0),
    _Arm("yarn_mscale01", 0.1),
    _Arm("yarn_mscale02", 0.2),
)


# Data: datakit phase-1 mixture only, from step 0. Shared across arms.
_data_train = LmDataConfig(
    tokenizer=marin_tokenizer,
    cache_dir=None,
    components=_datakit_components(),
    train_weights=[(0, _phase_weights(1))],
    auto_build_caches=False,
    mixture_block_size=_MIXTURE_BLOCK_SIZE,
)
_data = add_validation_sets_to_mixture(_data_train, default_validation_sets(tokenizer=marin_tokenizer))

_heuristic = MoeMuonHHeuristic(min_lr_ratio=_MIN_LR_RATIO)
_model_base = _heuristic.build_model_config(_DIM, seq_len=_SEQ)
_model_base = dataclasses.replace(
    _model_base,
    disable_pko=True,
    disable_long_rope=True,
    sliding_window=2048,
    use_array_stacked_blocks=True,
)
_QK_MULT_BASELINE: float = _model_base.qk_mult  # heuristic default (1.3)

# LR schedule uses the FULL cooldown horizon (_LR_END_STEP=42,150) even though
# we only train 20 steps -- this way the 20 training steps we do walk the
# exact same LR trajectory as the eventual full-length run.
_tokens = float(_LR_END_STEP * _BS * _SEQ)
_optimizer_base = _heuristic.build_muonh_config(_BS, _tokens, _DIM, seq_len=_SEQ)
_optimizer_base = dataclasses.replace(_optimizer_base, rmsnorm_to_adam=True)
_optimizer_replaced = dataclasses.replace(
    _optimizer_base,
    learning_rate=_LR_SOURCE_PEAK,  # pin to source peak so floor = source floor
    adam_lr=_LR_SOURCE_PEAK / _ADAMH_RATIO,
    min_lr_ratio=_MIN_LR_RATIO,
)
_optimizer = GrugMoeMuonHResumeConfig(
    **dataclasses.asdict(_optimizer_replaced),
    resume_step=_RESUME_STEP,
    ramp_end_step=_RESUME_STEP + 1,
    end_step=_LR_END_STEP,
    lr_at_resume=_LR_AT_RESUME,
    lr_at_ramp_end=_LR_AT_RESUME,
)


def _build_arm_step(arm: _Arm) -> ExecutorStep:
    mscale = arm.mscale_coef * math.log(_SEQ / _ORIG_SEQ) + 1.0
    model_cfg = dataclasses.replace(_model_base, qk_mult=_QK_MULT_BASELINE * mscale)
    run_id = (
        f"moe_67b_a2b_d{_DIM}_ep{_EP}_rep{_REPLICA_AXIS}_bs{_BS}_"
        f"seq{_SEQ}_sw2k_v4_2048_muon_cooldown_step39k_{arm.tag}_test{_TEST_TRAIN_STEPS}"
    )
    return ExecutorStep(
        name=f"grug/{run_id}",
        fn=run_grug_moe_trial_2x_bs,
        config=GrugMoeLaunchConfig2xBS(
            model=versioned(model_cfg),
            data=_data,
            output_path=this_output_path(),
            run_id=run_id,
            resources=versioned(ResourceConfig.with_tpu(_SLICE, preemptible=False)),
            steps=versioned(_STOP_STEP),
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
                    f"bs{_BS}",
                    f"seq{_SEQ}",
                    "cooldown",
                    "step39k",
                    "phase1_only",
                    "test20",
                    arm.tag,
                    "disable_pko",
                    "no_long_rope",
                    "stacked",
                    "logit_z_loss",
                    "rmsadam",
                    "muon",
                    "v4_2048",
                ],
                group="june-tpu-67b-a2b-cooldown-yarn-test20",
                name=None,
            ),
            optimizer=versioned(_optimizer),
            expert_parallel=_EP,
            checkpoint_keep=[{"every": 500}],
            save_interval_minutes=60,
            initialize_from_path=_RESUME_CKPT_PATH,
            source_batch_size=None,
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
                    eval_batch_size=1024,
                    steps_per_eval=500,
                    max_eval_batches=1,
                    eval_current=True,
                    eval_ema=False,
                )
            ),
        ),
    )


steps: list[ExecutorStep] = [_build_arm_step(arm) for arm in _ARMS]


if __name__ == "__main__":
    executor_main(
        steps=steps,
        description=(
            f"YaRN attn-scale sweep TEST from step {_RESUME_STEP} of "
            f"muon_resume15k_v2_10T. {len(_ARMS)} arms: {[a.tag for a in _ARMS]}. "
            f"seq={_SEQ} (8x), BS={_BS} (/8). Training STOPS after "
            f"{_TEST_TRAIN_STEPS} steps (step {_STOP_STEP}), but the LR schedule "
            f"is built with end_step={_LR_END_STEP} so the LR trajectory over "
            f"those 20 steps matches the eventual full-length "
            f"{_COOLDOWN_STEPS}-step cooldown. Datakit phase-1 (mix 2) only. "
            f"LR at step {_RESUME_STEP}: {_LR_AT_RESUME:.6f}; final (step "
            f"{_LR_END_STEP}) if run to completion: {_LR_FLOOR:.7f} "
            f"(= source peak {_LR_SOURCE_PEAK} x {_MIN_LR_RATIO} -- same "
            f"absolute floor the source targets)."
        ),
    )
