# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""YaRN mscale sweep -- long-attention layers ONLY.

Twin of ``..._muon_10T_yarn`` but the mscale multiplier is applied ONLY on
the every-4th-and-last long-attention layers, not on the short
sliding-window layers. This uses the new ``qk_mult_long_scale`` field on
``GrugModelConfig`` (defaults to 1.0 → no-op for existing code paths).

Rationale: the temperature-sharpening problem YaRN's mscale fixes is a
long-context issue -- softmax entropy over N keys grows like log(N), which
scaling ``q`` by ``mscale = coef * log(N/N0) + 1.0`` counteracts. But short
layers here use ``sliding_window=2048``, so their attention population is
capped at 2,048 keys regardless of ``seq_len``. Applying mscale to the
short layers pumps the temperature on a problem those layers don't have.
Restricting mscale to the long branch keeps the correction where it's
physically motivated.

Arms::

    coef=0.0  → long-branch mscale=1.000  (control -- validates new plumbing)
    coef=0.1  → long-branch mscale=1.208
    coef=0.2  → long-branch mscale=1.416

``qk_mult`` stays at its 1.3 baseline everywhere; only long layers see
``q *= 1.3 * mscale``, short layers see ``q *= 1.3 * 1.0``.

Short-run TEST mode: trains only 20 steps past resume; LR trajectory over
those steps matches what a full 3,150-step cooldown would use for its
first 20 steps (built with end_step=42,150).

Submit (us-central2, production, --no-preemptible)::

    WANDB_KEY=$(python3 -c "import os; print(os.environ['WANDB_API_KEY'])") && \\
    .venv/bin/iris --cluster=marin job run --no-wait --region us-central2 \\
        --priority production --no-preemptible -e WANDB_API_KEY "$WANDB_KEY" \\
        -- python -m experiments.grug.moe.moe_67b_a2b_d2560_cooldown_step39k_seq64k_bs1024_rep8_muon_10T_yarn_longonly
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
_SEQ: int = 65_536
_ORIG_SEQ: int = 8_192
_EP: int = 1
_REPLICA_AXIS: int = 8
_SLICE: str = "v4-2048"
_LOGIT_Z_LOSS_WEIGHT: float = 1e-4

_RESUME_STEP: int = 39_000
_COOLDOWN_STEPS: int = 3_150
_LR_END_STEP: int = _RESUME_STEP + _COOLDOWN_STEPS  # 42,150
_TEST_TRAIN_STEPS: int = 20
_STOP_STEP: int = _RESUME_STEP + _TEST_TRAIN_STEPS  # 39,020

_LR_AT_RESUME: float = 0.004270
_LR_SOURCE_PEAK: float = 0.005281
_MIN_LR_RATIO: float = 0.05
_LR_FLOOR: float = _LR_SOURCE_PEAK * _MIN_LR_RATIO  # 0.0002641
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


_ARMS: tuple[_Arm, ...] = (
    _Arm("yarn_long_mscale00", 0.0),
    _Arm("yarn_long_mscale01", 0.1),
    _Arm("yarn_long_mscale02", 0.2),
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

_heuristic = MoeMuonHHeuristic(min_lr_ratio=_MIN_LR_RATIO)
_model_base = _heuristic.build_model_config(_DIM, seq_len=_SEQ)
_model_base = dataclasses.replace(
    _model_base,
    disable_pko=True,
    disable_long_rope=True,
    sliding_window=2048,
    use_array_stacked_blocks=True,
)

_tokens = float(_LR_END_STEP * _BS * _SEQ)
_optimizer_base = _heuristic.build_muonh_config(_BS, _tokens, _DIM, seq_len=_SEQ)
_optimizer_base = dataclasses.replace(_optimizer_base, rmsnorm_to_adam=True)
_optimizer_replaced = dataclasses.replace(
    _optimizer_base,
    learning_rate=_LR_SOURCE_PEAK,
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
    # Long-branch ONLY: qk_mult unchanged, qk_mult_long_scale=mscale.
    # Short layers still see q *= 1.3, exactly as in the source run.
    model_cfg = dataclasses.replace(_model_base, qk_mult_long_scale=mscale)
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
                    "yarn_long_only",
                    arm.tag,
                    "disable_pko",
                    "no_long_rope",
                    "stacked",
                    "logit_z_loss",
                    "rmsadam",
                    "muon",
                    "v4_2048",
                ],
                group="june-tpu-67b-a2b-cooldown-yarn-longonly-test20",
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
            f"YaRN attn-scale sweep (LONG LAYERS ONLY) TEST from step "
            f"{_RESUME_STEP} of muon_resume15k_v2_10T. {len(_ARMS)} arms: "
            f"{[a.tag for a in _ARMS]}. mscale applied via "
            f"qk_mult_long_scale, so short (sliding-window) layers see "
            f"unchanged q *= 1.3. seq={_SEQ} (8x), BS={_BS} (/8). Training "
            f"STOPS after {_TEST_TRAIN_STEPS} steps (step {_STOP_STEP}); "
            f"LR trajectory those 20 steps matches the full-length "
            f"{_COOLDOWN_STEPS}-step cooldown (end_step={_LR_END_STEP}). "
            f"Datakit phase-1 (mix 2) only. LR at step {_RESUME_STEP}: "
            f"{_LR_AT_RESUME:.6f}; floor if run to completion: "
            f"{_LR_FLOOR:.7f} (= source floor)."
        ),
    )
