# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Final 1 T-token cooldown from step-141,000 of the ``..._muon_resume15k_v2_10T`` run.

Third cooldown in the seq=64k / BS=1024 family (after step-39k with 211 B tokens
and step-102k with 211 B tokens). This one is the intended production tail:
resume at step 141,000 (~9 T tokens seen, ~89.5% of the 157,500-step horizon)
and train ~1 T more tokens at extended context.

- ``seq_len``: 8,192 → 65,536 (8x longer sequences).
- ``batch_size``: 8,192 → 1,024 sequences (8x smaller, so
  ``tokens_per_step`` stays at 8,192 * 8,192 = 67,108,864 tokens).
- ``total tokens``: 15,000 cooldown steps at 67.1 M tokens/step = ~1.007 T
  tokens. Brings the model's lifetime tokens seen from ~9 T (step 141k) to
  ~10 T -- essentially the horizon's compute in the extended-context regime.
- ``mixture``: datakit phase-1 (mix 2) weights from step 0, with **exact**
  token-position continuation from main run via ``resume_data_offset``.
  Levanter's TokenSeqDataset uses ``offsets = indices × seq_len``, so
  cooldown mixture-sample-index N (at seq=65,536) corresponds to token
  offset N × 65,536 in each underlying bucket. Main run consumed
  13,500 × 8,192 = 110,592,000 samples of mix-2 at seq=8,192 =
  905,969,664,000 tokens. To land the cooldown at the same underlying
  token offset, we need cooldown sample-index 905,969,664,000 / 65,536 =
  13,824,000 at step 141,000. The launcher's ``resume_data_offset``
  field constructs a 3-stage BS schedule
  ``[(0, 98), (135000, 99), (141000, 1024)]`` that yields exactly
  13,824,000 cumulative samples at step 141,000 -- **zero
  re-consumption AND zero skip**.
- ``LR schedule``: start at the source LR at step 141,000 (~0.000822),
  linear decay over the 15,000 cooldown steps to the same absolute floor
  the source targets: ``0.005281 * 0.05 = 0.000264`` (source peak x
  ``min_lr_ratio``). No ramp / no warmup -- we join the schedule exactly
  where the source left it.

Everything else (mesh: rep=8, EP=1, GQA 4:1 attention, ``sliding_window=2048``,
``disable_pko``, ``disable_long_rope``, stacked blocks, mp policy,
``qk_mult=1.57`` from the step-102k cooldown result) is unchanged.

The mesh is ``(replica_dcn=8, data=128, expert=1, model=1)`` on v4-2048.
With BS=1,024 sequences and ``batch_shards = 8 * 128 * 1 = 1,024``, each
chip gets **1 sequence of length 64 k** per step. Full-attention "long"
layers (every 4th + last, 7 of 26) at seq=64 k are the memory tight spot;
short (sliding-window) layers stay bounded by ``sliding_window=2048``.

Submit (us-central2, production, --no-preemptible)::

    WANDB_KEY=$(python3 -c "import os; print(os.environ['WANDB_API_KEY'])") && \\
    .venv/bin/iris --cluster=marin job run --no-wait --region us-central2 \\
        --priority production --no-preemptible --max-retries 100 -e WANDB_API_KEY "$WANDB_KEY" \\
        -- python -m experiments.grug.moe.moe_67b_a2b_d2560_cooldown_step141k_seq64k_bs1024_rep8_muon_10T
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
_BS: int = 1024  # 1024 sequences * 65,536 tokens = 67,108,864 tokens/step (same as source)
_SEQ: int = 65_536  # 8x source seq_len
_EP: int = 1
_REPLICA_AXIS: int = 8
_SLICE: str = "v4-2048"
_LOGIT_Z_LOSS_WEIGHT: float = 1e-4

# Cooldown schedule
_RESUME_STEP: int = 141_000
_COOLDOWN_STEPS: int = 15_000  # 1.007 T tokens (~= "last 1 T at 65k context")
_TOTAL_STEPS: int = _RESUME_STEP + _COOLDOWN_STEPS  # 156,000

# LR at step 141,000 on the source's piecewise schedule:
#   ramp_end (step 15,100)     = 0.005078
#   floor    (step 157,500)    = 0.005281 * 0.05  = 0.000264
#   at 141,000: linear interp within [15,100, 157,500]
#     frac = (141,000 - 15,100) / (157,500 - 15,100) = 0.88413
#     LR   = 0.005078 - frac * (0.005078 - 0.000264)  = 0.000822
_LR_AT_RESUME: float = 0.000822
# Cooldown decays to the same ABSOLUTE floor the source targets: 0.005281 * 0.05
# = 0.000264. The `learning_rate` field in the resume schedule is the abstract
# "peak" that `min_lr_ratio` is applied to; pinning it to the source peak
# preserves the floor. Only `lr_at_resume` / `lr_at_ramp_end` (= 0.000822)
# determine where we JOIN the schedule.
_LR_SOURCE_PEAK: float = 0.005281  # source `learning_rate` (never reached in cooldown)
_MIN_LR_RATIO: float = 0.05
_LR_FLOOR: float = _LR_SOURCE_PEAK * _MIN_LR_RATIO  # 0.0002641 (= source floor)
_ADAMH_RATIO: float = 13.0 / 3.0

# Batch-shard math -- mesh (8, 128, 1, 1) on v4-2048 → batch_shards = 1024,
# so per_device_parallelism = 1 sequence per chip at BS=1024.
_BATCH_SHARDS: int = _REPLICA_AXIS * (1024 // _REPLICA_AXIS // _EP) * _EP  # = 1024
_PER_DEVICE_PARALLELISM: int = _BS // _BATCH_SHARDS  # = 1

# Source checkpoint: the resume-v2 run's own step-141,000. Loaded once on
# first launch via initialize_from_path; iris preemption / crash restarts
# then auto-resume from THIS run's own output dir.
_RESUME_CKPT_PATH: str = (
    "gs://marin-us-central2/grug/"
    "moe_67b_a2b_d2560_ep1_rep8_bs8192_seq8192_sw2k_v4_2048_muon_resume15k_v2_10T-9fcc1f/"
    "checkpoints/step-141000/"
)

# Data: datakit phase-1 (mix 2) weights from step 0. Clean continuation
# from main run's mix-2 position at step 141,000 is achieved via
# _SOURCE_BS below (see the source_batch_size parameter on the launcher
# config). Under the hood, TokenSeqDataset.get_batch uses
# ``offsets = indices * seq_len``, so mixture-sample-index N in the
# cooldown (at seq=65,536) corresponds to token offset N × 65,536 in the
# underlying bucket. Main run consumed 13,500 × 8,192 = 110.592 M samples
# of mix-2 at seq=8,192 = 906 B tokens. To land the cooldown at that same
# 906 B token position: cooldown mixture-sample-index at start = 906 B /
# 65,536 = 13,824,000. We achieve this with source_batch_size × resume_step
# = 13,824,000, i.e. source_batch_size ≈ 98 (0.04% short of exact -- would
# need a non-integer BS to hit exactly, and the tolerance is negligible).
_data_train = LmDataConfig(
    tokenizer=marin_tokenizer,
    cache_dir=None,
    components=_datakit_components(),
    train_weights=[(0, _phase_weights(1))],
    auto_build_caches=False,
    mixture_block_size=_MIXTURE_BLOCK_SIZE,
)
_data = add_validation_sets_to_mixture(_data_train, default_validation_sets(tokenizer=marin_tokenizer))

# EXACT continuation from main run's mix-2 finish position.
# Main run consumed 13,500 × 8,192 = 110,592,000 samples of mix-2 at
# seq=8,192, i.e. 905,969,664,000 tokens. Cooldown at seq=65,536 wants
# to start at the same underlying token offset, which is sample-index
# 905,969,664,000 / 65,536 = 13,824,000. The launcher's new
# ``resume_data_offset`` field constructs a 3-stage BS schedule that
# yields exactly that cumulative sample count at resume_step, so
# continuation is exact (zero re-consumption AND zero skip).
_MAIN_RUN_MIX2_TOKENS: int = 13_500 * 8_192 * 8_192  # 905,969,664,000
_RESUME_DATA_OFFSET: int = _MAIN_RUN_MIX2_TOKENS // _SEQ  # 13,824,000

# YaRN attention temperature scale, applied across ALL layers. Rounded to
# exactly 1.57 (from 1.3 * mscale(coef=0.1) = 1.3 * (0.1 * log(8) + 1.0) =
# 1.5703). This matches the step-39k and step-102k cooldowns and the 20-step
# YaRN probe's tied-best-Paloma arm (see issue #6811).
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
    # Honest FLOPs: 19 of 26 layers run sw=2048, so at seq=65k the naive
    # all-full-attention count overstates work by ~2.28x. Matches the
    # step-39k cooldown's MFU accounting.
    hybrid_attention_flops_accounting=True,
)

# Optimizer: reuse the resume subclass with a no-op ramp -- we're joining
# the schedule at exactly the source's step-141,000 LR, so lr_at_resume ==
# lr_at_ramp_end == learning_rate. Only the decay segment does work.
_tokens = float(_TOTAL_STEPS * _BS * _SEQ)  # only used by the LR-formula sanity path
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
    ramp_end_step=_RESUME_STEP + 1,  # no-op ramp, one-step transition to satisfy optax
    end_step=_TOTAL_STEPS,
    lr_at_resume=_LR_AT_RESUME,
    lr_at_ramp_end=_LR_AT_RESUME,  # same → no ramp
)

_run_id = f"moe_67b_a2b_d{_DIM}_ep{_EP}_rep{_REPLICA_AXIS}_bs{_BS}_" f"seq{_SEQ}_sw2k_v4_2048_muon_cooldown_step141k"
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
                f"bs{_BS}",
                f"seq{_SEQ}",
                "cooldown",
                "step141k",
                "final_1T",
                "phase1_only",
                "yarn_mscale01",
                "disable_pko",
                "no_long_rope",
                "stacked",
                "logit_z_loss",
                "rmsadam",
                "muon",
                "v4_2048",
            ],
            group="june-tpu-67b-a2b-cooldown",
            name=None,
        ),
        optimizer=versioned(_optimizer),
        expert_parallel=_EP,
        # Save every 1,500 steps -- 10 permanent saves across the 15k-step
        # final 1 T; matches steps_per_eval below so eval + checkpoint
        # cadences line up.
        checkpoint_keep=[{"every": 1_500}],
        save_interval_minutes=60,
        initialize_from_path=_RESUME_CKPT_PATH,
        # resume_data_offset: exact-target extension of source_batch_size.
        # See the data-config comment above. The launcher builds a 3-stage
        # BatchSchedule that lands the DataLoader at exactly
        # _RESUME_DATA_OFFSET cumulative samples at step _RESUME_STEP,
        # matching main run's mix-2 finish position to the sample.
        resume_data_offset=_RESUME_DATA_OFFSET,
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
                # rep=8 * data=128 * expert=1 = 1024 batch shards, so
                # eval_batch_size must be divisible by 1024.
                eval_batch_size=1024,
                steps_per_eval=1_500,
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
            f"Final 1 T-token cooldown from step {_RESUME_STEP} of the "
            f"muon_resume15k_v2_10T run. seq={_SEQ} (8x), BS={_BS} (/8), "
            f"{_COOLDOWN_STEPS} steps (~{_COOLDOWN_STEPS * _BS * _SEQ / 1e12:.3f} T "
            f"tokens ~= last 1 T of the 10 T horizon). Datakit phase-1 "
            f"mixture only. LR linear decay {_LR_AT_RESUME:.6f} -> {_LR_FLOOR:.7f} "
            f"over the cooldown, no ramp. qk_mult={_QK_MULT} (matches step-102k "
            f"cooldown result)."
        ),
    )
