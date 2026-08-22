# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Context extension probe from step-156,000 of the ``..._cooldown_step141k`` run.

Extends context 4x (65,536 -> 262,144) with context parallelism, running a
short LR decay over 1,000 steps at BS=256.

- ``seq_len``: 65,536 -> 262,144 (4x longer).
- ``batch_size``: 1,024 -> 256 (so tokens/step stays at 256 * 262,144 =
  67,108,864 = matches the step-141k cooldown's 67.1 M tokens/step).
- ``total tokens``: 1,000 steps * 67.1 M = 67.1 B tokens.
- ``mesh``: ``(replica_dcn=8, data=32, context=4, expert=1, model=1)`` on
  v4-2048 (1024 chips). ``batch_shards = 8 * 32 * 1 = 256`` so
  ``per_device_parallelism = 1``; ``context_axis_size=4`` shards the seq axis
  across 4 chips, giving ``seq_per_device = 65,536`` -- identical to the
  step-141k cooldown's chip-local seq length, which is our known-good memory
  point.
- ``LR schedule``: linear decay from 5% of the source peak (0.005281 * 0.05
  = 0.000264, the value at the end of the step-141k cooldown) to 1% of the
  source peak (0.005281 * 0.01 = 0.0000528) over the 1,000 cooldown steps.
  No ramp -- join the schedule at the source's step-156,000 LR.
  Implementation: pin the resume schedule's abstract "peak" to 0.000264
  (the actual start LR) with ``min_lr_ratio=0.2`` so the floor equals
  ``0.000264 * 0.2 = 0.0000528``.
- ``qk_mult``: 1.75 = 1.3 * mscale(coef=0.1) = 1.3 * (0.1 * log(32) + 1) =
  1.7506, rounded. Matches the "1.3 * (0.1 * ln(s) + 1)" convention used at
  s=8 (=1.57) for the 65k cooldown.
- ``data``: datakit phase-1 (mix 2) only, with exact per-token continuation
  from the step-141k cooldown's finish position. Main run consumed
  905,969,664,000 tokens of mix-2, cooldown consumed 1,006,632,960,000
  more (15,000 * 1024 * 65,536), for a total of 1,912,602,624,000 tokens.
  At seq=262,144, that is 7,296,000 mixture samples -- an exact integer
  (both source consumption counts divide by 262,144 cleanly). The
  launcher's ``resume_data_offset`` builds a 3-stage BS schedule
  ``[(0, 46), (36000, 47), (156000, 256)]`` that yields exactly
  7,296,000 cumulative samples at step 156,000 -- zero re-consumption AND
  zero skip.

Data budget: after this run, cumulative mix-2 consumption reaches
1,979,711,488,000 tokens (7,552,000 samples * 262,144). Mix-2's design budget
is ~2.00 T tokens (20% of the 10 T horizon), so we land at 98.9% of budget
after this run.

Everything else (rep-8 layout intent, EP=1, GQA 4:1 attention, sliding_window
2,048, disable_pko, disable_long_rope, stacked blocks, mp policy) unchanged
from the step-141k cooldown. The new axis is ``context=4`` for CP; K/V still
get all-gathered inside the splash attention kernel, only Q is seq-sharded
(the "all-gather-KV" flavor).

Submit (us-central2, production, --no-preemptible)::

    WANDB_KEY=$(python3 -c "import os; print(os.environ['WANDB_API_KEY'])") && \\
    .venv/bin/iris --cluster=marin job run --no-wait --region us-central2 \\
        --priority production --no-preemptible --max-retries 100 -e WANDB_API_KEY "$WANDB_KEY" \\
        -- python -m experiments.grug.moe.moe_67b_a2b_d2560_ctxext_step156k_seq262k_bs256_ctx4_muon
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
_BS: int = 256  # 256 sequences * 262,144 tokens = 67,108,864 tokens/step
_SEQ: int = 262_144  # 4x the step-141k cooldown's 65,536
_EP: int = 1
# replica=1 (not 8 like the step-141k cooldown): CP absorbs 4x the devices
# that used to be batch shards, so pushing all remaining devices into "data"
# maximises FSDP shard count. This gives data=256 -> 8x more FSDP than the
# naive replica=8 config, and 2x more than the step-141k cooldown's 128-way
# FSDP -- necessary to keep params + optim state per chip under HBM budget
# once the batch shrinks from 1024 to 256 sequences.
_REPLICA_AXIS: int = 1
_CONTEXT_AXIS: int = 4  # seq shards; seq_per_device = 262144 / 4 = 65,536
_SLICE: str = "v4-2048"
_LOGIT_Z_LOSS_WEIGHT: float = 1e-4

# 1,000-step run: 156,000 -> 157,000.
_RESUME_STEP: int = 156_000
_STAGE_STEPS: int = 1_000  # 67.1 B tokens
_TOTAL_STEPS: int = _RESUME_STEP + _STAGE_STEPS  # 157,000

_CHECKPOINT_EVERY: int = 250  # 4 permanent checkpoints (156250, 156500, 156750, 157000)
_STEPS_PER_EVAL: int = 250

# LR: 5% -> 1% of source peak (linear decay, no ramp).
# The step-141k cooldown ended at LR = 0.000264 = 5% of source peak
# (0.005281). We continue linearly to 1% = 0.0000528 over the 1,000 steps.
# The resume schedule expresses "decay to floor" as
#   floor = learning_rate * min_lr_ratio
# so pin the abstract "peak" to the actual start LR and set min_lr_ratio=0.2
# to land at 20% of that -- which equals 1% of the source peak.
_LR_AT_RESUME: float = 0.000264
_LR_END: float = 0.0000528  # = 0.000264 * 0.2 = 0.005281 * 0.01
_MIN_LR_RATIO: float = 0.2  # so peak * 0.2 == _LR_END with peak = _LR_AT_RESUME
_ADAMH_RATIO: float = 13.0 / 3.0

# Batch-shard math -- mesh (replica=1, data=256, context=4, expert=1, model=1)
# on v4-2048 -> batch_shards = replica * data * expert = 1 * 256 * 1 = 256, so
# per_device_parallelism = 1 sequence per chip at BS=256. Context axis splits
# the seq into 4 shards.
_BATCH_SHARDS: int = _REPLICA_AXIS * (1024 // _REPLICA_AXIS // _CONTEXT_AXIS // _EP) * _EP  # = 256
_PER_DEVICE_PARALLELISM: int = _BS // _BATCH_SHARDS  # = 1

# Source checkpoint: the step-141k cooldown's step-156,000. Loaded once on
# first launch via initialize_from_path; iris preemption / crash restarts
# then auto-resume from THIS run's own output dir.
_RESUME_CKPT_PATH: str = (
    "gs://marin-us-central2/grug/"
    "moe_67b_a2b_d2560_ep1_rep8_bs1024_seq65536_sw2k_v4_2048_muon_cooldown_step141k-a30ef8/"
    "checkpoints/step-156000/"
)

# Data: datakit phase-1 (mix 2) weights from step 0. Exact continuation
# from the step-141k cooldown's finish position (see docstring above for
# the sample-count derivation).
_data_train = LmDataConfig(
    tokenizer=marin_tokenizer,
    cache_dir=None,
    components=_datakit_components(),
    train_weights=[(0, _phase_weights(1))],
    auto_build_caches=False,
    mixture_block_size=_MIXTURE_BLOCK_SIZE,
)
_data = add_validation_sets_to_mixture(_data_train, default_validation_sets(tokenizer=marin_tokenizer))

# EXACT per-token continuation from cooldown's mix-2 finish position.
# main:      13,500 * 8,192 * 8,192   = 905,969,664,000 tokens
# cooldown:  15,000 * 1,024 * 65,536 = 1,006,632,960,000 tokens
# total:                              1,912,602,624,000 tokens
# at seq=262,144:                     7,296,000 samples (exact integer)
_MAIN_RUN_MIX2_TOKENS: int = 13_500 * 8_192 * 8_192  # 905,969,664,000
_COOLDOWN_MIX2_TOKENS: int = 15_000 * 1_024 * 65_536  # 1,006,632,960,000
_PRIOR_MIX2_TOKENS: int = _MAIN_RUN_MIX2_TOKENS + _COOLDOWN_MIX2_TOKENS  # 1,912,602,624,000
_RESUME_DATA_OFFSET: int = _PRIOR_MIX2_TOKENS // _SEQ  # 7,296,000

# YaRN attention temperature scale, applied Q-only across ALL layers.
# 1.75 = 1.3 * mscale(coef=0.1) = 1.3 * (0.1 * log(32) + 1) = 1.75055
# (rounded). Same "1.3 * (0.1 * ln(s) + 1)" convention that gave 1.57 at s=8.
_QK_MULT: float = 1.75

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

# Optimizer: reuse the resume subclass with a no-op ramp; the schedule joins
# the source's step-156,000 LR (0.000264) and linearly decays to 0.0000528.
_tokens = float(_TOTAL_STEPS * _BS * _SEQ)  # only used by the LR-formula sanity path
_optimizer_base = _heuristic.build_muonh_config(_BS, _tokens, _DIM, seq_len=_SEQ)
_optimizer_base = dataclasses.replace(_optimizer_base, rmsnorm_to_adam=True)
_optimizer_replaced = dataclasses.replace(
    _optimizer_base,
    learning_rate=_LR_AT_RESUME,  # abstract "peak" pinned so floor = peak * min_lr_ratio = _LR_END
    adam_lr=_LR_AT_RESUME / _ADAMH_RATIO,
    min_lr_ratio=_MIN_LR_RATIO,
)
_optimizer = GrugMoeMuonHResumeConfig(
    **dataclasses.asdict(_optimizer_replaced),
    resume_step=_RESUME_STEP,
    ramp_end_step=_RESUME_STEP + 1,  # no-op ramp
    end_step=_TOTAL_STEPS,
    lr_at_resume=_LR_AT_RESUME,
    lr_at_ramp_end=_LR_AT_RESUME,  # same -> no ramp
)

_run_id = f"moe_67b_a2b_d{_DIM}_ep{_EP}_rep{_REPLICA_AXIS}_ctx{_CONTEXT_AXIS}_bs{_BS}_seq{_SEQ}_ctxext_step156k"
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
                "yarn_mscale01",
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
        # 4 permanent saves across the 1,000-step run at steps 156,250 /
        # 156,500 / 156,750 / 157,000. Cadence matches steps_per_eval so
        # eval + checkpoint line up.
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
                # rep=8 * data=32 * expert=1 = 256 batch shards; eval_batch_size
                # must be divisible by 256.
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
            f"Context extension probe from step {_RESUME_STEP} of the step-141k cooldown. "
            f"seq={_SEQ} (4x), BS={_BS} (/4), context={_CONTEXT_AXIS}, {_STAGE_STEPS} steps "
            f"(~{_STAGE_STEPS * _BS * _SEQ / 1e9:.1f} B tokens). Datakit phase-1 mixture only, "
            f"exact continuation from cooldown mix-2 finish position via resume_data_offset. "
            f"LR linear decay {_LR_AT_RESUME:.6f} -> {_LR_END:.7f} over the stage, no ramp. "
            f"qk_mult={_QK_MULT} (= 1.3 * (0.1 * ln(32) + 1))."
        ),
    )
