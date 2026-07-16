# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Early cooldown from step-39,000 of the ``..._muon_resume15k_v2_10T`` run.

Loads step-39,000 (mid-decay in the original doubled-BS schedule), switches
to a long-context, small-batch configuration for a 10% cooldown, and decays
the LR to 5% of its current value::

- ``seq_len``: 8,192 → 65,536 (8x longer sequences).
- ``batch_size``: 8,192 → 1,024 sequences (8x smaller, so ``tokens_per_step``
  stays at 8,192 * 8,192 = 67,108,864 tokens).
- ``total tokens``: 10% of the 2.114 T seen at step 39,000 = ~211 B tokens
  = 3,150 cooldown steps at 67.1 M tokens/step.
- ``mixture``: datakit phase-1 weights only, from step 0 -- no phase-0
  section, no phase-0 -> phase-1 transition.
- ``simulated epoching``: off.
- ``LR schedule``: start at the source LR at step 39,000 (~0.004270),
  linear decay over the 3,150 cooldown steps to the same absolute floor
  the source targets: ``0.005281 * 0.05 = 0.000264`` (source peak x
  ``min_lr_ratio``), NOT 5% of the resume-value 0.004270. No ramp / no
  warmup -- we join the schedule exactly where the source left it.

Everything else (mesh: rep=8, EP=1, GQA 4:1 attention, ``sliding_window=2048``,
``disable_pko``, ``disable_long_rope``, stacked blocks, mp policy) is
unchanged from the source resume launcher.

The mesh is ``(replica_dcn=8, data=128, expert=1, model=1)`` on v4-2048.
With BS=1,024 sequences and ``batch_shards = 8 * 128 * 1 = 1,024``, each
chip gets **1 sequence of length 64 k** per step. Full-attention "long"
layers (every 4th + last, 7 of 26) at seq=64 k are the memory tight spot;
short (sliding-window) layers stay bounded by ``sliding_window=2048``.

Submit (us-central2, production, --no-preemptible)::

    WANDB_KEY=$(python3 -c "import os; print(os.environ['WANDB_API_KEY'])") && \\
    .venv/bin/iris --cluster=marin job run --no-wait --region us-central2 \\
        --priority production --no-preemptible -e WANDB_API_KEY "$WANDB_KEY" \\
        -- python -m experiments.june_tpu_67b_a2b.moe.moe_67b_a2b_d2560_cooldown_step39k_seq64k_bs1024_rep8_muon_10T \\
           --version 2026.07.16 --run

``--version`` sets the checkpoint version (required) and ``--run`` builds it; without ``--run`` the
lowered plan is printed. Pin a calendar version for a run to keep; pass ``--version dev`` to iterate.
"""

import dataclasses
import math

from fray.cluster import ResourceConfig
from levanter.data.text.datasets import LmDataConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import experiment_main
from marin.experiment.namespacing import user_namespaced_name
from marin.training.training import LevanterCheckpoint

from experiments.datasets.paloma import paloma_datasets
from experiments.datasets.uncheatable import uncheatable_datasets
from experiments.june_tpu_67b_a2b.moe.heuristic_muonh import MoeMuonHHeuristic
from experiments.june_tpu_67b_a2b.moe.launch_2x_bs import (
    GrugMoeLaunchConfig2xBS,
    GrugMoeMuonHResumeConfig,
    run_grug_moe_trial_2x_bs,
)
from experiments.june_tpu_67b_a2b.moe.launch_datakit_moe_mix import (
    _MIXTURE_BLOCK_SIZE,
    _datakit_components,
    _phase_weights,
    _validation_component,
)
from experiments.june_tpu_67b_a2b.moe.train import GrugEvalConfig, GrugTrainerConfig
from experiments.marin_tokenizer import marin_tokenizer

_DIM: int = 2560
_BS: int = 1024  # 1024 sequences * 65,536 tokens = 67,108,864 tokens/step (same as source)
_SEQ: int = 65_536  # 8x source seq_len
_EP: int = 1
_REPLICA_AXIS: int = 8
_SLICE: str = "v4-2048"
_LOGIT_Z_LOSS_WEIGHT: float = 1e-4

# Cooldown schedule
_RESUME_STEP: int = 39_000
_COOLDOWN_STEPS: int = 3_150  # 10% of 2.114 T tokens at 67.1 M tokens/step
_TOTAL_STEPS: int = _RESUME_STEP + _COOLDOWN_STEPS  # 42,150

# LR at step 39,000 on the source's piecewise schedule:
#   ramp_end (step 15,100)     = 0.005078
#   floor    (step 157,500)    = 0.005281 * 0.05  = 0.000264
#   at 39,000: linear interp within [15,100, 157,500]
#     frac = (39,000 - 15,100) / (157,500 - 15,100) = 0.16784
#     LR   = 0.005078 - frac * (0.005078 - 0.000264)  = 0.004270
_LR_AT_RESUME: float = 0.004270
# Cooldown decays to the same ABSOLUTE floor the source targets: 0.005281 * 0.05
# = 0.000264, NOT to 5% of 0.004270 (which would be 0.000214 -- too low by ~19%).
# The `learning_rate` field in the resume schedule is the abstract "peak" that
# `min_lr_ratio` is applied to; pinning it to the source peak preserves the
# floor. Only `lr_at_resume` / `lr_at_ramp_end` (= 0.004270) determine where
# we JOIN the schedule.
_LR_SOURCE_PEAK: float = 0.005281  # source `learning_rate` (never reached in cooldown)
_MIN_LR_RATIO: float = 0.05
_LR_FLOOR: float = _LR_SOURCE_PEAK * _MIN_LR_RATIO  # 0.0002641 (= source floor)
_ADAMH_RATIO: float = 13.0 / 3.0

# Batch-shard math -- mesh (8, 128, 1, 1) on v4-2048 → batch_shards = 1024,
# so per_device_parallelism = 1 sequence per chip at BS=1024.
_BATCH_SHARDS: int = _REPLICA_AXIS * (1024 // _REPLICA_AXIS // _EP) * _EP  # = 1024
_PER_DEVICE_PARALLELISM: int = _BS // _BATCH_SHARDS  # = 1

# Source checkpoint: the resume-v2 run's own step-39,000. Loaded once on
# first launch via initialize_from_path; iris preemption / crash restarts
# then auto-resume from THIS run's own output dir.
_RESUME_CKPT_PATH: str = (
    "gs://marin-us-central2/grug/"
    "moe_67b_a2b_d2560_ep1_rep8_bs8192_seq8192_sw2k_v4_2048_muon_resume15k_v2_10T-9fcc1f/"
    "checkpoints/step-39000/"
)

_VALIDATION = {
    **{f"paloma/{name}": step for name, step in paloma_datasets(tokenizer=marin_tokenizer).items()},
    **{f"uncheatable_eval/{name}": step for name, step in uncheatable_datasets(tokenizer=marin_tokenizer).items()},
}

# YaRN attention temperature scale, applied across ALL layers (not just long):
# mscale = 0.1 * log(65536/8192) + 1.0 = 1.2079  ->  qk_mult = 1.3 * mscale = 1.5703.
# The all-layers coef=0.1 arm was tied for the best final Paloma macro loss in
# the 20-step probe (see issue #6811); we're committing to that setting for the
# full-length cooldown.
_YARN_MSCALE_COEF: float = 0.1
_YARN_MSCALE: float = _YARN_MSCALE_COEF * math.log(_SEQ / 8_192) + 1.0
_QK_MULT_BASELINE: float = 1.3
_QK_MULT: float = _QK_MULT_BASELINE * _YARN_MSCALE  # 1.5703

_heuristic = MoeMuonHHeuristic(min_lr_ratio=_MIN_LR_RATIO)
_model_base = _heuristic.build_model_config(_DIM, seq_len=_SEQ)
_model = dataclasses.replace(
    _model_base,
    disable_pko=True,
    disable_long_rope=True,
    sliding_window=2048,
    use_array_stacked_blocks=True,
    qk_mult=_QK_MULT,
)

# Optimizer: reuse the resume subclass with a no-op ramp -- we're joining
# the schedule at exactly the source's step-39,000 LR, so lr_at_resume ==
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

_run_id = f"moe_67b_a2b_d{_DIM}_ep{_EP}_rep{_REPLICA_AXIS}_bs{_BS}_" f"seq{_SEQ}_sw2k_v4_2048_muon_cooldown_step39k"


def build(*, version: str | None = None) -> ArtifactStep[LevanterCheckpoint]:
    """Build the vendored June TPU 67B cooldown run."""
    name = f"grug/{_run_id}"
    version = resolve_version(name, version)

    def build_config(ctx: StepContext) -> GrugMoeLaunchConfig2xBS:
        if ctx.is_fingerprint:
            validation_components = {
                name: _validation_component(ctx.artifact_path(dep)) for name, dep in _VALIDATION.items()
            }
        else:
            validation_components = {name: ctx.resolved(dep).as_component() for name, dep in _VALIDATION.items()}

        validation_weights = {name: 0.0 for name in validation_components}
        data = LmDataConfig(
            tokenizer=marin_tokenizer,
            cache_dir=None,
            components={**_datakit_components(), **validation_components},
            train_weights=[(0, {**_phase_weights(1), **validation_weights})],
            auto_build_caches=False,
            mixture_block_size=_MIXTURE_BLOCK_SIZE,
        )

        return GrugMoeLaunchConfig2xBS(
            model=_model,
            data=data,
            output_path=ctx.output_path,
            run_id=_run_id,
            resources=ctx.runtime_arg("train_resources"),
            steps=_TOTAL_STEPS,
            batch_size=_BS,
            seed=0,
            mp="params=float32,compute=bfloat16,output=bfloat16",
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
            optimizer=_optimizer,
            expert_parallel=_EP,
            checkpoint_keep=[{"every": 500}],
            save_interval_minutes=60,
            initialize_from_path=_RESUME_CKPT_PATH,
            source_batch_size=None,
            resume_step=_RESUME_STEP,
            per_device_parallelism=_PER_DEVICE_PARALLELISM,
            grug_trainer=GrugTrainerConfig(
                z_loss_weight=_LOGIT_Z_LOSS_WEIGHT,
                ema_beta=None,
                log_every=1,
                replica_axis_size=_REPLICA_AXIS,
            ),
            eval=GrugEvalConfig(
                # rep=8 * data=128 * expert=1 = 1024 batch shards, so
                # eval_batch_size must be divisible by 1024.
                eval_batch_size=1024,
                steps_per_eval=500,
                max_eval_batches=1,
                eval_current=True,
                eval_ema=False,
            ),
        )

    return ArtifactStep(
        name=user_namespaced_name(name, version),
        version=version,
        artifact_type=LevanterCheckpoint,
        run=run_grug_moe_trial_2x_bs,
        build_config=build_config,
        deps=tuple(_VALIDATION.values()),
        runtime_args={"train_resources": ResourceConfig.with_tpu(_SLICE, preemptible=False)},
    )


if __name__ == "__main__":
    experiment_main(build)()
