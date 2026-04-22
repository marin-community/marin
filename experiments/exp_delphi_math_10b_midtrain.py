# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Midtraining sweep: continue AdamH pretrain checkpoints on Nemotron-CC-Math v1.

Continues training from two of the smallest existing AdamH-trained Marin
checkpoints on 10 B tokens of ``nemotron_cc_math_v1/4plus``, sweeping the
peak LR factor around 2/3 of each base's own pretrain peak. Both bases share
the same optimizer family (AdamH + Complete(d)P); see Will Held's blog at
https://oa.williamheld.com/blog/delphi/ and ``experiments.scaling_law_sweeps
.completed_adamh``.

This file enumerates ``len(BASES) * len(LR_FACTORS) = 6`` :class:`ExecutorStep`
runs. **Each sweep point must be launched as its own top-level ``iris job
run`` coordinator** — do NOT put all 6 under a single coordinator, because
Marin's ``run_levanter_train_lm`` submits its iris child with the hardcoded
name ``train_lm`` (``lib/marin/src/marin/training/training.py:307``) and
concurrent same-name submits collapse onto one handle via the iris
``EXISTING_JOB_POLICY_KEEP`` policy (``lib/iris/src/iris/cluster/controller
/service.py:1113``). Symptom on v10: 1 of 6 actually trained; the other 5
marked SUCCESS with empty artifacts.

To launch a single sweep point, set the env vars below so this script
builds only the matching step:

  iris --cluster=marin job run --cpu 1 --memory 3GB --disk 9GB \\
    --region us-central1 --job-name delphi-math-10b-1e21-lr0.67 --no-wait \\
    -e MARIN_I_WILL_PAY_FOR_ALL_FEES 1 -e WANDB_API_KEY "$WANDB_API_KEY" \\
    -e MIDTRAIN_SELECT_BASE 1e21-v5 -e MIDTRAIN_SELECT_LR 0.67 \\
    -- python experiments/exp_delphi_math_10b_midtrain.py

With no env vars set, all 6 steps are generated (useful for dry-runs /
introspection; do NOT actually ``executor_main`` on the full list).

See ``.agents/logbooks/midtraining_delphi.md`` for the full rationale,
numbers, and verification plan.
"""

import os

from levanter.optim import AdamHConfig

from experiments.defaults import default_train
from experiments.midtraining_data_buckets import BUCKET_2
from experiments.scaling_law_sweeps.completed_adamh import completed_adamh_heuristic
from experiments.simple_train_config import SimpleTrainConfig
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, executor_main

# ----------------------------------------------------------------------------
# Fixed knobs (both bases)
# ----------------------------------------------------------------------------

SEQ_LEN: int = 4096
BATCH_SIZE: int = 512
TOKEN_BUDGET: int = 10_000_000_000
# ceil(TOKEN_BUDGET / (BATCH_SIZE * SEQ_LEN)); 4768 * 512 * 4096 ≈ 1.00e10.
NUM_TRAIN_STEPS: int = 4768
WARMUP_STEPS: int = 500
DECAY_STEPS: int = NUM_TRAIN_STEPS - WARMUP_STEPS  # 4268
MIN_LR_RATIO: float = 0.1
# v5p-64 (32 chips) is the smallest slice that fits this config without
# gradient checkpointing per marin.scaling_laws.tpu_utils.pick_v5p_type.
# v5p pool is us-central1-a + us-east5-a (see lib/iris/examples/marin.yaml).
TPU_TYPE: str = "v5p-64"

STEPS_PER_EVAL: int = 200
STEPS_PER_EXPORT: int = 1000
STEPS_PER_HF_EXPORT: int = 1000

# Heuristic-derived constants shared across the suite.
BETA1: float = 0.9
MAX_GRAD_NORM: float = 0.1


# ----------------------------------------------------------------------------
# Base-model slots
# ----------------------------------------------------------------------------
# peak_lr / peak_adam_lr / beta2 / epsilon are read verbatim from each run's
# wandb config (not recomputed via the heuristic formula — the config is the
# source of truth for what the weights were optimized against).

BASES: dict[str, dict] = {
    # ~1.9 B AdamH isoflop scan point at 3e20 FLOPs (compute-optimal).
    # Stands in for "1e20" — no optimal-training run exists at 1e20 FLOPs
    # in the AdamH scaling ladder, only the sweep points up to 3e20.
    # Pre-copied from us-central2 into us-central1 so Marin's same-region
    # training-config guard (rigging.filesystem.check_gcs_paths_same_region)
    # passes. Job is pinned to --region us-central1 accordingly.
    "1e20-iso-d2048-L21": dict(
        ckpt=(
            "gs://marin-us-central1/checkpoints/isoflop/isoflop-3e+20-d2048-L21-B128-adamh_scaling_v5/checkpoints/step-46915/"
        ),
        hidden_dim=2048,
        peak_lr=4.483e-3,
        peak_adam_lr=7.382e-5,
        beta2=0.99980,
        epsilon=4.11e-8,
    ),
    # Canonical Delphi 1e21 v5 (~3.4 B). Seed replicates v5-seed42,
    # v5-seed62746, and v6 converge within 0.001 c4-en-loss of this run.
    "1e21-v5": dict(
        ckpt=("gs://marin-us-central1/adamh-scaling-ladder-nemotron-optimal-1e+21-v5-019021/checkpoints/step-21979/"),
        hidden_dim=2560,
        peak_lr=7.425e-3,
        peak_adam_lr=4.314e-4,
        beta2=0.99920,
        epsilon=2.81e-8,
    ),
}

LR_FACTORS: tuple[float, ...] = (0.5, 0.67, 0.83)


# ----------------------------------------------------------------------------
# Data: 100% nemotron_cc_math_v1/4plus via BUCKET_2's ExecutorStep. The
# raw HF download already landed in `gs://marin-us-east5/raw/nemotron_cc_math_v1-322fe4/`
# (46 parquet shards, ~62 GB), but normalize and tokenize have not yet been
# run. When this job lands on a us-east5 coordinator (see --region flags),
# the executor walks the dep chain:
#   download (skipped — raw already local)
#   → normalize (runs fresh, writes to us-east5)
#   → tokenize (runs fresh, writes to us-east5)
#   → training
# All stages read/write in us-east5; no cross-region egress for the data.
MATH_TRAIN_STEP = BUCKET_2["nemotron_cc_math_v1/4plus"]


# ----------------------------------------------------------------------------


def _build_adamh(base: dict, lr_factor: float) -> AdamHConfig:
    return AdamHConfig(
        learning_rate=base["peak_lr"] * lr_factor,
        adam_lr=base["peak_adam_lr"] * lr_factor,
        beta1=BETA1,
        beta2=base["beta2"],
        epsilon=base["epsilon"],
        max_grad_norm=MAX_GRAD_NORM,
        # int → absolute step count (see exp898_deeper_cooldown.py).
        warmup=WARMUP_STEPS,
        decay=DECAY_STEPS,
        min_lr_ratio=MIN_LR_RATIO,
        lr_schedule="linear",
        nesterov=False,
    )


# Env-var filters: set these to restrict the generated sweep to a single
# point so each can be launched as its own iris coordinator job. Step hashes
# are unchanged by filtering — already-succeeded outputs (e.g. the v10
# `lr0.5-ba7b7f` run) stay cached and will be skipped automatically.
_SELECT_BASE = os.environ.get("MIDTRAIN_SELECT_BASE")  # e.g. "1e21-v5"
_SELECT_LR = os.environ.get("MIDTRAIN_SELECT_LR")  # e.g. "0.67"


def _lr_str(lr_factor: float) -> str:
    return f"{lr_factor:.2f}".rstrip("0").rstrip(".")


def _build_runs() -> list[ExecutorStep]:
    runs: list[ExecutorStep] = []
    for base_tag, base in BASES.items():
        if _SELECT_BASE is not None and base_tag != _SELECT_BASE:
            continue
        # Reconstruct the Qwen3Config exactly as the pretrain run built it,
        # so TensorStore weight restore matches every array shape.
        # Private method is intentional: it's the single source of truth for
        # Delphi architecture and it's what the pretrain path called.
        model_config = completed_adamh_heuristic._build_model_config(
            hidden_size=base["hidden_dim"],
            seq_len=SEQ_LEN,
        )

        for lr_factor in LR_FACTORS:
            if _SELECT_LR is not None and _lr_str(lr_factor) != _SELECT_LR:
                continue
            optimizer = _build_adamh(base, lr_factor)

            train_cfg = SimpleTrainConfig(
                resources=ResourceConfig.with_tpu(TPU_TYPE),
                train_batch_size=BATCH_SIZE,
                num_train_steps=NUM_TRAIN_STEPS,
                train_seq_len=SEQ_LEN,
                # `learning_rate` is a required SimpleTrainConfig field but
                # is unused when `optimizer_config` is provided. Set it to
                # the peak we actually use so logs remain consistent.
                learning_rate=optimizer.learning_rate,
                optimizer_config=optimizer,
                initialize_from_checkpoint_path=base["ckpt"],
                # Fresh data iterator: math mix is a different distribution
                # from the pretrain mix, so pretrain step counter + data
                # cursor should be discarded.
                reset_data_loader_on_init=True,
                steps_per_eval=STEPS_PER_EVAL,
                steps_per_export=STEPS_PER_EXPORT,
                steps_per_hf_export=STEPS_PER_HF_EXPORT,
            )

            lr_str = _lr_str(lr_factor)
            name = f"delphi-{base_tag}-math-10b-lr{lr_str}"

            runs.append(
                default_train(
                    name=name,
                    tokenized=MATH_TRAIN_STEP,
                    model_config=model_config,
                    train_config=train_cfg,
                    tags=(
                        "midtraining",
                        f"base={base_tag}",
                        "nemotron-cc-math-4plus",
                        f"lr_factor={lr_factor}",
                        f"peak_lr={optimizer.learning_rate:.3e}",
                        f"adam_lr={optimizer.adam_lr:.3e}",
                        "adamh",
                        "delphi-midtrain",
                    ),
                    eval_harness_tasks=(),
                )
            )
    return runs


runs: list[ExecutorStep] = _build_runs()


if __name__ == "__main__":
    executor_main(
        steps=runs,
        description="Delphi Nemotron-CC-Math 10B midtraining: LR sweep on two AdamH-trained base checkpoints.",
    )
