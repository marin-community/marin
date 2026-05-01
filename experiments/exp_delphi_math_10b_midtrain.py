# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Midtraining sweep: continue AdamH pretrain checkpoints on Nemotron-CC-Math v1.

Continues training from existing AdamH-trained Marin
checkpoints on 10 B tokens of ``nemotron_cc_math_v1/4plus``, sweeping the
peak LR factor around 2/3 of each base's own pretrain peak. All bases share
the same optimizer family (AdamH + Complete(d)P); see Will Held's blog at
https://oa.williamheld.com/blog/delphi/ and ``experiments.scaling_law_sweeps
.completed_adamh``.

This file enumerates ``len(BASES) * len(LR_FACTORS)`` :class:`ExecutorStep`
runs. **Each sweep point must be launched as its own top-level ``iris job
run`` coordinator** — do NOT put all generated steps under a single coordinator, because
Marin's ``run_levanter_train_lm`` submits its iris child with the hardcoded
name ``train_lm`` (``lib/marin/src/marin/training/training.py:307``) and
concurrent same-name submits collapse onto one handle via the iris
``EXISTING_JOB_POLICY_KEEP`` policy (``lib/iris/src/iris/cluster/controller
/service.py:1113``). Symptom on v10: 1 of 6 actually trained; the other 5
marked SUCCESS with empty artifacts.

To launch a single sweep point, set the env vars below so this script
builds only the matching step. The coordinator can land in either
us-central1-a or us-east5-a (the two v5p zones), but once it lands the
training child is pinned to that same region. Base ckpts are wrapped in
``mirrored(...)`` so MirrorFileSystem copies the ckpt from whichever
marin-<region> bucket has it on first open.

  iris --cluster=marin job run --cpu 1 --memory 3GB --disk 9GB \\
    --region us-central1 --region us-east5 \\
    --job-name delphi-math-10b-1e21-lr0.67 --no-wait \\
    -e WANDB_API_KEY "$WANDB_API_KEY" \\
    -e MIDTRAIN_SELECT_BASE 1e21-v5 -e MIDTRAIN_SELECT_LR 0.67 \\
    -- python experiments/exp_delphi_math_10b_midtrain.py

With no env vars set, all 9 steps are generated (useful for dry-runs /
introspection; do NOT actually ``executor_main`` on the full list).

See ``.agents/logbooks/midtraining_delphi.md`` for the full rationale,
numbers, and verification plan.
"""

import logging
import os
from dataclasses import dataclass

from fray.cluster import ResourceConfig
from haliax import Axis
from levanter.main.train_lm import CheckpointInitMode
from levanter.optim import AdamHConfig
from marin.execution.executor import ExecutorStep, MirroredValue, executor_main, mirrored
from rigging.filesystem import marin_region

from experiments.defaults import default_train
from experiments.midtrain_data_safety import assert_val_train_disjoint
from experiments.midtraining_data_buckets import BUCKET_2
from experiments.midtraining_mixes import (
    FULL_HIGHQUALITY_NEMO_MATH_NAME,
    MIDTRAIN_BUDGET_FRACTION,
    PRETRAIN_33P_MATH_67P_HIGHQUALITY_NEMO_MATH_NAME,
    PRETRAIN_67P_MATH_33P_HIGHQUALITY_NEMO_MATH_NAME,
    PRETRAIN_70P_MATH_30P_HIGHQUALITY_NEMO_MATH_NAME,
    log_partition_summary,
    midtrain_token_budget,
    midtraining_mix_by_name,
)
from experiments.scaling_law_sweeps.completed_adamh import completed_adamh_heuristic
from experiments.simple_train_config import SimpleTrainConfig

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Fixed knobs (all bases)
# ----------------------------------------------------------------------------

DEFAULT_SEQ_LEN: int = 4096
REFERENCE_WARMUP_BATCH_SIZE: int = 512
# Budget resolution order (per launch):
#   1. MIDTRAIN_TOKEN_BUDGET env var → hard override applied to every base
#      (legacy single-budget path; use only when running one base at a time).
#   2. MIDTRAIN_BUDGET_FRACTION env var → K override applied via the
#      midtrain_token_budget heuristic (per-base scaling).
#   3. midtraining_mixes.MIDTRAIN_BUDGET_FRACTION → default K = 0.20 from the
#      single-source-of-truth constant.
_TOKEN_BUDGET_HARD_OVERRIDE: int | None = (
    int(os.environ["MIDTRAIN_TOKEN_BUDGET"]) if os.environ.get("MIDTRAIN_TOKEN_BUDGET") else None
)
_BUDGET_FRACTION_OVERRIDE: float = float(os.environ.get("MIDTRAIN_BUDGET_FRACTION", MIDTRAIN_BUDGET_FRACTION))
_TOKEN_BUDGET_LABEL_OVERRIDE: str | None = os.environ.get("MIDTRAIN_TOKEN_BUDGET_LABEL")
# The original 1e20/1e21 sweep uses 500 warmup steps at global batch 512, or
# about 1.05B warmup tokens. Keep that warmup-token budget stable when the
# batch size changes for larger TPU slices.
WARMUP_TOKENS: int = 500 * REFERENCE_WARMUP_BATCH_SIZE * DEFAULT_SEQ_LEN
MIN_LR_RATIO: float = 0.0
TPU_TYPE_OVERRIDE: str | None = os.environ.get("MIDTRAIN_TPU_TYPE")
OVERRIDE_BATCH_SIZE: int | None = (
    int(os.environ["MIDTRAIN_BATCH_SIZE"]) if os.environ.get("MIDTRAIN_BATCH_SIZE") else None
)
OVERRIDE_PER_DEVICE_PARALLELISM: int | None = (
    int(os.environ["MIDTRAIN_PER_DEVICE_PARALLELISM"]) if os.environ.get("MIDTRAIN_PER_DEVICE_PARALLELISM") else None
)
OVERRIDE_TENSOR_PARALLEL_SIZE: int | None = (
    int(os.environ["MIDTRAIN_TENSOR_PARALLEL_SIZE"]) if os.environ.get("MIDTRAIN_TENSOR_PARALLEL_SIZE") else None
)
LR_MULTIPLIER: float = float(os.environ.get("MIDTRAIN_LR_MULTIPLIER", "1.0"))
MIDTRAIN_TRAIN_REGION: str | None = os.environ.get("MIDTRAIN_TRAIN_REGION")
MIDTRAIN_COORDINATOR_REGIONS: tuple[str, ...] = ("us-central1", "us-east5")

STEPS_PER_EVAL: int = 200
# Permanent checkpoint cadence is derived per-base from num_train_steps so
# every run gets ~10 evenly-spaced rollback points regardless of length
# (1e20 ~9.4k steps gets ~10 ckpts; 1e21 ~4.4k steps also gets ~10). Levanter's
# rolling temp checkpoint (save_interval=10min, set in experiments/defaults.py)
# is independent and handles preemption resume.
EXPORT_FRACTION_OF_RUN: float = 0.10
MIN_STEPS_PER_EXPORT: int = 50

# Heuristic-derived constants shared across the suite.
BETA1: float = 0.9
MAX_GRAD_NORM: float = 0.1


# ----------------------------------------------------------------------------
# Base-model slots
# ----------------------------------------------------------------------------
# peak_lr / peak_adam_lr / beta2 / epsilon are read verbatim from each run's
# wandb config (not recomputed via the heuristic formula — the config is the
# source of truth for what the weights were optimized against).


@dataclass(frozen=True)
class V5PComputeConfig:
    tpu_type: str
    per_device_parallelism: int = -1
    tensor_parallel_size: int = 1


@dataclass(frozen=True)
class MidtrainingBaseConfig:
    ckpt: str | MirroredValue[str]
    hidden_dim: int
    seq_len: int
    train_batch_size: int
    default_tpu_type: str
    v5p_compute: tuple[V5PComputeConfig, ...]
    peak_lr: float
    peak_adam_lr: float
    beta2: float
    epsilon: float
    pretrain_tokens: int
    """Tokens the base model was pretrained on (steps * pretrain_BS * seq_len).

    Drives the per-base midtrain budget via ``midtrain_token_budget``: same
    rule (``midtrain_tokens = pretrain_tokens * K``) applied to every scale.
    """

    def compute_config(self, tpu_type: str) -> V5PComputeConfig:
        for compute_config in self.v5p_compute:
            if compute_config.tpu_type == tpu_type:
                return compute_config
        allowed = ", ".join(compute_config.tpu_type for compute_config in self.v5p_compute)
        raise ValueError(f"{tpu_type!r} is not an approved v5p target for this base. Allowed: {allowed}")


# Checkpoint paths are wrapped in `mirrored(...)` so the executor rewrites
# them to `mirror://<rel>`. MirrorFileSystem locates the file in whichever
# marin-<region> bucket currently has it and copies to the local prefix on
# first open; Levanter's `latest_checkpoint_path` / `load_checkpoint` stage
# the mirror:// dir down to a concrete gs:// URL before TensorStore opens it
# (see `_stage_mirror_to_local` in lib/levanter/src/levanter/checkpoint.py).
# This lets the coordinator land in either us-central1-a or us-east5-a based
# on capacity. The training child is then pinned to that coordinator region,
# so concrete gs:// output/cache/checkpoint paths and TPU compute stay aligned.
BASES: dict[str, MidtrainingBaseConfig] = {
    # ~1.9 B AdamH isoflop scan point at 3e20 FLOPs (compute-optimal).
    # Stands in for "1e20" — no optimal-training run exists at 1e20 FLOPs
    # in the AdamH scaling ladder, only the sweep points up to 3e20.
    "1e20-iso-d2048-L21": MidtrainingBaseConfig(
        ckpt=mirrored(
            "checkpoints/isoflop/isoflop-3e+20-d2048-L21-B128-adamh_scaling_v5/checkpoints/step-46915",
            budget_gb=30,
        ),
        # Match the isoflop base run: d2048/L21, seq_len 4096, global B128.
        hidden_dim=2048,
        seq_len=DEFAULT_SEQ_LEN,
        train_batch_size=128,
        default_tpu_type="v5p-32",
        v5p_compute=(
            V5PComputeConfig("v5p-32"),
            V5PComputeConfig("v5p-64"),
            V5PComputeConfig("v5p-128"),
            V5PComputeConfig("v5p-256"),
            # At v5p-512 there are more chips than batch examples, so split the
            # model axis to keep the effective data axis at 128.
            V5PComputeConfig("v5p-512", tensor_parallel_size=2),
        ),
        peak_lr=4.483e-3,
        peak_adam_lr=7.382e-5,
        beta2=0.99980,
        epsilon=4.11e-8,
        # 47,064 pretrain steps * 128 BS * 4096 seq = 24.67 B tokens
        # (the 3e20-isoflop d2048-L21 pretrain ran the full 47,064 steps).
        pretrain_tokens=47_064 * 128 * 4096,
    ),
    # Canonical Delphi 1e21 v5 (~3.4 B). Seed replicates v5-seed42,
    # v5-seed62746, and v6 converge within 0.001 c4-en-loss of this run.
    "1e21-v5": MidtrainingBaseConfig(
        ckpt=mirrored(
            "adamh-scaling-ladder-nemotron-optimal-1e+21-v5-019021/checkpoints/step-21979",
            budget_gb=50,
        ),
        # Match the Delphi base run: d2560/L26, seq_len 4096, global B512.
        hidden_dim=2560,
        seq_len=DEFAULT_SEQ_LEN,
        train_batch_size=512,
        default_tpu_type="v5p-256",
        v5p_compute=(
            V5PComputeConfig("v5p-64"),
            V5PComputeConfig("v5p-128"),
            V5PComputeConfig("v5p-256"),
            V5PComputeConfig("v5p-512"),
        ),
        peak_lr=7.425e-3,
        peak_adam_lr=4.314e-4,
        beta2=0.99920,
        epsilon=2.81e-8,
        # 22,057 pretrain steps * 512 BS * 4096 seq = 46.27 B tokens
        # (Delphi v5 pretrain plan; checkpoint at step-21979 is 78 short of plan).
        pretrain_tokens=22_057 * 512 * 4096,
    ),
    # Canonical Delphi 1e22 v5 (~9.7 B). This base was trained at global
    # batch 1024 on v4-512 and finished at step 38206. v5p-512 ran cleanly at
    # B1024; smaller v5p slices keep B1024 with gradient accumulation by fixing
    # per-device parallelism at 4.
    "1e22-v5": MidtrainingBaseConfig(
        ckpt=mirrored(
            "adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e/checkpoints/step-38206",
            budget_gb=150,
        ),
        hidden_dim=3840,
        seq_len=DEFAULT_SEQ_LEN,
        train_batch_size=1024,
        default_tpu_type="v5p-512",
        v5p_compute=(
            V5PComputeConfig("v5p-128", per_device_parallelism=4),
            V5PComputeConfig("v5p-256", per_device_parallelism=4),
            V5PComputeConfig("v5p-512", per_device_parallelism=4),
        ),
        peak_lr=7.231797280729413e-3,
        peak_adam_lr=3.276222099351447e-4,
        beta2=0.9984011994401821,
        epsilon=3.70426657045089e-8,
        # 38,235 pretrain steps * 1024 BS * 4096 seq = 160.37 B tokens
        # (Delphi v5 pretrain plan; checkpoint at step-38206).
        pretrain_tokens=38_235 * 1024 * 4096,
    ),
}

# 10B/20B prior sweeps showed monotone lr0.5 < lr0.67 < lr0.83 on eval/loss
# for both mixes — optimum at or below 0.5. Shift the grid down to bracket
# the new minimum. See logbook §"2026-05-01 20:30 UTC — new sweep plan".
LR_FACTORS: tuple[float, ...] = (0.33, 0.5, 0.67)


# ----------------------------------------------------------------------------
# Data: by default, 100% nemotron_cc_math_v1/4plus via BUCKET_2's ExecutorStep.
# The raw HF download originally landed in us-east5, but this experiment must
# not bake concrete `gs://marin-<region>/...` strings into the run identity.
# Keep region selection at launch/materialization time so the child training
# job can be pinned to the coordinator's region.
#
# Set MIDTRAIN_MIX_NAME to a key from experiments.midtraining_mixes for new
# mixture runs, e.g. `70p_30m_highquality_nemo_math`. Leaving it unset preserves
# the original single-step data path and output namespace for currently-running
# 100% math jobs.
MATH_TRAIN_STEP = BUCKET_2["nemotron_cc_math_v1/4plus"]
_MIDTRAIN_MIX_NAME = os.environ.get("MIDTRAIN_MIX_NAME")
_MATH_TOKENIZED_OUTPUT_PATH_OVERRIDE = os.environ.get("MIDTRAIN_MATH_TOKENIZED_OUTPUT_PATH_OVERRIDE")
if _MATH_TOKENIZED_OUTPUT_PATH_OVERRIDE:
    if _MIDTRAIN_MIX_NAME:
        raise ValueError("MIDTRAIN_MATH_TOKENIZED_OUTPUT_PATH_OVERRIDE is only supported for the single math dataset")
    MATH_TRAIN_STEP = MATH_TRAIN_STEP.with_output_path(_MATH_TOKENIZED_OUTPUT_PATH_OVERRIDE)
_TOKENIZED_TRAIN_DATA = midtraining_mix_by_name(_MIDTRAIN_MIX_NAME) if _MIDTRAIN_MIX_NAME else MATH_TRAIN_STEP

_BASE_OUTPUT_TAGS = {
    "1e20-iso-d2048-L21": "1e20",
    "1e21-v5": "1e21",
    "1e22-v5": "1e22",
}

_MIX_OUTPUT_TAGS = {
    None: "math",
    FULL_HIGHQUALITY_NEMO_MATH_NAME: "math",
    PRETRAIN_70P_MATH_30P_HIGHQUALITY_NEMO_MATH_NAME: "p70m30",
    PRETRAIN_67P_MATH_33P_HIGHQUALITY_NEMO_MATH_NAME: "p67m33",
    PRETRAIN_33P_MATH_67P_HIGHQUALITY_NEMO_MATH_NAME: "p33m67",
}


# ----------------------------------------------------------------------------


def _token_budget_for_base(base: MidtrainingBaseConfig) -> int:
    """Resolve the midtrain token budget for one base.

    Hard-override (``MIDTRAIN_TOKEN_BUDGET``) takes precedence and applies the
    same absolute budget to every base. Otherwise the heuristic
    ``midtrain_token_budget(pretrain_tokens=base.pretrain_tokens, fraction=K)``
    derives a per-base budget so each scale gets a budget proportional to its
    own pretrain. K defaults to ``MIDTRAIN_BUDGET_FRACTION`` and can be
    overridden per-launch via the ``MIDTRAIN_BUDGET_FRACTION`` env var.
    """
    if _TOKEN_BUDGET_HARD_OVERRIDE is not None:
        return _TOKEN_BUDGET_HARD_OVERRIDE
    return midtrain_token_budget(pretrain_tokens=base.pretrain_tokens, fraction=_BUDGET_FRACTION_OVERRIDE)


def _budget_label(token_budget: int) -> str:
    """Human-readable budget label embedded in run names ("4p93b", "10b", ...)."""
    if _TOKEN_BUDGET_LABEL_OVERRIDE is not None:
        return _TOKEN_BUDGET_LABEL_OVERRIDE
    if token_budget % 1_000_000_000 == 0:
        return f"{token_budget // 1_000_000_000}b"
    return f"{token_budget / 1_000_000_000:.2f}b".replace(".", "p")


def _num_train_steps(token_budget: int, batch_size: int, seq_len: int) -> int:
    return max(1, round(token_budget / (batch_size * seq_len)))


def _steps_per_export(num_train_steps: int) -> int:
    """Permanent checkpoint cadence: ``EXPORT_FRACTION_OF_RUN`` of the run.

    Every base gets ~10 evenly-spaced permanent checkpoints regardless of total
    length, so 1e21 (4,411 steps) is as resilient to mid-run failure as 1e20
    (9,413 steps). HF exports use the same cadence (Levanter resolves
    ``steps_per_hf_export=None`` to ``steps_per_export``).
    """
    return max(MIN_STEPS_PER_EXPORT, int(num_train_steps * EXPORT_FRACTION_OF_RUN))


def _warmup_steps(batch_size: int, seq_len: int, num_train_steps: int) -> int:
    warmup_steps = max(1, round(WARMUP_TOKENS / (batch_size * seq_len)))
    return min(warmup_steps, num_train_steps - 1)


def _build_adamh(base: MidtrainingBaseConfig, lr_factor: float, warmup_steps: int, decay_steps: int) -> AdamHConfig:
    return AdamHConfig(
        learning_rate=base.peak_lr * lr_factor * LR_MULTIPLIER,
        adam_lr=base.peak_adam_lr * lr_factor * LR_MULTIPLIER,
        beta1=BETA1,
        beta2=base.beta2,
        epsilon=base.epsilon,
        max_grad_norm=MAX_GRAD_NORM,
        # int → absolute step count (see exp898_deeper_cooldown.py).
        warmup=warmup_steps,
        decay=decay_steps,
        min_lr_ratio=MIN_LR_RATIO,
        lr_schedule="linear",
        nesterov=False,
    )


def _selected_compute_config(base: MidtrainingBaseConfig) -> V5PComputeConfig:
    tpu_type = TPU_TYPE_OVERRIDE or base.default_tpu_type
    compute_config = base.compute_config(tpu_type)
    per_device_parallelism = (
        OVERRIDE_PER_DEVICE_PARALLELISM
        if OVERRIDE_PER_DEVICE_PARALLELISM is not None
        else compute_config.per_device_parallelism
    )
    tensor_parallel_size = (
        OVERRIDE_TENSOR_PARALLEL_SIZE
        if OVERRIDE_TENSOR_PARALLEL_SIZE is not None
        else compute_config.tensor_parallel_size
    )
    return V5PComputeConfig(
        tpu_type=compute_config.tpu_type,
        per_device_parallelism=per_device_parallelism,
        tensor_parallel_size=tensor_parallel_size,
    )


def _normalize_region(region_or_zone: str) -> str:
    region_or_zone = region_or_zone.lower()
    parts = region_or_zone.split("-")
    if len(parts) >= 3 and len(parts[-1]) == 1 and parts[-1].isalpha() and any(char.isdigit() for char in parts[-2]):
        return "-".join(parts[:-1])
    return region_or_zone


def _selected_train_region() -> str | None:
    explicit_region = MIDTRAIN_TRAIN_REGION is not None
    region = MIDTRAIN_TRAIN_REGION or marin_region()
    if region is None:
        return None

    region = _normalize_region(region)
    if region in MIDTRAIN_COORDINATOR_REGIONS:
        return region
    if explicit_region:
        allowed = ", ".join(MIDTRAIN_COORDINATOR_REGIONS)
        raise ValueError(f"Delphi midtraining must run in one of {{{allowed}}}, got {region!r}")
    return None


def _midtrain_tpu_resources(tpu_type: str) -> ResourceConfig:
    region = _selected_train_region()
    if region is None:
        return ResourceConfig.with_tpu(tpu_type)
    return ResourceConfig.with_tpu(tpu_type, regions=[region])


# Env-var filters: set these to restrict the generated sweep to a single
# point so each can be launched as its own iris coordinator job. Step hashes
# are unchanged by filtering — already-succeeded outputs (e.g. the v10
# `lr0.5-ba7b7f` run) stay cached and will be skipped automatically.
_SELECT_BASE = os.environ.get("MIDTRAIN_SELECT_BASE")  # e.g. "1e21-v5"
_SELECT_LR = os.environ.get("MIDTRAIN_SELECT_LR")  # e.g. "0.67"
_RUN_NAME_SUFFIX = os.environ.get("MIDTRAIN_RUN_NAME_SUFFIX")
_INIT_CKPT_PATH = os.environ.get("MIDTRAIN_INIT_CKPT_PATH")
_OUTPUT_PATH_OVERRIDE = os.environ.get("MIDTRAIN_OUTPUT_PATH_OVERRIDE")

if _OUTPUT_PATH_OVERRIDE and (_SELECT_BASE is None or _SELECT_LR is None):
    raise ValueError("MIDTRAIN_OUTPUT_PATH_OVERRIDE requires MIDTRAIN_SELECT_BASE and MIDTRAIN_SELECT_LR")


def _lr_str(lr_factor: float) -> str:
    return f"{lr_factor:.2f}".rstrip("0").rstrip(".")


def _base_output_tag(base_tag: str) -> str:
    if base_tag not in _BASE_OUTPUT_TAGS:
        raise ValueError(f"Missing short output tag for base {base_tag!r}")
    return _BASE_OUTPUT_TAGS[base_tag]


def _mix_output_tag(mix_name: str | None) -> str:
    if mix_name not in _MIX_OUTPUT_TAGS:
        raise ValueError(f"Missing short output tag for midtraining mix {mix_name!r}")
    return _MIX_OUTPUT_TAGS[mix_name]


def _build_run_name(base_tag: str, lr_factor: float, token_budget: int) -> str:
    lr_str = _lr_str(lr_factor)
    budget_label = _budget_label(token_budget)
    name = f"delphi-{_base_output_tag(base_tag)}-{_mix_output_tag(_MIDTRAIN_MIX_NAME)}-{budget_label}-lr{lr_str}"
    if _RUN_NAME_SUFFIX:
        name = f"{name}-{_RUN_NAME_SUFFIX}"
    if len(name) > 64:
        raise ValueError(f"Midtraining run name must stay within W&B's 64-char limit, got {len(name)}: {name}")
    return name


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
            hidden_size=base.hidden_dim,
            seq_len=base.seq_len,
        )

        token_budget = _token_budget_for_base(base)
        for lr_factor in LR_FACTORS:
            if _SELECT_LR is not None and _lr_str(lr_factor) != _SELECT_LR:
                continue
            compute_config = _selected_compute_config(base)
            batch_size = OVERRIDE_BATCH_SIZE or base.train_batch_size
            num_train_steps = _num_train_steps(token_budget, batch_size, base.seq_len)
            warmup_steps = _warmup_steps(batch_size, base.seq_len, num_train_steps)
            decay_steps = num_train_steps - warmup_steps
            optimizer = _build_adamh(base, lr_factor, warmup_steps, decay_steps)

            train_cfg = SimpleTrainConfig(
                resources=_midtrain_tpu_resources(compute_config.tpu_type),
                train_batch_size=batch_size,
                num_train_steps=num_train_steps,
                train_seq_len=base.seq_len,
                per_device_parallelism=compute_config.per_device_parallelism,
                tensor_parallel_size=compute_config.tensor_parallel_size,
                # `learning_rate` is a required SimpleTrainConfig field but
                # is unused when `optimizer_config` is provided. Set it to
                # the peak we actually use so logs remain consistent.
                learning_rate=optimizer.learning_rate,
                optimizer_config=optimizer,
                initialize_from_checkpoint_path=_INIT_CKPT_PATH or base.ckpt,
                # Fresh data iterator: math mix is a different distribution
                # from the pretrain mix, so pretrain step counter + data
                # cursor should be discarded.
                reset_data_loader_on_init=True,
                # MODEL_ONLY: drop pretrain opt_state so the fresh warmup/decay
                # schedule starts from count=0. The default FULL_STATE would
                # restore inject_hyperparams counts from the pretrain, clamping
                # the schedule to min_lr for the entire midtrain run.
                # See .agents/logbooks/midtraining_delphi.md.
                checkpoint_init_mode=CheckpointInitMode.MODEL_ONLY,
                steps_per_eval=STEPS_PER_EVAL,
                steps_per_export=_steps_per_export(num_train_steps),
                # HF export cadence: None → matches steps_per_export.
                steps_per_hf_export=None,
            )

            name = _build_run_name(base_tag, lr_factor, token_budget)

            data_tags = (
                (f"midtraining_mix={_MIDTRAIN_MIX_NAME}", "midtraining-mix")
                if _MIDTRAIN_MIX_NAME
                else ("nemotron-cc-math-4plus",)
            )

            runs.append(
                default_train(
                    name=name,
                    tokenized=_TOKENIZED_TRAIN_DATA,
                    model_config=model_config,
                    train_config=train_cfg,
                    tags=(
                        "midtraining",
                        f"base={base_tag}",
                        *data_tags,
                        f"lr_factor={lr_factor}",
                        f"batch_size={batch_size}",
                        f"seq_len={base.seq_len}",
                        f"tpu_type={compute_config.tpu_type}",
                        f"per_device_parallelism={compute_config.per_device_parallelism}",
                        f"tensor_parallel_size={compute_config.tensor_parallel_size}",
                        f"pretrain_tokens={base.pretrain_tokens}",
                        f"token_budget={token_budget}",
                        f"budget_fraction={token_budget / base.pretrain_tokens:.4f}",
                        f"num_train_steps={num_train_steps}",
                        f"peak_lr={optimizer.learning_rate:.3e}",
                        f"adam_lr={optimizer.adam_lr:.3e}",
                        "adamh",
                        "delphi-midtrain",
                    ),
                    eval_harness_tasks=(),
                    override_output_path=_OUTPUT_PATH_OVERRIDE,
                )
            )
    run_names = [run.name for run in runs]
    if len(run_names) != len(set(run_names)):
        raise ValueError(f"Generated duplicate midtraining run names: {run_names}")
    return runs


def _run_pre_flight_safety_checks() -> None:
    """Pre-flight: log val partition + verify val/train disjointness against
    the live cache. Always runs at sweep launch (under ``__main__``).

    Reads from GCS, so don't invoke at module import time. A failure means a
    real safety violation — do not proceed with the launch.
    """
    if not _MIDTRAIN_MIX_NAME:
        # Legacy single-step path bypasses LmDataConfig (no val carve-out
        # exists). Strongly recommend the LmDataConfig path going forward.
        logger.warning(
            "MIDTRAIN_MIX_NAME unset; skipping val/train disjointness check. "
            "Recommended: set MIDTRAIN_MIX_NAME=full_highquality_nemo_math (or a "
            "replay variant) so the run goes through the safety-asserted path."
        )
        return
    cfg = midtraining_mix_by_name(_MIDTRAIN_MIX_NAME)
    pos = Axis("position", DEFAULT_SEQ_LEN)
    log_partition_summary(cfg, pos)
    assert_val_train_disjoint(cfg, pos)


runs: list[ExecutorStep] = _build_runs()


if __name__ == "__main__":
    _run_pre_flight_safety_checks()
    executor_main(
        steps=runs,
        description="Delphi Nemotron-CC-Math 10B midtraining: LR sweep on AdamH-trained base checkpoints.",
    )
