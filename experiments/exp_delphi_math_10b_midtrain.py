# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Midtraining sweep: continue AdamH pretrain checkpoints on Nemotron-CC-Math v1.

Continues training from two of the smallest existing AdamH-trained Marin
checkpoints on 10 B tokens of ``nemotron_cc_math_v1/4plus``, sweeping the
peak LR factor around 2/3 of each base's own pretrain peak. Both bases share
the same optimizer family (AdamH + Complete(d)P); see Will Held's blog at
https://oa.williamheld.com/blog/delphi/ and ``experiments.scaling_law_sweeps
.completed_adamh``.

This file produces ``len(BASES) * len(LR_FACTORS) = 6`` :class:`ExecutorStep`
runs. See ``.agents/logbooks/midtraining_delphi.md`` for the full rationale,
numbers, and verification plan.
"""

from levanter.data.text import DatasetComponent, LMMixtureDatasetConfig, TextLmDatasetFormat
from levanter.optim import AdamHConfig

from experiments.defaults import default_train
from experiments.llama import llama3_tokenizer
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
    # mirror:// lets TPUs in any region read this ckpt by copying from
    # whichever marin-* bucket has it (us-central2) into the local prefix.
    "1e20-iso-d2048-L21": dict(
        ckpt=("mirror://checkpoints/isoflop/" "isoflop-3e+20-d2048-L21-B128-adamh_scaling_v5/checkpoints/step-46915/"),
        hidden_dim=2048,
        peak_lr=4.483e-3,
        peak_adam_lr=7.382e-5,
        beta2=0.99980,
        epsilon=4.11e-8,
    ),
    # Canonical Delphi 1e21 v5 (~3.4 B). Seed replicates v5-seed42,
    # v5-seed62746, and v6 converge within 0.001 c4-en-loss of this run.
    "1e21-v5": dict(
        ckpt=("mirror://" "adamh-scaling-ladder-nemotron-optimal-1e+21-v5-019021/checkpoints/step-21979/"),
        hidden_dim=2560,
        peak_lr=7.425e-3,
        peak_adam_lr=4.314e-4,
        beta2=0.99920,
        epsilon=2.81e-8,
    ),
}

LR_FACTORS: tuple[float, ...] = (0.5, 0.67, 0.83)


# ----------------------------------------------------------------------------
# Data mix: 100% nemotron_cc_math_v1/4plus. Point at the already-tokenized
# cache in us-central2 via mirror:// so the training workers can pull from
# any Marin bucket without re-running the raw + normalize + tokenize chain
# in their own region (that chain would redownload multi-TB of Nemotron-CC-
# Math-v1 from HF Hub — observed on the first launch attempt).
MATH_CACHE_MIRROR_PATH: str = "mirror://tokenized/nemotron_cc_math_v1/4plus-0bd79d"

math_mix: LMMixtureDatasetConfig = LMMixtureDatasetConfig(
    components={
        "nemotron_cc_math_v1/4plus": DatasetComponent(
            cache_dir=MATH_CACHE_MIRROR_PATH,
            format=TextLmDatasetFormat(),
        ),
    },
    train_weights={"nemotron_cc_math_v1/4plus": 1.0},
    tokenizer=llama3_tokenizer,
)


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


def _build_runs() -> list[ExecutorStep]:
    runs: list[ExecutorStep] = []
    for base_tag, base in BASES.items():
        # Reconstruct the Qwen3Config exactly as the pretrain run built it,
        # so TensorStore weight restore matches every array shape.
        # Private method is intentional: it's the single source of truth for
        # Delphi architecture and it's what the pretrain path called.
        model_config = completed_adamh_heuristic._build_model_config(
            hidden_size=base["hidden_dim"],
            seq_len=SEQ_LEN,
        )

        for lr_factor in LR_FACTORS:
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

            lr_str = f"{lr_factor:.2f}".rstrip("0").rstrip(".")
            name = f"delphi-{base_tag}-math-10b-lr{lr_str}"

            runs.append(
                default_train(
                    name=name,
                    tokenized=math_mix,
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
