# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Bolinas DNA enhancer-curation comparison.

Train a small ~0.6B gLM on each of two enhancer training datasets that share
the same 20-mammal subset but differ in curation strategy:

- ``seg_v20`` (segmentation): top-1% bins from a per-genome enhancer
  segmentation model, exon-masked.
- ``proj_v30`` (projection): ENCODE cCRE conserved enhancers projected from
  hg38 onto each genome via mmseqs2.

Evaluations:

- TraitGym Mendelian v2 (255 bp) via the lm_eval harness during training.
- LL gap = LL(functional) - LL(nonfunctional) on the v30 enhancer validation
  set, computed post-hoc from W&B as
  ``eval/val_v30_nonfunctional/loss - eval/val_v30_functional/loss``.
  Functional vs nonfunctional positions are encoded as uppercase / lowercase
  in the validation dataset (phyloP-split); see bolinas-dna#8 / #10.

Environment variables:
    SWEEP_DATASETS   CSV of dataset names to run (default: all in
                     ``TRAIN_DATASETS``). Useful for ``seg_v20`` or
                     ``proj_v30`` in isolation.
    WARMUP_MODE      'yes'/'no' (default 'no'). When 'yes', each run is
                     truncated to ``WARMUP_NUM_TRAIN_STEPS`` with
                     ``WARMUP_EVALS_PER_RUN`` evals so the LR schedule and
                     eval cadence (both derived from this) compress together.

https://github.com/Open-Athena/bolinas-dna/issues/136
"""

import dataclasses
import logging
import os
from datetime import timedelta
from functools import lru_cache

import jmp
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import DNALmDatasetFormat
from levanter.data.text.datasets import LmDataConfig
from levanter.eval_harness import LmEvalHarnessConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig

from experiments.defaults import default_tokenize
from experiments.dna.defaults import dna_effective_seq_len
from experiments.evals.task_configs import TRAITGYM_MENDELIAN_V2_255, convert_to_levanter_task_config
from experiments.qwen3 import qwen3_0_6b_hd128
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.execution.remote import remote
from marin.processing.tokenize import lm_mixture_data_config
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm

# =============================================================================
# Constants
# =============================================================================

VERSION = "v0.1"
TOKENIZER = "bolinas-dna/tokenizer-char-bos"
DNA_BASE_SEQ_LEN = 255  # bp (256 - 1 for BOS)

# Two enhancer training datasets that share the same 20-mammal subset but
# differ in curation strategy.
TRAIN_DATASETS = {
    "seg_v20": "bolinas-dna/genomes-v5-genome_set-enhancer_seg_mammals_v1-intervals-v20_255_128",
    "proj_v30": "bolinas-dna/genomes-v5-genome_set-mammals_seg20-intervals-v30_255_128",
}

# Single validation dataset with phyloP-derived uppercase/lowercase encoding;
# tokenize twice (functional and nonfunctional) for the LL-gap signal.
VALIDATION_DATASET = "bolinas-dna/genomes-v5-validation-intervals-v30_255_255"

# Training masks lowercase positions to 1% loss weight (consistent across
# Bolinas DNA experiments).
TRAIN_FORMAT = DNALmDatasetFormat(lowercase_weight=0.01)

# Validation tokenization specs — only the two terms of the LL gap; we
# deliberately skip a "default" matched-to-training variant since it adds eval
# cost without informing the strategy comparison.
VAL_SPECS: tuple[tuple[str, DNALmDatasetFormat], ...] = (
    ("functional", DNALmDatasetFormat(uppercase_weight=1.0, lowercase_weight=0.0)),
    ("nonfunctional", DNALmDatasetFormat(uppercase_weight=0.0, lowercase_weight=1.0)),
)

# Architecture: ~0.6B Qwen3 (h=1024, L=28, head_dim=128) — imported from
# the canonical preset so this file picks up any upstream changes
# automatically; max_seq_len is overridden to the DNA context size below.

# Resources & batching. Region is pinned so the inferred step hash stays
# stable across parent preemption / migration — without this, the marin
# executor auto-pins child regions to the parent's current iris region
# (executor.py:630), which changes the ResourceConfig (and hence step
# output_path) when the parent migrates, breaking checkpoint resume.
BATCH_SIZE = 4096
TPU_TYPES: tuple[str, ...] = ("v5p-8",)
TPU_REGIONS: tuple[str, ...] = ("us-central1",)

# Optimizer (AdamConfig defaults; schedule shape from exp_bolinas_4b_sweep.py).
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.1
BETA1 = 0.9
BETA2 = 0.95
EPSILON = 1e-8
MAX_GRAD_NORM = 1.0
WARMUP_FRACTION = 0.1
DECAY_FRACTION = 0.2
LR_SCHEDULE = "linear"
MIN_LR_RATIO = 0.0

# Training horizon. ~41M sequences seen at BATCH_SIZE=4096; the enhancer
# datasets are small enough that this is several epochs each.
NUM_TRAIN_STEPS = 10_000

# Eval cadence and checkpoint policy.
EVALS_PER_RUN = 10
CHECKPOINTS_PER_RUN = 3
CHECKPOINT_TIME_INTERVAL = timedelta(hours=1)

# Warmup mode (WARMUP_MODE=yes): smoke-test the full pipeline end-to-end.
WARMUP_NUM_TRAIN_STEPS = 100
WARMUP_EVALS_PER_RUN = 3

WANDB_PROJECT = "marin"

_EXPECTED_VOCAB_SIZE_WARNING = f"Tokenizer {TOKENIZER!r} not found in _KNOWN_VOCAB_SIZES"
logging.getLogger("marin.processing.tokenize.data_configs").addFilter(
    lambda record: _EXPECTED_VOCAB_SIZE_WARNING not in record.getMessage()
)


# =============================================================================
# Environment overrides
# =============================================================================


def _warmup_mode() -> bool:
    value = os.getenv("WARMUP_MODE", "no").lower()
    if value not in ("yes", "no"):
        raise ValueError(f"WARMUP_MODE must be 'yes' or 'no', got {value!r}")
    return value == "yes"


def _selected_datasets() -> dict[str, str]:
    """Return the subset of TRAIN_DATASETS named in SWEEP_DATASETS (or all if unset)."""
    raw = os.getenv("SWEEP_DATASETS")
    if not raw:
        return dict(TRAIN_DATASETS)
    requested = tuple(s.strip() for s in raw.split(","))
    invalid = [n for n in requested if n not in TRAIN_DATASETS]
    if invalid:
        raise ValueError(f"Invalid SWEEP_DATASETS {invalid}; available: {sorted(TRAIN_DATASETS)}")
    return {n: TRAIN_DATASETS[n] for n in requested}


# =============================================================================
# Builders
# =============================================================================


@lru_cache(maxsize=1)
def _model_seq_len() -> int:
    """Model context size = base DNA seq len + special tokens (BOS)."""
    return dna_effective_seq_len(DNA_BASE_SEQ_LEN, TOKENIZER)


def _tokenize(name: str, dataset: str, dataset_format: DNALmDatasetFormat) -> ExecutorStep:
    return default_tokenize(
        name=name,
        dataset=dataset,
        tokenizer=TOKENIZER,
        format=dataset_format,
    )


def _build_data_mixture(strategy: str, dataset: str) -> LmDataConfig:
    """One training component + the v30 validation set tokenized per VAL_SPEC.

    Validation entries are absent from ``weights`` and so receive weight=0 via
    ``missing_weights_are_validation=True`` — sampled only at eval time.
    """
    components: dict[str, ExecutorStep] = {
        strategy: _tokenize(f"bolinas-v5-{strategy}-char-bos", dataset, TRAIN_FORMAT),
    }
    for suffix, fmt in VAL_SPECS:
        key = f"val_v30_{suffix}"
        components[key] = _tokenize(f"bolinas-v5-{key}-char-bos", VALIDATION_DATASET, fmt)
    return lm_mixture_data_config(
        components=components,
        weights={strategy: 1.0},
    )


def _build_model_config() -> Qwen3Config:
    return dataclasses.replace(qwen3_0_6b_hd128, max_seq_len=_model_seq_len())


def _build_optimizer() -> AdamConfig:
    return AdamConfig(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        beta1=BETA1,
        beta2=BETA2,
        epsilon=EPSILON,
        max_grad_norm=MAX_GRAD_NORM,
        warmup=WARMUP_FRACTION,
        decay=DECAY_FRACTION,
        lr_schedule=LR_SCHEDULE,
        min_lr_ratio=MIN_LR_RATIO,
    )


def _eval_harness_config() -> LmEvalHarnessConfig:
    return LmEvalHarnessConfig(
        task_spec=convert_to_levanter_task_config([TRAITGYM_MENDELIAN_V2_255]),
        include_path="experiments/evals/custom_tasks",
        max_packed_segments=1,
    )


def _checkpointer(num_train_steps: int) -> CheckpointerConfig:
    return CheckpointerConfig(
        save_interval=CHECKPOINT_TIME_INTERVAL,
        keep=[dict(every=max(1, num_train_steps // CHECKPOINTS_PER_RUN))],
    )


def _build_train_step(strategy: str, dataset: str) -> ExecutorStep:
    if _warmup_mode():
        num_train_steps = WARMUP_NUM_TRAIN_STEPS
        evals_per_run = WARMUP_EVALS_PER_RUN
    else:
        num_train_steps = NUM_TRAIN_STEPS
        evals_per_run = EVALS_PER_RUN
    steps_per_eval = max(1, num_train_steps // evals_per_run)

    warmup_suffix = "-warmup" if _warmup_mode() else ""
    run_name = f"dna-bolinas-enhancer-curation-{VERSION}{warmup_suffix}-{strategy}"
    tags = ("dna", "exp136", "enhancer_curation", VERSION, f"strategy={strategy}")
    if _warmup_mode():
        tags = (*tags, "warmup")

    inner = TrainLmConfig(
        data=_build_data_mixture(strategy, dataset),
        model=_build_model_config(),
        train_seq_len=_model_seq_len(),
        optimizer=_build_optimizer(),
        eval_harness=_eval_harness_config(),
        eval_harness_steps=steps_per_eval,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                project=WANDB_PROJECT,
                tags=list(tags),
                group=f"exp136-enhancer-curation-{VERSION}",
                name=run_name,
                replicate_path=this_output_path(),
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=BATCH_SIZE,
            num_train_steps=num_train_steps,
            steps_per_eval=steps_per_eval,
            checkpointer=_checkpointer(num_train_steps),
            mesh=MeshConfig(axes={"replica": 1, "data": -1, "model": 1}),
            allow_nondivisible_batch_size=True,
        ),
    )
    pod_config = TrainLmOnPodConfig(
        train_config=inner,
        resources=ResourceConfig.with_tpu(TPU_TYPES, regions=list(TPU_REGIONS)),
        output_path=this_output_path(),
    )
    return ExecutorStep(
        name=os.path.join("checkpoints", run_name),
        fn=remote(run_levanter_train_lm, resources=ResourceConfig.with_cpu()),
        config=pod_config,
    )


def main():
    selected = _selected_datasets()
    steps = [_build_train_step(strategy, dataset) for strategy, dataset in selected.items()]
    executor_main(steps=steps, description=f"DNA Bolinas enhancer-curation comparison {VERSION}")


if __name__ == "__main__":
    main()
