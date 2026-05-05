# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Bolinas DNA zoonomia projection sanity-check (v1-v1 and v1-v2).

First end-to-end run of the new cross-mammal projection pipeline (#158, #149)
— halLiftover Cactus 447m alignments, 108 family-deduplicated mammals,
conservation-filtered 255 bp human source windows. Two HF datasets share that
projection cohort and differ only in source-window selection:

- ``zoonomia-v1-v1``: every conserved human window (phyloP-447way ≥ 2.2162,
  ≥ 20% conserved bases per window) projected onto 108 mammals. Whole-genome
  scope.
- ``zoonomia-v1-v2``: same projection cohort, lazy ``query_name.is_in(...)``
  subset to windows whose human anchor overlaps ``[TSS - 256, TSS + 256]`` for
  any Ensembl rel 115 ``protein_coding`` transcript. TSS-proximal subset.

Different alignment backend, species set, and source-interval definition than
the mmseqs2 / ``mammals_seg20`` cCRE family used in #136-#142, so this is a
sanity-check (does the pipeline produce viable training data?) rather than a
controlled methodology comparison. The v1 vs v2 head-to-head is a secondary
read on TSS-proximal source curation at this scale.

Evaluations:

- TraitGym Mendelian v2 (255 bp) via the lm_eval harness during training,
  mirroring #142.
- LL gap = LL(functional) - LL(nonfunctional) on each of four region-specific
  validation sets, computed post-hoc from W&B as
  ``eval/val_{region}_nonfunctional/loss - eval/val_{region}_functional/loss``:

    - ``v30`` enhancers, ``v5`` CDS, ``v1`` upstream (promoter), ``v15``
      downstream (3'-UTR).

  None of the four matches the v1/v2 training distribution (those are
  conservation-filtered cross-mammal windows, not genomes-v5 annotation
  regions), so the LL-gap result is a cross-domain signal for both arms with
  no obvious favouritism between v1 and v2.

Setup mirrors #142: 0.6B Qwen3 (h=1024, L=28, head_dim=128), max_seq_len=256,
BATCH_SIZE=4096, NUM_TRAIN_STEPS=10_000, AdamConfig lr=1e-3 linear
warmup=0.1/decay=0.2, v5p-8 (~14h per arm). ``ResourceConfig`` is unpinned;
zone restriction is applied at the iris CLI (``--zone us-central1-a``) per
the #142 op note, so cross-region preempt+resume doesn't drop checkpoints.

Environment variables:
    SWEEP_DATASETS   CSV of dataset names to run (default: all in
                     ``TRAIN_DATASETS``). Useful for ``zoonomia_v1`` or
                     ``zoonomia_v2`` in isolation.

https://github.com/Open-Athena/bolinas-dna/issues/160
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

# Two arms of the new zoonomia projection pipeline (#158): same projection
# cohort, differing only in source-window selection (whole-genome vs
# TSS-proximal subset).
TRAIN_DATASETS = {
    "zoonomia_v1": "bolinas-dna/zoonomia-v1-v1",  # whole-genome cohort
    "zoonomia_v2": "bolinas-dna/zoonomia-v1-v2",  # TSS-proximal subset of v1
}

# Four region-specific validation sets, each tokenized functional + nonfunctional
# for the LL-gap signal. Region IDs match the genomes-v5 interval definitions
# on HF; comments give the biological annotation each ID corresponds to.
VAL_DATASETS: tuple[tuple[str, str], ...] = (
    ("v30", "bolinas-dna/genomes-v5-validation-intervals-v30_255_255"),  # enhancers
    ("v5", "bolinas-dna/genomes-v5-validation-intervals-v5_255_255"),  # CDS
    ("v1", "bolinas-dna/genomes-v5-validation-intervals-v1_255_255"),  # upstream (promoter)
    ("v15", "bolinas-dna/genomes-v5-validation-intervals-v15_255_255"),  # downstream (3'-UTR)
)

# Training masks lowercase positions to 1% loss weight (consistent across
# Bolinas DNA experiments).
TRAIN_FORMAT = DNALmDatasetFormat(lowercase_weight=0.01)

# Validation tokenization specs — only the two terms of the LL gap; we
# deliberately skip a "default" matched-to-training variant since it adds eval
# cost without informing the v1 vs v2 comparison.
VAL_SPECS: tuple[tuple[str, DNALmDatasetFormat], ...] = (
    ("functional", DNALmDatasetFormat(uppercase_weight=1.0, lowercase_weight=0.0)),
    ("nonfunctional", DNALmDatasetFormat(uppercase_weight=0.0, lowercase_weight=1.0)),
)

# Architecture: ~0.6B Qwen3 (h=1024, L=28, head_dim=128) — imported from
# the canonical preset so this file picks up any upstream changes
# automatically; max_seq_len is overridden to the DNA context size below.

BATCH_SIZE = 4096
TPU_TYPES: tuple[str, ...] = ("v5p-8",)

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

# Training horizon. Matches the #142 reference scale (~14h on v5p-8).
NUM_TRAIN_STEPS = 10_000

# Eval cadence and checkpoint policy.
EVALS_PER_RUN = 10
CHECKPOINTS_PER_RUN = 3
CHECKPOINT_TIME_INTERVAL = timedelta(hours=1)

WANDB_PROJECT = "marin"

_EXPECTED_VOCAB_SIZE_WARNING = f"Tokenizer {TOKENIZER!r} not found in _KNOWN_VOCAB_SIZES"
logging.getLogger("marin.processing.tokenize.data_configs").addFilter(
    lambda record: _EXPECTED_VOCAB_SIZE_WARNING not in record.getMessage()
)


# =============================================================================
# Environment overrides
# =============================================================================


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


# Marin's only CPU worker pool is n2-highmem-2 (2 vCPU / 16 GiB) and is
# typically near-saturated by other users' small coordinators, so override
# default_tokenize's cpu=4 ask down to a minimal orchestrator footprint that
# can pack alongside an existing job on the same VM. The tokenize step is
# pure orchestration — heavy work runs on zephyr workers spawned at their
# own resource spec — so cpu=1 is plenty.
_TOKENIZE_RESOURCES = ResourceConfig.with_cpu(cpu=1, ram="4g", disk="10g")


def _tokenize(name: str, dataset: str, dataset_format: DNALmDatasetFormat) -> ExecutorStep:
    return default_tokenize(
        name=name,
        dataset=dataset,
        tokenizer=TOKENIZER,
        format=dataset_format,
        resources=_TOKENIZE_RESOURCES,
    )


def _build_data_mixture(strategy: str, dataset: str) -> LmDataConfig:
    """One training component + four validation sets each tokenized per VAL_SPEC.

    Validation entries are absent from ``weights`` and so receive weight=0 via
    ``missing_weights_are_validation=True`` — sampled only at eval time. Val
    tokenization names use the ``bolinas-v5-val_{region}_{suffix}-char-bos``
    convention from #142, letting the ``val_v30_*`` tokenizations dedupe with
    that experiment's executor cache.
    """
    components: dict[str, ExecutorStep] = {
        strategy: _tokenize(f"bolinas-zoonomia-v1-{strategy}-char-bos", dataset, TRAIN_FORMAT),
    }
    for region, val_dataset in VAL_DATASETS:
        for suffix, fmt in VAL_SPECS:
            key = f"val_{region}_{suffix}"
            components[key] = _tokenize(f"bolinas-v5-{key}-char-bos", val_dataset, fmt)
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
    steps_per_eval = max(1, NUM_TRAIN_STEPS // EVALS_PER_RUN)
    run_name = f"dna-bolinas-zoonomia-v1-v2-{VERSION}-{strategy}"
    tags = ("dna", "exp160", "zoonomia_v1_v2", VERSION, f"strategy={strategy}")

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
                group=f"exp160-zoonomia-v1-v2-{VERSION}",
                name=run_name,
                replicate_path=this_output_path(),
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=BATCH_SIZE,
            num_train_steps=NUM_TRAIN_STEPS,
            steps_per_eval=steps_per_eval,
            checkpointer=_checkpointer(NUM_TRAIN_STEPS),
            mesh=MeshConfig(axes={"replica": 1, "data": -1, "model": 1}),
            allow_nondivisible_batch_size=True,
        ),
    )
    pod_config = TrainLmOnPodConfig(
        train_config=inner,
        resources=ResourceConfig.with_tpu(TPU_TYPES),
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
    executor_main(steps=steps, description=f"DNA Bolinas zoonomia v1 vs v2 sanity-check {VERSION}")


if __name__ == "__main__":
    main()
