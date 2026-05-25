# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exp 30: WSD vs cosine LR schedule on the m11 protein-docs mixture.

Tracks ``Open-Athena/MarinFold#30``. Two trials at 100M / batch=128 / ~4.3B
tokens on mixture m11 (size-proportional H/M/L blend over the quality-bucketed
``eczech/marinfold-exp11-protein-docs`` re-publication, loss-masked to
``protein_train_common.distance_masked_components()``):

* ``wsd`` — linear-decay over the trailing ``LR_DECAY`` fraction of steps
  (the WSD-style schedule used by the exp11 data-mix sweeps).
* ``cosine`` — Levanter's ``AdamConfig`` defaults (lr_schedule="cosine",
  decay=None ⇒ full decay), matching the recipe in
  ``train_protein_1_5b_distance_masked.py``.

The recipe is adapted from ``train_protein_1_5b_distance_masked.py``:
batch=128, seq=8192, weight_decay=0.01, warmup=0.1. ``LEARNING_RATE`` is
rescaled to the 100M model via ``LR_CONSTANT * sqrt(batch) / hidden`` (the
1.5B recipe solves LR_CONSTANT from its own (batch=128, hidden=2048, lr=3.5e-4)
point and reuses it here). ``BETA2`` follows the noise-scale heuristic
(0.98 at batch=128). The model architecture is fixed to Llama —
``protein_llama_100m`` from ``train_protein_100m_distance_masked.py``.

Env vars: ``RUNS`` (CSV substring filter on variant id), ``PREVIEW=yes``
(list targets, submit nothing), ``TPU`` (override worker TPU; default v5p-8).

Submission (one Fray training job per variant, four iris jobs in total
across exp29 + exp30)::

    TIMESTAMP=$(date +%Y%m%d-%H%M)
    for variant in wsd cosine; do
        uv run iris --cluster=marin job run --user $USERNAME --no-wait \\
            --job-name prot-exp30-${variant}-${TIMESTAMP} \\
            --region us-east5 --memory=1GB \\
            -e HF_TOKEN "$HF_TOKEN" -e WANDB_API_KEY "$WANDB_API_KEY" \\
            -e RUNS $variant \\
            -- python -m experiments.protein.exp30_lrsch_sweep
    done
"""

import dataclasses
import logging
import math
import os
from dataclasses import dataclass
from datetime import timedelta

from fray import ResourceConfig, current_client
from fray.types import Entrypoint, JobRequest, create_environment
from levanter.callbacks.watch import WatchConfig
from levanter.data.text import DatasetComponent, LmDataConfig, TextLmDatasetFormat, UrlDatasetSourceConfig
from marin.execution.executor import versioned
from marin.execution.sweep import SweepTarget, claim_and_run
from marin.training.training import extras_for_resources, resolve_training_env

from experiments.defaults import _run_training_on_worker, prepare_lm_train
from experiments.llama import compute_num_parameters
from experiments.protein.protein_train_common import (
    PROTEIN_TOKENIZER,
    distance_bin_only_loss_weight,
    protein_docs_val_tokenized,
)
from experiments.protein.train_protein_100m_distance_masked import protein_llama_100m
from experiments.simple_train_config import SimpleTrainConfig

logger = logging.getLogger(__name__)


# --- Identity ----------------------------------------------------------------

# Bump to fork run names + sweep-root claim dir on a recipe change.
VERSION: str = "v1"

RUN_NAME_PREFIX: str = "prot-exp30"
SWEEP_ROOT: str = f"gs://marin-us-east5/sweeps/prot-exp30-lrsch/run_lrsch_sweep-{VERSION}"
WANDB_GROUP: str = "exp30-lrsch"

# --- Data --------------------------------------------------------------------

# Quality-bucketed re-publication of cd-train tokens (H=round0, M=round1,
# L=round2..4), tokenized once by exp11_data_mix_sweep into per-(quality,split)
# Levanter caches. Hardcoded so this script is independent of exp11.
HF_DATASET_ID: str = "eczech/marinfold-exp11-protein-docs"
HF_REVISION: str = "41b2ec71070cb9e8799311cd8f78877e747f6754"

CACHE_PREFIX: str = "gs://marin-us-east5/tokenized/exp11-data-mix-41b2ec7-v1"
H_TRAIN_CACHE: str = f"{CACHE_PREFIX}/high-train/"
M_TRAIN_CACHE: str = f"{CACHE_PREFIX}/medium-train/"
L_TRAIN_CACHE: str = f"{CACHE_PREFIX}/low-train/"
H_VAL_CACHE: str = f"{CACHE_PREFIX}/high-val/"
M_VAL_CACHE: str = f"{CACHE_PREFIX}/medium-val/"
L_VAL_CACHE: str = f"{CACHE_PREFIX}/low-val/"
H_TEST_CACHE: str = f"{CACHE_PREFIX}/high-test/"
M_TEST_CACHE: str = f"{CACHE_PREFIX}/medium-test/"
L_TEST_CACHE: str = f"{CACHE_PREFIX}/low-test/"

CD_VAL_CACHE: str = protein_docs_val_tokenized.override_output_path

COMPONENT_H_TRAIN: str = "protein-docs-high-train"
COMPONENT_M_TRAIN: str = "protein-docs-medium-train"
COMPONENT_L_TRAIN: str = "protein-docs-low-train"
COMPONENT_H_VAL: str = "protein-docs-high-val"
COMPONENT_M_VAL: str = "protein-docs-medium-val"
COMPONENT_L_VAL: str = "protein-docs-low-val"
COMPONENT_H_TEST: str = "protein-docs-high-test"
COMPONENT_M_TEST: str = "protein-docs-medium-test"
COMPONENT_L_TEST: str = "protein-docs-low-test"
COMPONENT_CD_VAL: str = "protein-docs-cd-val"
COMPONENT_CD_VAL_UNMASKED: str = "protein-docs-cd-val-unmasked"

# Mixture m11: per-quality train weights proportional to per-cell sizes
# (H ≈ 1.68M / 5.39M, M ≈ 1.42M / 5.39M, L ≈ 2.29M / 5.39M, rounded to 2dp).
# Identical to ``exp11_data_mix_sweep.Mixture(id="m11", ...)``.
M11_WEIGHT_H: float = 0.31
M11_WEIGHT_M: float = 0.26
M11_WEIGHT_L: float = 0.43

IID_EVAL_SEQS_PER_TRAIN: int = 4096
MAX_EVAL_BATCHES: int = 16

# --- Model -------------------------------------------------------------------

# Lineage: ``protein_llama_100m`` is the 100M Llama defined in
# ``train_protein_100m_distance_masked.py`` (h=768, l=12, dff=3072, heads=12,
# kv=4). The arch sweep (exp29) varies this; here it's fixed.
MODEL_CONFIG = protein_llama_100m

PROTEIN_VOCAB_SIZE: int = 2840
HIDDEN_DIM: int = 768

# --- Optimizer / schedule ----------------------------------------------------

BATCH_SIZE: int = 128
SEQ_LEN: int = 8192
NUM_TRAIN_STEPS: int = 4104  # 4.3B tokens / (128 * 8192) ≈ 4104 (snapped to a multiple of 8 evals)
STEPS_PER_EVAL: int = 513    # NUM_TRAIN_STEPS / 8 evals
WEIGHT_DECAY: float = 0.01
WARMUP: float = 0.1

# LR scaling — adapted from ``train_protein_1_5b_distance_masked.py``:
#   reference: lr=3.5e-4 @ batch=128, hidden=2048
#   solve: LR_CONSTANT = lr * hidden / sqrt(batch)
#   apply: lr_here = LR_CONSTANT * sqrt(BATCH_SIZE) / HIDDEN_DIM
# At BATCH_SIZE=128 / HIDDEN_DIM=768 this resolves to ~9.33e-4.
LR_REF: float = 3.5e-4
LR_REF_BATCH: int = 128
LR_REF_HIDDEN: int = 2048
LR_CONSTANT: float = LR_REF * LR_REF_HIDDEN / math.sqrt(LR_REF_BATCH)
LEARNING_RATE: float = LR_CONSTANT * math.sqrt(BATCH_SIZE) / HIDDEN_DIM

# Adam β₂ scaled per the noise-scale heuristic (0.98 at batch=128).
BETA2: float = 0.98 ** (BATCH_SIZE / LR_REF_BATCH)

# WSD = linear decay over the trailing LR_DECAY fraction of steps; cosine
# leaves both unset so SimpleTrainConfig falls back to AdamConfig defaults
# (lr_schedule="cosine", decay=None ⇒ full decay).
WSD_SCHEDULE: str = "linear"
WSD_DECAY: float = 0.2

# Pinned so cross-run comparisons share the same data permutation. Distinct
# from the smoke/mix/scale stages of exp11 (DATA_SEED=1729) and from exp29
# (DATA_SEED=73) so this training stream doesn't overlap.
DATA_SEED: int = 42

SHUFFLE: bool = True
PERMUTATION_TYPE: str = "feistel"
MIXTURE_BLOCK_SIZE: int = 2048

TEMP_CHECKPOINT_INTERVAL = timedelta(minutes=8)

# --- Resources --------------------------------------------------------------

PROTEIN_ZONE: str = "us-east5-a"
DEFAULT_TPU: str = "v5p-8"


def tpu() -> str:
    return os.environ.get("TPU") or DEFAULT_TPU


def resources() -> ResourceConfig:
    return ResourceConfig.with_tpu(tpu(), zone=PROTEIN_ZONE)


# --- Variants ----------------------------------------------------------------


@dataclass(frozen=True)
class Variant:
    """One trial's LR schedule + variant tag.

    Both arms in this sweep share an LR/optimizer recipe and the Llama model;
    only the post-warmup decay schedule varies. ``lr_schedule=None`` +
    ``decay=None`` falls through to AdamConfig defaults (cosine + full decay).
    """

    variant_id: str  # "wsd" | "cosine"; baked into the run name and W&B tags
    lr_schedule: str | None
    decay: float | None


VARIANTS: tuple[Variant, ...] = (
    Variant(variant_id="wsd", lr_schedule=WSD_SCHEDULE, decay=WSD_DECAY),
    Variant(variant_id="cosine", lr_schedule=None, decay=None),
)


# --- Helpers ----------------------------------------------------------------


def fmt_count(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1e9:.3f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)


def fmt_lr(lr: float) -> str:
    mantissa, exponent = f"{lr:.1e}".split("e")
    return f"{mantissa}e{int(exponent)}"


TOKENS_TAG: str = fmt_count(BATCH_SIZE * SEQ_LEN * NUM_TRAIN_STEPS)  # "4.303B"


def trial_name(variant: Variant) -> str:
    """Stable per-trial run id (also the per-target dir under ``SWEEP_ROOT``)."""
    return (
        f"{RUN_NAME_PREFIX}-100m-{TOKENS_TAG}-m11-{variant.variant_id}-"
        f"lr{fmt_lr(LEARNING_RATE)}-{VERSION}"
    )


# --- Data config ------------------------------------------------------------


def empty_source_component(cache_dir: str, *, masked: bool) -> DatasetComponent:
    """Cache-only component: empty URL lists short-circuit Levanter's cache-build."""
    fmt = TextLmDatasetFormat(text_key="document")
    source = UrlDatasetSourceConfig(
        train_urls=[],
        validation_urls=[],
        cache_dir=cache_dir,
        format=fmt,
        tags=[],
    )
    return DatasetComponent(
        source=source,
        cache_dir=cache_dir,
        format=fmt,
        pack=True,
        tags=[],
        loss_weight_fn=distance_bin_only_loss_weight if masked else None,
    )


def build_data_config() -> LmDataConfig:
    """LmDataConfig over m11 (H/M/L train weights) + 6 heldout cells + cd-val."""
    components: dict[str, DatasetComponent] = {
        COMPONENT_H_TRAIN: empty_source_component(H_TRAIN_CACHE, masked=True),
        COMPONENT_M_TRAIN: empty_source_component(M_TRAIN_CACHE, masked=True),
        COMPONENT_L_TRAIN: empty_source_component(L_TRAIN_CACHE, masked=True),
        COMPONENT_H_VAL: empty_source_component(H_VAL_CACHE, masked=True),
        COMPONENT_M_VAL: empty_source_component(M_VAL_CACHE, masked=True),
        COMPONENT_L_VAL: empty_source_component(L_VAL_CACHE, masked=True),
        COMPONENT_H_TEST: empty_source_component(H_TEST_CACHE, masked=True),
        COMPONENT_M_TEST: empty_source_component(M_TEST_CACHE, masked=True),
        COMPONENT_L_TEST: empty_source_component(L_TEST_CACHE, masked=True),
        COMPONENT_CD_VAL: empty_source_component(CD_VAL_CACHE, masked=True),
        COMPONENT_CD_VAL_UNMASKED: empty_source_component(CD_VAL_CACHE, masked=False),
    }
    train_weights: dict[str, float] = {
        COMPONENT_H_TRAIN: M11_WEIGHT_H,
        COMPONENT_M_TRAIN: M11_WEIGHT_M,
        COMPONENT_L_TRAIN: M11_WEIGHT_L,
        COMPONENT_H_VAL: 0.0,
        COMPONENT_M_VAL: 0.0,
        COMPONENT_L_VAL: 0.0,
        COMPONENT_H_TEST: 0.0,
        COMPONENT_M_TEST: 0.0,
        COMPONENT_L_TEST: 0.0,
        COMPONENT_CD_VAL: 0.0,
        COMPONENT_CD_VAL_UNMASKED: 0.0,
    }
    num_validation_sequences: dict[str, int] = {
        COMPONENT_H_TRAIN: IID_EVAL_SEQS_PER_TRAIN,
        COMPONENT_M_TRAIN: IID_EVAL_SEQS_PER_TRAIN,
        COMPONENT_L_TRAIN: IID_EVAL_SEQS_PER_TRAIN,
    }
    return LmDataConfig(
        components=components,
        train_weights=train_weights,
        tokenizer=PROTEIN_TOKENIZER,
        cache_dir=None,
        block_cross_document_attention=True,
        shuffle=SHUFFLE,
        permutation_type=PERMUTATION_TYPE,
        mixture_block_size=MIXTURE_BLOCK_SIZE,
        num_validation_sequences=num_validation_sequences,
    )


# --- Trial construction -----------------------------------------------------


def build_trial(variant: Variant) -> tuple[str, object]:
    """Build one trial's ``(job_name, raw_config)`` for ``prepare_lm_train``."""
    train_config = SimpleTrainConfig(
        resources=resources(),
        train_batch_size=BATCH_SIZE,
        num_train_steps=NUM_TRAIN_STEPS,
        learning_rate=versioned(LEARNING_RATE),
        weight_decay=WEIGHT_DECAY,
        beta2=BETA2,
        warmup=WARMUP,
        decay=variant.decay,
        lr_schedule=variant.lr_schedule,
        train_seq_len=SEQ_LEN,
        steps_per_eval=STEPS_PER_EVAL,
        steps_per_export=None,
        max_eval_batches=MAX_EVAL_BATCHES,
        data_seed=DATA_SEED,
        per_device_parallelism=-1,
    )
    params = compute_num_parameters(MODEL_CONFIG, PROTEIN_VOCAB_SIZE)
    tokens = BATCH_SIZE * SEQ_LEN * NUM_TRAIN_STEPS
    job_name, raw_config = prepare_lm_train(
        name=trial_name(variant),
        tokenized=build_data_config(),
        model_config=MODEL_CONFIG,
        train_config=train_config,
        tags=[
            "protein",
            "exp30",
            "lrsch",
            variant.variant_id,
            "llama",
            "100m",
            "m11",
            f"params={fmt_count(params)}",
            f"params_exact={params}",
            f"tokens={fmt_count(tokens)}",
            f"tokens_exact={tokens}",
            f"steps={NUM_TRAIN_STEPS}",
        ],
        eval_harness_tasks=[],
        use_default_validation=False,
        wandb_group=WANDB_GROUP,
    )
    raw_config = dataclasses.replace(
        raw_config,
        trainer=dataclasses.replace(
            raw_config.trainer,
            checkpointer=dataclasses.replace(
                raw_config.trainer.checkpointer,
                save_interval=TEMP_CHECKPOINT_INTERVAL,
            ),
            watch=WatchConfig(watch_targets=[], interval=0),
        ),
    )
    return job_name, raw_config


# --- Selection / preview ----------------------------------------------------


def selected_variants() -> tuple[Variant, ...]:
    """``RUNS`` is a CSV substring filter on ``variant_id``; empty = all."""
    raw = os.environ.get("RUNS", "")
    needles = tuple(s.strip() for s in raw.split(",") if s.strip())
    if not needles:
        return VARIANTS
    return tuple(v for v in VARIANTS if any(n in v.variant_id for n in needles))


def preview() -> bool:
    return os.environ.get("PREVIEW", "").strip().lower() in {"yes", "true", "1"}


def print_preview(variants: tuple[Variant, ...]) -> None:
    print(f"PREVIEW: exp30 lrsch sweep would run {len(variants)} target(s):", flush=True)
    for v in variants:
        schedule_str = (
            "cosine,full" if v.lr_schedule is None and v.decay is None else f"{v.lr_schedule},decay={v.decay}"
        )
        print(f"  {trial_name(v)}", flush=True)
        print(
            f"    arch=llama batch={BATCH_SIZE} steps={NUM_TRAIN_STEPS} "
            f"steps_per_eval={STEPS_PER_EVAL} lr={LEARNING_RATE:.4g} beta2={BETA2:.4g} "
            f"tokens={TOKENS_TAG} schedule={schedule_str} data_seed={DATA_SEED}",
            flush=True,
        )
        print(
            f"    mixture m11: H={M11_WEIGHT_H} M={M11_WEIGHT_M} L={M11_WEIGHT_L}",
            flush=True,
        )
        print(flush=True)


# --- Worker + launcher ------------------------------------------------------


def worker_entrypoint(variants: tuple[str, ...]) -> None:
    """One Fray worker: take all variants and run them via ``claim_and_run``.

    Fan-out across Fray workers is unused here — the bash launcher submits one
    iris job per variant, so each worker sees a single-element ``variants``
    tuple. ``claim_and_run`` is kept for idempotency (re-running a completed
    target is a no-op).
    """
    targets = [SweepTarget(target_id=trial_name(v), config=v.variant_id) for v in VARIANTS if v.variant_id in variants]
    logger.info("Worker assigned %d/%d target(s): %s", len(targets), len(VARIANTS), [t.target_id for t in targets])

    def run_one(target: SweepTarget) -> None:
        variant = next(v for v in VARIANTS if v.variant_id == target.config)
        name, raw_config = build_trial(variant)
        _run_training_on_worker(name=name, raw_config=raw_config, override_output_path=None, resources=resources())

    claim_and_run(SWEEP_ROOT, targets, run_one)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    variants = selected_variants()
    if not variants:
        raise ValueError(f"No variants matched RUNS={os.environ.get('RUNS', '')!r}")

    if preview():
        print_preview(variants)
        return

    res = resources()
    env = resolve_training_env(base_env=None, resources=res)
    extras = extras_for_resources(res)
    logger.info("Submitting %d Fray worker job(s); variants=%s", 1, [v.variant_id for v in variants])

    client = current_client()
    request = JobRequest(
        name=f"{RUN_NAME_PREFIX}-lrsch-w0",
        entrypoint=Entrypoint.from_callable(worker_entrypoint, args=[tuple(v.variant_id for v in variants)]),
        resources=res,
        environment=create_environment(env_vars=env, extras=extras),
    )
    handle = client.submit(request)
    logger.info("Submitted worker: %s", request.name)
    handle.wait(raise_on_failure=True)
    logger.info("Worker finished")


if __name__ == "__main__":
    main()
