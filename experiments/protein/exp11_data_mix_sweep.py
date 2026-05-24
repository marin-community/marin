# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exp 11: ``cd`` data-mix sweep (Fray-native, no Executor).

Tracks ``Open-Athena/MarinFold#11``. Compares mixtures derived from the
quality-bucketed re-publication ``eczech/marinfold-exp11-protein-docs``
(H=round0, M=round1, L=round2..4), loss-masked to match
``protein_train_common.distance_masked_components()``.

Hyperparameter reference recipe (from ``train_protein_1_5b_distance_masked.py``):
batch=128, seq=8192, lr=3.5e-4, weight_decay=0.01, warmup=0.1. Current runs
use ``BATCH_SIZE`` (set below) with LR rescaled via
``scaled_lr(b, h) = LR_CONSTANT * sqrt(b) / h`` and
``beta2 = 0.98 ** (BATCH_SIZE / 128)``.

Subcommands (``COMMAND`` env var):

* ``tokenize`` — one Fray job per (quality, split) cell.
* ``run_smoke`` — single trial: m9 (staged), 100M, ~400M tokens; m9
  exercises the staged-mixture path.
* ``run_mix_sweep`` — all 9 mix mixtures (m1..m9) at 100M, ~4.3B tokens each.
* ``run_scale_sweep`` — 6 scale mixtures (m10..m15) at 1.5B on v5p-8,
  ~21.5B tokens each (one job per mixture).

Env vars: ``COMMAND`` (required), ``RUNS`` (CSV substring filter on target
ids), ``PREVIEW=yes`` (list targets, submit nothing), ``NUM_WORKERS``
(default per stage), ``TPU`` (override worker TPU; default v5p-8).

Smoke usage::

    uv run iris --cluster=marin job run --user $USERNAME --no-wait \\
        --job-name prot-exp11-smoke-$(date +%Y%m%d-%H%M) \\
        --region us-east5 --memory=1GB \\
        -e HF_TOKEN "$HF_TOKEN" -e WANDB_API_KEY "$WANDB_API_KEY" \\
        -e COMMAND run_smoke \\
        -- python -m experiments.protein.exp11_data_mix_sweep

Preview without submitting::

    COMMAND=run_mix_sweep PREVIEW=yes \\
        uv run python -m experiments.protein.exp11_data_mix_sweep
"""

import dataclasses
import logging
import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta

from fray import ResourceConfig, current_client
from fray.types import Entrypoint, JobRequest, create_environment
from levanter.data.text import DatasetComponent, LmDataConfig, TextLmDatasetFormat, UrlDatasetSourceConfig
from marin.execution.executor import versioned
from marin.execution.sweep import SweepTarget, claim_and_run
from marin.processing.tokenize.tokenize import TokenizeConfig, tokenize
from marin.training.training import extras_for_resources, resolve_training_env
from rigging.filesystem import marin_prefix

from experiments.defaults import _run_training_on_worker, prepare_lm_train
from experiments.llama import compute_num_parameters
from experiments.protein.protein_train_common import (
    PROTEIN_TOKENIZER,
    distance_bin_only_loss_weight,
    protein_docs_val_tokenized,
)
from experiments.protein.train_protein_1_5b_distance_masked import protein_llama_1_5b
from experiments.protein.train_protein_100m_distance_masked import protein_llama_100m
from experiments.simple_train_config import SimpleTrainConfig

logger = logging.getLogger(__name__)

# --- Data source -------------------------------------------------------------

# SHA-pinned so an HF re-shard can't silently change tokenize inputs;
# ``DATA_VERSION`` forks the cache independently.
HF_DATASET_ID = "eczech/marinfold-exp11-protein-docs"
HF_REVISION = "41b2ec71070cb9e8799311cd8f78877e747f6754"
HF_REVISION_SHORT = HF_REVISION[:7]

# Quality bucket -> HF config name; HF layout is ``<config>/{train,val,test}/*.parquet``.
QUALITY_CONFIGS: dict[str, str] = {"H": "high", "M": "medium", "L": "low"}
HF_SPLITS: tuple[str, ...] = ("train", "val", "test")

# --- Cache layout ------------------------------------------------------------

# Bump to fork the cache when tokenize semantics change; ``HF_REVISION_SHORT``
# is baked in so an HF rev bump also forks.
DATA_VERSION = "v1"
CACHE_SUBPATH = f"tokenized/exp11-data-mix-{HF_REVISION_SHORT}-{DATA_VERSION}"


@dataclass(frozen=True)
class Cell:
    """One (quality bucket, HF split) tokenize cell."""

    quality: str  # "H" | "M" | "L"
    split: str  # "train" | "val" | "test"

    @property
    def hf_config(self) -> str:
        return QUALITY_CONFIGS[self.quality]

    @property
    def suffix(self) -> str:
        return f"{self.hf_config}-{self.split}"

    @property
    def component_name(self) -> str:
        return f"protein-docs-{self.suffix}"

    @property
    def is_train(self) -> bool:
        return self.split == "train"


ALL_CELLS: tuple[Cell, ...] = tuple(Cell(q, s) for q in QUALITY_CONFIGS for s in HF_SPLITS)


def cell_input_glob(cell: Cell) -> str:
    """HF parquet glob for one cell."""
    return f"hf://datasets/{HF_DATASET_ID}@{HF_REVISION}/{cell.hf_config}/{cell.split}/*.parquet"


def cell_cache_path(cell: Cell) -> str:
    """Levanter cache root for one tokenized cell.

    Train cells write under ``<cache>/train/``, val/test under ``<cache>/validation/``.
    """
    return f"{marin_prefix()}/{CACHE_SUBPATH}/{cell.suffix}/"


# --- Tokenize ---------------------------------------------------------------

# Coordinator-side is slack; zephyr children do the heavy work.
TOKENIZE_RESOURCES = ResourceConfig(cpu=2, ram="16G", disk="50G")


def _tokenize_one_cell(cell: Cell) -> None:
    """Tokenize one HF cell into its cache path. Idempotent on cache hit."""
    glob = [cell_input_glob(cell)]
    config = TokenizeConfig(
        train_paths=glob if cell.is_train else [],
        validation_paths=[] if cell.is_train else glob,
        cache_path=cell_cache_path(cell),
        tokenizer=PROTEIN_TOKENIZER,
        format=TextLmDatasetFormat(text_key="document"),
        # HF "test" splits feed the validation cache for eval, not train —
        # bypass the safety assert.
        allow_test_in_train=True,
    )
    tokenize(config)


def _tokenize_main(cells: list[Cell]) -> None:
    logger.info("Submitting %d tokenize job(s)", len(cells))
    client = current_client()
    env = create_environment(extras=extras_for_resources(TOKENIZE_RESOURCES))
    handles = []
    for cell in cells:
        handles.append(
            client.submit(
                JobRequest(
                    name=f"prot-exp11-tok-{cell.suffix}-{DATA_VERSION}",
                    entrypoint=Entrypoint.from_callable(_tokenize_one_cell, args=[cell]),
                    resources=TOKENIZE_RESOURCES,
                    environment=env,
                )
            )
        )

    failures = 0
    for h, cell in zip(handles, cells, strict=True):
        try:
            h.wait(raise_on_failure=True)
            logger.info("Finished cell %s", cell.suffix)
        except Exception:
            failures += 1
            logger.exception("Cell %s failed", cell.suffix)
    if failures:
        raise RuntimeError(f"{failures}/{len(cells)} tokenize job(s) failed")


# --- Models -----------------------------------------------------------------

SEQ_LEN = 8192

# Pulled from the existing model configs so arch changes propagate here.
HIDDEN_100M: int = protein_llama_100m.hidden_dim  # 768
HIDDEN_1_5B: int = protein_llama_1_5b.hidden_dim  # 2048

# --- Optimizer / LR scaling --------------------------------------------------

# Source-of-truth recipe from ``train_protein_1_5b_distance_masked.py``:
LR_REF: float = 3.5e-4
LR_REF_BATCH: int = 128
LR_REF_HIDDEN: int = HIDDEN_1_5B  # 2048

# Solve ``lr = lr_constant * sqrt(batch) / hidden`` from the 1.5B recipe;
# reuse to scale LR at other (batch, hidden) pairs.
LR_CONSTANT: float = LR_REF * LR_REF_HIDDEN / math.sqrt(LR_REF_BATCH)


def scaled_lr(batch_size: int, hidden_size: int) -> float:
    """LR per the source recipe's sqrt-batch / inverse-hidden scaling."""
    return LR_CONSTANT * math.sqrt(batch_size) / hidden_size


# Operating point for all sweeps. LR scales by sqrt(BATCH_SIZE/128) via
# scaled_lr; beta2 follows the noise-scale heuristic
# beta2 = 0.98 ** (BATCH_SIZE / 128). At BATCH_SIZE=256 these resolve to:
#   smoke/mix (HIDDEN_100M=768):  lr = 1.32e-3
#   scale     (HIDDEN_1_5B=2048): lr = 4.95e-4
#   beta2 (all sweeps): 0.9604
BATCH_SIZE: int = 256
BETA2: float = 0.98 ** (BATCH_SIZE / 128)

WEIGHT_DECAY: float = 0.01
WARMUP: float = 0.1

# Linear decay to ``LR_DECAY * peak`` post-warmup. Overrides Levanter's cosine
# default so smoke/mix/scale runs are directly comparable.
LR_SCHEDULE: str = "linear"
LR_DECAY: float = 0.2

# Rolling temp-checkpoint cadence. Overrides defaults.py's 10-min default so
# preemption/host-loss costs ≤8 min of progress.
TEMP_CHECKPOINT_INTERVAL = timedelta(minutes=8)

# Evenly-spaced permanent checkpoints per run.
# steps_per_export = num_train_steps // NUM_PERMANENT_CHECKPOINTS.
NUM_PERMANENT_CHECKPOINTS: int = 5

# Vocab for the legacy 2840-vocab tokenizer pinned in PROTEIN_TOKENIZER.
# Hardcoded so it lands in run-config hashes / wandb tags; bump on pin change.
PROTEIN_VOCAB_SIZE: int = 2840

# IID held-out sequences carved per train cell; ~4096 ≈ 33.5M tokens / cell.
IID_EVAL_SEQS_PER_TRAIN: int = 4096

# At BATCH_SIZE=256, seq=8192: 16 * 256 = 4096 examples, ~33.6M tokens / component.
MAX_EVAL_BATCHES: int = 16

# Pinned so cross-run comparisons share the same data permutation.
DATA_SEED: int = 1729

# Full Feistel (not Levanter's hierarchical-block default): clean
# cross-mixture comparisons, and per-cell sizes (≤5.28M packed seqs) stay in
# Feistel PRP's cheap regime.
SHUFFLE: bool = True
PERMUTATION_TYPE: str = "feistel"

# MixtureDataset's per-block proportional guarantee. ``block_size`` must be a
# multiple of train_batch_size for staged mixtures to land on a valid boundary;
# ``_resolve_mixture_weights`` enforces this.
MIXTURE_BLOCK_SIZE: int = 2048

# --- Resources --------------------------------------------------------------

# us-east5-a co-locates TPUs with the ``marin-us-east5`` checkpoint bucket.
PROTEIN_ZONE = "us-east5-a"

# All stages run BATCH_SIZE with ``scaled_lr`` so v5p-8 fits.
_DEFAULT_TPU = "v5p-8"


def _tpu() -> str:
    return os.environ.get("TPU") or _DEFAULT_TPU


def _resources() -> ResourceConfig:
    return ResourceConfig.with_tpu(_tpu(), zone=PROTEIN_ZONE)


# --- Mixtures ---------------------------------------------------------------


@dataclass(frozen=True)
class Mixture:
    """One named train-data mixture over (H, M, L) train cells.

    Either ``static`` (flat dict) or ``staged`` (list of ``(stage_frac, weights)``
    entries). Fractions are resolved by :func:`_resolve_mixture_weights`, which
    snaps each boundary to a ``MixtureDataset`` block edge.
    """

    id: str
    static: dict[str, float] | None = None
    staged: tuple[tuple[float, dict[str, float]], ...] | None = None


def _resolve_mixture_weights(
    mixture: Mixture, num_train_steps: int, *, batch_size: int
) -> dict[str, float] | list[tuple[int, dict[str, float]]]:
    """Resolve a mixture to MixtureDataset's expected weights form.

    Staged: snaps ``frac * num_train_steps`` DOWN to a multiple of
    ``MIXTURE_BLOCK_SIZE // batch_size`` so ``step * batch_size`` lands on a
    block boundary. First stage pinned to 0; monotonicity re-asserted after
    snapping in case two stages collapse.
    """
    if mixture.static is not None:
        return dict(mixture.static)
    assert mixture.staged is not None

    assert MIXTURE_BLOCK_SIZE % batch_size == 0, (
        f"MIXTURE_BLOCK_SIZE ({MIXTURE_BLOCK_SIZE}) must be a multiple of "
        f"batch_size ({batch_size}) so stage boundaries can land on block edges"
    )
    alignment = MIXTURE_BLOCK_SIZE // batch_size

    stages: list[tuple[int, dict[str, float]]] = []
    for i, (frac, w) in enumerate(mixture.staged):
        if i == 0:
            step = 0
        else:
            raw_step = round(frac * num_train_steps)
            step = (raw_step // alignment) * alignment
            prev = stages[-1][0]
            if step <= prev:
                step = prev + alignment  # keep stages strictly increasing
        stages.append((step, dict(w)))
    if stages[-1][0] >= num_train_steps:
        raise ValueError(
            f"mixture {mixture.id}: last stage step {stages[-1][0]} >= num_train_steps "
            f"{num_train_steps} after alignment; bump num_train_steps or rethink stages"
        )
    return stages


# Mixtures for ``run_mix_sweep``: every variant at 100M, ~4.3B tokens.
# m6 ratios are proportional to per-cell sizes:
#   H ≈ 1.68M / 5.39M = 0.3117 → 0.31
#   M ≈ 1.42M / 5.39M = 0.2634 → 0.26
#   L ≈ 2.29M / 5.39M = 0.4249 → 0.43
MIX_MIXTURES: tuple[Mixture, ...] = (
    Mixture(id="m1", static={"H": 1.0}),
    Mixture(id="m2", static={"M": 1.0}),
    Mixture(id="m3", static={"L": 1.0}),
    Mixture(id="m4", static={"H": 0.80, "M": 0.10, "L": 0.10}),
    Mixture(id="m5", static={"H": 0.60, "M": 0.30, "L": 0.10}),
    Mixture(id="m6", static={"H": 0.31, "M": 0.26, "L": 0.43}),
    Mixture(id="m7", staged=((0.0, {"L": 1.0}), (0.5, {"H": 1.0}))),
    Mixture(id="m8", staged=((0.0, {"H": 1.0}), (0.5, {"L": 1.0}))),
    Mixture(
        id="m9",
        staged=(
            (0.0, {"L": 0.5, "M": 0.5}),
            (1.0 / 3.0, {"L": 1.0 / 3.0, "M": 1.0 / 3.0, "H": 1.0 / 3.0}),
            (2.0 / 3.0, {"H": 1.0}),
        ),
    ),
)

# Mixtures for ``run_scale_sweep``: a focused subset re-run at 1.5B scale.
# m10 == m1 (H-only), m11 == m6 (size-proportional blend), m12 == m7
# (L→H staged). m13 is the staged analogue of m11: a three-stage L→M→H
# curriculum whose per-stage fractions match m6/m11 (L=0.43, M=0.26,
# H=0.31), so transitions land at 0.43 and 0.43+0.26=0.69.
# m14 == m2 (M-only), m15 == m3 (L-only) — single-quality scale baselines.
SCALE_MIXTURES: tuple[Mixture, ...] = (
    Mixture(id="m10", static={"H": 1.0}),
    Mixture(id="m11", static={"H": 0.31, "M": 0.26, "L": 0.43}),
    Mixture(id="m12", staged=((0.0, {"L": 1.0}), (0.5, {"H": 1.0}))),
    Mixture(
        id="m13",
        staged=(
            (0.0, {"L": 1.0}),
            (0.43, {"M": 1.0}),
            (0.69, {"H": 1.0}),
        ),
    ),
    Mixture(id="m14", static={"M": 1.0}),
    Mixture(id="m15", static={"L": 1.0}),
)
MIXTURE_BY_ID: dict[str, Mixture] = {m.id: m for m in (*MIX_MIXTURES, *SCALE_MIXTURES)}

# --- Components --------------------------------------------------------------

# Always present in every mixture's components dict; weight=0 for qualities a
# mixture omits.
TRAIN_CELLS: tuple[Cell, ...] = tuple(Cell(q, "train") for q in QUALITY_CONFIGS)

# Default heldout eval cells: all 6 (H/M/L) x (val/test). Scale narrows this
# via ``StageSpec.heldout_cells``.
HELDOUT_CELLS: tuple[Cell, ...] = tuple(Cell(q, s) for q in QUALITY_CONFIGS for s in HF_SPLITS if s != "train")

# Reuse the existing cd-val cache (~440M tokens). Masked matches training
# loss; unmasked is the explicit additional metric per the issue.
CD_VAL_COMPONENT_MASKED = "protein-docs-cd-val"
CD_VAL_COMPONENT_UNMASKED = "protein-docs-cd-val-unmasked"


def _quality_to_train_component_name(quality: str) -> str:
    return Cell(quality, "train").component_name


def _train_weights_for(
    mixture: Mixture,
    num_train_steps: int,
    *,
    batch_size: int,
    heldout_cells: tuple[Cell, ...],
    include_cd_val_unmasked: bool,
) -> dict[str, float] | list[tuple[int, dict[str, float]]]:
    """Resolve mixture per-quality weights to per-component weights.

    Always lists all 3 train components (zero for unused qualities) AND every
    heldout-eval component at weight 0. ``MixtureDataset`` drops zero-weight
    datasets from sampling but still loads them for eval.
    """
    eval_zeros = {cell.component_name: 0.0 for cell in heldout_cells}
    eval_zeros[CD_VAL_COMPONENT_MASKED] = 0.0
    if include_cd_val_unmasked:
        eval_zeros[CD_VAL_COMPONENT_UNMASKED] = 0.0

    def _expand(qw: dict[str, float]) -> dict[str, float]:
        train_part = {_quality_to_train_component_name(q): float(qw.get(q, 0.0)) for q in QUALITY_CONFIGS}
        return {**train_part, **eval_zeros}

    weights = _resolve_mixture_weights(mixture, num_train_steps, batch_size=batch_size)
    if isinstance(weights, dict):
        return _expand(weights)
    return [(step, _expand(w)) for step, w in weights]


def _cd_val_cache_path() -> str:
    """On-disk cd-val cache path (``override_output_path`` on the source step)."""
    return protein_docs_val_tokenized.override_output_path


def _empty_source_component(cache_dir: str, *, masked: bool) -> DatasetComponent:
    """Cache-only component: empty url lists short-circuit Levanter's cache-build.

    Loads ``<cache_dir>/{train,validation}/`` if present; loss masking applied
    iff ``masked``.
    """
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


def _components(heldout_cells: tuple[Cell, ...], *, include_cd_val_unmasked: bool) -> dict[str, DatasetComponent]:
    """All train + eval components for one stage.

    3 train cells (loss-masked) feed sampling + IID-carve eval. Heldout
    evals: ``heldout_cells`` + cd-val masked, plus cd-val unmasked when
    requested.
    """
    components: dict[str, DatasetComponent] = {}
    for cell in TRAIN_CELLS:
        components[cell.component_name] = _empty_source_component(cell_cache_path(cell), masked=True)
    for cell in heldout_cells:
        components[cell.component_name] = _empty_source_component(cell_cache_path(cell), masked=True)
    cd_val_cache = _cd_val_cache_path()
    components[CD_VAL_COMPONENT_MASKED] = _empty_source_component(cd_val_cache, masked=True)
    if include_cd_val_unmasked:
        components[CD_VAL_COMPONENT_UNMASKED] = _empty_source_component(cd_val_cache, masked=False)
    return components


def _has_nonzero_weight(train_weights: dict[str, float] | list[tuple[int, dict[str, float]]], name: str) -> bool:
    """Mirror of Levanter's ``LmDataConfig._has_nonzero_weight``.

    ``build_caches("train")`` skips components that are zero in every stage;
    if ``num_validation_sequences`` names one of those, the IID-carve step
    KeyErrors. Keep this in lockstep with the upstream check.
    """
    if isinstance(train_weights, dict):
        return train_weights.get(name, 0) > 0
    return any(w.get(name, 0) > 0 for _, w in train_weights)


def build_mixture(
    mixture: Mixture,
    num_train_steps: int,
    *,
    batch_size: int,
    heldout_cells: tuple[Cell, ...] = HELDOUT_CELLS,
    include_cd_val_unmasked: bool = True,
) -> LmDataConfig:
    """Build the LmDataConfig for one mixture at a given (step, batch) shape.

    ``num_train_steps`` + ``batch_size`` resolve staged-mixture transitions
    into Levanter's ``(step, weights)`` schedule and align them to block edges.
    Defaults match the smoke/mix recipe; scale narrows the heldout set.
    """
    components = _components(heldout_cells, include_cd_val_unmasked=include_cd_val_unmasked)
    train_weights = _train_weights_for(
        mixture,
        num_train_steps,
        batch_size=batch_size,
        heldout_cells=heldout_cells,
        include_cd_val_unmasked=include_cd_val_unmasked,
    )
    # IID-carve only from active train cells: listing a zero-weight cell here
    # KeyErrors against Levanter's build_caches skip (m1/m2/m3 single-cell,
    # m7/m8 omit M, etc.).
    num_validation_sequences = {
        cell.component_name: IID_EVAL_SEQS_PER_TRAIN
        for cell in TRAIN_CELLS
        if _has_nonzero_weight(train_weights, cell.component_name)
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


# --- Helpers ----------------------------------------------------------------


def _fmt_count(n: int) -> str:
    """Compact precise count, e.g. 120.7M, 1.007B."""
    if n >= 1_000_000_000:
        return f"{n / 1e9:.3f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)


def _fmt_lr(lr: float) -> str:
    """Format LR for run names, e.g. ``9.3e-4``."""
    mantissa, exponent = f"{lr:.1e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def _schedule(target_tokens: int, num_evals: int, batch_size: int) -> tuple[int, int]:
    """Return ``(total_steps, steps_per_eval)`` for a token target.

    Rounds so ``total_steps`` is exactly ``num_evals * steps_per_eval``.
    """
    tokens_per_step = batch_size * SEQ_LEN
    spe = max(1, round(target_tokens / num_evals / tokens_per_step))
    return spe * num_evals, spe


# --- Selection / preview ----------------------------------------------------


def _selected_runs() -> tuple[str, ...]:
    """CSV of substrings from ``RUNS``; empty = all."""
    raw = os.environ.get("RUNS", "")
    return tuple(s.strip() for s in raw.split(",") if s.strip())


def _preview() -> bool:
    return os.environ.get("PREVIEW", "").strip().lower() in {"yes", "true", "1"}


def _format_weights(weights: dict[str, float]) -> str:
    return ", ".join(f"{k}={v:.3g}" for k, v in weights.items())


def _describe_target(spec: "StageSpec", target: SweepTarget) -> str:
    """Multi-line target description for preview: schedule, mixture, stages."""
    (mixture_id,) = target.config
    quality_weights = _resolve_mixture_weights(
        MIXTURE_BY_ID[mixture_id], spec.num_train_steps, batch_size=spec.batch_size
    )
    tokens_str = _fmt_count(spec.batch_size * SEQ_LEN * spec.num_train_steps)
    lines = [
        f"  {target.target_id}",
        f"    model={spec.model_tag} batch={spec.batch_size} steps={spec.num_train_steps} "
        f"steps_per_eval={spec.steps_per_eval} lr={spec.learning_rate:.4g} tokens={tokens_str} "
        f"lr_schedule={LR_SCHEDULE} decay={LR_DECAY}",
    ]
    if isinstance(quality_weights, dict):
        lines.append(f"    mixture (static): {_format_weights(quality_weights)}")
    else:
        lines.append(f"    mixture (staged, {len(quality_weights)} stages):")
        for step, w in quality_weights:
            lines.append(f"      step {step:>6} (seq {step * spec.batch_size:>10}): {_format_weights(w)}")
    return "\n".join(lines)


def _print_preview(spec: "StageSpec", targets: list[SweepTarget]) -> None:
    print(f"PREVIEW: {spec.name} would run {len(targets)} target(s):", flush=True)
    for t in targets:
        print(_describe_target(spec, t), flush=True)
        print(flush=True)


def _print_tokenize_preview(suffixes: list[str]) -> None:
    print(f"PREVIEW: tokenize would run {len(suffixes)} target(s):", flush=True)
    for s in suffixes:
        print(f"  {s}", flush=True)


# ============================================================================
# Sweep stage specs. Add/edit stages in :func:`_make_stage_specs`; the
# dispatch table and launcher pick them up automatically. Bump ``version``
# to fork run names + sweep-root lock dir.
# ============================================================================
SWEEP_ROOT_PREFIX = "gs://marin-us-east5/sweeps/prot-exp11-data-mix"
RUN_NAME_PREFIX = "prot-exp11-dm"


@dataclass(frozen=True)
class StageSpec:
    """All knobs that vary across sweep stages."""

    name: str  # "run_smoke" | "run_mix_sweep" | "run_scale_sweep" (the COMMAND value)
    label: str  # short tag baked into target_id ("smoke"/"mix"/"scale")
    model_tag: str  # short model tag baked into target_id ("100m"/"1_5b")
    model_config: object  # LlamaConfig for this stage's model
    resources_fn: Callable[[], ResourceConfig]
    mixture_ids: tuple[str, ...]  # which mixtures from MIXTURE_BY_ID this stage runs
    batch_size: int
    num_train_steps: int
    steps_per_eval: int
    learning_rate: float
    version: str  # bump to fork run names + sweep-root when recipe changes
    num_workers: int  # default Fray-worker count for this stage's launcher
    # Defaults match the smoke/mix recipe; override per-stage to narrow.
    heldout_cells: tuple[Cell, ...] = HELDOUT_CELLS
    include_cd_val_unmasked: bool = True
    steps_per_export: int | None = None  # None = keep no permanent intermediates


def _trial_name(spec: StageSpec, mixture_id: str) -> str:
    tokens_str = _fmt_count(spec.batch_size * SEQ_LEN * spec.num_train_steps)
    return (
        f"{RUN_NAME_PREFIX}-{spec.label}-{spec.model_tag}-{tokens_str}-"
        f"{mixture_id}-lr{_fmt_lr(spec.learning_rate)}-{spec.version}"
    )


def _stage_targets(spec: StageSpec) -> list[SweepTarget]:
    return [SweepTarget(target_id=_trial_name(spec, mid), config=(mid,)) for mid in spec.mixture_ids]


def _build_trial(spec: StageSpec, mixture_id: str) -> tuple[str, object]:
    """Build one trial's ``(job_name, raw_config)`` for ``prepare_lm_train``."""
    data = build_mixture(
        MIXTURE_BY_ID[mixture_id],
        spec.num_train_steps,
        batch_size=spec.batch_size,
        heldout_cells=spec.heldout_cells,
        include_cd_val_unmasked=spec.include_cd_val_unmasked,
    )
    train_config = SimpleTrainConfig(
        resources=spec.resources_fn(),
        train_batch_size=spec.batch_size,
        num_train_steps=spec.num_train_steps,
        learning_rate=versioned(spec.learning_rate),
        weight_decay=WEIGHT_DECAY,
        beta2=BETA2,
        warmup=WARMUP,
        decay=LR_DECAY,
        lr_schedule=LR_SCHEDULE,
        train_seq_len=SEQ_LEN,
        steps_per_eval=spec.steps_per_eval,
        steps_per_export=spec.steps_per_export,
        max_eval_batches=MAX_EVAL_BATCHES,
        data_seed=DATA_SEED,
    )
    params = compute_num_parameters(spec.model_config, PROTEIN_VOCAB_SIZE)
    tokens = spec.batch_size * SEQ_LEN * spec.num_train_steps
    job_name, raw_config = prepare_lm_train(
        name=_trial_name(spec, mixture_id),
        tokenized=data,
        model_config=spec.model_config,
        train_config=train_config,
        tags=[
            "protein",
            "exp11",
            "data-mix",
            spec.label,
            "llama",
            spec.model_tag,
            mixture_id,
            f"params={_fmt_count(params)}",
            f"params_exact={params}",
            f"tokens={_fmt_count(tokens)}",
            f"tokens_exact={tokens}",
            f"steps={spec.num_train_steps}",
        ],
        eval_harness_tasks=[],
        use_default_validation=False,
        wandb_group="exp11-data-mix",
    )
    # Override defaults.py's 10-min cadence; see TEMP_CHECKPOINT_INTERVAL.
    raw_config = dataclasses.replace(
        raw_config,
        trainer=dataclasses.replace(
            raw_config.trainer,
            checkpointer=dataclasses.replace(
                raw_config.trainer.checkpointer,
                save_interval=TEMP_CHECKPOINT_INTERVAL,
            ),
        ),
    )
    return job_name, raw_config


def _make_stage_specs() -> dict[str, StageSpec]:
    # Full cd-train size = 43_301_511_168 packed tokens; scale target is 5x
    # the mix schedule (~half-epoch coverage), not full cd-train.
    smoke_steps, smoke_spe = _schedule(400_000_000, num_evals=2, batch_size=BATCH_SIZE)
    mix_steps, mix_spe = _schedule(4_300_000_000, num_evals=8, batch_size=BATCH_SIZE)
    scale_steps, scale_spe = _schedule(5 * 4_300_000_000, num_evals=32, batch_size=BATCH_SIZE)
    return {
        # m9 exercises MixtureDataset's weight_stages path (alignment + rescaling).
        "run_smoke": StageSpec(
            name="run_smoke",
            label="smoke",
            model_tag="100m",
            model_config=protein_llama_100m,
            resources_fn=_resources,
            mixture_ids=("m9",),
            batch_size=BATCH_SIZE,
            num_train_steps=smoke_steps,
            steps_per_eval=smoke_spe,
            learning_rate=scaled_lr(BATCH_SIZE, HIDDEN_100M),
            version="v6",
            num_workers=1,
        ),
        "run_mix_sweep": StageSpec(
            name="run_mix_sweep",
            label="mix",
            model_tag="100m",
            model_config=protein_llama_100m,
            resources_fn=_resources,
            mixture_ids=tuple(m.id for m in MIX_MIXTURES),
            batch_size=BATCH_SIZE,
            num_train_steps=mix_steps,
            steps_per_eval=mix_spe,
            learning_rate=scaled_lr(BATCH_SIZE, HIDDEN_100M),
            version="v3",
            # 9 trials, one v5p-8 per worker; 9 workers ⇒ one trial each.
            num_workers=9,
        ),
        "run_scale_sweep": StageSpec(
            name="run_scale_sweep",
            label="scale",
            model_tag="1_5b",
            model_config=protein_llama_1_5b,
            resources_fn=_resources,
            mixture_ids=tuple(m.id for m in SCALE_MIXTURES),
            batch_size=BATCH_SIZE,
            num_train_steps=scale_steps,
            steps_per_eval=scale_spe,
            learning_rate=scaled_lr(BATCH_SIZE, HIDDEN_1_5B),
            version="v6",
            # 6 trials, one v5p-8 per worker → 1 trial each.
            num_workers=6,
            heldout_cells=(Cell("H", "val"), Cell("H", "test")),
            include_cd_val_unmasked=False,
            steps_per_export=scale_steps // NUM_PERMANENT_CHECKPOINTS,
        ),
    }


STAGE_SPECS: dict[str, StageSpec] = _make_stage_specs()
SWEEP_STAGES: tuple[str, ...] = tuple(STAGE_SPECS.keys())


# ============================================================================
# Stage dispatch / worker entrypoint
# ============================================================================


def _resolve_targets(spec: StageSpec, runs: tuple[str, ...]) -> list[SweepTarget]:
    """Stage target list after the ``RUNS`` substring filter."""
    targets = _stage_targets(spec)
    if runs:
        targets = [t for t in targets if any(r in t.target_id for r in runs)]
    return targets


def _sweep_root_for(spec: StageSpec) -> str:
    return f"{SWEEP_ROOT_PREFIX}/{spec.name}-{spec.version}"


def _worker_entrypoint(stage: str, rank: int, num_workers: int, runs: tuple[str, ...]) -> None:
    """One worker: take a rank-stride slice of targets and run claim_and_run.

    Re-resolving targets here (vs shipping them as args) keeps JobRequest
    tiny; rank-stride slicing yields disjoint subsets so workers don't fight.
    """
    spec = STAGE_SPECS[stage]
    targets = _resolve_targets(spec, runs)
    my_targets = targets[rank::num_workers]
    logger.info(
        "Worker rank=%d/%d assigned %d/%d target(s): %s",
        rank,
        num_workers,
        len(my_targets),
        len(targets),
        [t.target_id for t in my_targets],
    )
    sweep_root = _sweep_root_for(spec)
    resources = spec.resources_fn()

    def _run_one(target: SweepTarget) -> None:
        (mixture_id,) = target.config
        name, raw_config = _build_trial(spec, mixture_id)
        _run_training_on_worker(name=name, raw_config=raw_config, override_output_path=None, resources=resources)

    claim_and_run(sweep_root, my_targets, _run_one)


# ============================================================================
# Launcher
# ============================================================================


def _tokenize_launcher() -> None:
    runs = _selected_runs()
    cells = [c for c in ALL_CELLS if not runs or any(r in c.suffix for r in runs)]
    if not cells:
        raise ValueError(f"tokenize: no cells matched RUNS={runs!r}")
    if _preview():
        _print_tokenize_preview([c.suffix for c in cells])
        return
    _tokenize_main(cells)


def _sweep_launcher(stage: str) -> None:
    spec = STAGE_SPECS[stage]
    runs = _selected_runs()
    targets = _resolve_targets(spec, runs)
    if not targets:
        raise ValueError(f"{stage}: no targets matched RUNS={runs!r}")

    if _preview():
        _print_preview(spec, targets)
        return

    num_workers = int(os.environ.get("NUM_WORKERS", str(spec.num_workers)))
    num_workers = min(num_workers, len(targets))
    resources = spec.resources_fn()
    env = resolve_training_env(base_env=None, resources=resources)
    extras = extras_for_resources(resources)

    logger.info(
        "Stage=%s targets=%d workers=%d runs=%s resources=%s",
        stage,
        len(targets),
        num_workers,
        runs,
        resources,
    )

    client = current_client()
    handles = []
    for rank in range(num_workers):
        request = JobRequest(
            name=f"{RUN_NAME_PREFIX}-{stage}-w{rank}",
            entrypoint=Entrypoint.from_callable(_worker_entrypoint, args=[stage, rank, num_workers, runs]),
            resources=resources,
            environment=create_environment(env_vars=env, extras=extras),
        )
        handles.append(client.submit(request))
        logger.info("Submitted worker rank=%d/%d: %s", rank, num_workers, request.name)

    failures = 0
    for rank, h in enumerate(handles):
        try:
            h.wait(raise_on_failure=True)
            logger.info("Worker rank=%d finished", rank)
        except Exception:
            failures += 1
            logger.exception("Worker rank=%d failed", rank)
    if failures:
        raise RuntimeError(f"{failures}/{num_workers} workers failed")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    command = os.environ.get("COMMAND")
    if command == "tokenize":
        _tokenize_launcher()
        return
    if command in SWEEP_STAGES:
        _sweep_launcher(command)
        return
    raise ValueError(f"Set COMMAND to one of: tokenize, {', '.join(SWEEP_STAGES)}. Got {command!r}.")


if __name__ == "__main__":
    main()
