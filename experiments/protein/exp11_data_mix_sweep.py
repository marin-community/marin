# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exp 11: ``cd`` data-mix sweep (Fray-native, no Executor).

Tracks ``Open-Athena/MarinFold#11``. Compares mixtures derived from the
quality-bucketed re-publication ``eczech/marinfold-exp11-protein-docs``
(H=round0, M=round1, L=round2..4), with the same loss-mask recipe as
``protein_train_common.distance_masked_components()``.

Hyperparameters trace back to ``train_protein_1_5b_distance_masked.py``
(the source of truth): batch=128, seq=8192, lr=3.5e-4, weight_decay=0.01,
warmup=0.1, ``optimizer`` defaults. For other (model, batch) pairs the LR
is recomputed via ``scaled_lr(b, h) = LR_CONSTANT * sqrt(b) / h`` where
``LR_CONSTANT`` is derived from the source recipe.

Subcommands (``COMMAND`` env var):

* ``tokenize`` — one Fray job per (quality, split) cell that materializes
  the tokenized cache at a deterministic path.
* ``run_smoke`` — single trial: mixture m9 (staged), 100M model, ~200M
  tokens. m9 is chosen so the smoke exercises the staged-mixture path
  (transition-step alignment + MixtureDataset weight_stages).
* ``run_mix_sweep`` — all 9 mixtures at 100M scale, ~4.3B tokens each.
* ``run_scale_sweep`` — m1, m4, m6 at 1.5B scale on v5p-32 (one job per
  mixture), ~43B tokens each.

Env vars: ``COMMAND`` (required), ``RUNS`` (CSV substring filter on target
ids), ``PREVIEW=yes`` (list targets, submit nothing), ``NUM_WORKERS``
(default depends on stage), ``TPU`` (override the worker TPU type — only
``run_smoke`` / ``run_mix_sweep`` honor this; the scale sweep is pinned to
v5p-32 because LR is calibrated to batch=512).

Smoke usage::

    uv run iris --cluster=marin job run --user $USERNAME --no-wait \\
        --job-name prot-exp11-smoke-$(date +%Y%m%d-%H%M) \\
        --region us-east5 --memory=3.5GB \\
        -e HF_TOKEN "$HF_TOKEN" -e WANDB_API_KEY "$WANDB_API_KEY" \\
        -e COMMAND run_smoke \\
        -- python -m experiments.protein.exp11_data_mix_sweep

Preview without submitting::

    COMMAND=run_mix_sweep PREVIEW=yes \\
        uv run python -m experiments.protein.exp11_data_mix_sweep
"""

import logging
import math
import os
from collections.abc import Callable
from dataclasses import dataclass

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

# Quality-bucketed re-publication of contacts-and-distances-v1-5x, pinned by
# sha so a future re-shard of the HF repo doesn't silently change tokenize
# inputs. Cache path versioning (``DATA_VERSION``) is independent.
HF_DATASET_ID = "eczech/marinfold-exp11-protein-docs"
HF_REVISION = "41b2ec71070cb9e8799311cd8f78877e747f6754"
HF_REVISION_SHORT = HF_REVISION[:7]

# Mapping: quality bucket -> HF config name (subdir on HF).
QUALITY_CONFIGS: dict[str, str] = {"H": "high", "M": "medium", "L": "low"}
# Splits on the HF side: train/val/test. HF lays them out as
# ``<config>/train|val|test/*.parquet``.
HF_SPLITS: tuple[str, ...] = ("train", "val", "test")

# --- Cache layout ------------------------------------------------------------

# Bump to fork the tokenize cache root when tokenize semantics change. Pinned
# in path so the sweep can reference the cache directly without an Executor
# step. ``HF_REVISION_SHORT`` is also baked in so a data revision bump
# automatically forks the cache.
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

    Tokenize writes train cells under ``<cache>/train/`` and val/test cells
    under ``<cache>/validation/``. Component lookup in the sweep uses this
    same root.
    """
    return f"{marin_prefix()}/{CACHE_SUBPATH}/{cell.suffix}/"


# --- Tokenize ---------------------------------------------------------------

# Coordinator-side resources are slack — zephyr children do the heavy work.
# Matches eac-plm/tokenize_quality_splits.TOKENIZE_RESOURCES.
TOKENIZE_RESOURCES = ResourceConfig(cpu=2, ram="16G", disk="50G")


def _tokenize_one_cell(cell: Cell) -> None:
    """Fray entrypoint: tokenize one HF cell into its cache path.

    Re-running is a fast no-op on cache hit (marin.tokenize writes a shard
    ledger and short-circuits when present).
    """
    glob = [cell_input_glob(cell)]
    config = TokenizeConfig(
        train_paths=glob if cell.is_train else [],
        validation_paths=[] if cell.is_train else glob,
        cache_path=cell_cache_path(cell),
        tokenizer=PROTEIN_TOKENIZER,
        format=TextLmDatasetFormat(text_key="document"),
        # We name HF splits explicitly (train/val/test); the cell's role
        # (train vs validation cache) is encoded above.
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

# Re-use the existing shapes from this repo so any future arch change to those
# scripts propagates to this sweep without manual sync.
SEQ_LEN = 8192

# Hidden widths for the LR-scaling formula.
HIDDEN_100M: int = protein_llama_100m.hidden_dim  # 768
HIDDEN_1_5B: int = protein_llama_1_5b.hidden_dim  # 2048

# --- Optimizer / LR scaling --------------------------------------------------

# Source-of-truth recipe from ``train_protein_1_5b_distance_masked.py``:
LR_REF: float = 3.5e-4
LR_REF_BATCH: int = 128
LR_REF_HIDDEN: int = HIDDEN_1_5B  # 2048

# Solve ``lr = lr_constant * sqrt(batch) / hidden`` for ``lr_constant`` using
# the 1.5B source recipe; reuse to scale LR at other (batch, hidden) pairs.
LR_CONSTANT: float = LR_REF * LR_REF_HIDDEN / math.sqrt(LR_REF_BATCH)


def scaled_lr(batch_size: int, hidden_size: int) -> float:
    """LR per the source recipe's sqrt-batch / inverse-hidden scaling."""
    return LR_CONSTANT * math.sqrt(batch_size) / hidden_size


# Per source recipe; unchanged across stages.
WEIGHT_DECAY: float = 0.01
WARMUP: float = 0.1

# LR schedule (uniform across stages). Linear decay to ``DECAY * peak`` over
# the post-warmup window. Deliberately overrides Levanter's cosine default so
# the smoke/mix/scale runs are directly comparable.
LR_SCHEDULE: str = "linear"
DECAY: float = 0.2

# Vocab size for the legacy ``timodonnell/protein-docs-tokenizer@83f597d88e9b``
# revision (pinned in PROTEIN_TOKENIZER). Hardcoded so it lands in run-config
# hashes / wandb tags; bump if the tokenizer pin changes. Matches the value
# used in eac-plm's quality-splits sweep, but for the legacy 2840-vocab.
PROTEIN_VOCAB_SIZE: int = 2840

# Sequences carved off each train cell as an IID held-out eval. Matches the
# eac-plm sweep default; ~4096 sequences ≈ 33.5M tokens / cell / eval pass.
IID_EVAL_SEQS_PER_TRAIN: int = 4096

# Per-component cap. At batch=128, seq=8192 this is ~33.6M tokens / cell / eval.
MAX_EVAL_BATCHES: int = 32

# Pinned across stages so cross-run comparisons share the same data permutation.
DATA_SEED: int = 1729

# Per-component shuffle: full Feistel permutation across the whole cell.
# Pick: full-Feistel over Levanter's hierarchical-block default because (a) we
# want clean cross-mixture comparisons without window-boundary correlations
# inside a single cell, and (b) per-cell sizes (≤5.28M packed sequences) are
# well within Feistel PRP's cheap-evaluation regime. Matches the eac-plm
# quality-splits sweep precedent.
SHUFFLE: bool = True
PERMUTATION_TYPE: str = "feistel"

# MixtureDataset's per-block proportional guarantee. Pinned (Levanter default)
# so stage-boundary alignment math below stays predictable. block_size must be
# a multiple of train_batch_size for staged mixtures to land on a valid
# boundary; helper ``_resolve_mixture_weights`` enforces this.
MIXTURE_BLOCK_SIZE: int = 2048

# --- Resources --------------------------------------------------------------

# us-east5-a keeps TPUs co-located with the ``marin-us-east5`` checkpoint
# bucket (matches ``PROTEIN_RESOURCES_USE5``). Override the 100M TPU with
# the ``TPU`` env var.
PROTEIN_ZONE = "us-east5-a"

# Default for the smoke + mix stages; scale stage is pinned (see below).
_DEFAULT_100M_TPU = "v5p-8"


def _tpu_100m() -> str:
    return os.environ.get("TPU") or _DEFAULT_100M_TPU


def _resources_100m() -> ResourceConfig:
    return ResourceConfig.with_tpu(_tpu_100m(), zone=PROTEIN_ZONE)


def _resources_1_5b() -> ResourceConfig:
    # Pinned — LR for the scale sweep is calibrated to batch=512, which in
    # turn assumes v5p-32 utilization. Don't honor TPU env var here.
    return ResourceConfig.with_tpu("v5p-32", zone=PROTEIN_ZONE)


# --- Mixtures ---------------------------------------------------------------


@dataclass(frozen=True)
class Mixture:
    """One named train-data mixture over (H, M, L) train cells.

    Either ``static`` (one flat dict) or ``staged`` (a list of
    ``(stage_frac, weights)`` entries). Stage fractions are resolved to
    actual training-step indices by :func:`_resolve_mixture_weights`, which
    also snaps each boundary to a multiple of
    ``MIXTURE_BLOCK_SIZE // batch_size`` so it lands on a valid
    ``MixtureDataset`` block boundary.
    """

    id: str
    static: dict[str, float] | None = None
    staged: tuple[tuple[float, dict[str, float]], ...] | None = None


def _resolve_mixture_weights(
    mixture: Mixture, num_train_steps: int, *, batch_size: int
) -> dict[str, float] | list[tuple[int, dict[str, float]]]:
    """Resolve a mixture to MixtureDataset's expected weights form.

    For staged mixtures, snaps each stage's ``(frac * num_train_steps)`` step
    DOWN to a multiple of ``alignment = MIXTURE_BLOCK_SIZE // batch_size`` so
    ``step * batch_size`` (the seq-index passed to MixtureDataset) lands on
    a block boundary. First stage is pinned to step 0. Monotonicity is
    re-asserted after snapping in case two stages snap together.
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


# m6 ratios are proportional to per-cell sizes:
#   H ≈ 1.68M / 5.39M = 0.3117 → 0.31
#   M ≈ 1.42M / 5.39M = 0.2634 → 0.26
#   L ≈ 2.29M / 5.39M = 0.4249 → 0.43
ALL_MIXTURES: tuple[Mixture, ...] = (
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
MIXTURE_BY_ID: dict[str, Mixture] = {m.id: m for m in ALL_MIXTURES}

# --- Components --------------------------------------------------------------

# Subset of training cells (the three train splits). Always present in every
# mixture's components dict; weight=0 for any cell a mixture doesn't draw on.
TRAIN_CELLS: tuple[Cell, ...] = tuple(Cell(q, "train") for q in QUALITY_CONFIGS)
HELDOUT_CELLS: tuple[Cell, ...] = tuple(Cell(q, s) for q in QUALITY_CONFIGS for s in HF_SPLITS if s != "train")


def _quality_to_train_component_name(quality: str) -> str:
    return Cell(quality, "train").component_name


def _train_weights_for(
    mixture: Mixture, num_train_steps: int, *, batch_size: int
) -> dict[str, float] | list[tuple[int, dict[str, float]]]:
    """Resolve mixture per-quality weights to per-component weights.

    Always includes all three train components (zero-weight for any quality a
    mixture doesn't draw on) AND every heldout-eval component at weight 0.
    Levanter's ``MixtureDataset`` drops zero-weight datasets from the sampling
    pool but still loads them for eval.
    """
    eval_zeros = {cell.component_name: 0.0 for cell in HELDOUT_CELLS}
    eval_zeros[CD_VAL_COMPONENT_MASKED] = 0.0
    eval_zeros[CD_VAL_COMPONENT_UNMASKED] = 0.0

    def _expand(qw: dict[str, float]) -> dict[str, float]:
        # Every train component listed, zero if absent from the mixture.
        train_part = {_quality_to_train_component_name(q): float(qw.get(q, 0.0)) for q in QUALITY_CONFIGS}
        return {**train_part, **eval_zeros}

    weights = _resolve_mixture_weights(mixture, num_train_steps, batch_size=batch_size)
    if isinstance(weights, dict):
        return _expand(weights)
    return [(step, _expand(w)) for step, w in weights]


# --- cd-val (existing legacy cache, both masked and unmasked) ----------------

# Reference the existing protein-docs-cd-val tokenized cache from
# protein_train_common so we don't re-tokenize ~440M tokens. The issue
# requires evaluation on the combined cd validation split both masked
# (consistent with training loss) and unmasked (the explicit exception in
# the issue): "Eval on unmasked contacts-and-distances-v1-5x/validation too".
CD_VAL_COMPONENT_MASKED = "protein-docs-cd-val"
CD_VAL_COMPONENT_UNMASKED = "protein-docs-cd-val-unmasked"


def _cd_val_cache_path() -> str:
    """Resolve the existing cd-val cache via the protein_train_common step."""
    # ``override_output_path`` is set on this step, so the field is the
    # final on-disk path.
    return protein_docs_val_tokenized.override_output_path


def _empty_source_component(cache_dir: str, *, masked: bool) -> DatasetComponent:
    """Cache-only component: ``UrlDatasetSourceConfig`` with empty url lists.

    The empty-url source short-circuits Levanter's cache-build path: it just
    loads ``<cache_dir>/{train,validation}/`` if present, otherwise the split
    is skipped (matches the eac-plm pattern). Loss masking is applied iff
    ``masked``.
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


def _components() -> dict[str, DatasetComponent]:
    """All train + eval components, identical across mixtures.

    Train cells: 3 (loss-masked). Per-quality heldout: 6 (loss-masked).
    Combined cd-val: 2 (one masked, one unmasked).
    """
    components: dict[str, DatasetComponent] = {}
    for cell in TRAIN_CELLS:
        components[cell.component_name] = _empty_source_component(cell_cache_path(cell), masked=True)
    for cell in HELDOUT_CELLS:
        components[cell.component_name] = _empty_source_component(cell_cache_path(cell), masked=True)
    cd_val_cache = _cd_val_cache_path()
    components[CD_VAL_COMPONENT_MASKED] = _empty_source_component(cd_val_cache, masked=True)
    components[CD_VAL_COMPONENT_UNMASKED] = _empty_source_component(cd_val_cache, masked=False)
    return components


def _has_nonzero_weight(train_weights: dict[str, float] | list[tuple[int, dict[str, float]]], name: str) -> bool:
    """Mirror of Levanter's ``LmDataConfig._has_nonzero_weight``.

    Levanter's ``build_caches("train")`` skips components that have zero
    weight in every stage; if ``num_validation_sequences`` lists one of
    those names, the IID-carve step KeyErrors. Keep this in lockstep with
    the upstream check at ``lib/levanter/src/levanter/data/text/datasets.py``.
    """
    if isinstance(train_weights, dict):
        return train_weights.get(name, 0) > 0
    return any(w.get(name, 0) > 0 for _, w in train_weights)


def build_mixture(mixture: Mixture, num_train_steps: int, *, batch_size: int) -> LmDataConfig:
    """Build the LmDataConfig for one mixture at a given (step, batch) shape.

    ``num_train_steps`` + ``batch_size`` resolve staged mixtures' transition
    points into Levanter's ``(step, weights)`` schedule and align each step
    to a MixtureDataset block boundary.
    """
    components = _components()
    train_weights = _train_weights_for(mixture, num_train_steps, batch_size=batch_size)
    # IID-carve only from train cells with nonzero weight in this mixture.
    # Levanter's build_caches("train") skips components that are zero in every
    # stage, so listing a skipped cell here KeyErrors in the IID-carve step of
    # _validation_datasets_unwrapped. Matters for single-cell mixtures (m1/m2/m3)
    # and staged mixtures that omit a quality (m7/m8 never use M).
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
        f"lr_schedule={LR_SCHEDULE} decay={DECAY}",
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
# Sweep stage specs
# ============================================================================
# Each stage = one StageSpec consumed by ``_trial_name`` / ``_build_trial`` /
# ``_stage_targets``. Add or change a stage by editing :func:`_make_stage_specs`
# below; the dispatch table (`STAGE_SPECS`) and the worker / launcher pick it
# up automatically. Bump ``version`` to fork run names (and the sweep-root
# lock dir) when a stage's recipe changes but its identity doesn't.
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
    data = build_mixture(MIXTURE_BY_ID[mixture_id], spec.num_train_steps, batch_size=spec.batch_size)
    train_config = SimpleTrainConfig(
        resources=spec.resources_fn(),
        train_batch_size=spec.batch_size,
        num_train_steps=spec.num_train_steps,
        learning_rate=versioned(spec.learning_rate),
        weight_decay=WEIGHT_DECAY,
        warmup=WARMUP,
        decay=DECAY,
        lr_schedule=LR_SCHEDULE,
        train_seq_len=SEQ_LEN,
        steps_per_eval=spec.steps_per_eval,
        max_eval_batches=MAX_EVAL_BATCHES,
        data_seed=DATA_SEED,
        env_vars={"WANDB_ENTITY": "timodonnell"},
    )
    params = compute_num_parameters(spec.model_config, PROTEIN_VOCAB_SIZE)
    tokens = spec.batch_size * SEQ_LEN * spec.num_train_steps
    return prepare_lm_train(
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


def _make_stage_specs() -> dict[str, StageSpec]:
    # Issue EDA: cd train = 43,301,511,168 packed tokens (taken at face value;
    # see the user-confirmed "skip the 43B verification" instruction).
    smoke_steps, smoke_spe = _schedule(200_000_000, num_evals=2, batch_size=128)
    mix_steps, mix_spe = _schedule(4_300_000_000, num_evals=8, batch_size=128)
    scale_steps, scale_spe = _schedule(43_301_511_168, num_evals=8, batch_size=512)
    return {
        # m9 chosen over a static mixture so smoke also exercises
        # MixtureDataset's weight_stages path (alignment + rescaling).
        "run_smoke": StageSpec(
            name="run_smoke",
            label="smoke",
            model_tag="100m",
            model_config=protein_llama_100m,
            resources_fn=_resources_100m,
            mixture_ids=("m9",),
            batch_size=128,
            num_train_steps=smoke_steps,
            steps_per_eval=smoke_spe,
            learning_rate=scaled_lr(128, HIDDEN_100M),
            version="v5",
            num_workers=1,
        ),
        "run_mix_sweep": StageSpec(
            name="run_mix_sweep",
            label="mix",
            model_tag="100m",
            model_config=protein_llama_100m,
            resources_fn=_resources_100m,
            mixture_ids=tuple(m.id for m in ALL_MIXTURES),
            batch_size=128,
            num_train_steps=mix_steps,
            steps_per_eval=mix_spe,
            learning_rate=scaled_lr(128, HIDDEN_100M),
            version="v2",
            # 9 trials, one v5p-8 per worker; 9 workers ⇒ one trial each.
            num_workers=9,
        ),
        "run_scale_sweep": StageSpec(
            name="run_scale_sweep",
            label="scale",
            model_tag="1_5b",
            model_config=protein_llama_1_5b,
            resources_fn=_resources_1_5b,
            mixture_ids=("m1", "m4", "m6"),
            batch_size=512,
            num_train_steps=scale_steps,
            steps_per_eval=scale_spe,
            learning_rate=scaled_lr(512, HIDDEN_1_5B),
            version="v1",
            # 3 trials, one v5p-32 per worker → 1 trial each.
            num_workers=3,
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
    """One worker: re-resolve targets, take its rank-stride slice, run claim_and_run.

    Re-resolving on the worker (instead of shipping the full list as args)
    keeps the JobRequest payload tiny and round-robins disjoint subsets so
    workers don't fight for the same lock.
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
