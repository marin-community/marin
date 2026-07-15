# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exp 117: contacts-v1 1.5B LR / weight-decay / epochs tuning, one point per launch.

Single-point trainer for the contacts-v1 1.5B tuning sweep of MarinFold #117
(https://github.com/Open-Athena/MarinFold/issues/117), the multi-epoch extension of #75.
This module trains **exactly one explicit ``(epochs, lr, wd, tpu, region)`` point per
invocation** and nothing else: no grid, no ladder, no scheduling. The search over
``(epochs, lr, wd)`` is owned by the ``design-adaptive-sweep`` / ``run-adaptive-sweep``
skills, which drive this script one point at a time. ``SMOKE=yes`` overrides the point
with a tiny, identity-isolated end-to-end run (see below) for validating a launch on a
target slice/region; it never resumes or overwrites a real sweep point.

The recipe (Qwen3 1.5B, ``seq_len=8192``, global batch 128, AdamW + cosine with 10%
warmup, unmasked loss, pack-prefix-only, full Feistel shuffle at ``seed=0``, the contacts-v1
documents tokenized in-pipeline) mirrors #75 / MarinFold #70. Where #117 and #75 differ,
**#117 wins** -- notably one permanent checkpoint every epoch (in addition to the 10-minute
rolling resumption checkpoint).

Objective being optimized: **final-step ``eval/tokenized/contacts-v1-val/loss``** (read from W&B).

Interface for the adaptive-sweep skills (identity-vs-execution split):
  * ``TPU`` selects the slice and drives per-device batch sizing without touching run identity:
    global batch stays 128 on every slice, so a same-region slice change resumes the same run.
  * ``REGION`` sets the storage prefix (cache + checkpoint locality) and the run id. A region
    change starts a fresh run; a same-region re-dispatch resumes. Documents are tokenized
    in-pipeline into a region-local cache (raw docs are staged in every eligible region).

Preview a point (builds nothing, submits nothing)::

    EPOCHS=8 LR=1e-2 WD=0.1 TPU=v5p-8 REGION=us-east5 PREVIEW=yes \\
        uv run python -m experiments.protein.exp117_sweep

Launch one run (secrets come from the iris command, never baked into the config)::

    source ~/marin.env && uv run iris --cluster marin job run \\
        --user "$USERNAME" --no-wait --region us-east5 --memory=1GB \\
        -e WANDB_API_KEY "$WANDB_API_KEY" \\
        -e HUGGING_FACE_HUB_TOKEN "$HF_TOKEN" \\
        -e WANDB_ENTITY "$WANDB_ENTITY" -e WANDB_PROJECT "$WANDB_PROJECT" \\
        -e EPOCHS 8 -e LR 1e-2 -e WD 0.1 -e TPU v5p-8 -e REGION us-east5 \\
        -- python -m experiments.protein.exp117_sweep

Smoke test one slice/region (a real but tiny end-to-end run: ~20 steps, one eval, one
permanent checkpoint, under an isolated ``exp117-smoke`` identity that never touches a real
point). EPOCHS/LR/WD are optional in smoke mode but honored if given; ``SMOKE_STEPS``
overrides the step budget. ``CORRECTION_FACTOR`` overrides the batch-fit correction factor (else the
slice's calibrated per-family default) -- a smaller value packs a larger per-device batch. In smoke
mode the slice and effective correction factor are folded into the run identity, so each
``(slice, region, correction_factor)`` is a distinct W&B run::

    source ~/marin.env && uv run iris --cluster marin job run \\
        --user "$USERNAME" --no-wait --region us-east5 --memory=1GB \\
        -e WANDB_API_KEY "$WANDB_API_KEY" \\
        -e HUGGING_FACE_HUB_TOKEN "$HF_TOKEN" \\
        -e WANDB_ENTITY "$WANDB_ENTITY" -e WANDB_PROJECT "$WANDB_PROJECT" \\
        -e SMOKE yes -e TPU v6e-4 -e REGION us-east5 -e SMOKE_BATCH 64 \\
        -- python -m experiments.protein.exp117_sweep
"""

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TypeVar

from fray.cluster import ResourceConfig
from fray.types import tpu_family
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim.config import AdamConfig
from marin.execution.lazy import ArtifactStep, lower
from marin.execution.step_runner import StepRunner
from marin.experiment.data import tokenized
from marin.experiment.train import train_lm
from marin.processing.tokenize.tokenize import TokenizedCache
from marin.rl.placement import marin_prefix_for_region, singleton_region_list
from marin.training.training import LevanterCheckpoint

from experiments.coral.batch_calibration import (
    adam_optimizer_bytes,
    batch_memory_bytes,
    dense_transformer_bytes,
    tpu_batch_config,
)

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


# --- Identity ----------------------------------------------------------------

# Calendar versions (marin's ArtifactStep requires calendar ``YYYY.MM.DD[.N]`` or a mutable
# ``dev``). Split so the training campaign and the tokenized cache version independently:
#   * SWEEP_VERSION keys the run / checkpoint identity (``{run_id}/{SWEEP_VERSION}/``).
#   * CACHE_VERSION keys the tokenize cache path (``tokenized/contacts-v1{,-val}/{CACHE_VERSION}/``).
# Bump SWEEP_VERSION to fork a fresh training campaign while reusing the existing cache; bump
# CACHE_VERSION to rebuild the cache. The train handle depends on the cache handles, so if
# SWEEP_VERSION is a (fixed) calendar version the cache versions must be too -- a fixed handle
# cannot depend on a mutable ``dev`` dep (marin ``_lower``).
SWEEP_VERSION: str = "2026.07.13.1"
CACHE_VERSION: str = "2026.07.13.1"
RUN_PREFIX: str = "prot-exp117"
WANDB_GROUP: str = "exp117-contacts-v1-tune"

# Smoke mode: a throwaway end-to-end validation run under an isolated identity (distinct prefix,
# W&B group, "smoke" tag, tiny cadence) so it never resumes or shares a run with a real point.
SMOKE_RUN_PREFIX: str = "prot-exp117-smoke"
SMOKE_WANDB_GROUP: str = "exp117-smoke"
# Manual smoke identity token (not dated). Folded into the smoke run id so a re-run after a code
# or recipe change gets a FRESH run + checkpoint path instead of resuming/merging the prior smoke
# run of the same (point, region, tpu, correction_factor). Bump to fork a clean smoke run; the old
# run and its checkpoints are left in place untouched.
SMOKE_VERSION: str = "v3"  # v3: overhead_factor -> per-family correction_factor (identity token oh->cf)
SMOKE_STEPS_DEFAULT: int = 20
SMOKE_NUM_EVALS: int = 2  # evals (and permanent checkpoints) spread across the smoke run
# Nominal point used when EPOCHS/LR/WD are omitted in smoke mode; overridden by any that are
# set. The values are immaterial to what smoke validates (the pipeline mechanics), but must
# be valid so the optimizer builds.
SMOKE_POINT_DEFAULTS: tuple[int, float, float] = (1, 1e-3, 0.1)


# --- Data (contacts-v1 documents, tokenized in-pipeline) ---------------------

# contacts-v1 tokenizer (2845 vocab), pinned to the immutable revision the corpus is
# tokenized with.
TOKENIZER: str = "timodonnell/contacts-v1-tokenizer@5d68a24a899f"
VOCAB_SIZE: int = 2845
TEXT_KEY: str = "document"

# Raw contacts-v1 documents (MarinFold exp53), region-relative so they resolve region-local.
_DOCS_BASE: str = "protein-structure/MarinFold/exp53_contacts_v1_5x/documents"
TRAIN_DOCS: str = f"{_DOCS_BASE}/train/*.parquet"
VAL_DOCS: str = f"{_DOCS_BASE}/val/*.parquet"

# Tokenized-cache handle names. Each name is BOTH the storage subpath under the region prefix
# (-> ``gs://marin-<region>/{name}/{CACHE_VERSION}/``) AND the mixture component key / eval
# series prefix (``eval/<name>/loss``). The ``tokenized/`` prefix nests the caches under the
# conventional ``tokenized/`` location instead of the bucket root; the val name is the swept
# objective's series.
COMPONENT_TRAIN: str = "tokenized/contacts-v1"
COMPONENT_VAL: str = "tokenized/contacts-v1-val"

# Exact train-corpus token count for the contacts-v1 train split at TOKENIZER. Steps/epoch
# are derived from this, not estimated. Matches the #70 cache; re-derive if the tokenizer
# revision or corpus changes.
TRAIN_TOKENS: int = 4_676_753_425


# --- Model (Qwen3 1.47B; #75 / exp44 dims + Llama3 rope) ---------------------

MODEL_CONFIG = Qwen3Config(
    max_seq_len=8192,
    hidden_dim=2048,
    intermediate_dim=8192,
    num_heads=32,
    num_kv_heads=8,
    num_layers=24,
    rope=Llama3RotaryEmbeddingsConfig(),
)
MODEL_SIZE: str = "1_5b"  # size label for the run id and tags; matches MODEL_CONFIG


# --- Fixed training recipe (mirrors #70/#75; only LR/WD/epochs vary per launch) ---

BATCH_SIZE: int = 128
SEQ_LEN: int = 8192
WARMUP: float = 0.1  # 10% warmup
LR_SCHEDULE: str = "cosine"  # AdamW + cosine decay
NUM_EVALS_PER_EPOCH: int = 2

TOKENS_PER_STEP: int = BATCH_SIZE * SEQ_LEN
STEPS_PER_EPOCH: int = round(TRAIN_TOKENS / TOKENS_PER_STEP)
STEPS_PER_EVAL: int = max(1, round(STEPS_PER_EPOCH / NUM_EVALS_PER_EPOCH))

# Per-family correction factor for the batch-fit estimate, calibrated against measured per-chip
# ceilings (see exp117_batch_calibration.md): each value sits inside its family's measured range with
# margin. batch_fit uses the slice's family value by default; a launch may override via CORRECTION_FACTOR.
CORRECTION_FACTORS: dict[str, float] = {"v5e": 0.5, "v6e": 0.3, "v5p": 0.45}


# --- A single trial point ----------------------------------------------------


@dataclass(frozen=True)
class Point:
    """One trial: explicit epoch count, peak LR, and weight decay."""

    epochs: int
    learning_rate: float
    weight_decay: float

    @property
    def num_train_steps(self) -> int:
        return self.epochs * STEPS_PER_EPOCH

    @property
    def point_id(self) -> str:
        return f"e{self.epochs}-lr{_fmt_lr(self.learning_rate)}-wd{_fmt_wd(self.weight_decay)}"


def run_id(point: Point, region: str, prefix: str = RUN_PREFIX) -> str:
    """The W&B run name, trainer/resume id, and run-output subpath for a ``(point, region)``.

    Keyed on the point and the region -- never the TPU -- so a region change is a fresh run
    while any compatible-slice re-dispatch in the same region resumes the same run. ``prefix``
    selects the identity namespace (``SMOKE_RUN_PREFIX`` isolates smoke runs from real ones).
    Run outputs (checkpoints, W&B mirror) land at ``gs://marin-<region>/{run_id}/{SWEEP_VERSION}/``.
    """
    return f"{prefix}-cv1-{MODEL_SIZE}-{point.point_id}-{region}"


# --- Env inputs (one point per launch) ---------------------------------------


def _env_value(name: str, cast: Callable[[str], _T], default: _T | None) -> _T:
    """Read env var ``name`` through ``cast``; fall back to ``default`` (``None`` == required)."""
    raw = os.environ.get(name)
    if raw is not None:
        return cast(raw)
    if default is None:
        raise SystemExit(f"missing required env var '{name}'; set EPOCHS, LR, WD, TPU, REGION")
    return default


def parse_point(defaults: Point | None = None) -> Point:
    """Read the ``(epochs, lr, wd)`` point from the environment.

    A full sweep launch passes ``defaults=None`` and requires every variable. Smoke mode
    passes a fallback point so EPOCHS/LR/WD are optional (any that are set still win).
    """
    epochs = _env_value("EPOCHS", int, defaults.epochs if defaults else None)
    learning_rate = _env_value("LR", float, defaults.learning_rate if defaults else None)
    weight_decay = _env_value("WD", float, defaults.weight_decay if defaults else None)
    if epochs < 1:
        raise SystemExit(f"EPOCHS must be >= 1, got {epochs}")
    if learning_rate <= 0:
        raise SystemExit(f"LR must be > 0, got {learning_rate}")
    if weight_decay < 0:
        raise SystemExit(f"WD must be >= 0, got {weight_decay}")
    return Point(epochs=epochs, learning_rate=learning_rate, weight_decay=weight_decay)


def parse_tpu() -> str:
    tpu = os.environ.get("TPU")
    if not tpu:
        raise SystemExit("missing required env var TPU (e.g. v5p-8, v6e-16)")
    return tpu


def parse_region() -> str:
    region = os.environ.get("REGION")
    if not region:
        raise SystemExit("missing required env var REGION (e.g. us-east5)")
    return region.lower()


def preview() -> bool:
    return os.environ.get("PREVIEW", "").strip().lower() in {"yes", "true", "1"}


def smoke() -> bool:
    return os.environ.get("SMOKE", "").strip().lower() in {"yes", "true", "1"}


def smoke_steps() -> int:
    steps = _env_value("SMOKE_STEPS", int, SMOKE_STEPS_DEFAULT)
    if steps < 1:
        raise SystemExit(f"SMOKE_STEPS must be >= 1, got {steps}")
    return steps


def parse_correction_factor() -> float | None:
    """Optional per-launch override of the batch-fit correction factor via the ``CORRECTION_FACTOR`` env.

    ``None`` (env unset) means use the slice's calibrated per-family default. A smaller value shrinks
    the estimate, admitting a larger per-device batch (less gradient accumulation).
    """
    raw = os.environ.get("CORRECTION_FACTOR")
    if raw is None or not raw.strip():
        return None
    factor = float(raw)
    if factor < 0:
        raise SystemExit(f"CORRECTION_FACTOR must be >= 0, got {factor}")
    return factor


def correction_factor_for(tpu: str, override: float | None) -> float:
    """Effective correction factor: the per-launch ``override`` if set, else the slice's family default."""
    if override is not None:
        return override
    family = tpu_family(tpu)
    if family not in CORRECTION_FACTORS:
        raise SystemExit(f"no calibrated correction_factor for TPU family {family!r}; set CORRECTION_FACTOR")
    return CORRECTION_FACTORS[family]


def parse_smoke_batch() -> int | None:
    """Global-batch override for the direct max-batch probe (``SMOKE_BATCH``); smoke mode only.

    When set, the run trains at exactly this global batch with ``per_device_parallelism=-1``
    (full per-chip batch, NO gradient accumulation), bypassing the HBM estimate and correction factor
    entirely. The job then either fits (succeeds) or OOMs, so a binary search over powers of 2 finds
    each slice's true batch ceiling -- the empirical ground truth the estimate is calibrated against.
    Folded into the run identity so every probe is a distinct W&B run.
    """
    raw = os.environ.get("SMOKE_BATCH")
    if raw is None or not raw.strip():
        return None
    batch = int(raw)
    if batch < 1:
        raise SystemExit(f"SMOKE_BATCH must be >= 1, got {batch}")
    return batch


# --- Helpers -----------------------------------------------------------------


def _fmt_lr(lr: float) -> str:
    """Compact, path-safe LR tag to ~4 sig figs, e.g. ``1e-2``, ``3p162e-3``."""
    mantissa, exponent = f"{lr:.3e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".").replace(".", "p")
    return f"{mantissa}e{int(exponent)}"


def _fmt_wd(wd: float) -> str:
    """Path-safe weight-decay tag, e.g. ``0.1`` -> ``0p1``, ``1.6`` -> ``1p6``."""
    return f"{wd:g}".replace(".", "p")


def _fmt_correction(correction_factor: float) -> str:
    """Path-safe correction-factor tag, e.g. ``0.3`` -> ``0p3``, ``1.0`` -> ``1``."""
    return f"{correction_factor:g}".replace(".", "p")


def _batch_bytes(correction_factor: float) -> int:
    """Estimated HBM to place the full global batch (params + Adam state + activations)."""
    params = MODEL_CONFIG.total_trainable_params(VOCAB_SIZE)
    param_bytes, activation_bytes = dense_transformer_bytes(
        parameter_count=params,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        hidden_dim=MODEL_CONFIG.hidden_dim,
        intermediate_dim=MODEL_CONFIG.intermediate_dim,
        num_layers=MODEL_CONFIG.num_layers,
    )
    return batch_memory_bytes(
        param_bytes=param_bytes,
        optimizer_bytes=adam_optimizer_bytes(params),
        activation_bytes=activation_bytes,
        correction_factor=correction_factor,
    )


def batch_fit(tpu: str, correction_factor: float | None) -> tuple[int, int]:
    """``(per_device_parallelism, gradient_accumulation)`` that fits global batch 128 on ``tpu``.

    ``per_device_parallelism == -1`` means the full per-chip batch fits with no accumulation. The
    global batch is always 128, so the objective is comparable across every slice. ``correction_factor``
    is the per-launch override, or ``None`` to use the slice's calibrated per-family default.
    """
    return tpu_batch_config(tpu, BATCH_SIZE, _batch_bytes(correction_factor_for(tpu, correction_factor)))


def _tokenize_cache(name: str, docs: str, *, validation: bool) -> ArtifactStep[TokenizedCache]:
    """Tokenize a split of the raw documents region-local into a levanter cache handle.

    ``name`` is the cache storage subpath and the mixture component key (see COMPONENT_TRAIN/VAL).
    ``pack`` is applied later on the assembled data config (:func:`_apply_recipe_overrides`).
    """
    return tokenized(
        name=name,
        tokenizer=TOKENIZER,
        version=CACHE_VERSION,
        paths=[docs],
        text_key=TEXT_KEY,
        validation=validation,
        tags=["protein", "contacts-v1", name],
    )


def _training_env(region: str) -> dict[str, str]:
    """W&B metadata for the dispatched job. No ``MARIN_PREFIX``: the worker resolves
    ``marin_prefix()`` from its own (pinned) region, so caches and checkpoints land region-local.
    Secrets come from the iris command, not the config.
    """
    env = {"WANDB_PROJECT": os.environ.get("WANDB_PROJECT", "marin")}
    for key in ("WANDB_ENTITY", "WANDB_MODE"):
        if value := os.environ.get(key):
            env[key] = value
    return env


def _tags(point: Point, region: str) -> list[str]:
    # Stable, identity-bearing facts only. TPU / per-device parallelism are deliberately
    # omitted: a run may migrate slices over its life, so they are neither stable tags nor
    # part of run identity -- keeping them out also leaves the fingerprint TPU-independent.
    params = MODEL_CONFIG.total_trainable_params(VOCAB_SIZE)
    tokens = TOKENS_PER_STEP * point.num_train_steps
    return [
        "protein",
        "exp117",
        "contacts-v1",
        "qwen3",
        "unmasked",
        f"model_size={MODEL_SIZE}",
        f"global_batch={BATCH_SIZE}",
        f"params={params}",
        f"epochs={point.epochs}",
        f"lr={point.learning_rate:g}",
        f"wd={point.weight_decay:g}",
        f"region={region}",
        f"steps={point.num_train_steps}",
        f"tokens={tokens}",
        f"sweep_version={SWEEP_VERSION}",
        f"cache_version={CACHE_VERSION}",
    ]


# --- Trial construction ------------------------------------------------------


@dataclass(frozen=True)
class RunShape:
    """The mode-dependent shape of one dispatched run: identity plus step/eval/ckpt cadence.

    Separates the two run modes -- a full sweep point vs. a smoke check -- without a boolean
    flag: the optimizer hyperparameters come from the :class:`Point`, and everything about
    identity and cadence comes from here. :func:`production_shape` and :func:`smoke_shape`
    are the only constructors.
    """

    run_id: str
    wandb_group: str
    num_train_steps: int
    steps_per_eval: int
    checkpoint_every: int  # permanent-checkpoint keep interval (in steps)
    tags: list[str]
    batch_override: int | None = None  # SMOKE_BATCH: force this global batch + per_device_parallelism=-1


def production_shape(point: Point, region: str) -> RunShape:
    """Full sweep point: #117 cadence -- steps from the epoch count, one permanent ckpt/epoch."""
    return RunShape(
        run_id=run_id(point, region),
        wandb_group=WANDB_GROUP,
        num_train_steps=point.num_train_steps,
        steps_per_eval=STEPS_PER_EVAL,
        checkpoint_every=STEPS_PER_EPOCH,
        tags=_tags(point, region),
    )


def smoke_shape(
    point: Point, region: str, steps: int, tpu: str, correction_factor: float | None, batch_override: int | None
) -> RunShape:
    """Tiny, identity-isolated end-to-end run: ``SMOKE_NUM_EVALS`` evals + permanent ckpts.

    Reuses the real caches, model, and TPU batch-fit so it validates the actual launch path, but under
    a smoke-only identity and with an eval/checkpoint every ``steps // SMOKE_NUM_EVALS`` so at least one
    of each fires within the short run. The slice, effective ``correction_factor``, and ``SMOKE_VERSION``
    are folded into the run id (and tagged), and ``SMOKE_VERSION`` forks a fresh identity after a
    recipe/library change so a re-run never resumes/merges a prior smoke run rather than colliding.
    """
    every = max(1, steps // SMOKE_NUM_EVALS)
    base = run_id(point, region, SMOKE_RUN_PREFIX)
    cf = correction_factor_for(tpu, correction_factor)
    identity = f"{base}-{tpu}-cf{_fmt_correction(cf)}-{SMOKE_VERSION}"
    tags = [
        *_tags(point, region),
        "smoke",
        "batch-calib",
        f"tpu={tpu}",
        f"correction_factor={cf:g}",
        f"smoke_version={SMOKE_VERSION}",
    ]
    if batch_override is not None:
        identity = f"{identity}-b{batch_override}"
        tags.append(f"smoke_batch={batch_override}")
    return RunShape(
        run_id=identity,
        wandb_group=SMOKE_WANDB_GROUP,
        num_train_steps=steps,
        steps_per_eval=every,
        checkpoint_every=every,
        tags=tags,
        batch_override=batch_override,
    )


def _apply_recipe_overrides(
    step: ArtifactStep[LevanterCheckpoint],
    tpu: str,
    checkpoint_every: int,
    correction_factor: float | None,
    force_full_batch: bool,
) -> ArtifactStep[LevanterCheckpoint]:
    """Apply the #117 knobs :func:`train_lm` does not expose.

    Identity-bearing (always applied, so they enter the fingerprint): a permanent checkpoint
    every ``checkpoint_every`` steps alongside the 10-minute rolling resumption checkpoint,
    pack-prefix-only components (documents are never concat-and-split), and full-eval with no
    downsampling (``max_eval_batches=None``, per #117). Execution-only (run time
    only, so run identity stays TPU-independent): fit the global batch to the actual TPU via
    per-device parallelism.
    """
    base_build_config = step.build_config

    def build_config(ctx):
        pod = base_build_config(ctx)
        trainer = replace(
            pod.train_config.trainer,
            # #117 requires the full held-out val split with NO downsampling. This is levanter's
            # default (None = evaluate every batch), but pin it explicitly so the guarantee is
            # part of the run identity and can't be silently capped by an upstream default change.
            max_eval_batches=None,
            checkpointer=replace(
                pod.train_config.trainer.checkpointer,
                # levanter's CheckpointerConfig.keep is a list of dicts (it builds
                # CheckpointInterval(**k) itself); a single interval needs no ``until``.
                keep=[{"every": checkpoint_every}],
            ),
        )
        data = pod.train_config.data
        data = replace(data, components={key: replace(c, pack=True) for key, c in data.components.items()})
        if not ctx.is_fingerprint:
            # force_full_batch (SMOKE_BATCH probe): place the whole per-chip batch with no
            # accumulation and skip the estimate, so the run directly tests whether this global
            # batch fits the slice's HBM.
            per_device_parallelism = -1 if force_full_batch else batch_fit(tpu, correction_factor)[0]
            eval_parallelism = (
                trainer.per_device_eval_parallelism if per_device_parallelism == -1 else per_device_parallelism
            )
            trainer = replace(
                trainer,
                per_device_parallelism=per_device_parallelism,
                per_device_eval_parallelism=eval_parallelism,
            )
        train_config = replace(pod.train_config, trainer=trainer, data=data)
        return replace(pod, train_config=train_config)

    return replace(step, build_config=build_config)


def build_run(
    point: Point, shape: RunShape, tpu: str, region: str, correction_factor: float | None
) -> ArtifactStep[LevanterCheckpoint]:
    """Assemble one training run as an ``ArtifactStep``.

    ``point`` supplies only the optimizer hyperparameters; ``shape`` supplies identity and
    step/eval/checkpoint cadence (production vs. smoke). ``tpu`` and ``correction_factor`` drive the
    per-device batch fit at run time; ``region`` scopes storage and cache locality.
    """
    name = shape.run_id
    batch_size = shape.batch_override or BATCH_SIZE
    train_cache = _tokenize_cache(COMPONENT_TRAIN, TRAIN_DOCS, validation=False)
    val_cache = _tokenize_cache(COMPONENT_VAL, VAL_DOCS, validation=True)
    step = train_lm(
        name=name,
        model=MODEL_CONFIG,
        optimizer=AdamConfig(
            learning_rate=point.learning_rate,
            weight_decay=point.weight_decay,
            warmup=WARMUP,
            lr_schedule=LR_SCHEDULE,
        ),
        datasets={train_cache: 1.0},
        validation=[val_cache],
        batch_size=batch_size,
        seq_len=SEQ_LEN,
        num_train_steps=shape.num_train_steps,
        z_loss_weight=None,
        evals=None,  # no lm-eval-harness; the objective is the weight-0 val-loss eval
        resources=ResourceConfig.with_tpu(tpu, regions=singleton_region_list(region)),
        version=SWEEP_VERSION,
        steps_per_eval=shape.steps_per_eval,
        wandb_project=os.environ.get("WANDB_PROJECT", "marin"),
        wandb_group=shape.wandb_group,
        run_id=name,
        tags=shape.tags,
        env_vars=_training_env(region),
    )
    return _apply_recipe_overrides(
        step, tpu, shape.checkpoint_every, correction_factor, force_full_batch=shape.batch_override is not None
    )


# --- Preview + entry point ---------------------------------------------------


def _print_preview(
    point: Point, shape: RunShape, tpu: str, region: str, mode: str, correction_factor: float | None
) -> None:
    if shape.batch_override is not None:
        per_device_parallelism, grad_accum, batch = -1, 1, shape.batch_override
    else:
        per_device_parallelism, grad_accum = batch_fit(tpu, correction_factor)
        batch = BATCH_SIZE
    params = MODEL_CONFIG.total_trainable_params(VOCAB_SIZE)
    tokens = batch * SEQ_LEN * shape.num_train_steps
    print(
        f"PREVIEW exp117 [{mode}] -- no submit\n"
        f"  run_id={shape.run_id} wandb_group={shape.wandb_group}\n"
        f"  epochs={point.epochs} lr={point.learning_rate:g} wd={point.weight_decay:g}\n"
        f"  steps={shape.num_train_steps} (steps/eval={shape.steps_per_eval}, "
        f"permanent ckpt every {shape.checkpoint_every} steps)\n"
        f"  tokens={tokens / 1e9:.3f}B params={params / 1e9:.3f}B schedule={LR_SCHEDULE} warmup={WARMUP}\n"
        f"  tpu={tpu} region={region} prefix={marin_prefix_for_region(region)}\n"
        f"  correction_factor={correction_factor_for(tpu, correction_factor):g} "
        f"per_device_parallelism={per_device_parallelism} grad_accum={grad_accum} (global batch {batch})\n"
        f"  objective=eval/{COMPONENT_VAL}/loss (final step, minimize)",
        flush=True,
    )


def _resolve_run(region: str, tpu: str, correction_factor: float | None) -> tuple[Point, RunShape, str]:
    """Resolve the ``(point, shape, mode)`` for this invocation from the environment."""
    if smoke():
        point = parse_point(defaults=Point(*SMOKE_POINT_DEFAULTS))
        shape = smoke_shape(point, region, smoke_steps(), tpu, correction_factor, parse_smoke_batch())
        return point, shape, "smoke"
    point = parse_point()
    return point, production_shape(point, region), "sweep"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    tpu = parse_tpu()
    region = parse_region()
    correction_factor = parse_correction_factor()

    # No MARIN_PREFIX: driver and worker each resolve marin_prefix() from their own region (both
    # pinned via --region / ResourceConfig), so caches and outputs land region-local.
    point, shape, mode = _resolve_run(region, tpu, correction_factor)

    if preview():
        _print_preview(point, shape, tpu, region, mode, correction_factor)
        return

    StepRunner().run([lower(build_run(point, shape, tpu, region, correction_factor))])


if __name__ == "__main__":
    main()
