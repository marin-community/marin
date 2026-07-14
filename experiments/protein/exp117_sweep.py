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

Objective being optimized: **final-step ``eval/contacts-v1-val/loss``** (read from W&B).

Interface for the adaptive-sweep skills (identity-vs-execution split):
  * ``TPU`` selects the slice and drives per-device batch sizing
    (:func:`~experiments.coral.batch_config.tpu_batch_config`) *without* touching run
    identity: the global batch stays 128 on every slice (so the objective is comparable),
    and a same-region change to any compatible slice resumes the same run.
  * ``REGION`` sets the storage prefix (checkpoint locality) and the W&B / trainer run id.
    A region change starts a fresh run under a fresh regional identity; a same-region
    re-dispatch resumes the rolling checkpoint. The contacts-v1 documents are tokenized
    in-pipeline into a region-local cache; the raw docs currently live only in us-east5, so the
    run must be dispatched there.

Preview a point (builds nothing, submits nothing)::

    EPOCHS=8 LR=1e-2 WD=0.1 TPU=v5p-8 REGION=us-east5 PREVIEW=yes \\
        uv run python -m experiments.protein.exp117_sweep

Launch one run (secrets come from the iris command, never baked into the config)::

    source ~/marin.env && uv run iris --cluster marin job run \\
        --user "$USERNAME" --no-wait --region us-east5 --memory=1GB \\
        -e WANDB_API_KEY "$WANDB_API_KEY" \\
        -e HUGGING_FACE_HUB_TOKEN "$HUGGING_FACE_HUB_TOKEN" \\
        -e WANDB_ENTITY "$WANDB_ENTITY" -e WANDB_PROJECT "$WANDB_PROJECT" \\
        -e EPOCHS 8 -e LR 1e-2 -e WD 0.1 -e TPU v5p-8 -e REGION us-east5 \\
        -- python -m experiments.protein.exp117_sweep

Smoke test one slice/region (a real but tiny end-to-end run: ~20 steps, one eval, one
permanent checkpoint, under an isolated ``exp117-smoke`` identity that never touches a real
point). EPOCHS/LR/WD are optional in smoke mode but honored if given; ``SMOKE_STEPS``
overrides the step budget::

    source ~/marin.env && uv run iris --cluster marin job run \\
        --user "$USERNAME" --no-wait --region us-east5 --memory=1GB \\
        -e WANDB_API_KEY "$WANDB_API_KEY" \\
        -e HUGGING_FACE_HUB_TOKEN "$HUGGING_FACE_HUB_TOKEN" \\
        -e WANDB_ENTITY "$WANDB_ENTITY" -e WANDB_PROJECT "$WANDB_PROJECT" \\
        -e SMOKE yes -e TPU v6e-4 -e REGION us-east5 \\
        -- python -m experiments.protein.exp117_sweep
"""

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TypeVar

from fray.cluster import ResourceConfig
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

from experiments.coral.batch_config import (
    adam_optimizer_bytes,
    batch_memory_bytes,
    dense_transformer_bytes,
    tpu_batch_config,
)

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


# --- Identity ----------------------------------------------------------------

# Fixed calendar version: keeps the run in the shared namespace and makes the run id,
# checkpoint path, and fingerprint stable across invocations so re-runs of a point are
# idempotent and resumable. Bump to fork a fresh campaign over the same recipe.
# .1: tokenize the contacts-v1 documents in-pipeline (canonical dataset pattern), no adopt/
# mirror:// and no lib hacks. (This is the same in-pipeline tokenize recipe as the earlier .1
# campaign, so its already-built tokenize cache is reused; .0/.2-.4 explored other data paths.)
VERSION: str = "2026.07.13.1"
RUN_PREFIX: str = "prot-exp117"
WANDB_GROUP: str = "exp117-contacts-v1-tune"

# Smoke mode: a throwaway end-to-end validation run under a separate identity, so it can
# never resume, overwrite, or share a run/checkpoint with a real sweep point. Distinct run
# prefix + W&B group + a "smoke" tag, plus tiny step/eval/checkpoint cadence, keep it fully
# isolated and cheap while still exercising data -> train -> eval -> checkpoint on the slice.
SMOKE_RUN_PREFIX: str = "prot-exp117-smoke"
SMOKE_WANDB_GROUP: str = "exp117-smoke"
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

# Raw contacts-v1 documents (MarinFold exp53, the corpus #70's exp67 caches were built from),
# tokenized in-pipeline into a region-local levanter cache (built once, fingerprint-cached,
# reused) -- the canonical dataset pattern, no ``mirror://``/adopt. Bucket-relative so the raw
# docs resolve region-local; they currently live only in us-east5.
_DOCS_BASE: str = "protein-structure/MarinFold/exp53_contacts_v1_5x/documents"
TRAIN_DOCS: str = f"{_DOCS_BASE}/train/*.parquet"
VAL_DOCS: str = f"{_DOCS_BASE}/val/*.parquet"

# Mixture component keys -> the prefix of each ``eval/<component>/loss`` W&B series. The
# val key is the swept objective.
COMPONENT_TRAIN: str = "contacts-v1"
COMPONENT_VAL: str = "contacts-v1-val"

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


# --- Fixed training recipe (mirrors #70/#75; only LR/WD/epochs vary per launch) ---

BATCH_SIZE: int = 128
SEQ_LEN: int = 8192
WARMUP: float = 0.1  # 10% warmup
LR_SCHEDULE: str = "cosine"  # AdamW + cosine decay
NUM_EVALS_PER_EPOCH: int = 2

TOKENS_PER_STEP: int = BATCH_SIZE * SEQ_LEN
STEPS_PER_EPOCH: int = round(TRAIN_TOKENS / TOKENS_PER_STEP)
STEPS_PER_EVAL: int = max(1, round(STEPS_PER_EPOCH / NUM_EVALS_PER_EPOCH))

# HBM overhead multiplier for the batch-fit estimate (experiments.coral.batch_config).
# 1.0 matches the coral calibration-study default and is conservative for this model: it
# predicts at least as much microbatching as #75's hand-verified per-slice configs, so it
# fits. Calibrate against the coral batch-config study if throughput headroom matters.
HBM_OVERHEAD_FACTOR: float = 1.0


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
    """The W&B run name, trainer/resume id, and checkpoint subpath for a ``(point, region)``.

    Keyed on the point and the region -- never the TPU -- so a region change is a fresh run
    while any compatible-slice re-dispatch in the same region resumes the same run. ``prefix``
    selects the identity namespace (``SMOKE_RUN_PREFIX`` isolates smoke runs from real ones).
    """
    return f"{prefix}-cv1-1_5b-{point.point_id}-{region}"


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


# --- Helpers -----------------------------------------------------------------


def _fmt_lr(lr: float) -> str:
    """Compact, path-safe LR tag to ~4 sig figs, e.g. ``1e-2``, ``3p162e-3``."""
    mantissa, exponent = f"{lr:.3e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".").replace(".", "p")
    return f"{mantissa}e{int(exponent)}"


def _fmt_wd(wd: float) -> str:
    """Path-safe weight-decay tag, e.g. ``0.1`` -> ``0p1``, ``1.6`` -> ``1p6``."""
    return f"{wd:g}".replace(".", "p")


def _batch_bytes() -> int:
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
        overhead_factor=HBM_OVERHEAD_FACTOR,
    )


def batch_fit(tpu: str) -> tuple[int, int]:
    """``(per_device_parallelism, gradient_accumulation)`` that fits global batch 128 on ``tpu``.

    ``per_device_parallelism == -1`` means the full per-chip batch fits with no accumulation.
    The global batch is always 128, so the objective is comparable across every slice.
    """
    return tpu_batch_config(tpu, BATCH_SIZE, _batch_bytes())


def _tokenize_cache(name: str, docs: str, *, validation: bool) -> ArtifactStep[TokenizedCache]:
    """Tokenize a split of the contacts-v1 documents in-pipeline into a levanter cache handle.

    ``docs`` is a bucket-relative raw-doc glob (resolved region-local). ``name`` is the mixture
    component key; for the val cache it becomes the ``eval/<name>/loss`` objective series. The
    cache is built once (fingerprint-cached) under ``{name}/{version}``; ``pack`` is applied on
    the assembled data config (:func:`_apply_recipe_overrides`), not here.
    """
    return tokenized(
        name=name,
        tokenizer=TOKENIZER,
        version=VERSION,
        paths=[docs],
        text_key=TEXT_KEY,
        validation=validation,
        tags=["protein", "contacts-v1", name],
    )


def _training_env(region: str) -> dict[str, str]:
    """Environment for the dispatched training job: W&B metadata only.

    Deliberately does *not* set ``MARIN_PREFIX``: the worker resolves ``marin_prefix()`` from
    its own region (GCE metadata), and the job is pinned to ``region`` via ``ResourceConfig``,
    so the worker's prefix is the target regional bucket -- where the tokenize output cache and
    checkpoints land region-local. Secrets (WANDB_API_KEY, HUGGING_FACE_HUB_TOKEN) are passed by
    the iris command, not baked into the config.
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
    tokens = TOKENS_PER_STEP * point.num_train_steps
    return [
        "protein",
        "exp117",
        "contacts-v1",
        "1_5b",
        "qwen3",
        "unmasked",
        f"epochs={point.epochs}",
        f"lr={point.learning_rate:g}",
        f"wd={point.weight_decay:g}",
        f"region={region}",
        f"steps={point.num_train_steps}",
        f"tokens_exact={tokens}",
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


def smoke_shape(point: Point, region: str, steps: int) -> RunShape:
    """Tiny, identity-isolated end-to-end run: ``SMOKE_NUM_EVALS`` evals + permanent ckpts.

    Reuses the real caches, model, and TPU batch-fit so it validates the actual launch path,
    but under a smoke-only identity and with an eval/checkpoint every ``steps //
    SMOKE_NUM_EVALS`` so at least one of each fires within the short run.
    """
    every = max(1, steps // SMOKE_NUM_EVALS)
    return RunShape(
        run_id=run_id(point, region, SMOKE_RUN_PREFIX),
        wandb_group=SMOKE_WANDB_GROUP,
        num_train_steps=steps,
        steps_per_eval=every,
        checkpoint_every=every,
        tags=[*_tags(point, region), "smoke"],
    )


def _apply_recipe_overrides(
    step: ArtifactStep[LevanterCheckpoint], tpu: str, checkpoint_every: int
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
            per_device_parallelism, _ = batch_fit(tpu)
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


def build_run(point: Point, shape: RunShape, tpu: str, region: str) -> ArtifactStep[LevanterCheckpoint]:
    """Assemble one training run as an ``ArtifactStep``.

    ``point`` supplies only the optimizer hyperparameters; ``shape`` supplies identity and
    step/eval/checkpoint cadence (production vs. smoke). ``tpu`` drives the per-device batch
    fit at run time; ``region`` scopes storage and cache locality.
    """
    name = shape.run_id
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
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        num_train_steps=shape.num_train_steps,
        z_loss_weight=None,
        evals=None,  # no lm-eval-harness; the objective is the weight-0 val-loss eval
        resources=ResourceConfig.with_tpu(tpu, regions=singleton_region_list(region)),
        version=VERSION,
        steps_per_eval=shape.steps_per_eval,
        wandb_project=os.environ.get("WANDB_PROJECT", "marin"),
        wandb_group=shape.wandb_group,
        run_id=name,
        tags=shape.tags,
        env_vars=_training_env(region),
    )
    return _apply_recipe_overrides(step, tpu, shape.checkpoint_every)


# --- Preview + entry point ---------------------------------------------------


def _print_preview(point: Point, shape: RunShape, tpu: str, region: str, mode: str) -> None:
    per_device_parallelism, grad_accum = batch_fit(tpu)
    params = MODEL_CONFIG.total_trainable_params(VOCAB_SIZE)
    tokens = TOKENS_PER_STEP * shape.num_train_steps
    print(
        f"PREVIEW exp117 [{mode}] -- no submit\n"
        f"  run_id={shape.run_id} wandb_group={shape.wandb_group}\n"
        f"  epochs={point.epochs} lr={point.learning_rate:g} wd={point.weight_decay:g}\n"
        f"  steps={shape.num_train_steps} (steps/eval={shape.steps_per_eval}, "
        f"permanent ckpt every {shape.checkpoint_every} steps)\n"
        f"  tokens={tokens / 1e9:.3f}B params={params / 1e9:.3f}B schedule={LR_SCHEDULE} warmup={WARMUP}\n"
        f"  tpu={tpu} region={region} prefix={marin_prefix_for_region(region)}\n"
        f"  per_device_parallelism={per_device_parallelism} grad_accum={grad_accum} (global batch {BATCH_SIZE})\n"
        f"  objective=eval/{COMPONENT_VAL}/loss (final step, minimize)",
        flush=True,
    )


def _resolve_run(region: str) -> tuple[Point, RunShape, str]:
    """Resolve the ``(point, shape, mode)`` for this invocation from the environment."""
    if smoke():
        point = parse_point(defaults=Point(*SMOKE_POINT_DEFAULTS))
        return point, smoke_shape(point, region, smoke_steps()), "smoke"
    point = parse_point()
    return point, production_shape(point, region), "sweep"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    tpu = parse_tpu()
    region = parse_region()

    # No MARIN_PREFIX is set: the driver resolves marin_prefix() from its own region (it is
    # submitted with --region), and the worker resolves it from its (pinned) region, where the
    # mirror:// caches and the checkpoint output land region-local.
    point, shape, mode = _resolve_run(region)

    if preview():
        _print_preview(point, shape, tpu, region, mode)
        return

    StepRunner().run([lower(build_run(point, shape, tpu, region))])


if __name__ == "__main__":
    main()
