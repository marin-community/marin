# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exp 146: contacts-v1 model-size / optimizer / batch-size / epoch sweep.

Runs exactly one explicit ``(model_size, epochs, lr, wd, batch_size, tpu, region)``
point per invocation. External adaptive-sweep tooling owns point selection and dispatch.
``SMOKE=yes`` launches a tiny identity-isolated run through the same data, model, and
batch-fitting path.

The data recipe and token caches are identical to exp117. The production grid is exactly
MarinFold issue #146: models ``{1.5B, 3B, 6B}``, batch sizes ``{64, 128, 256}``,
learning rates ``{3.1623e-4, 1e-3, 3.1623e-3}``, weight decays
``{0.1, 0.2, 0.4, 0.8, 1.6}``, and epochs ``{2, 4, 8}``. The optimized metric is
final-step ``eval/tokenized/contacts-v1-val/loss``.

Preview a point (builds and submits nothing)::

    MODEL_SIZE=1_5b EPOCHS=8 LR=1e-3 WD=0.1 BATCH_SIZE=128 \
        TPU=v6e-128 REGION=us-east5 PREVIEW=yes \
        uv run python -m experiments.protein.exp146_sweep

Launch one run (secrets come from the iris command, never the config)::

    source ~/marin.env && uv run iris --cluster marin job run \
        --user "$USERNAME" --no-wait --region us-east5 --memory=1GB \
        -e WANDB_API_KEY "$WANDB_API_KEY" \
        -e HUGGING_FACE_HUB_TOKEN "$HF_TOKEN" \
        -e WANDB_ENTITY "$WANDB_ENTITY" -e WANDB_PROJECT "$WANDB_PROJECT" \
        -e MODEL_SIZE 1_5b -e EPOCHS 8 -e LR 1e-3 -e WD 0.1 -e BATCH_SIZE 128 \
        -e TPU v6e-128 -e REGION us-east5 \
        -- python -m experiments.protein.exp146_sweep

Smoke-test one model/slice/region under an isolated identity. Point coordinates are
optional in smoke mode but honored when supplied; ``SMOKE_STEPS`` overrides the step
budget, and ``CORRECTION_FACTOR`` overrides the model-size/TPU-family batch estimate::

    source ~/marin.env && uv run iris --cluster marin job run \
        --user "$USERNAME" --no-wait --region us-east5 --memory=1GB \
        -e WANDB_API_KEY "$WANDB_API_KEY" \
        -e HUGGING_FACE_HUB_TOKEN "$HF_TOKEN" \
        -e WANDB_ENTITY "$WANDB_ENTITY" -e WANDB_PROJECT "$WANDB_PROJECT" \
        -e SMOKE yes -e MODEL_SIZE 1_5b -e TPU v6e-4 -e REGION us-east5 \
        -- python -m experiments.protein.exp146_sweep
"""

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TypeVar

from fray.cluster import ResourceConfig
from fray.types import tpu_family
from levanter.callbacks.watch import WatchConfig
from levanter.data.text.datasets import BlockShuffleConfig
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
    TpuBatchConfig,
    adam_optimizer_bytes,
    batch_memory_bytes,
    dense_transformer_bytes,
    tpu_batch_config,
)

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


# --- Identity ----------------------------------------------------------------

# Sweep and data identities are deliberately independent. Bump SWEEP_SUBVERSION for a
# fresh campaign; change CACHE_VERSION only when the tokenizer or source documents change.
VERSION_DATE: str = "2026.07.23"
SWEEP_SUBVERSION: str = "01"
SWEEP_VERSION: str = f"{VERSION_DATE}.{SWEEP_SUBVERSION}"
# Preserve exp117's cache handles exactly; this date intentionally does not follow VERSION_DATE.
CACHE_VERSION: str = "2026.07.13.1"
DATA_SUBVERSION: str = "1"
RUN_PREFIX: str = "prot-exp146"
WANDB_GROUP: str = "exp146-contacts-v1-model-size-tune"

# Smoke mode: a throwaway end-to-end validation run under an isolated identity (distinct prefix,
# W&B group, "smoke" tag, tiny cadence) so it never resumes or shares a run with a real point.
SMOKE_RUN_PREFIX: str = "prot-exp146-smoke"
SMOKE_WANDB_GROUP: str = "exp146-smoke"
# Manual smoke identity token (not dated). Folded into the smoke run id so a re-run after a code
# or recipe change gets a FRESH run + checkpoint path instead of resuming/merging the prior smoke
# run of the same (point, region, tpu, correction_factor). Bump to fork a clean smoke run; the old
# run and its checkpoints are left in place untouched.
SMOKE_VERSION: str = "v1"
CALIBRATION_VERSION: str = "batchcal-tp1-bs1024-exp117fix-v1"
CALIBRATION_BATCH_SIZE: int = 1024
CALIBRATION_MODEL_SIZES: frozenset[str] = frozenset({"3b", "6b"})
CALIBRATION_TPU_RAM_ENV: str = "CALIBRATION_TPU_RAM"
SMOKE_STEPS_DEFAULT: int = 20
SMOKE_NUM_EVALS: int = 2  # evals (and permanent checkpoints) spread across the smoke run
# Representative point used when EPOCHS/LR/WD/BATCH_SIZE are omitted in smoke mode; any explicit
# values override it.
SMOKE_MODEL_SIZE_DEFAULT: str = "1_5b"
SMOKE_POINT_DEFAULTS: tuple[int, float, float, int] = (1, 1e-3, 0.1, 128)


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


# --- Models (exactly MarinFold issue #146) -----------------------------------


MODEL_CONFIGS: dict[str, Qwen3Config] = {
    "1_5b": Qwen3Config(
        max_seq_len=8192,
        hidden_dim=2048,
        intermediate_dim=8192,
        num_layers=24,
        num_heads=32,
        num_kv_heads=8,
        head_dim=64,
        rope=Llama3RotaryEmbeddingsConfig(),
    ),
    "3b": Qwen3Config(
        max_seq_len=8192,
        hidden_dim=2560,
        intermediate_dim=10240,
        num_layers=30,
        num_heads=48,
        num_kv_heads=16,
        head_dim=64,
        rope=Llama3RotaryEmbeddingsConfig(),
    ),
    "6b": Qwen3Config(
        max_seq_len=8192,
        hidden_dim=3200,
        intermediate_dim=12800,
        num_layers=37,
        num_heads=64,
        num_kv_heads=32,
        head_dim=64,
        rope=Llama3RotaryEmbeddingsConfig(),
    ),
}
MODEL_SIZE_ALIASES: dict[str, str] = {"1.5b": "1_5b", "1-5b": "1_5b", "1500m": "1_5b"}

# Production coordinates from issue #146. Smoke runs deliberately use a tiny point
# outside this grid.
SWEEP_BATCH_SIZES: frozenset[int] = frozenset({64, 128, 256})
SWEEP_LEARNING_RATES: frozenset[float] = frozenset({3.1623e-4, 1e-3, 3.1623e-3})
SWEEP_WEIGHT_DECAYS: frozenset[float] = frozenset({0.1, 0.2, 0.4, 0.8, 1.6})
SWEEP_EPOCHS: frozenset[int] = frozenset({2, 4, 8})

# --- Fixed training recipe (mirrors #70/#75) ---------------------------------

SEQ_LEN: int = 8192
DATA_SEED: int = 0
SHUFFLE = BlockShuffleConfig(io_block_size=256, window_blocks=512, perm_type="feistel")
WARMUP: float = 0.1  # 10% warmup
LR_SCHEDULE: str = "cosine"  # AdamW + cosine decay
NUM_EVALS_PER_EPOCH: int = 2
WANDB_WATCH_CONFIG = WatchConfig(watch_targets=[], interval=0)

# Batch-fit correction factors. The 1.5B row comes from exp117 calibration; the
# 3B row and 6B/v6e were confirmed with representative calibration smoke runs
# that exercised training, checkpointing, and eval. The 6B/v5p value is an
# extrapolated guess from 3B/v5p and 6B/v6e because v5p capacity did not become
# available for a full run. The 6B/v5e (v5litepod) value is not a viable
# calibrated result: 6B still OOMed at TP=1 once per-device batch reached 1, so
# correction-factor tuning alone cannot make that case representative.
CORRECTION_FACTORS: dict[str, dict[str, float]] = {
    "1_5b": {"v5e": 0.50, "v6e": 0.30, "v5p": 0.45},
    "3b": {"v5e": 1.20, "v6e": 0.30, "v5p": 0.30},
    "6b": {"v5e": 0.80, "v6e": 0.70, "v5p": 0.70},
}


# --- A single trial point ----------------------------------------------------


@dataclass(frozen=True)
class Point:
    """One explicit model and optimizer trial."""

    model_size: str
    epochs: int
    learning_rate: float
    weight_decay: float
    batch_size: int

    @property
    def model_config(self) -> Qwen3Config:
        return MODEL_CONFIGS[self.model_size]

    @property
    def tokens_per_step(self) -> int:
        return self.batch_size * SEQ_LEN

    @property
    def steps_per_epoch(self) -> int:
        return round(TRAIN_TOKENS / self.tokens_per_step)

    @property
    def steps_per_eval(self) -> int:
        return max(1, round(self.steps_per_epoch / NUM_EVALS_PER_EPOCH))

    @property
    def num_train_steps(self) -> int:
        return self.epochs * self.steps_per_epoch

    @property
    def point_id(self) -> str:
        return (
            f"{self.model_size}-e{self.epochs}-lr{_fmt_lr(self.learning_rate)}-"
            f"wd{_fmt_wd(self.weight_decay)}-bs{self.batch_size}"
        )


def run_id(point: Point, region: str, prefix: str = RUN_PREFIX) -> str:
    """The W&B run name, trainer/resume id, and run-output subpath for a ``(point, region)``.

    Keyed on the point and the region -- never the TPU -- so a region change is a fresh run
    while any compatible-slice re-dispatch in the same region resumes the same run. ``prefix``
    selects the identity namespace (``SMOKE_RUN_PREFIX`` isolates smoke runs from real ones).
    Run outputs (checkpoints, W&B mirror) land at ``gs://marin-<region>/{run_id}/{SWEEP_VERSION}/``.
    """
    return f"{prefix}-cv1-s{SWEEP_SUBVERSION}-{point.point_id}-{region}"


# --- Env inputs (one point per launch) ---------------------------------------


def _env_value(name: str, cast: Callable[[str], _T], default: _T | None) -> _T:
    """Read env var ``name`` through ``cast``; fall back to ``default`` (``None`` == required)."""
    raw = os.environ.get(name)
    if raw is not None:
        return cast(raw)
    if default is None:
        raise SystemExit(f"missing required env var '{name}'; set MODEL_SIZE, EPOCHS, LR, WD, BATCH_SIZE, TPU, REGION")
    return default


def parse_point(defaults: Point | None = None) -> Point:
    """Read and validate one model/hyperparameter point from the environment."""
    raw_model_size = _env_value("MODEL_SIZE", str, defaults.model_size if defaults else None)
    model_size = raw_model_size.strip().lower()
    model_size = MODEL_SIZE_ALIASES.get(model_size, model_size)
    if model_size not in MODEL_CONFIGS:
        valid = ", ".join(MODEL_CONFIGS)
        raise SystemExit(f"unknown MODEL_SIZE {raw_model_size!r}; choose one of: {valid}")

    epochs = _env_value("EPOCHS", int, defaults.epochs if defaults else None)
    learning_rate = _env_value("LR", float, defaults.learning_rate if defaults else None)
    weight_decay = _env_value("WD", float, defaults.weight_decay if defaults else None)
    batch_size = _env_value("BATCH_SIZE", int, defaults.batch_size if defaults else None)
    if epochs < 1:
        raise SystemExit(f"EPOCHS must be >= 1, got {epochs}")
    if learning_rate <= 0:
        raise SystemExit(f"LR must be > 0, got {learning_rate}")
    if weight_decay < 0:
        raise SystemExit(f"WD must be >= 0, got {weight_decay}")
    if batch_size < 1:
        raise SystemExit(f"BATCH_SIZE must be >= 1, got {batch_size}")
    point = Point(
        model_size=model_size,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
    )
    if defaults is None:
        validate_sweep_point(point)
    return point


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


def validate_sweep_point(point: Point) -> None:
    """Reject production coordinates outside the grid in MarinFold issue #146."""
    invalid = []
    if point.batch_size not in SWEEP_BATCH_SIZES:
        invalid.append(f"BATCH_SIZE={point.batch_size}")
    if point.learning_rate not in SWEEP_LEARNING_RATES:
        invalid.append(f"LR={point.learning_rate:g}")
    if point.weight_decay not in SWEEP_WEIGHT_DECAYS:
        invalid.append(f"WD={point.weight_decay:g}")
    if point.epochs not in SWEEP_EPOCHS:
        invalid.append(f"EPOCHS={point.epochs}")
    if invalid:
        raise SystemExit(f"point is outside the MarinFold issue #146 grid: {', '.join(invalid)}")


def validate_sweep_target(point: Point, tpu: str, correction_factor: float | None) -> None:
    """Validate topology, model placement, and minimum microbatch memory."""
    batch_fit(point, tpu, correction_factor)


def preview() -> bool:
    return os.environ.get("PREVIEW", "").strip().lower() in {"yes", "true", "1"}


def smoke() -> bool:
    return os.environ.get("SMOKE", "").strip().lower() in {"yes", "true", "1"}


def calibration() -> bool:
    return os.environ.get("CALIBRATION", "").strip().lower() in {"yes", "true", "1"}


def calibration_tpu_ram() -> str | None:
    """Optional calibration-only TPU host RAM override, e.g. ``192g``."""
    if not calibration():
        return None
    ram = os.environ.get(CALIBRATION_TPU_RAM_ENV)
    if ram is None:
        return None
    ram = ram.strip()
    if not ram:
        return None
    return ram


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
    if factor <= 0:
        raise SystemExit(f"CORRECTION_FACTOR must be > 0, got {factor}")
    return factor


def correction_factor_for(model_size: str, tpu: str, override: float | None) -> float:
    """Resolve the per-launch override or this model-size/TPU-family estimate."""
    if override is not None:
        return override
    family = tpu_family(tpu)
    factors = CORRECTION_FACTORS[model_size]
    if family not in factors:
        raise SystemExit(
            f"no correction factor for MODEL_SIZE={model_size!r}, TPU family={family!r}; set CORRECTION_FACTOR"
        )
    return factors[family]


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


def _fmt_path_token(value: str) -> str:
    """Path-safe token for short env-driven identity fragments."""
    return value.lower().replace(".", "p").replace("/", "-")


def _batch_bytes(point: Point, correction_factor: float) -> int:
    """Estimate full-global-batch HBM with the model from PR #7380."""
    model = point.model_config
    params = model.total_trainable_params(VOCAB_SIZE)
    parameter_bytes, activation_bytes = dense_transformer_bytes(
        parameter_count=params,
        batch_size=point.batch_size,
        seq_len=SEQ_LEN,
        hidden_dim=model.hidden_dim,
        intermediate_dim=model.intermediate_dim,
        num_layers=model.num_layers,
    )
    return batch_memory_bytes(
        parameter_bytes=parameter_bytes,
        optimizer_bytes=adam_optimizer_bytes(params),
        activation_bytes=activation_bytes,
        correction_factor=correction_factor,
    )


def batch_fit(point: Point, tpu: str, correction_factor: float | None) -> TpuBatchConfig:
    """Derive DP, TP, per-device parallelism, and accumulation using PR #7380."""
    factor = correction_factor_for(point.model_size, tpu, correction_factor)
    return tpu_batch_config(tpu, point.batch_size, _batch_bytes(point, factor))


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


def _tags(point: Point, region: str, *, num_train_steps: int) -> list[str]:
    # Stable, identity-bearing facts only. TPU / per-device parallelism are deliberately
    # omitted: a run may migrate slices over its life, so they are neither stable tags nor
    # part of run identity -- keeping them out also leaves the fingerprint TPU-independent.
    params = point.model_config.total_trainable_params(VOCAB_SIZE)
    tokens = point.batch_size * SEQ_LEN * num_train_steps
    return [
        "protein",
        "exp146",
        "contacts-v1",
        "qwen3",
        f"model_size={point.model_size}",
        f"global_batch={point.batch_size}",
        f"params={params}",
        f"epochs={point.epochs}",
        f"lr={point.learning_rate:g}",
        f"wd={point.weight_decay:g}",
        f"region={region}",
        f"steps={num_train_steps}",
        f"tokens={tokens}",
        f"sweep_subversion={int(SWEEP_SUBVERSION)}",
        f"data_subversion={int(DATA_SUBVERSION)}",
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


def production_shape(point: Point, region: str) -> RunShape:
    """Full point with two evals and one permanent checkpoint per epoch."""
    return RunShape(
        run_id=run_id(point, region),
        wandb_group=WANDB_GROUP,
        num_train_steps=point.num_train_steps,
        steps_per_eval=point.steps_per_eval,
        checkpoint_every=point.steps_per_epoch,
        tags=_tags(point, region, num_train_steps=point.num_train_steps),
    )


def smoke_shape(
    point: Point,
    region: str,
    steps: int,
    tpu: str,
    correction_factor: float | None,
    *,
    calibration_run: bool = False,
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
    cf = correction_factor_for(point.model_size, tpu, correction_factor)
    identity = f"{base}-{tpu}-cf{_fmt_correction(cf)}-{SMOKE_VERSION}"
    tags = [
        *_tags(point, region, num_train_steps=steps),
        "smoke",
        f"tpu={tpu}",
        f"correction_factor={cf:g}",
        f"smoke_version={SMOKE_VERSION}",
    ]
    if calibration_run:
        identity = f"{identity}-{CALIBRATION_VERSION}"
        tags.extend(["batch-calibration", f"calibration_version={CALIBRATION_VERSION}"])
        if ram := calibration_tpu_ram():
            identity = f"{identity}-ram{_fmt_path_token(ram)}"
            tags.append(f"{CALIBRATION_TPU_RAM_ENV}={ram}")
    return RunShape(
        run_id=identity,
        wandb_group=SMOKE_WANDB_GROUP,
        num_train_steps=steps,
        steps_per_eval=every,
        checkpoint_every=every,
        tags=tags,
    )


def _apply_recipe_overrides(
    step: ArtifactStep[LevanterCheckpoint],
    point: Point,
    tpu: str,
    checkpoint_every: int,
    correction_factor: float | None,
) -> ArtifactStep[LevanterCheckpoint]:
    """Apply recipe settings not exposed directly by :func:`train_lm`.

    Batch fitting is runtime-only so TPU migrations do not change run identity. Data,
    evaluation, checkpoint, and W&B settings remain fingerprinted.
    """
    base_build_config = step.build_config

    def build_config(ctx):
        pod = base_build_config(ctx)
        trainer = replace(
            pod.train_config.trainer,
            max_eval_batches=None,
            watch=WANDB_WATCH_CONFIG,
            checkpointer=replace(
                pod.train_config.trainer.checkpointer,
                keep=[{"every": checkpoint_every}],
            ),
        )
        data = replace(
            pod.train_config.data,
            shuffle=SHUFFLE,
            components={
                key: replace(component, pack=True)
                for key, component in pod.train_config.data.components.items()
            },
        )
        if not ctx.is_fingerprint:
            config = batch_fit(point, tpu, correction_factor)
            trainer = replace(
                trainer,
                per_device_parallelism=config.per_device_parallelism,
                per_device_eval_parallelism=config.per_device_parallelism,
            )
        train_config = replace(pod.train_config, trainer=trainer, data=data, data_seed=DATA_SEED)
        return replace(pod, train_config=train_config)

    return replace(step, build_config=build_config)


def build_run(
    point: Point, shape: RunShape, tpu: str, region: str, correction_factor: float | None
) -> ArtifactStep[LevanterCheckpoint]:
    """Assemble one production or smoke training run."""
    name = shape.run_id
    train_cache = _tokenize_cache(COMPONENT_TRAIN, TRAIN_DOCS, validation=False)
    val_cache = _tokenize_cache(COMPONENT_VAL, VAL_DOCS, validation=True)
    batch_config = batch_fit(point, tpu, correction_factor)
    resource_kwargs = {"regions": singleton_region_list(region)}
    if ram := calibration_tpu_ram():
        resource_kwargs["ram"] = ram
    step = train_lm(
        name=name,
        model=point.model_config,
        optimizer=AdamConfig(
            learning_rate=point.learning_rate,
            weight_decay=point.weight_decay,
            warmup=WARMUP,
            lr_schedule=LR_SCHEDULE,
        ),
        datasets={train_cache: 1.0},
        validation=[val_cache],
        batch_size=point.batch_size,
        tensor_parallel_size=batch_config.tensor_parallelism,
        seq_len=SEQ_LEN,
        num_train_steps=shape.num_train_steps,
        z_loss_weight=None,
        evals=None,
        resources=ResourceConfig.with_tpu(tpu, **resource_kwargs),
        version=SWEEP_VERSION,
        steps_per_eval=shape.steps_per_eval,
        wandb_project=os.environ.get("WANDB_PROJECT", "marin"),
        wandb_group=shape.wandb_group,
        run_id=name,
        tags=shape.tags,
        env_vars=_training_env(region),
    )
    return _apply_recipe_overrides(step, point, tpu, shape.checkpoint_every, correction_factor)


# --- Preview + entry point ---------------------------------------------------


def _print_preview(
    point: Point, shape: RunShape, tpu: str, region: str, mode: str, correction_factor: float | None
) -> None:
    config = batch_fit(point, tpu, correction_factor)
    params = point.model_config.total_trainable_params(VOCAB_SIZE)
    tokens = point.batch_size * SEQ_LEN * shape.num_train_steps
    print(
        f"PREVIEW exp146 [{mode}] -- no submit\n"
        f"  run_id={shape.run_id} wandb_group={shape.wandb_group}\n"
        f"  model_size={point.model_size} epochs={point.epochs} lr={point.learning_rate:g} "
        f"wd={point.weight_decay:g} batch_size={point.batch_size}\n"
        f"  steps={shape.num_train_steps} (steps/eval={shape.steps_per_eval}, "
        f"permanent ckpt every {shape.checkpoint_every} steps)\n"
        f"  tokens={tokens / 1e9:.3f}B params={params / 1e9:.3f}B "
        f"schedule={LR_SCHEDULE} warmup={WARMUP}\n"
        f"  tpu={tpu} region={region} prefix={marin_prefix_for_region(region)}\n"
        f"  correction_factor={correction_factor_for(point.model_size, tpu, correction_factor):g} "
        f"data_parallelism={config.data_parallelism} tensor_parallelism={config.tensor_parallelism}\n"
        f"  per_device_parallelism={config.per_device_parallelism} "
        f"gradient_accumulation={config.gradient_accumulation}\n"
        f"  calibration_tpu_ram={calibration_tpu_ram() or '<default>'}\n"
        f"  objective=eval/{COMPONENT_VAL}/loss (final step, minimize)",
        flush=True,
    )


def validate_calibration_smoke(point: Point, tpu: str, correction_factor: float | None) -> None:
    """Keep one-off batch calibration probes isolated from production sweep semantics."""
    if not smoke():
        raise SystemExit("CALIBRATION=yes requires SMOKE=yes")
    if point.model_size not in CALIBRATION_MODEL_SIZES:
        valid = ", ".join(sorted(CALIBRATION_MODEL_SIZES))
        raise SystemExit(f"CALIBRATION=yes only supports MODEL_SIZE in {{{valid}}}; got {point.model_size}")
    if point.batch_size != CALIBRATION_BATCH_SIZE:
        raise SystemExit(f"CALIBRATION=yes requires BATCH_SIZE={CALIBRATION_BATCH_SIZE}; got {point.batch_size}")
    config = batch_fit(point, tpu, correction_factor)
    if config.tensor_parallelism != 1:
        raise SystemExit(
            f"CALIBRATION=yes must not use tensor parallelism; resolved tensor_parallelism={config.tensor_parallelism}"
        )


def _resolve_run(region: str, tpu: str, correction_factor: float | None) -> tuple[Point, RunShape, str]:
    """Resolve the point and mode for this invocation."""
    calibration_run = calibration()
    if smoke():
        point = parse_point(defaults=Point(SMOKE_MODEL_SIZE_DEFAULT, *SMOKE_POINT_DEFAULTS))
        if calibration_run:
            validate_calibration_smoke(point, tpu, correction_factor)
        shape = smoke_shape(point, region, smoke_steps(), tpu, correction_factor, calibration_run=calibration_run)
        return point, shape, "calibration-smoke" if calibration_run else "smoke"
    if calibration_run:
        raise SystemExit("CALIBRATION=yes requires SMOKE=yes")
    point = parse_point()
    validate_sweep_target(point, tpu, correction_factor)
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
