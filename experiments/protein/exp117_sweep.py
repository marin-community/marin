# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exp 117: contacts-v1 1.5B LR / weight-decay / epochs tuning, one point per launch.

Single-point trainer for the contacts-v1 1.5B tuning sweep of MarinFold #117
(https://github.com/Open-Athena/MarinFold/issues/117), the multi-epoch extension of #75.
This module trains **exactly one explicit ``(epochs, lr, wd, tpu, region)`` point per
invocation** and nothing else: no grid, no ladder, no smoke test, no scheduling. The
search over ``(epochs, lr, wd)`` is owned by the ``design-adaptive-sweep`` /
``run-adaptive-sweep`` skills, which drive this script one point at a time.

The recipe (Qwen3 1.5B, ``seq_len=8192``, global batch 128, AdamW + cosine with 10%
warmup, unmasked loss, pack-prefix-only, full Feistel shuffle at ``seed=0``, the reused
contacts-v1 caches) mirrors #75 / MarinFold #70. Where #117 and #75 differ, **#117
wins** -- notably one permanent checkpoint every epoch (in addition to the 10-minute
rolling resumption checkpoint).

Objective being optimized: **final-step ``eval/contacts-v1-val/loss``** (read from W&B).

Interface for the adaptive-sweep skills (identity-vs-execution split):
  * ``TPU`` selects the slice and drives per-device batch sizing
    (:func:`~experiments.coral.batch_config.tpu_batch_config`) *without* touching run
    identity: the global batch stays 128 on every slice (so the objective is comparable),
    and a same-region change to any compatible slice resumes the same run.
  * ``REGION`` sets the storage prefix (checkpoint locality) and the W&B / trainer run id.
    A region change starts a fresh run under a fresh regional identity; a same-region
    re-dispatch resumes the rolling checkpoint. The contacts-v1 caches are read
    region-local through ``mirror://`` (copied once if absent).

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
"""

import logging
import os
from dataclasses import dataclass, replace

from fray.cluster import ResourceConfig
from levanter.checkpoint import CheckpointInterval
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim import AdamConfig
from marin.execution.lazy import ArtifactStep, lower
from marin.execution.step_runner import StepRunner
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


# --- Identity ----------------------------------------------------------------

# Fixed calendar version: keeps the run in the shared namespace and makes the run id,
# checkpoint path, and fingerprint stable across invocations so re-runs of a point are
# idempotent and resumable. Bump to fork a fresh campaign over the same recipe.
VERSION: str = "2026.07.13"
RUN_PREFIX: str = "prot-exp117"
WANDB_GROUP: str = "exp117-contacts-v1-tune"


# --- Data (reused contacts-v1 caches from MarinFold #70) ---------------------

# contacts-v1 tokenizer (2845 vocab), loaded from HF at the immutable revision the caches
# were tokenized from.
TOKENIZER: str = "timodonnell/contacts-v1-tokenizer@5d68a24a899f"
VOCAB_SIZE: int = 2845
TEXT_KEY: str = "document"

# Immutable levanter caches published by #70 under the exp67 prefix. Referenced through
# ``mirror://`` (bucket-relative key) so the training job reads them from its own region,
# copying once if absent. NB: a full copy of the train cache exceeds the 10 GB
# cross-region TransferBudget, so running outside us-east5 -- where the caches already
# live -- needs explicit approval (see experiments/AGENTS.md); in us-east5 the read is local.
_CACHE_BASE: str = "mirror://protein-structure/MarinFold/exp67_contacts_v1_1_5b/tokenized"
TRAIN_CACHE: str = f"{_CACHE_BASE}/contacts-v1-663ba6"
VAL_CACHE: str = f"{_CACHE_BASE}/contacts-v1-val-92827b"

# Mixture component keys -> the prefix of each ``eval/<component>/loss`` W&B series. The
# val key is the swept objective.
COMPONENT_TRAIN: str = "contacts-v1"
COMPONENT_VAL: str = "contacts-v1-val"

# Exact train-corpus token count read from the reused cache ledger
# (contacts-v1-663ba6 train/.stats.json). Steps/epoch are derived from this, not estimated.
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


def run_id(point: Point, region: str) -> str:
    """The W&B run name, trainer/resume id, and checkpoint subpath for a ``(point, region)``.

    Keyed on the point and the region -- never the TPU -- so a region change is a fresh run
    while any compatible-slice re-dispatch in the same region resumes the same run.
    """
    return f"{RUN_PREFIX}-cv1-1_5b-{point.point_id}-{region}"


# --- Env inputs (one point per launch) ---------------------------------------


def parse_point() -> Point:
    try:
        epochs = int(os.environ["EPOCHS"])
        learning_rate = float(os.environ["LR"])
        weight_decay = float(os.environ["WD"])
    except KeyError as e:
        raise SystemExit(f"missing required env var {e}; set EPOCHS, LR, WD, TPU, REGION") from e
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


def _adopt_cache(name: str, source: str) -> ArtifactStep[TokenizedCache]:
    """Adopt a prebuilt contacts-v1 levanter cache, read region-local via ``mirror://``.

    ``name`` is the mixture component key; for the val cache it becomes the
    ``eval/<name>/loss`` objective series. ``pack`` is applied on the assembled data config
    (see :func:`_apply_recipe_overrides`), not here.
    """
    return ArtifactStep.adopt(
        name,
        VERSION,
        source,
        kind=TokenizedCache,
        config={
            "tokenizer": TOKENIZER,
            "format": {"text_key": TEXT_KEY},
            "tags": ["protein", "contacts-v1", name],
        },
    )


def _training_env(region: str) -> dict[str, str]:
    """Environment for the dispatched training job: region prefix + W&B metadata.

    Secrets (WANDB_API_KEY, HUGGING_FACE_HUB_TOKEN) are passed by the iris command, not
    baked into the config.
    """
    env = {
        "MARIN_PREFIX": marin_prefix_for_region(region),
        "WANDB_PROJECT": os.environ.get("WANDB_PROJECT", "marin"),
    }
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


def _apply_recipe_overrides(step: ArtifactStep[LevanterCheckpoint], tpu: str) -> ArtifactStep[LevanterCheckpoint]:
    """Apply the #117 knobs :func:`train_lm` does not expose.

    Identity-bearing (always applied, so they enter the fingerprint): one permanent
    checkpoint every epoch alongside the 10-minute rolling resumption checkpoint, and
    pack-prefix-only components (documents are never concat-and-split). Execution-only
    (run time only, so run identity stays TPU-independent): fit the global batch to the
    actual TPU via per-device parallelism.
    """
    base_build_config = step.build_config

    def build_config(ctx):
        pod = base_build_config(ctx)
        trainer = replace(
            pod.train_config.trainer,
            checkpointer=replace(
                pod.train_config.trainer.checkpointer,
                keep=[CheckpointInterval(every=STEPS_PER_EPOCH)],
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


def build_run(point: Point, tpu: str, region: str) -> ArtifactStep[LevanterCheckpoint]:
    """Assemble the single ``(point, tpu, region)`` training run as an ``ArtifactStep``."""
    name = run_id(point, region)
    train_cache = _adopt_cache(COMPONENT_TRAIN, TRAIN_CACHE)
    val_cache = _adopt_cache(COMPONENT_VAL, VAL_CACHE)
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
        num_train_steps=point.num_train_steps,
        z_loss_weight=None,
        evals=None,  # no lm-eval-harness; the objective is the weight-0 val-loss eval
        resources=ResourceConfig.with_tpu(tpu, regions=singleton_region_list(region)),
        version=VERSION,
        steps_per_eval=STEPS_PER_EVAL,
        wandb_project=os.environ.get("WANDB_PROJECT", "marin"),
        wandb_group=WANDB_GROUP,
        run_id=name,
        tags=_tags(point, region),
        env_vars=_training_env(region),
    )
    return _apply_recipe_overrides(step, tpu)


# --- Preview + entry point ---------------------------------------------------


def _print_preview(point: Point, tpu: str, region: str) -> None:
    per_device_parallelism, grad_accum = batch_fit(tpu)
    params = MODEL_CONFIG.total_trainable_params(VOCAB_SIZE)
    tokens = TOKENS_PER_STEP * point.num_train_steps
    print(
        "PREVIEW exp117 -- one point, no submit\n"
        f"  run_id={run_id(point, region)}\n"
        f"  epochs={point.epochs} lr={point.learning_rate:g} wd={point.weight_decay:g}\n"
        f"  steps={point.num_train_steps} (steps/epoch={STEPS_PER_EPOCH}, steps/eval={STEPS_PER_EVAL}, "
        "permanent ckpt/epoch)\n"
        f"  tokens={tokens / 1e9:.3f}B params={params / 1e9:.3f}B schedule={LR_SCHEDULE} warmup={WARMUP}\n"
        f"  tpu={tpu} region={region} prefix={marin_prefix_for_region(region)}\n"
        f"  per_device_parallelism={per_device_parallelism} grad_accum={grad_accum} (global batch {BATCH_SIZE})\n"
        f"  objective=eval/{COMPONENT_VAL}/loss (final step, minimize)",
        flush=True,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    point = parse_point()
    tpu = parse_tpu()
    region = parse_region()

    # Region is explicit: pin the driver's storage prefix to the region bucket so output
    # (checkpoints) lands in-region regardless of where the driver runs.
    os.environ["MARIN_PREFIX"] = marin_prefix_for_region(region)

    if preview():
        _print_preview(point, tpu, region)
        return

    StepRunner().run([lower(build_run(point, tpu, region))])


if __name__ == "__main__":
    main()
