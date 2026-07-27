# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exp 166: contacts-v1 amino-acid augmentation ablation.

Runs the four best completed 8-epoch configurations from MarinFold #117
(https://github.com/Open-Athena/MarinFold/issues/117) with training-time amino-acid
augmentation. Each configuration is trained both from scratch and from its corresponding
exp117 checkpoint, for eight logical trials total.

``TRIAL`` identifies a logical trial and excludes region. ``REGION`` is included in the
W&B run and suggested Iris job identities so three regional executions can race without
changing trial identity. ``TPU`` is execution-only and must be a 64-256 chip v5e/v6e slice
or a v5p slice with at least 32 chips (``v5p-64`` or larger).

The amino-acid augmentation hook is intentionally a fail-closed stub. Preview works, but
lowering a run raises until the augmentation behavior is designed and implemented.

Preview one regional execution without lowering or submitting it::

    TRIAL=lr3p162e-3-wd0p2-bs64-scratch TPU=v6e-64 REGION=europe-west4 PREVIEW=yes \\
        uv run python -m experiments.protein.exp166_sweep
"""

import logging
import math
import os
from dataclasses import dataclass, replace
from enum import StrEnum

from fray.cluster import ResourceConfig
from fray.types import get_tpu_topology, tpu_family
from levanter.data.text.datasets import BlockShuffleConfig, LmDataConfig
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

# --- Identity and data -------------------------------------------------------

VERSION: str = "2026.07.27.1"
EXP117_VERSION: str = "2026.07.13.02"
CACHE_VERSION: str = "2026.07.13.1"
RUN_PREFIX: str = "prot-exp166-cv1-aaaug"
WANDB_GROUP: str = "exp166-contacts-v1-aa-augmentation"

TOKENIZER: str = "timodonnell/contacts-v1-tokenizer@5d68a24a899f"
VOCAB_SIZE: int = 2845
TEXT_KEY: str = "document"
_DOCS_BASE: str = "protein-structure/MarinFold/exp53_contacts_v1_5x/documents"
TRAIN_DOCS: str = f"{_DOCS_BASE}/train/*.parquet"
VAL_DOCS: str = f"{_DOCS_BASE}/val/*.parquet"
COMPONENT_TRAIN: str = "tokenized/contacts-v1"
COMPONENT_VAL: str = "tokenized/contacts-v1-val"
TRAIN_TOKENS: int = 4_676_753_425


# --- Fixed recipe ------------------------------------------------------------

MODEL_CONFIG = Qwen3Config(
    max_seq_len=8192,
    hidden_dim=2048,
    intermediate_dim=8192,
    num_heads=32,
    num_kv_heads=8,
    num_layers=24,
    rope=Llama3RotaryEmbeddingsConfig(),
)
MODEL_SIZE: str = "1_5b"
EPOCHS: int = 8
SEQ_LEN: int = 8192
DATA_SEED: int = 0
SHUFFLE = BlockShuffleConfig(io_block_size=256, window_blocks=512, perm_type="feistel")
WARMUP: float = 0.1
LR_SCHEDULE: str = "cosine"
NUM_EVALS_PER_EPOCH: int = 2
CORRECTION_FACTORS: dict[str, float] = {"v5e": 0.5, "v6e": 0.3, "v5p": 0.45}


# --- Four exp117 winners x two initialization modes -------------------------


@dataclass(frozen=True)
class Point:
    """One of the four selected exp117 hyperparameter configurations."""

    key: str
    learning_rate: float
    weight_decay: float
    batch_size: int
    exp117_run: str
    exp117_loss: float

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
        return EPOCHS * self.steps_per_epoch


POINTS: tuple[Point, ...] = (
    Point(
        key="lr3p162e-3-wd0p2-bs64",
        learning_rate=3.1623e-3,
        weight_decay=0.2,
        batch_size=64,
        exp117_run="prot-exp117-cv1-s02-1_5b-e8-lr3p162e-3-wd0p2-bs64-europe-west4",
        exp117_loss=2.7130589485168457,
    ),
    Point(
        key="lr3p162e-4-wd1p6-bs64",
        learning_rate=3.1623e-4,
        weight_decay=1.6,
        batch_size=64,
        exp117_run="prot-exp117-cv1-s02-1_5b-e8-lr3p162e-4-wd1p6-bs64-us-east5",
        exp117_loss=2.733017921447754,
    ),
    Point(
        key="lr3p162e-3-wd0p1-bs128",
        learning_rate=3.1623e-3,
        weight_decay=0.1,
        batch_size=128,
        exp117_run="prot-exp117-cv1-s02-1_5b-e8-lr3p162e-3-wd0p1-bs128-europe-west4",
        exp117_loss=2.736903667449951,
    ),
    Point(
        key="lr1e-3-wd0p8-bs128",
        learning_rate=1e-3,
        weight_decay=0.8,
        batch_size=128,
        exp117_run="prot-exp117-cv1-s02-1_5b-e8-lr1e-3-wd0p8-bs128-us-east5",
        exp117_loss=2.7450978755950928,
    ),
)


class Initialization(StrEnum):
    SCRATCH = "scratch"
    EXP117 = "exp117-init"


@dataclass(frozen=True)
class Trial:
    """A logical trial: one selected point and one initialization mode."""

    point: Point
    initialization: Initialization

    @property
    def trial_id(self) -> str:
        return f"{self.point.key}-{self.initialization.value}"


TRIALS: dict[str, Trial] = {
    trial.trial_id: trial
    for point in POINTS
    for trial in (Trial(point, Initialization.SCRATCH), Trial(point, Initialization.EXP117))
}


def parse_trial() -> Trial:
    key = os.environ.get("TRIAL", "").strip().lower()
    if key not in TRIALS:
        choices = ", ".join(TRIALS)
        raise SystemExit(f"TRIAL must be one of: {choices}")
    return TRIALS[key]


def parse_tpu() -> str:
    tpu = os.environ.get("TPU", "").strip().lower()
    if not tpu:
        raise SystemExit("missing required env var TPU")
    validate_tpu(tpu)
    return tpu


def parse_region() -> str:
    region = os.environ.get("REGION", "").strip().lower()
    if not region:
        raise SystemExit("missing required env var REGION")
    return region


def preview() -> bool:
    return os.environ.get("PREVIEW", "").strip().lower() in {"yes", "true", "1"}


def validate_tpu(tpu: str) -> None:
    """Enforce the large-slice floor while allowing every region."""
    family = tpu_family(tpu)
    chips = get_tpu_topology(tpu).chip_count
    if family in {"v5e", "v6e"} and 64 <= chips <= 256:
        return
    if family == "v5p" and chips >= 32:
        return
    message = f"unsupported TPU {tpu!r} ({family=}, {chips=}): "
    message += "use 64-256 chip v5e/v6e or v5p-64+ (at least 32 chips)"
    raise SystemExit(message)


def regional_run_id(trial: Trial, region: str) -> str:
    """W&B/resume identity for one regional execution of a logical trial."""
    return f"{RUN_PREFIX}-{MODEL_SIZE}-e{EPOCHS}-{trial.trial_id}-{region}"


def regional_job_id(trial: Trial, region: str, tpu: str) -> str:
    """Suggested Iris job identity; attempts may append ``-aN``."""
    return f"{regional_run_id(trial, region)}-{tpu}"


# --- Checkpoint initialization ----------------------------------------------


def exp117_checkpoint(point: Point) -> ArtifactStep[LevanterCheckpoint]:
    """Adopt the relocated exp117 output through the region-local mirror filesystem."""
    return ArtifactStep.adopt(
        name=f"adopted/exp117/{point.key}",
        version=EXP117_VERSION,
        source=f"mirror://checkpoints/protein/{point.exp117_run}/{EXP117_VERSION}",
        kind=LevanterCheckpoint,
    )


def initial_checkpoint(trial: Trial) -> ArtifactStep[LevanterCheckpoint] | None:
    if trial.initialization is Initialization.SCRATCH:
        return None
    return exp117_checkpoint(trial.point)


# --- Placement ---------------------------------------------------------------


def placement_axes(tpu: str, batch_size: int) -> tuple[int, int]:
    """Return ``(data_axis_size, tensor_parallel_size)`` for the selected slice."""
    chip_count = get_tpu_topology(tpu).chip_count
    data_axis_size = math.gcd(chip_count, batch_size)
    return data_axis_size, chip_count // data_axis_size


def _batch_bytes(batch_size: int, correction_factor: float) -> int:
    params = MODEL_CONFIG.total_trainable_params(VOCAB_SIZE)
    param_bytes, activation_bytes = dense_transformer_bytes(
        parameter_count=params,
        batch_size=batch_size,
        seq_len=SEQ_LEN,
        hidden_dim=MODEL_CONFIG.hidden_dim,
        intermediate_dim=MODEL_CONFIG.intermediate_dim,
        num_layers=MODEL_CONFIG.num_layers,
    )
    return batch_memory_bytes(
        parameter_bytes=param_bytes,
        optimizer_bytes=adam_optimizer_bytes(params),
        activation_bytes=activation_bytes,
        correction_factor=correction_factor,
    )


def batch_fit(tpu: str, batch_size: int) -> tuple[int, int]:
    family = tpu_family(tpu)
    correction_factor = CORRECTION_FACTORS[family]
    data_axis_size, _ = placement_axes(tpu, batch_size)
    return tpu_batch_config(
        tpu,
        batch_size,
        _batch_bytes(batch_size, correction_factor),
        data_axis_size=data_axis_size,
    )


# --- Data and augmentation ---------------------------------------------------


def _tokenize_cache(name: str, docs: str, *, validation: bool) -> ArtifactStep[TokenizedCache]:
    return tokenized(
        name=name,
        tokenizer=TOKENIZER,
        version=CACHE_VERSION,
        paths=[docs],
        text_key=TEXT_KEY,
        validation=validation,
        tags=["protein", "contacts-v1", name],
    )


def augment_amino_acids(_data: LmDataConfig) -> LmDataConfig:
    """Randomize already-randomized amino acids in training examples only.

    Deliberate stub: we will design the token-level transformation, randomness, and
    validation isolation together before enabling experiment launches.
    """
    raise NotImplementedError("exp166 amino-acid augmentation is not implemented")


def _training_env() -> dict[str, str]:
    env = {"WANDB_PROJECT": os.environ.get("WANDB_PROJECT", "marin")}
    for key in ("WANDB_ENTITY", "WANDB_MODE"):
        if value := os.environ.get(key):
            env[key] = value
    return env


def _tags(trial: Trial, region: str) -> list[str]:
    point = trial.point
    params = MODEL_CONFIG.total_trainable_params(VOCAB_SIZE)
    tags = [
        "protein",
        "exp166",
        "contacts-v1",
        "aa-augmentation",
        "qwen3",
        f"trial_id={trial.trial_id}",
        f"initialization={trial.initialization.value}",
        f"model_size={MODEL_SIZE}",
        f"global_batch={point.batch_size}",
        f"params={params}",
        f"epochs={EPOCHS}",
        f"lr={point.learning_rate:g}",
        f"wd={point.weight_decay:g}",
        f"exp117_loss={point.exp117_loss:.8f}",
        f"region={region}",
        f"steps={point.num_train_steps}",
        f"tokens={point.batch_size * SEQ_LEN * point.num_train_steps}",
        f"version={VERSION}",
        f"cache_version={CACHE_VERSION}",
    ]
    if trial.initialization is Initialization.EXP117:
        tags.append(f"source_checkpoint={point.exp117_run}")
    return tags


# --- Run construction --------------------------------------------------------


def _apply_recipe_overrides(
    step: ArtifactStep[LevanterCheckpoint],
    *,
    tpu: str,
    checkpoint_every: int,
) -> ArtifactStep[LevanterCheckpoint]:
    """Apply the exp117 data/checkpoint recipe and the exp166 augmentation hook."""
    base_build_config = step.build_config

    def build_config(ctx):
        pod = base_build_config(ctx)
        trainer = replace(
            pod.train_config.trainer,
            max_eval_batches=None,
            checkpointer=replace(
                pod.train_config.trainer.checkpointer,
                keep=[{"every": checkpoint_every}],
            ),
        )
        data = pod.train_config.data
        components = {key: replace(component, pack=True) for key, component in data.components.items()}
        data = replace(data, shuffle=SHUFFLE, components=components)
        data = augment_amino_acids(data)
        if not ctx.is_fingerprint:
            per_device_parallelism = batch_fit(tpu, pod.train_config.trainer.train_batch_size)[0]
            eval_parallelism = (
                trainer.per_device_eval_parallelism if per_device_parallelism == -1 else per_device_parallelism
            )
            trainer = replace(
                trainer,
                per_device_parallelism=per_device_parallelism,
                per_device_eval_parallelism=eval_parallelism,
            )
        train_config = replace(pod.train_config, trainer=trainer, data=data, data_seed=DATA_SEED)
        return replace(pod, train_config=train_config)

    return replace(step, build_config=build_config)


def build_run(trial: Trial, tpu: str, region: str) -> ArtifactStep[LevanterCheckpoint]:
    point = trial.point
    name = regional_run_id(trial, region)
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
        batch_size=point.batch_size,
        seq_len=SEQ_LEN,
        num_train_steps=point.num_train_steps,
        z_loss_weight=None,
        evals=None,
        resources=ResourceConfig.with_tpu(tpu, regions=singleton_region_list(region)),
        version=VERSION,
        init_from=initial_checkpoint(trial),
        tensor_parallel_size=placement_axes(tpu, point.batch_size)[1],
        steps_per_eval=point.steps_per_eval,
        wandb_project=os.environ.get("WANDB_PROJECT", "marin"),
        wandb_group=WANDB_GROUP,
        run_id=name,
        tags=_tags(trial, region),
        env_vars=_training_env(),
    )
    return _apply_recipe_overrides(step, tpu=tpu, checkpoint_every=point.steps_per_epoch)


def _print_preview(trial: Trial, tpu: str, region: str) -> None:
    point = trial.point
    per_device_parallelism, grad_accum = batch_fit(tpu, point.batch_size)
    data_axis_size, tensor_parallel_size = placement_axes(tpu, point.batch_size)
    init = initial_checkpoint(trial)
    checkpoint = init.adopt_source if init is not None else "random initialization"
    print(
        "PREVIEW exp166 -- no lower or submit\n"
        f"  trial_id={trial.trial_id}\n"
        f"  run_id={regional_run_id(trial, region)}\n"
        f"  suggested_job_id={regional_job_id(trial, region, tpu)}\n"
        f"  lr={point.learning_rate:g} wd={point.weight_decay:g} batch_size={point.batch_size}\n"
        f"  exp117_loss={point.exp117_loss:.8f}\n"
        f"  steps={point.num_train_steps} steps/eval={point.steps_per_eval} "
        f"checkpoint_every={point.steps_per_epoch}\n"
        f"  initialization={checkpoint}\n"
        f"  tpu={tpu} region={region} prefix={marin_prefix_for_region(region)}\n"
        f"  per_device_parallelism={per_device_parallelism} grad_accum={grad_accum}\n"
        f"  data_axis_size={data_axis_size} tensor_parallel_size={tensor_parallel_size}\n"
        "  amino_acid_augmentation=STUB (launch intentionally blocked)",
        flush=True,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    trial = parse_trial()
    tpu = parse_tpu()
    region = parse_region()
    if preview():
        _print_preview(trial, tpu, region)
        return
    StepRunner().run([lower(build_run(trial, tpu, region))])


if __name__ == "__main__":
    main()
