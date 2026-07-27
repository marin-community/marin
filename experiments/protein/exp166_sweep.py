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

At training time, every packed example receives a fresh deterministic re-permutation of
the two-token ``<pN> <AA>`` sequence statements (and the two terminus statements) in each
document. Position assignments, contacts, segment boundaries, and validation data are
unchanged.

Preview one regional execution without lowering or submitting it::

    TRIAL=lr3p162e-3-wd0p2-bs64-scratch TPU=v6e-64 REGION=europe-west4 PREVIEW=yes \\
        uv run python -m experiments.protein.exp166_sweep

Launch one run (secrets come from the Iris command, never the config)::

    source ~/marin.env && uv run iris --cluster marin job run \\
        --user "$USERNAME" --no-wait --region europe-west4 --memory=1GB \\
        -e WANDB_API_KEY "$WANDB_API_KEY" \\
        -e HUGGING_FACE_HUB_TOKEN "$HF_TOKEN" \\
        -e WANDB_ENTITY "$WANDB_ENTITY" -e WANDB_PROJECT "$WANDB_PROJECT" \\
        -e TRIAL lr3p162e-3-wd0p2-bs64-scratch \\
        -e TPU v6e-64 -e REGION europe-west4 \\
        -- python -m experiments.protein.exp166_sweep

Smoke-test the same path under an isolated identity. ``SMOKE_STEPS`` defaults to 2::

    source ~/marin.env && uv run iris --cluster marin job run \\
        --user "$USERNAME" --no-wait --region europe-west4 --memory=1GB \\
        -e WANDB_API_KEY "$WANDB_API_KEY" \\
        -e HUGGING_FACE_HUB_TOKEN "$HF_TOKEN" \\
        -e WANDB_ENTITY "$WANDB_ENTITY" -e WANDB_PROJECT "$WANDB_PROJECT" \\
        -e SMOKE yes -e TRIAL lr3p162e-3-wd0p2-bs64-scratch \\
        -e TPU v6e-4 -e REGION europe-west4 \\
        -- python -m experiments.protein.exp166_sweep
"""

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, fields, replace
from enum import StrEnum

import jax
import numpy as np
from fray.cluster import ResourceConfig
from fray.types import get_tpu_topology, tpu_family
from haliax import Axis
from jaxtyping import PRNGKeyArray
from levanter.data.dataset import AsyncDataset
from levanter.data.text.datasets import BlockShuffleConfig, LmDataConfig
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.lm_model import LmExample
from levanter.models.qwen import Qwen3Config
from levanter.optim.config import AdamConfig
from levanter.schedule import BatchSchedule
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

# --- Identity and data -------------------------------------------------------

VERSION: str = "2026.07.27.1"
EXP117_VERSION: str = "2026.07.13.02"
CACHE_VERSION: str = "2026.07.13.1"
RUN_PREFIX: str = "prot-exp166-cv1-aaaug"
WANDB_GROUP: str = "exp166-contacts-v1-aa-augmentation"
SMOKE_RUN_PREFIX: str = "prot-exp166-aaaug-smoke"
SMOKE_WANDB_GROUP: str = "exp166-aaaug-smoke"
SMOKE_VERSION: str = "v4"
SMOKE_STEPS_DEFAULT: int = 2
SMOKE_MAX_EVAL_BATCHES: int = 1

TOKENIZER: str = "timodonnell/contacts-v1-tokenizer@5d68a24a899f"
VOCAB_SIZE: int = 2845
TEXT_KEY: str = "document"
_DOCS_BASE: str = "protein-structure/MarinFold/exp53_contacts_v1_5x/documents"
TRAIN_DOCS: str = f"{_DOCS_BASE}/train/*.parquet"
VAL_DOCS: str = f"{_DOCS_BASE}/val/*.parquet"
COMPONENT_TRAIN: str = "tokenized/contacts-v1"
COMPONENT_VAL: str = "tokenized/contacts-v1-val"
TRAIN_TOKENS: int = 4_676_753_425

# Token ids from the pinned contacts-v1 tokenizer. The augmentation validates these
# at training startup before touching examples.
CONTACTS_V1_TOKEN_IDS: dict[str, int] = {
    "<contacts-v1>": 2,
    "<begin_sequence>": 8,
    "<begin_statements>": 9,
}
AA_AUGMENTATION_SEED: int = 166
AA_AUGMENTATION_LOG_LIMIT: int = 4
_augmentation_log_count = 0


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
    validate_tpu(tpu, allow_smaller=smoke())
    return tpu


def parse_region() -> str:
    region = os.environ.get("REGION", "").strip().lower()
    if not region:
        raise SystemExit("missing required env var REGION")
    return region


def preview() -> bool:
    return os.environ.get("PREVIEW", "").strip().lower() in {"yes", "true", "1"}


def smoke() -> bool:
    return os.environ.get("SMOKE", "").strip().lower() in {"yes", "true", "1"}


def smoke_steps() -> int:
    raw = os.environ.get("SMOKE_STEPS", str(SMOKE_STEPS_DEFAULT))
    steps = int(raw)
    if steps < 1:
        raise SystemExit(f"SMOKE_STEPS must be >= 1, got {steps}")
    return steps


def validate_tpu(tpu: str, *, allow_smaller: bool = False) -> None:
    """Enforce the large-slice floor while allowing every region."""
    family = tpu_family(tpu)
    chips = get_tpu_topology(tpu).chip_count
    if allow_smaller and family in CORRECTION_FACTORS:
        return
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


def smoke_run_id(trial: Trial, region: str, tpu: str) -> str:
    """Identity-isolated smoke run for one trial, region, and slice."""
    return f"{SMOKE_RUN_PREFIX}-{MODEL_SIZE}-{trial.trial_id}-{region}-{tpu}-{SMOKE_VERSION}"


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


def batch_fit(tpu: str, batch_size: int) -> TpuBatchConfig:
    """Derive DP, TP, per-device parallelism, and accumulation using the current calibrator."""
    family = tpu_family(tpu)
    correction_factor = CORRECTION_FACTORS[family]
    return tpu_batch_config(
        tpu,
        batch_size,
        _batch_bytes(batch_size, correction_factor),
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


@dataclass(frozen=True)
class AugmentationStats:
    """Observable effect of re-randomizing sequence statements in one packed example."""

    documents: int = 0
    residue_statements: int = 0
    moved_statements: int = 0
    changed_token_positions: int = 0


def shuffle_amino_acid_statements(
    token_ids: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, AugmentationStats]:
    """Re-permute each contacts-v1 sequence section without changing its meaning.

    A sequence section consists entirely of two-token statements: one ``<pN> <AA>``
    statement per residue plus the N/C terminus statements. The source corpus shuffles
    these statements once during document generation. This augmentation applies a fresh
    statement-level permutation to every document in a packed training example. Position
    assignments and the structure section are left byte-for-byte unchanged.
    """
    if token_ids.ndim != 1:
        raise ValueError(f"expected one token sequence, got shape {token_ids.shape}")

    augmented = token_ids.copy()
    begin_sequence_id = CONTACTS_V1_TOKEN_IDS["<begin_sequence>"]
    begin_statements_id = CONTACTS_V1_TOKEN_IDS["<begin_statements>"]
    documents = 0
    residue_statements = 0
    moved_statements = 0
    cursor = 0

    while cursor < augmented.size:
        begin_offsets = np.flatnonzero(augmented[cursor:] == begin_sequence_id)
        if begin_offsets.size == 0:
            break
        begin = cursor + int(begin_offsets[0])
        structure_offsets = np.flatnonzero(augmented[begin + 1 :] == begin_statements_id)
        if structure_offsets.size == 0:
            raise ValueError("contacts-v1 sequence marker has no following structure marker")
        structure = begin + 1 + int(structure_offsets[0])
        sequence_length = structure - begin - 1
        if sequence_length % 2:
            raise ValueError(f"contacts-v1 sequence section has odd token count {sequence_length}")

        statement_count = sequence_length // 2
        if statement_count < 2:
            raise ValueError(f"contacts-v1 sequence section has only {statement_count} statement(s)")
        statements = augmented[begin + 1 : structure].reshape(statement_count, 2).copy()
        permutation = rng.permutation(statement_count)
        augmented[begin + 1 : structure] = statements[permutation].reshape(-1)
        documents += 1
        # Every valid contacts-v1 document has exactly two terminus statements.
        residue_statements += statement_count - 2
        moved_statements += int(np.count_nonzero(permutation != np.arange(statement_count)))
        cursor = structure + 1

    return augmented, AugmentationStats(
        documents=documents,
        residue_statements=residue_statements,
        moved_statements=moved_statements,
        changed_token_positions=int(np.count_nonzero(augmented != token_ids)),
    )


def _augmentation_rng(seed: int, index: int) -> np.random.Generator:
    if index < 0:
        raise ValueError(f"dataset index must be nonnegative, got {index}")
    entropy = [seed, index & 0xFFFFFFFF, index >> 32]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def _augment_lm_example(example: LmExample, *, seed: int, index: int) -> LmExample:
    global _augmentation_log_count

    original = np.asarray(jax.device_get(example.tokens.array))
    augmented, stats = shuffle_amino_acid_statements(original, _augmentation_rng(seed, index))
    if stats.documents == 0:
        raise ValueError("packed contacts-v1 training example contains no complete document")

    token_array = jax.device_put(augmented, example.tokens.array.sharding)
    result = replace(example, tokens=replace(example.tokens, array=token_array))

    if jax.process_index() == 0 and _augmentation_log_count < AA_AUGMENTATION_LOG_LIMIT:
        logging.getLogger(__name__).info(
            "exp166 AA augmentation runtime effect: documents=%d residue_statements=%d "
            "moved_statements=%d changed_token_positions=%d",
            stats.documents,
            stats.residue_statements,
            stats.moved_statements,
            stats.changed_token_positions,
        )
        _augmentation_log_count += 1
    return result


class AminoAcidAugmentedDataset(AsyncDataset[LmExample]):
    """Apply deterministic, occurrence-indexed augmentation without invoking JAX PRNGs."""

    def __init__(self, dataset: AsyncDataset[LmExample], seed: int):
        self.dataset = dataset
        self.seed = seed

    async def async_len(self) -> int:
        return await self.dataset.async_len()

    def is_finite(self) -> bool:
        return self.dataset.is_finite()

    async def get_batch(self, indices: Sequence[int]) -> Sequence[LmExample]:
        examples = await self.dataset.get_batch(indices)
        return [
            _augment_lm_example(example, seed=self.seed, index=index)
            for index, example in zip(indices, examples, strict=True)
        ]


def _validate_contacts_v1_tokenizer(data: LmDataConfig) -> None:
    tokenizer = data.the_tokenizer
    observed = tokenizer.convert_tokens_to_ids(list(CONTACTS_V1_TOKEN_IDS))
    expected = list(CONTACTS_V1_TOKEN_IDS.values())
    if observed != expected or len(tokenizer) != VOCAB_SIZE:
        message = f"contacts-v1 tokenizer contract changed: {observed=}, {expected=}, vocab_size={len(tokenizer)}"
        raise ValueError(message)


@dataclass(frozen=True)
class AminoAcidAugmentedDataConfig(LmDataConfig):
    """LmDataConfig variant that augments only the global-indexed training stream."""

    augmentation_seed: int = AA_AUGMENTATION_SEED

    def train_set(
        self,
        Pos: Axis,
        batch_schedule: BatchSchedule,
        *,
        key: PRNGKeyArray,
    ) -> AsyncDataset[LmExample]:
        _validate_contacts_v1_tokenizer(self)
        dataset = super().train_set(Pos, batch_schedule, key=key)
        # This wrapper sits outside MixtureDataset, so ``index`` is the global
        # training occurrence rather than the finite cache index. A document
        # therefore gets a fresh deterministic view each time the mixture restarts.
        # NumPy owns the augmentation RNG so data loading never mixes a CPU JAX key
        # with the active TPU mesh.
        return AminoAcidAugmentedDataset(dataset, self.augmentation_seed)


def augment_amino_acids(data: LmDataConfig) -> LmDataConfig:
    """Enable training-only sequence-statement augmentation."""
    values = {field.name: getattr(data, field.name) for field in fields(LmDataConfig)}
    return AminoAcidAugmentedDataConfig(**values)


def _training_env() -> dict[str, str]:
    env = {"WANDB_PROJECT": os.environ.get("WANDB_PROJECT", "marin")}
    for key in ("WANDB_ENTITY", "WANDB_MODE"):
        if value := os.environ.get(key):
            env[key] = value
    return env


def _tags(trial: Trial, region: str, *, num_train_steps: int) -> list[str]:
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
        f"steps={num_train_steps}",
        f"tokens={point.batch_size * SEQ_LEN * num_train_steps}",
        f"version={VERSION}",
        f"cache_version={CACHE_VERSION}",
    ]
    if trial.initialization is Initialization.EXP117:
        tags.append(f"source_checkpoint={point.exp117_run}")
    return tags


# --- Run construction --------------------------------------------------------


@dataclass(frozen=True)
class RunShape:
    """Identity and bounded cadence for a production or smoke execution."""

    run_id: str
    wandb_group: str
    num_train_steps: int
    steps_per_eval: int
    checkpoint_keep: list[dict[str, int]] | None
    max_eval_batches: int | None
    tags: list[str]
    mode: str


def production_shape(trial: Trial, region: str) -> RunShape:
    point = trial.point
    return RunShape(
        run_id=regional_run_id(trial, region),
        wandb_group=WANDB_GROUP,
        num_train_steps=point.num_train_steps,
        steps_per_eval=point.steps_per_eval,
        checkpoint_keep=[{"every": point.steps_per_epoch}],
        max_eval_batches=None,
        tags=_tags(trial, region, num_train_steps=point.num_train_steps),
        mode="production",
    )


def smoke_shape(trial: Trial, region: str, tpu: str, steps: int) -> RunShape:
    every = max(1, steps // 2)
    tags = [
        *_tags(trial, region, num_train_steps=steps),
        "smoke",
        f"tpu={tpu}",
        f"smoke_version={SMOKE_VERSION}",
    ]
    return RunShape(
        run_id=smoke_run_id(trial, region, tpu),
        wandb_group=SMOKE_WANDB_GROUP,
        num_train_steps=steps,
        steps_per_eval=every,
        checkpoint_keep=None,
        max_eval_batches=SMOKE_MAX_EVAL_BATCHES,
        tags=tags,
        mode="smoke",
    )


def _apply_recipe_overrides(
    step: ArtifactStep[LevanterCheckpoint],
    *,
    tpu: str,
    shape: RunShape,
) -> ArtifactStep[LevanterCheckpoint]:
    """Apply the exp117 data/checkpoint recipe and the exp166 augmentation hook."""
    base_build_config = step.build_config

    def build_config(ctx):
        pod = base_build_config(ctx)
        trainer = replace(
            pod.train_config.trainer,
            max_eval_batches=shape.max_eval_batches,
            checkpointer=replace(
                pod.train_config.trainer.checkpointer,
                keep=shape.checkpoint_keep,
            ),
        )
        data = pod.train_config.data
        components = {key: replace(component, pack=True) for key, component in data.components.items()}
        data = replace(data, shuffle=SHUFFLE, components=components)
        data = augment_amino_acids(data)
        if not ctx.is_fingerprint:
            batch_config = batch_fit(tpu, pod.train_config.trainer.train_batch_size)
            trainer = replace(
                trainer,
                per_device_parallelism=batch_config.per_device_parallelism,
                per_device_eval_parallelism=batch_config.per_device_parallelism,
            )
        train_config = replace(pod.train_config, trainer=trainer, data=data, data_seed=DATA_SEED)
        return replace(pod, train_config=train_config)

    return replace(step, build_config=build_config)


def build_run(trial: Trial, shape: RunShape, tpu: str, region: str) -> ArtifactStep[LevanterCheckpoint]:
    point = trial.point
    name = shape.run_id
    batch_config = batch_fit(tpu, point.batch_size)
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
        num_train_steps=shape.num_train_steps,
        z_loss_weight=None,
        evals=None,
        resources=ResourceConfig.with_tpu(tpu, regions=singleton_region_list(region)),
        version=VERSION,
        init_from=initial_checkpoint(trial),
        tensor_parallel_size=batch_config.tensor_parallelism,
        steps_per_eval=shape.steps_per_eval,
        wandb_project=os.environ.get("WANDB_PROJECT", "marin"),
        wandb_group=shape.wandb_group,
        run_id=name,
        tags=shape.tags,
        env_vars=_training_env(),
    )
    return _apply_recipe_overrides(step, tpu=tpu, shape=shape)


def _print_preview(trial: Trial, shape: RunShape, tpu: str, region: str) -> None:
    point = trial.point
    batch_config = batch_fit(tpu, point.batch_size)
    init = initial_checkpoint(trial)
    checkpoint = init.adopt_source if init is not None else "random initialization"
    print(
        f"PREVIEW exp166 [{shape.mode}] -- no lower or submit\n"
        f"  trial_id={trial.trial_id}\n"
        f"  run_id={shape.run_id} wandb_group={shape.wandb_group}\n"
        f"  suggested_job_id={regional_job_id(trial, region, tpu)}\n"
        f"  lr={point.learning_rate:g} wd={point.weight_decay:g} batch_size={point.batch_size}\n"
        f"  exp117_loss={point.exp117_loss:.8f}\n"
        f"  steps={shape.num_train_steps} steps/eval={shape.steps_per_eval} "
        f"checkpoint_keep={shape.checkpoint_keep}\n"
        f"  initialization={checkpoint}\n"
        f"  tpu={tpu} region={region} prefix={marin_prefix_for_region(region)}\n"
        f"  per_device_parallelism={batch_config.per_device_parallelism} "
        f"grad_accum={batch_config.gradient_accumulation}\n"
        f"  data_parallelism={batch_config.data_parallelism} "
        f"tensor_parallelism={batch_config.tensor_parallelism}\n"
        "  amino_acid_augmentation=per-training-occurrence sequence-statement permutation",
        flush=True,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    trial = parse_trial()
    tpu = parse_tpu()
    region = parse_region()
    shape = smoke_shape(trial, region, tpu, smoke_steps()) if smoke() else production_shape(trial, region)
    if preview():
        _print_preview(trial, shape, tpu, region)
        return
    StepRunner().run([lower(build_run(trial, shape, tpu, region))])


if __name__ == "__main__":
    main()
