# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Single-run trainer for the exp166 contacts-v1 amino-acid augmentation ablation on GPUs.

Port of MarinFold #166 from TRC TPUs to CoreWeave. The augmentation question is unchanged;
the regional machinery the TPU version was built around is gone, because every CoreWeave
cluster reads the same bucket. See ``exp166_cw_plan.md``.

Each invocation trains one logical trial: an exp117 hyperparameter configuration, an
initialization mode, and whether amino-acid augmentation is on. ``CLUSTER`` and ``NODES``
are placement only and never enter run identity, so any re-dispatch resumes the same run
from the same checkpoint.

Eight trials. Six train from random weights with augmentation, one per exp117
configuration. Two continue from exp117's ``lr3p162e-3-wd0p2-bs64`` checkpoint and differ
only in whether augmentation is on; the no-augmentation one measures what eight further
epochs buy on their own.

At training time every packed example receives a fresh deterministic re-permutation of the
two-token ``<pN> <AA>`` sequence statements in each document. Position assignments,
contacts, segment boundaries and validation data are unchanged.

Preview one trial (builds and submits nothing)::

    TRIAL=lr3p162e-3-wd0p2-bs64-scratch-aug CLUSTER=cw-us-east-08a NODES=4 PREVIEW=yes \\
        uv run python -m experiments.protein.exp166_cw_sweep

Launch one run. The driver is a small CPU job **inside** the target cluster: Iris only
federates root jobs, so the gang lands wherever the driver runs, and ``--priority batch``
on the driver is inherited by the gang::

    set -a; source ~/marin.env; set +a
    uv run iris --config lib/iris/config/marin.yaml job run \\
        --user "$USERNAME" --target-cluster cw-us-east-08a --priority batch \\
        --no-wait --memory 3GB \\
        -e WANDB_API_KEY "$WANDB_API_KEY" -e HUGGING_FACE_HUB_TOKEN "$HF_TOKEN" \\
        -e WANDB_ENTITY "$WANDB_ENTITY" -e WANDB_PROJECT "$WANDB_PROJECT" \\
        -e TRIAL lr3p162e-3-wd0p2-bs64-scratch-aug \\
        -e CLUSTER cw-us-east-08a -e NODES 4 \\
        -- python -m experiments.protein.exp166_cw_sweep

Smoke or calibrate under an isolated identity, on TTL'd temp storage. ``PER_DEVICE`` forces
the microbatch, which is how an unmeasured GPU type gets its ceiling probed::

    ... -e SMOKE yes -e PER_DEVICE 8 -e CLUSTER cw-us-east-08a -e NODES 1 \\
        -- python -m experiments.protein.exp166_cw_sweep
"""

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, fields, replace
from enum import StrEnum

import fsspec
import jax
import numpy as np
from fray.cluster import ResourceConfig
from haliax import Axis
from jaxtyping import PRNGKeyArray
from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import latest_checkpoint_path
from levanter.data.dataset import AsyncDataset
from levanter.data.text.datasets import BlockShuffleConfig, LmDataConfig
from levanter.layers.attention import AttentionBackend
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
from marin.training.training import LevanterCheckpoint, temporary_checkpoint_base_path
from rigging.filesystem import marin_prefix, marin_temp_bucket

from experiments.coral.batch_calibration import GpuBatchConfig, gpu_batch_config

logger = logging.getLogger(__name__)


# --- Storage -----------------------------------------------------------------

# Everything this experiment writes lives under its own MarinFold-prefixed directory so it
# can be bulk-removed later (#108's standing rule) without touching another experiment.
STORAGE_PREFIX: str = "s3://marin-us-east-02a/MarinFold/exp166cw_qwen_contacts_v1"

# Every byte this experiment reads or writes must live on this CoreWeave bucket. The guard
# below is not defensive boilerplate: a CoreWeave pod resolves the cluster config to GCS
# (scheme ``gs``) because ``MARIN_CLUSTER`` is unset there. Paths stay on S3 only because
# ``MARIN_PREFIX`` is forwarded to the gang; if that stops happening, path resolution
# silently drifts toward GCS and a checkpoint write becomes cross-cloud egress.
STORAGE_BUCKET: str = "s3://marin-us-east-02a/"

# The token cache is READ IN PLACE from the exp153 prefix, never copied and never written.
# ``ArtifactStep.path()`` returns a pin verbatim when it carries a URL scheme instead of
# joining it onto MARIN_PREFIX, and a pinned step records no provenance, so this is a
# read-only reference. Consequence: exp154's ``tokenized/`` is a live dependency of this
# experiment and must not be pruned.
_EXP154_PREFIX: str = "s3://marin-us-east-02a/MarinFold/exp154_qwen_contacts_v1"
CACHE_VERSION: str = "2026.07.25"
TRAIN_CACHE_PIN: str = f"{_EXP154_PREFIX}/tokenized/contacts-v1/{CACHE_VERSION}"
VAL_CACHE_PIN: str = f"{_EXP154_PREFIX}/tokenized/contacts-v1-val/{CACHE_VERSION}"

# Raw contacts-v1 parquet, staged to the CoreWeave bucket by MarinFold exp108. Only read
# if a cache pin is ever removed; the pinned caches make tokenizing unnecessary.
_DOCS_BASE: str = "s3://marin-us-east-02a/MarinFold/data/document_structures/contacts_v1"
TRAIN_DOCS: str = f"{_DOCS_BASE}/train/*.parquet"
VAL_DOCS: str = f"{_DOCS_BASE}/val/*.parquet"

# Region-local is not a concept here: one bucket serves every cluster. The seed namespace
# keeps the staged exp117 weights from colliding with this experiment's own run outputs.
SEED_NAMESPACE: str = "checkpoints/exp166-init"
EXP117_VERSION: str = "2026.07.13.02"


# --- Identity ----------------------------------------------------------------

SWEEP_VERSION: str = "2026.07.29.01"
RUN_PREFIX: str = "prot-exp166cw-cv1-aaaug"
WANDB_GROUP: str = "exp166cw-contacts-v1-aa-augmentation"
CHECKPOINT_ROOT: str = "checkpoints"

SMOKE_RUN_PREFIX: str = "prot-exp166cw-aaaug-smoke"
SMOKE_WANDB_GROUP: str = "exp166cw-aaaug-smoke"
# Bump to fork a clean smoke after a recipe or library change; otherwise the executor
# prunes the step as already-succeeded and the "run" is a silent cache hit.
SMOKE_VERSION: str = "v1"
SMOKE_STEPS_DEFAULT: int = 20
SMOKE_TTL_DAYS: int = 1
SMOKE_NUM_EVALS: int = 2
SMOKE_MAX_EVAL_BATCHES: int = 4

GPU_RAM_ENV: str = "GPU_RAM"


# --- Data --------------------------------------------------------------------

TOKENIZER: str = "timodonnell/contacts-v1-tokenizer@5d68a24a899f"
VOCAB_SIZE: int = 2845
TEXT_KEY: str = "document"

# Cache names double as the mixture component key and the eval series prefix
# (``eval/<name>/loss``). They must match the pinned caches' own component names, since the
# objective metric key is compared against #117 and #153.
COMPONENT_TRAIN: str = "tokenized/contacts-v1"
COMPONENT_VAL: str = "tokenized/contacts-v1-val"

# Exact train-corpus token count for contacts-v1 at TOKENIZER, from the #117 cache.
TRAIN_TOKENS: int = 4_676_753_425

# Token ids from the pinned contacts-v1 tokenizer. The augmentation validates these at
# training startup before touching any example.
CONTACTS_V1_TOKEN_IDS: dict[str, int] = {
    "<contacts-v1>": 2,
    "<begin_sequence>": 8,
    "<begin_statements>": 9,
}
AA_AUGMENTATION_SEED: int = 166
AA_AUGMENTATION_LOG_LIMIT: int = 4
_augmentation_log_count = 0


# --- Model -------------------------------------------------------------------

MODEL_SIZE: str = "1_5b"
# Architecture is fixed by #117 and is not a tuning axis.
MODEL_CONFIG = Qwen3Config(
    max_seq_len=8192,
    hidden_dim=2048,
    intermediate_dim=8192,
    num_heads=32,
    num_kv_heads=8,
    num_layers=24,
    rope=Llama3RotaryEmbeddingsConfig(),
    # On GPU levanter defaults to the Transformer Engine backend, which is absent from the
    # `gpu` extra and silently falls back to the O(seq^2) reference kernel — unusable at
    # seq 8192 (MarinFold #108; upstream marin#7013 is still open). JAX_FLASH is the
    # memory-safe blocked-flash path that needs no TE. The TPU version of this sweep does
    # not set this and must not be copied verbatim.
    attn_backend=AttentionBackend.JAX_FLASH,
)


# --- Training recipe ---------------------------------------------------------

EPOCHS: int = 8
SEQ_LEN: int = 8192
DATA_SEED: int = 0
SHUFFLE = BlockShuffleConfig(io_block_size=256, window_blocks=512, perm_type="feistel")
WARMUP: float = 0.1
LR_SCHEDULE: str = "cosine"
NUM_EVALS_PER_EPOCH: int = 2
WANDB_WATCH_CONFIG = WatchConfig(watch_targets=[], interval=0)

# Measured from the exp117 HF checkpoints: a 1.5B Levanter save is 16.44 GiB and its HF
# export 5.48 GiB. Used only to report expected storage in the preview.
LEVANTER_CHECKPOINT_BYTES: int = int(16.44 * 1024**3)
HF_EXPORT_BYTES: int = int(5.48 * 1024**3)

# Measured sequence capacity per GPU: the largest per-device microbatch under which a run
# survives training AND the eval pass AND a checkpoint save, which are three distinct
# memory peaks. One number per GPU type, measured on the SMALLEST supported gang and valid
# for every larger one, because parameters, gradients and optimizer state are FSDP-sharded
# over the data axis while activations are not.
#
# EMPTY ON PURPOSE. exp153's H100 value of 8 was measured on the 6B and does not transfer
# to this 1.5B. Fill each entry from a calibration probe (SMOKE=yes with PER_DEVICE,
# stepping up until it stops surviving) before any production run on that GPU type.
MAX_SEQS_PER_DEVICE: dict[str, int] = {}


# --- Targets -----------------------------------------------------------------


@dataclass(frozen=True)
class ClusterSpec:
    """One CoreWeave training cluster: its node shape and per-node container ask.

    ``gpus_per_node`` is the node's full complement. Sub-node GPU requests do schedule on
    CoreWeave but fragment the InfiniBand gang pool, so a gang is always whole nodes.
    """

    gpu_variant: str
    gpus_per_node: int
    cpu: int
    ram: str
    disk: str


CLUSTERS: dict[str, ClusterSpec] = {
    # gb200-4x: 4x GB200-186GB per node. The primary target: 208 schedulable nodes against
    # the H100 fleet's 36, and the parity runs put it slightly ahead of v6e-4.
    "cw-us-east-08a": ClusterSpec("GB200", 4, cpu=32, ram="256g", disk="256g"),
    # gd-8xh100ib-i128: 8x H100-80GB + InfiniBand, 128 vCPU / 2 TB per node.
    "cw-us-east-02a": ClusterSpec("H100", 8, cpu=32, ram="256g", disk="256g"),
}

# Gang ceiling. A 1.5B at seq 8192 does roughly a quarter the compute per step that the 6B
# did, so the communication-to-compute ratio is worse and scaling falls off sooner than
# exp153's 94%-at-8-nodes. Eight modest gangs running at once finish the set faster than
# eight large ones in sequence.
MAX_NODES: int = 4


# --- The six exp117 configurations -------------------------------------------


@dataclass(frozen=True)
class Point:
    """One exp117 hyperparameter configuration, with the result it reached there."""

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
    Point(
        key="lr3p162e-4-wd1p6-bs128",
        learning_rate=3.1623e-4,
        weight_decay=1.6,
        batch_size=128,
        exp117_run="prot-exp117-cv1-s02-1_5b-e8-lr3p162e-4-wd1p6-bs128-us-east1",
        exp117_loss=2.7489023208618164,
    ),
    Point(
        key="lr1e-3-wd0p2-bs64",
        learning_rate=1e-3,
        weight_decay=0.2,
        batch_size=64,
        exp117_run="prot-exp117-cv1-s02-1_5b-e8-lr1e-3-wd0p2-bs64-us-east5",
        exp117_loss=2.750558853149414,
    ),
)

# The one exp117 configuration whose final checkpoint is published on
# huggingface.co/open-athena/marinfold-exp117 and therefore staged into this prefix.
# exp166 pairs a configuration with the checkpoint from the exp117 run that used the same
# hyperparameters, so a configuration without a published checkpoint cannot be continued.
SEEDED_POINT_KEY: str = "lr3p162e-3-wd0p2-bs64"


class Initialization(StrEnum):
    SCRATCH = "scratch"
    EXP117 = "exp117init"


@dataclass(frozen=True)
class Trial:
    """One logical trial: a configuration, an initialization mode, and augmentation on/off."""

    point: Point
    initialization: Initialization
    augment: bool

    @property
    def trial_id(self) -> str:
        return f"{self.point.key}-{self.initialization.value}-{'aug' if self.augment else 'noaug'}"


def _build_trials() -> dict[str, Trial]:
    """Six from-scratch augmented runs, plus the augmented/unaugmented continuation pair."""
    seeded = next(p for p in POINTS if p.key == SEEDED_POINT_KEY)
    trials = [Trial(point, Initialization.SCRATCH, augment=True) for point in POINTS]
    trials.append(Trial(seeded, Initialization.EXP117, augment=True))
    trials.append(Trial(seeded, Initialization.EXP117, augment=False))
    return {trial.trial_id: trial for trial in trials}


TRIALS: dict[str, Trial] = _build_trials()


# --- Env inputs --------------------------------------------------------------


def parse_trial() -> Trial:
    key = os.environ.get("TRIAL", "").strip().lower()
    if key not in TRIALS:
        choices = "\n  ".join(TRIALS)
        raise SystemExit(f"TRIAL must be one of:\n  {choices}")
    return TRIALS[key]


def parse_cluster() -> ClusterSpec:
    name = os.environ.get("CLUSTER", "").strip().lower()
    if name not in CLUSTERS:
        raise SystemExit(f"CLUSTER must be one of: {', '.join(CLUSTERS)}")
    return CLUSTERS[name]


def parse_nodes() -> int:
    raw = os.environ.get("NODES")
    if raw is None:
        raise SystemExit("missing required env var NODES")
    nodes = int(raw)
    if nodes < 1:
        raise SystemExit(f"NODES must be >= 1, got {nodes}")
    if nodes > MAX_NODES:
        raise SystemExit(f"NODES must be <= {MAX_NODES} (gang ceiling), got {nodes}")
    if nodes & (nodes - 1):
        raise SystemExit(f"NODES must be a power of two; {nodes} yields a degenerate mesh")
    return nodes


def preview() -> bool:
    return os.environ.get("PREVIEW", "").strip().lower() in {"yes", "true", "1"}


def smoke() -> bool:
    return os.environ.get("SMOKE", "").strip().lower() in {"yes", "true", "1"}


def smoke_steps() -> int:
    steps = int(os.environ.get("SMOKE_STEPS", str(SMOKE_STEPS_DEFAULT)))
    if steps < 1:
        raise SystemExit(f"SMOKE_STEPS must be >= 1, got {steps}")
    return steps


def resource_ram() -> str | None:
    return os.environ.get(GPU_RAM_ENV) or None


def per_device_override() -> int | None:
    """Forced per-device microbatch for a ceiling search, or ``None`` to use the table."""
    raw = os.environ.get("PER_DEVICE")
    if raw is None:
        return None
    value = int(raw)
    if value < 1:
        raise SystemExit(f"PER_DEVICE must be >= 1, got {value}")
    return value


def max_seqs_per_device(spec: ClusterSpec) -> int:
    """Per-device sequence capacity for this GPU type, measured not guessed."""
    if (override := per_device_override()) is not None:
        if not smoke():
            raise SystemExit("PER_DEVICE is a calibration probe; it is only allowed with SMOKE=yes")
        return override
    capacity = MAX_SEQS_PER_DEVICE.get(spec.gpu_variant)
    if capacity is None:
        raise SystemExit(
            f"no measured capacity for {spec.gpu_variant} at this model size; measured types are "
            f"{sorted(MAX_SEQS_PER_DEVICE) or '(none)'}. Run the ceiling search on the smallest "
            "gang (SMOKE=yes with PER_DEVICE, stepping up until it stops surviving) and record "
            "the result in MAX_SEQS_PER_DEVICE before any production run."
        )
    return capacity


def batch_fit(trial: Trial, spec: ClusterSpec, nodes: int) -> GpuBatchConfig:
    """Resolve DP, TP, per-device parallelism and accumulation for one gang shape."""
    return gpu_batch_config(
        spec.gpus_per_node,
        nodes,
        trial.point.batch_size,
        max_seqs_per_device(spec),
    )


# --- Identity helpers --------------------------------------------------------


def run_id(trial: Trial) -> str:
    """W&B run name, trainer/resume id, and run-output subpath.

    Keyed on the trial alone. Cluster and node count are placement, not identity: every
    CoreWeave cluster reads the same bucket, so any re-dispatch resumes the same run from
    the same checkpoint. The ``exp166cw`` prefix keeps these clear of the exp166 runs the
    TPU attempt left in W&B, whose names all end in a region.
    """
    return f"{RUN_PREFIX}-{MODEL_SIZE}-e{EPOCHS}-{trial.trial_id}"


def smoke_run_id(trial: Trial, spec: ClusterSpec, nodes: int) -> str:
    per_device = per_device_override()
    probe = f"-pd{per_device}" if per_device is not None else ""
    return f"{SMOKE_RUN_PREFIX}-{MODEL_SIZE}-{trial.trial_id}-{spec.gpu_variant}x{nodes}{probe}-{SMOKE_VERSION}"


# --- Checkpoint initialization -----------------------------------------------


@dataclass(frozen=True)
class SeededCheckpointConfig:
    """The staged exp117 checkpoint an ``exp117init`` trial initializes from."""

    checkpoint_root: str


def _verify_seeded_checkpoint(config: SeededCheckpointConfig) -> None:
    """Confirm the staged exp117 seed is present and complete.

    Nothing is copied here and nothing may be: the seed is placed once, out of band, from
    the published HF checkpoint. A missing seed is a setup error, never something a
    training job repairs.
    """
    if not config.checkpoint_root.startswith(STORAGE_BUCKET):
        raise RuntimeError(
            f"exp117 seed root {config.checkpoint_root} is not on {STORAGE_BUCKET}; "
            "refusing to read weights across a cloud boundary"
        )

    checkpoints = f"{config.checkpoint_root.rstrip('/')}/checkpoints"
    checkpoint = latest_checkpoint_path(checkpoints)
    if checkpoint is None:
        raise FileNotFoundError(f"no staged exp117 seed under {checkpoints}; stage it before running this trial")

    fs, path = fsspec.core.url_to_fs(checkpoint)
    objects = {p: info for p, info in fs.find(path, detail=True).items() if info.get("type") != "directory"}
    if not objects:
        raise FileNotFoundError(f"empty exp117 seed at {checkpoint}")

    total_bytes = sum(int(info.get("size", 0)) for info in objects.values())
    logger.info("EXP166CW SEED path=%s objects=%d size=%.2fGiB", checkpoint, len(objects), total_bytes / 1024**3)


def exp117_checkpoint(point: Point) -> ArtifactStep[LevanterCheckpoint]:
    """Resolve the staged exp117 seed for this configuration.

    The seed is named for the exp117 run it came from, which records the exact source run
    and its origin region, and the ``exp166-init`` namespace keeps that name from colliding
    with this experiment's own run outputs.
    """
    return ArtifactStep(
        name=f"{SEED_NAMESPACE}/{point.exp117_run}",
        version=EXP117_VERSION,
        artifact_type=LevanterCheckpoint,
        run=_verify_seeded_checkpoint,
        build_config=lambda ctx: SeededCheckpointConfig(checkpoint_root=ctx.output_path),
    )


def initial_checkpoint(trial: Trial) -> ArtifactStep[LevanterCheckpoint] | None:
    if trial.initialization is Initialization.SCRATCH:
        return None
    return exp117_checkpoint(trial.point)


# --- Amino-acid augmentation -------------------------------------------------


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
        logger.info(
            "exp166cw AA augmentation runtime effect: documents=%d residue_statements=%d "
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
        # This wrapper sits outside MixtureDataset, so ``index`` is the global training
        # occurrence rather than the finite cache index. A document therefore gets a fresh
        # deterministic view each time the mixture restarts. NumPy owns the augmentation
        # RNG so data loading never mixes a CPU JAX key with the active device mesh.
        return AminoAcidAugmentedDataset(dataset, self.augmentation_seed)


def augment_amino_acids(data: LmDataConfig) -> LmDataConfig:
    """Enable training-only sequence-statement augmentation."""
    values = {field.name: getattr(data, field.name) for field in fields(LmDataConfig)}
    return AminoAcidAugmentedDataConfig(**values)


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


def _tags(trial: Trial, *, num_train_steps: int) -> list[str]:
    point = trial.point
    params = MODEL_CONFIG.total_trainable_params(VOCAB_SIZE)
    tags = [
        "protein",
        "exp166cw",
        "contacts-v1",
        "aa-augmentation",
        "qwen3",
        f"trial_id={trial.trial_id}",
        f"initialization={trial.initialization.value}",
        f"augmentation={trial.augment}",
        f"model_size={MODEL_SIZE}",
        f"global_batch={point.batch_size}",
        f"params={params}",
        f"epochs={EPOCHS}",
        f"lr={point.learning_rate:g}",
        f"wd={point.weight_decay:g}",
        f"exp117_loss={point.exp117_loss:.8f}",
        f"steps={num_train_steps}",
        f"tokens={point.batch_size * SEQ_LEN * num_train_steps}",
        f"version={SWEEP_VERSION}",
        f"cache_version={CACHE_VERSION}",
    ]
    if trial.initialization is Initialization.EXP117:
        tags.append(f"source_checkpoint=exp117/{point.key}")
    return tags


def production_shape(trial: Trial) -> RunShape:
    """Full run: two evals per epoch, one permanent checkpoint at the final step."""
    point = trial.point
    return RunShape(
        run_id=run_id(trial),
        wandb_group=WANDB_GROUP,
        num_train_steps=point.num_train_steps,
        steps_per_eval=point.steps_per_eval,
        # Levanter still forces the final save when ``keep`` is None. No per-epoch
        # permanents: the objective is the final val loss, and the R-precision-vs-tokens
        # curve that motivated per-epoch keeps belongs to #117.
        checkpoint_keep=None,
        max_eval_batches=None,
        tags=_tags(trial, num_train_steps=point.num_train_steps),
        mode="production",
    )


def smoke_shape(trial: Trial, spec: ClusterSpec, nodes: int, steps: int) -> RunShape:
    """Tiny, identity-isolated end-to-end run with evals and a final checkpoint."""
    tags = [
        *_tags(trial, num_train_steps=steps),
        "smoke",
        f"cluster_gpu={spec.gpu_variant}",
        f"nodes={nodes}",
        f"smoke_version={SMOKE_VERSION}",
    ]
    if (per_device := per_device_override()) is not None:
        tags.append(f"per_device_probe={per_device}")
    return RunShape(
        run_id=smoke_run_id(trial, spec, nodes),
        wandb_group=SMOKE_WANDB_GROUP,
        num_train_steps=steps,
        steps_per_eval=max(1, steps // SMOKE_NUM_EVALS),
        checkpoint_keep=None,
        max_eval_batches=SMOKE_MAX_EVAL_BATCHES,
        tags=tags,
        mode="smoke",
    )


def _tokenize_cache(name: str, docs: str, pin: str, *, validation: bool) -> ArtifactStep[TokenizedCache]:
    """A handle to the already-tokenized contacts-v1 cache.

    ``pin`` is an absolute location, so this resolves to the existing cache instead of
    tokenizing anything, and the step records no provenance — the pinned location is never
    written to. ``pack`` is applied later on the assembled data config.
    """
    return tokenized(
        name=name,
        tokenizer=TOKENIZER,
        version=CACHE_VERSION,
        paths=[docs],
        text_key=TEXT_KEY,
        validation=validation,
        pin=pin,
        tags=["protein", "contacts-v1", name],
    )


def _training_env() -> dict[str, str]:
    """W&B and storage metadata for the dispatched gang.

    The gang runs in a separate pod from the driver and does not inherit its environment,
    so ``MARIN_PREFIX`` must be forwarded explicitly — otherwise the workers fall back to
    the GCS cluster config and every write becomes cross-cloud egress.
    """
    env = {
        "WANDB_PROJECT": os.environ.get("WANDB_PROJECT", "marin"),
        "MARIN_PREFIX": STORAGE_PREFIX,
    }
    for key in ("WANDB_ENTITY", "WANDB_MODE"):
        if value := os.environ.get(key):
            env[key] = value
    return env


def _apply_recipe_overrides(
    step: ArtifactStep[LevanterCheckpoint],
    trial: Trial,
    shape: RunShape,
    spec: ClusterSpec,
    nodes: int,
) -> ArtifactStep[LevanterCheckpoint]:
    """Apply recipe settings not exposed directly by :func:`train_lm`.

    Batch fitting is runtime-only so gang-shape changes do not change run identity. Data,
    evaluation, checkpoint and W&B settings remain fingerprinted.
    """
    base_build_config = step.build_config

    def build_config(ctx):
        pod = base_build_config(ctx)
        trainer = replace(
            pod.train_config.trainer,
            max_eval_batches=shape.max_eval_batches,
            watch=WANDB_WATCH_CONFIG,
            checkpointer=replace(
                pod.train_config.trainer.checkpointer,
                keep=shape.checkpoint_keep,
                # Keep the ten-minute resumption checkpoints inside the run's own output
                # directory. The default routes them to the bucket root's `tmp/ttl=Nd/`,
                # outside the MarinFold prefix this experiment stays within (#108).
                temporary_base_path=None,
            ),
        )
        data = replace(
            pod.train_config.data,
            shuffle=SHUFFLE,
            components={
                key: replace(component, pack=True) for key, component in pod.train_config.data.components.items()
            },
        )
        if trial.augment:
            data = augment_amino_acids(data)
        if not ctx.is_fingerprint:
            config = batch_fit(trial, spec, nodes)
            trainer = replace(
                trainer,
                per_device_parallelism=config.per_device_parallelism,
                per_device_eval_parallelism=config.per_device_parallelism,
            )
        # An exp117-init trial is an ablation from pretrained weights, not a continuation
        # of exp117's finished optimizer and schedule. Move the staged checkpoint onto
        # Levanter's strict model-only path: fresh optimizer, fresh data order, step 0.
        initialize_model_from_checkpoint_path = pod.train_config.initialize_from_checkpoint_path
        train_config = replace(
            pod.train_config,
            trainer=trainer,
            data=data,
            data_seed=DATA_SEED,
            initialize_from_checkpoint_path=None,
            initialize_model_from_checkpoint_path=initialize_model_from_checkpoint_path,
            # One HF export, at the end. Levanter's export hook is installed with
            # ``every=hf_save_steps`` but still fires on the final step, so a period past
            # the run's length yields exactly one export.
            hf_save_steps=shape.num_train_steps + 1,
        )
        return replace(pod, train_config=train_config)

    return replace(step, build_config=build_config)


def build_run(trial: Trial, shape: RunShape, spec: ClusterSpec, nodes: int) -> ArtifactStep[LevanterCheckpoint]:
    """Assemble one production or smoke training run."""
    train_cache = _tokenize_cache(COMPONENT_TRAIN, TRAIN_DOCS, TRAIN_CACHE_PIN, validation=False)
    val_cache = _tokenize_cache(COMPONENT_VAL, VAL_DOCS, VAL_CACHE_PIN, validation=True)
    batch_config = batch_fit(trial, spec, nodes)
    point = trial.point
    step = train_lm(
        name=f"{CHECKPOINT_ROOT}/{shape.run_id}",
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
        tensor_parallel_size=batch_config.tensor_parallelism,
        seq_len=SEQ_LEN,
        num_train_steps=shape.num_train_steps,
        z_loss_weight=None,
        evals=None,
        resources=ResourceConfig.with_gpu(
            spec.gpu_variant,
            count=spec.gpus_per_node,
            replicas=nodes,
            cpu=spec.cpu,
            ram=resource_ram() or spec.ram,
            disk=spec.disk,
        ),
        version=SWEEP_VERSION,
        init_from=initial_checkpoint(trial),
        steps_per_eval=shape.steps_per_eval,
        wandb_project=os.environ.get("WANDB_PROJECT", "marin"),
        wandb_group=shape.wandb_group,
        run_id=shape.run_id,
        tags=shape.tags,
        env_vars=_training_env(),
    )
    if smoke():
        step = replace(step, override_path=smoke_output_path(shape.run_id))
    return _apply_recipe_overrides(step, trial, shape, spec, nodes)


def smoke_output_path(run_id_: str) -> str:
    """Temp-storage output path for a smoke or calibration run.

    A smoke produces a checkpoint nobody wants to keep and the experiment prefix has no
    lifecycle rules, so this output goes to the bucket's TTL'd temp area and expires on its
    own. A deliberate exception to keeping everything under ``MarinFold/`` (#108): the
    point of that rule is bulk removability, which a one-day lifecycle satisfies more
    reliably than a manual sweep.
    """
    return marin_temp_bucket(SMOKE_TTL_DAYS, f"{CHECKPOINT_ROOT}/{run_id_}/{SWEEP_VERSION}")


def _output_path(shape: RunShape) -> str:
    if smoke():
        return smoke_output_path(shape.run_id)
    return f"{STORAGE_PREFIX}/{CHECKPOINT_ROOT}/{shape.run_id}/{SWEEP_VERSION}"


def checkpoint_bytes(shape: RunShape) -> int:
    """Storage one completed run leaves behind: one permanent save plus one HF export."""
    permanents = 1 if shape.checkpoint_keep is None else max(1, shape.num_train_steps)
    return permanents * LEVANTER_CHECKPOINT_BYTES + HF_EXPORT_BYTES


def assert_paths_on_bucket(trial: Trial, shape: RunShape) -> dict[str, str]:
    """Resolve every path this run touches and refuse to proceed if one leaves the bucket.

    A cross-cloud write is expensive and effectively irreversible, so this fails before
    submission rather than surfacing on the storage bill.
    """
    output = _output_path(shape)
    paths = {
        "marin_prefix": marin_prefix(),
        "output": output,
        "temporary_checkpoints": temporary_checkpoint_base_path(output),
        "compilation_cache": f"{marin_prefix().rstrip('/')}/compilation-cache",
        "train_cache": TRAIN_CACHE_PIN,
        "val_cache": VAL_CACHE_PIN,
        "train_docs": TRAIN_DOCS,
        "val_docs": VAL_DOCS,
    }
    if trial.initialization is Initialization.EXP117:
        paths["exp117_seed"] = f"{STORAGE_PREFIX}/{SEED_NAMESPACE}/{trial.point.exp117_run}/{EXP117_VERSION}"
    for name, path in paths.items():
        if not path.startswith(STORAGE_BUCKET):
            raise SystemExit(
                f"path {name!r} resolved to {path!r}, which is not on {STORAGE_BUCKET}. "
                "Refusing to run: writing from CoreWeave to another cloud is cross-cloud "
                "egress. Check that MARIN_PREFIX reaches the gang."
            )
    # The pinned caches are read-only by construction, but a pin pointed at this
    # experiment's own prefix would mean the cache was copied after all, which the plan
    # rules out. Fail loudly rather than silently diverging from it.
    for name in ("train_cache", "val_cache"):
        if paths[name].startswith(STORAGE_PREFIX):
            raise SystemExit(f"{name} {paths[name]!r} is inside {STORAGE_PREFIX}; the cache is pinned, not copied")
    return paths


# --- Preview + entry point ---------------------------------------------------


def _print_preview(trial: Trial, shape: RunShape, spec: ClusterSpec, nodes: int) -> None:
    config = batch_fit(trial, spec, nodes)
    point = trial.point
    params = MODEL_CONFIG.total_trainable_params(VOCAB_SIZE)
    tokens = point.batch_size * SEQ_LEN * shape.num_train_steps
    init = (
        f"exp117 seed {trial.point.exp117_run}"
        if trial.initialization is Initialization.EXP117
        else "random initialization"
    )
    print(
        f"PREVIEW exp166cw [{shape.mode}] -- no submit\n"
        f"  trial_id={trial.trial_id}\n"
        f"  run_id={shape.run_id} wandb_group={shape.wandb_group}\n"
        f"  model_size={MODEL_SIZE} epochs={EPOCHS} lr={point.learning_rate:g} "
        f"wd={point.weight_decay:g} batch_size={point.batch_size}\n"
        f"  initialization={init}\n"
        f"  augmentation={'on' if trial.augment else 'OFF (control)'}\n"
        f"  exp117_loss={point.exp117_loss:.8f}\n"
        f"  steps={shape.num_train_steps} (steps/eval={shape.steps_per_eval}, "
        f"permanent ckpts=final only)\n"
        f"  storage={checkpoint_bytes(shape) / 1024**3:.2f} GiB "
        f"(1 HF export at the final step; nothing here expires automatically)\n"
        f"  tokens={tokens / 1e9:.3f}B params={params / 1e9:.3f}B "
        f"schedule={LR_SCHEDULE} warmup={WARMUP}\n"
        f"  gpu={spec.gpu_variant}x{spec.gpus_per_node} nodes={nodes} "
        f"devices={spec.gpus_per_node * nodes}\n"
        f"  attn_backend={MODEL_CONFIG.attn_backend}\n"
        f"  data_parallelism={config.data_parallelism} "
        f"tensor_parallelism={config.tensor_parallelism}\n"
        f"  per_device_parallelism={config.per_device_parallelism} "
        f"gradient_accumulation={config.gradient_accumulation}\n"
        f"  output={_output_path(shape)}\n"
        f"  storage_paths_verified_on={STORAGE_BUCKET}\n"
        f"  objective=eval/{COMPONENT_VAL}/loss (final step, minimize)",
        flush=True,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    # Set before any path resolution: the driver pod's ambient config points at GCS.
    os.environ["MARIN_PREFIX"] = STORAGE_PREFIX

    trial = parse_trial()
    spec = parse_cluster()
    nodes = parse_nodes()
    shape = smoke_shape(trial, spec, nodes, smoke_steps()) if smoke() else production_shape(trial)

    checked = assert_paths_on_bucket(trial, shape)

    if preview():
        _print_preview(trial, shape, spec, nodes)
        for name, path in checked.items():
            print(f"    {name}: {path}", flush=True)
        return

    StepRunner().run([lower(build_run(trial, shape, spec, nodes))])


if __name__ == "__main__":
    main()
