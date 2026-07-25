# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Single-point trainer for the exp153 contacts-v1 6B adaptive sweep on CoreWeave GPUs.

Each invocation trains one ``(epochs, lr, wd, batch_size)`` point on an explicit GPU gang
(a cluster plus a node count). ``SMOKE=yes`` runs the same path under an isolated identity.

Unlike the TPU sweeps this ports from (exp117 / exp146), the cluster is **placement only**
and never enters run identity: every CoreWeave cluster reads the same object-storage bucket,
so a trial can move between clusters and resume its checkpoint. There is no regional run
identity and no cross-region restart.

Preview a point (builds and submits nothing)::

    EPOCHS=8 LR=1e-3 WD=0.1 BATCH_SIZE=128 \
        CLUSTER=cw-us-east-02a NODES=4 PREVIEW=yes \
        uv run python -m experiments.protein.exp153_sweep

Launch one run. The driver is a small CPU job **inside** the target cluster: Iris only
federates root jobs, so the training gang lands wherever the driver runs, and ``--priority
batch`` on the driver is inherited by the gang (the scheduler resolves an unspecified child
band up the parent chain)::

    set -a; source ~/marin.env; set +a
    uv run iris --config lib/iris/config/marin.yaml job run \
        --user "$USERNAME" --target-cluster cw-us-east-02a --priority batch \
        --no-wait --memory 3GB \
        -e WANDB_API_KEY "$WANDB_API_KEY" -e HUGGING_FACE_HUB_TOKEN "$HF_TOKEN" \
        -e WANDB_ENTITY "$WANDB_ENTITY" -e WANDB_PROJECT "$WANDB_PROJECT" \
        -e EPOCHS 8 -e LR 1e-3 -e WD 0.1 -e BATCH_SIZE 128 \
        -e CLUSTER cw-us-east-02a -e NODES 4 \
        -- python -m experiments.protein.exp153_sweep

Smoke-test one gang shape under an isolated identity. Point coordinates are optional in
smoke mode but honored when supplied; ``SMOKE_STEPS`` overrides the step budget::

    ... -e SMOKE yes -e CLUSTER cw-us-east-02a -e NODES 1 \
        -- python -m experiments.protein.exp153_sweep

Tokenize the corpus without training (run this once, before the first trial)::

    ... -e TOKENIZE_ONLY yes -- python -m experiments.protein.exp153_sweep
"""

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TypeVar

from fray.cluster import ResourceConfig
from levanter.callbacks.watch import WatchConfig
from levanter.data.text.datasets import BlockShuffleConfig
from levanter.layers.attention import AttentionBackend
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim.config import AdamConfig
from marin.execution.lazy import ArtifactStep, lower
from marin.execution.step_runner import StepRunner
from marin.experiment.data import tokenized
from marin.experiment.train import train_lm
from marin.processing.tokenize.tokenize import TokenizedCache
from marin.training.training import LevanterCheckpoint, temporary_checkpoint_base_path
from rigging.filesystem import marin_prefix, marin_temp_bucket

from experiments.coral.batch_calibration import GpuBatchConfig, gpu_batch_config

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


# --- Storage -----------------------------------------------------------------

# Everything this experiment writes lives under one MarinFold-prefixed directory so it can
# be bulk-removed later (MarinFold #108's standing rule). The directory is named for the
# parent scaling sweep (#154); the runs inside it are exp153.
STORAGE_PREFIX: str = "s3://marin-us-east-02a/MarinFold/exp154_qwen_contacts_v1"

# Every byte this experiment reads or writes must live on this CoreWeave bucket. The guard
# below is not defensive boilerplate: a CoreWeave pod resolves ``data_config()`` to the GCS
# cluster config (scheme ``gs``, GCP region buckets), because ``MARIN_CLUSTER`` is unset
# there and the default is ``marin``. Paths stay on S3 only because ``MARIN_PREFIX`` is
# forwarded to the gang; if that ever stops happening, path resolution silently falls back
# toward GCS and a 96 GB checkpoint write becomes cross-cloud egress.
STORAGE_BUCKET: str = "s3://marin-us-east-02a/"

# Raw contacts-v1 parquet, staged to the CoreWeave bucket by MarinFold exp108.
_DOCS_BASE: str = "s3://marin-us-east-02a/MarinFold/data/document_structures/contacts_v1"
TRAIN_DOCS: str = f"{_DOCS_BASE}/train/*.parquet"
VAL_DOCS: str = f"{_DOCS_BASE}/val/*.parquet"


# --- Identity ----------------------------------------------------------------

# Sweep and data identities are deliberately independent. Bump SWEEP_SUBVERSION for a
# fresh campaign; change CACHE_VERSION only when the tokenizer or source documents change.
VERSION_DATE: str = "2026.07.25"
SWEEP_SUBVERSION: str = "01"
SWEEP_VERSION: str = f"{VERSION_DATE}.{SWEEP_SUBVERSION}"
CACHE_VERSION: str = "2026.07.25"
DATA_SUBVERSION: str = "1"
RUN_PREFIX: str = "prot-exp153"
WANDB_GROUP: str = "exp153-contacts-v1-6b-tune"
CHECKPOINT_ROOT: str = "checkpoints"

SMOKE_RUN_PREFIX: str = "prot-exp153-smoke"
SMOKE_WANDB_GROUP: str = "exp153-smoke"
# Manual smoke identity token. Bump to fork a clean smoke run after a recipe/library change,
# otherwise the executor prunes the step as already-succeeded and the "run" is a cache hit.
# v2: main merged in (229 commits) 2026-07-25.
SMOKE_VERSION: str = "v2"
SMOKE_STEPS_DEFAULT: int = 20
# Lifecycle window for smoke/calibration output only. Production checkpoints are
# permanent and live under STORAGE_PREFIX; they are never routed through temp storage.
SMOKE_TTL_DAYS: int = 1
SMOKE_NUM_EVALS: int = 2
SMOKE_POINT_DEFAULTS: tuple[int, float, float, int] = (1, 1e-3, 0.1, 128)

# Optional resource-only host RAM override, passed through to the gang's ResourceConfig.
GPU_RAM_ENV: str = "GPU_RAM"


# --- Data --------------------------------------------------------------------

# contacts-v1 tokenizer (2845 vocab), pinned to the immutable revision the corpus is
# tokenized with.
TOKENIZER: str = "timodonnell/contacts-v1-tokenizer@5d68a24a899f"
VOCAB_SIZE: int = 2845
TEXT_KEY: str = "document"

# Cache names double as the storage subpath under STORAGE_PREFIX and as the mixture
# component key / eval series prefix (``eval/<name>/loss``).
COMPONENT_TRAIN: str = "tokenized/contacts-v1"
COMPONENT_VAL: str = "tokenized/contacts-v1-val"

# Exact train-corpus token count for contacts-v1 at TOKENIZER, from the #117 cache.
# Steps/epoch are derived from this, not estimated.
TRAIN_TOKENS: int = 4_676_753_425

# Zephyr fan-out for the tokenize step. The CoreWeave CPU pools are small — cw-us-east-02a
# has 4 cd-gp-a192-genoa nodes (192 vCPU each) that also host the controller — so the 4096
# default would leave most workers pending. The coordinator override is required, not a
# tuning choice: Kubernetes enforces the memory request as a hard limit and the task image
# runs a full workspace `uv sync` at startup, which OOM-kills Zephyr's small default.
TOKENIZE_MAX_WORKERS: int = 256
TOKENIZE_COORDINATOR_RESOURCES = ResourceConfig(cpu=1, ram="6g", disk="16g", preemptible=False)


# --- Model -------------------------------------------------------------------

MODEL_SIZE: str = "6b"
# Architecture is fixed by MarinFold #153 — do not retune it here.
MODEL_CONFIG = Qwen3Config(
    max_seq_len=8192,
    hidden_dim=3200,
    intermediate_dim=12800,
    num_layers=37,
    num_heads=64,
    num_kv_heads=32,
    head_dim=64,
    rope=Llama3RotaryEmbeddingsConfig(),
    # On GPU levanter defaults to the Transformer Engine backend, which is absent from the
    # `gpu` extra and silently falls back to the O(seq^2) reference kernel — unusable at
    # seq 8192 (MarinFold #108; upstream marin#7013 is still open). JAX_FLASH is the
    # memory-safe blocked-flash path that needs no TE.
    attn_backend=AttentionBackend.JAX_FLASH,
)


# --- Training recipe ---------------------------------------------------------

SEQ_LEN: int = 8192
DATA_SEED: int = 0
SHUFFLE = BlockShuffleConfig(io_block_size=256, window_blocks=512, perm_type="feistel")
WARMUP: float = 0.1
LR_SCHEDULE: str = "cosine"
NUM_EVALS_PER_EPOCH: int = 2
WANDB_WATCH_CONFIG = WatchConfig(watch_targets=[], interval=0)

# Measured on the 2026-07-25 1-node smoke: one permanent save is ~72 GB of Levanter state
# (params + Adam moments, fp32) and an HF export is ~24 GB. Used only to report expected
# storage in the preview.
LEVANTER_CHECKPOINT_BYTES: int = 72 * 1024**3
HF_EXPORT_BYTES: int = 24 * 1024**3

# Measured sequence capacity per GPU: the largest per-device microbatch under which a run
# survives training AND the eval pass AND a checkpoint save, which are three distinct memory
# peaks. One number per GPU type, measured once on the SMALLEST supported gang, and valid
# for every larger one.
#
# Why one measurement extrapolates. Per-device memory is roughly
#
#     mem(dp, pd) ~= sharded_state/dp + per_seq_activations*pd + fixed
#
# because parameters, gradients, and optimizer state are FSDP-sharded over the data axis
# while activations are not. Capacity is therefore monotonically non-decreasing in gang
# size: a microbatch that fits on the smallest gang fits on every larger one, since the
# only term that grows per-device is the one that shrinks with dp. Calibrating at the
# smallest gang is thus conservative by construction rather than by guesswork.
#
# Being conservative costs nothing here. Measured 2026-07-25 at 8xH100: pd=4 (ga=4) and
# pd=8 (ga=2) both run at ~33.5 s/step, so throughput is flat in the microbatch. The only
# job this number has is to avoid an OOM, not to find an optimum.
#
# H100 = 8: pd=8 trains, evaluates and checkpoints cleanly at 8 devices; pd=16 dies in the
# allocator (a single 61.6 GiB request). The true ceiling is in [8, 16); 8 is taken as the
# measured value rather than bisecting further, since nothing is gained by going higher.
#
# Specific to this model at this sequence length, optimizer, precision, and remat setting --
# do not transplant it. An unmeasured GPU type fails loudly rather than guessing, because
# guessing wrong costs a multi-day trial.
MAX_SEQS_PER_DEVICE: dict[str, int] = {
    "H100": 8,
}


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
    # gd-8xh100ib-i128: 8x H100-80GB + InfiniBand, 128 vCPU / 2 TB per node.
    "cw-us-east-02a": ClusterSpec("H100", 8, cpu=32, ram="256g", disk="256g"),
    # gb200-4x: 4x GB200-186GB per node.
    "cw-us-east-08a": ClusterSpec("GB200", 4, cpu=32, ram="256g", disk="256g"),
}


# --- A single trial point ----------------------------------------------------


@dataclass(frozen=True)
class Point:
    """One sweep point."""

    epochs: int
    learning_rate: float
    weight_decay: float
    batch_size: int

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
        return f"e{self.epochs}-lr{_fmt_lr(self.learning_rate)}-wd{_fmt_wd(self.weight_decay)}-bs{self.batch_size}"


def run_id(point: Point, prefix: str = RUN_PREFIX) -> str:
    """The W&B run name, trainer/resume id, and run-output subpath for a point.

    Keyed on the point alone. Cluster and node count are placement, not identity: every
    CoreWeave cluster reads the same bucket, so any re-dispatch — same cluster or not —
    resumes the same run from the same checkpoint. Run outputs land at
    ``{STORAGE_PREFIX}/checkpoints/{run_id}/{SWEEP_VERSION}/``.
    """
    return f"{prefix}-cv1-s{SWEEP_SUBVERSION}-{MODEL_SIZE}-{point.point_id}"


# --- Env inputs (one point per launch) ---------------------------------------


def _env_value(name: str, cast: Callable[[str], _T], default: _T | None) -> _T:
    """Read env var ``name`` through ``cast``; fall back to ``default`` (``None`` == required)."""
    raw = os.environ.get(name)
    if raw is not None:
        return cast(raw)
    if default is None:
        raise SystemExit(f"missing required env var '{name}'; set EPOCHS, LR, WD, BATCH_SIZE, CLUSTER, NODES")
    return default


def parse_point(defaults: Point | None = None) -> Point:
    """Read and validate one hyperparameter point from the environment."""
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
    return Point(epochs=epochs, learning_rate=learning_rate, weight_decay=weight_decay, batch_size=batch_size)


def parse_cluster() -> ClusterSpec:
    cluster = os.environ.get("CLUSTER")
    if not cluster:
        raise SystemExit(f"missing required env var CLUSTER (one of {sorted(CLUSTERS)})")
    spec = CLUSTERS.get(cluster)
    if spec is None:
        raise SystemExit(f"unknown CLUSTER {cluster!r}; expected one of {sorted(CLUSTERS)}")
    return spec


def parse_nodes() -> int:
    nodes = _env_value("NODES", int, None)
    if nodes < 1:
        raise SystemExit(f"NODES must be >= 1, got {nodes}")
    return nodes


def preview() -> bool:
    return os.environ.get("PREVIEW", "").strip().lower() in {"yes", "true", "1"}


def smoke() -> bool:
    return os.environ.get("SMOKE", "").strip().lower() in {"yes", "true", "1"}


def tokenize_only() -> bool:
    return os.environ.get("TOKENIZE_ONLY", "").strip().lower() in {"yes", "true", "1"}


def resource_ram() -> str | None:
    """Optional GPU-worker host RAM override, e.g. ``512g``."""
    ram = os.environ.get(GPU_RAM_ENV)
    if ram is None:
        return None
    return ram.strip() or None


def smoke_steps() -> int:
    steps = _env_value("SMOKE_STEPS", int, SMOKE_STEPS_DEFAULT)
    if steps < 1:
        raise SystemExit(f"SMOKE_STEPS must be >= 1, got {steps}")
    return steps


def per_device_override() -> int | None:
    """Forced per-device microbatch for a ceiling search, or ``None`` to use the table.

    Only meaningful in smoke mode: it is how an unmeasured ``(gpu, devices)`` combination
    gets probed, stepping the value up until the run stops surviving.
    """
    raw = os.environ.get("PER_DEVICE")
    if raw is None:
        return None
    value = int(raw)
    if value < 1:
        raise SystemExit(f"PER_DEVICE must be >= 1, got {value}")
    return value


def max_seqs_per_device(spec: ClusterSpec, nodes: int) -> int:
    """Per-device sequence capacity for this gang shape.

    One measured number per GPU type, applied at every gang size: capacity only grows with
    the gang (see :data:`MAX_SEQS_PER_DEVICE`), so the single-gang measurement is a safe
    lower bound everywhere. ``nodes`` is accepted to make that dependence explicit at the
    call site even though the answer does not currently vary with it.
    """
    if (override := per_device_override()) is not None:
        if not smoke():
            raise SystemExit("PER_DEVICE is a calibration probe; it is only allowed with SMOKE=yes")
        return override
    capacity = MAX_SEQS_PER_DEVICE.get(spec.gpu_variant)
    if capacity is None:
        raise SystemExit(
            f"no measured capacity for {spec.gpu_variant}; measured types are "
            f"{sorted(MAX_SEQS_PER_DEVICE)}. Run the ceiling search on the smallest gang "
            "(SMOKE=yes with PER_DEVICE, stepping up until it OOMs) before using this GPU."
        )
    return capacity


# --- Helpers -----------------------------------------------------------------


def _fmt_lr(lr: float) -> str:
    """Compact, path-safe LR tag to ~4 sig figs, e.g. ``1e-2``, ``3p162e-3``."""
    mantissa, exponent = f"{lr:.3e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".").replace(".", "p")
    return f"{mantissa}e{int(exponent)}"


def _fmt_wd(wd: float) -> str:
    """Path-safe weight-decay tag, e.g. ``0.1`` -> ``0p1``, ``1.6`` -> ``1p6``."""
    return f"{wd:g}".replace(".", "p")


def batch_fit(point: Point, spec: ClusterSpec, nodes: int) -> GpuBatchConfig:
    """Resolve DP, TP, per-device parallelism, and accumulation for one gang shape."""
    return gpu_batch_config(
        spec.gpus_per_node,
        nodes,
        point.batch_size,
        max_seqs_per_device(spec, nodes),
    )


def smoke_output_path(run_id_: str) -> str:
    """Temp-storage output path for a smoke or calibration run.

    Smoke runs produce a ~96 GB checkpoint that nobody wants to keep, and the experiment
    prefix has no lifecycle rules, so their output goes to the bucket's TTL'd temp area
    instead and expires on its own. This is a deliberate exception to keeping everything
    under ``MarinFold/`` (#108): the point of that rule is bulk removability, which a
    one-day lifecycle satisfies more reliably than a manual sweep would.
    """
    return marin_temp_bucket(SMOKE_TTL_DAYS, f"{CHECKPOINT_ROOT}/{run_id_}/{SWEEP_VERSION}")


def _tokenize_cache(name: str, docs: str, *, validation: bool) -> ArtifactStep[TokenizedCache]:
    """Tokenize a split of the raw documents into a levanter cache handle.

    ``name`` is the cache storage subpath and the mixture component key (see
    COMPONENT_TRAIN/VAL). ``pack`` is applied later on the assembled data config.
    """
    return tokenized(
        name=name,
        tokenizer=TOKENIZER,
        version=CACHE_VERSION,
        paths=[docs],
        text_key=TEXT_KEY,
        validation=validation,
        tags=["protein", "contacts-v1", name],
        max_workers=TOKENIZE_MAX_WORKERS,
        coordinator_resources=TOKENIZE_COORDINATOR_RESOURCES,
    )


def _training_env() -> dict[str, str]:
    """W&B and storage metadata for the dispatched gang.

    The gang runs in a separate pod from the driver and does not inherit its environment,
    so ``MARIN_PREFIX`` must be forwarded explicitly — otherwise the workers fall back to
    the cluster default (``s3://marin-us-east-02a/marin``) and write outside MarinFold.
    Secrets come from the iris command, not from here.
    """
    env = {
        "MARIN_PREFIX": STORAGE_PREFIX,
        "WANDB_PROJECT": os.environ.get("WANDB_PROJECT", "marin"),
    }
    for key in ("WANDB_ENTITY", "WANDB_MODE"):
        if value := os.environ.get(key):
            env[key] = value
    return env


def _tags(point: Point, *, num_train_steps: int) -> list[str]:
    # Stable, identity-bearing facts only. Cluster and node count are deliberately omitted:
    # a run may migrate clusters over its life, so they are neither stable tags nor part of
    # run identity.
    params = MODEL_CONFIG.total_trainable_params(VOCAB_SIZE)
    tokens = point.batch_size * SEQ_LEN * num_train_steps
    return [
        "protein",
        "exp153",
        "contacts-v1",
        "qwen3",
        "coreweave",
        f"model_size={MODEL_SIZE}",
        f"global_batch={point.batch_size}",
        f"params={params}",
        f"epochs={point.epochs}",
        f"lr={point.learning_rate:g}",
        f"wd={point.weight_decay:g}",
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
    """The mode-dependent shape of one dispatched run: identity plus step/eval/ckpt cadence."""

    run_id: str
    wandb_group: str
    num_train_steps: int
    steps_per_eval: int
    checkpoint_keep: list[dict[str, int]] | None
    checkpoint_policy: str
    tags: list[str]


def _production_checkpoint_policy(point: Point) -> tuple[list[dict[str, int]] | None, str]:
    """Return the permanent checkpoint keep policy for a production sweep point.

    Levanter still saves the forced final checkpoint when ``keep`` is ``None``;
    only 8-epoch runs get periodic permanent checkpoints before the final save, because the
    R-precision-vs-tokens analysis (#117) needs the intermediate points and the top rung is
    where they are worth 72 GB each. Every other rung keeps the final checkpoint only.
    """
    if point.epochs == 8:
        return [{"every": point.steps_per_epoch}], f"every {point.steps_per_epoch} steps"
    return None, "final only"


def checkpoint_bytes(shape: RunShape, point: Point) -> int:
    """Estimated storage one completed run leaves behind.

    The bucket has no lifecycle rule under ``MarinFold/`` and no object versioning, so
    nothing here is reclaimed automatically — surfacing the number in the preview is what
    keeps the sweep's storage bill from being a surprise. Sizes are measured from the
    2026-07-25 smoke: 72 GB of Levanter state and 24 GB of HF export per save.
    """
    permanents = 1
    if shape.checkpoint_keep:
        permanents = max(1, shape.num_train_steps // point.steps_per_epoch)
    return permanents * LEVANTER_CHECKPOINT_BYTES + HF_EXPORT_BYTES


def production_shape(point: Point) -> RunShape:
    """Full point with two evals per epoch and epoch checkpoints only for 8ep runs."""
    checkpoint_keep, checkpoint_policy = _production_checkpoint_policy(point)
    return RunShape(
        run_id=run_id(point),
        wandb_group=WANDB_GROUP,
        num_train_steps=point.num_train_steps,
        steps_per_eval=point.steps_per_eval,
        checkpoint_keep=checkpoint_keep,
        checkpoint_policy=checkpoint_policy,
        tags=_tags(point, num_train_steps=point.num_train_steps),
    )


def smoke_shape(point: Point, steps: int, spec: ClusterSpec, nodes: int) -> RunShape:
    """Tiny, identity-isolated end-to-end run with evals and a final checkpoint.

    Reuses the real caches, model, and batch fit so it validates the actual launch path,
    but under a smoke-only identity. The gang shape, resolved microbatch, and
    ``SMOKE_VERSION`` are folded into the run id so smoke re-runs do not collide across
    resource or recipe changes.

    Smoke runs still evaluate and still write a checkpoint, deliberately: the eval pass and
    the checkpoint save are memory peaks distinct from the training step, so a capacity
    measured without them would not be the capacity that matters. Their output goes to
    TTL'd temp storage (:func:`smoke_output_path`) rather than the experiment prefix.
    """
    every = max(1, steps // SMOKE_NUM_EVALS)
    base = run_id(point, SMOKE_RUN_PREFIX)
    pd = batch_fit(point, spec, nodes).per_device_parallelism
    identity = f"{base}-{spec.gpu_variant.lower()}x{spec.gpus_per_node}n{nodes}-pd{pd}-{SMOKE_VERSION}"
    tags = [
        *_tags(point, num_train_steps=steps),
        "smoke",
        f"gpu={spec.gpu_variant}",
        f"nodes={nodes}",
        f"per_device_parallelism={pd}",
        f"smoke_version={SMOKE_VERSION}",
    ]
    return RunShape(
        run_id=identity,
        wandb_group=SMOKE_WANDB_GROUP,
        num_train_steps=steps,
        steps_per_eval=every,
        checkpoint_keep=None,
        checkpoint_policy="final only",
        tags=tags,
    )


def _apply_recipe_overrides(
    step: ArtifactStep[LevanterCheckpoint],
    point: Point,
    shape: RunShape,
    spec: ClusterSpec,
    nodes: int,
) -> ArtifactStep[LevanterCheckpoint]:
    """Apply recipe settings not exposed directly by :func:`train_lm`.

    Batch fitting is runtime-only so gang-shape changes do not change run identity. Data,
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
                keep=shape.checkpoint_keep,
                # Keep time-policy checkpoints inside the run's own output directory.
                # The default routes them to the bucket root's `tmp/ttl=Nd/`, outside the
                # MarinFold prefix this experiment is required to stay within (#108).
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
        if not ctx.is_fingerprint:
            config = batch_fit(point, spec, nodes)
            trainer = replace(
                trainer,
                per_device_parallelism=config.per_device_parallelism,
                per_device_eval_parallelism=config.per_device_parallelism,
            )
        train_config = replace(
            pod.train_config,
            trainer=trainer,
            data=data,
            data_seed=DATA_SEED,
            # One HF export, at the end. Levanter's export hook is installed with
            # ``every=hf_save_steps`` but still fires on the final step, so a period past
            # the run's length yields exactly one export. The default (10,000) would emit
            # three extra 24 GB copies of an 8-epoch run for no new information: the HF
            # format is derived, regenerable from any Levanter checkpoint, and only needed
            # for runs actually exported.
            hf_save_steps=shape.num_train_steps + 1,
        )
        return replace(pod, train_config=train_config)

    return replace(step, build_config=build_config)


def build_run(point: Point, shape: RunShape, spec: ClusterSpec, nodes: int) -> ArtifactStep[LevanterCheckpoint]:
    """Assemble one production or smoke training run."""
    train_cache = _tokenize_cache(COMPONENT_TRAIN, TRAIN_DOCS, validation=False)
    val_cache = _tokenize_cache(COMPONENT_VAL, VAL_DOCS, validation=True)
    batch_config = batch_fit(point, spec, nodes)
    resource_kwargs = {"cpu": spec.cpu, "ram": resource_ram() or spec.ram, "disk": spec.disk}
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
            **resource_kwargs,
        ),
        version=SWEEP_VERSION,
        steps_per_eval=shape.steps_per_eval,
        wandb_project=os.environ.get("WANDB_PROJECT", "marin"),
        wandb_group=shape.wandb_group,
        run_id=shape.run_id,
        tags=shape.tags,
        env_vars=_training_env(),
    )
    if smoke():
        step = replace(step, override_path=smoke_output_path(shape.run_id))
    return _apply_recipe_overrides(step, point, shape, spec, nodes)


# --- Preview + entry point ---------------------------------------------------


def assert_paths_on_bucket(shape: RunShape) -> dict[str, str]:
    """Resolve every path this run touches and refuse to proceed if one leaves the bucket.

    Returns the resolved paths so the preview can show exactly what was checked. Raises
    ``SystemExit`` on the first path that is not on :data:`STORAGE_BUCKET` — a cross-cloud
    write is expensive and effectively irreversible, so this fails before submission rather
    than surfacing as a surprise on the storage bill.
    """
    output = _output_path(shape)
    paths = {
        "marin_prefix": marin_prefix(),
        "output": output,
        "temporary_checkpoints": temporary_checkpoint_base_path(output),
        "compilation_cache": f"{marin_prefix().rstrip('/')}/compilation-cache",
        "train_cache": f"{STORAGE_PREFIX}/{COMPONENT_TRAIN}/{CACHE_VERSION}",
        "val_cache": f"{STORAGE_PREFIX}/{COMPONENT_VAL}/{CACHE_VERSION}",
        "train_docs": TRAIN_DOCS,
        "val_docs": VAL_DOCS,
    }
    for name, path in paths.items():
        if not path.startswith(STORAGE_BUCKET):
            raise SystemExit(
                f"path {name!r} resolved to {path!r}, which is not on {STORAGE_BUCKET}. "
                "Refusing to run: writing from CoreWeave to another cloud is cross-cloud "
                "egress. Check that MARIN_PREFIX reaches the gang."
            )
    return paths


def _output_path(shape: RunShape) -> str:
    """Where this run's checkpoints land — temp storage for smokes, the prefix otherwise."""
    if smoke():
        return smoke_output_path(shape.run_id)
    return f"{STORAGE_PREFIX}/{CHECKPOINT_ROOT}/{shape.run_id}/{SWEEP_VERSION}"


def _print_preview(point: Point, shape: RunShape, spec: ClusterSpec, nodes: int, mode: str) -> None:
    config = batch_fit(point, spec, nodes)
    params = MODEL_CONFIG.total_trainable_params(VOCAB_SIZE)
    tokens = point.batch_size * SEQ_LEN * shape.num_train_steps
    print(
        f"PREVIEW exp153 [{mode}] -- no submit\n"
        f"  run_id={shape.run_id} wandb_group={shape.wandb_group}\n"
        f"  model_size={MODEL_SIZE} epochs={point.epochs} lr={point.learning_rate:g} "
        f"wd={point.weight_decay:g} batch_size={point.batch_size}\n"
        f"  steps={shape.num_train_steps} (steps/eval={shape.steps_per_eval}, "
        f"permanent ckpts={shape.checkpoint_policy})\n"
        f"  storage={checkpoint_bytes(shape, point) / 1024**4:.2f} TiB "
        f"(1 HF export at the final step; nothing here expires automatically)\n"
        f"  tokens={tokens / 1e9:.3f}B params={params / 1e9:.3f}B "
        f"schedule={LR_SCHEDULE} warmup={WARMUP}\n"
        f"  gpu={spec.gpu_variant}x{spec.gpus_per_node} nodes={nodes} "
        f"devices={spec.gpus_per_node * nodes}\n"
        f"  output={_output_path(shape)}\n"
        f"  data_parallelism={config.data_parallelism} "
        f"tensor_parallelism={config.tensor_parallelism}\n"
        f"  per_device_parallelism={config.per_device_parallelism} "
        f"gradient_accumulation={config.gradient_accumulation}\n"
        f"  resource_ram={resource_ram() or spec.ram}\n"
        f"  storage_paths_verified_on={STORAGE_BUCKET}\n"
        f"  objective=eval/{COMPONENT_VAL}/loss (final step, minimize)",
        flush=True,
    )


def _resolve_run(spec: ClusterSpec, nodes: int) -> tuple[Point, RunShape, str]:
    """Resolve the point and mode for this invocation."""
    if smoke():
        point = parse_point(defaults=Point(*SMOKE_POINT_DEFAULTS))
        return point, smoke_shape(point, smoke_steps(), spec, nodes), "smoke"
    point = parse_point()
    batch_fit(point, spec, nodes)
    return point, production_shape(point), "sweep"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    # Resolve every path under the experiment's own prefix rather than the cluster default.
    # Set before any step is built, since StepContext reads it at build time.
    os.environ["MARIN_PREFIX"] = STORAGE_PREFIX

    if tokenize_only():
        # Validation split first: it is ~100x smaller, so a broken tokenize path fails in
        # seconds rather than after the train split has burned the CPU pool.
        caches = [
            _tokenize_cache(COMPONENT_VAL, VAL_DOCS, validation=True),
            _tokenize_cache(COMPONENT_TRAIN, TRAIN_DOCS, validation=False),
        ]
        logger.info("tokenizing contacts-v1 into %s (cache version %s)", STORAGE_PREFIX, CACHE_VERSION)
        StepRunner().run([lower(cache) for cache in caches])
        return

    spec = parse_cluster()
    nodes = parse_nodes()
    point, shape, mode = _resolve_run(spec, nodes)

    checked = assert_paths_on_bucket(shape)

    if preview():
        _print_preview(point, shape, spec, nodes, mode)
        for name, path in checked.items():
            print(f"    {name}: {path}", flush=True)
        return

    StepRunner().run([lower(build_run(point, shape, spec, nodes))])


if __name__ == "__main__":
    main()
