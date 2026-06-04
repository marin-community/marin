# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exp 49 (AA mixture sweep): does adding AA-sequence pretraining data help the structure-token task?

Two mixtures x 3 seeds (6 trials): ``m1`` = 100% structure docs (500M tokens);
``m2`` = 50/50 structure docs + AA-sequence docs (1B tokens — additive, so
structure-doc exposure stays at 500M in both arms). Metric: held-out cd-val
distance-bin loss. Structure docs + cd-val use the distance-bin loss mask; the
seq cell is unmasked (standard LM loss on the AA tokens).

Region-agnostic (pass ``--region`` at submission; only ``SWEEP_ROOT`` is pinned).
Token caches resolve per-region via ``marin_prefix()`` and build once under a
lock (no Executor). Training fans out ``WORKERS`` TPU workers that claim trials
via ``claim_and_run`` and resolve placeholders in the worker's own region.

Pre-build the caches (CPU, no fan-out)::

    set -a; source ~/marin.env; set +a
    export PATH="$HOME/google-cloud-sdk/bin:$HOME/.local/bin:$PATH"
    uv run iris --cluster=marin job run --user "$USERNAME" --no-wait \\
        --job-name prot-exp49-tokenize \\
        --region us-east5 --cpu=4 --memory=16GB --extra=cpu --enable-extra-resources \\
        -e HF_TOKEN "$HF_TOKEN" -e HUGGING_FACE_HUB_TOKEN "$HF_TOKEN" \\
        -e TOKENIZE yes \\
        -- python -m experiments.protein.exp49_aamix_sweep

Launch the sweep::

    uv run iris --cluster=marin job run --user "$USERNAME" --no-wait \\
        --job-name prot-exp49-sweep \\
        --region us-east5 --memory=1GB \\
        -e HF_TOKEN "$HF_TOKEN" -e HUGGING_FACE_HUB_TOKEN "$HF_TOKEN" \\
        -e WANDB_API_KEY "$WANDB_API_KEY" -e WANDB_ENTITY "$WANDB_ENTITY" \\
        -e WANDB_PROJECT "$WANDB_PROJECT" \\
        -- python -m experiments.protein.exp49_aamix_sweep

Env vars: ``RUNS`` (CSV substring filter on config id, e.g. ``m1``, ``m2-s1``),
``PREVIEW=yes`` (list targets, submit nothing), ``TOKENIZE=yes`` (build caches
and exit), ``TPU`` (single-host slice; default v5p-8), ``WORKERS`` (fan-out
count; default = number of selected configs). Preview::

    RUNS=m2 PREVIEW=yes uv run python -m experiments.protein.exp49_aamix_sweep
"""

import dataclasses
import logging
import math
import os
from dataclasses import dataclass
from datetime import timedelta

from fray import ResourceConfig, current_client
from fray.types import Entrypoint, JobRequest, create_environment
from levanter.callbacks.watch import WatchConfig
from levanter.data.text import DatasetComponent, LmDataConfig, TextLmDatasetFormat, UrlDatasetSourceConfig
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config
from marin.execution.executor_step_status import STATUS_SUCCESS, StepAlreadyDone, step_lock
from marin.execution.sweep import SweepTarget, claim_and_run
from marin.execution.types import versioned
from marin.processing.tokenize import TokenizeConfig, tokenize
from marin.training.training import extras_for_resources, resolve_training_env
from rigging.filesystem import marin_prefix

from experiments.defaults import _run_training_on_worker, prepare_lm_train
from experiments.llama import compute_num_parameters
from experiments.protein.protein_train_common import (
    HF_DATASET_BASE,
    PROTEIN_TOKENIZER,
    distance_bin_only_loss_weight,
)
from experiments.simple_train_config import SimpleTrainConfig

logger = logging.getLogger(__name__)


# --- Identity ----------------------------------------------------------------

# Two independent version knobs. Bump SWEEP_VERSION to fork run names + the
# (region-pinned) sweep lock root for a fresh run on the SAME caches. Bump
# TOKENIZE_VERSION only when the tokenization recipe changes: it forks the cache
# dirs (forcing a re-tokenize) and is otherwise decoupled from the sweep, so a
# v2/v3 sweep can reuse the v1 caches.
SWEEP_VERSION: str = "v1"
TOKENIZE_VERSION: str = "v1"

RUN_NAME_PREFIX: str = "prot-exp49"
SWEEP_ROOT: str = f"gs://marin-us-east5/sweeps/prot-exp49-aamix/run-{SWEEP_VERSION}"
WANDB_GROUP: str = "exp49-aamix"

# Cache namespace; the per-TOKENIZE_VERSION dir under it is the decoupling point.
CACHE_NAMESPACE: str = "protein/exp49"

# --- TPU sizing (size-tpu-train-config skill, marin-agent-kb) ----------------


@dataclass(frozen=True)
class TpuStats:
    chips: int
    hbm_gib: int
    tflops: int


# Single-host slices: (chips, HBM GiB, bf16 TFLOP/s) per chip. Extend as needed.
SINGLE_HOST_TPUS: dict[str, TpuStats] = {
    "v4-8": TpuStats(chips=4, hbm_gib=32, tflops=275),
    "v5litepod-1": TpuStats(chips=1, hbm_gib=16, tflops=197),
    "v5litepod-2": TpuStats(chips=2, hbm_gib=16, tflops=197),
    "v5litepod-4": TpuStats(chips=4, hbm_gib=16, tflops=197),
    "v5litepod-8": TpuStats(chips=8, hbm_gib=16, tflops=197),
    "v5p-8": TpuStats(chips=4, hbm_gib=95, tflops=459),
    "v6e-1": TpuStats(chips=1, hbm_gib=32, tflops=918),
    "v6e-4": TpuStats(chips=4, hbm_gib=32, tflops=918),
    "v6e-8": TpuStats(chips=8, hbm_gib=32, tflops=918),
}

HBM_FLOOR_GIB: int = 16

# Hand-tuned (no estimation): examples/chip that fit a 16 GiB v5e chip for this
# 1.47B / seq-8192 model, scaled by HBM per slice (see per_device_parallelism).
# At 4: v6e-4 -> microbatch 32 (accum 4); v5p-8 -> microbatch 64 (accum 2).
# v6e-4 OOM'd at 8 (microbatch 64), so this is the global step-down.
PER_CHIP_MICROBATCH: int = 4


def per_device_parallelism(tpu_type: str, global_batch: int, per_chip_microbatch: int) -> int:
    """``-1`` (no accumulation) if the full per-chip load fits, else the largest
    divisor of ``global_batch // chips`` within the HBM-scaled cap."""
    stats = SINGLE_HOST_TPUS[tpu_type]
    if global_batch % stats.chips:
        raise ValueError(f"global batch {global_batch} not divisible by {stats.chips} chips ({tpu_type})")
    cap = per_chip_microbatch * (stats.hbm_gib // HBM_FLOOR_GIB)
    full = global_batch // stats.chips
    if full <= cap:
        return -1
    return next(d for d in range(cap, 0, -1) if full % d == 0)


# --- Data --------------------------------------------------------------------

DOCS_HF_DATASET_ID: str = "eczech/marinfold-exp11-protein-docs"
DOCS_HF_REVISION: str = "41b2ec71070cb9e8799311cd8f78877e747f6754"

SEQ_HF_DATASET_ID: str = "eczech/marinfold-exp11-protein-docs-seq"
SEQ_HF_REVISION: str = "1fe8de92e638e50aabf0ce05a83590654d7ceb09"

# Mixture row keys / prefix of every ``eval/<component>/loss`` series in W&B.
COMPONENT_DOCS: str = "protein-docs-low-train"
COMPONENT_SEQ: str = "protein-docs-seq-low-train"
COMPONENT_CD_VAL: str = "protein-docs-cd-val"

DOC_FORMAT = TextLmDatasetFormat(text_key="document")


@dataclass(frozen=True)
class CacheSpec:
    """Region-independent tokenize job; the leaf ``name`` resolves to a
    region-local, TOKENIZE_VERSION-scoped dir (see ``ensure_cache``)."""

    name: str
    source_url: str
    is_validation: bool


# Cache leaf keys (resolved under
# ``marin_prefix()/tokenized/<CACHE_NAMESPACE>/<TOKENIZE_VERSION>/``); the
# revision tag in each leaf means a dataset bump also writes a fresh cache.
DOCS_SPEC = CacheSpec(
    name="docs-low-41b2ec7",
    source_url=f"hf://datasets/{DOCS_HF_DATASET_ID}@{DOCS_HF_REVISION}/low/train/",
    is_validation=False,
)
SEQ_SPEC = CacheSpec(
    name="seq-low-1fe8de9",
    source_url=f"hf://datasets/{SEQ_HF_DATASET_ID}@{SEQ_HF_REVISION}/low/train/",
    is_validation=False,
)
CD_VAL_SPEC = CacheSpec(
    name="cd-val",
    source_url=f"{HF_DATASET_BASE}/val/",
    is_validation=True,
)

_ensured_caches: dict[str, str] = {}


def ensure_cache(spec: CacheSpec) -> str:
    """Region-local cache dir, built once under a lock if absent.

    ``tokenize`` runs in-process (no Executor) and is idempotent. ``step_lock``
    serializes same-region workers: the first builds and writes ``STATUS_SUCCESS``,
    the rest wait then skip via ``StepAlreadyDone``.
    """
    cache_dir = f"{marin_prefix()}/tokenized/{CACHE_NAMESPACE}/{TOKENIZE_VERSION}/{spec.name}"
    if cache_dir in _ensured_caches:
        return cache_dir
    try:
        with step_lock(cache_dir, spec.name) as status:
            tokenize(
                TokenizeConfig(
                    train_paths=[] if spec.is_validation else [spec.source_url],
                    validation_paths=[spec.source_url] if spec.is_validation else [],
                    cache_path=cache_dir,
                    tokenizer=PROTEIN_TOKENIZER,
                    format=DOC_FORMAT,
                )
            )
            status.write_status(STATUS_SUCCESS)
    except StepAlreadyDone:
        pass  # a peer already built it in this region
    _ensured_caches[cache_dir] = cache_dir
    return cache_dir


# --- Model -------------------------------------------------------------------

# exp44's 1.47B dims, LlamaConfig -> Qwen3Config with Llama3 rope.
MODEL_CONFIG = Qwen3Config(
    max_seq_len=8192,
    hidden_dim=2048,
    intermediate_dim=8192,
    num_heads=32,
    num_kv_heads=8,
    num_layers=24,
    rope=Llama3RotaryEmbeddingsConfig(),
)

PROTEIN_VOCAB_SIZE: int = 2840  # legacy 2840-vocab tokenizer pinned in PROTEIN_TOKENIZER
HIDDEN_DIM: int = MODEL_CONFIG.hidden_dim

# --- Optimizer / schedule ----------------------------------------------------

BATCH_SIZE: int = 128
SEQ_LEN: int = 8192
WEIGHT_DECAY: float = 0.01
WARMUP: float = 0.1

# Peak LR from the lr * hidden / sqrt(batch) heuristic; resolves to the 3.5e-4
# reference at batch=128 / hidden=2048.
LR_REF: float = 3.5e-4
LR_REF_BATCH: int = 128
LR_REF_HIDDEN: int = 2048
LR_CONSTANT: float = LR_REF * LR_REF_HIDDEN / math.sqrt(LR_REF_BATCH)
LEARNING_RATE: float = LR_CONSTANT * math.sqrt(BATCH_SIZE) / HIDDEN_DIM

BETA2: float = 0.98 ** (BATCH_SIZE / LR_REF_BATCH)  # noise-scale heuristic (0.98 @ batch 128)

# WSD: warmup -> constant -> linear decay over the trailing LR_DECAY fraction.
LR_SCHEDULE: str = "linear"
LR_DECAY: float = 0.2

# Per replicate, the seed sets both trainer seed and data_seed.
SEEDS: tuple[int, ...] = (1729, 1730, 1731)

SHUFFLE: bool = True
PERMUTATION_TYPE: str = "feistel"
MIXTURE_BLOCK_SIZE: int = 2048

NUM_EVALS: int = 5
EVAL_EXAMPLES: int = 8192  # cd-val sequences per eval; max_eval_batches solved per slice
TEMP_CHECKPOINT_INTERVAL = timedelta(minutes=8)

# --- Resources --------------------------------------------------------------

# Region-agnostic: no zone pinned (pass --region/--zone at submission);
# checkpoints + caches resolve to the worker's region via marin_prefix().
DEFAULT_TPU: str = "v5p-8"


def tpu() -> str:
    return os.environ.get("TPU") or DEFAULT_TPU


def resources() -> ResourceConfig:
    name = tpu()
    if name not in SINGLE_HOST_TPUS:
        raise ValueError(f"unsupported TPU {name!r}; single-host options: {sorted(SINGLE_HOST_TPUS)}")
    return ResourceConfig.with_tpu(name)


# --- Mixtures ----------------------------------------------------------------

# Additive design: m2 = 500M docs + 500M seq holds docs exposure at 500M (== m1),
# so the cd-val structure loss isolates the added seq data. m2 is 2x m1's tokens
# by construction (a fixed-compute split would confound it against the eval).
M1_TOTAL_TOKENS: int = 500_000_000
M2_TOTAL_TOKENS: int = 1_000_000_000


@dataclass(frozen=True)
class Mixture:
    id: str
    weights: dict[str, float]
    target_tokens: int

    def schedule(self) -> tuple[int, int]:
        """``(num_train_steps, steps_per_eval)`` rounded so total = evals * spe."""
        tokens_per_step = BATCH_SIZE * SEQ_LEN
        spe = max(1, round(self.target_tokens / NUM_EVALS / tokens_per_step))
        return spe * NUM_EVALS, spe


MIXTURES: tuple[Mixture, ...] = (
    Mixture(id="m1", weights={COMPONENT_DOCS: 1.0}, target_tokens=M1_TOTAL_TOKENS),
    Mixture(id="m2", weights={COMPONENT_DOCS: 0.5, COMPONENT_SEQ: 0.5}, target_tokens=M2_TOTAL_TOKENS),
)


# --- Configs (mixture x seed) ------------------------------------------------


@dataclass(frozen=True)
class Config:
    """One trial: a (mixture, seed) pair. Region-independent; carried in SweepTarget."""

    mixture: Mixture
    seed_index: int

    @property
    def seed(self) -> int:
        return SEEDS[self.seed_index]

    @property
    def config_id(self) -> str:
        return f"{self.mixture.id}-s{self.seed_index}"


ALL_CONFIGS: tuple[Config, ...] = tuple(Config(m, i) for m in MIXTURES for i in range(len(SEEDS)))
CONFIG_BY_ID: dict[str, Config] = {c.config_id: c for c in ALL_CONFIGS}


# --- Helpers ----------------------------------------------------------------


def fmt_count(n: int) -> str:
    """Compact precise count, e.g. 498.0M, 1.001B."""
    if n >= 1_000_000_000:
        return f"{n / 1e9:.3f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)


def fmt_lr(lr: float) -> str:
    mantissa, exponent = f"{lr:.1e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def trial_name(config: Config) -> str:
    """Stable per-trial run id (also the per-target dir under SWEEP_ROOT)."""
    num_train_steps, _ = config.mixture.schedule()
    tokens_tag = fmt_count(BATCH_SIZE * SEQ_LEN * num_train_steps)
    return (
        f"{RUN_NAME_PREFIX}-1_5b-{config.mixture.id}-{tokens_tag}-"
        f"s{config.seed_index}-lr{fmt_lr(LEARNING_RATE)}-{SWEEP_VERSION}"
    )


# --- Data config ------------------------------------------------------------


def cache_only_component(cache_dir: str, *, masked: bool) -> DatasetComponent:
    """Cache-only component (empty URLs load ``<cache_dir>/{train,validation}/``);
    distance-bin loss mask applied iff ``masked``."""
    source = UrlDatasetSourceConfig(
        train_urls=[],
        validation_urls=[],
        cache_dir=cache_dir,
        format=DOC_FORMAT,
        tags=[],
    )
    return DatasetComponent(
        source=source,
        cache_dir=cache_dir,
        format=DOC_FORMAT,
        pack=True,
        tags=[],
        loss_weight_fn=distance_bin_only_loss_weight if masked else None,
    )


def build_data_config(config: Config) -> LmDataConfig:
    """LmDataConfig for one mixture (resolved on the worker; caches ensured first).

    Docs + cd-val are distance-bin-masked; seq is unmasked and present only when
    the mixture uses it. cd-val is eval-only (train weight 0).
    """
    components: dict[str, DatasetComponent] = {
        COMPONENT_DOCS: cache_only_component(ensure_cache(DOCS_SPEC), masked=True),
        COMPONENT_CD_VAL: cache_only_component(ensure_cache(CD_VAL_SPEC), masked=True),
    }
    train_weights: dict[str, float] = {
        COMPONENT_DOCS: float(config.mixture.weights.get(COMPONENT_DOCS, 0.0)),
        COMPONENT_CD_VAL: 0.0,
    }
    if config.mixture.weights.get(COMPONENT_SEQ, 0.0) > 0.0:
        components[COMPONENT_SEQ] = cache_only_component(ensure_cache(SEQ_SPEC), masked=False)
        train_weights[COMPONENT_SEQ] = float(config.mixture.weights[COMPONENT_SEQ])

    return LmDataConfig(
        components=components,
        train_weights=train_weights,
        tokenizer=PROTEIN_TOKENIZER,
        cache_dir=None,
        block_cross_document_attention=True,
        shuffle=SHUFFLE,
        permutation_type=PERMUTATION_TYPE,
        mixture_block_size=MIXTURE_BLOCK_SIZE,
    )


# --- Trial construction (worker side) ---------------------------------------


@dataclass(frozen=True)
class BatchPlan:
    per_device_parallelism: int  # -1 == BATCH_SIZE // chips (no accumulation)
    per_device_eval_parallelism: int
    max_eval_batches: int
    grad_accum_steps: int


def plan_batch(res: ResourceConfig) -> BatchPlan:
    """Size train/eval parallelism for the chosen single-host slice."""
    stats = SINGLE_HOST_TPUS[res.device.variant]
    chips = stats.chips
    if chips != res.chip_count():
        raise ValueError(f"chip mismatch: table {chips} != resources {res.chip_count()} ({res.device.variant})")
    pdp = per_device_parallelism(res.device.variant, BATCH_SIZE, PER_CHIP_MICROBATCH)
    eval_pdp = BATCH_SIZE // chips if pdp == -1 else pdp  # eval does not accumulate
    max_eval_batches = max(1, EVAL_EXAMPLES // (eval_pdp * chips))
    grad_accum = 1 if pdp == -1 else (BATCH_SIZE // chips) // pdp
    return BatchPlan(
        per_device_parallelism=pdp,
        per_device_eval_parallelism=eval_pdp,
        max_eval_batches=max_eval_batches,
        grad_accum_steps=grad_accum,
    )


def build_trial(config: Config, res: ResourceConfig) -> tuple[str, object]:
    """Build one trial's ``(job_name, raw_config)`` for ``_run_training_on_worker``."""
    num_train_steps, steps_per_eval = config.mixture.schedule()
    plan = plan_batch(res)

    train_config = SimpleTrainConfig(
        resources=res,
        train_batch_size=BATCH_SIZE,
        num_train_steps=num_train_steps,
        learning_rate=versioned(LEARNING_RATE),
        weight_decay=WEIGHT_DECAY,
        beta2=BETA2,
        warmup=WARMUP,
        decay=LR_DECAY,
        lr_schedule=LR_SCHEDULE,
        train_seq_len=SEQ_LEN,
        steps_per_eval=steps_per_eval,
        steps_per_export=None,  # one permanent (final) checkpoint
        max_eval_batches=plan.max_eval_batches,
        data_seed=config.seed,
        per_device_parallelism=plan.per_device_parallelism,
        per_device_eval_parallelism=plan.per_device_eval_parallelism,
    )
    params = compute_num_parameters(MODEL_CONFIG, PROTEIN_VOCAB_SIZE)
    tokens = BATCH_SIZE * SEQ_LEN * num_train_steps
    tags = [
        "protein",
        "exp49",
        "aamix",
        "1_5b",
        "qwen3",
        config.mixture.id,
        f"sweep={SWEEP_VERSION}",
        f"tok={TOKENIZE_VERSION}",
        f"seed={config.seed}",
        f"params={fmt_count(params)}",
        f"params_exact={params}",
        f"tokens={fmt_count(tokens)}",
        f"tokens_exact={tokens}",
        f"steps={num_train_steps}",
        f"tpu={res.device.variant}",
        f"grad_accum={plan.grad_accum_steps}",
    ]
    job_name, raw_config = prepare_lm_train(
        name=trial_name(config),
        tokenized=build_data_config(config),
        model_config=MODEL_CONFIG,
        train_config=train_config,
        tags=tags,
        eval_harness_tasks=[],
        use_default_validation=False,
        wandb_group=WANDB_GROUP,
    )
    # Per-replicate trainer seed; 8-min temp checkpoints; no per-param watch (HBM).
    raw_config = dataclasses.replace(
        raw_config,
        trainer=dataclasses.replace(
            raw_config.trainer,
            seed=config.seed,
            checkpointer=dataclasses.replace(
                raw_config.trainer.checkpointer,
                save_interval=TEMP_CHECKPOINT_INTERVAL,
            ),
            watch=WatchConfig(watch_targets=[], interval=0),
        ),
    )
    return job_name, raw_config


# --- Selection / preview ----------------------------------------------------


def selected_configs() -> tuple[Config, ...]:
    """Configs passing the ``RUNS`` substring filter (empty = all)."""
    raw = os.environ.get("RUNS", "")
    needles = tuple(s.strip() for s in raw.split(",") if s.strip())
    if not needles:
        return ALL_CONFIGS
    return tuple(c for c in ALL_CONFIGS if any(n in c.config_id for n in needles))


def preview() -> bool:
    return os.environ.get("PREVIEW", "").strip().lower() in {"yes", "true", "1"}


def tokenize_only() -> bool:
    return os.environ.get("TOKENIZE", "").strip().lower() in {"yes", "true", "1"}


def num_workers(num_configs: int) -> int:
    """Fan-out count: ``WORKERS`` env, else one worker per selected config."""
    raw = os.environ.get("WORKERS", "").strip()
    return int(raw) if raw else num_configs


def _format_weights(weights: dict[str, float]) -> str:
    return ", ".join(f"{k}={v:.3g}" for k, v in weights.items())


def print_preview(configs: tuple[Config, ...], res: ResourceConfig) -> None:
    params = compute_num_parameters(MODEL_CONFIG, PROTEIN_VOCAB_SIZE)
    plan = plan_batch(res)
    pdp = BATCH_SIZE // res.chip_count() if plan.per_device_parallelism == -1 else plan.per_device_parallelism
    print(
        f"PREVIEW: exp49 aamix would run {len(configs)} target(s) "
        f"(model={fmt_count(params)} params, lr={LEARNING_RATE:.4g}, beta2={BETA2:.4g}, "
        f"tpu={res.device.variant} chips={res.chip_count()} per_device={pdp} "
        f"grad_accum={plan.grad_accum_steps}):",
        flush=True,
    )
    for c in configs:
        num_train_steps, steps_per_eval = c.mixture.schedule()
        tokens_tag = fmt_count(BATCH_SIZE * SEQ_LEN * num_train_steps)
        print(f"  {c.config_id}  {trial_name(c)}", flush=True)
        print(
            f"    mixture={c.mixture.id} seed={c.seed} batch={BATCH_SIZE} "
            f"steps={num_train_steps} steps_per_eval={steps_per_eval} "
            f"tokens={tokens_tag} schedule=WSD(warmup={WARMUP},decay={LR_DECAY})",
            flush=True,
        )
        print(
            f"    train_weights: {_format_weights(c.mixture.weights)}  (seq masked=False, docs masked=True)", flush=True
        )
        print(flush=True)


# --- Worker + launcher ------------------------------------------------------


def _run_one(target: SweepTarget, res: ResourceConfig) -> None:
    """Resolve one trial under this worker's region and train inline."""
    config: Config = target.config
    name, raw_config = build_trial(config, res)
    _run_training_on_worker(name=name, raw_config=raw_config, override_output_path=None, resources=res)


def _sweep_worker_entrypoint(sweep_root: str, targets: list[SweepTarget], res: ResourceConfig) -> None:
    """One TPU sweep worker: claim a target via the lock, train inline, repeat.

    ``targets``/``res`` ride in the entrypoint args (shipped by value), so the
    worker doesn't depend on its own env for the trial set or TPU type.
    """
    claim_and_run(sweep_root, targets, lambda t: _run_one(t, res))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    if tokenize_only():
        for spec in (DOCS_SPEC, SEQ_SPEC, CD_VAL_SPEC):
            logger.info("Ensuring cache %s -> %s", spec.name, ensure_cache(spec))
        return

    configs = selected_configs()
    if not configs:
        raise ValueError(f"No configs matched RUNS={os.environ.get('RUNS', '')!r}")

    res = resources()

    if preview():
        print_preview(configs, res)
        return

    targets = [SweepTarget(target_id=trial_name(c), config=c) for c in configs]
    workers = num_workers(len(configs))
    env = resolve_training_env(base_env=None, resources=res)
    extras = extras_for_resources(res)
    logger.info("Submitting %d TPU worker(s) for %d config(s) on %s", workers, len(configs), res.device.variant)

    client = current_client()
    handles = []
    for i in range(workers):
        request = JobRequest(
            name=f"{RUN_NAME_PREFIX}-w{i}",
            entrypoint=Entrypoint.from_callable(_sweep_worker_entrypoint, args=[SWEEP_ROOT, targets, res]),
            resources=res,
            environment=create_environment(env_vars=env, extras=extras),
        )
        handles.append(client.submit(request))
        logger.info("Submitted worker: %s", request.name)
    for handle in handles:
        handle.wait(raise_on_failure=True)
    logger.info("All %d worker(s) finished", workers)


if __name__ == "__main__":
    main()
