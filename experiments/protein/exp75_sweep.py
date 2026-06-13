# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exp 75: contacts-v1 1.5B LR / weight-decay / epochs tuning, one run per launch.

Reimplements the contacts-v1 1.5B recipe from MarinFold PR #70
(https://github.com/Open-Athena/MarinFold/pull/70) as a self-contained marin
monolith (fray only, NO marin Executor), exposed as a launcher that trains
**exactly one explicit (epochs, lr, wd) point per invocation**. The model is the
exp49 Qwen3 1.47B config; data, loss, packing, shuffle and eval mirror #70.

This file is deliberately NOT a fixed grid. The search over (epochs, lr, wd) is
agent-driven and reviewed by hand wave-by-wave -- see
``experiments/protein/exp75_sweep.md`` for the procedure (staged coarse-to-fine
zoom, warm-started up the epoch ladder, boundary handling) and the running log.
The launcher's only job is: take three values, build the trial, submit one TPU
job that trains it inline, and key the lock / run name on the exact values so
reruns are idempotent and identifiable.

Objective being optimized: **final-step ``eval/contacts-v1-val/loss``** (read
from W&B after the run finishes).

Concurrency: bounded by the iris budget (interactive) plus discretionary off-budget
``batch`` capacity, not a fixed run count -- the goal is to finish the sweep fast.
Each invocation here is a single run; a wave is many separate ``iris job run``
submissions across slices/bands. See exp75_sweep.md "Scheduling — finish fast".

Required env per launch: ``EPOCHS`` (int >= 1), ``LR`` (float > 0), ``WD``
(float >= 0). ``PREVIEW=yes`` resolves the point and submits nothing. ``TPU``
picks the slice (single-host ``v6e-8`` (default) / ``v5p-8`` / ``v6e-4`` or
multi-host ``v5p-16/32/64`` / ``v6e-16/32``); ``BAND`` picks the priority band
independently (``interactive`` default = counts toward budget; ``batch`` =
off-budget, lower priority). Region/zone are set on the iris command
(``--region us-east5``), never here. See exp75_sweep.md.

Preview one point::

    EPOCHS=1 LR=3.5e-4 WD=0.05 PREVIEW=yes \\
        uv run python -m experiments.protein.exp75_sweep

Launch one run::

    source ~/marin.env && uv run iris --cluster marin job run \\
        --user "$USERNAME" --no-wait --region us-east5 --memory=1GB \\
        -e WANDB_API_KEY "$WANDB_API_KEY" \\
        -e HUGGING_FACE_HUB_TOKEN "$HUGGING_FACE_HUB_TOKEN" \\
        -e WANDB_ENTITY "$WANDB_ENTITY" -e WANDB_PROJECT "$WANDB_PROJECT" \\
        -e EPOCHS 1 -e LR 3.5e-4 -e WD 0.05 \\
        -- python -m experiments.protein.exp75_sweep

Monitor / resubmit failures with the ``monitor-sweep`` skill.
"""

import dataclasses
import logging
import os
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum

from fray import ResourceConfig, current_client
from fray.types import Entrypoint, JobRequest, create_environment
from iris.rpc import job_pb2
from levanter.callbacks.watch import WatchConfig
from levanter.data.text import DatasetComponent, LmDataConfig, TextLmDatasetFormat, UrlDatasetSourceConfig
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config
from marin.execution.sweep import SweepTarget
from marin.execution.types import versioned
from marin.training.run_environment import extras_for_resources
from marin.training.training import resolve_training_env

from experiments.defaults import _run_training_on_worker, prepare_lm_train
from experiments.llama import compute_num_parameters
from experiments.simple_train_config import SimpleTrainConfig

logger = logging.getLogger(__name__)


# --- Identity ----------------------------------------------------------------

# Bump SWEEP_VERSION to fork run names + the (region-pinned) lock root for a
# fresh campaign over the same recipe and caches.
SWEEP_VERSION: str = "v1"

RUN_NAME_PREFIX: str = "prot-exp75"
SWEEP_ROOT: str = f"gs://marin-us-east5/sweeps/prot-exp75-contacts-v1/run-{SWEEP_VERSION}"
WANDB_GROUP: str = "exp75-contacts-v1-tune"

# The canonical epoch ladder the agent walks (see exp75_sweep.md). Not enforced
# -- any integer >= 1 is launchable -- but documents the intended rungs.
CANONICAL_EPOCHS: tuple[int, ...] = (1, 2, 4, 8)


# --- Data (reused contacts-v1 caches from MarinFold PR #70) ------------------

# contacts-v1 tokenizer (2845 vocab). Training loads it from HF at the immutable
# revision; the cache below was tokenized from this exact revision by #70.
TOKENIZER_REPO: str = "timodonnell/contacts-v1-tokenizer"
TOKENIZER_REVISION: str = "5d68a24a899f"
TOKENIZER: str = f"{TOKENIZER_REPO}@{TOKENIZER_REVISION}"
VOCAB_SIZE: int = 2845

# Immutable levanter token caches published by #70 under the MarinFold exp67
# prefix (us-east5). The train cache holds ``train/`` and the val cache holds
# ``validation/``. We point cache-only components straight at them -- no
# tokenize step runs in this sweep.
_CONTACTS_V1_CACHE_BASE: str = "gs://marin-us-east5/protein-structure/MarinFold/exp67_contacts_v1_1_5b/tokenized"
TRAIN_CACHE: str = f"{_CONTACTS_V1_CACHE_BASE}/contacts-v1-663ba6"
VAL_CACHE: str = f"{_CONTACTS_V1_CACHE_BASE}/contacts-v1-val-92827b"

# Exact train-corpus token count, read from the reused cache's ledger
# (``contacts-v1-663ba6/train/.stats.json``: total_tokens over 4,129,682 docs).
# Steps/epoch are derived from this, not estimated.
TRAIN_TOKENS: int = 4_676_753_425

# Mixture row keys -> prefix of every ``eval/<component>/loss`` series in W&B.
COMPONENT_TRAIN: str = "contacts-v1"
COMPONENT_VAL: str = "contacts-v1-val"

DOC_FORMAT = TextLmDatasetFormat(text_key="document")


# --- Model (exp49 Qwen3 1.47B; exp44 dims + Llama3 rope) ---------------------

MODEL_CONFIG = Qwen3Config(
    max_seq_len=8192,
    hidden_dim=2048,
    intermediate_dim=8192,
    num_heads=32,
    num_kv_heads=8,
    num_layers=24,
    rope=Llama3RotaryEmbeddingsConfig(),
)
HIDDEN_DIM: int = MODEL_CONFIG.hidden_dim


# --- Fixed training recipe (mirrors #70; only LR/WD/epochs vary per launch) --

BATCH_SIZE: int = 128
SEQ_LEN: int = 8192
WARMUP: float = 0.1  # 10% warmup
LR_SCHEDULE: str = "cosine"  # AdamW + cosine decay (per #70, not exp49's WSD)

# Fixed data shuffle: full Feistel permutation over the whole train stream with
# a fixed seed (the corpus shards are round-descending, highest-pLDDT last).
DATA_SEED: int = 0
SHUFFLE: bool = True
PERMUTATION_TYPE: str = "feistel"
MIXTURE_BLOCK_SIZE: int = 2048

# 1 epoch ~= TRAIN_TOKENS / (batch * seq) steps.
TOKENS_PER_STEP: int = BATCH_SIZE * SEQ_LEN
STEPS_PER_EPOCH: int = round(TRAIN_TOKENS / TOKENS_PER_STEP)

# Eval: full held-out val split each eval (max_eval_batches=None), 2 evals/epoch.
NUM_EVALS_PER_EPOCH: int = 2
STEPS_PER_EVAL: int = max(1, round(STEPS_PER_EPOCH / NUM_EVALS_PER_EPOCH))

# Checkpoints: rolling temp every 10 min; one permanent (final) checkpoint.
TEMP_CHECKPOINT_INTERVAL = timedelta(minutes=10)


# --- TPU sizing (size-tpu-train-config skill; ported from exp49) -------------


class Band(StrEnum):
    """iris priority band, a free per-launch choice via the ``BAND`` env.

    INTERACTIVE counts toward the per-user budget cap; BATCH is excluded from
    budget but lower priority (scheduled/preempted after interactive). Any slice
    may run under either band -- pick whatever finishes the sweep fastest while
    keeping interactive spend roughly under cap (see exp75_sweep.md).
    """

    INTERACTIVE = "interactive"
    BATCH = "batch"


# Our band -> the iris proto enum, forwarded straight through as JobRequest.priority.
_BAND_TO_PRIORITY: dict[Band, int] = {
    Band.INTERACTIVE: job_pb2.PRIORITY_BAND_INTERACTIVE,
    Band.BATCH: job_pb2.PRIORITY_BAND_BATCH,
}

DEFAULT_BAND: Band = Band.INTERACTIVE


@dataclass(frozen=True)
class TpuStats:
    chips: int
    hbm_gib: int
    tflops: int


# Allow-list of slices (resources() rejects anything else): single-host
# v6e-8/v5p-8/v6e-4 and larger multi-host v5p-16/32/64, v6e-16/32. Band is NOT
# tied to slice -- any slice runs under either priority band (BAND env). Choose
# the slice by MEASURED throughput, not size: empirically the v5p family wins
# (see exp75_throughput.md), and the best tok/s is not always the biggest slice.
# (chips, HBM GiB/chip, bf16 TFLOP/s/chip). chips per fray TPU_TOPOLOGIES
# (v5p-N = N/2 chips, v6e-N = N chips); HBM/TFLOPs per chip from tpu-stats.
TPUS: dict[str, TpuStats] = {
    "v6e-8": TpuStats(chips=8, hbm_gib=32, tflops=918),
    "v5p-8": TpuStats(chips=4, hbm_gib=95, tflops=459),
    "v6e-4": TpuStats(chips=4, hbm_gib=32, tflops=918),
    "v5p-16": TpuStats(chips=8, hbm_gib=95, tflops=459),
    "v5p-32": TpuStats(chips=16, hbm_gib=95, tflops=459),
    "v5p-64": TpuStats(chips=32, hbm_gib=95, tflops=459),
    "v6e-16": TpuStats(chips=16, hbm_gib=32, tflops=918),
    "v6e-32": TpuStats(chips=32, hbm_gib=32, tflops=918),
}

HBM_FLOOR_GIB: int = 16
# Examples/chip that fit a 16 GiB chip for this 1.47B / seq-8192 model, scaled by
# HBM per slice (cap = PER_CHIP_MICROBATCH * hbm_gib // 16). Per-device
# parallelism (pdp) and grad-accum at global batch 128 (full = 128 // chips;
# pdp = -1 when the full per-chip load fits the cap -- no accumulation -- else the
# largest divisor of full <= cap). Code-verified (PREVIEW):
#   tpu       chips  hbm  cap  full  pdp  grad_accum
#   v6e-8         8   32    8    16    8       2
#   v6e-4         4   32    8    32    8       4
#   v5p-8         4   95   20    32   16       2
#   v5p-16        8   95   20    16   -1       1
#   v5p-32       16   95   20     8   -1       1
#   v5p-64       32   95   20     4   -1       1
#   v6e-16       16   32    8     8   -1       1
#   v6e-32       32   32    8     4   -1       1
# Multi-host slices (>= 8 chips) fit the full per-chip load (<= 16) with NO
# accumulation (grad_accum 1). v6e-16 sits at exactly 8/chip (the v6e ceiling;
# 16/chip OOMs a 32 GiB chip). Global batch stays 128 on every slice, so val loss
# is comparable across all slices and bands.
# NB: PCM 4-6 yield this same plan; PCM=7 lifts v5p-8 to full 32/chip; PCM=8 OOMs v6e.
PER_CHIP_MICROBATCH: int = 4


def per_device_parallelism(tpu_type: str, global_batch: int, per_chip_microbatch: int) -> int:
    """``-1`` (no accumulation) if the full per-chip load fits, else the largest
    divisor of ``global_batch // chips`` within the HBM-scaled cap."""
    stats = TPUS[tpu_type]
    if global_batch % stats.chips:
        raise ValueError(f"global batch {global_batch} not divisible by {stats.chips} chips ({tpu_type})")
    cap = per_chip_microbatch * (stats.hbm_gib // HBM_FLOOR_GIB)
    full = global_batch // stats.chips
    if full <= cap:
        return -1
    return next(d for d in range(cap, 0, -1) if full % d == 0)


@dataclass(frozen=True)
class BatchPlan:
    per_device_parallelism: int  # -1 == BATCH_SIZE // chips (no accumulation)
    per_device_eval_parallelism: int
    grad_accum_steps: int


def plan_batch(res: ResourceConfig) -> BatchPlan:
    """Size train/eval parallelism for the chosen single-host slice."""
    stats = TPUS[res.device.variant]
    chips = stats.chips
    if chips != res.chip_count():
        raise ValueError(f"chip mismatch: table {chips} != resources {res.chip_count()} ({res.device.variant})")
    pdp = per_device_parallelism(res.device.variant, BATCH_SIZE, PER_CHIP_MICROBATCH)
    eval_pdp = BATCH_SIZE // chips if pdp == -1 else pdp  # eval does not accumulate
    grad_accum = 1 if pdp == -1 else (BATCH_SIZE // chips) // pdp
    return BatchPlan(per_device_parallelism=pdp, per_device_eval_parallelism=eval_pdp, grad_accum_steps=grad_accum)


# --- Resources ---------------------------------------------------------------

# `TPU` selects the slice, `BAND` the priority band -- independent choices.
# Region/zone are NEVER set here; pass them on the `iris job run` command
# (`--region us-east5`, always). The band IS set here (forwarded as
# JobRequest.priority) because a fray-submitted child does not inherit the
# driver's band. Default slice is v6e-8, default band interactive.
DEFAULT_TPU: str = "v6e-8"


def tpu() -> str:
    return os.environ.get("TPU") or DEFAULT_TPU


def resources() -> ResourceConfig:
    name = tpu()
    if name not in TPUS:
        raise ValueError(f"unsupported TPU {name!r}; supported: {list(TPUS)}")
    return ResourceConfig.with_tpu(name)


def band() -> Band:
    """Priority band for this launch (``BAND`` env; default interactive). Any slice
    may use either band -- choose to finish fastest while keeping interactive spend
    roughly under cap."""
    raw = os.environ.get("BAND")
    if not raw:
        return DEFAULT_BAND
    try:
        return Band(raw.strip().lower())
    except ValueError:
        raise SystemExit(f"BAND must be one of {[b.value for b in Band]}, got {raw!r}") from None


# --- A single trial point ----------------------------------------------------


@dataclass(frozen=True)
class Config:
    """One trial: explicit epoch count, peak LR, and weight decay."""

    epochs: int
    learning_rate: float
    weight_decay: float

    @property
    def num_train_steps(self) -> int:
        return self.epochs * STEPS_PER_EPOCH

    @property
    def config_id(self) -> str:
        return f"e{self.epochs}-lr{fmt_lr(self.learning_rate)}-wd{fmt_wd(self.weight_decay)}"


def parse_point() -> Config:
    """Build the single trial from the ``EPOCHS`` / ``LR`` / ``WD`` env vars."""
    try:
        epochs = int(os.environ["EPOCHS"])
        learning_rate = float(os.environ["LR"])
        weight_decay = float(os.environ["WD"])
    except KeyError as e:
        raise SystemExit(f"missing required env var {e}; set EPOCHS, LR and WD (one run per launch)") from e
    if epochs < 1:
        raise SystemExit(f"EPOCHS must be >= 1, got {epochs}")
    if learning_rate <= 0:
        raise SystemExit(f"LR must be > 0, got {learning_rate}")
    if weight_decay < 0:
        raise SystemExit(f"WD must be >= 0, got {weight_decay}")
    if epochs not in CANONICAL_EPOCHS:
        logger.warning("EPOCHS=%d is off the canonical ladder %s", epochs, CANONICAL_EPOCHS)
    return Config(epochs=epochs, learning_rate=learning_rate, weight_decay=weight_decay)


# --- Helpers -----------------------------------------------------------------


def fmt_count(n: int) -> str:
    """Compact precise count, e.g. 4.677B, 17.8k."""
    if n >= 1_000_000_000:
        return f"{n / 1e9:.3f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)


def fmt_lr(lr: float) -> str:
    """Compact, collision-safe LR tag to ~3 sig figs, e.g. 3.5e-4, 2.75e-4."""
    mantissa, exponent = f"{lr:.2e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exponent)}"


def fmt_wd(wd: float) -> str:
    """Path/name-safe weight decay, e.g. 0.05 -> ``0p05``, 0.0 -> ``0``."""
    return f"{wd:g}".replace(".", "p")


def trial_name(config: Config) -> str:
    """Stable per-trial run id (also the per-target lock dir under SWEEP_ROOT)."""
    return f"{RUN_NAME_PREFIX}-cv1-1_5b-{config.config_id}-{SWEEP_VERSION}"


def _job_safe(text: str) -> str:
    return text.replace(".", "p")


# --- Data config -------------------------------------------------------------


def cache_only_component(cache_dir: str) -> DatasetComponent:
    """Unmasked cache-only component: empty URLs load ``<cache_dir>/{train,validation}/``.

    contacts-v1 has no ``<distance>`` statements, so there is NO loss mask --
    every token position contributes to the loss. ``pack=True`` (prefix-only;
    documents are never concat-and-split, which would orphan headers).
    """
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
        loss_weight_fn=None,
    )


def build_data_config() -> LmDataConfig:
    """LmDataConfig over the reused contacts-v1 train cache + held-out val cache.

    The val cache is eval-only (train weight 0). Identical for every trial.
    """
    return LmDataConfig(
        components={
            COMPONENT_TRAIN: cache_only_component(TRAIN_CACHE),
            COMPONENT_VAL: cache_only_component(VAL_CACHE),
        },
        train_weights={COMPONENT_TRAIN: 1.0, COMPONENT_VAL: 0.0},
        tokenizer=TOKENIZER,
        cache_dir=None,
        block_cross_document_attention=True,
        shuffle=SHUFFLE,
        permutation_type=PERMUTATION_TYPE,
        mixture_block_size=MIXTURE_BLOCK_SIZE,
    )


# --- Trial construction (worker side) ----------------------------------------


def build_trial(config: Config, res: ResourceConfig) -> tuple[str, object]:
    """Build one trial's ``(job_name, raw_config)`` for ``_run_training_on_worker``."""
    plan = plan_batch(res)
    num_train_steps = config.num_train_steps

    train_config = SimpleTrainConfig(
        resources=res,
        train_batch_size=BATCH_SIZE,
        num_train_steps=versioned(num_train_steps),
        learning_rate=versioned(config.learning_rate),
        weight_decay=versioned(config.weight_decay),
        warmup=WARMUP,
        lr_schedule=LR_SCHEDULE,
        train_seq_len=SEQ_LEN,
        steps_per_eval=STEPS_PER_EVAL,
        steps_per_export=None,  # one permanent (final) checkpoint
        max_eval_batches=None,  # full held-out val split each eval
        data_seed=DATA_SEED,
        per_device_parallelism=plan.per_device_parallelism,
        per_device_eval_parallelism=plan.per_device_eval_parallelism,
    )

    params = compute_num_parameters(MODEL_CONFIG, VOCAB_SIZE)
    tokens = TOKENS_PER_STEP * num_train_steps
    tags = [
        "protein",
        "exp75",
        "contacts-v1",
        "1_5b",
        "qwen3",
        "unmasked",
        f"sweep={SWEEP_VERSION}",
        f"epochs={config.epochs}",
        f"lr={config.learning_rate:g}",
        f"wd={config.weight_decay:g}",
        f"params={fmt_count(params)}",
        f"params_exact={params}",
        f"tokens={fmt_count(tokens)}",
        f"tokens_exact={tokens}",
        f"steps={num_train_steps}",
        f"tpu={res.device.variant}",
        f"band={band()}",
        f"grad_accum={plan.grad_accum_steps}",
    ]
    job_name, raw_config = prepare_lm_train(
        name=trial_name(config),
        tokenized=build_data_config(),
        model_config=MODEL_CONFIG,
        train_config=train_config,
        tags=tags,
        eval_harness_tasks=[],
        use_default_validation=False,
        wandb_group=WANDB_GROUP,
        wandb_name=trial_name(config),
    )
    # 10-min temp checkpoints; no per-parameter watch (HBM pressure on 1.5B).
    raw_config = dataclasses.replace(
        raw_config,
        trainer=dataclasses.replace(
            raw_config.trainer,
            checkpointer=dataclasses.replace(
                raw_config.trainer.checkpointer,
                save_interval=TEMP_CHECKPOINT_INTERVAL,
            ),
            watch=WatchConfig(watch_targets=[], interval=0),
        ),
    )
    return job_name, raw_config


# --- Preview -----------------------------------------------------------------


def preview() -> bool:
    return os.environ.get("PREVIEW", "").strip().lower() in {"yes", "true", "1"}


def print_preview(config: Config, res: ResourceConfig) -> None:
    params = compute_num_parameters(MODEL_CONFIG, VOCAB_SIZE)
    plan = plan_batch(res)
    pdp = BATCH_SIZE // res.chip_count() if plan.per_device_parallelism == -1 else plan.per_device_parallelism
    tokens = fmt_count(TOKENS_PER_STEP * config.num_train_steps)
    print(
        f"PREVIEW: exp75 would launch 1 run -- {config.config_id}\n"
        f"  trial_name={trial_name(config)}\n"
        f"  epochs={config.epochs} lr={config.learning_rate:g} wd={config.weight_decay:g}\n"
        f"  steps={config.num_train_steps} (steps/epoch={STEPS_PER_EPOCH}) "
        f"steps/eval={STEPS_PER_EVAL} tokens={tokens}\n"
        f"  model={fmt_count(params)} params schedule={LR_SCHEDULE} warmup={WARMUP}\n"
        f"  tpu={res.device.variant} band={band()} "
        f"chips={res.chip_count()} per_device={pdp} grad_accum={plan.grad_accum_steps}",
        flush=True,
    )


# --- Worker + launcher -------------------------------------------------------


def _run_one(target: SweepTarget, res: ResourceConfig) -> None:
    """Resolve the trial under this worker's region and train inline."""
    config: Config = target.config
    name, raw_config = build_trial(config, res)
    _run_training_on_worker(name=name, raw_config=raw_config, override_output_path=None, resources=res)


def _sweep_worker_entrypoint(sweep_root: str, targets: list[SweepTarget], res: ResourceConfig) -> None:
    """Train the single target inline on every host of the slice.

    ``targets`` is always a SINGLE-element list (one run per launch).

    NB: ``claim_and_run`` is intentionally DISABLED for now. It takes a GCS lock
    keyed on ``target_id``; on a MULTI-HOST slice every host runs this entrypoint
    and races that one lock -- one wins, the rest die with "Lost lock race ..."
    and the gang fails. We manage runs individually (one launch per point), so the
    dedup/idempotency guard isn't needed. CAVEAT: with the lock off there is no
    guard against running the same (epochs, lr, wd) twice -- don't double-submit a
    point. To re-enable, restore the ``claim_and_run`` import + call below.
    #   claim_and_run(sweep_root, targets, lambda t: _run_one(t, res))
    """
    for target in targets:
        _run_one(target, res)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    config = parse_point()
    res = resources()

    if preview():
        print_preview(config, res)
        return

    env = resolve_training_env(base_env=None, resources=res)
    extras = extras_for_resources(res)

    target = SweepTarget(target_id=trial_name(config), config=config)
    request = JobRequest(
        name=_job_safe(f"{RUN_NAME_PREFIX}-{config.config_id}"),
        entrypoint=Entrypoint.from_callable(_sweep_worker_entrypoint, args=[SWEEP_ROOT, [target], res]),
        resources=res,
        environment=create_environment(env_vars=env, extras=extras),
        priority=_BAND_TO_PRIORITY[band()],
    )
    client = current_client()
    handle = client.submit(request)
    logger.info("Submitted %s (%s)", request.name, trial_name(config))
    handle.wait(raise_on_failure=True)
    logger.info("Run finished: %s", trial_name(config))


if __name__ == "__main__":
    main()
