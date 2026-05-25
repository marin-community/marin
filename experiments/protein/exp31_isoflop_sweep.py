# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exp 31: protein iso-FLOP sweep (Open-Athena/MarinFold#31).

See ``exp31_isoflop_sweep.md`` for the design.

Env vars: ``RUNS`` (CSV substring filter on target_id), ``PREVIEW=yes`` (print
the iso-FLOP table and target list, submit nothing), ``LIST_RUNS=yes`` (print
resolved trial names one per line), ``NUM_WORKERS`` (default 4), ``TPU``
(default v5p-8).

Bulk submission (one iris job, NUM_WORKERS Fray workers via rank-stride)::

    uv run iris --cluster=marin job run --user $USERNAME --no-wait \\
        --job-name prot-exp31-iso-$(date +%Y%m%d-%H%M) \\
        --region us-east5 --memory=1GB \\
        -e HF_TOKEN "$HF_TOKEN" -e WANDB_API_KEY "$WANDB_API_KEY" \\
        -- python -m experiments.protein.exp31_isoflop_sweep

Per-trial submission (one iris job per trial, each named after its trial)::

    TIMESTAMP=$(date +%Y%m%d-%H%M)
    for trial in $(LIST_RUNS=yes uv run python -m experiments.protein.exp31_isoflop_sweep); do
        uv run iris --cluster=marin job run --user $USERNAME --no-wait \\
            --job-name ${trial}-${TIMESTAMP} \\
            --region us-east5 --memory=1GB \\
            -e HF_TOKEN "$HF_TOKEN" -e WANDB_API_KEY "$WANDB_API_KEY" \\
            -e RUNS "$trial" -e NUM_WORKERS 1 \\
            -- python -m experiments.protein.exp31_isoflop_sweep
    done
"""

import dataclasses
import logging
import math
import os
from dataclasses import dataclass

from fray import ResourceConfig, current_client
from fray.types import Entrypoint, JobRequest, create_environment
from levanter.callbacks.watch import WatchConfig
from levanter.data.text import DatasetComponent, LmDataConfig, TextLmDatasetFormat, UrlDatasetSourceConfig
from levanter.models.qwen import Qwen3Config
from levanter.utils.flop_utils import lm_flops_per_token
from marin.execution.executor import versioned
from marin.execution.sweep import SweepTarget, claim_and_run
from marin.training.training import extras_for_resources, resolve_training_env

from experiments.defaults import _run_training_on_worker, prepare_lm_train
from experiments.llama import compute_num_parameters
from experiments.protein.protein_train_common import (
    PROTEIN_TOKENIZER,
    distance_bin_only_loss_weight,
    protein_docs_tokenized,
    protein_docs_val_tokenized,
)
from experiments.simple_train_config import SimpleTrainConfig

logger = logging.getLogger(__name__)


# === Constants ==============================================================

# --- Identity ---------------------------------------------------------------

# Bump VERSION to fork run names + sweep-root claim dir on a recipe change.
VERSION: str = "v2"
RUN_NAME_PREFIX: str = "prot-exp31-iso"
SWEEP_ROOT: str = f"gs://marin-us-east5/sweeps/prot-exp31-isoflop/run_isoflop_sweep-{VERSION}"
WANDB_GROUP: str = "exp31-isoflop"

# --- Data -------------------------------------------------------------------

CD_TRAIN_CACHE: str = protein_docs_tokenized.override_output_path
CD_VAL_CACHE: str = protein_docs_val_tokenized.override_output_path
COMPONENT_CD_TRAIN: str = "protein-docs-cd"
COMPONENT_CD_VAL: str = "protein-docs-cd-val"

# Pinned to the legacy 2840-vocab revision in PROTEIN_TOKENIZER.
PROTEIN_VOCAB_SIZE: int = 2840

# Packed cd-train tokens (43.3B). Source: MarinFold#11 EDA.
DATASET_TOKENS: int = 43_301_511_168

# --- Model / sweep grid -----------------------------------------------------

# Four iso-FLOP budgets at ~sqrt(10) ratio (half-decade spacing).
BUDGETS: tuple[float, ...] = (3e17, 1e18, 3e18, 1e19)

# Hidden grid; all values divisible by HEAD_DIM=64.
HIDDEN_MIN: int = 256
HIDDEN_MAX: int = 1280
HIDDEN_STEP: int = 128

HEAD_DIM: int = 64
MLP_RATIO: int = 4
SEQ_LEN: int = 8192

# --- Iso-FLOP solver --------------------------------------------------------

# Single value across all budgets; tuned via PREVIEW=yes (24/36 at 10000).
STEPS_PER_RUN: int = 10000
BATCH_MIN: int = 8  # v5p-8 chip count
BATCH_MAX: int = 128  # v5p-8 HBM limit
LR_MAX: float = 0.03
FLOP_TOLERANCE: float = 0.01
MAX_PARAMS: int = 2_000_000_000
MAX_TRAIN_TOKENS: int = 5 * DATASET_TOKENS  # ~5 epochs of cd-train (safety)

# Levanter's lm_flops_per_token is forward-only; x3 = training FLOPs (C=6ND).
FWD_TO_TRAIN_FLOPS: int = 3

# --- Optimizer / schedule ---------------------------------------------------

# LR scaling solved from the 1.5B recipe: lr=3.5e-4 @ batch=128, hidden=2048.
# Reused as lr(b, h) = LR_CONSTANT * sqrt(b) / h.
LR_REF: float = 3.5e-4
LR_REF_BATCH: int = 128
LR_REF_HIDDEN: int = 2048
LR_CONSTANT: float = LR_REF * LR_REF_HIDDEN / math.sqrt(LR_REF_BATCH)

# beta2(b) = BETA2_BASE ** (b / 128) -- noise-scale heuristic.
BETA2_BASE: float = 0.95

WEIGHT_DECAY: float = 0.01
WARMUP: float = 0.1
LR_SCHEDULE: str = "linear"
LR_DECAY: float = 0.2

# Distinct from exp11 (1729), exp29 (73).
DATA_SEED: int = 31

# --- Resources --------------------------------------------------------------

DEFAULT_TPU: str = "v5p-8"
DEFAULT_NUM_WORKERS: int = 4


# === Helpers ================================================================


def scaled_lr(batch_size: int, hidden_size: int) -> float:
    return LR_CONSTANT * math.sqrt(batch_size) / hidden_size


def scaled_beta2(batch_size: int) -> float:
    return BETA2_BASE ** (batch_size / LR_REF_BATCH)


def tpu() -> str:
    return os.environ.get("TPU") or DEFAULT_TPU


def resources() -> ResourceConfig:
    return ResourceConfig.with_tpu(tpu())


def fmt_count(n: int | float) -> str:
    """Compact precise count, e.g. 120.7M, 1.007B."""
    if n >= 1_000_000_000:
        return f"{n / 1e9:.3f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(int(n))


def fmt_lr(lr: float) -> str:
    mantissa, exponent = f"{lr:.1e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def fmt_budget(budget: float) -> str:
    """Format a FLOP budget, e.g. ``1e18``, ``3.2e18``."""
    mantissa, exponent = f"{budget:.1e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exponent)}"


def round_to_power_of_two_ceil(x: float) -> int:
    if x <= 1:
        return 1
    return 2 ** math.ceil(math.log2(x))


def num_layers_for(hidden: int) -> int:
    """exp2101 depth formula."""
    return round(hidden / (64 + math.log2(hidden) * 4 - 8))


def build_qwen3(hidden: int) -> Qwen3Config:
    num_layers = num_layers_for(hidden)
    num_heads = hidden // HEAD_DIM
    return Qwen3Config(
        max_seq_len=SEQ_LEN,
        hidden_dim=hidden,
        intermediate_dim=hidden * MLP_RATIO,
        num_heads=num_heads,
        num_kv_heads=num_heads,
        num_layers=num_layers,
    )


def total_flops(batch: int, steps: int, model_config: Qwen3Config) -> float:
    """Total training FLOPs (forward + backward) for the given shape."""
    fwd_per_token = lm_flops_per_token(
        hidden_dim=model_config.hidden_dim,
        intermediate_dim=model_config.intermediate_dim,
        num_layers=model_config.num_layers,
        num_kv_heads=model_config.num_kv_heads,
        num_heads=model_config.num_heads,
        seq_len=SEQ_LEN,
        vocab_size=PROTEIN_VOCAB_SIZE,
        glu=True,
    )
    return FWD_TO_TRAIN_FLOPS * fwd_per_token * batch * steps * SEQ_LEN


# === Iso-FLOP solver ========================================================


@dataclass(frozen=True)
class IsoFlopRun:
    """One resolved (budget, hidden) sweep point."""

    budget: float
    hidden: int
    num_layers: int
    intermediate: int
    num_heads: int
    num_kv_heads: int
    batch_exact: float
    batch: int
    halvings: int
    train_steps: int
    train_tokens: int
    achieved_flops: float
    params: int
    lr: float
    beta2: float
    model_config: Qwen3Config


@dataclass(frozen=True)
class DroppedRun:
    """One (budget, hidden) candidate that didn't survive the filters."""

    budget: float
    hidden: int
    reason: str
    batch_exact: float
    batch: int
    halvings: int
    lr: float
    train_steps: int | None
    train_tokens: int | None
    achieved_flops: float | None
    params: int | None


def _resolve_one(budget: float, hidden: int) -> IsoFlopRun | DroppedRun:
    model_cfg = build_qwen3(hidden)
    params = compute_num_parameters(model_cfg, PROTEIN_VOCAB_SIZE)

    flops_at_b1_step1 = total_flops(batch=1, steps=1, model_config=model_cfg)
    batch_exact = budget / (flops_at_b1_step1 * STEPS_PER_RUN)
    batch = round_to_power_of_two_ceil(batch_exact)

    # Halve only on lr > LR_MAX. Configs above BATCH_MAX get dropped below.
    halvings = 0
    lr = scaled_lr(batch, hidden)
    while lr > LR_MAX and batch > 1:
        batch //= 2
        halvings += 1
        lr = scaled_lr(batch, hidden)

    def drop(
        reason: str,
        *,
        train_steps: int | None = None,
        train_tokens: int | None = None,
        achieved_flops: float | None = None,
    ) -> DroppedRun:
        return DroppedRun(
            budget=budget,
            hidden=hidden,
            reason=reason,
            batch_exact=batch_exact,
            batch=batch,
            halvings=halvings,
            lr=lr,
            train_steps=train_steps,
            train_tokens=train_tokens,
            achieved_flops=achieved_flops,
            params=params,
        )

    if batch > BATCH_MAX:
        return drop(f"batch={batch} > BATCH_MAX={BATCH_MAX}")
    if batch < BATCH_MIN:
        return drop(f"batch={batch} < BATCH_MIN={BATCH_MIN}")

    flops_at_b_step1 = total_flops(batch=batch, steps=1, model_config=model_cfg)
    train_steps = round(budget / flops_at_b_step1)
    achieved_flops = flops_at_b_step1 * train_steps
    train_tokens = batch * SEQ_LEN * train_steps

    if abs(achieved_flops - budget) / budget > FLOP_TOLERANCE:
        return drop(
            f"achieved_flops={achieved_flops:.3e} outside +/-{FLOP_TOLERANCE:.0%} of budget",
            train_steps=train_steps,
            train_tokens=train_tokens,
            achieved_flops=achieved_flops,
        )
    if params > MAX_PARAMS:
        return drop(
            f"params={params:,} > MAX_PARAMS={MAX_PARAMS:,}",
            train_steps=train_steps,
            train_tokens=train_tokens,
            achieved_flops=achieved_flops,
        )
    if train_tokens > MAX_TRAIN_TOKENS:
        return drop(
            f"train_tokens={fmt_count(train_tokens)} > MAX_TRAIN_TOKENS={fmt_count(MAX_TRAIN_TOKENS)}",
            train_steps=train_steps,
            train_tokens=train_tokens,
            achieved_flops=achieved_flops,
        )

    return IsoFlopRun(
        budget=budget,
        hidden=hidden,
        num_layers=model_cfg.num_layers,
        intermediate=model_cfg.intermediate_dim,
        num_heads=model_cfg.num_heads,
        num_kv_heads=model_cfg.num_kv_heads,
        batch_exact=batch_exact,
        batch=batch,
        halvings=halvings,
        train_steps=train_steps,
        train_tokens=train_tokens,
        achieved_flops=achieved_flops,
        params=params,
        lr=lr,
        beta2=scaled_beta2(batch),
        model_config=model_cfg,
    )


def all_candidates() -> list[IsoFlopRun | DroppedRun]:
    hidden_grid = list(range(HIDDEN_MIN, HIDDEN_MAX + 1, HIDDEN_STEP))
    return [_resolve_one(b, h) for b in BUDGETS for h in hidden_grid]


def valid_runs() -> list[IsoFlopRun]:
    return [r for r in all_candidates() if isinstance(r, IsoFlopRun)]


# === Run identity / tags ====================================================


def trial_name(run: IsoFlopRun) -> str:
    return (
        f"{RUN_NAME_PREFIX}-F{fmt_budget(run.budget)}-"
        f"P{fmt_count(run.params)}-T{fmt_count(run.train_tokens)}-{VERSION}"
    )


def trial_tags(run: IsoFlopRun) -> list[str]:
    return [
        "protein",
        "exp31",
        "isoflop",
        "qwen3",
        f"budget={fmt_budget(run.budget)}",
        f"budget_exact={run.budget:.1e}",
        f"params={fmt_count(run.params)}",
        f"params_exact={run.params}",
        f"tokens={fmt_count(run.train_tokens)}",
        f"tokens_exact={run.train_tokens}",
        f"batch={run.batch}",
        f"lr={fmt_lr(run.lr)}",
        f"lr_exact={run.lr:.4e}",
        f"beta2={run.beta2:.4f}",
    ]


# === Data config ============================================================


def empty_source_component(cache_dir: str, *, masked: bool) -> DatasetComponent:
    """Cache-only component: empty URL lists short-circuit Levanter's cache build."""
    fmt = TextLmDatasetFormat(text_key="document")
    source = UrlDatasetSourceConfig(
        train_urls=[],
        validation_urls=[],
        cache_dir=cache_dir,
        format=fmt,
        tags=[],
    )
    return DatasetComponent(
        source=source,
        cache_dir=cache_dir,
        format=fmt,
        pack=True,
        tags=[],
        loss_weight_fn=distance_bin_only_loss_weight if masked else None,
    )


def build_data_config() -> LmDataConfig:
    """cd-train (loss-masked) for sampling, cd-val (loss-masked) for eval."""
    components: dict[str, DatasetComponent] = {
        COMPONENT_CD_TRAIN: empty_source_component(CD_TRAIN_CACHE, masked=True),
        COMPONENT_CD_VAL: empty_source_component(CD_VAL_CACHE, masked=True),
    }
    train_weights: dict[str, float] = {COMPONENT_CD_TRAIN: 1.0, COMPONENT_CD_VAL: 0.0}
    return LmDataConfig(
        components=components,
        train_weights=train_weights,
        tokenizer=PROTEIN_TOKENIZER,
        cache_dir=None,
        block_cross_document_attention=True,
        shuffle=True,
        permutation_type="feistel",
        num_validation_sequences={},
    )


# === Trial construction =====================================================


def build_trial(run: IsoFlopRun) -> tuple[str, object]:
    train_config = SimpleTrainConfig(
        resources=resources(),
        train_batch_size=run.batch,
        num_train_steps=run.train_steps,
        learning_rate=versioned(run.lr),
        weight_decay=WEIGHT_DECAY,
        beta2=run.beta2,
        warmup=WARMUP,
        decay=LR_DECAY,
        lr_schedule=LR_SCHEDULE,
        train_seq_len=SEQ_LEN,
        steps_per_eval=run.train_steps,
        steps_per_export=None,
        max_eval_batches=None,
        data_seed=DATA_SEED,
        per_device_parallelism=-1,
    )
    job_name, raw_config = prepare_lm_train(
        name=trial_name(run),
        tokenized=build_data_config(),
        model_config=run.model_config,
        train_config=train_config,
        tags=trial_tags(run),
        eval_harness_tasks=[],
        use_default_validation=False,
        wandb_group=WANDB_GROUP,
    )
    raw_config = dataclasses.replace(
        raw_config,
        trainer=dataclasses.replace(
            raw_config.trainer,
            watch=WatchConfig(watch_targets=[], interval=0),
        ),
    )
    return job_name, raw_config


# === Selection / preview ====================================================


def selected_runs() -> tuple[str, ...]:
    raw = os.environ.get("RUNS", "")
    return tuple(s.strip() for s in raw.split(",") if s.strip())


def preview() -> bool:
    return os.environ.get("PREVIEW", "").strip().lower() in {"yes", "true", "1"}


def list_runs() -> bool:
    return os.environ.get("LIST_RUNS", "").strip().lower() in {"yes", "true", "1"}


def filter_targets(runs: list[IsoFlopRun], needles: tuple[str, ...]) -> list[IsoFlopRun]:
    if not needles:
        return runs
    return [r for r in runs if any(n in trial_name(r) for n in needles)]


def print_preview(candidates: list[IsoFlopRun | DroppedRun], targets: list[IsoFlopRun]) -> None:
    print(
        f"PREVIEW: {len(targets)}/{len(candidates)} target(s) after filters (STEPS_PER_RUN={STEPS_PER_RUN}):", flush=True
    )
    print(flush=True)
    by_budget: dict[float, list[IsoFlopRun]] = {b: [] for b in BUDGETS}
    for r in candidates:
        if isinstance(r, IsoFlopRun):
            by_budget.setdefault(r.budget, []).append(r)
    print(f"{'budget':>10}  {'survivors':>10}", flush=True)
    for b in BUDGETS:
        print(f"{fmt_budget(b):>10}  {len(by_budget.get(b, [])):>10}", flush=True)
    print(flush=True)
    # `halve` = LR-driven batch halvings. `D/N` = tokens / params (Chinchilla ~= 20).
    header = (
        f"{'budget':>8}  {'h':>5}  {'L':>3}  {'params':>9}  "
        f"{'b_exact':>10}  {'batch':>6}  {'halve':>5}  {'steps':>8}  {'tokens':>9}  "
        f"{'D/N':>6}  {'lr':>9}  {'beta2':>7}  {'flops':>10}  status"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for c in candidates:
        if isinstance(c, IsoFlopRun):
            dn = c.train_tokens / c.params if c.params else 0.0
            line = (
                f"{fmt_budget(c.budget):>8}  {c.hidden:>5}  {c.num_layers:>3}  {fmt_count(c.params):>9}  "
                f"{c.batch_exact:>10.2f}  {c.batch:>6}  {c.halvings:>5}  {c.train_steps:>8}  "
                f"{fmt_count(c.train_tokens):>9}  {dn:>6.1f}  "
                f"{c.lr:>9.3e}  {c.beta2:>7.4f}  {c.achieved_flops:>10.3e}  OK"
            )
        else:
            ach = f"{c.achieved_flops:.3e}" if c.achieved_flops is not None else "-"
            steps = f"{c.train_steps}" if c.train_steps is not None else "-"
            toks = fmt_count(c.train_tokens) if c.train_tokens is not None else "-"
            params = fmt_count(c.params) if c.params is not None else "-"
            dn = f"{c.train_tokens / c.params:.1f}" if c.train_tokens is not None and c.params else "-"
            line = (
                f"{fmt_budget(c.budget):>8}  {c.hidden:>5}  {'-':>3}  {params:>9}  "
                f"{c.batch_exact:>10.2f}  {c.batch:>6}  {c.halvings:>5}  {steps:>8}  {toks:>9}  "
                f"{dn:>6}  {c.lr:>9.3e}  {'-':>7}  {ach:>10}  DROP: {c.reason}"
            )
        print(line, flush=True)
    print(flush=True)
    print(f"Selected targets ({len(targets)}):", flush=True)
    for r in targets:
        print(f"  {trial_name(r)}", flush=True)


# === Worker + launcher ======================================================


def worker_entrypoint(rank: int, num_workers: int, needles: tuple[str, ...]) -> None:
    """One Fray worker: claim_and_run on ``targets[rank::num_workers]``."""
    runs = filter_targets(valid_runs(), needles)
    my_runs = runs[rank::num_workers]
    logger.info(
        "Worker rank=%d/%d assigned %d/%d target(s): %s",
        rank,
        num_workers,
        len(my_runs),
        len(runs),
        [trial_name(r) for r in my_runs],
    )
    targets = [SweepTarget(target_id=trial_name(r), config=r) for r in my_runs]
    res = resources()

    def run_one(target: SweepTarget) -> None:
        name, raw_config = build_trial(target.config)
        _run_training_on_worker(name=name, raw_config=raw_config, override_output_path=None, resources=res)

    claim_and_run(SWEEP_ROOT, targets, run_one)


def fray_job_name(rank: int, num_workers: int, runs: list[IsoFlopRun]) -> str:
    """Name a single-trial Fray job after its trial; otherwise use rank."""
    if len(runs) == 1 and num_workers == 1:
        return trial_name(runs[0])
    return f"{RUN_NAME_PREFIX}-w{rank}"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    candidates = all_candidates()
    needles = selected_runs()
    runs = filter_targets(valid_runs(), needles)
    if not runs:
        raise ValueError(f"No targets matched RUNS={needles!r} (out of {len(valid_runs())} valid runs)")

    if list_runs():
        for r in runs:
            print(trial_name(r), flush=True)
        return

    if preview():
        print_preview(candidates, runs)
        return

    num_workers = int(os.environ.get("NUM_WORKERS", str(DEFAULT_NUM_WORKERS)))
    num_workers = max(1, min(num_workers, len(runs)))
    res = resources()
    env = resolve_training_env(base_env=None, resources=res)
    extras = extras_for_resources(res)

    logger.info(
        "Submitting %d Fray worker(s); targets=%d runs=%s resources=%s",
        num_workers,
        len(runs),
        needles,
        res,
    )

    client = current_client()
    handles = []
    for rank in range(num_workers):
        request = JobRequest(
            name=fray_job_name(rank, num_workers, runs),
            entrypoint=Entrypoint.from_callable(worker_entrypoint, args=[rank, num_workers, needles]),
            resources=res,
            environment=create_environment(env_vars=env, extras=extras),
        )
        handles.append(client.submit(request))
        logger.info("Submitted worker rank=%d/%d: %s", rank, num_workers, request.name)

    failures = 0
    for rank, h in enumerate(handles):
        try:
            h.wait(raise_on_failure=True)
            logger.info("Worker rank=%d finished", rank)
        except Exception:
            failures += 1
            logger.exception("Worker rank=%d failed", rank)
    if failures:
        raise RuntimeError(f"{failures}/{num_workers} workers failed")


if __name__ == "__main__":
    main()
