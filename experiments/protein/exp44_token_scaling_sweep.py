# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exp 44: 1.5B ``cd`` token-scaling sweep.

Tracks ``Open-Athena/MarinFold#44``. Re-runs Issue #11's six ``cd`` scale
mixtures (m10..m15) at *reduced* token budgets to ask: how few tokens can we
train on and still recover the stage-2 mixture ranking / final
``protein-docs-cd-val`` loss?

Same data, tokenizer, 1.47B Llama, and optimizer recipe as
``exp11_data_mix_sweep.run_scale_sweep`` — only the token budget varies. The
full 21.5B-token run is **not** re-run here; it is the reused exp11 reference.

Token budgets (``num_evals=8`` each, so the step counts match the issue):

* ``t1`` — 0.5B tokens, 240 steps
* ``t2`` — 1B tokens, 480 steps
* ``t3`` — 2B tokens, 952 steps

18 configs = 3 budgets x 6 mixtures. Each config has id ``<budget>-<mixture>``
(e.g. ``t1-m10``) and is selected with ``RUNS``.

Structure mirrors ``exp29_arch_sweep`` / ``exp11_data_mix_sweep``: this script
runs as a lightweight CPU driver that submits ONE Fray training job (a v5p-8
worker via ``ResourceConfig.with_tpu``) for the selected config(s);
``claim_and_run`` makes re-running a completed config a no-op. The tokenized
caches were built once by ``exp11_data_mix_sweep`` (hardcoded paths below), so
there is no tokenize step here.

Env vars: ``RUNS`` (CSV substring filter on config id), ``PREVIEW=yes`` (list
targets, submit nothing), ``TPU`` (override worker TPU; default v5p-8).

Submission — one iris CPU driver per config, ``RUNS=<config>`` (hold <=12 at
once; shortest budget ``t1`` first). The driver is a CPU job
(``--region us-east5 --memory=1GB``, no ``--tpu``); the script then submits the
Fray ``with_tpu("v5p-8")`` worker that does the training. This is the exact
form used to launch the sweep::

    set -a; source ~/marin.env; set +a            # USERNAME, HF_TOKEN, WANDB_*
    export PATH="$HOME/google-cloud-sdk/bin:$HOME/.local/bin:$PATH"
    TIMESTAMP=$(date +%Y%m%d-%H%M)
    for cfg in t1-m10 t1-m11 t1-m12 t1-m13 t1-m14 t1-m15 \\
               t2-m10 t2-m11 t2-m12 t2-m13 t2-m14 t2-m15 \\
               t3-m10 t3-m11 t3-m12 t3-m13 t3-m14 t3-m15; do
        uv run iris --cluster=marin job run --user "$USERNAME" --no-wait \\
            --job-name prot-exp44-ts-${cfg}-${TIMESTAMP} \\
            --region us-east5 --memory=1GB \\
            -e HF_TOKEN "$HF_TOKEN" -e HUGGING_FACE_HUB_TOKEN "$HF_TOKEN" \\
            -e WANDB_API_KEY "$WANDB_API_KEY" -e WANDB_ENTITY "$WANDB_ENTITY" \\
            -e WANDB_PROJECT "$WANDB_PROJECT" \\
            -e RUNS ${cfg} \\
            -- python -m experiments.protein.exp44_token_scaling_sweep
    done

Preview without submitting::

    RUNS=t1-m10 PREVIEW=yes \\
        uv run python -m experiments.protein.exp44_token_scaling_sweep
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
from marin.execution.sweep import SweepTarget, claim_and_run
from marin.execution.types import versioned
from marin.training.training import extras_for_resources, resolve_training_env

from experiments.defaults import _run_training_on_worker, prepare_lm_train
from experiments.llama import compute_num_parameters
from experiments.protein.protein_train_common import (
    PROTEIN_TOKENIZER,
    distance_bin_only_loss_weight,
    protein_docs_val_tokenized,
)
from experiments.protein.train_protein_1_5b_distance_masked import protein_llama_1_5b
from experiments.simple_train_config import SimpleTrainConfig

logger = logging.getLogger(__name__)


# --- Identity ----------------------------------------------------------------

# Bump to fork run names + sweep-root claim dir on a recipe change (never
# deletes prior wandb/gcs data — a new VERSION writes to new paths).
VERSION: str = "v1"

RUN_NAME_PREFIX: str = "prot-exp44-ts"
SWEEP_ROOT: str = f"gs://marin-us-east5/sweeps/prot-exp44-token-scaling/run_token_sweep-{VERSION}"
WANDB_GROUP: str = "exp44-token-scaling"

# --- Data --------------------------------------------------------------------

# Quality-bucketed re-publication of cd-train tokens (H=round0, M=round1,
# L=round2..4), tokenized once by exp11_data_mix_sweep into per-(quality,split)
# Levanter caches. Hardcoded so this script is independent of exp11.
HF_DATASET_ID: str = "eczech/marinfold-exp11-protein-docs"
HF_REVISION: str = "41b2ec71070cb9e8799311cd8f78877e747f6754"

# Per-(quality, split) tokenized cache paths (Levanter cache layout
# ``<prefix>/<suffix>/{train,validation}/``). Identical to exp11's cache.
CACHE_PREFIX: str = "gs://marin-us-east5/tokenized/exp11-data-mix-41b2ec7-v1"
H_TRAIN_CACHE: str = f"{CACHE_PREFIX}/high-train/"
M_TRAIN_CACHE: str = f"{CACHE_PREFIX}/medium-train/"
L_TRAIN_CACHE: str = f"{CACHE_PREFIX}/low-train/"
H_VAL_CACHE: str = f"{CACHE_PREFIX}/high-val/"
H_TEST_CACHE: str = f"{CACHE_PREFIX}/high-test/"

# Existing cd-val cache built by ``protein_train_common`` (the primary metric).
CD_VAL_CACHE: str = protein_docs_val_tokenized.override_output_path

# Component names — row keys in MixtureDataset weights and the prefix of every
# ``eval/<component>/loss`` series in W&B.
COMPONENT_H_TRAIN: str = "protein-docs-high-train"
COMPONENT_M_TRAIN: str = "protein-docs-medium-train"
COMPONENT_L_TRAIN: str = "protein-docs-low-train"
COMPONENT_H_VAL: str = "protein-docs-high-val"
COMPONENT_H_TEST: str = "protein-docs-high-test"
COMPONENT_CD_VAL: str = "protein-docs-cd-val"

# Quality -> train component name (for resolving mixture weights).
QUALITY_TRAIN_COMPONENT: dict[str, str] = {
    "H": COMPONENT_H_TRAIN,
    "M": COMPONENT_M_TRAIN,
    "L": COMPONENT_L_TRAIN,
}

# Eval-only heldout cells (matches exp11's scale sweep: H val/test + cd-val).
HELDOUT_COMPONENTS: tuple[str, ...] = (COMPONENT_H_VAL, COMPONENT_H_TEST, COMPONENT_CD_VAL)

# IID held-out sequences carved per active train cell; ~4096 ≈ 33.5M tokens.
IID_EVAL_SEQS_PER_TRAIN: int = 4096

# Per-component eval batch cap. With grad_accum_steps=2 the eval batch is the
# microbatch (per_device_eval_parallelism=per_device_parallelism=32, x4 chips
# on v5p-8 = 128), so this evaluates 16 * 128 = 2048 sequences (~16.8M tokens),
# identical to exp11's run_scale_sweep cd-val eval.
MAX_EVAL_BATCHES: int = 16

# --- Model -------------------------------------------------------------------

# gradient_checkpointing="nested" — multi-level scan saves only sqrt(num_layers)
# ≈ 5 carries instead of 24. Combined with grad_accum_steps=2 to fit the
# post-May-2026 v5p-8 HBM budget. Identical to exp11's scale model.
MODEL_CONFIG = dataclasses.replace(protein_llama_1_5b, gradient_checkpointing="nested")
GRAD_ACCUM_STEPS: int = 2

# Vocab for the legacy 2840-vocab tokenizer pinned in PROTEIN_TOKENIZER.
PROTEIN_VOCAB_SIZE: int = 2840

HIDDEN_DIM: int = protein_llama_1_5b.hidden_dim  # 2048

# --- Optimizer / schedule ----------------------------------------------------

BATCH_SIZE: int = 256
SEQ_LEN: int = 8192
WEIGHT_DECAY: float = 0.01
WARMUP: float = 0.1

# LR scaling — adapted from ``train_protein_1_5b_distance_masked.py``:
#   reference: lr=3.5e-4 @ batch=128, hidden=2048
#   solve: LR_CONSTANT = lr * hidden / sqrt(batch)
#   apply: lr_here = LR_CONSTANT * sqrt(BATCH_SIZE) / HIDDEN_DIM
# At BATCH_SIZE=256 / HIDDEN_DIM=2048 this resolves to ~4.95e-4.
LR_REF: float = 3.5e-4
LR_REF_BATCH: int = 128
LR_REF_HIDDEN: int = 2048
LR_CONSTANT: float = LR_REF * LR_REF_HIDDEN / math.sqrt(LR_REF_BATCH)
LEARNING_RATE: float = LR_CONSTANT * math.sqrt(BATCH_SIZE) / HIDDEN_DIM

# Adam β₂ scaled per the noise-scale heuristic (0.98 at batch=128 → 0.9604 at 256).
BETA2: float = 0.98 ** (BATCH_SIZE / LR_REF_BATCH)

# WSD-style linear decay to ``LR_DECAY * peak`` over the trailing fraction.
# warmup/decay are fractions of each run's own num_train_steps, so the schedule
# shape is preserved across budgets and fully decayed at every run's end.
LR_SCHEDULE: str = "linear"
LR_DECAY: float = 0.2

# Pinned so cross-run comparisons share the same data permutation (matches the
# exp11 scale sweep so cd-val numbers stay directly comparable).
DATA_SEED: int = 1729
SHUFFLE: bool = True
PERMUTATION_TYPE: str = "feistel"
MIXTURE_BLOCK_SIZE: int = 2048

# Evals per run. With BATCH_SIZE=256, seq=8192 this lands the budgets on the
# step counts quoted in the issue (0.5B→240, 1B→480, 2B→952).
NUM_EVALS: int = 8

# Rolling temp-checkpoint cadence; permanent intermediate checkpoints are not
# kept (steps_per_export=None) — the metric is the final/eval cd-val loss.
TEMP_CHECKPOINT_INTERVAL = timedelta(minutes=8)

# --- Resources --------------------------------------------------------------

# us-east5-a co-locates TPUs with the ``marin-us-east5`` checkpoint bucket.
PROTEIN_ZONE: str = "us-east5-a"
DEFAULT_TPU: str = "v5p-8"


def tpu() -> str:
    return os.environ.get("TPU") or DEFAULT_TPU


def resources() -> ResourceConfig:
    return ResourceConfig.with_tpu(tpu(), zone=PROTEIN_ZONE)


# --- Mixtures ----------------------------------------------------------------


@dataclass(frozen=True)
class Mixture:
    """One named train-data mixture over (H, M, L) train cells.

    Either ``static`` (flat dict) or ``staged`` (list of ``(stage_frac, weights)``
    entries). Staged fractions are resolved by :func:`resolve_mixture_weights`,
    which snaps each boundary to a ``MixtureDataset`` block edge.
    """

    id: str
    static: dict[str, float] | None = None
    staged: tuple[tuple[float, dict[str, float]], ...] | None = None


# The six ``cd`` scale mixtures, identical to exp11's ``SCALE_MIXTURES``:
# m10 H-only, m11 size-proportional (H-31/M-26/L-43), m12 L→H staged,
# m13 three-stage L→M→H (transitions at 0.43 and 0.43+0.26=0.69), m14 M-only,
# m15 L-only.
SCALE_MIXTURES: tuple[Mixture, ...] = (
    Mixture(id="m10", static={"H": 1.0}),
    Mixture(id="m11", static={"H": 0.31, "M": 0.26, "L": 0.43}),
    Mixture(id="m12", staged=((0.0, {"L": 1.0}), (0.5, {"H": 1.0}))),
    Mixture(
        id="m13",
        staged=(
            (0.0, {"L": 1.0}),
            (0.43, {"M": 1.0}),
            (0.69, {"H": 1.0}),
        ),
    ),
    Mixture(id="m14", static={"M": 1.0}),
    Mixture(id="m15", static={"L": 1.0}),
)


def resolve_mixture_weights(
    mixture: Mixture, num_train_steps: int
) -> dict[str, float] | list[tuple[int, dict[str, float]]]:
    """Resolve a mixture to MixtureDataset's expected per-quality weights form.

    Staged: snaps ``frac * num_train_steps`` DOWN to a multiple of
    ``MIXTURE_BLOCK_SIZE // BATCH_SIZE`` so ``step * batch`` lands on a block
    boundary. First stage pinned to 0; monotonicity re-asserted after snapping.
    """
    if mixture.static is not None:
        return dict(mixture.static)
    assert mixture.staged is not None

    assert MIXTURE_BLOCK_SIZE % BATCH_SIZE == 0, (
        f"MIXTURE_BLOCK_SIZE ({MIXTURE_BLOCK_SIZE}) must be a multiple of "
        f"BATCH_SIZE ({BATCH_SIZE}) so stage boundaries land on block edges"
    )
    alignment = MIXTURE_BLOCK_SIZE // BATCH_SIZE

    stages: list[tuple[int, dict[str, float]]] = []
    for i, (frac, w) in enumerate(mixture.staged):
        if i == 0:
            step = 0
        else:
            raw_step = round(frac * num_train_steps)
            step = (raw_step // alignment) * alignment
            prev = stages[-1][0]
            if step <= prev:
                step = prev + alignment  # keep stages strictly increasing
        stages.append((step, dict(w)))
    if stages[-1][0] >= num_train_steps:
        raise ValueError(
            f"mixture {mixture.id}: last stage step {stages[-1][0]} >= num_train_steps "
            f"{num_train_steps} after alignment; bump num_train_steps or rethink stages"
        )
    return stages


# --- Token budgets -----------------------------------------------------------


@dataclass(frozen=True)
class TokenBudget:
    """One reduced token budget; step count is derived via :func:`schedule`."""

    id: str  # "t1" | "t2" | "t3"
    target_tokens: int
    num_evals: int = NUM_EVALS

    def schedule(self) -> tuple[int, int]:
        """``(num_train_steps, steps_per_eval)`` rounded so total = evals*spe."""
        tokens_per_step = BATCH_SIZE * SEQ_LEN
        spe = max(1, round(self.target_tokens / self.num_evals / tokens_per_step))
        return spe * self.num_evals, spe


# 0.5B → 240 steps, 1B → 480 steps, 2B → 952 steps (BATCH_SIZE=256, seq=8192).
# The full 21.5B run is the reused exp11 reference and is intentionally absent.
TOKEN_BUDGETS: tuple[TokenBudget, ...] = (
    TokenBudget(id="t1", target_tokens=500_000_000),
    TokenBudget(id="t2", target_tokens=1_000_000_000),
    TokenBudget(id="t3", target_tokens=2_000_000_000),
)


# --- Configs (budget x mixture) ----------------------------------------------


@dataclass(frozen=True)
class Config:
    """One trial: a (token budget, mixture) pair."""

    budget: TokenBudget
    mixture: Mixture

    @property
    def config_id(self) -> str:
        """Short selector id, e.g. ``t1-m10`` (used by ``RUNS``)."""
        return f"{self.budget.id}-{self.mixture.id}"


ALL_CONFIGS: tuple[Config, ...] = tuple(Config(b, m) for b in TOKEN_BUDGETS for m in SCALE_MIXTURES)
CONFIG_BY_ID: dict[str, Config] = {c.config_id: c for c in ALL_CONFIGS}


# --- Helpers ----------------------------------------------------------------


def fmt_count(n: int) -> str:
    """Compact precise count, e.g. 120.7M, 1.007B."""
    if n >= 1_000_000_000:
        return f"{n / 1e9:.3f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)


def fmt_lr(lr: float) -> str:
    """Format LR for run names, e.g. ``4.9e-4``."""
    mantissa, exponent = f"{lr:.1e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def trial_name(config: Config) -> str:
    """Stable per-trial run id (also the per-target dir under ``SWEEP_ROOT``)."""
    num_train_steps, _ = config.budget.schedule()
    tokens_tag = fmt_count(BATCH_SIZE * SEQ_LEN * num_train_steps)
    return (
        f"{RUN_NAME_PREFIX}-1_5b-{config.budget.id}-{tokens_tag}-"
        f"{config.mixture.id}-lr{fmt_lr(LEARNING_RATE)}-{VERSION}"
    )


# --- Data config ------------------------------------------------------------


def empty_source_component(cache_dir: str, *, masked: bool) -> DatasetComponent:
    """Cache-only component: empty URL lists short-circuit Levanter's cache-build.

    Loads ``<cache_dir>/{train,validation}/`` if present; loss masking applied
    iff ``masked``.
    """
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


def _has_nonzero_weight(train_weights: dict[str, float] | list[tuple[int, dict[str, float]]], name: str) -> bool:
    """Mirror of Levanter's ``LmDataConfig._has_nonzero_weight``.

    ``build_caches("train")`` skips components zero in every stage; an
    IID-carve request on such a component KeyErrors. Keep in lockstep.
    """
    if isinstance(train_weights, dict):
        return train_weights.get(name, 0) > 0
    return any(w.get(name, 0) > 0 for _, w in train_weights)


def _expand_quality_weights(qw: dict[str, float]) -> dict[str, float]:
    """Per-quality weights -> per-component weights (+ heldout cells at 0)."""
    out = {comp: float(qw.get(q, 0.0)) for q, comp in QUALITY_TRAIN_COMPONENT.items()}
    for comp in HELDOUT_COMPONENTS:
        out[comp] = 0.0
    return out


def build_data_config(config: Config, num_train_steps: int) -> LmDataConfig:
    """LmDataConfig for one config's mixture at a given step count.

    The three train cells always appear in ``components``/``train_weights``
    (zero for unused qualities); ``num_validation_sequences`` only references
    active (non-zero) train cells.
    """
    components: dict[str, DatasetComponent] = {
        COMPONENT_H_TRAIN: empty_source_component(H_TRAIN_CACHE, masked=True),
        COMPONENT_M_TRAIN: empty_source_component(M_TRAIN_CACHE, masked=True),
        COMPONENT_L_TRAIN: empty_source_component(L_TRAIN_CACHE, masked=True),
        COMPONENT_H_VAL: empty_source_component(H_VAL_CACHE, masked=True),
        COMPONENT_H_TEST: empty_source_component(H_TEST_CACHE, masked=True),
        COMPONENT_CD_VAL: empty_source_component(CD_VAL_CACHE, masked=True),
    }

    weights = resolve_mixture_weights(config.mixture, num_train_steps)
    if isinstance(weights, dict):
        train_weights: dict | list = _expand_quality_weights(weights)
    else:
        train_weights = [(step, _expand_quality_weights(w)) for step, w in weights]

    num_validation_sequences = {
        comp: IID_EVAL_SEQS_PER_TRAIN
        for comp in (COMPONENT_H_TRAIN, COMPONENT_M_TRAIN, COMPONENT_L_TRAIN)
        if _has_nonzero_weight(train_weights, comp)
    }
    return LmDataConfig(
        components=components,
        train_weights=train_weights,
        tokenizer=PROTEIN_TOKENIZER,
        cache_dir=None,
        block_cross_document_attention=True,
        shuffle=SHUFFLE,
        permutation_type=PERMUTATION_TYPE,
        mixture_block_size=MIXTURE_BLOCK_SIZE,
        num_validation_sequences=num_validation_sequences,
    )


# --- Trial construction -----------------------------------------------------


def build_trial(config: Config) -> tuple[str, object]:
    """Build one trial's ``(job_name, raw_config)`` for ``prepare_lm_train``."""
    num_train_steps, steps_per_eval = config.budget.schedule()
    res = resources()

    # grad_accum_steps=2 splits each step into 2 microbatches to halve
    # activation HBM at compile (dodges the v5p-8 CompileTimeHbmOom on 1.5B).
    microbatch = BATCH_SIZE // GRAD_ACCUM_STEPS
    chips = res.chip_count()
    if BATCH_SIZE % GRAD_ACCUM_STEPS != 0:
        raise ValueError(f"batch_size ({BATCH_SIZE}) not divisible by grad_accum_steps ({GRAD_ACCUM_STEPS})")
    if microbatch % chips != 0:
        raise ValueError(f"microbatch ({microbatch}) not divisible by chip_count ({chips})")
    per_device_parallelism = microbatch // chips

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
        steps_per_export=None,
        max_eval_batches=MAX_EVAL_BATCHES,
        data_seed=DATA_SEED,
        per_device_parallelism=per_device_parallelism,
    )
    params = compute_num_parameters(MODEL_CONFIG, PROTEIN_VOCAB_SIZE)
    tokens = BATCH_SIZE * SEQ_LEN * num_train_steps
    job_name, raw_config = prepare_lm_train(
        name=trial_name(config),
        tokenized=build_data_config(config, num_train_steps),
        model_config=MODEL_CONFIG,
        train_config=train_config,
        tags=[
            "protein",
            "exp44",
            "token-scaling",
            "1_5b",
            config.mixture.id,
            config.budget.id,
            f"params={fmt_count(params)}",
            f"params_exact={params}",
            f"tokens={fmt_count(tokens)}",
            f"tokens_exact={tokens}",
            f"steps={num_train_steps}",
        ],
        eval_harness_tasks=[],
        use_default_validation=False,
        wandb_group=WANDB_GROUP,
    )
    # Override defaults.py's 10-min checkpoint cadence; disable per-parameter
    # watch tracking (matches exp11 scale's HBM-pressure workaround).
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


# --- Selection / preview ----------------------------------------------------


def selected_configs() -> tuple[Config, ...]:
    """``RUNS`` is a CSV substring filter on ``config_id``; empty = all 18."""
    raw = os.environ.get("RUNS", "")
    needles = tuple(s.strip() for s in raw.split(",") if s.strip())
    if not needles:
        return ALL_CONFIGS
    return tuple(c for c in ALL_CONFIGS if any(n in c.config_id for n in needles))


def preview() -> bool:
    return os.environ.get("PREVIEW", "").strip().lower() in {"yes", "true", "1"}


def _format_weights(weights: dict[str, float]) -> str:
    return ", ".join(f"{k}={v:.3g}" for k, v in weights.items())


def print_preview(configs: tuple[Config, ...]) -> None:
    print(f"PREVIEW: exp44 token-scaling would run {len(configs)} target(s):", flush=True)
    for c in configs:
        num_train_steps, steps_per_eval = c.budget.schedule()
        tokens_tag = fmt_count(BATCH_SIZE * SEQ_LEN * num_train_steps)
        weights = resolve_mixture_weights(c.mixture, num_train_steps)
        print(f"  {c.config_id}  {trial_name(c)}", flush=True)
        print(
            f"    budget={c.budget.id} batch={BATCH_SIZE} steps={num_train_steps} "
            f"steps_per_eval={steps_per_eval} lr={LEARNING_RATE:.4g} beta2={BETA2:.4g} "
            f"tokens={tokens_tag} schedule={LR_SCHEDULE},decay={LR_DECAY} data_seed={DATA_SEED}",
            flush=True,
        )
        if isinstance(weights, dict):
            print(f"    mixture {c.mixture.id} (static): {_format_weights(weights)}", flush=True)
        else:
            print(f"    mixture {c.mixture.id} (staged, {len(weights)} stages):", flush=True)
            for step, w in weights:
                print(f"      step {step:>6} (seq {step * BATCH_SIZE:>10}): {_format_weights(w)}", flush=True)
        print(flush=True)


# --- Worker + launcher ------------------------------------------------------


def worker_entrypoint(config_ids: tuple[str, ...]) -> None:
    """One Fray worker: run the selected config(s) via ``claim_and_run``.

    The bash launcher submits one iris job per config, so each worker normally
    sees a single config id. ``claim_and_run`` keeps re-running a completed
    config a no-op.
    """
    targets = [
        SweepTarget(target_id=trial_name(c), config=c.config_id) for c in ALL_CONFIGS if c.config_id in config_ids
    ]
    logger.info("Worker assigned %d/%d target(s): %s", len(targets), len(ALL_CONFIGS), [t.target_id for t in targets])

    def run_one(target: SweepTarget) -> None:
        config = CONFIG_BY_ID[target.config]
        name, raw_config = build_trial(config)
        _run_training_on_worker(name=name, raw_config=raw_config, override_output_path=None, resources=resources())

    claim_and_run(SWEEP_ROOT, targets, run_one)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    configs = selected_configs()
    if not configs:
        raise ValueError(f"No configs matched RUNS={os.environ.get('RUNS', '')!r}")

    if preview():
        print_preview(configs)
        return

    res = resources()
    env = resolve_training_env(base_env=None, resources=res)
    extras = extras_for_resources(res)
    logger.info("Submitting 1 Fray worker job; configs=%s", [c.config_id for c in configs])

    client = current_client()
    request = JobRequest(
        name=f"{RUN_NAME_PREFIX}-w0",
        entrypoint=Entrypoint.from_callable(worker_entrypoint, args=[tuple(c.config_id for c in configs)]),
        resources=res,
        environment=create_environment(env_vars=env, extras=extras),
    )
    handle = client.submit(request)
    logger.info("Submitted worker: %s", request.name)
    handle.wait(raise_on_failure=True)
    logger.info("Worker finished")


if __name__ == "__main__":
    main()
