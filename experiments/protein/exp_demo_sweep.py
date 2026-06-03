# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exp demo: does AA-sequence pretraining data help the structure-token task?

Hypothesis
----------
Training on amino-acid *sequence* tokens (no structure statements, standard LM
loss) in addition to the structure documents improves performance on the
structure (distance-bin) task. The eval metric is the held-out
``protein-docs-cd-val`` distance-bin loss.

Two mixtures, 3 seeds each (6 runs):

* ``m1`` — 100% structure docs (``eczech/marinfold-exp11-protein-docs``,
  config ``low``, split ``train``), distance-bin-masked loss, **500M tokens**.
* ``m2`` — 50/50 mix of the same structure docs and an *additional* 500M
  tokens of AA-sequence-only docs (``eczech/marinfold-exp11-protein-docs-seq``,
  config ``low``, split ``train``), **1B tokens total**.

Design note — additive, not fixed-compute
------------------------------------------
``m2`` is 1B tokens (500M docs + 500M seq), NOT 500M total. This holds the
structure-doc exposure fixed at 500M in *both* arms, so any change in cd-val
loss is attributable to the added sequence data rather than to seeing less
structure data. A fixed-compute reading (250M docs + 250M seq) would confound
the two. Flip :data:`M2_TOTAL_TOKENS` to ``500_000_000`` for the fixed-compute
variant.

Loss masking
------------
The structure docs carry ``<distance> ... <d_value>`` statements; the loss is
zeroed everywhere except the ``<d_value>`` bin (``distance_bin_only_loss_weight``,
same as exp44). The sequence docs are ``<begin_sequence> <AA>...`` ONLY — they
contain no ``<distance>`` token, so the distance mask would zero *all* of their
loss. The seq component is therefore trained **unmasked** (standard next-token
LM loss over the AA tokens); that is the whole point of the ablation. The
cd-val eval component is distance-bin-masked (it measures the structure task).

Packing is identical in all cases: ``pack=True`` with
``block_cross_document_attention=True`` (no cross-document attention).

Recipe (model / optimizer / schedule)
-------------------------------------
Model is exp44's 1.47B config dims, swapped from ``LlamaConfig`` to
``Qwen3Config`` (h=2048, dff=8192, heads=32, kv=8, layers=24). The global
batch is lowered to 128 (from exp44's 256), so there is no nested gradient
checkpointing and no gradient accumulation. Qwen3 seq_len 8192. WSD LR
schedule (10% warmup + 20% linear decay, constant in between) with the peak LR
solved from the ``lr * hidden / sqrt(batch)`` heuristic — at batch=128/h=2048
this lands on the 3.5e-4 reference. β₂ via the noise-scale heuristic (0.98 at
batch 128). 5 evals per run; rolling temp checkpoints every 8 min plus one
permanent (final) checkpoint.

Execution (mirrors exp44 / write-sweep / run-iris-job)
------------------------------------------------------
This script is a lightweight CPU driver that submits ONE Fray training job (a
v5p-8 ``with_tpu`` worker) for the selected config(s); ``claim_and_run`` makes
re-running a completed config a no-op. Caches are prebuilt: the docs and cd-val
caches already exist (reused from exp11 / protein_train_common); the seq cache
is built once by the ``TOKENIZE`` step below.

Step 0 — build the seq cache once. The driver is a lightweight CPU coordinator:
``executor_main`` dispatches the ``default_tokenize`` step to the cluster (it
runs on its own ``cpu=4, ram=16g`` worker), so the driver itself needs only
modest resources::

    set -a; source ~/marin.env; set +a
    export PATH="$HOME/google-cloud-sdk/bin:$HOME/.local/bin:$PATH"
    uv run iris --cluster=marin job run --user "$USERNAME" --no-wait \\
        --job-name prot-exp-demo-tokenize-seq \\
        --region us-east5 --cpu=1 --memory=1GB --extra=cpu \\
        -e HF_TOKEN "$HF_TOKEN" -e HUGGING_FACE_HUB_TOKEN "$HF_TOKEN" \\
        -e TOKENIZE yes \\
        -- python -m experiments.protein.exp_demo_sweep

Step 1 — launch the sweep, one CPU driver per config::

    TIMESTAMP=$(date +%Y%m%d-%H%M)
    for cfg in m1-s0 m1-s1 m1-s2 m2-s0 m2-s1 m2-s2; do
        uv run iris --cluster=marin job run --user "$USERNAME" --no-wait \\
            --job-name prot-exp-demo-${cfg}-${TIMESTAMP} \\
            --region us-east5 --memory=1GB \\
            -e HF_TOKEN "$HF_TOKEN" -e HUGGING_FACE_HUB_TOKEN "$HF_TOKEN" \\
            -e WANDB_API_KEY "$WANDB_API_KEY" -e WANDB_ENTITY "$WANDB_ENTITY" \\
            -e WANDB_PROJECT "$WANDB_PROJECT" \\
            -e RUNS ${cfg} \\
            -- python -m experiments.protein.exp_demo_sweep
    done

Preview without submitting::

    PREVIEW=yes uv run python -m experiments.protein.exp_demo_sweep
    RUNS=m2 PREVIEW=yes uv run python -m experiments.protein.exp_demo_sweep

Env vars: ``RUNS`` (CSV substring filter on config id, e.g. ``m1``, ``m2-s1``),
``PREVIEW=yes`` (list targets, submit nothing), ``TOKENIZE=yes`` (build the seq
cache and exit), ``TPU`` (override worker TPU; default v5p-8).
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
from marin.execution.executor import executor_main
from marin.execution.sweep import SweepTarget, claim_and_run
from marin.execution.types import versioned
from marin.training.training import extras_for_resources, resolve_training_env

from experiments.defaults import _run_training_on_worker, default_tokenize, prepare_lm_train
from experiments.llama import compute_num_parameters
from experiments.protein.protein_train_common import (
    PROTEIN_TOKENIZER,
    distance_bin_only_loss_weight,
    protein_docs_val_tokenized,
)
from experiments.simple_train_config import SimpleTrainConfig

logger = logging.getLogger(__name__)


# --- Identity ----------------------------------------------------------------

# Bump to fork run names + sweep-root claim dir on a recipe change (never
# deletes prior wandb/gcs data — a new VERSION writes to new paths).
VERSION: str = "v1"

RUN_NAME_PREFIX: str = "prot-exp-demo"
SWEEP_ROOT: str = f"gs://marin-us-east5/sweeps/prot-exp-demo-seq-ablation/run-{VERSION}"
WANDB_GROUP: str = "exp-demo-seq-ablation"

# --- Data --------------------------------------------------------------------

# Structure docs: cd-train tokens, quality bucket "low". Reuse exp11's existing
# Levanter cache (built from eczech/marinfold-exp11-protein-docs@41b2ec7,
# config=low, split=train); empty source URLs load it cache-only.
DOCS_TRAIN_CACHE: str = "gs://marin-us-east5/tokenized/exp11-data-mix-41b2ec7-v1/low-train/"

# AA-sequence-only docs: NEW data with no prebuilt cache. Built once by the
# TOKENIZE step (default_tokenize -> SEQ_TRAIN_CACHE/train/) and then loaded
# cache-only by the sweep, identical to the docs/cd-val caches.
SEQ_HF_DATASET_ID: str = "eczech/marinfold-exp11-protein-docs-seq"
SEQ_HF_REVISION: str = "1fe8de92e638e50aabf0ce05a83590654d7ceb09"
SEQ_TRAIN_URL: str = f"hf://datasets/{SEQ_HF_DATASET_ID}@{SEQ_HF_REVISION}/low/train/"
SEQ_TRAIN_CACHE: str = "gs://marin-us-east5/tokenized/exp-demo-protein-docs-seq-low-1fe8de9-v1"

# Existing cd-val cache built by ``protein_train_common`` (the primary metric):
# timodonnell/protein-docs, config contacts-and-distances-v1-5x, split val.
CD_VAL_CACHE: str = protein_docs_val_tokenized.override_output_path

# Component names — row keys in the mixture weights and the prefix of every
# ``eval/<component>/loss`` series in W&B.
COMPONENT_DOCS: str = "protein-docs-low-train"
COMPONENT_SEQ: str = "protein-docs-seq-low-train"
COMPONENT_CD_VAL: str = "protein-docs-cd-val"

# Tokenize step for the new seq data; pinned output path so re-running is a
# no-op once the cache exists. Run via ``TOKENIZE=yes`` (a CPU iris job).
seq_tokenized = dataclasses.replace(
    default_tokenize(
        name="exp-demo-protein-docs-seq-low",
        dataset=SEQ_TRAIN_URL,
        tokenizer=PROTEIN_TOKENIZER,
        format=TextLmDatasetFormat(text_key="document"),
    ),
    override_output_path=SEQ_TRAIN_CACHE,
)

# --- Model -------------------------------------------------------------------

# exp44's 1.47B config dims (h=2048, dff=8192, heads=32, kv=8, layers=24),
# swapped from LlamaConfig to Qwen3Config. Llama3 rope to match the repo's
# other qwen3.py configs. No nested gradient checkpointing and no gradient
# accumulation: the global batch is lowered to 128 (from exp44's 256), so the
# default checkpointing fits the v5p-8 HBM budget on its own.
MODEL_CONFIG = Qwen3Config(
    max_seq_len=8192,
    hidden_dim=2048,
    intermediate_dim=8192,
    num_heads=32,
    num_kv_heads=8,
    num_layers=24,
    rope=Llama3RotaryEmbeddingsConfig(),
)

# Vocab for the legacy 2840-vocab tokenizer pinned in PROTEIN_TOKENIZER.
PROTEIN_VOCAB_SIZE: int = 2840

HIDDEN_DIM: int = MODEL_CONFIG.hidden_dim  # 2048

# --- Optimizer / schedule ----------------------------------------------------

BATCH_SIZE: int = 128
SEQ_LEN: int = 8192
WEIGHT_DECAY: float = 0.01
WARMUP: float = 0.1

# LR scaling — adapted from ``train_protein_1_5b_distance_masked.py``:
#   reference: lr=3.5e-4 @ batch=128, hidden=2048
#   solve: LR_CONSTANT = lr * hidden / sqrt(batch)
#   apply: lr_here = LR_CONSTANT * sqrt(BATCH_SIZE) / HIDDEN_DIM
# At BATCH_SIZE=128 / HIDDEN_DIM=2048 this resolves back to the 3.5e-4 reference.
LR_REF: float = 3.5e-4
LR_REF_BATCH: int = 128
LR_REF_HIDDEN: int = 2048
LR_CONSTANT: float = LR_REF * LR_REF_HIDDEN / math.sqrt(LR_REF_BATCH)
LEARNING_RATE: float = LR_CONSTANT * math.sqrt(BATCH_SIZE) / HIDDEN_DIM

# Adam β₂ scaled per the noise-scale heuristic (0.98 at batch=128).
BETA2: float = 0.98 ** (BATCH_SIZE / LR_REF_BATCH)

# WSD: linear warmup (WARMUP) -> constant -> linear decay over the trailing
# LR_DECAY fraction. warmup/decay are fractions of each run's own
# num_train_steps, so both mixtures fully decay at their own end.
LR_SCHEDULE: str = "linear"
LR_DECAY: float = 0.2

# Data-ordering seed sweep: 3 seeds per mixture. Each seed sets BOTH the
# trainer seed (model init / training key) and the data_seed (data permutation),
# so the three runs are fully independent replicates.
SEEDS: tuple[int, ...] = (1729, 1730, 1731)

SHUFFLE: bool = True
PERMUTATION_TYPE: str = "feistel"
MIXTURE_BLOCK_SIZE: int = 2048

# Evals per run.
NUM_EVALS: int = 5

# Eval on 8192 cd-val sequences: eval batch = per_device_eval_parallelism *
# chips = 32 * 4 = 128, so 64 batches = 8192 sequences.
EVAL_EXAMPLES: int = 8192

# Rolling temp-checkpoint cadence; one permanent (final) checkpoint only
# (steps_per_export=None) — the metric is the final/eval cd-val loss.
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

# m2's total token budget. Additive design: 500M docs + 500M seq holds the
# structure-doc exposure fixed at 500M (== m1) so the ablation isolates the
# effect of the added seq data. Set to 500_000_000 for the fixed-compute
# variant (250M docs + 250M seq).
M1_TOTAL_TOKENS: int = 500_000_000
M2_TOTAL_TOKENS: int = 1_000_000_000


@dataclass(frozen=True)
class Mixture:
    """One named train mixture over the (docs, seq) cells + its token budget."""

    id: str
    weights: dict[str, float]  # over COMPONENT_DOCS / COMPONENT_SEQ
    target_tokens: int

    def schedule(self) -> tuple[int, int]:
        """``(num_train_steps, steps_per_eval)`` rounded so total = evals*spe."""
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
    """One trial: a (mixture, seed) pair."""

    mixture: Mixture
    seed_index: int

    @property
    def seed(self) -> int:
        return SEEDS[self.seed_index]

    @property
    def config_id(self) -> str:
        """Short selector id, e.g. ``m1-s0``."""
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
    """Format LR for run names, e.g. ``3.5e-4``."""
    mantissa, exponent = f"{lr:.1e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def trial_name(config: Config) -> str:
    """Stable per-trial run id (also the per-target dir under ``SWEEP_ROOT``)."""
    num_train_steps, _ = config.mixture.schedule()
    tokens_tag = fmt_count(BATCH_SIZE * SEQ_LEN * num_train_steps)
    return (
        f"{RUN_NAME_PREFIX}-1_5b-{config.mixture.id}-{tokens_tag}-"
        f"s{config.seed_index}-lr{fmt_lr(LEARNING_RATE)}-{VERSION}"
    )


# --- Data config ------------------------------------------------------------


def cache_only_component(cache_dir: str, *, masked: bool) -> DatasetComponent:
    """Cache-only component: empty URL lists short-circuit Levanter's cache-build.

    Loads ``<cache_dir>/{train,validation}/`` if present. ``pack=True`` with
    cross-document attention blocked (set on LmDataConfig). Distance-bin-only
    loss mask applied iff ``masked``; otherwise standard next-token LM loss.
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


def build_data_config(config: Config) -> LmDataConfig:
    """LmDataConfig for one config's mixture.

    Both train cells always appear in ``components``/``train_weights`` (zero for
    unused cells). Structure docs + cd-val are distance-bin-masked; the seq cell
    is unmasked (standard LM loss on the AA tokens). The cd-val component has
    only a validation split (train weight 0) so it is eval-only.
    """
    components: dict[str, DatasetComponent] = {
        COMPONENT_DOCS: cache_only_component(DOCS_TRAIN_CACHE, masked=True),
        COMPONENT_SEQ: cache_only_component(SEQ_TRAIN_CACHE, masked=False),
        COMPONENT_CD_VAL: cache_only_component(CD_VAL_CACHE, masked=True),
    }
    train_weights = {
        COMPONENT_DOCS: float(config.mixture.weights.get(COMPONENT_DOCS, 0.0)),
        COMPONENT_SEQ: float(config.mixture.weights.get(COMPONENT_SEQ, 0.0)),
        COMPONENT_CD_VAL: 0.0,
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
    )


# --- Trial construction -----------------------------------------------------


def build_trial(config: Config) -> tuple[str, object]:
    """Build one trial's ``(job_name, raw_config)`` for ``prepare_lm_train``."""
    num_train_steps, steps_per_eval = config.mixture.schedule()
    res = resources()

    chips = res.chip_count()
    if BATCH_SIZE % chips != 0:
        raise ValueError(f"batch_size ({BATCH_SIZE}) not divisible by chip_count ({chips})")
    per_device_parallelism = BATCH_SIZE // chips
    # Eval batch = per_device_eval_parallelism * chips; pick the cap so we
    # evaluate EVAL_EXAMPLES sequences (8192 / 128 = 64 batches).
    eval_batch = per_device_parallelism * chips
    max_eval_batches = max(1, EVAL_EXAMPLES // eval_batch)

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
        max_eval_batches=max_eval_batches,
        data_seed=config.seed,
        per_device_parallelism=per_device_parallelism,
        per_device_eval_parallelism=per_device_parallelism,
    )
    params = compute_num_parameters(MODEL_CONFIG, PROTEIN_VOCAB_SIZE)
    tokens = BATCH_SIZE * SEQ_LEN * num_train_steps
    tags = [
        "protein",
        "exp-demo",
        "seq-ablation",
        "1_5b",
        "qwen3",
        config.mixture.id,
        f"seed={config.seed}",
        f"params={fmt_count(params)}",
        f"params_exact={params}",
        f"tokens={fmt_count(tokens)}",
        f"tokens_exact={tokens}",
        f"steps={num_train_steps}",
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
    # Temp checkpoint cadence = 8 min; set the trainer seed (model init /
    # training key) per replicate; disable per-parameter watch tracking (HBM).
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


def _format_weights(weights: dict[str, float]) -> str:
    return ", ".join(f"{k}={v:.3g}" for k, v in weights.items())


def print_preview(configs: tuple[Config, ...]) -> None:
    params = compute_num_parameters(MODEL_CONFIG, PROTEIN_VOCAB_SIZE)
    print(
        f"PREVIEW: exp-demo seq-ablation would run {len(configs)} target(s) "
        f"(model={fmt_count(params)} params, lr={LEARNING_RATE:.4g}, beta2={BETA2:.4g}):",
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


def worker_entrypoint(config_ids: tuple[str, ...]) -> None:
    """One Fray worker: run the selected config(s) via ``claim_and_run``."""
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

    # TOKENIZE=yes: build the seq cache (CPU) and exit. Run once before the sweep.
    if tokenize_only():
        logger.info("Building seq cache -> %s", SEQ_TRAIN_CACHE)
        executor_main(steps=[seq_tokenized])
        return

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
