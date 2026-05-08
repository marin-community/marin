# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Bolinas DNA zoonomia 1-epoch scaling: 1B + 4B Qwen3 gLMs.

See https://github.com/Open-Athena/bolinas-dna/issues/166 for full context.

Trains 1B and 4B Qwen3 gLMs on ``bolinas-dna/zoonomia-v1-v1`` (108-species
cross-mammal projection, whole-genome) for one full epoch (~57.3B tokens),
both on ``v6e-4``. Single-mixture training — no per-region weighting (cf. the
parallel CDS/upstream/downstream mix sweep in exp135).

Hyperparameters are transferred from the v0.6 DNA reference sweep via the
``CompletedAdamHHeuristic``. Both sizes share the same (B, T), so they receive
identical optimizer hparams (lr / adam_lr / epsilon / beta2 depend only on
batch and token horizon, not model size).

Architecture sizes are derived by the heuristic's depth-from-width formula:
- ``hidden=1920``  → 19 layers → ~1.12B params (exp109's ``TRANSFER_HIDDEN_SIZES[-1]``)
- ``hidden=2944``  → 29 layers → ~4.02B params (exp109's largest scaling-sweep size)

Validation: TraitGym Mendelian v2 (255 bp) via the lm_eval harness, plus
LL-gap on the four region-specific genomes-v5 validation sets (enhancer, CDS,
upstream, downstream), each tokenized functional + nonfunctional only.
None of the four matches the zoonomia training distribution, so LL-gap is a
cross-domain signal — same setup as #160.

Environment variables:
    SWEEP_HIDDEN_SIZES  CSV of hidden sizes to run (default: all in
                        ``MODEL_HIDDEN_SIZES``). Useful for ``1920`` (1B-only)
                        or ``2944`` (4B-only) launches.
    WARMUP_MODE         ``yes``/``no`` (default ``no``). ``yes`` truncates
                        training to ``WARMUP_NUM_TRAIN_STEPS`` with
                        ``WARMUP_EVALS_PER_RUN`` evals; the optimizer is still
                        built at the full token count so the LR schedule
                        endpoint matches production.
    PREVIEW_MODE        ``yes``/``no`` (default ``no``). ``yes`` prints token
                        counts, per-size param counts, and reference +
                        transferred hparams, then exits without submitting.
"""

import logging
import math
import os
from dataclasses import dataclass, replace
from datetime import timedelta

import jmp
from fray.cluster import ResourceConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import DNALmDatasetFormat
from levanter.eval_harness import LmEvalHarnessConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.optim import AdamHConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.execution.remote import remote
from marin.processing.tokenize import lm_mixture_data_config
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm

from experiments.defaults import default_tokenize
from experiments.dna.defaults import dna_effective_seq_len
from experiments.evals.task_configs import TRAITGYM_MENDELIAN_V2_255, convert_to_levanter_task_config
from experiments.scaling_law_sweeps.completed_adamh import CompletedAdamHHeuristic

# =============================================================================
# Constants
# =============================================================================

VERSION = "v0.1"
TOKENIZER = "bolinas-dna/tokenizer-char-bos"
DNA_BASE_SEQ_LEN = 255  # bp (256 - 1 for BOS)

# Single training mixture: whole-genome cross-mammal projection (#149, #158).
# 223,880,280 RC-augmented sequences across 64 shards.
TRAIN_DATASET = "bolinas-dna/zoonomia-v1-v1"
TRAIN_SAMPLE_COUNT = 223_880_280

# Four region-specific validation sets, each tokenized functional + nonfunctional
# for the LL-gap signal. Region IDs match the genomes-v5 interval definitions
# on HF; comments give the biological annotation each ID corresponds to.
VAL_DATASETS: tuple[tuple[str, str], ...] = (
    ("v30", "bolinas-dna/genomes-v5-validation-intervals-v30_255_255"),  # enhancers
    ("v5", "bolinas-dna/genomes-v5-validation-intervals-v5_255_255"),  # CDS
    ("v1", "bolinas-dna/genomes-v5-validation-intervals-v1_255_255"),  # upstream (promoter)
    ("v15", "bolinas-dna/genomes-v5-validation-intervals-v15_255_255"),  # downstream (3'-UTR)
)

# Training masks lowercase positions to 1% loss weight (consistent across
# Bolinas DNA experiments). Zoonomia datasets use ``sequence`` as the text
# field (vs the older ``seq`` default used by genomes-v5 datasets), so override
# ``text_key`` accordingly. (#160 PR #5518 op note 3.)
TRAIN_FORMAT = DNALmDatasetFormat(text_key="sequence", lowercase_weight=0.01)

# Validation tokenization specs — only the two terms of the LL gap; we
# deliberately skip a "default" matched-to-training variant (it adds eval cost
# without informing the LL-gap measurement). Matches exp160.
VAL_SPECS: tuple[tuple[str, DNALmDatasetFormat], ...] = (
    ("functional", DNALmDatasetFormat(uppercase_weight=1.0, lowercase_weight=0.0)),
    ("nonfunctional", DNALmDatasetFormat(uppercase_weight=0.0, lowercase_weight=1.0)),
)

# Architecture: heuristic-derived from hidden_size. Depth, intermediate dim,
# n_heads (= n_kv_heads) all come from ``CompletedAdamHHeuristic._build_model_config``.
# 1920 → ~1.12B params (19 layers); 2944 → ~4.02B params (29 layers).
# Both sizes appear in exp109's ``SCALING_HIDDEN_SIZES`` — picking the two
# endpoints of the AdamH-calibrated regime.
MODEL_HIDDEN_SIZES: tuple[int, ...] = (1920, 2944)

# Resources & batching. v6e-4 / europe-west4-a matches exp109's parameter
# scaling sweep, so 1B and 4B are directly comparable on wall-clock,
# throughput, and sample efficiency.
BATCH_SIZE = 8192
TPU_TYPES: tuple[str, ...] = ("v6e-4",)

# Per-device parallelism per hidden size. 4B at PDP=1024 OOMed on v6e-4 in the
# first prod-v0.2 launch (allocation 41.7 GB for the bf16[29, 1024, 256, 2944]
# activations tensor exceeded the 31.5 GB per-chip HBM); halve to 512. 1B fits
# at 1024. Gradient accumulation absorbs the difference (4B: 4 microbatches/step
# vs 1B: 2 microbatches/step) — effective batch size is BATCH_SIZE for both, so
# training dynamics are unchanged; only per-step wall-clock differs slightly.
PER_DEVICE_PARALLELISM_BY_HIDDEN: dict[int, int] = {
    1920: 1024,
    2944: 512,
}


# =============================================================================
# Reference hparams + DNA-calibrated heuristic
# =============================================================================


@dataclass(frozen=True)
class ReferenceHparams:
    """v0.6 Vizier-optimal hparams used as the heuristic's reference point."""

    lr: float
    adam_lr: float
    beta1: float
    beta2: float
    epsilon: float
    max_grad_norm: float
    z_loss_weight: float
    initializer_range: float


# Source: rank 1/183 of wandb group ``dna-bolinas-reference-sweep-v0.6``,
# eval/loss=1.228545, hidden=512, B0=16384, T0=2.5e9.
# https://wandb.ai/eric-czech/marin/runs/dna-bolinas-ref-v0.6-IR0.02-E1-loop8-trial3-abad72
REFERENCE_HPARAMS = ReferenceHparams(
    lr=0.015566099981405093,
    adam_lr=0.02989514059663958,
    beta1=0.6675603345321236,
    beta2=0.9067269880630742,
    epsilon=1e-15,
    max_grad_norm=0.9951880136348765,
    z_loss_weight=4.312883184368223e-06,
    initializer_range=0.02,
)
REFERENCE_BATCH_SIZE = 16384
REFERENCE_TOKENS = 2_500_000_000

# DNA-calibrated heuristic: re-anchored from text defaults to the v0.6 DNA
# reference regime. Constraints (max_lr, beta2 range, batch_size range) are
# explicit DNA-regime values, mirroring exp135's ``DNA_SCALING_HEURISTIC``.
DNA_SCALING_HEURISTIC = CompletedAdamHHeuristic(
    tokenizer=TOKENIZER,
    reference_batch_size=REFERENCE_BATCH_SIZE,
    reference_tokens=REFERENCE_TOKENS,
    lr_base=REFERENCE_HPARAMS.lr,
    adam_lr_base=REFERENCE_HPARAMS.adam_lr,
    epsilon_base=REFERENCE_HPARAMS.epsilon,
    beta1=REFERENCE_HPARAMS.beta1,
    beta2_base=REFERENCE_HPARAMS.beta2,
    max_grad_norm=REFERENCE_HPARAMS.max_grad_norm,
    z_loss_weight=REFERENCE_HPARAMS.z_loss_weight,
    max_learning_rate=0.03,
    min_beta2=0.5,
    max_beta2=0.9999,
    min_batch_size=8,
    max_batch_size=8192,
)

assert (
    BATCH_SIZE <= DNA_SCALING_HEURISTIC.max_batch_size
), f"BATCH_SIZE={BATCH_SIZE} exceeds heuristic max_batch_size={DNA_SCALING_HEURISTIC.max_batch_size}"


# Eval cadence and checkpoint policy. CHECKPOINTS_PER_RUN matches EVALS_PER_RUN
# so every eval point has a saved checkpoint to re-run evals against offline.
EVALS_PER_RUN = 10
CHECKPOINTS_PER_RUN = 10
CHECKPOINT_TIME_INTERVAL = timedelta(hours=1)

# Warmup mode: smoke-test the full pipeline. Only ``num_train_steps`` and eval
# count are reduced; the optimizer is built at the full target token count so
# the LR schedule endpoint matches the full run.
WARMUP_NUM_TRAIN_STEPS = 100
WARMUP_EVALS_PER_RUN = 3

WANDB_PROJECT = "marin"
WANDB_GROUP = f"dna-bolinas-exp166-{VERSION}"

_EXPECTED_VOCAB_SIZE_WARNING = f"Tokenizer {TOKENIZER!r} not found in _KNOWN_VOCAB_SIZES"
logging.getLogger("marin.processing.tokenize.data_configs").addFilter(
    lambda record: _EXPECTED_VOCAB_SIZE_WARNING not in record.getMessage()
)


# =============================================================================
# Environment overrides
# =============================================================================


def _env_yes_no(name: str, default: str = "no") -> bool:
    value = os.getenv(name, default).lower()
    if value not in ("yes", "no"):
        raise ValueError(f"{name} must be 'yes' or 'no', got {value!r}")
    return value == "yes"


def _warmup_mode() -> bool:
    return _env_yes_no("WARMUP_MODE")


def _preview_mode() -> bool:
    return _env_yes_no("PREVIEW_MODE")


def _selected_hidden_sizes() -> tuple[int, ...]:
    """Return the subset of MODEL_HIDDEN_SIZES named in SWEEP_HIDDEN_SIZES (or all)."""
    raw = os.getenv("SWEEP_HIDDEN_SIZES")
    if not raw:
        return MODEL_HIDDEN_SIZES
    requested = tuple(int(s.strip()) for s in raw.split(","))
    invalid = [h for h in requested if h not in MODEL_HIDDEN_SIZES]
    if invalid:
        raise ValueError(f"Invalid SWEEP_HIDDEN_SIZES {invalid}; available: {list(MODEL_HIDDEN_SIZES)}")
    return requested


# =============================================================================
# Builders
# =============================================================================


# Marin's only CPU worker pool is n2-highmem-2 (2 vCPU / 16 GiB) and is
# typically near-saturated by other users' small coordinators, so override
# default_tokenize's cpu=4 ask down to a minimal orchestrator footprint that
# can pack alongside an existing job on the same VM. The tokenize step is
# pure orchestration — heavy work runs on zephyr workers spawned at their
# own resource spec — so cpu=1 is plenty. RAM is bumped to 12g (just under
# the 16g VM cap, leaving headroom for the parent's 1g) because the
# coordinator's shutdown/aggregation phase OOMs at 4g on the larger
# whole-genome zoonomia_v1 dataset. Inherited verbatim from exp160.
_TOKENIZE_RESOURCES = ResourceConfig.with_cpu(cpu=1, ram="12g", disk="10g")


def _model_seq_len() -> int:
    return dna_effective_seq_len(DNA_BASE_SEQ_LEN, TOKENIZER)


def _model_config(hidden_size: int):
    base = DNA_SCALING_HEURISTIC._build_model_config(hidden_size, _model_seq_len())
    return replace(base, initializer_range=REFERENCE_HPARAMS.initializer_range)


def _num_params(hidden_size: int) -> int:
    return _model_config(hidden_size).total_trainable_params(DNA_SCALING_HEURISTIC.vocab_size)


def _format_params(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{round(n / 1e9)}B"
    return f"{round(n / 1e6)}M"


def _tokenize(name: str, dataset: str, dataset_format: DNALmDatasetFormat) -> ExecutorStep:
    return default_tokenize(
        name=name,
        dataset=dataset,
        tokenizer=TOKENIZER,
        format=dataset_format,
        resources=_TOKENIZE_RESOURCES,
    )


def _full_num_train_steps() -> int:
    """Steps for one effective epoch over the zoonomia training set."""
    return math.ceil(TRAIN_SAMPLE_COUNT / BATCH_SIZE)


def _num_train_steps() -> int:
    if _warmup_mode():
        return WARMUP_NUM_TRAIN_STEPS
    return _full_num_train_steps()


def _full_target_tokens() -> int:
    """Tokens consumed in a full (non-warmup) run; feeds the heuristic's T."""
    return _full_num_train_steps() * BATCH_SIZE * _model_seq_len()


def _steps_per_eval(num_train_steps: int) -> int:
    evals = WARMUP_EVALS_PER_RUN if _warmup_mode() else EVALS_PER_RUN
    return max(1, num_train_steps // evals)


def _build_data_mixture():
    """One training component + cross-product of validation regions x specs.

    Tokenize names match exp160's convention so val tokenizations dedupe with
    that experiment's executor cache: ``bolinas-zoonomia-v1-zoonomia_v1-char-bos``
    for train, ``bolinas-v5-val_{region}_{suffix}-char-bos`` for val.
    """
    components: dict[str, ExecutorStep] = {
        "zoonomia_v1": _tokenize("bolinas-zoonomia-v1-zoonomia_v1-char-bos", TRAIN_DATASET, TRAIN_FORMAT),
    }
    for region, val_dataset in VAL_DATASETS:
        for suffix, fmt in VAL_SPECS:
            key = f"val_{region}_{suffix}"
            components[key] = _tokenize(f"bolinas-v5-{key}-char-bos", val_dataset, fmt)
    return lm_mixture_data_config(
        components=components,
        weights={"zoonomia_v1": 1.0},
    )


def _build_optimizer() -> AdamHConfig:
    """AdamH config heuristic-scaled to the full one-epoch token count.

    Depends only on (B, T), not on hidden size, so 1B and 4B share this config.
    """
    return DNA_SCALING_HEURISTIC.build_optimizer_config(BATCH_SIZE, _full_target_tokens())


def _eval_harness_config() -> LmEvalHarnessConfig:
    return LmEvalHarnessConfig(
        task_spec=convert_to_levanter_task_config([TRAITGYM_MENDELIAN_V2_255]),
        include_path="experiments/evals/custom_tasks",
        max_packed_segments=1,
    )


def _checkpointer(num_train_steps: int) -> CheckpointerConfig:
    return CheckpointerConfig(
        save_interval=CHECKPOINT_TIME_INTERVAL,
        keep=[dict(every=max(1, num_train_steps // CHECKPOINTS_PER_RUN))],
    )


def _build_train_step(hidden_size: int) -> ExecutorStep:
    num_train_steps = _num_train_steps()
    steps_per_eval = _steps_per_eval(num_train_steps)
    optimizer = _build_optimizer()
    target_tokens = _full_target_tokens()
    num_params = _num_params(hidden_size)
    params_label = _format_params(num_params)
    warmup_suffix = "-warmup" if _warmup_mode() else ""
    run_name = f"dna-bolinas-exp166-{VERSION}-p{params_label}{warmup_suffix}"
    tags = [
        "dna",
        "bolinas",
        "exp166",
        "zoonomia",
        "scaling",
        VERSION,
        f"hidden={hidden_size}",
        f"params={num_params}",
        f"bs={BATCH_SIZE}",
        f"tokens={target_tokens}",
        f"lr={optimizer.learning_rate}",
        f"adam_lr={optimizer.adam_lr}",
        f"beta1={optimizer.beta1}",
        f"beta2={optimizer.beta2}",
        f"eps={optimizer.epsilon}",
        f"mgn={optimizer.max_grad_norm}",
        f"zloss={REFERENCE_HPARAMS.z_loss_weight}",
    ]
    if _warmup_mode():
        tags.append("warmup")

    inner = TrainLmConfig(
        data=_build_data_mixture(),
        model=_model_config(hidden_size),
        train_seq_len=_model_seq_len(),
        z_loss_weight=REFERENCE_HPARAMS.z_loss_weight,
        optimizer=optimizer,
        eval_harness=_eval_harness_config(),
        eval_harness_steps=steps_per_eval,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                project=WANDB_PROJECT,
                tags=tags,
                group=WANDB_GROUP,
                name=run_name,
                replicate_path=this_output_path(),
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=BATCH_SIZE,
            num_train_steps=num_train_steps,
            steps_per_eval=steps_per_eval,
            checkpointer=_checkpointer(num_train_steps),
            mesh=MeshConfig(axes={"replica": 1, "data": -1, "model": 1}),
            allow_nondivisible_batch_size=True,
            per_device_parallelism=PER_DEVICE_PARALLELISM_BY_HIDDEN[hidden_size],
        ),
    )
    pod_config = TrainLmOnPodConfig(
        train_config=inner,
        resources=ResourceConfig.with_tpu(TPU_TYPES, ram="300g"),
        output_path=this_output_path(),
    )
    return ExecutorStep(
        name=os.path.join("checkpoints", run_name),
        fn=remote(run_levanter_train_lm, resources=ResourceConfig.with_cpu()),
        config=pod_config,
    )


# =============================================================================
# Preview
# =============================================================================


def _print_preview(selected: tuple[int, ...]) -> None:
    """Print token counts, per-size param counts, and reference + transferred hparams."""
    full_steps = _full_num_train_steps()
    target_tokens = _full_target_tokens()
    optimizer = _build_optimizer()
    print("=" * 78)
    print(f"DNA Bolinas exp166 zoonomia 1-epoch scaling {VERSION} — preview")
    print(f"  train_dataset={TRAIN_DATASET}  samples={TRAIN_SAMPLE_COUNT:,}")
    pdp_str = ", ".join(f"h={h}:pdp={p}" for h, p in PER_DEVICE_PARALLELISM_BY_HIDDEN.items())
    print(f"  batch_size={BATCH_SIZE}  seq_len={_model_seq_len()}  per_device_parallelism={{ {pdp_str} }}")
    print(f"  full_num_train_steps={full_steps}  target_tokens={target_tokens:.3e}")
    if _warmup_mode():
        print(f"  WARMUP_MODE=yes -> num_train_steps clamped to {WARMUP_NUM_TRAIN_STEPS}")
    print()
    print("Models:")
    for hidden_size in selected:
        n = _num_params(hidden_size)
        cfg = _model_config(hidden_size)
        print(
            f"  hidden={hidden_size:4d}  layers={cfg.num_layers:2d}  "
            f"intermediate={cfg.intermediate_dim}  n_heads={cfg.num_heads}  "
            f"params={n:,} (~{_format_params(n)})  tok/param={target_tokens / n:.1f}"
        )
    print()
    print(f"Reference hparams (v0.6 Vizier optimum at B0={REFERENCE_BATCH_SIZE}, T0={REFERENCE_TOKENS:.2e}):")
    for field, value in (
        ("lr", REFERENCE_HPARAMS.lr),
        ("adam_lr", REFERENCE_HPARAMS.adam_lr),
        ("beta1", REFERENCE_HPARAMS.beta1),
        ("beta2", REFERENCE_HPARAMS.beta2),
        ("epsilon", REFERENCE_HPARAMS.epsilon),
        ("max_grad_norm", REFERENCE_HPARAMS.max_grad_norm),
        ("z_loss_weight", REFERENCE_HPARAMS.z_loss_weight),
        ("initializer_range", REFERENCE_HPARAMS.initializer_range),
    ):
        print(f"  {field:20s} {value:.6g}")
    print(
        f"  heuristic constraints: max_lr={DNA_SCALING_HEURISTIC.max_learning_rate} "
        f"beta2_range=[{DNA_SCALING_HEURISTIC.min_beta2}, {DNA_SCALING_HEURISTIC.max_beta2}] "
        f"batch_size_range=[{DNA_SCALING_HEURISTIC.min_batch_size}, {DNA_SCALING_HEURISTIC.max_batch_size}]"
    )
    print()
    print("Transferred hparams (depend only on B and T; identical for 1B and 4B):")
    for field, value in (
        ("lr", optimizer.learning_rate),
        ("adam_lr", optimizer.adam_lr),
        ("beta1", optimizer.beta1),
        ("beta2", optimizer.beta2),
        ("epsilon", optimizer.epsilon),
        ("max_grad_norm", optimizer.max_grad_norm),
        ("z_loss_weight", REFERENCE_HPARAMS.z_loss_weight),
        ("warmup", optimizer.warmup),
        ("decay", optimizer.decay),
        ("min_lr_ratio", optimizer.min_lr_ratio),
    ):
        print(f"  {field:20s} {value:.6g}")
    print(f"  {'lr_schedule':20s} {optimizer.lr_schedule}")
    print(f"  {'nesterov':20s} {optimizer.nesterov}")
    print()
    print(f"Cadence: evals_per_run={EVALS_PER_RUN} checkpoints_per_run={CHECKPOINTS_PER_RUN}")
    print(
        f"  steps_per_eval={_steps_per_eval(_num_train_steps())}  "
        f"checkpoint_time_interval={CHECKPOINT_TIME_INTERVAL}"
    )
    print("=" * 78)


# =============================================================================
# Entry point
# =============================================================================


def main():
    selected = _selected_hidden_sizes()
    if _preview_mode():
        _print_preview(selected)
        return
    steps = [_build_train_step(hidden_size) for hidden_size in selected]
    executor_main(
        steps=steps,
        description=f"DNA Bolinas exp166 zoonomia 1-epoch scaling {VERSION}",
    )


if __name__ == "__main__":
    main()
