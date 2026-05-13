# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Bolinas DNA mix sweep at the 1B scale.

See https://github.com/Open-Athena/bolinas-dna/issues/135 for full context.

Compares CDS / upstream / downstream training mixture weights at hidden=1920
(~1.12B params), with optimizer hparams transferred from the v0.6 reference
sweep via ``CompletedAdamHHeuristic``. Hparams are recomputed per-mix because
each mix trains on a different total token count: region-only mixes
(``cds_only`` / ``upstream_only`` / ``downstream_only``) train on the full
per-region dataset (``MAX_TRAIN_EXAMPLES_PER_REGION``), while the uniform mix
caps each active component at ``UNIFORM_MAX_EXAMPLES_PER_COMPONENT``
(upstream's full size, ~68.3M).

For the uniform mix, downstream (~20.5M) is smaller than the per-component
cap, so its slice is clamped to the dataset size and the mixture cycles via
Levanter's ``RESTART_STRATEGY`` (each downstream sequence repeats ~3.33x).
Cycling reuses the same Feistel-permuted underlying examples but with
different per-block within-block ordering, so repeated examples are
reinterleaved across cycles. The optimizer's target T counts all drawn tokens
including repeats, so the LR schedule is sized for the actual number of
gradient updates. Tokenization names use the ``-5149`` suffix to create fresh
cache keys for the post-issue-5149 fix
(https://github.com/marin-community/marin/issues/5149).

Validation adds 6 metrics on top of the existing 3: each region tokenized with
functional (uppercase-only) and nonfunctional (lowercase-only) masks per
https://github.com/Open-Athena/bolinas-dna/issues/10.

Environment variables:
    SWEEP_MIX_NAMES   CSV of mix names to run (default: all in ``MIX_CONFIGS``).
    WARMUP_MODE       ``yes``/``no`` (default ``no``). ``yes`` truncates training
                      to ``WARMUP_NUM_TRAIN_STEPS`` with ``WARMUP_EVALS_PER_RUN``
                      evals; the optimizer is still built at the full token count
                      so the LR schedule endpoint matches production.
    PREVIEW_MODE      ``yes``/``no`` (default ``no``). ``yes`` prints token
                      counts, param count, and reference + per-mix transferred
                      hparams, then exits without submitting.
"""

import logging
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

VERSION = "v0.9"
TOKENIZER = "bolinas-dna/tokenizer-char-bos"
DNA_BASE_SEQ_LEN = 255  # bp (256 - 1 for BOS)

# Uniform mix: cap each active component at upstream's full dataset size
# (~68.3M). cds (~244M) is sliced to ~68.3M unique examples; upstream gets one
# full epoch; downstream (~20.5M) cycles ~3.33x via Levanter's
# ``RESTART_STRATEGY``. The optimizer's target T tracks the total drawn-token
# count (~52.4B) so the LR schedule matches the actual gradient trajectory.
# See https://github.com/Open-Athena/bolinas-dna/issues/109.
UNIFORM_MAX_EXAMPLES_PER_COMPONENT = 68_286_166

# Region-only mixes train on the full per-region dataset; values are the
# training-example counts of each per-region dataset, so the cap is exactly
# one epoch over that region.
MAX_TRAIN_EXAMPLES_PER_REGION: dict[str, int] = {
    "cds": 242_334_716,
    "upstream": 68_286_166,
    "downstream": 20_501_856,
}

TRAIN_DATASETS = {
    "cds": "bolinas-dna/genomes-v5-genome_set-animals-intervals-v5_255_128",
    "upstream": "bolinas-dna/genomes-v5-genome_set-animals-intervals-v1_255_128",
    "downstream": "bolinas-dna/genomes-v5-genome_set-animals-intervals-v15_255_128",
}
VALIDATION_DATASETS = {
    "val_cds": "bolinas-dna/genomes-v5-validation-intervals-v5_255_255",
    "val_upstream": "bolinas-dna/genomes-v5-validation-intervals-v1_255_255",
    "val_downstream": "bolinas-dna/genomes-v5-validation-intervals-v15_255_255",
}

# Tokenization name template — the ``-5149`` suffix forces a fresh cache key for
# the post-fix tokenization (https://github.com/marin-community/marin/issues/5149).
TOKENIZE_NAME = "bolinas-v5-{key}-char-bos-5149"

# Training mask: 1% loss weight on lowercase positions.
TRAIN_FORMAT = DNALmDatasetFormat(lowercase_weight=0.01)

# Architecture: hidden=1920 -> ~1.12B params. Matches exp109's
# ``TRANSFER_HIDDEN_SIZES[-1]`` so this study anchors directly on the
# hparam-validated regime. The heuristic derives intermediate/heads/layers.
MODEL_HIDDEN_SIZE = 1920

# Resources & batching.
BATCH_SIZE = 8192
PER_DEVICE_PARALLELISM = 1024
TPU_TYPES: tuple[str, ...] = ("v5p-8",)


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
# explicit DNA-regime values, mirroring exp109's ``DNA_SCALING_HEURISTIC``.
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


# Eval cadence and checkpoint policy.
EVALS_PER_RUN = 10
CHECKPOINTS_PER_RUN = 3
CHECKPOINT_TIME_INTERVAL = timedelta(minutes=15)

# Warmup mode: smoke-test the full pipeline. Only ``num_train_steps`` and eval
# count are reduced; the optimizer is built at the full target token count so
# the LR schedule endpoint matches the full run.
WARMUP_NUM_TRAIN_STEPS = 100
WARMUP_EVALS_PER_RUN = 3

# Number of times to repeat the full mixture. With EPOCHS > 1, the trainer
# drives EPOCHS * single_epoch steps. Levanter's MixtureDataset has three
# stop strategies: FIRST_STOP (end when any component is exhausted), ALL_STOP
# (end when all are), and the default RESTART (cycle exhausted components
# indefinitely). We rely on RESTART so a component whose slice runs out is
# sampled cyclically while training continues. Optimizer hparams are anchored
# at single-epoch T -- LR/beta2/decay don't change with EPOCHS; only the
# LR-schedule fractions stretch over the run.
EPOCHS = 1

WANDB_PROJECT = "marin"

_EXPECTED_VOCAB_SIZE_WARNING = f"Tokenizer {TOKENIZER!r} not found in _KNOWN_VOCAB_SIZES"
logging.getLogger("marin.processing.tokenize.data_configs").addFilter(
    lambda record: _EXPECTED_VOCAB_SIZE_WARNING not in record.getMessage()
)


# =============================================================================
# Mix configurations
# =============================================================================


@dataclass(frozen=True)
class MixConfig:
    """One run in the sweep: a name and per-region training weights.

    Regions absent from ``weights`` (or with weight 0) are omitted from the
    mixture entirely — neither tokenized nor sampled. Validation datasets are
    always tokenized and evaluated regardless of training weights.

    ``max_train_examples``, when set, overrides the per-region cap logic in
    ``_train_example_caps`` with a total-budget split: each active region's
    cap is ``max_train_examples * weight / sum(active_weights)``. Used by
    continuation runs that train for a fixed (small) token budget.

    ``initialize_from_hf``, when set, is forwarded to
    ``TrainLmConfig.initialize_from_hf`` so the run warm-starts model weights
    from a HuggingFace-format checkpoint (config.json + model.safetensors).
    Only the model is loaded; optimizer state, step, and RNG keys stay fresh,
    so the LR schedule starts at the beginning of warmup. The Levanter-format
    counterpart (``initialize_from_checkpoint_path``) is not used here because
    it deserializes the full state including the ``inject_hyperparams`` count
    in ``opt_state``, which would leave the new run's schedule clamped at 0.

    ``data_seed``, when set, is forwarded to ``TrainLmConfig.data_seed`` and
    overrides only the data shuffle / mixture sampling key. Model init and
    training noise are unaffected. Useful for continuation runs that share an
    HF init but want a distinct data ordering.
    """

    name: str
    weights: dict[str, float]
    max_train_examples: int | None = None
    initialize_from_hf: str | None = None
    data_seed: int | None = None

    def __post_init__(self):
        unknown = set(self.weights) - set(TRAIN_DATASETS)
        if unknown:
            raise ValueError(f"{self.name}: unknown regions {unknown}; expected subset of {set(TRAIN_DATASETS)}")
        if not any(w > 0 for w in self.weights.values()):
            raise ValueError(f"{self.name}: at least one weight must be > 0")
        if self.max_train_examples is not None and self.max_train_examples <= 0:
            raise ValueError(f"{self.name}: max_train_examples must be positive, got {self.max_train_examples}")

    @property
    def active_regions(self) -> tuple[str, ...]:
        return tuple(r for r, w in self.weights.items() if w > 0)


MIX_CONFIGS: tuple[MixConfig, ...] = (
    MixConfig(name="uniform", weights={"cds": 1 / 3, "upstream": 1 / 3, "downstream": 1 / 3}),
    MixConfig(name="cds_only", weights={"cds": 1.0}),
    MixConfig(name="upstream_only", weights={"upstream": 1.0}),
    MixConfig(name="downstream_only", weights={"downstream": 1.0}),
    # Continuation run: warm-start model weights from the final HF checkpoint
    # of the i=2 upstream_only run and train a 50/50 cds+upstream mixture for
    # ~one downstream-dataset worth of examples (~20.5M → ~5.25B tokens). We
    # load via the HF path (model-only) so the LR schedule starts fresh; see
    # the ``MixConfig`` docstring for why ``initialize_from_checkpoint_path``
    # is unsuitable here.
    MixConfig(
        name="cont_upstream_to_cds_2",
        weights={"cds": 0.5, "upstream": 0.5},
        max_train_examples=MAX_TRAIN_EXAMPLES_PER_REGION["downstream"],
        initialize_from_hf=(
            "gs://marin-us-east5/checkpoints/dna-bolinas-mix-v0.9-p1B-i2-upstream_only-68544e/hf/step-8334/"
        ),
    ),
    # Replay of cont_upstream_to_cds_2 with a distinct ``data_seed`` so the
    # data shuffle / mixture sampling differs while the model init and recipe
    # are identical.
    MixConfig(
        name="cont_upstream_to_cds_2.1",
        weights={"cds": 0.5, "upstream": 0.5},
        max_train_examples=MAX_TRAIN_EXAMPLES_PER_REGION["downstream"],
        initialize_from_hf=(
            "gs://marin-us-east5/checkpoints/dna-bolinas-mix-v0.9-p1B-i2-upstream_only-68544e/hf/step-8334/"
        ),
        data_seed=2,
    ),
    # Same warm-start as cont_upstream_to_cds_2 (downstream-sized token budget)
    # but with a 50/30/20 upstream/downstream/cds mixture instead of 50/50.
    MixConfig(
        name="cont_upstream_to_cds_3",
        weights={"upstream": 0.5, "downstream": 0.3, "cds": 0.2},
        max_train_examples=MAX_TRAIN_EXAMPLES_PER_REGION["downstream"],
        initialize_from_hf=(
            "gs://marin-us-east5/checkpoints/dna-bolinas-mix-v0.9-p1B-i2-upstream_only-68544e/hf/step-8334/"
        ),
    ),
    # Same warm-start and token budget as the cont_upstream_to_cds_* mixes but
    # with a 50/50 upstream/downstream mixture (no cds). Uses a distinct
    # ``data_seed`` so the shuffle / mixture sampling differs from the other
    # continuation runs.
    MixConfig(
        name="cont_upstream_to_downstream_1",
        weights={"upstream": 0.5, "downstream": 0.5},
        max_train_examples=MAX_TRAIN_EXAMPLES_PER_REGION["downstream"],
        initialize_from_hf=(
            "gs://marin-us-east5/checkpoints/dna-bolinas-mix-v0.9-p1B-i2-upstream_only-68544e/hf/step-8334/"
        ),
        data_seed=1,
    ),
)


# =============================================================================
# Validation specs
# =============================================================================


@dataclass(frozen=True)
class ValSpec:
    """One validation tokenization variant applied to every region.

    ``suffix`` is appended to each region key (with an underscore) when non-empty;
    empty suffix produces the default validation set whose mask matches training.
    Functional vs nonfunctional masks isolate loss to upper- vs lowercase
    (conserved vs nonconserved) positions — see
    https://github.com/Open-Athena/bolinas-dna/issues/10.
    """

    suffix: str
    format: DNALmDatasetFormat


VAL_SPECS: tuple[ValSpec, ...] = (
    ValSpec(suffix="", format=DNALmDatasetFormat(lowercase_weight=0.01)),
    ValSpec(suffix="functional", format=DNALmDatasetFormat(lowercase_weight=0.0, uppercase_weight=1.0)),
    ValSpec(suffix="nonfunctional", format=DNALmDatasetFormat(lowercase_weight=1.0, uppercase_weight=0.0)),
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


def _selected_mix_configs() -> tuple[MixConfig, ...]:
    """Return the subset of MIX_CONFIGS named in SWEEP_MIX_NAMES (or all if unset)."""
    raw = os.getenv("SWEEP_MIX_NAMES")
    if not raw:
        return MIX_CONFIGS
    requested = tuple(s.strip() for s in raw.split(","))
    available = {c.name: c for c in MIX_CONFIGS}
    invalid = [n for n in requested if n not in available]
    if invalid:
        raise ValueError(f"Invalid SWEEP_MIX_NAMES {invalid}; available: {sorted(available)}")
    return tuple(available[n] for n in requested)


# =============================================================================
# Builders
# =============================================================================


def _model_seq_len() -> int:
    return dna_effective_seq_len(DNA_BASE_SEQ_LEN, TOKENIZER)


def _model_config():
    base = DNA_SCALING_HEURISTIC._build_model_config(MODEL_HIDDEN_SIZE, _model_seq_len())
    return replace(base, initializer_range=REFERENCE_HPARAMS.initializer_range)


def _num_params() -> int:
    return _model_config().total_trainable_params(DNA_SCALING_HEURISTIC.vocab_size)


def _format_params(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{round(n / 1e9)}B"
    return f"{round(n / 1e6)}M"


def _tokenize(key: str, dataset: str, dataset_format: DNALmDatasetFormat) -> ExecutorStep:
    return default_tokenize(
        name=TOKENIZE_NAME.format(key=key),
        dataset=dataset,
        tokenizer=TOKENIZER,
        format=dataset_format,
    )


def _train_example_caps(mix: MixConfig) -> dict[str, int]:
    """Per-region training-example cap for this mix.

    If ``mix.max_train_examples`` is set, the total budget is split across
    active regions in proportion to their weights (used by fixed-budget
    continuation runs). Otherwise: region-only mixes (single active region)
    cap at the region's full dataset size (``MAX_TRAIN_EXAMPLES_PER_REGION``)
    and multi-region mixes (uniform) cap each active component at
    ``UNIFORM_MAX_EXAMPLES_PER_COMPONENT``.
    """
    if mix.max_train_examples is not None:
        active_weight_sum = sum(mix.weights[r] for r in mix.active_regions)
        return {
            region: int(mix.max_train_examples * mix.weights[region] / active_weight_sum)
            for region in mix.active_regions
        }
    if len(mix.active_regions) == 1:
        region = mix.active_regions[0]
        return {region: MAX_TRAIN_EXAMPLES_PER_REGION[region]}
    return {region: UNIFORM_MAX_EXAMPLES_PER_COMPONENT for region in mix.active_regions}


def _train_batch_caps(mix: MixConfig) -> dict[str, int]:
    return {region: cap // BATCH_SIZE for region, cap in _train_example_caps(mix).items()}


def _full_num_train_steps(mix: MixConfig) -> int:
    """Steps for one effective epoch over active regions, ignoring warmup truncation."""
    return sum(_train_batch_caps(mix).values())


def _num_train_steps(mix: MixConfig) -> int:
    if _warmup_mode():
        return WARMUP_NUM_TRAIN_STEPS
    return EPOCHS * _full_num_train_steps(mix)


def _full_target_tokens(mix: MixConfig, *, per_epoch: bool) -> int:
    """Tokens consumed during training.

    ``per_epoch=True`` returns one epoch's worth (used to anchor the optimizer
    heuristic). ``per_epoch=False`` returns the full run total
    (``EPOCHS * one_epoch``), used for logging/tags.
    """
    one_epoch = _full_num_train_steps(mix) * BATCH_SIZE * _model_seq_len()
    return one_epoch if per_epoch else EPOCHS * one_epoch


def _steps_per_eval(num_train_steps: int) -> int:
    evals = WARMUP_EVALS_PER_RUN if _warmup_mode() else EVALS_PER_RUN
    return max(1, num_train_steps // evals)


def _build_data_mixture(mix: MixConfig):
    """Tokenize active train regions + cross-product of validation regions x specs."""
    components = {region: _tokenize(region, TRAIN_DATASETS[region], TRAIN_FORMAT) for region in mix.active_regions}
    for region_key, dataset in VALIDATION_DATASETS.items():
        for spec in VAL_SPECS:
            key = f"{region_key}_{spec.suffix}" if spec.suffix else region_key
            components[key] = _tokenize(key, dataset, spec.format)
    train_weights = {region: mix.weights[region] for region in mix.active_regions}
    return lm_mixture_data_config(
        components=components,
        weights=train_weights,
        max_train_batches=_train_batch_caps(mix),
    )


def _build_optimizer(mix: MixConfig) -> AdamHConfig:
    """Per-mix AdamH config — anchored at single-epoch T (epoch-invariant)."""
    return DNA_SCALING_HEURISTIC.build_optimizer_config(BATCH_SIZE, _full_target_tokens(mix, per_epoch=True))


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


def _build_train_step(index: int, mix: MixConfig) -> ExecutorStep:
    num_train_steps = _num_train_steps(mix)
    steps_per_eval = _steps_per_eval(num_train_steps)
    optimizer = _build_optimizer(mix)
    target_tokens = _full_target_tokens(mix, per_epoch=False)
    num_params = _num_params()
    params_label = _format_params(num_params)
    warmup_suffix = "-warmup" if _warmup_mode() else ""
    epochs_suffix = f"-e{EPOCHS}" if EPOCHS > 1 else ""
    run_name = f"dna-bolinas-mix-{VERSION}-p{params_label}{warmup_suffix}{epochs_suffix}-i{index}-{mix.name}"
    tags = [
        "sweep",
        "dna",
        "bolinas",
        "mix",
        VERSION,
        f"mix={mix.name}",
        f"i={index}",
        f"hidden={MODEL_HIDDEN_SIZE}",
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
        data=_build_data_mixture(mix),
        model=_model_config(),
        train_seq_len=_model_seq_len(),
        z_loss_weight=REFERENCE_HPARAMS.z_loss_weight,
        optimizer=optimizer,
        initialize_from_hf=mix.initialize_from_hf or False,
        data_seed=mix.data_seed,
        eval_harness=_eval_harness_config(),
        eval_harness_steps=steps_per_eval,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                project=WANDB_PROJECT,
                tags=tags,
                group=f"dna-bolinas-mix-sweep-{VERSION}",
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
            per_device_parallelism=PER_DEVICE_PARALLELISM,
        ),
    )
    pod_config = TrainLmOnPodConfig(
        train_config=inner,
        resources=ResourceConfig.with_tpu(TPU_TYPES, ram="128g"),
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


def _print_preview(selected: tuple[MixConfig, ...]) -> None:
    """Print token counts, param count, and reference + per-mix transferred hparams."""
    num_params = _num_params()
    print("=" * 78)
    print(f"DNA Bolinas mix sweep {VERSION} — preview")
    print(f"  hidden={MODEL_HIDDEN_SIZE}  params={num_params:,} (~{_format_params(num_params)})")
    print(f"  batch_size={BATCH_SIZE}  seq_len={_model_seq_len()}  per_device_parallelism={PER_DEVICE_PARALLELISM}")
    if _warmup_mode():
        print(f"  WARMUP_MODE=yes -> num_train_steps clamped to {WARMUP_NUM_TRAIN_STEPS}")
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
    for mix in selected:
        full_steps = _full_num_train_steps(mix)
        opt_tokens = _full_target_tokens(mix, per_epoch=True)
        total_tokens = _full_target_tokens(mix, per_epoch=False)
        opt = _build_optimizer(mix)
        print(
            f"Mix {mix.name}: active=[{','.join(mix.active_regions)}]  "
            f"epoch_steps={full_steps}  total_steps={EPOCHS * full_steps} (EPOCHS={EPOCHS})  "
            f"opt_T={opt_tokens:.3e}  total_T={total_tokens:.3e} "
            f"(~{opt_tokens / num_params:.1f} tok/param/epoch)"
        )
        if mix.initialize_from_hf is not None:
            print(f"    init_from_hf        {mix.initialize_from_hf}")
        for field, value in (
            ("lr", opt.learning_rate),
            ("adam_lr", opt.adam_lr),
            ("beta1", opt.beta1),
            ("beta2", opt.beta2),
            ("epsilon", opt.epsilon),
            ("max_grad_norm", opt.max_grad_norm),
            ("z_loss_weight", REFERENCE_HPARAMS.z_loss_weight),
            ("warmup", opt.warmup),
            ("decay", opt.decay),
            ("min_lr_ratio", opt.min_lr_ratio),
        ):
            print(f"    {field:20s} {value:.6g}")
        print(f"    {'lr_schedule':20s} {opt.lr_schedule}")
        print(f"    {'nesterov':20s} {opt.nesterov}")
        print()
    print("=" * 78)


# =============================================================================
# Entry point
# =============================================================================


def main():
    selected = _selected_mix_configs()
    if _preview_mode():
        _print_preview(selected)
        return
    # Preserve original MIX_CONFIGS indices so run names stay stable across SWEEP_MIX_NAMES filters.
    index_by_name = {c.name: i for i, c in enumerate(MIX_CONFIGS)}
    steps = [_build_train_step(index_by_name[mix.name], mix) for mix in selected]
    executor_main(steps=steps, description=f"DNA Bolinas mix sweep {VERSION}")


if __name__ == "__main__":
    main()
