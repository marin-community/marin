# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Bolinas DNA transferred scaling sweep (marin#4251).

DNA parameter scaling sweep using Completed AdamH heuristics, adapted from the
text reference sweep (marin#2432). See bolinas-dna#109 for full context.

Subcommands:
    run_smoke_test                 ~20-step infrastructure validation
    run_reference_tuning_sweep     Vizier Bayesian optimization over AdamH hparams
    run_transfer_validation_sweep  Sweep key hypers in isolation at largest scale
    run_parameter_scaling_sweep    IsoFLOP parameter scaling via CompletedAdamH
"""

import logging
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import timedelta
from functools import lru_cache

import jmp
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import DNALmDatasetFormat
from levanter.eval_harness import LmEvalHarnessConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.optim import AdamHConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig

from experiments.defaults import default_tokenize, default_train
from experiments.dna.defaults import dna_effective_seq_len
from experiments.evals.task_configs import TRAITGYM_MENDELIAN_V2_255, convert_to_levanter_task_config
from experiments.references.reference_hyperparameter_sweep import (
    SUGGESTIONS_FILENAME,
    VIZIER_DB_FILENAME,
    VizierOptimalConfig,
    VizierSuggestConfig,
    VizierUpdateConfig,
    _extract_adamh_hparams,
    _load_suggestions,
    run_vizier_optimal,
    run_vizier_suggest,
    run_vizier_update,
)
from experiments.scaling_law_sweeps.completed_adamh import CompletedAdamHHeuristic
from experiments.simple_train_config import SimpleTrainConfig
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.execution.remote import remote
from marin.processing.tokenize import lm_mixture_data_config
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm

# =============================================================================
# Module-level constants
# =============================================================================

TOKENIZER = "bolinas-dna/tokenizer-char-bos"
DNA_BASE_SEQ_LEN = 255  # bp (256 - 1 for BOS)

TRAIN_DATASETS = {
    "cds": "bolinas-dna/genomes-v5-genome_set-animals-intervals-v5_255_128",
    "upstream": "bolinas-dna/genomes-v5-genome_set-animals-intervals-v1_255_128",
    "downstream": "bolinas-dna/genomes-v5-genome_set-animals-intervals-v15_255_128",
}
TRAIN_WEIGHTS = {"cds": 0.7319, "upstream": 0.2062, "downstream": 0.0619}

VALIDATION_DATASETS = {
    "val_cds": "bolinas-dna/genomes-v5-validation-intervals-v5_255_255",
    "val_upstream": "bolinas-dna/genomes-v5-validation-intervals-v1_255_255",
    "val_downstream": "bolinas-dna/genomes-v5-validation-intervals-v15_255_255",
}

REFERENCE_TPU_TYPE = "v5p-8"
TRANSFER_TPU_TYPE = "v5p-8"

# Reference sweep sizing
REFERENCE_HIDDEN_SIZE = 512  # ~25M params with vocab_size=7
INITIALIZER_RANGES = (0.04, 0.02, 0.01, 0.005, 0.0025)
EPOCHS = (1,)

# Vizier search space (same as text reference sweep)
SEARCH_SPACE = {
    "lr": (0.00005, 0.03),
    "beta1": (0.5, 1.0),
    "adam_lr": (0.00005, 0.03),
    "beta2": (0.5, 1.0),
    "epsilon": (1e-15, 1e-3),
    "max_grad_norm": (0.1, 1.0),
    "z_loss_weight": (1e-7, 0.1),
}

# Vizier sweep parameters
NUM_LOOPS = 10
SUGGESTIONS_PER_LOOP = 4
TARGET_TOKENS = 2_500_000_000  # ~100:1 token-to-param for ~25M
REFERENCE_BATCH_SIZE = 16384  # tokens/batch: 16384*256 = 4M (16x reference_hyperparameter_sweep 64*4096=256K)
METRIC_KEY = "eval/loss"  # TODO: confirm from smoke test tracker_metrics.jsonl
METRIC_FILE = "tracker_metrics.jsonl"
METRIC_MODE = "min"
VIZIER_ALGORITHM = "DEFAULT"
STUDY_OWNER = "marin"
WANDB_PROJECT = "marin"
REFERENCE_VERSION = "v0.6"


def _get_initializer_ranges() -> tuple[float, ...]:
    """Parse SWEEP_IR_VALUES env var (comma-separated) or return all IR values."""
    raw = os.getenv("SWEEP_IR_VALUES")
    if not raw:
        return INITIALIZER_RANGES
    values = tuple(float(v) for v in raw.split(","))
    invalid = set(values) - set(INITIALIZER_RANGES)
    if invalid:
        raise ValueError(f"Invalid IR values {invalid}. Must be in {INITIALIZER_RANGES}")
    return values


def _warmup_mode() -> bool:
    """Check WARMUP_MODE env var. Warmup submits a subset of jobs to validate the pipeline.

    All step configs are constructed identically regardless of this flag — warmup
    is purely a filter on which steps are passed to executor_main. This ensures
    warmup runs have the same hashes and won't be recomputed when warmup is disabled.
    """
    value = os.getenv("WARMUP_MODE", "no").lower()
    if value not in ("yes", "no"):
        raise ValueError(f"WARMUP_MODE must be 'yes' or 'no', got {value!r}")
    return value == "yes"


def _preview_mode() -> bool:
    """Check PREVIEW_MODE env var. Preview prints sweep configuration and exits."""
    value = os.getenv("PREVIEW_MODE", "no").lower()
    if value not in ("yes", "no"):
        raise ValueError(f"PREVIEW_MODE must be 'yes' or 'no', got {value!r}")
    return value == "yes"


# Schedule (same as text reference)
LR_SCHEDULE = "linear"
WARMUP_FRACTION = 0.1
DECAY_FRACTION = 0.2


# DNA-specific heuristic instance (vocab_size=7 affects param counting)
DNA_HEURISTIC = CompletedAdamHHeuristic(tokenizer=TOKENIZER)

_EXPECTED_VOCAB_SIZE_WARNING = f"Tokenizer {TOKENIZER!r} not found in _KNOWN_VOCAB_SIZES"
logging.getLogger("marin.processing.tokenize.data_configs").addFilter(
    lambda record: _EXPECTED_VOCAB_SIZE_WARNING not in record.getMessage()
)


# =============================================================================
# Shared builders
# =============================================================================


@lru_cache(maxsize=1)
def _model_seq_len() -> int:
    """Model context size = DNA base seq len + special tokens (BOS)."""
    return dna_effective_seq_len(DNA_BASE_SEQ_LEN, TOKENIZER)


def _num_train_steps(target_tokens: int = TARGET_TOKENS, batch_size: int = REFERENCE_BATCH_SIZE) -> int:
    return target_tokens // (batch_size * _model_seq_len())


def _build_model_config(hidden_size: int, initializer_range: float = 0.02):
    """Build Qwen3Config via the heuristic's architecture formula."""
    config = DNA_HEURISTIC._build_model_config(hidden_size, _model_seq_len())
    return replace(config, initializer_range=initializer_range)


def _tokenize_region() -> str | None:
    """Parse PIN_TOKENIZE_REGION env var. When set, force tokenize steps to that region.

    When unset, the marin executor infers regions from GCS path dependencies.
    Set to e.g. "us-east5" when the parent executor lands in a different region
    from the tokenized artifacts and the inferred-region guard would otherwise
    hard-fail step resolution.
    """
    value = os.getenv("PIN_TOKENIZE_REGION")
    return value if value else None


def _tokenize_dataset(name: str, dataset: str) -> ExecutorStep:
    region = _tokenize_region()
    resources = ResourceConfig.with_cpu(regions=[region]) if region else None
    return default_tokenize(
        name=name,
        dataset=dataset,
        tokenizer=TOKENIZER,
        format=DNALmDatasetFormat(lowercase_weight=0.01),
        resources=resources,
    )


def _build_data_mixture():
    """Tokenize train + validation datasets; validation has weight=0 (eval-only)."""
    tokenized = {
        region: _tokenize_dataset(f"bolinas-v5-{region}-char-bos", dataset) for region, dataset in TRAIN_DATASETS.items()
    }
    for region, dataset in VALIDATION_DATASETS.items():
        tokenized[region] = _tokenize_dataset(f"bolinas-v5-{region}-char-bos", dataset)
    # TRAIN_WEIGHTS only covers train keys; missing_weights_are_validation=True (default)
    # assigns weight=0 to val_* keys, making them validation-only.
    return lm_mixture_data_config(components=tokenized, weights=TRAIN_WEIGHTS)


def _dna_eval_harness_config() -> LmEvalHarnessConfig:
    """VEP eval harness config shared across sweeps."""
    return LmEvalHarnessConfig(
        task_spec=convert_to_levanter_task_config([TRAITGYM_MENDELIAN_V2_255]),
        include_path="experiments/evals/custom_tasks",
        max_packed_segments=1,
    )


# =============================================================================
# Smoke test
# =============================================================================


def run_smoke_test():
    """Run ~20-step infrastructure validation with the sweep's exact config."""
    mixture = _build_data_mixture()
    model_config = _build_model_config(REFERENCE_HIDDEN_SIZE)

    train_config = SimpleTrainConfig(
        resources=ResourceConfig.with_tpu(REFERENCE_TPU_TYPE),
        train_batch_size=32,
        num_train_steps=20,
        learning_rate=1e-3,
        steps_per_eval=10,
        steps_per_task_eval=10,
        steps_per_export=20,
    )

    train_step = default_train(
        name=f"dna-bolinas-smoke-{REFERENCE_VERSION}",
        tokenized=mixture,
        model_config=model_config,
        train_config=train_config,
        tags=["dna", "bolinas", "smoke_test", REFERENCE_VERSION],
        eval_harness_tasks=[TRAITGYM_MENDELIAN_V2_255],
        eval_harness_max_packed_segments=1,
        use_default_validation=False,
    )

    executor_main(steps=[train_step], description=f"DNA Bolinas smoke test {REFERENCE_VERSION}")


# =============================================================================
# Reference tuning sweep
# =============================================================================

# --- AdamH config builder ---


def _build_adamh_config(
    *,
    learning_rate: float,
    beta1: float,
    adam_learning_rate: float,
    beta2: float,
    epsilon: float,
    max_grad_norm: float,
) -> AdamHConfig:
    return AdamHConfig(
        learning_rate=learning_rate,
        adam_lr=adam_learning_rate,
        min_lr_ratio=0.0,
        warmup=WARMUP_FRACTION,
        decay=DECAY_FRACTION,
        lr_schedule=LR_SCHEDULE,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        max_grad_norm=max_grad_norm,
        nesterov=False,
    )


# --- Base training config ---


def _final_checkpoint_only(num_steps: int) -> CheckpointerConfig:
    """Save a single permanent checkpoint at the final training step."""
    return CheckpointerConfig(
        save_interval=timedelta(days=365),  # no time-based saves
        keep=[dict(every=num_steps)],  # permanent checkpoint at final step only
    )


def _build_base_train_config(
    model_config,
    data_mixture,
    *,
    wandb_group: str,
    base_tags: tuple[str, ...],
    checkpointer: CheckpointerConfig,
    steps_per_eval: int,
) -> TrainLmOnPodConfig:
    """Build TrainLmOnPodConfig with placeholder optimizer values.

    Placeholders for lr/batch_size/steps/z_loss are replaced at runtime
    inside run_dna_vizier_train once Vizier suggestions are available.
    """
    placeholder_lr = SEARCH_SPACE["lr"][0]
    placeholder_beta1 = SEARCH_SPACE["beta1"][0]
    placeholder_adam_lr = SEARCH_SPACE["adam_lr"][0]
    placeholder_beta2 = SEARCH_SPACE["beta2"][0]
    placeholder_epsilon = SEARCH_SPACE["epsilon"][0]
    placeholder_max_grad_norm = SEARCH_SPACE["max_grad_norm"][0]
    placeholder_z_loss = SEARCH_SPACE["z_loss_weight"][0]
    placeholder_steps = TARGET_TOKENS // (REFERENCE_BATCH_SIZE * _model_seq_len())

    inner = TrainLmConfig(
        data=data_mixture,
        model=model_config,
        train_seq_len=_model_seq_len(),
        z_loss_weight=placeholder_z_loss,
        optimizer=_build_adamh_config(
            learning_rate=placeholder_lr,
            beta1=placeholder_beta1,
            adam_learning_rate=placeholder_adam_lr,
            beta2=placeholder_beta2,
            epsilon=placeholder_epsilon,
            max_grad_norm=placeholder_max_grad_norm,
        ),
        trainer=TrainerConfig(
            tracker=WandbConfig(
                project=WANDB_PROJECT,
                tags=list(base_tags),
                group=wandb_group,
                name=None,
                replicate_path=this_output_path(),
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=REFERENCE_BATCH_SIZE,
            num_train_steps=placeholder_steps,
            steps_per_eval=steps_per_eval,
            checkpointer=checkpointer,
            mesh=MeshConfig(axes={"replica": 1, "data": -1, "model": 1}),
            allow_nondivisible_batch_size=True,
            crash_on_nan=False,
            crash_on_inf=False,
        ),
    )

    return TrainLmOnPodConfig(
        train_config=inner,
        resources=ResourceConfig.with_tpu(REFERENCE_TPU_TYPE),
        output_path=this_output_path(),
    )


# --- Vizier train config and function ---


@dataclass(frozen=True)
class DnaVizierTrainConfig:
    """Config passed to run_dna_vizier_train at execution time."""

    suggestions_path: str
    suggestion_index: int
    base_train_config: TrainLmOnPodConfig
    target_tokens: int
    seq_len: int
    batch_size: int
    loop_index: int
    initializer_range: float
    epochs: int
    version: str
    wandb_group: str
    base_tags: tuple[str, ...]


def run_dna_vizier_train(config: DnaVizierTrainConfig) -> None:
    """Train a DNA model for a single Vizier suggestion."""
    suggestions = _load_suggestions(config.suggestions_path)["suggestions"]
    if config.suggestion_index >= len(suggestions):
        raise IndexError(f"Suggestion index {config.suggestion_index} out of range")

    suggestion = suggestions[config.suggestion_index]
    hparams = _extract_adamh_hparams(suggestion)
    batch_size = config.batch_size
    num_steps = config.target_tokens // (batch_size * config.seq_len)
    trial_id = int(suggestion["trial_id"])

    base = config.base_train_config
    num_params = base.train_config.model.total_trainable_params(DNA_HEURISTIC.vocab_size)

    run_name = (
        f"dna-bolinas-reference-{config.version}"
        f"-IR{config.initializer_range}-E{config.epochs}"
        f"-L{config.loop_index}-T{trial_id}"
    )
    new_tags = [
        *config.base_tags,
        f"lr={hparams['lr']}",
        f"beta1={hparams['beta1']}",
        f"adam_lr={hparams['adam_lr']}",
        f"beta2={hparams['beta2']}",
        f"eps={hparams['epsilon']}",
        f"mgn={hparams['max_grad_norm']}",
        f"zloss={hparams['z_loss_weight']}",
        f"bs={batch_size}",
        f"params={num_params}",
        f"tokens={config.target_tokens}",
        f"trial={trial_id}",
        f"loop={config.loop_index}",
    ]
    inner = replace(
        base.train_config,
        optimizer=_build_adamh_config(
            learning_rate=hparams["lr"],
            beta1=hparams["beta1"],
            adam_learning_rate=hparams["adam_lr"],
            beta2=hparams["beta2"],
            epsilon=hparams["epsilon"],
            max_grad_norm=hparams["max_grad_norm"],
        ),
        z_loss_weight=hparams["z_loss_weight"],
        trainer=replace(
            base.train_config.trainer,
            num_train_steps=num_steps,
            train_batch_size=batch_size,
            tracker=replace(
                base.train_config.trainer.tracker,
                tags=new_tags,
                name=run_name,
            ),
        ),
    )
    pod_config = replace(base, train_config=inner)
    run_levanter_train_lm(pod_config)


# --- Step builders ---


def _build_dna_suggest_step(
    study_id: str,
    loop_index: int,
    input_db_path,
) -> ExecutorStep:
    client_id = f"{study_id}-loop-{loop_index}"
    return ExecutorStep(
        name=f"{study_id}-suggest-loop{loop_index}",
        fn=remote(run_vizier_suggest, resources=ResourceConfig.with_cpu(), pip_dependency_groups=["vizier"]),
        config=VizierSuggestConfig(
            study_owner=STUDY_OWNER,
            study_id=study_id,
            input_db_path=input_db_path,
            output_path=this_output_path(),
            num_suggestions=SUGGESTIONS_PER_LOOP,
            client_id=client_id,
            metric_key=METRIC_KEY,
            mode=METRIC_MODE,
            algorithm=VIZIER_ALGORITHM,
            search_space=SEARCH_SPACE,
            loop_index=loop_index,
        ),
    )


def _build_dna_train_step(
    suggest_step: ExecutorStep,
    suggestion_index: int,
    base_train_config: TrainLmOnPodConfig,
    *,
    loop_index: int,
    study_id: str,
    initializer_range: float,
    epochs: int,
    base_tags: tuple[str, ...],
    wandb_group: str,
    version: str,
) -> ExecutorStep:
    return ExecutorStep(
        name=os.path.join(
            "checkpoints",
            f"{study_id}-loop{loop_index}-trial{suggestion_index}",
        ),
        fn=remote(run_dna_vizier_train, resources=ResourceConfig.with_cpu()),
        config=DnaVizierTrainConfig(
            suggestions_path=suggest_step / SUGGESTIONS_FILENAME,
            suggestion_index=suggestion_index,
            base_train_config=base_train_config,
            target_tokens=TARGET_TOKENS,
            seq_len=_model_seq_len(),
            batch_size=REFERENCE_BATCH_SIZE,
            loop_index=loop_index,
            initializer_range=initializer_range,
            epochs=epochs,
            version=version,
            wandb_group=wandb_group,
            base_tags=base_tags,
        ),
    )


def _build_dna_update_step(
    study_id: str,
    loop_index: int,
    suggest_step: ExecutorStep,
    training_steps: list[ExecutorStep],
) -> ExecutorStep:
    study_resource_name = f"owners/{STUDY_OWNER}/studies/{study_id}"
    return ExecutorStep(
        name=f"{study_id}-update-loop{loop_index}",
        fn=remote(run_vizier_update, resources=ResourceConfig.with_cpu(), pip_dependency_groups=["vizier"]),
        config=VizierUpdateConfig(
            study_id=study_id,
            study_resource_name=study_resource_name,
            input_db_path=suggest_step / VIZIER_DB_FILENAME,
            suggestions_path=suggest_step / SUGGESTIONS_FILENAME,
            run_paths=[step.as_input_name() for step in training_steps],
            metric_file=METRIC_FILE,
            metric_key=METRIC_KEY,
            mode=METRIC_MODE,
            output_path=this_output_path(),
            loop_index=loop_index,
        ),
    )


def _build_dna_optimal_step(
    study_id: str,
    last_update_step: ExecutorStep,
) -> ExecutorStep:
    study_resource_name = f"owners/{STUDY_OWNER}/studies/{study_id}"
    return ExecutorStep(
        name=f"{study_id}-optimal",
        fn=remote(run_vizier_optimal, resources=ResourceConfig.with_cpu(), pip_dependency_groups=["vizier"]),
        config=VizierOptimalConfig(
            study_id=study_id,
            study_resource_name=study_resource_name,
            input_db_path=last_update_step / VIZIER_DB_FILENAME,
            output_path=this_output_path(),
        ),
    )


# --- Sweep orchestration ---


def run_reference_tuning_sweep():
    """Vizier Bayesian optimization over AdamH hparams.

    Outer loop: EPOCHS x INITIALIZER_RANGES (independent Vizier studies).
    Inner: suggest -> train x N -> update, repeated for num_loops.
    Final: extract optimal trials per study.
    """
    version = REFERENCE_VERSION
    mixture = _build_data_mixture()
    all_optimal_steps = []

    num_loops = NUM_LOOPS
    suggestions_per_loop = SUGGESTIONS_PER_LOOP
    initializer_ranges = _get_initializer_ranges()

    for epochs in EPOCHS:
        for init_range in initializer_ranges:
            study_id = f"dna-bolinas-ref-{version}-IR{init_range}-E{epochs}"
            wandb_group = f"dna-bolinas-reference-sweep-{version}"
            base_tags = (
                "sweep",
                "dna",
                "bolinas",
                "reference",
                version,
                f"epochs={epochs}",
                f"initializer_range={init_range}",
            )

            model_config = _build_model_config(REFERENCE_HIDDEN_SIZE, init_range)
            num_steps = _num_train_steps()
            base_config = _build_base_train_config(
                model_config,
                mixture,
                wandb_group=wandb_group,
                base_tags=base_tags,
                checkpointer=_final_checkpoint_only(num_steps),
                steps_per_eval=num_steps // 2,  # 3 evals per run: step 0 (forced), midpoint, final
            )

            previous_update_step = None
            for loop_index in range(num_loops):
                input_db_path = previous_update_step / VIZIER_DB_FILENAME if previous_update_step else None
                suggest_step = _build_dna_suggest_step(study_id, loop_index, input_db_path)

                training_steps = [
                    _build_dna_train_step(
                        suggest_step,
                        i,
                        base_config,
                        loop_index=loop_index,
                        study_id=study_id,
                        initializer_range=init_range,
                        epochs=epochs,
                        base_tags=base_tags,
                        wandb_group=wandb_group,
                        version=version,
                    )
                    for i in range(suggestions_per_loop)
                ]

                update_step = _build_dna_update_step(study_id, loop_index, suggest_step, training_steps)
                previous_update_step = update_step

            optimal_step = _build_dna_optimal_step(study_id, previous_update_step)
            all_optimal_steps.append(optimal_step)

    # WARMUP_MODE=yes: submit only the first IR study to validate the full pipeline.
    # All steps are constructed identically — warmup is purely a filter on which are submitted.
    if _warmup_mode():
        all_optimal_steps = all_optimal_steps[:1]

    executor_main(steps=all_optimal_steps, description=f"DNA Bolinas reference sweep {version}")


# =============================================================================
# Transfer validation sweep
# =============================================================================

# --- Reference hparams from wandb ---


@dataclass(frozen=True)
class ReferenceHparams:
    """Best hparams from the reference Vizier sweep, pulled from wandb.

    Single source of truth for all wandb-sourced values used by transfer/scaling
    sweeps. Update this when re-running the reference sweep or adding new fields
    (e.g. epochs).
    """

    # Optimizer base values (feed into CompletedAdamHHeuristic)
    lr: float
    adam_lr: float
    beta1: float
    beta2: float
    epsilon: float
    max_grad_norm: float
    z_loss_weight: float
    # Model config
    initializer_range: float


# Source: best run from wandb group 'dna-bolinas-reference-sweep-v0.6'
# 183 finished runs, rank 1/183, eval/loss=1.228545
# Run: dna-bolinas-reference-v0.6-IR0.02-E1-L8-T32
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


# --- Heuristic and scaling ---

# DNA-calibrated heuristic: CompletedAdamHHeuristic re-parameterized with the
# DNA reference sweep's optimal values as the base point. The scaling formulas
# (completed_adamh.py:162-209) then operate relative to the DNA reference
# regime (B0=16384, T0=2.5B) rather than the text defaults. Used by both the
# transfer validation sweep and the parameter scaling sweep.
DNA_SCALING_HEURISTIC = CompletedAdamHHeuristic(
    tokenizer=TOKENIZER,
    # Reference point (from Vizier-optimized sweep)
    reference_batch_size=REFERENCE_BATCH_SIZE,
    reference_tokens=TARGET_TOKENS,
    lr_base=REFERENCE_HPARAMS.lr,
    adam_lr_base=REFERENCE_HPARAMS.adam_lr,
    epsilon_base=REFERENCE_HPARAMS.epsilon,
    beta1=REFERENCE_HPARAMS.beta1,
    beta2_base=REFERENCE_HPARAMS.beta2,
    max_grad_norm=REFERENCE_HPARAMS.max_grad_norm,
    z_loss_weight=REFERENCE_HPARAMS.z_loss_weight,
    # Constraints — all explicitly set for DNA regime rather than relying on text defaults.
    # build_optimizer_config clips lr/adam_lr to max_learning_rate and beta2 to [min_beta2, max_beta2].
    # Reference optimal adam_lr (0.0299) nearly hits the text default clip of 0.01.
    max_learning_rate=0.03,
    min_beta2=0.5,
    max_beta2=0.9999,
    # Batch size limits — each downstream sweep asserts its own batch size against max_batch_size.
    min_batch_size=8,
    max_batch_size=8192,
)

TRANSFER_VERSION = "v0.14"
TRANSFER_HIDDEN_SIZE = 1920  # ~1.12B params
TRANSFER_TARGET_TOKENS = 10_000_000_000
TRANSFER_BATCH_SIZE = 4096
TRANSFER_NUM_POINTS = 7

assert (
    TRANSFER_BATCH_SIZE <= DNA_SCALING_HEURISTIC.max_batch_size
), f"TRANSFER_BATCH_SIZE={TRANSFER_BATCH_SIZE} exceeds heuristic max_batch_size={DNA_SCALING_HEURISTIC.max_batch_size}"

# Transferred optimizer config: the center of the sweep grid and the positive control.
# Scales reference-optimal hparams from (B0=16384, T0=2.5B) to (B=4096, T=10B).
TRANSFER_OPTIMIZER = DNA_SCALING_HEURISTIC.build_optimizer_config(TRANSFER_BATCH_SIZE, TRANSFER_TARGET_TOKENS)


# --- Sweep axes ---


@dataclass(frozen=True)
class TransferSweepAxis:
    """One axis of the transfer validation grid.

    Bounds are fixed feasibility guard rails. The grid is centered at the
    transferred optimizer value for this field (from TRANSFER_OPTIMIZER).
    """

    field: str  # field name on AdamHConfig (e.g. "learning_rate")
    low: float
    high: float
    log_scale: bool


# Feasibility bounds for sweep — derived from heuristic constraints, not hardcoded separately.
TRANSFER_BOUNDS: dict[str, tuple[float, float]] = {
    "learning_rate": (1e-5, DNA_SCALING_HEURISTIC.max_learning_rate),
    "beta2": (DNA_SCALING_HEURISTIC.min_beta2, DNA_SCALING_HEURISTIC.max_beta2),
}

TRANSFER_SWEEP_AXES = tuple(
    TransferSweepAxis(
        field=field,
        low=TRANSFER_BOUNDS[field][0],
        high=TRANSFER_BOUNDS[field][1],
        log_scale=(field == "learning_rate"),
    )
    for field in ("learning_rate", "beta2")
)


def _build_transfer_grid(axis: TransferSweepAxis, center: float, num_points: int = TRANSFER_NUM_POINTS) -> list[float]:
    """Generate `num_points` grid values centered at `center` within [axis.low, axis.high].

    Returns exactly `num_points` values: (num_points-1)//2 below center, center
    itself, and the remainder above. Spacing is log or linear per `axis.log_scale`.
    """
    assert (
        axis.low <= center <= axis.high
    ), f"Transferred center {center} outside bounds [{axis.low}, {axis.high}] for {axis.field}"
    n_below = (num_points - 1) // 2
    n_above = num_points - 1 - n_below

    if axis.log_scale:
        log_center = math.log(center)
        log_span = min(log_center - math.log(axis.low), math.log(axis.high) - log_center)
        log_low = log_center - log_span
        log_high = log_center + log_span
        below = [math.exp(log_low + i * (log_center - log_low) / n_below) for i in range(n_below)]
        above = [math.exp(log_center + (i + 1) * (log_high - log_center) / n_above) for i in range(n_above)]
    else:
        span = min(center - axis.low, axis.high - center)
        low = center - span
        high = center + span
        below = [low + i * (center - low) / n_below for i in range(n_below)]
        above = [center + (i + 1) * (high - center) / n_above for i in range(n_above)]

    grid = [*below, center, *above]
    assert all(axis.low <= v <= axis.high for v in grid), f"Grid values outside bounds for {axis.field}: {grid}"
    return grid


# --- Preview ---


def _print_transfer_preview():
    """Print transfer sweep configuration: transferred hparams, bounds, and grid values."""
    transferred_fields = ("learning_rate", "adam_lr", "epsilon", "beta2")  # scaled by build_optimizer_config
    swept_fields = {axis.field for axis in TRANSFER_SWEEP_AXES}

    negative_optimizer = DNA_SCALING_HEURISTIC.build_optimizer_config(
        DNA_SCALING_HEURISTIC.reference_batch_size,
        DNA_SCALING_HEURISTIC.reference_tokens,
    )

    print("=" * 70)
    print(f"Transfer validation sweep preview — {TRANSFER_VERSION}")
    print(f"  model: hidden={TRANSFER_HIDDEN_SIZE}, batch={TRANSFER_BATCH_SIZE}, tokens={TRANSFER_TARGET_TOKENS:.0e}")
    num_steps = TRANSFER_TARGET_TOKENS // (TRANSFER_BATCH_SIZE * _model_seq_len())
    print(f"  num_steps: {num_steps}")
    print()

    print("Transferred hparams (scaled by CompletedAdamH):")
    for field in transferred_fields:
        pos = getattr(TRANSFER_OPTIMIZER, field)
        neg = getattr(negative_optimizer, field)
        swept = "SWEPT" if field in swept_fields else ""
        print(f"  {field:20s}  positive={pos:<12.6g}  negative={neg:<12.6g}  {swept}")
    print()

    print("Passed-through hparams (not scaled):")
    for field in ("beta1", "min_lr_ratio", "warmup", "max_grad_norm", "lr_schedule", "decay", "nesterov"):
        pos = getattr(TRANSFER_OPTIMIZER, field)
        swept = "SWEPT" if field in swept_fields else ""
        print(f"  {field:20s}  value={pos!s:<12}  {swept}")
    print()

    print("Sweep grid per axis:")
    for axis in TRANSFER_SWEEP_AXES:
        center = getattr(TRANSFER_OPTIMIZER, axis.field)
        grid = _build_transfer_grid(axis, center)
        off_center = [v for v in grid if v != center]
        print(f"  {axis.field}  bounds=[{axis.low:.6g}, {axis.high:.6g}]  center={center:.6g}  log={axis.log_scale}")
        print(f"    grid ({len(grid)} points, {len(off_center)} off-center): {[f'{v:.6g}' for v in grid]}")
    print()

    total = (
        1
        + 1
        + sum(len(_build_transfer_grid(ax, getattr(TRANSFER_OPTIMIZER, ax.field))) - 1 for ax in TRANSFER_SWEEP_AXES)
    )
    print(f"Total runs: {total} (1 positive + 1 negative + {total - 2} off-center)")
    print("=" * 70)


# --- Checkpointing ---


def _periodic_checkpoint(*, keep_every: int, interval_hours: int) -> CheckpointerConfig:
    """Save time-based checkpoints and permanent checkpoints at keep_every steps."""
    return CheckpointerConfig(
        save_interval=timedelta(hours=interval_hours),
        keep=[dict(every=keep_every)],
    )


# --- Step builder ---


def _build_scaled_train_step(
    optimizer: AdamHConfig,
    model_config,
    data_mixture,
    *,
    run_name: str,
    wandb_group: str,
    tags: tuple[str, ...],
    batch_size: int,
    num_train_steps: int,
    tpu_type: str | Sequence[str],
    checkpointer: CheckpointerConfig,
    run_eval_harness: bool,
) -> ExecutorStep:
    """Build an ExecutorStep for a single scaled (transfer or parameter-scaling) run.

    `tpu_type` may be a list when flexible fallback scheduling is desired.
    """
    steps_per_eval = max(1, num_train_steps // 32)

    inner = TrainLmConfig(
        data=data_mixture,
        model=model_config,
        train_seq_len=_model_seq_len(),
        z_loss_weight=REFERENCE_HPARAMS.z_loss_weight,
        optimizer=optimizer,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                project=WANDB_PROJECT,
                tags=list(tags),
                group=wandb_group,
                name=run_name,
                replicate_path=this_output_path(),
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=batch_size,
            num_train_steps=num_train_steps,
            steps_per_eval=steps_per_eval,
            checkpointer=checkpointer,
            mesh=MeshConfig(axes={"replica": 1, "data": -1, "model": 1}),
            allow_nondivisible_batch_size=True,
        ),
    )
    if run_eval_harness:
        inner = replace(inner, eval_harness=_dna_eval_harness_config(), eval_harness_steps=steps_per_eval)

    pod_config = TrainLmOnPodConfig(
        train_config=inner,
        resources=ResourceConfig.with_tpu(tpu_type, ram="300g"),
        output_path=this_output_path(),
    )

    return ExecutorStep(
        name=os.path.join("checkpoints", run_name),
        fn=remote(run_levanter_train_lm, resources=ResourceConfig.with_cpu()),
        config=pod_config,
    )


# --- Sweep orchestration ---


def _get_transfer_steps(all_steps: list[ExecutorStep]) -> list[ExecutorStep]:
    """Parse SWEEP_TRANSFER_INDICES env var (CSV of indices into all_steps) or return all."""
    raw = _csv_env("SWEEP_TRANSFER_INDICES", ())
    if not raw:
        return all_steps
    indices = tuple(int(i) for i in raw)
    n = len(all_steps)
    invalid = [i for i in indices if not 0 <= i < n]
    if invalid:
        raise ValueError(f"Invalid indices {invalid}. Must be in [0, {n})")
    return [all_steps[i] for i in indices]


def run_transfer_validation_sweep():
    """Sweep LR, beta1, beta2 in isolation at full-epoch scale with ~4B model.

    Positive control: transferred optimizer (scaled via DNA_SCALING_HEURISTIC).
    Negative control: heuristic at reference point (unscaled, to validate transfer helps).
    Per-axis sweeps: 7 points centered at transferred value, 3 axes, center deduplicated.
    Total: 1 positive + 1 negative + 18 off-center = 20 runs.
    """
    if _preview_mode():
        _print_transfer_preview()
        return

    version = TRANSFER_VERSION
    mixture = _build_data_mixture()
    model_config = _build_model_config(TRANSFER_HIDDEN_SIZE, REFERENCE_HPARAMS.initializer_range)
    num_params = model_config.total_trainable_params(DNA_SCALING_HEURISTIC.vocab_size)
    wandb_group = f"dna-bolinas-transfer-sweep-{version}"

    base_tags = (
        "sweep",
        "dna",
        "bolinas",
        "transfer",
        version,
        f"params={num_params}",
        f"tokens={TRANSFER_TARGET_TOKENS}",
        f"bs={TRANSFER_BATCH_SIZE}",
    )

    num_steps = TRANSFER_TARGET_TOKENS // (TRANSFER_BATCH_SIZE * _model_seq_len())
    checkpointer = _periodic_checkpoint(keep_every=num_steps // 3, interval_hours=1)

    def _make_step(optimizer: AdamHConfig, run_name: str, tags: tuple[str, ...]) -> ExecutorStep:
        return _build_scaled_train_step(
            optimizer=optimizer,
            model_config=model_config,
            data_mixture=mixture,
            run_name=run_name,
            wandb_group=wandb_group,
            tags=tags,
            batch_size=TRANSFER_BATCH_SIZE,
            num_train_steps=num_steps,
            tpu_type=TRANSFER_TPU_TYPE,
            checkpointer=checkpointer,
            run_eval_harness=False,
        )

    all_steps: list[ExecutorStep] = []

    # Positive control: transferred (scaled) optimizer
    all_steps.append(
        _make_step(
            TRANSFER_OPTIMIZER,
            f"dna-bolinas-transfer-{version}-positive-control",
            (*base_tags, "role=positive-control"),
        )
    )

    # Negative control: heuristic evaluated at its reference point (B0, T0) — i.e. the
    # raw reference-optimal hparams without transfer scaling. If scaling works, the
    # positive control (transferred) should outperform this.
    negative_optimizer = DNA_SCALING_HEURISTIC.build_optimizer_config(
        DNA_SCALING_HEURISTIC.reference_batch_size,
        DNA_SCALING_HEURISTIC.reference_tokens,
    )
    all_steps.append(
        _make_step(
            negative_optimizer,
            f"dna-bolinas-transfer-{version}-negative-control",
            (*base_tags, "role=negative-control"),
        )
    )

    # Per-axis sweeps: 7 points each, center deduplicated as positive control above
    if not _warmup_mode():
        for axis in TRANSFER_SWEEP_AXES:
            center = getattr(TRANSFER_OPTIMIZER, axis.field)
            grid = _build_transfer_grid(axis, center)
            for i, value in enumerate(grid):
                if value == center:
                    continue  # deduplicated as positive control
                swept_optimizer = replace(TRANSFER_OPTIMIZER, **{axis.field: value})
                all_steps.append(
                    _make_step(
                        swept_optimizer,
                        f"dna-bolinas-transfer-{version}-{axis.field}-{i}",
                        (*base_tags, f"axis={axis.field}", f"{axis.field}={value}"),
                    )
                )

    # TODO: Remove this filter — temporary hack to run a subset of steps
    # _skip = ("-negative-control", "-positive-control", *[f"-learning_rate-{i}" for i in range(TRANSFER_NUM_POINTS)])
    # all_steps = [s for s in all_steps if not any(s.name.endswith(k) for k in _skip)]
    all_steps = [s for s in all_steps if "-beta2-" in s.name]

    all_steps = _get_transfer_steps(all_steps)

    executor_main(steps=all_steps, description=f"DNA Bolinas transfer validation {version}")


# =============================================================================
# Parameter scaling sweep
# =============================================================================

SCALING_VERSION = "v0.5"
SCALING_TPU_TYPES: tuple[str, ...] = ("v6e-8",)
SCALING_BATCH_SIZE = 1536
SCALING_WARMUP_STEPS = 100

# Total training tokens available: sum of examples across CDS/upstream/downstream mixtures,
# each example = DNA_BASE_SEQ_LEN + 1 BOS = 256 tokens. Source: exp109 .md.
SCALING_TRAIN_EXAMPLES = 331_122_738

# 8 model sizes spanning ~46M → ~4.02B params. Centered on 1920 (the transfer validation
# sweep's 1.12B model) so the scaling study anchors on a hparam-validated regime.
# Consecutive param ratios: 1.63x, 1.71x, 2.00x, 1.87x, 2.36x, 2.02x, 1.77x.
# All sizes satisfy hidden_size % hidden_head_ratio (128) == 0, required by CompletedAdamH.
SCALING_HIDDEN_SIZES: tuple[int, ...] = (640, 768, 896, 1152, 1408, 1920, 2432, 2944)

assert (
    SCALING_BATCH_SIZE <= DNA_SCALING_HEURISTIC.max_batch_size
), f"SCALING_BATCH_SIZE={SCALING_BATCH_SIZE} exceeds heuristic max_batch_size={DNA_SCALING_HEURISTIC.max_batch_size}"
assert TRANSFER_HIDDEN_SIZE in SCALING_HIDDEN_SIZES, "Scaling sweep must include the transfer validation model"


def _scaling_target_tokens() -> int:
    """Total tokens in one pass over the training mixture."""
    return SCALING_TRAIN_EXAMPLES * _model_seq_len()


def _csv_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    """Parse a comma-separated env var (stripping whitespace) or return `default`."""
    raw = os.getenv(name)
    if not raw:
        return default
    return tuple(s.strip() for s in raw.split(","))


def _get_scaling_hidden_sizes() -> tuple[int, ...]:
    """Parse SWEEP_MODEL_INDICES env var (CSV of indices into SCALING_HIDDEN_SIZES) or return all."""
    raw = _csv_env("SWEEP_MODEL_INDICES", ())
    if not raw:
        return SCALING_HIDDEN_SIZES
    indices = tuple(int(i) for i in raw)
    n = len(SCALING_HIDDEN_SIZES)
    invalid = [i for i in indices if not 0 <= i < n]
    if invalid:
        raise ValueError(f"Invalid indices {invalid}. Must be in [0, {n})")
    return tuple(SCALING_HIDDEN_SIZES[i] for i in indices)


def _format_params(n: int) -> str:
    """Compact param-count label: 46M, 255M, 1B, 4B, etc."""
    if n >= 1_000_000_000:
        return f"{round(n / 1e9)}B"
    return f"{round(n / 1e6)}M"


def run_parameter_scaling_sweep():
    """Parameter scaling sweep: 8 model sizes (46M to 4B) trained for one epoch each.

    Each model uses the AdamH optimizer config produced by DNA_SCALING_HEURISTIC at
    (SCALING_BATCH_SIZE, target_tokens). No controls — every run is at the transferred
    optimum for its (B, T). num_train_steps is floored so total tokens consumed never
    exceed one pass over the mixture.
    """
    version = SCALING_VERSION
    mixture = _build_data_mixture()
    target_tokens = _scaling_target_tokens()
    full_num_steps = target_tokens // (SCALING_BATCH_SIZE * _model_seq_len())
    # WARMUP_MODE=yes: cap each run to SCALING_WARMUP_STEPS. Optimizer is still built at full
    # target_tokens so the LR schedule endpoint matches production — the run just stops early.
    num_steps = SCALING_WARMUP_STEPS if _warmup_mode() else full_num_steps
    optimizer = DNA_SCALING_HEURISTIC.build_optimizer_config(SCALING_BATCH_SIZE, target_tokens)
    checkpointer = _periodic_checkpoint(keep_every=max(1, num_steps // 3), interval_hours=1)
    wandb_group = f"dna-bolinas-scaling-sweep-{version}"

    tpu_types = _csv_env("TPU_TYPES", SCALING_TPU_TYPES)
    hidden_sizes = _get_scaling_hidden_sizes()
    all_steps: list[ExecutorStep] = []
    for hidden_size in hidden_sizes:
        model_config = _build_model_config(hidden_size, REFERENCE_HPARAMS.initializer_range)
        num_params = model_config.total_trainable_params(DNA_SCALING_HEURISTIC.vocab_size)
        tags = (
            "sweep",
            "dna",
            "bolinas",
            "scaling",
            version,
            f"hidden={hidden_size}",
            f"params={num_params}",
            f"tokens={target_tokens}",
            f"bs={SCALING_BATCH_SIZE}",
        )
        all_steps.append(
            _build_scaled_train_step(
                optimizer=optimizer,
                model_config=model_config,
                data_mixture=mixture,
                run_name=f"dna-bolinas-scaling-{version}-h{hidden_size}-p{_format_params(num_params)}",
                wandb_group=wandb_group,
                tags=tags,
                batch_size=SCALING_BATCH_SIZE,
                num_train_steps=num_steps,
                tpu_type=tpu_types,
                checkpointer=checkpointer,
                run_eval_harness=True,
            )
        )

    executor_main(steps=all_steps, description=f"DNA Bolinas parameter scaling {version}")


# =============================================================================
# Entry point
# =============================================================================

COMMANDS = (
    "run_smoke_test",
    "run_reference_tuning_sweep",
    "run_transfer_validation_sweep",
    "run_parameter_scaling_sweep",
)

if __name__ == "__main__":
    command = os.environ.get("SWEEP_COMMAND")
    if command is None or command not in COMMANDS:
        raise ValueError(f"Set SWEEP_COMMAND to one of: {', '.join(COMMANDS)}")
    globals()[command]()
