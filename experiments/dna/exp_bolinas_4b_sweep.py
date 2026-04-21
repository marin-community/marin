# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Bolinas DNA 4B sweep.

Vizier sweep over (lr, beta1, beta2, epsilon, weight_decay) at 4B model
scale with vanilla AdamConfig, uniform mixture of CDS / upstream / downstream,
and one epoch to maximize ``lm_eval/traitgym_mendelian_v2_255/auprc``.
"""

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import timedelta
from functools import lru_cache

import jmp
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import DNALmDatasetFormat
from levanter.eval_harness import LmEvalHarnessConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig

from experiments.defaults import default_tokenize
from experiments.dna.defaults import dna_effective_seq_len
from experiments.evals.task_configs import TRAITGYM_MENDELIAN_V2_255, convert_to_levanter_task_config
from experiments.references.reference_hyperparameter_sweep import (
    SUGGESTIONS_FILENAME,
    VIZIER_DB_FILENAME,
    VizierOptimalConfig,
    VizierSuggestConfig,
    VizierUpdateConfig,
    _load_suggestions,
    run_vizier_optimal,
    run_vizier_suggest,
    run_vizier_update,
)
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.execution.remote import remote
from marin.processing.tokenize import lm_mixture_data_config
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm

# =============================================================================
# Constants
# =============================================================================

VERSION = "v0.3"
TOKENIZER = "bolinas-dna/tokenizer-char-bos"
DNA_BASE_SEQ_LEN = 255  # bp (256 - 1 for BOS)

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
# Uniform weights for each training mixture component.
TRAIN_WEIGHTS = {name: 1.0 / len(TRAIN_DATASETS) for name in TRAIN_DATASETS}

# Dataset-imposed ceiling on any single training mixture component:
# smallest componenent "downstream" has ~20.5M examples;
# see https://github.com/Open-Athena/bolinas-dna/issues/109
COMPONENT_MAX_EXAMPLES = 20_501_856

# Per-component cap actually used for training each epoch (≤ COMPONENT_MAX_EXAMPLES).
# Sized to target ~3hr per-trial runtimes.
COMPONENT_TRAIN_EXAMPLES = 516_096
assert COMPONENT_TRAIN_EXAMPLES <= COMPONENT_MAX_EXAMPLES, (
    f"COMPONENT_TRAIN_EXAMPLES ({COMPONENT_TRAIN_EXAMPLES}) must be "
    f"<= COMPONENT_MAX_EXAMPLES ({COMPONENT_MAX_EXAMPLES})"
)

# Accelerator batch size.
BATCH_SIZE = 4096
TPU_TYPES: tuple[str, ...] = ("v5p-8",)

# Architecture (~4.03B params at vocab_size=7).
MODEL_HIDDEN_DIM = 2944
MODEL_INTERMEDIATE_DIM = MODEL_HIDDEN_DIM * 4  # 11776
MODEL_NUM_HEADS = MODEL_HIDDEN_DIM // 128  # 23 (head_dim = 128)
MODEL_NUM_LAYERS = 29

# WSD schedule: warmup → constant stable → linear decay.
LR_SCHEDULE = "linear"
WARMUP_FRACTION = 0.1
DECAY_FRACTION = 0.2

# Vizier sweep: fixed grad norm, weight_decay in (0, 3) per https://arxiv.org/abs/2509.14786.
MAX_GRAD_NORM = 1.0
SEARCH_SPACE: dict[str, tuple[float, float]] = {
    "lr": (0.00005, 0.03),
    "beta1": (0.5, 1.0),
    "beta2": (0.5, 1.0),
    "epsilon": (1e-15, 1e-3),
    "weight_decay": (0.0, 3.0),
}
NUM_LOOPS = 10
SUGGESTIONS_PER_LOOP = 4

# Warmup mode (WARMUP_MODE=yes): smoke-test the full pipeline end-to-end
WARMUP_NUM_TRAIN_STEPS = 50
WARMUP_NUM_LOOPS = 1
WARMUP_SUGGESTIONS_PER_LOOP = 2

# Evaluation metrics: target overall Mendelian VEP AUPRC w/ 3 evals per trial.
METRIC_KEY = "lm_eval/traitgym_mendelian_v2_255/auprc"
METRIC_FILE = "tracker_metrics.jsonl"
METRIC_MODE = "max"
VIZIER_ALGORITHM = "DEFAULT"
STUDY_OWNER = "marin"
WANDB_PROJECT = "marin"
EVALS_PER_RUN = 3

_EXPECTED_VOCAB_SIZE_WARNING = f"Tokenizer {TOKENIZER!r} not found in _KNOWN_VOCAB_SIZES"
logging.getLogger("marin.processing.tokenize.data_configs").addFilter(
    lambda record: _EXPECTED_VOCAB_SIZE_WARNING not in record.getMessage()
)


def _warmup_mode() -> bool:
    value = os.getenv("WARMUP_MODE", "no").lower()
    if value not in ("yes", "no"):
        raise ValueError(f"WARMUP_MODE must be 'yes' or 'no', got {value!r}")
    return value == "yes"


# =============================================================================
# Shared builders
# =============================================================================


@lru_cache(maxsize=1)
def _model_seq_len() -> int:
    """Model context size = base DNA seq len + special tokens (BOS)."""
    return dna_effective_seq_len(DNA_BASE_SEQ_LEN, TOKENIZER)


def _max_train_batches_per_component() -> int:
    """Per-component cap expressed in batches (= floor(examples / batch_size))."""
    return COMPONENT_TRAIN_EXAMPLES // BATCH_SIZE


def _num_train_steps() -> int:
    """One epoch over the capped mixture: sum of per-component batch caps.

    In warmup mode, returns ``WARMUP_NUM_TRAIN_STEPS`` so the LR schedule and
    eval cadence (all derived from this) compress together.
    """
    if _warmup_mode():
        return WARMUP_NUM_TRAIN_STEPS
    return len(TRAIN_DATASETS) * _max_train_batches_per_component()


def _tokenize_dataset(name: str, dataset: str) -> ExecutorStep:
    return default_tokenize(
        name=name,
        dataset=dataset,
        tokenizer=TOKENIZER,
        format=DNALmDatasetFormat(lowercase_weight=0.01),
    )


def _build_data_mixture():
    """Tokenize train + validation datasets; validation has weight=0 (eval-only)."""
    tokenized = {
        region: _tokenize_dataset(f"bolinas-v5-{region}-char-bos", dataset)
        for region, dataset in {**TRAIN_DATASETS, **VALIDATION_DATASETS}.items()
    }
    max_batches = _max_train_batches_per_component()
    return lm_mixture_data_config(
        components=tokenized,
        weights=TRAIN_WEIGHTS,
        max_train_batches={name: max_batches for name in TRAIN_DATASETS},
    )


def _build_model_config() -> Qwen3Config:
    return Qwen3Config(
        hidden_dim=MODEL_HIDDEN_DIM,
        intermediate_dim=MODEL_INTERMEDIATE_DIM,
        num_layers=MODEL_NUM_LAYERS,
        num_heads=MODEL_NUM_HEADS,
        num_kv_heads=MODEL_NUM_HEADS,
        max_seq_len=_model_seq_len(),
        rope=Llama3RotaryEmbeddingsConfig(),
    )


def _dna_eval_harness_config() -> LmEvalHarnessConfig:
    return LmEvalHarnessConfig(
        task_spec=convert_to_levanter_task_config([TRAITGYM_MENDELIAN_V2_255]),
        include_path="experiments/evals/custom_tasks",
        max_packed_segments=1,
    )


def _build_adam_config(
    *,
    learning_rate: float,
    weight_decay: float,
    beta1: float,
    beta2: float,
    epsilon: float,
) -> AdamConfig:
    return AdamConfig(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        min_lr_ratio=0.0,
        warmup=WARMUP_FRACTION,
        decay=DECAY_FRACTION,
        lr_schedule=LR_SCHEDULE,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        max_grad_norm=MAX_GRAD_NORM,
    )


def _checkpointer(num_steps: int) -> CheckpointerConfig:
    """Hourly time-based saves (for crash recovery) + one permanent at the final step."""
    return CheckpointerConfig(
        save_interval=timedelta(hours=1),
        keep=[dict(every=num_steps)],
    )


def _build_base_train_config(*, wandb_group: str, base_tags: tuple[str, ...]) -> TrainLmOnPodConfig:
    """TrainLmOnPodConfig with placeholder optimizer — swapped in per-trial at runtime."""
    num_steps = _num_train_steps()
    steps_per_eval = num_steps // EVALS_PER_RUN

    inner = TrainLmConfig(
        data=_build_data_mixture(),
        model=_build_model_config(),
        train_seq_len=_model_seq_len(),
        optimizer=_build_adam_config(
            learning_rate=SEARCH_SPACE["lr"][0],
            weight_decay=SEARCH_SPACE["weight_decay"][0],
            beta1=SEARCH_SPACE["beta1"][0],
            beta2=SEARCH_SPACE["beta2"][0],
            epsilon=SEARCH_SPACE["epsilon"][0],
        ),
        eval_harness=_dna_eval_harness_config(),
        eval_harness_steps=steps_per_eval,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                project=WANDB_PROJECT,
                tags=list(base_tags),
                group=wandb_group,
                name=None,
                replicate_path=this_output_path(),
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=BATCH_SIZE,
            num_train_steps=num_steps,
            steps_per_eval=steps_per_eval,
            checkpointer=_checkpointer(num_steps),
            mesh=MeshConfig(axes={"replica": 1, "data": -1, "model": 1}),
            allow_nondivisible_batch_size=True,
            crash_on_nan=False,
            crash_on_inf=False,
        ),
    )
    return TrainLmOnPodConfig(
        train_config=inner,
        resources=ResourceConfig.with_tpu(TPU_TYPES),
        output_path=this_output_path(),
    )


# =============================================================================
# Vizier train step
# =============================================================================

_HPARAM_FIELDS = ("lr", "beta1", "beta2", "epsilon", "weight_decay")


def _extract_hparams(suggestion: dict) -> dict[str, float]:
    parameters = suggestion["parameters"]
    if not isinstance(parameters, Mapping):
        raise ValueError(f"Expected suggestion parameters mapping, got {type(parameters)!r}")
    return {name: float(parameters[name]) for name in _HPARAM_FIELDS}


@dataclass(frozen=True)
class VizierTrainConfig:
    """Config passed to ``run_vizier_train`` at execution time."""

    suggestions_path: str
    suggestion_index: int
    base_train_config: TrainLmOnPodConfig
    loop_index: int
    version: str
    base_tags: tuple[str, ...]


def run_vizier_train(config: VizierTrainConfig) -> None:
    """Train one Vizier suggestion: swap the optimizer + wandb tags, then train."""
    suggestions = _load_suggestions(config.suggestions_path)["suggestions"]
    if config.suggestion_index >= len(suggestions):
        raise IndexError(f"Suggestion index {config.suggestion_index} out of range")

    suggestion = suggestions[config.suggestion_index]
    hparams = _extract_hparams(suggestion)
    trial_id = int(suggestion["trial_id"])
    run_name = f"dna-bolinas-4b-{config.version}-L{config.loop_index}-T{trial_id}"

    base = config.base_train_config
    assert isinstance(base.train_config, TrainLmConfig)  # narrow from object
    assert isinstance(base.train_config.trainer.tracker, WandbConfig)  # narrow from union
    new_tags = [
        *config.base_tags,
        *[f"{k}={v}" for k, v in hparams.items()],
        f"bs={BATCH_SIZE}",
        f"trial={trial_id}",
        f"loop={config.loop_index}",
    ]
    inner = replace(
        base.train_config,
        optimizer=_build_adam_config(
            learning_rate=hparams["lr"],
            weight_decay=hparams["weight_decay"],
            beta1=hparams["beta1"],
            beta2=hparams["beta2"],
            epsilon=hparams["epsilon"],
        ),
        trainer=replace(
            base.train_config.trainer,
            tracker=replace(base.train_config.trainer.tracker, tags=new_tags, name=run_name),
        ),
    )
    run_levanter_train_lm(replace(base, train_config=inner))


# =============================================================================
# Step builders
# =============================================================================


def _build_suggest_step(study_id: str, loop_index: int, input_db_path, num_suggestions: int) -> ExecutorStep:
    return ExecutorStep(
        name=f"{study_id}-suggest-loop{loop_index}",
        fn=remote(run_vizier_suggest, resources=ResourceConfig.with_cpu(), pip_dependency_groups=["vizier"]),
        config=VizierSuggestConfig(
            study_owner=STUDY_OWNER,
            study_id=study_id,
            input_db_path=input_db_path,
            output_path=this_output_path(),
            num_suggestions=num_suggestions,
            client_id=f"{study_id}-loop-{loop_index}",
            metric_key=METRIC_KEY,
            mode=METRIC_MODE,
            algorithm=VIZIER_ALGORITHM,
            search_space=SEARCH_SPACE,
            loop_index=loop_index,
        ),
    )


def _build_train_step(
    suggest_step: ExecutorStep,
    suggestion_index: int,
    base_train_config: TrainLmOnPodConfig,
    *,
    study_id: str,
    loop_index: int,
    base_tags: tuple[str, ...],
) -> ExecutorStep:
    return ExecutorStep(
        name=os.path.join("checkpoints", f"{study_id}-loop{loop_index}-trial{suggestion_index}"),
        fn=remote(run_vizier_train, resources=ResourceConfig.with_cpu()),
        config=VizierTrainConfig(
            suggestions_path=suggest_step / SUGGESTIONS_FILENAME,
            suggestion_index=suggestion_index,
            base_train_config=base_train_config,
            loop_index=loop_index,
            version=VERSION,
            base_tags=base_tags,
        ),
    )


def _build_update_step(
    study_id: str,
    loop_index: int,
    suggest_step: ExecutorStep,
    training_steps: list[ExecutorStep],
) -> ExecutorStep:
    return ExecutorStep(
        name=f"{study_id}-update-loop{loop_index}",
        fn=remote(run_vizier_update, resources=ResourceConfig.with_cpu(), pip_dependency_groups=["vizier"]),
        config=VizierUpdateConfig(
            study_id=study_id,
            study_resource_name=f"owners/{STUDY_OWNER}/studies/{study_id}",
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


def _build_optimal_step(study_id: str, last_update_step: ExecutorStep) -> ExecutorStep:
    return ExecutorStep(
        name=f"{study_id}-optimal",
        fn=remote(run_vizier_optimal, resources=ResourceConfig.with_cpu(), pip_dependency_groups=["vizier"]),
        config=VizierOptimalConfig(
            study_id=study_id,
            study_resource_name=f"owners/{STUDY_OWNER}/studies/{study_id}",
            input_db_path=last_update_step / VIZIER_DB_FILENAME,
            output_path=this_output_path(),
        ),
    )


# =============================================================================
# Sweep
# =============================================================================


def run_sweep():
    """Single Vizier sweep: NUM_LOOPS * SUGGESTIONS_PER_LOOP trials at 4B scale.

    Warmup mode (WARMUP_MODE=yes) shrinks num_loops, suggestions/loop, and
    num_train_steps and routes outputs to a separate ``-warmup`` study/group so
    production state stays clean.
    """
    warmup = _warmup_mode()
    suffix = f"{VERSION}-warmup" if warmup else VERSION
    study_id = f"dna-bolinas-4b-{suffix}"
    wandb_group = f"dna-bolinas-4b-sweep-{suffix}"
    base_tags = ("sweep", "dna", "bolinas", "4b", VERSION, *(("warmup",) if warmup else ()))

    num_loops = WARMUP_NUM_LOOPS if warmup else NUM_LOOPS
    suggestions_per_loop = WARMUP_SUGGESTIONS_PER_LOOP if warmup else SUGGESTIONS_PER_LOOP

    base_config = _build_base_train_config(wandb_group=wandb_group, base_tags=base_tags)

    previous_update_step: ExecutorStep | None = None
    for loop_index in range(num_loops):
        input_db_path = previous_update_step / VIZIER_DB_FILENAME if previous_update_step else None
        suggest_step = _build_suggest_step(study_id, loop_index, input_db_path, suggestions_per_loop)
        training_steps = [
            _build_train_step(
                suggest_step,
                i,
                base_config,
                study_id=study_id,
                loop_index=loop_index,
                base_tags=base_tags,
            )
            for i in range(suggestions_per_loop)
        ]
        previous_update_step = _build_update_step(study_id, loop_index, suggest_step, training_steps)

    optimal_step = _build_optimal_step(study_id, previous_update_step)
    executor_main(steps=[optimal_step], description=f"DNA Bolinas 4B sweep {suffix}")


if __name__ == "__main__":
    run_sweep()
