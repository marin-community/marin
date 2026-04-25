# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Bolinas DNA mixing sweep at the 2B scale.

See https://github.com/Open-Athena/bolinas-dna/issues/135 for full context.

Compares CDS / upstream / downstream training mixture weights with all other
hyperparameters held to the 2B reference run
``dna-bolinas-scaling-v0.5-h2432-p2B-f5f484`` (AdamH at h=2432, ~2.27B params).

Each active component is capped at 20,501,856 examples (the smallest component,
``downstream``); each run trains for one effective epoch over its active
components. Tokenization names use the ``-5149`` suffix to create fresh cache
keys for the post-issue-5149 tokenization fix.

Environment variables:
    SWEEP_MIX_NAMES   CSV of mix names to run (default: all in ``MIX_CONFIGS``).
    WARMUP_MODE       ``yes``/``no`` (default ``no``); when ``yes`` each run
                      is truncated to ``WARMUP_NUM_TRAIN_STEPS`` steps with
                      ``WARMUP_EVALS_PER_RUN`` evals.
"""

import logging
import os
from dataclasses import dataclass
from datetime import timedelta
from functools import lru_cache

import jmp
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import DNALmDatasetFormat
from levanter.eval_harness import LmEvalHarnessConfig
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim import AdamHConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig

from experiments.defaults import default_tokenize
from experiments.dna.defaults import dna_effective_seq_len
from experiments.evals.task_configs import TRAITGYM_MENDELIAN_V2_255, convert_to_levanter_task_config
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.execution.remote import remote
from marin.processing.tokenize import lm_mixture_data_config
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm

# =============================================================================
# Constants
# =============================================================================

VERSION = "v0.2"
TOKENIZER = "bolinas-dna/tokenizer-char-bos"
DNA_BASE_SEQ_LEN = 255  # bp (256 - 1 for BOS)

# Smallest training component (`downstream`) has ~20.5M examples; cap each
# active component at this size for one effective epoch per component.
# See https://github.com/Open-Athena/bolinas-dna/issues/109.
MAX_EXAMPLES_PER_COMPONENT = 20_501_856

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

# Tokenization name template — the `-5149` suffix forces a fresh cache key for
# the post-fix tokenization (https://github.com/marin-community/marin/issues/5149).
TOKENIZE_NAME = "bolinas-v5-{key}-char-bos-5149"

# Training data masks lowercase positions to 1% loss weight (consistent across
# experiments; the validation default below mirrors this for comparability).
TRAIN_FORMAT = DNALmDatasetFormat(lowercase_weight=0.01)

# Architecture: Qwen3 ~2.27B params at vocab_size=7. Matches the 2B reference run
# https://wandb.ai/eric-czech/marin/runs/dna-bolinas-scaling-v0.5-h2432-p2B-f5f484
MODEL_HIDDEN_DIM = 2432
MODEL_INTERMEDIATE_DIM = 9728  # = hidden * 4
MODEL_NUM_HEADS = 19  # = hidden // 128 (head_dim = 128)
MODEL_NUM_KV_HEADS = 19
MODEL_NUM_LAYERS = 24
MODEL_INITIALIZER_RANGE = 0.02

# Resources & batching.
BATCH_SIZE = 4096
PER_DEVICE_PARALLELISM = 512
TPU_TYPES: tuple[str, ...] = ("v5p-8",)

# AdamH optimizer hyperparameters — derived by re-running CompletedAdamHHeuristic
# at (B=BATCH_SIZE, T=331_122_738 * seq_len) with the v0.6 reference Vizier optima
# as base hparams. The 2B reference run used B=1536; bumping to B=4096 here scales
# lr, adam_lr, beta2, and epsilon per the heuristic's CompletedP-style formulas.
# Reference run anchor: https://wandb.ai/eric-czech/marin/runs/dna-bolinas-scaling-v0.5-h2432-p2B-f5f484
LEARNING_RATE = 0.002704351455928664
ADAM_LR = 0.0025670015091924167
BETA1 = 0.6675603345321236
BETA2 = 0.9758186981611812
EPSILON = 1.1645938068047589e-14
MAX_GRAD_NORM = 0.9951880136348765
WEIGHT_DECAY = 0.1
WARMUP_FRACTION = 0.1
DECAY_FRACTION = 0.2
LR_SCHEDULE = "linear"
MIN_LR_RATIO = 0.0
NESTEROV = False
Z_LOSS_WEIGHT = 4.312883184368223e-06

# Eval cadence and checkpoint policy (10 evals + 1 permanent checkpoint per run).
EVALS_PER_RUN = 10
CHECKPOINTS_PER_RUN = 3
CHECKPOINT_TIME_INTERVAL = timedelta(hours=1)

# Warmup mode: smoke-test the full pipeline (LR schedule, eval cadence, harness)
# in ~minutes. Activated via WARMUP_MODE=yes.
WARMUP_NUM_TRAIN_STEPS = 100
WARMUP_EVALS_PER_RUN = 3

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

    Regions absent from `weights` (or with weight 0) are omitted from the
    mixture entirely — neither tokenized nor sampled. Validation datasets are
    always tokenized and evaluated regardless of training weights.
    """

    name: str
    weights: dict[str, float]

    def __post_init__(self):
        unknown = set(self.weights) - set(TRAIN_DATASETS)
        if unknown:
            raise ValueError(f"{self.name}: unknown regions {unknown}; expected subset of {set(TRAIN_DATASETS)}")
        if not any(w > 0 for w in self.weights.values()):
            raise ValueError(f"{self.name}: at least one weight must be > 0")

    @property
    def active_regions(self) -> tuple[str, ...]:
        return tuple(r for r, w in self.weights.items() if w > 0)


MIX_CONFIGS: tuple[MixConfig, ...] = (
    MixConfig(name="uniform", weights={"cds": 1 / 3, "upstream": 1 / 3, "downstream": 1 / 3}),
    MixConfig(name="cds_only", weights={"cds": 1.0}),
    MixConfig(name="upstream_only", weights={"upstream": 1.0}),
    MixConfig(name="downstream_only", weights={"downstream": 1.0}),
)


# =============================================================================
# Validation specs
# =============================================================================


@dataclass(frozen=True)
class ValSpec:
    """One validation tokenization variant applied to every region.

    `suffix` is appended to each region key (with an underscore) when non-empty;
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


def _warmup_mode() -> bool:
    value = os.getenv("WARMUP_MODE", "no").lower()
    if value not in ("yes", "no"):
        raise ValueError(f"WARMUP_MODE must be 'yes' or 'no', got {value!r}")
    return value == "yes"


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


@lru_cache(maxsize=1)
def _model_seq_len() -> int:
    return dna_effective_seq_len(DNA_BASE_SEQ_LEN, TOKENIZER)


def _tokenize(key: str, dataset: str, dataset_format: DNALmDatasetFormat) -> ExecutorStep:
    return default_tokenize(
        name=TOKENIZE_NAME.format(key=key),
        dataset=dataset,
        tokenizer=TOKENIZER,
        format=dataset_format,
    )


def _max_batches_per_component() -> int:
    return MAX_EXAMPLES_PER_COMPONENT // BATCH_SIZE


def _num_train_steps(mix: MixConfig) -> int:
    """One full pass at the per-component cap, summed across active regions.

    In warmup mode the run is truncated to ``WARMUP_NUM_TRAIN_STEPS``; the LR
    schedule and eval cadence (both derived from this value) compress together.
    """
    if _warmup_mode():
        return WARMUP_NUM_TRAIN_STEPS
    return len(mix.active_regions) * _max_batches_per_component()


def _steps_per_eval(num_train_steps: int) -> int:
    evals = WARMUP_EVALS_PER_RUN if _warmup_mode() else EVALS_PER_RUN
    return max(1, num_train_steps // evals)


def _build_data_mixture(mix: MixConfig):
    """Tokenize active train regions + cross-product of validation regions x specs.

    Train regions with weight 0 are omitted entirely. Each validation region is
    tokenized once per ValSpec; specs with empty `suffix` keep the original key.
    """
    components = {region: _tokenize(region, TRAIN_DATASETS[region], TRAIN_FORMAT) for region in mix.active_regions}
    for region_key, dataset in VALIDATION_DATASETS.items():
        for spec in VAL_SPECS:
            key = f"{region_key}_{spec.suffix}" if spec.suffix else region_key
            components[key] = _tokenize(key, dataset, spec.format)
    train_weights = {region: mix.weights[region] for region in mix.active_regions}
    cap = _max_batches_per_component()
    return lm_mixture_data_config(
        components=components,
        weights=train_weights,
        max_train_batches={region: cap for region in mix.active_regions},
    )


def _build_model_config() -> Qwen3Config:
    return Qwen3Config(
        hidden_dim=MODEL_HIDDEN_DIM,
        intermediate_dim=MODEL_INTERMEDIATE_DIM,
        num_layers=MODEL_NUM_LAYERS,
        num_heads=MODEL_NUM_HEADS,
        num_kv_heads=MODEL_NUM_KV_HEADS,
        max_seq_len=_model_seq_len(),
        initializer_range=MODEL_INITIALIZER_RANGE,
        rope=Llama3RotaryEmbeddingsConfig(),
    )


def _build_optimizer() -> AdamHConfig:
    return AdamHConfig(
        learning_rate=LEARNING_RATE,
        adam_lr=ADAM_LR,
        beta1=BETA1,
        beta2=BETA2,
        epsilon=EPSILON,
        max_grad_norm=MAX_GRAD_NORM,
        weight_decay=WEIGHT_DECAY,
        warmup=WARMUP_FRACTION,
        decay=DECAY_FRACTION,
        lr_schedule=LR_SCHEDULE,
        min_lr_ratio=MIN_LR_RATIO,
        nesterov=NESTEROV,
    )


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
    warmup_suffix = "-warmup" if _warmup_mode() else ""
    run_name = f"dna-bolinas-mix-{VERSION}{warmup_suffix}-i{index}-{mix.name}"
    tags = ("sweep", "dna", "bolinas", "mix", VERSION, f"mix={mix.name}", f"i={index}")
    if _warmup_mode():
        tags = (*tags, "warmup")

    inner = TrainLmConfig(
        data=_build_data_mixture(mix),
        model=_build_model_config(),
        train_seq_len=_model_seq_len(),
        z_loss_weight=Z_LOSS_WEIGHT,
        optimizer=_build_optimizer(),
        eval_harness=_eval_harness_config(),
        eval_harness_steps=steps_per_eval,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                project=WANDB_PROJECT,
                tags=list(tags),
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
        resources=ResourceConfig.with_tpu(TPU_TYPES, ram="300g"),
        output_path=this_output_path(),
    )
    return ExecutorStep(
        name=os.path.join("checkpoints", run_name),
        fn=remote(run_levanter_train_lm, resources=ResourceConfig.with_cpu()),
        config=pod_config,
    )


def main():
    selected = _selected_mix_configs()
    # Preserve original MIX_CONFIGS indices so run names stay stable across SWEEP_MIX_NAMES filters.
    index_by_name = {c.name: i for i, c in enumerate(MIX_CONFIGS)}
    steps = [_build_train_step(index_by_name[mix.name], mix) for mix in selected]
    executor_main(steps=steps, description=f"DNA Bolinas mixing sweep {VERSION}")


if __name__ == "__main__":
    main()
