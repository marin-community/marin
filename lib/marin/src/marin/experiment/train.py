# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The inline training helper: ``train_lm`` and the small pieces it composes.

``train_lm`` exposes exactly the *experimental decisions* — model, data,
optimizer, token budget, precision, parallelism, evals, checkpoint/export
cadence, init source — and nothing else. Framework invariants (the MoE
compute-mapping, ``allow_nondivisible_batch_size``, the checkpoint heartbeat,
``this_output_path`` wiring) are set here, hidden from the protocol. Runtime
bindings (region, run id, W&B group) are deliberately absent; the runner sets
them at launch.

It does not accept arbitrary ``SimpleTrainConfig`` fields — that would make it a
config object by another name. New knobs are added only when they are genuine
experimental decisions.
"""

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta

import jmp
from fray import ResourceConfig
from haliax.partitioning import ResourceAxis
from levanter.adaptor import NoAdaptorConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import LmDataConfig
from levanter.eval_harness import LmEvalHarnessConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.models.lm_model import LmConfig
from levanter.optim import AdamConfig, OptimizerConfig
from levanter.tracker.wandb import WandbConfig, truncate_wandb_run_name
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig

from marin.evaluation.evaluation_config import convert_to_levanter_task_config
from marin.execution.executor import unwrap_versioned_value
from marin.execution.types import ExecutorStep, this_output_path
from marin.experiment.evals import EvalSuite
from marin.processing.tokenize import TokenizerStep, add_validation_sets_to_mixture, lm_mixture_data_config
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm

# Precision shorthands -> jmp policy strings. Compute in low precision, keep
# master params/optimizer state in f32. "bf16" is the marin default.
_PRECISION_POLICIES = {
    "bf16": "p=f32,c=bfloat16",
    "f32": "p=f32,c=f32",
}

# Wall-clock heartbeat for the rolling resumption checkpoint.
_CHECKPOINT_INTERVALS = {
    "10min": timedelta(minutes=10),
}


@dataclass(frozen=True)
class Parallelism:
    """Model (tensor) parallelism. ``tensor>1`` shards weights across devices."""

    tensor: int = 1


@dataclass(frozen=True)
class WandbTracker:
    """A W&B tracker target. Project + tags are protocol; name/group are runtime."""

    project: str
    tags: tuple[str, ...] = ()


def wandb(*, project: str, tags: Sequence[str] = ()) -> WandbTracker:
    return WandbTracker(project=project, tags=tuple(tags))


def adam(
    *,
    lr: float,
    weight_decay: float | None = None,
    beta1: float | None = None,
    beta2: float | None = None,
    epsilon: float | None = None,
    max_grad_norm: float | None = None,
    warmup: float | None = None,
    decay: float | None = None,
    rewarmup: float | None = None,
    lr_schedule: str | None = None,
    cycle_length: int | list[int] | None = None,
    min_lr_ratio: float | None = None,
    skip_bad_steps: bool = False,
) -> AdamConfig:
    """Build an ``AdamConfig``, filling unset fields with Adam's own defaults."""
    d = AdamConfig()
    return AdamConfig(
        learning_rate=lr,
        weight_decay=weight_decay if weight_decay is not None else d.weight_decay,
        beta1=beta1 if beta1 is not None else d.beta1,
        beta2=beta2 if beta2 is not None else d.beta2,
        epsilon=epsilon if epsilon is not None else d.epsilon,
        max_grad_norm=max_grad_norm if max_grad_norm is not None else d.max_grad_norm,
        warmup=warmup if warmup is not None else d.warmup,
        rewarmup=rewarmup if rewarmup is not None else d.rewarmup,
        decay=decay if decay is not None else d.decay,
        lr_schedule=lr_schedule if lr_schedule is not None else d.lr_schedule,
        cycle_length=cycle_length,
        min_lr_ratio=min_lr_ratio if min_lr_ratio is not None else d.min_lr_ratio,
        skip_bad_steps=skip_bad_steps,
    )


def mixture(
    components: Mapping[str, TokenizerStep],
    weights: Mapping[str, float],
    *,
    validation: Mapping[str, TokenizerStep] | None = None,
) -> LmDataConfig:
    """A training data mixture from named tokenized components and their weights.

    ``validation`` sets are folded in as zero-weight components (eval-only).
    """
    config = lm_mixture_data_config(components=dict(components), weights=dict(weights))
    if validation:
        config = add_validation_sets_to_mixture(config, dict(validation))
    return config


def _steps_for_tokens(tokens: float, train_batch_size: int, train_seq_len: int) -> int:
    return int(tokens) // (train_batch_size * train_seq_len)


def _train_length(train_seq_len: int, model: LmConfig) -> int:
    actual = unwrap_versioned_value(model)
    if train_seq_len > actual.max_seq_len:
        raise ValueError(f"train_seq_len {train_seq_len} exceeds model max_seq_len {actual.max_seq_len}.")
    return train_seq_len


def train_lm(
    *,
    name: str,
    model: LmConfig,
    data: LmDataConfig,
    optimizer: OptimizerConfig,
    train_batch_size: int,
    train_seq_len: int,
    resources: ResourceConfig,
    tracker: WandbTracker,
    tokens: float | None = None,
    num_train_steps: int | None = None,
    precision: str = "bf16",
    parallelism: Parallelism = Parallelism(),
    z_loss: float | None = None,
    evals: EvalSuite | None = None,
    eval_every: int = 1000,
    checkpoint_every: str = "10min",
    export_every: int | None = None,
    hf_export_every: int | None = None,
    init_from: str | None = None,
) -> ExecutorStep:
    """Construct a language-model training step from inline experimental decisions.

    Exactly one of ``tokens`` or ``num_train_steps`` must be given.
    """
    if (tokens is None) == (num_train_steps is None):
        raise ValueError("Specify exactly one of `tokens` or `num_train_steps`.")

    steps = (
        num_train_steps if num_train_steps is not None else _steps_for_tokens(tokens, train_batch_size, train_seq_len)
    )
    length = _train_length(train_seq_len, model)

    harness = LmEvalHarnessConfig(task_spec=convert_to_levanter_task_config(list(evals.tasks))) if evals else None
    harness_steps = evals.every if evals else 10000

    inner_config = TrainLmConfig(
        data=data,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                project=tracker.project,
                name=None,
                tags=list(tracker.tags),
                group=None,
                replicate_path=this_output_path(),
            ),
            mp=jmp.get_policy(_PRECISION_POLICIES[precision]),
            train_batch_size=train_batch_size,
            per_device_parallelism=-1,
            num_train_steps=steps,
            steps_per_eval=eval_every,
            checkpointer=CheckpointerConfig(
                save_interval=_CHECKPOINT_INTERVALS[checkpoint_every],
                keep=[dict(every=export_every)] if export_every is not None else [],
            ),
            model_averaging=None,
            mesh=MeshConfig(
                axes={"replica": 1, "data": -1, "model": parallelism.tensor},
                compute_mapping={
                    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                },
            ),
            allow_partial_checkpoint=False,
            per_device_eval_parallelism=-1,
            max_eval_batches=None,
            allow_nondivisible_batch_size=True,
            quantization=None,
            initialize_from=None,
            use_explicit_mesh_axes=False,
        ),
        initialize_from_checkpoint_path=None,
        initialize_from_hf=init_from or False,
        pad_tokenizer_to_match_model=False,
        z_loss_weight=z_loss,
        train_seq_len=length,
        model=model,
        optimizer=optimizer,
        hf_save_steps=hf_export_every if hf_export_every is not None else export_every,
        hf_generation_eos_token_ids=None,
        data_seed=None,
        eval_harness_steps=harness_steps,
        eval_harness=harness,
        adapter=NoAdaptorConfig(),
    )

    config = TrainLmOnPodConfig(
        train_config=inner_config,
        resources=resources,
        output_path=this_output_path(),
        env_vars=None,
    )

    return ExecutorStep(
        name=os.path.join("checkpoints", truncate_wandb_run_name(name)),
        description=f"Train {name} for {steps} steps * {train_batch_size} batch * {length} seq.",
        fn=run_levanter_train_lm,
        resources=resources,
        config=config,
    )
