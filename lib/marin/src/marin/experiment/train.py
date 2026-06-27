# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A generic language-model training assembler.

:func:`train_lm` turns the *meaningful* decisions of a training run — the model, the
optimizer, the data, the token budget, the regularization, the evals — into a lazy
:class:`~marin.execution.lazy.Checkpoint`. Every one of those is a required argument:
the helper defaults none of them, so reading the call shows the whole experiment. What
it *does* own is the mechanical marin-on-TPU plumbing that is identical across runs and
carries no experiment meaning: the data-parallel mesh and token ``compute_mapping``,
the rolling resumption checkpointer, the eval-harness wiring, the WandB replication
path, and the Fray dispatch of the training job. That split is this design's
identity-vs-execution line — *what is computed* is the caller's, *how/where it runs* is
the library's.

This is deliberately **not** a ``default_train``: it bakes in no optimizer, no mixture,
no default eval suite, no learning rate. It only removes boilerplate.
"""

from collections.abc import Callable, Sequence
from datetime import timedelta

import jmp
from fray.types import ResourceConfig
from haliax.partitioning import ResourceAxis
from levanter.adaptor import NoAdaptorConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import LmDataConfig
from levanter.eval_harness import LmEvalHarnessConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.models.lm_model import LmConfig
from levanter.optim import OptimizerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig

from marin.evaluation.evaluation_config import convert_to_levanter_task_config
from marin.execution.lazy import Artifact, Checkpoint, Recipe, RunContext
from marin.execution.remote import remote
from marin.experiment.evals import EvalSuite
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm

DEFAULT_VERSION = "v1"

# Compute in bf16, keep master params and optimizer state in f32. The universal marin
# precision policy; it bears identity (it changes numerics), so overriding it is a
# deliberate experiment, but it is the same across essentially every marin LM run.
MARIN_PRECISION = "p=f32,c=bfloat16"

# The marin token axis maps onto the data-parallel mesh. This is hardware plumbing, not
# an experiment choice: it says nothing about what is computed, only how the sequence
# axis is laid out across the pod.
_TOKEN_AXES = (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)

# The checkpoint Levanter writes under a run's output, relative to ``ctx.out``.
_CHECKPOINTS_SUBDIR = "checkpoints"

# Rolling resumption checkpoint cadence. Operational (it governs recovery, not the
# trained model), so it is not an experiment knob.
_RESUMPTION_INTERVAL = timedelta(minutes=10)


def _marin_mesh(tensor_parallel_size: int) -> MeshConfig:
    """The standard marin training mesh: data parallel, optional tensor sharding.

    ``model`` is the tensor-parallel width (1 = no sharding); ``data`` absorbs the rest
    of the pod. The token axes ride the replica/data axes the marin path expects.
    """
    return MeshConfig(
        axes={"replica": 1, "data": -1, "model": tensor_parallel_size},
        compute_mapping={"token": _TOKEN_AXES, "token_repeat": _TOKEN_AXES},
    )


def _train_job(pod_config: TrainLmOnPodConfig) -> None:
    """Dispatch the assembled config as its own Fray training job."""
    remote(run_levanter_train_lm, resources=pod_config.resources)(pod_config)


def train_lm(
    *,
    name: str,
    model: LmConfig,
    optimizer: OptimizerConfig,
    data: Callable[[RunContext], LmDataConfig],
    deps: Sequence[Artifact],
    batch_size: int,
    seq_len: int,
    num_train_steps: int,
    z_loss_weight: float | None,
    evals: EvalSuite | None,
    resources: ResourceConfig,
    init_from: Checkpoint | None = None,
    version: str = DEFAULT_VERSION,
    mp: str = MARIN_PRECISION,
    tensor_parallel_size: int = 1,
    steps_per_eval: int = 1000,
    wandb_project: str = "marin",
    wandb_group: str | None = None,
    run_id: str | None = None,
    tags: Sequence[str] = (),
    env_vars: dict[str, str] | None = None,
) -> Checkpoint:
    """Assemble a language-model training run as a lazy :class:`Checkpoint`.

    The required arguments are the run's identity-bearing decisions; the helper defaults
    none of them. ``data`` is a builder called with the run's :class:`RunContext` (use it
    to assemble a :func:`~marin.experiment.data.mixture` from dataset handles); ``deps``
    are the handles that builder consumes, so they materialize first. ``evals=None``
    opts out of harness evals explicitly — there is no implicit default suite.

    The remaining parameters are execution choices that do not define the experiment:
    ``mp`` (the standard marin precision, identity-bearing but universal),
    ``tensor_parallel_size`` (model sharding width), eval/checkpoint cadence, tracker
    metadata, and ``resources`` (the TPU the job is dispatched onto — a run-arg, so it
    never enters the checkpoint's fingerprint). ``init_from`` chains this run onto another
    checkpoint (it becomes a dep and seeds ``initialize_from_checkpoint_path``).
    """
    harness = (
        LmEvalHarnessConfig(task_spec=convert_to_levanter_task_config(list(evals.tasks))) if evals is not None else None
    )
    all_deps = (*deps, init_from) if init_from is not None else tuple(deps)

    def build_config(ctx: RunContext) -> TrainLmOnPodConfig:
        init_path = f"{ctx.path(init_from)}/{_CHECKPOINTS_SUBDIR}" if init_from is not None else None
        inner = TrainLmConfig(
            data=data(ctx),
            trainer=TrainerConfig(
                id=run_id,
                tracker=WandbConfig(
                    project=wandb_project,
                    name=run_id,
                    tags=[*tags],
                    group=wandb_group,
                    # Mirror metrics next to the run's output so they outlive the job.
                    replicate_path=ctx.out,
                ),
                mp=jmp.get_policy(mp),
                train_batch_size=batch_size,
                per_device_parallelism=-1,
                num_train_steps=num_train_steps,
                steps_per_eval=steps_per_eval,
                checkpointer=CheckpointerConfig(save_interval=_RESUMPTION_INTERVAL, keep=[]),
                mesh=_marin_mesh(tensor_parallel_size),
                per_device_eval_parallelism=-1,
                allow_nondivisible_batch_size=True,
            ),
            model=model,
            optimizer=optimizer,
            z_loss_weight=z_loss_weight,
            train_seq_len=seq_len,
            initialize_from_checkpoint_path=init_path,
            eval_harness=harness,
            eval_harness_steps=evals.every if evals is not None else None,
            adapter=NoAdaptorConfig(),
        )
        return TrainLmOnPodConfig(
            train_config=inner,
            resources=ctx.run_arg("train_resources"),
            output_path=ctx.out,
            env_vars=env_vars,
        )

    return Checkpoint(
        name=name,
        version=version,
        recipe=Recipe(
            fn=_train_job,
            build_config=build_config,
            deps=all_deps,
            run_args={"train_resources": resources},
        ),
    )
