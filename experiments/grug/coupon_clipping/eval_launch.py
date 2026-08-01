# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline Paloma evaluation for coupon-clipping checkpoints."""

import dataclasses
import json
import os
from dataclasses import dataclass

import jax
import jmp
import levanter.tracker
from fray.cluster import ResourceConfig
from haliax.partitioning import set_mesh
from levanter.checkpoint import latest_checkpoint_path, load_checkpoint
from levanter.data.text.datasets import LmDataConfig
from levanter.eval import eval_model
from levanter.grug.sharding import compact_grug_mesh
from levanter.tracker import TrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.data import mixture
from marin.experiment.namespacing import user_namespaced_name
from marin.training.training import LevanterCheckpoint
from rigging.filesystem import open_url, prefix_join

from experiments.datasets.paloma import paloma_datasets
from experiments.grug.coupon_clipping.config import CouponClippingArm, build_model_config
from experiments.grug.coupon_clipping.model import GrugModelConfig, Transformer
from experiments.grug.coupon_clipping.train import GrugEvalConfig, build_tagged_evaluator
from experiments.grug.dispatch import dispatch_grug_training_run
from experiments.marin_tokenizer import marin_tokenizer

_EVAL_RESOURCES = ResourceConfig.with_gpu("GB200", count=4, cpu=32, ram="256g", disk="256g", replicas=4)
_MIXED_PRECISION = "params=float32,compute=bfloat16,output=bfloat16"
_WANDB_PROJECT = "marin"
_WANDB_GROUP = "cc16-7836-paloma"
_DEFAULT_EVAL_BATCH_SIZE = 64
_DEFAULT_MAX_EVAL_BATCHES = 8


@dataclass(frozen=True)
class CouponClippingEvalConfig:
    """Inputs for one checkpoint-only Paloma evaluation."""

    model: GrugModelConfig
    data: LmDataConfig
    checkpoint_path: str
    output_path: str
    run_id: str
    resources: ResourceConfig
    tracker: TrackerConfig
    eval_batch_size: int
    max_eval_batches: int


def _run_coupon_clipping_eval_local(config: CouponClippingEvalConfig) -> None:
    tracker = (
        dataclasses.replace(config.tracker, name=config.run_id)
        if isinstance(config.tracker, WandbConfig)
        else config.tracker
    )
    trainer = TrainerConfig(
        id=config.run_id,
        seed=0,
        train_batch_size=config.eval_batch_size,
        num_train_steps=0,
        mp=jmp.get_policy(_MIXED_PRECISION),
        tracker=tracker,
        use_explicit_mesh_axes=True,
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
    )
    trainer.initialize()
    levanter.tracker.log_configuration(config)

    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    with set_mesh(mesh):

        @jax.jit
        def init_model(key):
            return trainer.mp.cast_to_param(Transformer.init(config.model, key=key))

        model = init_model(jax.random.PRNGKey(trainer.seed))
        concrete_checkpoint_path = latest_checkpoint_path(config.checkpoint_path)
        if concrete_checkpoint_path is None:
            raise FileNotFoundError(f"no checkpoint found under {config.checkpoint_path}")
        loaded = load_checkpoint(
            {"params": model},
            concrete_checkpoint_path,
            axis_mapping=None,
            mesh=mesh,
            allow_partial=False,
        )
        model = trainer.mp.cast_to_compute(loaded["params"])

        evaluator = build_tagged_evaluator(
            data_config=config.data,
            max_seq_len=config.model.max_seq_len,
            mesh=mesh,
            eval_cfg=GrugEvalConfig(
                eval_batch_size=config.eval_batch_size,
                steps_per_eval=None,
                max_eval_batches=config.max_eval_batches,
                eval_current=True,
                eval_ema=False,
                compute_bpb=True,
            ),
        )
        if evaluator is None:
            raise ValueError("Paloma evaluation has no validation datasets")
        metrics = eval_model(evaluator, model, prefix="eval")
        levanter.tracker.log(metrics, step=0)

    if jax.process_index() == 0:
        metrics_path = prefix_join(config.output_path, "metrics.json")
        with open_url(metrics_path, "w") as metrics_file:
            json.dump(metrics, metrics_file, default=float, indent=2, sort_keys=True)
    levanter.tracker.current_tracker().finish()


def run_coupon_clipping_eval(config: CouponClippingEvalConfig) -> None:
    """Dispatch a checkpoint-only Paloma evaluation through Fray."""
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")
    dispatch_grug_training_run(
        run_id=config.run_id,
        config=config,
        local_entrypoint=_run_coupon_clipping_eval_local,
        resources=config.resources,
        processes_per_task=1,
    )


def build_paloma_eval(
    checkpoint: ArtifactStep[LevanterCheckpoint],
    *,
    label: str,
    version: str | None = None,
    eval_batch_size: int = _DEFAULT_EVAL_BATCH_SIZE,
    max_eval_batches: int = _DEFAULT_MAX_EVAL_BATCHES,
) -> ArtifactStep[Artifact]:
    """Build a bounded offline Paloma evaluation for a full-size checkpoint."""
    if eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be positive")
    if max_eval_batches <= 0:
        raise ValueError("max_eval_batches must be positive")

    model = build_model_config(CouponClippingArm.C0_P0)
    validation = tuple(paloma_datasets(tokenizer=marin_tokenizer).values())
    step_name = f"grug/coupon-clipping/eval/paloma-{label}"
    resolved_version = resolve_version(step_name, version)
    run_id = f"cc16-paloma-{label}"

    def build_config(ctx: StepContext) -> CouponClippingEvalConfig:
        return CouponClippingEvalConfig(
            model=model,
            data=mixture(ctx, {}, validation=validation),
            checkpoint_path=prefix_join(ctx.artifact_path(checkpoint), "checkpoints"),
            output_path=ctx.output_path,
            run_id=run_id,
            resources=ctx.runtime_arg("eval_resources"),
            tracker=WandbConfig(
                project=_WANDB_PROJECT,
                tags=["grug", "moe", "coupon-clipping", "gb200", "paloma", label],
                group=_WANDB_GROUP,
                name=None,
                replicate_path=ctx.output_path,
            ),
            eval_batch_size=eval_batch_size,
            max_eval_batches=max_eval_batches,
        )

    return ArtifactStep(
        name=user_namespaced_name(step_name, resolved_version),
        version=resolved_version,
        artifact_type=Artifact,
        run=run_coupon_clipping_eval,
        build_config=build_config,
        deps=(checkpoint, *validation),
        runtime_args={"eval_resources": _EVAL_RESOURCES},
    )
