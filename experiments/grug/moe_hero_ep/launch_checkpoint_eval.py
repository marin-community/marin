# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score one hero checkpoint with the dropless held-out eval at several eval lengths, without training.

The job restores only the checkpoint's parameters, directly onto the expert-collapsed mesh that the
training-time dropless eval uses (`expert` folded into `data`), and runs Paloma and Uncheatable once per
eval length under the local dropless MoE backend. Each length logs under
``eval_dropless/seq_len_<N>/...`` at the checkpoint's step. Nothing is written to the checkpoint's tree.
"""

import dataclasses
import functools
import logging
import os
from dataclasses import dataclass

import click
import jax
import jax.numpy as jnp
import jmp
import levanter.tracker
from fray.cluster import ResourceConfig
from haliax.partitioning import set_mesh
from jax._src import config as jax_config
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from levanter.callbacks.profiler import ProfilerConfig
from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import CheckpointerConfig, load_checkpoint
from levanter.data.text.datasets import LmDataConfig
from levanter.eval import eval_model
from levanter.grug.grug_moe import MoeImplementation
from levanter.grug.sharding import compact_grug_mesh
from levanter.store.jagged_array import set_jagged_array_read_cache_bytes
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_namespaced_name
from rigging.filesystem.storage_path import prefix_join

from experiments.datasets.uncheatable import uncheatable_datasets
from experiments.grug.dispatch import dispatch_grug_training_run
from experiments.grug.moe_hero_ep.hero_recipe import (
    DEFAULT_WANDB_PROJECT,
    HERO_EP_NODES,
    HERO_GPUS_PER_NODE,
    HERO_MODEL_CONFIG,
    HERO_NODE_CPU,
    HERO_NODE_DISK,
    HERO_NODE_RAM,
    HERO_PROCESSES_PER_TASK,
    HERO_TENSORSTORE_CACHE_BYTES,
    HERO_TOKENS_PER_RACK,
    HeroThroughputResult,
    hero_trainer_config,
    validation_datasets,
    with_transport_remat_mode,
)
from experiments.grug.moe_hero_ep.model import GrugModelConfig, Transformer
from experiments.grug.moe_hero_ep.train import (
    DEFAULT_DROPLESS_MOE_IMPLEMENTATION,
    GrugEvalConfig,
    _apply_hero_ep_runtime_defaults,
    _to_dropless_local,
    build_tagged_evaluator,
    restore_template_from,
)
from experiments.marin_tokenizer import marin_tokenizer

logger = logging.getLogger(__name__)

DROPLESS_EVAL_PREFIX = "eval_dropless"
# The hero's model at its 16K phase (steps 257553-372000), before #10820 moved it to 65K with
# qk_mult 1.66 and receiver capacity 1.5. Capacity does not enter the dropless eval.
HERO_16K_MODEL_CONFIG = with_transport_remat_mode(
    dataclasses.replace(HERO_MODEL_CONFIG, max_seq_len=16384, capacity_factor=1.15, qk_mult=1.48)
)
HERO_EVAL_SEQ_LENS = (4096, 16384, 65536)
CHECKPOINT_EVAL_MAX_RETRIES = 3


@dataclass(frozen=True)
class CheckpointEvalConfig:
    """Dropless held-out eval of one checkpoint at several eval lengths."""

    model: GrugModelConfig
    data: LmDataConfig
    trainer: TrainerConfig
    resources: ResourceConfig
    checkpoint_path: str
    seq_lens: tuple[int, ...]
    # Sequences per eval batch at each length are this budget divided by the length.
    eval_tokens_per_batch: int
    dropless_moe_implementation: MoeImplementation = DEFAULT_DROPLESS_MOE_IMPLEMENTATION
    compute_bpb: bool = True
    tensorstore_cache_bytes: int | None = None
    processes_per_task: int = 1

    def __post_init__(self):
        if not self.seq_lens or len(set(self.seq_lens)) != len(self.seq_lens):
            raise ValueError(f"seq_lens must be non-empty and distinct, got {self.seq_lens}")
        for seq_len in self.seq_lens:
            if seq_len < 2 or self.eval_tokens_per_batch % seq_len:
                raise ValueError(f"eval length {seq_len} must divide eval_tokens_per_batch={self.eval_tokens_per_batch}")


def load_checkpoint_params(
    model_config: GrugModelConfig, checkpoint_path: str, *, mesh: Mesh, mp: jmp.Policy
) -> tuple[Transformer, int]:
    """Read the checkpoint's parameters and step onto ``mesh``; every other leaf stays unread."""
    params = jax.jit(lambda key: mp.cast_to_param(Transformer.init(model_config, key=key)))(jax.random.PRNGKey(0))
    exemplar = {
        "step": jax.sharding.reshard(jnp.array(0, dtype=jnp.int32), P()),
        "params": restore_template_from(params),
    }
    loaded = load_checkpoint(exemplar, checkpoint_path, axis_mapping=None, mesh=mesh)
    return loaded["params"], int(loaded["step"])


def evaluate_checkpoint(config: CheckpointEvalConfig) -> tuple[int, dict[str, float]]:
    """Return the checkpoint step and the dropless metrics at every eval length."""
    # Fold `expert` into `data` as the training-time dropless eval does; the local backend needs expert=1.
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    metrics: dict[str, float] = {}
    with set_mesh(mesh):
        params, step = load_checkpoint_params(config.model, config.checkpoint_path, mesh=mesh, mp=config.trainer.mp)
        logger.info("Loaded parameters of step %d from %s", step, config.checkpoint_path)
        for seq_len in config.seq_lens:
            eval_cfg = GrugEvalConfig(
                eval_batch_size=config.eval_tokens_per_batch // seq_len,
                compute_bpb=config.compute_bpb,
                eval_current=False,
                eval_ema=False,
                dropless_eval=True,
                dropless_eval_moe_implementation=config.dropless_moe_implementation,
            )
            evaluator = build_tagged_evaluator(
                data_config=config.data,
                max_seq_len=seq_len,
                mesh=mesh,
                eval_cfg=eval_cfg,
                mp=config.trainer.mp,
                model_transform=functools.partial(_to_dropless_local, implementation=config.dropless_moe_implementation),
            )
            if evaluator is None:
                raise ValueError("The data config has no validation sets")
            with jax_config.enable_pgle(False):
                length_metrics = eval_model(evaluator, params, prefix=f"{DROPLESS_EVAL_PREFIX}/seq_len_{seq_len}")
            levanter.tracker.log(length_metrics, step=step)
            metrics.update(length_metrics)
    return step, metrics


def _evaluate_checkpoint_local(config: CheckpointEvalConfig) -> None:
    if config.tensorstore_cache_bytes is not None:
        set_jagged_array_read_cache_bytes(config.tensorstore_cache_bytes)
    config.trainer.initialize()
    levanter.tracker.log_configuration(config)
    evaluate_checkpoint(config)
    levanter.tracker.current_tracker().finish()


def run_checkpoint_eval(config: CheckpointEvalConfig) -> None:
    """Dispatch the checkpoint eval through Fray onto the configured GPU gang."""
    _apply_hero_ep_runtime_defaults(
        inline_watch_enabled=False,
        processes_per_task=config.processes_per_task,
        moe_implementation=config.model.moe_implementation,
        remat_mode=config.model.remat_mode,
    )
    run_id = config.trainer.id
    if run_id is None:
        raise ValueError("trainer.id must be set before dispatching the checkpoint eval")
    dispatch_grug_training_run(
        run_id=run_id,
        config=config,
        local_entrypoint=_evaluate_checkpoint_local,
        resources=config.resources,
        processes_per_task=config.processes_per_task,
        # The eval keeps no state, so a retry after a gang bootstrap failure repeats it from the start.
        max_retries_failure=CHECKPOINT_EVAL_MAX_RETRIES,
    )


def build_checkpoint_eval(
    *,
    run_id: str,
    checkpoint_path: str,
    qk_mult: float,
    qk_mult_long_scale: float,
    seq_lens: tuple[int, ...] = HERO_EVAL_SEQ_LENS,
    version: str | None = None,
) -> ArtifactStep[HeroThroughputResult]:
    """One rack evaluating the hero's 16K-phase model from ``checkpoint_path`` at ``seq_lens``.

    Only ``qk_mult`` and ``qk_mult_long_scale`` differ from the hero's 16K model. Every eval batch holds
    one hero rack's 4,194,304 tokens, one 65K sequence per GPU.
    """
    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    model = dataclasses.replace(HERO_16K_MODEL_CONFIG, qk_mult=qk_mult, qk_mult_long_scale=qk_mult_long_scale)
    resources = ResourceConfig.with_gpu(
        "GB200",
        count=HERO_GPUS_PER_NODE,
        cpu=HERO_NODE_CPU,
        ram=HERO_NODE_RAM,
        disk=HERO_NODE_DISK,
        replicas=HERO_EP_NODES,
    )
    name = f"grug/{run_id}"
    version = resolve_version(name, version)
    validation = [*validation_datasets(), *uncheatable_datasets(tokenizer=marin_tokenizer).values()]
    wandb_project = os.environ.get("WANDB_PROJECT") or DEFAULT_WANDB_PROJECT

    def build_config(ctx: StepContext) -> CheckpointEvalConfig:
        trainer = hero_trainer_config(
            run_id=run_id,
            seed=0,
            train_batch_size=HERO_TOKENS_PER_RACK // model.max_seq_len,
            num_train_steps=1,
            profiler=ProfilerConfig(enabled=False),
            tracker=WandbConfig(
                save_code=False,
                entity="marin-community",
                project=wandb_project,
                tags=["grug", "moe", "hero", "checkpoint-eval", "gb200"],
                group="moe-hero-ep-checkpoint-eval",
                name=run_id,
                replicate_path=ctx.output_path,
            ),
            watch=WatchConfig(interval=0),
            # The eval never saves; the checkpointer only needs a path inside this run's own tree.
            checkpointer=CheckpointerConfig(base_path=prefix_join(ctx.output_path, "checkpoints")),
        )
        val_components = {item.name: ctx.resolved(item).as_component() for item in validation}
        data = LmDataConfig(
            tokenizer=marin_tokenizer,
            cache_dir=None,
            components=val_components,
            train_weights={name: 0.0 for name in val_components},
            auto_build_caches=False,
        )
        return CheckpointEvalConfig(
            model=model,
            data=data,
            trainer=trainer,
            resources=ctx.runtime_arg("resources"),
            checkpoint_path=checkpoint_path,
            seq_lens=seq_lens,
            eval_tokens_per_batch=HERO_TOKENS_PER_RACK,
            tensorstore_cache_bytes=HERO_TENSORSTORE_CACHE_BYTES,
            processes_per_task=HERO_PROCESSES_PER_TASK,
        )

    return ArtifactStep(
        name=user_namespaced_name(name, version),
        version=version,
        artifact_type=HeroThroughputResult,
        run=run_checkpoint_eval,
        build_config=build_config,
        deps=tuple(validation),
        runtime_args={"resources": resources},
    )


@click.command()
@click.option("--run-id", required=True, help="Run identifier for the artifact and W&B names.")
@click.option("--checkpoint", "checkpoint_path", required=True, help="Concrete checkpoint directory (step-N).")
@click.option("--qk-mult", type=click.FloatRange(min=0, min_open=True), required=True, help="Q multiplier, all layers.")
@click.option(
    "--qk-mult-long-scale",
    type=click.FloatRange(min=0, min_open=True),
    default=1.0,
    show_default=True,
    help="Extra multiplier on --qk-mult for the global (full-causal) layers only.",
)
@click.option(
    "--seq-len",
    "seq_lens",
    type=click.IntRange(min=2),
    multiple=True,
    default=HERO_EVAL_SEQ_LENS,
    show_default=True,
    help="Eval length; repeat for several.",
)
@build_options
def main(
    run_id: str, checkpoint_path: str, qk_mult: float, qk_mult_long_scale: float, seq_lens: tuple[int, ...]
) -> ArtifactStep[HeroThroughputResult]:
    return build_checkpoint_eval(
        run_id=run_id,
        checkpoint_path=checkpoint_path,
        qk_mult=qk_mult,
        qk_mult_long_scale=qk_mult_long_scale,
        seq_lens=tuple(seq_lens),
    )


if __name__ == "__main__":
    main()
