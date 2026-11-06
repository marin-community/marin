# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score hero checkpoints with the dropless held-out eval at several eval lengths, without training.

The job restores each checkpoint's parameters in turn, directly onto the expert-collapsed mesh that the
training-time dropless eval uses (`expert` folded into `data`), and runs Paloma and Uncheatable once per
eval length under the local dropless MoE backend. Each checkpoint keeps its own ``qk_mult``. Each length logs
under ``eval_dropless/seq_len_<N>/...`` at the checkpoint's step, so one W&B run holds the whole sweep.
Nothing is written to a checkpoint's tree.
"""

import dataclasses
import functools
import logging
import os
from dataclasses import dataclass
from enum import StrEnum

import click
import jax
import jax.numpy as jnp
import jmp
import levanter.tracker
import numpy as np
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
from levanter.utils.jax_utils import leaf_key_paths
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_namespaced_name
from rigging.filesystem.storage_path import prefix_join

from experiments.datasets.uncheatable import uncheatable_datasets
from experiments.grug.dispatch import dispatch_grug_training_run
from experiments.grug.moe.launch_datakit_moe_mix import _val_component
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
from experiments.grug.moe_hero_ep.optimizer import GrugMoeMuonHConfig
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


class MergeSpace(StrEnum):
    """How a weight merge combines its sources."""

    LINEAR = "linear"
    """The weighted average of the raw parameters."""
    HYPERBALL = "hyperball"
    """The weighted average, then each MuonH / AdamH leaf rescaled to its sources' norm over the optimizer's sphere
    axes, so the merge stays on the hyperball the hero trained on."""


@dataclass(frozen=True)
class EvalCheckpoint:
    """The model to score: one checkpoint directory (``step-N``), or a weighted average of several (a weight merge),
    and the ``qk_mult`` its model trained with. ``sources`` pairs each directory with its weight; the weights sum to 1.
    """

    sources: tuple[tuple[str, float], ...]
    qk_mult: float
    label: str = ""
    """A name for the merge in logs (empty for a single checkpoint)."""
    space: MergeSpace = MergeSpace.LINEAR

    def __post_init__(self):
        if not self.sources:
            raise ValueError("an eval checkpoint needs at least one source")
        if abs(sum(weight for _, weight in self.sources) - 1.0) > 1e-6 or any(w <= 0 for _, w in self.sources):
            raise ValueError(f"merge weights must be positive and sum to 1, got {[w for _, w in self.sources]}")

    @classmethod
    def single(cls, path: str, qk_mult: float) -> "EvalCheckpoint":
        return cls(((path, 1.0),), qk_mult)


def hyperball_axes(path: str, ndim: int, group: str) -> tuple[int, ...] | None:
    """The axes whose joint Frobenius norm the hero's optimizer holds fixed for a leaf, or None for an Adam leaf.

    Mirrors ``_scale_invariant_hyperball_updates`` (MuonH) and ``scale_by_adamh`` (AdamH) in
    ``experiments/grug/moe_hero_ep/optimizer.py`` with the hero's ``ExpertNormalization.ALL_EXPERTS``: a matrix is one
    sphere, a stacked leaf has one sphere per leading (layer) index, and a routed expert bank is one sphere per layer.
    """
    if group == "adam":
        return None
    if ndim <= 2:
        return tuple(range(ndim))
    if group == "muonh" and path.endswith((".expert_mlp.w_gate", ".expert_mlp.w_up", ".expert_mlp.w_down")):
        return tuple(range(ndim - 3, ndim))
    return tuple(range(1, ndim))


@dataclass(frozen=True)
class CheckpointEvalConfig:
    """Dropless held-out eval of checkpoints, in order, at several eval lengths."""

    model: GrugModelConfig
    """The model; each checkpoint replaces only its ``qk_mult``."""
    data: LmDataConfig
    trainer: TrainerConfig
    resources: ResourceConfig
    checkpoints: tuple[EvalCheckpoint, ...]
    seq_lens: tuple[int, ...]
    # Sequences per eval batch at each length are this budget divided by the length.
    eval_tokens_per_batch: int
    dropless_moe_implementation: MoeImplementation = DEFAULT_DROPLESS_MOE_IMPLEMENTATION
    compute_bpb: bool = True
    tensorstore_cache_bytes: int | None = None
    processes_per_task: int = 1

    def __post_init__(self):
        if not self.checkpoints:
            raise ValueError("checkpoints must not be empty")
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


def merged_params(
    model_config: GrugModelConfig, checkpoint: EvalCheckpoint, *, mesh: Mesh, mp: jmp.Policy
) -> tuple[Transformer, int, dict[str, float]]:
    """The weighted average of ``checkpoint.sources``' parameters, the latest source's step, and the merge's norm
    metrics. Sources load one at a time into a float32 running sum, so at most two copies of the parameters are held;
    floating leaves are cast back to their dtype, and any other leaf is taken from the last source.

    The metrics give the merged-to-source norm ratio on the hyperball leaves (min and mean over spheres), measured
    before any ``MergeSpace.HYPERBALL`` rescale: how far a linear average falls inside the spheres.
    """
    totals, target_norms, axes, dtypes, treedef, last_step = None, None, None, None, None, 0
    for path, weight in checkpoint.sources:
        params, step = load_checkpoint_params(model_config, path, mesh=mesh, mp=mp)
        logger.info("Loaded step %d from %s (merge weight %g)", step, path, weight)
        last_step = max(last_step, step)
        leaves, leaf_treedef = jax.tree.flatten(params)
        if treedef is None:
            treedef = leaf_treedef
            dtypes = [x.dtype for x in leaves]
            paths = jax.tree.leaves(leaf_key_paths(params))
            groups = jax.tree.leaves(GrugMoeMuonHConfig().create_mask(params))
            axes = [
                hyperball_axes(p, x.ndim, g) if _is_float(x) else None
                for x, p, g in zip(leaves, paths, groups, strict=True)
            ]
            totals = [jnp.zeros_like(x, dtype=jnp.float32) if _is_float(x) else x for x in leaves]
            target_norms = [None if a is None else 0.0 for a in axes]
        elif leaf_treedef != treedef:
            raise ValueError(f"{path} has a different parameter structure than the merge's first source")
        totals = [_weighted_sum(t, x, weight=weight) for t, x in zip(totals, leaves, strict=True)]
        target_norms = [
            None if a is None else n + weight * _sphere_norm(x, a)
            for n, x, a in zip(target_norms, leaves, axes, strict=True)
        ]
        del params, leaves  # free this source before loading the next
    spheres = [(i, a) for i, a in enumerate(axes) if a is not None]
    ratios = [np.asarray(_sphere_norm(totals[i], a) / target_norms[i]).ravel() for i, a in spheres]
    ratios = np.concatenate(ratios)
    norm_metrics = {"merge/norm_ratio_min": float(ratios.min()), "merge/norm_ratio_mean": float(ratios.mean())}
    logger.info("Merged-to-source norm ratio over %d hyperball leaves: %s", len(spheres), norm_metrics)
    if checkpoint.space == MergeSpace.HYPERBALL:
        for i, a in spheres:
            totals[i] = totals[i] * (target_norms[i] / _sphere_norm(totals[i], a))
    merged = [t.astype(d) if jnp.issubdtype(d, jnp.floating) else t for t, d in zip(totals, dtypes, strict=True)]
    return jax.tree.unflatten(treedef, merged), last_step, norm_metrics


def _sphere_norm(x, axes: tuple[int, ...]):
    """The float32 Frobenius norm of ``x`` over ``axes``, kept as size-1 dims."""
    return jnp.sqrt(jnp.sum(jnp.square(x.astype(jnp.float32)), axis=axes, keepdims=True))


def _weighted_sum(total, x, *, weight: float):
    """``total + weight * x`` in float32 for floating leaves; ``x`` itself for any other leaf."""
    return total.astype(jnp.float32) + weight * x.astype(jnp.float32) if _is_float(x) else x


def _is_float(x) -> bool:
    return jnp.issubdtype(x.dtype, jnp.floating)


def evaluate_checkpoints(config: CheckpointEvalConfig) -> list[tuple[int, dict[str, float]]]:
    """Each checkpoint's step and dropless metrics at every eval length, in ``config.checkpoints`` order (logged at
    each step, which must therefore increase)."""
    # Fold `expert` into `data` as the training-time dropless eval does; the local backend needs expert=1.
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    results: list[tuple[int, dict[str, float]]] = []
    with set_mesh(mesh):
        evaluators = {}
        for seq_len in config.seq_lens:
            eval_cfg = GrugEvalConfig(
                eval_batch_size=config.eval_tokens_per_batch // seq_len,
                compute_bpb=config.compute_bpb,
                eval_current=False,
                eval_ema=False,
                dropless_eval=True,
                dropless_eval_moe_implementation=config.dropless_moe_implementation,
            )
            evaluators[seq_len] = build_tagged_evaluator(
                data_config=config.data,
                max_seq_len=seq_len,
                mesh=mesh,
                eval_cfg=eval_cfg,
                mp=config.trainer.mp,
                model_transform=functools.partial(_to_dropless_local, implementation=config.dropless_moe_implementation),
            )
            if evaluators[seq_len] is None:
                raise ValueError("The data config has no validation sets")
        for checkpoint in config.checkpoints:
            model = dataclasses.replace(config.model, qk_mult=checkpoint.qk_mult)
            params, step, norm_metrics = merged_params(model, checkpoint, mesh=mesh, mp=config.trainer.mp)
            if results and step <= results[-1][0]:
                raise ValueError(f"checkpoint steps must increase: {checkpoint.sources[-1][0]} is step {step}")
            what = f"merge {checkpoint.label}" if checkpoint.label else "parameters"
            logger.info("Scoring %s of step %d (qk_mult %g)", what, step, checkpoint.qk_mult)
            metrics: dict[str, float] = dict(norm_metrics) if len(checkpoint.sources) > 1 else {}
            for seq_len, evaluator in evaluators.items():
                with jax_config.enable_pgle(False):
                    length_metrics = eval_model(evaluator, params, prefix=f"{DROPLESS_EVAL_PREFIX}/seq_len_{seq_len}")
                metrics.update(length_metrics)
            levanter.tracker.log(metrics, step=step)
            results.append((step, metrics))
            del params  # free the parameters before loading the next checkpoint
    return results


def _evaluate_checkpoint_local(config: CheckpointEvalConfig) -> None:
    if config.tensorstore_cache_bytes is not None:
        set_jagged_array_read_cache_bytes(config.tensorstore_cache_bytes)
    config.trainer.initialize()
    levanter.tracker.log_configuration(config)
    evaluate_checkpoints(config)
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
    checkpoints: tuple[EvalCheckpoint, ...],
    qk_mult_long_scale: float,
    seq_lens: tuple[int, ...] = HERO_EVAL_SEQ_LENS,
    version: str | None = None,
) -> ArtifactStep[HeroThroughputResult]:
    """One rack evaluating the hero's model from each of ``checkpoints`` in turn, at ``seq_lens``.

    Only each checkpoint's ``qk_mult`` and the shared ``qk_mult_long_scale`` differ from the hero's 16K model.
    Every eval batch holds one hero rack's 4,194,304 tokens.
    """
    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    model = dataclasses.replace(HERO_16K_MODEL_CONFIG, qk_mult_long_scale=qk_mult_long_scale)
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
        if ctx.is_fingerprint:
            val_components = {item.name: _val_component(ctx.artifact_path(item)) for item in validation}
        else:
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
            checkpoints=checkpoints,
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
@click.option(
    "--checkpoint",
    "checkpoints",
    type=(str, click.FloatRange(min=0, min_open=True)),
    multiple=True,
    help="A checkpoint directory (step-N) and its Q multiplier for all layers; repeat, in increasing step order.",
)
@click.option(
    "--merge-source",
    "merge_sources",
    type=(str, click.FloatRange(min=0, min_open=True)),
    multiple=True,
    help="Instead of --checkpoint: score one weight merge. A checkpoint directory and its relative weight; repeat "
    "(weights are normalized to sum to 1).",
)
@click.option(
    "--merge-qk-mult",
    type=click.FloatRange(min=0, min_open=True),
    default=None,
    help="The Q multiplier for the merged model (required with --merge-source).",
)
@click.option(
    "--merge-space",
    type=click.Choice([space.value for space in MergeSpace]),
    default=MergeSpace.LINEAR.value,
    show_default=True,
    help="linear: average the raw parameters; hyperball: then rescale each MuonH/AdamH sphere to its sources' norm.",
)
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
    run_id: str,
    checkpoints: tuple[tuple[str, float], ...],
    merge_sources: tuple[tuple[str, float], ...],
    merge_qk_mult: float | None,
    merge_space: str,
    qk_mult_long_scale: float,
    seq_lens: tuple[int, ...],
) -> ArtifactStep[HeroThroughputResult]:
    if bool(checkpoints) == bool(merge_sources):
        raise click.UsageError("pass either --checkpoint or --merge-source")
    if merge_sources:
        if merge_qk_mult is None:
            raise click.UsageError("--merge-source needs --merge-qk-mult")
        total = sum(weight for _, weight in merge_sources)
        sources = tuple((path, weight / total) for path, weight in merge_sources)
        evaluated = (EvalCheckpoint(sources, merge_qk_mult, label=run_id, space=MergeSpace(merge_space)),)
    else:
        evaluated = tuple(EvalCheckpoint.single(path, qk) for path, qk in checkpoints)
    return build_checkpoint_eval(
        run_id=run_id,
        checkpoints=evaluated,
        qk_mult_long_scale=qk_mult_long_scale,
        seq_lens=tuple(seq_lens),
    )


if __name__ == "__main__":
    main()
