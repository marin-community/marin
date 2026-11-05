# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score hero checkpoints with dropless evals at several eval lengths and on MRCR, without training.

The job restores each checkpoint's parameters in turn, directly onto the expert-collapsed mesh that the
training-time dropless eval uses (`expert` folded into `data`). It runs paired MRCR once and Paloma and
Uncheatable once per eval length under the local dropless MoE backend. Each checkpoint keeps its own
``qk_mult``. Each length logs under ``eval_dropless/seq_len_<N>/...`` and MRCR under ``eval/mrcr/...``, at the
checkpoint's step, so one W&B run holds the whole sweep. Nothing is written to a checkpoint's tree. A text
stream's final partial sequence is kept with masked right padding, since at long lengths most Paloma subsets
are shorter than one sequence.

With a merge plan, the job instead scores raw checkpoints and in-memory merges of them in one newest-first pass,
under ``eval_merge/<name>/seq_len_<N>/...`` and ``eval_merge/<name>/mrcr/...``. Merge metrics log at a pass index,
since the checkpoint steps fall.
"""

import dataclasses
import functools
import itertools
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import get_args

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
from levanter.eval import TaggedEvaluator, eval_model
from levanter.grug.attention import ContextAttentionLayout
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

from experiments.datasets.mrcr import MRCR_CONTEXT_CAP, MRCR_DATASET_REVISION, mrcr_dataset
from experiments.datasets.uncheatable import uncheatable_datasets
from experiments.grug.dispatch import dispatch_grug_training_run
from experiments.grug.moe.launch_datakit_moe_mix import _val_component
from experiments.grug.moe_hero_ep.checkpoint_merge import (
    BiasSource,
    MergePlan,
    distance_stats,
    merge_plan_from_json,
    reproject_to_reference,
    running_form,
    weighted_sum,
    with_expert_banks_only,
    with_mixed_bias,
)
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
from experiments.grug.moe_hero_ep.mrcr_eval import MrcrEvalConfig, evaluate_mrcr
from experiments.grug.moe_hero_ep.train import (
    DEFAULT_DROPLESS_MOE_IMPLEMENTATION,
    GrugEvalConfig,
    _apply_hero_ep_runtime_defaults,
    _to_dropless_local,
    _tree_to_memory_kind,
    build_tagged_evaluator,
    restore_template_from,
)
from experiments.marin_tokenizer import marin_tokenizer

logger = logging.getLogger(__name__)

DROPLESS_EVAL_PREFIX = "eval_dropless"
MERGE_EVAL_PREFIX = "eval_merge"
HERO_EVAL_SEQ_LENS = (4096, 16384, 65536)
CHECKPOINT_EVAL_MAX_RETRIES = 3


@dataclass(frozen=True)
class EvalCheckpoint:
    """A checkpoint directory (``step-N``) and the ``qk_mult`` to score it with."""

    path: str
    qk_mult: float


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
    context_axis_size: int = 1
    mrcr: MrcrEvalConfig | None = None
    dropless_moe_implementation: MoeImplementation = DEFAULT_DROPLESS_MOE_IMPLEMENTATION
    compute_bpb: bool = True
    tensorstore_cache_bytes: int | None = None
    processes_per_task: int = 1
    merge: MergePlan | None = None
    """Score this plan's merges instead of ``checkpoints``."""

    def __post_init__(self):
        if bool(self.checkpoints) == (self.merge is not None):
            raise ValueError("Set exactly one of checkpoints and merge")
        if not (self.seq_lens or self.mrcr) or len(set(self.seq_lens)) != len(self.seq_lens):
            raise ValueError(f"seq_lens must be distinct and non-empty without MRCR, got {self.seq_lens}")
        for seq_len in self.seq_lens:
            if seq_len < 2 or self.eval_tokens_per_batch % seq_len:
                raise ValueError(f"eval length {seq_len} must divide eval_tokens_per_batch={self.eval_tokens_per_batch}")
            if seq_len % self.context_axis_size:
                raise ValueError(f"context_axis_size={self.context_axis_size} must divide eval length {seq_len}")


def eval_tokens_per_batch(*, nodes: int, context_axis_size: int, seq_lens: tuple[int, ...]) -> int:
    """Return the tokens in one eval batch on ``nodes``, which keep one rack's tokens per GPU.

    Every length must fill a whole number of sequences on each batch shard, one per ``data`` group.
    """
    if HERO_EP_NODES % nodes:
        raise ValueError(f"nodes={nodes} must divide the rack's {HERO_EP_NODES} nodes")
    tokens = HERO_TOKENS_PER_RACK * nodes // HERO_EP_NODES
    gpus = nodes * HERO_GPUS_PER_NODE
    if gpus % context_axis_size:
        raise ValueError(f"context_axis_size={context_axis_size} must divide {gpus} GPUs")
    batch_shards = gpus // context_axis_size
    for seq_len in seq_lens:
        if tokens % seq_len or (tokens // seq_len) % batch_shards:
            raise ValueError(
                f"eval length {seq_len} does not give whole sequences on {batch_shards} batch shards "
                f"from {tokens} tokens per batch"
            )
    return tokens


def _params_template(model_config: GrugModelConfig, mp: jmp.Policy) -> Transformer:
    params = jax.jit(lambda key: mp.cast_to_param(Transformer.init(model_config, key=key)))(jax.random.PRNGKey(0))
    return restore_template_from(params)


def _load_params(template: Transformer, checkpoint_path: str, *, mesh: Mesh) -> tuple[Transformer, int]:
    exemplar = {"step": jax.sharding.reshard(jnp.array(0, dtype=jnp.int32), P()), "params": template}
    loaded = load_checkpoint(exemplar, checkpoint_path, axis_mapping=None, mesh=mesh)
    return loaded["params"], int(loaded["step"])


def load_checkpoint_params(
    model_config: GrugModelConfig, checkpoint_path: str, *, mesh: Mesh, mp: jmp.Policy
) -> tuple[Transformer, int]:
    """Read the checkpoint's parameters and step onto ``mesh``; every other leaf stays unread."""
    return _load_params(_params_template(model_config, mp), checkpoint_path, mesh=mesh)


def load_router_biases(template: Transformer, checkpoint_path: str, *, mesh: Mesh) -> dict[BiasSource, jax.Array]:
    """Read only the checkpoint's stored bias and its pending bias, in the bias's sign convention."""
    bias = template.stacked_blocks.stacked.mlp.router_bias
    exemplar = {
        "params": {"stacked_blocks": {"stacked": {"mlp": {"router_bias": bias}}}},
        "pending_qb_betas": bias,
    }
    loaded = load_checkpoint(exemplar, checkpoint_path, axis_mapping=None, mesh=mesh)
    return {
        BiasSource.PARAMS: loaded["params"]["stacked_blocks"]["stacked"]["mlp"]["router_bias"],
        BiasSource.PENDING: -loaded["pending_qb_betas"],
    }


def _build_evaluators(config: CheckpointEvalConfig, mesh: Mesh) -> dict[int, TaggedEvaluator]:
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
        evaluators[seq_len] = evaluator
    return evaluators


def _mrcr_metrics(
    params: Transformer,
    config: CheckpointEvalConfig,
    model: GrugModelConfig,
    mesh: Mesh,
    *,
    name: str,
    checkpoint_path: str,
    step: int,
) -> dict[str, float]:
    """Paired MRCR for one scored model, with its artifact under ``name`` in the run's own output."""
    mrcr = config.mrcr
    if mrcr is None:
        raise ValueError("_mrcr_metrics needs an MRCR config")
    return evaluate_mrcr(
        params,
        dataclasses.replace(mrcr, output_path=prefix_join(mrcr.output_path, name)),
        mesh=mesh,
        mp=config.trainer.mp,
        implementation=config.dropless_moe_implementation,
        record={
            "dataset_revision": MRCR_DATASET_REVISION,
            "checkpoint_path": checkpoint_path,
            "checkpoint_step": step,
            "qk_mult": model.qk_mult,
            "qk_mult_long_scale": model.qk_mult_long_scale,
            "log_scaling_n_floor": model.log_scaling_n_floor,
            "log_scaling_alpha": model.log_scaling_alpha,
            "context_axis_size": config.context_axis_size,
            "max_batches": mrcr.max_batches,
        },
    )


def evaluate_checkpoints(config: CheckpointEvalConfig) -> list[tuple[int, dict[str, float]]]:
    """Each checkpoint's step and dropless metrics, in ``config.checkpoints`` order.

    Metrics log at each checkpoint's step, which must therefore increase.
    """
    # Fold `expert` into `data` as the training-time dropless eval does; the local backend needs expert=1.
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1, context_axis_size=config.context_axis_size)
    results: list[tuple[int, dict[str, float]]] = []
    with set_mesh(mesh):
        evaluators = _build_evaluators(config, mesh)
        for checkpoint in config.checkpoints:
            model = dataclasses.replace(config.model, qk_mult=checkpoint.qk_mult)
            params, step = load_checkpoint_params(model, checkpoint.path, mesh=mesh, mp=config.trainer.mp)
            if results and step <= results[-1][0]:
                raise ValueError(f"checkpoint steps must increase: {checkpoint.path} is step {step}")
            logger.info("Loaded parameters of step %d (qk_mult %g) from %s", step, checkpoint.qk_mult, checkpoint.path)
            metrics: dict[str, float] = {}
            if config.mrcr is not None:
                metrics.update(
                    _mrcr_metrics(
                        params, config, model, mesh, name=f"step-{step}", checkpoint_path=checkpoint.path, step=step
                    )
                )
            for seq_len, evaluator in evaluators.items():
                with jax_config.enable_pgle(False):
                    metrics.update(eval_model(evaluator, params, prefix=f"{DROPLESS_EVAL_PREFIX}/seq_len_{seq_len}"))
            levanter.tracker.log(metrics, step=step)
            results.append((step, metrics))
            del params  # free the parameters before loading the next checkpoint
    return results


def evaluate_merge_plan(config: CheckpointEvalConfig) -> tuple[dict[str, dict[str, float]], dict[int, dict[str, float]]]:
    """Return each scored name's metrics and each source step's distance stats.

    One newest-first pass reads every source's parameters once. The newest source, the running sum and any
    candidate's own accumulator stay in host memory, so at most three parameter copies are on a GPU at a time
    and only the scored candidate during its eval.
    """
    plan = config.merge
    if plan is None:
        raise ValueError("evaluate_merge_plan needs a merge plan")
    # As in `evaluate_checkpoints`: `expert` folded into `data`, as the training-time dropless eval does.
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1, context_axis_size=config.context_axis_size)
    read_order = plan.source_steps()
    forms = {candidate.name: running_form(candidate.weights, read_order) for candidate in plan.candidates}
    last_running_index = max((f.last_index for f in forms.values() if f is not None and f.running), default=0)
    last_indices = {
        candidate.name: max(read_order.index(entry.step) for entry in candidate.weights) for candidate in plan.candidates
    }
    log_step = itertools.count()
    scores: dict[str, dict[str, float]] = {}
    overlaps: dict[int, dict[str, float]] = {}

    # Under jit, as the train step moves its offloaded optimizer state; the helper needs the tracer's abstract mesh.
    to_host = jax.jit(lambda tree: _tree_to_memory_kind(tree, "pinned_host"))
    to_device = jax.jit(lambda tree: _tree_to_memory_kind(tree, "device"))

    def score(name: str, params: Transformer, *, step: int) -> None:
        metrics: dict[str, float] = {}
        if config.mrcr is not None:
            mrcr = _mrcr_metrics(
                params, config, model, mesh, name=name, checkpoint_path=plan.source_path(step), step=step
            )
            metrics.update({f"{MERGE_EVAL_PREFIX}/{name}/{key.removeprefix('eval/')}": v for key, v in mrcr.items()})
        for seq_len, evaluator in evaluators.items():
            with jax_config.enable_pgle(False):
                metrics.update(eval_model(evaluator, params, prefix=f"{MERGE_EVAL_PREFIX}/{name}/seq_len_{seq_len}"))
        levanter.tracker.log(metrics, step=next(log_step))
        scores[name] = metrics

    with set_mesh(mesh):
        evaluators = _build_evaluators(config, mesh)
        model = dataclasses.replace(config.model, qk_mult=plan.qk_mult)
        template = _params_template(model, config.trainer.mp)
        biases = {step: load_router_biases(template, plan.source_path(step), mesh=mesh) for step in read_order}
        newest: Transformer | None = None
        running: Transformer | None = None
        accumulators: dict[str, Transformer] = {}
        for index, step in enumerate(read_order):
            source, loaded_step = _load_params(template, plan.source_path(step), mesh=mesh)
            if loaded_step != step:
                raise ValueError(f"{plan.source_path(step)} holds step {loaded_step}")
            logger.info("Loaded merge source %d of %d: step %d", index + 1, len(read_order), step)
            if newest is None:
                newest = running = to_host(source)
            else:
                newest_device = to_device(newest)
                running_device = None if running is None else to_device(running)
                overlaps[step] = distance_stats(source, newest_device, running_device, index)
                levanter.tracker.log(
                    {f"{MERGE_EVAL_PREFIX}/distance/{name}": value for name, value in overlaps[step].items()},
                    step=next(log_step),
                )
                del newest_device
                if running_device is not None:
                    running = to_host(weighted_sum([(1.0, running_device), (1.0, source)]))
                del running_device
                for candidate in plan.candidates:
                    weights = {entry.step: entry.weight for entry in candidate.weights}
                    if forms[candidate.name] is not None or step not in weights:
                        continue
                    terms = [(weights[step], source)]
                    if candidate.name in accumulators:
                        terms.append((1.0, to_device(accumulators[candidate.name])))
                    elif read_order[0] in weights:
                        terms.append((weights[read_order[0]], to_device(newest)))
                    accumulators[candidate.name] = to_host(weighted_sum(terms))
                    del terms
            if step in plan.score_raw_steps:
                score(f"step-{step}", source, step=step)
            del source
            for candidate in plan.candidates:
                if last_indices[candidate.name] != index:
                    continue
                form = forms[candidate.name]
                if form is None:
                    merged = to_device(accumulators.pop(candidate.name))
                else:
                    terms = []
                    if form.newest:
                        terms.append((form.newest, to_device(newest)))
                    if form.running:
                        terms.append((form.running, to_device(running)))
                    merged = weighted_sum(terms)
                    del terms
                if candidate.reproject_to_newest:
                    merged = reproject_to_reference(merged, to_device(newest))
                if candidate.expert_banks_only:
                    merged = with_expert_banks_only(merged, to_device(newest))
                if candidate.bias is not None:
                    merged = with_mixed_bias(merged, candidate.bias, biases)
                logger.info("Scoring merge candidate %s after step %d", candidate.name, step)
                score(candidate.name, merged, step=step)
                del merged
            if index >= last_running_index:
                running = None  # no later candidate reads it
    return scores, overlaps


def _evaluate_checkpoint_local(config: CheckpointEvalConfig) -> None:
    if config.tensorstore_cache_bytes is not None:
        set_jagged_array_read_cache_bytes(config.tensorstore_cache_bytes)
    config.trainer.initialize()
    levanter.tracker.log_configuration(config)
    if config.merge is None:
        evaluate_checkpoints(config)
    else:
        evaluate_merge_plan(config)
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
    merge: MergePlan | None = None,
    log_scaling_n_floor: int | None = None,
    log_scaling_alpha: float = HERO_MODEL_CONFIG.log_scaling_alpha,
    seq_lens: tuple[int, ...] = HERO_EVAL_SEQ_LENS,
    context_axis_size: int = 1,
    nodes: int = HERO_EP_NODES,
    attention_context_layout: ContextAttentionLayout = HERO_MODEL_CONFIG.attention_context_layout,
    mrcr: bool = False,
    mrcr_max_batches: int | None = None,
    version: str | None = None,
) -> ArtifactStep[HeroThroughputResult]:
    """``nodes`` evaluating the hero's model from each of ``checkpoints`` at ``seq_lens`` and, optionally, on MRCR.

    Only each checkpoint's ``qk_mult``, ``qk_mult_long_scale``, ``log_scaling_n_floor`` and ``log_scaling_alpha``
    differ from the hero's model. An eval batch holds a rack's 4,194,304 tokens at the full 16 nodes, and the same
    tokens per GPU on fewer. Capacity does not enter the dropless eval.
    """
    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    longest = max((*seq_lens, MRCR_CONTEXT_CAP) if mrcr else seq_lens)
    tokens_per_batch = eval_tokens_per_batch(
        nodes=nodes,
        context_axis_size=context_axis_size,
        seq_lens=(*seq_lens, MRCR_CONTEXT_CAP) if mrcr else seq_lens,
    )
    model = with_transport_remat_mode(
        dataclasses.replace(
            HERO_MODEL_CONFIG,
            max_seq_len=longest,
            qk_mult_long_scale=qk_mult_long_scale,
            log_scaling_n_floor=log_scaling_n_floor,
            log_scaling_alpha=log_scaling_alpha,
            attention_context_layout=attention_context_layout,
        )
    )
    resources = ResourceConfig.with_gpu(
        "GB200",
        count=HERO_GPUS_PER_NODE,
        cpu=HERO_NODE_CPU,
        ram=HERO_NODE_RAM,
        disk=HERO_NODE_DISK,
        replicas=nodes,
    )
    name = f"eval/mrcr-hero/{run_id}"
    version = resolve_version(name, version)
    validation = [*validation_datasets(), *uncheatable_datasets(tokenizer=marin_tokenizer).values()] if seq_lens else []
    mrcr_data = mrcr_dataset(tokenizer=marin_tokenizer) if mrcr else None
    wandb_project = os.environ.get("WANDB_PROJECT") or DEFAULT_WANDB_PROJECT

    def build_config(ctx: StepContext) -> CheckpointEvalConfig:
        trainer = hero_trainer_config(
            run_id=run_id,
            seed=0,
            train_batch_size=tokens_per_batch // model.max_seq_len,
            context_axis_size=context_axis_size,
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
            validation_remainder="pad",
        )
        mrcr_config = None
        if mrcr_data is not None:
            mrcr_config = MrcrEvalConfig(
                data_path=ctx.artifact_path(mrcr_data),
                output_path=ctx.output_path,
                tokenizer=marin_tokenizer,
                batch_size=tokens_per_batch // MRCR_CONTEXT_CAP,
                max_batches=mrcr_max_batches,
            )
        return CheckpointEvalConfig(
            model=model,
            data=data,
            trainer=trainer,
            resources=ctx.runtime_arg("resources"),
            checkpoints=checkpoints,
            merge=merge,
            seq_lens=seq_lens,
            eval_tokens_per_batch=tokens_per_batch,
            context_axis_size=context_axis_size,
            mrcr=mrcr_config,
            tensorstore_cache_bytes=HERO_TENSORSTORE_CACHE_BYTES,
            processes_per_task=HERO_PROCESSES_PER_TASK,
        )

    return ArtifactStep(
        name=user_namespaced_name(name, version),
        version=version,
        artifact_type=HeroThroughputResult,
        run=run_checkpoint_eval,
        build_config=build_config,
        deps=(*validation, *((mrcr_data,) if mrcr_data is not None else ())),
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
    "--merge",
    "merge_plan_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="A merge plan JSON, such as hero_merge_plan_stage1.json, to score instead of --checkpoint.",
)
@click.option(
    "--qk-mult-long-scale",
    type=click.FloatRange(min=0, min_open=True),
    default=1.0,
    show_default=True,
    help="Extra multiplier on --qk-mult for the global (full-causal) layers only.",
)
@click.option(
    "--log-scaling-n-floor",
    type=click.IntRange(min=1),
    default=None,
    help="Fixed log scaling of global-layer queries past this many keys in their document.",
)
@click.option(
    "--log-scaling-alpha",
    type=click.FloatRange(min=0, min_open=True),
    default=HERO_MODEL_CONFIG.log_scaling_alpha,
    show_default=True,
    help="Slope of the log scaling set by --log-scaling-n-floor.",
)
@click.option(
    "--seq-len",
    "seq_lens",
    type=click.IntRange(min=2),
    multiple=True,
    default=HERO_EVAL_SEQ_LENS,
    show_default=True,
    help="Paloma and Uncheatable eval length; repeat for several.",
)
@click.option("--no-held-out", is_flag=True, help="Skip Paloma and Uncheatable; requires --mrcr.")
@click.option(
    "--context-axis-size",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Sequence shards. 262144-token sequences need 4.",
)
@click.option(
    "--nodes",
    type=click.IntRange(min=1),
    default=HERO_EP_NODES,
    show_default=True,
    help="GB200 nodes. Must divide a rack; the batch shrinks with it, keeping the tokens per GPU.",
)
@click.option(
    "--attention-context-layout",
    type=click.Choice(get_args(ContextAttentionLayout)),
    default=HERO_MODEL_CONFIG.attention_context_layout,
    show_default=True,
    help="FA4 query-row layout under context parallelism; a diagnostic control for layout effects.",
)
@click.option("--mrcr", is_flag=True, help="Also run the paired 262144-token MRCR eval.")
@click.option("--mrcr-max-batches", type=click.IntRange(min=1), default=None, help="Stop MRCR early, as a smoke test.")
@build_options
def main(
    run_id: str,
    checkpoints: tuple[tuple[str, float], ...],
    merge_plan_path: str | None,
    qk_mult_long_scale: float,
    log_scaling_n_floor: int | None,
    log_scaling_alpha: float,
    seq_lens: tuple[int, ...],
    no_held_out: bool,
    context_axis_size: int,
    nodes: int,
    attention_context_layout: ContextAttentionLayout,
    mrcr: bool,
    mrcr_max_batches: int | None,
) -> ArtifactStep[HeroThroughputResult]:
    return build_checkpoint_eval(
        run_id=run_id,
        checkpoints=tuple(EvalCheckpoint(path, qk_mult) for path, qk_mult in checkpoints),
        merge=None if merge_plan_path is None else merge_plan_from_json(Path(merge_plan_path).read_text()),
        qk_mult_long_scale=qk_mult_long_scale,
        log_scaling_n_floor=log_scaling_n_floor,
        log_scaling_alpha=log_scaling_alpha,
        seq_lens=() if no_held_out else tuple(seq_lens),
        context_axis_size=context_axis_size,
        nodes=nodes,
        attention_context_layout=attention_context_layout,
        mrcr=mrcr,
        mrcr_max_batches=mrcr_max_batches,
    )


if __name__ == "__main__":
    main()
