# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dropless re-eval sweep (issue #8062): re-evaluate a trained run's checkpoints under dropless MoE.

A trained run (EP `fixed_all_to_all`, or FSDP `sonic_cute` chunked) drops routed assignments above its
capacity. This job loads each saved checkpoint of that run one at a time, loads the params into a model
whose only difference is the MoE backend -- `sonic_cute` at one chunk, which computes every assignment
(dropless) -- and runs the same paloma/uncheatable perplexity eval. Logging each checkpoint's eval loss
against its step gives the loss the run *would* have had without training-time drops, isolating the drop
cost. Params, optimizer state, and routing are identical to the source run; only the dispatch changes.

One job per source run, on one GB200 rack (dropless FSDP: expert axis 1, full FSDP over `data`).
"""

import dataclasses
import logging
import os
from dataclasses import dataclass

import click
import jax
import jmp
import levanter.tracker
from fray.cluster import ResourceConfig
from haliax.partitioning import set_mesh
from levanter.eval import eval_model
from levanter.grug.sharding import compact_grug_mesh
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_namespaced_name

from experiments.grug.checkpointing import _scan_checkpoint_root, restore_grug_state_from_checkpoint
from experiments.grug.dispatch import dispatch_grug_training_run
from experiments.grug.moe.launch_datakit_moe_mix import _datakit_data_config, _val_component
from experiments.grug.moe_hero_ep.heuristic import MoeHeuristic
from experiments.grug.moe_hero_ep.launch import (
    DEFAULT_WANDB_PROJECT,
    HERO_MIXED_PRECISION,
    HERO_PROCESSES_PER_TASK,
    HeroThroughputResult,
)
from experiments.grug.moe_hero_ep.small_scale_abl_launch import (
    _VALIDATION,
    EVAL_BATCH_SIZE,
    SMALL_SHAPES,
    TARGETS,
    _root_component,
    _small_model,
)
from experiments.grug.moe_hero_ep.train import (
    GrugEvalConfig,
    GrugRunConfig,
    GrugTrainerConfig,
    _apply_hero_ep_runtime_defaults,
    build_tagged_evaluator,
    initial_state,
)

logger = logging.getLogger(__name__)

SEQ_LEN = 4096
MARIN_CHECKPOINT_PREFIX = "s3://marin-us-east-02a/marin/grug"


@dataclass(frozen=True)
class DroplessEvalConfig:
    """A trained run's checkpoints plus the dropless model to re-evaluate them with."""

    run: GrugRunConfig
    checkpoint_dir: str
    # None evaluates every checkpoint found under ``checkpoint_dir``; a tuple keeps only those steps.
    checkpoint_steps: tuple[int, ...] | None


def run_dropless_eval(config: DroplessEvalConfig) -> None:
    """Entry point: dispatch the checkpoint-sweep eval to a GB200 rack."""
    trainer = config.run.trainer.trainer
    if trainer.id is None:
        raise ValueError("trainer.id must be set before dispatching dropless eval.")
    # Dispatch snapshots os.environ for the child, so apply runtime defaults first (PGLE off for eval).
    _apply_hero_ep_runtime_defaults(inline_watch_enabled=False, enable_pgle=config.run.trainer.enable_pgle)
    dispatch_grug_training_run(
        run_id=trainer.id,
        config=config,
        local_entrypoint=_run_dropless_eval_local,
        resources=config.run.resources,
        processes_per_task=config.run.processes_per_task,
    )


def _run_dropless_eval_local(config: DroplessEvalConfig) -> None:
    """Runs on the gang: build the dropless model once, then load/eval/log each checkpoint in turn."""
    rc = config.run
    trainer = rc.trainer.trainer
    trainer.initialize()
    levanter.tracker.log_configuration(rc)

    optimizer = rc.optimizer.build(trainer.num_train_steps)
    mesh = compact_grug_mesh(
        expert_axis_size=rc.trainer.expert_axis_size,
        replica_axis_size=rc.trainer.replica_axis_size,
    )
    with set_mesh(mesh):

        @jax.jit
        def _init(model_rng):
            # Full state exemplar (params + opt_state + pending_qb_betas): its tree/sharding drives how
            # the checkpoint loads. Same arch and optimizer as the source, so the shapes match exactly.
            return initial_state(rc.model, optimizer=optimizer, mp=trainer.mp, key=model_rng, ema_beta=None)

        state_template = _init(jax.random.PRNGKey(0))
        evaluator = build_tagged_evaluator(
            data_config=rc.data, max_seq_len=rc.model.max_seq_len, mesh=mesh, eval_cfg=rc.eval, mp=trainer.mp
        )
        if evaluator is None:
            raise ValueError("dropless eval has no tagged eval sets")

        # Discover the run's saved checkpoints rather than assume a fixed schedule, so this works for
        # any rung. ``checkpoint_steps``, when set, keeps only those steps.
        found = [(step, path) for step, _, path in _scan_checkpoint_root(config.checkpoint_dir) if step >= 0]
        if config.checkpoint_steps:
            wanted = set(config.checkpoint_steps)
            found = [(step, path) for step, path in found if step in wanted]
        found.sort()
        if not found:
            raise ValueError(f"no checkpoints found under {config.checkpoint_dir}")
        logger.info("dropless eval over %d checkpoints: %s", len(found), [s for s, _ in found])

        for step, ckpt_path in found:
            loaded = restore_grug_state_from_checkpoint(
                state_template,
                checkpoint_search_paths=[ckpt_path],
                load_checkpoint_setting=True,
                mesh=mesh,
                allow_partial=False,
            )
            log_dict = eval_model(evaluator, loaded.params, prefix=rc.eval.prefix)
            levanter.tracker.log(log_dict, step=step)
            logger.info("dropless eval step %d: %s", step, log_dict.get("eval/paloma/macro_loss", log_dict))

    levanter.tracker.current_tracker().finish()


def _dropless_checkpoint_dir(source_run_id: str, version: str) -> str:
    return f"{MARIN_CHECKPOINT_PREFIX}/{source_run_id}/{version}/checkpoints"


def build_dropless_eval_run(
    *,
    source_run_id: str,
    size: str = "d768",
    num_experts: int = 192,
    num_experts_per_token: int = 4,
    use_latent: bool = True,
    intermediate_dim: int | None = None,
    checkpoint_dir: str | None = None,
    checkpoint_steps: tuple[int, ...] | None = None,
    source_version: str = "2026.08.08",
    version: str | None = None,
) -> ArtifactStep[HeroThroughputResult]:
    """Re-eval ``source_run_id``'s checkpoints with the dropless (``sonic_cute`` x1) MoE.

    ``num_experts`` / ``use_latent`` must match the source run's architecture so the checkpoints load;
    only the MoE backend changes. ``checkpoint_dir`` defaults to the source run's own checkpoint prefix.
    """
    if size not in SMALL_SHAPES:
        raise ValueError(f"size must be one of {sorted(SMALL_SHAPES)}, got {size!r}")
    shape = SMALL_SHAPES[size]
    fleet = TARGETS["gb200-rack"]
    ckpt_dir = checkpoint_dir if checkpoint_dir is not None else _dropless_checkpoint_dir(source_run_id, source_version)
    run_id = f"{source_run_id}-dropless-eval"

    # Dropless FSDP: sonic_cute at one chunk (computes every assignment), expert axis 1. Capacity factor
    # is irrelevant to the dropless kernel; the QB estimator is unused at eval (the loaded router bias is
    # what routes), so the cheap top-k default is fine.
    model = _small_model(
        shape,
        1.0,
        fleet.attention_implementation,
        "sonic_cute",
        1,
        SEQ_LEN,
        num_experts,
        num_experts_per_token,
        intermediate_dim if intermediate_dim is not None else (shape.hidden_dim if use_latent else None),
        shape.hidden_dim // 2 if use_latent else None,
    )

    optimizer = dataclasses.replace(
        MoeHeuristic().build_optimizer_config(
            num_train_steps=1, batch_size=EVAL_BATCH_SIZE, hidden_dim=model.hidden_dim, seq_len=SEQ_LEN
        ),
        use_syrk=fleet.use_syrk,
    )
    grug_trainer = GrugTrainerConfig(
        data_seed=None,
        log_every=1,
        ema_beta=None,
        z_loss_weight=1e-4,
        offload_opt_state=False,
        save_checkpoints=False,
        expert_axis_size=1,
        replica_axis_size=1,
        sharding_dump_path=None,
        enable_pgle=False,
    )
    train_resources = ResourceConfig.with_gpu(
        fleet.accelerator,
        count=fleet.gpus_per_node,
        cpu=fleet.cpu,
        ram=fleet.ram,
        disk=fleet.disk,
        replicas=fleet.nodes,
    )
    name = f"grug/{run_id}"
    version = resolve_version(name, version)

    def build_config(ctx: StepContext) -> DroplessEvalConfig:
        trainer = TrainerConfig(
            id=run_id,
            seed=0,
            train_batch_size=EVAL_BATCH_SIZE,
            num_train_steps=1,
            mp=jmp.get_policy(HERO_MIXED_PRECISION),
            tracker=WandbConfig(
                entity="marin-community",
                project=os.environ.get("WANDB_PROJECT") or DEFAULT_WANDB_PROJECT,
                tags=["grug", "moe", "hero", "ep", "dropless-eval", f"shape-{size}", f"source-{source_run_id}"],
                group="moe-hero-ep-dropless-eval",
                name=run_id,
                replicate_path=ctx.output_path,
            ),
            use_explicit_mesh_axes=True,
            require_accelerator=True,
            allow_nondivisible_batch_size=False,
        )
        if ctx.is_fingerprint:
            val_components = {v.name: _val_component(ctx.artifact_path(v)) for v in _VALIDATION}
        else:
            val_components = {v.name: ctx.resolved(v).as_component() for v in _VALIDATION}
        data = _datakit_data_config(
            total_steps=1,
            batch_size=EVAL_BATCH_SIZE,
            max_seq_len=SEQ_LEN,
            enable_simulated_epoching=False,
            val_components=val_components,
        )
        data = dataclasses.replace(
            data, components={n: _root_component(component) for n, component in data.components.items()}
        )
        run = GrugRunConfig(
            model=model,
            data=data,
            resources=ctx.runtime_arg("train_resources"),
            optimizer=optimizer,
            trainer=dataclasses.replace(grug_trainer, trainer=trainer),
            eval=GrugEvalConfig(
                eval_batch_size=EVAL_BATCH_SIZE,
                steps_per_eval=1,
                max_eval_batches=8,  # match the training-time eval subset so the curves are comparable
                eval_current=True,
                eval_ema=False,
            ),
            processes_per_task=HERO_PROCESSES_PER_TASK,
        )
        return DroplessEvalConfig(run=run, checkpoint_dir=ckpt_dir, checkpoint_steps=checkpoint_steps)

    return ArtifactStep(
        name=user_namespaced_name(name, version),
        version=version,
        artifact_type=HeroThroughputResult,
        run=run_dropless_eval,
        build_config=build_config,
        deps=tuple(_VALIDATION),
        runtime_args={"train_resources": train_resources},
    )


@click.command()
@click.option("--source-run-id", required=True, help="Trained run whose checkpoints to re-eval.")
@click.option("--size", type=click.Choice(sorted(SMALL_SHAPES)), default="d768", show_default=True)
@click.option("--num-experts", type=click.IntRange(min=1), default=192, show_default=True, help="Source expert count.")
@click.option(
    "--num-experts-per-token", type=click.IntRange(min=1), default=4, show_default=True, help="Source routed top-k."
)
@click.option("--latent/--no-latent", default=True, show_default=True, help="Whether the source used LatentMoE.")
@click.option(
    "--intermediate-dim", type=click.IntRange(min=1), default=None, help="Source expert width (else hidden/hidden-2)."
)
@click.option("--checkpoint-dir", default=None, help="Override the source checkpoint prefix (else derived).")
@build_options
def main(
    source_run_id: str,
    size: str,
    num_experts: int,
    num_experts_per_token: int,
    latent: bool,
    intermediate_dim: int | None,
    checkpoint_dir: str | None,
) -> ArtifactStep[HeroThroughputResult]:
    return build_dropless_eval_run(
        source_run_id=source_run_id,
        size=size,
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_token,
        use_latent=latent,
        intermediate_dim=intermediate_dim,
        checkpoint_dir=checkpoint_dir,
    )


if __name__ == "__main__":
    main()
