# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test-time depth-scaling eval harness for the re-entrant grug experiments.

E3 trains one weight-tied core whose loop count is sampled per step from a small
set (e.g. {2, 4, 8}). The headline question is whether looping the SAME trained
checkpoint MORE times at eval time keeps lowering loss. Because the parameters
are weight-tied and independent of the loop count, we restore the checkpoint
once and re-evaluate it at several recurrence depths R by swapping only the
(static) ``recurrence_steps`` on the model config -- ``Transformer.__call__``
applies the shared core that many times.

This file is eval-only: it loads a checkpoint, builds the evaluator once, sweeps
R, and records the validation/paloma loss at each R. It does not train and does
not modify the model.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import os
from dataclasses import dataclass, field
from datetime import timedelta

import jax
import jmp
import levanter.tracker
from fray.cluster import ResourceConfig
from haliax.partitioning import set_mesh
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import LmDataConfig
from levanter.eval import EvalResult, TaggedEvaluator
from levanter.grug.sharding import compact_grug_mesh
from levanter.optim import OptimizerConfig
from levanter.tracker import TrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned
from marin.training.training import temporary_checkpoint_base_path

from experiments.grug.checkpointing import restore_grug_state_from_checkpoint
from experiments.grug.dispatch import dispatch_grug_training_run
from experiments.grug.reentrant.launch import (
    _E3_MODEL,
    _E6_MODEL,
    _REENTRANT_RESOURCES,
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    _baseline_optimizer,
    _resolve_run_id,
    env_int,
)
from experiments.grug.reentrant.model import GrugModelConfig, Transformer
from experiments.grug.reentrant.train import (
    GrugEvalConfig,
    GrugRunConfig,
    GrugTrainerConfig,
    build_tagged_evaluator,
    initial_state,
)

logger = logging.getLogger(__name__)

# wandb x-axis / per-R metric prefixes for the depth sweep.
_SWEEP_PREFIX = "sweep"
_DEFAULT_RECURRENCE_VALUES: tuple[int, ...] = (2, 4, 8, 16, 32)

# Which model config to restore the checkpoint with. E4's anytime supervision adds
# no params, so its checkpoint shares E3's param tree (restore with E3, no readout
# overhead). E64 shares E6's tree (depth-conditioned router bias present, anytime
# off for eval). Keyed by SWEEP_MODEL (default e3).
_SWEEP_MODELS = {"e3": _E3_MODEL, "e4": _E3_MODEL, "e6": _E6_MODEL, "e64": _E6_MODEL}


def _resolve_sweep_model() -> GrugModelConfig:
    name = os.environ.get("SWEEP_MODEL", "e3").strip()
    if name not in _SWEEP_MODELS:
        raise ValueError(f"SWEEP_MODEL must be one of {sorted(_SWEEP_MODELS)}, got {name!r}")
    return _SWEEP_MODELS[name]


# A per-tag macro loss whose tag name contains this substring is surfaced as the
# headline paloma loss. The default validation sets carry a "paloma" parent tag.
_PALOMA_TAG_SUBSTRING = "paloma"


def _model_at_depth(model: Transformer, recurrence_steps: int) -> Transformer:
    """Return a copy of ``model`` whose forward loops the core ``recurrence_steps`` times.

    The params are weight-tied and independent of the loop count, so we only swap
    the static config. ``randomize_recurrence`` is forced off (eval is a single
    fixed depth, not a per-step sample) and ``recurrence_choices`` cleared so the
    config is a valid plain re-entrant model at depth R.
    """
    eval_cfg = dataclasses.replace(
        model.config,
        recurrence_steps=recurrence_steps,
        randomize_recurrence=False,
        recurrence_choices=(),
    )
    return dataclasses.replace(model, config=eval_cfg)


def evaluate_at_depths(
    model: Transformer,
    evaluator: TaggedEvaluator,
    recurrence_values: tuple[int, ...],
) -> dict[int, EvalResult]:
    """Evaluate one weight-tied model at several recurrence depths.

    For each R in ``recurrence_values`` we build a model variant that loops its
    shared core R times (swapping only the static config) and run the SAME
    evaluator. Returns R -> the evaluator's ``EvalResult`` (macro/micro losses and
    per-tag breakdown). Pure: no logging or I/O, so the multi-R plumbing is
    testable on a tiny model without a checkpoint.
    """
    results: dict[int, EvalResult] = {}
    for recurrence_steps in recurrence_values:
        model_at_r = _model_at_depth(model, recurrence_steps)
        results[recurrence_steps] = evaluator.evaluate(model_at_r)
    return results


def _paloma_macro_loss(result: EvalResult) -> float | None:
    """Extract the paloma parent-tag macro loss from an eval result, if present."""
    for tag, loss in result.tag_macro_losses.items():
        if _PALOMA_TAG_SUBSTRING in tag.lower():
            return loss
    return None


def _log_sweep_result(recurrence_steps: int, result: EvalResult) -> None:
    """Log one depth's losses to wandb, keyed both by step=R and by an explicit per-R key."""
    metrics: dict[str, float] = {
        f"{_SWEEP_PREFIX}/recurrence_steps": float(recurrence_steps),
        f"{_SWEEP_PREFIX}/macro_loss": result.macro_avg_loss,
        f"{_SWEEP_PREFIX}/micro_loss": result.micro_avg_loss,
    }
    if result.macro_bpb is not None:
        metrics[f"{_SWEEP_PREFIX}/macro_bpb"] = result.macro_bpb
    paloma_loss = _paloma_macro_loss(result)
    if paloma_loss is not None:
        metrics[f"{_SWEEP_PREFIX}/paloma_macro_loss"] = paloma_loss
    for tag, loss in result.tag_macro_losses.items():
        metrics[f"{_SWEEP_PREFIX}/tag/{tag}/macro_loss"] = loss

    # step=R gives a wandb x-axis of recurrence depth for sweep/* line plots.
    levanter.tracker.log(metrics, step=recurrence_steps)
    # Also pin each depth's headline loss under a flat key so the run summary keeps
    # the whole curve even though successive log() calls share no monotonic step.
    summary: dict[str, float] = {f"{_SWEEP_PREFIX}/macro_loss_at_R{recurrence_steps}": result.macro_avg_loss}
    if paloma_loss is not None:
        summary[f"{_SWEEP_PREFIX}/paloma_macro_loss_at_R{recurrence_steps}"] = paloma_loss
    levanter.tracker.log_summary(summary)


def _print_sweep_table(results: dict[int, EvalResult]) -> None:
    """Print a clean stdout table of R vs macro loss (and paloma if present)."""
    lines = ["", "depth-scaling eval sweep", f"{'R':>6}  {'macro_loss':>12}  {'paloma_loss':>12}"]
    for recurrence_steps in sorted(results):
        result = results[recurrence_steps]
        paloma_loss = _paloma_macro_loss(result)
        paloma_str = f"{paloma_loss:>12.4f}" if paloma_loss is not None else f"{'n/a':>12}"
        lines.append(f"{recurrence_steps:>6}  {result.macro_avg_loss:>12.4f}  {paloma_str}")
    print("\n".join(lines))


def _run_eval_sweep_local(
    config: GrugRunConfig,
    *,
    checkpoint_path: str,
    recurrence_values: tuple[int, ...],
) -> None:
    """Eval-only entrypoint: restore one checkpoint, sweep recurrence depth.

    Mirrors the mesh/optimizer/initial_state/restore setup of
    ``experiments.grug.reentrant.train._run_grug_local`` but runs no training: it
    builds the evaluator once and evaluates the restored params at each depth in
    ``recurrence_values``.
    """
    trainer = config.trainer.trainer
    trainer.initialize()
    levanter.tracker.log_configuration(config)

    # An optimizer is needed only to shape the train state for the checkpoint
    # restore (the same GrugTrainState layout was saved during training); it is
    # never stepped here.
    optimizer = config.optimizer.build(trainer.num_train_steps)

    model_key = jax.random.split(jax.random.PRNGKey(trainer.seed), 2)[1]

    mesh = compact_grug_mesh(
        expert_axis_size=config.trainer.expert_axis_size,
        replica_axis_size=config.trainer.replica_axis_size,
    )
    with set_mesh(mesh):

        @jax.jit
        def _init_state(model_rng):
            return initial_state(
                config.model,
                optimizer=optimizer,
                mp=trainer.mp,
                key=model_rng,
                ema_beta=config.trainer.ema_beta,
            )

        state = _init_state(model_key)
        state = restore_grug_state_from_checkpoint(
            state,
            checkpoint_search_paths=[checkpoint_path],
            load_checkpoint_setting=True,
            mesh=mesh,
            allow_partial=False,
        )

        eval_cfg = config.eval
        if eval_cfg is None:
            raise ValueError("eval config is required for the depth-scaling sweep")
        evaluator = build_tagged_evaluator(
            data_config=config.data,
            max_seq_len=config.model.max_seq_len,
            mesh=mesh,
            eval_cfg=eval_cfg,
        )
        if evaluator is None:
            raise ValueError("no evaluation datasets configured; cannot run the depth sweep")

        logger.info("Running depth-scaling eval sweep over R=%s", recurrence_values)
        results = evaluate_at_depths(state.params, evaluator, recurrence_values)
        for recurrence_steps in recurrence_values:
            _log_sweep_result(recurrence_steps, results[recurrence_steps])
        _print_sweep_table(results)

    levanter.tracker.current_tracker().finish()


def run_eval_sweep(
    config: GrugRunConfig,
    *,
    checkpoint_path: str,
    recurrence_values: tuple[int, ...],
) -> None:
    """Dispatch the depth-scaling eval sweep through Fray jobs.

    Mirrors ``experiments.grug.reentrant.train.run_grug`` but binds the checkpoint
    path and recurrence values into the entrypoint, since
    ``dispatch_grug_training_run`` calls ``local_entrypoint(config)``.
    """
    trainer = config.trainer.trainer
    if trainer.id is None:
        raise ValueError("trainer.id must be set before dispatching the eval sweep.")

    entrypoint = functools.partial(
        _run_eval_sweep_local,
        checkpoint_path=checkpoint_path,
        recurrence_values=recurrence_values,
    )
    dispatch_grug_training_run(
        run_id=trainer.id,
        config=config,
        local_entrypoint=entrypoint,
        resources=config.resources,
    )


@dataclass(frozen=True)
class GrugEvalSweepLaunchConfig:
    """Last-mile config for the re-entrant depth-scaling eval sweep."""

    model: GrugModelConfig
    data: LmDataConfig
    output_path: str
    run_id: str
    resources: ResourceConfig
    batch_size: int
    seed: int
    mp: str  # jmp policy string, e.g. "params=float32,compute=bfloat16,output=bfloat16".
    tracker: TrackerConfig
    optimizer: OptimizerConfig
    checkpoint_path: str
    recurrence_values: tuple[int, ...]
    grug_trainer: GrugTrainerConfig = field(default_factory=GrugTrainerConfig)
    eval: GrugEvalConfig | None = field(default_factory=GrugEvalConfig)


def run_grug_eval_sweep(config: GrugEvalSweepLaunchConfig) -> None:
    """Map launch knobs onto a GrugRunConfig and dispatch the sweep.

    ``num_train_steps=1`` keeps the LR-schedule build valid; no training runs.
    The checkpointer points at a temporary path because the sweep never saves.
    """
    tracker = config.tracker
    if isinstance(tracker, WandbConfig):
        tracker = dataclasses.replace(tracker, name=config.run_id)

    trainer = TrainerConfig(
        id=config.run_id,
        seed=config.seed,
        train_batch_size=config.batch_size,
        num_train_steps=1,
        mp=jmp.get_policy(config.mp),
        tracker=tracker,
        use_explicit_mesh_axes=True,
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        checkpointer=CheckpointerConfig(
            base_path=os.path.join(config.output_path, "checkpoints"),
            temporary_base_path=temporary_checkpoint_base_path(config.output_path),
            append_run_id_to_base_path=False,
            save_interval=timedelta(days=365),
            keep=None,
        ),
    )

    grug_trainer = dataclasses.replace(config.grug_trainer, trainer=trainer)
    run_config = GrugRunConfig(
        model=config.model,
        data=config.data,
        resources=config.resources,
        optimizer=config.optimizer,
        trainer=grug_trainer,
        eval=config.eval,
    )
    run_eval_sweep(
        run_config,
        checkpoint_path=config.checkpoint_path,
        recurrence_values=config.recurrence_values,
    )


def _parse_recurrence_values(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError(f"RECURRENCE_VALUES must list at least one integer, got {raw!r}")
    if any(v < 1 for v in values):
        raise ValueError(f"RECURRENCE_VALUES must all be >= 1, got {values}")
    return values


def _build_eval_sweep_step() -> ExecutorStep:
    """Build the ExecutorStep for a depth-scaling eval sweep from env config.

    The eval reuses the exact train-time model/data/resources so the restored
    checkpoint matches its train-state layout. SWEEP_MODEL (default e3) selects
    which variant's model config to restore with; CHECKPOINT_PATH (the gs://
    checkpoint dir) is required; RECURRENCE_VALUES defaults to "2,4,8,16,32".
    """
    checkpoint_path = os.environ.get("CHECKPOINT_PATH", "")
    if not checkpoint_path:
        raise ValueError("CHECKPOINT_PATH must be set to the gs:// checkpoint dir.")
    sweep_model_name = os.environ.get("SWEEP_MODEL", "e3").strip()
    sweep_model = _resolve_sweep_model()
    recurrence_values = _parse_recurrence_values(
        os.environ.get("RECURRENCE_VALUES", ",".join(str(v) for v in _DEFAULT_RECURRENCE_VALUES))
    )

    # Match the E3 train eval knobs but evaluate every batch (no train-time
    # max_eval_batches throttle) so the depth comparison is over the full set.
    eval_cfg = GrugEvalConfig(
        eval_batch_size=512,
        steps_per_eval=None,
        max_eval_batches=None,
        eval_current=True,
        eval_ema=False,
    )

    return ExecutorStep(
        name=f"grug/reentrant_{sweep_model_name}_eval_sweep",
        fn=run_grug_eval_sweep,
        config=GrugEvalSweepLaunchConfig(
            model=versioned(sweep_model),
            data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
            output_path=this_output_path(),
            run_id=_resolve_run_id(f"reentrant_{sweep_model_name}_eval_sweep"),
            resources=versioned(_REENTRANT_RESOURCES),
            batch_size=versioned(env_int("EVAL_BATCH_SIZE", 512)),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                project="marin_moe",
                tags=["moe", "reentrant", sweep_model_name, "depth-sweep"],
                group="reentrant-eval-sweep",
                name=None,
            ),
            optimizer=versioned(_baseline_optimizer),
            checkpoint_path=versioned(checkpoint_path),
            recurrence_values=versioned(recurrence_values),
            grug_trainer=versioned(GrugTrainerConfig(z_loss_weight=1e-4, ema_beta=None, log_every=1)),
            eval=versioned(eval_cfg),
        ),
    )


if __name__ == "__main__":
    executor_main(
        steps=[_build_eval_sweep_step()],
        description="Re-entrant grug E3 test-time depth-scaling eval sweep (d512).",
    )
