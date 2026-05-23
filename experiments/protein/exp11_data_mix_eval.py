# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exp 11 eval-only: run val/test eval on final checkpoints of mix/scale sweeps.

Loads the latest ``step_N`` checkpoint for each ``(stage, mixture)`` trial and
runs Levanter's :class:`TaggedEvaluator` on the same heldout components the
training run reported. Reuses :func:`build_mixture` via
:func:`_build_trial` so the ``LmDataConfig`` (and therefore the IID-carve
permutation + heldout cells + masking) matches training exactly.

W&B run names are ``<training run id>-eval-<EVAL_VERSION>`` so eval lands in
fresh runs that don't collide with the training run's step axis. Bump
``EVAL_VERSION`` to fork eval runs when eval semantics change (data shape,
masking, batch count, etc.) — training-run versions are untouched.

Subcommands (``COMMAND`` env var):

* ``run_mix_sweep``  — eval m1..m9 (100M) on H/M/L x val/test + cd-val (masked + unmasked).
* ``run_scale_sweep`` — eval m10..m15 (1.5B) on H x val/test + cd-val (masked).

Env vars: ``COMMAND`` (required), ``RUNS`` (CSV substring filter on eval target
ids), ``PREVIEW=yes`` (list targets, submit nothing), ``NUM_WORKERS`` (default
1; eval is short, one TPU sequential is cheap), ``TPU`` (override worker TPU;
default v5p-8), ``EVAL_MAX_BATCHES`` (default ``MAX_EVAL_BATCHES`` from the
sweep, currently 16).

Preview::

    COMMAND=run_scale_sweep PREVIEW=yes \\
        uv run python -m experiments.protein.exp11_data_mix_eval

Submit::

    uv run iris --cluster=marin job run --user $USERNAME --no-wait \\
        --job-name prot-exp11-eval-scale-$(date +%Y%m%d-%H%M) \\
        --region us-east5 --memory=1GB \\
        -e HF_TOKEN "$HF_TOKEN" -e WANDB_API_KEY "$WANDB_API_KEY" \\
        -e COMMAND run_scale_sweep \\
        -- python -m experiments.protein.exp11_data_mix_eval
"""

import logging
import os

import jmp
from fray import current_client
from fray.types import Entrypoint, JobRequest, create_environment
from levanter.main.eval_lm import EvalLmConfig
from levanter.main.eval_lm import main as eval_lm_main
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.execution.executor import compute_output_path
from marin.execution.sweep import SweepTarget, claim_and_run
from marin.training.training import extras_for_resources, resolve_training_env

from experiments.protein.exp11_data_mix_sweep import (
    MAX_EVAL_BATCHES,
    RUN_NAME_PREFIX,
    SEQ_LEN,
    STAGE_SPECS,
    SWEEP_ROOT_PREFIX,
    StageSpec,
    _build_trial,
    _preview,
    _resolve_targets,  # noqa: F401  # imported for parity; eval uses its own target ids
    _selected_runs,
    _trial_name,
)

logger = logging.getLogger(__name__)

# Bump to fork eval runs without touching training-run versions. Used in both
# the W&B run name and the claim_and_run sweep root.
EVAL_VERSION: str = "v1"

# Stages eligible for eval. Smoke is excluded — it's a 400M-token sanity run.
EVAL_STAGES: tuple[str, ...] = ("run_mix_sweep", "run_scale_sweep")

# Default eval batch count. Matches the training-time eval budget so the same
# number of examples per component is scored. Override via EVAL_MAX_BATCHES.
DEFAULT_EVAL_MAX_BATCHES: int = MAX_EVAL_BATCHES

# W&B group for eval runs. Distinct from training's "exp11-data-mix" so they
# don't clutter the same group view. Runs land under the wandb API key owner's
# entity (currently eric-czech/marin), same as training.
EVAL_WANDB_GROUP: str = "exp11-data-mix-eval"


def _eval_max_batches() -> int:
    raw = os.environ.get("EVAL_MAX_BATCHES")
    return int(raw) if raw else DEFAULT_EVAL_MAX_BATCHES


def _eval_run_name(spec: StageSpec, mixture_id: str) -> str:
    """Eval run id: ``<training run id>-eval-<EVAL_VERSION>``."""
    return f"{_trial_name(spec, mixture_id)}-eval-{EVAL_VERSION}"


def _eval_targets(spec: StageSpec) -> list[SweepTarget]:
    return [SweepTarget(target_id=_eval_run_name(spec, mid), config=(mid,)) for mid in spec.mixture_ids]


def _resolve_eval_targets(spec: StageSpec, runs: tuple[str, ...]) -> list[SweepTarget]:
    targets = _eval_targets(spec)
    if runs:
        targets = [t for t in targets if any(r in t.target_id for r in runs)]
    return targets


def _eval_sweep_root(spec: StageSpec) -> str:
    return f"{SWEEP_ROOT_PREFIX}/eval-{spec.name}-{spec.version}-{EVAL_VERSION}"


def _trial_checkpoint_dir(spec: StageSpec, mixture_id: str) -> str:
    """Recompute the trial's checkpoint dir from its StageSpec.

    Re-runs the same ``prepare_lm_train`` pipeline training used, so the
    ``compute_output_path`` hash matches. Returns ``<output_path>/checkpoints``;
    Levanter's ``latest_checkpoint_path`` walks it to find the latest ``step_N``.
    """
    name, raw_config = _build_trial(spec, mixture_id)
    output_path = compute_output_path(name, raw_config, override_output_path=None)
    return f"{output_path}/checkpoints"


def _build_eval_config(spec: StageSpec, mixture_id: str, *, max_eval_batches: int) -> EvalLmConfig:
    """Build an :class:`EvalLmConfig` for one trial.

    Reuses ``_build_trial`` so the data config is bit-identical to training's:
    same heldout cells, same ``num_validation_sequences`` (IID-carve), same
    ``data_seed`` (= ``DATA_SEED`` constant). ``per_device_eval_parallelism``
    is left at -1 so eval_batch_size resolves to ``train_batch_size``, matching
    the eval batch the training loop used (so ``max_eval_batches`` semantics
    are directly comparable across train and eval).
    """
    name, raw_config = _build_trial(spec, mixture_id)
    output_path = compute_output_path(name, raw_config, override_output_path=None)
    checkpoint_dir = f"{output_path}/checkpoints"
    run_name = _eval_run_name(spec, mixture_id)

    return EvalLmConfig(
        checkpoint_path=checkpoint_dir,
        model=spec.model_config,
        data=raw_config.data,
        max_eval_length=SEQ_LEN,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                project="marin",
                name=run_name,
                group=EVAL_WANDB_GROUP,
                tags=[
                    "protein",
                    "exp11",
                    "data-mix",
                    "eval",
                    spec.label,
                    spec.model_tag,
                    mixture_id,
                    f"eval={EVAL_VERSION}",
                ],
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            # Match training's eval batch so EVAL_MAX_BATCHES counts the same
            # number of examples per component as the training-time eval.
            train_batch_size=spec.batch_size,
            max_eval_batches=max_eval_batches,
        ),
    )


def _worker_entrypoint(stage: str, rank: int, num_workers: int, runs: tuple[str, ...], max_eval_batches: int) -> None:
    """One eval worker: slice targets rank-stride, claim, and run eval_lm per trial."""
    spec = STAGE_SPECS[stage]
    targets = _resolve_eval_targets(spec, runs)
    my_targets = targets[rank::num_workers]
    logger.info(
        "Eval worker rank=%d/%d assigned %d/%d target(s): %s",
        rank,
        num_workers,
        len(my_targets),
        len(targets),
        [t.target_id for t in my_targets],
    )
    sweep_root = _eval_sweep_root(spec)

    def _run_one(target: SweepTarget) -> None:
        (mixture_id,) = target.config
        config = _build_eval_config(spec, mixture_id, max_eval_batches=max_eval_batches)
        logger.info("Running eval for %s from %s", target.target_id, config.checkpoint_path)
        eval_lm_main(config)

    claim_and_run(sweep_root, my_targets, _run_one)


# ============================================================================
# Launcher / preview
# ============================================================================


def _print_eval_preview(spec: StageSpec, targets: list[SweepTarget], *, max_eval_batches: int) -> None:
    print(
        f"PREVIEW: eval {spec.name} ({spec.version}/{EVAL_VERSION}) would run "
        f"{len(targets)} target(s); max_eval_batches={max_eval_batches}:",
        flush=True,
    )
    for t in targets:
        (mid,) = t.config
        ckpt = _trial_checkpoint_dir(spec, mid)
        print(f"  {t.target_id}", flush=True)
        print(f"    checkpoint: {ckpt}", flush=True)


def _eval_launcher(stage: str) -> None:
    spec = STAGE_SPECS[stage]
    runs = _selected_runs()
    targets = _resolve_eval_targets(spec, runs)
    if not targets:
        raise ValueError(f"eval {stage}: no targets matched RUNS={runs!r}")

    max_eval_batches = _eval_max_batches()

    if _preview():
        _print_eval_preview(spec, targets, max_eval_batches=max_eval_batches)
        return

    # Default to 1 worker: eval is short, paying TPU startup for parallelism
    # is rarely worth it. Bump via NUM_WORKERS to parallelize.
    num_workers = int(os.environ.get("NUM_WORKERS", "1"))
    num_workers = min(num_workers, len(targets))
    resources = spec.resources_fn()
    env = resolve_training_env(base_env=None, resources=resources)
    extras = extras_for_resources(resources)

    logger.info(
        "Eval stage=%s targets=%d workers=%d runs=%s max_eval_batches=%d resources=%s",
        stage,
        len(targets),
        num_workers,
        runs,
        max_eval_batches,
        resources,
    )

    client = current_client()
    handles = []
    for rank in range(num_workers):
        request = JobRequest(
            name=f"{RUN_NAME_PREFIX}-eval-{stage}-w{rank}",
            entrypoint=Entrypoint.from_callable(
                _worker_entrypoint,
                args=[stage, rank, num_workers, runs, max_eval_batches],
            ),
            resources=resources,
            environment=create_environment(env_vars=env, extras=extras),
        )
        handles.append(client.submit(request))
        logger.info("Submitted eval worker rank=%d/%d: %s", rank, num_workers, request.name)

    failures = 0
    for rank, h in enumerate(handles):
        try:
            h.wait(raise_on_failure=True)
            logger.info("Eval worker rank=%d finished", rank)
        except Exception:
            failures += 1
            logger.exception("Eval worker rank=%d failed", rank)
    if failures:
        raise RuntimeError(f"{failures}/{num_workers} eval workers failed")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    command = os.environ.get("COMMAND")
    if command in EVAL_STAGES:
        _eval_launcher(command)
        return
    raise ValueError(f"Set COMMAND to one of: {', '.join(EVAL_STAGES)}. Got {command!r}.")


if __name__ == "__main__":
    main()
