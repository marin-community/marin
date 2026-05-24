# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exp 11 eval-only: run val/test eval on one trial's final checkpoint.

Designed to run directly on a TPU VM (no Fray coordinator). Each invocation
evaluates exactly one (stage, mixture) trial. Parallelize at the top level by
submitting separate ``iris job run`` commands, one per mixture.

Reuses :func:`build_mixture` via :func:`_build_trial` so the ``LmDataConfig``
(heldout cells, IID-carve permutation, masking) matches training exactly.
W&B run name is ``<training run id>-eval-<EVAL_VERSION>`` so eval lands in a
fresh run that doesn't collide with the training run's step axis. Bump
``EVAL_VERSION`` to fork eval runs when eval semantics change.

Env vars:

* ``COMMAND`` (required) — ``"run_mix_sweep"`` or ``"run_scale_sweep"``.
* ``RUN`` (required) — mixture id (e.g. ``"m10"``); must exactly match one
  of the stage's mixtures.
* ``EVAL_MAX_BATCHES`` (optional) — cap on batches per heldout component.
  Unset = no cap (evaluate every batch). Set to an int for smoke tests.
* ``PREVIEW=yes`` (optional) — describe target, don't run.

Submission example::

    uv run iris job run \\
        --region us-east5 --user eczech --no-wait --priority interactive \\
        --tpu v6e-8 --enable-extra-resources --extra tpu --extra lm_eval \\
        --memory 128GB \\
        --job-name prot-exp11-eval-scale-m10-$(date +%Y%m%d-%H%M) \\
        -e HF_TOKEN "$HF_TOKEN" -e WANDB_API_KEY "$WANDB_API_KEY" \\
        -e COMMAND run_scale_sweep -e RUN m10 \\
        -- python -m experiments.protein.exp11_data_mix_eval
"""

import dataclasses
import logging
import os
from typing import Any

import jmp
from levanter.checkpoint import latest_checkpoint_path
from levanter.main.eval_lm import EvalLmConfig
from levanter.main.eval_lm import main as eval_lm_main
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.execution.executor import compute_output_path

from experiments.protein.exp11_data_mix_sweep import (
    HIDDEN_100M,
    SEQ_LEN,
    STAGE_SPECS,
    StageSpec,
    _build_trial,
    _trial_name,
    scaled_lr,
)

logger = logging.getLogger(__name__)

# Bump to fork eval runs without touching training-run versions. Goes into the
# W&B run name suffix.
EVAL_VERSION: str = "v2"

# Stages eligible for eval. Smoke is excluded — it's a 400M-token sanity run.
EVAL_STAGES: tuple[str, ...] = ("run_mix_sweep", "run_scale_sweep")

# Sweep version actually trained (and thus targeted by eval) per stage. The
# live ``STAGE_SPECS`` in the sweep file may have advanced past these — when
# that's the case, ``_STAGE_SPEC_OVERRIDES`` below supplies the (batch, lr,
# steps, version) values needed to recompute the trained-checkpoint path.
# Bump these (and the override entry below) once a newer training version
# has actually produced checkpoints on GCS.
MIX_TRAIN_VERSION: str = "v2"
SCALE_TRAIN_VERSION: str = "v6"

# Per-stage StageSpec field overrides applied at eval-build time. Only fields
# that affect ``_trial_name`` or the ``compute_output_path`` hash need to
# appear. An empty dict means the live ``STAGE_SPECS[stage]`` already matches
# the trained version (no override needed).
#
# Mix override pins to the trained v2 sweep (batch=128, scaled-LR, 4103 steps).
# Scale needs no override — STAGE_SPECS["run_scale_sweep"].version is v6,
# which matches SCALE_TRAIN_VERSION.
_STAGE_SPEC_OVERRIDES: dict[str, dict[str, Any]] = {
    "run_mix_sweep": {
        "version": MIX_TRAIN_VERSION,
        "batch_size": 128,
        "learning_rate": scaled_lr(128, HIDDEN_100M),
        # num_train_steps=4104 matches the trained path hash. Levanter saves
        # the final checkpoint at step-(N-1), so the on-disk dir is step-4103.
        "num_train_steps": 4104,
    },
    "run_scale_sweep": {},
}


def _resolve_spec(stage: str) -> StageSpec:
    """Return the StageSpec the eval should target for ``stage``.

    Falls back to ``STAGE_SPECS[stage]`` unchanged when no override is
    registered for the stage. ``dataclasses.replace`` mutates a single field
    list at a time so overrides can be added incrementally.
    """
    spec = STAGE_SPECS[stage]
    overrides = _STAGE_SPEC_OVERRIDES.get(stage) or {}
    if not overrides:
        return spec
    return dataclasses.replace(spec, **overrides)


# Sanity-check at import time that the scale spec hasn't drifted past
# SCALE_TRAIN_VERSION without an override entry.
assert STAGE_SPECS["run_scale_sweep"].version == SCALE_TRAIN_VERSION or _STAGE_SPEC_OVERRIDES["run_scale_sweep"], (
    f"STAGE_SPECS['run_scale_sweep'].version={STAGE_SPECS['run_scale_sweep'].version!r} "
    f"!= SCALE_TRAIN_VERSION={SCALE_TRAIN_VERSION!r}; add an override entry."
)

# Default eval batch count. ``None`` = evaluate every batch in every heldout
# component (no truncation). Override via ``EVAL_MAX_BATCHES`` if you want to
# cap eval cost — e.g. for quick smoke tests.
DEFAULT_EVAL_MAX_BATCHES: int | None = None

# W&B group for eval runs. Distinct from training's "exp11-data-mix" so they
# don't clutter the same group view. Runs land under the wandb API key owner's
# entity (currently eric-czech/marin), same as training.
EVAL_WANDB_GROUP: str = "exp11-data-mix-eval"


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Required env var {name!r} not set")
    return value


def _eval_max_batches() -> int | None:
    raw = os.environ.get("EVAL_MAX_BATCHES")
    return DEFAULT_EVAL_MAX_BATCHES if raw is None else int(raw)


def _preview() -> bool:
    return os.environ.get("PREVIEW", "").strip().lower() in {"yes", "true", "1"}


def _eval_run_name(spec: StageSpec, mixture_id: str) -> str:
    """Eval run id: ``<training run id>-eval-<EVAL_VERSION>``."""
    return f"{_trial_name(spec, mixture_id)}-eval-{EVAL_VERSION}"


def _resolve_mixture(spec: StageSpec, run: str) -> str:
    """Resolve ``RUN`` to exactly one of ``spec.mixture_ids`` by exact match.

    Exact match (not substring) so ``RUN=m1`` can't accidentally also match
    m10, m11, etc.
    """
    if run not in spec.mixture_ids:
        raise ValueError(f"RUN={run!r} not in {spec.name} mixtures {spec.mixture_ids}")
    return run


def _build_eval_config(spec: StageSpec, mixture_id: str, *, max_eval_batches: int) -> EvalLmConfig:
    """Build an :class:`EvalLmConfig` for one trial.

    Reuses ``_build_trial`` so the data config is bit-identical to training's:
    same heldout cells, same ``num_validation_sequences`` (IID-carve), same
    ``data_seed``. ``per_device_eval_parallelism`` is left at -1 so
    eval_batch_size resolves to ``train_batch_size``, matching the eval batch
    the training loop used (so ``max_eval_batches`` semantics are directly
    comparable across train and eval).
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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    command = _required_env("COMMAND")
    if command not in EVAL_STAGES:
        raise ValueError(f"COMMAND must be one of {EVAL_STAGES}. Got {command!r}.")
    spec = _resolve_spec(command)
    mixture_id = _resolve_mixture(spec, _required_env("RUN"))
    max_eval_batches = _eval_max_batches()
    config = _build_eval_config(spec, mixture_id, max_eval_batches=max_eval_batches)
    run_name = _eval_run_name(spec, mixture_id)

    if _preview():
        print(f"PREVIEW: eval {command} ({spec.version}/{EVAL_VERSION})", flush=True)
        print(f"  target:         {run_name}", flush=True)
        print(f"  checkpoint dir: {config.checkpoint_path}", flush=True)
        try:
            resolved = latest_checkpoint_path(config.checkpoint_path)
            print(f"  resolved:       {resolved}", flush=True)
        except FileNotFoundError as e:
            print(f"  resolved:       <not found: {e}>", flush=True)
        print(f"  max_eval_batches: {max_eval_batches}", flush=True)
        return

    logger.info("Running eval for %s from %s", run_name, config.checkpoint_path)
    eval_lm_main(config)


if __name__ == "__main__":
    main()
