# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grug-MoE swarm on the CoreWeave datakit store: one training run per mixture.

The successor to ``experiments/mixing/v0/launch_swarm.py`` on the current framework and on
CoreWeave. Reads a set of candidate mixtures and emits one ``ArtifactStep`` per mixture, all at
the *same* model/optimizer/regime and resources -- only the 168-bucket two-phase weights vary,
exactly as the original 840-run swarm did. ``build()`` returns the list of steps; ``run`` builds
them with the shared ``--max-concurrent`` cap.

The model/optimizer/regime/resources are inherited verbatim from
:mod:`experiments.grug.moe.launch_datakit_moe_mix` (configured via the same ``DATAKIT_*`` env
vars), so a swarm run is byte-identical to a single ``launch_datakit_moe_mix`` run except for the
mixture. Set ``budget_tokens`` in the surrogate to the REAL per-run token budget these runs train
at (~1e11 for the swarm regime), not the simulated store size.

Mixtures come from ``SWARM_MIXTURES`` (default ``proposals.json``), which is either:
  * the surrogate's ``sample.py`` export -- a dict with a ``"proposals"`` list, each entry having
    ``weights: {"phase0": {bucket: w}, "phase1": {bucket: w}}``; or
  * a plain list of ``{"phase0": {...}, "phase1": {...}}`` mixtures.
Bucket names are the 168 grug buckets (``cNNqT`` + ``tail``), which match the datakit mixture space
1:1. Each phase is renormalized to sum to 1.

    # 1. propose mixtures with the surrogate (real run budget!)
    python experiments/mixture_surrogate/sample.py --budget-tokens 1e11 --top-k 50 --out proposals.json
    # 2. launch the swarm on CoreWeave via federated Iris
    SWARM_MIXTURES=proposals.json .venv/bin/iris --cluster=cw-rno2a job run --no-wait --cpu 4 --extra cpu \
        -- python -m experiments.grug.moe.launch_datakit_swarm_cw --version dev --run --max-concurrent 16
"""

import json
import os
from typing import NamedTuple

from levanter.checkpoint import CheckpointerConfig
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import experiment_main
from marin.experiment.namespacing import user_namespaced_name
from marin.training.training import LevanterCheckpoint

import experiments.grug.moe.launch_datakit_moe_mix as dk
from experiments.grug.moe.launch import GrugMoeLaunchConfig, env_int, run_grug_moe_trial

_MIXTURES_PATH = os.environ.get("SWARM_MIXTURES", "proposals.json")
_LIMIT = env_int("SWARM_LIMIT", 0)  # 0 = all mixtures in the file


class Mixture(NamedTuple):
    label: str
    phase0: dict[str, float]  # bucket -> weight, renormalized to 1
    phase1: dict[str, float]


def _load_mixtures(path: str) -> list[Mixture]:
    """Parse ``SWARM_MIXTURES``: the surrogate's ``proposals.json`` (a dict with ``"proposals"``) or
    a plain list of ``{"phase0": {...}, "phase1": {...}}`` mixtures. Each phase is renormalized to 1.
    """
    with open(path) as fh:
        raw = json.load(fh)
    items = raw["proposals"] if isinstance(raw, dict) and "proposals" in raw else raw
    if not items:
        raise ValueError(f"no mixtures in {path}")
    mixtures = []
    for i, item in enumerate(items):
        weights = item.get("weights", item)
        phase0 = dk._normalize({b: float(v) for b, v in weights["phase0"].items()})
        phase1 = dk._normalize({b: float(v) for b, v in weights["phase1"].items()})
        mixtures.append(Mixture(str(item.get("rank", item.get("label", i))), phase0, phase1))
    return mixtures


def _make_build_config(mixture: Mixture, run_id: str):
    def build_config(ctx: StepContext) -> GrugMoeLaunchConfig:
        if ctx.is_fingerprint:
            val_components = {v.name: dk._val_component(ctx.artifact_path(v)) for v in dk._VALIDATION}
        else:
            val_components = {v.name: ctx.resolved(v).as_component() for v in dk._VALIDATION}
        data = dk._datakit_data_config(
            phase0_weights=mixture.phase0,
            phase1_weights=mixture.phase1,
            total_steps=dk._steps,
            batch_size=dk._batch_size,
            max_seq_len=dk._model.max_seq_len,
            enable_simulated_epoching=dk.ENABLE_SIMULATED_EPOCHING,
            val_components=val_components,
        )
        tracker: WandbConfig | JsonLoggerConfig = (
            WandbConfig(project="marin_moe", tags=["moe", "datakit_swarm", run_id], group="datakit-swarm", name=None)
            if dk._USE_WANDB
            else JsonLoggerConfig(logger_name="datakit_swarm.metrics")
        )
        # Node-local disposable checkpoints: a swarm wants the eval metric per run, not durable
        # weights, and CW S3 tensorstore writes are fragile.
        checkpointer = CheckpointerConfig(
            base_path=f"/tmp/datakit-swarm-ckpt/{run_id}",
            append_run_id_to_base_path=False,
            save_interval=None,
            keep=None,
        )
        return dk.build_launch_config(ctx, data=data, run_id=run_id, tracker=tracker, checkpointer=checkpointer)

    return build_config


def _step_for(mixture: Mixture) -> ArtifactStep[LevanterCheckpoint]:
    name = f"grug/datakit_swarm_{dk._SLUG}/mix{mixture.label}"
    version = resolve_version(name, None)
    return ArtifactStep(
        name=user_namespaced_name(name, version),
        version=version,
        artifact_type=LevanterCheckpoint,
        run=run_grug_moe_trial,
        build_config=_make_build_config(mixture, f"datakit-swarm-{mixture.label}"),
        deps=tuple(dk._VALIDATION),
        runtime_args={"train_resources": dk._TRAIN_RESOURCES},
    )


def build() -> list[ArtifactStep[LevanterCheckpoint]]:
    """One training step per mixture in ``SWARM_MIXTURES`` (capped by ``SWARM_LIMIT``)."""
    mixtures = _load_mixtures(_MIXTURES_PATH)
    if _LIMIT:
        mixtures = mixtures[:_LIMIT]
    return [_step_for(mixture) for mixture in mixtures]


if __name__ == "__main__":
    experiment_main(build)()
