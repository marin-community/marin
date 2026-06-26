# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hyperparameter sweeps as lazy artifacts: fan out trials, select the best.

A sweep is fan-out then fan-in. :func:`sweep` builds one trial handle per grid
point; :func:`select` is a reducer that depends on every trial, reads each trial's
recorded metrics, and writes the winner. Both are ordinary
:class:`~marin.execution.lazy.Artifact`\\ s, so a sweep lowers and runs through the
normal pipeline — every trial materializes (each its own job), then the selection.

The trial contract: a trial's step fn produces a *metrics mapping* as its artifact
payload — the value it returns, which the runner persists via
:class:`marin.execution.artifact.Artifact`. :func:`select` reads that mapping back
by a metric key.
"""

import itertools
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from marin.execution.artifact import Artifact as ArtifactIO
from marin.execution.lazy import Artifact, Recipe, RunContext


def grid(**axes: Sequence[Any]) -> list[dict[str, Any]]:
    """The Cartesian product of named axes, as a list of parameter dicts.

    ``grid(learning_rate=[1e-3, 3e-3], weight_decay=[0.0, 0.1])`` yields the four
    ``{"learning_rate": ..., "weight_decay": ...}`` combinations, in row-major order
    (the last axis varies fastest).
    """
    keys = list(axes)
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*axes.values())]


def sweep(trial: Callable[..., Artifact], **axes: Sequence[Any]) -> list[Artifact]:
    """One trial handle per grid point: ``trial(**params)`` for each combination.

    ``axes`` are the swept dimensions (see :func:`grid`); ``trial`` maps one parameter
    set to a handle. The swept values are literals in each trial's config, so every
    grid point gets a distinct fingerprint and ``name@version``.
    """
    return [trial(**params) for params in grid(**axes)]


def select(
    name: str,
    version: str,
    trials: Sequence[Artifact],
    *,
    metric: str,
    mode: str = "min",
) -> Artifact:
    """A reducer artifact: the trial whose metrics ``min``/``max`` ``metric``.

    Depends on every trial. At run time it reads each trial's metrics mapping (its
    artifact payload) and writes its own payload — ``{"winner", "score",
    "winner_path", "metrics", "scores"}`` — so a consumer can read the chosen run
    through this one handle (its build fn reads ``winner_path`` from the payload).

    ``metric`` and ``mode`` bear identity (they enter the fingerprint): selecting by a
    different metric or direction is a different artifact. The trial *values* do not
    (they are read at run time), so the selection is identified by its inputs'
    ``name@version`` and the selection rule.
    """
    if mode not in ("min", "max"):
        raise ValueError(f"select mode must be 'min' or 'max', got {mode!r}")
    trials = tuple(trials)
    if not trials:
        raise ValueError("select needs at least one trial")

    def build_config(ctx: RunContext) -> dict[str, Any]:
        return {"metric": metric, "mode": mode, "trials": {t.name: ctx.path(t) for t in trials}}

    def choose(config: Mapping[str, Any]) -> dict[str, Any]:
        maximize = config["mode"] == "max"
        scores: dict[str, float] = {}
        metrics: dict[str, Mapping[str, Any]] = {}
        best: tuple[str, float, str] | None = None
        for trial_name, trial_path in config["trials"].items():
            payload = ArtifactIO.from_path(trial_path)
            if not isinstance(payload, Mapping):
                raise TypeError(f"trial {trial_name!r} produced no metrics payload at {trial_path}")
            metrics[trial_name] = payload
            score = payload[config["metric"]]
            scores[trial_name] = score
            if best is None or (score > best[1] if maximize else score < best[1]):
                best = (trial_name, score, trial_path)
        assert best is not None  # config["trials"] is non-empty (guarded at build time)
        winner_name, winner_score, winner_path = best
        return {
            "winner": winner_name,
            "score": winner_score,
            "winner_path": winner_path,
            "metrics": dict(metrics[winner_name]),
            "scores": scores,
        }

    return Artifact(name=name, version=version, recipe=Recipe(fn=choose, build_config=build_config, deps=trials))
