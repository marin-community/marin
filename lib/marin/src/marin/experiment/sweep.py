# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hyperparameter sweeps over lazy checkpoints: fan out trials, select the best.

A sweep is fan-out then fan-in over ``Lazy[Checkpoint]``. :func:`sweep` builds one
checkpoint handle per grid point; :func:`select` reduces them to the one whose recorded
metric is best, as a ``Lazy[Selection]``. The whole thing lowers and runs through the
normal pipeline — every trial trains (its own job), then the selection runs inline.

A trial is just a checkpoint-producing function (e.g. ``lambda **p: train_lm(...)``):
there is no metrics payload to return. :func:`select` reads each trial's metrics from
where the trial *wrote* them — its output path — with a single ``reader`` (a sweep is
homogeneous, so every trial writes its metrics the same way). The default reader,
:func:`read_replicated_metrics`, reads the ``tracker_metrics.jsonl`` that a
:func:`~marin.experiment.train.train_lm` run mirrors next to its checkpoints (the WandB
``replicate_path``). A custom metric source is a ``reader=`` argument to :func:`select`,
not a per-trial wrapper.
"""

import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from marin.execution.artifact import Checkpoint, JsonArtifact
from marin.execution.lazy import Lazy, Recipe, RunContext
from marin.scaling_laws.eval_metrics_reader import read_eval_records


class Selection(JsonArtifact):
    """The outcome of a :func:`select`: the winning trial and the scores it ranked."""

    winner: str
    """The winning trial's ``name@version``."""
    score: float
    """The winner's value of the selection ``metric``."""
    winner_path: str
    """The winner's resolved output path."""
    scores: dict[str, float]
    """Every trial's ``name@version`` -> its ``metric`` value."""
    metrics: dict[str, Any]
    """The winner's full recorded metrics summary."""


def read_replicated_metrics(output_path: str) -> Mapping[str, Any]:
    """A ``train_lm`` run's final metrics, read back from its output.

    ``train_lm`` mirrors WandB's final summary to ``<output>/tracker_metrics.jsonl``
    (the trainer's ``replicate_path``). This returns that summary mapping — the
    default reader for sweep trials that train with ``train_lm``. Selecting on
    ``train/loss`` or any logged ``eval/.../loss`` works without any per-trial wiring.
    """
    records = read_eval_records([output_path])
    if not records:
        raise FileNotFoundError(f"no recorded metrics for trial at {output_path}")
    return records[-1]["summary"]


def grid(**axes: Sequence[Any]) -> list[dict[str, Any]]:
    """The Cartesian product of named axes, as a list of parameter dicts.

    ``grid(learning_rate=[1e-3, 3e-3], weight_decay=[0.0, 0.1])`` yields the four
    ``{"learning_rate": ..., "weight_decay": ...}`` combinations, in row-major order
    (the last axis varies fastest).
    """
    keys = list(axes)
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*axes.values())]


def sweep(trial: Callable[..., Lazy[Checkpoint]], **axes: Sequence[Any]) -> list[Lazy[Checkpoint]]:
    """One trial handle per grid point: ``trial(**params)`` for each combination.

    ``axes`` are the swept dimensions (see :func:`grid`); ``trial`` maps one parameter
    set to a ``Lazy[Checkpoint]``. The trial must fold the swept values into both its
    config (so each grid point gets a distinct fingerprint) and its ``name`` (so each
    gets a distinct, readable address); :func:`select` rejects trials whose
    ``name@version`` collide.
    """
    return [trial(**params) for params in grid(**axes)]


@dataclass(frozen=True)
class _SelectConfig:
    """The reducer's config: how to rank, and where each trial's output lives."""

    metric: str
    mode: str
    trials: dict[str, str]
    """Trial ``name@version`` -> resolved output path (a placeholder ``name@version`` at
    fingerprint time). Keyed by full identity, not bare name, so two trials that differ only
    by version do not collapse."""


def select(
    name: str,
    version: str,
    trials: Sequence[Lazy[Checkpoint]],
    *,
    metric: str,
    mode: str = "min",
    reader: Callable[[str], Mapping[str, Any]] = read_replicated_metrics,
) -> Lazy[Selection]:
    """The trial whose recorded ``metric`` is best (``min``/``max``), as a ``Selection``.

    Depends on every trial. At run time it reads each trial's metrics with ``reader``,
    ranks them by ``metric``, and produces a :class:`Selection`
    (``winner``/``score``/``winner_path``/``scores``/``metrics``). Read the selection back
    (its ``winner``/``winner_path``) to drive a follow-on run from the winning trial.

    ``metric`` and ``mode`` bear identity (they enter the fingerprint): selecting by a
    different metric or direction is a different artifact. The trial *values* and the
    ``reader`` do not — values are read at run time, and a callable has no stable
    fingerprint, so swapping readers does not re-identify the selection (bump ``version``
    if a reader change should be a new artifact).
    """
    if mode not in ("min", "max"):
        raise ValueError(f"select mode must be 'min' or 'max', got {mode!r}")
    trials = tuple(trials)
    if not trials:
        raise ValueError("select needs at least one trial")
    ids = [f"{t.name}@{t.version}" for t in trials]
    if len(set(ids)) != len(ids):
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"select trials must have distinct name@version; duplicates: {duplicates}")

    def build_config(ctx: RunContext) -> _SelectConfig:
        return _SelectConfig(
            metric=metric, mode=mode, trials={tid: ctx.path(t) for tid, t in zip(ids, trials, strict=True)}
        )

    def choose(config: _SelectConfig) -> Selection:
        maximize = config.mode == "max"
        scores: dict[str, float] = {}
        summaries: dict[str, Mapping[str, Any]] = {}
        best: tuple[str, float, str] | None = None
        for trial_id, trial_path in config.trials.items():
            summary = reader(trial_path)
            summaries[trial_id] = summary
            score = summary[config.metric]
            scores[trial_id] = score
            if best is None or (score > best[1] if maximize else score < best[1]):
                best = (trial_id, score, trial_path)
        assert best is not None  # config.trials is non-empty (guarded at build time)
        winner_id, winner_score, winner_path = best
        return Selection(
            winner=winner_id,
            score=winner_score,
            winner_path=winner_path,
            scores=scores,
            metrics=dict(summaries[winner_id]),
        )

    return Lazy(
        name=name,
        version=version,
        recipe=Recipe(fn=choose, build_config=build_config, deps=trials),
        result_type=Selection,
    )
