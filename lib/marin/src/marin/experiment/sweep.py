# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hyperparameter sweeps over lazy checkpoints: fan out trials, select the best.

A sweep is fan-out then fan-in over ``Lazy[Checkpoint]``. :func:`sweep` builds one
checkpoint handle per grid point; :func:`select` reduces them to the one whose
recorded metric is best. Both are ordinary
:class:`~marin.execution.lazy.Artifact`\\ s, so a sweep lowers and runs through the
normal pipeline — every trial trains (its own job), then the selection.

A trial is just a checkpoint-producing function (e.g. ``lambda **p: train_lm(...)``):
there is no metrics payload to return. Selection reads each trial's metric from where
the trial *wrote* it — its output path. The handle that pairs a checkpoint with "how
to read my own metrics" is an :class:`AnnotatedCheckpoint`; :func:`annotate` wraps a
plain :class:`~marin.execution.lazy.Checkpoint` into one. The default reader,
:func:`read_replicated_metrics`, reads the ``tracker_metrics.jsonl`` that a
:func:`~marin.experiment.train.train_lm` run mirrors next to its checkpoints (the
WandB ``replicate_path``). :func:`select` itself stays generic: it knows nothing about
where metrics live (the trial's reader does that) — only which ``metric`` key to rank
by and which direction.
"""

import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from marin.execution.lazy import Checkpoint, Recipe, RunContext
from marin.scaling_laws.eval_metrics_reader import read_eval_records


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


@dataclass(frozen=True, eq=False)
class AnnotatedCheckpoint(Checkpoint):
    """A :class:`~marin.execution.lazy.Checkpoint` that also knows how to read its
    own recorded metrics.

    ``metrics_reader(output_path) -> Mapping`` reads the trial's metrics from its
    materialized output (where training wrote them), so a sweep can rank trials
    without each trial returning a metrics payload. The reader does not bear on
    identity — it is read at run time, not built into the fingerprint.
    """

    metrics_reader: Callable[[str], Mapping[str, Any]] = read_replicated_metrics


def annotate(
    checkpoint: Checkpoint,
    *,
    metrics_reader: Callable[[str], Mapping[str, Any]] = read_replicated_metrics,
) -> AnnotatedCheckpoint:
    """Pair a checkpoint with a reader for its recorded metrics, for selection.

    Wraps a plain :class:`~marin.execution.lazy.Checkpoint` (e.g. from
    :func:`~marin.experiment.train.train_lm`) without changing its identity: same
    ``name``/``version``/recipe, so it lowers and caches exactly as before.
    """
    return AnnotatedCheckpoint(
        name=checkpoint.name,
        version=checkpoint.version,
        recipe=checkpoint.recipe,
        override_path=checkpoint.override_path,
        adopt_source=checkpoint.adopt_source,
        expected_fingerprint=checkpoint.expected_fingerprint,
        metrics_reader=metrics_reader,
    )


def grid(**axes: Sequence[Any]) -> list[dict[str, Any]]:
    """The Cartesian product of named axes, as a list of parameter dicts.

    ``grid(learning_rate=[1e-3, 3e-3], weight_decay=[0.0, 0.1])`` yields the four
    ``{"learning_rate": ..., "weight_decay": ...}`` combinations, in row-major order
    (the last axis varies fastest).
    """
    keys = list(axes)
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*axes.values())]


def sweep(trial: Callable[..., AnnotatedCheckpoint], **axes: Sequence[Any]) -> list[AnnotatedCheckpoint]:
    """One trial handle per grid point: ``trial(**params)`` for each combination.

    ``axes`` are the swept dimensions (see :func:`grid`); ``trial`` maps one parameter
    set to an :class:`AnnotatedCheckpoint`. The trial must fold the swept values into
    both its config (so each grid point gets a distinct fingerprint) and its ``name``
    (so each gets a distinct, readable address); :func:`select` rejects trials whose
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
    trials: Sequence[AnnotatedCheckpoint],
    *,
    metric: str,
    mode: str = "min",
) -> Checkpoint:
    """The trial whose recorded ``metric`` is best (``min``/``max``), as a checkpoint.

    Depends on every trial. At run time it reads each trial's metrics through that
    trial's own reader (an :class:`AnnotatedCheckpoint` annotation), ranks them by
    ``metric``, and writes its payload — ``{"winner", "score", "winner_path",
    "metrics", "scores"}``. The returned handle is a
    :class:`~marin.execution.lazy.Checkpoint` addressed at ``name@version``: read the
    selection through it (``winner``/``winner_path``) to drive a follow-on run from the
    winning trial.

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
    ids = [f"{t.name}@{t.version}" for t in trials]
    if len(set(ids)) != len(ids):
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"select trials must have distinct name@version; duplicates: {duplicates}")
    readers = {tid: t.metrics_reader for tid, t in zip(ids, trials, strict=True)}

    def build_config(ctx: RunContext) -> _SelectConfig:
        return _SelectConfig(
            metric=metric, mode=mode, trials={tid: ctx.path(t) for tid, t in zip(ids, trials, strict=True)}
        )

    def choose(config: _SelectConfig) -> dict[str, Any]:
        maximize = config.mode == "max"
        scores: dict[str, float] = {}
        summaries: dict[str, Mapping[str, Any]] = {}
        best: tuple[str, float, str] | None = None
        for trial_id, trial_path in config.trials.items():
            summary = readers[trial_id](trial_path)
            summaries[trial_id] = summary
            score = summary[config.metric]
            scores[trial_id] = score
            if best is None or (score > best[1] if maximize else score < best[1]):
                best = (trial_id, score, trial_path)
        assert best is not None  # config.trials is non-empty (guarded at build time)
        winner_id, winner_score, winner_path = best
        return {
            "winner": winner_id,
            "score": winner_score,
            "winner_path": winner_path,
            "metrics": dict(summaries[winner_id]),
            "scores": scores,
        }

    return Checkpoint(name=name, version=version, recipe=Recipe(fn=choose, build_config=build_config, deps=trials))
