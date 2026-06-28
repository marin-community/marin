# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hyperparameter sweeps: fan out one trial handle per grid point.

A sweep is fan-out over a grid of parameters; :func:`grid` enumerates the points and
:func:`sweep` builds one handle per point. There is no framework ``select``: selection is
ordinary code over the resolved, typed outputs — e.g. reduce ``trial.resolve()`` by
``LevanterCheckpoint.training_metrics().eval_loss`` — and the chosen result is persisted, when
wanted, with the low-level :func:`marin.execution.artifact.write_artifact`.
"""

import itertools
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

T = TypeVar("T")


def grid(**axes: Sequence[Any]) -> list[dict[str, Any]]:
    """The Cartesian product of named axes, as a list of parameter dicts.

    ``grid(learning_rate=[1e-3, 3e-3], weight_decay=[0.0, 0.1])`` yields the four
    ``{"learning_rate": ..., "weight_decay": ...}`` combinations, in row-major order
    (the last axis varies fastest).
    """
    keys = list(axes)
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*axes.values())]


def sweep(trial: Callable[..., T], **axes: Sequence[Any]) -> list[T]:
    """One trial handle per grid point: ``trial(**params)`` for each combination.

    ``axes`` are the swept dimensions (see :func:`grid`); ``trial`` maps one parameter set to a
    handle. The trial must fold the swept values into both its config (so each grid point gets a
    distinct fingerprint) and its ``name`` (so each gets a distinct, readable address).
    """
    return [trial(**params) for params in grid(**axes)]
