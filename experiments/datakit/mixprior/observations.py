# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Group repeated mixture designs without splitting replicates across model fits."""

from dataclasses import dataclass, replace

import numpy as np

from experiments.datakit.mixprior.hf import Data


@dataclass
class GroupedObservations:
    data: Data
    counts: np.ndarray


def group_observations(data: Data) -> GroupedObservations:
    """Order designs by first appearance and aggregate their replicated outcomes."""
    flat = np.round(data.weights.reshape(len(data.weights), -1), 12)
    _, first, groups = np.unique(flat, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first)
    counts = np.bincount(groups)
    outcomes = np.column_stack([np.bincount(groups, weights=column) / counts for column in data.outcomes.T])
    first = first[order]
    grouped = replace(
        data,
        weights=data.weights[first],
        outcomes=outcomes[order],
        groups=[data.groups[i] for i in first],
        observation_ids=[data.observation_ids[i] for i in first],
    )
    return GroupedObservations(grouped, counts[order])
