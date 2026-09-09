# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline grouped-design replay; benchmark policy stays outside the optimizer."""

import argparse
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from experiments.datakit.mixprior.hf import Data, load_data
from experiments.datakit.mixprior.model import fit
from experiments.datakit.mixprior.objective import fit_objective
from experiments.datakit.mixprior.search import Predictor, posterior_mean


def replay(
    data: Data,
    prefixes: list[int],
    fit_model: Callable[[Data, np.ndarray, np.ndarray], Predictor] = fit,
) -> list[dict]:
    """Fit prefix designs and rank the remaining measured mixtures.

    Replicate designs stay together. The objective uses fixed full-campaign
    reference/noise calibration, so this is a selection replay, not a fresh
    evaluation of calibration on unseen data.
    """
    values, variances = fit_objective(data)(data.outcomes)
    flat = np.round(data.weights.reshape(len(values), -1), 12)
    _, first, groups = np.unique(flat, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first)
    precision = np.bincount(groups, weights=1 / variances)
    y = np.bincount(groups, weights=values / variances) / precision
    variance = 1 / precision
    first, y, variance = first[order], y[order], variance[order]
    data = replace(
        data,
        weights=data.weights[first],
        outcomes=data.outcomes[first],
        groups=[data.groups[i] for i in first],
        observation_ids=[data.observation_ids[i] for i in first],
    )
    rows = []
    for prefix in prefixes:
        if not 2 <= prefix < len(y):
            raise ValueError("Each prefix must leave both training and held-out designs")
        train = replace(
            data,
            weights=data.weights[:prefix],
            outcomes=data.outcomes[:prefix],
            groups=data.groups[:prefix],
            observation_ids=data.observation_ids[:prefix],
        )
        model = fit_model(train, y[:prefix], variance[:prefix])
        predictions = posterior_mean(model, data.weights[prefix:])
        actual = y[prefix:]
        winner = int(predictions.argmax())
        rows.append(
            {
                "prefix": prefix,
                "held_out": len(actual),
                "spearman": float(spearmanr(predictions, actual).statistic),
                "winner_rank": int((actual > actual[winner]).sum() + 1),
                "winner_regret": float(actual.max() - actual[winner]),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--swarm", required=True)
    parser.add_argument("--prefixes", nargs="+", type=int, default=[128, 384])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay(load_data(args.data, args.swarm), args.prefixes)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
