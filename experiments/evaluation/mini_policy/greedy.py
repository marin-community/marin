# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Training-family cross-validated greedy selection with a tournament objective."""

import numpy as np
from sklearn.model_selection import GroupKFold

from experiments.evaluation.mini_policy.costs import covered_candidates
from experiments.evaluation.mini_policy.methods import standardized


def prediction_loss(predictions, actual, sems, predicted_sem):
    # Candidate, validation model, benchmark. The discrete term follows the
    # full tournament's decisive benchmark comparisons; MAE breaks plateaus.
    clipped = np.clip(predictions, 0, 1)
    mae = np.mean(np.abs(clipped - actual[None, :, :]), axis=(1, 2))
    if len(actual) < 2:
        return mae
    losses = []
    for i in range(len(actual)):
        for j in range(i):
            threshold = 1.96 * np.hypot(sems[i], sems[j]) / np.sqrt(0.9)
            difference = actual[i] - actual[j]
            truth = np.where(difference > threshold, 1, np.where(difference < -threshold, -1, 0))
            delta = clipped[:, i] - clipped[:, j]
            cutoff = 1.96 * np.sqrt(2) * predicted_sem / np.sqrt(0.9)
            estimated = np.where(delta > cutoff, 1, np.where(delta < -cutoff, -1, 0))
            decisive = truth != 0
            if decisive.any():
                losses.append(np.mean(estimated[:, decisive] != truth[None, decisive], axis=1))
    return np.mean(losses, axis=0) + 0.05 * mae if losses else mae


def greedy_cv_order(response, target, sems, groups, pool, size, seed, costs, domains=None):
    """Choose tasks using inner-family prediction improvement per estimated second."""
    rng = np.random.default_rng(seed)
    states = []
    splits = GroupKFold(n_splits=min(3, len(set(groups)))).split(response, groups=groups)
    for train, valid in splits:
        z, mean, scale = standardized(response[train][:, pool])
        new = (np.where(np.isfinite(response[valid][:, pool]), response[valid][:, pool], mean) - mean) / scale
        center = target[train].mean(0)
        states.append(
            {
                "x": z,
                "v": new,
                "inverse": np.eye(len(train)) / 10,
                "cross": np.zeros((len(valid), len(train))),
                "target": target[train] - center,
                "prediction": np.repeat(center[None, :], len(valid), axis=0),
                "actual": target[valid],
                "sem": sems[valid],
                "predicted_sem": np.median(sems[train], axis=0),
            }
        )
    available = np.ones(len(pool), dtype=bool)
    selected = []
    for step in range(size):
        candidates = covered_candidates(np.flatnonzero(available), pool, domains, step)
        if len(candidates) > 1024:
            candidates = rng.choice(candidates, 1024, replace=False)
        losses = []
        baseline = []
        for state in states:
            u = state["x"][:, candidates].T
            v = state["v"][:, candidates].T
            projected = u @ state["inverse"]
            denominator = 1 + np.sum(projected * u, axis=1)
            correction = v - projected @ state["cross"].T
            coefficient = projected @ state["target"] / denominator[:, None]
            prediction = state["prediction"][None, :, :] + correction[:, :, None] * coefficient[:, None, :]
            losses.append(prediction_loss(prediction, state["actual"], state["sem"], state["predicted_sem"]))
            baseline.append(
                prediction_loss(state["prediction"][None, :, :], state["actual"], state["sem"], state["predicted_sem"])[
                    0
                ]
            )
        loss = np.mean(losses, axis=0)
        gain = float(np.mean(baseline)) - loss
        positive = gain > 0
        if positive.any():
            utility = np.where(positive, gain / costs[pool[candidates]], -np.inf)
            best = int(candidates[np.argmax(utility)])
        else:
            # An exact fixed budget still needs probes after discrete loss
            # saturates. Minimize predicted loss; prefer cheaper exact ties.
            order = np.lexsort((costs[pool[candidates]], loss))
            best = int(candidates[order[0]])
        selected.append(int(pool[best]))
        available[best] = False
        for state in states:
            u = state["x"][:, best]
            v = state["v"][:, best]
            projected = state["inverse"] @ u
            denominator = 1 + u @ projected
            state["prediction"] += np.outer(v - state["cross"] @ projected, projected @ state["target"]) / denominator
            state["inverse"] -= np.outer(projected, projected) / denominator
            state["cross"] += np.outer(v, u)
    return np.array(selected)
