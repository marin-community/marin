# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Task selectors and response models used by the frozen outer-fold experiment."""

import json
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from experiments.evaluation.mini_policy.costs import covered_candidates


@dataclass
class Mirt:
    ability: np.ndarray
    discrimination: np.ndarray
    intercept: np.ndarray
    continuous: np.ndarray
    expected_counts: np.ndarray
    converged: bool
    iterations: int
    loss: float
    gradient_error: float


def standardized(x):
    mean = np.nanmean(x, axis=0)
    mean = np.nan_to_num(mean, nan=0.5)
    scale = np.maximum(np.nanstd(x, axis=0), 0.15)
    return (np.where(np.isfinite(x), x, mean) - mean) / scale, mean, scale


def random_order(pool, tasks, seed):
    rng = np.random.default_rng(seed)
    buckets = {}
    for j in pool:
        task = tasks[j]
        label = json.dumps(task.get("strata") or {}, sort_keys=True, default=str)
        buckets.setdefault(task["benchmark"], {}).setdefault(label, []).append(j)
    for strata in buckets.values():
        for values in strata.values():
            rng.shuffle(values)
    per_benchmark = {}
    for benchmark, strata in buckets.items():
        keys = list(strata)
        rng.shuffle(keys)
        order = []
        while keys:
            next_keys = []
            for key in keys:
                if strata[key]:
                    order.append(strata[key].pop())
                if strata[key]:
                    next_keys.append(key)
            keys = next_keys
        per_benchmark[benchmark] = order
    benchmarks = list(per_benchmark)
    rng.shuffle(benchmarks)
    order = []
    while benchmarks:
        remaining = []
        for benchmark in benchmarks:
            values = per_benchmark[benchmark]
            if values:
                order.append(values.pop(0))
            if values:
                remaining.append(benchmark)
        benchmarks = remaining
    return np.asarray(order, dtype=int)


def supervised_order(x, y, pool, size, seed, method):
    rng = np.random.default_rng(seed)
    z, _, _ = standardized(x[:, pool])
    target = (y - y.mean(0)) / np.maximum(y.std(0), 0.05)
    if method == "regularized":
        kernel = z @ z.T / z.shape[1]
        dual = np.linalg.solve(kernel + np.eye(len(z)), target)
        relevance = np.linalg.norm(z.T @ dual, axis=1)
    elif method == "mi_mrmr":
        teacher = target.mean(1)
        ybin = np.searchsorted(np.quantile(teacher, [1 / 3, 2 / 3]), teacher)
        xbin = np.where(x[:, pool] <= 0, 0, np.where(x[:, pool] >= 1, 3, np.where(x[:, pool] < 0.5, 1, 2)))
        valid = np.isfinite(x[:, pool])
        joint = np.empty((4, 3, len(pool)))
        for a in range(4):
            for b in range(3):
                joint[a, b] = np.sum((xbin == a) & (ybin[:, None] == b) & valid, axis=0) + 0.5
        joint /= joint.sum((0, 1), keepdims=True)
        independent = joint.sum(1, keepdims=True) * joint.sum(0, keepdims=True)
        relevance = np.sum(joint * np.log(joint / independent), axis=(0, 1))
    else:
        raise ValueError(method)
    relevance /= max(float(relevance.max()), 1e-9)
    # Ridge leverage rewards response directions not already represented.
    inverse = np.eye(len(z))
    selected = []
    available = np.ones(len(pool), dtype=bool)
    for _ in range(size):
        candidates = np.flatnonzero(available)
        if len(candidates) > 2048:
            candidates = rng.choice(candidates, 2048, replace=False)
        vectors = z[:, candidates].T
        leverage = np.maximum(np.einsum("ij,jk,ik->i", vectors, inverse, vectors), 0)
        utility = relevance[candidates] * np.log1p(leverage)
        best = int(candidates[int(np.argmax(utility))])
        selected.append(int(pool[best]))
        available[best] = False
        v = z[:, best]
        projected = inverse @ v
        inverse -= np.outer(projected, projected) / (1 + v @ projected)
    return np.asarray(selected)


def ridge_prediction(train, test, target, chosen):
    z, mean, scale = standardized(train[:, chosen])
    new = (np.where(np.isfinite(test[:, chosen]), test[:, chosen], mean) - mean) / scale
    center = target.mean(0)
    dual = np.linalg.solve(z @ z.T + 10 * np.eye(len(z)), target - center)
    return np.clip(center + new @ z.T @ dual, 0, 1)


def fit_mirt(response, counts, benchmarks, continuous, dimension, seed, mask=None, maxiter=160):
    n, m = response.shape
    observed = np.isfinite(response)
    if mask is not None:
        observed &= mask
    y = np.nan_to_num(response, nan=0.5)
    weights = (observed * np.maximum(counts, 1)).astype(float)
    for benchmark in set(benchmarks):
        columns = np.array([b == benchmark for b in benchmarks])
        total = weights[:, columns].sum()
        if total:
            weights[:, columns] /= total
    weights *= n * m / max(weights.sum(), 1)
    weights /= n
    mean = np.divide(np.sum(y * observed, axis=0), observed.sum(0), out=np.full(m, 0.5), where=observed.sum(0) > 0)
    b0 = np.log(np.clip(mean, 0.02, 0.98) / np.clip(1 - mean, 0.02, 0.98))
    centered = np.where(observed, y - mean, 0)
    eigen, u = np.linalg.eigh(centered @ centered.T)
    order = np.argsort(-eigen)[:dimension]
    theta = u[:, order] * np.sqrt(n)
    a = centered.T @ theta / n * 2
    parameter_scale = np.sqrt(m / n)
    start = np.concatenate([theta.ravel(), a.ravel() / parameter_scale, b0 / parameter_scale])

    def unpack(v):
        return (
            v[: n * dimension].reshape(n, dimension),
            v[n * dimension : n * dimension + m * dimension].reshape(m, dimension) * parameter_scale,
            v[-m:] * parameter_scale,
        )

    def objective(v):
        theta, a, b = unpack(v)
        eta = theta @ a.T + b
        p = expit(eta)
        loss = np.logaddexp(0, eta) - y * eta
        gradient = p - y
        loss[:, continuous] = 2 * (p[:, continuous] - y[:, continuous]) ** 2
        gradient[:, continuous] = 4 * (p[:, continuous] - y[:, continuous]) * p[:, continuous] * (1 - p[:, continuous])
        loss_value = float(
            np.sum(weights * loss) / m + 0.02 * np.mean(theta**2) + 0.02 * np.mean(a**2) + 0.005 * np.mean((b - b0) ** 2)
        )
        residual = weights * gradient / m
        gt = residual @ a + 0.04 * theta / theta.size
        ga = residual.T @ theta + 0.04 * a / a.size
        gb = residual.sum(0) + 0.01 * (b - b0) / m
        return loss_value, np.concatenate([gt.ravel(), ga.ravel() * parameter_scale, gb * parameter_scale])

    # Check independent parameter blocks before trusting the analytic gradient.
    _, initial_gradient = objective(start)
    rng = np.random.default_rng(seed)
    gradient_error = 0.0
    for begin, end in [(0, n * dimension), (n * dimension, n * dimension + m * dimension), (len(start) - m, len(start))]:
        direction = np.zeros_like(start)
        direction[begin:end] = rng.normal(size=end - begin)
        direction /= np.linalg.norm(direction)
        epsilon = 1e-4
        numeric = (objective(start + epsilon * direction)[0] - objective(start - epsilon * direction)[0]) / (2 * epsilon)
        analytic = float(initial_gradient @ direction)
        error = abs(numeric - analytic) / max(abs(numeric), abs(analytic), 1e-7)
        gradient_error = max(gradient_error, error)
        if abs(numeric - analytic) > 1e-8 + 1e-4 * max(abs(numeric), abs(analytic)):
            raise ArithmeticError(f"MIRT gradient check failed: {numeric} != {analytic}")
    fit = minimize(
        objective,
        start,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "gtol": 1e-7, "ftol": 1e-9, "maxcor": 7},
    )
    theta, a, b = unpack(fit.x)
    return Mirt(
        theta,
        a,
        b,
        np.array(continuous),
        np.maximum(np.median(counts, axis=0), 1),
        bool(fit.success),
        int(fit.nit),
        float(fit.fun),
        gradient_error,
    )


def estimate_ability(model, selected, response, counts, initial=None):
    chosen = np.asarray(selected)
    valid = np.isfinite(response[chosen])
    chosen = chosen[valid]
    a = model.discrimination[chosen]
    b = model.intercept[chosen]
    y = response[chosen]
    w = np.maximum(counts[chosen], 1)
    continuous = model.continuous[chosen]

    def objective(theta):
        eta = a @ theta + b
        p = expit(eta)
        loss = np.logaddexp(0, eta) - y * eta
        gradient = p - y
        loss[continuous] = 2 * (p[continuous] - y[continuous]) ** 2
        gradient[continuous] = 4 * (p[continuous] - y[continuous]) * p[continuous] * (1 - p[continuous])
        return float(np.sum(w * loss) + 0.1 * np.sum(theta**2)), a.T @ (w * gradient) + 0.2 * theta

    start = np.zeros(a.shape[1]) if initial is None else initial
    return minimize(objective, start, jac=True, method="L-BFGS-B", options={"maxiter": 80, "gtol": 1e-7}).x


def information_order(model, pool, size, seed, response=None, counts=None, costs=None, domains=None):
    rng = np.random.default_rng(seed)
    available = np.ones(len(pool), dtype=bool)
    selected = []
    inverse = np.eye(model.ability.shape[1])
    theta = np.zeros(model.ability.shape[1])
    checkpoints = {}
    static = response is None
    if static:
        p = expit(model.ability @ model.discrimination[pool].T + model.intercept[pool])
        information = np.mean(p * (1 - p), axis=0)
        information[model.continuous[pool]] = 4 * np.mean(
            (p[:, model.continuous[pool]] * (1 - p[:, model.continuous[pool]])) ** 2, axis=0
        )
        information *= model.expected_counts[pool]
    for step in range(size):
        if not static and selected and step % 10 == 0:
            theta = estimate_ability(model, selected, response, counts, theta)
            seen = np.asarray(selected)[np.isfinite(response[selected])]
            a = model.discrimination[seen]
            p = expit(a @ theta + model.intercept[seen])
            info = p * (1 - p)
            continuous = model.continuous[seen]
            info[continuous] = 4 * (p[continuous] * (1 - p[continuous])) ** 2
            inverse = np.linalg.inv(np.eye(len(theta)) + a.T @ ((info * np.maximum(counts[seen], 1))[:, None] * a))
        candidates = covered_candidates(np.flatnonzero(available), pool, domains, step)
        if len(candidates) > 2048:
            candidates = rng.choice(candidates, 2048, replace=False)
        actual = pool[candidates]
        a = model.discrimination[actual]
        if static:
            info = information[candidates]
        else:
            p = expit(a @ theta + model.intercept[actual])
            info = p * (1 - p)
            continuous = model.continuous[actual]
            info[continuous] = 4 * (p[continuous] * (1 - p[continuous])) ** 2
            info *= model.expected_counts[actual]
        leverage = np.einsum("ij,jk,ik->i", a, inverse, a) * info
        utility = leverage if costs is None else leverage / costs[actual]
        position = int(np.argmax(utility))
        best = int(candidates[position])
        task = int(pool[best])
        selected.append(task)
        available[best] = False
        v = model.discrimination[task] * np.sqrt(max(info[position], 1e-12))
        projected = inverse @ v
        inverse -= np.outer(projected, projected) / (1 + v @ projected)
        if step + 1 in {210, 840} and not static:
            checkpoints[step + 1] = estimate_ability(model, selected, response, counts, theta)
    return np.asarray(selected), checkpoints


def predict_panel(model, theta, task_benchmarks, target_benchmarks, eligible):
    p = expit(model.discrimination @ theta + model.intercept)
    return np.array([np.mean(p[(task_benchmarks == b) & eligible]) for b in target_benchmarks])
