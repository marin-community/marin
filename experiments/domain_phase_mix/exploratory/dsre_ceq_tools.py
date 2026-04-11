# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""DS-RE-CEQ fitting utilities with sensitivity extraction."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
from scipy.optimize import minimize

from experiments.domain_phase_mix.exploratory.general_scaling_models import (
    DatasetSpec,
    _as_3d,
    _broadcast_epoch_mult,
    _ces_mean_stable,
    _compute_epochs,
    _huber_delta,
    _parse_small,
    _pseudo_huber,
    _softplus_scaled,
)


@dataclass(frozen=True)
class DsreCeqArtifacts:
    """Fit artifacts needed for selection and diagnostics."""

    spec: DatasetSpec
    predict_fn: Callable[[np.ndarray], np.ndarray]
    forward_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
    final_p: np.ndarray
    n_params: int

    def jacobian(self, weights: np.ndarray, rel_eps: float = 1e-4) -> np.ndarray:
        """Return a finite-difference Jacobian of predictions wrt parameters."""
        weights_3d = _as_3d(weights)
        base = self.final_p
        steps = rel_eps * np.maximum(1.0, np.abs(base))
        jac = np.empty((weights_3d.shape[0], base.size), dtype=float)

        for idx, step in enumerate(steps):
            p_plus = base.copy()
            p_minus = base.copy()
            p_plus[idx] += step
            p_minus[idx] -= step
            plus = self.forward_fn(p_plus, weights_3d)
            minus = self.forward_fn(p_minus, weights_3d)
            jac[:, idx] = (plus - minus) / (2.0 * step)

        return jac


def fit_dsre_ceq_artifacts(
    spec: DatasetSpec,
    *,
    gate: bool = True,
    satiety: bool = True,
    per_domain_pi: bool = True,
    interference: bool = True,
    n_restarts: int = 8,
    seed: int = 0,
    maxiter: int = 500,
    reg: float = 1e-4,
) -> DsreCeqArtifacts:
    """Fit DS-RE-CEQ and expose artifacts for D-optimal design."""
    rng = np.random.default_rng(seed)
    weights = spec.weights
    targets = spec.y
    _, n_phases, n_domains = weights.shape

    epoch_multipliers = _broadcast_epoch_mult(spec.epoch_multipliers, n_phases, n_domains)
    epochs = _compute_epochs(weights, epoch_multipliers)
    small_domains = _parse_small(spec.small_domains, n_domains)
    huber_delta = _huber_delta(targets)

    use_gate = gate and interference
    use_lambda = interference
    n_pi_sets = n_domains if per_domain_pi else 1

    n_params = (
        3
        + (n_domains - 1)
        + 1
        + n_pi_sets * max(n_phases - 1, 0)
        + (max(n_phases - 1, 0) * n_domains if use_lambda else 0)
        + (n_domains if satiety else 0)
        + (n_phases if use_gate else 0)
        + 1
    )
    median_tau = float(np.median(epochs[:, :, small_domains].sum(axis=(1, 2))) + 1e-6)

    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits)
        exp = np.exp(shifted)
        return exp / exp.sum()

    def _sigmoid(value: float) -> float:
        return 1.0 / (1.0 + np.exp(-np.clip(value, -50.0, 50.0)))

    def unpack(params: np.ndarray):
        idx = 0
        c0 = float(params[idx])
        log_a = float(params[idx + 1])
        log_b = float(params[idx + 2])
        idx += 3

        logits = np.zeros(n_domains)
        if n_domains > 1:
            logits[: n_domains - 1] = params[idx : idx + (n_domains - 1)]
            idx += n_domains - 1
        domain_weights = _softmax(logits)

        rho = float(np.clip(5.0 * np.tanh(params[idx]), -10.0, 0.99))
        idx += 1

        phase_importance = np.zeros((n_domains, n_phases))
        if per_domain_pi:
            for domain_idx in range(n_domains):
                logits_free = np.array([])
                if n_phases > 1:
                    logits_free = params[idx : idx + (n_phases - 1)]
                    idx += n_phases - 1
                full = np.zeros(n_phases)
                if n_phases > 1:
                    full[: n_phases - 1] = logits_free
                phase_importance[domain_idx] = _softmax(full)
        else:
            shared = np.zeros(n_phases)
            if n_phases > 1:
                shared[: n_phases - 1] = params[idx : idx + (n_phases - 1)]
                idx += n_phases - 1
            shared_importance = _softmax(shared)
            for domain_idx in range(n_domains):
                phase_importance[domain_idx] = shared_importance

        interference_lambda = np.zeros((n_phases, n_domains))
        if use_lambda and n_phases > 1:
            raw = params[idx : idx + (n_phases - 1) * n_domains].reshape(n_phases - 1, n_domains)
            idx += (n_phases - 1) * n_domains
            interference_lambda[1:] = np.exp(np.clip(raw, -8.0, 8.0))

        if satiety:
            satiety_memory = np.array([_sigmoid(float(params[idx + domain_idx])) for domain_idx in range(n_domains)])
            idx += n_domains
        else:
            satiety_memory = np.ones(n_domains)

        if use_gate:
            conflict_gate = np.array([_sigmoid(float(params[idx + phase_idx])) for phase_idx in range(n_phases)])
            idx += n_phases
            conflict_gate[0] = 0.0
        elif interference:
            conflict_gate = np.ones(n_phases)
            conflict_gate[0] = 0.0
        else:
            conflict_gate = np.zeros(n_phases)

        tau = float(np.exp(np.clip(params[idx], -8.0, 8.0)))
        return (
            c0,
            log_a,
            log_b,
            domain_weights,
            rho,
            phase_importance,
            interference_lambda,
            satiety_memory,
            conflict_gate,
            tau,
        )

    def forward(params: np.ndarray, input_weights: np.ndarray) -> np.ndarray:
        (
            c0,
            log_a,
            log_b,
            domain_weights,
            rho,
            phase_importance,
            interference_lambda,
            satiety_memory,
            conflict_gate,
            tau,
        ) = unpack(params)
        utility_scale = float(np.exp(np.clip(log_a, -10.0, 10.0)))
        penalty_scale = float(np.exp(np.clip(log_b, -10.0, 10.0)))

        input_weights = _as_3d(input_weights)
        input_epochs = _compute_epochs(input_weights, epoch_multipliers)
        inverse_weights = 1.0 - input_weights

        cumulative = np.zeros_like(input_epochs)
        if n_phases > 1:
            cumulative[:, 1:, :] = np.cumsum(input_epochs[:, :-1, :], axis=1)

        prior = satiety_memory[None, None, :] * cumulative
        signal = np.log1p(prior + input_epochs) - np.log1p(prior)

        gated_lambda = interference_lambda * conflict_gate[:, None]
        conflict = inverse_weights * gated_lambda[None, :, :]
        suffix = np.cumsum(conflict[:, ::-1, :], axis=1)[:, ::-1]
        exp_term = np.zeros_like(conflict)
        if n_phases > 1:
            exp_term[:, :-1, :] = suffix[:, 1:, :]
        retention = np.exp(-exp_term)

        retained_state = np.zeros((input_weights.shape[0], n_domains))
        retained_exposure = np.zeros((input_weights.shape[0], n_domains))
        for domain_idx in range(n_domains):
            phase_weights = phase_importance[domain_idx][None, :]
            retained_state[:, domain_idx] = np.sum(
                phase_weights * retention[:, :, domain_idx] * signal[:, :, domain_idx],
                axis=1,
            )
            retained_exposure[:, domain_idx] = np.sum(
                phase_weights * retention[:, :, domain_idx] * input_epochs[:, :, domain_idx],
                axis=1,
            )

        utility = _ces_mean_stable(np.maximum(retained_state, 1e-12), domain_weights[None, :], rho)
        small_exposure = retained_exposure[:, small_domains].sum(axis=1)
        penalty = _softplus_scaled(small_exposure - tau, 1.0) ** 2
        return c0 - utility_scale * utility + penalty_scale * penalty

    def objective(params: np.ndarray) -> float:
        prediction = forward(params, weights)
        loss = float(np.sum(_pseudo_huber(prediction - targets, huber_delta)))
        loss += float(reg) * float(np.sum(params * params))
        return loss

    best_value = np.inf
    best_params = None

    for _ in range(n_restarts):
        params0 = np.zeros(n_params, dtype=float)
        idx = 0
        params0[idx] = float(np.median(targets))
        params0[idx + 1] = float(rng.normal(0.0, 1.0))
        params0[idx + 2] = float(rng.normal(-2.0, 1.0))
        idx += 3

        if n_domains > 1:
            params0[idx : idx + (n_domains - 1)] = rng.normal(0.0, 0.5, n_domains - 1)
            idx += n_domains - 1
        params0[idx] = float(rng.normal(0.0, 0.7))
        idx += 1

        n_pi_total = n_pi_sets * max(n_phases - 1, 0)
        if n_pi_total > 0:
            params0[idx : idx + n_pi_total] = rng.normal(0.0, 0.5, n_pi_total)
            idx += n_pi_total

        if use_lambda and n_phases > 1:
            params0[idx : idx + (n_phases - 1) * n_domains] = rng.normal(-1.0, 0.5, (n_phases - 1) * n_domains)
            idx += (n_phases - 1) * n_domains

        if satiety:
            params0[idx : idx + n_domains] = rng.normal(2.0, 0.6, n_domains)
            idx += n_domains

        if use_gate:
            params0[idx : idx + n_phases] = np.linspace(-2.0, 2.0, n_phases) + rng.normal(0.0, 0.6, n_phases)
            idx += n_phases

        params0[idx] = float(np.log(median_tau) + rng.normal(0.0, 0.3))

        try:
            result = minimize(
                objective,
                params0,
                method="L-BFGS-B",
                options={"maxiter": maxiter, "ftol": 1e-10},
            )
        except Exception:
            continue

        if np.isfinite(result.fun) and result.fun < best_value:
            best_value = float(result.fun)
            best_params = np.asarray(result.x, dtype=float)

    if best_params is None:
        raise RuntimeError("DS-RE-CEQ optimization failed to converge")

    final_params = best_params.copy()

    def predict(input_weights: np.ndarray) -> np.ndarray:
        return forward(final_params, _as_3d(input_weights))

    return DsreCeqArtifacts(
        spec=spec,
        predict_fn=predict,
        forward_fn=forward,
        final_p=final_params,
        n_params=n_params,
    )
