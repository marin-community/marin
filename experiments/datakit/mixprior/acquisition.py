# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Metric GP acquisition with additive domain and quality features."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from scipy.spatial.distance import pdist

from experiments.datakit.mixprior.hf import Data
from experiments.datakit.mixprior.metric import condition_metrics, expected_score, raw_features
from experiments.datakit.mixprior.model import KERNEL_VARIANCE, MAD_SCALE, kernel, squared_distances
from experiments.datakit.mixprior.objective import Objective

FEATURE_FLOOR = 0.01


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class KernelConfig:
    """Covariance amplitudes persisted with each fitted metric model."""

    matern: ArrayLike = KERNEL_VARIANCE
    intercept: ArrayLike = 25.0
    features: ArrayLike = 1.0
    trend: ArrayLike = 0.1


def standardization(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return features.mean(0), np.maximum(features.std(0), FEATURE_FLOOR) * np.sqrt(features.shape[1])


def root_length(weights: np.ndarray) -> float:
    distances = pdist(np.concatenate([np.sqrt(weights[:, 0]), np.sqrt(weights[:, 1])]))
    length = float(2 * np.median(distances[distances**2 > np.finfo(float).eps]))
    if not np.isfinite(length):
        raise ValueError("Feature fitting requires distinct phase profiles")
    return length


def memberships(data: Data, quality: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    domains = sorted(set(data.domains))
    domain = np.asarray([[name == value for value in domains] for name in data.domains], dtype=float)
    if quality.shape != (len(data.components),) or np.any(quality < 0):
        raise ValueError("Nonnegative integer quality levels must align with components")
    quality_membership = np.eye(int(quality.max()) + 1)[quality]
    return domain, quality_membership, domain * (quality >= 3)[:, None]


def trend_features(weights: jax.Array, domain: jax.Array, quality: jax.Array) -> jax.Array:
    return jnp.concatenate(
        [(weights @ domain).reshape(len(weights), -1), (weights @ quality).reshape(len(weights), -1)], axis=1
    )


def bounded_trend(features: jax.Array, center: jax.Array, scale: jax.Array) -> jax.Array:
    width = jnp.sqrt(features.shape[1])
    return 2 * jnp.tanh((features - center) / scale * width / 2) / width


def additive_kernel(
    roots: jax.Array,
    features: jax.Array,
    trend: jax.Array,
    other_roots: jax.Array,
    other_features: jax.Array,
    other_trend: jax.Array,
    length: jax.Array,
    lengths: jax.Array,
    config: KernelConfig,
) -> jax.Array:
    covariance = (
        kernel(roots, other_roots, length, config.matern) + config.intercept + config.trend * (trend @ other_trend.T)
    )
    first = roots.shape[1]
    ends = (0, first, first + first // 2, features.shape[1])
    block_count = len(ends) - 1
    for block in range(block_count):
        start, end = ends[block], ends[block + 1]
        covariance += jnp.exp(
            -0.5 * squared_distances(features[:, start:end], other_features[:, start:end]) / lengths[block] ** 2
        ) * (config.features / block_count)
    return covariance


@jax.tree_util.register_dataclass
@dataclass
class AdditiveMetricGP:
    kernel_config: KernelConfig
    domain: jax.Array
    quality: jax.Array
    high_quality: jax.Array
    feature_center: jax.Array
    feature_scale: jax.Array
    trend_center: jax.Array
    trend_scale: jax.Array
    train_roots: jax.Array
    train_features: jax.Array
    train_trend: jax.Array
    length: jax.Array
    lengths: jax.Array
    projection: jax.Array
    inverse: jax.Array
    alpha: jax.Array
    center: jax.Array
    scale: jax.Array
    target: jax.Array
    hinge: jax.Array
    bias: jax.Array
    variance_multiplier: jax.Array
    observation_noise: jax.Array

    @jax.jit
    def predict_metrics(self, weights: ArrayLike) -> tuple[jax.Array, jax.Array]:
        """Return calibrated metric means and latent variances in objective-standardized units."""
        weights = jnp.asarray(weights)
        roots = jnp.sqrt(weights).reshape(len(weights), -1)
        features = (raw_features(weights, self.high_quality) - self.feature_center) / self.feature_scale
        trend = bounded_trend(trend_features(weights, self.domain, self.quality), self.trend_center, self.trend_scale)
        covariance = additive_kernel(
            roots,
            features,
            trend,
            self.train_roots,
            self.train_features,
            self.train_trend,
            self.length,
            self.lengths,
            self.kernel_config,
        )
        mean = self.center + self.scale * (covariance @ self.alpha)
        prior = (
            self.kernel_config.matern
            + self.kernel_config.intercept
            + self.kernel_config.features
            + self.kernel_config.trend * jnp.sum(trend**2, axis=1)
        )
        variance = self.scale**2 * jnp.maximum(prior[:, None] - (covariance @ self.projection) ** 2 @ self.inverse, 0)
        variance = self.variance_multiplier * (variance + self.observation_noise) - self.observation_noise
        return mean + self.bias, variance

    @jax.jit
    def acquisition(self, weights: ArrayLike) -> jax.Array:
        mean, variance = self.predict_metrics(weights)
        return expected_score(mean, variance, self.target, self.hinge)


def fit_additive(
    data: Data,
    objective: Objective,
    counts: np.ndarray,
    device: jax.Device,
    *,
    kernel_config: KernelConfig = KernelConfig(),
) -> AdditiveMetricGP:
    """Fit grouped metric outcomes with replicate-scaled noise on the selected device."""
    if not jax.config.x64_enabled:
        raise ValueError("Metric acquisition requires jax_enable_x64=True")
    if objective.epsilon != 0 or not objective.target_mask.any() or objective.target_mask.all():
        raise ValueError("Metric acquisition requires targets, guardrails, and zero hinge tolerance")
    if counts.shape != (len(data.weights),) or not np.isfinite(counts).all() or np.any(counts <= 0):
        raise ValueError("Positive finite replicate counts must align with training designs")
    if not np.isfinite(data.outcomes[:, objective.columns]).all():
        raise ValueError("Metric observations must be finite")
    domain, quality_membership, high = memberships(data, data.quality)
    dd, qd, hd, weights = jax.device_put((domain, quality_membership, high, data.weights), device)
    features = np.asarray(raw_features(weights, hd))
    first = data.weights.shape[1] * data.weights.shape[2]
    ends = (0, first, first + first // 2, features.shape[1])
    centers, scales, lengths = [], [], []
    block_count = len(ends) - 1
    for block in range(block_count):
        raw = features[:, ends[block] : ends[block + 1]]
        center, scale = standardization(raw)
        centers.append(center)
        scales.append(scale)
        distances = pdist((raw - center) / scale)
        lengths.append(2 * np.median(distances[distances > 1e-12]))
    if not np.isfinite(lengths).all():
        raise ValueError("Additive fitting requires distinct features in each block")
    fc, fs = np.concatenate(centers), np.concatenate(scales)
    tc, ts = standardization(np.asarray(trend_features(weights, dd, qd)))
    values = (data.outcomes[:, objective.columns] - objective.mean) / objective.scale
    center = np.median(values, axis=0)
    scale = np.maximum(MAD_SCALE * np.median(np.abs(values - center), axis=0), 1)
    target = objective.target_mask / objective.target_mask.sum()
    hinge = target + (~objective.target_mask) / (~objective.target_mask).sum()
    fc, fs, tc, ts, features, length, lengths, center, scale, target, hinge = jax.device_put(
        (
            fc,
            fs,
            tc,
            ts,
            (features - fc) / fs,
            root_length(data.weights),
            np.array(lengths),
            center,
            scale,
            target,
            hinge,
        ),
        device,
    )
    roots = jnp.sqrt(weights).reshape(len(weights), -1)
    trend = bounded_trend(trend_features(weights, dd, qd), tc, ts)
    kernel_config = jax.device_put(kernel_config, device)
    covariance = additive_kernel(roots, features, trend, roots, features, trend, length, lengths, kernel_config)
    targets, noise, counts = jax.device_put(
        (
            np.clip((values - np.asarray(center)) / np.asarray(scale), -8, 8),
            np.diag(objective.noise_covariance) / objective.scale**2 / np.asarray(scale) ** 2,
            counts,
        ),
        device,
    )
    projection, inverse, alpha = condition_metrics(covariance, counts, targets, noise)
    return AdditiveMetricGP(
        kernel_config,
        dd,
        qd,
        hd,
        fc,
        fs,
        tc,
        ts,
        roots,
        features,
        trend,
        length,
        lengths,
        projection,
        inverse,
        alpha,
        center,
        scale,
        target,
        hinge,
        jax.device_put(np.zeros(len(objective.columns)), device),
        jax.device_put(np.ones(len(objective.columns)), device),
        jax.device_put(np.diag(objective.noise_covariance) / objective.scale**2, device),
    )
