# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class TokenSequenceStats:
    """Compact Phase 0 summary for one tokenization family."""

    num_examples: int
    min_length: int
    max_length: int
    mean_length: float
    median_length: float
    p90_length: float
    p95_length: float
    p99_length: float
    unique_tokens: int

    def to_dict(self) -> dict[str, int | float]:
        """Serialize the summary for JSON-friendly reporting."""

        return asdict(self)


@dataclass(frozen=True)
class ReconstructionMetrics:
    """Per-example reconstruction metrics for lossy tokenizers."""

    mse: float
    psnr: float
    ssim: float
    dssim: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class AggregateMetrics:
    """Aggregate summary for reconstruction metrics over a corpus."""

    mean: float
    median: float
    p90: float
    p95: float
    p99: float
    min: float
    max: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def compute_token_sequence_stats(sequences: Sequence[np.ndarray]) -> TokenSequenceStats:
    """Summarize sequence lengths and approximate vocabulary size."""

    if not sequences:
        raise ValueError("Expected at least one token sequence")

    arrays = [np.asarray(sequence, dtype=np.int32) for sequence in sequences]
    lengths = np.asarray([len(sequence) for sequence in arrays], dtype=np.int32)
    unique_tokens = len(np.unique(np.concatenate(arrays)))
    return TokenSequenceStats(
        num_examples=len(arrays),
        min_length=int(lengths.min()),
        max_length=int(lengths.max()),
        mean_length=float(lengths.mean()),
        median_length=float(np.median(lengths)),
        p90_length=float(np.percentile(lengths, 90)),
        p95_length=float(np.percentile(lengths, 95)),
        p99_length=float(np.percentile(lengths, 99)),
        unique_tokens=int(unique_tokens),
    )


def compute_reconstruction_metrics(reference: np.ndarray, reconstructed: np.ndarray) -> ReconstructionMetrics:
    """Compute pixel and perceptual-proxy reconstruction metrics."""

    reference_f = np.asarray(reference, dtype=np.float32)
    reconstructed_f = np.asarray(reconstructed, dtype=np.float32)
    if reference_f.shape != reconstructed_f.shape:
        raise ValueError(f"Mismatched shapes: {reference_f.shape} vs {reconstructed_f.shape}")

    mse = float(np.mean(np.square(reference_f - reconstructed_f)))
    if mse == 0.0:
        psnr = float("inf")
    else:
        psnr = float(20.0 * np.log10(255.0) - 10.0 * np.log10(mse))

    ssim = float(compute_ssim(reference_f, reconstructed_f))
    return ReconstructionMetrics(
        mse=mse,
        psnr=psnr,
        ssim=ssim,
        dssim=float((1.0 - ssim) / 2.0),
    )


def compute_ssim(reference: np.ndarray, reconstructed: np.ndarray) -> float:
    """Compute SSIM using a Gaussian window as a perceptual proxy."""

    reference = np.asarray(reference, dtype=np.float32)
    reconstructed = np.asarray(reconstructed, dtype=np.float32)
    if reference.shape != reconstructed.shape:
        raise ValueError(f"Mismatched shapes: {reference.shape} vs {reconstructed.shape}")

    k1 = 0.01
    k2 = 0.03
    data_range = 255.0
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    mu_x = gaussian_filter(reference, sigma=1.5, truncate=3.5)
    mu_y = gaussian_filter(reconstructed, sigma=1.5, truncate=3.5)

    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x_sq = gaussian_filter(reference * reference, sigma=1.5, truncate=3.5) - mu_x_sq
    sigma_y_sq = gaussian_filter(reconstructed * reconstructed, sigma=1.5, truncate=3.5) - mu_y_sq
    sigma_xy = gaussian_filter(reference * reconstructed, sigma=1.5, truncate=3.5) - mu_xy

    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    ssim_map = numerator / np.maximum(denominator, 1e-12)
    return float(np.clip(ssim_map.mean(), -1.0, 1.0))


def summarize_metric(values: Sequence[float]) -> AggregateMetrics:
    """Summarize a scalar metric across a set of examples."""

    if not values:
        raise ValueError("Expected at least one metric value")
    array = np.asarray(values, dtype=np.float64)
    return AggregateMetrics(
        mean=float(array.mean()),
        median=float(np.median(array)),
        p90=float(np.percentile(array, 90)),
        p95=float(np.percentile(array, 95)),
        p99=float(np.percentile(array, 99)),
        min=float(array.min()),
        max=float(array.max()),
    )


__all__ = [
    "AggregateMetrics",
    "ReconstructionMetrics",
    "TokenSequenceStats",
    "compute_reconstruction_metrics",
    "compute_ssim",
    "compute_token_sequence_stats",
    "summarize_metric",
]
