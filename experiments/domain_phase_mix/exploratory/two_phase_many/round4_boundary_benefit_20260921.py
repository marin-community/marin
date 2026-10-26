# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Shared target fraction of learned benefit at the phase boundary.

For total benefit B, actual prefix benefit X, and tied prefix benefit T, the
benefit column is -B + kappa * ((X-q*B)**2 - (T-q*B)**2). Each bucket's existing
nonnegative MARINER benefit coefficient multiplies this column; no independent
phase amplitude is fitted. Positive curvature is in benefit coordinates, not
necessarily physical mixture coordinates. The construction assumes the frozen
constant-exposure geometry contract.
"""

import math

import benchmark_two_phase_20260921 as benchmark
import numpy as np
from nextgen_two_phase_surrogate_20260921 import check_geometry

mariner = benchmark.mariner


def design_matrix(panel: benchmark.data.BenchPanel, shape: dict[str, float], q: float, kappa: float) -> np.ndarray:
    """Return native-width aggregate features with a shared boundary penalty.

    Args:
        panel: Constant-exposure policies, including physical phase doses.
        shape: Native MARINER rate, power, and harm threshold.
        q: Target fraction of each bucket's final benefit acquired in phase zero.
        kappa: Nonnegative strength of the squared benefit-fraction departure.
    """
    check_geometry(panel)
    if not math.isfinite(q) or not 0 <= q <= 1 or not math.isfinite(kappa) or kappa < 0:
        raise ValueError("Require q in [0,1] and finite nonnegative kappa")
    aggregate = mariner.design_matrix(panel.epochs, shape)
    if kappa == 0:
        return aggregate
    width = len(panel.buckets)
    total = -aggregate[:, :width]
    prefix = mariner.benefit(panel.phase0_epochs, shape["rate"], shape["power"])
    tied = mariner.benefit(panel.alpha * panel.epochs, shape["rate"], shape["power"])
    # Factoring the squares avoids cancellation. Under constant exposure,
    # exactly equal phase weights imply exactly zero phase correction.
    delta = np.where(panel.phase0 == panel.phase1, 0.0, prefix - tied)
    aggregate[:, :width] += kappa * delta * (prefix + tied - 2 * q * total)
    return aggregate
