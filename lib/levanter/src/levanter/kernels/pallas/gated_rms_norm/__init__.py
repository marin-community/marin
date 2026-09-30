# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Fused RMSNorm followed by a low-rank sigmoid gate (GatedNorm)."""

from .api import Implementation, gated_rms_norm
from .config import GatedRmsNormBlockSizes
from .pallas_gpu import expected_bytes_moved, pallas_gated_rms_norm_available
from .reference import gated_rms_norm_reference

__all__ = [
    "GatedRmsNormBlockSizes",
    "Implementation",
    "expected_bytes_moved",
    "gated_rms_norm",
    "gated_rms_norm_reference",
    "pallas_gated_rms_norm_available",
]
