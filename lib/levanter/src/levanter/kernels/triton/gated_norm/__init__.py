# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Fused ``GatedNorm(RMSNorm(x))``: a restructured backward with Triton elementwise steps, and its reference."""

from .api import Implementation, gated_rms_norm
from .reference import gated_rms_norm_reference

__all__ = [
    "Implementation",
    "gated_rms_norm",
    "gated_rms_norm_reference",
]
