# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared style constants for two-phase-many convergence plots."""

from __future__ import annotations

GRP_COLOR = "#E24731"
REGMIX_COLOR = "#59A14F"
OLMIX_COLOR = "#CC79A7"
BEST_OBSERVED_BPB_COLOR = "#4C78A8"
PROPORTIONAL_BPB_COLOR = "#6C6F7D"

PREDICTED_LINESTYLE = "--"
VALIDATED_LINESTYLE = "-"


def model_bpb_color(model_name: str) -> str:
    """Return the canonical BPB track color for a convergence-model label."""
    normalized = model_name.lower()
    if "regmix" in normalized:
        return REGMIX_COLOR
    if "olmix" in normalized:
        return OLMIX_COLOR
    return GRP_COLOR
