# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The mixing methods that the paper compares, by their names in the dataset and the names the paper prints."""

from __future__ import annotations

from enum import StrEnum


class Method(StrEnum):
    """The methods the figures and tables compare, by their names in the dataset's method, model and predictor
    columns."""

    PROPORTIONAL = "proportional"
    UNIMAX = "unimax8"
    OLMIX = "olmix"
    MARINER = "mariner"
    REGMIX = "regmix"  # tuned RegMix
    REGMIX_RELEASED = "regmix_official"
    SPLINE = "spline"
    QUADRATIC = "quadratic"
    KERNEL_RIDGE = "krr"
    MARINER_CONVEX = "mariner_cvx"
    MARINER_ADDITIVE = "mariner_add"
    MARINER_BUCKET_THRESHOLDS = "mariner_pbt"


METHOD_LABELS = {
    Method.PROPORTIONAL: "Proportional",
    Method.UNIMAX: "UniMax-8",
    Method.OLMIX: "Olmix",
    Method.MARINER: "MARINER",
    Method.REGMIX: "RegMix (tuned)",
    Method.REGMIX_RELEASED: "RegMix (released)",
    Method.SPLINE: "Cubic spline",
    Method.QUADRATIC: "Quadratic",
    Method.KERNEL_RIDGE: "Hellinger kernel ridge",
    Method.MARINER_CONVEX: "MARINER, fully convex objective",
    Method.MARINER_ADDITIVE: "MARINER, additive form",
    Method.MARINER_BUCKET_THRESHOLDS: "MARINER, per-bucket thresholds",
}
