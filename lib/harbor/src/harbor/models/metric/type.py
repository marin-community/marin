# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class MetricType(str, Enum):
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    MEAN = "mean"
    UV_SCRIPT = "uv-script"
