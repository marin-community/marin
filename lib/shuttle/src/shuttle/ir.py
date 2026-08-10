# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared typed IR vocabulary."""

from enum import StrEnum


class DType(StrEnum):
    """Dtypes whose precision affects rewrite legality."""

    BOOL = "bool"
    BF16 = "bf16"
    FP32 = "fp32"
    FP64 = "fp64"
    INT32 = "int32"
