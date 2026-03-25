# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Compatibility shim for the historical attention module path."""

from .pallas_gpu import DEFAULT_MASK_VALUE, BlockSizes, mha

__all__ = [
    "BlockSizes",
    "DEFAULT_MASK_VALUE",
    "mha",
]
