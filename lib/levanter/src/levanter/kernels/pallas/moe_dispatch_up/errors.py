# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared exceptions for the MoE dispatch-up subkernel."""


class MosaicGpuUnsupportedError(NotImplementedError):
    """Raised when the MoE dispatch-up Mosaic GPU backend is requested but unavailable."""
