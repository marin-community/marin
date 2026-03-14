# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""DeepEP-backed JAX kernel helpers."""

from .layout_ffi import deepep_get_dispatch_layout

__all__ = ["deepep_get_dispatch_layout"]
