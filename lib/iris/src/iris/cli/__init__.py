# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris CLI package.

Re-exports the top-level ``iris`` Click group so that the entry point
``iris = "iris.cli:iris"`` keeps working.
"""

from iris.cli.main import iris

__all__ = ["iris"]
