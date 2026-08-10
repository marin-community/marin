# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris CLI package.

Re-exports the top-level ``iris`` Click group and the ``main`` entry point
(``iris = "iris.cli:main"``), which wraps the group to render auth errors cleanly.
"""

from iris.cli.main import iris, main

__all__ = ["iris", "main"]
