# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Top-level ``ray`` package that forwards to the fakeray shim.

When this distribution is installed in place of real Ray (via uv
``override-dependencies``), ``import ray`` and ``from ray import ...`` resolve
here, and every public name is re-exported from :mod:`fakeray`. The
``ray.exceptions`` submodule is provided as a real module file so that
``import ray.exceptions`` (which smallpond does at module load) resolves through
``sys.modules`` — a package ``__getattr__`` alone does not serve dotted
submodule imports.
"""

from __future__ import annotations

from fakeray import (
    ObjectRef,
    RemoteFunction,
    exceptions,
    get,
    init,
    is_initialized,
    put,
    remote,
    shutdown,
    timeline,
    wait,
)

__all__ = [
    "ObjectRef",
    "RemoteFunction",
    "exceptions",
    "get",
    "init",
    "is_initialized",
    "put",
    "remote",
    "shutdown",
    "timeline",
    "wait",
]
