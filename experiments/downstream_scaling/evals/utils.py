# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for downstream-scaling evals."""

from __future__ import annotations

from marin.execution.executor import InputName, MirroredValue, versioned


def version_path(path: str | InputName | MirroredValue) -> str | InputName | MirroredValue:
    if isinstance(path, str):
        return versioned(path)  # type: ignore[return-value]
    return path
