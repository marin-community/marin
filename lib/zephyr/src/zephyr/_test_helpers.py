# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Helpers shared by tests to avoid pickling import issues on Ray workers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SampleDataclass:
    """Tiny dataclass used in serialization round-trip tests."""

    name: str
    value: int


__all__ = ["SampleDataclass"]
