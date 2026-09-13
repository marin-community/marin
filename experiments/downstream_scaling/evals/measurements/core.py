# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Composition protocol for downstream-scaling statistics."""

from __future__ import annotations

from typing import Protocol

from thalas.execution.executor import ExecutorStep, InputName, MirroredValue


class Statistic(Protocol):
    def make_statistic_step(
        self,
        *,
        name: str,
        prompts_path: str | InputName | MirroredValue,
        alg_output_path: str | InputName | MirroredValue,
    ) -> ExecutorStep:
        """Return a step whose output directory contains statistics.jsonl.gz."""
        ...
