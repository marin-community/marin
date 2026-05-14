# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass


@dataclass(frozen=True)
class TraceLabeledEvalOutput:
    """Executor artifact produced by a completed trace-labeled evaluation step."""

    results_path: str
