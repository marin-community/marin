# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass


@dataclass(frozen=True)
class TraceMaskedEvalOutput:
    """Executor artifact produced by a completed trace-masked evaluation step."""

    results_path: str
