# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""What every exporter needs: the public bucket and task ids and the aggregate weights."""

import dataclasses
from pathlib import Path

from experiments.domain_phase_mix.release import naming


@dataclasses.dataclass
class ExportContext:
    buckets: list[str]
    tasks: list[str]
    weights: dict[str, dict[str, float]]


def required_sources() -> list[Path]:
    """The files behind every exporter's names and aggregate weights (`naming`)."""
    return [naming.BUCKET_NAMES, naming.TASK_METADATA, naming.OBJECTIVE_WEIGHTS]


def build() -> ExportContext:
    return ExportContext(buckets=naming.BUCKETS, tasks=naming.TASKS, weights=naming.task_weights())
