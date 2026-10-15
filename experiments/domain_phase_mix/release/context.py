# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""What every exporter needs: the observation registry, the public id lists and the aggregate weights."""

import dataclasses
from pathlib import Path

from experiments.domain_phase_mix.release import keys, naming, paths


@dataclasses.dataclass
class ExportContext:
    registry: keys.ObservationRegistry
    buckets: list[str]
    tasks: list[str]
    weights: dict[str, dict[str, float]]


def required_sources() -> list[Path]:
    """The files behind every exporter's names and aggregate weights (`naming`)."""
    return [naming.BUCKET_NAMES, naming.TASK_METADATA, naming.OBJECTIVE_WEIGHTS]


def build(private_dir: Path = paths.PRIVATE) -> ExportContext:
    salt = keys.load_salt(private_dir / "salt")
    registry = keys.ObservationRegistry(salt=salt, map_path=private_dir / "observation_map.csv")
    return ExportContext(registry=registry, buckets=naming.BUCKETS, tasks=naming.TASKS, weights=naming.task_weights())
