# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The lowered task envelope: a validated TaskSpec and its runtime machine shapes.

A rebuilt task publishes one envelope row per task; the rollout runtime, the
snapshot counter and the liveness runner all read that envelope through these
records, so its schema is checked at load rather than walked by string keys.
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass

from taskcompendium.models import TaskSpec

LOWERED_TASK_SPEC_COLUMN = "lowered_task_spec"
"""The parquet column carrying a rebuilt task's envelope."""


@dataclass(frozen=True)
class MachineShape:
    """The cpus, memory, storage and gpu count a runtime declares for one machine."""

    cpus: int | None = None
    memory_mb: int | None = None
    storage_mb: int | None = None
    gpus: int = 0

    @classmethod
    def from_mapping(cls, shape: Mapping) -> "MachineShape":
        return cls(
            cpus=shape.get("cpus"),
            memory_mb=shape.get("memory_mb"),
            storage_mb=shape.get("storage_mb"),
            gpus=shape.get("gpus", 0),
        )


@dataclass(frozen=True)
class LoweredSpec:
    """One lowered task envelope: its validated TaskSpec and runtime machine shapes."""

    task: TaskSpec
    task_machine: MachineShape
    verifier_machine: MachineShape

    @classmethod
    def from_document(cls, document: dict) -> "LoweredSpec":
        runtime = document.get("runtime") or {}
        return cls(
            TaskSpec.model_validate(document.get("task") or {}),
            MachineShape.from_mapping(runtime.get("task_machine") or {}),
            MachineShape.from_mapping(runtime.get("verifier_machine") or {}),
        )

    def envelope(self) -> dict:
        """The JSON envelope the rollout runtime and the measurement code consume."""
        return {
            "task": self.task.model_dump(mode="json"),
            "runtime": {
                "task_machine": asdict(self.task_machine),
                "verifier_machine": asdict(self.verifier_machine),
            },
        }
