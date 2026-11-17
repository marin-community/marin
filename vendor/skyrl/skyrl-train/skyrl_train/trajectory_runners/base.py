from typing import Protocol

from skyrl_train.trajectory_runners.types import (
    BatchMetadata as BatchMetadata,
    ConversationType as ConversationType,
    TrajectoryBatch as TrajectoryBatch,
    TrajectoryID as TrajectoryID,
    TrajectoryRequestBatch as TrajectoryRequestBatch,
    TrainingPhase as TrainingPhase,
)
from skyrl_train.rollouts.buffer import RolloutTask, RolloutWriter
from skyrl_train.trajectory_runners.trajectory_retention import RetentionSink


class TrajectoryRunner(Protocol):
    """Training batches, buffer submission, and worker resource lifecycle."""

    async def run(self, input_batch: TrajectoryRequestBatch) -> TrajectoryBatch: ...

    async def run_task(self, task: RolloutTask, writer: RolloutWriter) -> int:
        """Commit a leased prompt group and return its response token count."""
        ...

    def set_trajectory_sink(self, sink: RetentionSink) -> None: ...

    async def startup(self) -> None: ...

    async def shutdown(self) -> None: ...

    async def start_eval_session(self, *, run_name: str, eval_step: int, val_set_name: str | None = None) -> None: ...

    async def stop_eval_session(self) -> None: ...
