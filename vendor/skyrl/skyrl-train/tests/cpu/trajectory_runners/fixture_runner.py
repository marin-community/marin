"""Shared lifecycle for scripted test batches."""

from types import MappingProxyType

from skyrl_train.rollouts.buffer import RolloutTask, RolloutWriter, write_trajectory_batch
from skyrl_train.rollouts.finalization import finalize_trajectory_batch
from skyrl_train.trajectory_runners.trajectory_retention import RetentionSink
from skyrl_train.trajectory_runners.types import TrajectoryBatch, TrajectoryRequestBatch


class FixtureRunner:
    trajectory_runner_cfg = MappingProxyType({})
    trajectory_sink: RetentionSink | None = None

    async def run(self, input_batch: TrajectoryRequestBatch) -> TrajectoryBatch:
        output = await self._run(input_batch)
        return await finalize_trajectory_batch(input_batch, output, self.trajectory_runner_cfg, self.trajectory_sink)

    async def run_task(self, task: RolloutTask, writer: RolloutWriter) -> int:
        output = await self.run(task.request)
        return await write_trajectory_batch(task, writer, output)

    def set_trajectory_sink(self, sink: RetentionSink) -> None:
        sink.bind_runner(type(self).__name__)
        self.trajectory_sink = sink

    async def _run(self, input_batch: TrajectoryRequestBatch) -> TrajectoryBatch:
        raise NotImplementedError

    async def start_eval_session(self, *, run_name: str, eval_step: int, val_set_name: str | None = None) -> None:
        pass

    async def stop_eval_session(self) -> None:
        pass

    async def startup(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass
