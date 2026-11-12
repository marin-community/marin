"""Exercise source items, registered graders, sample delivery, and reduction together."""

import asyncio
from collections.abc import Callable, Generator
from typing import Any

from eval.execution.engine import CollectSamples, Evaluator
from eval.execution.grading import Graders
from eval.execution.metrics import SourceMetrics
from eval.graders.specs import GRADERS
from eval.runtime.checkpoints import CheckpointRow, input_digest


class FixedEndpoint:
    """Endpoint fake that returns a fixed response and retains submitted requests."""

    def __init__(self, response: Any | Callable[[Any], Any]):
        self.response = response
        self.requests: list[Any] = []

    async def execute(self, request: Any) -> Any:
        self.requests.append(request)
        return self.response(request) if callable(self.response) else self.response


class MemoryCheckpoints:
    """In-memory operation journal that records finalization for lifecycle assertions."""

    def __init__(self):
        self.rows: dict[tuple[int, str], CheckpointRow] = {}
        self.closed = False

    def records(self) -> Generator[CheckpointRow, None, None]:
        yield from sorted(self.rows.values(), key=lambda row: (row.position, input_digest(row.unit)))

    def record(self, row: CheckpointRow) -> None:
        self.rows[row.position, input_digest(row.unit)] = row

    def finalize(self) -> None:
        self.closed = True


def evaluate_source(source, options, backend, *, judges=None, checkpoints=None, workers=2):
    sink = CollectSamples()
    grading = Graders(GRADERS)
    metrics = SourceMetrics(repeated=getattr(source, "n_repeat", 1) > 1, policy=source.REDUCER)
    engine = Evaluator(backend, grading, sink, workers=workers, judges=judges, checkpoints=checkpoints)

    async def run():
        async for completed in engine.run(source.iter_specs(options)):
            if completed.error is not None:
                raise completed.error
            metrics.add(completed.result)

    try:
        asyncio.run(run())
    finally:
        grading.close()
    return metrics.finish(options.task)["results"][options.task], sink.samples[options.task]
