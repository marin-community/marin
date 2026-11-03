"""Common grader invocation and reductions over per-item measurements."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from eval.contracts.eval_spec import EvalSpec, Grade, JudgePlan
from eval.execution.engine import ItemResult
from eval.passk import estimate_pass_at_k
from eval.runtime.resources import ExecutionMode, Resources


@dataclass(frozen=True)
class Grader:
    function: Callable[[EvalSpec, Sequence[Any]], Grade | JudgePlan]
    mode: ExecutionMode = ExecutionMode.SERIAL
    extraction_filters: tuple[str, ...] = ("none",)


class Graders:
    """Invoke the selected grading function under its declared resource policy."""

    def __init__(self, definitions: Mapping[str, Grader], workers: int = 1):
        self.definitions = dict(definitions)
        self.resources = Resources(workers)

    async def __call__(self, spec: EvalSpec, responses: Sequence[Any]) -> Grade | JudgePlan:
        grader = self.definitions[spec.grader.name]
        return await self.resources.call(grader.function, (spec, responses), grader.mode)

    def close(self) -> None:
        self.resources.close()


class MeanMetrics:
    """Online weighted means for numeric sample metrics."""

    def __init__(self):
        self.sums: dict[str, float] = defaultdict(float)
        self.counts: dict[str, float] = defaultdict(float)

    def add(self, metrics: Mapping[str, Any], weights: Mapping[str, float] | None = None) -> None:
        for name, value in metrics.items():
            if isinstance(value, (float, int)):
                weight = (weights or {}).get(name, 1.0)
                self.sums[name] += float(value) * weight
                self.counts[name] += weight

    def finish(self) -> dict[str, float]:
        return {name: total / self.counts[name] if self.counts[name] else 0.0 for name, total in self.sums.items()}


class TrialMetrics:
    """Common repeated-trial and pass@k reduction, grouped by source example."""

    def __init__(self, k: Sequence[int], metric: str):
        self.k = tuple(k)
        self.metric = metric
        self.items: dict[str, list[float | None]] = defaultdict(list)

    def add(self, result: ItemResult) -> None:
        value = result.grade.metrics.get(self.metric)
        if value is not None and value not in (0.0, 1.0):
            raise ValueError(f"pass@k requires binary grades; {self.metric} returned {value}")
        self.items[result.record["sample_id"]].append(value)

    def finish(self) -> dict[str, float | None]:
        if any(value is None for scores in self.items.values() for value in scores):
            # A missing judgment is neither a failed trial nor a smaller sampled
            # cohort. Its pass@k estimate is unavailable for the planned run.
            return {
                f"pass@{k}": None for k in self.k
                if self.items and k <= min(len(scores) for scores in self.items.values())
            }
        metrics = MeanMetrics()
        for scores in self.items.values():
            metrics.add(
                {
                    f"pass@{k}": float(
                        estimate_pass_at_k(len(scores), [int(sum(value for value in scores if value is not None))], k)[0]
                    )
                    for k in self.k
                    if k <= len(scores)
                }
            )
        return metrics.finish()
