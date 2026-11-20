"""Check a SkyRL training log against its run spec."""

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

# `WANDB_MIRROR kind=train step=2 metrics={"policy/policy_loss": 0.1, ...}`, embedded in a
# loguru line, so the prefix is matched loosely and the JSON object runs to end of line.
METRIC_LINE = re.compile(r"WANDB_MIRROR kind=(?P<kind>\w+) step=(?P<step>\d+) metrics=(?P<metrics>\{.*\})\s*$")

# The trainer's loguru sink colorizes even when piped, so the payload arrives wrapped in SGR
# escapes (`\x1b[32m...WANDB_MIRROR...}\x1b[0m`). The trailing reset defeats the `\}\s*$` anchor,
# which silently parses zero steps out of a perfectly healthy run -- strip the escapes first.
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


class MetricKind(StrEnum):
    TRAIN = "train"
    EVAL = "eval"


class FailureKind(StrEnum):
    WALL_CLOCK = "wall_clock"
    TRAIN_STEPS = "train_steps"
    CONFLICTING_PAYLOAD = "conflicting_payload"
    MISSING_METRIC = "missing_metric"
    NONFINITE = "nonfinite"
    BOUNDS = "bounds"
    OBSERVATIONS = "observations"
    TREND = "trend"
    LOG_PATTERN = "log_pattern"


@dataclass(frozen=True)
class GateFailure:
    kind: FailureKind
    message: str
    metric: str | None = None
    stream: str | None = None
    step: int | None = None


@dataclass(frozen=True)
class StepMetrics:
    """The metrics the trainer logged for one step."""

    kind: str
    step: int
    values: dict[str, object]


@dataclass(frozen=True)
class MetricBound:
    """Require all values, or at least minimum_count values, inside the range."""

    minimum: float = -math.inf
    maximum: float = math.inf
    minimum_count: int | None = None
    inclusive_minimum: bool = True
    inclusive_maximum: bool = True

    def contains(self, value: float) -> bool:
        lower = value >= self.minimum if self.inclusive_minimum else value > self.minimum
        upper = value <= self.maximum if self.inclusive_maximum else value < self.maximum
        return lower and upper


@dataclass(frozen=True)
class SeriesTrend:
    window: int
    min_improvement: float

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError("trend window must be positive")


@dataclass(frozen=True)
class MetricSeries:
    """Ordered payloads for one metric, with statistics over finite observations."""

    kind: MetricKind
    metric: str
    observations: tuple[StepMetrics, ...]

    def finite_values(self) -> tuple[float, ...]:
        return tuple(
            float(row.values[self.metric]) for row in self.observations if _is_finite(row.values.get(self.metric))
        )

    def count_in_range(self, bound: MetricBound) -> int:
        return sum(bound.contains(value) for value in self.finite_values())

    def improvement(self, window: int) -> float:
        values = self.finite_values()
        return math.fsum(value / window for value in values[-window:]) - math.fsum(
            value / window for value in values[:window]
        )


@dataclass(frozen=True)
class MetricGate:
    """Required finite observations selected at one step or up to a completed step."""

    kind: MetricKind
    metric: str
    min_observations: int
    bounds: MetricBound | None = None
    trend: SeriesTrend | None = None
    step: int | Literal["first", "last"] | None = None
    max_step: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", MetricKind(self.kind))

    def series(self, steps: list[StepMetrics]) -> MetricSeries:
        rows = [row for row in steps if row.kind == self.kind]
        if isinstance(self.step, int):
            rows = [row for row in rows if row.step == self.step]
        elif self.step == "first":
            rows = rows[:1]
        elif self.step == "last":
            rows = rows[-1:]
        if self.max_step is not None:
            rows = [row for row in rows if row.step <= self.max_step]
        return MetricSeries(self.kind, self.metric, tuple(rows))

    def check(self, series: MetricSeries) -> list[GateFailure]:
        failures = []
        for row in series.observations:
            value = row.values.get(self.metric)
            if self.metric not in row.values:
                failures.append(
                    GateFailure(FailureKind.MISSING_METRIC, "metric missing", self.metric, self.kind, row.step)
                )
            elif not _is_finite(value):
                failures.append(
                    GateFailure(FailureKind.NONFINITE, f"nonfinite value {value!r}", self.metric, self.kind, row.step)
                )
        if failures:
            return failures
        values = series.finite_values()
        if len(values) < self.min_observations:
            failures.append(
                GateFailure(
                    FailureKind.OBSERVATIONS,
                    f"{len(values)} finite observations; need {self.min_observations}",
                    self.metric,
                    self.kind,
                )
            )
        if self.bounds is not None:
            count = series.count_in_range(self.bounds)
            required = len(values) if self.bounds.minimum_count is None else self.bounds.minimum_count
            if count < required:
                failures.append(
                    GateFailure(
                        FailureKind.BOUNDS,
                        f"{count} observations in range {self.bounds}; need {required}",
                        self.metric,
                        self.kind,
                    )
                )
        if self.trend is not None:
            window = self.trend.window
            if len(values) < 2 * window:
                failures.append(
                    GateFailure(
                        FailureKind.OBSERVATIONS,
                        f"{len(values)} observations; need {2 * window} for trend",
                        self.metric,
                        self.kind,
                    )
                )
            elif series.improvement(window) < self.trend.min_improvement:
                failures.append(
                    GateFailure(
                        FailureKind.TREND,
                        f"gain {series.improvement(window):+.4f}; need {self.trend.min_improvement:+.4f}",
                        self.metric,
                        self.kind,
                    )
                )
        return failures


@dataclass(frozen=True)
class LogPatternBound:
    pattern: str
    minimum: int
    maximum: int | None = None


@dataclass(frozen=True)
class GateSpec:
    min_train_steps: int
    finite_metrics: tuple[str, ...]
    bounds: dict[str, MetricBound]
    max_wall_clock_seconds: float
    required_log_patterns: dict[str, LogPatternBound] | None = None
    metric_gates: tuple[MetricGate, ...] = ()


def load_spec(path: Path) -> GateSpec:
    raw = json.loads(path.read_text())
    rows = []
    for value in raw.get("metric_gates", ()):
        row = dict(value)
        if "bounds" in row:
            row["bounds"] = MetricBound(**row["bounds"])
        if "trend" in row:
            row["trend"] = SeriesTrend(**row["trend"])
        rows.append(MetricGate(**row))
    return GateSpec(
        min_train_steps=raw["min_train_steps"],
        finite_metrics=tuple(raw.get("finite_metrics", ())),
        bounds={name: MetricBound(**bound) for name, bound in raw.get("bounds", {}).items()},
        max_wall_clock_seconds=raw["max_wall_clock_seconds"],
        required_log_patterns={
            name: LogPatternBound(**bound) for name, bound in raw.get("required_log_patterns", {}).items()
        },
        metric_gates=tuple(rows),
    )


def parse_metrics(log_text: str) -> list[StepMetrics]:
    """Read native WANDB_MIRROR payloads in their logged order."""
    steps = []
    for line in ANSI_ESCAPE.sub("", log_text).splitlines():
        match = METRIC_LINE.search(line)
        if match is not None:
            steps.append(StepMetrics(match["kind"], int(match["step"]), json.loads(match["metrics"])))
    return steps


def _is_finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _distinct_steps(steps: list[StepMetrics]) -> tuple[list[StepMetrics], list[GateFailure]]:
    by_key = {}
    failures = []
    for row in steps:
        if row.kind not in (MetricKind.TRAIN, MetricKind.EVAL):
            continue
        key = (row.kind, row.step)
        prior = by_key.get(key)
        if prior is not None and json.dumps(prior.values, sort_keys=True) != json.dumps(row.values, sort_keys=True):
            failures.append(
                GateFailure(FailureKind.CONFLICTING_PAYLOAD, "conflicting payloads", stream=row.kind, step=row.step)
            )
        by_key[key] = row
    return sorted(by_key.values(), key=lambda row: (row.kind, row.step)), failures


def check_run(steps: list[StepMetrics], spec: GateSpec, wall_clock_seconds: float) -> list[GateFailure]:
    """Return structured failures for the run's timing, steps and metric rows."""
    rows, failures = _distinct_steps(steps)
    if wall_clock_seconds > spec.max_wall_clock_seconds:
        failures.append(
            GateFailure(
                FailureKind.WALL_CLOCK, f"run took {wall_clock_seconds:.0f}s; limit {spec.max_wall_clock_seconds:.0f}s"
            )
        )
    train = [row for row in rows if row.kind == MetricKind.TRAIN]
    if len(train) < spec.min_train_steps:
        failures.append(
            GateFailure(
                FailureKind.TRAIN_STEPS,
                f"{len(train)} training steps; need {spec.min_train_steps}",
                stream=MetricKind.TRAIN,
            )
        )
    for metric in dict.fromkeys((*spec.finite_metrics, *spec.bounds)):
        requirement = MetricGate(MetricKind.TRAIN, metric, 1, bounds=spec.bounds.get(metric), step="last")
        failures.extend(requirement.check(requirement.series(rows)))
    for requirement in spec.metric_gates:
        failures.extend(requirement.check(requirement.series(rows)))
    return failures


def check_log_patterns(log_text: str, spec: GateSpec) -> list[GateFailure]:
    """Check named events against inclusive occurrence counts."""
    failures = []
    for name, bound in (spec.required_log_patterns or {}).items():
        count = len(re.findall(bound.pattern, ANSI_ESCAPE.sub("", log_text)))
        if count < bound.minimum or (bound.maximum is not None and count > bound.maximum):
            failures.append(
                GateFailure(
                    FailureKind.LOG_PATTERN, f"{count} occurrences; require [{bound.minimum}, {bound.maximum}]", name
                )
            )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True, help="training run log to read")
    parser.add_argument("--spec", type=Path, required=True, help="gate spec to check against")
    parser.add_argument(
        "--wall-clock-seconds",
        type=float,
        required=True,
        help="how long the run took, measured by the caller",
    )
    args = parser.parse_args()
    spec = load_spec(args.spec)
    log_text = args.log.read_text()
    steps = parse_metrics(log_text)
    failures = check_run(steps, spec, args.wall_clock_seconds) + check_log_patterns(log_text, spec)
    rows, _ = _distinct_steps(steps)
    train = [row for row in rows if row.kind == MetricKind.TRAIN]
    print(f"parsed {len(train)} training steps from {args.log} in {args.wall_clock_seconds:.0f}s")
    if train:
        print(f"final step {train[-1].step}: {json.dumps(train[-1].values, sort_keys=True)}")
    if failures:
        print(f"\nFAILED against {args.spec}:")
        for failure in failures:
            print(f"  - {failure.kind}: {failure.stream}/{failure.metric} step={failure.step}: {failure.message}")
        return 1
    print(f"\nOK against {args.spec}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
