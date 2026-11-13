# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sequential rank-pinned pilot with pacing, repeat checks, and latency stops."""

import json
import statistics
import time
from dataclasses import dataclass, field
from typing import TextIO

from marin.evaluation.likelihood import LikelihoodScore, ScoringClient

from experiments.evals.vep.targets import Target

COMPLETION_GAP = 1.0
SENTINEL_TOLERANCE = 1e-6
SENTINEL_INTERVAL = 48
MAX_COMPLETION_LATENCY = 5.0
INITIAL_LATENCY_SAMPLES = 8
RECENT_LATENCY_SAMPLES = 5
RECENT_LATENCY_MULTIPLIER = 3
MIN_RECENT_LATENCY_LIMIT = 1.5
DATA_PARALLEL_RANK = 0


@dataclass(kw_only=True)
class PacedClient(ScoringClient):
    sentinels: list[Target] = field(default_factory=list)
    guard_log: TextIO
    baseline: list[float] = field(default_factory=list)
    latencies: list[float] = field(default_factory=list)
    last_completion: float | None = None
    scored_targets: int = 0

    def request(self, route: str, payload: dict | None) -> dict:
        if route == "v1/completions" and self.last_completion is not None:
            time.sleep(max(0, COMPLETION_GAP - (time.monotonic() - self.last_completion)))
        started = time.monotonic()
        result = super().request(route, payload)
        if route == "v1/completions":
            self.last_completion = time.monotonic()
            elapsed = self.last_completion - started
            self.latencies.append(elapsed)
            if elapsed > MAX_COMPLETION_LATENCY:
                raise RuntimeError("Completion latency exceeded five-second stop threshold")
            if len(self.latencies) >= INITIAL_LATENCY_SAMPLES + RECENT_LATENCY_SAMPLES:
                initial = statistics.median(self.latencies[:INITIAL_LATENCY_SAMPLES])
                recent = statistics.median(self.latencies[-RECENT_LATENCY_SAMPLES:])
                if recent > max(MIN_RECENT_LATENCY_LIMIT, RECENT_LATENCY_MULTIPLIER * initial):
                    raise RuntimeError("Sustained completion latency exceeded stop threshold")
        return result

    def check_sentinels(self) -> None:
        for index, target in enumerate(self.sentinels):
            result = super().score(target.prefix, target.text, target.single_token)
            value = result.logprob
            if len(self.baseline) <= index:
                self.baseline.append(value)
            delta = abs(value - self.baseline[index])
            self.guard_log.write(
                json.dumps({"after_targets": self.scored_targets, "sentinel": index, "logprob": value, "delta": delta})
                + "\n"
            )
            self.guard_log.flush()
            if delta > SENTINEL_TOLERANCE:
                raise RuntimeError(f"Sentinel {index} drifted by {delta}; stopping")

    def score(self, prefix: str, target: str, single_token: bool) -> LikelihoodScore:
        if self.scored_targets and self.scored_targets % SENTINEL_INTERVAL == 0:
            self.check_sentinels()
        result = super().score(prefix, target, single_token)
        self.scored_targets += 1
        return result
