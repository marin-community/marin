"""Exact weighted route mixtures and ordered stage sequences for training prompts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import math
import random
from typing import Any

import datasets

from skyrl_train.rollouts.loader import JudgedGroup

ROUTE_COLUMN = "teacher_route"


def weighted_quotas(total: int, weights: Mapping[str, float], batch_index: int) -> dict[str, int]:
    """Allocate an exact batch with stable, rotating ties."""
    if total <= 0 or not weights or any(not math.isfinite(weight) or weight <= 0 for weight in weights.values()):
        raise ValueError("weighted route quotas require a positive batch size and positive weights")
    names = tuple(weights)
    scale = sum(weights.values())
    raw = {name: total * weights[name] / scale for name in names}
    quotas = {name: math.floor(raw[name]) for name in names}
    remainder = total - sum(quotas.values())
    priority = sorted(
        names,
        key=lambda name: (-(raw[name] - quotas[name]), (names.index(name) - batch_index) % len(names)),
    )
    for name in priority[:remainder]:
        quotas[name] += 1
    return quotas


class DomainWeightedOrder:
    """Cycle shuffled route pools, composing every window of ``window_size`` rows with an exact weighted mixture.

    Rows within one window are unique. Groups can finish out of order or be discarded, so a training batch
    matches the mixture in expectation rather than exactly.
    """

    def __init__(self, dataframe: datasets.Dataset, weights: Mapping[str, float], *, seed: int, window_size: int):
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if ROUTE_COLUMN not in dataframe.column_names:
            raise ValueError(f"Domain-weighted sampling requires a top-level {ROUTE_COLUMN!r} column")
        self.weights = dict(weights)
        self.names = tuple(self.weights)
        self.window_size = window_size
        pools: dict[str, list[int]] = defaultdict(list)
        for index, route in enumerate(dataframe[ROUTE_COLUMN]):
            pools[str(route)].append(index)
        unknown = set(pools) - set(self.names)
        if unknown:
            raise ValueError(f"Dataset has unknown teacher routes: {sorted(unknown)}")
        max_quotas = {
            name: max(weighted_quotas(window_size, self.weights, window)[name] for window in range(len(self.names)))
            for name in self.names
        }
        underfilled = {name: quota for name, quota in max_quotas.items() if len(pools[name]) < quota}
        if underfilled:
            raise ValueError(f"Domain-weighted route pools cannot fill a unique window: {underfilled}")
        self.rng = random.Random(seed)
        self.pools = {name: list(pools[name]) for name in self.names}
        for rows in self.pools.values():
            self.rng.shuffle(rows)
        self.cursors = {name: 0 for name in self.names}
        self.window_index = 0
        self.window: list[int] = []
        self.window_position = 0

    def next_index(self) -> int:
        if self.window_position == len(self.window):
            self._next_window()
        index = self.window[self.window_position]
        self.window_position += 1
        return index

    def observe(self, groups: Sequence[JudgedGroup]) -> dict[str, float]:
        return {}

    def _take(self, route: str, count: int) -> list[int]:
        selected: list[int] = []
        seen: set[int] = set()
        while len(selected) < count:
            cursor = self.cursors[route]
            pool = self.pools[route]
            row = pool[cursor]
            cursor += 1
            if cursor == len(pool):
                self.rng.shuffle(pool)
                cursor = 0
            self.cursors[route] = cursor
            if row not in seen:
                selected.append(row)
                seen.add(row)
        return selected

    def _next_window(self) -> None:
        quotas = weighted_quotas(self.window_size, self.weights, self.window_index)
        self.window = [row for name in self.names for row in self._take(name, quotas[name])]
        self.rng.shuffle(self.window)
        self.window_position = 0
        self.window_index += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "rng_state": self.rng.getstate(),
            "pools": {name: list(rows) for name, rows in self.pools.items()},
            "cursors": dict(self.cursors),
            "window_index": self.window_index,
            "window": list(self.window),
            "window_position": self.window_position,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.rng.setstate(state["rng_state"])
        self.pools = {name: list(rows) for name, rows in state["pools"].items()}
        self.cursors = dict(state["cursors"])
        self.window_index = state["window_index"]
        self.window = list(state["window"])
        self.window_position = state["window_position"]


class OrderedStages:
    """Consume route pools in a declared stage sequence.

    Each stage draws from one route's pool: ``quota`` prompts per stage visit, or the pool's row
    count when the stage declares no quota. When a stage's quota is spent the sequence advances, and past
    the last stage it restarts from the first. Draws are deterministic given the seed, and rows stay
    unique within each window of ``window_size`` draws, as :class:`DomainWeightedOrder` requires.
    """

    def __init__(
        self,
        dataframe: datasets.Dataset,
        stages: Sequence[Mapping[str, Any]],
        *,
        seed: int,
        window_size: int,
    ):
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        declared = []
        for stage in stages:
            label = stage.get("source")
            quota = stage.get("quota")
            if not isinstance(label, str) or not label:
                raise ValueError(f"ordered stages name a route label, got {label!r}")
            if quota is not None and (not isinstance(quota, int) or isinstance(quota, bool) or quota <= 0):
                raise ValueError(f"stage {label!r} quota must be a positive integer or null, got {quota!r}")
            declared.append((label, quota))
        if not declared:
            raise ValueError("ordered stage sampling requires at least one stage")
        self.window_size = window_size
        self.stages = declared
        pools: dict[str, list[int]] = defaultdict(list)
        for index, route in enumerate(dataframe[ROUTE_COLUMN]):
            pools[str(route)].append(index)
        unknown = {label for label, _ in declared} - set(pools)
        if unknown:
            raise ValueError(f"Ordered stages name routes the dataset does not have: {sorted(unknown)}")
        # The most rows one label can be asked for inside a single window is the maximum count of
        # that label's draws over any ``window_size``-wide stretch of the cyclic stage sequence. A
        # window can span repeated stage cycles, so one visit per declared stage undercounts. Walk
        # one full period of the deterministic visit sequence plus a window of wrap-around and take
        # the sliding-window maximum per label; the pool must supply that many unique rows.
        visit_lengths = [quota if quota is not None else len(pools[label]) for label, quota in declared]
        period_sequence = [label for (label, _), length in zip(declared, visit_lengths) for _ in range(length)]
        period = sum(visit_lengths)
        draws_needed = period + window_size - 1
        repeats = -(-draws_needed // period)
        extended = (period_sequence * repeats)[:draws_needed]
        counts: dict[str, int] = defaultdict(int)
        for label in extended[:window_size]:
            counts[label] += 1
        demand = dict(counts)
        for offset in range(1, period):
            counts[extended[offset - 1]] -= 1
            counts[extended[offset + window_size - 1]] += 1
            for label, count in counts.items():
                if count > demand.get(label, 0):
                    demand[label] = count
        underfilled = {label: wanted for label, wanted in demand.items() if wanted > len(pools[label])}
        if underfilled:
            raise ValueError(f"Ordered stage pools cannot supply a window-unique stage visit: {underfilled}")
        self.rng = random.Random(seed)
        self.pools = {label: list(pools[label]) for label in demand}
        for rows in self.pools.values():
            self.rng.shuffle(rows)
        self.cursors = {label: 0 for label in self.pools}
        self.stage_index = 0
        self.stage_draws = 0
        self.window: list[int] = []
        self.window_position = 0

    def next_index(self) -> int:
        if self.window_position == self.window_size:
            self.window = []
            self.window_position = 0
        label, quota = self.stages[self.stage_index]
        spent = quota if quota is not None else len(self.pools[label])
        if self.stage_draws >= spent:
            self.stage_index = (self.stage_index + 1) % len(self.stages)
            self.stage_draws = 0
            label, quota = self.stages[self.stage_index]
            if quota is None:
                # A quota-less stage visit is one exact pass over the pool: reshuffle on entry so
                # the visit covers every row once, whatever an earlier visit left on the cursor.
                self.rng.shuffle(self.pools[label])
                self.cursors[label] = 0
        row = self._take(label)
        self.window.append(row)
        self.window_position += 1
        self.stage_draws += 1
        return row

    def observe(self, groups: Sequence[JudgedGroup]) -> dict[str, float]:
        return {}

    def _take(self, label: str) -> int:
        """The next window-unique row from a label's pool, advancing the pool order.

        The cursor never reshuffles mid-visit: a quota-less visit consumes one shuffled pool order
        from its entry, so it covers the pool exactly once."""
        pool = self.pools[label]
        start = self.cursors[label] % len(pool)
        for offset in range(len(pool)):
            cursor = (start + offset) % len(pool)
            row = pool[cursor]
            if row not in self.window:
                self.cursors[label] = (cursor + 1) % len(pool)
                return row
        raise RuntimeError(
            f"route {label!r} has no window-unique row left: its pool of {len(pool)} rows cannot supply "
            f"the declared stage visits within a {self.window_size}-draw window"
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "rng_state": self.rng.getstate(),
            "pools": {name: list(rows) for name, rows in self.pools.items()},
            "cursors": dict(self.cursors),
            "stage_index": self.stage_index,
            "stage_draws": self.stage_draws,
            "window": list(self.window),
            "window_position": self.window_position,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.rng.setstate(state["rng_state"])
        self.pools = {name: list(rows) for name, rows in state["pools"].items()}
        self.cursors = dict(state["cursors"])
        self.stage_index = state["stage_index"]
        self.stage_draws = state["stage_draws"]
        self.window = list(state["window"])
        self.window_position = state["window_position"]
