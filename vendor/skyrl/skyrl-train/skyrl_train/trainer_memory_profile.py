"""Warm-state trainer sizing that restores training state before admitting rollouts."""

import copy
import random
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Protocol

import numpy as np
import torch
from torch.utils._pytree import tree_map


class Stateful(Protocol):
    def state_dict(self) -> dict: ...

    def load_state_dict(self, state: dict) -> Any: ...


def _cpu_snapshot(state: dict) -> dict:
    return tree_map(lambda value: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value), state)


@contextmanager
def preserve_training_state(
    objects: Sequence[Stateful], *, optimizer_parameters: Sequence[torch.Tensor], cuda_devices: list[int]
) -> Iterator[None]:
    """Restore model/optimizer/scheduler and process RNG state after a sizing attempt.

    Snapshots live on the host so the memory check does not count another GPU copy of the model.
    Distributed RNG trackers must be preserved by the strategy alongside this process-local state.
    """
    states = [_cpu_snapshot(obj.state_dict()) for obj in objects]
    # DistributedOptimizer.state_dict() intentionally omits parameter-dependent state.
    # The caller supplies its underlying torch optimizers and their actual master shards.
    masters = [parameter.detach().cpu().clone() for parameter in optimizer_parameters]
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            try:
                yield
            finally:
                with torch.no_grad():
                    for parameter, original in zip(optimizer_parameters, masters, strict=True):
                        parameter.copy_(original)
                for obj, state in zip(objects, states, strict=True):
                    obj.load_state_dict(state)
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)


def warm_state_profile(
    valid_token_counts: Sequence[int],
    *,
    step: Callable[[int], None],
    measure: Callable[[int, Callable[[], None], bool], dict],
) -> list[dict]:
    """Initialize Adam once, then measure an optimizer-inclusive step for each token count."""
    if not valid_token_counts:
        raise ValueError("Trainer memory profile requires at least one token count")
    smallest = min(valid_token_counts)
    reports = [measure(smallest, lambda: step(smallest), False)]
    reports.extend(measure(tokens, lambda tokens=tokens: step(tokens), True) for tokens in valid_token_counts)
    return reports
