"""Trainer settings scoped to one frozen-token probe forward."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any, Protocol

NATIVE_MODE = "native"
REPEAT_MODE = "repeat"
REPLAY_MODE = "router_replay"
FILTERED_REPLAY_MODE = "router_replay_filtered"


class ProbeWorker(Protocol):
    model: Any


def _native(worker: ProbeWorker, settings: Mapping[str, Any]) -> AbstractContextManager:
    return nullcontext()


def _replay(worker: ProbeWorker, settings: Mapping[str, Any]) -> AbstractContextManager:
    controller = worker.model.router_replay
    if controller is None:
        raise ValueError("router replay probe mode requires an installed controller")
    return controller.scoring_mode(settings["probe_mode"], settings["probe_keep_fraction"])


@dataclass(frozen=True)
class ModeSpec:
    context: Callable[[ProbeWorker, Mapping[str, Any]], AbstractContextManager]
    requires_routes: bool = False
    requires_keep_fraction: bool = False


TRAINER_MODES: dict[str, ModeSpec] = {
    NATIVE_MODE: ModeSpec(_native),
    REPEAT_MODE: ModeSpec(_native),
    REPLAY_MODE: ModeSpec(_replay, requires_routes=True),
    FILTERED_REPLAY_MODE: ModeSpec(_replay, requires_routes=True, requires_keep_fraction=True),
}
