# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The endpoint an evaluation queries and the health of whatever serves it."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from fray.client import JobHandle

from marin.inference.config import EffectiveServing
from marin.inference.types import InferenceBackendState, RunningModel


class InferenceSession(Protocol):
    """An OpenAI-compatible endpoint with the serving jobs, health and settings behind it."""

    @property
    def model(self) -> RunningModel: ...

    @property
    def jobs(self) -> tuple[JobHandle, ...]: ...

    @property
    def metrics_url(self) -> str | None: ...

    @property
    def effective_serving(self) -> EffectiveServing | None: ...

    def check_alive(self) -> None: ...

    def preemption_count(self) -> int: ...

    def backend_state(self) -> InferenceBackendState: ...

    def wait_until_ready(self, *, check_dependencies: Callable[[], None] | None = None) -> None: ...


@dataclass(frozen=True)
class ApiInferenceSession:
    """An external API endpoint: no Marin jobs serve it, so it is always ready.

    Provider outages surface as request failures, which the evaluator reports in item coverage.
    """

    model: RunningModel
    jobs: tuple[JobHandle, ...] = ()
    metrics_url: str | None = None
    effective_serving: EffectiveServing | None = None

    def check_alive(self) -> None:
        return None

    def preemption_count(self) -> int:
        return 0

    def backend_state(self) -> InferenceBackendState:
        return InferenceBackendState.READY

    def wait_until_ready(self, *, check_dependencies: Callable[[], None] | None = None) -> None:
        return None
