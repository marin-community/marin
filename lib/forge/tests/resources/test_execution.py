# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from forge.effects.issues import Restart, Transient
from forge.resources import execution
from forge.resources.env import RunCommand


async def test_retry_backoff_is_exponential_capped_and_stops_after_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    delays: list[float] = []
    attempts = 0

    async def wait(delay: float) -> None:
        delays.append(delay)

    async def fail() -> None:
        nonlocal attempts
        attempts += 1
        raise Transient("controller not ready")

    monkeypatch.setattr(execution, "sleep", wait)
    monkeypatch.setattr(execution, "MAX_ATTEMPTS", 9)
    with pytest.raises(Restart, match="controller not ready"):
        await execution.retry_effect(RunCommand("true", 1), fail, lambda _: None)
    assert attempts == 9
    assert delays == [1, 2, 4, 8, 16, 32, 60, 60]

