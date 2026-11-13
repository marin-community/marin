# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

daytona = pytest.importorskip("daytona")
snapshots = pytest.importorskip("experiments.post_training.tmax_reserve.snapshots")


class _Factory:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    async def prepare(self, spec):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_a_provider_error_while_building_is_followed_by_one_more_prepare(capsys):
    factory = _Factory([daytona.DaytonaError("Bad Gateway", status_code=502), "shellbox-b"])

    assert asyncio.run(snapshots.prepare_once_more_on_error(factory, spec=None)) == "shellbox-b"
    assert factory.calls == 2
    assert "DaytonaError: Bad Gateway" in capsys.readouterr().err


def test_a_second_provider_error_propagates():
    factory = _Factory([daytona.DaytonaError("first"), daytona.DaytonaError("build failed")])

    with pytest.raises(daytona.DaytonaError, match="build failed"):
        asyncio.run(snapshots.prepare_once_more_on_error(factory, spec=None))
    assert factory.calls == 2
