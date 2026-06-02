# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Actor + placement-group surface over Fray LocalClient.

These cover the Ray features actor-heavy frameworks (e.g. SkyRL) need beyond the
stateless-task surface: ``@ray.remote`` classes, actor-method calls, named-actor
lookup, ``.options(num_gpus=, scheduling_strategy=)``, and the
``ray.util.placement_group`` shim. Mirrors /tmp's SkyRL fail-fast probe.
"""

from __future__ import annotations

import fakeray as ray
import pytest
from fakeray._scheduler import FakeRayConfig


@pytest.fixture(autouse=True)
def _fresh_cluster():
    ray.set_config(FakeRayConfig(pool_size=2))
    ray.init()
    yield
    ray.shutdown()


def test_function_remote_still_works():
    @ray.remote
    def f(x):
        return x + 1

    assert ray.get(f.remote(41)) == 42


def test_stateful_actor_method_calls():
    @ray.remote
    class Counter:
        def __init__(self):
            self.n = 0

        def add(self, x):
            self.n += x
            return self.n

    h = Counter.remote()
    assert ray.get(h.add.remote(5)) == 5
    assert ray.get(h.add.remote(3)) == 8  # state persists across calls


def test_actor_options_resources():
    @ray.remote
    class A:
        def ping(self):
            return "pong"

    h = A.options(num_gpus=1, num_cpus=2).remote()
    assert ray.get(h.ping.remote()) == "pong"


def test_named_actor_lookup():
    @ray.remote
    class Info:
        def who(self):
            return "info"

    Info.options(name="InfoActor").remote()
    h = ray.get_actor("InfoActor")
    assert ray.get(h.who.remote()) == "info"


def test_get_actor_missing_raises():
    with pytest.raises(ValueError, match="nope"):
        ray.get_actor("nope")


def test_placement_group_and_scheduling_strategy():
    from ray.util.placement_group import placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    pg = placement_group([{"CPU": 1}, {"CPU": 1}], strategy="PACK")
    ray.get(pg.ready())

    @ray.remote
    class W:
        def go(self):
            return 1

    # actor pinned to a bundle via the scheduling strategy (best-effort)
    h = W.options(
        scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_bundle_index=0)
    ).remote()
    assert ray.get(h.go.remote()) == 1


def test_actor_method_exception_propagates():
    @ray.remote
    class Boom:
        def explode(self):
            raise ValueError("kaboom")

    h = Boom.remote()
    with pytest.raises(ValueError, match="kaboom"):
        ray.get(h.explode.remote())
