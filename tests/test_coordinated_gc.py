# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gc
import weakref

import pytest

from experiments.grug.moe_hero_ep.coordinated_gc import coordinated_gc


class CyclicNode:
    def __init__(self):
        self.link = self


def test_scheduled_collection_reclaims_cycles_at_global_step_boundary():
    with coordinated_gc(100) as collect:
        node = CyclicNode()
        reference = weakref.ref(node)
        del node
        # A resumed loop at 99 must use the global boundary, not 100 local iterations.
        collect(99)
        assert reference() is not None
        collect(100)
        assert reference() is None


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("fail", [True, False])
def test_gc_policy_restored_after_training_exit(enabled, fail):
    original = gc.isenabled()
    try:
        gc.enable() if enabled else gc.disable()
        try:
            with coordinated_gc(100):
                assert not gc.isenabled()
                if fail:
                    raise RuntimeError("training failed")
        except RuntimeError as error:
            assert str(error) == "training failed"
        assert gc.isenabled() == enabled
    finally:
        gc.enable() if original else gc.disable()
