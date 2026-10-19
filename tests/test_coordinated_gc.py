# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gc
import weakref
from contextlib import nullcontext

import jax.numpy as jnp
import pytest
from levanter.callbacks.state_adapter import StateCallbackRunner

from experiments.grug.moe_hero_ep.coordinated_gc import coordinated_gc
from experiments.grug.moe_hero_ep.train import _collect_after_eval


class CyclicNode:
    def __init__(self):
        self.link = self


def test_scheduled_collection_reclaims_cycles_at_global_step_boundary():
    runner = StateCallbackRunner(
        step_getter=lambda step: jnp.asarray(step),
        model_getter=lambda _: None,
        eval_model_getter=lambda _: None,
        opt_state_getter=lambda _: None,
    )
    with coordinated_gc() as collect:
        runner.add_hook(collect, every=100)
        node = CyclicNode()
        reference = weakref.ref(node)
        del node
        # A resumed loop at 99 must use the global boundary, not 100 local iterations.
        runner.run(99, loss=0.0, step_duration=0.0)
        assert reference() is not None
        runner.run(100, loss=0.0, step_duration=0.0)
        assert reference() is None

        node = CyclicNode()
        reference = weakref.ref(node)
        del node
        runner.run(150, loss=0.0, step_duration=0.0, force=True)
        assert reference() is None


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("fail", [True, False])
def test_gc_policy_restored_after_training_exit(enabled, fail):
    original = gc.isenabled()
    try:
        gc.enable() if enabled else gc.disable()
        with pytest.raises(RuntimeError) if fail else nullcontext():
            with coordinated_gc():
                assert not gc.isenabled()
                # A callback may change the process-global policy during training.
                gc.enable()
                if fail:
                    raise RuntimeError("training failed")
        assert gc.isenabled() == enabled
    finally:
        gc.enable() if original else gc.disable()


def test_entering_policy_reclaims_warmup_cycles():
    original = gc.isenabled()
    gc.disable()
    try:
        node = CyclicNode()
        reference = weakref.ref(node)
        del node
        with coordinated_gc():
            assert reference() is None
    finally:
        gc.enable() if original else gc.disable()


def test_eval_releases_array_cycles_before_next_training_step():
    references = []

    def evaluate():
        node = CyclicNode()
        node.array = jnp.ones(8)
        references.append(weakref.ref(node.array))

    with coordinated_gc():
        _collect_after_eval(evaluate)()
        assert references[0]() is None
