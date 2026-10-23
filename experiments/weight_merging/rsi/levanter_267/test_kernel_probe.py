# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check that the reproducer consumes both distinct conditional scan results."""

import cloudpickle
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np

from experiments.weight_merging.rsi.levanter_267 import kernel_probe


def test_conditional_probe_carries_dependency_and_returns_both_layers(monkeypatch):
    def fake_attention(q, *args, sliding_window=None, **kwargs):
        return q * 2 if sliding_window is None else q + 1

    monkeypatch.setattr(kernel_probe.attention, "_do_tpu_ragged_paged_attention", fake_attention)
    q = hax.named(jnp.ones((2, 5, 4, 128)), ("position", "kv_head", "q_heads_per_group", "head_size"))
    kv = hax.named(jnp.ones((1, 1, 10, 128)), ("page", "page_size", "kv_head", "head_size"))
    lens = hax.named(jnp.array([2]), "seq")
    indices = hax.named(jnp.array([[0]]), ("seq", "page"))
    cu = hax.named(jnp.array([0, 2]), "seq")
    num = jnp.array(1)
    cloudpickle.register_pickle_by_value(kernel_probe)
    try:
        remote_function = cloudpickle.loads(cloudpickle.dumps(kernel_probe._invoke))
    finally:
        cloudpickle.unregister_pickle_by_value(kernel_probe)
    compiled_invoke = jax.jit(remote_function, static_argnames=("window", "conditional"))
    lowered = compiled_invoke.lower(q, kv, lens, indices, cu, num, window=2048, conditional=True)
    hlo = lowered.compiler_ir("hlo").as_hlo_text()
    assert "conditional" in hlo and "while" in hlo
    result = compiled_invoke(q, kv, lens, indices, cu, num, window=2048, conditional=True)
    np.testing.assert_array_equal(result[0], 2)
    np.testing.assert_array_equal(result[1], 4)
