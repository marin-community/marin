# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
import textwrap

import jax.numpy as jnp
import numpy as np

from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss
from levanter.kernels.pallas.fused_cross_entropy_loss import BlockSizes


def test_chunked_fused_cross_entropy_matches_unchunked_loss():
    hidden = jnp.arange(32, dtype=jnp.float32).reshape(8, 4) / 10
    lm_head = jnp.arange(24, dtype=jnp.float32).reshape(4, 6) / 20
    labels = jnp.arange(8, dtype=jnp.int32) % 6
    weight = jnp.asarray([1.0, 0.5, 1.0, 1.0, 0.0, 1.0, 0.25, 1.0], dtype=jnp.float32)

    expected = fused_linear_softmax_cross_entropy_loss(
        hidden,
        lm_head,
        labels,
        weight=weight,
        reduction="none",
        implementation="xla",
    )
    actual = fused_linear_softmax_cross_entropy_loss(
        hidden,
        lm_head,
        labels,
        weight=weight,
        reduction="none",
        implementation="xla",
        block_sizes=BlockSizes(b_block_size=128, h_block_size=128, v_block_size=128),
        batch_chunk_size=4,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_chunked_fused_cross_entropy_preserves_context_sharding():
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    script = """
        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

        from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss

        mesh = Mesh(
            np.asarray(jax.devices()).reshape(2, 2),
            axis_names=("data", "context"),
            axis_types=(AxisType.Explicit, AxisType.Explicit),
        )
        hidden = jnp.arange(48, dtype=jnp.float32).reshape(2, 4, 6) / 10
        lm_head = jnp.arange(42, dtype=jnp.float32).reshape(6, 7) / 20
        labels = jnp.arange(8, dtype=jnp.int32).reshape(2, 4) % 7
        weights = jnp.asarray([[1.0, 0.5, 1.0, 0.0], [0.25, 1.0, 1.0, 0.5]], dtype=jnp.float32)

        expected = fused_linear_softmax_cross_entropy_loss(
            hidden,
            lm_head,
            labels,
            weight=weights,
            reduction="none",
            implementation="xla",
            batch_chunk_size=2,
        )
        expected_mean = jnp.sum(expected) / jnp.sum(weights)

        token_sharding = NamedSharding(mesh, P("data", "context"))
        hidden_sharding = NamedSharding(mesh, P("data", "context", None))
        with jax.set_mesh(mesh):
            actual = fused_linear_softmax_cross_entropy_loss(
                jax.device_put(hidden, hidden_sharding),
                jax.device_put(lm_head, NamedSharding(mesh, P(None, None))),
                jax.device_put(labels, token_sharding),
                weight=jax.device_put(weights, token_sharding),
                reduction="none",
                implementation="xla",
                batch_chunk_size=2,
            )
            actual_mean = fused_linear_softmax_cross_entropy_loss(
                jax.device_put(hidden, hidden_sharding),
                jax.device_put(lm_head, NamedSharding(mesh, P(None, None))),
                jax.device_put(labels, token_sharding),
                weight=jax.device_put(weights, token_sharding),
                reduction="mean",
                implementation="xla",
                batch_chunk_size=2,
            )

        assert actual.sharding.spec == P("data", "context")
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(np.asarray(actual_mean), np.asarray(expected_mean), rtol=1e-6, atol=1e-6)
    """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
