# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Zero-update TPU check of the fused head against its dense vocabulary path."""

import jax
import jax.numpy as jnp
import numpy as np

from experiments.grug.moe_cplm import belay, plus


def validate_head(variant, width):
    """Compare loss and VJP at the real hidden width before any optimizer step."""
    # 1024 rows satisfy the fused CE block. A small vocabulary exercises the
    # same width-dependent kernel tiles without allocating the full LM state.
    shapes = [(1, 1024, width), (width, 512), (width, width), (width, width), (width,), (width,)]
    if variant == "plus":
        shapes += [(width, width), (width,), (width,), ()]
    with jax.default_device(jax.local_devices()[0]):
        params = tuple(
            (0.02 * jax.random.normal(k, s)).astype(jnp.bfloat16)
            for k, s in zip(jax.random.split(jax.random.key(16), len(shapes)), shapes, strict=True)
        )
        tokens = (jnp.arange(1024) % 31)[None]
        labels = jnp.roll(tokens, -1, axis=1)
        segments = (jnp.arange(1024) // 256)[None]
        classes = jnp.arange(512) // 2
        members = (jnp.arange(512) // 2)[:, None] * 2 + jnp.arange(2)

        def loss(p, reference):
            h, head, qw, kw, gw, sink, *extra = p
            if variant == "belay":
                return belay.vocabulary_statistics(h, head, labels, reference=reference)[0].mean()
            fn = plus.local_head if variant == "plus" else belay.local_head
            args = (h, head, qw, kw, gw, sink, tokens, labels, segments)
            if variant == "plus":
                args += (*extra, classes, members)
            return fn(*args, reference=reference)[0].mean()

        fused = jax.jit(jax.value_and_grad(lambda p: loss(p, False)))
        dense = jax.jit(jax.value_and_grad(lambda p: loss(p, True)))
        got, got_grad = jax.device_get(fused(params))
        want, want_grad = jax.device_get(dense(params))
        # BF16 cotangents are rounded at the head's input boundary. Compare
        # norm error as well as the loss; tiny zero entries cannot set a scale.
        np.testing.assert_allclose(got, want, rtol=2e-4, atol=2e-4)
        errors = []
        for a, b in zip(got_grad, want_grad, strict=True):
            a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
            assert np.isfinite(a).all()
            error = float(np.linalg.norm(a - b) / max(float(np.linalg.norm(b)), 1e-9))
            assert error < 0.02, error
            errors.append(error)
        fused.clear_cache()
        dense.clear_cache()
        return {
            "loss": float(got),
            "reference_loss": float(want),
            "gradient_relative_errors": errors,
            "training_updates": 0,
            "width": width,
            "variant": variant,
        }
