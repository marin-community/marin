# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The wire quantizer and the expert kernels' quantizer must agree bit-exactly.

`levanter.grug._moe.mxfp8_wire` carries its own copy of the MXFP8 feature-axis
quantizer because levanter cannot import from `experiments/`. If the two drift,
the payload the wire ships stops matching what the grouped kernels would have
produced from the same activations, and the "quantizing before the dispatch is
free" property in #7665 quietly stops holding.

This test lives here rather than in `lib/levanter/tests` because only this side
of the dependency direction may import both.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from levanter.grug._moe.mxfp8_wire import quantize_mxfp8_rows

sys.path.insert(0, str(Path(__file__).parent / "standalone"))
from mxfp8_grouped.quantize import quantize_mxfp8


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_wire_quantizer_matches_the_kernel_reference(seed):
    key = jax.random.PRNGKey(seed)
    k_base, k_tok = jax.random.split(key)
    x = (
        (
            jax.random.normal(k_base, (128, 256), dtype=jnp.float32)
            * jnp.exp(jax.random.normal(k_tok, (128, 1), dtype=jnp.float32) * 1.5)
        )
        .astype(jnp.bfloat16)
        .astype(jnp.float32)
    )

    wire_q, wire_sf = quantize_mxfp8_rows(x)
    ref_q, ref_sf = quantize_mxfp8(x)

    assert jnp.array_equal(wire_sf, ref_sf)
    assert jnp.array_equal(wire_q.view(jnp.uint8), ref_q.view(jnp.uint8))


def test_all_zero_blocks_quantize_to_zero_on_the_wire_path():
    """The one intentional difference from the reference.

    An all-zero block has amax 0, so its e8m0 scale byte is 0, which decodes to
    the subnormal 2^-127. The reference divides by it; backends that flush
    denormals turn that into 0/0. Dropped slots and pad rows make all-zero rows
    routine, so the wire masks instead. Whether the reference actually produces
    NaN is backend-dependent (it does on XLA CPU), so only the wire's behaviour
    is asserted here.
    """
    zeros = jnp.zeros((32, 64), jnp.float32)

    wire_q, _ = quantize_mxfp8_rows(zeros)
    wire_values = wire_q.astype(jnp.float32)

    assert not bool(jnp.any(jnp.isnan(wire_values)))
    assert bool(jnp.all(wire_values == 0))
