# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Parallax attention (arXiv 2605.29157): causal softmax attention minus the attention-weighted key-value
covariance probed by a second query-like projection ``r``.

``parallax_attention(q, r, k, v, segment_ids, sm_scale=...)`` takes ``[B, L, H, D]`` queries and probes and
``[B, L, H_kv, D]`` keys and values (GQA when ``H_kv < H``). ``implementation`` selects the Pallas Triton GPU
kernels (``"pallas_gpu"``), the same kernels in Pallas's interpreter (``"pallas_interpret"``, for CPU tests), or the
dense fp32 reference (``"reference"``).
"""

import functools
from typing import Literal, TypeAlias

import jax
from jaxtyping import Array, Float, Int

from .pallas_gpu import BlockSizes, parallax_backward, parallax_forward
from .reference import parallax_reference

Implementation: TypeAlias = Literal["reference", "pallas_gpu", "pallas_interpret"]
IMPLEMENTATIONS: tuple[Implementation, ...] = ("reference", "pallas_gpu", "pallas_interpret")


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7))
def _parallax_kernel(q, r, k, v, segment_ids, scale: float, block_sizes: BlockSizes, interpret: bool):
    return parallax_forward(q, r, k, v, segment_ids, scale=scale, block_sizes=block_sizes, interpret=interpret)[0]


def _parallax_kernel_fwd(q, r, k, v, segment_ids, scale, block_sizes, interpret):
    o, barv, m, d1, bart = parallax_forward(
        q, r, k, v, segment_ids, scale=scale, block_sizes=block_sizes, interpret=interpret
    )
    return o, (q, r, k, v, segment_ids, o, barv, m, d1, bart)


def _parallax_kernel_bwd(scale, block_sizes, interpret, residuals, do):
    q, r, k, v, segment_ids, o, barv, m, d1, bart = residuals
    dq, dr, dk, dv = parallax_backward(
        q,
        r,
        k,
        v,
        segment_ids,
        o,
        barv,
        m,
        d1,
        bart,
        do.astype(q.dtype),
        scale=scale,
        block_sizes=block_sizes,
        interpret=interpret,
    )
    return dq, dr, dk, dv, None


_parallax_kernel.defvjp(_parallax_kernel_fwd, _parallax_kernel_bwd)


def parallax_attention(
    q: Float[Array, "B L H D"],
    r: Float[Array, "B L H D"],
    k: Float[Array, "B L Hkv D"],
    v: Float[Array, "B L Hkv D"],
    segment_ids: Int[Array, "B L"] | None = None,
    *,
    sm_scale: float,
    implementation: Implementation = "pallas_gpu",
    block_sizes: BlockSizes | None = None,
) -> Float[Array, "B L H D"]:
    """Causal (same-segment) Parallax attention; ``r . k`` is unscaled, ``q . k`` is scaled by ``sm_scale``."""
    if implementation == "reference":
        return parallax_reference(q, r, k, v, segment_ids, sm_scale=sm_scale).astype(q.dtype)
    if implementation not in IMPLEMENTATIONS:
        raise ValueError(f"implementation must be one of {IMPLEMENTATIONS}, got {implementation!r}")
    blocks = BlockSizes() if block_sizes is None else block_sizes
    return _parallax_kernel(q, r, k, v, segment_ids, sm_scale, blocks, implementation == "pallas_interpret")
