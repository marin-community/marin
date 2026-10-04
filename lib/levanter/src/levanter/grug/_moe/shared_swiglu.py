# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""A shared expert's SwiGLU MLP as two per-shard ops: the gate/up projection, then the down projection.

The split lets a caller run the gate/up projection somewhere other than the down projection, for
example beside a collective, and hand the projection's output across. For bfloat16 on SM100 with
QuACK, the gate/up op is one GEMM against interleaved gate/up weights with SwiGLU in its epilogue, and the
down op's backward runs the SwiGLU backward in the dh GEMM's epilogue. Elsewhere both ops are XLA
einsums. The two compute the same function up to rounding: QuACK applies SwiGLU to the fp32
accumulator, where XLA rounds gate and up to the compute dtype first.

Both ops take whole (unsharded) weights and this shard's tokens, so they run inside a shard map.
"""

import dataclasses
from collections.abc import Callable
from typing import Protocol

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Array, Float

from levanter.grug._moe.availability import quack_grouped_gemm_available


class SharedSwigluMlp(Protocol):
    """``down(gate_up(x, w_gate, w_up), w_down)`` is ``act(x @ w_gate) * (x @ w_up) @ w_down``.

    ``gate_up`` returns a tuple of token-major ``[T, ...]`` arrays that only ``down`` interprets.
    """

    def gate_up(
        self, x: Float[Array, "T D"], w_gate: Float[Array, "D I"], w_up: Float[Array, "D I"]
    ) -> tuple[jax.Array, ...]: ...

    def down(self, gate_up: tuple[jax.Array, ...], w_down: Float[Array, "I D"]) -> Float[Array, "T D"]: ...


@dataclasses.dataclass(frozen=True)
class _XlaSharedSwiglu:
    activation_fn: Callable[[jax.Array], jax.Array]

    def gate_up(self, x, w_gate, w_up):
        return jnp.einsum("td,dm->tm", x, w_gate), jnp.einsum("td,dm->tm", x, w_up)

    def down(self, gate_up, w_down):
        gate, up = gate_up
        return jnp.einsum("tm,md->td", self.activation_fn(gate) * up, w_down)


@dataclasses.dataclass(frozen=True)
class _CuteSharedSwiglu:
    def gate_up(self, x, w_gate, w_up):
        # QuACK and CUTLASS DSL are installed only with the CUDA 13 GPU extra.
        from levanter.grug._moe.sonic_cute import shared_gate_up  # noqa: PLC0415

        preact, h = shared_gate_up(x, w_gate, w_up)
        # `down` sends its whole input gradient to `preact`, through the dh GEMM's dSwiGLU epilogue,
        # and none to `h`. Stopping `h`'s gradient here keeps it a symbolic zero for `shared_gate_up`'s
        # backward even when the pair crosses another custom VJP on its way to `down`, as the ragged
        # MoE's `DispatchOverlap` staging does. That VJP hands back dense zeros, which would cost an
        # elementwise SwiGLU backward over [T, 2I] that adds nothing.
        return preact, jax.lax.stop_gradient(h)

    def down(self, gate_up, w_down):
        from levanter.grug._moe.sonic_cute import shared_swiglu_down  # noqa: PLC0415

        preact, h = gate_up
        return shared_swiglu_down(preact, h, w_down)


def select_shared_swiglu_mlp(activation_fn: Callable[[jax.Array], jax.Array], dtype: DTypeLike) -> SharedSwigluMlp:
    """The QuACK ops for SiLU on bfloat16 on an SM100 GPU with the GPU extra, the XLA einsums otherwise.

    ``dtype`` is the operands' common dtype. QuACK's gated GEMM writes its SwiGLU output in it, and
    its epilogues take no 32-bit outputs, so float32 operands keep the einsums; float16 does too, since
    only bfloat16 is tested.
    """
    if activation_fn is jax.nn.silu and jnp.dtype(dtype) == jnp.bfloat16 and quack_grouped_gemm_available():
        return _CuteSharedSwiglu()
    return _XlaSharedSwiglu(activation_fn)
