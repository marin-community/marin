# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Full TransformerEngine MoE block backend (issue #7331 follow-up).

Unlike ``nccl_ep`` (TE NCCL_EP dispatch/combine around the QuACK expert FFN),
this backend hands the ENTIRE MoE layer to TE's fused block —
``transformer_engine.jax.moe.moe``: fused sigmoid+top-k router, NCCL_EP
dispatch, TE grouped-quantize + grouped GEMMs, NCCL_EP combine — under a
single ``custom_vjp``. This is NVIDIA's recommended NCCL_EP integration path
(the "MoE Block" from TE #2912/#3116), benchmarked here against the
dispatch/combine-only seam.

Router parity with the bench's QB-routed MoEMLP: TE's sigmoid path computes
``sigmoid(logits)``, adds ``expert_bias`` (no gradient through top-k, matching
the bench's ``stop_gradient``), takes top-k, reverts the bias, sum-normalizes
the selected scores, and multiplies by ``scaling_factor`` — the bench's
"renormalize to ``_ROUTING_RENORM_SUM``" with ``scaling_factor = renorm sum``.
One residual difference: TE adds the bias in *score* space (post-sigmoid)
while the bench adds it in logit space, so once QB updates make the bias
nonzero the two arms can select different experts on marginal tokens.

Process-global requirements are the same as ``ep_nccl.py`` (process-per-GPU,
eager ``ep_bootstrap`` under the TE shard guard, TE imported before the JAX
CUDA client exists), plus ``record_ep_bootstrap_signature_for_moe`` after
bootstrap: ``moe()`` re-derives its staging needs per call — with 128-token
per-expert alignment padding, see :func:`te_moe_recv_capacity` — and asserts
the recorded bootstrap covers them.
"""

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

try:
    from transformer_engine.jax.moe import (
        moe as _te_moe_fused,
        record_ep_bootstrap_signature_for_moe as record_ep_bootstrap_signature_for_moe,
    )

    _TE_IMPORT_ERROR = None
except ImportError as _e:  # optional dep: transformer-engine with the MoE block
    _te_moe_fused = None
    record_ep_bootstrap_signature_for_moe = None
    _TE_IMPORT_ERROR = _e

# TE moe.py's per-expert dispatch-slot alignment (moe._ALIGN_SIZE): NCCL EP HT
# lays out variable per-expert zones in one flat recv buffer, each non-empty
# zone padded to this many tokens.
_TE_MOE_ALIGN_TOKENS = 128


def te_moe_recv_capacity(tokens_per_rank: int, ep_size: int, top_k: int, num_local_experts: int) -> int:
    """Per-rank recv capacity TE's ``moe()`` will demand of the bootstrap.

    Replicates the bound in ``transformer_engine/jax/moe.py::_moe_fwd_rule``
    (no-drop worst case plus per-expert alignment padding). ``ep_bootstrap``
    must be sized with THIS value, not the bare ``ep × tokens × top_k``
    product, or the per-call compatibility assert fires.
    """
    tokens_per_ep_group = ep_size * tokens_per_rank
    max_local_assignments = tokens_per_ep_group * min(top_k, num_local_experts)
    max_nonempty_experts = min(num_local_experts, max_local_assignments)
    padded_total_bound = max_local_assignments + (_TE_MOE_ALIGN_TOKENS - 1) * max_nonempty_experts
    aligned_total_bound = -(-padded_total_bound // _TE_MOE_ALIGN_TOKENS) * _TE_MOE_ALIGN_TOKENS
    per_expert_bound = num_local_experts * (-(-tokens_per_ep_group // _TE_MOE_ALIGN_TOKENS) * _TE_MOE_ALIGN_TOKENS)
    return min(per_expert_bound, aligned_total_bound)


def _moe_mlp_te_block(
    x: jax.Array,
    gate_kernel: jax.Array,
    w_gate: jax.Array,
    w_up: jax.Array,
    w_down: jax.Array,
    router_bias: jax.Array,
    *,
    num_experts: int,
    top_k: int,
    combine_renorm_sum: float,
    mesh,
    data_axes: tuple[str, ...],
) -> jax.Array:
    """Run TE's fused MoE block on ``x`` [B, S, H] -> [B, S, H].

    ``gate_kernel`` [H, E]; ``w_gate``/``w_up`` [E, H, I]; ``w_down`` [E, I, H]
    (TE order: ``wi_0`` = gate, ``wi_1`` = up, ``silu(gate) * up``).
    ``data_axes`` are the mesh batch axes outside ``expert``; batch rows must
    arrive sharded ``((*data_axes, "expert"), None, None)``.
    """
    if _te_moe_fused is None:
        raise ModuleNotFoundError(
            "moe_implementation='te_moe' requires a transformer-engine build with the JAX MoE block"
        ) from _TE_IMPORT_ERROR

    batch_spec = P((*data_axes, "expert"), None, None)

    def _body(x_b, gate_b, w_gate_b, w_up_b, w_down_b, bias_b):
        out, aux_loss = _te_moe_fused(
            x_b,
            gate_b,
            w_gate_b,
            w_up_b,
            w_down_b,
            expert_bias=bias_b,
            num_experts=num_experts,
            num_experts_per_tok=top_k,
            activation_type="silu",
            score_function="sigmoid",
            scaling_factor=float(combine_renorm_sum),
            aux_loss_coeff=0.0,
            ep_axis="expert",
            data_parallelism_axes=data_axes,
            # Empty logical-axis tuples: skip TE's flax logical-axis
            # constraints; sharding is pinned by auto_axes out_sharding and
            # the explicit weight specs below.
            input_axes=(),
            gate_kernel_axes=(),
            wi_kernel_axes=(),
            wo_kernel_axes=(),
            dtype=x_b.dtype,
        )
        assert aux_loss is None
        return out

    # The bench mesh types every axis Explicit; TE's moe() issues its own
    # with_sharding_constraint calls written for auto-sharding. Same move as
    # ep_nccl.py: run the block under auto axes, pin only the output.
    return jax.sharding.auto_axes(_body, axes=tuple(mesh.axis_names), out_sharding=batch_spec)(
        x,
        gate_kernel,
        w_gate,
        w_up,
        w_down,
        router_bias.astype(jnp.float32),
    )
