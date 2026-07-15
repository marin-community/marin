# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""TransformerEngine NCCL-EP fused-MoE backend for Grug.

This backend replaces grug's router + expert dispatch/compute/combine with the
fused ``transformer_engine.jax.moe`` layer, which internally runs:

    router (gate_kernel einsum + top-k score fn)
      -> tex.ep_dispatch  (NCCL-EP, hierarchical HT transport)
      -> tex.grouped_gemm (cuBLAS grouped GEMM, FC1)
      -> activation
      -> tex.grouped_gemm (FC2)
      -> tex.ep_combine   (NCCL-EP)

Unlike grug's other EP backends (``ep_ring`` / ``ep_deepep``) this is NOT a
``shard_local_fn`` invoked inside a per-shard ``shard_map``: ``te.moe`` is a
global ``custom_vjp`` primitive with its own SPMD partitioning over a named
``ep_axis``. It therefore plugs in at the ``MoEMLP`` block level, not at the
``moe_mlp`` dispatch boundary, and requires:

  * one GPU per JAX process (multi-controller; grug ``processes_per_task=8``),
  * an eager ``tex.ep_bootstrap`` + ``record_ep_bootstrap_signature_for_moe``
    before the first jit (see ``ensure_te_ep_bootstrap``),
  * an active ``Mesh`` with the ``ep_axis`` and ``global_shard_guard`` set.

ROUTER SEMANTICS CAVEAT: grug uses a bespoke quantile-bias (QB) router with
sigmoid combine weights renormalized to ``_ROUTING_RENORM_SUM`` and an
auxiliary-loss-free bias updated from a QB-beta statistic. ``te.moe`` uses its
own ``fused_topk_with_score_function`` (softmax/sigmoid top-k) and does not
emit the QB-beta statistic. This backend therefore does NOT reproduce grug's
exact routing; it is a throughput/feasibility bring-up, and the router-bias
update callback is inert under it.
"""

from __future__ import annotations

import threading

import jax

# Import guarded: TE with the NCCL-EP build is an optional, GPU-only dependency.
try:
    import transformer_engine.jax as te  # noqa: F401
    from transformer_engine.jax import cpp_extensions as tex
    from transformer_engine.jax.moe import moe as te_moe
    from transformer_engine.jax.moe import record_ep_bootstrap_signature_for_moe

    _TE_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised only without the TE build
    te = None  # type: ignore[assignment]
    tex = None  # type: ignore[assignment]
    te_moe = None  # type: ignore[assignment]
    record_ep_bootstrap_signature_for_moe = None  # type: ignore[assignment]
    _TE_IMPORT_ERROR = exc


_BOOTSTRAP_LOCK = threading.Lock()
_BOOTSTRAP_DONE = False


def te_available() -> bool:
    return _TE_IMPORT_ERROR is None


def require_te() -> None:
    if _TE_IMPORT_ERROR is not None:
        raise RuntimeError(
            "transformer_engine.jax NCCL-EP backend is unavailable. Build TE from source with "
            "NVTE_WITH_NCCL_EP=1 against a NCCL>=2.30.4 device-API build on sm_90. "
            f"Underlying import error: {_TE_IMPORT_ERROR!r}"
        )


def ensure_te_ep_bootstrap(
    *,
    num_experts: int,
    max_tokens_per_rank: int,
    recv_capacity_per_rank: int,
    hidden_dim: int,
    ep_size: int,
) -> None:
    """Eagerly initialize the NCCL-EP communicator once per process.

    Must be called inside an active ``Mesh`` + ``global_shard_guard`` context and
    BEFORE the first jit that calls ``te.moe``. ``max_tokens_per_rank`` is the
    per-rank dispatch token count (local_batch_per_gpu * seq_len) and is capped
    by the kernel ``MAX_SUPPORTED_TOKENS_PER_RANK`` (8192 on this branch).
    """
    global _BOOTSTRAP_DONE
    require_te()
    with _BOOTSTRAP_LOCK:
        if _BOOTSTRAP_DONE:
            return
        tex.ep_bootstrap(
            world_size=jax.process_count(),
            rank=jax.process_index(),
            num_experts=num_experts,
            max_tokens_per_rank=max_tokens_per_rank,
            recv_capacity_per_rank=recv_capacity_per_rank,
            hidden_dim=hidden_dim,
        )
        record_ep_bootstrap_signature_for_moe(
            num_experts=num_experts,
            max_tokens_per_rank=max_tokens_per_rank,
            recv_capacity_per_rank=recv_capacity_per_rank,
            hidden_dim=hidden_dim,
            ep_size=ep_size,
        )
        _BOOTSTRAP_DONE = True


def moe_te_nccl(
    x_bsd: jax.Array,
    router_kernel: jax.Array,
    w_gate: jax.Array,
    w_up: jax.Array,
    w_down: jax.Array,
    *,
    num_experts: int,
    num_experts_per_token: int,
    ep_axis: str,
    data_parallelism_axes: tuple[str, ...],
    recv_capacity_per_rank: int,
    score_function: str = "sigmoid",
) -> tuple[jax.Array, jax.Array | None]:
    """Run one grug MoE sublayer through TE's fused NCCL-EP ``moe``.

    Args:
        x_bsd: activations ``[B, S, H]``.
        router_kernel: grug router weight ``[H, E]`` -> TE ``gate_kernel``.
        w_gate/w_up: grug expert gate/up weights ``[E, H, I]`` -> ``wi_0``/``wi_1``.
        w_down: grug expert down weight ``[E, I, H]`` -> ``wo``.
    """
    require_te()
    # grug expert weights are [E, H, I] (gate/up) and [E, I, H] (down); TE expects
    # wi_* as [E, embed, mlp] and wo as [E, mlp, embed] which matches directly.
    output, aux_loss = te_moe(
        x_bsd,
        router_kernel,
        w_gate,
        w_up,
        w_down,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_token,
        activation_type="silu",
        score_function=score_function,
        ep_axis=ep_axis,
        data_parallelism_axes=data_parallelism_axes,
        dtype=x_bsd.dtype,
    )
    return output, aux_loss
