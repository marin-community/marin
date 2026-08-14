# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Correctness simulator for the Marin EP kernel.

Executable model of `experiments/marin_ep/SPEC.md` Sections 3-4: `S` device
programs exchanging messages through an abstract machine with a single
primitive, `put` (remote contiguous write), executed bulk-synchronously per
phase. The GPU kernel realizes the same algorithm with NVLink remote stores
and tile arrival flags; both must produce identical results (SPEC Section 6
invariants).

Tensor math uses eager jax.numpy (fp32 accumulation via
`preferred_element_type`, dtype cast points pinned by the spec); index and
routing metadata use NumPy. Every remote write and expert GEMM emits a
trace event; the perf model consumes these traces.

Structure of a forward run:
  F1 count -> F2 count exchange -> F3 dispatch puts -> F4 expert GEMMs
  -> F5 combine-return puts -> F6 fp32 reduce
Backward reuses the forward routing metadata (no recount):
  B1 dc -> B2 dZ -> B3 dispatch puts -> B4 expert MLP backward
  -> B5 return puts -> B6 token-grad reduce
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from experiments.marin_ep.oracle import expert_capacity

_F32 = jnp.float32


@dataclass(frozen=True)
class CommEvent:
    """One aggregated remote write (src == dst means NVLink is not crossed)."""

    src: int
    dst: int
    num_bytes: int
    phase: str


@dataclass(frozen=True)
class ComputeEvent:
    device: int
    flops: int
    phase: str


@dataclass
class Trace:
    events: list[CommEvent | ComputeEvent] = field(default_factory=list)

    def put(self, src: int, dst: int, num_bytes: int, phase: str) -> None:
        self.events.append(CommEvent(src, dst, num_bytes, phase))

    def compute(self, device: int, flops: int, phase: str) -> None:
        self.events.append(ComputeEvent(device, flops, phase))

    def comm_bytes(self, *, phase: str | None = None, remote_only: bool = True) -> int:
        return sum(
            e.num_bytes
            for e in self.events
            if isinstance(e, CommEvent) and (phase is None or e.phase == phase) and not (remote_only and e.src == e.dst)
        )

    def total_flops(self, *, phase: str | None = None) -> int:
        return sum(e.flops for e in self.events if isinstance(e, ComputeEvent) and (phase is None or e.phase == phase))


@dataclass
class Routing:
    """Forward routing metadata, shared verbatim by the backward pass.

    `routes[d][g] = (slots, offsets)`: device `d`'s kept assignments to
    expert `g` — flat local assignment slots (`t*K + k`, ascending) and
    their row offsets in the owner's receive buffer for `g`.
    """

    capacity: int
    counts: np.ndarray  # [S, E] per-device assignment counts
    base: np.ndarray  # [S, E] exclusive prefix over devices
    keep: np.ndarray  # [S, T, K] kept-assignment mask
    routes: list[list[tuple[np.ndarray, np.ndarray]]]
    valid_rows: np.ndarray  # [E] rows received per expert: min(N_g, C)
    dropped_total: int


@dataclass
class Saved:
    """Residuals the forward pass saves for backward."""

    routing: Routing
    recv_x: list[jax.Array]  # per owner device: [El, C, H] dispatched inputs
    returned: jax.Array  # [S, T, K, H] per-assignment expert outputs


@dataclass
class ForwardResult:
    y: jax.Array  # [S, T, H]
    dropped_total: int
    saved: Saved
    trace: Trace


@dataclass
class BackwardResult:
    dx: jax.Array  # [S, T, H]
    dw13: jax.Array  # [S, El, H, 2I], fp32
    dw2: jax.Array  # [S, El, I, H], fp32
    dcombine_weights: jax.Array  # [S, T, K], fp32
    trace: Trace


def _routing(selected_experts: np.ndarray, *, num_experts: int, capacity_factor: float) -> Routing:
    """Phases F1-F2: per-device counts, count exchange, offsets, keep rule."""
    num_devices, tokens, topk = selected_experts.shape
    assignments = tokens * topk
    capacity = expert_capacity(num_devices * assignments, num_experts, capacity_factor)

    counts = np.zeros((num_devices, num_experts), dtype=np.int64)
    rank_local = np.zeros((num_devices, assignments), dtype=np.int64)
    for d in range(num_devices):
        flat = selected_experts[d].reshape(assignments)
        counts[d] = np.bincount(flat, minlength=num_experts)
        order = np.argsort(flat, kind="stable")
        segment_start = np.cumsum(counts[d]) - counts[d]
        rank_sorted = np.arange(assignments, dtype=np.int64) - segment_start[flat[order]]
        rank_local[d, order] = rank_sorted

    base = np.cumsum(counts, axis=0) - counts

    keep = np.zeros((num_devices, assignments), dtype=bool)
    routes: list[list[tuple[np.ndarray, np.ndarray]]] = []
    for d in range(num_devices):
        flat = selected_experts[d].reshape(assignments)
        keep[d] = base[d, flat] + rank_local[d] < capacity
        device_routes = []
        for g in range(num_experts):
            slots = np.nonzero((flat == g) & keep[d])[0]
            offsets = base[d, g] + rank_local[d, slots]
            device_routes.append((slots, offsets))
        routes.append(device_routes)

    total_counts = counts.sum(axis=0)
    return Routing(
        capacity=capacity,
        counts=counts,
        base=base,
        keep=keep.reshape(num_devices, tokens, topk),
        routes=routes,
        valid_rows=np.minimum(total_counts, capacity),
        dropped_total=int(np.maximum(total_counts - capacity, 0).sum()),
    )


def _emit_count_exchange(trace: Trace, num_devices: int, num_experts: int, phase: str) -> None:
    for src in range(num_devices):
        for dst in range(num_devices):
            if src != dst:
                trace.put(src, dst, num_experts * 4, phase)


def _swiglu(hidden: jax.Array, intermediate_dim: int, activation_fn: Callable) -> jax.Array:
    gate, up = jnp.split(hidden, [intermediate_dim], axis=-1)
    return activation_fn(gate) * up


def simulate_forward(
    x: Float[Array, "S T H"],
    selected_experts: np.ndarray,
    combine_weights: Float[Array, "S T K"],
    w13: Float[Array, "S El H I2"],
    w2: Float[Array, "S El I H"],
    *,
    num_experts: int,
    capacity_factor: float,
    activation_fn: Callable[[jax.Array], jax.Array],
) -> ForwardResult:
    """Run SPEC Section 3 (F1-F6) and return output, residuals, and trace."""
    num_devices, tokens, hidden_dim = x.shape
    topk = selected_experts.shape[2]
    local_experts = w13.shape[1]
    intermediate_dim = w2.shape[2]
    if local_experts * num_devices != num_experts:
        raise ValueError(f"E={num_experts} must equal S*El={num_devices}*{local_experts}")
    compute_dtype = x.dtype
    itemsize = jnp.dtype(compute_dtype).itemsize
    trace = Trace()

    routing = _routing(np.asarray(selected_experts), num_experts=num_experts, capacity_factor=capacity_factor)
    _emit_count_exchange(trace, num_devices, num_experts, "fwd_count_exchange")

    # F3: dispatch puts into per-owner receive buffers [El, C, H].
    recv_x = [jnp.zeros((local_experts, routing.capacity, hidden_dim), compute_dtype) for _ in range(num_devices)]
    x_assign = jnp.asarray(x)  # [S, T, H]; row for assignment slot a is x[d, a // K]
    for d in range(num_devices):
        for g in range(num_experts):
            slots, offsets = routing.routes[d][g]
            if slots.size == 0:
                continue
            owner, local_g = g // local_experts, g % local_experts
            rows = x_assign[d, slots // topk].astype(compute_dtype)
            recv_x[owner] = recv_x[owner].at[local_g, offsets].set(rows)
            trace.put(d, owner, slots.size * hidden_dim * itemsize, "fwd_dispatch")

    # F4: expert GEMMs (fp32 accumulation, outputs in compute dtype).
    expert_out: list[list[jax.Array]] = []
    for owner in range(num_devices):
        outs = []
        for local_g in range(local_experts):
            g = owner * local_experts + local_g
            m = int(routing.valid_rows[g])
            rows = recv_x[owner][local_g, :m]
            hidden = jnp.matmul(rows, w13[owner, local_g], preferred_element_type=_F32).astype(compute_dtype)
            act_out = _swiglu(hidden, intermediate_dim, activation_fn)
            z = jnp.matmul(act_out, w2[owner, local_g], preferred_element_type=_F32).astype(compute_dtype)
            outs.append(z)
            trace.compute(owner, 6 * m * hidden_dim * intermediate_dim, "fwd_expert_gemm")
        expert_out.append(outs)

    # F5: combine-return puts into per-source [T*K, H] slot buffers.
    returned = jnp.zeros((num_devices, tokens * topk, hidden_dim), compute_dtype)
    for d in range(num_devices):
        for g in range(num_experts):
            slots, offsets = routing.routes[d][g]
            if slots.size == 0:
                continue
            owner, local_g = g // local_experts, g % local_experts
            returned = returned.at[d, slots].set(expert_out[owner][local_g][offsets])
            trace.put(owner, d, slots.size * hidden_dim * itemsize, "fwd_combine")

    # F6: fp32 reduce over the K slots, cast to x dtype.
    returned = returned.reshape(num_devices, tokens, topk, hidden_dim)
    y = jnp.einsum(
        "stkh,stk->sth",
        returned,
        jnp.asarray(combine_weights).astype(compute_dtype),
        preferred_element_type=_F32,
    ).astype(compute_dtype)

    saved = Saved(routing=routing, recv_x=recv_x, returned=returned)
    return ForwardResult(y=y, dropped_total=routing.dropped_total, saved=saved, trace=trace)


def simulate_backward(
    dy: Float[Array, "S T H"],
    combine_weights: Float[Array, "S T K"],
    w13: Float[Array, "S El H I2"],
    w2: Float[Array, "S El I H"],
    saved: Saved,
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
) -> BackwardResult:
    """Run SPEC Section 4 (B1-B6), reusing forward routing and residuals.

    Token activations are not an input: the dispatched rows live in
    `saved.recv_x`, exactly as the kernel's backward reads them.
    """
    routing = saved.routing
    num_devices, tokens, hidden_dim = dy.shape
    local_experts = w13.shape[1]
    intermediate_dim = w2.shape[2]
    num_experts = num_devices * local_experts
    topk = saved.returned.shape[2]
    compute_dtype = dy.dtype
    itemsize = jnp.dtype(compute_dtype).itemsize
    trace = Trace()

    keep = jnp.asarray(routing.keep)
    dy = jnp.asarray(dy)
    weights = jnp.asarray(combine_weights)

    # B1: combine-weight grads (fp32, zero at dropped slots).
    dcombine = jnp.einsum("stkh,sth->stk", saved.returned.astype(_F32), dy.astype(_F32))
    dcombine = jnp.where(keep, dcombine, 0.0)

    # B2: per-assignment cotangents.
    dz_slots = jnp.where(
        keep[..., None],
        weights.astype(compute_dtype)[..., None] * dy[:, :, None, :],
        jnp.zeros((), compute_dtype),
    ).reshape(num_devices, tokens * topk, hidden_dim)

    # B3: dispatch dZ along the forward routes.
    recv_dz = [jnp.zeros((local_experts, routing.capacity, hidden_dim), compute_dtype) for _ in range(num_devices)]
    for d in range(num_devices):
        for g in range(num_experts):
            slots, offsets = routing.routes[d][g]
            if slots.size == 0:
                continue
            owner, local_g = g // local_experts, g % local_experts
            recv_dz[owner] = recv_dz[owner].at[local_g, offsets].set(dz_slots[d, slots])
            trace.put(d, owner, slots.size * hidden_dim * itemsize, "bwd_dispatch")

    # B4: expert MLP backward (recomputes the hidden from saved inputs).
    dw13 = jnp.zeros(w13.shape, _F32)
    dw2 = jnp.zeros(w2.shape, _F32)
    dx_expert: list[list[jax.Array]] = []
    for owner in range(num_devices):
        outs = []
        for local_g in range(local_experts):
            g = owner * local_experts + local_g
            m = int(routing.valid_rows[g])
            rows = saved.recv_x[owner][local_g, :m]
            dz = recv_dz[owner][local_g, :m]
            hidden = jnp.matmul(rows, w13[owner, local_g], preferred_element_type=_F32).astype(compute_dtype)
            gate, up = jnp.split(hidden, [intermediate_dim], axis=-1)
            act_gate, act_vjp = jax.vjp(activation_fn, gate)
            act_out = act_gate * up

            d_act_out = jnp.matmul(dz, w2[owner, local_g].T, preferred_element_type=_F32).astype(compute_dtype)
            dw2 = dw2.at[owner, local_g].set(jnp.matmul(act_out.T, dz, preferred_element_type=_F32))
            dgate = act_vjp(d_act_out * up)[0]
            dup = d_act_out * act_gate
            dgu = jnp.concatenate([dgate, dup], axis=-1)
            dw13 = dw13.at[owner, local_g].set(jnp.matmul(rows.T, dgu, preferred_element_type=_F32))
            dx_rows = jnp.matmul(dgu, w13[owner, local_g].T, preferred_element_type=_F32).astype(compute_dtype)
            outs.append(dx_rows)
            # dW13 + dX are 4*m*H*I each; dW2 + d_act_out are 2*m*H*I each.
            trace.compute(owner, 12 * m * hidden_dim * intermediate_dim, "bwd_expert_gemm")
        dx_expert.append(outs)

    # B5: return dX rows to source assignment slots.
    dx_slots = jnp.zeros((num_devices, tokens * topk, hidden_dim), compute_dtype)
    for d in range(num_devices):
        for g in range(num_experts):
            slots, offsets = routing.routes[d][g]
            if slots.size == 0:
                continue
            owner, local_g = g // local_experts, g % local_experts
            dx_slots = dx_slots.at[d, slots].set(dx_expert[owner][local_g][offsets])
            trace.put(owner, d, slots.size * hidden_dim * itemsize, "bwd_combine")

    # B6: fp32 sum over K slots, cast to x dtype.
    dx = dx_slots.reshape(num_devices, tokens, topk, hidden_dim).astype(_F32).sum(axis=2).astype(compute_dtype)
    return BackwardResult(dx=dx, dw13=dw13, dw2=dw2, dcombine_weights=dcombine, trace=trace)
