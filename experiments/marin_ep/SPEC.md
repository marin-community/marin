# Marin EP kernel — behavior specification

This spec defines the *intended behavior* of the Marin EP fused MoE kernel,
independent of any implementation (oracle, simulator, or GPU kernel). Code
conforms to this document; when they disagree, one of them has a bug and the
discrepancy is resolved here first. Design rationale and milestones live in
`.agents/projects/20260814_marin_ep_kernel.md`.

Version: v1 (2026-08-14). Sections marked *(v2)* are planned extensions that
v1 must not preclude.

## 1. Contract

The kernel implements the body of a `jax.shard_map` over mesh axis
`"expert"` of size `S`, drop-in compatible with the backends dispatched by
`levanter.grug.grug_moe.moe_mlp` (`shard_local_fn` signature):

Inputs, per device `d ∈ [0, S)`:

| tensor | shape | dtype | meaning |
|---|---|---|---|
| `x_d` | `[T, H]` | compute dtype (bf16 in production) | routed activations (LatentMoE width) |
| `e_d` | `[T, K]` | int32 | selected expert ids, values in `[0, E)` |
| `c_d` | `[T, K]` | float | combine weights |
| `w13_d` | `[El, H, 2I]` | compute dtype | gate/up weights, `El = E / S` local experts |
| `w2_d` | `[El, I, H]` | compute dtype | down weights |

Static parameters: `activation_fn` (SwiGLU gate activation, silu in
production), `num_experts = E`, `capacity_factor = cf`.

Outputs, per device:

- `y_d [T, H]` in `x` dtype: combined MoE output (Section 4).
- `dropped_total []` int32: **global** count of dropped assignments,
  identical on every device (psum semantics).

Expert ownership: global expert `g` is owned by device `g // El` with local
index `g % El` (matches `fixed_all_to_all`).

Derived sizes: `A = T*K` assignments per device, `A_global = S*A`.

Duplicate expert ids within one token's `K` entries are permitted and
treated as distinct assignments (same as existing backends).

## 2. Drop rule: pooled per-expert capacity

Unlike `fixed_all_to_all` (per `(source shard, expert)` cell capacity, an
idle cell cannot lend rows to a hot cell), Marin EP pools capacity per
expert across all source shards:

- Per-expert capacity:
  `C = min(max(1, ceil(cf * A_global / E)), A_global)` rows (the clamp makes
  any sufficiently large `cf` exactly dropless without unbounded buffers).
  Per-device receive pool is `El * C = ceil-ish(cf * A)` rows — the same
  memory bound as the ragged backend's receiver pool.
- **Arrival order** of assignments to expert `g` is source-major: sort by
  `(source device id, local flat assignment index)`, where the local flat
  index is `t*K + k`. This is the global flat assignment order when tokens
  are partitioned contiguously across devices.
- An assignment with arrival rank `r_g` for expert `g` is **kept** iff
  `r_g < C`.
- `dropped_total = Σ_g max(0, N_g − C)` where `N_g` is the global count of
  assignments to expert `g`.

Consequences (enforced as test invariants, Section 6):

- The kept set is a pure function of the *global* routing, independent of
  `S`. EP1 / EP4 / EP64 produce identical outputs for the same global batch
  and contiguous token partitioning.
- `cf` large enough ⇒ zero drops ⇒ output equals the dense reference.
- Dropped assignments contribute exactly zero to `y`; combine weights are
  **not** renormalized after drops (matches existing backends).

Rank computation is distributed-friendly: with per-device counts
`n_d[g]` (`N` is the `[S, E]` count matrix), the base offset of device
`d`'s rows for expert `g` is `base[d, g] = Σ_{d' < d} n_{d'}[g]`, and an
assignment's arrival rank is `base[d, g] + r_local` where `r_local` is its
rank among device-`d` assignments to `g` in local flat order. Keep iff
`base[d, g] + r_local < C`.

## 3. Forward algorithm (abstract machine)

The kernel is specified against a message-passing machine with primitives
`put(dst_device, buffer, offset, rows)` (remote contiguous write) and
per-region arrival signaling. The correctness simulator implements this
machine bulk-synchronously; the GPU kernel implements it with NVLink
remote stores + tile arrival flags. Both must produce identical results.

Phase F1 — **count**: device `d` computes `n_d[g]` for all `g`, plus local
sort metadata (assignments grouped by expert in local flat order).

Phase F2 — **count exchange**: device `d` sends `n_d[·]` (int32, `E`
entries) to every other device (or: to each owner, the `El` relevant
entries; v1 sim sends the full row — a few KB). After F2 every device
knows `base[d, g]` and `N_g` for its own experts, and every sender knows
`base[d, g]` for all `g` (computable from the full count matrix).

Phase F3 — **dispatch**: for each local expert `g` on owner `o(g)`, a
receive buffer `X_g` of `C` rows (width `H`, wire dtype) exists. Device
`d` writes its kept assignments' activation rows to
`X_g[base[d,g] + r_local]` via `put`. Rows are the token's `x` row (an
`x` row is written once per kept assignment, i.e. up to `K` times total).
Wire dtype v1 = `x` dtype. *(v2: fp8 wire with per-token scales; scales
must never be shared across tokens along the sequence axis.)*

Phase F4 — **expert GEMM**: for each local expert `g`, with
`m_g = min(N_g, C)` valid rows:
`Hd = X_g[:m_g] @ w13[g]` → split `[.., :I]` gate / `[.., I:]` up (contiguous
split, not interleaved) → `Z_g = (act(gate) * up) @ w2[g]`.
GEMM accumulation in fp32; `Hd` and `Z_g` stored in compute dtype.

Phase F5 — **combine return**: owner writes each row `Z_g[j]` back to its
source device `d(j)` at that assignment's slot in a `[T, K, H]` return
buffer (slot = the local flat index `t*K + k` of the assignment). Slots of
dropped assignments remain zero.

Phase F6 — **reduce**: `y_d[t] = cast_x( Σ_k c_d[t,k] * R_d[t,k,:] )` with
the sum accumulated in fp32 (`R` is the return buffer read in compute
dtype). This matches the `fixed_all_to_all` einsum with
`preferred_element_type=f32`.

Metadata retained for backward: local sort metadata, count matrix /
offsets, keep mask, and the saved tensors `X_g` (dispatched inputs) and
per-assignment expert outputs `R` (or recompute — v1 saves; remat policy
integration mirrors `MOE_REMAT_SAVE_NAMES`).

## 4. Backward algorithm

Given `dy_d [T, H]` (cotangent of `y_d`, in compute dtype):

Phase B1 — **combine-weight grad** (local):
`dc_d[t,k] = Σ_h dy_d[t,h] * R_d[t,k,h]`, fp32, zero for dropped slots.

Phase B2 — **assignment cotangents** (local):
`dZ_d[t,k,:] = c_d[t,k] * dy_d[t,:]` for kept slots, zero otherwise.

Phase B3 — **dispatch of `dZ`**: identical routes and offsets as F3 (reuse
forward metadata; no recount). Owner receives `dZ_g [m_g, H]`.

Phase B4 — **expert MLP backward** (local to owner), per expert `g`:
- `dW2[g] = act_out^T @ dZ_g` where `act_out = act(gate) * up`
  (recomputed from saved `X_g` and `w13[g]`; v1 recomputes the `[m_g, 2I]`
  hidden rather than saving it).
- `dHd = dZ_g @ w2[g]^T` → SwiGLU backward → `dGU [m_g, 2I]`.
- `dW13[g] = X_g^T @ dGU`.
- `dX_g = dGU @ w13[g]^T`.
All GEMMs accumulate fp32; weight grads emitted in fp32. Weight grads are
local (expert-sharded) — no collective.

Phase B5 — **return of `dX_g`**: same transport as F5, into a `[T, K, H]`
buffer on the source device.

Phase B6 — **token grad reduce** (local):
`dx_d[t] = cast_x( Σ_k dX_d[t,k,:] )`, fp32 accumulation; dropped slots
contribute zero. (Note the current `fixed_all_to_all` backward materializes
this as a 6 GiB fp32 buffer at hero shape; the kernel must fuse the k-sum.)

The transport pattern is symmetric: forward and backward each use one
dispatch-direction and one combine-direction transfer of `A` rows × `H` ×
wire bytes (pre-drop).

## 5. Numerics

- All GEMMs (F4, B4) accumulate in fp32 regardless of input dtype.
- Combine reduces (F6, B1, B6) accumulate in fp32; outputs cast to `x`
  dtype (`dc` stays fp32 until the caller casts).
- Activation `act` is evaluated in compute dtype (matches existing
  backends evaluating on bf16 tensors).
- Reduction *order* within a token's `k` slots is fixed (slot order), so
  results are deterministic and independent of message arrival order.
- Tolerances for conformance testing: fp32 end-to-end ⇒
  `rtol=atol=1e-5` vs oracle; bf16 wire/compute ⇒ `rtol=atol=2e-2` vs
  fp32 oracle and `rtol=atol=1e-6` between any two conforming bf16
  implementations with identical cast points (exactness only where cast
  points are pinned by this spec).

## 6. Invariants (fuzz-enforced)

- **I1 dropless-parity**: with `cf` such that `C ≥ max_g N_g`, output
  equals the dense oracle (up to dtype tolerance) and `dropped_total = 0`.
- **I2 drop accounting**: `dropped_total = Σ_g max(0, N_g − C)` exactly.
- **I3 EP invariance**: for the same global `(x, e, c)` contiguously
  partitioned, outputs and `dropped_total` are identical for every `S`
  dividing `E` (up to dtype tolerance at pinned cast points).
- **I4 drop semantics**: zeroing the kept-mask complement in the oracle
  reproduces the kernel's output exactly — drops are per-assignment zero
  contributions, nothing else.
- **I5 gradient conformance**: explicit backward (Section 4) matches JAX
  autodiff of the keep-masked oracle for `dx`, `dW13`, `dW2`, `dc`.
- **I6 determinism**: repeated evaluation is bitwise identical (fixed
  reduction orders; no atomics-order dependence observable in results).

## 7. Buffers (hero shape, per device, forward)

For reference at `T=65536, K=4, H=3072, E=192, El=3, S=64, cf=1.33`:
`A = 262144`, `C = ceil(1.33 * 16,777,216 / 192) = 116,218` rows,
receive pool `El*C = 348,654` rows ≈ 2.0 GiB bf16 (same bound as today's
send cells), return buffer `[T, K, H]` ≈ 1.5 GiB bf16. No fp32 `[A, H]`
intermediate anywhere (contrast: current backward's 6 GiB `grad_rows`).

## 8. Out of scope for v1 (must not be precluded)

- fp8 wire dtype with per-token scales (#7665 composition).
- fp8/MXFP8 expert GEMMs.
- Tile-granular overlap of transport with the surrounding attention
  layer's compute.
- Multi-rack EP (> NVLink domain) with hierarchical dispatch.
