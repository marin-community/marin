FLA-style decomposition directive

Context:
The reference FlashLinearAttention (FLA) implementation structures the chunkwise GatedDeltaRule into a small set of
specialized kernels, rather than one fully-fused monolith. On TPU, this split can sometimes *increase* overall
throughput because each kernel becomes MXU-dense and easier for the compiler to schedule.

FLA (GPU) forward pipeline (conceptual):
1) `g = chunk_local_cumsum(g)`
2) `A = chunk_scaled_dot_kkt_fwd(k, g, beta)`
3) `A = solve_tril(A)`  (triangular inverse / WY representation)
4) `w, u = recompute_w_u_fwd(k, v, beta, A, g)`
5) `h, v_new = chunk_gated_delta_rule_fwd_h(k, w, u, g, initial_state)`
6) `o = chunk_fwd_o(q, k, v_new, h, g, scale)`

TPU experiment directive:

Pick **one** of these decomposition experiments and fully execute it (end-to-end + profile). Do not do partial refactors.

## Experiment A: 2-kernel split (Solve vs Recurrent)

Goal: remove the largest fusion barrier by splitting the segment kernel into:

- Kernel 1 (“solve kernel”): per-chunk compute of the chunk-local solve outputs needed downstream.
  Output candidates:
  - `k_cumdecay` (Ct×K)
  - `v_pseudo`   (Ct×V)
  - optional: `attn` or any chunk-only factors used in output/state update

- Kernel 2 (“recurrent kernel”): sequentially apply `S_prev` across chunks (still segmented or pipelined), producing
  `out` and `chunk_starts`, using the outputs of Kernel 1.

Why this can win:
- Kernel 1 is dominated by Ct×Ct / Ct×K dots + triangular solve.
- Kernel 2 is dominated by Ct×K @ K×V and K×Ct @ Ct×V updates; it is a clean MXU workload and a great candidate for
  **tiling over V blocks** (see Experiment B).

## Experiment B: Recurrent kernel tiles V (and optionally K)

Goal: reduce VMEM pressure and increase parallelism by running the recurrent kernel with grid axes:
`grid = (NH, V_blocks)` (and optionally `K_blocks`). Each program holds state `S_prev[K, Vb]` for a slice of V.

Implementation notes:
- Keep chunk-only quantities that are shared across V blocks either:
  - recomputed cheaply per block, or
  - computed once and staged via SMEM/VMEM scratch.
- In early iterations, accept some redundant compute if it enables large VMEM savings and better scheduling.

## Experiment C: FLA parity (multi-kernel)

Goal: mirror the FLA kernel boundaries on TPU and tune each kernel independently.

Implementation plan:
1) Start with forward-only correctness parity against existing reference.
2) Add backward one stage at a time.
3) Profile and tune only after correctness is solid.

Acceptance criteria for any experiment:
- Pass `lib/levanter/tests/test_gdn_kernels.py` and `lib/levanter/tests/test_gdn_layer.py`.
- Show a measurable reduction in XProf time for the dominant GDN custom calls.
- Update the hillclimb log with:
  - what split was attempted,
  - kernel boundaries and I/O tensors,
  - the exact profile run and trace link,
  - before/after hotspot timing.
