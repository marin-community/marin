# mfu30 agent B: MoE routing marshal

Branch `research/mcwitt/mfu30-routing` (worktree `~/projects/marin.mfu30-routing`), based on the campaign
branch `research/mcwitt/mfu30` (main `f38da1173d`). Campaign logbook: `.agents/logbooks/mfu30.md`.
Run ids `m30b-<short>-<NN>`, Iris JAX ports 33200-33249. Scripts: `autoresearch/loop-260930-mfu30/b/`.

## M30B-001 Marshal anatomy of the Sep 24 main trace (2026-09-30)

Source: `remeasure-main-144k` (rank 0, 3 steps), compute stream only, seconds per step. MoE geometry
from the HLO: T=65536 tokens per shard, K=8, TK=524288, H=3072, I=3072, E=384 (6 local experts per
shard, 2 chunks of 3), chunk capacity C=301466 rows (1.15x). A [TK, H] bf16 buffer is 3.2 GB; a
[C, H] buffer is 1.85 GB. 48 calls per step per MoE kernel.

| op (HLO) | phase | s/step | what it is |
|---|---|---|---|
| `loop_or_fusion`, `.1` (moe_chunk_{0,1}/or) | bwd | 0.125 | SwiGLU backward over [C, 2I], gate/up packed in u32. Not a mask: the "or" packs the two bf16 outputs. 9.3 GB per call at ~7 TB/s |
| `wrapped_add` (moe_chunk_0/add_any) | bwd | 0.063 | sum of the two chunks' dispatch-operand cotangents, [TK, H]; the two are disjoint |
| `loop_select_fusion.3` (moe_chunk_1/select_n) | bwd | 0.043 | `_ragged_a2a` transpose masking the chained return buffer's cotangent, [TK, H] |
| `combine/jit(argsort)/sort` (cub, memsets) | fwd | 0.068 | inverse permutation for the combine gather-sum |
| `dispatch/jit(argsort)` sort + `wrapped_iota.6.remat` | bwd | 0.039 | inverse permutation for the dispatch gradient; the s32 iota alone reads 457 us/call (stalled behind the a2a) |
| `dispatch/gather` | fwd + remat | 0.120 | [TK, H] expert-sorted gather |
| sonic gather-sum fwd / remat / bwd, dispatch bwd | all | 0.137 | at 5-7 TB/s; one pass each |
| zero inits (`_loop_local_zeros`) | all | ~0.17 | fwd [C,H] x2 0.026, return [TK,H] 0.024, remat same, bwd [TK,H] x2 0.052, [C,H] x2 0.025 |

The per-kernel durations of the small sort/iota/memset kernels are inflated by SM contention with the
device-initiated ragged all-to-all (an s32[524288] iota at 457 us), so their savings on the critical path
are uncertain until a trace shows them.

## M30B-002 Inverse routing port (d1ccdd9959)

Cherry-picked `c2b3a4e8db` onto main without conflicts; hunks verified in the intended functions. It
replaces the three `jnp.argsort(permutation)` sites with a unique-index scatter. Index values are
identical, so outputs must be bitwise equal. CPU: `lib/levanter/tests/grug/` 145 passed. The Sep 20
repeatability pair `/mwittmann/hero2-repeatability-20260920-1317-p1` is not recoverable: no W&B run and
no Iris record on the hub or either peer.

## M30B-003 Chained a2a cotangents (c67ee1f965)

The chunk loop writes each chunk's transport rows into buffers that are zero on those rows, but
`_ragged_a2a` differentiated an overwrite. Replaced it with `_ragged_a2a_add` (output cotangent passes to
the init unchanged) and `_ragged_a2a_add_forwarding` (dispatch forwards the sorted buffer to the next
chunk, so the backward writes into the later chunks' cotangent). Lowered StableHLO on 4 CPU devices: the
[TK, H] `add` and `select` and two of four [TK, H] broadcasts disappear. Expected saving 0.13 s/step
(add 0.063 + select 0.043 + zero fill 0.021-0.031), exact.

Gate: `autoresearch/loop-260930-mfu30/b/routing_gate.py` compares main's module (frozen copy), the
inverse-only module and the branch module on a GB200x4 node: bitwise output and gradients (x, combine
weights, w13, w2), control repeatability, and component timing at hero per-shard shapes.

## M30B-004 GB200x4 gate: both changes bitwise exact (2026-09-30)

Job `/mwittmann/m30b-gate-chain-02` (branch c67ee1f965 + gate script), 4 GB200, expert axis 4, hero env
(`cuda_async`, fraction 0.75, device-kernel ragged a2a with symmetric buffers). The first attempt
(`m30b-gate-inv-01`) ran with the default BFC allocator, and NCCL failed to register the whole 138 GiB pool as
symmetric memory; the hero env fixes it.

| case | T/shard, H=I | drops | inverse vs main | inverse+chain vs main | fwd+bwd ms: main / inverse / both |
|---|---|---|---|---|---|
| small-uniform | 2048, 512 | 0 | bitwise | bitwise | - |
| small-skewed-drops | 2048, 512 | 12685 | bitwise | bitwise | - |
| small-padded | 2048, 512 | 0 | bitwise | bitwise | - |
| small-one-hot | 2048, 512 | 3668 | bitwise | bitwise | - |
| hero-uniform | 65536, 3072 | 0 | bitwise | bitwise | 86.81 / 86.52 / 84.73 |
| hero-skewed-drops (+padding) | 65536, 3072 | 358395 | bitwise | bitwise | 83.13 / 82.78 / 80.61 |

Bitwise covers the output, the drop count and the gradients of x, the combine weights, w13 and w2. Main is
repeatable (two runs bitwise equal). One MoE layer, forward plus backward without remat: inverse routing
saves 0.3 ms (0.3%), the chained cotangents a further 1.8-2.2 ms (2.4-3.1% combined). Compiler temp drops
by 1.37 GB (one [TK, H] buffer). Scaled to 48 layers: ~0.1 s/step, before any remat or contention effect.
In isolation the inverse sort costs far less than the 0.107 s/step the training trace attributes to it,
consistent with its kernels being stretched by the concurrent device-initiated a2a in training.

## M30B-005 Unfilled transport buffers: hoisting probe (2026-09-30)

After B, the transport buffers still get a zero fill each: fwd [C, H] x2 0.026 + [TK, H] 0.024, the same
again in the remat, bwd [TK, H] ~0.025 + [C, H] x2 0.025, ~0.15 s/step in the Sep 24 trace. The
consumers can be made to read only written rows, so the fills are removable; the question is how to get
an unfilled buffer that stays inside the layer loop (#8822).

Probe `scan_probe.py` (3-layer rematted scan, T/shard 16384, H=I=1024, 4 GB200; jobs
`m30b-probe-empty-0{1..5}`, the first with loop-invariant routing, which confounds the forward):

| init | fwd loop | bwd loop (remat + bwd) | step ms |
|---|---|---|---|
| loop-local zeros (main) | 3 fills | 6 fills | 18.71 |
| `jax.lax.empty` (`AllocateBuffer`, no operands) | hoisted to entry by JAX's scan partial eval, 3 copies per layer | 6 allocations, in loop | 18.46 |
| Triton kernel that writes nothing, operand = loop-carried marker | nothing | 2 copies [C, H] | 18.23 |

The two remaining copies feed a synchronous `ragged-all-to-all.N.2` clone made by XLA's
HloRematerialization (the `.remat`/`.2` family agent A traced to the missing host-offload flag); a
custom-call output cannot be cloned, so the clone and the original share it and one gets a copy.

## M30B-006 Candidate C: unfilled transport buffers (commit after c67ee1f965)

- `_transport_buffer` (was `_loop_local_zeros`): on GPU a no-op Triton kernel (`sonic.unwritten_buffer`)
  whose operand is the same loop-carried `min(tie[0], -site) + site` marker; zeros elsewhere.
- Expert MLP protocol: rows past the active count in `x_dispatch` and in the output cotangent are
  unspecified. QuACK bounds every grouped GEMM by `cu`; the portable `ragged_dot` path masks input and
  output rows with `where`.
- Accepted-assignment mask (`_accepted_assignments`: rank within the expert group < accepted prefix),
  computed before the dispatch gather. The combine weights are `where(accepted, w, 0)`, the dispatch
  gradient sums only accepted slots, and the Sonic gather-sum kernel does not load zero-weight rows.
  The weight gradient read from a dropped row is discarded by the `where` transpose (a select, so NaN
  garbage cannot leak).
- Numerics: identical values; only the sign of an all-zero weight gradient on a dropped slot can differ.
- CPU tests: new behavior tests with NaN in the unspecified rows (dispatch gradient, combine, portable
  expert MLP) and an accepted-prefix reference test. Gate adds `quack_contract.py` (each QuACK grouped
  GEMM with NaN vs zero rows past `cu[-1]`) and `scan_compare.py` (control / chain / candidate in a
  rematted scan: fills, copies, temp bytes, step time).

## M30B-007 Candidate C gate: exact, no fills, +1.5-3% per MoE layer over B (2026-09-30)

Job `/mwittmann/m30b-gate-unfilled-01` (GB200x4, hero env), candidate C at e612b34244.

- Routing gate, all six cases (drops, padding, one-hot, hero shapes): candidate C is bitwise equal to main
  in the output, drop count and all four gradients. Main repeatable.
- One MoE layer fwd+bwd at hero per-shard shapes, main / chain (A+B) / C: uniform 87.74 / 84.83 / 83.73 ms
  (C +4.8% vs main); skewed with drops 83.50 / 81.57 / 79.95 ms (+4.4%).
- `quack_contract.py`: gated forward, down forward, dh, dx, dw2 and dw13 give identical finite active
  rows (weight gradients: identical) with NaN or zero rows past `cu[-1]`.
- `scan_compare.py` (3 rematted layers, T/shard 32768, H=I=2048): main has 7 fills in the backward loop
  and 3 in the forward loop, chain 6 + 3, C none and no copies (10 no-op kernels). Temp 8.419 / 8.393 /
  8.394 GB. Final loss identical. Step 91.25 / 88.48 / 85.96 ms: chain +3.1%, C +6.1% vs main.
- GPU pytest (`test_grugformer_moe.py`): 79 passed, 3 failed, the same three that fail with the inverse-only
  and chain branches: `test_moe_ep_path_lowers_on_abstract_mesh[ragged_all_to_all]`,
  `test_moe_mlp_runs_with_ep_axis_when_available`,
  `test_moe_mlp_reports_positive_drop_count_in_ragged_a2a_when_over_capacity`. All three feed float32
  activations into the SM100 QuACK path, which asserts "gated aux output must be 16-bit". CI runs these on
  CPU, where the ragged path is skipped, so they are GPU-only failures independent of this branch.

## M30B-008 Rack arm: A+B+C stacked, profiled (2026-09-30 18:41 UTC)

Cancelled `m30b-invchain-01` (A+B; queued since 18:00 UTC, gang never admitted) and submitted
`m30b-unfilled-01` (`/mwittmann/m30b-unfilled-01-coord`, port 33201, code e612b34244, branch head
fcec70f93d) with `--profile-start-step 180021 --profile-steps 3`. All three changes are bitwise exact on
the gate, so one stacked arm measures the deployable set; the profile attributes the removed kernels.
Expected: +1.5-2% throughput (0.2-0.25 s/step if the component wins carry over), loss equal to
`mhep-ctx4k-s0-20260930` at every step if the hero step is deterministic, peak HBM unchanged or lower.

## M30B-009 Pricing: expert-side router gradient + saved MoE output (2026-09-30)

Fidelity ruling (orchestrator): rounding/reassociation changes are allowed inside the same-code loss
band; anything that changes drops, capacity or chunking is not.

Idea: the backward needs the down-projection output y only through `returned`, for (a) the combine's
router-weight gradient dS[t,k] = <dout[t], y[t,k]> and (b) the combine output, which feeds the W_up
weight gradient and the SConv input. Save the combine output (the latent MoE output) and compute dS on
the expert side from values the MLP backward already has: dy = w * dout rows, dh = dy @ W2^T, so
<dout, y> = rowsum(dh * h) / w. The remat then no longer needs y, `returned` or the combine.

Design (keeps the forward and dx, dW13, dW2 bitwise, changes only dS at rounding level):
- One `custom_vjp` over `_moe_mlp_ep_ragged_a2a_local`. Its backward is the current one written out:
  the weighted gather of dout into sorted order, then per chunk in reverse: reverse-return a2a, the QuACK
  backward, the reverse dispatch chained into one buffer. It adds s = rowsum(dh * h) per expert row
  (fused into the SwiGLU backward or a separate [C, I] reduce), one small [C, 1] f32 a2a per chunk that
  returns s to the token side, and dS = s / w on accepted assignments with w != 0 (0 otherwise; the
  router weights are renormalized sigmoids in bf16, so an exactly-zero accepted weight is not expected).
- Remat policy saves the MoE latent output (checkpoint name, on device). JAX's remat DCE then drops the
  recomputed down GEMM, both return a2as, and the combine gather-sum.
- The chunk barrier currently ties chunk c+1's dispatch to chunk c's `returned`. That would keep the
  return alive in the remat. It must tie to chunk c's SwiGLU output h instead, which lets dispatch c+1
  start after the gated GEMM c in the forward pass (one extra [C, H] buffer live, schedule change).

Price from the Sep 24 trace, s/step:
| item | change |
|---|---|
| remat down GEMM, 2 chunks (`ffi_call.341/.342`) | -0.279 |
| remat return a2a, exposed (`ragged-all-to-all.1.1` 0.181, `.3.1` 0.001) | -0.18 |
| remat combine gather-sum (`triton_kernel_call.17`) | -0.027 |
| combine backward: Sonic bwd (reads `returned`) -> gather-multiply | -0.015 to -0.03 |
| SM contention: `.3.1` (0.207 busy) no longer shares SMs with GEMMs (~22% per C) | ~-0.04, low confidence |
| s = rowsum(dh * h): fused, or a separate [C, I] reduce | 0 to +0.05 |
| 96 small a2as in the backward | <= +0.005 |
| stacking the saved output per layer | +0.005 |
| **net** | **-0.40 to -0.50** |

HBM: saved latent output [65536, 3072] bf16 = 402.7 MB/layer, 48 layers = 19.3 GB = 18.0 GiB on device.
The recompute no longer materializes `returned` (3.2 GB) or the two y chunks (1.85 GB each); whether that
lowers the backward peak depends on where the peak sits. Fallback: offload the saved output to pinned host
like the carry (0.4 GB per layer each way).

## M30B-010 Expert-side routing-weight gradient: built and gated (2026-09-30)

Commits 992a33a975 (custom_vjp + saved routed output), then the row dot fused into the SwiGLU backward,
then a barrier ordering the backward transports after the recomputed dispatch.

- `_routed_experts` (custom_vjp) covers dispatch to combine. Its backward is the previous one written out
  plus s = <h, dh> per expert row (h recomputed in fp32 from gu inside the SwiGLU-backward fusion), one
  [C, 1] f32 ragged a2a per chunk returning s along the forward return's routes, and dS = s / w on
  accepted assignments with w != 0 (0 otherwise, by `where`, not by division).
- `_ExpertMlp` now has `forward`/`backward` (QuACK: `_expert_mlp_quack_wgrad_fwd` +
  `_expert_mlp_quack_wgrad_backward`; portable: ragged_dot with masks, its backward via `jax.vjp`).
- Hero model: `MOE_OUTPUT_REMAT_NAME` tags the routed output before W_up; the offload_carry policy saves it.
- The chunk barrier ties dispatch c+1 to chunk c's expert-MLP residuals, not to its return.

Gate `m30b-gate-sonic-02` (GB200x4, hero XLA flags incl. overlap limit 1):
- Single layer, 6 routing cases: out, drops, dx, dW13, dW2 equal to main; dS max 0.4-0.6% of the largest
  gradient, median 0.53% elementwise (about one bf16 ulp). Single-layer fwd+bwd without remat: +4.0% vs
  main (A+B+C: +3.4%); the remat saving does not show here by construction.
- 3-layer rematted scan, T/shard 65536, H=I=3072: HLO confirms the recompute lost both down-projection
  QuACK GEMMs, both return a2as and the combine gather-sum (no copies). dx, dW13, dW2 equal to main, dS
  max 0.40% (median 0). Step: main 338.8 ms, A+B+C 326.0 (+3.9%), A+B+C+D 290.7 (+16.5%, i.e. -11.8 ms per
  layer vs A+B+C, ~0.57 s/step at 48 layers). Temp: 35.30 / 32.08 / 31.92 GB: the saved outputs (1.21 GB
  for 3 layers) are more than offset, peak excluding the stack drops ~1.4 GB.
- Row dot fused vs separate (reading the saved h): 292.5 vs 291.3 ms (noise), temp 31.92 vs 33.66 GB. Fused kept.
- pytest: 85 passed, same 3 GPU-only failures as main (f32 into QuACK). New: dS in the EP dense-parity test,
  QuACK row-dot test.
- dS is 0 for accepted assignments with an exactly-zero weight (the exact value is <dout, y>). The first
  gate's one-hot case built weights from logits boosted by 100, so 7 of 8 weights were exactly 0 in bf16 and
  dS disagreed there. Router sigmoids reach 0 only below ~1e-38, where d w / d logit is as small; the gate now
  draws weights from separate logits.

Race under collective overlap > 1: without `--xla_gpu_experimental_parallel_collective_overlap_limit=1`
(first scan run, and `m30b-diag-sonic-02` at hero shapes), the candidate's gradients are not repeatable
run to run (main is). The backward's first transport no longer depends on the recompute, so a backward
ragged a2a can be in flight with a recomputed dispatch. With the hero's forced limit of 1 (`m30b-diag-sonic-01`,
32768/2048, all configurations) everything is repeatable and exact. A barrier restoring main's order is in
test (`m30b-diag-sonic-03`, no flags).

HBM interaction: XLA's post-schedule HloRematerialization already binds on the hero (A's M30A-002: it counts
the 36 GiB pinned-host carry stack as device memory and recomputes 10.6 GiB in the backward body). The 18 GiB
saved-output stack is caller usage at the backward while loop, so without A's H-A4 flag
(`--xla_gpu_enable_host_memory_offloading=true`) XLA would likely rematerialize ~18 GiB more in the backward.
The D arm should run with that flag.

## M30B-011 Overlap > 1 races; barrier reverted; ragged runs force overlap 1 (2026-09-30)

`m30b-diag-sonic-02` (hero shapes, no overlap flag): main repeatable; the candidate's gradients differ run
to run, and the next configuration hung until cancelled. `m30b-diag-sonic-03` added a barrier holding the
backward transports until the recompute's last dispatch (ea3e4d6194): the candidate configuration hung
again, so the barrier does not remove the hazard (other backward transports are independent too) and it
would cost overlap under limit 1. Reverted (eea9d79d4b). `train.py` now forces
`--xla_gpu_experimental_parallel_collective_overlap_limit=1` for every ragged run (ce112504f1), as the
carry offload already did; the hero configuration was already at 1. Under limit 1 (`m30b-diag-sonic-01`,
`m30b-scan-sonic-02`, `m30b-gate-sonic-02`) all configurations are repeatable and exact.

## M30B-012 Rack arm status (2026-09-30)

`m30b-unfilled-01` cancelled (the coordinator `--timeout` counts Kueue queue time). Resubmitted without a
timeout as `m30b-unfilled-02` (`/mwittmann/m30b-unfilled-02-coord`, port 33202, from worktree
`~/projects/marin.mfu30-routing-arm` pinned at fcec70f93d = e612b34244 lib, profiled 180021-180023).
Scoring (orchestrator): median over 180011-180059 against `mhep-ctx4k-s0-20260930` (28.255%, 13.892 s,
peak 103.09 GiB; loss 180000 1.261413, 180001 1.234596, 180002 1.200221, 180003 1.256785).

## M30B-013 D rack arm submitted (2026-09-30 20:12 UTC)

`m30b-sonic-01` (`/mwittmann/m30b-sonic-01-coord`, port 33203), code dd45f27c17 (= ce112504f1 + TF_CPP_
forwarding), queued alongside `m30b-unfilled-02` by exception (FIFO queue ~5 h). Env:
`XLA_FLAGS=--xla_gpu_enable_host_memory_offloading=true`, `TF_CPP_MIN_LOG_LEVEL=0`,
`TF_CPP_VMODULE=hlo_rematerialization=1`; profile 180021-180023. It doubles as the restore smoke. Valid only
if loss at 180000 == 1.261413 exactly, 180001-180003 within ~1e-4 of `mhep-ctx4k-s0-20260930`, and drops and
router balancing/QB statistics stay in family through 180059. Attribution: D+flag vs `m30b-unfilled-02`
(A+B+C) and A's `m30a-hmo-02` (flag alone).

Correctness constraint of D (record in any landing PR): the backward's all-to-alls no longer depend on the
recomputed forward's, so D requires a collective overlap limit of 1. ce112504f1 forces it for every ragged
run, which changes non-hero ragged configurations that inherited the default limit (4).

## M30B-014 PR #9481 transport folded into `_routed_experts` (a1d699e67e)

- Mirror parameters: the backward sends each transport's cotangent along its mirror transfer (reverse
  return along `dispatch_params`, reverse dispatch and the row-dot return along `return_params`) and drops
  `_reverse_ragged_a2a`'s two offset all-to-alls. #9481's test that the two parameter sets are each
  other's transpose is included.
- Pipelined chunks: plans first, dispatch chunk 0, then in chunk c dispatch c+1 behind
  `barrier((sorted_x, x_dispatch_c))` and hold chunk c's return behind `barrier((out_c, next_x_dispatch))`,
  keeping only `out_c` from the second barrier so the recompute can still drop the return and down GEMM.
- Not on the D arm; for C's stack. Gate `m30b-gate-fold-01` (variants control / unfilled / stack = C's port of
  #9481 on A+B+C / sonic = D / candidate = D + fold).

## M30B-015 #9481 fold measured: mirror params kept, pipelining dropped (2026-09-30)

Gate `m30b-gate-fold-01` (fold a1d699e67e = D + mirror params + pipelined chunks): single layer exact for
out/drops/dx/dW13/dW2 and identical dS to D; QuACK contract passes; pytest 86 passed (the #9481 transpose test
included), same 3 GPU-only failures. Recompute still drops the down GEMMs and the combine (HLO census).

3-layer rematted scan at hero per-shard shapes (`m30b-gate-fold-01`, `m30b-scan-fold-03`), median step:
| variant | ms | temp GB |
|---|---|---|
| main | 345.4 / 346.3 | 35.30 |
| A+B+C (unfilled) | 332.1 | 32.08 |
| A+B+C + #9481 (C's stack port, full remat) | 323.6 | 31.96 |
| D (sonic, ce112504f1) | 298.4 / 298.6 | 31.92 |
| D + mirror params | 299.0 | 31.92 |
| D + mirror + pipelined chunks (a1d699e67e) | 307.7 / 307.4 | 31.21 |

On four GPUs the pipelining costs 8.7 ms per 3-layer step inside D (it helps +2.6% on A+B+C), and the mirror
parameters are neutral (their saving is per-rank latency of two offset all-to-alls, larger at EP64). Branch head
6a6bb78853 keeps the mirror parameters and sequential chunks; a1d699e67e is the pipelined variant if the rack
says otherwise. Gate for 6a6bb78853: `m30b-gate-mirror-01`.

## M30B-016 Gates: D + mirror (6a6bb78853) passes; E (SwiGLU backward in the dh GEMM epilogue) built

`m30b-gate-mirror-01` (6a6bb78853 = D + mirror params, sequential chunks): single layer identical to D (out,
drops, dx, dW13, dW2 equal to main; dS within rounding); scan main 341.2 / D 299.4 / D+mirror 298.8 ms;
pytest 86 passed + the same 3 GPU-only failures. This is the version for C's stack.

E (12643e682c): QuACK's grouped dh GEMM with a custom `packed_cd_b16x2` epilogue (`_dswiglu_row_dot_epilogue`,
QuACK's `dswiglu` + a scaled `ColVecReduce` for <h, dh>) replaces dh + the XLA SwiGLU-backward pass.
Single GB200, one chunk at hero per-shard shapes (C=301466, 262144 active rows, H=I=3072), `m30b-dswiglu-01`:
fused 3.54 ms vs dh GEMM + XLA pass 5.10 ms (dh GEMM alone 3.04 ms): -1.56 ms per call, ~-0.15 s/step at
96 calls. d_gate_up max 0.83% of the largest value (mean 5e-5), row dot max 0.17%: rounding level (fp32
SwiGLU backward on the unrounded accumulator). Changes dx and dW13 at rounding level, so the gate now
compares those to rounding (median and max) and keeps out/drops/dW2 exact. Gate `m30b-gate-epi-02`.

## M30B-017 E gate passes (`m30b-gate-epi-02`, 12643e682c = D + mirror + E)

Controls run the pre-E QuACK backward (frozen `unfused_backward.py`). Single layer, all six cases: out, drops and
dW2 equal to main; dx, dW13 and dS median 0.49-0.56% elementwise (about one bf16 ulp), max <= 0.82% of the largest
magnitude. Single-layer fwd+bwd: main 84.4 / D 81.4 / D+mirror+E 79.3 ms (uniform), 81.3 / 78.3 / 75.7 (skewed).
3-layer rematted scan: main 342.2 / D 298.7 / D+mirror+E 290.3 ms: E is -2.8 ms per layer (~-0.13 s/step at 48
layers on 4 GPUs); temp 35.30 / 31.92 / 31.82 GB; gradients vs main to rounding (dx, dW13 median 0.49%). pytest 88
passed (two new QuACK dswiglu tests) + the same 3 GPU-only failures.

## M30B-018 Idle and exposed collectives on the Sep 24 baseline (rank 0, 3 steps)

Tool: `b/exposure.py` (per-step head/tail/internal idle, gap context, exposed time per collective
instruction with per-instance duration spread). Fixed `exposed_detail.py`'s list/tuple sort crash.

Idle (0.346 s/step) is not an MFU lever. 0.323 s is the step tail (last kernel to the next step's launch:
callbacks, logging, data), which `throughput/duration` excludes (it times dispatch through
`block_until_ready(loss)`); 0.007 s is the head (dispatch to first kernel); internal gaps total 0.017 s, all
under 50 us (D2D copies and kernel boundaries).

Exposed collectives (1.681 s/step), by cause:
| class | s/step | instructions | removable by |
|---|---|---|---|
| ragged a2a, chunk 0 | 0.669 | fwd dispatch `.8.1` 0.183, fwd return `.9.1` 0.153, remat dispatch `.13` 0.153, remat return `.1.1` 0.181 | D removes the remat return (0.181). The rest is transfer time (min 2.9 ms ~ median 3.2-4.0 ms per instance), not skew |
| rank-skew waits | ~0.40 | u32 drop-count all-reduce `all-reduce.254` 0.241 (1/step, min 11 ms, median 97 ms), QB `pmin`/`pmax` in the forward 0.04, group-size all-gather in the remat `all-gather.123` 0.048, norm all-reduce 0.019 | structural: removing one sync moves the wait to the next collective |
| XLA remat clones | 0.246 | `all-gather.127.remat` 0.227 (96/step, synchronous), `all-gather.127` 0.019 | A's H-A4 flag |
| FSDP weight gathers / grad reduce-scatter | ~0.27 | fwd `all-gather.101/.108/.109/.104/.110/.107`, bwd `.26/.28/.30/.23`, `reduce-scatter.18` 0.058 | mostly time above the fastest instance, i.e. waits; scheduling/PGLE |
| offset all-to-alls in the transport backward | 0.010 | `all_to_all.30.1` | mirror parameters (6a6bb78853) |

Why chunk 0's transports are exposed, from one forward layer's kernel order: dispatch c0 (3.8 ms) runs with
nothing on the compute stream; gated + down GEMMs c0; the QB `pmin` (0.75 ms, a wait); return c0 (3.1 ms) again
alone; then dispatch c1 overlaps the four shared-expert GEMMs (~6 ms) and return c1 overlaps two more (~3.6 ms).
The latency-hiding scheduler spends the shared-expert GEMMs (~10 ms per layer) on chunk 1's transports and leaves
chunk 0's (6.9 ms per layer) bare. With pipelined chunks (dispatch c+1 under MLP c, return c under MLP c+1) the
shared-expert GEMMs are free to cover dispatch c0 and return c1, so every transport has compute under it. That is
the case #9481's pipelining targets, and a four-GPU scan without a shared expert cannot show it. Pipelined variant
for the rack: branch `research/mcwitt/mfu30-routing-pipelined` @ 64909b24d0 (= 12643e682c D+mirror+E with the
pipelined forward of a1d699e67e; values identical).

## M30B-019 Review of C's stack integration (`research/mcwitt/mfu30-stack` @ 8eff7b8ec4, `-pipelined` @ 7393a9ae26)

- `lib/levanter/src/levanter/grug/_moe/` is byte-identical to 12643e682c (sequential); the pipelined branch
  adds only 64909b24d0's forward. `train.py` (overlap-limit forcing) and `dispatch.py` (TF_CPP forwarding)
  unchanged from my lineage. `model.py` keeps `MOE_OUTPUT_REMAT_NAME` and the offload_carry save policy.
- #9481 model commits against D: the QB `_forward_barrier((s_minus_alpha, moe_out))` holds the whole MoE
  output, but QB statistics are not differentiated, so the recompute drops the barrier; the MLP-weight
  prefetch barrier sits upstream of the routed MLP. Checked with `b/model_dce.py` (the model smoke config, grad
  jaxpr traced on CPU, per scan body): forward scan 4 ragged a2a (2 dispatch + 2 return); backward scan 8
  (2 recomputed dispatches + 2 reverse returns + 2 reverse dispatches + 2 row-dot returns) on both branches,
  so the recomputed returns stay dead at model level. The pipelined branch shows 4 forward barriers (2 chunk +
  QB + prefetch) and 2 in the backward (the dead return-path barrier is gone).
- The portable ragged_dot expert MLP keeps y as a residual for its row dot, so only the QuACK path drops the
  recomputed down projection (by design; the hero is QuACK). The model smoke checks finiteness and norm
  parity, not the GPU-level drop of the QuACK down GEMM; my layer scan (`m30b-gate-epi-02`) covers that.
- No blocking issues found.

## M30B-020 Short-conv pricing (baseline trace; no build)

Per layer: SConv after the K projection (C=1536, attributed to attention), after the attention output and after
the MoE branch output (C=6144, [16, 4096, 6144] bf16 = 0.805 GB per tensor). Short-conv scope, s/step: forward
0.043 (two 6144 kernels, 0.46 ms each), remat 0.025 (one), backward 0.119 (two, 1.24-1.34 ms each). The kernel's own
docstring measures 3.0 HBM passes forward (floor 2) and 6.7 backward (floor 3): Pallas Triton cannot slice a
register tile, so each of the W=4 taps is an overlapping reload. At the floor and ~6.5 TB/s: backward 0.37 ms
per call (-0.87 ms x 96 = -0.083 s/step), forward 0.25 ms (-0.22 ms x 144 = -0.032 s/step). A causal-conv1d-style
CUDA/CuTe kernel with an SMEM/register ring carry (segment-id resets, fp32 dw partials) is worth ~0.08-0.11
s/step; a Triton rewrite cannot get there for the reason the docstring gives. Also available: summing the three
addends of the MoE-branch input (W_up + two shared down projections) inside the conv's load saves one pass,
~0.02 s/step.

## M30B-021 Streaming Triton short conv (`sconv_implementation=triton_gpu`)

Build: `lib/levanter/src/levanter/kernels/pallas/short_conv/triton_gpu.py` (raw Triton via jax_triton, not
CUDA/CuTe), selected by `short_conv(..., implementation="triton_gpu")`; hero switch `GrugModelConfig.
sconv_implementation` (all three ShortConv sites) and `launch_diagnostics.py --sconv-implementation`. The
implementation is a static module field, so the checkpoint tree is unchanged. Each program walks one sequence
chunk of one channel block with the previous 3 x rows (and, backward, the next 3 dy rows) in registers with their
segment ids; segment ids reset taps; the backward writes one fp32 dw partial per chunk, summed by the caller.
Launch shape per direction (`TritonShortConvTiles`): forward chunk 32, 1024 channels, 4 warps, 8 rows per step;
backward chunk 128, 256 channels, 4 warps, 8 rows per step.

Rounding: with the reference's per-op bf16 rounding written as f32 round trips, LLVM folds them into bf16 ops and
contracts multiply+add into `fma.rn.bf16x2` (one rounding where the reference has two), about 1 ulp off. The exact
path issues `mul.rn.bf16x2` / `add.rn.bf16x2` as inline PTX; the compiled PTX has no bf16 FMA.

Correctness (GB200, `m30b-sconv-02` default tiles, `m30b-sconv-03` tuned tiles; [16,4096,6144] and
[16,4096,1536]; unpacked, packed, 1-3 token documents, -1 padding): output and dx bit-identical to
`short_conv_reference` and to the Pallas kernel in every case; dw error against a float64 oracle equals the Pallas
kernel's (rel 1.9e-3 to 3.7e-3, the bf16 output rounding). The sweep (`m30b-sconv-sweep-03`) checked out/dx parity
for all 56 launch shapes at both widths: no shape breaks it. GPU tests added to `tests/kernels/test_short_conv.py`.

Per call, ten calls per jit (`b/sconv_sweep.py`), ms:

| shape | Pallas fwd | Triton fwd | Pallas bwd | Triton bwd | floor fwd / bwd (XLA 2- / 3-pass elementwise) |
|---|---|---|---|---|---|
| [16,4096,6144] | 0.437 | 0.262 | 1.065 | 0.480 | 0.242 / 0.350 |
| [16,4096,1536] | 0.119 | 0.082 | 0.295 | 0.151 | 0.073 / 0.099 |

The backward is occupancy-bound: 8 channels per thread needs 168 registers; 2 per thread needs 72-96 and runs at
~5 TB/s. Pipelining the row loop (`num_stages` 2-3) did nothing.

Block benchmark (`b/sconv_block_bench.py`, real hero Block on one GB200 with the routed MoE stubbed, remat +
backward, `m30b-sconv-block-02`): short-conv kernel time per layer 4.39 -> 2.20 ms (-2.19 ms); block step median
121.9 -> 120.3 ms (-1.6 ms, 3 interleaved reps each); temp 16.83 -> 16.22 GiB; loss bitwise equal. Gradient
leaves that differ Pallas vs Triton are the same leaves, at the same size (~1e-5 rel-rms), as Pallas vs a Pallas
rerun (attention weights, x, and `sconv_k` whose dy is attention's dk): run-to-run nondeterminism in the attention
backward. The only Triton-specific differences are the `sconv_attn` / `sconv_mlp` weight gradients at 1.7e-7
rel-rms (fp32 partial order). Over 48 layers the kernel saving is ~0.105 s/step (~0.76% of 13.892 s, ~+0.2 MFU
points if it stays on the critical path).

Hand-off to C: `b/sconv_on_stack.patch` is the full short-conv change (kernel, API, tests, model switch, launch
flag) resolved against `research/mcwitt/mfu30-stack` @ 8eff7b8ec4; it also applies cleanly to `-pipelined` @
7393a9ae26 (`git apply --index`). The only conflicts were the two switch plumbings side by side with C's
`gated_norm_implementation`; both are kept. Short-conv and hero-model CPU tests pass on the result. Arm flag:
`--sconv-implementation triton_gpu`. Not done: the three-addend summation in the conv load (skipped per the
orchestrator).

## M30B-022 Exposure of `m30b-unfilled-02` (A+B+C), and `m30b-sonic-02` submitted

Score (orchestrator): 28.668% / 13.692 s vs control 28.258% / 13.891 s (+0.41 MFU, -0.199 s/step), peak 103.75 GiB,
loss at 180000 exact (1.2614134550); later steps drift by up to 9.6e-4 (attention-backward nondeterminism, see
M30B-021; same-code repeat m30-ctl-s0-r2 queued to calibrate).

Profile: rank 0, steps 180021-180023 (`b/exposure.py`, `anatomy.py`, `exposed_detail.py`). Compared with the Sep 24
main profile (M30B-018) and A's `m30a-hmo-02` (current main + host-offloading flag); there is no profile of the
Sep 30 control.

| s/step | Sep 24 main | hmo-02 | unfilled-02 |
|---|---|---|---|
| span (profiled steps) | 14.439 | 13.766 | 13.762 |
| compute | 11.793 | 11.627 | 11.533 |
| exposed collectives | 1.681 | 1.443 | 1.516 |
| exposed copies | 0.619 | 0.618 | 0.631 |
| idle (tail) | 0.346 (0.323) | 0.077 (0.055) | 0.082 (0.055) |

- Routing marshal compute fell 0.37 s/step against Sep 24 (0.42 against hmo-02): `moe_expert_elementwise`
  0.372 -> 0.136 (backward -0.23), `moe_ep_other` 0.046 -> 0, `moe_combine` 0.150 -> 0.116, `moe_dispatch`
  0.194 -> 0.173, unmapped 0.066 -> 0.035. Against hmo-02 it pays ~0.23 s more remat compute (norms, attention
  elementwise, shared MLP in the recompute), which the flag removes; the two arms land at the same profiled span,
  so D + flag (sonic-02) is where they stack.
- Ragged a2a exposure is unchanged (0.838 vs 0.804): chunk 0's four transports are still bare (0.66 s; `.1.1` 0.199,
  `.9.1` 0.158, `.8.1` 0.151, `.13` 0.151). D removes the remat return; the pipelined variant targets the rest.
- The rank-skew wait moved, as M30B-018 predicted: the u32 drop-count all-reduce (0.241) is gone and the latent
  projection's backward reduce-scatter `reduce-scatter.18` now waits (exposed 0.058 -> 0.305, median 0.18 -> 1.58 ms
  per instance, 0.327 above the fastest instance). hmo-02 shows the same move (0.255), so it is a property of the
  current main, not of A+B+C.
- XLA remat-clone all-gathers (`all-gather.127.remat`, 0.227 on Sep 24) are gone in both current-main profiles
  (0.021 / 0.001).
- Exposed copies (0.63, unchanged, A's area): backward reloads of the offloaded carry (H2D `dynamic_slice` 0.218)
  and D2H `copy-start.97/98/99` (0.21), 0.37 of it in the last tenth of the step.

`m30b-sonic-02` submitted 2026-10-01 01:41 UTC (`/mwittmann/m30b-sonic-02-coord`, port 33210) from
`~/projects/marin.mfu30-routing-arm` at dd45f27c17 (D, unchanged from sonic-01, which the orchestrator cancelled
before it started). Env: `XLA_FLAGS=--xla_gpu_enable_host_memory_offloading=true
--xla_gpu_memory_limit_slop_factor=105`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.78` (A's pair), `TF_CPP_MIN_LOG_LEVEL=0`,
`TF_CPP_VMODULE=hlo_rematerialization=1`; seed 0, `--num-steps 180060`, profile 180021-180023, no timeout. Checks:
rank-0 "Rematerialized N instructions in module jit_train_step" N <~ 10, "Peak memory for main" (> ~110 -> try slop
110), W&B memory/limit_gib ~143.76, memory/peak_gib < ~139, loss at 180000 == 1.2614134550. The delta to
unfilled-02 is D + memory settings.

## M30B-023 `m30b-sonic-02` (A+B+C + D + H-A4, fraction 0.78 / slop 105): remat logs and exposure

Score (orchestrator): 29.783% / 13.179 s vs control 28.258% / 13.891 s (+1.52 MFU, -0.712 s/step), steady state
29.70-29.85, W&B memory/peak_gib 125.02 of 143.75; loss at 180000 exact; later dloss inside the same-code band.

Remat (rank logs, `TF_CPP_VMODULE=hlo_rematerialization=1`): "Rematerialized 0 instructions in module
jit_train_step" on all 64 processes. Rank 0: "HloRematerialization() with memory limit of 187.74GiB", "Peak memory
for main.808_spmd: 105.08GiB", "Peak memory usage of module (before): 178.72GiB", unchanged after. Limit and module
peak differ from A's 114.1 GiB device limit and the main peak by the same 73.64 GiB, which I read as host-space
buffers counted on both sides: main peaks at 105.08 GiB against 114.1, 9.0 GiB of headroom, nothing recomputed.

Profile (rank 0, steps 180021-180023), against unfilled-02 (s/step):

| | unfilled-02 | sonic-02 |
|---|---|---|
| span (profiled) | 13.762 | 13.229 |
| compute | 11.533 | 11.188 |
| exposed collectives | 1.516 | 1.336 (ragged a2a 0.838 -> 0.736) |
| exposed copies | 0.631 | 0.622 |
| idle | 0.082 | 0.084 |

Compute: D drops the recomputed down projection (`moe_expert_gemm` remat -0.263) and the recomputed return
(`moe_dispatch` remat -0.063, `moe_combine` remat -0.030); the flag drops XLA's remat clones (norms -0.083, attention
elementwise -0.087, shared MLP remat -0.120); D's expert-side backward adds `moe_combine` bwd +0.076,
`moe_expert_elementwise` bwd +0.069, `moe_dispatch` bwd +0.014. Attention is +0.08 in every phase (unexplained).

Ragged all-to-alls per layer, identified from operand shapes and consumers in the HLO (`b/ragged_order.py`):
- Forward: dispatch c0 (`.8.1`, 3.4 ms) 10% covered, 0.144 exposed; dispatch c1 (`.9.1`), return c0 (`.10.1`) and
  return c1 (`.11.1`) run under the shared-expert GEMMs and the c0 down projection (0.020 exposed). unfilled-02's
  forward exposed 0.339.
- Backward, in order: remat dispatch c0 (`.1.1`, median 5.6 ms, fastest 3.0 ms) 0% covered, 0.254 exposed, 0.111 of
  it above the fastest instance (rank skew at the MoE backward's entry); remat expert GEMMs c0 (5.9 ms) with no
  transport under them; remat dispatch c1 (`.3.1`, 3.4 ms) 0.169 exposed, held behind the c0 recompute by the chunk
  barrier; reverse return of dy c1 (`.2.1`, 3.1 ms) 0.149 exposed, issued directly behind it; reverse return of dy c0
  (`.13`) under the c1 recompute; row-dot returns (`.6.1` 0.9 ms, `.7.1` 1.2 ms, 0.112 busy) fully covered;
  reverse dispatches of dx (`.4.1`, `.5.1`, ~4.1 ms) under the shared-expert backward GEMMs. The recomputed
  return (unfilled-02 `.1.1`, 0.199 exposed) is gone.

C's checks: `stack/carry_stall.py` 2.8 ms/step exposed carry D2H (healthy; unfilled-02 15.6). `stack/copy_schedule.py`:
momentum H2D `copy-start.44/43/42` (10.12 GiB each) after the backward loop and D2H `copy-start.97/98/99` at the
end, the same placement as unfilled-02. Exposed copies (0.62): carry reloads in the backward (H2D `dynamic_slice`)
0.219, optimizer D2H `97/98/99` 0.211, H2D `44` 0.064; 0.364 of it in the last tenth of the step.

Remaining exposure by lever (s/step):
- Pipelined chunks (forward, and the recompute that reuses it): forward dispatch c0 0.144 (the shared-expert
  GEMMs could cover it once MLP c0 covers dispatch c1); remat dispatch c1 0.169 (no barrier between dispatch c1 and
  the c0 recompute, so it can run under the 5.9 ms of c0 remat GEMMs). Up to ~0.31.
- Scheduling / PGLE: reverse return dy c1 0.149 (dy is ready at the MoE backward's entry; the c0 remat GEMMs have
  no transport under them); FSDP all-gathers ~0.27 (`remat_carry` bwd ring 0.110, forward ring 0.104, `moe_dispatch`
  forward 0.053).
- Not schedulable: rank-skew waits, `reduce-scatter.18` 0.274 + ring 0.066 (0.296 above the fastest instance) and
  0.111 inside remat dispatch c0. Remat dispatch c0's transfer (0.143) has only the shared-expert backward GEMMs
  as candidate cover, and they currently cover the reverse dispatches.
- Copies (A's area): 0.62, mostly at the step end.

## M30B-024 Recompute pipelining check and forward-order backward

(a) Does the pipelined branch's recompute pipeline? `b/recompute_deps.py` traces the stack model's grad jaxpr on
CPU, flattens the backward scan body into one dataflow graph, and lists each ragged all-to-all's transitive
inputs. `mfu30-stack-pipelined` (MoE module = 64909b24d0): the recomputed dispatch c1 depends only on the
recomputed dispatch c0 (barrier on `x_dispatch` c0), no c0 expert matmul, so it can run under the c0 recompute
GEMMs; the return-path barrier is dead in the recompute and the backward body has 8 all-to-alls, no return or
down GEMM. `mfu30-stack` (sequential): the recomputed dispatch c1 has the two c0 recompute matmuls among its
inputs (barrier on c0's expert-MLP residuals), the serialization behind sonic-02's 0.169 s. Both reverse returns of
dy have no inputs but dy, in both variants: their placement is purely the scheduler's.

(b) Forward-order backward: `_routed_experts_bwd` runs chunks 0, 1 instead of 1, 0, so chunk 0's backward and its
returns depend only on the c0 recompute and dy c0's reverse return, never on dispatch c1 or the c1 recompute
(`recompute_deps.py`). Commits: sequential 547bf2ad20 (`research/mcwitt/mfu30-routing`), pipelined 2cc470d88f
(`research/mcwitt/mfu30-routing-pipelined`); each applies cleanly to C's matching stack branch (ff0ce13b29,
62095665a6).

Gate `m30b-reorder-gate-01` (GB200x4, `b/reorder_gate.sh`): out/dropped/dx/dS/dW13/dW2 bitwise for forward order
vs D, pipelined forward order vs pipelined, and pipelined vs D, in all six routing cases (small uniform, skewed
with 12,685 drops, padded, one-hot; hero uniform, hero skewed+padded with 357,643 drops). 3-layer rematted scans:
gradients bitwise for both pairs; step time unchanged (D 0.2917 s vs forward order 0.2919; pipelined 0.2966 vs
0.2968). pytest 88 passed, plus the same 3 GPU-only failures as main (f32 into QuACK). In the MoE-only scan the
scheduler still leaves most incoming transports without a GEMM under them in either order (it has no shared
experts or attention to place), so the scan cannot show the hero effect; that needs a rack trace. If the hero
schedule does not move, the deterministic option is a manual remat: the bwd rule recomputes the dispatch and
gate/up itself, so barriers inside it can order the transports.

## M30B-025 Attribution: `m30c-stackpipe-trace-03` vs `m30b-sonic-02`

Scores (orchestrator): stackpipe-03 29.768% / 13.186 s vs sonic-02 29.783% / 13.179 s, net zero, with mirror + E +
#9481 model commits (MLP-weight prefetch, QB after the MLP, attention re-gather on) + Triton short conv + pipelined
chunks on top of D. Profiled step 2 of stackpipe-03 is a 14.41 s outlier (two 55-68 ms idle gaps, rank stall), so
the comparison uses profiled steps 1 and 3 of each trace (`b/split_steps.py`; spans without the step tail: sonic-02
13.177 s, stackpipe-03 13.144 s).

| s/step | sonic-02 | stackpipe-03 | delta |
|---|---|---|---|
| compute | 11.211 | 10.820 | -0.391 |
| exposed collectives | 1.324 | 1.527 | +0.203 |
| exposed copies | 0.620 | 0.774 | +0.154 |
| idle | 0.022 | 0.025 | +0.003 |

Compute (-0.39): E -0.15 (`moe_expert_elementwise` -0.198, `moe_expert_gemm` bwd +0.050); Triton short conv -0.110
(kernel time 0.218 -> 0.108; its kernels fall into other scopes, so the scope table shows -0.188 and +0.08
elsewhere); GEMM contention moved, -0.05 net (shared-expert forward GEMMs -0.132 now run alone, expert GEMMs fwd and
remat +0.081 now run under the pipelined transports); attention -0.05, router/dispatch/shared elementwise -0.07.

Exposed collectives (+0.20), all of it ragged all-to-all (+0.214):
- forward return c1 0.021 -> 0.188 (+0.167): the shared-expert GEMMs, which covered return c1 (and dispatch c1, return
  c0) in sonic-02, now run after the routed MoE, between the QB collectives; the pipelined MoE GEMMs cover dispatch c1
  and return c0 instead;
- forward dispatch c0 0.141 -> 0.179 (+0.038), still bare;
- backward: recomputed dispatch c1 0.167 -> 0.150, still bare: the scheduler issues recomputed dispatch c0, dy c1's
  reverse return and recomputed dispatch c1 back to back at the MoE backward's entry, then runs the c0 recompute under
  dy c0's return. dy c1's reverse return 0.149 (unchanged, bare). Recomputed dispatch c0 +0.026 (skew).
- #9481's collectives net ~+0.01: the prefetch removes the in-MoE weight gather (-0.052); the attention re-gather's
  forward gathers are +0.093 under a new scope, against -0.061 on the old FSDP gathers; backward `remat_carry` gathers
  +0.034.

Exposed copies (+0.15): the forward carry D2H stall, `carry_stall.py` 147 ms/step (sonic-02 2.8). The carry copy
shares a memcpy stream with the first weight slice a layer needs; any program change can redraw it.

Loss: max |d| 6.8e-4, late mean +7.7e-5, 38/49 positive, above the same-code max of 3.3e-4. Of the stacked changes
only E changes values (dx and dW13 at bf16-rounding level, median ~1 ulp); D's dS change is the same as in
sonic-02 (inside the band), the short conv is bitwise in out/dx with dw at 1e-7, pipelining and the mirror and #9481
model commits are bitwise. So E is the expected source; consecutive-step loss deltas are strongly correlated, so
38/49 is not evidence of bias on its own.

## M30B-026 Attribution: `m30c-stackseq-trace-03` vs sonic-02 and stackpipe-03

Profiled steps 1 and 3 of each trace (no outlier in stackseq-03: 13.173, 13.159, 13.080 s). Spans without the
step tail: sonic-02 13.177, stackpipe-03 13.144, stackseq-03 13.087 s (-0.090 vs sonic-02, ~+0.2 MFU points from
the profiled steps alone; the score is the orchestrator's).

| s/step | sonic-02 | stackpipe-03 | stackseq-03 |
|---|---|---|---|
| compute | 11.211 | 10.820 | 10.804 |
| exposed collectives | 1.324 | 1.527 | 1.491 |
| of which ragged a2a | 0.732 | 0.947 | 0.838 |
| exposed copies | 0.620 | 0.774 | 0.769 |

Ragged all-to-alls (s/step exposed; instructions identified from the HLO):

| transport | sonic-02 | stackpipe-03 | stackseq-03 |
|---|---|---|---|
| fwd dispatch c0 | 0.141 | 0.179 | 0.187 |
| fwd dispatch c1 / return c0 | 0 / 0 | 0 / 0 | 0.008 / 0 |
| fwd return c1 | 0.021 | 0.188 | 0.184 |
| bwd recomputed dispatch c0 | 0.254 | 0.280 | 0.292 |
| bwd dy c1 reverse return | 0.149 | 0.149 | 0 (under the c0 recompute GEMMs) |
| bwd recomputed dispatch c1 | 0.167 | 0.150 | 0.167 |

- Shared experts after the MoE: also in the sequential arm, so not caused by pipelining. In both -03 arms the
  shared-expert forward GEMMs run after the routed MoE, interleaved with the QB collectives (psum, pmin, pmax,
  psum_invariant, 0.15-0.9 ms each) that #9481's QB-after-MLP commit moved behind the MoE output; in sonic-02 they
  covered dispatch c1, return c0 and return c1. Return c1 is bare in both (+0.16), and the expert GEMMs, now the
  ones under the transports, are slower (+0.12 in both) while the shared GEMMs, alone, are faster (-0.105). The
  shared experts do not depend on the MoE output, so this is the scheduler's choice; QB-after-MLP is the likely
  trigger, not yet isolated.
- Carry stall: hit in stackseq-03 too (`carry_stall.py` 141 ms/step), as in stackpipe-03 (147); sonic-02 2.8.
  Both -03 arms run the production wheel; A's stream fix is in F1.
- Sequential vs pipelined: sequential exposes 0.109 less ragged time, because its scheduler put dy c1's reverse
  return under the c0 recompute GEMMs; pipelined's ran bare between the two recomputed dispatches.
- Compute in stackseq-03 matches stackpipe-03 (E -0.15, Triton short conv -0.11, GEMM contention shift ~0,
  attention and small items -0.15).
- Re-gather: forward attention gathers +0.096 under the new scope against -0.043 on the old ones, backward
  `remat_carry` gathers +0.046; net about +0.1 (dropped in F1).

## M30B-027 `m30-f1-seq-02` attribution; QB-after-MLP isolation attempts

F1-seq-02 (final-seq d4234c88e7, stream fix on, re-gather off): 30.219% / 12.990 s vs stackseq-03 29.928% / 13.115 s
(orchestrator). Profiled steps 1 and 3 (12.985, 12.969 s without the tail):

| s/step | stackseq-03 | F1-seq-02 | delta |
|---|---|---|---|
| span | 13.087 | 12.977 | -0.110 |
| compute | 10.803 | 10.854 | +0.051 (expert GEMMs +0.037, attention projections +0.014) |
| exposed collectives | 1.491 | 1.473 | -0.018 |
| exposed copies | 0.769 | 0.625 | -0.144 (carry stall gone: `carry_stall.py` 3.7 ms/step) |

- Re-gather off: forward attention gathers return to the FSDP scope; net forward all-gather exposure -0.025, backward
  `remat_carry` gathers unchanged (0.161 -> 0.168).
- Forward order, at the MoE backward entry: recomputed dispatch c1 moved under chunk 0's backward GEMMs (0.167 ->
  0.010), but the scheduler now puts dy c0's reverse return directly behind recomputed dispatch c0 with nothing
  under it (0.149) and dy c1's under the c0 recompute GEMMs. Net ragged exposure unchanged (0.838 -> 0.843).

Remaining exposure in F1-seq-02 (s/step), the baseline for the noqb comparison:

| transport | exposed |
|---|---|
| fwd dispatch c0 | 0.192 |
| fwd dispatch c1 / return c0 | 0.006 / 0 |
| fwd return c1 | 0.186 |
| bwd recomputed dispatch c0 | 0.300 (fastest instance ~3 ms; the rest is skew) |
| bwd dy c0 reverse return | 0.149 |
| bwd dy c1 reverse return | 0 |
| bwd recomputed dispatch c1 | 0.010 |
| row-dot and dx returns | ~0.002 |

Other: latent-projection reduce-scatters 0.284 (skew), backward `remat_carry` gathers 0.168, forward FSDP gathers
0.149, exposed copies 0.625 (optimizer-state copies at the step end, carry reloads).

QB-after-MLP isolation: a GB200x4 compile-only schedule (`stack/forward_schedule.py`, model smoke and a hero-shaped
EP4 model) places the shared-expert GEMMs under both returns with and without the QB barrier, unlike EP64, so it
cannot attribute the rack placement. Variant branch `research/mcwitt/mfu30-final-seq-noqb` @ 3b88a218cc (clean revert
of 0b6113396b). Value check (`stack/qb_values.py`, model smoke, reference attention, 3 runs per job): within a job
the runs are identical (loss, all metrics, 35/36 gradient leaves); across the two branches the loss differs at 1e-7
relative, layer-0 margin_max at 6e-5, and qb_beta by 2.6% (0.2837 vs 0.2762, several histogram bins). Pending: the
same comparison with autotuning off, and a same-code repeat across jobs, to separate the barrier's numerics from
cross-job GEMM autotuning.

## M30B-028 noqb trace and the QB value check

`m30-f1-noqb-01` (final-seq with QB-after-MLP reverted): 30.127% / 13.029 s vs F1-seq-02 30.219% / 12.990 s
(orchestrator). Profiled steps 1 and 3: span 13.034 vs 12.977 s.
- QB-after-MLP is what put the shared experts after the MoE: with it reverted, the shared-expert forward GEMMs are back
  inside the MoE section, under dispatch c1 and return c0 (as in sonic-02). But return c1 stays bare (0.214 vs
  0.186): the sequential chunks' expert GEMMs would have covered dispatch c1 and return c0 anyway, and the shared
  GEMMs are spent there. Forward dispatch c0 is a little better covered (0.131 vs 0.192). Ragged total -0.037.
- What makes noqb slower: the QB statistics' compute and collectives are back before the MoE (`other:jvp` forward
  compute +0.055, a `moe_dispatch` forward all-gather exposed +0.057), and the shared GEMMs, now under transports, are
  slower (+0.071) while the expert GEMMs are faster (-0.072).
- Covering return c1 needs compute that only becomes available at the end of the MoE: e.g. one shared expert held
  back until the last chunk's MLP output exists, so the scheduler has nothing else to run beside return c1. A
  JAX-level split of the shared experts with a barrier on the last chunk's MLP output would do it; not built.

QB value check (`b/qb_values.py`, model smoke, reference attention, 3 runs per job; all 3 runs identical in every
job):
- Same code, two jobs (`final-02` vs `final-04`, autotuning on): loss 7.657164574 vs 7.657171249, 40/88 metrics and
  35/36 gradient leaves differ, qb_beta 0.2837 vs 0.2769. XLA's GEMM autotuning picks different algorithms per job,
  so same-code runs are not bitwise across jobs.
- Autotuning off (`--xla_gpu_autotune_level=0`), final-seq vs noqb: loss equal, qb_beta, margins and router bias
  equal; only the logged router z-loss (layer 0 and total) and layer 2's load-balancing loss differ, at 1 fp32 ulp,
  and two gradient leaves (`output_proj`, `token_embed`) differ bitwise. The QB statistics are unchanged by the move;
  the residuals are reduction-order effects of the barrier, inside the rounding ruling.

## M30B-029 Shared-expert holdback (`research/mcwitt/mfu30-final-seq-holdback` @ ab78bbe3ad)

Off by default; `--held-back-shared-experts 1`. The routed MoE releases a gradient-free copy of `mlp_in` only after
an `optimization_barrier` on the last expert chunk's MLP residuals (inside `_routed_experts`, through
`moe_mlp_with_holdback` / `MoEExpertMlp.call_with_holdback`); `Block` gates the last shared expert's input on that
release with `_forward_barrier`, so its GEMMs become ready beside the last chunk's down projection and return. Tied
to the residuals, not the down-projection output, so the recompute (which produces the residuals anyway) gains no
GEMM. Parameters unchanged; `moe_mlp` untouched.

Checks: `b/recompute_deps.py held_back_shared_experts=1`: backward body 8 ragged all-to-alls and 16
`ragged_dot_general`, as without it (4 barriers vs 2); no new collectives. Gate `m30b-holdback-gate-02` (GB200x4):
module `moe_mlp` vs `moe_mlp_with_holdback` bitwise in out, drops, dx, dS, dW13, dW2 on small skewed+drops, small
padded and hero-shape skewed+padded; model smoke (0 vs 1 in one process): parameters, loss and all router metrics
bitwise, 4 of 36 gradient leaves differ (`shared[1].w_gate/w_up`, `token_embed`, `attn.w_q`); pytest 80 passed plus
the 3 GPU-only failures main has. The first version passed the held-back expert's cotangent back through the MoE
(31 leaves differed, bf16 cotangent summation order); the gradient-free token fixed that.

`m30b-holdback-diff-01` (`b/holdback_diff.py`): without the holdback XLA merges the two shared experts' gate/up GEMMs
with the latent down projection into one GEMM on `mlp_in` (output [2048, 2560] in the smoke); with it the held-back
expert's GEMMs are separate. Differing leaves: `shared[1].w_gate` max_rel 4.2e-3, `w_up` 4.0e-3 (about 1 bf16 ulp,
0.05% of elements), `token_embed` 8.3e-3 (2.6% of elements); medians 0. fp32 reassociation from the GEMM split,
inside the rounding ruling; the orchestrator accepted it.
