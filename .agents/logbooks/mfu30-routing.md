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
