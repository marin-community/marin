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
