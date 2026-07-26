# AGENT_LOG — ep25-d5 (does the EP25 stack transfer to the REAL hero-run shape, d6144 4-of-256?)

Worktree: `/home/marin/projects/marin/.worktrees/ep25-d5-d6144`, branch `agent/ep25-d5-d6144`,
base `agent/ep25-d1-adjoint` @ 17f886f3d (custom adjoint + gather dispatch + drops metric + spill).
d1's own log is preserved in its own worktree; this file is ep25-d5's log from here down.

Mission (from coordinator):
1. Honest baseline at **d6144 · 4-of-256 · 48L**, QB-on, cf1.0, custom adjoint, drops, 120 steps,
   1 GB200 rack, EP64. Fall back to 4-of-128 if 707B does not fit.
2. Does **MXFP8 flip** at the fatter expert GEMM (i3072 vs i1280)? d2 measured -2.83pp at d5120/i1280
   and explicitly named "fatter expert GEMMs, such as d6144/i2560" as a reopening condition.
3. The **drop picture at top-4**: per-(sender,expert) bucket mean halves 2048 -> 1024, so the
   uniform-routing floor rises ~0.9% -> ~1.25%.

## Check-in 1 — config resolved, param count matches the issue exactly

Built the model config through `build_scale_model()` with
`SCALE_HIDDEN_DIM=6144 SCALE_NUM_LAYERS=48 SCALE_NUM_EXPERTS=256 SCALE_TOP_K=4 SCALE_SEQ_LEN=4096`:

    hidden 6144 · layers 48 · heads 48 · kv 12 · intermediate 3072 · shared_intermediate 6144
    attn 4.53B | shared 5.44B | routed 695.78B | embed 1.58B | router 0.075B | TOTAL 707.4B
    active (excl. embed) 20.9B, active (incl. embed) 22.5B

That reproduces issue #7201's 4-of-256 candidate (707.5B total / 695.8B routed / 22.6B active) with
NO explicit `SCALE_INTERMEDIATE` / `SCALE_SHARED_INTERMEDIATE` / `SCALE_NUM_KV_HEADS`: the May-Recipe
heuristic already gives i = ceil(6144/2/128)*128 = 3072, shared = hidden = 6144, and kv = 48/4 = 12.
So the only launcher deltas vs d1's d5120 control are HIDDEN_DIM 5120->6144, TOP_K 8->4, and dropping
the INTERMEDIATE/SHARED_INTERMEDIATE overrides.

Memory arithmetic at EP64 (`Pfsdp = ("data","expert")`, so non-expert weights shard over the expert
axis when data=1; the embedding table is replicated by design):
  routed/GPU 10.87B + non-expert/GPU 0.18B = 11.05B -> fp32 params 41.2 GiB, +MuonH momentum ~41 GiB,
  bf16 compute copies ~21 GiB, replicated fp32 embed 5.9 GiB.
  Scan residual stack [L,T,H] bf16 = 48 x 65,536 x 6144 x 2 = ~38.7 GiB.
  MoE fixed-capacity buffers are SMALLER than the proxy: capacity = 262,144/256 = 1024, send_size =
  4 x 64 x 1024 = 262,144 rows x 6144 x 2 = 3.2 GiB (vs 5.4 GiB at d5120 top-8).
  Rough steady total ~150 GiB against 186 GiB HBM and a 0.75 default BFC fraction (139 GiB) =>
  OOM is a live risk. Mitigation ladder, in order: `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async` (what the
  hero-run candidate commands use), then `SCALE_OFFLOAD_OPT_STATE=1`, then fall back to 4-of-128.

Note on knobs: `SCALE_CAPACITY_FACTOR` does not exist on this branch lineage — capacity factor is the
module constant `_DEFAULT_EP_CAPACITY_FACTOR = 1.0` in `experiments/grug/moe/model.py`. cf1.0 is what
I need, so no change; a cf sweep at this shape would need that constant lifted to an env knob.

Confidence: 7/10 that the d6144 4-of-256 EP64 baseline is measurable on one rack (memory is the risk,
not the code). Next: drop-floor simulation at top-4, MXFP8 port from d2, EP4 smoke.
