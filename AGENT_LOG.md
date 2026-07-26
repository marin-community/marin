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

## Check-in 2 — deliverable 3 answered ahead of the cluster: top-4 does NOT change the drop regime

Simulated routing at true per-shard scale (T = 65,536 tokens/shard, capacity = the per-(sender,expert)
bucket mean at cf1.0, so the residual is purely statistical + burstiness). Routing model: 16 document
blocks per shard; each block draws a Dirichlet expert preference; a token's distribution is
`(1-burst)*uniform + burst*block_pref`; top-k by exact Gumbel-top-k.

Uniform-routing floor (burst = 0), which is the quantity my brief flagged as rising:

| shape | bucket mean | simulated floor | 0.3989/sqrt(mu) |
|---|---|---|---|
| d5120 proxy, 256 experts, top-8 | 2048 | 0.96% | 0.88% |
| cand A, d6144, 256 experts, top-4 | 1024 | 1.24% | 1.25% |
| cand B, d6144, 128 experts, top-4 | 2048 | 0.84% | 0.88% |

Confirms the arithmetic: the floor rises 0.88 -> 1.25% at top-4 of 256. But the live system does not
sit at the floor. Calibrating `burst` so the PROXY shape reproduces the observed live steady tail
(d4's 350-step tail-100 mean 7.3%) gives burst = 0.764, and at that same burstiness:

    d5120 proxy 256/top8: 7.57%   |   cand A 256/top4: 7.76%   |   cand B 128/top4: 7.35%
    (3 seeds each; seed spread ~0.5pp, larger than the between-shape spread)

PRE-REGISTERED PREDICTION for my live leg: drops at d6144 4-of-256 land in the same 6-8% band as the
proxy, NOT materially worse. Mechanism: in the burstiness-dominated regime the floor is a small
additive term, and doubling k while halving the bucket mean roughly cancels. If the live leg confirms
it, the whole fidelity story — including d1's spill result (drops 7.3% -> 3.7% tail at -0.13pp with
BETTER loss) — transfers to the hero shape unchanged.
Caveat: this is a model, and its absolute calibration is model-dependent (my burst parameterization
needs 0.764 where d1's needed 0.30 for the same 7%); only the BETWEEN-SHAPE comparison at matched
burstiness is being claimed.

## Check-in 3 — MXFP8 ported, jobs in flight

MXFP8 port (commit d1785580c): kernels, `mxfp8_expert_mlp`, and the CUTLASS resolution
(`nvidia-cutlass-dsl[cu13]>=4.5.2,<4.6` + the base-wheel exclusion) taken verbatim from
agent/ep25-d2-bakeoff; `uv sync` reproduces d2's lock with no changes. Wired as a third arm of
`_fixed_a2a_core` (all locals dispatch, one grouped MXFP8 GEMM, then per-expert combine), mutually
exclusive with `SCALE_A2A_BATCH_EXPERTS`. 24 kernel tests pass on CPU including 2 new ones.
Note d2's own verdict names this exact reopening condition: "fatter expert GEMMs, such as d6144/i2560".

Jobs (all mine, `submit_d5.sh`):
- /mwittmann/ep25d5-smoke-bf16-0726-0201 — EP4 4-GPU smoke, d6144/i3072, 64 experts top-4 (bucket mean
  1024, same as the rack target), 4 layers, 40 steps. RUNNING; sentinel line confirms the modified
  module shipped and hparams confirm i3072/kv12/qb_routing.
- /mwittmann/ep25d5-smoke-mxfp8-0726-0202 — same + SCALE_MOE_MXFP8=1. RUNNING.
- /mwittmann/ep25d5-d6144-e256-bf16-120-0726-0205 — THE RACK BASELINE (coordinator approved one leg,
  bf16 alone): 16 replicas / 64 GPUs, EP64, 256 experts top-4, d6144, 48 layers, batch 1024, seq 4096,
  120 steps, QB-on, cf1.0, custom adjoint, drops, `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`.
  OOM ladder if it fails: cuda_async (in) -> BFC fraction 0.85/0.90 -> SCALE_OFFLOAD_OPT_STATE=1 ->
  fall back to 4-of-128. Offload would make the number non-comparable to d1's 22.66% d5120 control
  and will be labeled if used.

## Check-in 4 — spill transfers to top-4 unchanged, and the drop model validates against d1's live spill

Re-ran the routing model through the SHIPPING `_assign_with_spill` kernel (not a re-implementation),
same calibrated burstiness, capacity = bucket mean at cf1.0:

| shape | capacity | m=0 | m=1 | m=2 | m=3 |
|---|---|---|---|---|---|
| d5120 proxy 256/top-8 | 2048 | 7.29% | 4.62% (-37%) | 3.44% (-53%) | 2.73% (-62%) |
| cand A d6144 256/top-4 | 1024 | 7.52% | 4.81% (-36%) | 3.58% (-52%) | 2.88% (-62%) |
| cand B d6144 128/top-4 | 2048 | 7.10% | 4.29% (-40%) | 3.11% (-56%) | 2.55% (-64%) |

Two things fall out.
1. INDEPENDENT VALIDATION of the model: d1's LIVE spill m=2 leg measured a 3.7% tail against a 7.3%
   no-spill tail at the proxy shape. The model, calibrated only on the no-spill number, predicts
   3.44% at m=2. Within 0.3pp on a quantity it was not fitted to. That materially raises my confidence
   in the between-shape comparison above.
2. Spill's reclaim RATE is shape-invariant (-36/-52/-62% at cand A vs -37/-53/-62% at the proxy). So
   the compliant-config story does not change at top-4: m=3 is the setting that reaches the 3% bar,
   landing ~2.9% at cand A and ~2.6% at cand B, exactly as it does on the proxy. Whatever MFU price
   d1 measures for spill (m=2 cost -0.13pp) is the price at the hero shape too.

## Check-in 5 — the comparison my leg is actually against (coordinator's steer: make it, don't disclaim it)

Comparators from #7201's own measured table, on the same hardware, at the same shapes, with the same
2.5 PF/s per-device denominator, all 12-step probes:

| row | shape | parallelism | QB | MFU |
|---|---|---|---|---|
| chunk-4, batch 2304 | d6144 4-of-256 | 2-rack FSDP, 128 GPUs | ON | 18.6% (p50 18.63) |
| chunk-2, batch 1024 | d6144 4-of-128 | 1-rack FSDP, 64 GPUs | OFF | 22.7% (p50 22.71) |
| full-feature + host offload | d6144 4-of-128 | 1-rack FSDP, 64 GPUs | ON | 23.1% |

Precision matters here: only the 18.6% and 23.1% rows carry `SCALE_MOE_QB 1`; the 22.7% chunk-2 row
does not, so the honest QB-on comparators are 18.6% (4-of-256) and 23.1% (4-of-128).

My leg is EP64 on ONE rack at 4-of-256, so the direct comparator is the 18.6% row. Caveats, and the
direction each one pushes:
- Mine is 1-rack EP, theirs is 2-rack FSDP. Their cross-rack penalty is the stated reason that row is
  18.6% while 4-of-128 on one rack is 23.1%. Any advantage I show is partly "one rack beats two".
  #7201's own projection rule is the one to use rather than inventing one: 1-rack -> 12-rack applies
  -7% weak scaling (tok/s x 12 x 0.93), 2-rack -> 12-rack applies -5%.
- I run sliding_window 2048 where the candidate runs sw512 with 5:1 local:global. My configuration
  does MORE attention work, so any MFU advantage I measure is CONSERVATIVE — the candidate's own
  attention config would read higher still.
- I do not run XSA / attn-gate / GatedNorm / host offload; the 18.6% and 23.1% rows do.
- Theirs are 12-step probes; mine is a 120-step p50 with the drop series, which is the stricter number.

## Check-in 6 — RESULT: 707B at EP64 does not fit one rack (measured), and MXFP8 does not flip

### Finding 1 — the memory wall at 4-of-256 / EP64, one rack

/mwittmann/ep25d5-d6144-e256-bf16-120-0726-0205 failed at 09:11:45Z. All 16 tasks reported the same
`gpu_cudamallocasync_allocator` failure, so this is the shape, not a straggler:

    Limit     138.22 GiB   (= 0.75 x 184.3 GiB physical; cuda_async honors XLA_PYTHON_CLIENT_MEM_FRACTION)
    InUse      92.02 GiB
    MaxInUse  117.83 GiB
    failing request: ONE allocation of 114,492,278,784 bytes = 106.63 GiB

92.02 + 106.63 = 198.65 GiB required against 184.3 GiB physical. No memory fraction closes a 14 GiB
gap to the whole device, so the 0.85/0.90 rung is ruled out by arithmetic rather than by a burned rack
leg. A single 106.63 GiB buffer is 58% of HBM in one piece, i.e. compile-time planned, so the remedy
has to be structural (offload / more sharding / fewer per-GPU tokens), not allocator tuning.
Corroborating detail from the same histogram: six live 14,495,514,624-byte buffers, which is exactly
4 local experts x 48 layers x 6144 x 6144 in bf16 — the w13 expert stacks.

Went straight to the offload rung: /mwittmann/ep25d5-d6144-e256-bf16-120-0726-0219-v2 adds
`SCALE_OFFLOAD_OPT_STATE=1` and fraction 0.90. MuonH momentum on 10.87B expert params is ~43.5 GiB of
that 92 GiB resident set, so the requirement should fall to ~155 GiB against a 165.9 GiB limit.
Comparability actually improves rather than degrades: BOTH #7201 d6144 candidate commands already set
`SCALE_OFFLOAD_OPT_STATE=1`, so offload matches the candidate; it is only non-comparable to d1's
d5120 control, which will be stated on the number.

### Finding 2 — MXFP8 at the fatter GEMM: matched EP4 pair, still negative

d2's verdict named "fatter expert GEMMs, such as d6144/i2560" as the reopening condition. Tested at
d6144/i3072 (4 GPUs, EP4, 64 experts top-4, 4 layers, 40 steps, everything else identical):

| arm | p10 | p50 | p90 | tok/s | step | drop@39 | loss@39 |
|---|---|---|---|---|---|---|---|
| bf16 | 8.990 | **9.067** | 9.150 | 55,450 | 1.182s | 0.1480 | 6.7372 |
| MXFP8 | 8.680 | **8.754** | 8.788 | 53,465 | 1.226s | 0.1440 | 6.7418 |

-0.313pp p50 (-3.5% relative); bands do not overlap (bf16 p10 8.990 > MXFP8 p90 8.788). The drop
series are near-identical throughout, so this is a MATCHED-REGIME comparison and the delta is real
throughput, not the drop artifact. Loss differs by 0.005 => MXFP8's numerics are fine at this horizon;
the loss is purely speed. Same SIGN as d2's -2.83pp at d5120/i1280, measured at the shape that was
supposed to reverse it.
Magnitude caveat (not sign): EP4 gives the grouped kernel 16 local experts x 4,096 rows where EP64
would give 4 x 65,536, which is friendlier to it, and a 4-layer model dilutes the MoE share of the
step. So -0.31pp is not the rack number — but the reopening condition has been tested and did not
deliver, which is enough to keep MXFP8 off the rack queue.
Positive by-product: the MXFP8 port runs clean end-to-end on GB200 (40/40 steps, descending loss),
which confirms the 4.5.2/cu13 CUTLASS resolution ports correctly onto this branch.

Confidence: 9/10 on the OOM finding (16/16 tasks, identical arithmetic); 7/10 that MXFP8 stays
negative at EP64 rack scale (sign is measured at the right GEMM shape, scale extrapolation is the gap);
6/10 the offload rung lands the baseline.

## Check-in 7 — the 106.63 GiB buffer identified: NOT a sharding gap, it is the temp arena

The failing job's allocator dump also printed the histogram of live allocations, which sums to exactly
the reported 92.02 GiB InUse over 324 allocations. Decomposing it:

| bytes | count | GiB | identity |
|---|---|---|---|
| 14,495,514,624 | 6 | 81.00 | fp32 `[4, 48, 6144, 3072]` — see below |
| 3,152,019,456 | 3 | 8.81 | fp32 `[128256, 6144]` embedding-shaped (embed, lm_head, one Adam slot) |
| everything else | 315 | ~2.2 | router, norms, scalars |

14,495,514,624 B = 3,623,878,656 fp32 elements = 4 local experts x 48 layers x 6144 x 3072 exactly.
The routed expert weights are `w13 [4,48,6144,6144]` = TWO such units and `w2 [4,48,3072,6144]` = ONE,
so params = 3 units = 40.5 GiB and the MuonH momentum = 3 more = 40.5 GiB. Six units, 81.00 GiB, with
nothing left over. So the resident set is exactly "expert params fp32 + expert momentum fp32 + the
replicated embedding trio", i.e. ALL persistent state and no activations — the job died on the very
first step's temporaries. Nothing is replicated that should be sharded; the per-GPU parameter shard is
what it should be at EP64.

The 114,492,278,784-byte request is therefore not a tensor at all. Two independent legs, after a
correction: my first draft called 1,433,447 prime, which is WRONG — it is 929 x 1543, so the full
factorization is 2^11 x 3 x 13 x 929 x 1543 and the "trailing prime" argument is void. The dimensional
argument that does hold is divisibility, verified with sympy:

    as fp32: 28,623,069,696 elements = 2^9 x 3 x 13 x 929 x 1543
    as bf16: 57,246,139,392 elements = 2^10 x 3 x 13 x 929 x 1543
    NEITHER is divisible by H = 6144 (which needs 2^11 x 3), by T = 65,536, or by V = 128,256.

Every candidate tensor in this model carries a 6144 (hidden, or 2I which is also 6144) or a 128,256
(vocab) dimension, and every activation additionally carries the 65,536 per-shard token count. So no
single such tensor can produce this size. That is the negative leg. The positive leg is the
reconstruction below, which accounts for the request without residue. Together with XLA:GPU allocating
ONE contiguous temp buffer per executable, sized as the sum of the module's live intermediates, the
conclusion is that this is the arena. Reconstructing what has to be live at the peak of one step:

    fp32 expert-weight gradient accumulators (same 3 units as params)   40.5 GiB
    bf16 scan residual stack [48, 65536, 6144]                          36.0 GiB
    per-layer recompute working set (MoE send/recv/hidden + attention)   ~20-30 GiB
                                                                        ~96-106 GiB

which lands on the observed 106.63 GiB. VERDICT: legitimate consequence of the shape, not a bug. The
compressible items, in order: the fp32 gradient accumulators (40.5 GiB), the scan residual stack
(36 GiB, shrinks with per-GPU tokens or residual offload), and the optimizer state (40.5 GiB, but that
is RESIDENT rather than arena — which is exactly why `SCALE_OFFLOAD_OPT_STATE=1` is the right rung:
it takes resident 92.02 -> ~51.5 GiB, so 51.5 + 106.63 = ~158 GiB against the 165.9 GiB limit at
fraction 0.90).

### Method note (for the final report)

Three confident intermediate claims on this thread have now been caught by cross-checking rather than
by measurement: the coordinator's drop-model reframing, d1's toy-scale (mu=32) drop model, and my
"1,433,447 is prime" argument. Each was corrected before it reached a conclusion, and in each case the
conclusion survived on better evidence. Worth one sentence in the write-up: a reader should trust the
surviving numbers more, not less, knowing the failed ones were caught by the same process.

## Check-in 8 — projected memory for the 4-of-128 fallback, from the now-verified decomposition

The same accounting applied to cand B (d6144, 128 experts, top-4, EP64 -> 2 local experts/GPU):

    resident: expert params fp32 20.25 GiB + MuonH momentum 20.25 + embedding trio 8.81  = ~49 GiB
    temp arena: fp32 grad accumulators 20.25 + bf16 residual stack 36.0 (unchanged, it scales
                with tokens not experts) + per-layer working set ~10-15               = ~70 GiB
    total ~119 GiB against the 138.22 GiB limit at the DEFAULT 0.75 fraction.

So the prediction is that 4-of-128 fits one rack at EP64 with no offload and no fraction bump, while
4-of-256 does not fit even at fraction 1.0. Note this mirrors #7201's own choices exactly: their
4-of-128 candidate is a 1-rack command and their 4-of-256 candidate is a 2-rack command. The EP64 path
hits the same wall at the same place as their FSDP path, which is a coherence check on both.
Ready to fire as `./submit_d5.sh rack-e128` if the offload rung fails.

## Check-in 9 — offload rung: NO OOM, but two infra deaths; resubmitted v3

/mwittmann/ep25d5-d6144-e256-bf16-120-0726-0219-v2 (offload + fraction 0.90) reached `worker_failed`
without ever producing a step, and the important negative is that it never OOM'd: ZERO
"failed to allocate" lines across the full 20,346-line warning stream, versus 16/16 tasks reporting one
within 9 minutes on the no-offload leg. So the offload rung did what the accounting predicted; the
deaths were infra.

Two of them, both during the long compile:
- 09:23:02Z — `preemption_notifier.cc:90 SIGTERM caught` on tasks 0/2/3/6/12 simultaneously, i.e. an
  eviction, not a crash. Note `ResourceConfig.with_gpu` sets `preemptible: true`, so these rack workers
  are evictable; the ~30 minute compile at 707B is a long exposure window.
- 09:53:28Z — gang abort. All 15 other tasks logged "another task died"; task 6 is the only one absent
  from that list and its log shows it back at `[iris setup] step 2/3` at 09:54, i.e. it went first with
  no primary error of its own. This is the known transient the brief describes.

Per the standing policy (operational friction never closes a direction) resubmitted unchanged as
/mwittmann/ep25d5-d6144-e256-bf16-120-0726-1000-v3. One hypothesis to watch on this attempt, since
offload is new at this size: it parks ~40.5 GiB of optimizer state per GPU in PINNED HOST memory, so
4 GPUs/node x 40.5 = ~162 GiB of pinned RAM against the launcher's `ram="256g"` request. If v3 dies
the same way, a host-memory OOM is the first thing to check, and the answer is either a larger RAM
request or the 4-of-128 fallback.

## Check-in 10 — operational finding worth lifting out of this direction: rack workers are preemptible

`ResourceConfig.with_gpu` in `experiments/grug/moe/launch_cw_scale.py` sets `preemptible: true` (visible
in every run's hparams dump), so rack GPU workers on this cluster are evictable. That gives every job an
eviction window proportional to its COMPILE time, and a 707B model's ~30-minute compile is close to the
worst case anyone here runs. My v2 leg produced unambiguous evidence rather than an inference:

    09:23:02.720759 - 09:23:02.721244Z, tasks 3, 12, 2, 6, 0: `preemption_notifier.cc:90 SIGTERM caught`

Five tasks receiving SIGTERM within 500 microseconds is an eviction, not a crash. Worth noting for the
wider session: an evicted task that comes back carries a NEW incarnation, and a peer still holding the
old one reports exactly the "unexpectedly tried to connect with a different incarnation" gang-abort that
has been hitting several agents and rav all session. That does not prove the two are the same
phenomenon, but it is a concrete mechanism that produces the observed symptom, and it is testable by
grepping any suspect job's logs for `preemption_notifier` before blaming the code under test.

Deliberately NOT attempted, per the known GB200 failure: enabling the JAX persistent compilation cache
to shorten the exposure window. A leader process starting with a populated cache is necessary and
sufficient to deadlock NCCL clique init on this hardware, and it presents as a HANG at init — which is
indistinguishable at a glance from the gang aborts above and would have cost hours to separate.
Confirmed clean: no submission of mine sets a cache directory, and every run's hparams shows
`jax_compilation_cache_dir: null`.

Added `SCALE_RAM` (default 256g, unchanged) so the host-memory request can be raised without a code
edit at submit time; verified it flows through to the ResourceConfig. Host offload parks ~40.5 GiB per
GPU in pinned host memory = ~162 GiB per 4-GPU node against the 256g default.

## Check-in 11 — v3 crashlooped silently; host-memory hypothesis survives; v4 raises RAM to 600g

/mwittmann/ep25d5-d6144-e256-bf16-120-0726-1000-v3 reached failures=9 with restart cycles as short as
2m45s. Classification across its 40,004-line warning stream and 52,238-line full stream:

    "failed to allocate" (HBM OOM)        0
    "SIGTERM caught" (preemption)         0
    "another task died" (gang victim)     49
    primary exception / traceback         NONE

So one process disappears and the rest report it. A process that dies with no exception, no HBM OOM
and no SIGTERM is the signature of a kernel SIGKILL, which is what a container host-memory OOM looks
like from inside. That is exactly the hypothesis I pre-registered when v2's offload rung first ran, and
it survives this look, so per the pre-authorized ladder the RAM request goes first.

The arithmetic supports it: `SCALE_PROCESSES_PER_TASK` defaults to 1, so ONE container per node holds
all 4 GPUs' offloaded optimizer state in pinned host memory = 4 x 40.5 = ~162 GiB, plus the JAX host
allocations, the loader prefetch (32 buffers) and the process itself, against the launcher's hardcoded
`ram="256g"`. GB200 nodes have 960 GB (`lib/iris/config/cw-us-east-08a.yaml`: gb200-4x, 144 vCPU,
960GB LPDDR5X), so the 256g request was leaving ~800 GB of the node unused while the container hit its
own limit.

/mwittmann/ep25d5-d6144-e256-bf16-120-0726-1023-v4 = v3 + `SCALE_RAM=600g` (the knob added this
session). Note the two are otherwise identical, so if v4 clears init this is a clean attribution.
Stopped v3 before resubmitting; it was crashlooping the rack to no purpose.

## TRIAGE RECIPE — three failure modes that all look like "another task died"

Lift this out of the log; it is not specific to my direction. On this cluster a rack job dying with
every task reporting `Terminating process because the JAX distributed service detected fatal errors ...
another task died` has at least three distinct causes with completely different fixes. All three have
occurred THIS session, and this thread and peers have repeatedly mistaken them for bugs in the code
under test. Classification takes two minutes.

Fetch the log once at warning level (the default 1000-line window is too small and will hide the
primary cause; `--level warning` keeps it manageable):

    iris --cluster=marin job logs <JOB> --level warning --max-lines 40000 > /tmp/j.log

Then run all three greps. The combination of what IS and ISN'T present is the classifier:

| grep | class | reading |
|---|---|---|
| `grep -c "failed to allocate" /tmp/j.log` | 3. device HBM OOM | any hit => HBM. The adjacent `Stats: Limit / InUse / MaxInUse` lines and the `allocator.cc:71` histogram give the whole picture; the histogram sums to InUse and its entries can be matched to tensor shapes. |
| `grep -c "SIGTERM caught" /tmp/j.log` | 1. preemption | hits, especially several tasks within the same instant, => eviction, not a crash. `ResourceConfig.with_gpu` sets `preemptible: true`, so exposure scales with COMPILE time. |
| all three of the above zero, plus no traceback | 2. container host-memory OOM | a process died by kernel SIGKILL, which leaves no log. Suspect immediately when host offload or other large pinned host allocations are in play. |

Confirm class 2 positively by finding the task that is NOT in the victim list: every other task logs
"another task died", so the one absent from that set went first. Its own log will show it already back
at `[iris setup]` with no error of its own.

    grep -oE "grug-train-[^ ]*/[0-9]+ \|" /tmp/j.log | grep -f <(echo "another task died") ...
    # simpler in practice: list the task indices that logged the victim message and diff against 0..N-1

Fixes by class:
1. preemption — resubmit with a `-vN` suffix; nothing is wrong with the code. Shorten the compile if you
   can, but NOT via the JAX persistent compilation cache: a leader process starting with a populated
   cache deadlocks NCCL clique init on GB200 and presents as a HANG, which is a fourth look-alike.
2. host-memory OOM — raise the per-node request. `SCALE_RAM` (added this session) does it without a
   code edit; see the launcher limitation below.
3. HBM OOM — the allocator report tells you the resident set and the failing request; decide between
   offload, remat, fewer per-GPU tokens, or more sharding from those numbers.

### Launcher limitation this exposed (ops item, one-line-fix shaped)

`experiments/grug/moe/launch_cw_scale.py` hardcoded `ram="256g"` in `ResourceConfig.with_gpu`, while a
`gb200-4x` node has 960 GB of LPDDR5X (`lib/iris/config/cw-us-east-08a.yaml`: 144 vCPU, 960GB). With
`SCALE_PROCESSES_PER_TASK` defaulting to 1, ONE container per node holds all 4 GPUs' host-offloaded
optimizer state — ~162 GiB at d6144 4-of-256 — so roughly 800 GB per node sits idle while the container
hits a cap it did not choose. Any workload combining `SCALE_OFFLOAD_OPT_STATE=1` with this model scale
will hit it. Now overridable via `SCALE_RAM` (default unchanged at 256g).
