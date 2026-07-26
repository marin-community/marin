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
| `grep -cE "failed to allocate\|RESOURCE_EXHAUSTED.*Out of memory" /tmp/j.log` | 3. device HBM OOM (cuda_async logs the first string, BFC the second) | any hit => HBM. The adjacent `Stats: Limit / InUse / MaxInUse` lines and the `allocator.cc:71` histogram give the whole picture; the histogram sums to InUse and its entries can be matched to tensor shapes. |
| `grep -c "SIGTERM caught" /tmp/j.log` | 1. preemption | hits, especially several tasks within the same instant, => eviction, not a crash. `ResourceConfig.with_gpu` sets `preemptible: true`, so exposure scales with COMPILE time. |
| all three of the above zero, plus no traceback | 2. container host-memory OOM (SUSPECT, then TEST) | a process died by kernel SIGKILL, which leaves no log. Suspect when host offload or other large pinned host allocations are in play — but confirm by raising the request and re-running, because in my own case raising 256g -> 600g changed nothing, so this branch also covers residual transient gang aborts. |

Confirm class 2 positively by finding the task that is NOT in the victim list: every other task logs
"another task died", so the one absent from that set went first. Its own log will show it already back
at `[iris setup]` with no error of its own.

    grep -oE "grug-train-[^ ]*/[0-9]+ \|" /tmp/j.log | grep -f <(echo "another task died") ...
    # simpler in practice: list the task indices that logged the victim message and diff against 0..N-1

| `grep -cE "ncclAlltoAll\|unhandled cuda error\|Cuda failure 2" /tmp/j.log` | 5. NCCL memory starvation | **THE INVERSION — read this before reaching for headroom.** NCCL allocates transport buffers OUTSIDE the XLA arena, so a class-3 XLA OOM says RAISE `XLA_PYTHON_CLIENT_MEM_FRACTION` while a class-5 NCCL OOM says LOWER it. The class-3 grep (`failed to allocate`) returns ZERO for class 5, so the natural diagnostic silently points the wrong way: anyone who OOMs at EP scale, greps for "failed to allocate", finds nothing, and reaches for more headroom will reproduce this failure exactly. Concretely at 184.3 GiB HBM: fraction 0.90 leaves ~18 GiB for NCCL + cuBLAS workspaces + CUDA context combined; 0.75 leaves ~46 GiB. At EP64 the all-to-all does not fit in 18. |

Fixes by class:
1. preemption — resubmit with a `-vN` suffix; nothing is wrong with the code. Shorten the compile if you
   can, but NOT via the JAX persistent compilation cache: a leader process starting with a populated
   cache deadlocks NCCL clique init on GB200 and presents as a HANG, which is a fourth look-alike.
2. host-memory OOM — raise the per-node request. `SCALE_RAM` (added this session) does it without a
   code edit; see the launcher limitation below.
3. HBM OOM — the allocator report tells you the resident set and the failing request; decide between
   offload, remat, fewer per-GPU tokens, or more sharding from those numbers. Raising the fraction is
   the tempting fix and at EP scale it can convert this into class 5; prefer the structural options.
5. NCCL OOM — LOWER the fraction back toward the 0.75 default. Do not confuse with class 3.

### Launcher limitation this exposed (ops item, one-line-fix shaped)

`experiments/grug/moe/launch_cw_scale.py` hardcoded `ram="256g"` in `ResourceConfig.with_gpu`, while a
`gb200-4x` node has 960 GB of LPDDR5X (`lib/iris/config/cw-us-east-08a.yaml`: 144 vCPU, 960GB). With
`SCALE_PROCESSES_PER_TASK` defaulting to 1, ONE container per node holds all 4 GPUs' host-offloaded
optimizer state — ~162 GiB at d6144 4-of-256 — so roughly 800 GB per node sits idle while the container
hits a cap it did not choose. Any workload combining `SCALE_OFFLOAD_OPT_STATE=1` with this model scale
will hit it. Now overridable via `SCALE_RAM` (default unchanged at 256g).

## Check-in 12 — CORRECTION to every modelled spill number, and a sharper finding falls out

d1 recovered true tail-100 statistics (the iris log default truncates at 1000 lines, which had been
biasing every "steady state" number low) and its live spill measurements moved: m=2 from 3.7% to
4.14%, m=3 to 3.66%, with the no-spill baseline still 7.30%. So my "within 0.3pp out-of-sample
validation" was measured against a biased number and must be withdrawn in that form: against the true
4.14%, my model's 3.44% is 0.70pp optimistic, i.e. ~17% relative. The model over-predicts how much
spill reclaims, which makes mechanical sense — an idealized model overestimates how many free buckets
later attempts find, because a token's alternative experts are correlated with its first choice.

CORRECTION FACTORS, derived self-consistently against MY model on the proxy shape (this differs from
the 1.36 / 1.54 quoted to me, which were derived against d1's own more-optimistic model; correcting my
numbers requires my model's error, and I note the discrepancy rather than adopt a factor whose baseline
is not mine — under the larger factors every conclusion below only gets stronger):

    m=2: measured 4.14% / my model 3.44% = x1.204
    m=3: measured 3.66% / my model 2.73% = x1.338
    the factor grows with m by about +0.134 per attempt, consistent with the correlation mechanism.

CORRECTED projection against the 3% bar at cf1.0:

| shape | m_max | model m=2 | corrected | model m=3 | corrected | verdict |
|---|---|---|---|---|---|---|
| d5120 proxy 256/top-8 | 7 | 3.44% | 4.14% | 2.73% | **3.66%** | fails at m=3, but has budget left |
| cand A d6144 256/top-4 | 3 | 3.58% | 4.31% | 2.88% | **3.86%** | fails, and SATURATED |
| cand B d6144 128/top-4 | 3 | 3.11% | 3.74% | 2.55% | **3.42%** | fails, and SATURATED |

THE SHARPER FINDING, and it is now MEASURED rather than asserted: top-k IS the spill recovery budget.
Running the shipping kernel out to m=7 shows the top-4 shapes stop improving at m=3 exactly —

    cand A: m3 2.88% | m4 2.88% | m5 2.88% | m6 2.88% | m7 2.88%   (saturated)
    proxy:  m3 2.73% | m4 2.28% | m5 1.97% | m6 1.75% | m7 1.57%   (still improving)

because an assignment can only be re-offered to experts the token itself selected, so top-4 leaves at
most 3 alternatives. CONSEQUENCE FOR THE RUN DECISION: neither top-4 candidate can reach a 3% drop bar
by spill alone at cf1.0, and unlike the top-8 proxy they have no further attempts to spend. They would
need capacity headroom on top, at the measured price of about -0.58pp MFU per +0.05 cf. The top-8
proxy retains headroom the hero candidates do not have — so the proxy was FLATTERING on fidelity, and
every compliance claim carried over from it needs this correction.

What does NOT change: the between-shape comparison, because the correction is multiplicative at fixed
m and applies to all shapes equally. The ordering (cand B slightly better than proxy, cand A slightly
worse) and the ~0.4pp spread survive. Also unchanged: the no-spill prediction, since m=0 needs no
correction — the hero shape should still land in the same 6-8% band as the proxy, which my live leg
tests directly.

## Check-in 13 — the corrected compliant-config recommendation, per candidate

Swept capacity factor x spill attempts through the shipping kernel at all three shapes, then applied
the m-dependent correction factors from check-in 12. Corrected drop %, `*` clears the 3% bar:

| shape | cf | m=0 | m=1 | m=2 | m=3 |
|---|---|---|---|---|---|
| proxy 256/top-8 | 1.00 | 7.29 | 4.95 | 4.14 | 3.66 |
| | 1.05 | 5.22 | 2.82* | 1.93* | 1.37* |
| | 1.10 | 3.67 | 1.55* | 0.81* | 0.42* |
| | 1.15 | 2.46* | 0.79* | 0.30* | 0.12* |
| cand A 256/top-4 | 1.00 | 7.52 | 5.15 | 4.31 | 3.85 |
| | 1.05 | 5.42 | 2.99* | 2.03* | 1.47* |
| | 1.10 | 3.81 | 1.69* | 0.91* | 0.50* |
| | 1.15 | 2.62* | 0.85* | 0.34* | 0.14* |
| cand B 128/top-4 | 1.00 | 7.10 | 4.59 | 3.74 | 3.42 |
| | 1.05 | 5.07 | 2.54* | 1.60* | 1.01* |
| | 1.10 | 3.57 | 1.34* | 0.60* | 0.26* |
| | 1.15 | 2.33* | 0.72* | 0.25* | 0.07* |

Read with the measured MFU prices (both from the proxy): capacity costs about -0.58pp per +0.05 cf,
spill m=2 costs -0.13pp, and the old strict config cf1.15 + m=0 cost -1.75pp.

RECOMMENDED COMPLIANT CONFIG, corrected, per candidate:
- cand A (d6144 4-of-256): **cf1.05 + spill m=2** -> ~2.0% drops for about -0.7pp total. cf1.05 + m=1
  also technically clears at 2.99%, but with no margin at all on a modelled number, so m=2 is the
  recommendation. cf1.0 is not reachable at any m, because m saturates at 3.
- cand B (d6144 4-of-128): **cf1.05 + spill m=1** -> ~2.5% for about -0.65pp, or m=2 -> 1.6% for
  roughly the same price if margin is wanted.
- Either is about 1pp cheaper than the cf1.15 + m=0 config the effort has been quoting (-1.75pp).

Cross-agent consequence to carry into the report: d1's combination ranking (m=2 at cf1.05 winning on
expected compliant MFU) was derived at the PROXY shape, where the spill axis still has room. At a
top-4 shape the ranking must shift toward capacity, because the spill axis is pinned at m=3 — the
saturation table is what forces that. The table above is the top-4 version of that ranking. If d1's
cf1.15 leg validates the model's capacity response, this becomes a measured recommendation rather than
a modelled one.

Caveat, and it runs in the conservative direction: the correction factors were derived at cf1.0. With
capacity headroom, later spill attempts should find free buckets MORE easily than the idealized model
assumes is hard, so the model's optimism ought to shrink as cf rises — meaning the starred cells are
probably pessimistic rather than optimistic.

## Check-in 14 — v4 (RAM 600g) also failed; falling back to 4-of-128 per pre-authorization

Triaged v4 with my own recipe (20,894-line warning stream):

    failed to allocate  0      SIGTERM caught  0      another task died  58      Traceback  3

The three tracebacks are `step_runner._harvest` re-raising the child failure plus faulthandler thread
dumps from victims — no primary exception. So the RAM rung did NOT fix it: raising the request from
256g to 600g changed nothing observable, which means either the host-memory hypothesis is wrong or
600g was not the binding constraint. I am recording that as a FAILED prediction of mine rather than
quietly dropping it; the arithmetic that motivated it (4 x 40.5 GiB pinned in one container) still
looks right, but it did not produce the fix it predicted, so the class-2 diagnosis of v3 is now
unconfirmed rather than established. The triage recipe's class-2 branch should be read as "suspect and
test", not "diagnose" — I have weakened the wording accordingly.

Tally at the 4-of-256 shape: 4 attempts, 0 steps. One measured HBM OOM (16/16 tasks, the real finding),
one preemption, two gang aborts with no primary cause. Per the standing pre-authorization I am not
spending a fifth attempt there.

FALLBACK FIRED: /mwittmann/ep25d5-d6144-e128-bf16-120-0726-1100 — d6144, 128 experts, top-4, 48
layers, EP64, batch 1024, seq 4096, 120 steps, QB-on, cf1.0, custom adjoint, drops. Deliberately NO
host offload and the DEFAULT BFC allocator at fraction 0.90: my projection says ~119 GiB against the
138.22 GiB limit at the default 0.75, so the headroom is there without offload, and dropping offload
removes the host-memory variable entirely AND keeps the step's data movement identical to d1's d5120
control. The only deviation from that control is the fraction (0.90 vs 0.75), which a peer measured as
performance-neutral on this workload. So this number will be comparable BOTH to d1's 22.66% control
and to #7201's 4-of-128 rows.

---

# STANDALONE RESULTS — ep25-d5, written to survive without the narrative above

Question: does the EP25 stack (custom scatter-add adjoint, gather dispatch, QB balancing, spill)
transfer from the d5120 8-of-256 proxy to the REAL hero-run candidates, d6144 4-of-256 (707B) and
d6144 4-of-128 (360B) from issue #7201?

## R1. The proxy was FLATTERING on fidelity: top-k is the spill recovery budget

MEASURED through the shipping `_assign_with_spill` kernel, identical settings, m = spill attempts:

    cand A d6144 256/top-4:  m3 2.88 | m4 2.88 | m5 2.88 | m6 2.88 | m7 2.88   <- stops dead at m=3
    d5120 proxy 256/top-8:   m3 2.73 | m4 2.28 | m5 1.97 | m6 1.75 | m7 1.57   <- keeps improving

A flat line beside a declining one at identical settings is the whole argument, and it needs no
modelling assumptions: an assignment can only be re-offered to experts the token ITSELF selected, so
top-4 leaves at most 3 alternatives while top-8 leaves 7. Every compliance claim in this effort was
measured at the top-8 proxy, which has more than twice the recovery budget of either hero candidate.

## R2. Corrected drop projections, and neither top-4 candidate is compliant by spill alone

Model calibrated on the proxy's live no-spill tail, then corrected by the factors implied by d1's live
truncation-corrected measurements (x1.204 at m=2, x1.338 at m=3; the factor grows with m because an
idealized model overestimates how many free buckets later attempts find):

| shape | m_max | corrected m=2 | corrected m=3 | 3% bar |
|---|---|---|---|---|
| d5120 proxy 256/top-8 | 7 | 4.14% | 3.66% | fails at m=3, has budget left |
| cand A d6144 256/top-4 | 3 | 4.31% | 3.86% | fails, SATURATED |
| cand B d6144 128/top-4 | 3 | 3.74% | 3.42% | fails, SATURATED |

The no-spill (m=0) prediction needs no correction and is unchanged: at matched burstiness the three
shapes land within 0.4pp of each other (7.57 / 7.76 / 7.35), so the uniform-routing floor rising from
0.88% to 1.25% at top-4 does NOT change the regime. The worry that motivated this direction is
answered: drops at the hero shape are not materially worse than at the proxy. What IS worse is the
recovery budget.

## R3. The corrected compliant-config recommendation, per candidate

Capacity costs about -0.58pp MFU per +0.05 cf; spill m=2 costs -0.13pp; the cf1.15 + m=0 config this
effort has been quoting costs -1.75pp.

- cand A (4-of-256): cf1.05 + spill m=2 -> ~2.0% drops, about -0.7pp. cf1.0 is unreachable at any m.
- cand B (4-of-128): cf1.05 + spill m=1 -> ~2.5%, about -0.65pp; m=2 -> ~1.6% for roughly the same.
- Both are about 1pp cheaper than cf1.15 + m=0.
d1's ranking (m=2 at cf1.05) was derived at the proxy, where the spill axis still has room; at a top-4
shape the ranking must shift toward capacity because spill is pinned at m=3.

## R4. 707B / 4-of-256 at EP64 does not fit one rack

MEASURED, 16/16 tasks, identical report: allocator limit 138.22 GiB (0.75 x 184.3 physical), 92.02 GiB
resident, and a single failing 106.63 GiB request. 92.02 + 106.63 = 198.65 GiB against 184.3 GiB of
physical HBM, so no memory fraction closes it. The resident set decomposes exactly — six units of
4 x 48 x 6144 x 3072 fp32 (= expert params 3 units + MuonH momentum 3 units = 81.00 GiB) plus the
embedding trio (8.81 GiB) — confirming nothing is replicated that should be sharded. The 106.63 GiB
request is XLA:GPU's single temp arena, not a tensor: as fp32 or bf16 its element count is not
divisible by H = 6144, T = 65,536, or V = 128,256, and the reconstruction from known intermediates
(fp32 grad accumulators 40.5 + bf16 residual stack 36.0 + per-layer working set ~20-30) lands on the
observed value. This is consistent with #7201's own 4-of-256 candidate command, which already assumes
2 racks AND host offload.

NOT AFFECTED BY THE LATER MEMORY-FRACTION CONFUSION. This result was measured on the FIRST leg at the
DEFAULT 0.75 fraction, with an explicit XLA allocator report, and 16/16 tasks agreeing. The
fraction-0.90 mistake (which starved NCCL, see the triage recipe class 5) came later and affected only
the subsequent attempts. A reader who sees that confusion should not discount this number.

## R5. MXFP8 does not flip at the fatter expert GEMM

d2 measured -2.83pp at d5120/i1280 and named "fatter expert GEMMs, such as d6144/i2560" as the
reopening condition. Matched EP4 pair at d6144/i3072, identical drop regime (0.1480 vs 0.1440 at step
39), 40 steps: bf16 p50 9.067% (p10 8.990 / p90 9.150) vs MXFP8 p50 8.754% (p10 8.680 / p90 8.788) =
**-0.313pp**, bands non-overlapping, loss parity 0.005. Same sign as d2's result, at the shape that was
supposed to reverse it. Magnitude caveat: EP4 gives the grouped kernel 16 local experts x 4,096 rows
where EP64 would give 4 x 65,536, and a 4-layer model dilutes the MoE share. The port itself works
(40/40 steps clean on GB200), so the 4.5.2/cu13 CUTLASS resolution transfers.

## R6. Operational findings (lift these out of the direction)

- Rack GPU workers are `preemptible: true` via `ResourceConfig.with_gpu`, so every job carries an
  eviction window proportional to COMPILE time; a 707B compile is ~30 minutes. Evidence: five tasks
  taking `preemption_notifier SIGTERM caught` within 500 microseconds. An evicted task returning with
  a new incarnation is exactly how the "different incarnation" gang-abort surfaces.
- `ram` was hardcoded at 256g while a gb200-4x node has 960 GB; with one container per node holding
  all 4 GPUs' offloaded optimizer state (~162 GiB), roughly 800 GB per node sat idle. Now overridable
  via `SCALE_RAM` (default unchanged).
- The three-way triage recipe for look-alike gang aborts is above; class 2 is "suspect, then test",
  because raising 256g -> 600g did not fix my case.

## R8. Expert-count reduction is NOT the memory lever it looks like — per-GPU tokens is

Two MEASURED temp-arena sizes at EP64, same batch, same layers, same everything but expert count:

    d6144 4-of-256 (4 local experts/GPU):  106.63 GiB
    d6144 4-of-128 (2 local experts/GPU):   90.64 GiB     -> halving experts cut the arena 15%

The arena is dominated by the token-scaled bf16 residual stack — 36.0 GiB, IDENTICAL at both shapes,
because it is [L, tokens_per_shard, H] and neither L, tokens nor H changed. The expert-scaled fp32
gradient accumulators DO halve (40.5 -> 20.25 GiB) and that accounts for essentially the whole 16 GiB
delta. Nothing else moved.

So anyone who sees 4-of-256 fail to fit and reaches for 4-of-128 as the memory fix is reaching for a
knob that moves 15% of the problem. The levers that actually move it, in order:
1. per-GPU tokens — scales the residual stack (36.0 GiB) and most of the per-layer working set;
   halving the batch would take roughly 18 GiB out of the arena at either shape.
2. residual offload / more aggressive remat — attacks the same 36.0 GiB.
3. optimizer-state host offload — ~40.5 GiB at 4-of-256, ~20.25 at 4-of-128, but note this comes off
   the RESIDENT set, not the arena, so it helps the total without touching the peak intermediate.
4. expert count — the intuitive one, and the weakest.

Methodological note: I projected ~70 GiB and measured 90.64, a 20 GiB miss. The pair is more
informative than a correct projection would have been — a projection that had matched would have told
us nothing about WHY, whereas the miss forced the decomposition that produced this finding.

## R7. RESOLVED — EP64 beats FSDP at the one-rack hero candidate shape

/mwittmann/ep25d5-d6144-e128-bf16-120-0726-1140-v3 completed 120/120 steps, 16/16 tasks.
STEADY TAIL (steps 90-119): p50 **24.594%**, p10 24.434 / p90 24.771, sd 0.118, 276,413 tok/s,
15.174 s/step, drops ~9-13% (120-step run, so step 119 is end-of-anneal), loss 5.59.

STATE THE HEADLINE AS A PAIR. The raw number must never travel alone:

    throughput frontier   24.59%   at ~9-13% drops   vs #7201's 23.1% FSDP row = +1.5pp, +8.3% tok/s
    compliant projection  ~23.9%   at <3% drops      (cf1.05 + spill m=1..2, about -0.65pp) -- still
                                                     above 23.1%, and this is the ROBUST version

Because: **#7201's 4-of-128 rows do not report drop fractions.** Their runs compute the per-layer count
but never emit it — the tracker-logging bug d1 found and fixed in 2d4a87395. My 24.59% is measured at
~9-13% drops, and this session established that heavy-drop runs read HIGHER MFU (dropped assignments
gather a zero pad row and do less real work at the same step accounting). So the EP-versus-FSDP
comparison is CROSS-DROP-REGIME IN AN UNKNOWN DIRECTION: if their drops are materially lower than mine,
some portion of the +1.5pp is drop artifact rather than parallelization advantage. This is the single
largest threat to the headline — larger than any configuration caveat, all of which have a known sign —
and a reader cannot derive it from the record, so it must be stated next to the number.
It does not overturn the result: the compliant projection prices the fidelity and still clears 23.1%.
RECOMMENDATION: anyone rerunning those FSDP rows should apply the drop-metric fix (2d4a87395) FIRST.
One job's cost resolves this ambiguity permanently, and the ambiguity currently sits on the most
decision-relevant comparison in the effort.

On a like-for-like measurement window (mine at steps 10-24, the regime a 12-step probe actually
samples) it reads 27.04%.
All surviving caveats point the same way — sw2048 vs the candidate's sw512 + 5:1, and no
XSA/attn-gate/GatedNorm where the 23.1% row has them — so the advantage is CONSERVATIVE. Offload is on
in both; default allocator and default fraction, so no memory-knob caveat. Compliant configuration
(cf1.05 + spill m=1..2, about -0.65pp) lands near ~23.9%, still above the FSDP row.
SCOPE, stated plainly: 4-of-128 is a #7201 top candidate and is the ONE-RACK answer, but it is NOT the
4-of-256 or 8-of-256 shape the original goal named. 4-of-256 has no MFU number and cannot get one on a
single rack at EP64 (R4); 8-of-256 measures 22.66% at the d5120 proxy. The honest framing is "the
one-rack candidate is both faster and EP-favorable", not "the goal is met at 256 experts".

Full detail in the RESULT section at the end of this log.

### Still open

The 4-of-256 shape has no MFU number and will not get one on a single rack (R4). Spill is unmeasured
live at any d6144 shape — R1-R3 are kernel-exact but not a training leg. The original framing for this
section, retained because it was written before the result and states the falsification cleanly:

The 4-of-128 EP64 leg is the direct EP-versus-FSDP test at the shape most likely to be chosen. #7201's comparators are 22.7% (QB-off chunk-2) and 23.1%
(QB-on full-feature), both 1-rack FSDP 12-step probes. If the EP64 number beats 23.1%, that is the
first evidence EP plus the custom adjoint wins at the candidate shape and is the headline of this
effort. If it lands below, that is equally important and must be reported just as prominently: it
would mean FSDP is the better parallelization at the one-rack hero candidate and that the EP line —
adjoint, spill, and all — is optimizing a path the run should not take at that shape.
Caveats to attach either way: my sw2048 versus the candidate's sw512 + 5:1 local:global means my
configuration does MORE attention work, so an EP advantage is conservative and an EP deficit is
overstated; I run no XSA / attn-gate / GatedNorm / host offload where the 23.1% row does; mine is a
120-step p50 with a drop series where theirs is a 12-step probe; and I run BFC at fraction 0.90 rather
than the default 0.75, which a peer measured as performance-neutral on this workload.
Also unmeasured: spill live at any d6144 shape — everything in R1-R3 is kernel-exact but not a live
training leg.

## Check-in 15 — a FIFTH failure class, and this one was my own configuration error

The 4-of-128 leg compiled, reached execution, and then died with a REAL primary error — the first one
in five rack attempts:

    jax.errors.JaxRuntimeError: INTERNAL: NCCL operation ncclAlltoAll(...) failed: unhandled cuda
    error ... Last NCCL warning(error) log entry 'Cuda failure 2 'out of memory''
    [executable_name='jit_train_step']

reported by many tasks at 11:08:08Z, i.e. inside the a2a. NCCL allocates its transport buffers OUTSIDE
the XLA allocator. I had set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90` "for headroom", which with the BFC
allocator pre-reserves 90% of HBM at startup and leaves ~18 GiB for NCCL, cuBLAS workspaces and the
CUDA context combined. At EP64 the all-to-all needs more than that, so I starved it. The default 0.75
leaves ~46 GiB.

This is a fifth class for the triage recipe and it is worth as much as the other four, because the
signature actively misleads: an XLA-allocator grep (`failed to allocate`) returns ZERO — the memory
failure is inside NCCL, and it surfaces as an INTERNAL NCCL error naming the collective.

    grep -c "ncclAlltoAll\|unhandled cuda error\|Cuda failure 2" j.log
    non-zero => memory starvation OUTSIDE the XLA allocator. LOWER XLA_PYTHON_CLIENT_MEM_FRACTION,
    do not raise it. Raising the fraction to fix an XLA OOM can CAUSE this one.

Fixed the submitter so `MEM_FRACTION` is empty by default (leave 0.75 alone) rather than 0.90, with the
mechanism documented in the script. Resubmitted as
/mwittmann/ep25d5-d6144-e128-bf16-120-0726-1125-v2 at the DEFAULT fraction and the DEFAULT BFC
allocator, no offload — which is now byte-for-byte d1's d5120 control configuration with only the
model shape changed, the cleanest possible comparison. My projection (~119 GiB against the 138.22 GiB
limit at 0.75) says it fits without the headroom I was trying to buy.

Retrospective worth noting: v2/v3/v4 at 4-of-256 ALSO ran at fraction 0.90. They died with no primary
error, which is not this signature, but I can no longer rule out that the 0.90 fraction contributed —
a process killed inside a CUDA/NCCL allocation failure can die before it logs. That is a HYPOTHESIS,
not a finding, and it does not touch the 4-of-256 memory result, which was measured at the DEFAULT
0.75 fraction on the very first leg with an explicit XLA allocator report.

## Check-in 16 — 4-of-128 MEASURED arena: 90.64 GiB. My ~70 GiB projection was too low.

The default-fraction, default-allocator, no-offload 4-of-128 leg gave the cleanest possible failure —
a textbook class 3, with all four other classes reading zero:

    failed to allocate 0 | SIGTERM caught 0 | another task died 0 | ncclAlltoAll 0 | Cuda failure 0
    jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 90.64GiB
    [executable_name='jit_train_step']

(The class-3 grep string differs between allocators: cuda_async logs `failed to allocate`, BFC logs
`RESOURCE_EXHAUSTED ... Out of memory while trying to allocate`. Both are class 3. Recipe updated.)

So the temp arena at 4-of-128 is 90.64 GiB, against 106.63 GiB at 4-of-256. I had projected ~70 GiB
and was wrong by 20 GiB — a FAILED PREDICTION, recorded as such. What the two measurements together
show is more useful than my projection was: halving the expert count cut the arena by only 15%,
because the arena is dominated by the token-scaled bf16 residual stack (36.0 GiB, identical at both
shapes) rather than by anything expert-scaled. The fp32 gradient accumulators do halve (40.5 -> 20.25
GiB), which accounts for essentially the whole 16 GiB difference.

Consequence: 4-of-128 at EP64 misses fitting at the default fraction by only ~2 GiB (resident ~49 +
arena 90.64 = ~140 GiB against the 138.22 GiB limit). Two ways to close 2 GiB:
- raise the fraction slightly — REJECTED. That is the exact knob that starved NCCL at 0.90, and I am
  not going to re-enter that failure mode to buy 2 GiB.
- host offload — CHOSEN. At 4-of-128 it frees ~20 GiB of resident (49 -> ~29), giving ~120 GiB against
  138.22 with real margin, and it needs only ~81 GiB of pinned host memory per node (vs ~162 at
  4-of-256), comfortably inside the 256g default. Better still, it makes the leg MORE comparable to
  the number that matters: #7201's 23.1% full-feature 4-of-128 row also sets
  `SCALE_OFFLOAD_OPT_STATE=1`. The cost is comparability to d1's d5120 control, which does not.

Fired: /mwittmann/ep25d5-d6144-e128-bf16-120-0726-1140-v3 = 4-of-128, EP64, offload on, default BFC
allocator at the default 0.75 fraction, no other change.

## RESULT — the 4-of-128 EP64 leg COMPLETED: 24.59% steady p50, and EP beats FSDP at the candidate shape

/mwittmann/ep25d5-d6144-e128-bf16-120-0726-1140-v3 — SUCCEEDED, 16/16 tasks, all 120 steps.
d6144, 128 experts, top-4, 48 layers, EP64 on one GB200 rack, batch 1024, seq 4096, QB-on, cf1.0,
custom adjoint + gather dispatch, drops reported, host offload on, default BFC allocator at the
default 0.75 fraction, sliding window 2048.

| window | p10 | p50 | p90 | sd | tok/s | step |
|---|---|---|---|---|---|---|
| steady tail, steps 90-119 | 24.434 | **24.594** | 24.771 | 0.118 | 276,413 | 15.174s |
| whole run past warmup, steps 10-119 | 24.537 | 24.944 | 26.861 | 0.794 | | |
| early window, steps 10-24 | 26.373 | 27.042 | 27.318 | 0.319 | 300,047 | 13.979s |

Drop series (120-STEP RUN — the LR schedule is defined over num_train_steps, so step 119 here is
END-OF-ANNEAL and is NOT comparable to a 350-step run's step 119 at ~68% of peak LR):
0.169 @0 -> 0.859 @10 (QB warming up) -> 0.607 @30 -> 0.304 @40 -> 0.191 @60 -> 0.128 @90 ->
0.124 @100 -> 0.103 @110 -> 0.089 @119. Tail-30 sits ~0.09-0.13.
Do not place this 0.089@119/120-step figure in the same column as the proxy's 0.071 tail-100/350-step
figure without the annotation; they are different points on different schedules.
Loss 10.31 @2 -> 5.59 @119, descending cleanly throughout.

### The EP-versus-FSDP comparison, both ways, honestly

#7201's 4-of-128 comparators are 1-rack FSDP 12-STEP probes: 22.7% (QB-off, chunk-2) and 23.1%
(QB-on, full-feature + host offload).

1. STEADY-STATE, the honest number: mine is **24.59%** at ~9-13% drops. It is the only steady-state
   number that exists at this shape — theirs stop at 12 steps. Against their best (23.1%) that is
   **+1.5pp and +8.3% tok/s** (276.4K vs 255.3K).
2. LIKE-FOR-LIKE measurement window: a 12-step probe cannot have converged QB. At step 12 my own drop
   fraction is ~0.86, and by my own protocol caveat heavy-drop runs READ HIGHER because dropped
   assignments gather the zero pad row. My matched early window (steps 10-24, drops 0.86 -> 0.66)
   reads **27.04%**. If their probes sat in a similar regime, the like-for-like gap is nearer +3.9pp.
   I cannot verify their drop trajectory, so I present (1) as the result and (2) as the caveat that
   makes the comparison fair rather than as a second claim.

Direction of the remaining caveats, stated so the reader can sign them:
- CONSERVATIVE: I run sliding window 2048 where the candidate runs sw512 with 5:1 local:global, so my
  configuration does MORE attention work. The candidate's own attention config would read higher.
- CONSERVATIVE: I run no XSA / attn-gate / GatedNorm; the 23.1% row does.
- NEUTRAL: host offload is on in both mine and the 23.1% row. Default allocator and default 0.75
  fraction in mine, so no memory-knob caveat to sign.
- AGAINST comparability with d1's d5120 control (22.66%): that control has no offload. This leg is
  built to match #7201's row, not d1's.

VERDICT: at the one-rack hero candidate shape, EP64 with the custom adjoint beats the FSDP
parallelization that #7201's candidate assumes, on the honest steady-state number, with the surviving
caveats all pointing the same way. This is the first evidence that the EP line is optimizing the path
the run should actually take at the shape most likely to be chosen.

### What this does NOT say
It says nothing about 4-of-256, which does not fit one rack at EP64 (R4) and whose candidate command
already assumes 2 racks. It does not compare against a steady-state FSDP number, because none exists.
And 24.59% is a THROUGHPUT number at ~9-13% drops: the compliant configuration costs extra, and per R3
the corrected recommendation at this shape is cf1.05 + spill m=1..2 for about -0.65pp, landing near
**~23.9%** compliant — still above the 23.1% FSDP row, which is the comparison that matters.

## Check-in 17 — three qualifications that MUST travel with the 24.59% headline

### Q1. The largest uncertainty: the comparison is cross-drop-regime in an UNKNOWN direction

#7201's 4-of-128 rows do NOT log drops — that is the emission bug d1 found, where the per-layer count
is computed but never reaches the tracker. So their drop rate at 22.7% / 23.1% is unknown.

This matters because of a result this effort established itself: heavy-drop runs read HIGHER MFU,
since dropped assignments gather a zero pad row and do less real work at the same step accounting. My
24.59% is measured at ~9-13% drops. If their drops were LOWER than mine, some part of my +1.5pp is the
drop artifact rather than the parallelization. If HIGHER, my advantage is understated. The direction is
genuinely unknown, and this is the single largest uncertainty on the headline — larger than any of the
configuration caveats, all of which at least have a known sign.

CONSEQUENCE FOR HOW TO REPORT IT: lead with the PAIR, not the raw number.

    throughput frontier   24.59%  at ~9-13% drops   (cross-drop-regime vs 23.1%, direction unknown)
    compliant projection  ~23.9%  at <3% drops      (cf1.05 + spill m=1..2, about -0.65pp)

The compliant figure is the more robust claim precisely because it prices the fidelity instead of
leaving it as an unpriced difference between the two sides of the comparison — and it still clears the
23.1% FSDP row.

ONE-JOB FIX AVAILABLE: anyone rerunning the #7201 FSDP rows should apply the drop-metric fix
(d1's 2d4a87395) FIRST. Those runs already compute the per-layer drop count; only the tracker emission
is missing. One job's cost would resolve this ambiguity permanently, and it currently sits on the most
decision-relevant comparison in the effort.

### Q2. Every drop figure must carry its run length

d1 established that the LR schedule is defined over `num_train_steps`, so step 119 of my 120-step run
is END-OF-ANNEAL while step 119 of a 350-step run is mid-schedule at ~68% of peak LR. Different
optimization states produce different routing concentration, so the numbers are not interchangeable.

    my 4-of-128 leg:  drops 0.089 @119 of 120 steps   (end-of-anneal)
    proxy reference:  drops 0.071 tail-100 of 350 steps (mid-to-late schedule)

These must never sit in the same column without the annotation. It also means my model's cand-B
prediction of 7.10%, calibrated against 350-step tails, is not directly checkable against this leg's
8.9% — the run lengths differ, so this leg neither confirms nor refutes the m=0 projection. Confirming
it needs a 350-step leg at the d6144 shape, which is now the top unmeasured item in my direction.

### Q3. Scope: this does not answer the goal as posed

The original goal named 4- or 8-of-256. 4-of-128 is a genuine #7201 top candidate and the one-rack
answer, but it is a DIFFERENT shape. Stated plainly:
- 4-of-256: no MFU number, and none is obtainable on one rack at EP64 (R4). Its own candidate command
  assumes 2 racks.
- 8-of-256: 22.66% at the d5120 proxy, which is where this effort's optimization work was done.
- 4-of-128 (this result): 24.59% steady, EP-favorable.
So the honest framing is "the one-rack candidate is both faster and EP-favorable", NOT "the goal is
met at 256 experts".

### Why the magnitude is credible rather than surprising

Top-4 halves the assignments relative to top-8, which halves both the a2a bytes and the expert-GEMM
rows. At the same batch and sequence length that is a direct throughput reduction in the two costs this
effort has spent all session optimizing, so a top-4 shape SHOULD read higher than a top-8 shape
independent of parallelization. That makes granularity a throughput lever in its own right, which the
run decision should weigh alongside the parallelization choice.
Caveat on the cleanliness of that inference: my 24.59% (d6144, i3072) and the 22.66% proxy (d5120,
i1280) differ in hidden and intermediate dimension as well as in top-k, so the comparison is a
mechanism argument, not a controlled A/B. I did not find an 8-of-256 row inside #7201 to check it
against; its 4-of-256 row is 18.6% but at 2 racks, where the cross-rack penalty dominates. A clean
test would be one d6144 leg at top-8 of 128 against this one.
