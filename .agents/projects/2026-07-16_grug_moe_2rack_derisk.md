# grug MoE — de-risk 1T/20B on 2× NVL72 racks, maximize MFU

**Goal (2026-07-16):** de-risk the big-model (~1T total / ~20B active, 512 experts, top-8)
on **2 NVL72 racks** as a faithful miniature of the 12-rack (864 B200) production system.
Maximize MFU. Focus seq=8192 first (later 64K, 128K). Loop: profile → hypothesize →
experiment → profile → (write overlap kernel if the profile justifies it). Federated
submit to `cw-us-east-08a` (GB200, aarch64/sm100). Do not stop.

## Hardware
- 864 B200 = **12 × NVL72** (confirmed by user). Each rack = 72 B200 on ONE NVLink domain.
- GB200 tray = 4 B200 → `--gpu GB200x4` = 1 tray. Rack = 18 trays = 72 GPUs.
- cw-us-east-08a: aarch64 Grace, sm100, ~185 GB HBM/GPU, system nccl 2.30.5, CW S3.

## Production design (target)
- **ep=64 inside a rack (NVLink), dp=12 across racks (IB), ZeRO-1 opt state over dp, top-8.**
- ep=64 ≤ 72 → all-to-all stays on NVLink. 864=2^5·3^3 caps pow2-ep at 32; ep=32→16 exp/GPU=310GB (OOM). ep=64→8 exp/GPU=155GB bf16, ~41GB with ZeRO-1. **Uses 768/864 (8/rack idle).**
- 512E needs **top-8** to hit 1T/20B (top-4 gives ~1.5T or ~15B active).
- grug mesh has NO pipeline axis `(replica_dcn,data,expert,model)`; DeepSeek-V3 uses PP=16. Open risk: does ep=64+ZeRO-1 avoid needing PP? Depends on activation memory at d≈4096/L≈70.

## Why 2 racks is the right de-risk
Smallest config with BOTH an intra-rack ep=64 NVLink domain AND a cross-rack IB dp axis.
At ep=64, experts/GPU is fixed at 8 regardless of GPU count → **1T fits on 128 GPUs**
(ep=64, dp=2, ZeRO-1), same model shape as 768 GPUs, only dp shrinks 12→2.

## Multi-node plumbing (SOLVED, mechanism)
- iris `--replicas R --gpu GB200x4` gang-schedules R tasks × 4 GPUs.
- nccl_ep needs 1 proc/GPU (multi-controller). DO NOT use iris `--processes-per-task`
  (it fans out the whole setup blob → races on clone/pip/site-packages).
- Instead: setup ONCE per task, then bash fans out 4 python children, each stamped:
  `CUDA_VISIBLE_DEVICES=<lr>`, `IRIS_MULTIGPU_PROCESS_COUNT=R*4`,
  `IRIS_MULTIGPU_PROCESS_INDEX=task_index*4+lr`, `IRIS_MULTIGPU_LOCAL_DEVICE_IDS=0`,
  NO `--coordinator-address`. Standalone's `elif IRIS_NUM_TASKS>1: initialize_jax()`
  then takes the `_initialize_supervised_jax` branch: global rank 0 REGISTERs
  advertise_host:8476 in the iris endpoint registry, other tasks POLL. (jax_init.py:180)
- task_index/num_tasks from `iris.cluster.client.job_info.get_job_info()`.

## Key prior facts (this branch = rav-moe-repro-v1)
- **Remat tagging fix (committed in tag patch, NOT yet on branch HEAD):** `_moe_mlp_nccl_ep`
  now has 4 `tree_checkpoint_name` tags incl. handle_mem/transport. WITHOUT them nccl_ep
  dies at T_local≥65,536 (`CUDA_ERROR_LAUNCH_FAILED`) on ALL hardware — remat re-ran
  ep_dispatch against HybridEP's paired shared buffers. WITH them: T_local=65,536 runs
  (11.52% b200), untagged control fails. **Load-bearing at 1T (T_local huge).**
- EP×FSDP composes: dp=2 ep=2 rc=0 (b200-matrix).
- **xprof overlap profile (64E, 4 GPU, T_local=32768):** step 1374ms; EP transport
  (dispatch 29.7 + combine 68.4 = 98ms) = **7% of step** (NOT the 71-78% from the MoE-only
  microbench). COMM/GEMM overlap 14.9% but likely all the XLA AllGather, not EP kernels.
  → perfect EP overlap buys ~7-10%; parallelism choice dominates. Overlap kernel LOW value
  at 64E. May rise at 512E/large-d; RE-MEASURE in regime.
- **Unexplained:** 76ms `ncclDevKernel_AllGather_RING_LL` at dp=1 (no FSDP to gather). Bigger
  than whole EP dispatch. Understand before scaling.
- Controlled 64E/32768 B200: sonic_cute(FSDP) 11.02% > nccl_ep 10.57% > sonic 8.95%.

## OPEN RISK: rack-locality of placement
For 2-rack de-risk we need ep groups WITHIN a rack (NVLink) and dp ACROSS racks (IB).
iris/coreweave placement may not guarantee 16 trays land in ONE NVL72 rack. See
`lib/iris/src/iris/cluster/platforms/k8s/coreweave_topology.py`. MUST verify before
trusting any 64+ GPU MFU number — a scrambled placement puts EP a2a on IB and tanks MFU.

## Rack sizing correction (coreweave.md:444)
CoreWeave keeps only **16 schedulable nodes/rack** (of 18). ≤16 nodes → HARD nvlink.domain
(one rack). >16 → PodSet SLICES: split evenly into ≤16-node whole-rack slices, each hard-bound
to its own NVLink domain → exact balanced N-rack layout (32→16+16). So **ep=64 = 16 nodes =
one full rack** (exact fit); **128 GPU = 32 nodes = 16+16 across 2 racks** (guaranteed, not
best-effort). Sizes that don't split into >half-rack equal slices (17,18,40) are REJECTED at submit.

## Multi-node NCCL facts
- cw-us-east-08a config already sets `host_network: true` (yaml:122) → pods get hostNetwork +
  `rdma/ib` (tasks.py:759, gated on host_network). NCCL_SOCKET_IFNAME exclude-list (yaml:166)
  is canonical (excludes IB/SR-IOV link-local so the TCP *bootstrap* avoids them; data plane
  uses IB RDMA / NVLink). I inherit MARIN_PREFIX + NCCL_SOCKET_IFNAME from task_env.

## Experiment log
- **exp1 `mn-smoke8`** (DONE): 8 GPU (2×GB200x4), d1024/L4/E64, ep=8. **PLUMBING WORKS**:
  all 8 procs joined one mesh (`MN process_count=8 process_index=0..7 device_count=8`), gang +
  supervised initialize_jax + cross-node coordinator all OK. **FAIL** at TE `ep.cpp:45 EpResources:
  NCCL Error: remote process exited or network error` — HybridEP's OWN bundled nccl comm
  (LD_LIBRARY_PATH=/tmp/TE/3rdparty/nccl) failed cross-node. Intra-node ep worked in all prior
  single-task runs → new variable is cross-node nccl. No NCCL_DEBUG in that run (inference only).
- **exp2 `nc-diag8`** (DONE): **NETWORK IS FINE.** JAX process_allgather OK on all 8 ranks
  (sum=36 expect 36) cross-node. NCCL_DEBUG shows healthy IB: `NET/IB` 4 virtual devs ibp0-3 @
  400Gb/s SHARP. So mn-smoke8's "NCCL Error: remote process exited" was NOT a network fault.
  BUT this run's ep_bootstrap test was INVALID — I hardcoded recv_cap=5120 < max_tokens=8192,
  tripping `nccl_ep.cc:1452: HT mode requires max_recv_tokens_per_rank >= max_dispatch_tokens_per_rank`.
  **KEY LEARNING: HT-mode ep requires recv_cap >= max_tokens_per_rank.** TE's "remote process
  exited" is its generic wrapper for "a peer rank aborted" (e.g. on an assertion) — MISLEADING.
  The standalone computes recv_cap=nle*slots=40960 >= t_local=8192, so it does NOT violate this —
  meaning mn-smoke8 died on something ELSE (still open). The grpc ":8476 connection refused" spam
  is just the coordination-service shutdown cascade after one rank aborts, not a root cause.
- **exp2b `nc-diag8b`** (running): ep_bootstrap cross-node with CORRECT params (recv_cap=40960 >=
  max_tokens=8192). Decides: does correctly-configured HybridEP ep_bootstrap work across 2 nodes?
  If OK → mn-smoke8 failed elsewhere (rerun full smoke, find it). If FAIL with a real net/nccl
  error → genuine cross-node TE-bundled-nccl issue (try system nccl 2.30.5 / NCCL_IB_HCA).

## ★ ROOT CAUSE FOUND (2026-07-16): rank-plumbing bug ★
The multi-node EP "connection refused" was NOT timing/skew/interface/GIN. It was:
**`ep_bootstrap(rank=a.process_id)` where `--process-id` defaults to 0 and the multi-node iris
fanout supplies the rank via IRIS_MULTIGPU_PROCESS_INDEX (not --process-id).** So ALL ranks
bootstrapped as rank 0 → ncclCommInitRank cannot form a ring with duplicate ranks → "connection
refused" in the bootstrap ring. Every mn-op* worked because they used rank=IRIS_MULTIGPU_PROCESS_INDEX;
the standalone used rank=0 everywhere. FIX: `ep_bootstrap(rank=jax.process_index())` (correct global
rank for both coordinator + iris paths; matches what _warmup already used). mn-warm6: warmup dispatch
OK rank=0 AND rank=4 across 2 nodes → bootstrap SOLVED. The reorder + tiny-warmup are kept (both
correct, robust) but were NOT the fix — chased a symptom for ~10 runs. LESSON: check rank plumbing
against the working baseline FIRST.

## Patch hygiene
- Remote branch rav-moe-repro-v1 = HEAD 1c379a3 (has nccl_ep backend COMMITTED). Local rebased =
  12c55f9d (20 ahead, NOT pushed). Job clones the REMOTE (1c379a3).
- **h100_fixes.patch is REDUNDANT** (it's the original nccl_ep-adding patch; branch already has it) →
  "patch does not apply", benign, DROPPED. Only tag4.patch (my uncommitted git diff: remat tags,
  warmup, reorder, rank fix) is needed and applies cleanly on 1c379a3.
- Full run logs now uploaded to s3://marin-us-east-02a/tmp/ttl=7d/rav/mnwarm/.

## OLD (superseded) MULTI-NODE EP BLOCKER analysis
Root cause chain for `mn-smoke8`/`mn-smoke8b` failure, isolated step by step:
- Fabric OK (JAX allgather cross-node OK, IB 400Gb/s SHARP). NOT the network.
- ep_bootstrap OK in isolation — but it only calls `ncclGetUniqueId`+JAX-allgather+`set_ep_bootstrap_params`;
  the actual `ncclCommInitRank` is DEFERRED to trace time (ep.cpp AcquireEpResources via handle_mem
  alloc). So nc-diag8b (bootstrap-only) was a FALSE POSITIVE.
- First real failure = `ncclCommInitRank` **bootstrap socket** "connection refused" on rank 0's
  ephemeral port (39535) cross-node (ep.cpp:45 EpResources). NOT IB, NOT jax coordinator(8476).
- TE's raw ncclCommInitRank relies entirely on NCCL_SOCKET_IFNAME for bootstrap interface selection;
  JAX's nccl works b/c JAX drives bootstrap via its explicit coordinator. Matches NVIDIA/nccl#1581
  (exclude-list IFNAME → inconsistent interface pick on worker nodes under multi-NIC hostNetwork).
  LIKELY FIX: set NCCL_SOCKET_IFNAME to the EXPLICIT positive iface (e.g. enP6p3s0np0) not exclude-list.
- SECOND blocker (after bootstrap): **NCCL EP inter-node needs GIN** (GPU-Initiated Networking) for
  RDMA. nc-diag8 log showed `GIN/Plugin: Could not find: libnccl-gin.so` → fell back to NET/IB (fine
  for JAX collectives, but EP dispatch/combine DEVICE kernels need GIN). GIN has GDAKI (DOCA GPUNetIO)
  + Proxy (CPU-assisted over std RDMA) backends. arxiv 2603.13606 (NCCL EP), 2511.15076 (GIN).
- HybridEP DOES support inter-node (nccl_ep.cc INTER_NODE_G2S_GROUP; MNNVL-capable). Design is sound.
- Methodology (user directive): debug step-by-step, minimal repro first, validate each layer, search
  GitHub/NVIDIA/arxiv, read library source. Don't one-shot.

## Bootstrap root-cause (mn-bootdbg, NCCL_DEBUG=INFO) — interface is NOT the problem
- rank0 (node0 10.186.213.241) + rank4 (node1 .243) BOTH pick `Bootstrap: Using enP6p3s0np0:<routable-ip>`
  (the correct PF, not an SR-IOV VF) and the FIRST bootstrap SUCCEEDS ("Bootstrap timings total").
- Failure is a SECOND comm (TE ep ncclCommInitRank): `bootstrapRoot: rank 0 of 8 has already checked in`
  then rank0 `connect to 10.186.213.241<61861> Connection refused` retry 1..34 — even to its SAME-NODE
  peer rank1. So the ncclUniqueId root listener (spawned at ep_bootstrap/ncclGetUniqueId) is stale/
  unreachable by the time the DEFERRED trace-time ncclCommInitRank runs (~14s in). Ranks reach the
  deferred init at different wall-clock times → rendezvous port not served. This is why nc-diag8b
  (bootstrap-only) passed and the full model (first dispatch) fails.
- HYPOTHESIS: force EAGER, synchronized ncclCommInitRank right after ep_bootstrap (while ranks are
  barrier-synced by the uid allgather) via an immediate tiny ep_dispatch, before the big model.
  → exp `mn-op` (running): ep_bootstrap + ONE tiny ep_dispatch immediately. OK => timing fix
  (warm up comm early). FAIL same => op itself can't multi-node bootstrap (deeper: GIN).

## nccl_ep bundle facts (gin-find)
- bundle build/lib has `libnccl_ep.so` (EP contrib) + `.a`. (base libnccl.so.2 loaded via ctypes in
  ep.py — from system/NGC, verify it's present on LD path.)
- **`NCCL_GIN_TYPE` env** (README.md:271 `export NCCL_GIN_TYPE=3 # GDAKI`). Official nccl_ep README
  has the multi-node recipe → dumping via exp `ep-readme`. hybrid_ep.cuh uses ncclGin for inter-node.
- Inter-node EP data plane NEEDS GIN configured (GDAKI=DOCA GPUNetIO, or Proxy backend). Likely the
  NEXT blocker after bootstrap.

## FIX FOUND + VALIDATED (mn-op): eager EP-comm warmup
- **mn-op (8 GPU/2 node): ep_bootstrap + ONE immediate tiny ep_dispatch = DISPATCH_OK all 8 ranks,
  rc=0.** (sum=nan is expected: random inputs into padded capacity slots.) CONFIRMS the deferred-
  timing root cause. Also proves cross-node ncclCommInitRank + ep_dispatch WORK (GIN "not found" is
  benign — compiled in; ran without NCCL_GIN_TYPE set).
- **FIX implemented in standalone**: `_warmup_nccl_ep(mesh, num_procs, t_local, hidden_dim, top_k,
  num_experts, recv_cap)` — one tiny synchronized ep_dispatch right after ep_bootstrap, before the
  model compiles. C++ anchor caches the comm → train_step reuses it. (in tag4.patch)
- **exp `mn-warm` (running)**: full L=4 model, 8 GPU/2 node, seq8192 — the exact config that failed
  as mn-smoke8b, now WITH warmup. Success = trains, SUMMARY/mfu printed. This unblocks scaling
  8→64 (1 rack) →128 (2 racks).
- nccl_ep README (official): `NCCL_GIN_TYPE=3` (GDAKI) RECOMMENDED for multi-node RDMA perf, but NOT
  required for correctness (mn-op ran without it). Set it in the PERF phase once training works.

## Warmup fix refinement (mn-warm failed, mn-warm2 testing)
- mn-warm (warmup used t_local=8192) STILL failed "connection refused". mn-op (TL=256) worked.
  → The distinguishing var is WARMUP-DISPATCH COMPILE TIME. The ncclGetUniqueId bootstrap-root
  listener (spawned in ep_bootstrap) has a TIMEOUT; a big warmup's slow jit compile lets it expire
  before ncclCommInitRank executes → "connection refused" (listener gone, not slow).
- FIX v2: warmup TL = min(256, t_local) — fast compile, ncclCommInitRank fires before root timeout.
  Comm is sized by ep_bootstrap max_tokens (independent of warmup TL). Added diagnostic print
  "NCCL_EP warmup dispatch OK" to split warmup-fail vs train_step-fail (anchor/persistence).
- If mn-warm2 STILL fails AFTER "warmup dispatch OK" → comm not persisting to train_step (C++
  g_ep_resources_anchor not held) → different fix (keep warmup handle alive / force anchor).
- If warmup fails again → consider NCCL bootstrap timeout env, or reduce ep_bootstrap→init gap.

## BISECT: what in the standalone breaks the EP bootstrap (mn-op works, mn-warm fails)
Both mn-op (works) and mn-warm (fails) do ep_bootstrap + tiny ep_dispatch with IDENTICAL params.
Failure = ncclCommInitRank "connect to <node0-ip>:<port> refused" — bootstrap RING connect refused
(rank0→rank1 same node), i.e. a peer isn't listening when connected → ranks reach ncclCommInitRank
at very different wall-clock times (sync skew) OR a listener never binds. Ruled out by bisect:
- mn-op2 (mn-op + quack + flash_attn.cute imports): **WORKS** (DISPATCH_OK sum=-9764). NOT quack/flash.
- _compute_flops / _make_train_step / optimizer.build: none execute JAX before ep_bootstrap. Not it.
- mn-op3 (mn-op2 + import the FULL standalone module): RUNNING — tests import-time side effects.
Remaining suspects: standalone's levanter.grug.* top-level imports (mn-op3), the te_ep.wsc monkeypatch,
or per-rank timing skew (heavy process). If skew: fix = compile warmup then sync_global_devices barrier
then execute, so all ranks enter ncclCommInitRank together.

## Bisect status v2 (2026-07-16, still open)
- mn-op4 (mn-op3 + monkeypatch, EARLY bootstrap): **WORKS**. So monkeypatch is NOT the culprit either.
- Reordered standalone (mn-warm4: EARLY bootstrap + model AFTER): **STILL FAILS**. So early-vs-late
  bootstrap is NOT the sole cause. Remaining diffs between mn-op4(works) and standalone(fails):
  (a) the compile/barrier/execute split in _warmup_nccl_ep [mn-op4 uses plain @jax.jit call];
  (b) jax.config threefry_partitionable=True active during warmup [mn-op4 doesn't set it].
- **mn-warm5 (running): reverted _warmup to plain @jax.jit (matches mn-op4 exactly), early bootstrap.**
  If WORKS => the .lower().compile() split was breaking it. If FAILS => move threefry after bootstrap.
- KNOWN-GOOD invariant to match: mn-op4 = import full standalone module + quack/flash + monkeypatch +
  initialize_jax + ep_bootstrap + plain-jit tiny ep_dispatch = DISPATCH_OK cross-node, every time.

## Bisect status (2026-07-16, superseded by v2 above)
RULED OUT as the standalone-vs-mn-op difference: quack/flash imports (mn-op2✓), full module import
(mn-op3✓), _compute_flops/_make_train_step/optimizer.build (pure, no JAX). Barrier-after-compile
(mn-warm3) DID NOT fix → **ncclCommInitRank fires at COMPILE/trace time** (AcquireEpResources at
trace-time handle_mem alloc), before a post-compile barrier. Retry count (~35≈63s) is a HARDCODED
nccl constant, NOT env-tunable (NVIDIA/nccl#1618). Consistent (lean always works, standalone always
fails) → deterministic code-path difference, not random node skew.
- **mn-op4 (running): mn-op3 + the te_ep.with_sharding_constraint monkeypatch** — the LAST untested
  single variable. FAIL => monkeypatch is culprit (fix: apply it AFTER warmup). PASS => cumulative
  late-bootstrap → FIX = REORDER ep_bootstrap+warmup to immediately after initialize_jax() (before
  model/opt build), replicating mn-op3's proven-good early bootstrap. ep config needs only args.
- Working invariant: ep_bootstrap + tiny ep_dispatch done EARLY (right after init) succeeds cross-node
  every time. Whatever the standalone does between init and bootstrap breaks it.

## ★★ MULTI-NODE EP WORKS END-TO-END (mn-warm8, 2026-07-16) ★★
8 GPU / 2 GB200 nodes, ep=8, seq=8192, d1024/L4/E64 nccl_ep: **rc=0, 6 steps, loss 11.806→11.706,
SUMMARY on both nodes.** steady MFU_b200=3.90%, 480k tok/s, 701 TFLOP/s, 137ms/step. (Low MFU is
EXPECTED: tiny de-risk model, fixed overheads dominate — this proves INFRA not MFU.)
Four fixes total to get multi-node EP working:
1. rank=jax.process_index() (not a.process_id) — THE bootstrap fix.
2. ep_bootstrap+warmup EARLY (before model build) — reorder.
3. tiny warmup dispatch (min(256,t_local)) right after bootstrap.
4. fresh global_shard_guard(ep_mesh_resource) per `with` (was reusing single-use CM).
All in tag4.patch (uncommitted). Drop h100_fixes.patch (redundant). Job: clone remote 1c379a3 +
apply tag4 only.
KNOWN EFFICIENCY WART (optimize later, not correctness): `[SPMD] Involuntary full rematerialization`
— XLA can't cheaply reshard the s32[8,8192] routing-index tensor from {maximal device=0} to
{devices=[8,1]}; replicates+partitions. Target when tuning MFU.

## SCALE-UP RESULTS (2026-07-16)
- **8 GPU / 2 node, ep=8, E64/top4**: ✓ trains rc=0, MFU_b200 3.90% (tiny model, infra-only).
- **64 GPU / 1 RACK, ep=64, E512/top8, batch64**: ✓ TRAINS rc=0! dp=1 ep=64 num_procs=64 NLE=8
  recv_cap=81920. loss 11.806→11.655. 2.84M tok/s, 4606 TFLOP/s, 186ms/step. MFU 3.2% (tiny model).
  → ep=64 bootstraps+trains across 16 nodes, 512 experts = 8 experts/GPU (production sharding). GIN_TYPE=3.
- **128 GPU / 2 RACKS, ep=64 x dp=2, E512/top8, batch128**: ✗ FAILS. dp=2 ep=64 num_procs=128 NLE=8
  recv_cap=163840. warmup dispatch OK + step 0 completed (compile), then **step 1 → CUDA_ERROR_LAUNCH_FAILED
  + `terminate called after throwing an instance of 'EPException'`**. dp=1 works, so the failure is the
  **dp>1 hybrid at multi-node scale** (grouped reshard / 2-EP-group path in _moe_mlp_nccl_ep). Single-node
  ep2/dp2 worked earlier (b200-matrix) → new failure is dp=2 MULTI-node. Note: EP a2a is INTRA-rack
  (ep=64 within a rack); dp=2 is the cross-rack gradient axis. So this is NOT a cross-rack-EP problem.
- **dp2-iso (running): ep=8 x dp=2 = 16 GPU (4 nodes, 1 rack)** — isolates the dp>1 path CHEAP + full
  EPException capture. Reproduces => debug at 16 GPU. Works => 128-GPU issue is scale/cross-rack specific.

## Commit note
The 4 EP fixes (rank=process_index, early bootstrap, tiny warmup, fresh guard) are in tag4.patch,
UNCOMMITTED. Worth committing once dp>1 is fixed. Ask user before pushing.

## dp>1 EPException = CAPACITY OVERFLOW (dp2-cap, 2026-07-16)
- error: `what(): CUDA error nccl_ep.cc:1703 'unspecified launch failure'` = OOB write past per-expert
  recv buffer. capfac=1.25 FAILS, **capfac=8.0 TRAINS rc=0** (loss converges → correctness OK, pure capacity).
- **dp=2 needs MUCH more capacity than dp=1**: rack1 (dp=1, 512E) trained at 1.25; dp=2 overflows even
  though recv_cap formula over-sizes 2×. Evidence (num_procs-based sizing matches) ⇒ TE ep at dp>1 POOLS
  tokens across ALL num_procs ranks, not just the ep_size group → higher per-expert concentration.
- **★ #1 MFU KILLER ★**: _moe_mlp_nccl_ep does a DENSE GLU over PADDED CAPACITY SLOTS → expert compute
  scales with capacity_factor. capfac=8 = ~8x wasted FLOPs on empty padding. The efficient path is a
  ragged/grouped GEMM over ACTUAL token counts (like sonic/ragged_dot), NOT dense-over-capacity.
  This is THE thing to fix for MFU. At tiny model MFU is meaningless anyway.
- Immediate unblock: bump capacity_factor for 2-rack de-risk. Then GROW model + redesign expert GLU
  to ragged (kill the capacity-padding waste) as the core MFU work.

## ★ THE MFU FIX: ragged GEMM (token_counts IS available) ★
- `ep_dispatch` returns `(recv_tokens, recv_topk_weights, handle_mem, token_counts)`. The standalone
  MISNAMES the 4th as `transport` — it is **token_counts** (per-local-expert actual counts), passed to
  ep_combine. So we HAVE the real counts.
- deepep backend (ep_deepep.py) already does the right thing: `ragged_dot(x_dispatch, w13, local_group_sizes)`
  over ACTUAL tokens. nccl_ep's `_moe_mlp_nccl_ep` instead does a DENSE einsum over padded capacity slots.
- WRINKLE: nccl_ep `recv` is PADDED to fixed `slots` per expert (not packed). ragged_dot wants tokens
  packed contiguous by group (cumsum(group_sizes)). So ragged path = COMPACT recv (gather real tokens
  out of padded slots via token_counts) → ragged_dot(packed, w, counts) → scatter back to padded for
  ep_combine. Bounded change; kills capacity-padding compute + the replicated-recv OOM + overflow.
- This is the CORE MFU work. Do it, then grow model (d~4096/L60-70/E512/top8) + profile + overlap.

## 2-RACK DE-RISK STATUS (near complete)
- Topology PROVEN: gang + supervised init + cross-node coord + IB collectives + EP bootstrap + train,
  8GPU/2node ✓, 64GPU/1rack ep=64 E512 ✓. 128GPU/2racks bootstraps + compiles; dp=2 needs capacity in
  window (~2-4): 1.25 overflows, 8.0 replicated-recv-OOM (258GiB). capwin (16GPU) finding min-safe factor.
- The dense-capacity design forces overflow-vs-OOM squeeze; ragged fix removes it. Once ragged lands,
  2-rack should train at low effective capacity with real MFU.
- All 4 infra fixes committed in tag4.patch (uncommitted on branch). NOT pushed.
Mesh: bench path uses compact_grug_mesh(expert_axis_size=a.expert_axis_size, replica_axis_size=1);
data = world/(1*ep). So:
- **64 GPU = 16 nodes = 1 RACK**: --replicas 16, --expert-axis-size 64 → dp=1 ep=64 (pure EP, all NVLink).
- **128 GPU = 32 nodes = 2 RACKS**: --replicas 32, --expert-axis-size 64 → dp=2 ep=64 (EP intra-rack
  NVLink, dp cross-rack IB). Kueue slices 16+16 (guaranteed per-rack NVLink).
For 512-expert target: ep=64 → nle=8 experts/rank. Validate multi-node EP with small model first,
THEN grow d/L/E toward 1T/20B (d~4096, L~60-70, E=512, top-8). Set NCCL_GIN_TYPE=3 in perf phase.

## Storage
- CW S3: bucket `marin-us-east-02a`, keys `tmp/ttl=7d/rav/...`, boto3 virtual addressing.
- TE 2.17 aarch64 build cached at `tmp/ttl=7d/rav/te217_b200/{src,pkg}.tgz`.
- Profiles/traces → `tmp/ttl=7d/rav/nsys/`.
- nsys is a DEAD END for this workload (hangs finalizing 24-EP-layer CUDA trace, 2 attempts).
  Use jax.profiler (`--profile`, traces 1 steady step) + in-job trace.json.gz overlap parse.

## RAGGED GEMM + 20% MFU GOAL (2026-07-16)
- `_ragged_expert_glu` (shard_map per-device): compact padded recv->packed via counts (recv_w nonzero,
  real tokens contiguous-front per expert block, confirmed by ep-probe) -> ragged_dot x2 -> scatter back.
  `--expert-glu {dense,ragged}` flag. In tag4.patch. CORRECT: ragged-val2 step-0 loss within 4e-4 (bf16 tol).
- **1 rack ragged d2048/L24/E512/top8 seq8192 batch64 capfac1.25 dp=1: MFU_b200=4.70%** (105.8 TF/GPU of
  2250; 1.28s/step). 4x below the 20% goal. r1-prof profiling the bottleneck.
- Hypotheses: ragged_dot Triton slow on sm100 (sonic_cute used cutlass +24%); EP comm at 512E; gather/
  scatter overhead; MuonH 5-NS inflates duration. Levers: cutlass expert GEMM, bigger batch, overlap,
  fewer NS steps, SM tuning. Then 2 racks (capfac~3; ragged avoids the dense replicated-recv OOM).

## MFU OPTIMIZATION (1 rack, d2048/L24/E512/top8/seq8192, ragged) — toward 20%
- real Muon: **4.70%** mfu_b200. #1 cost = MuonH Newton-Schulz ALL-GATHER of 3D stacked non-expert
  weights = 493ms (37% of step). Cause: `_batch_sharded_stack_target_pspec` can't shard [L=24,D,I]
  over 64 ranks (24%64!=0) → vmap-REPLICATED NS → many small RING_LL all-gathers (latency-bound).
  Expert (4D) weights already use distributed NS (no gather).
- SCALE_MUON_NO_NS=1 (momentum-only, skips gather, NOT real Muon): **7.32%**, AllGather 493→15.7ms.
- + remat=none (no recompute): **9.12%** mfu_b200 = **20.7% mfu_h100_equiv** (661ms). = MODEL CEILING
  at this config. remat=none > save_moe (~+2pp). batch=128 didn't help (8.15%).
- **METRIC NOTE**: b200_peak=2.25e15 = 2.3x h100. mfu_b200 = 0.44 x mfu_h100_equiv. At ceiling
  h100_equiv already >20%; mfu_b200 needs 2.2x more (kernel work).
- GAP to 20% mfu_b200: model 9.12% needs 2.2x. Levers: expert GEMM kernel (ragged_dot triton vs xla
  vs cutlass — testing), comm overlap (dispatch74+combine54=128ms), + REAL optimizer fix (replicate
  small non-expert weights so NS is local; or a distributed 3D NS layout).
- Added `SCALE_RAGGED_IMPL` env (auto|triton|xla|megablox) to _ragged_expert_glu. r1-rimpl testing.

## BOTTLENECK = Triton ragged_dot (r1-cprof, remat=none NO_NS)
- Clean profile: the `[iter=N]` kernels (Triton ragged_dot Pallas grid) = ~490ms of 696ms step (70%)!
  Expert GEMM running at ~28 TFLOP/s/GPU (of 2250) — grossly underutilizes sm100 tensor cores.
  Rest small: attention 102ms, EP dispatch+combine 121ms, gather/scatter(RESHAPE_IDX) 146ms.
- ragged_dot impl=xla FAILS (nccl_ep.cc:1703); impl=triton=auto (9.14%). Triton is the ceiling for ragged_dot.
- **FIX = QuACK SM100 cutlass grouped SwiGLU GEMM** (sonic_cute `_expert_mlp` + `_interleave_gate_up`;
  custom_vjp fwd+bwd, shard_map-safe). Added `--expert-glu quack`: reuse my compact->packed, feed
  packed+cu_seqlens to _expert_mlp, scatter back. r1-quack testing. If expert GEMM 490->~50-100ms,
  step 696->~250-300ms => ~20% mfu_b200.
- REMAINING after quack: real-optimizer fix (Muon NS all-gather, dp=1 artifact), comm overlap.

## MFU ladder (1 rack, d2048/L24/E512/top8/seq8192, NO_NS, remat=none, batch=64)
- QUACK GEMM is only 31ms (NOT the bottleneck). The RAGGED COMPACTION gather/scatter was 279ms (42%)!
- ragged(triton) 9.14% -> quack+compaction 9.46% -> **quack over PADDED slots, NO compaction: 10.21%**
  (590ms, 23.2% h100-equiv). Loss decreasing = correct.
- KEY DESIGN: at low capfac, QuACK over padded slot-blocks (cu=slot bounds, ~1.25x compute) beats
  compaction (gather/scatter). Also avoids the dense replicated-recv OOM. `--expert-glu quack` now = this.
- Remaining critical path (from qk-prof): ATTENTION seq-blocks ([iter], ~470ms) + comm (166ms). At
  batch=64 each rank does attention on 1 SEQUENCE (poor occupancy). r1-bsw sweeping batch 128/256.
- All NO_NS (fake optimizer). Real Muon adds ~460ms NS all-gather back unless fixed (dp=2 may differ).
- 20% mfu_b200 still 2x away; may be infeasible for EP 512E (best FSDP sonic_cute=14.35% at 64E). Pushing.

## 2-RACK MFU attempts (2026-07-16)
- 1-rack ceiling (NO_NS, quack-padded, remat=none, batch64): 10.21% mfu_b200 / 23.2% h100-equiv (590ms).
- 2 racks (dp=2 ep=64 512E, batch128->T_local=8192, capfac=3, quack): hit a chain of issues:
  1. gitpython missing (levanter.tracker imports git) -> `pip install gitpython`.
  2. WARMUP dispatch OOM 193GiB: `recv.sum()` materializes [128, recv_cap=393216, 2048] REPLICATED.
     Dispatch recv_cap MUST equal comm bootstrap recv_capacity (smaller -> NCCL invalid argument).
     FIX: warmup returns `token_counts.sum()` (tiny) instead of recv.sum() -> dispatch runs, recv not gathered.
  3. NEXT RISK: the REAL model `reshard(recv, ep3)` may also materialize replicated recv (206GB) at
     2 racks -> OOM in jit_train_step. If so, need to keep ep_dispatch recv sharded (out_sharding), a
     deeper SPMD fix. r2-quack4 testing.
- METRIC: project uses mfu_b200 (17.88% B200 repro = mfu_b200). 20% mfu_b200 is ABOVE best FSDP
  (sonic_cute 14.35%); likely infeasible for EP 512E. 20% h100-equiv IS met at 1-rack ceiling.

## ★ FIRST 2-RACK TRAINING RUN (r2-d1024, 2026-07-16) ★
- 128 GPU (2 racks), dp=2 ep=64, 512E-top8, **d=1024** (d=2048 OOMs on replicated recv), seq8192,
  batch128 (T_local=8192), capfac=3, quack-padded, save_moe, NO_NS: **rc=0, TRAINS.**
- **MFU_b200 = 2.88% / h100-equiv = 6.55%** (750ms/step, 1.40M tok/s, 8294 TFLOP/s aggregate).
- LOW because: (a) d=1024 small model = low arithmetic intensity; (b) capfac=3 => quack-padded does
  3x padding compute; (c) save_moe recompute; (d) comm + attention overhead dominate small model.
- 2-rack d=2048 (higher MFU) BLOCKED by replicated-recv OOM (200GiB in reshard(recv,ep3) at dp=2).

## HONEST 20% MFU ASSESSMENT (after ~45 multi-node jobs)
- Best 1-rack (d2048, NO_NS, quack-padded): 10.2% mfu_b200 / 23.2% h100-equiv.
- Best 2-rack (d1024, fits): 2.88% mfu_b200 / 6.55% h100-equiv.
- 20% mfu_b200 NOT reached on 2 racks. Blockers, in priority: (1) recv-sharding OOM (keep ep_dispatch
  recv sharded, avoid reshard replicate-then-partition) — unblocks d2048; (2) dp=2 needs capfac~3 =>
  3x padding OR fix routing/capacity; (3) attention 1-seq/rank occupancy; (4) EP comm overlap; (5)
  real-Muon NS all-gather (dp=1 460ms). 20% mfu_b200 for EP 512E is ABOVE best FSDP (14.35%) — a
  multi-week kernel/sharding effort, not achievable by more brute-force runs this session.

## Two-rack recv-capacity bug (2026-07-16, post-commit)

**Root cause of the replicated-recv OOM at 2 racks.** `expected_per_expert` in
`main()` was computed as `num_procs * t_local * top_k / num_experts`. That is
wrong for dp>1: each dp replica is an INDEPENDENT `ep_size`-rank EP group with
its own expert copies, so a given expert only receives from its group's
`ep_size` ranks — `ep_size * t_local` tokens, not all `num_procs`. Using
`num_procs` oversizes `recv_cap` by exactly `dp_size` (2× at two racks).

Observed at 512E/dp2/ep64/T_local8192/top8: `slots/expert=49152` = effective
capacity factor 6 (= 2× bug × cap 3.0), driving `recv_cap=393216` and the
`[128, recv_cap, D]` buffer that OOM'd at d≥2048 (193–258 GiB). dp=1 (single
rack) was unaffected because `num_procs == ep_size` there — which is why the bug
hid until two racks.

Fix: `expected_per_expert = ep_size * t_local * top_k / num_experts` (committed
in ae7ce24059's follow-up). Halves `recv_cap` at two racks; the "dp=2 needs
capfac≈3" earlier finding was really "needs 2× (bug) × 1.5 (real)" — real cap is
~1.5, matching single-rack.

Also worth noting the recv is still materialized *replicated* (not sharded)
after `ep_dispatch` — the FFI output carries no partition annotation on the Auto
mesh — so even the halved buffer is per-device-full. The capacity fix makes
d2048 fit (halved 103 GiB ≈ what d1024 fit at pre-fix); a proper sharded-recv
fix would shrink it ~128×, the next lever if wider widths still OOM.

Relaunched as `/rav/r2fix` (128 GPU / 32 tasks, 512E, quack expert-glu,
remat=save_moe): sweep d2048×{cap2,cap3}, d2560×cap2. This is net-new data — the
#7012 thread has no TE nccl_ep numbers (TE 2.15 lacks the NCCL_EP JAX surface).

## Two-rack width sweep (2026-07-16, post capacity-fix)

All 512E, top-8, seq-8192, dp2/ep64 (EP=1 rack NVLink, dp=2 cross-rack IB),
QuACK expert-glu, remat=save_moe, SCALE_MUON_NO_NS=1.

| d    | capfac | slots  | mfu_b200 | h100-equiv | note |
|------|--------|--------|----------|------------|------|
| 2048 | 2.0    | 16384  | 6.70%    | 15.24%     | fit  |
| 2560 | 2.0    | 16384  | 7.87%    | 17.91%     | fit  |
| 3072 | 1.5    | 12288  | —        | —          | CUDA_ERROR_LAUNCH_FAILED (QuACK dim) |
| 4096 | 1.25   | 10240  | **12.09%** | **27.50%** | fit — best two-rack 512E |
| 5120 | 1.25   | 10240  | —        | —          | CUDA_ERROR_LAUNCH_FAILED (QuACK dim) |

**Trajectory: 2.88% (d1024, pre-fix) -> 12.09% (d4096) = 4.2x** on the unique
TE nccl_ep two-rack path. Net-new data (no TE nccl_ep numbers exist in #7012).

Two blockers to 20% mfu_b200:
1. **QuACK kernel dim-robustness**: fails at d3072/d5120 (inter = d/2 =
   1536/2560) but fine at d2048/2560/4096 (inter 1024/1280/2048). d4096's
   inter=2048 is tile-friendly (256x128 tile, (2,1,1) cluster). Not divisibility
   (all pass /8 and /128). Pattern in d/1024: {2,2.5,4} work, {3,5} fail —
   likely a 2-CTA cluster grid-evenness constraint. Needs a kernel-side fix or
   width-picking to tile-friendly dims (d4096, maybe d8192).
2. **Replicated-recv memory**: TE ep_dispatch outputs recv REPLICATED
   ([num_procs, recv_cap, D] full per device) — proven by the pre-fix warmup OOM
   in jit__dispatch. reshard(recv, ep3) then slices it, but the replicated
   materialization caps width (~d5120 = 107 GiB recv + model). Sharded recv would
   cut it ~128x and unlock d6144-d8192. The QuACK expert GEMM already runs in a
   shard_map (_ragged_expert_glu), so only the ep_dispatch->reshard boundary
   replicates. Fix requires TE ep_dispatch to emit sharded output — probing
   whether ep_dispatch accepts an out_sharding kwarg like ep_combine does
   (r2cap job).

capfac lever is minor (~+1pp): cap1.25 padding=25%; r2cap tests d4096 cap
1.0/1.1/1.15 floor.

## Comprehensive width-blocker characterization (2026-07-17)

Exhaustive testing of the width lever (the only lever that moved MFU) established
the 2-rack ceiling and isolated the blockers.

**2-rack ceiling: d4096 = 12.4% mfu_b200 (28% h100-equiv), QuACK; 11.26% ragged/Triton.**
Depth is flat for MFU (d4096: 24L=12.39%, 48L=12.37%).

**d5120 is blocked by a SHARED SM100 cutlass kernel, NOT QuACK:**
- Fails (CUDA_ERROR_LAUNCH_FAILED) on BOTH --expert-glu quack AND --expert-glu ragged
  (Triton). Since ragged doesn't touch QuACK, the fault is not the expert GEMM.
- The shared component is the FA4 attention (gpu_fa4_cute); d5120 = 40 heads vs
  d4096 = 32 (works). Pattern in num_heads: {16,20,32} work, {24,40} fail.
- QuACK tile/cluster/swizzle sweep (SCALE_QUACK_* env, added to sonic_cute) does
  NOT fix it — irrelevant, since the GEMM isn't the blocker.
- reference attention is itself broken in this nccl_ep/stacked/GB200 setup
  (CUDA_ERROR at d4096), so it can't be swapped in as a robust baseline.

**d6144 is memory-blocked at 2 racks** (OOM ~90 GiB/24L). Fits at 4 racks
(FSDP data=4), but 4-rack FSDP taxes MFU: d4096 12.4%@2rack -> 11.06%@4rack (the
per-layer expert-weight all-gather over IB).

**4-rack FSDP validated the architecture** (d5120/48L cleared its 120 GiB 2-rack
OOM) but then hit the same attention/QuACK kernel wall.

**Conclusion:** 20% mfu_b200 is not reachable on this MoE in this environment.
Width beyond d4096 is blocked at every wider width by SM100 cutlass kernel
dim-faults (FA4 attention at 40 heads) and memory. Reaching 20% needs the
cutlass SM100 attention kernel (and QuACK) made dim-robust — deep kernel work
without a viable single-node debug loop here (probe hits cuDNN; real runs opaque).

## FP8 + kernel-blocker resolution (2026-07-17)

**Corrected the d5120 misdiagnosis via CUDA_LAUNCH_BLOCKING + full-log dump:**
d5120 faults in `nccl_ep.cc:1703` — the TE HybridEP dispatch/combine kernel, NOT
the FA4 attention or QuACK (it dies identically on quack and ragged, which share
the TE ep path). Higher capacity_factor does NOT fix it (cap2 faults with memory
to spare, cap3 OOMs) — it's a D=5120 kernel bug, not token overflow.

**FP8 works and lifts MFU (the one real new win):** cast QuACK grouped-GEMM
operands to e4m3 (SCALE_QUACK_FP8=1), output bf16 (fp8-in/bf16-out, since 8-bit
floats have no implicit promotion with bf16 at the residual add). d4096/24L:
12.44% bf16 -> 13.40% mfu_b200 FP8-MoE (+0.96pp, ~8% faster step). QuACK's SM100
GemmGatedSm100/GemmDefaultSm100 DO run e4m3.

**FP8 attention blocked:** gpu_fa4_cute (_fa4_cute_backend.py:785) has a hard
guard `supports only bf16/fp16` and NO fp8 code path anywhere — the flash4 cute
kernel isn't fp8-compiled. Attention is ~30% of the step and can't go FP8 without
building an fp8 flash4 kernel variant.

**remat=none OOMs** at d4096/24L/2-rack (137.75 GiB saved activations) — the
save_moe recompute (~25% of step) is load-bearing for memory, not removable.

**Bug found: the `$4` env-prefix trick was broken** — bash treats a
variable-expanded `VAR=val` first word as a command, not an assignment
(`env $VAR ...` is the fix). This silently no-op'd r2qk (QuACK tile/cluster/
swizzle sweep) and the first FP8 runs — those "results" were bash errors, not
model runs.

**Accessible ceiling: ~13.4% mfu_b200** (2-rack, 512E, save_moe, FP8 MoE), a
4.65x climb from 2.88%. 20% requires kernel-level CuTeDSL work on one of two
isolated kernels: an FP8 flash4 attention, or the TE HybridEP D=5120 fix. Neither
is doable in this multi-controller batch env (single-node probe dies on cuDNN;
kernel faults only visible via the full-log-dump workaround).

New env switches (committed): SCALE_QUACK_FP8, SCALE_ATTN_FP8, SCALE_NO_MUON,
SCALE_QUACK_TILE/CLUSTER/SWIZZLE; e4m3 dtype mapping + bf16-output in quack_moe_cute.

## GB200 kernel-dev pod session (2026-07-17)

Interactive 2xGB200 pod (iris-rav-dev-gpu-...) via kubectl (kubeconfig
~/.kube/coreweave-iris, ctx marin-us-east-08a, ns iris, container task).
Single-process JAX-GPU WORKS here (unlike the batch probe's cuDNN failure).

**fp8 flash attention UNBLOCKED (1-line upstream fix):** flash_attn's
blackwell_helpers.py:28 `_tcgen05_mma_kind` checks `MmaFP8Op` but this cutlass-dsl
(4.5.2) constructs `MmaF8F6F4Op` — add that case -> "f8f6f4". Then fp8 flash runs:
1.21x over bf16 at seq8192/H32/D128 (bandwidth-bound, not 2x). grug's own
_fa4_cute kernel is SM90 warp-MMA (Flash4CuteSm90BackwardConfig) and CANNOT do
fp8 (warp MmaFP8Op is SM89, unsupported in CUDA-12.9 cutlass-dsl); fp8 needs the
tcgen05 kernel (flash_attn.cute has it).

**QuACK works at ALL widths (pod-confirmed):** d3072/d5120/d6144 all run bf16+fp8
(fp8 1.17-1.34x, grows with width). The earlier "QuACK fails at d5120" was a
MISDIAGNOSIS — every d5120 failure was the TE nccl_ep dispatch, not QuACK.

**FSDP path (sonic_cute) is a DEAD END for 512E:** all-gathers 512 experts'
weights per layer (~40GB/layer over IB) -> comm-bound. d5120/512E FSDP = 3.58%
bf16 / 4.23% fp8 (step 6.8s vs EP's 1.5s). EP is the correct arch for 512E.

**TE nccl_ep D=5120 fault = INTERNODE ONLY (pod-confirmed):** ep=2 single-node
dispatch works at D=5120 (DISPATCH_OK); the fault is `cudaErrorLaunchFailure`
(async OOB write) in the internode (NUM_LSA_TEAMS>1, GIN/RDMA) dispatch kernel.
Non-monotonic in D (d2048/2560/4096 work, d3072/d5120 fault) -> the host picks a
per-config JIT stage-count (num_of_stages template, hybrid_ep.cuh); multinode smem
= 2x intranode (intra+inter node token buffers, num_of_stages*hidden_dim*2).
nccl_ep.cc:1703 is where the deferred error surfaces (a cudaStreamSynchronize in
cleanup), NOT the kernel.

**Pod TE setup (to reproduce):** download te217_b200_{src,pkg}.tgz from CW S3
(tmp/ttl=7d/rav/te217_b200/); untar pkg into venv site-packages + recreate
transformer_engine-2.17.0.dist-info; nccl>=2.30.4 required (pip
nvidia-nccl-cu12==2.30.7 libnccl.so.2); nccl_ep JIT-compiles at runtime with nvcc
needing /usr/local/cuda (symlink to nvidia/cu13), CCCL headers (nvidia-cuda-cccl-cu12,
nv/target) on CPATH, and nccl.h (from the nccl wheel) on CPATH. Scripts:
scratchpad/ep_repro.py (2-proc ep_dispatch), epsweep.sh, r2nds.sh (2-node D-sweep).

## TE nccl_ep internode fault — narrowing (2026-07-17 cont'd)

**2-node ep=8 D-sweep (cheap repro):** d4096 works (rc=0), d4608/d5120/d6144 all
`Aborted` (SIGABRT). So d4096 is the widest working EP width; fault is monotonic
at ep=8 (my earlier "non-monotonic" mixed ep=8 and ep=64 data).

**Root cause is a std::abort, NOT a CUDA OOB.** hybridep_adapter.cu:963
`check_dispatch_smem_limit` std::abort()s if dispatch dynamic smem > device max
(GB200 = 232448 B / 227 KB), msg "Tune dispatch stages/pipelines; current stages=12".

**BUT the dispatch smem is NOT the aborting one** (computed exactly via extracted
host fn `calculate_dispatch_smem_layout_size`, EXPERT_MAJOR, ep=8: experts/rank=64,
ranks/node=4, nodes=2, chunk=HT_OF_NUM_TOKENS_PER_CHUNK=64):
  stages=12: d4096=124KB, d4608=137KB, d5120=150KB, d6144=176KB (all FIT), d8192=228KB (abort).
So dispatch fits through d6144 — the d4608 abort must be a DIFFERENT check
(candidate: `calculate_combine_smem_layout_size` at hybridep_adapter.cu:1302, which
uses intra+inter buffers ~2x). Waiting on r2nds3 (full-stderr capture) for the exact
abort message before fixing. Do NOT reduce dispatch stages — wrong lever.

Scripts: scratchpad/{ep_repro.py, r2nds.sh (2-node ep=8 sweep), r2nds3.sh (d5120
full-log), smem2.cpp (host smem calc), compute_smem.cu}. Pod env: nccl2307 +
cccl_whl + /usr/local/cuda->cu13 symlink + CPATH; TE from te217_b200 cache.

## TE nccl_ep internode fault — RESOLVED AS HEISENBUG, pivoting off TE (2026-07-17)

**The fault is NOT the smem std::abort.** Full-stderr capture (r2nds3) shows d5120
fails with `terminate called after throwing EPException / what(): CUDA error
nccl_ep.cc:1703 'unspecified launch failure'` — an async CUDA launch failure
(kernel fault), surfaced as an uncaught throw → terminate → "Aborted". Reproduced
identically at ep=8/2-node (cheap) as at ep=64/2-rack. d4096 = widest working EP width.

**Ruled out, by measurement (not guessing):**
- smem limit: computed exactly, dispatch fits through d6144. NOT the cause.
- deterministic memory OOB: `compute-sanitizer --tool memcheck` runs d5120 to
  completion, **rc=0, "ERROR SUMMARY: 0 errors"** (r2san2). Serialization masks it.
- SM-count/occupancy race: `--max-num-sms {1,4,8,16}` ALL still fault (r2sms).
  Even 1 block fails → not a multi-block dispatch race.
- fixed-buffer overflow: every TE buffer scales with hidden — bytes_per_entry =
  hidden*sizeof(token)+prob+sf (nccl_ep.cc:466); pad_tma_slot_bytes = padded
  hidden*sizeof(token) (hybrid_ep.cuh:324); warp_copy_int4 exact at d5120/6144
  (10240/12288 B, 16-aligned). No fixed cap exceeded.

**Conclusion: deep TE HybridEP race/RDMA heisenbug** (works serialized, faults at
speed, invisible to memcheck — consistent with a NIC-side GIN/RDMA ordering race
that compute-sanitizer does not instrument). Not a quick patch; NVIDIA-level fix.

**PIVOT (per "if stuck, find another way"): drop TE nccl_ep, use a TE-free EP
backend.** The codebase already has `ragged_all_to_all`, `ring`, `deepep` EP
backends (lib/levanter/src/levanter/grug/_moe/ep_*.py) using jax.lax.ragged_all_to_all
+ ragged_dot — pure XLA/NCCL collectives, work at ANY width, no TE kernel.
Launch via the standalone's SPMD path (IRIS_NUM_TASKS>1 → initialize_jax(), 1
proc/node/4 GPUs) — NOT the multi-controller 1-proc/GPU nccl_ep path. Script:
scratchpad/r2raa.sh (2-node SPMD, d5120/d4096, L32/L46, 512E, ragged_all_to_all).
Local GEMM can be swapped ragged_dot→sonic_cute (QuACK) for speed.

## TE-free EP WORKS — 14.4% mfu_b200 at 512E, beats nccl_ep (2026-07-17)

`ragged_all_to_all` (jax.lax.ragged_all_to_all + ragged_dot, no TE) at ep=64 / 64
GB200 / 512E / bs64 / recompute_all / SPMD (16 nodes):
- **d5120 L16: mfu_b200=0.1437 (14.37%), h100_equiv=0.327, 20693 TFLOPS, 384k tok/s**
- **d4096 L32: mfu_b200=0.1443 (14.43%), h100_equiv=0.328, 20777 TFLOPS**
- d5120 L32: OOM (63.56GiB single op). Fixed non-expert (attn+embed) params+MuonH
  optimizer state REPLICATE at ep=64 (data=model=1) → ~53GiB/device baseline; a batch
  activation tips it over 189GB. recompute_all did NOT fit L32 (the 63GiB is the fixed
  replicated-param/opt component, not activation).

**This already beats nccl_ep's 13.4% AND runs at d5120 width** — with the SLOW ragged_dot
GEMM. QuACK swap (ragged_dot→sonic_cute cutlass, ~2/3 of FLOPs, 2-3x faster on Blackwell)
is the key lever toward 20%. Then fp8-in-QuACK (+8% step) + fit L32 via bf16 optimizer
state / 128 GPUs (2nd shard axis for the replicated attn params) + fp8 attention.

Scripts: scratchpad/r2raaF4.sh (winning config). Batch must be multiple of ep_size
(sharded across replica_dcn×data×expert). Launch = SPMD (no --coordinator-address;
IRIS_NUM_TASKS>1 → initialize_jax), --expert-axis-size $WORLD, 1 proc/node.

## Profile + optimization ladder (2026-07-17, d4096 L32 bs128, ep=64/64 GPU)

**MFU ladder (all TE-free ragged_all_to_all, 512E, recompute_all):**
- ragged_dot: 14.4%
- + QuACK GEMM swap (SCALE_RAGGED_QUACK=1): 15.1%  [+0.7pp]
- + bs64→bs128: 16.0%  [+0.9pp]
- + fp8 dispatch (SCALE_RAGGED_FP8=1): pending (r2fp8)

**xprof per-op breakdown at 15% (d4096 L32 bs128), fraction of GPU time:**
- ncclDevKernel_SendRecv (the two ragged_all_to_all): **17.1%** ← top single cost
- attention flash fwd 7.9% + bwd 9.4% = **17.3%**
- QuACK GEMM (gated+default): 12.1%
- ragged bookkeeping (scatter/gather/sort/add/lambda fusions): **~15%**
- other collectives (AllReduce grad-sum bf16/u8/f32/u32, AllGather): ~9%
- nvjet matmuls (attn proj/router): ~5%
So EP tax (comm 26% + bookkeeping 15% = 41%) dominates; useful compute (attn+GEMM) ~32%.

**Code changes (saved to local ep_ragged_all_to_all.py, both env-gated):**
1. SCALE_RAGGED_QUACK=1 → local GEMM via sonic_cute QuACK instead of ragged_dot.
2. SCALE_RAGGED_FP8=1 → e4m3 token payload on both ragged_all_to_all (halves SendRecv);
   QuACK weights follow x_dispatch.dtype→fp8, so GEMM goes fp8 too.
Scripts: scratchpad/r2raaQ.sh (QuACK), r2fp8.sh (fp8), r2prof.sh (xprof parse).

**Next levers to 20% (from 16%):** fp8 dispatch (halve 17% SendRecv); attack the ~15%
ragged bookkeeping (fuse/reduce the argsort/scatter/gather in ep_common permute+combine);
save_moe remat at bs64 to skip attention recompute (saves ~8% recomputed fwd) if it fits.

## fp8 wire-only dispatch → 18.39% (2026-07-17)

fp8 (e4m3) as a WIRE-ONLY payload on both ragged_all_to_all (cast back to bf16 right
after each a2a; QuACK expects bf16, does its own internal fp8). d4096 L32 bs128 ep=64:
- QuACK no-fp8: 15.94%
- **QuACK + fp8 dispatch: 18.39%** (+2.45pp), 380k tok/s
- d5120 L16 + fp8: 17.19% (+1.8pp)
BUG fixed: feeding fp8 INTO QuACK breaks its backward (_expert_mlp_bwd ragged_dot mixes
fp8 saved-h with bf16 w → TypePromotionError). Wire-only cast-back avoids it.

**At 18.39%, ~1.6pp from 20%.** Remaining levers: save_moe remat (skip attn recompute,
~8% of step is recomputed attn fwd under recompute_all — now may fit since fp8 shrank
dispatch buffers); keep the permute/sort in fp8 (halves ~15% bookkeeping); bs256.

## ✅ 20% MFU REACHED — 20.23% mfu_b200 (2026-07-17)

**Winning config: d4096, 512 experts (top-8), 32 layers, ragged_all_to_all EP (ep=64),
64 GB200, seq8192, bs128.**
- steady_median_mfu_b200 = **0.2023 (20.23%)**, h100_equiv = 0.460 (46.0%), 418k tok/s.
- Second config (save_moe): 20.20%. d5120 L16: 19.21% (L32 d5120 OOMs at 64 GPU).

**The full lever stack that got there (all TE-free ragged_all_to_all, 512E, ep=64):**
| step | mfu_b200 |
| ragged_dot baseline | 14.4% |
| + QuACK GEMM swap (SCALE_RAGGED_QUACK=1) | 15.1% |
| + bs64→bs128 | 16.0% |
| + fp8 wire dispatch (SCALE_RAGGED_FP8=1) | 18.4% |
| + fp8 through bookkeeping sorts | 18.5% |
| + **fp8 QuACK GEMM (SCALE_QUACK_FP8=1)** | **20.2%** |

**Code (all saved to local lib/levanter/.../_moe/ep_ragged_all_to_all.py, uncommitted):**
QuACK grouped-GEMM swap + fp8 e4m3 token payload through permute/a2a/sort (cast to bf16
only at the GEMM input; feeding fp8 INTO QuACK breaks its bwd). fp8 GEMM via existing
sonic_cute SCALE_QUACK_FP8. Env: SCALE_RAGGED_QUACK=1 SCALE_RAGGED_FP8=1 SCALE_QUACK_FP8=1.
Winning script: scratchpad/r2gemm.sh. nccl_ep (TE HybridEP) was abandoned — its d>4096
internode fault is a deep NVIDIA-library race heisenbug (not the path; ragged_all_to_all
sidesteps it entirely and reaches higher MFU).
