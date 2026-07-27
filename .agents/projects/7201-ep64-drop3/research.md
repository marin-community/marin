# EP64 HybridEP performance prior art

> The current operating state, results after v137, exact submit command, profile
> procedure, and takeover instructions are in [HANDOFF.md](HANDOFF.md). This
> file remains the detailed prior-art and early-experiment ledger.

## Background Research Brief

- Effort: high
- Stop rule: stop when the four requested primary sources no longer change the
  first three profile-driven hypotheses for the locked EP64/GB200 run.
- Date: 2026-07-26

### Current Target

- Reach at least 25% steady BF16 MFU on 64 GB200 GPUs; continue toward 30% after
  the first gate.
- Keep the shared-expert intermediate width at the original 5,120.
- Keep exact aggregate post-ECHO assignment drop at or below 3%.
- Qualify throughput and drop after QB has settled; early-step MFU is not
  decision-grade.

The 21,504-wide shared-expert series is retained as a separate architecture
option. Its padding-2 run
`marin-community/rav_moe/ep64-d5120-sh21504-pad2-qb200-v119-20260726-1853`
reached 25.5% tail MFU with about 2% exact drop before the target was revised.
That result cannot support a kernel-only claim because the always-on shared
expert is 4.2 times wider than the original configuration.

The original-capacity baseline is
`marin-community/rav_moe/ep64-d5120-sh5120-pad2-qb200-v128-20260726-1953`.
It keeps the receiver-ECHO kernel stack, top-4 routing, routed intermediate
2,048, padding-2 token envelope, QB, and exact drop logging, changing only the
shared intermediate to 5,120. The matched one-node profile is
`/rav/ep64-d5120-sh5120-pad2-qb-profile150-v129-20260726-1953`; it profiles
steps 150-153 with HLO proto enabled and CUDA graphs disabled.

### Question

Which techniques from the requested paper and EP implementations can move the
locked Grug MoE configuration from the stable HybridEP checkpoint beyond 30%
MFU while preserving receiver-side capacity semantics and at most 3% physical
token drop?

### Current Marin Context

The locked run is L48, d5120, E256, top-8, EP64, batch 1024, sequence length
4096, MuonH, BF16, and GB200. Marin currently calls HybridEP's fused
dispatch-permute and combine-unpermute through a JAX FFI. XLA requires a static
receive envelope. HybridEP writes only valid and expert-alignment rows, but the
Sonic CUTE expert MLP receives a group-size vector whose final group absorbs the
unused envelope.

Undefined rows made the weight gradients non-finite. Commit
`df28b3e2e49c8397cc54c099c39118566db25c7f` fixes correctness by masking the
unused rows before expert compute. Its 30-step EP64 run completed with 10.55%
p50 MFU, zero final overflow, and finite loss. Moving full-envelope zeroing into
the FFI reached 10.84%; passing device-resident group sizes to the expert GEMM
reached 10.86%. Removing the FFI's handle-token stream synchronizations did not
help: the 12-step A/B reached 10.78%.

The one-node HybridEP profile attributes about half the step to token transport.
It also shows that encoding exact ECHO destinations as synthetic HybridEP
experts expands each rank's top-8 route into a dense 1,024-column routing map.
A receiver-semantic ECHO path now packs those destinations into a static native
`all_to_all` envelope instead. Its two-step EP64 numerical gate passed all 54
parameter, gradient, and update checks. The stable top-8 treatment reached
17.02% p50 MFU with Sonic dispatch/combine, physically dropping 1.47% of
assignments on average and 2.50% in the worst layer. Switching to top-4 and a
512-token sliding window reached 18.82% p50 MFU with 0.49% mean and 1.26%
worst-layer physical drop.

The one-node receiver-semantic profile shows a different bottleneck from
HybridEP: NCCL send/recv accounts for 26.0% of exclusive device time, and all
communication for 32.2%. The reference dispatch scatter accounts for another
4.6%. Sonic dispatch removes most of the latter cost, but the three fixed
token/slot/return exchanges remain the primary ceiling.

The same profile also attributes about 10.5% of device time to generic gather
adjoints in dispatch and combine. Replacing both with the existing Sonic slot
gather raised the short-run p50 from 18.82% to 20.67% without changing routing
or clipping. The checkpoint-disabled 30-step qualification completed at 20.51%
p50 MFU (20.33/20.66% p10/p90) with finite falling loss and final 1.28% mean /
1.94% worst-layer physical drop.

Widening the routed intermediate from 1280 to 2048 and the shared intermediate
from 5120 to 6144 raised a 12-step qualification to 20.99% p50 MFU and 380.4k
tokens/s. Loss remained finite and declined to 8.33. Final physical overflow
was 1.98% mean / 2.97% worst layer; the largest transient worst-layer sample
was 3.41%.

### External Prior Art

#### NVIDIA Blackwell MoE training report

The report's GB200 recipe uses HybridEP, fused router/permutation, a
device-initiated grouped GEMM, paged temporary storage, and MXFP8. Its
device-initiated grouped GEMM reads token counts on the GPU, uses a static
launch shape, and skips empty/padded rows. The report also says EP overlap was
not helpful in its best GB200 configuration and that CPU/launch overhead
becomes important after NVL72 removes much of the communication bottleneck.
MXFP8 raises the reported GB200 throughput from 857 to 1048 TFLOP/s/GPU, but
changes numerical format and therefore is not the first experiment for this
BF16 correctness thread.

#### NVIDIA Megatron-LM PR 2368

The patch implements ECHO expert cloning and a device-initiated grouped GEMM.
It keeps static communication capacity without a device-to-host count sync,
then gives the expert GEMM device-side token counts so it can avoid processing
padding. Its ECHO planner uses spare expert slots only for overflow and
recomputes expert dispatch during backward. The patch is draft-quality and
contains experimental scaffolding, so it is evidence for mechanisms rather
than code to transplant wholesale.

#### DeepEP main and HybridEP branch

DeepEP V2's asynchronous path allocates a worst-case output but returns both
prefix sums and unaligned per-expert counts on the GPU. `do_zero_padding` zeros
only the alignment gaps in the dispatch epilogue. It does not require a
full-envelope memset.

HybridEP's fused dispatch-permute follows the same principle. The metadata scan
produces `tokens_per_expert` and a dense-to-expert map. Permute blocks copy only
valid routed rows and explicitly zero only per-expert alignment rows. The
non-blocking API documents that the unused output tail is garbage and must be
ignored using `tokens_per_expert`. Dispatch and combine are warp-specialized,
persistent TMA pipelines; chunk-ready flags let permute/unpermute blocks consume
data as communication produces it.

The branch defaults to 24 communication SMs for intranode dispatch and combine,
supports configurable 64/128-token chunks, and offers fused
dispatch-permute/combine-unpermute. Its own GB200 measurements distinguish
Torch API time from pure kernel time, which is important because the API gap is
large.

#### Loong-Megatron PR 7

The patch uses two parity-indexed token transport buffers so one-layer rank
drift cannot reuse a live buffer, and separate FC1/FC2 expert-dispatch buffers
for backward. CUDA events guard reuse. It overlaps each cloned-weight dispatch
with the corresponding home-expert GEMM, but deliberately launches expert
weight exchange after token all-to-all to avoid HBM/network contention. Home
and cloned expert GEMMs receive disjoint device count vectors, including
zero-count experts, so empty groups are skipped. The PR was closed and includes
heavy synchronization fallbacks, so its ordering rules need profile validation
in JAX before use.

### Negative / Failed Leads

- Broad communication/compute overlap is not the default next step. The
  Blackwell report's best GB200 configuration disables EP overlap, and
  Loong-Megatron warns that overlapping token and expert-weight exchange can
  reduce throughput through shared HBM/network contention.
- Full-envelope zeroing is a correctness fallback, not a performance design.
  DeepEP and HybridEP zero only alignment rows and rely on device counts to
  delimit valid work.
- CUDA graphs are useful in the paper, but the requested profile must disable
  them, and graphs cannot explain the current low GPU work efficiency before
  the static-envelope cost is measured.
- MXFP8 is a plausible later ceiling increase, not a clean first A/B: it changes
  arithmetic and requires a separate loss/stability qualification.
- ECHO-style cloned experts help only if per-rank expert imbalance is visible
  after the current same-expert routing. Implementing the planner without that
  measurement adds weight traffic and backward complexity without a falsifiable
  expected win.
- Sender-local fixed-capacity bucketing is not interchangeable with the desired
  receiver pool semantics. The requested <=3% physical drop constraint rules
  out claiming a speedup that silently changes where capacity is enforced.
- Full-tree numerical instrumentation is not a neutral large-model fit probe.
  It constructs finite masks for every parameter, gradient, and update inside
  the compiled step. A clean D6144/L32 run without that instrumentation still
  failed on a real 79.04-GiB allocation after reaching training-step
  compilation. D6144 therefore exceeds the default JAX device-memory pool;
  optimizer-state offload and a larger pool are separate live fit probes.
- Rowwise E4M3 token transport is not a throughput lever in its current form.
  Its EP64 run completed 12 steps at 20.49% p50 MFU, essentially equal to the
  BF16 path, because rowwise quantize/dequantize and the scale collective cancel
  the smaller payload. It also changes forward numerics and produced transient
  worst-layer overflow above the 3% target.
- Concatenating the two clone-weight exchanges into one collective is also not
  a throughput lever. The 12-step A/B regressed from 20.51% to 20.21% p50 MFU;
  the larger concatenated materialization costs more than the saved collective.
- The profile's generic expert weight-gradient kernels were actionable. Enabling
  the existing Blackwell variable-K QuACK weight-gradient path raised the wide
  configuration from 20.99% to 21.34% p50 MFU over 12 steps without changing
  routing semantics. Its final physical token drop was 2.04% mean and 3.14% in
  the worst layer, so a 30-step qualification is still required.
- Tightening the sparse clone-weight receiver envelope from 16 to 10 segments
  independently raised p50 from 20.99% to 21.23%. Final physical drop remained
  within the bound at 1.88% mean and 2.60% in the worst layer.
- The 30-step grouped-wgrad qualification finished at 21.28% p50 MFU with a
  finite falling loss and final physical drop of 1.31% mean / 1.92% max.
- Combining grouped wgrad with the ten-segment receiver envelope reached 21.52%
  p50 over 12 steps. Final physical drop was 1.95% mean / 2.976% max, inside
  the bound with only 0.024 percentage points of worst-layer headroom. A
  30-step qualification is running.
- Reducing token padding from four experts to three was throughput-neutral
  (21.46% p50) and exceeded the bound at 3.039% max final physical drop. Reject
  the three-slot envelope.
- Increasing expert intermediate width from 2048 to 3072 does not fit at either
  the default or 95% JAX pool. The 95% probe reached the first executable, then
  NCCL allocations failed across the rack; a 90% pool probe is running to leave
  more non-JAX headroom.
- Increasing the routed intermediate to 2560 while reducing the shared
  intermediate to 4096 keeps active MLP FLOPs unchanged but still does not fit:
  XLA requests an 80.34 GiB training executable. Persistent routed-expert
  parameters, rather than active width alone, set this boundary.
- D6144/L32 still does not fit after optimizer offload, 512 GiB host RAM, and a
  95% JAX pool. It reaches `jit_train_step`, then NCCL reports CUDA OOM across
  the rack. Host RAM fixed the earlier exit-137 ambiguity but not the device
  fit boundary.

### Evidence Map

#### Claim: static JAX shapes do not require computing or zeroing the full envelope

- Support:
  - DeepEP V2: asynchronous dispatch allocates worst-case output while retaining
    device-side per-expert prefix sums and unaligned counts.
  - HybridEP: non-blocking fused permute writes valid rows and alignment padding;
    its API explicitly treats the unused tail as undefined.
  - Megatron PR 2368 and the Blackwell report: device-initiated grouped GEMM
    consumes GPU counts under a static launch.
- Contradictions:
  - Marin's current Sonic CUTE wrapper expresses the unused suffix as work for
    the final group, so its present interface does require defined rows.
- Directness to Marin: high; this is the exact correctness/performance fault in
  the current EP64 run.
- Confidence: high.
- Action: profile the stable full-zero path, then change expert compute to use
  device counts or a valid-row predicate without materializing the entire
  envelope.

#### Claim: fuse initialization with dispatch/permute, not a separate 5.4 GB memset

- Support:
  - HybridEP's fused permute S2G warps initialize only the alignment padding
    they own while valid tokens are copied.
  - DeepEP V2 performs optional zero padding in the dispatch copy epilogue.
- Contradictions:
  - A suffix-only kernel is simpler and could already be bandwidth-cheap if the
    suffix is small; the profile must quantify it before deeper fusion.
- Directness to Marin: high.
- Confidence: high on correctness, profile-dependent on MFU impact.
- Action: measure `cudaMemsetAsync`; if material, either delete it by teaching
  GEMM to skip invalid rows or replace it with per-expert padding initialization
  inside the FFI dispatch path.

#### Claim: buffer parity and stream overlap are second-order unless a wait gap appears

- Support:
  - Loong-Megatron uses parity buffers and CUDA events to tolerate rank drift and
    prevent operation mismatch.
  - HybridEP uses monotonically increasing chunk flags and persistent
    producer/consumer pipelines inside a single fused kernel.
- Contradictions:
  - The current Marin run has already fixed dispatch handle lifetime and has not
    shown a buffer-reuse deadlock.
- Directness to Marin: medium.
- Confidence: exploratory.
- Action: inspect XProf for host gaps, stream serialization, flag polling, or
  cross-rank tail latency before adding buffers or streams.

### Recommended Next Experiments

#### 1. Profile the steady receiver-semantic native A2A path

- Result: the stable top-8 path reached 17.02% p50 MFU under the 3% drop bound.
  A one-node XProf with HLO proto enabled and CUDA graphs disabled attributes
  32.2% of exclusive device time to communication, including 26.0% to NCCL
  send/recv. The reference dispatch scatter contributes 4.6%.
- Consequence: prioritize reducing or overlapping the fixed token exchanges.
  Local dispatch/combine gather kernels remain useful but have a lower ceiling.
- Artifact:
  `s3://marin-us-east-02a/tmp/ttl=30d/xprof/ep64-echo-fixed-a2a-sonic-profile1n-v40-20260726-1223`.

#### 2. Device-count-aware expert GEMM with undefined static tail — completed

- Result: the two-step numerical gate passed all 54 checks. The 30-step run
  reached 10.86% p50 MFU versus 10.84% for FFI full zeroing, so padded expert
  work was not the primary bottleneck.
- Consequence: retain the device-count interface as the correct static-envelope
  design, but do not spend further rack time optimizing tail initialization.
- Sources: Blackwell report, Megatron PR 2368, DeepEP V2.

#### 3. Fused alignment-padding initialization — demoted

- Result: eliminating both full-envelope initialization and padded expert work
  changed p50 by only about 0.03 percentage points.
- Consequence: the profile falsifies padding initialization as a material MFU
  lever for this configuration.
- Sources: DeepEP V2 and HybridEP fused permute.

#### 4. Communication SM/chunk sweep after removing tail work

- Minimum experiment: sweep communication SMs around 16/24/32 and chunk size
  64/128 on a short stable EP64 run.
- Baseline/control: exact-count GEMM at current values.
- Expected signal: a clear API-level optimum; pure kernel bandwidth alone is
  insufficient.
- Falsifier: MFU changes less than 0.5 percentage points.
- Cost/risk: low implementation risk, several rack runs.
- Sources: HybridEP GB200 benchmark and configuration guide.

#### 5. ECHO and FP8 only after the BF16 path is efficient

- Result: a rowwise E4M3 transport prototype compresses forward dispatch and
  combine while retaining BF16 cotangent exchanges. Its two-step four-GPU smoke
  was finite, but the 12-step EP64 run reached only 20.49% p50 MFU and changed
  the loss trajectory. This implementation is rejected as the next lever.
- A future low-precision treatment would need fused quantization inside the
  transport kernel, then a matched-seed long loss qualification.
- Baseline/control: efficient BF16 receiver-count path.
- Expected signal: ECHO reduces straggler tail without exceeding 3% physical
  drop; low precision raises tensor-core throughput without unstable loss.
- Falsifier: balanced rank loads, clone traffic dominates, or loss diverges.
- Cost/risk: high numerical and implementation risk.
- Sources: Blackwell report, Megatron PR 2368, Loong-Megatron PR 7.

### Hypothesis Queue Update

- Add: native fixed-envelope transport for exact ECHO destinations; profile its
  sparse clone-weight exchange and fixed token A2A separately.
- Add: remove the receiver-slot metadata exchange by deriving the local
  permutation from globally known sender/expert counts, or fuse that metadata
  into the token transport.
- Add: prototype a chunked round-robin `ppermute` transport only if it pipelines
  peer arrivals with useful expert work; a serial decomposition of the same A2A
  does not address the profile.
- Revise: device-count-aware grouped GEMM is a correctness improvement but not a
  throughput lever here; treat parity buffers and round-robin overlap as
  profile-triggered remedies.
- Promote: use the existing QuACK grouped expert weight-gradient kernel; the
  isolated A/B gained 0.35 MFU points and the 30-step qualification sustained
  21.28% p50 under the drop bound.
- Promote with guardrail: combine grouped wgrad and ten receiver segments;
  21.52% is the current best, but the worst-layer drop is 2.976%, so retain
  four token-padding experts. The 30-step qualification sustained 21.39% p50
  MFU with final 0.93% mean / 1.78% max physical drop.
- Promote: optimize the fixed BF16 token transport before clone weights. The
  current-best one-node XProf attributes about 15.83 device-seconds to the
  fixed dispatch/combine A2As versus 2.59 device-seconds to sparse clone-weight
  ragged A2A.
- Reject: a serial 64-round `ppermute` decomposition of each fixed token A2A.
  It compiled, but its first `jit_train_step` failed on every rank when NCCL
  could not allocate GPU memory. A useful round-robin implementation must bound
  live transport state and pipeline peer arrivals with expert work.
- Reject in its current unfused form: the exact MNNVL peer-write transport on
  the current top-4, grouped-wgrad configuration. It retained receiver capacity
  semantics and finished 12/12 steps, but regressed from 21.39% to 20.36% p50
  MFU. Direct remote writes followed by a barrier and copy into an XLA-owned
  buffer do not beat the native collective; a future MNNVL path must expose
  chunk readiness directly to useful expert work.
- Reject at full-layer granularity: `save_moe` rematerialization on the exact
  current baseline. It attempted to retain the fixed transport residuals for
  all 48 layers and requested 374.32 GiB per GPU. Selective or compressed
  residuals remain a design lead, but the existing policy cannot fit.
- Blocked by a code bug, not infrastructure: scan unroll 2. All 64 process IDs
  connected exactly once, then tracing failed because scan's unroll rewrite
  built a `dynamic_update_slice` with incompatible explicit shardings
  (`int32[24,2]` versus `int32[1,2]`). Later connection errors were teardown.
- Reject: three token-padding experts; its 3.039% worst-layer drop exceeds the
  physical-drop bound without improving throughput.
- Reject: `intermediate_dim=2560`; the training executable requests 80.43 GiB
  with shared intermediate 6144, and 80.34 GiB even after reducing the shared
  intermediate to 4096. Neither variant fits.
- Falsify / stop: fusing both clone-weight exchanges through a single
  concatenated materialization.
- Falsify / stop: full-envelope initialization, padded expert work, sync-free
  handle tokens, and unmeasured blanket EP overlap as primary MFU levers.
- Promote: preserve receiver-side drop semantics while replacing the dense
  synthetic HybridEP route with a compact static transport layout.
- Promote pending 30-step qualification: scale the shared dense expert to
  intermediate width 21504 under the CUDA async allocator. The 12-step EP64
  probe reached 25.21% p50 MFU with 0.25% mean / 0.71% max physical drop.
- Reject: tighten the sparse receiver envelope from ten to eight segments. It
  regressed to 24.30% p50 and raised the worst-layer physical drop to 3.50%.
  Six segments additionally triggered illegal GPU memory accesses in the first
  training execution, so the ten-segment envelope remains the qualified value.
- Reject for this installed XLA: `xla_gpu_enable_custom_fusions` and
  `xla_gpu_enable_address_computation_fusion`; both flags are unknown and the
  process exits before distributed initialization.
- Promote: optimize or overlap the fixed BF16 token transport at the original
  shared width. The matched steady-QB profile attributes 30.13% of aggregate
  GPU kernel time to communication and 23.20% to NCCL SendRecv. The six
  `[64,4608,5120]` fixed token A2As account for 7.407 device-seconds across the
  four-GPU, four-step capture, about 78% of SendRecv time.
- Revise: the communication-free ceiling of the current 21.05% MFU control is
  only about 30.1%. Exceeding 30% therefore requires removing or hiding nearly
  all fixed token A2A time and improving at least one local compute path; a
  transport-only win is not enough margin.

### Source Ledger

| Source | Type | Location | Claim used for | Confidence | Notes |
|---|---|---|---|---|---|
| NVIDIA Blackwell MoE report, arXiv v2 | paper | `scratch/prior-art-20260726/arxiv-2603.07685/` | device-initiated GEMM, HybridEP, GB200 recipe, FP8 ceiling | high | 88-page PDF and source downloaded; PDF text read end to end |
| NVIDIA Megatron-LM PR 2368, head `a2b16b8733bb4ced46880d6adee6f19124732991` | PR / external code | `scratch/prior-art-20260726/megatron-pr2368/` | ECHO, GPU planner, device-count grouped GEMM | medium-high | full 8,587-line patch read; draft patch has rough edges |
| DeepEP main `dd758caf451848bd150e1046af3d0a73e5fff38d` | external code | `scratch/prior-art-20260726/DeepEP/` | worst-case async allocation plus device counts; padding-only zero | high | README, legacy docs, ElasticBuffer API and dispatch allocation reviewed locally |
| DeepEP HybridEP `94a9f8f6b146c07d97ec58f67cd6d303296d6098` | external code | `scratch/prior-art-20260726/DeepEP-hybrid/` | fused TMA pipeline, metadata counts, padding semantics, GB200 tuning | high | design docs, Python/C++ API, executor, buffers, permute, metadata scan, kernel launch, and tests reviewed locally |
| Loong-Megatron PR 7, head `b216e0a61ebbaaf147449acbe025d926c8ee137c` | PR / external code | `scratch/prior-art-20260726/loong-pr7/` | buffer parity, stream ordering, home/ECHO count vectors | medium | full 7,972-line patch read; PR closed |
| HybridEP JAX stable run | W&B | `marin-community/rav_moe/ep64-hybridep-zeropad-30-v24-20260726-0936` | correctness checkpoint and 10.55% p50 MFU | high | 30/30 steps, finite loss, zero final overflow |
| Receiver-semantic native A2A numerical gate | W&B | `marin-community/rav_moe/ep64-echo-fixed-a2a-numerics-v34-20260726-1152` | finite EP64 execution and physical-drop bound | high | all 54 numerical checks passed; 0.47% mean / 1.34% max physical drop |
| Receiver-semantic native A2A top-8 | W&B | `marin-community/rav_moe/ep64-echo-fixed-a2a-sonic-dispatch-v39-20260726-1221` | stable throughput under physical-drop bound | high | 17.02% p50 MFU; 1.47% mean / 2.50% max physical drop |
| Receiver-semantic native A2A top-4 | W&B | `marin-community/rav_moe/ep64-echo-fixed-top4-sw512-sonic-dispatch-v43-20260726-1232` | sparsity/config throughput under physical-drop bound | high | 18.82% p50 MFU; 0.49% mean / 1.26% max physical drop |
| Receiver-semantic Sonic slot gather | W&B | `marin-community/rav_moe/ep64-top4-slotgather-v44-20260726-1245` | profile-predicted removal of generic gather adjoints | high | 20.67% p50 at the last synced step; all 12 train steps completed, then the forced final local checkpoint failed |
| Receiver-semantic 30-step qualification | W&B | `marin-community/rav_moe/ep64-top4-slotgather-stable30-v49-20260726-1301` | steady throughput, finite loss, and physical-drop bound | high | 30/30 steps; 20.51% p50 MFU; finite falling loss; final 1.28% mean / 1.94% max physical drop |
| Wider receiver-semantic configuration | W&B | `marin-community/rav_moe/ep64-d5120-i2048-sh6144-steady-v53-20260726-1317` | routed/shared MLP width scaling | high | 12/12 steps; 20.99% p50 MFU; final 1.98% mean / 2.97% max physical drop |
| Fused clone-weight exchange | W&B | `marin-community/rav_moe/ep64-top4-fusedweights-v54-20260726-1334` | collective fusion A/B | high | 12/12 steps; regressed to 20.21% p50 MFU versus 20.51% matched control |
| QuACK grouped expert weight-gradient | W&B | `marin-community/rav_moe/ep64-d5120-wide-quackwgrad-v58-20260726-1343` | profile-derived expert wgrad A/B | high | 12/12 steps; 21.34% p50 MFU; finite loss; final 2.04% mean / 3.14% max physical drop |
| QuACK grouped expert weight-gradient qualification | W&B | `marin-community/rav_moe/ep64-d5120-wide-quackwgrad30-v64-20260726-1358` | steady grouped-wgrad throughput and stability | high | 30/30 steps; 21.28% p50 MFU; finite falling loss; final 1.31% mean / 1.92% max physical drop |
| Ten-segment sparse clone envelope | W&B | `marin-community/rav_moe/ep64-d5120-wide-seg10-v61-20260726-1350` | receiver-envelope A/B | high | 12/12 steps; 21.23% p50 MFU; final 1.88% mean / 2.60% max physical drop |
| Grouped wgrad plus ten-segment envelope | W&B | `marin-community/rav_moe/ep64-d5120-wide-wgrad-seg10-v65-20260726-1402` | combined throughput and drop bound | high | 12/12 steps; 21.52% p50 MFU; final 1.95% mean / 2.976% max physical drop |
| Grouped wgrad plus ten-segment qualification | W&B | `marin-community/rav_moe/ep64-d5120-wide-wgrad-seg10-30-v70-20260726-1425` | steady throughput, loss, and physical-drop bound | high | 30/30 steps; 21.39% p50 MFU; 386.2k tok/s; final loss 7.781; final 0.93% mean / 1.78% max physical drop |
| Three-token-padding envelope | W&B | `marin-community/rav_moe/ep64-d5120-i2048-pad3-v69-20260726-1410` | token-buffer fit/throughput boundary | high | 12/12 steps; 21.46% p50 MFU; rejected because final worst-layer physical drop reached 3.039% |
| FP8 token-transport smoke | W&B | `marin-community/rav_moe/ep4-fp8-transport-smoke-v47-20260726-1253` | GPU support and finite update gate for rowwise E4M3 transport | medium | two steps on four GB200 GPUs; all logged parameter, gradient, and update checks finite |
| FP8 token-transport EP64 | W&B | `marin-community/rav_moe/20260726-131005` | throughput and numerical A/B | high | 12/12 steps; 20.49% p50 MFU; throughput-neutral and numerically distinct |
| D6144/L32 clean fit probe | Iris | `/rav/ep64-d6144-L32-top4-nodebug-v52-20260726-1311` | default-pool fit boundary | high | transient startup retries followed by a true 79.04-GiB training-step allocation OOM |
| D6144/L32 90% device-pool fit probe | Iris | `/rav/ep64-d6144-L32-mem90-v56-20260726-1338` | reduced-pool fit boundary | high | training-step compilation still failed with CUDA out of memory |
| D6144/L32 optimizer-offload fit probe | Iris | `/rav/ep64-d6144-L32-offload-v55-20260726-1335` | host-offload fit boundary | medium | final retry lost rank 0 with exit 137 after dataset setup and before compilation; retry requires a larger host-RAM request |
| D6144/L32 optimizer-offload 512-GiB fit probe | Iris | `/rav/ep64-d6144-L32-offload-ram512-v67-20260726-1404` | device fit after removing the host-RAM ambiguity | high | reached `jit_train_step`; NCCL allocations failed with CUDA OOM across the rack despite optimizer offload and a 95% JAX pool |
| D5120/I3072 95%-pool fit probe | Iris | `/rav/ep64-d5120-i3072-mem95-v68-20260726-1410` | wider expert device-memory boundary | high | reached `jit_train_step`; NCCL allocations failed with CUDA OOM across the rack |
| D5120/I2560 fit probe | Iris | `/rav/ep64-d5120-i2560-wgrad-seg10-v74-20260726-1435` | routed-expert width fit boundary | high | failed at the first executable with an 80.43-GiB `jit_train_step` allocation |
| Receiver-semantic one-node XProf | XPlane / XProf | `s3://marin-us-east-02a/tmp/ttl=30d/xprof/ep64-echo-fixed-a2a-sonic-profile1n-v40-20260726-1223` | collective and local-kernel attribution | high | one node, steps 8-10, HLO proto enabled, CUDA graphs disabled; NCCL send/recv 26.0% |
| Current-best one-node XProf | XPlane / XProf | `s3://marin-us-east-02a/tmp/ttl=30d/xprof/ep64-d5120-wide-wgrad-seg10-profile1n-v72-20260726-1427` | current token/weight communication attribution | high | one node, steps 8-10, HLO proto enabled, CUDA graphs disabled; communication 28.3% of aggregate kernel time; fixed token A2As ~15.83 device-s, sparse clone-weight ragged A2A 2.59 device-s |
| Current-config MNNVL transport | Iris JSON tracker | `/rav/ep64-d5120-wide-wgrad-seg10-mnnvl-v79-20260726-1517` | direct fabric-write transport A/B | high | 12/12 steps; 20.36% p50 MFU; receiver overflow remained below 3%; current barrier-and-copy implementation regressed versus native v70 |
| Full MoE residual checkpointing | Iris | `/rav/ep64-d5120-wide-wgrad-seg10-savemoe-v80-20260726-1529` | avoid rematerialized EP collectives | high | training executable requested 374.32 GiB per GPU; existing `save_moe` policy is not viable at L48 |
| Two-layer scan unroll | Iris | `/rav/ep64-d5120-wide-wgrad-seg10-unroll2-v81-20260726-1532` | cross-layer weight-prefetch opportunity | high | all 64 ranks connected; tracing failed deterministically with an explicit-sharding mismatch in scan's `dynamic_update_slice`; full log saved locally |
| Shared-expert 8192 width | Iris JSON tracker | `/rav/ep64-d5120-i2048-sh8192-wgrad-seg10-v82-20260726-1538` | useful-compute capacity scaling | high | 12/12 steps; 22.40% p50 MFU during steady execution; 368.1k tok/s and finite loss 8.369; final physical drop 2.44% mean / 3.31% max, so the throughput result needs a slightly larger envelope to meet the strict 3% bound |
| Fused receiver-slot metadata, 16-byte row alignment | Iris JSON tracker | `/rav/ep64-d5120-wide-fusedslot-v83-20260726-1547` | eliminate separate receiver-slot all-to-all | high | 12/12 steps; regressed to about 20.75% p50 MFU and 376.8k tok/s. Appending eight BF16 words changed each row from 10240 to 10256 bytes, breaking 128-byte alignment for the much larger activation collective; an aligned 64-word variant is v85 |
| Fused receiver-slot metadata, 128-byte row alignment | Iris JSON tracker | `/rav/ep64-d5120-wide-fusedslot-align128-v85-20260726-1557` | test whether row alignment explains the fused-metadata regression | high | 12/12 steps; 20.72% p50 MFU, 376.1k tok/s at step 11, finite loss 9.021, and final 1.60% mean / 2.80% max physical drop. Restoring 128-byte row alignment did not recover throughput |
| Row-packed receiver-slot metadata | Iris JSON tracker | `/rav/ep64-d5120-wide-fusedslot-rows-v87-20260726-1601` | eliminate the slot all-to-all with negligible aligned activation traffic | high | 12/12 steps; 20.87% p50 MFU, 378.8k tok/s at step 11, finite loss 8.904, and final 1.73% mean / 2.69% max physical drop. Packing the exact int32 metadata into four full hidden-width rows still regressed, so the local prototype was removed |
| Shared-expert 12288 width | Iris JSON tracker | `/rav/ep64-d5120-sh12288-pad5-v86-20260726-1559` | useful-compute capacity scaling under the strict drop bound | high | 12/12 steps; 23.55% p50 MFU, 326.4k tok/s at step 11, finite loss 8.430, and final 1.85% mean / 2.54% max physical drop |
| Shared-expert 16384 width fit probe | local Iris log | `/rav/ep64-d5120-sh16384-pad5-v88-20260726-1602` | upper shared-capacity memory boundary | high | `jit_train_step` required 88.08 GiB and failed with `RESOURCE_EXHAUSTED` on multiple ranks before step 0. Coordinator connection failures were teardown consequences. Full 3.55-MB log saved at `/tmp/marin-job-logs/ep64-d5120-sh16384-pad5-v88-20260726-1602.log` |
| Shared-expert 14336 width | Iris JSON tracker | `/rav/ep64-d5120-sh14336-pad5-v89-20260726-1619` | largest known fitting shared capacity | high | 12/12 steps; 24.11% p50 MFU, 310.2k tok/s at step 11, finite loss 8.853, and final 0.44% mean / 1.54% max physical drop. Full 11.9-MB log saved at `/tmp/marin-job-logs/ep64-d5120-sh14336-pad5-v89-20260726-1619.log` |
| Same-expert ECHO chunking guard | local Iris log | `/rav/ep64-d5120-sh16384-ch2-pad5-v90-20260726-1619` | reduce the peak activation allocation of the 16384 shared-expert configuration | high | failed before compilation because the implementation explicitly rejected chunks greater than one. Full 4.46-MB log saved at `/tmp/marin-job-logs/ep64-d5120-sh16384-ch2-pad5-v90-20260726-1619.log`; the guard now has an exact value/gradient-tested chunk loop |
| Shared-expert 15360 width plus embedded metadata | local Iris log | `/rav/ep64-d5120-sh15360-embslot-pad5-v91-20260726-1636` | shared-width fit boundary and fused-metadata smoke | high | the training executable requested 85.18 GiB and failed before step 0. The 3.69-MB full rack log at `/tmp/marin-job-logs/ep64-d5120-sh15360-embslot-pad5-v91-20260726-1636.log` shows the allocation failure as the first root cause; later failures are teardown |
| Shared-expert 16384 width with two ECHO chunks | local Iris log | `/rav/ep64-d5120-sh16384-ch2-pad3-v92-20260726-1636` | reduce peak transport activation memory | high | the training executable still requested exactly 88.08 GiB, identical to the unchunked width-16384 probe. Python-level chunking did not alter the XLA peak and was removed. Full 3.62-MB rack log saved at `/tmp/marin-job-logs/ep64-d5120-sh16384-ch2-pad3-v92-20260726-1636.log` |
| Embedded slot metadata, compacted token rows | local Iris log | `/rav/ep64-d5120-sh14336-embslot-pad5-v93-20260726-1648` | remove the standalone receiver-slot all-to-all without increasing transport bytes | high | 12/12 steps; regressed to 23.14% p50 MFU and 297.5k tok/s versus 24.11% matched baseline. Final physical drop was 0.71% mean / 1.19% max. Compacting token rows around per-destination metadata padding adds full-payload copies, so this layout was replaced by a physical-layout-preserving implementation |
| Embedded slot metadata with four padding experts | local Iris log | `/rav/ep64-d5120-sh14336-embslot-pad4-v94-20260726-1648` | combine metadata fusion with a smaller fixed transport envelope | high | 12/12 steps; 22.94% p50 MFU and 295.1k tok/s. Final physical drop reached 2.48% mean / 3.17% max, violating the strict bound; both throughput and drop reject this variant |
| Embedded slot metadata, physical token layout | local Iris log | `/rav/ep64-d5120-sh14336-embslot-physical-v95-20260726-1702` | remove the standalone receiver-slot all-to-all without payload compaction | high | 12/12 steps; 24.27% p50 MFU, 312.9k tok/s, finite loss 9.082, and final 0.53% mean / 1.23% max physical drop. Preserving the fixed token layout recovers the compacted prototype's regression and improves about 0.15 MFU over the matched v89 baseline |
| Shared-expert 16384 width with CUDA async allocator | local Iris log | `/rav/ep64-d5120-sh16384-cudaasync-v96-20260726-1704` | cross the BFC allocator's apparent 14K shared-width memory ceiling | high | 12/12 steps; about 24.70% p50 MFU, finite final loss 8.435, and roughly 297k tok/s. The final physical overflow reached 2.23% mean / 3.33% max, so this is a throughput lead rather than a qualified result. Full 1.06-MB log saved at `/tmp/marin-job-logs/ep64-d5120-sh16384-cudaasync-v96-20260726-1704.log` |
| Two shared experts at aggregate width 16384 | local Iris log | `/rav/ep64-d5120-sh16384-split2-v97-20260726-1705` | test whether two 8192-wide shared MLPs reduce the FSDP gather peak | high | failed before step 0: the compiled executable still requested 89.49 GiB with the default BFC allocator. Splitting the shared width does not stream its layer-scanned weights or reduce the peak. Full 268-KB log saved at `/tmp/marin-job-logs/ep64-d5120-sh16384-split2-v97-20260726-1705.log`; later coordinator errors are teardown after the allocation failure |
| Shared-expert 17408 width with CUDA async allocator | local Iris log | `/rav/ep64-d5120-sh17408-cudaasync-v98-20260726-1714` | test whether more useful shared compute crosses 25% MFU | high | 12/12 steps; about 23.99% p50 MFU, 279.7k tok/s, finite loss 8.779, and final 1.54% mean / 2.28% max physical drop. The larger executable required much more XLA rematerialization and regressed despite the larger analytic-FLOP numerator. Full 1.05-MB log saved at `/tmp/marin-job-logs/ep64-d5120-sh17408-cudaasync-v98-20260726-1714.log` |
| Embedded physical-layout metadata at shared width 16384 | local Iris log | `/rav/ep64-d5120-sh16384-cudaasync-embslot-v99-20260726-1714` | combine the eliminated metadata collective with the 16K shared-width lead | high | 12/12 steps; about 24.45-24.50% p50 MFU, 293.9k tok/s, finite loss 8.425, and final 1.76% mean / 2.56% max physical drop. This is safely within the drop bound but does not cross 25%; full 1.16-MB log saved at `/tmp/marin-job-logs/ep64-d5120-sh16384-cudaasync-embslot-v99-20260726-1714.log` |
| Shared-expert 17408 width with six-expert receiver padding | local Iris log | `/rav/ep64-d5120-sh17408-pad6-cudaasync-v100-20260726-1724` | qualify the widest useful-compute configuration under the physical-drop bound | high | 12/12 steps; 24.53% p50 MFU, 284.5k tok/s, finite loss 8.609, and final 1.25% mean / 1.76% max physical drop. This is the current best fully compliant result; a one-node no-CUDA-graphs profile is v105 |
| Routed-expert 2560 plus shared-expert 8192 | local Iris log | `/rav/ep64-d5120-i2560-sh8192-pad6-cudaasync-v101-20260726-1727` | test whether moving useful compute into the routed grouped GEMMs improves utilization | high | 12/12 steps; only 21.76% p50 MFU despite 329.5k tok/s, with finite loss 8.361 and final 1.15% mean / 1.95% max physical drop. Larger routed grouped GEMMs are much less efficient than adding the same active compute to the shared dense path |
| Routed-expert 3072 plus shared-expert 6144 | local Iris log | `/rav/ep64-d5120-i3072-sh6144-pad6-cudaasync-v102-20260726-1728` | test the routed-width memory/performance ceiling under CUDA async allocation | high | failed before step 0 when NCCL's first training-step collective allocation ran out of device memory. Full 299-KB log saved at `/tmp/marin-job-logs/ep64-d5120-i3072-sh6144-pad6-cudaasync-v102-20260726-1728.log`; subsequent coordinator errors were teardown |
| Eight full-attention layers at shared width 16384 | local Iris log | `/rav/ep64-d5120-sh16384-pad6-global6-cudaasync-v103-20260726-1733` | test the requested 5:1 local/global schedule as useful compute | high | 12/12 steps; 24.30% p50 MFU, 285.6k tok/s, finite loss 8.557, and final 1.61% mean / 2.43% max physical drop. The eight full-attention layers are compliant but less efficient than the all-local 17408-wide shared configuration |
| Shared-expert 19456 width with six-expert receiver padding | local Iris log | `/rav/ep64-d5120-sh19456-pad6-cudaasync-v104-20260726-1737` | extend the compliant shared-dense scaling curve toward 25% MFU | high | 12/12 steps; 24.83% p50 MFU, 270.3k tok/s, finite loss 8.734, and final 0.42% mean / 1.34% max physical drop. This is the current highest compliant result and leaves substantial drop headroom for sparse clone-envelope tightening |
| Shared-expert 17408 no-CUDA-graphs profile | XPlane / XProf | `s3://marin-us-east-02a/tmp/ttl=30d/xprof/ep64-d5120-sh17408-pad6-profile1n-v105-20260726-1742` | current receiver-path communication and kernel attribution | high | one node, steps 8-11, HLO proto enabled, CUDA graphs disabled; 70.6% compute / 29.4% communication, with NCCL send/recv contributing 19.6% of aggregate kernel duration |
| Eight-segment receiver envelope | local Iris log | `/rav/ep64-d5120-sh17408-pad6-seg8-v106-20260726-1747` | reduce sparse clone envelope | high | 12/12 steps; 24.30% p50 MFU and 282.9k tok/s, but final physical drop reached 1.75% mean / 3.50% max, violating the strict bound |
| Six-segment receiver envelope | local Iris log | `/rav/ep64-d5120-sh17408-pad6-seg6-v107-20260726-1747` | lower sparse clone-envelope boundary | high | first training execution hit `CUDA_ERROR_ILLEGAL_ADDRESS` on multiple GPUs in `AsyncExecution::ExecutionGuard`; the remaining coordination and SIGTERM errors were teardown. Full 3.57-MB log saved locally |
| Shared-expert 21504 width with six-expert receiver padding | local Iris log | `/rav/ep64-d5120-sh21504-pad6-cudaasync-v108-20260726-1751` | cross 25% MFU under receiver-side capacity semantics | high | 12/12 steps; 25.21% p50 MFU, 258.2k tok/s, finite loss 8.949, and final 0.25% mean / 0.71% max exact post-ECHO assignment drop. Plain top-4 routing; QB is off. Unlike sender/expert clipping, ECHO preserves the selected expert and relocates its execution using sparse weight clones |
| Address-computation fusion flags | local Iris log | `/rav/ep64-d5120-sh19456-pad6-addressfusion-v109-20260726-1755` | fuse profile-visible layout copies | high | rejected before training because this XLA build does not recognize either `xla_gpu_enable_custom_fusions` or `xla_gpu_enable_address_computation_fusion`; full 1.23-MB log saved locally |
| Shared-expert 21504 30-step qualification | W&B | `marin-community/rav_moe/ep64-d5120-sh21504-pad6-wandb30-v110-20260726-1805` | steady >=25% MFU, loss, and physical-drop qualification | high | finished cleanly at 24.50% p50 MFU (p10 24.19 / p90 24.81), final loss 7.237, and 1.08% mean / 1.56% worst-layer exact post-ECHO assignment drop. This rejects the short v108 result as sufficient proof of stable >=25% |
| Shared-expert 21504 final one-node profile | XPlane / XProf | `s3://marin-us-east-02a/tmp/ttl=30d/xprof/ep64-d5120-sh21504-pad6-profile1n-v111-20260726-1807` | final >=25% configuration profile | high | succeeded: one node, HLO proto enabled, CUDA graphs disabled with `--xla_gpu_enable_command_buffer=`. Profile-window run reached 25.08% p50 MFU (p10 24.91 / p90 25.19) and 0.095% mean / 0.690% worst-layer exact drop. XPlane attribution is 78.3% compute / 21.7% communication; NCCL send/recv alone is 15.9% |
| Shared-expert 21504 QB-on qualification | W&B | `marin-community/rav_moe/ep64-d5120-sh21504-pad6-qb200-v112-20260726-1816` | measure steady QB effect on post-ECHO drops, clone traffic, and MFU | high | finished 200/200 with finite loss falling to 4.984. Tail-50 median MFU was 24.776% at 254.0k tokens/s, below target; tail-50 median exact aggregate post-ECHO drop was 0.684% and its maximum was 0.895%. The full-run 24.84% p50 confirms that the short 25% result was not a stable qualification |
| Shared-expert 22528 30-step qualification | W&B | `marin-community/rav_moe/ep64-d5120-sh22528-pad6-wandb30-v113-20260726-1830` | recover stable >=25% with a larger efficient shared dense expert | high | finished cleanly at 24.76% p50 MFU (p10 24.38 / p90 25.01), falling to 24.12% by step 29, with finite loss 7.387 and 0.93% mean / 1.33% worst-layer exact drop. Wider shared compute does not recover the long-run decline |
| Shared-expert 21504 steady-QB one-node profile | XPlane / XProf | `s3://marin-us-east-02a/tmp/ttl=30d/xprof/ep64-d5120-sh21504-pad6-qb-profile150-v114-20260726-1835` | capture the balanced-routing steady regime rather than QB's first 30 steps | high | 155-step QB-on run; profile steps 150-153 from process indices 0-3 only, HLO proto enabled, and CUDA graphs disabled with `--xla_gpu_enable_command_buffer=`. XPlane attribution is 78.37% compute / 21.63% communication, with NCCL send/recv at 16.2%. The six fixed token A2As total 2.217 device-s/step and sparse clone exchange totals 0.485 device-s/step, respectively 4.8% and 10.3% slower than the early QB-off v111 profile after normalizing for its three-step window. |
| Shared-expert 21504 QB-off qualification | W&B | `marin-community/rav_moe/ep64-d5120-sh21504-pad6-noqb200-v115-20260726-1840` | matched 200-step control for the QB-on v112 run | pending | matches v112's optimizer schedule and receiver-ECHO configuration, with only `SCALE_MOE_QB=0`; this avoids confounding QB with the different 30-step learning-rate schedule used by v110 |
| Shared-expert 21504 four-padding-expert QB qualification | W&B | `marin-community/rav_moe/ep64-d5120-sh21504-pad4-qb200-v116-20260726-1845` | reduce the profile-dominant fixed token envelope while checking exact settled drops | high | stopped after step 38 because it was dominated: 25.16% p50 / 24.76% current MFU with 1.00% exact aggregate drop. The 5,120-row message shape regressed relative to both padding 6 and padding 2 |
| Shared-expert 21504 three-padding-expert QB qualification | W&B | `marin-community/rav_moe/ep64-d5120-sh21504-pad3-qb200-v117-20260726-1848` | test the lowest empirically compliant token envelope over the full QB settling window | high | stopped after step 105 because padding 2 dominated it. Tail-50 median MFU was 24.9999% with 1.144% median / 1.446% maximum exact aggregate drop and finite falling loss. Padding 2 was about 0.6 MFU faster with comparable drop after 80+ steps |
| Shared-expert 21504 three-padding-expert embedded-metadata QB qualification | W&B | `marin-community/rav_moe/ep64-d5120-sh21504-pad3-embslot-qb200-v118-20260726-1850` | combine the reduced token envelope with the isolated slot-collective removal | high | stopped after step 35 because it was dominated by plain padding 3: 25.10% tail-20 median MFU versus 25.16%, with 1.72% versus 1.30% tail-20 median exact aggregate drop. The earlier short-run metadata gain did not reproduce under the matched long schedule |
| Shared-expert 21504 two-padding-expert QB qualification | W&B | `marin-community/rav_moe/ep64-d5120-sh21504-pad2-qb200-v119-20260726-1853` | preserve the capacity-maximized architecture option after restoring the target's shared width to 5120 | high | finished 200/200 with 25.566% full-run p50 MFU. Tail-100 median MFU was 25.501%, tail-50 was 25.478%, and tail-20 was 25.486%; tail-50 median throughput was 261.2k tok/s. Exact aggregate post-ECHO drop remained compliant: 2.024% tail-50 median, 2.054% tail-20 mean, and 1.986% final. Loss was finite and fell to 4.960. This is a reproducible architecture option, not evidence for the shared-5120 >30% kernel target. |
| Shared-expert 21504 zero-padding QB endpoint | Iris | `/rav/ep64-d5120-sh21504-pad0-qb200-v120-20260726-1857` | test whether QB-balanced per-sender traffic can use the exact-mean token envelope under the 3% drop bound | low | failed during Iris dependency synchronization before a nested training task or W&B run existed. This is not a kernel or capacity result and may be retried after the active frontier runs finish |
| Shared-expert 21504 three-padding-expert NCCL-64-CTA A/B | local Iris log | `/rav/ep64-d5120-sh21504-pad3-qb-nccl64-v121-20260726-1901` | test whether more NCCL communication CTAs accelerate the profile-dominant token exchanges | high | failed on the first `jit_train_step`: forcing 64 CTAs made `ncclAlltoAll` allocations fail with CUDA out of memory across multiple ranks. Later coordinator connection failures are teardown. This is not a performance result; the full log is `/tmp/marin-job-logs/ep64-d5120-sh21504-pad3-qb-nccl64-v121-20260726-1901.log` |
| Shared-expert 21504 one-padding-expert QB qualification | W&B | `marin-community/rav_moe/ep64-d5120-sh21504-pad1-qb200-v122-20260726-1904` | fill the capacity frontier between the two-padding and zero-padding endpoints | high | stopped after step 103 because it remained dominated by padding 2: current MFU was 25.08% with 2.758% exact aggregate drop, while padding 2 was sustaining about 25.5% with lower exact drop. The 4,352-row envelope is therefore neither faster nor safer on the observed QB trajectory. |
| Shared-expert 21504 zero-padding QB endpoint retry | W&B | `marin-community/rav_moe/ep64-d5120-sh21504-pad0-qb200-v123-20260726-1913` | retry the exact-mean token envelope after v120's pre-training setup failure | pending | exact v120 resubmission with QB on, 200 steps, and CUDA async allocation; the outer job uses a fresh run ID and coordinator port |
| Shared-expert 21504 two-padding-expert NCCL-48-CTA A/B | local Iris log | `/rav/ep64-d5120-sh21504-pad2-qb-nccl48-v124-20260726-1923` | increase communication parallelism for the profile-dominant token A2As without the 64-CTA allocation failure | high | failed on the first `jit_train_step`: NCCL `ncclAlltoAll` allocations ran out of HBM across many ranks. Later coordinator connection failures are teardown. Full 2.18-MB log: `/tmp/marin-job-logs/ep64-d5120-sh21504-pad2-qb-nccl48-v124-20260726-1923.log` |
| Shared-expert 21504 two-padding-expert steady one-node profile | XPlane / XProf | `s3://marin-us-east-02a/tmp/ttl=30d/xprof/ep64-d5120-sh21504-pad2-qb-profile150-v125-20260726-1926` | profile the current highest-MFU drop-compliant trajectory under the exact 200-step schedule | pending | QB on, padding 2, profile steps 150-153 from process indices 0-3 only, HLO proto enabled, and CUDA graphs disabled with `--xla_gpu_enable_command_buffer=`. Unlike v114, its optimizer schedule exactly matches the 200-step qualification run |
| Shared-expert 21504 two-padding-expert 350-step QB qualification | W&B | `marin-community/rav_moe/ep64-d5120-sh21504-pad2-qb350-v126-20260726-1930` | verify the highest-MFU capacity after the full QB settling horizon from echo #458 | pending | normal CUDA graph settings, QB on, padding 2, and 350 training steps. This is the final stability gate if default 32-CTA transport remains faster than the 48-CTA A/B |
| Shared-expert 21504 two-padding-expert NCCL-40-CTA A/B | local Iris log | `/rav/ep64-d5120-sh21504-pad2-qb-nccl40-v127-20260726-1932` | locate whether any communication-parallelism point above the default 32 CTAs fits HBM | high | failed on the first `jit_train_step`: NCCL reported CUDA out of memory while allocating for `ncclAlltoAll` on multiple ranks. The first root cause begins at log line 7010; later connection-refused errors are teardown. Together with the 48- and 64-CTA failures, this closes the above-default CTA lead. Full log: `/tmp/marin-job-logs/ep64-d5120-sh21504-pad2-qb-nccl40-v127-20260726-1932.log` |
| Original shared-5120 steady QB control | W&B | `marin-community/rav_moe/ep64-d5120-sh5120-pad2-qb200-v128-20260726-1953` | establish the stable matched-shape baseline for the >30% kernel target | high | finished 200/200. Full-run p50 was 21.050% and mean 21.067%; tail-100 median was 21.023%, tail-50 21.007%, and tail-20 20.997%. Tail-50 median exact aggregate post-ECHO drop was 1.784%; final loss was 5.098 and finite/falling. |
| Original shared-5120 steady-QB one-node profile | XPlane / XProf | `s3://marin-us-east-02a/tmp/ttl=30d/xprof/ep64-d5120-sh5120-pad2-qb-profile150-v129-20260726-1953` | attribute the removable gap from the matched 21.05% control to 30% | high | process indices 0-3 only, profile steps 150-154, HLO proto enabled, and CUDA graphs disabled with `--xla_gpu_enable_command_buffer=`. Aggregate GPU kernel time is 69.87% compute / 30.13% communication; NCCL SendRecv is 23.20%. The six fixed BF16 `[64,4608,5120]` token A2As total 7.407 device-s across four GPUs and four steps. |

### Handoff

- Suggested issue `Prior work` block: do not post unless explicitly requested.
- Suggested logbook entry: static receive capacity is compatible with
  device-dynamic expert work; full-envelope zeroing is only a correctness
  checkpoint.
- Open questions:
  - What is the steady p50 of the receiver-semantic native A2A path?
  - How much time does its sparse clone-weight exchange consume relative to
    token A2A and dispatch/combine gathers?
  - Can the existing Sonic dispatch/combine and clone-weight-gradient kernels
    improve that path without changing its accepted token set?
- Stop reason: the requested sources converge on the same next experiment;
  additional reading would not change the top-ranked profile question.
