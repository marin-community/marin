# Marin EP M7: fused transport+GEMM (consume behind arrival flags)

Coordinating issue: #8311. Logbook: `.agents/logbooks/marin-ep.md` (MEP-036
has the motivating profile). Branch: `marin-ep-8081`.

## Why now

The multi-process fused transport is validated in the production topology
(MEP-030..034: collective-metadata path + wheel v3 + dynamic-slice-fusion
off). EP64 fused = 17.75 s/step vs goal 12.0 s. Per the MEP-036 profile the
transport is 1.16 s busy and largely overlapped; M7's value is removing the
launch/sync structure around dispatch/combine (transport + the fusion mass
that pads it + the wait-for-all barrier semantics), estimated 1-1.5 s of
exposed time. M7 alone does NOT reach the goal; GEMM efficiency (4.8 s
busy) and XLA fusion mass (3.8 s) are independent, larger buckets.

## Design

Two kernels on one stream serialize, so "GEMM waits on flags" requires the
GEMM to live in the same kernel as (or run concurrently with) the puts.
Chunked XLA-level pipelining is dead (MEP-029: no async ragged; custom
calls share the main stream). The viable shape is a single warp-specialized
persistent Mosaic kernel:

- Transport warpgroup(s): today's `put_segments` loop (TMA gmem->smem->
  peer gmem via `remote_ref`), but signaling a per-(expert-chunk) arrival
  flag array in the owner's memory after each segment instead of one
  end-of-kernel semaphore per SM.
- Consumer warpgroups: grouped-GEMM tile scheduler that polls the flag
  array (plain global atomics) per K-slab of pool rows before issuing
  wgmma/tcgen05 MMA on those rows.
- The combine direction mirrors it: GEMM epilogue signals per-tile
  completion; transport warps put finished tiles back to sources.

Open question: the production GEMM is QuACK/cuDNN CuTeDSL (black box).
Either (a) write the grouped GEMM in Pallas/Mosaic (sm100 tcgen05; perf
risk vs cuDNN cute at d6144 hero shapes), or (b) CuTeDSL grouped GEMM via
cutlass_call polling a flag buffer passed as an extra operand (2.2 PF/s
grouped kernel exists from the MXFP8 campaign; flags are plain global
loads). (b) still needs the puts and the GEMM to run CONCURRENTLY -> puts
must come from a different stream or the same fused kernel; a cutlass_call
GEMM cannot share a kernel with Pallas puts. So (b) requires multi-stream
(XLA does not give custom calls streams) -> (a) is the real path.

## Milestones

- M7a: warp-specialized Pallas prototype at EP4 single-node: transport
  warpgroup + consumer warpgroup that polls flags and accumulates a
  checksum; prove flag-gated consumption with no deadlock, measure
  transport-hidden fraction vs sequential put+consume.
- M7b: replace checksum consumer with a bf16 grouped-GEMM tile loop
  (tcgen05 via plgpu); target >= 0.7x cuDNN-cute on the hero shard shape
  (2460x6144 x per-expert 6144x3072) before wiring further.
- M7c: EP16/EP64 integration behind `transport="fused_gemm"`; A/B vs
  17.75 s.
- Kill criterion: if M7b cannot reach 0.7x cuDNN-cute after tuning, fold
  back to the split path and redirect to GEMM/fusion buckets.

## State

- 2026-08-16: plan written; PGLE and command-buffer cheap arms in flight
  first (profile suggested scheduler slack). M7a not started.
