# 7333 — Source-push EP MoE on B200 production shapes (SPB series)

Coordinating issue: https://github.com/marin-community/marin/issues/7333
Branch: `research/mcwitt/7333-source-push-b200`

B200-phase continuation of the source-push EP MoE thread. Prior phases and
adjacent threads:

- H100 phase (SPF series): #7276, logbook `6841-source-push-followup.md` on
  `research/mcwitt/6841-source-push-followup`. Headlines: staged semantic path
  best honest API 61.0 ms vs 38.2 ms `ring` at EP8 H100; host planner ~380
  ms/plan is the hardware-independent production blocker (SPF-005); Lane
  kernels are `mgpu.wgmma` (sm_90a-only) and do not carry to Blackwell.
- Blackwell source-push transport: #6933 — staged transport validated at B300
  EP8 (1596 useful W13-equiv TFLOP/s/rank, zero drops); peer-ref Warpgroup
  kernels unsupported on Blackwell ("GMEM refs with peer ids are not supported
  in warpgroup lowering"); fused W13/SwiGLU epilogue blocked by a Mosaic
  `copy_gmem_to_smem` swizzle assertion.
- Production MFU thread: #7279 (B200MFU series) — 64-GPU GB200 production
  config d5120 L48 e64 top4 batch 1024; best 20.83% MFU (ring_cute EP4,
  recompute_all); collectives ~48% of device-busy time, latency-bound (NVLink
  at 8-14% of per-direction capability). Source-push instantiates that issue's
  hypothesis 1.

Tags: `spb`, `7333`, `b200`.

Venues: cw-us-east-08a GB200 (MNNVL default; `NCCL_MNNVL_ENABLE=0` for IB
A/B); shared academic B200/B300 Slurm cluster for single-node kernel work.

## Hypothesis queue

| ID | Hypothesis | Status |
| --- | --- | --- |
| SPB-001 | Census + gate: enough of the #6841 staged semantic path runs on GB200 (planner, transport, XLA combine; which Pallas kernels lower on sm_100) to measure staged source-push forward at the #7279 shapes vs ring_cute/a2a_cute bests; source-push reduces the ~48% collective share | OPEN — first experiment |
| SPB-002 | Tuned Blackwell compute epilogues (unblock #6933 swizzle assertion or CuTeDSL via `cutlass_call` per #7282) close the gap to fused local compute | BLOCKED on SPB-001 gate |
| SPB-003 | Device-side planner removes the ~380 ms/plan host gate (SPF-005 carry-over) | BLOCKED on SPB-001 gate |

## Entries

(append-only below)
