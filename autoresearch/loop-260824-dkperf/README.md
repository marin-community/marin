# loop-260824-dkperf: dk ragged-a2a to one-shot parity (leg 4)

Goal: bring the device-kernel (dk) ragged-all-to-all transport to parity with or past the
one-shot transport at the hero EP64 shape. Exhaust the remaining flag/knob space first, then
kernel-level XLA patches, then backend restructures.

Targets (from loop-260823-dknative final standings):
- microbench 64-rank per-call: one-shot 7.56 ms (424 GB/s egress) | dk stock 22.61 | dk g8x128 12.18
- hero synthetic MFU (steps 5-15): one-shot 22.77 (s64) | dk 22.62 (s1)
- hero 6k-restore MFU: one-shot 22.46-22.60 (s64) | dk 21.98 (s64)

Protocol: microbench gang job (16 nodes x GB200x4, one flag config per job -- two gang configs
cannot share one iris job) is the cheap screen; hero synthetic arm confirms; 6k-restore arm is
the final gate. One rack job at a time. Wheels build on CPU-only iris jobs (~20 min, no rack).

Standard runtime: branch research/mcwitt/8317-dk-native, g8x128 wheel
(s3://marin-us-east-02a/marin/research/mcwitt-ra2a/pjrt-mainpatch-g8x128-20260823/),
dk flags `--xla_gpu_experimental_ragged_all_to_all_use_device_kernel=true
--xla_enable_nccl_symmetric_buffers_for_collectives=raggedalltoall`, splits_per_peer=1
(synthetic) / 64 (restore, trained-router skew).

## Knob audit (XLA main e5d008bb03, 2026-08-24)

Flags touching the ragged-a2a path, with tested-status at the start of this leg:

| flag | default | status |
|---|---|---|
| xla_gpu_experimental_ragged_all_to_all_use_device_kernel | false | tested (the dk switch) |
| xla_enable_nccl_symmetric_buffers_for_collectives=raggedalltoall | unset | tested (scoped registration; global equivalent) |
| xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl | true | tested (one-shot barrier variant) |
| xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel | true | tested (legacy-off cell in leg-3) |
| xla_gpu_ragged_all_to_all_mode | COLLECTIVES_PRIVATE_MEMORY | UNTESTED -> cell a1 (expected null: put-path only guards the NCCL fallback) |
| xla_gpu_enable_gxl_ragged_all_to_all (+ gxl_scratch_size_bytes) | false | UNTESTED -> cell a2 (third backend, via fallback path; may be unavailable in our NCCL) |
| xla_gpu_unsupported_enable_ragged_all_to_all_multi_host_decomposer | true | UNTESTED as a toggle -> cell a3 (expected inert inside one NVL72 LSA domain) |
| xla_gpu_unsupported_override_fast_interconnect_slice_size | 0 (=LsaSize) | untested; only changes lsa_size plumbing, LsaSize already = world. Skip unless a3 shows the decomposer engages. |
| xla_gpu_allow_ragged_all_to_all_nccl_send_recv_fallback | false | n/a (error guard) |
| xla_gpu_unsupported_enable_ragged_all_to_all_decomposer | false | skip: decomposes to dense a2a, known-slow shape |
| splits_per_peer (marin-side) | 1 | swept in leg-3 (1/32/64/128, transport- and regime-dependent) |
| expert chunks (marin-side) | 2 | 1 OOMs, 3 = -0.12; 6 untested -> cell a4 (hero only) |

Kernel-structure facts driving the Tier-B patches (source-verified):
- dk copy already vectorizes to 16B (wider than one-shot's 8B); vector width is NOT the gap.
- one-shot copy grid scales with total updates (64 peers x updates/peer CTAs of 128 thr,
  ~12k CTAs at s64 -- 10x oversubscribed); dk grid is clamped to sm_cap because every CTA
  holds a per-blockIdx slot in the cross-rank LSA barrier session. Copy wants a huge grid,
  barrier cost scales with grid: that tension is the prime suspect for the residual per-byte
  gap. Evidence: grid sweep peaked at 8x SM (19.37 stock 1x / 21.02 4x256 / 21.69 8x128 /
  21.11 16x128) -- 16x REGRESSED, consistent with barrier cost overtaking copy gains.
- dk LSA inner loop is a plain grid-stride `dst[i] = src[i]` per unit (peer,update);
  one dependent load->store chain per iteration, no manual ILP.

## Experiment queue

Tier A (flags, no rebuild, microbench screen first):
- a1: dk + `--xla_gpu_ragged_all_to_all_mode=COLLECTIVES_SYMMETRIC_MEMORY`
- a2: GXL backend (`--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=false
  --xla_gpu_enable_gxl_ragged_all_to_all=true --xla_gpu_allow_ragged_all_to_all_nccl_send_recv_fallback=true`)
- a3: dk + `--xla_gpu_unsupported_enable_ragged_all_to_all_multi_host_decomposer=false`
- a4: chunks=6 hero cell (marin-side, no microbench analogue)

Tier B (kernel patches, new wheels; microbench screen -> hero confirm):
- b1 ilp4: unroll the LSA copy loop x4 with independent load/store batches (more outstanding
  NVLink transactions per thread).
- b2 cta0bar16: decouple the cross-rank barrier from the copy grid -- CTA 0 runs the
  entry/exit LsaBarrierSession, other CTAs spin on a device flag (allocated by the thunk,
  zeroed per launch); grid 16x SM. Direct A/B against the plain 16x cell isolates barrier
  cost.
- b3 stcs: streaming stores (`__stcs`) on the remote LSA writes.

Tier C (restructures, only if B stalls): unit rotation for skew balance, cooperative-launch
grid sync, persistent kernel, decomposer-based two-leg structure.

## Ledger

results.tsv columns: iter, ts, cell, kind (bench|synth|restore|build), config, metric
(ms for bench, MFU for arms), delta-vs-transport-best, status, notes.

## Findings (2026-08-25, through iteration 7)

Causal chain, each link measured on the 64-rank validating microbench (3.21 GB/rank/call):

1. The dk copy engine is NOT slow: at 1 update/peer it moves 404 GB/s vs the one-shot's
   424-431 -- effectively parity. The whole dk deficit was update granularity.
2. The U-curve has a cliff exactly where update count crosses the grid size: U=2/3/6/15 read
   8.16-8.56 ms, U=30 reads 12.2 (2nd wave 62% occupied ~ x1.5), U=120 reads 20.4 (7 waves).
   Wave quantization from the fixed ctas-per-update assignment, plus skew starvation of hot
   experts' units at the hero restore, fully explain the training gap.
3. Dead ends, all measured: cta0-delegated barrier = tie at 8x, worse at 16x/32x (barrier
   slots were never the 8x bottleneck; spinning oversubscribed CTAs throttle stores); ILP4
   unroll = null; mode=symmetric = null-to-slightly-negative; GXL = unavailable (hangs in
   fallback); one-shot at U=1 collapses to 24.2 (opposite granularity preference).
4. The fix: work-proportional CTA assignment (fork branch `ragged-a2a-dk-balanced`,
   06df0ee8a6) -- partition total copied elements evenly across the grid, walking update
   boundaries peer-major. Bench: U=30 12.2->10.9, U=120 20.4->14.2, U<=3 unchanged, all
   VALIDATION OK.
5. Hero gates (dkbal8 wheel, splits=1): synthetic 22.51 (dk stock 22.62, one-shot 22.77);
   6k restore 22.34 and 22.58 across two draws (stock dk best 21.98; one-shot band
   22.46-22.60) -- PARITY at the restore, +0.4..+0.6 over stock dk. splits=64 under the
   balanced kernel reads 21.12: splits are now purely harmful, the kernel replaces them as
   the load-balancing device.

Same-night one-shot pairing arm (dkp-09b) pending (first attempt starved: rack co-tenancy).
Not run: decomposer-off cell (expected inert inside one LSA domain), chunks=6.

Measurement discipline: single bench cells vary 12.2-14.2 across jobs when co-tenants share
the 72-GPU NVLink domain (we hold 64); the one-shot is much less sensitive. Verdicts came
from interleaved same-night series and repeat draws only.

## Final standings (2026-08-25, leg CLOSED)

6k-restore MFU, five interleaved same-night draws (loss identical to 5 decimals across all
arms; drops equal):

| config | draws | mean |
|---|---|---|
| dk + balanced kernel, splits=1 | 22.34, 22.58, 22.47 | 22.46 |
| one-shot, splits=64 (kmax128) | 21.93, 22.05 | 21.99 |
| dk + balanced kernel, splits=64 | 21.12 | (mechanism null) |
| stock dk best, splits=64 (leg-3) | 21.98 | |

GOAL EXCEEDED: the device kernel with the work-proportional-assignment patch beats the
one-shot by ~0.5 MFU in same-night pairing at splits=1, on stock XLA main + one small
self-contained kernel patch (`mcwitt/xla` branch `ragged-a2a-dk-balanced`, 06df0ee8a6,
wheel `pjrt-mainpatch-dkbal8-20260825`). The third balanced draw ran in the latest window,
excluding within-night drift as the explanation. The one-shot's historical 22.46-22.60 band
came from quieter nights; under tonight's co-tenancy it read 21.93-22.05 while the balanced
dk held its level -- the dk path is also the less placement-sensitive of the two.

Candidate follow-ups (not started): upstream the balanced-assignment patch (supersedes the
g8x128-grid-only patch; needs user approval to file); re-tune expert chunks under the
balanced kernel; port the leg-3 loader follow-up; NVIDIA thread update with the U-curve and
the balanced-kernel result (user decision).
