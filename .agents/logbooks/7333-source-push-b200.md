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
| SPB-001 | Census + gate: enough of the #6841 staged semantic path runs on GB200 (planner, transport, XLA combine; which Pallas kernels lower on sm_100) to measure staged source-push forward at the #7279 shapes vs ring_cute/a2a_cute bests; source-push reduces the ~48% collective share | CLOSED 2026-07-17 — census GREEN (staged path runs on sm_100/aarch64), gate FALSIFIED (1.98× slower best-case, ~530× honest API; see part 3/4 entries) |
| SPB-002 | Tuned Blackwell compute epilogues (unblock #6933 swizzle assertion or CuTeDSL via `cutlass_call` per #7282) close the gap to fused local compute | MOOT — compute stages already at 1.4–1.7 PF/s, only 14% of path time |
| SPB-003 | Device-side planner removes the ~380 ms/plan host gate (SPF-005 carry-over) | MOOT — planner measured 14.2 s/call at production routing volume, but even plan-free execution loses the gate |

## Entries

(append-only below)

### 2026-07-17 — SPB-001 (part 1): branch assembly and census plan

Assembled the working state for the GB200 census + gate on
`research/mcwitt/7333-source-push-b200`:

- Merged `origin/mcwitt/moe-standalone-ep` (65469cf38): the #7279 standalone
  MFU harness (`experiments/grug/moe/standalone/grug_moe_mfu.py`) plus the
  `ring_cute` / `ragged_all_to_all_cute` / `sonic_cute` backends.
- Merged `origin/codex/blackwell-source-push-stack` (e21dd73c5): the #6933
  staged source-push stack, including `source_push_inbox_blackwell.py`
  (B200/B300 tuning) and the `pallas_mgpu_source_push_blackwell` public
  implementation. Conflicts only in `_moe/common.py`
  (`_EP_MOE_IMPLEMENTATIONS` union) and trivially in `grug_moe.py`.
- Did NOT merge `research/mcwitt/6841-source-push-followup`: its semantic-path
  files diverged from the blackwell stack by thousands of lines
  (`source_push_forward.py` 1.9k-line diff, `source_push_mlp.py` 3.2k) and its
  Pallas kernels are `mgpu.wgmma` sm_90a-only. Portable SPF wins (SPF-004 XLA
  gather-sum combine, SPF-001 dy bf16) can be cherry-picked later if the gate
  passes.
- Extended `bench_source_push_forward_public_compare.py` `PUBLIC_EP_BACKENDS`
  with `ring_cute` / `ragged_all_to_all_cute` (e70df4f09) — the bench routes
  through public `moe_mlp`, so the merged dispatcher provides them.

Gate instrument: `lib/levanter/scripts/bench/bench_source_push_forward_public_compare.py`
(single-process, `jax.devices()[:ep_size]`, per-implementation timing +
correctness vs reference). Production gate shape from #7279 (64-GPU config,
per-32-GPU copy at EP4): `--ep-size 4 --tokens-per-rank 65536 --hidden-dim
5120 --intermediate-dim 2560 --experts-per-rank 16 --topk 4
--capacity-factor 1.0`. Paper check of `source_push_public` validation: EP4 ∈
2..8, I2560 % 128 = 0, GB200 in the blackwell device allowlist — passes.

Census instrument for per-stage anatomy: `bench_blackwell_source_push_forward_smoke.py`
(stages: input_prepare, destination_x_transport, w13, w2, return_transport,
combine).

### 2026-07-17 — SPB-001 (part 2): GB200 census — staged path RUNS on sm_100; eager-bench pitfall

Venue: 4×GB200 (one tray) via `dev_gpu.py` holder
`/mwittmann/dev-gpu-mwittmann-spb-b200` on cw-us-east-08a; aarch64 pod, jax
0.10.1 cuda13, `uv sync --all-packages --extra=gpu`. QuACK/CuTeDSL import
clean on aarch64.

**Census: the #6933 staged Blackwell source-push path lowers and runs on
GB200** (sm_100, aarch64) at EP4. Correctness compare at d2560/I1280/EP4/4k
tokens-per-rank, cf 1.25, roughly_balanced, `blackwell_staged` +
`staged_device_sync`:

```
bench_source_push_forward_public_compare.py --ep-size 4 --tokens-per-rank 4096 \
  --hidden-dim 2560 --intermediate-dim 1280 --experts-per-rank 16 --topk 4 \
  --capacity-factor 1.25 --routing roughly_balanced --entries-per-rank 80 --inbox-slots 24 \
  --source-push-implementation blackwell_staged --source-push-execution-mode staged_device_sync \
  --public-implementations ring,ring_cute,ragged_all_to_all_cute
```

vs ring / ring_cute / a2a_cute: mean |Δ| 0.224 / 0.299 / 0.254, max |Δ| 4–8
(bf16 accumulation-order scale; inter-backend spread is the same order),
dropped_route_delta = 0 everywhere. Two gotchas: (1) the package-private path
needs explicit queue sizing — defaults die with "entries_per_dst=2 but
required 75" at this shape; (2) `--entries-per-rank`/`--inbox-slots` must be
set from the shape, the public path auto-probes them.

**Bench pitfall (recorded as a durable methodology note):** the compare
bench's public timing called `moe_mlp` eagerly. At the production shape this
is overhead-bound, not compute-bound: ring_cute EP4 d5120/65k-tokens measured
13.6 s eager vs 27 ms jitted (~500×). Added `--jit-public` (a3d4b12b1) which
jits non-source-push backends (how production calls them); source-push stays
un-jitted since host planning is part of its honest public cost. Any prior
absolute numbers from this bench's eager public timing should be treated as
overhead-dominated at large shapes.

**Gate baselines (jitted, production shard shape #7279: EP4, d5120, I2560,
65536 tokens/rank, topk 4, cf 1.0, roughly_balanced, 4×GB200):**

| implementation | median fwd | useful TFLOP/s/rank | dropped |
| --- | --- | --- | --- |
| ring_cute (jit) | 26.96 ms | 765 | 589 |
| ragged_all_to_all_cute (jit, NCCL-path flag) | 33.42 ms | 617 | 589 |

Command: same shape flags plus `--public-timing --jit-public
--public-call-mode direct --public-implementations
ring_cute,ragged_all_to_all_cute --warmup 2 --steps 10 --repeat-runs 3`,
`XLA_FLAGS=--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=false`.

Source-push arms (preplanned amortized + direct honest-API) in flight.

### 2026-07-17 — SPB-001 (part 3): gate numbers — source-push loses 2× at best case, 530× honest

Same shape/venue as part 2 (EP4, d5120, I2560, 65536 tokens/rank, topk 4, cf
1.0, roughly_balanced, 4×GB200). Source-push arms via `--public-timing
--public-implementations pallas_mgpu_source_push_blackwell`:

| arm | median fwd | vs ring_cute |
| --- | --- | --- |
| ring_cute (jit, direct) | 26.96 ms | 1.00× |
| ragged_all_to_all_cute (jit, direct) | 33.42 ms | 1.24× |
| source-push blackwell, preplanned (planner amortized, `staged_device_sync`) | 53.5 ms | 1.98× |
| source-push blackwell, direct (honest API, host planner per call) | 14.27 s | ~530× |

- Preplanned = the impossible-best case (static routing, plan built once,
  transport + staged local compute only). It still loses to ring_cute by
  26.5 ms/layer-forward. Over 48 layers that is ~+1.3 s/step forward-only.
- Direct = the honest public API. Host planning at this shape (262144
  routes/rank × 4 ranks) costs ~14.2 s/call — SPF-005's 380 ms/plan scaled to
  production routing volume. Hardware-independent, unchanged on GB200.
- Roofline context: ring_cute's 27 ms is ~31% of bf16 peak on useful FLOPs
  against a ~6 ms NVLink comm floor (2×2.68 GB/rank) + ~8 ms GEMM floor —
  the backend ladder itself has headroom, which makes the source-push deficit
  worse than it looks.

Ops note: the full-shape (65k tokens/rank) `bench_blackwell_source_push_forward_smoke`
run OOM-killed the 128 GB pod (fp32 host reference arrays), which took down
the dev-gpu holder task and the reservation. Re-allocated with `--memory
240GB`; anatomy runs at 32k tokens/rank.

### 2026-07-17 — SPB-001 (part 4): anatomy + verdict — GATE FAILS, thread ends

Per-stage anatomy at 32768 tokens/rank (same shape otherwise; second holder
`/mwittmann/dev-gpu-mwittmann-spb-b200-2`, `--memory 240GB`;
`bench_blackwell_source_push_forward_smoke --stage-timing`, #6933-tuned knobs:
send=4, worker=32, slots=24, blocks 64/128/128, groups/job=2, entries 640):

| stage | median | share of stage sum |
| --- | --- | --- |
| input_prepare | 20.31 ms | 57% |
| destination_x_transport | 4.57 ms | 13% (298 GB/s effective) |
| w13 | 4.02 ms | 11% (1.73 PF/s) |
| w2 | 2.50 ms | 7% (1.39 PF/s) |
| return_transport | 2.85 ms | 8% |
| combine | 1.44 ms | 4% |
| stage sum | 35.7 ms | |
| end-to-end steady state | 46.9 ms | (+11 ms stage-glue/sync overhead) |

Same-shape jitted `ring_cute`: **13.96 ms** (738 TFLOP/s/rank). Scaling check:
ring_cute is linear in tokens (14.0→27.0 ms for 32k→65k); source-push is
sub-linear (46.9→53.5 ms) — fixed marshalling/sync overhead dominates it.

Findings:

1. **Compute is NOT the problem.** The #6933-tuned upstream Mosaic
   `blackwell_ragged_dot` kernels run at 1.4–1.7 PF/s on sm_100 — SPB-002
   (better compute epilogues) has almost nothing to win: w13+w2 are only
   6.5 ms of a 46.9 ms path.
2. **The staged path loses on marshalling and serialization**: 57% of stage
   time is `input_prepare` (device-side raw-token pack before transport), and
   ~11 ms more is stage-sync glue. Peer transport itself (7.4 ms at ~300 GB/s)
   is fine but can't be overlapped with compute — `staged_device_sync`
   serializes stages, and the fused persistent peer-ref kernels that would
   overlap are unsupported in Blackwell warpgroup lowering (#6933).
3. **Perfect-overlap floor** max(compute 6.5, transport 7.4) ≈ 7.4 ms < ring_cute
   14.0 ms is real on paper, but reaching it requires exactly the machinery
   Blackwell doesn't support (fused persistent kernels) plus a device-side
   planner (host planner ≈ 14.2 s/call at 65k tokens/rank, SPB-003). #7279's
   hypothesis 1 (chunked-pipeline decomposition in XLA-land) targets the same
   overlap headroom without either blocker.
4. Smoke-bench tolerance note: at cf 1.0 the smoke's internal reference
   fails its 2.0 max-abs output tolerance (max |Δ| 256, mean 12.1) while
   H-group diffs are tiny (max 1.0, mean 0.017) and dropped counts match the
   public backends exactly (589) — consistent with a drop-set mismatch
   between the smoke's reference and the staged path at capacity, not a
   kernel numerics bug (the cf 1.25 public compare in part 2 matched at bf16
   noise with dropped_route_delta 0). Not chased further given the verdict.

**Verdict: SPB-001 gate FAILS.** Best-case (preplanned, tuned) staged
source-push forward is 1.98× ring_cute at the production shape (53.5 vs
27.0 ms) and 3.4× at 32k (46.9 vs 14.0 ms); the honest API is ~530× (host
planner). Per the issue's stop criterion, the source-push thread on B200 ends
here; the ~48% collective headroom belongs to #7279's other avenues (chunked
pipeline, overlap, leg reduction).

## Hypothesis queue (final)

| ID | Hypothesis | Status |
| --- | --- | --- |
| SPB-001 | Staged source-push beats ring_cute/a2a_cute at #7279 shapes | FALSIFIED — 1.98× slower best-case, ~530× honest API |
| SPB-002 | Tuned Blackwell compute epilogues close the gap | MOOT — compute already 1.4–1.7 PF/s, only 14% of path time |
| SPB-003 | Device-side planner removes host-plan gate | MOOT for this thread — planner is 14.2 s/call at production routing volume, but even plan-free execution loses the gate |
