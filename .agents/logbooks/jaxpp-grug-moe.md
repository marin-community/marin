---
topic: jaxpp-grug-moe
description: JaxPP pipeline parallelism for experiments/grug/moe training
author: dlwh
---

# JaxPP Grug MoE: Task Logbook

## Scope
- Goal: Implement and debug an NVIDIA/JaxPP-based pipeline-parallel training path for `experiments/grug/moe/train.py`, then test it on 4x 8xH100 CoreWeave nodes with the May d=2560 shape.
- Primary metric(s): A pipelined train step can compile/run on the intended NVIDIA GPU environment; optimizer/loss semantics match the non-pipelined path for equivalent global batches; MFU is captured for relevant JaxPP schedules.
- Constraints: Keep default Grug MoE behavior unchanged when pipeline config is unset; do not add broad dependency churn until the runtime environment decision is explicit; require relative-L2 error at most `0.002` for loss and every gradient leaf.
- Coordinating issue/PR: https://github.com/marin-community/marin/issues/7024

## Current TL;DR
- `GRUG-JAXPP-001`: Explicit JaxPP MPMD training is implemented behind `GrugTrainerConfig.pipeline.implementation="explicit_mpmd"` with stage-local model/optimizer state, contiguous layer splits, explicit GPipe and explicit `std_1f1b` schedules. Milestone implementation commit: `abd979b82a` (`[grug] Add explicit JaxPP pipeline training`).
- The requested 24-layer, d2560, 256-expert, top-k 4 shape now fits and executes on 4x 8xH100 CoreWeave east02 after optimizer state is initialized from stage-local pipeline weights instead of from the full 61B-param model.
- Best completed throughput point: explicit MPMD `std_1f1b`, 24 layers, 64 experts, top-k 4, seq 4096, batch 8192, four physical/logical stages, 256 microbatches, ring EP, CuTe FA4 attention, Pallas-Triton ragged dot with eight warps, and `0.70` prealloc on RNO2A. Run `/dlwh/iris-run-job-20260711-080751/grug-train-jaxpp-rno2a-ring-l24-e64k4-b8192-s4096-p4m256-20260711-0107` reported mean MFU `18.2583`, p50 `18.3654`, p90 `18.3830`, and latest throughput `414,059.10` tokens/s. The `0.1248`-point gain over m128 confirms occupancy saturation, `1.7417` points below 20.
- RNO2A is healthy and performance-equivalent to east02 for this workload. The exact batch128/m4 baseline reported mean MFU `7.5946` and `170,329.97` tokens/s, within noise of the east02 `7.5981` result. NCCL debug logs confirmed `NET/IB/.../GDRDMA`, ruling out socket fallback.
- Profile captured for explicit MPMD GPipe at the stable batch-96 seq4096 point after commit `a0f8130985` (`[grug] Upload explicit MPMD profile artifacts`). W&B run: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-explicit-profile-l24-e256-b96-s4096-gpipe-xla070-artifact-20260709-0018>. Artifact: `marin-community/marin_moe/jaxpp-explicit-profile-l24-e256-b96-s4096-gpipe-xla070-artifact-20260709-0018-profiler:v0`.
- Profile readout: communication-dominated exclusive timeline, with `48.75%` communication, `45.10%` compute, and `6.15%` stall. Largest exclusive kernel is `ncclDevKernel_SendRecv` (`104` calls, about `7.35s` total exclusive in the profile window), followed by all-gather and reduce-scatter collectives.
- Follow-up explicit MPMD `std_1f1b` profile at the same 24L/256E/seq4096/batch96 shape also remains communication dominated. W&B run: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-explicit-profile-l24-e256-b96-s4096-std1f1b-xla070-artifact-20260709-0506>. It reported mean MFU `7.2713`; profile breakdown was `47.40%` communication, `46.91%` compute, and `5.69%` stall. `SendRecv` remained the top op, with average exclusive duration `64.5ms` versus `70.7ms` in the GPipe profile. The raw trace totals are not directly comparable because the `std_1f1b` trace captured a longer wall window than the GPipe trace.
- A fresh RNO2A batch256/m8 `std_1f1b` profile reported mean MFU `9.7654`, `226,714.36` tokens/s, and `3,199,065.76` GFLOP/s. The exclusive breakdown moved to `55.92%` compute, `38.96%` communication, and `5.12%` stall; `SendRecv` remained the top collective at `59.99ms` average. W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-profile-std1f1b-l24-e256-b256-s4096-p4m8-20260710-0238>. Artifact: `marin-community/marin_moe/jaxpp-rno2a-profile-std1f1b-l24-e256-b256-s4096-p4m8-20260710-0238-profiler:v0`.
- A profile of the 64-expert batch448/m14 shape reported mean MFU `11.2541` over the whole run and `11.3473` during profile steps 8-11. Its exclusive breakdown was `60.34%` compute, `33.85%` communication, and `5.81%` stall. Average `SendRecv` duration fell to `41.69ms`, but the average pre-op gap remained `628.72ms`, so pipeline rendezvous remains exposed. W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-profile-ring-l24-e64k4-b448-s4096-p4m14-20260710-1010>. Artifact: `marin-community/marin_moe/jaxpp-rno2a-profile-ring-l24-e64k4-b448-s4096-p4m14-20260710-1010-profiler:v0`.
- Decoding the XPlane-embedded HLO identified the dominant compute kernels as reference attention rather than expert GEMMs. The largest fusions materialize or mask bf16 `[4,20,4096,4096]` attention tensors in forward and backward; attention GEMMs are also among the leading kernels. Replacing reference attention with `gpu_fa4_cute` raises the matching batch448/m14 run from `11.6568` to `15.9684` mean MFU and cuts mean step duration from `6.9798s` to `5.1014s`.
- A CuTe FA4 profile at batch512/m16 reports `52.79%` compute, `39.61%` communication, and `7.59%` stall. Average `SendRecv` duration falls to `25.83ms` and its average pre-op gap to `426.17ms`. The largest remaining compute kernel is the Pallas-Triton ragged-dot `_lambda_` at `1.852ms` average, followed by FA4 forward at `1.530ms`. W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-profile-fa4cute-l24-e64k4-b512-s4096-p4m16-20260710-1110>. Artifact: `marin-community/marin_moe/jaxpp-rno2a-profile-fa4cute-l24-e64k4-b512-s4096-p4m16-20260710-1110-profiler:v0`.
- Pallas-Triton geometry controls did not produce a larger gain. Eight warps improves the default `block_k=32` point from `16.1040` to `16.2005` mean MFU, but increasing `block_k` to 64 regresses to `16.0189`. The remaining performance target is pipeline communication/dependency overlap rather than a broader blind tile sweep.
- Explicit `interleaved_gpipe` now supports more logical stages than physical ranks and follows the pinned JaxPP per-rank task queues. At 8 logical/4 physical stages, batch128/m4 reached mean MFU `9.2547` and batch192/m6 reached `10.2190`; batch224/m7 hit a compile-complexity cliff. The four-stage `std_1f1b` schedule remains faster and reaches `16.2005` MFU with CuTe FA4 and eight-warp Pallas-Triton ragged dot.
- Two 12-layer physical stages fit the exact L24/e64/top-k4/seq4096/b512/m16 model by using the extra `data=2` axis for FSDP, but regress matched mean MFU from `16.2005` to `12.9050` (`-20.34%`); p50 is `13.3542`. Halving pipeline boundaries does not offset data-axis collectives and larger stage tasks. Keep four six-layer stages.
- Unequal 24-layer splits, a narrower expert axis, ragged all-to-all, and DeepEP did not beat the ring baseline. Splits with a 7-layer stage either OOMed or failed to reach execution; expert-axis 4 stopped in compilation; ragged all-to-all reached only `3.6743` MFU at batch256/m8. DeepEP now builds and executes its SM90 transport on RNO2A, but training is numerically unstable: both 8-expert and 64-expert non-pipelined controls have a finite step-0 loss and NaN on step 1, while explicit JaxPP is NaN on step 0.
- Automatic JaxPP eager 1F1B is functionally executable but fails the accepted BF16 numerical gate. A four-rank H100 comparison with FP32 master parameters and the production BF16 compute cast passed loss at `3.44e-7` relative-L2, but only `34/72` gradient leaves passed `0.002`; 38 failed. The maximum was `0.1137` on a near-zero router gradient, while the largest non-router error was `0.01136`. The FP32 control passed all 72 leaves with maximum `0.001863`, isolating the failure to production mixed-precision lowering rather than schedule algebra. Do not scale automatic eager under the current policy. Standard 1F1B still deadlocks in DIME transfer setup, and zero-bubble remains blocked by multi-hour backward compilation.
- The Sonic whole-MLP non-return was reduced below JaxPP to concurrent `jax-tvm-ffi` handler invocation. Plain JAX reproduces the same fsdp=2/3 boundary with a single QuACK grouped GEMM at only three experts, one token per expert, and 8x8 matrices. A per-handler host mutex fixes the minimal case, a full 65,536-assignment FSDP-8 forward/backward control, and the exact four-stage target. Proper whole-MLP Sonic is operational but reaches only `13.9598` mean MFU, `2.2407` points below the `16.2005` ring-EP winner. Workaround/setup snapshot: `89bae1453c`.
- The exact proper-Sonic profile is `37.31%` compute, `50.47%` communication, and `12.22%` stall. AllGather remains nearly unchanged from old Sonic at `47.508s` aggregated over `48,192` device-track calls, 2.35x the ring profile's call count. Staggered once-per-stage materialization improved the reduced smoke but regressed the exact target from `13.9598` to `13.8155` MFU, confirming that the fine-grained gathers were better overlapped than the hoisted critical-path gather.
- Explicit Triton token-routing kernels remove the profiled rank-2 floating scatter families from `ring_fused` and pass H100 value/VJP parity, but a pinned 10-step reduced A/B found no throughput gain. After excluding isolated duration outliers, bulk ring averaged `9.1735` MFU and `0.214968s`; `ring_fused` averaged `9.1595` MFU and `0.215344s`. Do not scale this backend to L24 without a new kernel-level signal.
- Prioritizing explicit pipeline transfer construction ahead of local QB/loss/gradient accumulation is a clean negative. The pinned reduced run averaged `8.9243` MFU and `0.220987s`, versus `9.1735` and `0.214968s` for default ordering. Construction order is not the missing overlap mechanism; the next credible schedule change must split the combined activation/weight-gradient backward task.
- Splitting backward into input-gradient and weight-gradient tasks is operational and gradient-correct but a hard performance negative. The reduced H100 gate averaged `6.5356` central MFU and `0.301733s`, versus `9.1735` and `0.214968s` for the combined-backward control. Independent block rematerialization costs more than zero-bubble placement can recover.
- Exact two-chunk bulk ring preserves output/drop/VJP semantics and takes its fast path at the target EP8 microbatch, but direct H100 timing regresses forward by `5.30%` and forward-backward by `13.98%`. XLA latency hiding does not offset doubled collective and Pallas launch counts; do not register this backend for training.
- EP-local QuACK grouped GEMM is the first direct-kernel result to clear the performance gate: at the exact e64/top-k4/d2560 EP8 microbatch it improves median forward by `16.0%` and forward-backward by `10.8%`. A one-H100 exact-shape repro proves the output difference comes from QuACK's fused SwiGLU fast approximate `exp2`/reciprocal, not the EP adapter or grouped GEMMs: W13 and W2 are bitwise identical with shared inputs, while fused activation numerics reproduce the EP8 error. It remains benchmark-only pending an explicit decision about accepting approximate activation semantics.
- Approximate SwiGLU is now an explicit opt-in backend, `ring_quack_approx`. A paired 10-step one-node L2 training gate is finite and trajectory-stable: final loss differs by `+0.01356%`, while mean MFU improves from `21.1689` to `22.0161` (`+4.00%`) and median duration falls `4.04%`. The gain is real but insufficient by simple extrapolation to move the current L24 best above 20 MFU, so a reduced JaxPP gate is still required before the exact target.
- The contemporary reduced JaxPP A/B makes QuACK a pipeline performance negative: after excluding runtime stalls above `300ms`, ring averages `9.4613` MFU and QuACK `9.4786` (`+0.18%`), with p50 changing only `+0.09%`. Do not run L24 QuACK. The next transport hypothesis is FP8 over the existing ring wire, not another ordinary ragged-all-to-all retry.
- Packed FP8 inter-stage transfers are a modest directional win in the reduced L8 pipeline gate. Over 16 matched timed steps after excluding the two samples above `300ms`, p50 MFU improves from `9.4926` to `9.6398` (`+1.55%`) and p90 from `9.5406` to `9.6754` (`+1.41%`). All losses are finite and the final relative loss delta is `+0.00395%`. This is not enough to justify the exact m256 target directly; the next gate is L24/b512/m16 at the same 32-sequence microbatch.
- FP8 pipeline transfer remains positive at realistic L24 stage depth but is too small for the objective. The matched b512/m16 confirmation improves clean p50 MFU from `16.2447` to `16.5358` (`+1.79%`) and p90 by `1.81%`, with finite loss and `+0.0343%` final relative drift. Applying that gain to the `18.2583` m256 best projects only about `18.59` MFU. Do not launch exact m256 FP8; resume with a different transport/overlap mechanism.
- FP8 expert GEMMs are stable in reduced JaxPP training but too small for the objective. At L8/d2560/e64/top-k4/seq4096/b512/m16, FP8 raises mean MFU from `16.1532` to `16.4344` (`+1.74%`), reduces mean duration `1.69%`, and raises throughput `1.74%`. Loss remains finite; final relative loss drift is `+0.2122%`, accepted by the user for this approximate FP8 experiment. Applying the gain to the `18.2583` best projects only about `18.58` MFU, so do not scale unchanged FP8 expert GEMMs to L24.
- Transformer Engine NCCL_EP is no longer a pipeline candidate under the accepted numerical policy. Exact 524,288-row transport reaches only `5.0245%` MFU on the reduced pipeline versus `9.4180%` ring. Bounded 81,920-row transport is `1.37-1.45x` faster than ring in the direct full-MLP gate, but its best FP32-reference token-gradient relative-L2 is `0.2909%`, above the accepted `0.2%` ceiling. Replacing dispatch backward's token dgrad with FP32 TE combine-forward reproduces the same `0.2909485%` error exactly, so that hybrid is not a distinct numerical path.
- Cross-microbatch exact-ring fusion does not justify grouped JaxPP stage tasks. At group size 2, fused value-and-grad is numerically exact but only `1.0889x` faster than asynchronously queued single-microbatch calls, below the `1.11x` gate and projecting about `19.65` MFU after the four-stage bubble cost. Group size 4 falls to `1.0819x`, below its `1.134x` target, while shared W13/W2 gradient errors rise to `0.3518%`/`0.3375%`, above the accepted `0.2%`. Explicit dispatch/expert/combine phasing also regresses forward to `0.943-0.946x`.
- CUTLASS DSL 4.5.2's generated `_isa` helper caused the JaxPP-localized CuTe FA4 full-block exit `139` by constructing MLIR type wrappers while probing their type. Replacing that probe with the `isinstance` check used by CUTLASS DSL 4.6 clears both matched two-rank BF16 and FP8 full-block gates. The production JaxPP setup now applies the guard to a private venv copy so it cannot mutate UV's shared package cache. Integration snapshot: `30183d2d4f`.
- JAX 0.11's public device-initiated ragged-all-to-all is bitwise exact and `7.4349x` faster than the private-memory control at the target EP8 payload, but it does not complete reduced four-stage JaxPP training. Standard 1F1B, GPipe, transfer-priority, separate forward/reverse DIME communicators, NCCL implicit launch ordering, and disabled receive-buffer reuse all stop with ranks split between DIME control/receive waits and PJRT execution. Prewarming either DIME communicators or only CUDA streams before the first local MoE is allocator-incompatible: both prevent rank 2 from reserving the 51.469 GiB device-ragged symmetric arena. Two-rank, four-rank, bidirectional, and 16-microbatch synthetic compositions pass exactly; the repeated gate covers 96 transfers and 128 stage tasks. The current treatment reserves every stage's symmetric arena with a local ragged warmup before ordered DIME creation. Marin bug #7655 contains the pinned lower-bound package; no NVIDIA issue was filed.

## Hypothesis Queue

### Active
- `GRUG-JAXPP-002`: Router histograms need a dedicated cross-microbatch reducer before full metric parity is safe. Evidence: [2026-07-07 22:45 PDT - initial implementation](#2026-07-07-2245-pdt---initial-implementation). Next test: implement/validate a `SummaryStats` merge or compute summary metrics outside the pipeline loop.
- `GRUG-JAXPP-006`: Pipeline rendezvous remains exposed after increasing occupancy and reducing expert count. Evidence: the 64-expert batch448/m14 profile reduces communication share from `38.96%` to `33.85%` and average `SendRecv` duration from `59.99ms` to `41.69ms`, but its average pre-op gap remains `628.72ms`. Next test: reduce stage dependency wait or increase overlap at the working m14 boundary.
- `GRUG-JAXPP-008`: Attention was the largest compute bottleneck, and CuTe FA4 removes most of it, but the best full shape still needs another `9.5%` relative gain to reach 20 MFU. Evidence: HLO attribution maps the largest fusions to reference attention; `gpu_fa4_cute` raises the matching batch448/m14 result from `11.6568` to `15.9684`, and occupancy scaling reaches `18.2583` at batch8192/m256. Approximate QuACK is finite-step stable but neutral under the reduced pipeline, so it is no longer a scaling candidate.
- `GRUG-JAXPP-012`: More standard-schedule microbatches at fixed microbatch size 32 reduce the pipeline bubble but saturate below target. Evidence: b1024/m32 reaches `16.6677`, b2048/m64 `17.4430`, b4096/m128 `18.1334`, and b8192/m256 only `18.2583` mean MFU. Decision: stop batch scaling; a separate overlap/kernel gain is required.

### Blocked
- `GRUG-JAXPP-016`: Public device-initiated ragged-all-to-all has enough direct transport headroom to exceed 20 MFU, but the reduced four-stage JaxPP integration deadlocks before its first loss under standard 1F1B, GPipe, transfer-priority ordering, directional DIME communicators, NCCL implicit launch ordering, and disabled receive-buffer reuse. Synthetic two-rank and four-rank gates pass, including 16 microbatches, 96 transfers, and 128 stage tasks. Blocker: isolate the additional full-training condition and fix #7655. Current test: reserve each stage's device-ragged symmetric arena before globally ordered DIME creation in parent `/dlwh/iris-run-job-20260726-135924`. Resume when the L8/d2560/e64/top-k4/seq4096/b512/m16 gate completes finite steps with relative-L2 at most `0.002` for loss and every gradient.
- `GRUG-JAXPP-009`: DeepEP transport as a replacement for ring EP. Blocker: the pinned DeepEP FFI now builds and launches on RNO2A after adding CUDA runtime linkage, attention-only remat, and a 512-thread dispatch kernel, but both 8-expert and 64-expert non-pipelined controls become NaN after one finite update and the explicit pipeline is NaN on its first step. Resume after a DeepEP dispatch/combine VJP or runtime-state correctness fix.

### Falsified / Dead End
- `GRUG-JAXPP-005`: Automatic eager 1F1B executes, and its FP32 schedule-algebra control passes all 72 gradients, but the exact FP32-master/BF16-compute gate passes only `34/72` gradients at the accepted `0.002` ceiling. The largest non-router relative-L2 is `0.01136`. Do not run automatic eager performance scaling without a mixed-precision numerical correction. Evidence: [2026-07-25 23:31 PDT - automatic eager fails production mixed-precision parity](#2026-07-25-2331-pdt---automatic-eager-fails-production-mixed-precision-parity).
- Two 12-layer stages with `data=2` FSDP are operational at the exact sequence-4096 target but regress matched b512/m16 mean MFU by `20.34%` (`16.2005 -> 12.9050`). Keep the four-stage topology. Evidence: [2026-07-25 23:51 PDT - two-stage FSDP topology regresses](#2026-07-25-2351-pdt---two-stage-fsdp-topology-regresses).
- Delaying data-parallel gradient reduction until after microbatch accumulation is performance-neutral at batch128/m4: mean MFU `7.5648` versus `7.5946` baseline.
- Reusing opaque VJP residuals removes explicit backward recompute in a tiny smoke but does not fit the 61B shape. `save_moe` exhausts memory in forward residual storage; `recompute_all` moves the failure to backward scratch, including at 8 microbatches and with XLA preallocation disabled.
- Unequal stage splits do not expand the batch384/m12 capacity point. `7/5/7/5` OOMed stage 2 backward on a `19.65 GiB` request; `7/6/6/5` OOMed stage 0 forward or failed to reach execution at higher preallocation. Six layers per physical stage remains the practical limit.
- Reducing expert parallelism from 8 to 4 GPUs per stage did not reach execution after 8m51s of compilation. Ragged all-to-all was functional but regressed batch256/m8 to `3.6743` MFU and `73,425.13` tokens/s.
- CuTe FA4 does not clear the 8-logical/4-physical interleaved GPipe m8 compile cliff. The b256/m8 run compiled logical stages 0-3, then made no progress for 8m56s after starting `grug_interleaved_mb0_stage4_forward`; it was stopped without producing a train metric.
- Increasing the FA4 microbatch size from 32 to 40 at fixed m16 regresses mean MFU from `16.1040` at batch512 to `15.6037` at batch640. More standard-schedule batch capacity is not the route to 20 MFU at this shape.
- XLA ragged dot regresses the CuTe FA4 batch512/m16 point from `16.1040` to `8.1438` mean MFU. Pallas-Triton remains the grouped expert GEMM backend.
- Increasing Pallas-Triton `block_k` from 32 to 64 at eight warps regresses mean MFU from `16.2005` to `16.0189`; halving the kernel's K-loop iterations does not improve the full step.
- Proper whole-MLP Sonic without EP is operational but reaches only `13.9598` MFU. Staggering once-per-stage materialization regresses the exact target further to `13.8155`; eliminating repeated gather calls does not beat their existing overlap.
- XLA's zero-copy one-shot ragged all-to-all flags are accepted, but RNO2A pods cannot create the required exportable `FABRIC+POSIX_FD` CUDA VMM allocation. `cuMemCreate` returns `CUDA_ERROR_NOT_PERMITTED` and the process exits 139 before compilation completes. JAX 0.11 keeps symmetric output memory as the implementation, so upgrading JAX alone does not remove this blocker.
- The first true streamed `ring_ppermute` backend passes CPU output/gradient parity and removes global activation/output tensors, but its H100 smoke is numerically unstable after one finite step and over 1,100x slower on that first timed step because EP8 creates many small native XLA ragged GEMMs and collective permutes.
- Owner-local bulk-ring combine passes CPU output/gradient parity and removes the 640 MiB global output temporary, but seven owner-directed permutations make even the reduced L8 stage-3 backward graph compile for about 28 minutes without completing. It was stopped without a metric.
- Expert-axis 4 at e64 reaches every stage's first backward compile but then stalls in accumulation compilation for over 24 minutes; it was stopped after 34m37s with no metric. Doubling local expert state remains operationally intractable.
- Explicit Triton dispatch-backward and combine forward/backward routing kernels pass H100 value/VJP parity and remove rank-2 floating scatters from optimized HLO, but the pinned reduced A/B is neutral: central bulk-ring mean `9.1735` MFU versus `9.1595` for `ring_fused` (`-0.15%`). The apparent two-sample fused win did not replicate over ten steps.
- Enqueuing forward and backward pipeline transfers before local accumulation tasks regresses the pinned reduced median MFU by `2.90%` (`8.9006` versus `9.1660`) and increases median duration by `2.98%`. JaxPP task construction order alone does not improve rendezvous overlap.
- Input-gradient-first backward passes CPU loss, `d_hidden`, parameter-gradient, accumulated-gradient, and Adam-update parity in both remat modes and executes on 32 H100s, but regresses reduced central MFU by `28.75%` (`6.5356` versus `9.1735`). The extra per-block rematerialization dominates the intended bubble reduction.
- A private exact two-chunk bulk-ring prototype passes BF16 output/drop and four-input VJP parity, including overflow and globally consistent fallback. At the exact e64/top-k4/d2560 EP8 microbatch it regresses median forward from `10.389ms` to `10.939ms` and forward-backward from `22.948ms` to `26.156ms` despite XLA latency hiding.
- Grouped exact-ring stage tasks are not a scaling candidate. Group size 2 reaches only `1.0889x` direct value-and-grad speedup, below the `1.11x` gate needed to project above 20 MFU after pipeline bubbles. Group size 4 reaches `1.0819x`, below its `1.134x` target, and exceeds the accepted `0.2%` shared-gradient error ceiling. Separating dispatch, expert compute, and combine is slower than queued full calls.
- Approximate QuACK improves a one-node L2 whole step by `4.00%`, but the controlled reduced JaxPP A/B is steady-state neutral: clean mean MFU `9.4786` versus `9.4613` (`+0.18%`) and clean p50 `9.4928` versus `9.4843` (`+0.09%`). Pipeline/collective work hides or dominates the local expert-kernel gain.
- `GRUG-JAXPP-014`: Packed FP8 pipeline transfers are finite and consistently faster, but the L24/b512/m16 confirmation improves clean matched p50 MFU only `1.79%` (`16.2447 -> 16.5358`). The confirmed gain projects the `18.2583` m256 best to about `18.59`, so exact m256 scaling cannot reach 20 MFU.
- `GRUG-JAXPP-015`: Hopper FP8 expert GEMMs clear the CUTLASS/JaxPP compiler blocker and improve matched reduced mean MFU `1.74%` (`16.1532 -> 16.4344`), with finite loss and user-accepted `+0.2122%` final relative loss drift. The gain projects the `18.2583` best to only about `18.58`, so unchanged FP8 expert GEMMs cannot reach 20 MFU and should not be scaled to L24.
- `GRUG-JAXPP-013`: Exact NCCL_EP is numerically stable but reaches only `5.0245%` reduced MFU. Bounded NCCL_EP retains a direct `1.37-1.45x` speedup, but its token-gradient relative-L2 remains `0.2909%` against an FP32 ring reference after FP32 combine, token-scaled loss, FP32 dispatch, and an FP32 combine-forward token-dgrad control. The hybrid control reproduces the native dispatch-backward error exactly. This exceeds the accepted `0.2%` ceiling, so neither variant can scale to L24.

### Promoted
- `GRUG-JAXPP-001`: Explicit MPMD stage-local weights/optimizer state are the working pipeline implementation. Evidence: milestone commit `abd979b82a`, issue update <https://github.com/marin-community/marin/issues/7024#issuecomment-4919647823>, and the seq4096 perf/profile results.
- `GRUG-JAXPP-003`: GPU runtime availability is no longer blocked; CoreWeave east02 4x8 H100 jobs validated the implementation path.
- `GRUG-JAXPP-004`: The May d=2560 family runs with measurable MFU on 4x 8xH100. Best measured point is explicit `std_1f1b` seq4096, 64 experts, batch8192/m256 with CuTe FA4 and eight-warp Pallas-Triton ragged dot at mean MFU `18.2583`.

## Entry Log

### 2026-07-07 22:45 PDT - initial implementation
- Hypothesis: JaxPP can be introduced as an optional MoE train-step path by adding stage markers to the model and replacing the single global `value_and_grad` with a `jaxpp.treduce` over microbatches.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe` at baseline `5c36c4374a`.
- Command:
  - `git switch -c research/jaxpp-grug-moe`
  - `git clone --depth 1 https://github.com/NVIDIA/jaxpp.git /tmp/jaxpp`
  - `uv run python -m py_compile experiments/grug/moe/train.py experiments/grug/moe/model.py`
  - `uv run python - <<'PY' ... import GrugJaxPPConfig, GrugTrainerConfig; _pipeline_stage_end_layers(6, 3) ... PY`
  - `XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu uv run python - <<'PY' ... _compact_or_pipeline_grug_mesh ... _reshape_batch_for_pipeline ... PY`
  - `uv pip install -e /tmp/jaxpp`
  - `uv pip install --no-deps -e /tmp/jaxpp`
  - `XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu uv run python - <<'PY' ... jaxpp.treduce(... operation=((jaxpp.Add, jaxpp.Add), jaxpp.Add)) ... PY`
  - `XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu uv run python - <<'PY' ... jax.jit(_reshape_batch_for_pipeline) ... PY`
  - `./infra/pre-commit.py --fix experiments/grug/moe/train.py experiments/grug/moe/model.py`
- Config:
  - New `GrugJaxPPConfig(stages, microbatches, schedule="std_1f1b", stage_axis_name="pipeline")`.
  - `GrugTrainerConfig.pipeline=None` preserves the existing non-pipelined path.
  - Pipeline mesh shape is `(pipeline, replica_dcn, data, expert, model)`.
  - Stage cuts require `num_layers % stages == 0`.
- Result:
  - Added optional `jaxpp` imports with fail-fast errors when pipeline mode is requested without the package.
  - Added `pipeline_stages` threading through `Transformer.__call__`, `logits`, and `next_token_loss`.
  - Added `jaxpp.mark_stage_end` after each configured stage boundary, including the final stage marker required by current upstream JaxPP.
  - Added a pipelined train-step branch that reshapes batch leaves to `(microbatches, microbatch_size, ...)`, uses `jaxpp.treduce`, averages loss/grads/QB betas, and leaves default train-step behavior unchanged.
  - `uv pip install -e /tmp/jaxpp` failed on macOS because upstream `jaxpp==0.10.2` depends on `cupy-cuda13x`, which has no matching macOS wheel. `uv pip install --no-deps -e /tmp/jaxpp` succeeded for API smoke.
  - A review pass found that the first reshape implementation would skip tracer leaves inside `jax.jit`; fixed by accepting `jax.core.Tracer`, and the jitted reshape smoke now returns `(2, 2, 3)` for both token and loss-weight leaves.
  - A tiny full MoE CPU loss smoke with an unrealistic small config failed inside existing local MoE scatter shape handling, before showing a pipeline-specific failure. This is not treated as evidence about GPU JaxPP runtime correctness.
- Interpretation:
  - The structural implementation is ready for a real GPU smoke run, but runtime correctness and performance are unproven.
  - Full router metric parity is intentionally incomplete in the JaxPP path; the pipeline branch currently preserves `train/loss` and `qb_beta_per_layer` for QB updates, while avoiding incorrect `SummaryStats` reductions.
- Next action:
  - Launch or reserve a small NVIDIA GPU environment with `jaxpp` installed from NVIDIA/jaxpp and run a 1-2 step Grug MoE pipeline smoke.
  - Decide whether to add a narrow dependency hook for `jaxpp` to Grug GPU jobs or keep it as an explicit experimental environment prerequisite.

### 2026-07-07 23:02 PDT - tracking issue
- Hypothesis: The research thread needs a durable issue/logbook trail before cluster runs start.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe`.
- Command:
  - `gh issue list --repo marin-community/marin --state open --search "jaxpp grug moe pipeline parallelism in:title,body" --json number,title,url,labels,createdAt,updatedAt --limit 20`
  - `gh issue create --repo marin-community/marin --title "[grug] Track JaxPP pipeline-parallel MoE training" --label experiment --label agent-generated --body-file ...`
- Result:
  - Created coordinating experiment issue: https://github.com/marin-community/marin/issues/7024
  - Linked the issue to this logbook and recorded the 4x 8xH100 May d=2560 schedule/MFU objective.
- Next action:
  - Port or adapt the CoreWeave May d=2560 launcher and add JaxPP schedule knobs for the first GPU smoke.

### 2026-07-07 23:12 PDT - CoreWeave smoke launch plumbing
- Hypothesis: A 4-node H100 JaxPP smoke can use the May d=2560 shape with 24 layers, 256 experts, top-k 4, expert axis 8, four physical pipeline ranks, and eight microbatches.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe`.
- Command:
  - `experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --schedule std_1f1b --steps 2 --tracker json_logger`
  - `experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --schedule std_1f1b --steps 2 --tracker json_logger --run-id jaxpp-may-d2560-std-smoke-...`
  - `experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --schedule std_1f1b --steps 2 --tracker json_logger --run-id jaxpp-may-d2560-std-smoke-r2-...`
  - `experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --schedule std_1f1b --steps 2 --tracker json_logger --run-id jaxpp-may-d2560-std-direct-...`
- Config:
  - Default smoke model: `hidden_dim=2560`, `num_layers=24`, `seq_len=4096`, `num_experts=256`, `top_k=4`.
  - Mesh: `PP_MPMD_DIM=4`, `MAY_GPU_REPLICAS=4`, `MAY_EXPERT_AXIS=8`, `MAY_REPLICA_AXIS=1`.
  - Pipeline: `PP_SCHEDULE=std_1f1b`, `PP_STAGES=4`, `PP_MICROBATCHES=8`.
  - Data: deterministic synthetic `GrugLmExample` stream to isolate systems/runtime behavior from tokenization and object-store reads.
- Result:
  - Added `experiments/grug/moe/launch_cw_jaxpp_may_d2560.py` and `run_cw_jaxpp_may_d2560.sh`.
  - Added `post_setup_scripts` plumbing so GPU worker jobs can install pinned NVIDIA/JaxPP after normal Iris GPU sync.
  - Added JaxPP schedule names for `gpipe`, `std_1f1b`, `eager_1f1b`, `zero_bubble`, `interleaved_gpipe`, `interleaved_1f1b`, `dualpipe_v`, and `kimi_k2`.
  - First parent job `/dlwh/iris-run-job-20260708-060558` failed before launching training: `StepRunner` tried to create `s3://marin-na/...` and S3 returned `InvalidRegion`.
  - Second parent job `/dlwh/iris-run-job-20260708-060836` preserved the `dlwh` namespace but failed with the same `StepRunner` S3 artifact sidecar error.
  - Switched the launcher default to direct mode (`MAY_DIRECT=true`) so the CPU parent dispatches the GPU training job with local output/checkpoint paths and bypasses StepRunner artifact writes.
  - Third parent job `/dlwh/iris-run-job-20260708-061147` bypassed parent artifact writes, launched the 4-task GPU child, installed `cupy-cuda13x` and `jaxpp==0.10.2`, initialized JAX distributed, and reached JaxPP tracing.
  - The GPU child failed in tracing with `ValueError: use_abstract_mesh cannot change the size of the mesh` because the outer Grug mesh context was size 32 while JaxPP enters stage-local size-8 meshes.
  - Patched `_run_grug_local` so setup/init still runs under the global mesh, default non-pipeline train steps still run under `set_mesh(mesh)`, and JaxPP pipeline train steps run outside the global mesh context.
- Interpretation:
  - The first failure mode is unrelated to JaxPP or Grug; it is parent artifact bookkeeping against the cluster object-store configuration.
  - Direct mode is the correct path for disposable throughput probes until the CoreWeave/R2 artifact prefix handling is cleaned up.
  - JaxPP cannot trace inside the global Grug `set_mesh(mesh)` context; the stage-local mesh must be the active mesh during the pipeline train-step trace.
- Next action:
  - Resubmit the direct-mode `std_1f1b` smoke and watch for the next tracing/compile failure or first-step metrics.

### 2026-07-07 23:26 PDT - microbatch sharding failure
- Hypothesis: Once JaxPP traces outside the global Grug mesh, the next failure will identify whether the microbatch dimension can be consumed by `jaxpp.treduce` under the `(pipeline, replica_dcn, data, expert, model)` mesh.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe`.
- Command:
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job list --prefix /dlwh/iris-run-job-20260708-061923`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job logs ... /dlwh/iris-run-job-20260708-061923/grug-train-jaxpp-may-d2560-std-meshfix-20260708-061920`
  - `XLA_FLAGS=--xla_force_host_platform_device_count=32 JAX_PLATFORMS=cpu uv run python - <<'PY' ... _reshape_batch_for_pipeline ... PY`
  - `./infra/pre-commit.py --fix experiments/grug/moe/train.py .agents/logbooks/jaxpp-grug-moe.md`
- Config:
  - Parent job: `/dlwh/iris-run-job-20260708-061923`
  - GPU child: `/dlwh/iris-run-job-20260708-061923/grug-train-jaxpp-may-d2560-std-meshfix-20260708-061920`
  - Model: d2560, 24 layers, 256 experts, top-k 4, sequence length 4096.
  - Mesh: 4 physical pipeline ranks, expert axis 8, replica axis 1, batch 256.
  - Pipeline: `std_1f1b`, 4 logical stages, 8 microbatches.
- Result:
  - The run passed worker setup, installed pinned NVIDIA/JaxPP, initialized JAX distributed, and got past the earlier `use_abstract_mesh` size conflict.
  - It then failed inside JaxPP `treduce` when JaxPP internally called `jax.numpy.take(..., axis=0)` on the microbatch tree. JAX reported `ShardingTypeError` because the newly inserted microbatch axis inherited batch sharding: `operand=ShapedArray(int32[8@(replica_dcn,data,expert),32,4096])`.
  - Patched `_reshape_batch_for_pipeline` to call `x.reshape(..., out_sharding=P(None, ("replica_dcn", "data", "expert"), ...))`, leaving the leading microbatch axis unsharded and preserving the original per-microbatch batch sharding.
  - Added `JAX_COMPILATION_CACHE_DIR=/tmp/jax-compilation-cache` to the CoreWeave launcher environment so repeated debug smokes avoid the broken remote persistent compilation cache path.
- Interpretation:
  - The mesh-context fix was real progress: JaxPP now owns the active stage-local mesh during trace.
  - The current evidence points to a batch-reshape sharding bug in the experimental integration, not a model or optimizer failure.
- Next action:
  - Resubmit `std_1f1b` with the explicit microbatch-axis sharding and watch for compile progress, first-step loss, or the next JaxPP runtime error.

### 2026-07-07 23:34 PDT - native abort after JaxPP trace
- Hypothesis: With microbatch sharding fixed, the next failure will show whether `std_1f1b` can compile the 24-layer May d=2560 MoE step or whether another integration issue appears after JaxPP's second loop trace.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe`.
- Command:
  - `experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --schedule std_1f1b --steps 2 --tracker json_logger --run-id jaxpp-may-d2560-std-reshard-20260708-062725`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job summary /dlwh/iris-run-job-20260708-062728/grug-train-jaxpp-may-d2560-std-reshard-20260708-062725`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job logs --no-tail --max-lines 1400 /dlwh/iris-run-job-20260708-062728/grug-train-jaxpp-may-d2560-std-reshard-20260708-062725 ...`
  - `./infra/pre-commit.py --fix experiments/grug/moe/launch.py experiments/grug/moe/run_cw_jaxpp_may_d2560.sh experiments/grug/moe/launch_cw_jaxpp_may_d2560.py experiments/grug/moe/train.py .agents/logbooks/jaxpp-grug-moe.md`
- Config:
  - Parent job: `/dlwh/iris-run-job-20260708-062728`
  - GPU child: `/dlwh/iris-run-job-20260708-062728/grug-train-jaxpp-may-d2560-std-reshard-20260708-062725`
  - Model/pipeline: d2560, 24 layers, 256 experts, top-k 4, `std_1f1b`, 4 stages, 8 microbatches.
- Result:
  - The run passed worker setup, installed JaxPP, initialized JAX distributed, and got past the earlier microbatch `ShardingTypeError`.
  - No `train/loss` or MFU metrics were emitted.
  - Iris summary reported task 0 exited `139`; tasks 1-3 exited cleanly but were killed after max task failures.
  - Logs were dominated by `TrainerConfig.log_jaxprs=true` and `log_xla_hlo=true` output, including large JaxPP/JAX jaxpr dumps immediately before the native abort. There was no clean Python traceback.
  - Patched `experiments/grug/moe/launch.py` to honor `GRUG_LOG_JAXPRS`, `GRUG_LOG_XLA_HLO`, and `JAX_COMPILATION_CACHE_DIR` when building `TrainerConfig`.
  - Patched the CoreWeave JaxPP wrapper to default `GRUG_LOG_JAXPRS=false`, `GRUG_LOG_XLA_HLO=false`, and `JAX_COMPILATION_CACHE_DIR=/tmp/jax-compilation-cache`.
- Interpretation:
  - The explicit microbatch sharding fix worked, but the run now reaches a native crash during or immediately after JaxPP/JAX tracing/compilation.
  - The next run should remove noisy IR logging as a confounder and make the failure surface smaller if the native crash persists.
- Next action:
  - Resubmit the same `std_1f1b` 2-step smoke with JaxPR/HLO logging disabled and collect either first metrics or a cleaner compiler/runtime failure.

### 2026-07-07 23:44 PDT - clean trace still segfaults
- Hypothesis: If the native abort was caused by giant JaxPR/HLO logging, disabling IR logging should let the 24-layer May d=2560 `std_1f1b` smoke reach compile or first metrics.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe`.
- Command:
  - `experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --schedule std_1f1b --steps 2 --tracker json_logger --run-id jaxpp-may-d2560-std-noir-20260708-063540`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job summary /dlwh/iris-run-job-20260708-063543/grug-train-jaxpp-may-d2560-std-noir-20260708-063540`
- Config:
  - Parent job: `/dlwh/iris-run-job-20260708-063543`
  - GPU child: `/dlwh/iris-run-job-20260708-063543/grug-train-jaxpp-may-d2560-std-noir-20260708-063540`
  - Model/pipeline: d2560, 24 layers, 256 experts, top-k 4, `std_1f1b`, 4 stages, 8 microbatches.
  - Debug knobs: `log_jaxprs=false`, `log_xla_hlo=false`, `jax_compilation_cache_dir=/tmp/jax-compilation-cache`.
- Result:
  - The worker hparams confirmed the logging/cache knobs landed.
  - JaxPP first-loop tracing completed on all tasks in about 6.4-6.8 seconds; second-loop tracing completed in about 0.33 seconds.
  - No `train/loss` or MFU metrics were emitted.
  - The job still aborted natively after about one minute in the train loop. Iris summary reported one task with exit `139`; the remaining tasks were killed after max task failures.
- Interpretation:
  - Logging volume was not the primary cause.
  - The failure now looks like a native compiler/runtime crash after JaxPP tracing. The next useful split is to shrink the model while keeping the same 4 physical pipeline stages.
- Next action:
  - Submit a 4-layer d2560 smoke with fewer experts and microbatches to test whether the JaxPP train-step path can run at all on this cluster/runtime.

### 2026-07-08 00:06 PDT - reduced MoE exposes JaxPP clustering failure
- Hypothesis: Shrinking the May d=2560 run to four layers and fewer experts should distinguish a size-related compiler crash from a structural JaxPP/Grug incompatibility.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe` at baseline `5c36c4374a`.
- Command:
  - `experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --schedule std_1f1b --steps 2 --tracker json_logger --layers 4 --experts 32 --top-k 2 --batch 32 --expert-axis 8 --microbatches 4 --run-id jaxpp-d2560-l4-e32-std-20260708-064441`
  - `experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --schedule std_1f1b --steps 2 --tracker json_logger --layers 4 --experts 32 --top-k 2 --batch 32 --seq-len 512 --expert-axis 1 --microbatches 4 --moe-implementation scatter --run-id jaxpp-d2560-l4-scatter-20260708-065703`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job summary /dlwh/iris-run-job-20260708-065712/grug-train-jaxpp-d2560-l4-scatter-20260708-065703`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job logs --no-tail --max-lines 2500 /dlwh/iris-run-job-20260708-065712/grug-train-jaxpp-d2560-l4-scatter-20260708-065703 ...`
- Config:
  - Ring/EP parent job: `/dlwh/iris-run-job-20260708-064444`
  - Scatter/no-EP parent job: `/dlwh/iris-run-job-20260708-065712`
  - Shared model shape: d2560, 4 layers, 32 experts, top-k 2, `std_1f1b`, 4 stages, 4 microbatches.
  - Scatter split: `seq_len=512`, `expert_axis=1`, `moe_implementation=scatter`, default CE implementation.
- Result:
  - The reduced ring/EP run still took about 284-288 seconds in JaxPP first-loop tracing and failed before `train/loss`; the large jaxpr included the EP ring path, `haliax.nn.ragged_dot`, Pallas calls, and EP collectives.
  - The scatter/no-EP run removed expert-parallel collectives but still failed before `train/loss`.
  - Scatter/no-EP first-loop tracing spent about 273-277 seconds, mostly in fused CE autotune for `batched_xla`; selected block sizes were `BlockSizes(b_block_size=1024, h_block_size=512, v_block_size=64)`.
  - The autotune cache write attempted `s3://marin-na/marin/levanter_kernel_autotune/fused_cross_entropy_loss/block_sizes_v1.json` and hit the same R2/S3 `InvalidRegion` warning seen in the parent StepRunner artifact path.
  - After tracing, task 1 raised `AssertionError: Failed on loop body jaxpr` inside JaxPP `cluster_jaxpr`; the printed jaxpr showed nested Grug MoE `shard_map` and Pallas `ragged_dot` calls even in the local scatter path.
- Interpretation:
  - Four-layer size reduction did not produce a working JaxPP step.
  - The reduced scatter failure is cleaner than the 24-layer native abort: JaxPP conservative loop clustering cannot assign all loop-body equations to pipeline stages when Grug MoE runs its nested mesh/shard_map/Pallas body.
  - CE autotune is also too expensive for these JaxPP traces and its remote cache path is region-misconfigured for this CoreWeave/R2 environment.
- Next action:
  - Relaunch the same scatter/no-EP shape with `loss_implementation=xla`, `LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS=false`, and `JAXPP_CONSERVATIVE_LOOP_CLUSTERING=false` to test whether the structural failure is only the conservative assignment check or a deeper lowering/runtime blocker.

### 2026-07-08 00:07 PDT - non-conservative clustering probe launched
- Hypothesis: Disabling JaxPP's conservative loop-clustering assertion will allow unclustered nested Grug MoE equations to fall into the final task, letting the reduced scatter/no-EP run reach compilation or first-step metrics.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe` at baseline `5c36c4374a`.
- Command:
  - `experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --schedule std_1f1b --steps 2 --tracker json_logger --layers 4 --experts 32 --top-k 2 --batch 32 --seq-len 512 --expert-axis 1 --microbatches 4 --moe-implementation scatter --loss-implementation xla --ce-autotune-on-miss false --conservative-loop-clustering false --run-id jaxpp-d2560-l4-scatter-xla-nonconservative-20260708-070725`
- Config:
  - Parent job: `/dlwh/iris-run-job-20260708-070725`
  - Model/pipeline: d2560, 4 layers, 32 experts, top-k 2, sequence length 512, `std_1f1b`, 4 stages, 4 microbatches.
  - Isolation knobs: `expert_axis=1`, `moe_implementation=scatter`, `loss_implementation=xla`, `LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS=false`, `JAXPP_CONSERVATIVE_LOOP_CLUSTERING=false`.
- Result:
  - The job failed quickly before `train/loss`.
  - Hparams confirmed `moe_implementation="scatter"`, `loss_implementation="xla"`, `expert_axis_size=1`, and the reduced d2560/4-layer/seq-512 shape.
  - Disabling conservative loop clustering got past the prior `Failed on loop body jaxpr` assertion, but JaxPP failed in `wrap_into_tasks_after_loop` with `AssertionError: After loop computation is not replicateable`.
  - The source stack pointed at `optax.apply_updates(qb_params, updates)`, specifically the `p + update` add inside Optax.
- Interpretation:
  - The loop-body clustering issue is not the only blocker.
  - Grug's after-loop optimizer update sees replicated/global parameters and MPMD-stage-local updates; JaxPP does not infer placement for the plain Optax add.
- Next action:
  - Apply pipeline updates with JaxPP's `place_with(param, update)` primitive and resubmit the same reduced probe.

### 2026-07-08 00:11 PDT - place_with update retry launched
- Hypothesis: Explicitly placing parameters with their corresponding updates before `p + update` will let JaxPP handle the after-loop optimizer application.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe` at baseline `5c36c4374a`.
- Command:
  - `./infra/pre-commit.py --fix experiments/grug/moe/train.py experiments/grug/moe/model.py experiments/grug/moe/launch_cw_jaxpp_may_d2560.py experiments/grug/moe/run_cw_jaxpp_may_d2560.sh .agents/logbooks/jaxpp-grug-moe.md`
  - `experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --schedule std_1f1b --steps 2 --tracker json_logger --layers 4 --experts 32 --top-k 2 --batch 32 --seq-len 512 --expert-axis 1 --microbatches 4 --moe-implementation scatter --loss-implementation xla --ce-autotune-on-miss false --conservative-loop-clustering false --run-id jaxpp-d2560-l4-scatter-xla-placewith-20260708-071127`
- Config:
  - Parent job: `/dlwh/iris-run-job-20260708-071127`
  - Same reduced scatter/no-EP/XLA-loss shape as the prior probe.
  - Code change: pipeline branch applies updates as `jaxpp.place_with(param, update) + update` instead of raw `optax.apply_updates`.
- Result:
  - The job failed quickly before `train/loss`.
  - Hparams confirmed the intended reduced scatter/no-EP/XLA-loss shape.
  - JaxPP first-loop/overall tracing completed in about 3.2 seconds, so the CE autotune delay was removed.
  - The same `AssertionError: After loop computation is not replicateable` remained, now sourced at the explicit `jaxpp.place_with(param, update)` call in the pipeline update helper.
- Interpretation:
  - `place_with` is not accepted as a post-loop replication bridge in this context.
  - The next likely shape is to turn the stage-local update into an explicitly cross-MPMD value before adding it to the replicated parameter leaf.
- Next action:
  - Replace `place_with(param, update) + update` with `param + jaxpp.cross_mpmd_all_reduce(update)` and resubmit the same reduced probe.

### 2026-07-08 00:14 PDT - cross-MPMD update retry launched
- Hypothesis: Explicitly reducing each Optax update leaf across MPMD ranks will make the post-loop parameter update replicable.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe` at baseline `5c36c4374a`.
- Command:
  - `uv run python -m py_compile experiments/grug/moe/train.py`
  - `./infra/pre-commit.py --fix experiments/grug/moe/train.py .agents/logbooks/jaxpp-grug-moe.md`
  - `experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --schedule std_1f1b --steps 2 --tracker json_logger --layers 4 --experts 32 --top-k 2 --batch 32 --seq-len 512 --expert-axis 1 --microbatches 4 --moe-implementation scatter --loss-implementation xla --ce-autotune-on-miss false --conservative-loop-clustering false --run-id jaxpp-d2560-l4-scatter-xla-crossreduce-...`
- Config:
  - Same reduced scatter/no-EP/XLA-loss probe.
  - Code change: pipeline update helper applies `jaxpp.cross_mpmd_all_reduce(update)` before adding each update leaf to its parameter leaf.
- Result:
  - Parent job: `/dlwh/iris-run-job-20260708-071424`.
  - The job failed quickly before `train/loss`.
  - Hparams confirmed the intended reduced scatter/no-EP/XLA-loss shape.
  - JaxPP first-loop/overall tracing completed in about 3.15 seconds.
  - The same `AssertionError: After loop computation is not replicateable` remained, sourced at `param + replicated_update`.
- Interpretation:
  - A one-input `cross_mpmd_all_reduce(update)` does not make the update available on the parameter leaf's placement. It preserves the update's existing MPMD placement, so the following add still has disjoint placements for at least one parameter leaf.
- Next action:
  - Retry the reduced probe with `jaxpp.cross_mpmd_all_reduce(update, jnp.zeros_like(param))`, which gives JaxPP both the update placement and the parameter placement without changing the numeric update.

### 2026-07-08 00:22 PDT - cross-MPMD update plus zero retry launched
- Hypothesis: Reducing each update leaf together with a zero leaf shaped like its corresponding parameter will materialize the update on both the update's stage and the parameter's stage, making the post-loop `param + update` add replicable.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe` at baseline `5c36c4374a`.
- Command:
  - `uv run python -m py_compile experiments/grug/moe/train.py`
  - `./infra/pre-commit.py --fix experiments/grug/moe/train.py .agents/logbooks/jaxpp-grug-moe.md`
  - `experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --schedule std_1f1b --steps 2 --tracker json_logger --layers 4 --experts 32 --top-k 2 --batch 32 --seq-len 512 --expert-axis 1 --microbatches 4 --moe-implementation scatter --loss-implementation xla --ce-autotune-on-miss false --conservative-loop-clustering false --run-id jaxpp-d2560-l4-scatter-xla-crosszero-...`
- Config:
  - Same reduced scatter/no-EP/XLA-loss probe.
  - Code change: pipeline update helper applies `jaxpp.cross_mpmd_all_reduce(update, jnp.zeros_like(param))` before adding each update leaf to its parameter leaf.
- Result:
  - Parent job: `/dlwh/iris-run-job-20260708-072104`.
  - Child job: `/dlwh/iris-run-job-20260708-072104/grug-train-jaxpp-d2560-l4-scatter-xla-crosszero-20260708-072232`.
  - The job failed before `train/loss`; task 2/3 showed the Python root cause and task 0 later aborted with exit 139 after coordination shutdown.
  - JaxPP tracing was short: first-loop tracing about 1.10 seconds, second-loop tracing about 0.055 seconds, total tracing about 2.78 seconds.
  - The failure remained `AssertionError: After loop computation is not replicateable`, sourced at `param + replicated_update` in the update helper.
- Interpretation:
  - The post-loop issue is not solved by adding zeros to the cross-MPMD reduce. The more fundamental problem is that the training state was still passed as SPMD/global arrays, so parameters were not partitioned into the pipeline-stage-owned structure JaxPP inferred for the train step.
- Next action:
  - Compile the JaxPP train step once, use `compiled.in_shardings` to obtain the state and batch `MpmdSharding` trees, convert the initial train state and per-step batches with `jaxpp.spmd_to_mpmd_reshard`, and restore ordinary `optax.apply_updates`.

### 2026-07-08 00:31 PDT - state and batch MPMD partitioning launched
- Hypothesis: Feeding JaxPP `MpmdArray` inputs with the compiled step's inferred shardings will partition weights by pipeline stage and remove the after-loop replicated/global parameter update conflict.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe` at baseline `5c36c4374a`.
- Command:
  - `uv run python -m py_compile experiments/grug/moe/train.py`
  - `./infra/pre-commit.py --fix experiments/grug/moe/train.py .agents/logbooks/jaxpp-grug-moe.md`
  - `experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --schedule std_1f1b --steps 2 --tracker json_logger --layers 4 --experts 32 --top-k 2 --batch 32 --seq-len 512 --expert-axis 1 --microbatches 4 --moe-implementation scatter --loss-implementation xla --ce-autotune-on-miss false --conservative-loop-clustering false --run-id jaxpp-d2560-l4-scatter-xla-mpmdstate-...`
- Config:
  - Same reduced scatter/no-EP/XLA-loss probe.
  - Code change: pipeline loop compiles the JaxPP step once, reads `compiled_pipeline_train_step.in_shardings`, reshards initial `GrugTrainState` and each batch with `jaxpp.spmd_to_mpmd_reshard`, keeps state as MPMD-owned across steps, disables watch-stat tracing for the pipeline path, and uses ordinary `optax.apply_updates`.
- Result:
  - Parent job: `/dlwh/iris-run-job-20260708-072741`.
  - Child job: `/dlwh/iris-run-job-20260708-072741/grug-train-jaxpp-d2560-l4-scatter-xla-mpmdstate-20260708-073124`.
  - The job failed before `train/loss` during `train_step.compile(...)`, so the post-compile `spmd_to_mpmd_reshard` path did not run.
  - JaxPP first-loop tracing took about 1.08-1.12 seconds, second-loop tracing about 0.05 seconds, and total tracing about 2.75-3.07 seconds.
  - The root source stack remained Optax's `p + u` in `optax.apply_updates`, and JaxPP again raised `AssertionError: After loop computation is not replicateable`.
- Interpretation:
  - Post-compile resharding is too late for this failure because JaxPP must first place the after-loop optimizer update while tracing.
  - JaxPP's after-loop pass explicitly rewrites JAX's `add_any` primitive across disjoint MPMD placements, but Optax emits ordinary `add`.
- Next action:
  - Keep the compile-then-reshard loop, but replace only the pipeline update application with a small `add_any`-based helper using `jax.interpreters.ad.add_jaxvals_p`; submit the same reduced probe.

### 2026-07-08 00:38 PDT - pipeline apply-updates with add_any launched
- Hypothesis: Using `add_any` for pipeline parameter updates will let JaxPP rewrite cross-placement parameter/update additions during after-loop placement, allowing compile to finish and then letting the loop reshard state/batches into inferred `MpmdSharding`s.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe` at baseline `5c36c4374a`.
- Command:
  - `uv run python -m py_compile experiments/grug/moe/train.py`
  - `./infra/pre-commit.py --fix experiments/grug/moe/train.py .agents/logbooks/jaxpp-grug-moe.md`
  - `experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --schedule std_1f1b --steps 2 --tracker json_logger --layers 4 --experts 32 --top-k 2 --batch 32 --seq-len 512 --expert-axis 1 --microbatches 4 --moe-implementation scatter --loss-implementation xla --ce-autotune-on-miss false --conservative-loop-clustering false --run-id jaxpp-d2560-l4-scatter-xla-addany-...`
- Config:
  - Same reduced scatter/no-EP/XLA-loss probe.
  - Code change: default/non-pipeline training still uses `optax.apply_updates`; the pipeline branch applies update leaves with `jax.interpreters.ad.add_jaxvals_p.bind(param, update)` so JaxPP sees `add_any`.
- Result:
  - Pending launch at time of entry.
- Interpretation:
  - Pending.
- Next action:
  - Submit and watch whether compile gets past after-loop placement.

### 2026-07-08 00:40 PDT - explicit MPMD pivot analysis
- Hypothesis: The automatic `treduce` path is fighting Grug's replicated state tree and after-loop optimizer update; the explicit `jaxpp.experimental.mpmd` pattern from `examples/mpmd.py` may be a cleaner fit because it initializes and updates stage-local parameter trees directly.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe` at baseline `5c36c4374a`.
- Command:
  - `sed -n '1,780p' /private/tmp/jaxpp/examples/mpmd.py`
  - `sed -n '715,950p' experiments/grug/moe/model.py`
- Observation:
  - `examples/mpmd.py` builds an explicit `MpmdMesh`, creates stage-local `NamedSharding`s, initializes parameter tuples directly on `mpmd_mesh.unstack[stage]`, and expresses the 1F1B schedule as named `mpmd.task` calls plus explicit activation/gradient `mpmd.transfer`s.
  - It avoids a global replicated parameter tree in the pipelined step. Stage params, grad accumulators, residuals, and activations all have explicit stage-local shardings before lowering.
  - Grug can be split around `Transformer.blocks`, but the stage contract must handle embedding on stage 0, intermediate block slices on middle stages, and final norm/LM head/cross entropy on the last stage.
  - The smallest Grug-compatible explicit-MPMD smoke should target 2 pipeline stages first, with 4 layers, scatter MoE, XLA CE, synthetic data, and no router metric parity beyond loss/QB beta needed for the smoke.
- Interpretation:
  - The current automatic `treduce` branch may still be useful if the pending `add_any` run compiles, but a robust implementation likely needs an explicit stage-owned state tree similar to `mpmd.py`.
  - A direct port is more than a one-line fix: it needs stage wrappers, stage-local state initialization/restoration, explicit activation shape/sharding definitions, and optimizer-state ownership by stage.
- Next action:
  - If the pending `add_any` smoke fails or remains stuck, start a new explicit-MPMD implementation slice for a two-stage smoke rather than continuing to patch after-loop placement in the automatic path.

### 2026-07-08 00:50 PDT - stage-local weight partition scaffolding
- Hypothesis: Explicit MPMD needs Grug weights and optimizer state represented as pipeline-stage-owned pytrees before the scheduler is wired; otherwise the implementation will repeat the automatic path's global-state placement conflict.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe` at baseline `5c36c4374a`.
- Command:
  - `uv run python -m py_compile experiments/grug/moe/model.py experiments/grug/moe/train.py`
  - `./infra/pre-commit.py --fix experiments/grug/moe/model.py experiments/grug/moe/train.py`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job summary /dlwh/iris-run-job-20260708-073255/grug-train-jaxpp-d2560-l4-scatter-xla-addany-20260708-073814`
- Config:
  - Local code refactor only; no new Iris launch.
  - `Transformer` now exposes `embed_tokens`, `block_range`, `finalize_hidden`, `hidden_next_token_loss`, `split_for_pipeline`, and `merge_pipeline_stages`.
  - `TransformerPipelineStage` owns only the params needed by one contiguous layer range: stage 0 owns embeddings, middle stages own block slices, and the last stage owns final norm plus LM head.
  - `GrugPipelineTrainState` stores `params: tuple[TransformerPipelineStage, ...]` and per-stage `opt_state: tuple[optax.OptState, ...]`.
- Result:
  - Targeted Python compile passed.
  - Targeted repo pre-commit hooks passed for `experiments/grug/moe/model.py` and `experiments/grug/moe/train.py`.
  - The add-any Iris smoke was still running with all 4 tasks in `building` after about 17 minutes, so it remains inconclusive.
- Interpretation:
  - The code now has the weight partition structure requested for pipeline parallelism, but the explicit `jaxpp.experimental.mpmd` scheduler is not wired yet.
  - The add-any automatic path has not produced evidence beyond the previous compile failures because it has not reached execution logs.
- Next action:
  - Wire a first explicit-MPMD two-stage, one-microbatch smoke using the stage-local state; keep router z-loss disabled and update params/opt state inside owning-stage tasks.

### 2026-07-08 10:14 PDT - explicit MPMD reaches runtime transfer
- Hypothesis: The `examples/mpmd.py` style explicit scheduler is the right shape for Grug because it keeps params, optimizer state, and QB router state stage-local before lowering instead of asking automatic JaxPP to infer ownership of a global train state.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe` at baseline `5c36c4374a`.
- Commands:
  - `uv run python -m py_compile experiments/grug/moe/train.py experiments/grug/moe/model.py experiments/grug/moe/launch_cw_jaxpp_may_d2560.py`
  - `./infra/pre-commit.py --fix experiments/grug/moe/train.py experiments/grug/moe/model.py experiments/grug/moe/launch_cw_jaxpp_may_d2560.py experiments/grug/moe/run_cw_jaxpp_may_d2560.sh`
  - `experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --implementation explicit_mpmd --physical-stages 2 --logical-stages 2 --microbatches 1 --nodes 2 --expert-axis 1 --layers 4 --experts 32 --top-k 2 --batch 16 --seq-len 128 --moe-implementation scatter --loss-implementation xla --steps 1 --tracker json_logger --run-id jaxpp-d2560-l4-explicit-mpmd-hoststep-...`
  - `experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --implementation explicit_mpmd --physical-stages 2 --logical-stages 2 --microbatches 1 --nodes 2 --expert-axis 1 --layers 4 --experts 16 --top-k 1 --vocab-size 32768 --batch 16 --seq-len 64 --moe-implementation scatter --loss-implementation xla --steps 1 --tracker json_logger --run-id jaxpp-d2560-l4-explicit-mpmd-vocab32k-b16-...`
- Changes:
  - Added an explicit two-stage MPMD train-step path using `jaxpp.experimental.mpmd` tasks and transfers.
  - Added `TransformerPipelineStage` stage-owned parameter wrappers and split/merge helpers so pipeline weights and optimizer state are partitioned before the MPMD step.
  - Fixed explicit-state conversion to use `jaxpp.spmd_to_mpmd_reshard` with `MpmdSharding` targets, while keeping `NamedSharding` trees for `mpmd.task` input/output shardings.
  - Duplicated batch leaves before sending the same logical batch to stage 0 and stage 1 because `spmd_to_mpmd_reshard` donates/consumes the input arrays.
  - Moved step bookkeeping out of the MPMD function because `jaxpp.experimental.mpmd` only accepts task/transfer/stack/slice primitives at top level and scalar step placement is not part of the stage computation.
  - Added smoke-only `MAY_VOCAB_SIZE` / `--vocab-size`; defaults still use the May heuristic vocabulary.
- Results:
  - `/dlwh/iris-run-job-20260708-170126/...batchcopy...` passed state reshards but failed on top-level `state.step + 1` inside the MPMD wrapper: `got jit`.
  - `/dlwh/iris-run-job-20260708-170453/...steptask...` moved the scalar add into a task but failed because the scalar step aval lived on a single pipeline mesh while the task context was the full pipeline mesh.
  - `/dlwh/iris-run-job-20260708-170755/...hoststep...` got past lowering, created DIME streams/communicator, compiled `grug_stage0_forward`, and then failed during activation transfer with NCCL reporting `Cuda failure 2 'out of memory'`.
  - `/dlwh/iris-run-job-20260708-171247/...vocab32k...` with `batch=8` failed before training in Levanter validation with `ZeroDivisionError` from `data_axis_size == 0`.
  - `/dlwh/iris-run-job-20260708-171442/...vocab32k-b16...` is the current reduced-vocab d2560 transfer smoke.
- Interpretation:
  - The current explicit path has cleared the earlier JaxPP tracing/after-loop placement failures and is now exercising runtime cross-stage transfer.
  - Full-vocab d2560, even at 4 layers, is memory-heavy enough that the first runtime transfer can OOM before any loss metric; the reduced-vocab smoke should distinguish implementation correctness from the memory envelope.
- Next action:
  - Poll `/dlwh/iris-run-job-20260708-171442`; if it reaches a loss, scale vocabulary/sequence/layers back up incrementally before the 24-layer target.

### 2026-07-08 10:44 PDT - explicit MPMD one-step smoke succeeds
- Hypothesis: The remaining failure after explicit MPMD task execution was host-side state materialization, not the JaxPP task graph.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe` at baseline `5c36c4374a`.
- Commands:
  - `uv run python -m py_compile experiments/grug/moe/train.py experiments/grug/moe/model.py experiments/grug/moe/launch_cw_jaxpp_may_d2560.py`
  - `./infra/pre-commit.py --fix experiments/grug/moe/train.py experiments/grug/moe/model.py experiments/grug/moe/launch_cw_jaxpp_may_d2560.py experiments/grug/moe/run_cw_jaxpp_may_d2560.sh .agents/logbooks/jaxpp-grug-moe.md`
  - `XLA_PYTHON_CLIENT_MEM_FRACTION=0.70 experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --implementation explicit_mpmd --physical-stages 2 --logical-stages 2 --microbatches 1 --nodes 2 --gpus-per-replica 1 --expert-axis 1 --layers 4 --experts 4 --top-k 1 --vocab-size 8192 --batch 2 --seq-len 32 --moe-implementation scatter --loss-implementation xla --steps 1 --tracker json_logger --run-id jaxpp-d2560-l4-explicit-mpmd-callback-20260708-174046`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job summary /dlwh/iris-run-job-20260708-174049`
- Result:
  - `/dlwh/iris-run-job-20260708-174049` succeeded.
  - Stage 0 logged `train/loss=9.066007614135742` and completed one train step.
  - The explicit path compiled and ran `grug_stage0_forward`, `grug_stage1_loss_backward`, `grug_stage1_update`, `grug_stage0_backward`, `grug_stage0_update`, and `grug_keep_step`.
- Changes since the previous failed smoke:
  - Added a local lowered-output adapter for `jaxpp.experimental.mpmd.LoweredMpmdFun` so nonlocal outputs reuse prior stage-local leaves instead of relying on JaxPP's empty-array placeholder construction.
  - Transferred the scalar loss back to stage 0 for logging.
  - Disabled checkpoint writes for explicit MPMD until stage-local arrays have a deliberate global checkpoint materialization path.
  - Gated explicit-MPMD host callbacks to stage 0 to avoid logging fallback loss values from non-loss-owning stages.
- Interpretation:
  - Weight partitioning and the explicit two-stage MPMD train graph are now functional for a reduced d2560/4-layer smoke.
  - Earlier full-vocab OOMs were sensitive to XLA preallocation; future scale-up smokes should use a lower `XLA_PYTHON_CLIENT_MEM_FRACTION` such as 0.50.
- Next action:
  - Launch a 24-layer reduced-vocab smoke with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.50`, then scale sequence/batch/vocab back up if it succeeds.

### 2026-07-08 10:49 PDT - 24-layer explicit MPMD smoke succeeds
- Hypothesis: Lowering XLA preallocation should avoid the earlier NCCL transfer OOM and allow a 24-layer stage-partitioned Grug smoke to complete.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe` at baseline `5c36c4374a`.
- Commands:
  - `XLA_PYTHON_CLIENT_MEM_FRACTION=0.50 experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --implementation explicit_mpmd --physical-stages 2 --logical-stages 2 --microbatches 1 --nodes 2 --gpus-per-replica 1 --expert-axis 1 --layers 24 --experts 4 --top-k 1 --vocab-size 8192 --batch 2 --seq-len 32 --moe-implementation scatter --loss-implementation xla --steps 1 --tracker json_logger --run-id jaxpp-d2560-l24-explicit-mpmd-xlamem50-20260708-174431`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job summary /dlwh/iris-run-job-20260708-174434`
- Result:
  - `/dlwh/iris-run-job-20260708-174434` succeeded.
  - Stage 0 logged `train/loss=9.049333572387695`.
  - Parameter count was `1,885,107,296`.
  - One step took about `90.9s` including first compilation; no OOM at `XLA_PYTHON_CLIENT_MEM_FRACTION=0.50`.
- Interpretation:
  - The explicit MPMD implementation now runs a 24-layer d2560 Grug model with stage-local weight/optimizer partitioning across two JaxPP stages.
  - Remaining gaps are productionizing checkpoint materialization for stage-local JaxPP arrays and scaling the smoke back toward larger vocab/sequence/batch/expert counts.
- Next action:
  - Keep `XLA_PYTHON_CLIENT_MEM_FRACTION` below the default for larger JaxPP smokes; try larger vocab or more GPUs per stage next.

### 2026-07-08 11:03 PDT - automatic full-shape retry reaches JaxPP compile
- Hypothesis: The full May d2560 shape on 4x8 H100s needs a higher XLA heap than the 0.50 smoke, but still benefits from lowering the default preallocation below the previous 0.95 launcher default.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe` at baseline `5c36c4374a`.
- Commands:
  - `TF_GPU_ALLOCATOR=cuda_malloc_async XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --schedule std_1f1b --steps 2 --tracker json_logger --layers 24 --experts 256 --top-k 4 --batch 256 --seq-len 4096 --expert-axis 8 --moe-implementation ring --loss-implementation xla --ce-autotune-on-miss false --conservative-loop-clustering false --run-id jaxpp-may-d2560-auto-std-4x8-xlamem90-async-20260708-175530`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job logs /dlwh/iris-run-job-20260708-175532 --tail --max-lines 700`
- Config:
  - Automatic JaxPP `std_1f1b`, `PP_STAGES=4`, `PP_MPMD_DIM=4`, `PP_MICROBATCHES=8`.
  - Full May target shape: d2560, 24 layers, 256 experts, top-k 4, vocab 128256, batch 256, seq 4096, ring MoE, XLA CE.
  - 4 CoreWeave H100 nodes, 8 GPUs per node, expert axis 8.
- Result:
  - `/dlwh/iris-run-job-20260708-175135` with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.50` failed during `jit__init_state` because XLA's base limit was about 42.5 GB while init needed about 70.47 GiB.
  - `/dlwh/iris-run-job-20260708-175532` with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90` and `TF_GPU_ALLOCATOR=cuda_malloc_async` got past model init on all four processes.
  - The run reported `parameter_count=61,969,583,104`, H100 theoretical throughput summary, and entered the train loop.
  - It then failed in JaxPP compile after `After loop replication` with `AssertionError` involving a `float32[24,256]` aux tensor and the `int32[256,4096]` token batch.
- Interpretation:
  - The full shape is no longer blocked at initialization when the XLA heap is raised from 0.50 to 0.90, but automatic `treduce` cannot carry the per-layer/per-expert QB aux tensor through the after-loop placement.
  - The launcher default `XLA_PYTHON_CLIENT_MEM_FRACTION` was lowered from `0.95` to `0.88`, and `--xla-memory-fraction` was added so runs record the choice explicitly. Full-shape runs may still need an override near 0.89-0.90.
- Next action:
  - Remove QB aux from the automatic JaxPP `treduce` output for schedule/MFU probes while preserving explicit-MPMD QB feedback, then relaunch a reduced automatic compile smoke before returning to full shape.

### 2026-07-08 11:03 PDT - PP-aware launcher mesh and reduced follow-up probes
- Hypothesis: Automatic JaxPP schedule probes can compile if the reducer returns only loss/grads, and small explicit-MPMD probes need Levanter validation to see the same pipeline/expert batch sharding as the custom Grug mesh.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe` at baseline `5c36c4374a`.
- Commands:
  - `uv run python -m py_compile experiments/grug/moe/train.py experiments/grug/moe/model.py experiments/grug/moe/launch.py experiments/grug/moe/launch_cw_jaxpp_may_d2560.py`
  - `./infra/pre-commit.py --fix experiments/grug/moe/train.py experiments/grug/moe/model.py experiments/grug/moe/launch.py experiments/grug/moe/launch_cw_jaxpp_may_d2560.py experiments/grug/moe/run_cw_jaxpp_may_d2560.sh`
  - `TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --schedule std_1f1b --implementation auto --physical-stages 4 --microbatches 4 --nodes 4 --gpus-per-replica 8 --expert-axis 8 --layers 4 --experts 8 --top-k 1 --vocab-size 8192 --batch 32 --seq-len 128 --moe-implementation ring --loss-implementation xla --steps 1 --tracker json_logger --xla-memory-fraction 0.70 --conservative-loop-clustering false --run-id jaxpp-d2560-l4-auto-std-qbskip-meshfix-20260708-180102`
  - `TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --implementation explicit_mpmd --physical-stages 4 --logical-stages 4 --microbatches 1 --nodes 4 --gpus-per-replica 8 --expert-axis 8 --layers 24 --experts 8 --top-k 4 --vocab-size 8192 --batch 8 --seq-len 32 --moe-implementation ring --loss-implementation xla --steps 1 --tracker json_logger --xla-memory-fraction 0.50 --run-id jaxpp-d2560-l24-explicit-4stage-meshfix-20260708-180102`
- Changes:
  - Automatic JaxPP `treduce` now reduces only loss and gradients; it keeps `state.pending_qb_betas` unchanged instead of returning `qb_beta_per_layer`.
  - `run_grug_moe_trial` now passes a PP-aware `MeshConfig` into `TrainerConfig` whenever `GrugJaxPPConfig` is present, with `pipeline`, `replica_dcn`, `data`, `expert`, and `model` axes.
  - The launcher wrapper now defaults `XLA_PYTHON_CLIENT_MEM_FRACTION` to `0.88`, accepts `--xla-memory-fraction`, prints it in dry runs, and passes through `TF_GPU_ALLOCATOR` / `XLA_PYTHON_CLIENT_PREALLOCATE` when set.
- Result:
  - Targeted Python compile passed.
  - Targeted pre-commit hooks passed.
  - Prior explicit 4-stage run `/dlwh/iris-run-job-20260708-175602` failed before model init with Levanter `ZeroDivisionError` because small `batch=8` inferred `per_device_parallelism=0` against the default all-data mesh.
  - New follow-up jobs are running:
    - `/dlwh/iris-run-job-20260708-180234`: automatic `std_1f1b` QB-skip/mesh-fix reduced smoke.
    - `/dlwh/iris-run-job-20260708-180235`: explicit-MPMD 4-stage mesh-fix reduced smoke.
- Interpretation:
  - The latest code addresses the two newest blockers: JaxPP after-loop QB aux placement on automatic schedules and Levanter validation using the wrong batch-shard topology for PP runs.
- Next action:
  - Poll `/dlwh/iris-run-job-20260708-180234` and `/dlwh/iris-run-job-20260708-180235`; if the automatic smoke succeeds, relaunch the full 24-layer 256-expert `std_1f1b` run with an explicit memory fraction near 0.89-0.90, then expand to other JaxPP schedules.

### 2026-07-08 11:06 PDT - move pipeline mesh axis to DCN for CoreWeave
- Hypothesis: The PP-aware Levanter mesh must place `pipeline` across CoreWeave slices/nodes, because each JAX process sees one 8-GPU slice and one pipeline rank.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe` at baseline `5c36c4374a`.
- Commands:
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job logs /dlwh/iris-run-job-20260708-180234 --tail --max-lines 1000`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job logs /dlwh/iris-run-job-20260708-180235 --tail --max-lines 1000`
  - `uv run python -m py_compile experiments/grug/moe/launch.py experiments/grug/moe/train.py experiments/grug/moe/launch_cw_jaxpp_may_d2560.py`
  - `./infra/pre-commit.py --fix experiments/grug/moe/launch.py experiments/grug/moe/train.py experiments/grug/moe/launch_cw_jaxpp_may_d2560.py .agents/logbooks/jaxpp-grug-moe.md`
  - `TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --schedule std_1f1b --implementation auto --physical-stages 4 --microbatches 4 --nodes 4 --gpus-per-replica 8 --expert-axis 8 --layers 4 --experts 8 --top-k 1 --vocab-size 8192 --batch 32 --seq-len 128 --moe-implementation ring --loss-implementation xla --steps 1 --tracker json_logger --xla-memory-fraction 0.70 --conservative-loop-clustering false --run-id jaxpp-d2560-l4-auto-std-qbskip-dcnmesh-20260708-180536`
  - `TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --implementation explicit_mpmd --physical-stages 4 --logical-stages 4 --microbatches 1 --nodes 4 --gpus-per-replica 8 --expert-axis 8 --layers 24 --experts 8 --top-k 4 --vocab-size 8192 --batch 8 --seq-len 32 --moe-implementation ring --loss-implementation xla --steps 1 --tracker json_logger --xla-memory-fraction 0.50 --run-id jaxpp-d2560-l24-explicit-4stage-dcnmesh-20260708-180551`
- Result:
  - `/dlwh/iris-run-job-20260708-180234` and `/dlwh/iris-run-job-20260708-180235` both failed in Levanter mesh validation with `ValueError: ICI product 32 does not divide devices_per_slice 8`.
  - The corrected `MeshConfig` keeps `expert`, `data`, `replica`, and `model` in ICI axes and places `pipeline` plus `replica_dcn` in DCN axes.
  - Targeted Python compile and pre-commit passed.
  - New follow-up jobs:
    - `/dlwh/iris-run-job-20260708-180542`: automatic `std_1f1b` QB-skip with pipeline in DCN mesh.
    - `/dlwh/iris-run-job-20260708-180600`: explicit-MPMD 4-stage with pipeline in DCN mesh.
- Interpretation:
  - CoreWeave exposes four slices of eight GPUs for the 4x8 run; the trainer validation mesh must reflect that topology even though the Grug training loop later builds its own concrete JaxPP mesh.
- Next action:
  - Poll `/dlwh/iris-run-job-20260708-180542` and `/dlwh/iris-run-job-20260708-180600` for hparams/loss/compile errors.

### 2026-07-08 11:09 PDT - remove QB leaf from automatic JaxPP state
- Hypothesis: The remaining automatic JaxPP after-loop assertion is caused by the unchanged `pending_qb_betas` leaf in `GrugTrainState`, not by the `treduce` aux output, so automatic schedule probes need to remove that state leaf entirely.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe` at baseline `5c36c4374a`.
- Commands:
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job logs /dlwh/iris-run-job-20260708-180542 --tail --max-lines 800`
  - `uv run python -m py_compile experiments/grug/moe/train.py experiments/grug/moe/launch.py experiments/grug/moe/model.py experiments/grug/moe/launch_cw_jaxpp_may_d2560.py`
  - `./infra/pre-commit.py --fix experiments/grug/moe/train.py experiments/grug/moe/launch.py experiments/grug/moe/model.py experiments/grug/moe/launch_cw_jaxpp_may_d2560.py .agents/logbooks/jaxpp-grug-moe.md`
  - `TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --schedule std_1f1b --implementation auto --physical-stages 4 --microbatches 4 --nodes 4 --gpus-per-replica 8 --expert-axis 8 --layers 4 --experts 8 --top-k 1 --vocab-size 8192 --batch 32 --seq-len 128 --moe-implementation ring --loss-implementation xla --steps 1 --tracker json_logger --xla-memory-fraction 0.70 --conservative-loop-clustering false --run-id jaxpp-d2560-l4-auto-std-noqbstate-20260708-180908`
- Result:
  - `/dlwh/iris-run-job-20260708-180542` reached hparams and train-loop setup with the corrected DCN mesh, then failed in JaxPP compile with `AssertionError` involving `float32[4,8]` and the `int32[32,128]` token batch.
  - Code change: `GrugTrainState.pending_qb_betas` is now optional; automatic JaxPP replaces it with `None` before compiling, while explicit pipeline splitting still requires real QB tensors.
  - Targeted Python compile and pre-commit passed.
  - New automatic follow-up job: `/dlwh/iris-run-job-20260708-180907`.
  - Explicit 4-stage follow-up `/dlwh/iris-run-job-20260708-180600` is still running.
- Interpretation:
  - Automatic JaxPP has now cleared trainer mesh validation and reduced the failing aux state to the QB leaf specifically; the next run tests whether removing that leaf is sufficient.
- Next action:
  - Poll `/dlwh/iris-run-job-20260708-180907` and `/dlwh/iris-run-job-20260708-180600`.

### 2026-07-08 11:13 PDT - explicit 4-stage run emits loss and MFU
- Hypothesis: The explicit `jaxpp.experimental.mpmd` implementation should scale from the earlier 2-stage reduced smoke to a 4-stage CoreWeave topology once the launcher mesh places `pipeline` in DCN axes.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe` at baseline `5c36c4374a`.
- Command:
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job logs /dlwh/iris-run-job-20260708-180600 --tail --max-lines 1200`
- Config:
  - Explicit MPMD, `PP_STAGES=4`, `PP_MPMD_DIM=4`, `PP_MICROBATCHES=1`.
  - 4 CoreWeave H100 nodes, 8 GPUs per node, `MAY_EXPERT_AXIS=8`.
  - d2560, 24 layers, 8 experts, top-k 4, vocab 8192, batch 8, seq 32, ring MoE, XLA CE.
  - `XLA_PYTHON_CLIENT_MEM_FRACTION=0.50`, `TF_GPU_ALLOCATOR=cuda_malloc_async`.
- Result:
  - `/dlwh/iris-run-job-20260708-180600` reached hparams on all four stages and reported `parameter_count=2,829,071,552`.
  - It compiled the expected explicit graph pieces: `grug_stage0_forward`, `grug_stage1_forward`, `grug_stage2_forward`, `grug_stage3_loss_backward`, stage updates/backwards, and `grug_keep_step`.
  - Stage 0 emitted one training result: `train/loss=9.041923522949219`, `throughput/mfu=8.186476590141406e-05`, `throughput/duration=108.68707649502903`, `tokens_per_second=2.3553858310993263`.
  - The Iris parent job was still marked running at the time of this entry, so the run has useful metrics but is not yet sealed as fully exited.
- Interpretation:
  - The explicit-MPMD path now demonstrates 4-stage, stage-local weight/optimizer partitioning on 4x8 H100 and emits an MFU datapoint.
  - The MFU includes first-compile cost and a tiny one-step reduced workload, so it is a correctness/placement signal rather than a throughput claim.
- Next action:
  - Keep polling `/dlwh/iris-run-job-20260708-180600` for clean exit or nonzero-stage hang; keep polling `/dlwh/iris-run-job-20260708-180907` for the automatic no-QB-state compile result.

### 2026-07-08 11:15 PDT - explicit clean exit; automatic blocked on batch placement
- Hypothesis: Removing QB tensors from the automatic pipeline state should leave no after-loop tensor that JaxPP rejects.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe` at baseline `5c36c4374a`.
- Commands:
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job summary /dlwh/iris-run-job-20260708-180600`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job summary /dlwh/iris-run-job-20260708-180907`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job logs /dlwh/iris-run-job-20260708-180907 --tail --max-lines 1200`
- Result:
  - `/dlwh/iris-run-job-20260708-180600` fully succeeded at the Iris level (`State: succeeded`) after `6 minutes and 45.44 seconds`.
  - `/dlwh/iris-run-job-20260708-180907` reached hparams and train setup, with `parameter_count=507,560,992`.
  - Automatic `std_1f1b` still failed during `train_step.compile`, but the assertion is now narrowed to the batch token input only: `AssertionError: ([Var(...):int32[32@(replica_dcn,data,expert),128]], [P()])`.
- Interpretation:
  - Explicit-MPMD is the working path for now: 4-stage CoreWeave topology, stage-local weights/optimizer state, clean job exit, and a first MFU datapoint.
  - The automatic `mpmd_jit_with_loop`/`treduce` path no longer fails on QB state, but JaxPP still tries to place the input token batch as replicated scalar-like `P()` after-loop state. The next automatic fix should focus on batch input placement/lifetime around `treduce`, not model weights.
- Next action:
  - For throughput work, scale the explicit-MPMD path first. Resume automatic schedule work only after understanding why JaxPP's after-loop replication includes the batch token var.

### 2026-07-08 11:16 PDT - launch explicit 64-expert scale probe
- Hypothesis: The explicit-MPMD path that succeeded at 8 experts should scale to a larger 64-expert, top-k 4 workload on the same 4-stage CoreWeave topology and produce a more useful MFU signal over multiple steps.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe` at baseline `5c36c4374a`.
- Command:
  - `TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --implementation explicit_mpmd --physical-stages 4 --logical-stages 4 --microbatches 1 --nodes 4 --gpus-per-replica 8 --expert-axis 8 --layers 24 --experts 64 --top-k 4 --vocab-size 8192 --batch 32 --seq-len 128 --moe-implementation ring --loss-implementation xla --steps 3 --tracker json_logger --xla-memory-fraction 0.80 --run-id jaxpp-d2560-l24-explicit-4stage-e64-b32-s128-20260708-1116`
- Config:
  - Explicit MPMD, 4 physical/logical stages, 1 microbatch.
  - 4 CoreWeave H100 nodes, 8 GPUs per node, expert axis 8.
  - d2560, 24 layers, 64 experts, top-k 4, vocab 8192, batch 32, seq 128, ring MoE, XLA CE, 3 train steps.
  - `XLA_PYTHON_CLIENT_MEM_FRACTION=0.80`, `TF_GPU_ALLOCATOR=cuda_malloc_async`.
- Result:
  - Submitted as `/dlwh/iris-run-job-20260708-181634`.
  - Pending at entry time.
- Interpretation:
  - This is the next explicit-MPMD scaling rung toward May shape; it keeps vocab/sequence reduced but materially increases experts, tokens/step, and step count.
- Next action:
  - Poll `/dlwh/iris-run-job-20260708-181634`; if it succeeds, try 256 experts at reduced sequence/batch or increase sequence/batch depending on the limiting factor.

### 2026-07-08 11:23 PDT - explicit 64-expert probe succeeds at 0.75 MFU
- Hypothesis: The explicit-MPMD path that succeeded at 8 experts should scale to a larger 64-expert, top-k 4 workload on the same 4-stage CoreWeave topology and produce a more useful MFU signal over multiple steps.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe` at baseline `5c36c4374a`.
- Commands:
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job summary /dlwh/iris-run-job-20260708-181634`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job logs /dlwh/iris-run-job-20260708-181634 --tail --max-lines 1200`
- Config:
  - Explicit MPMD, 4 physical/logical stages, 1 microbatch.
  - 4 CoreWeave H100 nodes, 8 GPUs per node, expert axis 8.
  - d2560, 24 layers, 64 experts, top-k 4, vocab 8192, batch 32, seq 128, ring MoE, XLA CE, 3 train steps.
  - `XLA_PYTHON_CLIENT_MEM_FRACTION=0.80`, `TF_GPU_ALLOCATOR=cuda_malloc_async`.
- Result:
  - `/dlwh/iris-run-job-20260708-181634` fully succeeded at the Iris level (`State: succeeded`) after `5 minutes and 1.93 seconds`.
  - Reported `parameter_count=16,044,571,136`.
  - Stage 0 emitted stable post-compile metrics by step 2: `train/loss=9.035113334655762`, `tokens_per_second=21483.455929944237`, `gflops_per_second=238404.24097989046`, and `throughput/mfu=0.7529189015282038`.
  - Summary metrics reported `throughput/mean_mfu=0.7529189015282038`, `p10/p50/p90=0.7529189015282038`, `sample_count=2`.
- Interpretation:
  - The explicit JaxPP/MpMD path is now a real throughput path, not just a compile smoke: stage-local weights/optimizer state work on a 4x8 H100 topology with a 16B-parameter 24-layer MoE.
  - The lowered prealloc fraction plus `cuda_malloc_async` avoided the earlier full-state init OOM pattern at this scale.
- Next action:
  - Launch the 256-expert top-k 4 rung with the same lowered-prealloc setup, keeping vocab and sequence reduced first. If memory fails, reduce batch before backing off expert count.

### 2026-07-08 11:24 PDT - launch explicit 256-expert scale probe
- Hypothesis: The explicit-MPMD implementation that reached 0.75 MFU at 64 experts can carry the requested 256-expert, top-k 4 expert scale if batch is reduced before token/sequence scale and XLA preallocation remains below the earlier 0.88-0.90 settings.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe` at baseline `5c36c4374a`.
- Command:
  - `TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --implementation explicit_mpmd --physical-stages 4 --logical-stages 4 --microbatches 1 --nodes 4 --gpus-per-replica 8 --expert-axis 8 --layers 24 --experts 256 --top-k 4 --vocab-size 8192 --batch 16 --seq-len 128 --moe-implementation ring --loss-implementation xla --steps 3 --tracker json_logger --xla-memory-fraction 0.80 --run-id jaxpp-d2560-l24-explicit-4stage-e256-b16-s128-20260708-1125`
- Config:
  - Explicit MPMD, 4 physical/logical stages, 1 microbatch.
  - 4 CoreWeave H100 nodes, 8 GPUs per node, expert axis 8.
  - d2560, 24 layers, 256 experts, top-k 4, vocab 8192, batch 16, seq 128, ring MoE, XLA CE, 3 train steps.
  - `XLA_PYTHON_CLIENT_MEM_FRACTION=0.80`, `TF_GPU_ALLOCATOR=cuda_malloc_async`.
- Result:
  - Submitted as `/dlwh/iris-run-job-20260708-182427`.
  - Initial poll showed Iris setup running.
- Interpretation:
  - This tests the expert-count target directly while holding vocabulary, sequence length, and batch below full May shape.
- Next action:
  - Poll `/dlwh/iris-run-job-20260708-182427`; if it OOMs, reduce batch or tune preallocation before reducing experts.

### 2026-07-08 11:28 PDT - 256-expert probe hits XLA base limit at 0.80
- Hypothesis: The explicit-MPMD implementation can carry 256 experts if batch is reduced and XLA preallocation stays below the earlier high settings.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe` at baseline `5c36c4374a`.
- Commands:
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job logs /dlwh/iris-run-job-20260708-182427 --tail --max-lines 5000 | rg -C 4 'RESOURCE_EXHAUSTED|out of memory|Out of memory|failed to allocate|Current allocation summary|bytes|GiB|MiB|Traceback|ERROR|RuntimeError'`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job stop /dlwh/iris-run-job-20260708-182427`
- Config:
  - Explicit MPMD, 4 physical/logical stages, d2560, 24 layers, 256 experts, top-k 4, vocab 8192, batch 16, seq 128.
  - `XLA_PYTHON_CLIENT_MEM_FRACTION=0.80`, `TF_GPU_ALLOCATOR=cuda_malloc_async`.
- Result:
  - The run reached JAX distributed startup and hparams on all four stages.
  - XLA reported `The byte size of input/output arguments (68289171792) exceeds the base limit (68015673538)`.
  - Rematerialization could not reduce below `63.60GiB (68289171784 bytes)`.
  - Workers then emitted repeated BFC OOM messages for tiny allocations (`1.25MiB`, then `4B`), so the job was stopped manually as `/dlwh/iris-run-job-20260708-182427`.
- Interpretation:
  - This failure is not driven by batch activations; the stage input/output state is already slightly larger than the `0.80` XLA base limit.
  - Reducing batch further is unlikely to fix the first failure. The next run should raise the fraction modestly while staying below the earlier `0.88-0.90` high-prealloc settings.
- Next action:
  - Retry the same 256-expert batch-16 shape at `XLA_PYTHON_CLIENT_MEM_FRACTION=0.84`.

### 2026-07-08 11:28 PDT - launch explicit 256-expert retry at 0.84
- Hypothesis: The 256-expert batch-16 shape failed at `0.80` only because the stage input/output footprint was slightly above the XLA base limit; `0.84` should leave enough headroom while still avoiding the earlier high `0.88-0.90` preallocation settings.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe` at baseline `5c36c4374a`.
- Command:
  - `TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --implementation explicit_mpmd --physical-stages 4 --logical-stages 4 --microbatches 1 --nodes 4 --gpus-per-replica 8 --expert-axis 8 --layers 24 --experts 256 --top-k 4 --vocab-size 8192 --batch 16 --seq-len 128 --moe-implementation ring --loss-implementation xla --steps 3 --tracker json_logger --xla-memory-fraction 0.84 --run-id jaxpp-d2560-l24-explicit-4stage-e256-b16-s128-xla084-20260708-1128`
- Config:
  - Explicit MPMD, 4 physical/logical stages, d2560, 24 layers, 256 experts, top-k 4, vocab 8192, batch 16, seq 128, ring MoE, XLA CE.
  - `XLA_PYTHON_CLIENT_MEM_FRACTION=0.84`, `TF_GPU_ALLOCATOR=cuda_malloc_async`.
- Result:
  - Submitted as `/dlwh/iris-run-job-20260708-182830`.
- Interpretation:
  - This is the same requested expert-scale target as the failed 0.80 attempt; only the memory fraction changed.
- Next action:
  - Poll `/dlwh/iris-run-job-20260708-182830` for init/compile progress and MFU.

### 2026-07-08 11:32 PDT - 256-expert retry clears base limit but OOMs at 400 MiB allocation
- Hypothesis: Raising the memory fraction from `0.80` to `0.84` should clear the XLA stage input/output base-limit failure for 256 experts while staying below the earlier high preallocation settings.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe` at baseline `5c36c4374a`.
- Commands:
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job summary /dlwh/iris-run-job-20260708-182830`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job logs /dlwh/iris-run-job-20260708-182830 --tail --max-lines 4000 | rg 'parameter_count|throughput/|train/loss|Compiling|Finished|compile|OOM|RESOURCE_EXHAUSTED|out of memory|failed to allocate|base limit|Current allocation summary|Traceback|AssertionError|ERROR|finished|summary'`
- Config:
  - Explicit MPMD, 4 physical/logical stages, d2560, 24 layers, 256 experts, top-k 4, vocab 8192, batch 16, seq 128.
  - `XLA_PYTHON_CLIENT_MEM_FRACTION=0.84`, `TF_GPU_ALLOCATOR=cuda_malloc_async`.
- Result:
  - `/dlwh/iris-run-job-20260708-182830` failed after `3 minutes and 13.64 seconds`.
  - The run cleared the earlier base-limit check and reported `parameter_count=61,354,855,424`.
  - It also reported `throughput/flops_per_token_analytic=3,722,629,120`, `throughput/flops_per_example_analytic=1,429,489,582,080`, and H100 theoretical FLOPs `3.1664e+16`.
  - It then failed on all stages with `jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 400.00MiB`.
- Interpretation:
  - The 256-expert 24-layer target reaches full model initialization/accounting with stage-local explicit MPMD, but the 0.84 allocator pool leaves too little runtime slack.
  - Since the failed allocation is only 400 MiB after a 61.35B-param model is resident, try one `0.88` run before reducing expert count. If `0.88` still fails, reduce to 192 or 128 experts rather than spending more runs on batch changes.
- Next action:
  - Retry 256 experts at `XLA_PYTHON_CLIENT_MEM_FRACTION=0.88`.

### 2026-07-08 11:33 PDT - launch explicit 256-expert retry at 0.88
- Hypothesis: The 256-expert batch-16 shape is close enough that the default-lower `0.88` fraction may clear the 400 MiB runtime allocation failure seen at `0.84`.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe` at baseline `5c36c4374a`.
- Command:
  - `TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --implementation explicit_mpmd --physical-stages 4 --logical-stages 4 --microbatches 1 --nodes 4 --gpus-per-replica 8 --expert-axis 8 --layers 24 --experts 256 --top-k 4 --vocab-size 8192 --batch 16 --seq-len 128 --moe-implementation ring --loss-implementation xla --steps 3 --tracker json_logger --xla-memory-fraction 0.88 --run-id jaxpp-d2560-l24-explicit-4stage-e256-b16-s128-xla088-20260708-1133`
- Config:
  - Explicit MPMD, 4 physical/logical stages, d2560, 24 layers, 256 experts, top-k 4, vocab 8192, batch 16, seq 128.
  - `XLA_PYTHON_CLIENT_MEM_FRACTION=0.88`, `TF_GPU_ALLOCATOR=cuda_malloc_async`.
- Result:
  - Submitted as `/dlwh/iris-run-job-20260708-183245`.
- Interpretation:
  - This is the last direct 256-expert memory-fraction probe before reducing expert count.
- Next action:
  - Poll `/dlwh/iris-run-job-20260708-183245`.

### 2026-07-08 11:36 PDT - 256-expert retry at 0.88 still OOMs
- Hypothesis: The 256-expert batch-16 shape may only need the default-lower `0.88` memory fraction to clear the 400 MiB allocation failure observed at `0.84`.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe` at baseline `5c36c4374a`.
- Commands:
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job summary /dlwh/iris-run-job-20260708-183245`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job logs /dlwh/iris-run-job-20260708-183245 --tail --max-lines 4500 | rg 'parameter_count|throughput/|train/loss|Compiling|Finished|compile|OOM|RESOURCE_EXHAUSTED|out of memory|failed to allocate|base limit|Current allocation summary|Traceback|AssertionError|ERROR|finished|summary'`
- Config:
  - Explicit MPMD, 4 physical/logical stages, d2560, 24 layers, 256 experts, top-k 4, vocab 8192, batch 16, seq 128.
  - `XLA_PYTHON_CLIENT_MEM_FRACTION=0.88`, `TF_GPU_ALLOCATOR=cuda_malloc_async`.
- Result:
  - `/dlwh/iris-run-job-20260708-183245` failed after `3 minutes and 4.54 seconds`.
  - It again reported `parameter_count=61,354,855,424` and the same analytic FLOP numbers as the `0.84` run.
  - It again failed on all stages with `jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 400.00MiB`.
- Interpretation:
  - The current explicit-MPMD state layout can initialize/account for the 256-expert 24-layer model but cannot execute it on 4x8 H100 at batch 16 with practical memory fractions up to `0.88`.
  - The repeated 400 MiB failure after full parameter accounting suggests reducing batch is not the right first fix; reduce expert count while preserving 24 layers/top-k 4.
- Next action:
  - Launch a 192-expert, top-k 4 rung at `XLA_PYTHON_CLIENT_MEM_FRACTION=0.80`.

### 2026-07-08 11:37 PDT - launch explicit 192-expert reduced rung
- Hypothesis: With 256 experts ruled out by repeated 400 MiB runtime OOMs, a 192-expert 24-layer top-k 4 run should preserve most of the expert-scale target while fitting under the lower `0.80` memory fraction.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe` at baseline `5c36c4374a`.
- Command:
  - `TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --implementation explicit_mpmd --physical-stages 4 --logical-stages 4 --microbatches 1 --nodes 4 --gpus-per-replica 8 --expert-axis 8 --layers 24 --experts 192 --top-k 4 --vocab-size 8192 --batch 16 --seq-len 128 --moe-implementation ring --loss-implementation xla --steps 3 --tracker json_logger --xla-memory-fraction 0.80 --run-id jaxpp-d2560-l24-explicit-4stage-e192-b16-s128-20260708-1137`
- Config:
  - Explicit MPMD, 4 physical/logical stages, d2560, 24 layers, 192 experts, top-k 4, vocab 8192, batch 16, seq 128.
  - `XLA_PYTHON_CLIENT_MEM_FRACTION=0.80`, `TF_GPU_ALLOCATOR=cuda_malloc_async`.
- Result:
  - Submitted as `/dlwh/iris-run-job-20260708-183650`.
- Interpretation:
  - This is the nearest reduced-expert rung after 256 experts failed at `0.80`, `0.84`, and `0.88`.
- Next action:
  - Poll `/dlwh/iris-run-job-20260708-183650` for fit and MFU.

### 2026-07-08 11:40 PDT - 192-expert lower-fraction run OOMs at 300 MiB allocation
- Hypothesis: A 192-expert 24-layer top-k 4 run should preserve most of the expert-scale target while fitting under the lower `0.80` memory fraction.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe` at baseline `5c36c4374a`.
- Commands:
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job summary /dlwh/iris-run-job-20260708-183650`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job logs /dlwh/iris-run-job-20260708-183650 --tail --max-lines 5000 | rg 'parameter_count|throughput/|train/loss|Compiling|Finished|compile|OOM|RESOURCE_EXHAUSTED|out of memory|failed to allocate|base limit|Current allocation summary|Traceback|AssertionError|ERROR|finished|summary'`
- Config:
  - Explicit MPMD, 4 physical/logical stages, d2560, 24 layers, 192 experts, top-k 4, vocab 8192, batch 16, seq 128.
  - `XLA_PYTHON_CLIENT_MEM_FRACTION=0.80`, `TF_GPU_ALLOCATOR=cuda_malloc_async`.
- Result:
  - `/dlwh/iris-run-job-20260708-183650` failed after `3 minutes and 4.93 seconds`.
  - It reported `parameter_count=46,251,427,328`, `throughput/flops_per_token_analytic=3,714,764,800`, and `throughput/flops_per_example_analytic=1,426,469,683,200`.
  - It failed on all stages with `jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 300.00MiB`.
- Interpretation:
  - 192 experts is much closer to fitting than 256, but the `0.80` pool is too tight after model state is resident.
  - Unlike the 256-expert run, a modest fraction increase has a plausible chance of fitting because the failed allocation is smaller and the parameter footprint is ~15.1B parameters lower.
- Next action:
  - Retry 192 experts at `XLA_PYTHON_CLIENT_MEM_FRACTION=0.84`.

### 2026-07-08 11:41 PDT - launch explicit 192-expert retry at 0.84
- Hypothesis: A modest memory-fraction increase from `0.80` to `0.84` should give the 192-expert shape enough runtime slack to clear the previous 300 MiB allocation failure.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe` at baseline `5c36c4374a`.
- Command:
  - `TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --implementation explicit_mpmd --physical-stages 4 --logical-stages 4 --microbatches 1 --nodes 4 --gpus-per-replica 8 --expert-axis 8 --layers 24 --experts 192 --top-k 4 --vocab-size 8192 --batch 16 --seq-len 128 --moe-implementation ring --loss-implementation xla --steps 3 --tracker json_logger --xla-memory-fraction 0.84 --run-id jaxpp-d2560-l24-explicit-4stage-e192-b16-s128-xla084-20260708-1141`
- Config:
  - Explicit MPMD, 4 physical/logical stages, d2560, 24 layers, 192 experts, top-k 4, vocab 8192, batch 16, seq 128.
  - `XLA_PYTHON_CLIENT_MEM_FRACTION=0.84`, `TF_GPU_ALLOCATOR=cuda_malloc_async`.
- Result:
  - Submitted as `/dlwh/iris-run-job-20260708-184109`.
- Interpretation:
  - This keeps the nearest reduced-expert target and uses a lower fraction than the direct 256-expert attempts at `0.88`.
- Next action:
  - Poll `/dlwh/iris-run-job-20260708-184109` for fit and MFU.

### 2026-07-08 11:45 PDT - 192-expert retry at 0.84 still OOMs
- Hypothesis: A modest memory-fraction increase from `0.80` to `0.84` should give the 192-expert shape enough runtime slack to clear the previous 300 MiB allocation failure.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe` at baseline `5c36c4374a`.
- Commands:
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job summary /dlwh/iris-run-job-20260708-184109`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job logs /dlwh/iris-run-job-20260708-184109 --tail --max-lines 6000 | rg 'parameter_count|throughput/|train/loss|Compiling|Finished|compile|OOM|RESOURCE_EXHAUSTED|out of memory|failed to allocate|base limit|Current allocation summary|Traceback|AssertionError|ERROR|finished|summary'`
- Config:
  - Explicit MPMD, 4 physical/logical stages, d2560, 24 layers, 192 experts, top-k 4, vocab 8192, batch 16, seq 128.
  - `XLA_PYTHON_CLIENT_MEM_FRACTION=0.84`, `TF_GPU_ALLOCATOR=cuda_malloc_async`.
- Result:
  - `/dlwh/iris-run-job-20260708-184109` failed after `3 minutes and 32.58 seconds`.
  - It again reported `parameter_count=46,251,427,328` and the same analytic FLOP numbers as the `0.80` run.
  - It again failed on all stages with `jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 300.00MiB`.
- Interpretation:
  - The current explicit-MPMD state layout does not fit 192 experts on 4x8 H100, even with `0.84`.
  - The next useful reduced expert count is 128. Since 64 experts succeeded at batch 32, try 128 experts with batch 32 and `0.80` to get a stronger MFU point.
- Next action:
  - Launch 128 experts, top-k 4, 24 layers, batch 32, seq 128 at `XLA_PYTHON_CLIENT_MEM_FRACTION=0.80`.

### 2026-07-08 11:46 PDT - launch explicit 128-expert batch-32 rung
- Hypothesis: Since 64 experts succeeded at batch 32 but 192 experts failed even at batch 16, 128 experts should fit at batch 32 and provide the strongest reduced-expert MFU point so far.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe` at baseline `5c36c4374a`.
- Command:
  - `TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --implementation explicit_mpmd --physical-stages 4 --logical-stages 4 --microbatches 1 --nodes 4 --gpus-per-replica 8 --expert-axis 8 --layers 24 --experts 128 --top-k 4 --vocab-size 8192 --batch 32 --seq-len 128 --moe-implementation ring --loss-implementation xla --steps 3 --tracker json_logger --xla-memory-fraction 0.80 --run-id jaxpp-d2560-l24-explicit-4stage-e128-b32-s128-20260708-1146`
- Config:
  - Explicit MPMD, 4 physical/logical stages, d2560, 24 layers, 128 experts, top-k 4, vocab 8192, batch 32, seq 128.
  - `XLA_PYTHON_CLIENT_MEM_FRACTION=0.80`, `TF_GPU_ALLOCATOR=cuda_malloc_async`.
- Result:
  - Submitted as `/dlwh/iris-run-job-20260708-184545`.
- Interpretation:
  - This keeps the lower preallocation fraction and restores the 64-expert run's token batch while doubling expert count.
- Next action:
  - Poll `/dlwh/iris-run-job-20260708-184545` for fit and MFU.

### 2026-07-08 11:52 PDT - explicit 128-expert batch-32 rung succeeds
- Hypothesis: Since 64 experts succeeded at batch 32 but 192 experts failed even at batch 16, 128 experts should fit at batch 32 and provide the strongest reduced-expert MFU point so far.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe` at baseline `5c36c4374a`.
- Commands:
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job summary /dlwh/iris-run-job-20260708-184545`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job logs /dlwh/iris-run-job-20260708-184545 --tail --max-lines 7000 | rg 'parameter_count|throughput/|train/loss|Compiling|Finished|compile|OOM|RESOURCE_EXHAUSTED|out of memory|failed to allocate|base limit|Current allocation summary|Traceback|AssertionError|ERROR|finished|summary|event": "metrics"'`
- Config:
  - Explicit MPMD, 4 physical/logical stages, d2560, 24 layers, 128 experts, top-k 4, vocab 8192, batch 32, seq 128.
  - `XLA_PYTHON_CLIENT_MEM_FRACTION=0.80`, `TF_GPU_ALLOCATOR=cuda_malloc_async`.
- Result:
  - `/dlwh/iris-run-job-20260708-184545` fully succeeded at the Iris level (`State: succeeded`) after `5 minutes and 42.93 seconds`.
  - Reported `parameter_count=31,147,999,232`.
  - Stage 0 emitted step-2 metrics: `train/loss=9.034143447875977`, `tokens_per_second=2945.7509327081166`, `gflops_per_second=32758.816639248496`, and `throughput/mfu=0.1034576068697843`.
  - Summary metrics reported `throughput/mean_mfu=0.1034576068697843`, `p10/p50/p90=0.1034576068697843`, and `mfu_sample_count=2`.
- Interpretation:
  - The explicit MPMD path supports 24-layer top-k 4 Grug MoE at 128 experts on 4x8 H100 with lower `0.80` preallocation and stage-local weights/optimizer state.
  - 192 and 256 experts initialize/account for params but fail runtime allocation under the current explicit state layout, so 128 experts is the largest completed reduced expert rung in this session.
- Next action:
  - Report the 64/128/192/256 expert ladder on issue #7024, including the automatic JaxPP batch-placement blocker and the memory limit for larger explicit-MPMD shapes.

### 2026-07-08 11:56 PDT - lower launcher memory fraction and move auto microbatch reshape outside JaxPP trace
- Hypothesis: The automatic JaxPP `std_1f1b` path's after-loop placement assertion on the token batch may be caused by reshaping the full batch inside the JaxPP-traced function. If the compiled step receives the already-microbatched batch tree, the full batch should no longer appear as an after-loop value. Separately, the default launcher memory fraction should match the successful lower-preallocation explicit runs.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe` at baseline `5c36c4374a`.
- Commands:
  - `./infra/pre-commit.py --fix experiments/grug/moe/train.py experiments/grug/moe/run_cw_jaxpp_may_d2560.sh`
  - `TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --schedule std_1f1b --implementation auto --steps 1 --tracker json_logger --layers 4 --experts 8 --top-k 1 --batch 32 --seq-len 128 --vocab-size 8192 --expert-axis 8 --microbatches 4 --moe-implementation ring --loss-implementation xla --ce-autotune-on-miss false --conservative-loop-clustering false --xla-memory-fraction 0.80 --run-id jaxpp-auto-std-l4-e8-microarg-20260708-185618`
- Config:
  - Automatic JaxPP, `std_1f1b`, 4 physical/logical stages, 4 microbatches, d2560, 4 layers, 8 experts, top-k 1, batch 32, seq 128, vocab 8192, ring MoE, XLA CE.
  - `XLA_PYTHON_CLIENT_MEM_FRACTION=0.80`, `TF_GPU_ALLOCATOR=cuda_malloc_async`.
  - Launcher default changed from `0.88` to `0.80`.
- Result:
  - Submitted as `/dlwh/iris-run-job-20260708-185621`.
  - The run failed before JaxPP compile with `ValueError: Using PartitionSpec when you are not under a mesh context is not allowed. Got P(None, ('replica_dcn', 'data', 'expert'), None)`.
  - The failure occurred at `_reshape_batch_for_pipeline(...)` in the outer Python loop after moving the reshape outside the JaxPP trace.
- Interpretation:
  - Moving the reshape out of the trace is still plausible, but it must run under the global Grug mesh so the `PartitionSpec` in `out_sharding` can be canonicalized.
  - Lowering the launcher default to `0.80` matches the successful explicit 64/128-expert runs and avoids the higher preallocation default unless explicitly requested.
- Next action:
  - Wrap the pre-JaxPP batch reshape in `set_mesh(mesh)` and rerun the same automatic reduced probe.

### 2026-07-08 12:00 PDT - auto microbatch pre-reshape clears batch placement but hits JaxPP sharding inference
- Hypothesis: Running `_reshape_batch_for_pipeline` under the global Grug mesh before JaxPP compile should preserve the unsharded microbatch axis without reintroducing the full batch as an after-loop value.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe` at baseline `5c36c4374a`.
- Commands:
  - `uv run python -m py_compile experiments/grug/moe/train.py`
  - `./infra/pre-commit.py --fix experiments/grug/moe/train.py experiments/grug/moe/run_cw_jaxpp_may_d2560.sh`
  - `TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --schedule std_1f1b --implementation auto --steps 1 --tracker json_logger --layers 4 --experts 8 --top-k 1 --batch 32 --seq-len 128 --vocab-size 8192 --expert-axis 8 --microbatches 4 --moe-implementation ring --loss-implementation xla --ce-autotune-on-miss false --conservative-loop-clustering false --xla-memory-fraction 0.80 --run-id jaxpp-auto-std-l4-e8-meshreshape-20260708-185958`
- Config:
  - Same reduced automatic `std_1f1b` probe as the prior entry, with the pre-reshape wrapped in the global mesh context.
- Result:
  - Submitted as `/dlwh/iris-run-job-20260708-190000`.
  - The prior batch after-loop placement assertion did not recur.
  - The run failed in `jaxpp/core.py` during `TraceableFunction.compile(...).infer_intermediate_shardings()`:
    `AssertionError` at `jaxpp/sharding_inference.py:613`, comparing equation input specs with environment specs.
  - The assertion payload includes Grug sharded parameter avals such as `float32[2560@data,2560@model]`, `float32[8@expert,2560@data,1280@model]`, and inferred specs like `P(None, None)` / `P('expert', None, None)`.
- Interpretation:
  - The batch lifetime issue appears fixed. The automatic schedule path is now blocked by JaxPP explicit sharding inference over Grug's manually sharded parameters.
  - This is a different blocker from the earlier token-batch after-loop assertion and occurs before any schedule-specific MFU can be measured.
- Next action:
  - Test whether automatic JaxPP can use a non-explicit mesh to route through JaxPP's non-explicit sharding reconciliation path.

### 2026-07-08 12:04 PDT - non-explicit mesh path is incompatible with Grug init
- Hypothesis: If the automatic JaxPP path uses a non-explicit stage mesh, JaxPP may avoid the stricter `infer_shardings_explicit` assertion and use ordinary sharding reconciliation instead.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe` at baseline `5c36c4374a`.
- Commands:
  - `uv run python -m py_compile experiments/grug/moe/train.py experiments/grug/moe/launch.py`
  - `./infra/pre-commit.py --fix experiments/grug/moe/train.py experiments/grug/moe/launch.py experiments/grug/moe/run_cw_jaxpp_may_d2560.sh`
  - `TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --schedule std_1f1b --implementation auto --steps 1 --tracker json_logger --layers 4 --experts 8 --top-k 1 --batch 32 --seq-len 128 --vocab-size 8192 --expert-axis 8 --microbatches 4 --moe-implementation ring --loss-implementation xla --ce-autotune-on-miss false --conservative-loop-clustering false --xla-memory-fraction 0.80 --run-id jaxpp-auto-std-l4-e8-implicitmesh-20260708-190436`
- Config:
  - Same reduced automatic `std_1f1b` probe, but with an experimental patch making automatic JaxPP use a non-explicit mesh and `TrainerConfig.use_explicit_mesh_axes=false`.
- Result:
  - Submitted as `/dlwh/iris-run-job-20260708-190439`.
  - The run failed during model initialization before JaxPP trace:
    `ValueError: PartitionSpec passed to reshard cannot contain axis names that are of type Auto or Manual. Got PartitionSpec: P('model', ('replica_dcn', 'data')) with axis name: model of type: AxisType.Auto`.
  - The failure points at `Transformer.init` when initializing `token_embed`.
  - The experimental non-explicit mesh patch was reverted.
- Interpretation:
  - Grug's model initialization depends on explicit mesh axes because it calls `reshard(..., PartitionSpec(...))` directly for model/data/expert-sharded parameters.
  - Automatic JaxPP therefore needs to handle Grug's explicit shardings, or we need a separate no-sharding model/launcher variant for schedule-only probes.
- Next action:
  - Keep automatic JaxPP on explicit axes and treat sharding inference as the current schedule-sweep blocker.

### 2026-07-08 12:10 PDT - 4x1 automatic no-intra-stage-sharding control hits topology validation
- Hypothesis: A 4x1 H100 automatic run with one GPU per physical pipeline stage might avoid real data/model/expert partitioning and show whether automatic schedules can run when intra-stage sharding is effectively absent.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe` at baseline `5c36c4374a`.
- Command:
  - `TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --schedule std_1f1b --implementation auto --steps 1 --tracker json_logger --nodes 4 --gpus-per-replica 1 --physical-stages 4 --microbatches 4 --layers 4 --experts 8 --top-k 1 --batch 4 --seq-len 128 --vocab-size 8192 --expert-axis 1 --moe-implementation ring --loss-implementation xla --ce-autotune-on-miss false --conservative-loop-clustering false --xla-memory-fraction 0.80 --run-id jaxpp-auto-std-l4-e8-4x1-20260708-190938`
- Config:
  - Automatic JaxPP, `std_1f1b`, 4 requested physical stages, 4 requested Iris replicas, 1 H100 per replica, d2560, 4 layers, 8 experts, top-k 1, batch 4, seq 128.
- Result:
  - Submitted as `/dlwh/iris-run-job-20260708-190941`.
  - The run failed before model init/JaxPP trace in Levanter mesh validation:
    `ValueError: DCN product 4 does not divide num_slices 2`.
- Interpretation:
  - This is a launcher/topology mismatch, not evidence for or against JaxPP schedules.
  - The current CoreWeave/Iris placement for `count=1, replicas=4` did not produce the four DCN slices required by `PP_MPMD_DIM=4`.
- Next action:
  - Either keep schedule debugging on the validated 4x8 topology or add a dedicated launcher path for smaller per-stage GPU counts before using 4x1 as a schedule control.

### 2026-07-08 12:33 PDT - explicit GPipe microbatch smoke succeeds with task-local reductions
- Hypothesis: The explicit MPMD GPipe path can support multiple microbatches if all cross-microbatch accumulation and averaging are expressed as `mpmd.task` calls rather than ordinary JAX arithmetic in the MPMD driver trace.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe` at baseline `5c36c4374a`.
- Commands:
  - `uv run python -m py_compile experiments/grug/moe/train.py`
  - `./infra/pre-commit.py --fix experiments/grug/moe/train.py`
  - `TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --schedule gpipe --implementation explicit_mpmd --physical-stages 4 --logical-stages 4 --microbatches 4 --nodes 4 --gpus-per-replica 8 --expert-axis 8 --layers 4 --experts 8 --top-k 1 --batch 32 --seq-len 128 --vocab-size 8192 --moe-implementation ring --loss-implementation xla --steps 1 --tracker json_logger --xla-memory-fraction 0.80 --run-id jaxpp-explicit-gpipe-l4-e8-taskreduce-20260708-192641`
- Config:
  - Explicit MPMD GPipe, 4 physical/logical stages, 4 microbatches, d2560, 4 layers, 8 experts, top-k 1, batch 32, seq 128, vocab 8192.
  - `XLA_PYTHON_CLIENT_MEM_FRACTION=0.80`, `TF_GPU_ALLOCATOR=cuda_malloc_async`.
- Result:
  - Parent launcher: `/dlwh/iris-run-job-20260708-192645`.
  - Child training job: `/dlwh/iris-run-job-20260708-192645/grug-train-jaxpp-explicit-gpipe-l4-e8-taskreduce-20260708-192641`.
  - Child succeeded on all 4 tasks.
  - Parameter count: `507,560,992`.
  - Step 0 metrics: loss `9.045722961425781`, tokens/s `58.40267300249191`, GFLOP/s `72.60306334899732`, MFU `0.00022929214044023916`.
- Interpretation:
  - The prior `jaxpp.experimental.mpmd ... got slice` failure was fixed by slicing microbatches outside the MPMD trace.
  - The follow-up `... got jit` failure was fixed by moving microbatch sum/average operations into named `mpmd.task` calls with stage-local shardings.
  - The measured MFU is compile-dominated because this smoke ran only one step on a tiny 4-layer model, but it proves functional multi-microbatch explicit GPipe execution.
- Next action:
  - Run the same explicit GPipe path with 24 layers and a few measured steps.

### 2026-07-08 12:39 PDT - 24-layer explicit GPipe run succeeds
- Hypothesis: The task-local GPipe implementation should scale from the 4-layer smoke to the user-accepted 24-layer depth while keeping stage-local weight and optimizer partitioning across 4 physical stages.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe` at baseline `5c36c4374a`.
- Command:
  - `TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --schedule gpipe --implementation explicit_mpmd --physical-stages 4 --logical-stages 4 --microbatches 4 --nodes 4 --gpus-per-replica 8 --expert-axis 8 --layers 24 --experts 8 --top-k 4 --batch 32 --seq-len 128 --vocab-size 8192 --moe-implementation ring --loss-implementation xla --steps 3 --tracker json_logger --xla-memory-fraction 0.80 --run-id jaxpp-explicit-gpipe-l24-e8-taskreduce-20260708-193310`
- Config:
  - Explicit MPMD GPipe, 4 physical/logical stages, 4 microbatches, d2560, 24 layers, 8 experts, top-k 4, batch 32, seq 128, vocab 8192.
  - `XLA_PYTHON_CLIENT_MEM_FRACTION=0.80`, `TF_GPU_ALLOCATOR=cuda_malloc_async`.
- Result:
  - Parent launcher: `/dlwh/iris-run-job-20260708-193316`.
  - Child training job: `/dlwh/iris-run-job-20260708-193316/grug-train-jaxpp-explicit-gpipe-l24-e8-taskreduce-20260708-193310`.
  - Child succeeded on all 4 tasks.
  - Parameter count: `2,829,071,552`.
  - Step 2 metrics: loss `9.036118507385254`, tokens/s `3806.936077293086`, GFLOP/s `42167.39284687717`, MFU `0.13317140237139077`.
  - Summary: `throughput/mean_mfu=0.13317140237139077`, `throughput/mfu_sample_count=2`.
- Interpretation:
  - 24-layer stage-split weights and optimizer state work under explicit MPMD GPipe with 4 microbatches.
  - This is a functional pipeline-parallel training run rather than only a compile smoke.
  - The MFU is lower than the earlier single-microbatch 64-expert rung because this 8-expert shape is much smaller; use a 64-expert GPipe run for a more meaningful schedule comparison.
- Next action:
  - Run explicit GPipe with 64 experts, top-k 4, 24 layers, batch 32, seq 128 at the same `0.80` memory fraction.

### 2026-07-08 12:47 PDT - 64-expert explicit GPipe run succeeds
- Hypothesis: The 24-layer GPipe path should remain stable at the earlier successful 64-expert parameter scale and provide a higher-signal MFU point than the 8-expert run.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe` at baseline `5c36c4374a`.
- Command:
  - `TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --schedule gpipe --implementation explicit_mpmd --physical-stages 4 --logical-stages 4 --microbatches 4 --nodes 4 --gpus-per-replica 8 --expert-axis 8 --layers 24 --experts 64 --top-k 4 --batch 32 --seq-len 128 --vocab-size 8192 --moe-implementation ring --loss-implementation xla --steps 3 --tracker json_logger --xla-memory-fraction 0.80 --run-id jaxpp-explicit-gpipe-l24-e64-taskreduce-20260708-193919`
- Config:
  - Explicit MPMD GPipe, 4 physical/logical stages, 4 microbatches, d2560, 24 layers, 64 experts, top-k 4, batch 32, seq 128, vocab 8192.
  - `XLA_PYTHON_CLIENT_MEM_FRACTION=0.80`, `TF_GPU_ALLOCATOR=cuda_malloc_async`.
- Result:
  - Parent launcher: `/dlwh/iris-run-job-20260708-193922`.
  - Child training job: `/dlwh/iris-run-job-20260708-193922/grug-train-jaxpp-explicit-gpipe-l24-e64-taskreduce-20260708-193919`.
  - Child succeeded on all 4 tasks.
  - Parameter count: `16,044,571,136`.
  - Step 2 metrics: loss `9.034236907958984`, tokens/s `13577.965711804029`, GFLOP/s `150676.15844160973`, MFU `0.47585952009098575`.
  - Summary: `throughput/mean_mfu=0.47585952009098575`, `throughput/mfu_sample_count=2`.
- Interpretation:
  - Explicit GPipe is stable at the same 64-expert scale where the single-microbatch explicit MPMD path previously reported `throughput/mean_mfu=0.7529189015282038`.
  - The GPipe implementation trades lower MFU for functional 4-microbatch pipeline execution; further tuning should compare 1F1B-style scheduling or reduce per-microbatch overhead.
- Next action:
  - Try the same GPipe path at 128 experts, matching the largest explicit single-microbatch rung that succeeded at batch 32.

### 2026-07-08 12:54 PDT - 128-expert explicit GPipe run succeeds
- Hypothesis: The GPipe implementation should still fit at the largest expert count that previously succeeded in the single-microbatch explicit MPMD path, despite storing multi-microbatch stage inputs for the backward pass.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe` at baseline `5c36c4374a`.
- Command:
  - `TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --schedule gpipe --implementation explicit_mpmd --physical-stages 4 --logical-stages 4 --microbatches 4 --nodes 4 --gpus-per-replica 8 --expert-axis 8 --layers 24 --experts 128 --top-k 4 --batch 32 --seq-len 128 --vocab-size 8192 --moe-implementation ring --loss-implementation xla --steps 3 --tracker json_logger --xla-memory-fraction 0.80 --run-id jaxpp-explicit-gpipe-l24-e128-taskreduce-20260708-194716`
- Config:
  - Explicit MPMD GPipe, 4 physical/logical stages, 4 microbatches, d2560, 24 layers, 128 experts, top-k 4, batch 32, seq 128, vocab 8192.
  - `XLA_PYTHON_CLIENT_MEM_FRACTION=0.80`, `TF_GPU_ALLOCATOR=cuda_malloc_async`.
- Result:
  - Parent launcher: `/dlwh/iris-run-job-20260708-194719`.
  - Child training job: `/dlwh/iris-run-job-20260708-194719/grug-train-jaxpp-explicit-gpipe-l24-e128-taskreduce-20260708-194716`.
  - Child succeeded on all 4 tasks.
  - Parameter count: `31,147,999,232`.
  - Step 2 metrics: loss `9.03332233428955`, tokens/s `10801.727899693138`, GFLOP/s `120122.79100860565`, MFU `0.3793670762020138`.
  - Summary: `throughput/mean_mfu=0.3793670762020138`, `throughput/mfu_sample_count=2`.
- Interpretation:
  - Explicit GPipe now works at 24 layers and 128 experts on 4x 8xH100 with stage-partitioned weights and optimizer state.
  - This is the largest completed GPipe result so far. 192/256 expert GPipe would likely face the same or worse memory pressure as the earlier single-microbatch explicit MPMD rungs.
  - Compared with the single-microbatch explicit 128-expert rung (`throughput/mean_mfu=0.1034576068697843`), GPipe improves throughput substantially at this scale, though the 64-expert single-microbatch rung remains the highest MFU datapoint.
- Next action:
  - Post this milestone to the tracking issue and run focused repo checks.

### 2026-07-08 13:02 PDT - two-stage explicit 1F1B smoke succeeds
- Hypothesis: A hand-written two-stage explicit MPMD `std_1f1b` path, patterned after NVIDIA/JaxPP `examples/mpmd.py`, can run Grug microbatching without the automatic JaxPP sharding-inference path.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe` at baseline `5c36c4374a`.
- Commands:
  - `uv run python -m py_compile experiments/grug/moe/train.py experiments/grug/moe/launch_cw_jaxpp_may_d2560.py`
  - `./infra/pre-commit.py --fix experiments/grug/moe/train.py`
  - `TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --schedule std_1f1b --implementation explicit_mpmd --physical-stages 2 --logical-stages 2 --microbatches 4 --nodes 2 --gpus-per-replica 8 --expert-axis 8 --layers 4 --experts 8 --top-k 1 --batch 32 --seq-len 128 --vocab-size 8192 --moe-implementation ring --loss-implementation xla --steps 1 --tracker json_logger --xla-memory-fraction 0.80 --run-id jaxpp-explicit-std1f1b-l4-e8-20260708-195905`
- Config:
  - Explicit MPMD `std_1f1b`, 2 physical/logical stages, 4 microbatches, d2560, 4 layers, 8 experts, top-k 1, batch 32, seq 128, vocab 8192.
  - `XLA_PYTHON_CLIENT_MEM_FRACTION=0.80`, `TF_GPU_ALLOCATOR=cuda_malloc_async`.
- Result:
  - Parent launcher: `/dlwh/iris-run-job-20260708-195908`.
  - Child training job: `/dlwh/iris-run-job-20260708-195908/grug-train-jaxpp-explicit-std1f1b-l4-e8-20260708-195905`.
  - Child compiled `grug_1f1b_*` forward/backward/update tasks and emitted finish metrics.
  - Parameter count: `507,560,992`.
  - Step 0 metrics: loss `9.045732498168945`, tokens/s `72.74860433222855`, GFLOP/s `90.43715394085764`, MFU `0.0005712301284793938`.
- Interpretation:
  - The explicit 1F1B path avoids the automatic `std_1f1b` sharding inference blocker.
  - This is only a compile-dominated 4-layer smoke, but it proves the new two-stage schedule function is valid under `jaxpp.experimental.mpmd`.
  - The implementation is intentionally limited to `explicit_mpmd + std_1f1b + stages=2 + microbatches>1`.
- Next action:
  - Scale the same explicit `std_1f1b` path to 24 layers and measure MFU.

### 2026-07-08 13:08 PDT - two-stage 24-layer explicit 1F1B run succeeds on 2x8 H100
- Hypothesis: The two-stage explicit `std_1f1b` implementation should scale from the 4-layer smoke to the user-accepted 24-layer depth.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe` at baseline `5c36c4374a`.
- Command:
  - `TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --schedule std_1f1b --implementation explicit_mpmd --physical-stages 2 --logical-stages 2 --microbatches 4 --nodes 2 --gpus-per-replica 8 --expert-axis 8 --layers 24 --experts 8 --top-k 4 --batch 32 --seq-len 128 --vocab-size 8192 --moe-implementation ring --loss-implementation xla --steps 3 --tracker json_logger --xla-memory-fraction 0.80 --run-id jaxpp-explicit-std1f1b-l24-e8-20260708-200218`
- Config:
  - Explicit MPMD `std_1f1b`, 2 physical/logical stages, 4 microbatches, d2560, 24 layers, 8 experts, top-k 4, batch 32, seq 128, vocab 8192.
  - 2x 8xH100, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.80`, `TF_GPU_ALLOCATOR=cuda_malloc_async`.
- Result:
  - Parent launcher: `/dlwh/iris-run-job-20260708-200220`.
  - Child training job: `/dlwh/iris-run-job-20260708-200220/grug-train-jaxpp-explicit-std1f1b-l24-e8-20260708-200218`.
  - Child succeeded on both tasks.
  - Parameter count: `2,829,071,552`.
  - Step 2 metrics: loss `9.036020278930664`, tokens/s `14352.960046017917`, GFLOP/s `158980.05442905021`, MFU `1.0041691158984982`.
- Interpretation:
  - Explicit `std_1f1b` works at 24 layers via the hand-written MPMD path.
  - The MFU is not directly comparable to the 4x8 GPipe runs because this run used only 16 H100s and the analytic FLOP accounting likely overstates the useful work for this small 8-expert shape.
- Next action:
  - Repeat the 24-layer 8-expert two-stage `std_1f1b` run on 4x8 H100 by using two `replica_dcn` groups under the two pipeline stages.

### 2026-07-08 13:12 PDT - 4x8 two-stage 1F1B batch-32 run needs larger microbatches
- Hypothesis: A two-stage explicit `std_1f1b` run can use 4x8 H100 by placing two pipeline stages across four Iris tasks with extra intra-stage data sharding.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe` at baseline `5c36c4374a`.
- Command:
  - `TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --schedule std_1f1b --implementation explicit_mpmd --physical-stages 2 --logical-stages 2 --microbatches 4 --nodes 4 --gpus-per-replica 8 --expert-axis 8 --layers 24 --experts 8 --top-k 4 --batch 32 --seq-len 128 --vocab-size 8192 --moe-implementation ring --loss-implementation xla --steps 3 --tracker json_logger --xla-memory-fraction 0.80 --run-id jaxpp-explicit-std1f1b-l24-e8-4x8-20260708-200826`
- Config:
  - Explicit MPMD `std_1f1b`, 2 physical/logical stages, 4 microbatches, d2560, 24 layers, 8 experts, top-k 4, batch 32, seq 128, vocab 8192, 4x 8xH100.
- Result:
  - Parent launcher: `/dlwh/iris-run-job-20260708-200828`.
  - Child training job: `/dlwh/iris-run-job-20260708-200828/grug-train-jaxpp-explicit-std1f1b-l24-e8-4x8-20260708-200826`.
  - The run reached parameter accounting (`2,829,071,552`) but failed before compile:
    `ValueError: Sharding spec ('replica_dcn', 'data', 'expert') implies that array axis 1 is partitioned 16 times, but does not evenly divide the dimension size 8. Got shape: (4, 8, 128) ... spec=P(None, ('replica_dcn', 'data', 'expert'), None)`.
- Interpretation:
  - The 4x8 two-stage topology is valid, but batch 32 with 4 microbatches produces per-microbatch batch axis size 8.
  - The local stage mesh shards that microbatch axis over `data=2, expert=8`, requiring divisibility by 16.
- Next action:
  - Retry the same 4x8 two-stage `std_1f1b` run with batch 64 so each of 4 microbatches has batch axis size 16.

### 2026-07-08 13:17 PDT - 4x8 two-stage 1F1B batch-64 run succeeds
- Hypothesis: Increasing global batch to 64 should make each of the 4 microbatches have batch axis size 16, matching the 4x8 two-stage local stage mesh sharding over `data=2` and `expert=8`.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe` at baseline `5c36c4374a`.
- Command:
  - `TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --schedule std_1f1b --implementation explicit_mpmd --physical-stages 2 --logical-stages 2 --microbatches 4 --nodes 4 --gpus-per-replica 8 --expert-axis 8 --layers 24 --experts 8 --top-k 4 --batch 64 --seq-len 128 --vocab-size 8192 --moe-implementation ring --loss-implementation xla --steps 3 --tracker json_logger --xla-memory-fraction 0.80 --run-id jaxpp-explicit-std1f1b-l24-e8-b64-4x8-20260708-201204`
- Config:
  - Explicit MPMD `std_1f1b`, 2 physical/logical pipeline stages, 4 microbatches, d2560, 24 layers, 8 experts, top-k 4, batch 64, seq 128, vocab 8192, 4x 8xH100.
  - The 4 Iris tasks form two pipeline stages plus two `replica_dcn` groups; each microbatch has batch axis size 16.
  - `XLA_PYTHON_CLIENT_MEM_FRACTION=0.80`, `TF_GPU_ALLOCATOR=cuda_malloc_async`.
- Result:
  - Parent launcher: `/dlwh/iris-run-job-20260708-201206`.
  - Child training job: `/dlwh/iris-run-job-20260708-201206/grug-train-jaxpp-explicit-std1f1b-l24-e8-b64-4x8-20260708-201204`.
  - Child succeeded on all 4 tasks.
  - Parameter count: `2,829,071,552`.
  - Stage-0 task 1 step 2 metrics: loss `9.026140213012695`, tokens/s `16488.05340780921`, GFLOP/s `182629.3405540302`, MFU `0.5767728036698782`.
  - Stage-0 task 0 step 2 metrics: loss `9.026140213012695`, tokens/s `16439.069656854095`, GFLOP/s `182086.77376864132`, MFU `0.5750592905780739`.
  - Summary: best reporting stage-0 task `throughput/mean_mfu=0.5767728036698782`, `throughput/mfu_sample_count=2`.
- Interpretation:
  - The hand-written JaxPP explicit MPMD `std_1f1b` path now has a successful 4x 8xH100, 24-layer run.
  - Lowering the default XLA client memory fraction to `0.80` is sufficient for this 8-expert shape and keeps the launcher conservative for larger expert-count probes.
  - The 1F1B result is not directly comparable to 128-expert GPipe because it uses only 8 experts; it is currently the schedule-functionality datapoint, while GPipe remains the larger-parameter datapoint.
- Next action:
  - Run focused compile/pre-commit checks, then post the 4x8 `std_1f1b` milestone to the tracking issue.

### 2026-07-08 13:24 PDT - 64-expert 4x8 two-stage 1F1B comparison succeeds
- Hypothesis: The same two-stage explicit `std_1f1b` path should scale from the 8-expert schedule-functionality point to the 64-expert parameter scale used by the earlier GPipe and single-microbatch comparisons.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe` at baseline `5c36c4374a`.
- Command:
  - `TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --schedule std_1f1b --implementation explicit_mpmd --physical-stages 2 --logical-stages 2 --microbatches 4 --nodes 4 --gpus-per-replica 8 --expert-axis 8 --layers 24 --experts 64 --top-k 4 --batch 64 --seq-len 128 --vocab-size 8192 --moe-implementation ring --loss-implementation xla --steps 3 --tracker json_logger --xla-memory-fraction 0.80 --run-id jaxpp-explicit-std1f1b-l24-e64-b64-4x8-20260708-2020`
- Config:
  - Explicit MPMD `std_1f1b`, 2 physical/logical pipeline stages, 4 microbatches, d2560, 24 layers, 64 experts, top-k 4, batch 64, seq 128, vocab 8192, 4x 8xH100.
  - `XLA_PYTHON_CLIENT_MEM_FRACTION=0.80`, `TF_GPU_ALLOCATOR=cuda_malloc_async`.
- Result:
  - Parent launcher: `/dlwh/iris-run-job-20260708-201923`.
  - Child training job: `/dlwh/iris-run-job-20260708-201923/grug-train-jaxpp-explicit-std1f1b-l24-e64-b64-4x8-20260708-2020`.
  - Child succeeded on all 4 tasks.
  - Parameter count: `16,044,571,136`.
  - Stage-0 task 1 step 2 metrics: loss `9.023824691772461`, tokens/s `4785.771846091158`, GFLOP/s `53108.22933660344`, MFU `0.16772432205850002`.
  - Stage-0 task 0 step 2 metrics: loss `9.023824691772461`, tokens/s `4693.756221608463`, GFLOP/s `52087.121969864034`, MFU `0.1644995009154372`.
  - Summary: best reporting stage-0 task `throughput/mean_mfu=0.16772432205850002`, `throughput/mfu_sample_count=2`.
- Interpretation:
  - The two-stage explicit `std_1f1b` path scales to the 16B-parameter 64-expert comparison point at the conservative `0.80` XLA memory fraction.
  - Throughput is materially lower than the 64-expert explicit GPipe result (`throughput/mean_mfu=0.47585952009098575`), so this hand-written 1F1B path is currently useful as schedule evidence rather than a performance win.
  - The stage-0 backward compile still emits XLA involuntary rematerialization warnings when repartitioning hidden activations, which is a plausible contributor to the performance gap.
- Next action:
  - Post this comparison to the tracking issue and decide whether the next work should generalize explicit 1F1B to 4 stages or return to the automatic JaxPP sharding-inference blocker.

### 2026-07-08 13:55 PDT - automatic JaxPP blocker advanced past sharding inference
- Hypothesis: The automatic `mpmd_jit_with_loop` path fails because JaxPP treats ClosedJaxpr consts as replicated inputs even when the const payloads are explicitly sharded Grug parameter arrays.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe` at baseline `5c36c4374a`.
- Code changes:
  - Added opt-in `GRUG_JAXPP_PATCH_CONST_SHARDINGS=1`, which monkey-patches `jaxpp.core.extract_params` to derive leading const input shardings from `params["jaxpr"].consts`.
  - Added `GRUG_JAXPP_AUTO_EXPLICIT_IN_SHARDINGS=1`, which rebuilds the automatic JaxPP train step with sampled explicit `(state, pipeline_batch)` `in_shardings`.
  - Forwarded `GRUG_JAXPP_*`, `XLA_PYTHON_CLIENT_*`, and `TF_GPU_ALLOCATOR` from the Grug dispatcher into child train tasks; before this, parent launcher envs were not necessarily present in the training child.
  - Kept the CoreWeave wrapper default at `XLA_PYTHON_CLIENT_MEM_FRACTION=0.80`.
- Failed setup run:
  - Parent `/dlwh/iris-run-job-20260708-203854`, child `/dlwh/iris-run-job-20260708-203854/grug-train-jaxpp-auto-std-l4-e8-constshard-20260708-2044`.
  - Intended command used `GRUG_JAXPP_AUTO_EXPLICIT_IN_SHARDINGS=1 GRUG_JAXPP_PATCH_CONST_SHARDINGS=1`, but the wrapper/dispatcher did not yet forward those vars to the child.
  - Result: same `jaxpp/sharding_inference.py:613` assertion as before; this was an env-propagation negative result, not a valid const-sharding patch test.
- Failed setup run:
  - Parent `/dlwh/iris-run-job-20260708-204218`, child `/dlwh/iris-run-job-20260708-204218/grug-train-jaxpp-auto-std-l4-e8-constshard2-20260708-2046`.
  - After env forwarding, JaxPP reached tracing and failed earlier with `ValueError: pjit does not support kwargs when in_shardings is specified`.
  - Fix: make `compute_watch` positional and compile the automatic pipeline path without keyword args.
- Failed setup run:
  - Parent `/dlwh/iris-run-job-20260708-204450`, child `/dlwh/iris-run-job-20260708-204450/grug-train-jaxpp-auto-std-l4-e8-constshard3-20260708-2049`.
  - JaxPP reached tracing and no longer hit the explicit sharding inference assertion.
  - It failed at call time with `AssertionError` in `jaxpp/core.py:3553`, `assert self.in_info.in_tree == in_tree`.
  - Fix: the automatic path now compiles and calls the JaxPP function with only the dynamic `(state, pipeline_batch)` inputs, since watch is disabled there.
- Failed setup run:
  - Parent `/dlwh/iris-run-job-20260708-204723`, child `/dlwh/iris-run-job-20260708-204723/grug-train-jaxpp-auto-std-l4-e8-constshard4-20260708-2052`.
  - Command shape:
    `GRUG_LOG_JAXPRS=0 GRUG_LOG_XLA_HLO=0 GRUG_JAXPP_AUTO_EXPLICIT_IN_SHARDINGS=1 GRUG_JAXPP_PATCH_CONST_SHARDINGS=1 TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --schedule std_1f1b --implementation auto --physical-stages 4 --logical-stages 4 --microbatches 4 --nodes 4 --gpus-per-replica 8 --expert-axis 8 --layers 4 --experts 8 --top-k 1 --batch 32 --seq-len 128 --vocab-size 8192 --moe-implementation ring --loss-implementation xla --steps 1 --tracker json_logger --xla-memory-fraction 0.80 --conservative-loop-clustering false --run-id jaxpp-auto-std-l4-e8-constshard4-20260708-2052`.
  - JaxPP completed first/second loop tracing and did not fail in `infer_shardings_explicit`.
  - New root failure:
    `ValueError: device_put's second argument must be a Device or a Sharding which represents addressable devices, but got NamedSharding(mesh=Mesh('pipeline': 4, 'replica_dcn': 1, 'data': 1, 'expert': 8, 'model': 1, ...), spec=P('expert', 'data', 'model'), memory_kind=device)`.
  - JaxPP logged the offending input as `state.params.blocks[3].mlp.expert_mlp.w_gate`, local shape `(8, 2560, 1280)`, target sharding `P('expert', 'data', 'model')`.
- Interpretation:
  - The const-sharding patch appears to clear the original automatic-path `infer_shardings_explicit` assertion for this reduced Grug shape.
  - The next automatic blocker is input placement: JaxPP's `_maybe_shard_inputs` tries to `device_put` a stage-local parameter array using a global sharding whose mesh still contains the non-addressable `pipeline` axis.
  - This is now downstream of schedule tracing/placement, so automatic `std_1f1b` is closer than before but still not runnable.
- Next action:
  - Either teach the automatic path to pass stage-local inputs with addressable stage mesh shardings before `GlobalMpmdFunction.__call__`, or keep relying on the explicit `mpmd.py`-style path for schedule results and performance comparisons.

### 2026-07-08 14:02 PDT - four-stage explicit 1F1B smoke succeeds
- Hypothesis: The `mpmd.py`-style explicit `std_1f1b` implementation can be generalized from two physical stages to four physical stages by recursively forcing forward/backward dependencies and transferring activations/activation-gradients between adjacent stages.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe` at baseline `5c36c4374a`.
- Code changes:
  - Removed the `stages == 2` guard for `implementation="explicit_mpmd", schedule="std_1f1b"`.
  - Replaced the two-stage-specific 1F1B step with a generic scheduler over `num_stages` and `microbatches`.
  - Each stage now has a local warmup/steady/drain task list, while recursive dependency helpers ensure upstream forwards and downstream backwards are materialized before the local task runs.
  - Kept stage-local params and optimizer state split by contiguous transformer layers; the smoke therefore exercises the same weight partitioning structure intended for 24-layer runs.
  - Lowered the CoreWeave wrapper default `XLA_PYTHON_CLIENT_MEM_FRACTION` from `0.80` to `0.70` for future JaxPP probes.
- Commands:
  - `uv run python -m py_compile experiments/grug/moe/train.py experiments/grug/moe/launch_cw_jaxpp_may_d2560.py`
  - `./infra/pre-commit.py --fix experiments/grug/moe/train.py`
  - `TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --schedule std_1f1b --implementation explicit_mpmd --physical-stages 4 --logical-stages 4 --microbatches 4 --nodes 4 --gpus-per-replica 8 --expert-axis 8 --layers 4 --experts 8 --top-k 1 --batch 32 --seq-len 128 --vocab-size 8192 --moe-implementation ring --loss-implementation xla --steps 1 --tracker json_logger --xla-memory-fraction 0.80 --run-id jaxpp-explicit-std1f1b4-l4-e8-20260708-2058`
- Config:
  - Explicit MPMD `std_1f1b`, four physical/logical pipeline stages, 4 microbatches, d2560, 4 layers, 8 experts, top-k 1, batch 32, seq 128, vocab 8192.
  - 4x 8xH100, `TF_GPU_ALLOCATOR=cuda_malloc_async`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.80` for this smoke.
- Result:
  - Parent launcher: `/dlwh/iris-run-job-20260708-205734`.
  - Child training job: `/dlwh/iris-run-job-20260708-205734/grug-train-jaxpp-explicit-std1f1b4-l4-e8-20260708-2058`.
  - Child succeeded on all 4 tasks.
  - Parameter count: `507,560,992`.
  - Stage-0 step 0 metrics: loss `9.045722007751465`, tokens/s `58.831121678516`, GFLOP/s `73.13568770962965`, MFU `0.00023097425375704164`.
- Interpretation:
  - Four-stage explicit `std_1f1b` now executes end to end on CoreWeave H100s, including stage-local weight partitioning and activation-gradient transfers.
  - The one-step 4-layer result is compile dominated and should not be used as a performance datapoint.
  - This validates the path requested by the user's pointer to JaxPP's `examples/mpmd.py`; it avoids the current automatic JaxPP non-addressable input-sharding blocker.
- Next action:
  - Launch a 24-layer 4-stage explicit `std_1f1b` run at lower XLA prealloc (`0.70`) and record comparable MFU.

### 2026-07-08 14:08 PDT - 24-layer 4-stage 1F1B run succeeds at lower prealloc
- Hypothesis: The generic four-stage explicit `std_1f1b` path should scale from the 4-layer smoke to the user-accepted 24-layer d2560 shape at a lower XLA prealloc fraction.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe` at baseline `5c36c4374a`.
- Command:
  - `TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --schedule std_1f1b --implementation explicit_mpmd --physical-stages 4 --logical-stages 4 --microbatches 4 --nodes 4 --gpus-per-replica 8 --expert-axis 8 --layers 24 --experts 8 --top-k 4 --batch 32 --seq-len 128 --vocab-size 8192 --moe-implementation ring --loss-implementation xla --steps 3 --tracker json_logger --xla-memory-fraction 0.70 --run-id jaxpp-explicit-std1f1b4-l24-e8-b32-xla070-20260708-2104`
- Config:
  - Explicit MPMD `std_1f1b`, four physical/logical pipeline stages, 4 microbatches, d2560, 24 layers, 8 experts, top-k 4, batch 32, seq 128, vocab 8192.
  - 4x 8xH100, `TF_GPU_ALLOCATOR=cuda_malloc_async`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.70`.
- Result:
  - Parent launcher: `/dlwh/iris-run-job-20260708-210253`.
  - Child training job: `/dlwh/iris-run-job-20260708-210253/grug-train-jaxpp-explicit-std1f1b4-l24-e8-b32-xla070-20260708-2104`.
  - Child succeeded on all 4 tasks.
  - Parameter count: `2,829,071,552`.
  - Stage-0 step 2 metrics: loss `9.036020278930664`, tokens/s `3388.72738839914`, GFLOP/s `37535.11909220262`, MFU `0.11854193750695625`.
  - Summary: `throughput/mean_mfu=0.11854193750695625`, `throughput/mfu_sample_count=2`.
- Interpretation:
  - The generalized 4-stage explicit 1F1B scheduler runs at the requested 24-layer depth and lower prealloc.
  - The 8-expert four-stage result is slower than the two-stage 8-expert batch-64 run, so the more important comparison is the 64-expert rung below.
- Next action:
  - Run a 64-expert 4-stage comparison to match the earlier 64-expert GPipe parameter scale.

### 2026-07-08 14:16 PDT - 64-expert 4-stage 1F1B comparison succeeds
- Hypothesis: The 4-stage explicit `std_1f1b` schedule should scale to the 64-expert comparison shape and produce a cleaner schedule comparison against 4-stage GPipe.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe` at baseline `5c36c4374a`.
- Command:
  - `TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --schedule std_1f1b --implementation explicit_mpmd --physical-stages 4 --logical-stages 4 --microbatches 4 --nodes 4 --gpus-per-replica 8 --expert-axis 8 --layers 24 --experts 64 --top-k 4 --batch 32 --seq-len 128 --vocab-size 8192 --moe-implementation ring --loss-implementation xla --steps 3 --tracker json_logger --xla-memory-fraction 0.70 --run-id jaxpp-explicit-std1f1b4-l24-e64-b32-xla070-20260708-2110`
- Config:
  - Explicit MPMD `std_1f1b`, four physical/logical pipeline stages, 4 microbatches, d2560, 24 layers, 64 experts, top-k 4, batch 32, seq 128, vocab 8192.
  - 4x 8xH100, `TF_GPU_ALLOCATOR=cuda_malloc_async`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.70`.
- Result:
  - Parent launcher: `/dlwh/iris-run-job-20260708-210906`.
  - Child training job: `/dlwh/iris-run-job-20260708-210906/grug-train-jaxpp-explicit-std1f1b4-l24-e64-b32-xla070-20260708-2110`.
  - Child succeeded on all 4 tasks.
  - Parameter count: `16,044,571,136`.
  - Stage-0 step 2 metrics: loss `9.034103393554688`, tokens/s `13848.62694765867`, GFLOP/s `153679.71553721954`, MFU `0.4853452360321486`.
  - Summary: `throughput/mean_mfu=0.4853452360321486`, `throughput/mfu_sample_count=2`.
- Interpretation:
  - Four-stage explicit `std_1f1b` now matches the 64-expert comparison scale and slightly exceeds the earlier 64-expert GPipe MFU (`0.47585952009098575`) under the short three-step synthetic probe.
  - Lowering XLA prealloc to `0.70` did not prevent the 64-expert 24-layer rung from compiling and running.
  - The full 256-expert May shape remains unproven; prior 192/256 expert attempts failed allocation in other schedule/topology settings, so the next capacity test should use this 4-stage 1F1B path.
- Next action:
  - Post the 4-stage `std_1f1b` results to issue 7024 and decide whether to attempt 128/192 experts with `0.70` or spend the next pass on the automatic JaxPP input-placement blocker.

### 2026-07-08 14:28 PDT - 128-expert 4-stage 1F1B succeeds, 192/256 expert rungs do not fit
- Hypothesis: The working 4-stage explicit `std_1f1b` path can push beyond 64 experts toward the requested 256-expert May shape at `0.70` XLA prealloc.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe` at baseline `5c36c4374a`.
- Successful 128-expert command:
  - `TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --schedule std_1f1b --implementation explicit_mpmd --physical-stages 4 --logical-stages 4 --microbatches 4 --nodes 4 --gpus-per-replica 8 --expert-axis 8 --layers 24 --experts 128 --top-k 4 --batch 32 --seq-len 128 --vocab-size 8192 --moe-implementation ring --loss-implementation xla --steps 3 --tracker json_logger --xla-memory-fraction 0.70 --run-id jaxpp-explicit-std1f1b4-l24-e128-b32-xla070-20260708-2119`
- Successful 128-expert result:
  - Parent launcher: `/dlwh/iris-run-job-20260708-211950`.
  - Child training job: `/dlwh/iris-run-job-20260708-211950/grug-train-jaxpp-explicit-std1f1b4-l24-e128-b32-xla070-20260708-2119`.
  - Child succeeded on all 4 tasks.
  - Parameter count: `31,147,999,232`.
  - Stage-0 step 2 metrics: loss `9.033186912536621`, tokens/s `11006.860721977533`, GFLOP/s `122404.011880775`, MFU `0.38657153827935514`.
  - Summary: `throughput/mean_mfu=0.38657153827935514`, `throughput/mfu_sample_count=2`.
- Failed 256-expert command:
  - `TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --schedule std_1f1b --implementation explicit_mpmd --physical-stages 4 --logical-stages 4 --microbatches 4 --nodes 4 --gpus-per-replica 8 --expert-axis 8 --layers 24 --experts 256 --top-k 4 --batch 32 --seq-len 128 --vocab-size 8192 --moe-implementation ring --loss-implementation xla --steps 3 --tracker json_logger --xla-memory-fraction 0.70 --run-id jaxpp-explicit-std1f1b4-l24-e256-b32-xla070-20260708-2129`
- Failed 256-expert result:
  - Parent launcher: `/dlwh/iris-run-job-20260708-212827`.
  - Child training job: `/dlwh/iris-run-job-20260708-212827/grug-train-jaxpp-explicit-std1f1b4-l24-e256-b32-xla070-20260708-2129`.
  - Hparams logged on all 4 tasks.
  - Parameter count before failure: `61,354,855,424`.
  - XLA reported input/output argument bytes `68,289,171,792` (`63.60GiB`) exceeding the `0.70` base limit `59,513,712,445`.
  - All ranks failed during `jit__init_state` with `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 400.00MiB`.
- Failed 192-expert command:
  - `TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --schedule std_1f1b --implementation explicit_mpmd --physical-stages 4 --logical-stages 4 --microbatches 4 --nodes 4 --gpus-per-replica 8 --expert-axis 8 --layers 24 --experts 192 --top-k 4 --batch 32 --seq-len 128 --vocab-size 8192 --moe-implementation ring --loss-implementation xla --steps 3 --tracker json_logger --xla-memory-fraction 0.70 --run-id jaxpp-explicit-std1f1b4-l24-e192-b32-xla070-20260708-2135`
- Failed 192-expert result:
  - Parent launcher: `/dlwh/iris-run-job-20260708-213340`.
  - Child training job: `/dlwh/iris-run-job-20260708-213340/grug-train-jaxpp-explicit-std1f1b4-l24-e192-b32-xla070-20260708-2135`.
  - Hparams logged on all 4 tasks.
  - All ranks failed while initializing optimizer state with `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 300.00MiB`.
- Interpretation:
  - The 4-stage explicit 1F1B path reaches a larger completed parameter scale than GPipe did in this session at the same short synthetic settings: 128 experts and `31.15B` parameters.
  - The requested 256-expert count still does not fit with the current optimizer-state initialization and four-stage split, even at `0.70` prealloc.
  - 192 experts also does not fit, so the practical capacity boundary for this exact optimizer/state setup is currently between 128 and 192 experts.
- Next action:
  - Post the capacity ladder to issue 7024. Future work to reach 192/256 should reduce optimizer-state memory or change the partitioning/state-init strategy rather than only retrying with schedule changes.

### 2026-07-08 15:06 PDT - Stage-local explicit init fits the 24-layer 256-expert rung
- Hypothesis: The 192/256-expert failures were caused by constructing full optimizer state before splitting the model into pipeline stages; explicit MPMD should fit the requested 256-expert shape if parameters and optimizer state are initialized stage-locally.
- Commit Hash: uncommitted working tree on `research/jaxpp-grug-moe` at baseline `5c36c4374a`.
- Code changes:
  - Added `initial_pipeline_state` for `explicit_mpmd`; it initializes model parameters, immediately reshards split pipeline stages into JaxPP MPMD shardings, and initializes optimizer state from those stage-local parameters instead of from the full `Transformer`.
  - Added stage-local scalar construction for `state.step` and Optax schedule counters using the same empty-array pattern as NVIDIA/jaxpp's `examples/mpmd.py`; this avoids full-mesh scalar avals inside stage-local tasks.
  - Added shape-based parameter counting for explicit MPMD because JaxPP `MpmdArray` leaves do not expose `.size` on the multi-process run path.
  - Kept the CoreWeave wrapper default at `XLA_PYTHON_CLIENT_MEM_FRACTION=0.70`; the earlier `0.70` failure was a full-state init issue, not evidence that lower prealloc would help.
- Local checks:
  - `uv run python -m py_compile experiments/grug/moe/train.py`
  - CPU-only four-device init probe with `GRUG_JAXPP_RESHARD_THRESHOLD_BYTES=1048576` and `XLA_FLAGS=--xla_force_host_platform_device_count=4`; verified stage-local `state.step` and Optax scalar leaves report pipeline mesh size `1`.
- H100 smoke:
  - Parent launcher: `/dlwh/iris-run-job-20260708-215602`.
  - Child training job: `/dlwh/iris-run-job-20260708-215602/grug-train-jaxpp-explicit-initstage-smoke4-l4-e8-20260708-2157`.
  - Config: explicit MPMD `std_1f1b`, four stages, 4 microbatches, d2560, 4 layers, 8 experts, top-k 1, batch 32, seq 128, 4x 8xH100, `TF_GPU_ALLOCATOR=cuda_malloc_async`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.70`.
  - Result: all four tasks succeeded after the stage-local scalar fixes.
- 24-layer 256-expert run:
  - Command:
    - `TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --schedule std_1f1b --implementation explicit_mpmd --physical-stages 4 --logical-stages 4 --microbatches 4 --nodes 4 --gpus-per-replica 8 --expert-axis 8 --layers 24 --experts 256 --top-k 4 --batch 32 --seq-len 128 --vocab-size 8192 --moe-implementation ring --loss-implementation xla --steps 2 --tracker json_logger --xla-memory-fraction 0.70 --run-id jaxpp-explicit-initstage-l24-e256-b32-xla070-20260708-2200`
  - Parent launcher: `/dlwh/iris-run-job-20260708-220012`.
  - Child training job: `/dlwh/iris-run-job-20260708-220012/grug-train-jaxpp-explicit-initstage-l24-e256-b32-xla070-20260708-2200`.
  - Result: all four tasks succeeded.
  - Parameter count: `61,354,855,424`.
  - Stage-0 final metrics: loss `9.045845985412598`, tokens/s `6630.2587767668865`, GFLOP/s `74045.98318658397`, MFU `0.23384911314610907`.
  - Summary: `throughput/mean_mfu=0.23384911314610907`, `throughput/mfu_sample_count=1`.
- Interpretation:
  - The user-requested 24-layer, 256-expert, top-k 4 shape now fits and executes on 4x 8xH100 with explicit JaxPP MPMD `std_1f1b`.
  - Stage-local optimizer initialization was the missing weight/state partitioning step; reducing XLA prealloc alone could not have fixed the prior full-state optimizer init OOM.
  - The short two-step synthetic run is compile-dominated and has only one MFU sample; use it as a correctness/capacity proof, not a final performance number.
- Next action:
  - Post the successful 256-expert result to issue 7024 and run pre-commit over the touched files.

### 2026-07-08 15:35 PDT - implementation snapshot and seq4096 perf sweep
- Hypothesis: Once the explicit MPMD implementation is snapshotted, larger sequence length and batch should produce a more representative MFU than the compile-dominated seq128 capacity proof.
- Commit Hash: `abd979b82a` (`[grug] Add explicit JaxPP pipeline training`).
- Commands:
  - `./infra/pre-commit.py --changed-files --fix`
  - `git add experiments/grug/moe/model.py experiments/grug/moe/train.py experiments/grug/moe/launch.py experiments/grug/moe/launch_cw_jaxpp_may_d2560.py experiments/grug/moe/run_cw_jaxpp_may_d2560.sh .agents/logbooks/jaxpp-grug-moe.md`
  - `git commit -m "[grug] Add explicit JaxPP pipeline training"`
  - `TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --implementation explicit_mpmd --physical-stages 4 --logical-stages 4 --microbatches 4 --nodes 4 --gpus-per-replica 8 --expert-axis 8 --layers 24 --experts 256 --top-k 4 --vocab-size 8192 --moe-implementation ring --loss-implementation xla --steps 6 --tracker wandb --xla-memory-fraction 0.70 ...`
- Config:
  - Core shape: d2560, 24 layers, 256 experts, top-k 4, vocab 8192, ring MoE, XLA loss, 4x 8xH100 east02, 4 physical/logical pipeline stages, 4 microbatches, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.70`, `TF_GPU_ALLOCATOR=cuda_malloc_async`.
  - Schedules: explicit MPMD `std_1f1b` and explicit MPMD `gpipe`.
- Results:

| Schedule | Seq | Batch | Run | Status | Tokens/s | GFLOP/s | Mean MFU | Notes |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | --- |
| `std_1f1b` | 512 | 64 | `/dlwh/iris-run-job-20260708-221312/grug-train-jaxpp-explicit-perf-l24-e256-b64-s512-xla070-20260708-1512` | succeeded | `54,404.6` | `623,077` | `1.9681` | first larger-token point |
| `std_1f1b` | 4096 | 16 | `/dlwh/iris-run-job-20260708-222010/grug-train-jaxpp-explicit-perf-l24-e256-b16-s4096-xla070-20260708-1520` | failed | | | | per-microbatch batch `4` not divisible by expert axis `8` |
| `std_1f1b` | 4096 | 32 | `/dlwh/iris-run-job-20260708-222319/grug-train-jaxpp-explicit-perf-l24-e256-b32-s4096-xla070-20260708-1524` | succeeded | `117,063` | `1,651,819` | `5.2170` | large seq improves MFU materially |
| `std_1f1b` | 4096 | 64 | `/dlwh/iris-run-job-20260708-222719/grug-train-jaxpp-explicit-perf-l24-e256-b64-s4096-xla070-20260708-1528` | succeeded | `151,157` | `2,132,909` | `6.7251` | best point before schedule/batch sweep |
| `std_1f1b` | 4096 | 96 | `/dlwh/iris-run-job-20260708-224507/grug-train-jaxpp-explicit-perf-l24-e256-b96-s4096-xla070-20260708-1545` | succeeded | `167,681.85` | `2,366,084.23` | `7.4980` | best `std_1f1b` point |
| `gpipe` | 4096 | 96 | `/dlwh/iris-run-job-20260708-224938/grug-train-jaxpp-explicit-perf-l24-e256-b96-s4096-gpipe-xla070-20260708-1550` | succeeded | `162,738.58` | `2,296,331.90` | `7.2628` | comparable stable profile target |
| `gpipe` | 4096 | 128 | `/dlwh/iris-run-job-20260708-225414/grug-train-jaxpp-explicit-perf-l24-e256-b128-s4096-gpipe-xla070-20260708-1555` | succeeded | `170,567.57` | `2,406,803.30` | `7.5981` | best completed point |
| `gpipe` | 4096 | 160 | `/dlwh/iris-run-job-20260708-225923/grug-train-jaxpp-explicit-perf-l24-e256-b160-s4096-gpipe-xla070-20260708-1559` | failed | | | | compile OOM allocating `22.62GiB` in stage-3 loss backward |
| automatic `eager_1f1b` | 4096 | 64 | `/dlwh/iris-run-job-20260708-230301/grug-train-jaxpp-auto-probe-l24-e256-b64-s4096-eager1f1b-xla070-20260708-1603` | failed | | | | full-state init OOM before training |

- Interpretation:
  - The earlier low MFU was real for tiny compile-dominated smokes, but not representative once seq length and batch are increased.
  - At the 61B-param 256-expert shape, seq4096 batch96/128 reaches roughly `7.3-7.6` in the repo's percent-style MFU metric.
  - Explicit MPMD remains the viable path. Automatic JaxPP schedules need stage-local init/input placement fixes before full-shape schedule comparisons are meaningful.
- Issue updates:
  - Milestone/perf sweep: <https://github.com/marin-community/marin/issues/7024#issuecomment-4919854060>
  - Schedule/batch sweep: <https://github.com/marin-community/marin/issues/7024#issuecomment-4919983732>
- Next action:
  - Capture a TensorBoard/XPlane profile at the stable seq4096 batch96 GPipe point, then use the profile to decide whether communication, compute, or stalls dominate.

### 2026-07-08 17:35 PDT - profile and artifact checkpoint
- Hypothesis: The stable explicit MPMD GPipe seq4096 batch96 point should produce a usable XPlane/TensorBoard profile, and the profile should explain whether the remaining throughput gap is dominated by pipeline communication or device compute.
- Commit Hash: `a0f8130985` (`[grug] Upload explicit MPMD profile artifacts`).
- Commands:
  - Profile attempt, batch128 GPipe: explicit MPMD GPipe, 24 layers, 256 experts, seq4096, batch128, `trainer.profiler.enabled=true`, `trainer.profiler.start_step=3`, `trainer.profiler.num_steps=6`.
  - Successful profile, batch96 GPipe: explicit MPMD GPipe, 24 layers, 256 experts, seq4096, batch96, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.70`, profiler enabled with artifact upload.
  - `uv run python lib/marin/tools/profile_summary.py summarize --profile-dir scratch/profiles/jaxpp_profile_b96_gpipe_artifact --breakdown-mode exclusive_global --output scratch/jaxpp_profile_b96_gpipe_artifact_summary.json`
  - `uv run python lib/marin/tools/profile_summary.py report --summary scratch/jaxpp_profile_b96_gpipe_artifact_summary.json --output scratch/jaxpp_profile_b96_gpipe_artifact_report.md`
  - `uv run --isolated --with tensorboard --with tensorboard-plugin-profile --with 'setuptools<81' python -m tensorboard.main --logdir /Users/dlwh/.codex/worktrees/04c0/marin/scratch/profiles/jaxpp_profile_b96_gpipe_artifact --host 127.0.0.1 --port 6006`
- Config:
  - Explicit MPMD GPipe, 4 physical/logical stages, 4 microbatches, d2560, 24 layers, 256 experts, top-k 4, seq4096, global batch 96, ring MoE, XLA loss, 4x 8xH100 east02, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.70`, `TF_GPU_ALLOCATOR=cuda_malloc_async`.
- Result:
  - Initial batch128 profile run `/dlwh/iris-run-job-20260708-230824/grug-train-jaxpp-explicit-profile-l24-e256-b128-s4096-gpipe-xla070-20260708-2308` captured trace files but failed in `levanter_barrier_sync_2::0` because only task 0 runs callbacks under explicit MPMD. The job was stopped to avoid leaving a failed profile run alive.
  - Patched Levanter profiler callback to accept `sync_after_stop=False` and passed that for explicit MPMD profiles, then added explicit profile artifact upload from `trainer.log_dir / run_id / "profiler"`.
  - Batch128 retry `/dlwh/iris-run-job-20260708-231945/grug-train-jaxpp-explicit-profile-l24-e256-b128-s4096-gpipe-xla070-20260708-2320` segfaulted inside the JaxPP MPMD compiled step before the profile window.
  - Successful batch96 profile run: `/dlwh/iris-run-job-20260709-001640/grug-train-jaxpp-explicit-profile-l24-e256-b96-s4096-gpipe-xla070-artifact-20260709-0018`.
  - W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-explicit-profile-l24-e256-b96-s4096-gpipe-xla070-artifact-20260709-0018>
  - Artifact: `marin-community/marin_moe/jaxpp-explicit-profile-l24-e256-b96-s4096-gpipe-xla070-artifact-20260709-0018-profiler:v0`
  - Local profile directory: `scratch/profiles/jaxpp_profile_b96_gpipe_artifact`
  - Local summary/report:
    - `scratch/jaxpp_profile_b96_gpipe_artifact_summary.json`
    - `scratch/jaxpp_profile_b96_gpipe_artifact_report.md`
  - Profiled metrics: `161,656.17` tokens/s, `2,281,058.48` GFLOP/s, mean MFU `7.2239`, `61,354,855,424` params.
  - TensorBoard served locally at `http://127.0.0.1:6006/#profile`; indexed run `2026_07_09_00_24_12`, host `g739bec`.
- Interpretation:
  - The profile is communication dominated: `48.75%` communication, `45.10%` compute, `6.15%` stall.
  - Top exclusive kernel is `ncclDevKernel_SendRecv` (`104` calls, about `7.35s` total exclusive in the profile window), followed by all-gather (`~1.14s`), reduce-scatter (`~0.83s`), and all-reduce (`~0.22s`).
  - The trace was not suspected truncated, but step markers were not present; use this profile for kernel/collective attribution rather than step-time histogramming.
  - TensorBoard launch required transient `setuptools<81` because current setuptools no longer provides TensorBoard's deprecated `pkg_resources` import.
- Issue update:
  - <https://github.com/marin-community/marin/issues/7024#issuecomment-4920404840>
- Next action:
  - Treat pipeline send/recv as the first optimization target. Candidate next tests are reducing activation transfer volume, improving overlap, or increasing useful compute per pipeline transfer, then re-profiling the same batch96 seq4096 shape.

### 2026-07-08 23:45 PDT - std1f1b profile comparison and main merge
- Hypothesis: If the GPipe profile's `SendRecv` time is mostly pipeline-rendezvous wait, explicit `std_1f1b` at the same 24L/256E/seq4096/batch96 shape should show less exposed communication or better overlap.
- Commit Hash:
  - Profile code baseline: `fb7965b783` plus prior profile artifact commit `a0f8130985`.
  - Branch update: merged `origin/main` at `233b4bf658`.
- Commands:
  - Merge: `git fetch origin main && git merge --no-edit origin/main`
  - Launch:
    - `TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --schedule std_1f1b --implementation explicit_mpmd --physical-stages 4 --logical-stages 4 --microbatches 4 --nodes 4 --gpus-per-replica 8 --expert-axis 8 --layers 24 --experts 256 --top-k 4 --batch 96 --seq-len 4096 --vocab-size 8192 --moe-implementation ring --loss-implementation xla --steps 14 --tracker wandb --xla-memory-fraction 0.70 --profiler-steps 6 --run-id jaxpp-explicit-profile-l24-e256-b96-s4096-std1f1b-xla070-artifact-20260709-0506`
  - Download artifact:
    - W&B API download of `marin-community/marin_moe/jaxpp-explicit-profile-l24-e256-b96-s4096-std1f1b-xla070-artifact-20260709-0506-profiler:v0` into `scratch/profiles/jaxpp_profile_b96_std1f1b_artifact`.
  - Summarize/report:
    - `uv run python lib/marin/tools/profile_summary.py summarize --profile-dir scratch/profiles/jaxpp_profile_b96_std1f1b_artifact --breakdown-mode exclusive_global --output scratch/jaxpp_profile_b96_std1f1b_artifact_summary.json`
    - `uv run python lib/marin/tools/profile_summary.py report --summary scratch/jaxpp_profile_b96_std1f1b_artifact_summary.json --output scratch/jaxpp_profile_b96_std1f1b_artifact_report.md`
  - Compare:
    - `uv run python lib/marin/tools/profile_summary.py compare --before scratch/jaxpp_profile_b96_gpipe_artifact_summary.json --after scratch/jaxpp_profile_b96_std1f1b_artifact_summary.json > scratch/jaxpp_profile_b96_gpipe_vs_std1f1b_compare.md`
- Config:
  - Explicit MPMD `std_1f1b`, 4 physical/logical stages, 4 microbatches, d2560, 24 layers, 256 experts, top-k 4, seq4096, batch96, ring MoE, XLA loss, 4x 8xH100 east02, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.70`, `TF_GPU_ALLOCATOR=cuda_malloc_async`.
- Result:
  - Parent launcher: `/dlwh/iris-run-job-20260709-050611`.
  - Child training job: `/dlwh/iris-run-job-20260709-050611/grug-train-jaxpp-explicit-profile-l24-e256-b96-s4096-std1f1b-xla070-artifact-20260709-0506`.
  - The child waited in Kueue `SchedulingGated` for about 85 minutes, then ran and succeeded on all 4 tasks.
  - W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-explicit-profile-l24-e256-b96-s4096-std1f1b-xla070-artifact-20260709-0506>
  - Artifact: `marin-community/marin_moe/jaxpp-explicit-profile-l24-e256-b96-s4096-std1f1b-xla070-artifact-20260709-0506-profiler:v0`
  - W&B metrics: loss `8.137009620666504`, tokens/s `165,358.39`, GFLOP/s `2,333,298.90`, mean MFU `7.271346232021006`, `mfu_sample_count=13`, params `61,354,855,424`.
  - Local files:
    - `scratch/profiles/jaxpp_profile_b96_std1f1b_artifact`
    - `scratch/jaxpp_profile_b96_std1f1b_artifact_summary.json`
    - `scratch/jaxpp_profile_b96_std1f1b_artifact_report.md`
    - `scratch/jaxpp_profile_b96_gpipe_vs_std1f1b_compare.md`
  - Profile overview: complete events `3,171,418`, total events `3,262,438`, no suspected truncation, no step markers counted.
  - Time breakdown: communication `47.40%`, compute `46.91%`, stall `5.69%`.
  - Communication breakdown: send-recv count `624`, total exclusive `40,255,675.58`, average `64,512.30`; all-gather count `7,680`, total `6,790,019.96`; reduce-scatter count `4,416`, total `4,984,968.49`; all-reduce count `3,648`, total `1,406,057.17`.
  - Top pre-op gap is before `ncclDevKernel_SendRecv`: `593` gaps, total `260,493,025.13`, max `2,388,744.05`, average `439,279.98`.
- Interpretation:
  - `std_1f1b` is not a clear fix for the pipeline-rendezvous bottleneck. Its MFU is slightly above the GPipe profiled run (`7.2713` vs `7.2239`), and its comm share is slightly lower (`47.40%` vs `48.75%`), but `SendRecv` remains the dominant top op.
  - Raw aggregate durations and counts should not be compared directly: the `std_1f1b` trace window was about `15.5s`, while the prior GPipe trace was about `3.1s`. Normalize by shares and per-call averages instead.
  - Per-call `SendRecv` average is slightly lower in `std_1f1b` (`64.5ms`) than GPipe (`70.7ms`), but the large pre-op gaps before `SendRecv` are worse on average in this trace. The bottleneck still looks like stage dependency/rendezvous wait more than raw network bandwidth.
  - After merging `origin/main`, the CoreWeave Iris configs expect consolidated kubeconfig contexts under `~/.kube/coreweave-iris`. This machine still has split/older kubeconfigs; direct rno2a/usw09b API hostnames from `~/.kube/cw-rno2a.yaml` and `~/.kube/cw-usw09b.yaml` currently fail DNS resolution from the local environment.
- Next action:
  - Do not treat `std_1f1b` alone as the communication fix. The next optimization target remains reducing pipeline boundary transfer/wait: smaller activation payloads, different layer/stage partitioning, more useful compute per transfer, or explicit overlap improvements.

### 2026-07-10 02:50 PDT - RNO2A parity, occupancy sweep, and current profile
- Hypothesis: The `7.6` MFU ceiling is caused by pipeline occupancy and stage rendezvous rather than an east02 fabric fault; increasing microbatches and global batch on RNO2A should improve utilization, while delayed reductions or saved VJP residuals should isolate secondary costs.
- Commit Hash:
  - Merged `origin/main` through `b09baa125a` in merge commit `2514456540`.
  - Experiment code after that merge was still uncommitted while the sweep ran.
- Commands:
  - Baseline/sweeps used `TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --cluster cw-rno2a --kubeconfig "$HOME/.kube/coreweave-iris" --schedule std_1f1b --implementation explicit_mpmd --physical-stages 4 --logical-stages 4 --nodes 4 --gpus-per-replica 8 --expert-axis 8 --layers 24 --experts 256 --top-k 4 --seq-len 4096 --vocab-size 8192 --moe-implementation ring --loss-implementation xla --xla-memory-fraction 0.70 --remat save_moe ...`.
  - The occupancy sweep varied `(batch, microbatches)` over `(128, 4)`, `(128, 8)`, `(128, 16)`, `(256, 8)`, `(256, 16)`, and `(384, 12)`.
  - Profile: the same command at batch256/m8 with `--steps 14 --profiler-steps 4`.
  - Summary/report: `uv run python lib/marin/tools/profile_summary.py summarize --profile-dir scratch/profiles/jaxpp_rno2a_profile_b256_m8 --breakdown-mode exclusive_global --output scratch/jaxpp_rno2a_profile_b256_m8_summary.json`, then `profile_summary.py report`.
- Config:
  - d2560, 24 layers, 256 experts, top-k 4, seq4096, vocab8192, ring MoE, XLA loss, explicit JaxPP MPMD, 4 physical/logical stages, 4x 8xH100 RNO2A, prealloc `0.70`.
- Results:

| Batch | Microbatches | Status | Tokens/s | GFLOP/s | Mean MFU | Duration |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 128 | 4 | succeeded | `170,329.97` | `2,403,450.60` | `7.5946` | `3.0781` |
| 128 | 8 | succeeded | `191,925.72` | `2,708,178.64` | `8.0597` | `2.7317` |
| 256 | 8 | succeeded | `221,450.39` | `3,124,788.13` | `9.7635` | `4.7350` |
| 384 | 12 | succeeded | `240,651.18` | `3,395,722.01` | `10.6043` | `6.5359` |

  - Exact RNO2A baseline: `/dlwh/iris-run-job-20260710-074857/grug-train-jaxpp-rno2a-baseline-gpipe-l24-e256-b128-s4096-m4-20260710-0053`; W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-baseline-gpipe-l24-e256-b128-s4096-m4-20260710-0053>.
  - Batch128/m8: `/dlwh/iris-run-job-20260710-085631/grug-train-jaxpp-rno2a-recompute-std1f1b-l24-e256-b128-s4096-p4m8-20260710-0157`; W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-recompute-std1f1b-l24-e256-b128-s4096-p4m8-20260710-0157>.
  - Batch256/m8: `/dlwh/iris-run-job-20260710-090212/grug-train-jaxpp-rno2a-recompute-std1f1b-l24-e256-b256-s4096-p4m8-20260710-0202`; W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-recompute-std1f1b-l24-e256-b256-s4096-p4m8-20260710-0202>.
  - Batch384/m12: `/dlwh/iris-run-job-20260710-090806/grug-train-jaxpp-rno2a-recompute-std1f1b-l24-e256-b384-s4096-p4m12-20260710-0208`; W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-recompute-std1f1b-l24-e256-b384-s4096-p4m12-20260710-0208>.
  - Batch128/m16 and batch256/m16 compiled all reusable leaf tasks but made no further progress for 8-12 minutes. Both were stopped; this is a compile-complexity cliff, not an execution OOM.
  - A 2-physical-stage/m4 run remained in active XLA compilation for one hour and was stopped. Larger per-stage graphs are operationally intractable without virtual chunks.
  - NCCL debug from a 2-stage probe showed `NET/IB/.../GDRDMA`. The low apparent TensorBoard `SendRecv` bandwidth is therefore mostly rendezvous wait, not socket transport.
  - Delayed gradient reduction run `/dlwh/iris-run-job-20260710-081419/grug-train-jaxpp-rno2a-delayedgrad-gpipe-l24-e256-b128-s4096-p4m4-20260710-0114` was performance-neutral: mean MFU `7.5648`, tokens/s `169,742.99`, duration `3.0887`.
  - Opaque VJP residual reuse passed a tiny 4-stage correctness smoke, but every full-shape memory strategy failed. `save_moe` needed another `8.13 GiB` in forward even at `0.90` prealloc; `recompute_all` moved failures to `14.02-16.91 GiB` backward allocations. At m8 and disabled preallocation, stage 0 still failed requesting `4.17 GiB`.
  - Batch256/m8 profile W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-profile-std1f1b-l24-e256-b256-s4096-p4m8-20260710-0238>. Artifact: `marin-community/marin_moe/jaxpp-rno2a-profile-std1f1b-l24-e256-b256-s4096-p4m8-20260710-0238-profiler:v0`.
  - Profile metrics: mean MFU `9.7654`, tokens/s `226,714.36`, GFLOP/s `3,199,065.76`; exclusive breakdown `55.92%` compute, `38.96%` communication, `5.12%` stall. `SendRecv` averaged `59.99ms`; the prior GPipe profile averaged `70.7ms`.
  - Local profile files: `scratch/profiles/jaxpp_rno2a_profile_b256_m8`, `scratch/jaxpp_rno2a_profile_b256_m8_summary.json`, `scratch/jaxpp_rno2a_profile_b256_m8_report.md`, and `scratch/jaxpp_profile_b96_gpipe_vs_rno2a_b256_m8_compare.md`.
- Interpretation:
  - RNO2A and east02 produce the same baseline, and IB/GDRDMA is active. The cluster and transport are not the reason MFU is low.
  - More useful work per pipeline flush raises MFU materially: batch384/m12 is `39.6%` above the batch128/m4 baseline, but still far below the `20` target.
  - The fresh profile is compute-major rather than communication-major by share, but send/recv remains the largest individual operation and has large upstream gaps. Pipeline occupancy and stage dependency remain primary constraints.
  - Delayed gradient reduction and opaque VJP residual caching are closed branches for this shape. The former does not move throughput; the latter does not fit.
- Next action:
  - Finish queue-correct explicit `interleaved_gpipe` at 8 logical/4 physical stages and test the full batch256/m8 shape.
  - If virtual stages do not compile or do not beat batch384/m12, profile stage latency and rebalance the contiguous layer split, especially the final stage with output/loss work.

### 2026-07-10 03:08 PDT - queue-correct interleaved GPipe sweep
- Hypothesis: Mapping eight logical pipeline stages onto four physical JaxPP ranks and executing the pinned `InterleavedGPipe.tasks()` queues should reduce exposed bubbles without compiling two oversized 12-layer physical stages.
- Commit Hash: uncommitted experiment code on merge baseline `2514456540`; the implementation was committed after this entry.
- Commands:
  - Queue-correct smoke and full runs used `--schedule interleaved_gpipe --implementation explicit_mpmd --physical-stages 4 --logical-stages 8` on `cw-rno2a`, with the same d2560/24-layer/256-expert/top-k-4/seq4096 model as the standard schedule sweep.
  - Performance points varied `(batch, microbatches)` over `(128, 4)`, `(192, 6)`, `(224, 7)`, and `(256, 8)`.
- Results:

| Batch | Microbatches | Status | Tokens/s | GFLOP/s | Mean MFU | Duration |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 128 | 4 | succeeded | `207,521.71` | `2,928,246.73` | `9.2547` | `2.5264` |
| 192 | 6 | succeeded | `229,314.55` | `3,235,755.81` | `10.2190` | `3.4295` |
| 224 | 7 | stopped in compile | - | - | - | - |
| 256 | 8 | stopped in compile | - | - | - | - |

  - The reduced 8-logical/4-physical GPU smoke `/dlwh/iris-run-job-20260710-093718/grug-train-jaxpp-rno2a-interleaved-queued-smoke-l8-e8-b32-s128-p4l8m4-20260710-0235` succeeded with finite loss.
  - Batch128/m4: `/dlwh/iris-run-job-20260710-095204/grug-train-jaxpp-rno2a-interleaved-queued-l24-e256-b128-s4096-p4l8m4-20260710-0251`; W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-interleaved-queued-l24-e256-b128-s4096-p4l8m4-20260710-0251>.
  - Batch192/m6: `/dlwh/iris-run-job-20260710-095726/grug-train-jaxpp-rno2a-interleaved-queued-l24-e256-b192-s4096-p4l8m6-20260710-0257`; W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-interleaved-queued-l24-e256-b192-s4096-p4l8m6-20260710-0257>.
  - Batch224/m7 `/dlwh/iris-run-job-20260710-095736` emitted no progress for 7.5 minutes after reaching stage-4 forward compilation and was stopped with its child. Batch256/m8 showed the same compile-complexity cliff and was also stopped.
  - A standard 4-stage batch480/m12 probe `/dlwh/iris-run-job-20260710-095255` failed at stage-3 loss backward with a `22.63 GiB` allocation OOM; its parent was stopped. This bounds the equal 6/6/6/6 layer split above the successful batch384/m12 point.
- Interpretation:
  - Interleaving is a real schedule improvement at fixed batch128/m4: `9.2547` versus the standard schedule's `7.5946` mean MFU, a `21.9%` gain.
  - Increasing to batch192/m6 raises interleaved MFU to `10.2190`, but it does not beat standard batch384/m12 at `10.6043`. Above m6, graph compilation becomes the practical limit before execution.
  - The equal layer split leaves the output head and loss backward on the same final stage as six transformer layers. The batch480 OOM and current profile make a lighter final-stage split, such as `7/6/6/5`, the highest-value next performance test.
- Next action:
  - Commit and push the queue-correct interleaved implementation and launcher changes.
  - Add explicit configurable stage layer counts, validate stage-local weight placement for an unequal split, and rerun batch384/m12 before spending more H100 time on higher microbatch-count schedules.

### 2026-07-10 05:30 PDT - unequal splits and alternate expert collectives
- Hypothesis: A lighter final stage or less expert-parallel communication can move the batch384/m12 point above `10.6043` MFU.
- Commit Hash: stage-count and backend harness checkpoint `e0b4d7c2d6`.
- Commands:
  - Unequal splits used the standard RNO2A command with `--stage-layer-counts 7,5,7,5` and `--stage-layer-counts 7,6,6,5` at 24 layers, 256 experts, batch384/m12, seq4096, ring MoE, and preallocation `0.70` or `0.85`.
  - Expert-axis probe used the same full model with `--expert-axis 4 --batch 128 --microbatches 8 --xla-memory-fraction 0.85`.
  - Ragged comparison used `--moe-implementation ragged_all_to_all --batch 256 --microbatches 8 --steps 8`.
- Results:
  - A reduced 8-layer `3/2/2/1` split smoke succeeded with finite loss, validating stage-local weight and optimizer-state partitioning for unequal counts.
  - Full `7/5/7/5` OOMed stage 2 backward on a `19.65 GiB` allocation. `7/6/6/5` OOMed stage 0 forward on an `8.97 GiB` request at `0.70`; at `0.85`, all leaf tasks compiled but execution did not start in 10m12s. Both jobs were stopped.
  - Expert axis 4 compiled for 8m51s without reaching execution and was stopped. The larger per-rank expert state makes this topology operationally worse.
  - Ragged all-to-all succeeded but reached only `73,425.13` tokens/s, `1,036,069.40` GFLOP/s, and mean MFU `3.6743` at batch256/m8. W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ragged-std1f1b-l24-e256-b256-s4096-p4m8-20260710-110607>.
- Interpretation:
  - Any seven-layer physical stage exceeds the current memory/compile envelope; the equal `6/6/6/6` split is the practical full-model layout.
  - Ring remains the only performant expert transport among the working backends tested here.
- Next action:
  - Test DeepEP because each physical pipeline rank is exactly one 8-GPU NVLink node, but require finite multi-step correctness before collecting performance numbers.

### 2026-07-10 06:59 PDT - DeepEP bring-up and numerical blocker
- Hypothesis: Replacing ring EP with DeepEP intranode dispatch/combine will reduce the `38.96%` communication share in the batch256/m8 profile.
- Commit Hashes:
  - `f8d5e029d3`: CUDA runtime linkage.
  - `f7015afd06`: attention-only remat around effectful DeepEP FFI calls.
  - `e9a60bafdd`: 512-thread SM90 dispatch to fit H100 dynamic shared memory.
  - `951d566085`: non-pipelined backend isolation mode.
- Commands:
  - Reduced pipeline smokes used `TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --cluster cw-rno2a --implementation explicit_mpmd --schedule std_1f1b --physical-stages 4 --logical-stages 4 --microbatches 4 --nodes 4 --gpus-per-replica 8 --expert-axis 8 --layers 8 --experts 8 --top-k 4 --batch 32 --seq-len 128 --vocab-size 8192 --moe-implementation deepep --loss-implementation xla --steps 3 --xla-memory-fraction 0.70 --remat save_moe ...`.
  - The control added `--no-pipeline --nodes 1` with the same model, batch, DeepEP backend, and CUDA settings.
- Results:
  - DeepEP setup progressed through missing source, `nvcc`, CUDA headers, CCCL, and `libcudart` linkage. The pinned source is `7febc6e25660af0f54d95dd781ecdcd62265ecca`; worker setup installs CUDA NVCC/NVVM `13.2.78`, CCCL `13.3.3.4.1`, and runtime `13.2.75`.
  - Whole-block remat failed during tracing because JAX checkpoint partial evaluation does not support `FfiEffect`. Splitting each block into rematerialized attention and non-rematerialized MoE cleared tracing and backward compilation.
  - Upstream's 768-thread SM90 dispatch requested 192 KiB dynamic shared memory and failed `cudaFuncSetAttribute` on RNO2A H100. `DEEPEP_DISPATCH_NUM_THREADS=512` reduced the request to 128 KiB and reached execution.
  - Explicit JaxPP run `/dlwh/iris-run-job-20260710-134537/grug-train-jaxpp-rno2a-deepep-t512-smoke-l8-e8-b32-s128-p4m4-20260710-0646` executed forward/backward but reported NaN loss on step 0. W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-deepep-t512-smoke-l8-e8-b32-s128-p4m4-20260710-0646>.
  - Non-pipelined control `/dlwh/iris-run-job-20260710-135240/grug-train-deepep-rno2a-nopipe-t512-l8-e8-b32-s128-20260710-0654` had finite loss `9.0462` on its first step and all 161 logged gradient norms were finite (maximum `0.71385`), then its next forward produced NaN. W&B: <https://wandb.ai/marin-community/marin_moe/runs/deepep-rno2a-nopipe-t512-l8-e8-b32-s128-20260710-0654>.
  - Distributed jobs that remained alive after fatal/NaN exits were explicitly stopped; the non-pipelined parent and child completed terminally.
- Interpretation:
  - The CUDA build, FFI registration, SM90 dispatch launch, forward, and backward all execute. The remaining blocker is numerical/runtime-state correctness, not environment setup.
  - A finite non-pipelined first forward followed by NaN after one finite-gradient update points at DeepEP dispatch/combine VJP or cached runtime state. JaxPP makes the corruption visible on the first pipeline step.
  - No DeepEP MFU comparison is valid yet. Do not run the 24-layer performance shape until a reduced multi-step test stays finite.
- Next action:
  - Return performance work to ring EP and target expert GEMM/ragged-dot compute efficiency. Resume DeepEP only with a focused transport/VJP correctness test.

### 2026-07-10 10:24 PDT - 4-of-64 ring sweep and profile
- Hypothesis: Keeping top-k 4 while reducing the total expert count from 256 to 64 should reduce expert-state and routing overhead enough to admit a larger batch, while a matching DeepEP control will show whether the earlier NaN was caused by the pathological 4-of-8 routing ratio.
- Commit Hash: experiment harness snapshot `0d97e81470` (`[grug] Record JaxPP backend experiments`).
- Commands:
  - Ring sweep base command: `TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --cluster cw-rno2a --kubeconfig "$HOME/.kube/coreweave-iris" --schedule std_1f1b --implementation explicit_mpmd --physical-stages 4 --logical-stages 4 --nodes 4 --gpus-per-replica 8 --expert-axis 8 --layers 24 --experts 64 --top-k 4 --seq-len 4096 --vocab-size 8192 --moe-implementation ring --loss-implementation xla --steps 8 --tracker wandb --xla-memory-fraction 0.70 --remat save_moe ...`.
  - Sweep points: `--batch 384 --microbatches 12`, `--batch 480 --microbatches 12`, `--batch 448 --microbatches 14`, and `--batch 512 --microbatches 16`.
  - DeepEP control: `TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --cluster cw-rno2a --kubeconfig "$HOME/.kube/coreweave-iris" --no-pipeline --nodes 1 --gpus-per-replica 8 --expert-axis 8 --layers 8 --experts 64 --top-k 4 --batch 32 --seq-len 128 --vocab-size 8192 --moe-implementation deepep --loss-implementation xla --steps 5 --tracker wandb --xla-memory-fraction 0.70 --remat save_moe --run-id deepep-rno2a-nopipe-t512-l8-e64k4-b32-s128-20260710-0710`.
  - Profile: ring base command with `--batch 448 --microbatches 14 --steps 14 --profiler-steps 4 --run-id jaxpp-rno2a-profile-ring-l24-e64k4-b448-s4096-p4m14-20260710-1010`.
  - Download: W&B API `artifact(..., type="profiler").download(root="scratch/profiles/jaxpp_rno2a_profile_e64_b448_m14_artifact")` for `marin-community/marin_moe/jaxpp-rno2a-profile-ring-l24-e64k4-b448-s4096-p4m14-20260710-1010-profiler:v0`.
  - Summarize: `uv run python lib/marin/tools/profile_summary.py summarize --profile-dir scratch/profiles/jaxpp_rno2a_profile_e64_b448_m14_artifact --breakdown-mode exclusive_global --output scratch/jaxpp_rno2a_profile_e64_b448_m14_summary.json`.
  - Compare: `uv run python lib/marin/tools/profile_summary.py compare --before scratch/jaxpp_rno2a_profile_b256_m8_summary.json --after scratch/jaxpp_rno2a_profile_e64_b448_m14_summary.json > scratch/jaxpp_rno2a_profile_e256_b256_vs_e64_b448_compare.md`.
- Config:
  - Ring performance runs: d2560, 24 layers, 64 experts, top-k 4, seq4096, ring EP, explicit JaxPP MPMD `std_1f1b`, 4 physical/logical stages, 4x 8xH100 RNO2A, preallocation `0.70`, `save_moe` remat.
  - DeepEP control: d2560, 8 layers, 64 experts, top-k 4, seq128, batch32, one 8xH100 RNO2A node, no pipeline, 512-thread dispatch.
- Results:

| Backend | Experts | Batch | Microbatches | Status | Tokens/s | GFLOP/s | Mean MFU | Run |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| ring | 64 | 384 | 12 | succeeded | `256,230.87` | `3,597,424.30` | `11.3624` | <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ring-l24-e64k4-b384-s4096-p4m12-20260710-0936> |
| ring | 64 | 480 | 12 | succeeded | `253,313.23` | `3,556,461.28` | `11.2317` | <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ring-l24-e64k4-b480-s4096-p4m12-20260710-0944> |
| ring | 64 | 448 | 14 | succeeded | `262,901.35` | `3,691,076.41` | `11.6568` | <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ring-l24-e64k4-b448-s4096-p4m14-20260710-0954> |
| ring | 64 | 512 | 16 | stopped | | | | `/dlwh/iris-run-job-20260710-165700` |
| DeepEP | 64 | 32 | n/a | finite step 0, NaN step 1 | | | | <https://wandb.ai/marin-community/marin_moe/runs/deepep-rno2a-nopipe-t512-l8-e64k4-b32-s128-20260710-0710> |

  - Ring parent jobs were `/dlwh/iris-run-job-20260710-163632`, `/dlwh/iris-run-job-20260710-164334`, `/dlwh/iris-run-job-20260710-165032`, and `/dlwh/iris-run-job-20260710-165700`, respectively.
  - Batch512/m16 made no progress after stage-0 backward compilation and was stopped after about 11 minutes. The parent, child, and tasks are terminal; this reproduced the high-microbatch compile-complexity cliff rather than an execution OOM.
  - The DeepEP control parent `/dlwh/iris-run-job-20260710-163450` succeeded terminally. Step-0 loss was finite at `9.046618461608887`; step 1 was NaN. Increasing the expert count therefore did not remove the DeepEP correctness failure.
  - Profile parent `/dlwh/iris-run-job-20260710-170925` succeeded. W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-profile-ring-l24-e64k4-b448-s4096-p4m14-20260710-1010>. Artifact: `marin-community/marin_moe/jaxpp-rno2a-profile-ring-l24-e64k4-b448-s4096-p4m14-20260710-1010-profiler:v0`.
  - The profiled run reported `253,151.33` tokens/s, `3,554,188.27` GFLOP/s, mean MFU `11.2541`, and profile-window mean MFU `11.3473` over steps 8-11.
  - The uncapped XPlane trace contains `5,610,611` complete events and no quality warnings. Its exclusive breakdown is `60.34%` compute, `33.85%` communication, and `5.81%` stall, compared with `55.92%`, `38.96%`, and `5.12%` for the 256-expert batch256/m8 profile.
  - `SendRecv` remains the largest collective. Its average exclusive duration fell from `59.99ms` to `41.69ms`, while the average pre-op gap was nearly unchanged (`633.26ms` to `628.72ms`). Raw totals are not comparable because the e64/m14 trace covers a longer window and more microbatches.
- Interpretation:
  - The requested 4-of-64 ring configuration is valid and improves the best mean MFU from `10.6043` to `11.6568`, a `9.9%` relative gain. The useful capacity boundary is batch448/m14; batch480/m12 fits but is slower, and m16 is not operationally compilable.
  - The profile confirms that lower expert count and higher occupancy reduce the relative communication burden and improve per-call `SendRecv` duration. The unchanged roughly `0.63s` average pre-op gap means pipeline dependency/rendezvous wait is still exposed.
  - DeepEP has no valid performance comparison. Reproducing finite step 0 followed by NaN at 64 experts rules out the 4-of-8 routing ratio as the primary cause and strengthens the dispatch/combine VJP or runtime-state hypothesis.
- Next action:
  - Keep ring EP for performance work. Attribute the dominant expert compute fusions/GEMMs in the e64 profile and test a targeted kernel/backend improvement at batch448/m14.
  - Resume DeepEP only after a reduced multi-step transport/VJP correctness test remains finite.

### 2026-07-10 10:58 PDT - reference-attention attribution and CuTe FA4 gain
- Hypothesis: The largest anonymous compute fusions in the 64-expert profile come from the default reference attention, so replacing them with the existing CuTe FlashAttention 4 backend should improve the same full-model pipeline point without changing MoE routing or pipeline semantics.
- Commit Hash: uncommitted launcher/logbook changes on `research/jaxpp-grug-moe` at baseline `6803f640e2`.
- Commands:
  - XPlane/HLO attribution: `uv run --with xprof --with protobuf python lib/marin/tools/profile_summary.py summarize --xplane-file scratch/profiles/jaxpp_rno2a_profile_e64_b448_m14_artifact/plugins/profile/2026_07_10_17_11_55/g5303b8.xplane.pb --xplane-output-dir scratch/profiles/jaxpp_rno2a_profile_e64_b448_m14_xprof --xplane-count-trace-events --breakdown-mode exclusive_global --output scratch/jaxpp_rno2a_profile_e64_b448_m14_xprof_summary.json`, followed by decoding the first length-delimited `HloModuleProto` field with `jaxlib._jax.HloModule.from_serialized_hlo_module_proto(...).to_string()`.
  - Smoke: `TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --cluster cw-rno2a --kubeconfig "$HOME/.kube/coreweave-iris" --schedule std_1f1b --implementation explicit_mpmd --physical-stages 4 --logical-stages 4 --microbatches 4 --nodes 4 --gpus-per-replica 8 --expert-axis 8 --layers 8 --experts 64 --top-k 4 --batch 32 --seq-len 4096 --vocab-size 8192 --attention-implementation gpu_fa4_cute --moe-implementation ring --loss-implementation xla --steps 3 --tracker wandb --xla-memory-fraction 0.70 --remat save_moe --run-id jaxpp-rno2a-fa4cute-smoke-l8-e64k4-b32-s4096-p4m4-20260710-1035`.
  - Full comparison: the same base command with `--layers 24 --batch 448 --microbatches 14 --steps 8 --run-id jaxpp-rno2a-fa4cute-l24-e64k4-b448-s4096-p4m14-20260710-1042`.
  - Capacity probe: the same full command with `--batch 512 --microbatches 16 --run-id jaxpp-rno2a-fa4cute-l24-e64k4-b512-s4096-p4m16-20260710-1054`.
- Config:
  - Comparison shape: d2560, 24 layers, 64 experts, top-k 4, seq4096, ring EP, explicit JaxPP MPMD `std_1f1b`, 4 physical/logical stages, 4x 8xH100 RNO2A, preallocation `0.70`, `save_moe` remat.
  - Only the attention backend changes from the model-default reference implementation to `gpu_fa4_cute`; CuTe FA4 preserves the current sliding-window semantics on H100. `gpu_fa4_thd` is not an equivalent control because its H100 path does not support the configured sliding window.
- Results:
  - The decoded HLO maps the largest anonymous kernels to reference attention. `fusion_1579` converts a backward/JVP attention tensor with shape `[4,20,4096,4096]`; `fusion_1567` and `fusion_1569` implement masked backward `_where/select_n` operations at that shape; `fusion_522` is the corresponding forward bf16 conversion. Attention forward/JVP/VJP GEMMs are also among the largest kernels. The MoE Pallas kernel appears separately as `_lambda_` / `moe_up_down`.
  - Reduced smoke parent `/dlwh/iris-run-job-20260710-173513` succeeded with finite final loss `8.94207763671875`, confirming end-to-end FA4 pipeline correctness before the full comparison. W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-fa4cute-smoke-l8-e64k4-b32-s4096-p4m4-20260710-1035>.
  - Full parent `/dlwh/iris-run-job-20260710-174151` succeeded. W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-fa4cute-l24-e64k4-b448-s4096-p4m14-20260710-1042>.
  - CuTe FA4 reports `359,706.33` tokens/s, `5,050,196.73` GFLOP/s, mean MFU `15.9684`, mean duration `5.1014s`, and final loss `7.911863803863525` over seven throughput samples.
  - The matching reference-attention result was `262,901.35` tokens/s, `3,691,076.41` GFLOP/s, mean MFU `11.6568`, and mean duration `6.9798s`. CuTe FA4 therefore improves MFU by `37.0%`, tokens/s by `36.8%`, and duration by `26.9%`.
  - Batch512/m16 parent `/dlwh/iris-run-job-20260710-175329` succeeded, so FA4 clears the previous reference-attention m16 compile-complexity cliff. W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-fa4cute-l24-e64k4-b512-s4096-p4m16-20260710-1054>.
  - Batch512/m16 reported mean MFU `16.1040`, final-sample throughput `363,197.10` tokens/s and `5,099,206.46` GFLOP/s, p10/p50/p90 MFU `16.0715/16.1041/16.1333`, and final loss `7.833186149597168` over seven throughput samples.
  - Increasing from batch448/m14 to batch512/m16 adds only `0.1357` MFU points (`0.85%` relative), showing that standard-schedule capacity is nearly saturated.
  - Interleaved GPipe parent `/dlwh/iris-run-job-20260710-180545` compiled logical stages 0-3, then made no progress for 8m56s after `grug_interleaved_mb0_stage4_forward` began. Parent and child were stopped cleanly after 13m12s total; W&B produced no loss or throughput metric: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-fa4cute-interleaved-l24-e64k4-b256-s4096-p4l8m8-20260710-1106>.
- Interpretation:
  - The previous working theory that core MoE compute dominated was incomplete. Attention was the single largest compute opportunity because the reference implementation materialized sequence-squared intermediates at seq4096.
  - The optimized backend closes most of the gap to the 20-MFU target, but batch448/m14 remains `4.0316` MFU points short. Capacity/schedule overlap and the post-FA4 profile now matter more than further reference-attention analysis.
- Next action:
  - Profile standard batch512/m16 in parent `/dlwh/iris-run-job-20260710-180932` and test standard batch640/m16 in parent `/dlwh/iris-run-job-20260710-181958`. Use the profile to choose the next kernel/backend axis rather than retrying interleaved m8.

### 2026-07-10 11:36 PDT - FA4 profile and grouped-GEMM backend target
- Hypothesis: After removing reference attention, the next full-step gain should come from the grouped expert GEMMs rather than from more batch capacity or another interleaved retry.
- Commit Hash: uncommitted launcher/logbook changes on `research/jaxpp-grug-moe` at baseline `6803f640e2`.
- Commands:
  - Profile: the standard CuTe FA4 batch512/m16 command with `--steps 14 --profiler-steps 4 --run-id jaxpp-rno2a-profile-fa4cute-l24-e64k4-b512-s4096-p4m16-20260710-1110`.
  - Download: W&B API artifact `marin-community/marin_moe/jaxpp-rno2a-profile-fa4cute-l24-e64k4-b512-s4096-p4m16-20260710-1110-profiler:v0` to `scratch/profiles/jaxpp_rno2a_profile_fa4_b512_m16_artifact`.
  - Summarize: `uv run python lib/marin/tools/profile_summary.py summarize --xplane-file scratch/profiles/jaxpp_rno2a_profile_fa4_b512_m16_artifact/plugins/profile/2026_07_10_18_21_30/g5303b8.xplane.pb --breakdown-mode exclusive_global --output scratch/jaxpp_rno2a_profile_fa4_b512_m16_summary.json`.
  - Capacity: the standard CuTe FA4 command with `--batch 640 --microbatches 16 --run-id jaxpp-rno2a-fa4cute-l24-e64k4-b640-s4096-p4m16-20260710-1120`.
  - Grouped-GEMM A/B: the standard batch512/m16 command with `--ragged-dot-implementation xla --run-id jaxpp-rno2a-fa4cute-raggedxla-l24-e64k4-b512-s4096-p4m16-20260710-1135`.
- Results:
  - Profile parent `/dlwh/iris-run-job-20260710-180932` succeeded. The run reported mean MFU `15.9665`; the profile window perturbed the whole-run mean, while non-profiled late steps remained near `16.3` MFU. W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-profile-fa4cute-l24-e64k4-b512-s4096-p4m16-20260710-1110>.
  - The uncapped XPlane trace contains `6,228,312` complete events with no truncation warning. Its exclusive breakdown is `52.79%` compute, `39.61%` communication, and `7.59%` stall, versus `60.34%`, `33.85%`, and `5.81%` for reference attention at batch448/m14.
  - `SendRecv` averages `25.83ms`, down from `41.69ms`; its average pre-op gap falls from `628.72ms` to `426.17ms`. FA4 removes enough compute that communication and pipeline wait now occupy a larger relative share despite lower absolute per-call duration and wait.
  - The largest remaining compute op is `_lambda_`, the Pallas-Triton ragged-dot / `moe_up_down` kernel, with `9,216` calls and `1.852ms` average. FA4 forward is next at `1.530ms` average; the largest NVJet GEMM averages `0.297ms`.
  - Batch640/m16 parent `/dlwh/iris-run-job-20260710-181958` succeeded but regressed to mean MFU `15.6037`; p50 was `15.8937` and final loss was `7.7064642906188965`. W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-fa4cute-l24-e64k4-b640-s4096-p4m16-20260710-1120>.
- Interpretation:
  - CuTe FA4 improves both compute and pipeline behavior, but it exposes grouped expert GEMM as the largest tunable compute kernel and raises communication to nearly 40% of the optimized trace.
  - Batch512/m16 is the useful standard-schedule capacity point. Increasing microbatch size to 40 makes steps slower and does not improve steady-state MFU.
  - The next cheapest discriminating experiment is the existing XLA ragged-dot implementation at the identical winning shape. If it loses, tune the Pallas-Triton block sizes rather than changing transport or schedule again.
- Next action:
  - Complete XLA ragged-dot parent `/dlwh/iris-run-job-20260710-183433`, then keep the faster grouped-GEMM backend and update issue #7024 with the FA4/profile/capacity conclusion.

### 2026-07-10 11:53 PDT - ragged-dot backend and warp-count A/B
- Hypothesis: The optimized FA4 batch512/m16 point can gain additional MFU by replacing or retuning the `1.852ms` Pallas-Triton grouped expert GEMM identified in the profile.
- Commit Hash: uncommitted launcher/kernel/logbook changes on `research/jaxpp-grug-moe` at baseline `6803f640e2`.
- Commands:
  - XLA: the standard CuTe FA4 batch512/m16 command with `--ragged-dot-implementation xla --run-id jaxpp-rno2a-fa4cute-raggedxla-l24-e64k4-b512-s4096-p4m16-20260710-1135`.
  - Triton eight warps: the standard CuTe FA4 batch512/m16 command with `--ragged-dot-implementation triton --ragged-dot-num-warps 8 --run-id jaxpp-rno2a-fa4cute-raggedtriton-w8-l24-e64k4-b512-s4096-p4m16-20260710-1140`.
- Config:
  - Both runs keep d2560, 24 layers, 64 experts, top-k 4, seq4096, batch512/m16, ring EP, CuTe FA4, explicit JaxPP MPMD `std_1f1b`, four physical/logical stages, 4x 8xH100 RNO2A, `0.70` preallocation, and `save_moe` remat.
  - The four-warp control is <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-fa4cute-l24-e64k4-b512-s4096-p4m16-20260710-1054> at mean MFU `16.1040`.
- Results:

| Ragged-dot backend | Warps | Mean MFU | Final tokens/s | Final GFLOP/s | Final duration | Run |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Pallas-Triton | 4 | `16.1040` | `363,197.10` | `5,099,206.46` | `5.7741s` | <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-fa4cute-l24-e64k4-b512-s4096-p4m16-20260710-1054> |
| XLA | n/a | `8.1438` | `185,375.96` | `2,602,637.17` | `11.3130s` | <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-fa4cute-raggedxla-l24-e64k4-b512-s4096-p4m16-20260710-1135> |
| Pallas-Triton | 8 | `16.2005` | `365,684.79` | `5,134,133.00` | `5.7349s` | <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-fa4cute-raggedtriton-w8-l24-e64k4-b512-s4096-p4m16-20260710-1140> |

  - XLA parent `/dlwh/iris-run-job-20260710-183433` and eight-warp parent `/dlwh/iris-run-job-20260710-184022` both succeeded terminally with finite final losses `7.833324432373047` and `7.833574295043945`, respectively.
  - Eight warps improves mean MFU by `0.0964` points (`0.60%` relative), tokens/s by `0.68%`, and final duration by `0.68%` over four warps. XLA regresses mean MFU by `49.4%`.
- Interpretation:
  - Pallas-Triton is decisively the correct grouped expert GEMM backend for this shape. Raising the warp count is a real but small gain and establishes `16.2005` as the new best result.
  - The profile's nearly `40%` communication share and the small whole-step response to this kernel launch change make a 20-MFU result unlikely from warp-count tuning alone. Any further kernel experiment should change block geometry or use an autotuned grouped GEMM; schedule/batch scaling has already saturated.
- Next action:
  - Inspect the existing ragged-dot block geometry and benchmark harnesses, then run at most the smallest justified block-shape control. Otherwise seal the FA4/8-warp milestone and move the remaining target to pipeline communication overlap.

### 2026-07-10 12:14 PDT - ragged-dot K-tile control
- Hypothesis: Increasing Pallas-Triton `block_k` from 32 to 64 will halve the inner-loop iterations for every profiled grouped-GEMM shape and improve the eight-warp FA4 batch512/m16 point without changing numerics or grid bounds.
- Commit Hash: `259c2bbe49` (`[haliax] Add ragged dot K-tile control`).
- Command:
  - `TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --cluster cw-rno2a --kubeconfig "$HOME/.kube/coreweave-iris" --schedule std_1f1b --implementation explicit_mpmd --physical-stages 4 --logical-stages 4 --microbatches 16 --nodes 4 --gpus-per-replica 8 --expert-axis 8 --layers 24 --experts 64 --top-k 4 --batch 512 --seq-len 4096 --vocab-size 8192 --attention-implementation gpu_fa4_cute --ragged-dot-implementation triton --ragged-dot-block-k 64 --ragged-dot-num-warps 8 --moe-implementation ring --loss-implementation xla --steps 8 --tracker wandb --xla-memory-fraction 0.70 --remat save_moe --run-id jaxpp-rno2a-fa4cute-raggedtriton-w8-k64-l24-e64k4-b512-s4096-p4m16-20260710-1200`.
- Config:
  - Identical to the `16.2005`-MFU winner except `HALIAX_RAGGED_DOT_TRITON_BLOCK_K=64` instead of the default 32. All relevant K dimensions (`1280`, `2560`, and typical per-expert dRHS rows near `10240`) divide by 64; accumulation remains FP32.
- Results:
  - Parent `/dlwh/iris-run-job-20260710-190045` and its four-task child succeeded terminally. W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-fa4cute-raggedtriton-w8-k64-l24-e64k4-b512-s4096-p4m16-20260710-1200>.
  - Mean MFU was `16.0189`, with p10/p50/p90 `15.9878/16.0185/16.0390`. The final sample reported `361,587.48` tokens/s, `5,076,607.74` GFLOP/s, `5.7998s` duration, and finite loss `7.832277297973633`.
  - Relative to `block_k=32` at eight warps, mean MFU fell by `0.1816` points (`1.12%`), final tokens/s fell by `1.12%`, and final duration increased by `1.13%`.
- Interpretation:
  - The isolated K-tile change is a clear negative. Fewer K-loop iterations do not compensate for the larger tile's occupancy, register, or scheduling costs at these grouped-GEMM shapes.
  - Do not expand the block geometry sweep without a per-shape forward/dLHS/dRHS benchmark or restored autotuning machinery. The full-step profile and the diminishing response to kernel controls now make pipeline communication/dependency overlap the higher-value target.
- Next action:
  - Keep `block_k=32`, eight warps, CuTe FA4, batch512/m16 as the best measured configuration. Investigate the exposed SendRecv pre-op gap and schedule-level overlap instead of another blind tile point.

### 2026-07-10 13:47 PDT - no-EP FSDP boundary and proper SonicMoE bring-up
- Hypothesis: Removing expert parallelism eliminates the exposed ring SendRecv dependency, while FSDP storage keeps the 64-expert parameters and Muon state resident; upstream SonicMoE's QuACK grouped GEMMs can recover local expert compute efficiency after materializing complete weights at the kernel boundary.
- Commit Hashes:
  - `e917a4924b`: explicit no-EP/FSDP materialization boundary and upstream benchmark.
  - `6140388738`, `412b167dea`, `36f605b8c9`: JAX TVM-FFI QuACK gated-GEMM proof and concatenated/interleaved layout corrections.
  - `23fbf07065`: end-to-end Levanter Sonic path using QuACK fused gated and grouped down-projection forward kernels with reference ragged-dot VJPs.
  - `9101a3e741`, `373139550a`: GPU gradient-parity test and direct pinned GPU dependencies.
- Commands:
  - Upstream exact-shape probe: one RNO2A H100, install `Dao-AILab/sonic-moe@0349404acd7952592f73d180ff0c1510f6d112c2` without dependency replacement, then `uv run python experiments/grug/moe/benchmark_upstream_sonic_moe.py`.
  - JAX exact routed-assignment probe: one RNO2A H100, then `uv run python experiments/grug/moe/benchmark_quack_jax_gated.py --tokens 65536 --hidden-dim 2560 --intermediate-dim 1280 --experts 64 --warmup 2 --iterations 5`.
  - GPU behavior gates: `uv run pytest -n 0 lib/levanter/tests/grug/test_grugformer_moe.py -q -k 'moe_mlp_sonic_matches_jax_gather_reference_on_gpu'` and the corresponding `moe_mlp_sonic_gradients_match_jax_reference_on_gpu` test.
  - Distributed smoke: `TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --cluster cw-rno2a --schedule std_1f1b --implementation explicit_mpmd --physical-stages 4 --logical-stages 4 --microbatches 4 --nodes 4 --gpus-per-replica 8 --expert-axis 1 --layers 8 --experts 64 --top-k 4 --batch 32 --seq-len 4096 --vocab-size 8192 --attention-implementation gpu_fa4_cute --ragged-dot-implementation triton --ragged-dot-num-warps 8 --moe-implementation sonic --loss-implementation xla --steps 3 --tracker wandb --xla-memory-fraction 0.65 --remat save_moe --run-id jaxpp-rno2a-sonicquack-smoke-l8-e64k4-b32-s4096-p4m4-20260710-1350`.
- Results:
  - Upstream SonicMoE on H100 at 16,384 tokens, d2560/i1280, 64 experts, top-k 4 completed forward and backward in `8.4459ms` mean (`8.4296ms` median, `8.3800ms` minimum) with `8.8182GiB` peak allocated memory. Parent `/dlwh/sonicmoe-upstream-e64k4-t16384-smoke-20260710-1325` succeeded.
  - The JAX TVM-FFI fused gated forward at the corresponding 65,536 routed assignments completed in `1.2454ms` mean (`1.2451ms` median, `1.2404ms` minimum). Interleaved preactivation was bit-identical to the independent reference; worst-case BF16 postactivation error was `0.125` under unit-variance inputs.
  - The integrated QuACK gated and down-projection forward path passed the end-to-end Sonic-vs-ragged-dot GPU test. Its custom VJP bridge passed independent gradients for activations, routing weights, W13, and W2 at `rtol=0.1`, `atol=2e-4`.
  - Parameters remain data-axis FSDP-sharded at rest. The no-EP `shard_map` boundary explicitly replicates complete expert matrices for each local kernel call; a two-CPU-device numerical regression matches the unsharded reference at `rtol=atol=1e-5`.
  - NVIDIA JAX 26.05 added XLA-cuDNN ragged-dot fusion and NCCL 2.28 copy-engine support for all-gather/all-to-all. The 26.06 container is JAX `0.10.1`, CUDA `13.3`, XLA `9b63591`; its release notes do not identify a new ragged-all-to-all primitive. Marin already uses JAX `0.10.1`, CUDA 13, and NCCL `>=2.28.3`, but not NVIDIA's downstream XLA build, so ring remains the validated EP fallback.
- Interpretation:
  - The proper upstream kernel is intrinsically fast enough to justify no-EP. The remaining risk is distributed integration and the cost of FSDP all-gather plus the current weight-layout transposes, not QuACK compute.
  - The checked-in VJP bridge is a correctness milestone, not the final performance implementation: forward uses QuACK, while backward still lowers through the existing Pallas-Triton ragged-dot kernels. Porting QuACK `gemm_dgated` and grouped weight-gradient kernels remains necessary if the bridge does not clear 20 MFU.
- Next action:
  - Babysit parent `/dlwh/iris-run-job-20260710-204625`. On finite success, run the full 24-layer batch512/m16 target comparison with `expert_axis=1`; on failure, fix the first MPMD/FFI/sharding/memory error before any performance claim.

### 2026-07-10 13:54 PDT - distributed no-EP gate and full target launch
- Hypothesis: The no-EP QuACK/Sonic path remains finite under four-host explicit MPMD pipeline execution, so the 24-layer batch512/m16 target can measure whether eliminating ring expert dispatch offsets FSDP expert all-gather and the reference backward bridge.
- Commit Hash: `6a413115d1` (`[docs] Record SonicMoE no-EP bring-up`).
- Commands:
  - Reduced gate: the distributed-smoke command in the preceding entry.
  - Full comparison: `TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --cluster cw-rno2a --kubeconfig "$HOME/.kube/coreweave-iris" --schedule std_1f1b --implementation explicit_mpmd --physical-stages 4 --logical-stages 4 --microbatches 16 --nodes 4 --gpus-per-replica 8 --expert-axis 1 --layers 24 --experts 64 --top-k 4 --batch 512 --seq-len 4096 --vocab-size 8192 --attention-implementation gpu_fa4_cute --ragged-dot-implementation triton --ragged-dot-num-warps 8 --moe-implementation sonic --loss-implementation xla --steps 8 --tracker wandb --xla-memory-fraction 0.65 --remat save_moe --run-id jaxpp-rno2a-sonicquack-l24-e64k4-b512-s4096-p4m16-20260710-2054`.
- Results:
  - Reduced parent `/dlwh/iris-run-job-20260710-204625` and its four-task child succeeded terminally. W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-sonicquack-smoke-l8-e64k4-b32-s4096-p4m4-20260710-1350>.
  - The three-step gate finished with finite loss `8.940884590148926`. Its two throughput samples reported mean MFU `6.0980`, `405,321.78` tokens/s, and `0.32338s` duration; this deliberately underfilled eight-layer smoke is a correctness gate, not a full-model performance result.
  - Full target parent `/dlwh/iris-run-job-20260710-205417` is submitted on `cw-rno2a`.
- Interpretation:
  - The distributed FFI registration, no-EP FSDP materialization, four-stage pipeline forward/backward, optimizer update, and W&B finalization all work across 32 H100s. This clears the integration gate for an exact-shape performance comparison.
- Next action:
  - Babysit the full parent to terminal state and compare its exact MFU distribution against the `16.2004883` ring-EP baseline. If it remains below `20`, capture a short profile before changing the backward implementation or expert weight layout.

### 2026-07-10 14:10 PDT - no-EP full comparison and profile launch
- Hypothesis: Eliminating ring expert dispatch will improve the exact 24-layer target despite FSDP expert materialization and the Pallas-Triton VJP bridge; if it does not, an exact-shape profile will identify whether weight collectives/layouts or backward grouped GEMMs dominate the regression.
- Commit Hash: `6a413115d1` (`[docs] Record SonicMoE no-EP bring-up`).
- Commands:
  - Full comparison: the 24-layer batch512/m16 command in the preceding entry.
  - Profile: the identical command with `--steps 14 --profiler-steps 4 --run-id jaxpp-rno2a-profile-sonicquack-l24-e64k4-b512-s4096-p4m16-20260710-2109`.
- Results:
  - Full parent `/dlwh/iris-run-job-20260710-205417` and all four child tasks succeeded. W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-sonicquack-l24-e64k4-b512-s4096-p4m16-20260710-2054>.
  - Mean MFU was `14.8455`, with p10/p50/p90 `14.8356/14.8396/14.8842` over seven samples. The final sample reported `334,671.66` tokens/s, `6.26630s` duration, and finite loss `7.803744316101074`.
  - The no-EP result is `1.3549` MFU points (`8.36%`) below the `16.2004883` ring-EP winner at the identical model, batch, sequence length, attention backend, schedule, and device count.
  - Profile parent `/dlwh/iris-run-job-20260710-210953` is submitted on `cw-rno2a`.
- Interpretation:
  - Removing EP does not currently pay for the no-EP path's extra local work. The QuACK forward kernel is validated, so likely costs are FSDP materialization, expert weight layout transposes, and the Pallas-Triton reference VJP bridge.
  - Capacity and schedule sweeps are lower-value until the profile attributes this regression. The same batch and microbatch count already saturate the ring baseline.
- Next action:
  - Download and summarize the profile artifact. Use the exact hotspots to choose between kernel-native expert storage/layout, QuACK backward kernels, or a collective/sharding correction; retain ring EP as the performance baseline until no-EP exceeds it.

### 2026-07-10 14:40 PDT - no-EP profile attribution and FSDP materialization hoist
- Hypothesis: The no-EP regression comes primarily from materializing FSDP expert weights inside every microbatch task; materializing a replicated compute view once per stage and optimizer step should preserve sharded storage and gradient outputs while amortizing the expert all-gather across all microbatches.
- Commit Hash: uncommitted schedule change at baseline `67400e6ec9`.
- Commands:
  - Profile: the command in the preceding entry. Download W&B artifact `marin-community/marin_moe/jaxpp-rno2a-profile-sonicquack-l24-e64k4-b512-s4096-p4m16-20260710-2109-profiler:v0` and summarize its XPlane with `uv run --with xprof --with protobuf python lib/marin/tools/profile_summary.py summarize --xplane-file .../g5303b8.xplane.pb --xplane-output-dir scratch/profiles/jaxpp_rno2a_profile_sonicquack_b512_m16_xprof --xplane-count-trace-events --breakdown-mode exclusive_global --output scratch/jaxpp_rno2a_profile_sonicquack_b512_m16_summary.json`.
  - Reduced hoist gate: the preceding eight-layer distributed Sonic smoke with run id `jaxpp-rno2a-sonicquack-hoist-smoke-l8-e64k4-b32-s4096-p4m4-20260710-1433`.
  - Full hoist comparison: the same exact 24-layer batch512/m16 command as the unhoisted run, with run id `jaxpp-rno2a-sonicquack-hoist-l24-e64k4-b512-s4096-p4m16-20260710-1439`.
- Results:
  - Profile parent `/dlwh/iris-run-job-20260710-210953` succeeded. W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-profile-sonicquack-l24-e64k4-b512-s4096-p4m16-20260710-2109>. Mean MFU was `14.5961`; p10/p50/p90 were `14.0390/14.7763/14.8247`; final throughput was `334,283.9` tokens/s with finite loss `6.96628`.
  - The four-step profile attributes `49.95%` to communication, `42.95%` to compute, and `7.10%` to uncovered stall. NCCL AllGather is the largest operation, followed by pipeline SendRecv and reduce-scatter. FA4, a generated ragged-dot backward kernel, and the QuACK down/up kernels lead compute.
  - Against the matching ring profile, AllGather exclusive duration rises from `20.20M` to `48.13M` microseconds and total collective duration rises `38.4%`. The no-EP path therefore replaced expert dispatch communication with repeated FSDP parameter gathers rather than removing communication.
  - A direct N-major QuACK view intended to avoid `swapaxes` failed GPU forward parity with maximum absolute error `9024`; the known-good K-major path was restored. Avoiding those transposes requires kernel-native stored weights or a separately validated QuACK layout configuration.
  - Hoist smoke parent `/dlwh/iris-run-job-20260710-213235` and all four child tasks succeeded with finite loss `8.9409246`. W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-sonicquack-hoist-smoke-l8-e64k4-b32-s4096-p4m4-20260710-1433>.
  - Full hoist parent `/dlwh/iris-run-job-20260710-213940` is submitted on `cw-rno2a`.
- Interpretation:
  - QuACK forward is not the primary regression. The schedule currently passes FSDP storage parameters independently to every microbatch executable, so no-EP forces each expert layer to all-gather complete weights repeatedly.
  - The schedule change preserves original FSDP parameters and optimizer state, creates a replicated Sonic-only compute view once per stage/step, reuses it across forward/backward microbatches, and constrains each microbatch gradient back to the original sharded layout before accumulation and update.
  - The reduced m4 smoke is not a performance discriminator because one-time materialization is poorly amortized; only the exact m16 comparison can validate the expected gain.
- Next action:
  - Babysit the full hoist parent and compare against `14.8455495` unhoisted no-EP and `16.2004883` ring EP. If the gather hoist is visible but remains below `20`, profile or port QuACK backward next rather than revisiting the invalid N-major shortcut.

### 2026-07-10 14:53 PDT - FSDP hoist negative and QuACK input-gradient launch
- Hypothesis: If serialized stage-level expert materialization does not improve wall time, restore the original FSDP schedule and replace only the two expert input-gradient products with QuACK grouped GEMMs, while retaining Pallas grouped weight gradients for a controlled backward A/B.
- Commit Hashes:
  - `f9fefc2cbe`: stage-level Sonic FSDP materialization hoist.
  - `6ec633a692`: QuACK grouped input-gradient VJP bridge.
- Commands:
  - Hoist comparison: the full command in the preceding entry.
  - GPU behavior gate: one RNO2A H100 running the Sonic forward and full-gradient parity tests under job `/dlwh/sonic-quack-inputgrad-parity-20260710-1443`.
  - Input-gradient comparison: the exact 24-layer batch512/m16 command with the hoist removed and run id `jaxpp-rno2a-sonicquack-inputgrad-l24-e64k4-b512-s4096-p4m16-20260710-1452`.
- Results:
  - Hoist parent `/dlwh/iris-run-job-20260710-213940` and all four child tasks succeeded. W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-sonicquack-hoist-l24-e64k4-b512-s4096-p4m16-20260710-1439>.
  - Mean MFU was `14.6531`, with p10/p50/p90 `14.6254/14.6614/14.6674`, `330,658.9` tokens/s, `6.34234s` final duration, and finite loss `7.80430` over seven samples.
  - The hoist regressed the unhoisted no-EP result by `0.1925` MFU points (`1.30%`) and remains `1.5474` points below ring EP. It is removed from the active implementation; commit `f9fefc2cbe` remains the reproducible negative snapshot.
  - The QuACK input-gradient bridge passed integrated forward and gradients for activations, combine weights, W13, and W2 on H100. It saves the fused-gated preactivation, computes the local SwiGLU pullback, uses QuACK for both grouped input-gradient GEMMs, and leaves only grouped weight gradients on Pallas.
  - Input-gradient parent `/dlwh/iris-run-job-20260710-215310` is submitted on `cw-rno2a`.
- Interpretation:
  - Stage-level materialization moved the all-gather onto the critical path and removed overlap; aggregate collective duration alone did not imply an equivalent wall-time saving. Keep FSDP's current task-local scheduling until a mechanism can prefetch rather than serialize weights.
  - QuACK input gradients are a narrower compute-only experiment and preserve the existing communication schedule, making their full-run delta directly attributable.
- Next action:
  - Babysit the input-gradient parent. If it improves but remains below `20`, port QuACK's variable-K grouped weight-gradient GEMM; if it regresses, restore the prior VJP and retain ring as the measured winner.

### 2026-07-10 16:01 PDT - proper QuACK recompute backward and full launch
- Hypothesis: The split input-gradient VJP retained a full gated preactivation across the pipeline and expanded XLA's differentiated graph. Matching QuACK's memory-efficient whole-MLP recompute boundary should restore practical compilation while moving all expert GEMMs, dSwiGLU, and grouped weight gradients onto upstream QuACK CuTe kernels.
- Commit Hash: `7952c5e5fd` (`[grug] Match Sonic recompute backward`).
- Commands:
  - Failed input-gradient comparison: the exact 24-layer batch512/m16 command from the preceding entry, parent `/dlwh/iris-run-job-20260710-215310`.
  - GPU behavior gate: one RNO2A H100 running the Sonic forward and full-gradient parity tests under `/dlwh/sonic-quack-recompute-parity-20260710-1604`.
  - Full recompute comparison: `TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --cluster cw-rno2a --kubeconfig "$HOME/.kube/coreweave-iris" --schedule std_1f1b --implementation explicit_mpmd --physical-stages 4 --logical-stages 4 --microbatches 16 --nodes 4 --gpus-per-replica 8 --expert-axis 1 --layers 24 --experts 64 --top-k 4 --batch 512 --seq-len 4096 --vocab-size 8192 --attention-implementation gpu_fa4_cute --ragged-dot-implementation triton --ragged-dot-num-warps 8 --moe-implementation sonic --loss-implementation xla --steps 8 --tracker wandb --xla-memory-fraction 0.65 --remat save_moe --run-id jaxpp-rno2a-sonicquack-recompute-l24-e64k4-b512-s4096-p4m16-20260710-1605`.
- Results:
  - The input-gradient-only parent was intentionally stopped after `57m 7s`; all four tasks terminated cleanly with no runtime failure, but XLA had spent roughly 52 minutes actively compiling after its last compiler log and produced no executable or training metric. Sampled compiler activity was initially about 43 CPU cores and later about nine cores, while GPUs remained idle at roughly 56.9 GiB resident.
  - The replacement uses one whole-MLP custom VJP and saves only dispatched input, W13, W2, and expert group sizes. Backward recomputes interleaved W13 preactivation, fuses `dout @ W2` with dSwiGLU and postactivation regeneration through `GemmDGatedSm90`, then uses QuACK variable-M/variable-K grouped GEMMs for `dx`, `dW13`, and `dW2`.
  - QuACK's `concat_layout=("B",)` and `concat_layout=("out",)` preserve Marin's concatenated W13 storage while the fused gated kernels consume and produce interleaved gate/up values.
  - The H100 forward and full-gradient gates passed: `2 passed, 19 deselected` in `25.38s`; Iris task duration was `52.16s` with exit `0`.
  - Exact full parent `/dlwh/iris-run-job-20260710-230029` is submitted on `cw-rno2a` from pushed commit `7952c5e5fd`.
- Interpretation:
  - The split VJP is a compile-time dead end and should not be benchmarked further. It contradicted upstream QuACK's recompute design by retaining the largest expert activation across the pipeline interval.
  - The whole-MLP boundary keeps XLA's backward graph to opaque custom calls and trades one extra grouped up-projection GEMM for lower saved-activation memory and a fused dSwiGLU path.
- Next action:
  - Babysit the exact full parent and compare compile time and MFU against `14.8455495` old no-EP Sonic and `16.2004883` ring EP. If it remains below `20`, profile this exact recompute implementation before changing FSDP scheduling again.

### 2026-07-10 17:00 PDT - whole-MLP recompute exact-shape compile negative
- Hypothesis: The H100-validated whole-MLP custom VJP will keep XLA's exact 24-layer JAXPP compile within the old no-EP Sonic run's practical window and expose the QuACK backward runtime improvement.
- Commit Hash: `7952c5e5fd` (`[grug] Match Sonic recompute backward`).
- Command: the full recompute comparison in the preceding entry.
- Results:
  - Parent `/dlwh/iris-run-job-20260710-230029` was intentionally stopped after `55m 42s`; its four-task child was stopped after `55m 29s`. All tasks terminated with exit `0` and zero failures, and no live failed job remains.
  - The run reached stage 3 loss/backward compilation by `23:05:32 UTC` but never reached `grug_1f1b_keep_step`, an executable, loss, or MFU. During the quiet interval all four Python processes continued consuming roughly 20-45 CPU cores; two stages also held all eight GPUs at 100%, while two stages had idle GPUs. This rules out a dead process but not pathological compiler/autotuner work.
  - The old no-EP Sonic baseline reached `grug_1f1b_keep_step` and began reporting W&B throughput within the same run; the new run's failure to reach keep-step after 55 minutes is a material compile regression, not normal exact-shape startup cost.
  - H100 behavior remains valid at small scale: the whole-MLP recompute path passed forward and full-gradient parity before this distributed launch.
- Interpretation:
  - Moving the differentiation boundary and eliminating saved preactivation fixed the upstream kernel/layout contract but did not make the exact JAXPP executable compile operationally viable. The additional recompute, DGated, and variable-K custom calls appear to trigger pathological stage-backward compilation or autotuning at full shape.
  - No MFU comparison is available, so the measured winner remains ring EP at `16.2004883`; old forward-QuACK/reference-backward no-EP remains `14.8455495`.
- Next action:
  - Do not relaunch the exact whole-MLP path unchanged. Capture a bounded reduced-stage XLA dump or compile profile to isolate the pass/custom call responsible, or retain ring EP while pursuing schedule overlap; require a practical compile gate before another 32-H100 full run.

### 2026-07-10 18:05 PDT - post-codegen JaxPP/QuACK execution isolation
- Hypothesis: The apparent exact-shape compile regression is either an XLA pass/codegen bottleneck, rank-local lazy compilation skew, or a QuACK custom-call execution problem specific to JaxPP MPMD.
- Commit Hashes:
  - `2c91e4a357`: bounded direct/two-rank JaxPP custom-VJP reproducer with synchronized rank shutdown.
  - `67400e6ec9`: legacy forward-QuACK/Pallas-VJP A/B source revision.
- Commands:
  - Exact current XLA dump: the 24-layer batch512/m16 recompute command with `TF_CPP_VMODULE=xla_compilation_cache=2,gpu_compiler=2` and per-rank `XLA_FLAGS=--xla_dump_to=... --xla_dump_hlo_pass_re=.* --xla_dump_hlo_as_text --xla_gpu_dump_llvmir --xla_gpu_dump_ptx`, parent `/dlwh/iris-run-job-20260711-000043`.
  - Exact legacy A/B: the same command and dump flags from detached revision `67400e6ec9`, parent `/dlwh/iris-run-job-20260711-002146`.
  - Eager-precompile probe: reduced L8/e64/top-k4/b32/m4/seq4096 job `/dlwh/iris-run-job-20260711-003809`, compiling every rank-local JaxPP task before a cross-rank barrier and then executing the compiled objects.
  - Direct QuACK controls: `experiments/grug/moe/repro_jaxpp_custom_vjp_compile.py --mode quack --runtime direct --layers 1 --experts 64 --tokens-per-expert {256,1024} --hidden 2560 --intermediate 1280` on one RNO2A H100.
  - Minimal JaxPP controls: the same reproducer on two local H100s with `--runtime jaxpp --fsdp 1`, comparing `--mode quack` with the opaque custom-VJP control.
- Results:
  - The current stage-3 graph completed numbered XLA passes, PTX, LLVM, and thunk metadata in about `92s` (948 files, `1.7449GB`), then stayed inside `pxla.__call__` for more than `11m` with no new dump/compiler output and all stage-3 GPUs idle. Artifacts and timeline are under `scratch/xla_sonic_compile_probe/`; the compact archive SHA256 is `8e825f272a668b643a25b7f5b875fabf01457a0e356354aca2b02e12c8b50dd8`.
  - The exact legacy graph completed stage-3 pass/codegen in `101.02s`, returned to JaxPP within the next second, reached `keep_step`, and completed step 0 with finite loss `9.0436745`. Parent and all four ranks succeeded. Its artifacts are under `scratch/xla_sonic_compile_probe/legacy_stage3/`.
  - Current and legacy optimized graphs are similar in size: `15,099/15,297` instructions, `304/298` custom calls, `1,157/1,169` fusions, and `1,782/1,795` thunks. The current graph has `126` custom-call thunks versus `90` in legacy.
  - Eager precompile completed all 7-8 local tasks on every rank in `62.87-64.96s`, and every rank crossed the barrier. Execution then hung for about `8m43s`; stages 0/1 were GPU-busy while stages 2/3 were idle. Stage 0 was blocked in JaxPP/NCCL transfer startup and stage 3 in `pxla.__call__`. The temporary eager-precompile patch was removed.
  - Standalone QuACK whole-MLP forward/backward returned at both production scales: `15.3191s` compile+execute for `16,384` assignments and `15.5425s` for `65,536` assignments. Both one-H100 Iris jobs succeeded.
  - The minimal opaque two-rank JaxPP custom-VJP control returned in `3.90s`. The equivalent two-rank QuACK case remained in rank 1's `pxla.__call__` until the distributed barrier failed. No related job remains live.
- Interpretation:
  - The earlier label "compile regression" was too broad. XLA pass/codegen completes faster than the successful legacy control; lazy task compilation skew is also falsified by eager precompile.
  - QuACK is healthy under ordinary JAX at both the reduced smoke and exact target assignment counts, while its opaque custom calls fail to return only inside JaxPP local task execution. This is now a minimal JaxPP/custom-call executable integration failure, not evidence of a slow QuACK kernel or an oversized HLO graph.
  - Keep ring EP at `16.2004883` MFU as the performance winner. Do not spend another 32-H100 allocation on whole-MLP QuACK until the minimal two-rank reproducer returns.
- Next action:
  - Track the bounded upstream-ready reproducer and control matrix in [#7110](https://github.com/marin-community/marin/issues/7110), linked to #7024. Nothing was filed upstream; the issue contains a draft for human filing. Use #7110 to seek a JaxPP fix or a narrowly scoped workaround, then resume the no-EP Sonic performance comparison.

### 2026-07-10 22:05 PDT - minimal multi-device TVM-FFI root cause and workaround
- Hypothesis: The non-return is caused by a shared mutable launch path below JaxPP, so serializing host-side invocation packing for each compiled TVM-FFI handler will preserve asynchronous GPU execution while preventing concurrent device threads from corrupting handler state.
- Commit Hashes:
  - `adc27843a9`: standalone minimal QuACK/JaxPP diagnostic.
  - `d0aa12393a`: pinned `jax-tvm-ffi` per-handler mutex patch and Iris setup hook.
- Commands:
  - Boundary/control matrix: `experiments/grug/moe/repro_jaxpp_quack_minimal.py --runtime {direct,jaxpp} --operation {quack,plain,opaque} --transform forward --transfer {none,scalar} --experts {2,3} --tokens-per-expert 1 --input-dim 8 --output-dim 8 --fsdp {2,3}` with bounded watchdogs on RNO2A H100s.
  - CUDA graph control: the failing direct fsdp3 case with `JAXPP_QUACK_ALLOW_CUDA_GRAPH=false`.
  - Production direct gate: whole-MLP Sonic forward/backward at d2560/i1280/e64 and 65,536 assignments across eight H100s with the per-handler patch.
  - Pipeline gate: explicit `std_1f1b`, four physical/logical stages, L8/e64/top-k4/b32/m4/seq4096, CuTe FA4, whole-MLP Sonic, and XLA preallocation `0.65` under parent `/dlwh/iris-run-job-20260711-044839`.
- Results:
  - The minimum failing QuACK shape is three experts, one token per expert, 8x8 matrices, and fsdp3. Direct JAX and JaxPP both fail; fsdp2 passes. Forward versus gradient, rank placement, transfer/no-transfer, and replicated versus sharded inputs do not change the boundary. Plain JAX and Pallas opaque controls pass.
  - Disabling QuACK CUDA graphs does not fix fsdp3. A process-global mutex around TVM-FFI stream setup/call/restore fixes it; narrowing the lock to one mutex per registered `JAXTVMFFIHandler` also fixes it.
  - The narrow patch completed the full FSDP-8, 65,536-assignment whole-MLP forward/backward control in `15.3541s` compile+execute.
  - The four-stage smoke and all four child tasks succeeded, completed 3/3 iterations, and finished W&B with finite final loss `8.9430628`. It reported `1.995244%` MFU, `132,619` tokens/s, and `0.98833s` duration; this reduced L8/b32 run is a correctness gate, not a performance comparison. W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-sonicquack-ffimutex-smoke-l8-e64k4-b32-s4096-p4m4-20260710-2156>.
- Interpretation:
  - The earlier JaxPP-specific diagnosis is falsified. The failure is concurrent multi-device invocation of a shared compiled QuACK TVM function through `jax-tvm-ffi`; JaxPP merely exposes the same direct-JAX fsdp3 defect at pipeline scale.
  - The workaround serializes only host-side argument/stream setup and launch per compiled handler. `safe_call` returns after enqueueing, so kernels on distinct device streams remain asynchronous; exact performance still needs measurement.
- Next action:
  - Run the exact 24-layer d2560/e64/top-k4/b512/m16/seq4096 whole-MLP Sonic comparison at `d0aa12393a`. Compare mean MFU against the `16.2004883` ring baseline and profile the exact path if it remains below `20`.

### 2026-07-10 22:15 PDT - exact Sonic packaging failure and fail-fast correction
- Hypothesis: The pinned setup hook will reconstruct the validated per-handler `jax-tvm-ffi` patch on every pipeline rank before the exact benchmark begins.
- Commit Hash: `6f44c3d3a1` (`[grug] Fail fast on TVM FFI patch setup`).
- Command: the exact L24/d2560/e64/top-k4/b512/m16/seq4096 command from the preceding entry, parent `/dlwh/iris-run-job-20260711-050056`.
- Results:
  - The zero-context patch artifact applied at syntactically valid but semantically wrong line offsets: `call_lock` landed in the handler constructor and `call_mutex_` outside the class. All four `jax-tvm-ffi` builds failed, but the setup shell continued to its final successful JaxPP install command.
  - The unpatched child was stopped before it could reproduce the known hang. Parent and all four child tasks are terminal; no live failed job remains. W&B has no metric samples, so this is not a performance result.
  - The patch now anchors each insertion to neighboring source text and was reapplied to the pinned upstream commit locally; inspection confirms the lock is inside `Call` immediately before stream setup and the mutex is a handler member. Setup begins with `set -euxo pipefail`, so future patch or build errors stop before training.
- Interpretation:
  - The failed launch says nothing about Sonic MFU. It identified a reproducibility defect in the patch packaging and a missing failure boundary in worker setup.
- Next action:
  - Relaunch the identical exact benchmark from `6f44c3d3a1`; require all ranks to log a successful patched wheel build before treating compilation or runtime behavior as valid.

### 2026-07-10 22:28 PDT - exact proper Sonic benchmark and NVIDIA transport check
- Hypothesis: Proper whole-MLP QuACK forward/backward without expert parallelism will recover enough expert compute time to exceed the `16.2004883` ring baseline and approach the `20` MFU target once the multi-device TVM-FFI race is serialized.
- Commit Hash: `89bae1453c` (`[docs] Record Sonic setup packaging failure`), including code fix `6f44c3d3a1`.
- Command: exact L24/d2560/e64/top-k4/b512/m16/seq4096 command from the preceding entries, run id `jaxpp-rno2a-sonicquack-ffimutex-r2-l24-e64k4-b512-s4096-p4m16-20260710-2213`.
- Results:
  - Parent `/dlwh/iris-run-job-20260711-051316` and all four child tasks succeeded. Every rank applied, built, and installed the patched `jax-tvm-ffi`; training completed 8/8 steps with finite final loss `7.873054`.
  - Mean MFU was `13.9598`, with p10/p50/p90 `12.3634/14.1616/14.2122`. Final throughput was `319,999` tokens/s and duration `6.55362s`. W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-sonicquack-ffimutex-r2-l24-e64k4-b512-s4096-p4m16-20260710-2213>.
  - Proper Sonic is `2.2407` MFU points below ring EP (`13.83%` relative) and `6.0402` points below the `20` target. It is also `0.8858` points below the old forward-QuACK/Pallas-backward no-EP result (`14.8455`).
  - [NVIDIA JAX 26.06](https://docs.nvidia.com/deeplearning/frameworks/jax-release-notes/rel-26-06.html) is the newest stable container and uses JAX `0.10.1`, CUDA `13.3`, and NCCL `2.30.4`; its release notes do not publish an H100 MoE ragged-all-to-all gain. [OpenXLA PR #41580](https://github.com/openxla/xla/pull/41580) adds a zero-copy one-shot path that removes two D2D copies and a 512 MiB scratch request behind `--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=true --xla_gpu_experimental_ragged_all_to_all_zero_copy=true`; [commit `8f9a718`](https://github.com/openxla/xla/commit/8f9a7182831f8533e120aed3ec277bdff4592e99) later made zero-copy the default after internal testing. [OpenXLA issue #33386](https://github.com/openxla/xla/issues/33386) still reports poor same-host utilization for ragged all-to-all on B200, so this is a bounded A/B candidate rather than evidence that ragged transport now beats ring.
- Interpretation:
  - The TVM-FFI mutex is a correctness workaround, not a performance win. Proper QuACK backward does not offset repeated no-EP FSDP materialization and any host-launch serialization at the exact shape.
  - Ring EP remains the measured winner. A profile is required before attempting layer-scoped prefetch or narrowing the mutex; the zero-copy ragged path merits only one fallback experiment because prior unflagged ragged all-to-all reached `3.6743` MFU.
- Next action:
  - Capture and summarize an exact proper-Sonic XPlane profile. Rank all-gather, TVM-FFI/QuACK kernels, and launch gaps against the existing ring and old-Sonic profiles; implement only the highest-evidence change.

### 2026-07-10 22:50 PDT - exact proper Sonic profile attribution
- Hypothesis: Proper QuACK backward reduces expert compute enough that the exact profile will expose either a removable host-launch mutex bottleneck or repeated no-EP FSDP weight materialization as the dominant remaining regression.
- Commit Hash: `ae5321e46e` (`[docs] Record exact Sonic MFU result`).
- Command: exact profile command matching the preceding benchmark with `--steps 14 --profiler-steps 4`, parent `/dlwh/iris-run-job-20260711-052815`.
- Results:
  - Parent and all four child tasks succeeded; training completed 14/14 finite steps and W&B finished. Mean MFU was `13.6730` including one profile-perturbed `11.2591` sample; p10/p50/p90 were `11.9932/13.9812/14.0848`, latest throughput was `317,095.99` tokens/s, and final loss was `7.0697079`. W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-profile-sonicquack-ffimutex-l24-e64k4-b512-s4096-p4m16-20260710-2228>.
  - Artifact `marin-community/marin_moe/jaxpp-rno2a-profile-sonicquack-ffimutex-l24-e64k4-b512-s4096-p4m16-20260710-2228-profiler:v0` has digest `89837b115d989db53c6f8e6b7a3e4066`. The uncapped trace contains `6,421,456` complete events with no quality warnings.
  - Exclusive breakdown moved from old Sonic's `42.95%` compute / `49.95%` communication / `7.10%` stall to `37.31%` / `50.47%` / `12.22%`. Ring FA4 remains `52.79%` / `39.61%` / `7.59%`.
  - AllGather totals `47.508s` over `48,192` calls (`0.986ms` average, `8.148ms` average pre-op gap), almost unchanged from old Sonic's `48.13s` and 2.35x the ring profile's call count. SendRecv totals `47.308s` over `2,272` calls (`20.822ms` average) with a `251.304ms` average pre-op gap. ReduceScatter totals `14.803s` over `7,168` calls.
  - The largest compute kernels are QuACK default GEMM (`22.180s`, `20,992` calls), CuTe FA4 forward (`18.636s`, `9,216`), QuACK gated GEMM (`12.030s`, `8,704`), NVJet NNT GEMM (`10.725s`, `38,528`), and QuACK dSwiGLU (`2.760s`, `3,072`). Proper QuACK removed the old Pallas backward kernels.
- Interpretation:
  - The dominant no-EP penalty is still repeated FSDP expert-weight materialization. The per-handler mutex may contribute to the additional `5.12` stall points and pipeline skew, but the trace does not establish causality and removing it reintroduces a proven correctness race.
  - The previous once-per-stage hoist reduced gather repetition but regressed because all stages materialized at startup. A narrower next test is a staggered prefetch chain: materialize stage 0, then overlap stage 1 materialization with stage 0 forward, and continue down the pipeline while reusing each replicated stage view across microbatches.
- Next action:
  - Implement the staggered materialization mode behind an explicit opt-in, validate a reduced four-stage smoke, then run the exact comparison only if the smoke is finite. Compare against `13.9598` proper Sonic and `16.2004883` ring EP.

### 2026-07-10 23:18 PDT - staggered Sonic materialization smoke gate
- Hypothesis: Materializing each stage's replicated Sonic expert weights once per step, then transferring a completion dependency downstream, will overlap stage N+1 gathering with stage N forward while eliminating per-microbatch weight gathers.
- Commit Hashes:
  - `8d7998dee2`: opt-in `staged_per_step` materialization mode and launcher control.
  - `1478cdd3e3`: sharding-safe completion dependency.
- Commands:
  - Initial L8/e64/top-k4/b32/m4/seq4096 four-stage smoke under parent `/dlwh/iris-run-job-20260711-060358`.
  - Corrected identical smoke under parent `/dlwh/iris-run-job-20260711-060935`.
- Results:
  - The initial smoke failed during JaxPP tracing because `weight.reshape(-1)[0]` selected a size-one result from an explicitly sharded dimension. No metric was produced; all tasks terminated and no failed live job remained.
  - The completion task now reduces one complete sharded vector from each `w_gate`, `w_up`, and `w_down` leaf. This is a real dependency on every replicated expert buffer and yields a legal replicated scalar without reducing an entire weight.
  - The corrected smoke compiled all four materialization tasks and stage 0-2 completion-token tasks, completed 3/3 finite steps, and finished W&B. Final loss was `8.9430542`, MFU `5.59637`, throughput `371,978` tokens/s, and duration `0.352365s`. W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-sonicquack-stagedprefetch-smoke-r2-l8-e64k4-b32-s4096-p4m4-20260710-2309>.
  - The matching per-task proper-Sonic smoke reported `1.995244` MFU and `132,619` tokens/s. The reduced staged result is a `2.80x` MFU/throughput improvement, though the short three-step smoke is a correctness and directional gate rather than the target benchmark.
- Interpretation:
  - Replicated compute parameters can remain live across all microbatch forward/backward tasks while gradients and optimizer state retain the original FSDP shardings. The explicit dependency chain executes as designed.
  - The reduced gain is large enough to justify the exact 24-layer comparison; no claim against ring or the 20-MFU target is made until that run completes.
- Next action:
  - Babysit exact parent `/dlwh/iris-run-job-20260711-061854` and compare its distribution against `13.9598` per-task Sonic, `16.2004883` ring EP, and the `20` target.

### 2026-07-10 23:35 PDT - exact staggered Sonic materialization negative
- Hypothesis: The `2.80x` reduced-smoke gain from staggered once-per-stage materialization will persist at L24/b512/m16 by eliminating repeated FSDP all-gathers and overlapping each downstream stage's one-time gather with upstream forward work.
- Commit Hash: `0d97125b50` (`[docs] Record staged Sonic smoke gate`), including implementation `8d7998dee2` and sharding fix `1478cdd3e3`.
- Command: exact L24/d2560/e64/top-k4/b512/m16/seq4096 staged command under parent `/dlwh/iris-run-job-20260711-061854`.
- Results:
  - Parent and all four child tasks succeeded; every materialization task, stage 0-2 completion token, and `keep_step` executed. Training completed 8/8 steps with finite final loss `7.8731637`.
  - Mean MFU was `13.8155`, with p10/p50/p90 `11.9343/14.0727/14.0813`, latest throughput `317,576.25` tokens/s, and duration `6.603617s`. W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-sonicquack-stagedprefetch-l24-e64k4-b512-s4096-p4m16-20260710-2318>.
  - Staging regressed per-task proper Sonic by `0.1443` MFU points and remains `2.3850` points below ring EP and `6.1845` points below the target.
- Interpretation:
  - The reduced smoke exaggerated startup/amortization effects. At the exact shape, fine-grained per-task FSDP all-gathers are sufficiently overlapped that replacing them with one dependency-chained gather per stage moves communication onto the critical path.
  - Keep `staged_per_step` as an explicit reproducible negative, but use `per_task` for Sonic. Ring EP remains the measured winner.
- Next action:
  - Run one e64/b512/m16 ragged-all-to-all comparison with XLA's barrier and zero-copy flags. Stop the ragged path if it remains materially below ring; do not migrate to a nightly container without a positive bounded signal.

### 2026-07-10 23:41 PDT - zero-copy ragged-all-to-all infrastructure blocker
- Hypothesis: The XLA zero-copy one-shot ragged-all-to-all path can remove two D2D copies and a 512 MiB scratch allocation on the exact e64/b512/m16 target without changing the container.
- Commit Hash: `d27b886fea` (`[docs] Record staged Sonic exact result`).
- Command: exact L24/d2560/e64/top-k4/b512/m16/seq4096 ragged-all-to-all command under parent `/dlwh/iris-run-job-20260711-063505`, with `XLA_FLAGS='--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=true --xla_gpu_experimental_ragged_all_to_all_zero_copy=true'`.
- Results:
  - Both flags were accepted and emitted on all four workers. During first-stage forward compilation, rank 0 repeatedly failed `VMM cuMemCreate ... FABRIC+POSIX_FD` with `CUDA_ERROR_NOT_PERMITTED`, then segfaulted with exit 139. Other ranks terminated after coordination loss.
  - Parent and all four child tasks are terminal; no live failed job remains. No compile completion, loss, or MFU was produced. W&B has no history: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ragged-zerocopy-l24-e64k4-b512-s4096-p4m16-20260710-2335>.
- Interpretation:
  - The implementation is present in the current JAX/XLA build, so a nightly NVIDIA container migration is not the missing step. The zero-copy path requires exportable CUDA VMM allocations that the current RNO2A pod permissions/runtime reject.
  - Do not retry zero-copy ragged all-to-all on RNO2A without an explicit runtime/permission change. Unflagged ragged all-to-all remains measured at only `3.6743` MFU on the earlier e256/b256/m8 shape; ring remains the only performant transport.
- Next action:
  - Increase ring pipeline occupancy at fixed microbatch size 32 using b1024/m32, then inspect the ring round scheduling for overlap opportunities.

### 2026-07-10 23:55 PDT - ring occupancy gain and transport readout
- Hypothesis: Doubling microbatch count while holding microbatch size at the efficient 32-token-batch shape will reduce the four-stage pipeline bubble without reproducing the b640/m16 larger-microbatch regression.
- Commit Hash: `fd3999a31c` (`[docs] Record zero-copy ragged blocker`).
- Command: exact L24/d2560/e64/top-k4/b1024/m32/seq4096 ring command under parent `/dlwh/iris-run-job-20260711-064117`.
- Results:
  - Parent and all four child tasks succeeded; training completed 8/8 finite steps with final loss `7.339579`.
  - Mean MFU was `16.6677`, with p10/p50/p90 `15.6381/17.2663/17.3222`; latest MFU was `17.3180`, throughput `390,572.87` tokens/s, and duration `10.738851s`. W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ring-l24-e64k4-b1024-s4096-p4m32-20260710-2341>.
  - Mean MFU improves b512/m16 by `0.4672` points; steady p50 improves by roughly `1.06` points. The run remains `3.3323` mean points below 20.
  - Code/HLO inspection shows the current `ring` backend is bulk AllGather/grouped-GEMM/ReduceScatter, not a streamed expert ring. Each GPU gathers local `bf16[16384,2560]` activations into `bf16[131072,2560]`, computes local-expert assignments, scatters into a 640 MiB global output, and reduce-scatters. AllGather overlaps unrelated compute for about 36% of its duration but overlaps Pallas expert GEMMs below 2%; there is no same-MoE communication/compute overlap.
- Interpretation:
  - Higher occupancy is a real but bounded gain. Removing the remaining m32 pipeline bubble cannot provide the additional roughly 16% relative improvement needed from the current mean.
  - A true `ppermute` ring can remove the 640 MiB global activation/output intermediates and expose per-round communication/GEMM overlap while preserving the working transport semantics. This is materially different from the sort/compact/inverse-sort ragged-all-to-all path that regressed.
- Next action:
  - Run b2048/m64 as the final fixed-microbatch occupancy point. In parallel, implement a separate streamed `ppermute` backend with output/gradient parity before a reduced H100 performance gate.

### 2026-07-11 00:12 PDT - m64 occupancy result and streamed ring prototype
- Hypothesis: Doubling fixed-size microbatch count again will recover more bubble overhead, while a true streamed ring can provide the separate communication gain needed to cross 20 MFU.
- Commit Hashes:
  - `91071b1e83`: occupancy and ring transport readout.
  - `251629c83c`: opt-in `ring_ppermute` backend.
- Commands:
  - Exact b2048/m64 ring parent `/dlwh/iris-run-job-20260711-065554`.
  - Four-device CPU output/overflow/gradient parity and abstract-lowering tests for `ring_ppermute`.
- Results:
  - The m64 parent and all four ranks succeeded; training completed 8/8 finite steps with final loss `6.6641207`. Mean MFU was `17.4430`, p10/p50/p90 `16.5874/17.7193/18.0223`, latest throughput `393,396.56` tokens/s, and duration `21.323542s`. W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ring-l24-e64k4-b2048-s4096-p4m64-20260710-2355>.
  - m64 improves m32 by `0.7753` mean points and `0.4530` p50 points. It remains `2.5570` mean points below 20; further bubble reduction alone cannot close the gap.
  - `ring_ppermute` rotates one source shard at a time with `lax.ppermute`, computes local-expert contributions, permutes partial outputs directly back to their owner, and avoids global activation/output tensors. Four-device output, dropped-count, activation/combine-weight/W13/W2 gradient parity passed (`5 passed`), and StableHLO contains collective-permute operations without a global `[EP*T,H]` activation tensor.
- Interpretation:
  - Occupancy is a validated contributor but has diminishing returns. The remaining target requires changing the expert transport or another similarly large bottleneck.
  - The streamed backend is correctness-ready for a reduced H100 gate. Its risks are EP8 smaller GEMMs, one assignment selection per round, and native XLA ragged-dot performance.
- Next action:
  - Compare L8/b32/m4 `ring_ppermute` against the matching bulk-ring `9.4180` MFU smoke. Scale only if the reduced result is competitive.

### 2026-07-11 00:24 PDT - streamed ring H100 hard negative
- Hypothesis: Removing global activation/output tensors and exposing per-round collective-permute/GEMM overlap will compensate for splitting the EP8 expert work into source-shard rounds.
- Commit Hash: `251629c83c` (`[grug] Add streamed expert ring backend`).
- Command: L8/d2560/e64/top-k4/b32/m4/seq4096 four-stage `ring_ppermute` smoke under parent `/dlwh/iris-run-job-20260711-071233`.
- Results:
  - Compilation completed through `keep_step` and the executable ran. Step 0 had finite loss `9.0438070`; step 1 produced NaN and training stopped. The parent was stopped after W&B finalized; all child tasks are terminal and no live failed job remains.
  - The only timed sample included startup and is not a steady-state metric: `0.0085168` MFU, `566.09` tokens/s, and `231.5379s` versus matching bulk-ring smoke `9.4180` MFU, `625,994.6` tokens/s, and `0.209382s`. W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ringppermute-smoke-l8-e64k4-b32-s4096-p4m4-20260711-0012>.
- Interpretation:
  - The one-source-per-round design is not viable with native XLA ragged GEMMs at EP8. It expands each layer into many small grouped GEMMs/collectives and fails multi-step numerical behavior despite CPU parity.
  - Do not scale or tune this backend. Preserve it only as a reproducible algorithmic negative; keep bulk ring as the working path.
- Next action:
  - Optimize the bulk ring combine locally: scatter only owner-local token rows into `zeros_like(x_local)` and `psum` that 80 MiB buffer, instead of scattering a 640 MiB global tensor and reduce-scattering it.

### 2026-07-11 01:07 PDT - m128 occupancy best and local-combine compile negative
- Hypothesis: m128 will recover more standard-schedule bubble overhead, while owner-local combine can reduce the bulk ring's peak output temporary without changing dispatch/GEMMs.
- Commit Hashes:
  - `a6d9214108`: streamed-ring negative snapshot.
  - `abe0428689`: opt-in `ring_local_combine` backend.
- Commands:
  - Exact b4096/m128 ring parent `/dlwh/iris-run-job-20260711-072442`.
  - L8/b32/m4 `ring_local_combine` smoke parent `/dlwh/iris-run-job-20260711-073658`.
- Results:
  - m128 and all four ranks succeeded with 8/8 finite steps and final loss `5.7604752`. Mean MFU was `18.1334`, p10/p50/p90 `17.7645/18.2729/18.3056`, latest throughput `400,643.27` tokens/s, and duration `41.875697s`. W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ring-l24-e64k4-b4096-s4096-p4m128-20260711-0024>.
  - m128 improves m64 by `0.6904` mean and `0.5536` p50 points, remaining `1.8666` mean points below 20.
  - A direct local-buffer `psum` is not semantically valid because local indices refer to different token owners. The parity-correct local-combine backend instead uses seven 80 MiB owner-directed collective permutes at EP8. CPU output/overflow/full-gradient parity passed (`6 passed`).
  - The reduced local-combine smoke reached stage-3 loss/backward compilation, then produced no new marker or metric for about 28 minutes. All ranks remained live with zero failures; the parent was intentionally stopped at 29m55s. All tasks are terminal and no failed live job remains.
- Interpretation:
  - Occupancy remains the only validated positive direction and now reaches 18.13 mean MFU, but its theoretical bubble asymptote still falls short of 20.
  - Owner-local combine preserves aggregate traffic and replaces one optimized reduce-scatter with seven explicit permutations; its differentiated graph is operationally intractable even at L8. Do not scale it.
- Next action:
  - Run b8192/m256 as the final occupancy point. For further code optimization, retain bulk collectives and target dispatch/scatter fusion within the existing AllGather/ReduceScatter dataflow rather than expanding collective count.

### 2026-07-11 01:48 PDT - occupancy saturation and e64 expert-axis4 compile negative
- Hypothesis: m256 may recover the last pipeline bubble, while reducing expert-axis size from 8 to 4 at e64 may halve bulk gather span without the e256 state/compile failure.
- Commit Hash: `d4bffe01b5` (`[docs] Record m128 and local combine results`).
- Commands:
  - Exact b8192/m256 EP8 ring parent `/dlwh/iris-run-job-20260711-080751`.
  - Exact b4096/m128 expert-axis4 ring parent `/dlwh/iris-run-job-20260711-081233`.
- Results:
  - m256 and all four ranks succeeded with 6/6 finite steps and final loss `5.5467644`. Mean MFU was `18.2583`, p10/p50/p90 `18.0479/18.3654/18.3830`, throughput `414,059.10` tokens/s, and duration `81.037785s`. W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ring-l24-e64k4-b8192-s4096-p4m256-20260711-0107>.
  - m256 improves m128 by only `0.1248` mean and `0.0925` p50 points, remaining `1.7417` mean points below 20. The occupancy curve is saturated.
  - Expert-axis4 reached stage 0-3 first backward compiles, then made no new progress for over 24 minutes after `mb1_stage0_accumulate_grads`. It was stopped at 34m37s without a metric; all tasks are terminal and no failed live job remains.
- Interpretation:
  - Further batch/microbatch scaling is not justified. Standard 1F1B bubble reduction asymptotes around the observed 18.3 MFU.
  - The e64 state fits initial compilation farther than the prior e256 probe, but doubled local expert state still makes accumulation/update compilation operationally intractable. Retain expert-axis8.
- Next action:
  - Run an exact m128 A/B with `--xla_gpu_enable_latency_hiding_scheduler=true`; the existing HLO already uses async collectives but overlaps expert GEMMs with AllGather below 2%.

### 2026-07-11 02:10 PDT - latency-hiding scheduler bounded negative
- Hypothesis: XLA's GPU latency-hiding scheduler will move the existing asynchronous expert AllGather and ReduceScatter collectives off the critical path enough to close a material fraction of the `1.7417` MFU-point gap to 20.
- Commit Hash: `64d21b7fc0` (`[docs] Record occupancy saturation`).
- Command: exact L24/d2560/e64/top-k4/b4096/m128/seq4096 EP8 bulk-ring command under parent `/dlwh/iris-run-job-20260711-084837`, with `XLA_FLAGS='--xla_gpu_enable_latency_hiding_scheduler=true'`.
- Results:
  - Parent and all four child tasks succeeded; the flag was accepted on every worker and W&B finished. Mean MFU was `18.2043`, with p10/p50/p90 `17.6869/18.4017/18.4242`, MFU standard deviation `0.3535`, final throughput `398,893.97` tokens/s, duration `42.059338s`, and finite final loss `5.7603002`. W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ring-lhs-l24-e64k4-b4096-s4096-p4m128-20260711-0148>.
  - Against the matching unflagged m128 run, mean MFU changed by `+0.0708` points (`+0.39%` relative) and p50 by `+0.1288` points, while final throughput changed by `-0.44%`. Five central samples clustered at `18.4013-18.4228` MFU, but the tail fell to `17.6869`.
- Interpretation:
  - The scheduler is functional, but its mean gain is smaller than run noise and only closes about `4%` of the remaining gap. It is not a credible path to 20 by itself.
  - Keep the flag as an optional small median improvement, but do not stack speculative scheduler flags without profile evidence. Retain the m256 unflagged result (`18.2583` mean) as the headline best.
- Next action:
  - Preserve the bulk AllGather/grouped-GEMM/ReduceScatter transport and target its large zero/scatter dispatch and combine fusions. Require full output/overflow/gradient parity and a reduced H100 gate before another exact run.

### 2026-07-11 02:14 PDT - bounded standalone TVM-FFI reproducer
- Hypothesis: Moving the direct-JAX probe behind a parent supervisor and making the known fsdp3 failure the zero-argument default will produce an upstream-ready artifact that terminates deterministically without requiring Marin, Levanter, or JaxPP.
- Commit Hash: `a7684fa8d9` (`[grug] Bound the QuACK TVM-FFI reproducer`).
- Commands:
  - Failing case: `CUDA_VISIBLE_DEVICES=0,1,2 XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 python -u experiments/grug/moe/repro_jaxpp_quack_minimal.py --timeout 30 --stack-after 10` under `/dlwh/quack-repro-bounded-hang-fsdp3-20260711` on three RNO2A H100s.
  - Passing control: `CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 python -u experiments/grug/moe/repro_jaxpp_quack_minimal.py --fsdp 2 --experts 2 --timeout 30 --stack-after 10` under `/dlwh/quack-repro-bounded-pass-fsdp2-20260711` on two RNO2A H100s.
- Results:
  - The fsdp2 control returned from `direct_eval` in `2.3943s`, emitted `verdict=pass`, and exited 0.
  - The fsdp3 case remained in `jax/_src/interpreters/pxla.py:388` during both 10-second stack dumps, emitted `watchdog_timeout` and `verdict=hang` at 30 seconds, and exited 124. Both Iris jobs are terminal.
  - The script now defaults to direct JAX at experts3/tokens-per-expert1/dims8x8/fsdp3, lazily imports JaxPP only for the optional MPMD control, verifies JAX/JAXLIB/QuACK/TVM-FFI versions, and emits a stable pass/hang/error verdict. The adjacent README pins the tested Python packages and JaxPP revision and documents both commands and external CUDA/H100 prerequisites.
- Interpretation:
  - The reproducer is self-contained below the Marin stack and safely bounded for upstream triage. The validated boundary is unchanged: fsdp2 passes and fsdp3 hangs, so JaxPP is not required to trigger the concurrent shared-handler defect.
- Next action:
  - Keep #7110 as the Marin tracking issue and package this commit for human-selected upstream filing; do not file it upstream automatically.

### 2026-07-11 02:28 PDT - output-oriented XLA ring combine hard negative
- Hypothesis: Replacing the bulk ring's zero-initialize plus atomic scatter-add combine with an assignment-to-dispatch inverse map and a top-k gather/reduction will let XLA emit one dense producer for the existing ReduceScatter operand.
- Commit Hashes:
  - `3636f7242c`: opt-in `ring_fused` backend with balanced/overflow output and full-gradient parity.
  - `18e86f5763`: launcher support for the backend.
- Command: L8/d2560/e64/top-k4/b32/m4/seq4096 four-stage EP8 `ring_fused` smoke under parent `/dlwh/iris-run-job-20260711-092136`, with CuTe FA4 and Pallas-Triton grouped GEMM (`block_k=32`, 8 warps).
- Results:
  - Parent and all four child tasks succeeded; all tasks are terminal. Compilation took about `195s`, and the first complete metric row followed after about `197.9s`.
  - The run completed finite steps with losses `11.7210522` and `11.7035103`. Mean MFU was `6.2896`, final throughput `301,763.37` tokens/s, and final duration `0.434354s`. W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ringfused-smoke-l8-e64k4-b32-s4096-p4m4-20260711-0220>.
  - The matching bulk-ring smoke reached `9.4180` MFU, `625,994.6` tokens/s, and `0.209382s`. The inverse-gather path regressed MFU by `33.22%`, throughput by `51.79%`, and step duration by `107.45%`.
  - XPlane/HLO attribution for the bulk baseline projects three 640 MiB scatter families at about `8.21s` per m256 step. Eliminating all three has an estimated ceiling near `20.32` MFU; combine alone has a ceiling near `18.79`.
- Interpretation:
  - XLA does not lower the logical `[tokens, top-k, hidden]` gather/reduction efficiently enough. Correctness and finite H100 execution are insufficient; this pure-XLA formulation is a hard performance negative and must not be scaled.
  - A viable continuation must use an explicit GPU kernel that writes `[tokens, hidden]` directly and explicit VJPs that replace dispatch-backward and combine-forward/backward scatters. Do not pursue more ordinary JAX rearrangements of the same inverse map.
- Next action:
  - Prototype the token-oriented primitive as a Mosaic GPU or Triton kernel with controlled backward. Require a reduced microbenchmark or L8 gate to beat bulk ring before any exact L24 run.

### 2026-07-11 04:55 PDT - explicit Triton routing is correct but performance-neutral
- Hypothesis: Reusing the token-grid Triton gather-sum kernel for combine forward and dispatch backward, plus a compact-grid combine-backward kernel, will remove the three profiled rank-2 floating scatter families and materially improve the reduced pipeline gate.
- Commit Hashes:
  - `0ecf680100`: explicit Triton routing kernels and custom VJPs.
  - `d149b66fa9`: corrected one-H100 GPU test binding.
- Commands:
  - One-H100 value/VJP and optimized-HLO test under `/dlwh/ring-fused-triton-gpu-test-r3-20260711`.
  - Bulk control parent `/dlwh/iris-run-job-20260711-114112`: `run_cw_jaxpp_may_d2560.sh --submit --cluster cw-rno2a --implementation explicit_mpmd --schedule std_1f1b --physical-stages 4 --layers 8 --experts 64 --top-k 4 --vocab-size 8192 --batch 32 --microbatches 4 --seq-len 4096 --moe-implementation ring --attention-implementation gpu_fa4_cute --ragged-dot-implementation triton --ragged-dot-block-k 32 --ragged-dot-num-warps 8 --loss-implementation xla --steps 10 --xla-memory-fraction 0.70`.
  - Fused comparison parent `/dlwh/iris-run-job-20260711-114818`: identical command with `--moe-implementation ring_fused`.
- Results:
  - The one-H100 test passed in `18.66s`. It validates output and VJP parity and confirms that the compiled backward HLO contains no rank-2 floating scatter.
  - A first corrected-vocab two-sample smoke ended at `9.5108` MFU and `0.207340s`, versus the historical bulk sample at `9.4180` and `0.209382s`; this apparent gain was too short to trust.
  - Both 10-step A/B parents and all child tasks succeeded, W&B finished, and losses matched within about `4e-6` at step 9. Bulk W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ring-v8192-ab-l8-e64k4-b32-s4096-p4m4-20260711-0442>. Fused W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ringfused-v8192-ab-l8-e64k4-b32-s4096-p4m4-20260711-0442>.
  - Excluding isolated duration outliers above `0.3s`, bulk ring had 7 central samples at mean/median `9.1735/9.1660` MFU and `0.214968/0.215138s`. `ring_fused` had 6 central samples at `9.1595/9.1559` MFU and `0.215344/0.215377s`.
  - The fused central mean changed by `-0.0139` MFU points (`-0.15%`) and duration by `+0.17%`. It also had two severe slow steps versus one for bulk ring.
- Interpretation:
  - The explicit kernels solve the targeted HLO-shape problem but do not produce a measurable end-to-end throughput gain. Other routing work, kernel launch overhead, or surrounding collective dependencies dominate enough to erase the projected scatter-only ceiling.
  - Do not run the L24/m256 comparison: a neutral reduced gate cannot plausibly close the `1.7417`-point gap from the `18.2583` MFU headline result to 20.
- Next action:
  - Keep `ring_fused` as a correctness-tested experimental backend and return to profile-guided pipeline/collective overlap work. Require a standalone kernel or reduced-pipeline gain comfortably above noise before another exact run.

### 2026-07-11 05:25 PDT - transfer-priority task ordering regresses
- Hypothesis: Constructing activation and `d_hidden` transfers immediately after their producers, before local QB/loss/parameter-gradient accumulation tasks, will reduce exposed pipeline rendezvous time without changing numerical dependencies.
- Commit Hash: `247fa5de02` (`[grug] Prioritize explicit pipeline transfers`).
- Command: pinned L8/d2560/e64/top-k4/seq4096/b32/m4/vocab8192/XLA-loss bulk-ring run with `--explicit-mpmd-schedule-mode transfer_priority --steps 10` under parent `/dlwh/iris-run-job-20260711-121447`.
- Results:
  - Parent and all four child tasks succeeded, W&B finished, and all eight timed rows were finite with no duration outlier above `0.3s`. W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ring-transferprio-v8192-ab-l8-e64k4-b32-s4096-p4m4-20260711-0515>.
  - Mean/median MFU was `8.9243/8.9006`; mean/median duration was `0.220987/0.221555s`; mean throughput was `593,180.44` tokens/s.
  - The matching default-order control reported mean/median `9.1735/9.1660` MFU and `0.214968/0.215138s`. Transfer priority regressed mean MFU by `2.72%`, median MFU by `2.90%`, and median duration by `2.98%`.
  - Final loss `8.7083950` remains close to control `8.7085800`; this is a performance negative, not a numerical failure.
- Interpretation:
  - Calling `mpmd.transfer` earlier does not force useful overlap and likely perturbs the per-rank queue away from the better default order. XLA latency scheduling and JaxPP task construction ordering are both now bounded negatives.
  - Do not scale this mode to L24. Keep it opt-in as a reproducible schedule-ordering experiment.
- Next action:
  - Prototype input-gradient-first backward: produce and transfer `d_hidden` before deferrable parameter-gradient work without recomputing the full stage backward twice. Require CPU value/gradient parity and at least a `5%` reduced H100 duration gain before an exact run.

### 2026-07-11 06:25 PDT - input-gradient-first backward is a hard negative
- Hypothesis: Splitting middle/last-stage backward into activation-gradient (`BWD_I`) and independently rematerialized per-block weight-gradient (`BWD_W`) tasks will transfer `d_hidden` earlier and fill pipeline bubbles without replaying the full six-layer cotangent chain.
- Commit Hash: `82ad228b18` (`[grug] Split input and weight gradients`).
- Validation:
  - Tiny FP32 CPU tests compare the split against combined `jax.grad(argnums=(0, 1))` for middle and final stages under `recompute_all` and `save_moe`. Loss, `d_hidden`, every parameter-gradient leaf, two-microbatch accumulation, and one Adam update match within `2e-5` (`4 passed`).
  - Two-device explicit-MPMD lowering produced two local programs.
- Command: pinned L8/d2560/e64/top-k4/seq4096/b32/m4/vocab8192/XLA-loss bulk-ring run with `--explicit-mpmd-schedule-mode input_gradient_first --steps 10` under parent `/dlwh/iris-run-job-20260711-131818`.
- Results:
  - The 32-H100 child compiled every `BWD_I`, `BWD_W`, accumulation, and update task and completed 10 finite steps. W&B finished: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ring-inputgradfirst-v8192-ab-l8-e64k4-b32-s4096-p4m4-20260711-0525>.
  - Excluding the final `0.9375s` runtime stall, seven central samples averaged/median `6.5356/6.5368` MFU, `434,406.26/434,488.75` tokens/s, and `0.301733/0.301669s`.
  - The matching combined-backward control averaged/median `9.1735/9.1660` MFU and `0.214968/0.215138s`. The split regressed central mean MFU by `28.75%` and increased mean duration by `40.36%`.
  - Final loss was finite at `8.7114143`; CPU parity and the close training trajectory rule out a gross gradient-semantics failure.
- Interpretation:
  - The intended schedule mechanism is real: `d_hidden` becomes available after `BWD_I`, and ZeroBubble places `BWD_W` separately. However, recomputing each block independently for both input and weight gradients adds enough work to dominate any recovered bubble.
  - Do not scale this mode to L24. A useful B/W split would need a kernel/autodiff primitive that emits input gradients and reusable weight-gradient residuals in one traversal, not JAX-level block rematerialization.
- Next action:
  - Move to the remaining profile-backed transport hypothesis: a two-chunk bulk ring that preserves AllGather/Pallas grouped-GEMM/ReduceScatter semantics while exposing chunk-N compute against chunk-N+1 communication. Require direct EP8 value/VJP parity and at least `5%` one-node kernel improvement before a pipeline gate.

### 2026-07-11 07:14 PDT - exact two-chunk bulk ring regresses
- Hypothesis: Splitting the proven bulk AllGather/grouped-GEMM/ReduceScatter dataflow into two equal token chunks will let XLA overlap chunk-1 gather with chunk-0 compute and chunk-0 ReduceScatter with chunk-1 compute while preserving total communication and padded GEMM work.
- Commit Hash: `0dffb28f07` (`[grug] Add two-chunk EP ring benchmark`).
- Validation:
  - The private backend performs one full global routing prepass, preserves source-major capacity/drop selection, and uses a globally consistent `pmin` gate to fall back to bulk when either half exceeds its capacity.
  - Forced-CPU EP8 tests cover balanced, overflow, boundary-spanning, odd-capacity, and one-half fallback routing. BF16 output/drop and VJPs for activations, combine weights, `w13`, and `w2` match bulk (`5 passed`); forward and `value_and_grad` lowering also pass.
- Command: one-node H100 job `/dlwh/ep-ring-two-chunk-benchmark-20260711`, exact e64/top-k4/d2560/i1280 microbatch32x4096, capacity factor `1.25`, Pallas-Triton `block_k=32`/8 warps, `XLA_FLAGS=--xla_gpu_enable_latency_hiding_scheduler=true`, 5 warmups and 30 blocked iterations.
- Results:
  - Job succeeded in `52.17s`; the two-chunk fast path was selected, no assignments dropped, and output plus all four gradient groups passed parity. No failed job remains.
  - Forward median: bulk `10.3885ms`, two-chunk `10.9390ms`, a `5.30%` regression.
  - Forward-backward median: bulk `22.9477ms`, two-chunk `26.1564ms`, a `13.98%` regression (`0.877x` speedup).
  - Forward added `0.5505ms`; forward-backward added `3.2086ms`, indicating that doubled backward collectives/Pallas launches serialize rather than hide.
- Interpretation:
  - The exactness and routing fallback are not the blocker. At this already-large chunk size, launch and backward collective overhead exceed any overlap exposed by the HLO DAG.
  - Do not register or pipeline-gate the backend. Profiling its slower timeline would explain the negative but would not make it a credible route to 20 without a lower-level asynchronous/autodiff primitive.
- Next action:
  - Preserve the single-chunk bulk transport and investigate a faster grouped-GEMM implementation at its exact forward/dLHS/dRHS shapes, including whether the existing QuACK/Sonic kernels can operate on EP-local expert weights without the no-EP FSDP materialization penalty.

### 2026-07-11 08:34 PDT - EP-local QuACK is faster but fails output parity
- Hypothesis: Replacing only the EP-local Pallas expert MLP with QuACK's Sonic grouped GEMM, while preserving the proven single-chunk ring routing and collectives, will improve the exact target kernel shape without materializing experts across the FSDP axis.
- Commit Hashes:
  - `b920782234`: private EP-local QuACK benchmark adapter.
  - `2427e8a3df`: sharding-safe host aggregation for diagnostic parity breakdowns.
- Command: one-node H100 job `/dlwh/ep-ring-quack-diagnostic-r4-20260711-082636`, exact e64/top-k4/d2560/i1280 microbatch32x4096, capacity factor `1.25`, balanced routing, Pallas-Triton `block_k=32`/8 warps, QuACK through the mutex-patched NVIDIA `jax-tvm-ffi` revision `e238a28483123efc8f56b9de358c2fb8b8de77e5`, 5 warmups and 30 blocked iterations. The benchmark ran with `--parity-mode diagnostic`; this mode reports timings but explicitly marks a parity failure non-promotable.
- Results:
  - The eight-H100 job succeeded in `91.29s` with zero task failures or preemptions. No failed live job remains.
  - Forward median improved from ring `10.410606ms` to `8.977512ms`, a `1.160x` speedup.
  - Forward-backward median improved from ring `23.021085ms` to `20.778980ms`, a `1.108x` speedup. This clears the predeclared roughly `10%` directional performance gate.
  - Output parity failed: relative L2 error `0.00612677`, mismatch fraction `0.02614194` (`8,771,779` elements), mean absolute error `0.00114951`, and maximum absolute error `0.03125`. No assignments were dropped.
  - The discrepancy is uniform across the eight ranks: mismatch fractions span `0.0261151-0.0261711` and relative L2 errors span `0.0061210-0.0061323`. This rules out a single-rank or routing-shard concentration.
  - Gradients for combine weights, `w13`, `w2`, and activations all pass the configured `allclose` tolerance with zero mismatch counts. Their maximum absolute errors are `4.47e-08`, `1.19e-06`, `1.86e-09`, and `5.09e-08`, respectively. Large relative errors for `w13` and activations come from very small reference norms.
- Interpretation:
  - QuACK is the first expert-kernel substitution in this series with enough direct forward-backward gain to plausibly close the remaining `9.5%` throughput gap from `18.2583` to 20 MFU.
  - The result cannot be promoted or pipeline-gated while output parity fails. Uniform rank errors and passing gradients point toward a deterministic BF16 numerical difference inside the grouped MLP rather than a ring routing or collective ownership bug, but that attribution remains a hypothesis.
  - Do not change tolerances to accept the result. Keep the adapter private and diagnostic-only until the output discrepancy is explained or removed under the existing gate.
- Next action:
  - Isolate the output discrepancy in the smallest direct QuACK-versus-Pallas grouped-MLP case at the exact dtype/activation shape, checking accumulation precision and activation implementation. Promote to a reduced JaxPP pipeline gate only after strict output parity passes.

### 2026-07-11 08:54 PDT - QuACK discrepancy isolated to approximate SwiGLU
- Hypothesis: The EP8 output mismatch is caused by either an adapter layout/ownership bug, grouped-GEMM accumulation differences, or QuACK's fused activation math.
- Commit Hashes:
  - `1a58bb61e8`: standalone one-H100 grouped-MLP numerical reproducer and metric test.
  - `b18cdd86a3`: comparison from a common recomputed preactivation.
  - `daf199f8ff`: decoding of QuACK's interleaved gated-preactivation layout.
- Command: one-H100 job `/dlwh/quack-grouped-mlp-numerics-r3-20260711-0848`, BF16, 8 experts, 16 rows per expert, hidden dimension 2560, intermediate dimension 1280, comparing Pallas-Triton, XLA, and QuACK W13, SwiGLU, W2, and full MLP outputs. The job succeeded with no task failures and no live resources remain.
- Results:
  - QuACK W13 output is bitwise identical to Pallas after decoding QuACK's internal interleaved `[gate_0, up_0, gate_1, up_1, ...]` storage.
  - QuACK W2 and Pallas W2 outputs are bitwise identical when supplied the same hidden activation. Pallas-Triton and XLA are bitwise identical end to end.
  - From the exact same preactivation, QuACK fused SwiGLU versus JAX `silu(gate) * up` has relative L2 error `0.00465063` and maximum absolute error `0.0625`. QuACK implements sigmoid with fast approximate FP32 `exp2` and reciprocal before converting the fused activation output to BF16.
  - The resulting full MLP output has relative L2 error `0.00519745`, maximum absolute error `0.01171875`, and mismatch fraction `2.5528%`. This closely reproduces the EP8 relative L2 `0.00612677` and mismatch fraction `2.614%`.
  - Focused validation passes: `uv run pytest -q lib/levanter/tests/grug/test_benchmark_ep_ring.py` reports `9 passed`; `./infra/pre-commit.py --changed-files --fix` passes.
- Interpretation:
  - The EP adapter, routing, weight layouts, and grouped GEMMs are exonerated. The output difference is deterministic, intrinsic to QuACK's fused approximate activation, and quantitatively reproduced without EP or JaxPP.
  - This is a semantics decision rather than an unresolved implementation bug. Strict parity correctly rejects the backend because it does not compute the same BF16 function as the current Pallas/JAX path, even though the approximation is bounded and gradients passed the prior gate.
  - Do not silently relax tolerances. A reduced pipeline training gate requires explicit approval to treat QuACK's approximate SwiGLU as an intentional model-kernel change, or an exact-SiLU QuACK variant that preserves the measured speedup.
- Next action:
  - Ask whether approximate SwiGLU is acceptable for this research branch. If approved, add an explicit opt-in semantic mode and run finite-step/loss-trajectory parity before the L24 performance comparison; otherwise investigate an exact activation variant and retain strict parity.

### 2026-07-22 09:35 PDT - approximate QuACK finite-step training gate passes
- Hypothesis: Treating QuACK's approximate fused SwiGLU as an explicit research-only semantic mode will preserve a close finite training trajectory while delivering a measurable end-to-end expert-kernel gain.
- Commit Hashes:
  - `8bf9459759`: register explicit EP backend `ring_quack_approx` without changing the default ring path.
  - `ea3470797b`: install the pinned mutex-patched TVM-FFI runtime for non-pipelined controls as well as JaxPP runs.
  - `8f23ed47e4`: update the launcher to the current CoreWeave kubeconfig path.
- Commands: paired one-node RNO2A H100x8, no-pipeline L2/d2560/e64/top-k4/seq4096/b32/vocab8192 runs with identical seed and synthetic data, CuTe FA4, Pallas-Triton `block_k=32`/8 warps, XLA loss, and 10 steps:
  - Ring parent `/dlwh/iris-run-job-20260722-162358`, W&B <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-no-pipeline-ring-l2-e64k4-b32-s4096-approx-ab-r2-20260722>.
  - QuACK parent `/dlwh/iris-run-job-20260722-162833`, W&B <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-no-pipeline-ringquackapprox-l2-e64k4-b32-s4096-approx-ab-r2-20260722>.
- Results:
  - Both parents succeeded with zero failures or preemptions; no live jobs remain. All ten loss and runtime rows are finite.
  - Ring mean/p50 MFU was `21.1689/21.1718`; QuACK was `22.0161/22.0633`, improving mean by `0.8472` points (`+4.00%`) and p50 by `4.21%`.
  - Mean throughput increased from `1,303,719` to `1,355,183` tokens/s (`+3.95%`); median step duration fell from `100.546ms` to `96.484ms` (`-4.04%`).
  - Step 0 losses are identical at `9.043948174`. QuACK-minus-ring loss drift increases monotonically from `+0.000225067` at step 1 to `+0.001182556` at step 9; final relative delta is `+0.01356%` and trajectory RMSE is `0.00079427`.
  - Across both layers and all steps, ring drops `32,463 / 10,485,760` assignments (`0.30959%`) and QuACK drops `33,722` (`0.32160%`), a `+0.01201`-point difference. Maximum router load-balance/z-loss/entropy deltas are `8.58e-6`, `4.58e-5`, and `2.34e-5`.
- Interpretation:
  - The explicit approximate semantics are stable over this bounded training gate. The small monotonic loss drift is expected from the accepted activation change and is not accompanied by instability or material router divergence.
  - The direct grouped-MLP benchmark's `10.8%` forward-backward gain becomes `4.0%` at whole-step L2. Simple extrapolation would move the L24 `18.2583` baseline only to about `18.99`, below target; pipeline interaction must be measured before paying for the exact run.
- Next action:
  - Run the pinned reduced L8/d2560/e64/top-k4/seq4096/b32/m4 explicit-MPMD `std_1f1b` QuACK gate against the matching ring baseline. Scale to L24 only if the pipeline gain is materially larger than the one-node whole-step gain or combines with another validated improvement to project above 20 MFU.

### 2026-07-22 10:20 PDT - approximate QuACK is neutral under JaxPP
- Hypothesis: QuACK's direct `10.8%` forward-backward expert-MLP gain, and `4.0%` one-node whole-step gain, will remain material inside the explicit-MPMD pipeline and justify an exact L24 run.
- Commit Hash: `51eaa57909` (`[docs] Record approximate QuACK training gate`), with backend implementation at `8bf9459759`.
- Commands: paired RNO2A 4x8 H100 explicit-MPMD `std_1f1b`, L8/d2560/e64/top-k4/seq4096/vocab8192/b32/m4, CuTe FA4, XLA loss, Pallas-Triton `block_k=32`/8 warps, 10 steps:
  - Ring parent `/dlwh/iris-run-job-20260722-170523`, W&B <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ring-v8192-ab2-l8-e64k4-b32-s4096-p4m4-20260722>.
  - QuACK parent `/dlwh/iris-run-job-20260722-170546`, W&B <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ringquackapprox-v8192-ab2-l8-e64k4-b32-s4096-p4m4-20260722>.
- Results:
  - Both parents and all eight child tasks succeeded with zero failures or preemptions; no live jobs remain.
  - Raw all-sample means are misleading because ring has two runtime stalls (`747.638ms`, `951.211ms`) and QuACK one (`688.421ms`). Raw mean MFU is `7.8723` ring versus `8.7453` QuACK, an outlier-count artifact.
  - Restricting to observed history rows below `300ms`, ring mean/p50 MFU is `9.4613/9.4843`; QuACK is `9.4786/9.4928`, only `+0.18%/+0.09%`.
  - Clean mean/p50 duration is `208.433/207.918ms` for ring and `208.050/207.732ms` for QuACK (`-0.18%/-0.09%`). Clean mean throughput changes from `628,870` to `630,024` tokens/s (`+0.18%`).
  - Loss remains finite and drifts monotonically as expected from approximate activation semantics: over retained steps 2-9, trajectory RMSE is `0.00355255`; final QuACK-minus-ring loss is `+0.004690170`, or `+0.05386%`.
- Interpretation:
  - QuACK's local grouped-MLP advantage is hidden by or small relative to pipeline transport, attention, and scheduling. Compile/setup duration differs, but steady-state throughput is neutral.
  - Do not launch the exact L24 QuACK run. A `+0.18%` reduced gain cannot close the `9.5%` relative gap from `18.2583` to 20 MFU.
  - Retain `ring_quack_approx` as an explicit research backend and the standalone numerical/training evidence, but stop performance scaling on this path.
- Next action:
  - Check current JAX/NVIDIA ragged-all-to-all support and the in-repo FP8-over-the-wire ring work. Prefer a bounded FP8 ring parity/microbenchmark because the prior zero-copy ragged path is already blocked by RNO2A CUDA VMM permissions and unflagged ragged transport was far below ring.

### 2026-07-22 10:31 PDT - upstream ragged and NCCL EP review
- Hypothesis: A current upstream JAX or NVIDIA transport may provide variable-count MoE dispatch/combine without the RNO2A exportable-VMM failure or the prior unflagged ragged-all-to-all regression.
- Commit Hash: research snapshot `f6e9040001` (`[docs] Record reduced QuACK pipeline result`).
- Sources:
  - [JAX 0.11.0](https://github.com/jax-ml/jax/releases/tag/jax-v0.11.0) and the OpenXLA changes making [NCCL barrier synchronization](https://github.com/openxla/xla/commit/26072180faecf3a4ee2ad72c55328729a560d050) and [symmetric output memory](https://github.com/openxla/xla/commit/c1fcf2507528eec72374aeb2eddd85d028939cc2) the ragged-all-to-all implementation.
  - [CUDA virtual memory management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html), which documents the IMEX permission requirement for fabric handles.
  - [NCCL EP v0.1.0](https://github.com/NVIDIA/nccl/releases/tag/nccl-ep-v0.1.0) and [Transformer Engine JAX NCCL EP integration](https://github.com/NVIDIA/TransformerEngine/pull/3036).
- Result:
  - JAX 0.11/OpenXLA does not fix the known RNO2A failure: ragged all-to-all still relies on symmetric/exportable output memory, so the environment's `CUDA_ERROR_NOT_PERMITTED` allocation remains the blocker. A JAX-only upgrade is not a credible next benchmark.
  - Transformer Engine `main` now exposes NCCL EP HT dispatch/combine through JAX FFI with custom VJPs and sharding rules. Its non-zero-copy path stages payloads and therefore avoids the failed XLA exportable-VMM allocation.
  - The integration requires one process per GPU, BF16, SM90 or newer, and NCCL 2.30.4 or newer. It is a separate runtime/integration path rather than a drop-in `jax.lax.ragged_all_to_all` improvement.
- Interpretation:
  - No released upstream primitive yet has a published end-to-end training win over Marin's EP8 ring at the target shape. Transformer Engine NCCL EP HT is nevertheless the first credible transport candidate that bypasses the established RNO2A blocker.
- Next action:
  - First, build a pinned Transformer Engine `main` plus NCCL 2.30.7 environment and benchmark non-zero-copy NCCL EP HT against Marin ring on one H100x8 node at BF16, EP8, d2560, top-k4, and 16,384 local tokens per rank. Sweep `max_num_sms={auto,8,16}` and require at least `10%` lower routed-MLP time with no more than `2%` expert-GEMM regression before integrating it into JaxPP; require at least `5%` matched full-training throughput gain before scaling to the 4x8 target.

### 2026-07-22 10:58 PDT - packed FP8 pipeline-wire implementation
- Hypothesis: Compressing inter-stage activations and gradients can reduce the profile's dominant SendRecv traffic enough to improve the saturated standard schedule without changing ring EP or adding a second collective.
- Commit Hash: `143fbb8752` (`[grug] Add FP8 pipeline wire experiment`).
- Config:
  - New research-only `explicit_mpmd_pipeline_wire_format="fp8"` mode for explicit-MPMD `std_1f1b` with more than one microbatch; the default remains `bf16`.
  - Forward activations use per-token current-scale E4M3 and backward `d_hidden` uses E5M2. The FP8 bytes and four FP32 scale bytes per token are packed into one rank-3 `uint8[..., H+4]` tensor, so each edge still issues one JaxPP transfer.
  - Same-rank edges bypass quantization. Cross-rank `.done()` remains immediately before receiver-side dequantization and the consuming stage task.
- Validation:
  - `uv run pytest -q tests/test_grug_moe_pipeline_wire.py`: `8 passed`, covering the byte-level wire shape, exact zero behavior, JIT round trips, and bounded E4M3/E5M2 error.
  - `uv run pytest -q tests/test_grug_moe_pipeline_wire.py tests/test_grug_moe_input_gradient_first.py`: `12 passed`; the existing split-backward parity suite is unchanged.
  - Valid and invalid launcher dry runs, shell syntax, Pyrefly, and `./infra/pre-commit.py --changed-files --fix` pass.
- Interpretation:
  - The wire payload falls from `2H` BF16 bytes to `H+4` bytes per token, or `50.08%` of baseline at `H=2560`. H100 pack/unpack overhead, MPMD transfer lowering, finite training drift, and end-to-end throughput remain unvalidated.
- Next action:
  - Run a paired reduced 4x8-H100 L8/d2560/e64/top-k4/seq4096/b32/m4 explicit `std_1f1b` comparison at the same commit. Scale to L24/b8192/m256 only if the reduced result credibly projects above 20 MFU.

### 2026-07-22 11:08 PDT - reduced FP8 pipeline-wire gate
- Hypothesis: Halving inter-stage activation and gradient payloads will improve steady-state explicit-MPMD throughput enough to justify a depth-matched confirmation.
- Commit Hash: run snapshot `87cc5f2b8f` (`[docs] Record FP8 pipeline wire hypothesis`), implementation `143fbb8752` (`[grug] Add FP8 pipeline wire experiment`).
- Commands: paired RNO2A 4x8 H100 explicit-MPMD `std_1f1b`, L8/d2560/e64/top-k4/seq4096/vocab8192/b32/m4, ring EP8, CuTe FA4, Pallas-Triton `block_k=32`/8 warps, XLA loss, `save_moe`, 20 steps, and XLA preallocation `0.70`. The only axis was `explicit_mpmd_pipeline_wire_format={bf16,fp8}`.
- Runs:
  - FP8 parent `/dlwh/iris-run-job-20260722-175111`; W&B <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ring-fp8wire-ab3-l8-e64k4-b32-s4096-p4m4-20260722>.
  - BF16 parent `/dlwh/iris-run-job-20260722-175133`; W&B <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ring-bf16wire-ab3-l8-e64k4-b32-s4096-p4m4-20260722>.
- Results:
  - Both parents, all eight child tasks, and both W&B runs succeeded at step 19 with zero failures, preemptions, or restarts. FP8 pack/unpack programs lowered and executed on all four stages.
  - Across all 18 matched timed steps, FP8 mean/p50/p90 MFU is `9.2079/9.6398/9.6753`; BF16 is `8.9517/9.4819/9.5396`. The raw means are distorted by one FP8 `1.0125s` stall and one BF16 `0.7160s` stall.
  - Removing matched step pairs where either duration exceeds `300ms` leaves 16 samples. FP8 mean/p50/p90 MFU is `9.6342/9.6398/9.6754`; BF16 is `9.3081/9.4926/9.5406`, or `+3.50%/+1.55%/+1.41%`. The clean mean remains influenced by one BF16 `299.493ms` sample just below the declared threshold, so p50/p90 are the robust readout.
  - Clean p50 duration falls from `207.737ms` to `204.564ms` (`-1.53%`). Clean mean tokens/s is `618,687` BF16 versus `640,361` FP8 (`+3.50%`), with the same `299.493ms` caveat.
  - All 18 logged losses are finite. FP8-minus-BF16 loss RMSE is `0.00022263`, maximum absolute delta is `0.00036240`, and the final delta is `+0.00033188` (`+0.003954%`). These runs did not emit router or dropped-assignment metrics, so no router/drop comparison is available.
- Interpretation:
  - FP8 transfer is consistently positive in robust central throughput, but the `+1.55%` p50 gain alone would move the current `18.2583` best only to about `18.54`, below target.
  - The L8 result is directional: the exact target has three times as many layers and four times as many tokens per pipeline microbatch. Do not launch the exact L24/b8192/m256 target from this gate.
- Next action:
  - Run only a paired L24/b512/m16 confirmation, which preserves the 32-sequence microbatch while increasing depth from two to six layers per stage. Scale further only if that confirmation materially exceeds the reduced projection with finite loss.

### 2026-07-22 11:31 PDT - L24 FP8 pipeline-wire confirmation
- Hypothesis: Increasing stage depth from two to six layers while holding the 32-sequence pipeline microbatch fixed will preserve or amplify the reduced FP8 transfer gain enough to justify the exact m256 target.
- Commit Hash: run snapshot `cafb45ccf5` (`[docs] Record reduced FP8 pipeline result`), implementation `143fbb8752` (`[grug] Add FP8 pipeline wire experiment`).
- Commands: paired RNO2A 4x8 H100 explicit-MPMD `std_1f1b`, L24/d2560/e64/top-k4/seq4096/vocab8192/b512/m16, ring EP8, CuTe FA4, Pallas-Triton `block_k=32`/8 warps, XLA loss, `save_moe`, 20 steps, and XLA preallocation `0.70`. The only axis was `explicit_mpmd_pipeline_wire_format={bf16,fp8}`.
- Runs:
  - FP8 parent `/dlwh/iris-run-job-20260722-181121`; W&B <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ring-fp8wire-ab4-l24-e64k4-b512-s4096-p4m16-20260722>.
  - BF16 parent `/dlwh/iris-run-job-20260722-181105`; W&B <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ring-bf16wire-ab4-l24-e64k4-b512-s4096-p4m16-20260722>.
- Results:
  - Both parents, all eight child tasks, and both W&B runs succeeded at step 19 with zero failures, preemptions, or runtime restarts. The initial concurrent FP8 parent submission collided with BF16's second-resolution autogenerated Iris ID before creating a job; resubmitting FP8 alone succeeded.
  - Across all 18 matched timed steps, FP8 mean/p50/p90 MFU is `16.1221/16.5358/16.5527`; BF16 is `15.8713/16.2407/16.2651`, or `+1.58%/+1.82%/+1.77%`. Mean duration improves `1.47%` and p50 improves `1.79%`.
  - The stall filter excludes a matched pair when either axis exceeds its own run-duration median by more than `300ms`. This removes six steps and leaves 12 matched samples. FP8 mean/p50/p90 MFU is `16.5392/16.5358/16.5525`; BF16 is `16.2464/16.2447/16.2587`, or `+1.80%/+1.79%/+1.81%`.
  - Clean mean/p50 duration falls from `5.72357/5.72417s` to `5.62224/5.62339s` (`-1.77%/-1.76%`). Clean mean tokens/s rises from `366,407` to `373,010` (`+1.80%`).
  - All 18 logged losses are finite. FP8-minus-BF16 loss RMSE is `0.00213983`, maximum absolute delta is `0.00240803`, and final relative delta is `+0.034324%`. These runs did not emit router or dropped-assignment metrics.
- Interpretation:
  - The FP8 gain replicates at realistic stage depth and is stable across mean, p50, and p90. It does not amplify beyond the reduced result.
  - Applying the clean p50 gain to the best L24/b8192/m256 mean MFU projects about `18.59`, still `1.41` points below the 20-MFU target. The exact m256 FP8 run is not justified.
- Next action:
  - Stop scaling packed FP8 pipeline transfer for this objective. Retain it as an explicit research mode and return to a qualitatively different mechanism, led by the one-node Transformer Engine NCCL EP HT gate or a schedule that actually overlaps the exposed SendRecv rendezvous.

### 2026-07-22 11:58 PDT - FP8 expert-GEMM gate prepared
- Hypothesis: Replacing only the two EP-local Pallas-Triton ragged expert GEMMs with Hopper FP8 tensor-core kernels can reduce the exact routed-MLP critical path while preserving the proven ring routing and BF16 collectives.
- Commit Hashes:
  - `97301bbdfb` (`[grug] Add FP8 ring GEMM benchmark gate`) adds the ring adapter and exact-shape diagnostic.
  - `8232eedf5f` through `a61081dd1a` port the reviewed Haliax FP8 ragged-dot forward, dgrad, and wgrad implementation.
- Config:
  - One RNO2A H100x8 node, EP8, d2560/i1280, 64 experts, top-k4, microbatch `32x4096`, capacity factor `1.25`, balanced routing, five warmups, and 30 timed iterations.
  - `ring_fp8_gemm` keeps dispatch/combine and collectives in BF16. Only W13 and W2 grouped GEMMs use E4M3 inputs; the JAX 0.10.1 gate also uses E4M3 output gradients because mixed E5M2-by-E4M3 WGMMA requires JAX 0.11.
  - This is an approximate diagnostic, not a training-semantics acceptance. Full JaxPP integration must separately preserve overwrite semantics for FP8 scale/amax state instead of summing those leaves across microbatches.
- Validation:
  - Focused tests report `29 passed, 8 skipped`; forced multi-device CPU ring/state tests report `2 passed`; focused Pyrefly and `./infra/pre-commit.py --changed-files --fix` pass.
- Promotion gate:
  - Require finite output and gradients for `x`, combine weights, W13, and W2; relative L2 error below `0.10`; FP8 WGMMA evidence; at least `1.05x` forward and `1.10x` forward-backward median speedup versus the eight-warp ring baseline; and no OOM at preallocation `0.70`.
- Next action:
  - Run the bounded one-node diagnostic. Stop this branch of work if forward-backward misses `1.10x`; otherwise run a matched reduced JaxPP A/B before any exact L24/m256 attempt.

### 2026-07-22 12:09 PDT - FP8 expert-GEMM EP8 gate passes
- Hypothesis: Replacing only the two EP-local Pallas-Triton expert GEMMs with Hopper E4M3 ragged GEMMs will clear the direct timing gate without OOM or non-finite output/gradients.
- Commit Hash: `db3af7b19c` (`[docs] Record FP8 expert GEMM gate`), with diagnostic implementation in `97301bbdfb`.
- Command: RNO2A H100x8 job `/dlwh/ep-ring-fp8-gemm-diagnostic-20260722`, running `experiments/grug/moe/benchmark_ep_ring.py` for `ring` and `ring_fp8_gemm` at d2560/i1280, e64/top-k4, EP8, `32x4096` tokens, capacity factor `1.25`, balanced routing, Pallas-Triton `block_k=32`/8 warps, E4M3 forward and reverse FP8, five warmups, and 30 timed iterations.
- Result:
  - Iris succeeded with exit `0`, zero failed or preempted tasks, no OOM, and no live task remaining. Repeated CUDA VMM `CUDA_ERROR_NOT_PERMITTED` messages were non-fatal fallback warnings.
  - Median forward falls from `10.3630919475ms` to `8.3562679356ms`, a `1.24015x` speedup (`+24.0%`). Median value-and-grad falls from `22.9142620228ms` to `18.1024125777ms`, a `1.26581x` speedup (`+26.6%`). Both exceed the declared `1.05x` and `1.10x` gates.
  - All outputs and gradients are finite and both backends drop zero assignments. Relative L2 / maximum absolute error is: output `0.0656761 / 0.15625`; grad-x `0.0799739 / 3.72529e-09`; grad-combine-weight `0.0387690 / 2.98023e-07`; grad-W13 `0.0826002 / 1.15484e-07`; grad-W2 `0.0693730 / 2.32831e-08`. Every relative-L2 value is below the declared `0.10` diagnostic limit.
  - The benchmark's BF16 allclose aggregate remains false because the approximate FP8 output exceeds exact BF16 tolerance; this is expected and is not being relabeled as exact parity. Gradient allclose checks pass under the benchmark's BF16 tolerances.
  - `MOSAIC_GPU_DUMP_PTXAS=1` shows Mosaic kernels compiling for `sm_90a`, consistent with the E4M3 FP8 path. The captured ptxas log does not print literal instruction text, so it is not direct evidence of a specific WGMMA opcode.
- Interpretation:
  - The direct expert-kernel gain is large enough to justify pipeline integration. Unlike approximate QuACK, the value-and-grad improvement is `26.6%`, leaving room for a useful whole-step gain after pipeline and collective dilution.
  - This remains an explicit approximate research mode. Promotion now depends on finite training, bounded loss drift, and a clean reduced-pipeline p50 gain; no default backend or acceptance tolerance changes.
- Next action:
  - Add explicit-MPMD accumulation semantics for Haliax `OverwriteWithGradient` FP8 amax/scale state. Overwrite leaves must be updated rather than summed or divided by the microbatch count. Then run matched L8/d2560/e64/top-k4/seq4096/b32/m4 BF16-wire `std_1f1b` ring versus FP8-expert-GEMM training jobs on RNO2A.

### 2026-07-22 12:28 PDT - BF16 control for the FP8 expert-GEMM pipeline gate
- Hypothesis: The reduced L8 explicit-MPMD control provides a stable matched baseline for deciding whether FP8 expert GEMMs retain at least a `5%` central-throughput gain after pipeline, attention, optimizer, and ring-collective overhead.
- Commit Hash: `84fc011f7d` (`[grug] Add research FP8 expert GEMM training`).
- Command: RNO2A 4x8 H100 explicit-MPMD `std_1f1b`, L8/d2560/e64/top-k4/seq4096/vocab8192/b32/m4, BF16 wire, ring EP8, CuTe FA4, Pallas-Triton `block_k=32`/8 warps, XLA loss, `save_moe`, 20 steps, and XLA preallocation `0.70`. Parent `/dlwh/iris-run-job-20260722-192841`; W&B <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ring-fp8expert-ab1-l8-e64k4-b32-s4096-p4m4-control-20260722>.
- Results:
  - Parent, child, and all four tasks succeeded with exit `0`, zero failures, preemptions, OOMs, or resubmits. W&B finished at step 19 and all 18 timed losses are finite; final loss is `8.392844200`.
  - The run-duration median is `0.208797334s`. Applying the established stall rule, duration greater than the run median plus `300ms`, excludes steps 3 and 10.
  - Clean mean/p50/p90 MFU is `9.458379/9.457171/9.529986`; clean mean/p50/p90 duration is `0.208492/0.208515/0.209718s`; clean mean/p50 tokens/s is `628,677/628,597`.
  - No router-load or dropped-assignment metrics were emitted.
- Interpretation:
  - The FP8 candidate must reach at least `9.930030` clean-p50 MFU to clear the predeclared `5%` promotion gate.
- Next action:
  - Compare the matched E4M3 expert-GEMM candidate using the same stall rule and require all finite losses before scaling.

### 2026-07-22 12:39 PDT - FP8 pipeline attempt 1 exposes missing CUDA staging
- Hypothesis: The research FP8 expert-GEMM state and explicit microbatch accumulation will compile and execute in the matched reduced JaxPP pipeline.
- Commit Hashes:
  - `84fc011f7d` adds the research-only training integration.
  - `655869c13a` (`[grug] Preserve CUDA setup with custom worker scripts`) fixes the setup failure found by this attempt.
- Failed run: parent `/dlwh/iris-run-job-20260722-192856`, child `/dlwh/iris-run-job-20260722-192856/grug-train-jaxpp-rno2a-ring-fp8expert-ab1-l8-e64k4-b32-s4096-p4m4-e4m3-20260722`.
- Result:
  - The first candidate failed while compiling stage-0 forward before any training step with `JaxRuntimeError: UNAVAILABLE: No PTX compilation provider is available. Neither ptxas/nvlink nor nvjitlink is available.` No FP8 optimizer/state semantics, numerical behavior, memory capacity, or throughput were exercised.
  - Root cause: Grug's custom JaxPP setup replaced Iris's automatic setup list. The GPU extra installed CUDA packages but omitted `cuda_toolchain_setup_script()`, so `ptxas` and `nvlink` were not staged onto the worker path.
  - Commit `655869c13a` preserves the default setup, inserts CUDA toolchain staging for GPU extras, and then runs custom JaxPP/DeepEP setup scripts. Pyrefly and changed-files precommit pass.
  - Relaunch parent `/dlwh/iris-run-job-20260722-193948` logs `staging CUDA toolchain` on all four workers and advances through stage-0, stage-1, and stage-2 compilation to `Compiling grug_1f1b_mb0_stage3_loss_backward`, proving the PTX-provider failure is fixed. As of 12:57 PDT it remains running and log-silent in that compile for about 14 minutes.
- Interpretation:
  - Attempt 1 is a setup failure, not a negative FP8 performance or stability result. The relaunch's prolonged stage-3 backward compile is a distinct possible JaxPP/XLA compile-stall result and must be resolved before the training gate can be interpreted.
- Next action:
  - Continue babysitting the relaunch without blind resubmission. If compile completes, collect all 20 training steps and compare to the control. If log freshness remains absent for two monitor cadences, collect task/process and compiler diagnostics, stop the failed parent, and package the stage-3 backward compile stall separately.

### 2026-07-22 13:01 PDT - FP8 expert-GEMM pipeline relaunch stalls compiling stage 3
- Hypothesis: After restoring `ptxas`/`nvlink`, the matched reduced FP8 candidate will compile all four stage programs and expose a finite training/performance result.
- Commit Hash: `655869c13a` (`[grug] Preserve CUDA setup with custom worker scripts`), with FP8 training integration at `84fc011f7d`.
- Command: same L8/d2560/e64/top-k4/seq4096/vocab8192/b32/m4, four-stage explicit-MPMD `std_1f1b`, BF16-wire, ring EP8, CuTe FA4, Pallas-Triton `block_k=32`/8 warps, XLA-loss control config, adding only `--research-fp8-expert-gemm`. Parent `/dlwh/iris-run-job-20260722-193948`; W&B <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ring-fp8expert-ab2-l8-e64k4-b32-s4096-p4m4-e4m3-20260722>.
- Results:
  - CUDA toolchain staging ran on all four workers. Compilation advanced at roughly 20-second intervals through stage-0 forward at 19:41:34 UTC, stage-1 forward and stage-0 accumulation at 19:41:54, stage-2 forward at 19:42:14, and stage-3 loss-backward at 19:42:35.
  - No further application log was emitted for more than 18 minutes. A live Iris thread dump on task 3 showed the main thread blocked in `jax._src.compiler.backend_compile_and_load`, called through `jaxpp.jax_primitives.apply_task`, `jaxpp.experimental._mpmd.eval_local`, and the Grug explicit-MPMD runner.
  - The run produced zero of 20 training steps, no loss or MFU history, and no PTX-provider, OOM, sharding, model, or Python exception. The W&B run remained marked running with zero rows when compute was stopped.
  - After two stale monitor cadences, only this parent was stopped. The parent and child are terminal `killed`; all four child tasks are terminal and no live resources remain.
- Interpretation:
  - The CUDA setup fix is validated, but the reduced FP8 pipeline training gate is blocked by a stage-3 backward backend compile stall. There is no finite-step or end-to-end performance evidence, so the direct `1.266x` expert-kernel result cannot be promoted or scaled.
  - A blind retry is not justified because the second attempt reached the same deterministic stage program and remained inside backend compilation without an infrastructure error.
- Next action:
  - Minimize stage-3 loss-backward compilation outside the full four-stage run, preserving the FP8 overwrite-state and microbatch-accumulation structure. Package an upstream-ready reproducer in Marin and file only against Marin before asking NVIDIA/JaxPP maintainers for a fix. Resume the paired performance gate only after the minimized program compiles or yields an actionable compiler error.

### 2026-07-22 14:22 PDT - isolated FP8 routed-MLP backward does not reproduce the stage-3 stall
- Hypothesis: The production FP8 expert GEMMs, delayed-scaling overwrite state, ring EP8 collectives, microbatch accumulation, and JaxPP task localization are sufficient to reproduce the stage-3 loss-backward compile stall without the rest of the Grug block or loss task.
- Commit Hashes:
  - `b496935ca7` adds the bounded direct/JaxPP FP8 expert-backward reproducer and watchdog event log.
  - `2c3b080dbf` adds the distributed-direct topology control.
  - `27aa048d06` preserves production ring sharding, collectives, and FP8 overwrite-state `pmax` semantics.
  - `1abcff61fc` adds the two-host external-worker mode used for eight devices per stage.
- Commands: exact fresh-name Iris commands, dependency pins, watchdog settings, and the topology ramp are recorded in `experiments/grug/moe/repro_jaxpp_fp8_expert_compile.README.md` at `1abcff61fc`. All GPU gates used JAX/JAXLIB `0.10.1`, JaxPP `7091a9b5ce02cd1a6bdc905f6a36e89370a5fba9`, CUDA 13, and `XLA_PYTHON_CLIENT_MEM_FRACTION=.50`.
- Results:
  - The one-device-per-stage ramp passed through L2, four microbatches, eight experts, hidden/intermediate `2560/1280`, and 65,536 tokens. The largest isolated JaxPP case compiled and executed in about `18.9s`; direct and distributed-direct controls stayed near `5.2-5.4s`.
  - Ring-sharded JaxPP FP8 passed with two devices per stage in `7.962s` and four devices per stage in `8.538s`. Matched distributed-direct BF16/FP8 and JaxPP BF16 controls also passed.
  - External two-host BF16 ring control `/dlwh/jaxpp-fp8-ring-dps8-bf16-20260722-211636` passed on 16 H100s. Rank 0/1 lower times were `0.0956/0.0925s`; JaxPP `eval_local` compile-and-execute times were `2.4112/4.5819s`.
  - Matched minimum FP8 ring `/dlwh/jaxpp-fp8-ring-dps8-fp8-20260722-211914` passed. Rank 0/1 lower times were `0.2500/0.2438s`; compile-and-execute times were `4.9585/9.8064s`, about `2.1x` the BF16 control.
  - Restored production expert-stage shape `/dlwh/jaxpp-fp8-ring-dps8-prodshape-20260722-212047` also passed: L2, four microbatches, 64 experts, top-k4, 32,768 tokens, hidden `2560`, intermediate `1280`, and eight devices per stage. Rank 0/1 lower times were `1.8194/1.8742s`; compile-and-execute times were `8.7470/24.3969s`.
  - Every job and task is terminal successful. No watchdog stack, timeout, OOM, compiler exception, resubmission, or cluster mutation occurred.
- Interpretation:
  - The isolated routed-MLP backward is insufficient to reproduce the production hang even at the production expert topology, capacity, width, layer count, microbatch count, and FP8 state shape. The result rules out those ingredients as a sufficient cause; it does not rule out an interaction with the complete stage-3 loss task.
  - FP8 increases localized compile-and-execute time materially, especially on rank 1, but the bounded cases complete in seconds. Packaging this version upstream would misrepresent the production failure because it has no failing case.
- Next action:
  - Add the smallest omitted stage-3 boundary, prioritizing the real language-model head/loss and complete value-and-grad output tree. Retain matched BF16 and direct controls, external dps8 topology, and hard watchdogs. File a separate Marin issue only after the minimized program reproduces or reaches a clearly actionable compiler failure; do not file NVIDIA upstream.

### 2026-07-22 14:43 PDT - last-stage head, loss, and full output tree still do not reproduce
- Hypothesis: Adding the production final normalization, language-model head, fused next-token loss, and complete last-stage value-and-grad result tree to the FP8 routed-MLP reproducer will trigger the stage-3 backend compile stall while a matched BF16 control completes.
- Commit Hash: `886326965c` (`[grug] Add last-stage FP8 compile repro boundary`).
- Config: two RNO2A H100x8 tasks, one JAX process and one JaxPP stage per task, L2/m4/e64/top-k4, 32,768 tokens as batch 8 by sequence 4096, hidden/intermediate `2560/1280`, vocab 8192, production ring EP8 sharding, JAX/JAXLIB `0.10.1`, JaxPP `7091a9b5ce02cd1a6bdc905f6a36e89370a5fba9`, XLA preallocation `.50`, stack dump after 120 seconds, and hard timeout after 1,200 seconds. The only A/B axis was BF16 versus E4M3 expert GEMMs and their delayed-scaling overwrite state.
- Runs:
  - BF16 control `/dlwh/jaxpp-last-stage-dps8-bf16-20260722-213810`.
  - FP8 candidate `/dlwh/jaxpp-last-stage-dps8-fp8-20260722-214009`.
- Results:
  - Both parents and all four tasks succeeded with exit `0`, zero failures or preemptions, and no live resources remaining.
  - BF16 rank 0/1 lower times were `0.53194/0.53780s`; JaxPP `eval_local` compile-and-execute times were `6.00513/17.55211s`.
  - FP8 rank 0/1 lower times were `1.92003/1.92225s`; compile-and-execute times were `10.93991/30.62062s`.
  - FP8 was about `3.6x` slower to lower and `1.7-1.8x` slower in localized compile-and-execute. Neither run emitted a watchdog stack, OOM, compiler error, traceback, timeout, or resubmit.
- Interpretation:
  - Final RMSNorm, gated norm, LM head, shifted labels, fused XLA cross-entropy, dynamic auxiliary output, full parameter-gradient tree, and all microbatch input cotangents are not sufficient to reproduce the production hang.
  - The monotonic FP8 compile penalty remains real, especially on rank 1, but the bounded task completes in about 31 seconds. The smallest missing production interaction now lies inside the transformer block/router/rematerialization path rather than at the head/loss or outer result-tree boundary.
- Next action:
  - Add the exact production block-rematerialization boundary around the routed MLP while keeping learned routing and attention excluded. Run another matched BF16/FP8 dps8 gate before adding a second structural axis.

### 2026-07-22 15:06 PDT - production save-MoE remat boundary also passes
- Hypothesis: The target launcher's block-level `save_moe` rematerialization policy interacting with FP8 ragged expert custom VJPs is sufficient to trigger the stage-3 compile stall.
- Commit Hash: `481bdf7c19` (`[grug] Add block remat FP8 compile repro mode`).
- Config: the completed next-token dps8 shape from the prior entry, adding only one `eqx.filter_checkpoint` per isolated residual expert block with production `jax.checkpoint_policies.save_only_these_names(*MOE_REMAT_SAVE_NAMES)` and default `prevent_cse=True`. This matches the target launcher's non-effectful `ring`/`save_moe` branch. Attention, learned routing, block norms, shared expert, optimizer, and the full scheduler remained excluded.
- Runs:
  - BF16 control `/dlwh/jaxpp-remat-dps8-bf16-20260722-215133`.
  - FP8 candidate `/dlwh/jaxpp-remat-dps8-fp8-20260722-215622`.
- Results:
  - Both jobs and all four tasks succeeded with exit `0`, zero failures/preemptions, and no live resources remaining.
  - BF16 rank 0/1 lower times were `0.97247/0.98491s`; JaxPP `eval_local` times were `5.98515/16.98365s`.
  - FP8 rank 0/1 lower times were `2.84200/2.83709s`; rank 0 `eval_local` was `10.43978s`. Finelog timed out before returning rank 1's exact completion event, but its successful completion barrier bounds `eval_local` below `30.47376s`.
  - Neither run emitted a watchdog stack, timeout, OOM, traceback, compiler failure, resubmit, or cluster mutation. The ad hoc post-completion Finelog timeout affected only log retrieval.
- Interpretation:
  - The exact production `save_moe` remat boundary around the isolated routed residual is not sufficient. Its compile-and-execute timing is close to the no-remat last-stage case for both BF16 and FP8.
  - The strongest remaining omitted compiler-shape delta is dynamic learned routing: router logits, top-k selection, capacity/group-size construction, dispatch ordering, combine weights, and router statistics. Pointwise per-block norms are less likely to explain an 18-minute backend stall and should not be bundled into the next test.
- Next action:
  - Add an opt-in production QB-routing boundary while keeping attention excluded and retaining `save_moe`, the last-stage loss, and full task result tree. Run a matched BF16/FP8 dps8 gate before adding attention or another structural axis.

### 2026-07-22 15:38 PDT - production learned routing also completes
- Hypothesis: Dynamic QB routing, top-k selection, assignment sorting/capacity construction, combine weights, and router statistics interacting with `save_moe` and FP8 expert VJPs are sufficient to trigger the stage-3 compile stall.
- Commit Hash: `e118667915` (`[grug] Add learned routing FP8 compile repro mode`).
- Config: the completed dps8 next-token/remat shape, replacing fixed balanced assignments with production `MoEMLP`/`MoEExpertMlp`: learned BF16 router parameters with FP32 logits, centered negative QB bias, biased top-(K+1), sigmoid and renormalized top-k combine weights, 1.25-capacity ring dispatch, assignment sorting, QB beta/router statistics, and zero-coefficient router z-loss matching the target launcher. Attention, attention and pre-MoE norms/gates, shared expert, optimizer, and outer scheduler remained excluded.
- Runs:
  - BF16 control `/dlwh/jaxpp-routing-dps8-bf16-20260722-221754`.
  - FP8 candidate `/dlwh/jaxpp-routing-dps8-fp8-20260722-222555`.
- Results:
  - Both jobs and all four tasks succeeded with exit `0`, zero failures/preemptions, and no live resources remaining.
  - BF16 task duration including setup was `96.92s`; FP8 task duration was `122.42s`. Both completed before the in-process 120-second watchdog because dependency setup precedes watchdog installation.
  - Finelog retrieval repeatedly timed out for these larger logs, so exact per-rank lower and `eval_local` events are unavailable. Structured Iris summaries remained available and are the terminal authority. No watchdog stack, OOM, compiler failure, resubmit, or cluster mutation occurred.
- Interpretation:
  - Production learned QB routing, differentiated top-k/sort/capacity, ring dispatch, `save_moe`, FP8 overwrite state, and the complete last-stage loss/output tree still are not sufficient to reproduce the hang.
  - The remaining major compute graph inside the localized task is the full block attention and normalization path. This is more plausible than the outer optimizer or scheduler because the observed production thread was compiling the localized stage-3 loss-backward task before optimizer execution.
- Next action:
  - Add an opt-in production block boundary with attention residual plus attention/pre-MoE RMS and GatedNorm around the learned routed MLP. Retain the target attention backend on H100 and a reference backend for CPU tests; keep optimizer and outer pipeline scheduling excluded.

### 2026-07-22 16:21 PDT - BF16 full-block failure is JaxPP-localization-specific
- Hypothesis: The actual final `TransformerPipelineStage` block, including CuTe FA4 attention, attention and MoE normalization/residual paths, learned routing, ring MoE, `save_moe`, final head/loss, and full gradients, is sufficient to reproduce the production JaxPP compile failure before adding FP8 expert GEMMs.
- Commit Hash: `c1b99d33be` (`[grug] Add full-block FP8 compile repro mode`).
- Commands:
  - BF16 JaxPP: the `Full-block CuTe FA4 dps8 gate` command in `experiments/grug/moe/repro_jaxpp_fp8_expert_compile.README.md`, using `--runtime jaxpp`.
  - BF16 direct control: the same dependency setup, topology, model arguments, and two H100x8 workers, changing only to `--runtime distributed_direct` and a fresh job name.
  - `uv run --package marin-iris --extra controller iris --cluster=cw-rno2a job summary /dlwh/jaxpp-full-block-dps8-bf16-20260722-230118`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-rno2a job summary /dlwh/full-block-dps8-bf16-direct-20260722-230952`
- Config: L8 final stage layers 6-7, d2560, 64 experts, top-k 4, sequence 4096, four microbatches, EP8 ring MoE, BF16 expert GEMMs, CuTe FA4 with 20 Q heads/5 KV heads/head dimension 128, 2048-token sliding window on layer 6, full causal layer 7, production learned QB routing, `save_moe`, final fused next-token loss, full parameter gradients, and input cotangents. JAX/JAXLIB `0.10.1`, JaxPP `7091a9b5ce02cd1a6bdc905f6a36e89370a5fba9`, patched JAX TVM FFI, and XLA preallocation `.50` were held fixed.
- Results:
  - JaxPP job `/dlwh/jaxpp-full-block-dps8-bf16-20260722-230118` failed terminally. Rank 1 exited `139` after `180.2s`; rank 0 became `cosched_failed`. The fatal stack was in `cutlass/_mlir/dialects/arith.py` `_isa`/`_is_any_of`/`_is_float_type`, `cutlass/_mlir/extras/types.py`, and `cutlass/base_dsl/typing.py:1594 from_mlir_type`. There was no OOM, missing dependency, or setup failure.
  - Distributed-direct job `/dlwh/full-block-dps8-bf16-direct-20260722-230952` succeeded with both tasks exit `0` in `100.63s` and `167.43s`, zero failures/preemptions, and no Cutlass DSL crash, OOM, watchdog termination, or compiler/setup error.
  - Finelog retained no phase events for the direct control, so separate lower/compile/execute timings are unavailable. Structured Iris summaries are the terminal authority. Neither job remains live.
- Interpretation:
  - BF16 is sufficient; the failure is not specific to FP8 expert GEMMs or overwrite state.
  - CuTe FA4 plus the full block under ordinary distributed JAX is not sufficient. JaxPP localization changes the compiler path in a way that triggers the Cutlass DSL segfault.
  - This is a smaller and more actionable boundary than the original 18-minute FP8 production stall, though the fatal symptom differs: native compiler segfault instead of a non-returning `backend_compile_and_load`.
  - Filed Marin bug [#7529](https://github.com/marin-community/marin/issues/7529) with `bug` and `agent-generated`; it is packaged for upstream review, but no NVIDIA issue was filed.
- Next action:
  - Do not launch the matched FP8 full-block run until the BF16 gate passes. Minimize the JaxPP/CuTe FA4 interaction within #7529 or consume a JaxPP/Cutlass fix, then resume the FP8 training A/B.

### 2026-07-23 00:01 PDT - private-output NCCL ragged all-to-all is functional but slower than ring
- Hypothesis: Forcing JAX 0.10.1's exact-count grouped NCCL send/receive fallback avoids the symmetric-memory failure of the one-shot ragged all-to-all path and improves the matched L24 pipeline over Marin's bulk ring transport.
- Commit Hash: `4d4c705cb6` (`[experiments] Add H100 NCCL_EP transport gate`).
- Command: RNO2A 4x8 H100 explicit-MPMD `std_1f1b`, L24/d2560/e64/top-k4/seq4096/vocab8192/b512/m16, ragged-all-to-all EP8, CuTe FA4, Pallas-Triton `block_k=32`/8 warps, XLA loss, `save_moe`, 10 steps, and XLA preallocation `0.70`. The launcher set both `--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=false` and `--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false`. Parent `/dlwh/iris-run-job-20260723-064055`; child `/dlwh/iris-run-job-20260723-064055/grug-train-jaxpp-rno2a-ragged-ncclfallback-l24-e64k4-b512-s4096-p4m16-20260722`; W&B <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ragged-ncclfallback-l24-e64k4-b512-s4096-p4m16-20260722>.
- Results:
  - Parent, child, and all four tasks succeeded. W&B finished after 10 finite steps with final loss `7.278883934`.
  - Mean/p10/p50/p90 MFU was `14.393416/14.356245/14.401264/14.415185`; latest throughput was `324,954.49` tokens/s, `4,562,288.73` GFLOP/s, and duration `6.453679s`.
  - The matched ring result at the same L24/b512/m16 geometry is `16.200488` mean MFU. The private-output fallback is `11.15%` slower.
  - FABRIC/POSIX_FD VMM warnings remained in the logs, but they were nonfatal because the selected fallback used ordinary output memory. Shutdown emitted PJRT coordination connection-refused noise only after task 0 completed; Iris classified the run as successful and no resources remain live.
- Source audit:
  - JAX 0.10.1 already contains this private-output host-metadata-sync plus grouped NCCL send/receive fallback. Current JAX 0.11 does not add a private-memory fast ragged kernel; its one-shot path remains symmetric-memory dependent.
  - Transformer Engine main at `4adad4c218c115cd9af235fb3d4e13ef4cec55a8` exposes a staged, non-zero-copy NCCL_EP transport with custom VJPs on H100. Its integrated grouped-GEMM MoE tests are SM100-only, so an H100 evaluation must first gate transport and then pair it with Marin GEMMs.
- Interpretation:
  - The exact-count NCCL fallback resolves the prior allocation failure but does not beat ring. At `11.15%` below ring it misses the predeclared within-10% promotion threshold, though it is not a catastrophic transport failure.
  - Ring remains the measured production winner. The remaining transport candidate is Transformer Engine NCCL_EP; a self-contained H100x8 transport gate is committed under `experiments/ncclep_h100`.
- Next action:
  - Run the H100x8 NCCL_EP transport gate. Stop if transport-only forward-backward exceeds its explicit `18.33144ms` sanity bound; otherwise build a paired eight-process NCCL_EP-plus-Marin-GEMM versus ring benchmark before changing the JaxPP transport.

### 2026-07-23 00:15 PDT - Transformer Engine NCCL_EP clears the H100 transport sanity gate
- Hypothesis: Transformer Engine's staged, non-zero-copy NCCL_EP transport has enough H100 headroom to justify a matched full routed-MLP comparison against Marin ring.
- Commit Hash: `34ce844e23` (`[docs] Record H100 NCCL ragged transport result`), with the gate implementation at `4d4c705cb6`.
- Command: the H100x8 Iris command in `experiments/ncclep_h100/README.md`. Job `/dlwh/ncclep-h100-ep8-gate-20260723-070226` built Transformer Engine `4adad4c218c115cd9af235fb3d4e13ef4cec55a8` for SM90, installed NCCL `2.30.7`, and launched eight supervised one-GPU JAX processes. Shape: EP8, 16,384 tokens/rank, d2560, e64/top-k4, BF16 token payloads, FP32 routing weights, uniform balanced routing, and 1.25 receive capacity.
- Results:
  - Iris succeeded in `8m52.57s`; the task and all eight ranks exited `0` with no failures or preemptions. The task-local TE wheel build, NCCL_EP JIT setup, bootstrap, dispatch, combine, and custom VJPs all completed. No resources remain live.
  - Dispatch-plus-combine forward median/p10/p90 was `14.26947/14.20567/14.90450ms`, corresponding to `41.1670` effective remote-wire GB/s.
  - Transport value-and-grad median/p10/p90 was `5.21039/5.19228/5.23367ms`, corresponding to `225.4849` effective remote-wire GB/s under the gate's two-round-trip byte model.
  - The value-and-grad median clears the predeclared `18.33144ms` unpaired sanity bound. Output, loss, token gradients, and routing-weight gradients were finite on every rank.
- Interpretation:
  - The gate promotes NCCL_EP to a paired full-MLP comparison; it does not establish a winner because the reference bound uses a different process topology and includes expert GEMMs.
  - Forward alone being slower than value-and-grad is counterintuitive. The historical TE microbenchmark notes that reverse-mode graph construction can eliminate FFI work when the primal is discarded; this gate used `value_and_grad`, but the inversion still warrants treating the absolute transport number as provisional. A full-MLP A/B with all parameter gradients and synchronized output leaves is the decisive measurement.
- Next action:
  - Compare current Marin ring against NCCL_EP plus the same Haliax/Pallas-Triton BF16 expert GEMMs in one eight-process H100 job. Require finite values, explicit output/loss/gradient parity, slowest-rank timing, and at least `10%` NCCL_EP p50 speedup before integrating it into JaxPP.

### 2026-07-23 08:03 PDT - paired full-MLP gate finds a small strict output mismatch
- Hypothesis: NCCL_EP dispatch/combine plus the same Pallas-Triton expert GEMMs will match Marin ring under the existing strict BF16 parity gate, allowing a paired full-gradient timing comparison.
- Commit Hashes:
  - `fbda960dd0` adds the paired H100x8 full routed-MLP A/B.
  - `5282a2f915` adds an explicit diagnostic-only timing mode after the strict run stopped before timing.
- Command: one RNO2A H100x8 task with eight one-GPU processes, EP8, 16,384 tokens/rank, d2560/i1280, e64/top-k4, BF16 inputs and weights, FP32 routing weights, uniform balanced routing, 1.25 capacity, and Haliax Pallas-Triton expert GEMMs at `block_k=32`/8 warps in both arms. Job `/dlwh/ncclep-h100-full-mlp-ab-20260723-072833`; strict parity mode.
- Results:
  - Iris succeeded in `8m56.82s`; the task and all eight ranks exited `0` with no failures or preemptions. No resources remain live.
  - Loss, token-gradient, routing-weight-gradient, W13-gradient, and W2-gradient checks all passed with zero elementwise mismatches. All values and gradients were finite and neither arm dropped assignments.
  - Output parity alone failed: `683,013 / 335,544,320` elements (`0.203554%`) exceeded `rtol=0.1, atol=0.0002`; relative-L2 error was `0.00296233`, mean absolute error `0.000289164`, and maximum absolute error `0.0078125`.
  - The harness stopped before timing as required. StableHLO showed ring with 4 all-gathers, 3 reduce-scatters, 5 all-reduces, and 6 custom calls; NCCL_EP showed no StableHLO collectives, 3 all-reduces, and 11 custom calls.
- Interpretation:
  - This is a narrow BF16 output discrepancy, not a gross adapter, routing, drop, gradient, or stability failure. It is smaller than the previously isolated approximate-QuACK discrepancy, but it still fails the declared strict gate and cannot be silently accepted.
  - No tolerance was changed. Diagnostic timing remains non-promotable while parity is false; it is useful only to decide whether isolating the transport/reduction-order difference is worth further work.
- Next action:
  - Rerun the same paired program in explicit diagnostic mode. Stop NCCL_EP if it does not beat ring by at least `10%`; if it does, isolate whether the output difference comes from combine accumulation order or another adapter detail before any JaxPP training integration.

### 2026-07-23 08:27 PDT - paired diagnostic shows a 1.45x NCCL_EP full-MLP gain
- Hypothesis: If NCCL_EP is materially faster than ring with identical expert GEMMs, the narrow output mismatch is worth isolating before deciding whether an exact or explicitly approximate backend is viable.
- Commit Hash: `cdd92278d1` (`[docs] Record strict NCCL_EP full-MLP parity gate`), with diagnostic support at `5282a2f915`.
- Command: the same paired H100x8 full routed-MLP program as the strict gate, launched remotely with `env PARITY_MODE=diagnostic`. Job `/dlwh/ncclep-h100-full-mlp-diagnostic-r2-20260723-151635`; six alternating warmup pairs and 20 alternating measured pairs, each sample aggregated as the slowest of eight ranks.
- Results:
  - Iris succeeded in `9m7.36s`; the task and all eight ranks exited `0` with no failures/preemptions. No resources remain live.
  - Marin bulk ring median/p10/p90 value-and-grad latency was `22.73490/22.58603/22.82575ms`.
  - Transformer Engine NCCL_EP plus the same Pallas-Triton expert GEMMs was `15.65238/15.62121/15.71140ms`.
  - NCCL_EP is `1.452489x` faster, a `31.1527%` latency reduction. The central distributions do not overlap.
  - The strict numerical result reproduced exactly: all values were finite, no assignments dropped, loss and all four gradient groups passed, while output retained `683,013` mismatches (`0.203554%`), relative-L2 `0.00296233`, and maximum absolute error `0.0078125`.
- Interpretation:
  - The gain easily clears the `10%` performance threshold and is large enough to matter at the pipeline level. It is the strongest exact-shape transport signal after ring.
  - The backend remains non-promotable: `status=stop` and the promotion criterion correctly stayed false because output parity failed. The result justifies one focused numerical isolation, not a tolerance change or immediate JaxPP integration.
- Next action:
  - Isolate TE dispatch/combine with top-k1 and top-k4 identity and per-expert-scaled identity cases. Determine whether the discrepancy is intrinsic BF16 combine accumulation order or an adapter error; only then choose an exact fix or request explicit approval for an approximate research backend.

### 2026-07-23 11:35 PDT - NCCL_EP combine isolation and pipeline promotion
- Hypothesis: Exact dispatch fingerprints plus controlled identity transforms can distinguish a route/adapter error from the expected BF16 accumulation-order difference in Transformer Engine's combine kernel.
- Commit Hashes:
  - `d208853243`, `df44f41344`, and `f211ba919b` add and harden the combine-isolation gate.
  - `df57afef9d` adds the scoped NCCL_EP Grug backend, four-group JaxPP bootstrap, task-local weight sharding, one-process-per-GPU launcher contract, and pinned Transformer Engine runtime setup.
- Command: RNO2A H100x8 job `/dlwh/ncclep-h100-combine-parity-r4-20260723-174749`, using Transformer Engine `4adad4c218c115cd9af235fb3d4e13ef4cec55a8`, NCCL `2.30.7`, JAX `0.10.1`, e64/EP8, d2560, 16,384 tokens/rank, and separate fresh process groups for top-k4 and top-k1.
- Results:
  - Iris succeeded in `9m14.15s`; both eight-rank groups exited `0`, with zero failed or preempted tasks and no live resources.
  - Dispatch counts, token-bit fingerprints, and routing-weighted token-bit fingerprints were exact for top-k4 and top-k1.
  - Top-k1 identity and scaled-identity combine were bitwise exact against both BF16 reference orders.
  - Top-k4 identity passed strict parity and exactly matched the reverse-order BF16 and FP32 references. Scaled identity also passed strict parity; its BF16 output was closest to FP32 accumulation at relative L2 `0.001981956`, maximum absolute error `0.0278320`, and at most one BF16 ULP versus that reference.
  - FP32 combine input is supported and passed strict parity, with relative L2 `0.000979293`, mean absolute error `0.000617520`, and maximum absolute error `0.0122070` against the FP32 reference.
  - The paired full routed-MLP result remains `22.73490ms` ring versus `15.65238ms` NCCL_EP median value-and-grad, a `1.452489x` speedup. Its only strict discrepancy is output: `0.203554%` of elements, relative L2 `0.00296233`, and maximum absolute error `0.0078125`; loss and all four gradient groups pass.
- Interpretation:
  - Route attachment and dispatch are correct. The top-k dependence, exact fingerprints, identity controls, and reference-order attribution localize the discrepancy to BF16 combine accumulation order.
  - The user explicitly accepted the observed approximately `0.2%` element-level output discrepancy under these circumstances. This approval is scoped to the NCCL_EP research backend; no global tolerance, loss check, or gradient check is weakened.
  - The performance gain and scoped numerical approval promote NCCL_EP to reduced JaxPP training. The integration creates four contiguous EP8 groups in a 32-process pipeline-major/expert-minor world and keeps expert weights partitioned over the `expert` mesh axis.
- Next action:
  - Run a four-stage L8/d2560/e64/top-k4/seq4096/b512/m16 explicit-MPMD `std_1f1b` smoke with CuTe FA4 and Pallas-Triton expert GEMMs. If finite and faster than the matched reduced ring result, scale to L24 and then the b8192/m256 capacity point.

### 2026-07-23 11:47 PDT - first NCCL_EP pipeline smoke finds worker setup activation bug
- Hypothesis: The pinned Transformer Engine setup and one-process-per-GPU launcher will reach NCCL_EP bootstrap on four RNO2A H100x8 workers.
- Commit Hash: `a3db0171e2` (`[grug] Activate worker venv for NCCL EP setup`) fixes the failure.
- Command: four-stage L8/d2560/e64/top-k4/seq4096/b512/m16 explicit-MPMD `std_1f1b`, CuTe FA4, Pallas-Triton `block_k=32`/8 warps, three steps, and preallocation `0.65`. Parent `/dlwh/iris-run-job-20260723-183643`; child `/dlwh/iris-run-job-20260723-183643/grug-train-jaxpp-rno2a-ncclep-smoke-l8-e64k4-b512-s4096-p4m16-20260723-1138`.
- Results:
  - The intended 32-process topology launched correctly, with task-local rank groups `0-7`, `8-15`, `16-23`, and `24-31`.
  - Iris custom setup ran outside `/app/.venv`. CUDA wheels were installed into that venv, but bare `python` in `cuda_wheels_env.sh` could not import `nvidia`, so `nvcc` was not found and Transformer Engine was not built.
  - The generated setup shell lacked fail-fast semantics and continued into rank launch; all ranks then failed with `ModuleNotFoundError: No module named 'transformer_engine'`.
  - Parent, child, and all tasks are terminal failed with no live resources. No NCCL_EP bootstrap, JaxPP compilation, training step, loss, or MFU result was produced.
- Interpretation:
  - This is a launcher setup failure, not a transport, topology, compiler, numerical, memory, or performance result. The rank layout validates the pipeline-major/expert-minor grouping assumption.
  - Commit `a3db0171e2` activates `$IRIS_VENV` and enables `set -euo pipefail` before CUDA/TE setup, making interpreter selection coherent and preventing a failed build from reaching rank launch.
- Next action:
  - Relaunch the identical reduced smoke from `a3db0171e2`; do not change the model or performance axes.

### 2026-07-23 11:56 PDT - corrected setup validates four EP8 groups and reaches JaxPP tracing
- Hypothesis: Activating the worker venv will build Transformer Engine consistently and advance the unchanged reduced smoke through multi-group NCCL_EP bootstrap.
- Commit Hash: `85600cf9d5` (`[grug] Trace NCCL EP under JaxPP abstract mesh`) fixes the tracing failure exposed by this run.
- Command: identical L8/d2560/e64/top-k4/seq4096/b512/m16 reduced smoke from the prior entry. Parent `/dlwh/iris-run-job-20260723-184218`; child `/dlwh/iris-run-job-20260723-184218/grug-train-jaxpp-rno2a-ncclep-smoke-r2-l8-e64k4-b512-s4096-p4m16-20260723-1150`.
- Results:
  - All four workers built Transformer Engine `2.19.0.dev0+4adad4c`, passed its NCCL_EP import probe, and completed all setup steps.
  - The 32-rank topology again mapped task-local groups to global ranks `0-7`, `8-15`, `16-23`, and `24-31`.
  - All ranks successfully bootstrapped four EP8 groups with `world=32`, `max_tokens_per_rank=16384`, and `recv_capacity_per_rank=81920`.
  - JaxPP then failed during `explicit_mpmd_train_step.lower()` in `jax.make_jaxpr`, before XLA compilation: `ValueError: Expected mesh of type jax.sharding.Mesh. Got jax._src.mesh.AbstractMesh` at the backend's inner `jax.set_mesh(mesh)`.
  - Parent, child, and all tasks are terminal failed with zero preemptions and no live resources. No training step, loss, MFU, or duration result was produced.
- Interpretation:
  - TE build, import, process ordering, communicator grouping, bootstrap capacity, and multi-group topology are validated. The failure is a redundant concrete-mesh context inside the backend, not an NCCL transport or JaxPP schedule failure.
  - JaxPP intentionally supplies an `AbstractMesh` while tracing stage-local programs. Commit `85600cf9d5` removes the inner `jax.set_mesh`; the outer runtime mesh and TE global shard guard remain authoritative.
- Next action:
  - Relaunch the unchanged reduced smoke from `85600cf9d5` and require it to pass JaxPR tracing before interpreting any later compiler or runtime result.

### 2026-07-23 12:56 PDT - NCCL_EP auto-axes tracing reaches nested expert shard map
- Hypothesis: Removing the redundant concrete mesh context will let JaxPP trace the Transformer Engine dispatch/combine body.
- Commit Hash: `741018a405` (`[grug] Match NCCL EP nested shard map context`) fixes the next tracing invariant.
- Command: unchanged reduced smoke. Parent `/dlwh/iris-run-job-20260723-185756`; child `/dlwh/iris-run-job-20260723-185756/grug-train-jaxpp-rno2a-ncclep-smoke-r3-l8-e64k4-b512-s4096-p4m16-20260723-1200`.
- Results:
  - TE build/import, 32-rank ordering, and four EP8 bootstraps all passed again with the expected `16384` tokens/rank and `81920` receive capacity.
  - The prior `jax.set_mesh(AbstractMesh)` failure is cleared.
  - JaxPP advanced through TE dispatch tracing into the nested local-expert `shard_map`, then failed before XLA compilation because `auto_axes` had installed an Auto-axis context while the nested map captured the original Explicit `AbstractMesh`.
  - All ranks reported the same mesh-context mismatch. Parent, child, and tasks are terminal failed with zero preemptions and no live resources. No training metric was produced.
- Interpretation:
  - TE custom-call tracing now advances beyond dispatch. The remaining mismatch is local to nested JAX sharding context, not communicator setup or transport semantics.
  - The validated one-node A/B already uses `jax.sharding.get_abstract_mesh()` inside the `auto_axes` body. Commit `741018a405` applies that same pattern so the nested expert FFN map uses the active Auto mesh rather than the captured Explicit mesh.
- Next action:
  - Relaunch the unchanged reduced smoke from `741018a405`; require JaxPR tracing to clear and capture the first lowering/compile/runtime result.

### 2026-07-23 13:21 PDT - JaxPP stage localization requires a one-group TE abstract view
- Hypothesis: Using the active Auto mesh in the nested expert map will complete the TE dispatch/local-FFN trace.
- Commit Hashes:
  - `ad8e7284c1` first applied TE's ordinary full-mesh compound receive spec.
  - `359a3effb9` supersedes that insufficient fix by separating the four physical communicators from each stage's one-group abstract JAX view.
- Command: unchanged reduced smoke. Parent `/dlwh/iris-run-job-20260723-195731`; child `/dlwh/iris-run-job-20260723-195731/grug-train-jaxpp-rno2a-ncclep-smoke-r4-l8-e64k4-b512-s4096-p4m16-20260723-1300`.
- Results:
  - TE setup, rank ordering, and four EP8 bootstraps passed again. The prior Explicit-versus-Auto nested-mesh mismatch is cleared.
  - Tracing reached the local FFN but saw `recv_tokens` shape `(4, 65536, 2560)` and failed when reshaping it to `(65536, 2560)`. No XLA compilation or training metric occurred.
  - The child failed after `9m08.13s`; the parent failed after `9m54.5s`. All tasks are terminal, with zero preemptions and no live resources.
- Interpretation:
  - TE caches `num_ep_groups=4` in a Python `EpConfig`, so its global dispatch abstract shape has leading extent `4 * 8 = 32`. JaxPP then localizes each stage onto an unstacked mesh where the `pipeline` axis is still named but has size 1. A compound `(pipeline, expert)` spec therefore divides by only eight and leaves four groups in each local shard.
  - Flattening those groups would be incorrect: token counts would contain 32 groups while each stage rank owns only eight local expert matrices.
  - The C++ bootstrap already configured each process with one physical EP8 communicator. At the pinned TE revision, the separate Python `EpConfig` controls abstract output dimensions, so `359a3effb9` changes only that tracing snapshot to `num_ep_groups=1` after physical bootstrap. It also pins stage-local inputs/receives/combines to the expert axis. The expected local FFN shapes are now receive `(1, 65536, 2560)`, counts `(1, 8)`, and eight expert matrices.
- Next action:
  - Relaunch the unchanged reduced smoke from `359a3effb9`. If tracing clears, capture the first XLA compile/runtime result before changing any capacity or performance axis.

### 2026-07-23 13:39 PDT - stage-local receive shape clears; TE backward needs an explicit spec
- Hypothesis: Separating TE's physical communicator config from its stage-local abstract config will clear the four-group receive shape and complete JaxPR construction.
- Commit Hash: `4bc4b059a7` (`[grug] Make NCCL EP dispatch backward sharding explicit`) fixes the next custom-VJP boundary.
- Command: unchanged reduced smoke. Parent `/dlwh/iris-run-job-20260723-202237`; child `/dlwh/iris-run-job-20260723-202237/grug-train-jaxpp-rno2a-ncclep-smoke-r5-l8-e64k4-b512-s4096-p4m16-20260723-1325`.
- Results:
  - TE setup and all four physical EP8 bootstraps passed. The prior `(4, 65536, 2560)` local receive reshape failure is cleared.
  - JaxPR construction advanced through forward dispatch, local expert FFN, and combine into TE's dispatch custom-VJP backward.
  - TE's public `_dispatch_bwd` then failed before XLA compilation while calling its private default output-spec helper: `AssertionError: Global mesh resource is not set`.
  - Child duration was `9m13.67s`; parent duration was `9m58.57s`. Parent, child, and all tasks are terminal with zero preemptions and no live resources. No loss or MFU metric was produced.
- Interpretation:
  - Physical topology and the stage-local receive layout are no longer blockers. The remaining error comes from TE's custom-VJP wrapper deriving backward sharding from ambient Python global state during JaxPP transposition.
  - `4bc4b059a7` retains TE's tested prepare/dispatch forward/backward primitives but wraps dispatch with an explicit custom VJP whose receive cotangents and returned token gradients use the stage-local expert-axis spec. It does not modify C++ kernels, communicator state, or numerical equations.
- Next action:
  - Relaunch the unchanged reduced smoke from `4bc4b059a7`. Require complete JaxPR construction and capture the first XLA partition/lower/compile/runtime result.

### 2026-07-23 14:00 PDT - full JaxPR clears; TE partition callbacks need the guard during compile
- Hypothesis: An explicit dispatch VJP will remove the final ambient TE dependency from JaxPR transposition and reach XLA compilation.
- Commit Hash: `c42fb432d6` (`[grug] Guard NCCL EP pipeline compilation`) fixes the compiler-callback lifecycle exposed by this run.
- Command: unchanged reduced smoke. Parent `/dlwh/iris-run-job-20260723-204042`; child `/dlwh/iris-run-job-20260723-204042/grug-train-jaxpp-rno2a-ncclep-smoke-r6-l8-e64k4-b512-s4096-p4m16-20260723-1345`.
- Results:
  - TE setup and four EP8 bootstraps passed. The explicit dispatch backward spec cleared the prior custom-VJP tracing failure.
  - The complete explicit-MPMD JaxPR lower succeeded for the first time.
  - Stage-local `eval_local()` compilation then failed in TE's `ep_prepare` custom partitioner because `_leading_axis_ok` called `global_mesh_resource()` after the earlier setup guard had exited.
  - Child duration was `13m30.09s`; parent duration was `14m34.49s`. All jobs/tasks are terminal with zero preemptions and no live resources. No training step or MFU metric was produced.
- Interpretation:
  - The forward and backward program is now representable as a JaxPP stage JaxPR. The failure is lifecycle-only: Grug intentionally exits the global mesh setup context before entering the pipeline loop, but the TE guard was coupled to that same context even though JaxPP invokes TE partition callbacks later during local compilation.
  - `c42fb432d6` makes a fresh TE guard context and enters it around explicit-MPMD train-step construction, lower, `eval_local()` compile, and execution. JaxPP remains outside the global Grug mesh as required.
- Next action:
  - Relaunch the unchanged reduced smoke from `c42fb432d6`. Require TE partitioning and stage-local XLA compilation to clear before interpreting runtime or performance.

### 2026-07-23 14:34 PDT - stage compilation clears; inconsistent NCCL_EP capacity fails first dispatch
- Hypothesis: Keeping the Transformer Engine shard guard active through lower, compile, and execute will let the reduced four-stage program enter its first forward microbatch.
- Commit Hash: `f038bdf9a1` (`[grug] Match NCCL EP bootstrap capacity to Grug`).
- Command: unchanged L8/d2560/e64/top-k4/seq4096/b512/m16 explicit-MPMD `std_1f1b` smoke. Parent `/dlwh/iris-run-job-20260723-210124`; child `/dlwh/iris-run-job-20260723-210124/grug-train-jaxpp-rno2a-ncclep-smoke-r7-l8-e64k4-b512-s4096-p4m16-20260723-1408`.
- Results:
  - TE setup, 32-rank ordering, four EP8 bootstraps, complete JaxPR construction, TE custom partitioning, and stage-0 XLA compilation all passed.
  - Execution reached `jit_grug_1f1b_mb0_stage0_forward`. Every stage-0 rank failed on its first TE dispatch custom call with `ep_backend.cpp:411 in dispatch: NCCL Error: invalid argument`; downstream ranks then blocked awaiting pipeline traffic.
  - The babysitter stopped the parent. Parent and child are terminal killed, with no live resources and no loss or MFU result.
- Interpretation:
  - Grug compiled each dispatch receive tensor with its model capacity factor `1.0`, or `65,536` rows/rank. Bootstrap incorrectly used Levanter's independent default `1.25`, registering `max_recv_tokens_per_rank=81,920`.
  - NCCL_EP's hybrid transport copies the full registered receive bound during CUDA graph capture and returns `ncclInvalidArgument` when the caller's receive token, weight, or index tensor has fewer than that many rows. The observed first-dispatch failure therefore follows directly from the compiled `65,536 < 81,920` contract violation.
  - The fix promotes Grug's existing `1.0` factor to one shared model constant and uses it for both model construction and NCCL_EP bootstrap. This preserves the matched ring model's routing capacity while making the registered and compiled receive bounds exactly `65,536`.
- Next action:
  - Relaunch the identical reduced smoke with matched bootstrap and model capacity. Require at least one finite training step before scaling or comparing performance.

### 2026-07-23 14:39 PDT - exact-average NCCL_EP capacity is too small for aligned live routing
- Hypothesis: Matching bootstrap and model receive capacity at Grug's existing factor `1.0` will satisfy the NCCL_EP fixed-shape contract and execute the first forward microbatch.
- Commit Hash: `df532f2e28` (`[grug] Reserve NCCL EP routing headroom`).
- Command: unchanged reduced smoke. Parent `/dlwh/iris-run-job-20260723-212224`; child `/dlwh/iris-run-job-20260723-212224/grug-train-jaxpp-rno2a-ncclep-smoke-r8-l8-e64k4-b512-s4096-p4m16-20260723-1424`.
- Results:
  - All 32 ranks bootstrapped four EP8 groups with the corrected matching `recv_capacity_per_rank=65,536`. Complete tracing/lowering passed, and stage-0 forward compilation began.
  - First dispatch failed with NCCL_EP's explicit device assertion: `padded EM slots 66368 > max_recv_tokens_per_rank 65536`.
  - The assertion poisoned the CUDA contexts; later launch, NCCL, and coordination errors are secondary. No loss, MFU, or duration metric was produced.
  - Parent, child, and all four tasks are terminal failed. No resources remain live.
- Interpretation:
  - Factor `1.0` is only the exact mean assignments/rank. Live learned routing plus NCCL_EP's expert-major alignment required 832 additional rows, or `1.27%` headroom, on this first microbatch.
  - The standalone transport and full-MLP gates already validated factor `1.25`. The next fix uses that bound for both NCCL_EP model buffers and bootstrap, while retaining Grug's existing factor `1.0` for ring and every other backend.
- Next action:
  - Relaunch the identical reduced smoke with the backend-scoped `1.25` receive bound. Require at least one finite training step before any performance comparison.

### 2026-07-23 14:55 PDT - NCCL_EP capacity clears and exposes local expert IDs before dispatch
- Hypothesis: A matching backend-scoped `1.25` receive bound will clear the first live dispatch and produce a finite reduced training step.
- Commit Hashes:
  - `df532f2e28` (`[grug] Reserve NCCL EP routing headroom`) was the code under test.
  - `460da06052` (`[grug] Preserve global expert IDs during routing`) fixes the route ownership failure exposed by the run.
- Command: unchanged L8/d2560/e64/top-k4/seq4096/b512/m16 explicit-MPMD `std_1f1b` smoke with CuTe FA4, Pallas-Triton `block_k=32`/8 warps, three steps, and preallocation `0.65`. Parent `/dlwh/iris-run-job-20260723-213929`; child `/dlwh/iris-run-job-20260723-213929/grug-train-jaxpp-rno2a-ncclep-smoke-r9-l8-e64k4-b512-s4096-p4m16-20260723-1442`.
- Results:
  - All 32 ranks bootstrapped four EP8 groups with the expected matching `recv_capacity_per_rank=81,920`.
  - Complete JaxPR construction, TE partitioning, lowering, stage-local compilation, and the first stage-0 forward dispatch all passed the prior shape and capacity checks.
  - Rank 0 then repeatedly reported `padded EM slots 524288 > max_recv_tokens_per_rank 81920`. The value is exactly the global microbatch assignment count: `131,072` tokens times top-k `4`.
  - The babysitter stopped the stalled parent. Parent, child, and all tasks are terminal with no live resources. No loss, MFU, duration, or W&B history was produced.
- Interpretation:
  - This is not a receive-capacity requirement: no rank can legitimately own every assignment in balanced e64/top-k4 routing, and increasing the bound to the global assignment count would only hide incorrect ownership.
  - The router score tensor's expert dimension was not explicitly replicated before `top_k`. Under JaxPP stage tracing, each EP rank selected among its local eight columns and emitted local indices `0..7`; NCCL_EP correctly interpreted those values as global expert IDs, so rank 0 claimed every assignment.
  - `460da06052` replicates the expert-score and bias dimensions before `top_k`, then explicitly shards selected global IDs and combine weights only over token axes. An independent EP8 CPU probe preserved winners `[63, 62, 61, 60]` through the same `auto_axes` boundary, and routing statistics remain correctly replicated/reduced.
- Next action:
  - Run the unchanged reduced smoke from merged commit `012111e22a` as r10. Require global expert ownership and at least one finite step before scaling to L24.

### 2026-07-23 20:50 PDT - r10 falsifies the router-score sharding diagnosis
- Hypothesis: Explicitly replicating all 64 router-score columns before top-k will preserve global expert IDs and let the unchanged reduced NCCL_EP pipeline smoke complete a finite step.
- Commit Hash: `012111e22a` (current `origin/main` merge plus `460da06052` global-route constraints).
- Command: unchanged L8/d2560/e64/top-k4/seq4096/b512/m16 explicit-MPMD `std_1f1b` smoke with four stages, 16 microbatches, CuTe FA4, Pallas-Triton `block_k=32`/8 warps, three steps, XLA preallocation `0.65`, and matching NCCL_EP receive capacity `81,920`. Parent `/dlwh/iris-run-job-20260724-025549`; child `/dlwh/iris-run-job-20260724-025549/grug-train-jaxpp-rno2a-ncclep-smoke-r10-l8-e64k4-b512-s4096-p4m16-20260723-1946`.
- Results:
  - All four workers completed setup, and all 32 ranks again bootstrapped four physical EP8 groups with `16,384` tokens/rank and `81,920` receive rows/rank.
  - JaxPP lowered and compiled all 30 distinct stage programs. The first stage-0 forward execution then reproduced `padded EM slots 524288 > max_recv_tokens_per_rank 81920` on task 0 local rank 0.
  - The assertion poisoned the CUDA context; the later launch, retry, telltale, and coordination failures are secondary. No finite step, loss, duration, MFU, or W&B history was produced. Parent, child, and all tasks are terminal with no live resources.
  - W&B: https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ncclep-smoke-r10-l8-e64k4-b512-s4096-p4m16-20260723-1946
- Interpretation:
  - The explicit router replication did not change the failure, so the r9 claim that JaxPP top-k emitted local expert IDs is not supported and is superseded by this result.
  - r8 is the critical control: with the same topology and model but a `65,536`-row receive shape, stage-0 local rank 3 reported a plausible `66,368` routed assignments. Changing the NCCL_EP receive shape to `81,920` was the only relevant transition before r9 began reporting the exact global assignment count on local rank 0.
  - Do not raise receive capacity to `524,288`; that would mask a TE/NCCL_EP prepare or stage-local partitioning defect and allocate an unjustified global-assignment buffer on every rank.
- Next action:
  - Add a bounded route fingerprint immediately before `ep_prepare` and rerun the unchanged shape once. Capture each local shard's route min/max and eight destination-rank counts to distinguish valid global routes from corruption inside TE/NCCL_EP. Preserve the backend-only accepted `0.203554%` forward mismatch; keep loss and gradient parity strict.

### 2026-07-23 21:32 PDT - r11b is infrastructure-invalid before NCCL_EP diagnostics
- Hypothesis: Route ownership and token-health fingerprints immediately before `ep_prepare` will distinguish a Transformer Engine prepare defect from corrupted output produced by an earlier NCCL_EP dispatch/combine.
- Commit Hash: `a8cc300327` (`[grug] Trace NCCL EP input health`).
- Command: unchanged L8/d2560/e64/top-k4/seq4096/b512/m16 explicit-MPMD `std_1f1b` NCCL_EP smoke with four stages, 16 microbatches, CuTe FA4, Pallas-Triton `block_k=32`/8 warps, three steps, and preallocation `0.65`. Parent `/dlwh/iris-run-job-20260724-035839`; child `/dlwh/iris-run-job-20260724-035839/grug-train-jaxpp-rno2a-ncclep-routefp-r11b-l8-e64k4-b512-s4096-p4m16-20260723-2058`.
- Results:
  - Attempt 0 failed before model execution when task 3 local rank 2 could not start its telltale server on port `39535`.
  - Attempts 1 and 2 failed before model execution when task 2/task 3 local rank 3 aborted in `CoordinationServiceAgent::Connect()` with exit `-6`; peers then exited from the failed distributed group.
  - The babysitter stopped attempt 3 during its repeated Transformer Engine rebuild. Parent, child, and all four tasks are terminal killed; counters are three failures and four preemptions, with no live resources.
  - The run emitted zero `NCCL_EP route fingerprint` lines, never reached `ep_prepare`, and produced no W&B run, loss, duration, throughput, or MFU.
- Interpretation:
  - This run contains no NCCL_EP correctness evidence. It neither supports nor falsifies the TE prepare defect and preceding-dispatch corruption hypotheses.
  - Preserve the bounded instrumentation for a clean allocation. Do not classify coordination bootstrap failures as transport failures or retry them through repeated expensive TE rebuilds.
- Next action:
  - Prioritize the now-unblocked FP8 expert path while retaining the NCCL_EP fingerprint probe for one later clean retry. Do not increase NCCL_EP capacity or weaken strict loss/gradient checks.

### 2026-07-23 21:48 PDT - CUTLASS type guard clears BF16 and FP8 full-block gates
- Hypothesis: CUTLASS DSL 4.5.2's `_isa` helper segfaults because it constructs MLIR wrapper types while probing a JaxPP-localized CuTe FA4 graph; using the `isinstance` test adopted by CUTLASS DSL 4.6 will remove that operation without changing the pinned FA4 or QuACK APIs.
- Commit Hashes:
  - `54242b3494` adds the minimal CUTLASS DSL 4.5.2 patch and updates the full-block reproducer commands.
  - `fe40556c56` copies the symlinked CUTLASS package into the worker venv before patching, preventing writes into UV's shared cache.
  - `d52c5b6e35` makes the helper portable to the minimal worker image.
  - `30183d2d4f` applies the validated helper in the production JaxPP setup.
- Commands:
  - Matched two-replica H100x8 BF16 and FP8 full-block commands in `experiments/grug/moe/repro_jaxpp_fp8_expert_compile.README.md`, using JAX/JAXLIB `0.10.1`, JaxPP `7091a9b5ce02cd1a6bdc905f6a36e89370a5fba9`, CUTLASS DSL `4.5.2`, CuTe FA4, learned QB routing, two layers/rank, 32,768 tokens, four microbatches, d2560/e64/top-k4, and `save_moe`.
  - BF16 job `/dlwh/jaxpp-full-block-cutlassisa-bf16-20260724-042818`.
  - Final FP8 job `/dlwh/jaxpp-full-block-cutlassisa-fp8-r3-20260724-044227`.
- Results:
  - BF16 succeeded 2/2. `jaxpp.lower()` returned in `2.3061s/2.3048s`; `lowered.eval_local()` returned in `11.4238s/109.7833s`; both ranks emitted pass verdicts. There was no segfault, traceback, OOM, or compiler exception.
  - FP8 succeeded 2/2. `jaxpp.lower()` returned in `4.3066s/4.2637s`; `lowered.eval_local()` returned in `16.6283s/123.4899s`; both ranks emitted pass verdicts. The 120-second watchdog showed active CUTLASS code generation on rank 1 rather than a hang.
  - The first two FP8 submissions were setup-only invalids: the direct patch mutated a symlinked UV cache entry, then the private-copy helper assumed `rg` existed in the worker image. Neither attempt reached external-worker bootstrap or FP8 compilation. The final helper is private-copying, idempotent, and uses `grep`.
  - The recurring FABRIC VMM `CUDA_ERROR_NOT_PERMITTED` warning fell back to simpler handle types and was nonfatal in both successful gates.
- Interpretation:
  - The exact unsafe MLIR type probe was causal for #7529's exit `139`. A broad CUTLASS 4.6/FA4/QuACK dependency migration is unnecessary for this experiment.
  - Both BF16 and FP8 full-block graphs now compile and execute under JaxPP localization, so FP8 expert GEMMs are unblocked for a real training comparison.
- Next action:
  - Babysit paired reduced parents `/dlwh/iris-run-job-20260724-044921` (BF16 ring) and `/dlwh/iris-run-job-20260724-044937` (FP8 expert), both L8/d2560/e64/top-k4/seq4096/b512/m16 for 20 steps. Scale only if the matched finite-step result credibly contributes the missing `9.5%` relative gain.

### 2026-07-23 22:12 PDT - reduced FP8 expert training is stable but only 1.74% faster
- Hypothesis: The direct expert-kernel gain will survive attention, routing, optimizer, and pipeline overhead strongly enough to contribute materially toward the missing `9.5%` relative MFU gain.
- Commit Hash: `34dd6b718f` (`[docs] Record Cutlass full-block recovery`), with FP8 expert training support from `84fc011f7d` and the CUTLASS guard from `30183d2d4f`.
- Commands:
  - Matched launcher configuration: explicit-MPMD `std_1f1b`, L8/d2560/e64/top-k4/seq4096/b512, four physical/logical stages, 16 microbatches, ring EP8, CuTe FA4, Pallas-Triton ragged dot with `block_k=32` and eight warps, `save_moe`, 20 steps, and preallocation `0.65`.
  - BF16 control parent `/dlwh/iris-run-job-20260724-044921`; child `/dlwh/iris-run-job-20260724-044921/grug-train-jaxpp-rno2a-cutlassisa-ring-ab5-l8-e64k4-b512-s4096-p4m16-20260723-2150`.
  - FP8 expert parent `/dlwh/iris-run-job-20260724-044937`; child `/dlwh/iris-run-job-20260724-044937/grug-train-jaxpp-rno2a-cutlassisa-fp8expert-ab5-l8-e64k4-b512-s4096-p4m16-20260723-2150`.
- Results:
  - Both runs succeeded on all four tasks and produced 18 usable timed rows at steps 2-19.
  - BF16 mean MFU was `16.15320381`, p10/p50/p90 `16.04736573/16.16456760/16.20825403`, with final duration `1.966142455s`, throughput `1,066,632.78` tokens/s, and loss `6.459357262`. W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-cutlassisa-ring-ab5-l8-e64k4-b512-s4096-p4m16-20260723-2150>.
  - FP8 mean MFU was `16.43438485`, p10/p50/p90 `16.37141857/16.43360554/16.49203307`, with mean duration `1.92009168s`, mean throughput `1,092,220.97` tokens/s, final duration `1.91554455s`, and final loss `6.473066`. W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-cutlassisa-fp8expert-ab5-l8-e64k4-b512-s4096-p4m16-20260723-2150>.
  - FP8 improved mean MFU by `0.28118104` points or `1.7407%`, reduced mean duration by `1.6853%`, and improved mean throughput by `1.7385%`.
  - All reported losses were finite. Matched-step loss RMSE was `0.01384159`; final absolute loss delta was `+0.013709`, or `+0.2122%` relative. The user accepted this approximately `0.2%` error for this experiment.
- Interpretation:
  - The CUTLASS guard and FP8 overwrite-state path are validated in real reduced training. The local expert-kernel speedup is mostly hidden by the rest of the pipeline step.
  - A `1.7407%` gain applied to the `18.2583` headline best projects about `18.58` MFU, still well below 20. This is a clean directional positive and a target-level negative.
- Next action:
  - Do not scale unchanged FP8 expert GEMMs to L24. Retry the reduced NCCL_EP pipeline with the existing bounded route/token fingerprint instrumentation on a clean allocation; its direct routed-MLP gain remains the only current result large enough to plausibly close the target gap.

### 2026-07-23 23:02 PDT - r11c exposes Transformer Engine build skew before model startup
- Hypothesis: A clean RNO2A allocation will reach the bounded NCCL_EP route/token fingerprints that r11b missed.
- Commit Hash: `bbb66af725` (`[docs] Record reduced FP8 expert comparison`).
- Command: `experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --cluster cw-rno2a --run-id jaxpp-rno2a-ncclep-routefp-r11c-l8-e64k4-b512-s4096-p4m16-20260723-2204 --schedule std_1f1b --implementation explicit_mpmd --physical-stages 4 --logical-stages 4 --stage-layer-counts 2,2,2,2 --microbatches 16 --nodes 4 --gpus-per-replica 8 --expert-axis 8 --layers 8 --experts 64 --top-k 4 --batch 512 --seq-len 4096 --moe-implementation nccl_ep --attention-implementation gpu_fa4_cute --ragged-dot-implementation triton --ragged-dot-block-k 32 --ragged-dot-num-warps 8 --xla-memory-fraction 0.65 --remat save_moe --steps 3 --tracker wandb`.
- Results:
  - Parent `/dlwh/iris-run-job-20260724-050423`; child `/dlwh/iris-run-job-20260724-050423/grug-train-jaxpp-rno2a-ncclep-routefp-r11c-l8-e64k4-b512-s4096-p4m16-20260723-2204`.
  - All four tasks built Transformer Engine `2.19.0.dev0+4adad4c` independently. Task 0 completed its wheel and launched its eight ranks at `05:13:56 UTC`; the gang failed at `06:01:42-06:01:48 UTC` when late ranks aborted in `CoordinationServiceAgent::Connect()`.
  - The child ended `worker_failed` with one direct failure and three coscheduled failures; the parent ended failed. All jobs are terminal and no resources remain live.
  - The run emitted zero `NCCL_EP route fingerprint`, `ep_prepare`, or padded-EM lines and produced no W&B history, loss, duration, throughput, or MFU.
- Interpretation:
  - This is the second infrastructure-invalid fingerprint retry. It contains no evidence about route ownership, token health, Transformer Engine prepare, or NCCL_EP transport correctness.
  - The per-node source build is allowed to finish at different times, but JAX's Iris bootstrap used a fixed 1,800-second initialization timeout. The fastest node entered the 32-rank barrier while slower nodes were still building, so the coordinator expired before the complete gang joined.
  - JAX 0.11 does not provide a useful fallback on RNO2A: native ragged all-to-all now requires symmetric/exportable memory, and this cluster rejects the required FABRIC allocation with `CUDA_ERROR_NOT_PERMITTED`. NCCL EP v0.1.0 plus Transformer Engine's merged-but-unreleased JAX binding remains the only released transport with a measured target-shape gain; the branch already pins that binding.
- Next action:
  - Make Iris's JAX distributed initialization timeout explicitly configurable and set this launcher to two hours. Validate locally, commit the startup fix, then run one final unchanged r11 fingerprint attempt before modifying model or transport code.

### 2026-07-24 11:58 PDT - r11d validates the longer JAX timeout and exposes a telltale port race
- Hypothesis: A two-hour distributed-initialization timeout will absorb per-node Transformer Engine build skew and let all 32 ranks reach the unchanged NCCL_EP route fingerprint diagnostic.
- Commit Hash: `6cd8d90938` (`[iris] Allow slow distributed JAX bootstrap`).
- Command: unchanged L8/d2560/e64/top-k4/seq4096/b512/m16 explicit-MPMD `std_1f1b` diagnostic with four stages, 16 microbatches, NCCL_EP, CuTe FA4, Pallas-Triton `block_k=32`/8 warps, preallocation `0.65`, three steps, and `--jax-init-timeout 7200`. Parent `/dlwh/iris-run-job-20260724-184256`; child `/dlwh/iris-run-job-20260724-184256/grug-train-jaxpp-rno2a-ncclep-routefp-r11d-l8-e64k4-b512-s4096-p4m16-20260724-1142`.
- Results:
  - All four workers completed Transformer Engine setup and launched ranks. Every observed rank logged `JAX_DISTRIBUTED_INITIALIZATION_TIMEOUT=7200`; peers reached supervised JAX initialization instead of reproducing r11c's 1,800-second coordinator expiry.
  - On task 1, local ranks 1 and 7 both selected telltale port `39535`. Rank 1 timed out after five seconds with `TimeoutError: telltale server did not start on port 39535`, exited `1`, and caused the supervisor to terminate its peers before distributed initialization completed.
  - The child is terminal killed with all four tasks complete, one failure, and four preemptions. The parent is terminal killed. Prefix inspection shows no running or pending resources.
  - The run emitted zero route fingerprints and never reached `ep_prepare`, JaxPP lowering, compilation, training, W&B history, loss, duration, throughput, or MFU.
- Interpretation:
  - The configurable timeout fix is validated at rank launch, but r11d remains infrastructure-invalid and contains no NCCL_EP correctness or performance evidence.
  - Telltale used `find_free_port()` to probe and release an ephemeral port before Uvicorn bound it. Concurrent supervised ranks could therefore select the same candidate. This is a startup race independent of NCCL_EP, model topology, capacity, or numerical behavior.
  - The fix makes Uvicorn bind port `0` directly and registers the kernel-assigned port from its live listener, eliminating the probe/release window.
- Next action:
  - Validate and snapshot the telltale fix, then launch one unchanged r11e diagnostic. Preserve the NCCL_EP-only accepted `0.203554%` top-k output mismatch; keep loss and all gradient groups strict.

### 2026-07-24 12:20 PDT - r11e clears startup and exposes JaxPP's four-minute DIME timeout
- Hypothesis: Atomic telltale binding and the two-hour JAX initialization timeout will let all 32 ranks reach the bounded NCCL_EP route/token fingerprint.
- Commit Hashes:
  - `80b0813105` makes telltale port binding atomic.
  - `827062a3a5` exposes JaxPP's coordination-client timeout and defaults it to two hours for this launcher.
- Command: unchanged L8/d2560/e64/top-k4/seq4096/b512/m16 explicit-MPMD `std_1f1b` diagnostic with four stages, 16 microbatches, NCCL_EP, CuTe FA4, Pallas-Triton `block_k=32`/8 warps, preallocation `0.65`, three steps, and `--jax-init-timeout 7200`. Parent `/dlwh/iris-run-job-20260724-185905`; child `/dlwh/iris-run-job-20260724-185905/grug-train-jaxpp-rno2a-ncclep-routefp-r11e-l8-e64k4-b512-s4096-p4m16-20260724-1159`.
- Results:
  - All 32 ranks registered unique telltale endpoints, initialized distributed JAX with timeout `7200`, and bootstrapped four physical NCCL_EP groups with `16,384` tokens/rank and `81,920` receive rows/rank.
  - At `19:09:28 UTC`, all task-0 ranks entered compilation of `grug_1f1b_mb0_stage0_forward`. Tasks 1-3 entered JaxPP `eval_local` and requested their DIME2 communicators, but task 0 did not finish compilation before the downstream rendezvous deadline.
  - At `19:13:29 UTC`, every downstream rank failed in `jaxpp/dime2.py:get_nccl_id` after exactly four minutes. Task 1 timed out on keys `0,8` through `7,15`; task 2 on `8,16` through `15,23`; task 3 on `16,24` through `23,31`.
  - Pinned JaxPP defines `JAXPP_CLIENT_TIMEOUT=240000` milliseconds. The launcher now validates and exports an explicit `--jaxpp-client-timeout-ms`, defaulting to `7,200,000` milliseconds.
  - No route/token fingerprint, `ep_prepare`, padded-EM assertion, finite step, loss, gradient, duration, throughput, or MFU was reached. W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ncclep-routefp-r11e-l8-e64k4-b512-s4096-p4m16-20260724-1159>.
  - The babysitter stopped only the failed parent. Parent and child are terminal killed; the prefix contains no live resources.
- Interpretation:
  - The startup fixes are validated. r11e is still infrastructure/runtime-invalid for NCCL_EP correctness because JaxPP's independent DIME rendezvous timeout expired during first-stage compilation.
  - Increasing the DIME wait changes no model, transport, capacity, or numerical behavior. It is a bounded correction to match the already-required two-hour compilation/startup budget.
  - The accepted approximately `0.2%` NCCL_EP output discrepancy was not exercised; loss and all gradient-group checks remain strict.
- Next action:
  - Run one final unchanged r11f fingerprint diagnostic with `JAXPP_CLIENT_TIMEOUT=7200000`. Do not increase NCCL_EP receive capacity or modify the model until the pre-`ep_prepare` fingerprints are captured.

### 2026-07-24 12:20 PDT - XLA device-initiated ragged all-to-all landed but is not yet consumable
- Hypothesis: A newly released JAX/XLA device-initiated ragged all-to-all path may replace the measured slow private-memory fallback and remove NCCL_EP as the remaining transport dependency.
- Evidence:
  - OpenXLA commit [`acb5aaffe4c0`](https://github.com/openxla/xla/commit/acb5aaffe4c0d844bacb57ad85234422f0ceaae0), from [PR #41903](https://github.com/openxla/xla/pull/41903), adds a single CUDA kernel using LSA for local peers and GIN for remote peers. It is opt-in with `--xla_gpu_experimental_ragged_all_to_all_use_device_kernel=true`.
  - The path also requires symmetric mode and symmetric buffers for `RaggedAllToAll`; it requests symmetric input and output windows. NCCL `>=2.29.0` is required, with full remote GIN support effectively requiring `>=2.29.7`.
  - The latest coherent public nightly, `0.11.1.dev20260724`, embeds JAX commit [`30a436d63e89`](https://github.com/jax-ml/jax/commit/30a436d63e895930cdefe4a87475502b9d6a5e49), which pins XLA [`b0512ae684f0`](https://github.com/openxla/xla/commit/b0512ae684f0a7d7c6e1e12d9ca3a9ba3bf9f822), 12 commits before the device kernel. JAX main advanced past it in [`36e7372f1318`](https://github.com/jax-ml/jax/commit/36e7372f13180bdae9bef8be4bf63176dfc236e0), so `20260725` or later is the first plausible nightly.
- Interpretation:
  - No released or nightly JAX artifact currently contains the new kernel. Upstream reports DeepSeek-v3 functional testing but no performance result.
  - The kernel does not eliminate RNO2A's main risk: symmetric-window registration remains mandatory, and prior FABRIC/POSIX-FD allocation on this cluster failed with `CUDA_ERROR_NOT_PERMITTED`. EP8 is node-local and should use LSA with `gin=0`, so the newer allocation fallback may still succeed; this is unproven.
- Next action:
  - When a nightly pins XLA at or after `acb5aaffe4c0`, first run a one-node H100x8 direct `jax.lax.ragged_all_to_all` gate at the exact EP8 microbatch geometry. Require output parity, `Device kernel: lsa_size=8 num_ranks=8 gin=0`, successful symmetric registration, and roughly 10% improvement over private-memory mode before testing reduced JaxPP or L24.

### 2026-07-24 12:35 PDT - exact-shape private ragged all-to-all baseline is bitwise correct
- Hypothesis: A standalone one-node H100x8 benchmark at the exact target EP8 microbatch transport shape will provide a stable correctness and timing baseline for the forthcoming XLA device-kernel path.
- Commit Hash: `b4b0d3add8` (`[grug] Add XLA ragged all-to-all benchmark`).
- Command: one H100x8 task on `cw-rno2a` running `experiments/grug/moe/benchmark_xla_ragged_all_to_all.py --assignments-per-rank 65536 --hidden-dim 2560 --warmup 5 --iterations 30` with `XLA_PYTHON_CLIENT_MEM_FRACTION=.50`. Job `/dlwh/xla-ragged-a2a-private-ep8-20260724-1232`.
- Results:
  - The job succeeded in `16.76s` with JAX/JAXLIB `0.10.1`, default `XLA_FLAGS=""`, and eight H100 80GB devices.
  - The BF16 output was bitwise exact: `mismatch_count=0`, checksum `3623878656`.
  - Transport timing for `65,536` assignments/rank, d2560, and `335,544,320` payload bytes/rank was mean `16.954763ms`, median `16.952521ms`, min `16.909279ms`, and max `17.007264ms`.
  - FABRIC+POSIX_FD allocation was not permitted, but XLA's private path retried with simpler handle types and completed. No failed resource remains live.
- Interpretation:
  - The benchmark isolates transport from input construction and supplies a bitwise-correct baseline for the future device-kernel A/B.
  - The user accepts approximately `0.2%` numerical error for the explicitly approximate QuACK, NCCL_EP, and FP8 research comparisons under discussion. This does not relax loss or gradient checks, and this exact XLA transport gate remains bitwise because it already passes at zero error.
- Next action:
  - Wait for a public JAX nightly whose pinned XLA includes `acb5aaffe4c0`, then run the identical benchmark with the required symmetric/device-kernel flags. Require `gin=0` LSA evidence, successful symmetric registration, and about 10% speedup before pipeline integration.

### 2026-07-24 12:46 PDT - automatic standard schedule clears input placement and exposes a free task variable
- Hypothesis: Localizing automatic JaxPP's compiled input shardings to the stage-local lowering mesh will clear the non-addressable global-mesh `device_put` failure and execute the tiny standard schedule.
- Commit Hash: `08f169803e` (`[grug] Localize automatic JaxPP inputs`).
- Command: automatic `std_1f1b`, L4/d2560/e8/top-k1/seq128/b32/m4, four H100x8 stages, ring MoE, two steps, and `GRUG_JAXPP_AUTO_EXPLICIT_IN_SHARDINGS=1`, `GRUG_JAXPP_PATCH_CONST_SHARDINGS=1`, `JAXPP_CONSERVATIVE_LOOP_CLUSTERING=false`. Parent `/dlwh/iris-run-job-20260724-193847`; child `/dlwh/iris-run-job-20260724-193847/grug-train-jaxpp-auto-localinput-std-l4-e8-b32-s128-r1-20260724-1245`.
- Results:
  - The new focused regression test passes both the non-addressable global-mesh case and the single-process no-op case. The patch preserves each input's `PartitionSpec` and memory kind while replacing only its device mesh with `mpmd_mesh.lowering_mesh()`.
  - All four ranks initialized and compiled their `before_loop_0_<rank>` tasks. The run did not reproduce the prior non-addressable input-sharding `device_put` failure.
  - Rank 0 then failed while compiling `fwd_0` in `jaxpp/jax_primitives.py:task_impl` with `KeyError: Var(...):bfloat16[1024@(replica_dcn,data,expert),2560]`; downstream ranks waited in DIME communicator creation.
  - No finite loss, duration, throughput, or MFU was produced. Parent, child, and all tasks are terminal with no live resources.
- Interpretation:
  - Input localization is validated and is no longer the automatic-schedule blocker.
  - Pinned JaxPP appends all unclustered loop equations to the last task when conservative clustering is disabled. The resulting stage task contains a free variable that is absent from its task environment; task-Jaxpr checking is disabled by default, so the defect appears later as a compile-time `KeyError`.
- Next action:
  - Run the matched tiny standard schedule with `JAXPP_CONSERVATIVE_LOOP_CLUSTERING=true` to capture the earlier transformation-time assertion and exact unclustered equation/Jaxpr. Do not test eager or zero-bubble schedules until standard executes.

### 2026-07-24 12:57 PDT - r11f localizes explicit MPMD compilation to XLA's sharded autotuner
- Hypothesis: Extending JaxPP's DIME client timeout to two hours will let the unchanged reduced NCCL_EP fingerprint diagnostic complete first-stage compilation.
- Commit Hash: `db66b180ab` (`[docs] Record JaxPP DIME timeout and XLA transport state`), with runtime fixes through `827062a3a5`.
- Command: unchanged L8/d2560/e64/top-k4/seq4096/b512/m16 explicit-MPMD `std_1f1b` NCCL_EP diagnostic with four stages, 16 microbatches, CuTe FA4, Pallas-Triton `block_k=32`/8 warps, preallocation `0.65`, three steps, `JAX_DISTRIBUTED_INITIALIZATION_TIMEOUT=7200`, and `JAXPP_CLIENT_TIMEOUT=7200000`. Parent `/dlwh/iris-run-job-20260724-192145`; child `/dlwh/iris-run-job-20260724-192145/grug-train-jaxpp-rno2a-ncclep-routefp-r11f-l8-e64k4-b512-s4096-p4m16-20260724-1221`.
- Results:
  - All 32 ranks registered unique telltales, initialized JAX, and bootstrapped four NCCL_EP groups. Stage 0 entered `grug_1f1b_mb0_stage0_forward` compilation at `19:32:13 UTC`; stages 1-3 entered DIME rendezvous one second later.
  - Native `py-spy` stacks on all eight stage-0 processes converged on `xla::Autotuner::Autotune -> DistributedKeyValueStore::Get -> CoordinationServiceAgent::GetKeyValue -> BlockingKeyValueGet`. Downstream rank 8 was independently blocked in `jaxpp.dime2.get_nccl_id -> BlockingKeyValueGet`. CPU counters were effectively idle.
  - OpenXLA's `xla_gpu_shard_autotuning` defaults to true. Its `ConfigAssigner` fingerprints the complete HLO module, partitions candidates across all participating compiler processes, publishes one result key per process, then waits up to 24 hours for every other process's module-fingerprint key. JaxPP MPMD stages compile different HLO modules, so those keys cannot match.
  - No route fingerprint, `ep_prepare`, padded-EM result, finite loss, duration, throughput, or MFU was reached. The W&B run has no training history: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ncclep-routefp-r11f-l8-e64k4-b512-s4096-p4m16-20260724-1221>.
  - The parent and child are terminal killed, all four child tasks exited `0`, and no matching pods remain.
- Interpretation:
  - r11f was not slow compilation. It was a deterministic contract mismatch between XLA's cross-process sharded autotuner and JaxPP's MPMD compilation model.
  - `--xla_gpu_shard_autotuning=false` is the narrow control: it preserves normal local online autotuning and disables only the cross-process partition/join layer. Disabling all autotuning with `--xla_gpu_autotune_level=0` is unnecessary for this diagnosis and would confound performance.
  - The accepted `0.2%` ceiling for approximate NCCL_EP comparisons was not exercised; exact loss and gradient checks remain required.
- Next action:
  - Babysit r11g parent `/dlwh/iris-run-job-20260724-195722`, which changes only `XLA_FLAGS` to include `--xla_gpu_shard_autotuning=false`. Require stage compilation to clear, capture the pre-`ep_prepare` fingerprints and padded-EM result, and produce finite training metrics before any L24 comparison.

### 2026-07-24 13:10 PDT - exact deployed XLA source confirms the MPMD autotuning deadlock
- Hypothesis: The sharded-autotuner behavior observed in r11f is present in the exact OpenXLA revision embedded by deployed JAX/JAXLIB `0.10.1`, rather than only current OpenXLA main.
- Evidence:
  - JAX tag `jax-v0.10.1` at `619764c15117fbefc4ba13ab941871cb514c23f6` pins OpenXLA commit [`9b635916ecc6`](https://github.com/openxla/xla/commit/9b635916ecc6df6efee62d8e4b0c7ef87ef84d69).
  - In that exact OpenXLA revision, `DebugOptions` initializes `xla_gpu_shard_autotuning=true`, and `AutotunerPass::RunImpl` enters the multiprocess overload whenever the flag is enabled and `process_count > 1`.
  - The multiprocess `Autotuner::Autotune` implementation builds each key from the complete HLO module fingerprint, backend fingerprint, and shard index. It publishes its local key and calls `kv_store.Get(..., absl::Hours(24))` for every other process's key.
- Result:
  - The deployed source behavior exactly matches the r11f native stacks and idle counters. Different JaxPP MPMD stages compile different module fingerprints while sharing one compiler process group, so no stage can publish the keys expected by the others.
  - The r11g control remains narrowly scoped: `--xla_gpu_shard_autotuning=false` bypasses only the multiprocess partition/join overload and preserves local online autotuning.
- Interpretation:
  - Confidence is high that r11f is a deterministic OpenXLA/JaxPP MPMD contract mismatch, not merely slow compilation.
  - No numerical acceptance policy is involved in this fix.
- Next action:
  - Require r11g to clear first-stage compilation before attributing any later failure to NCCL_EP routing or transport.

### 2026-07-24 13:10 PDT - conservative automatic clustering exposes disconnected router-bias gradients
- Hypothesis: Enabling JaxPP's conservative loop clustering on the tiny automatic standard schedule will reject the exact equations that become a free task variable when the fallback appends them to the final task.
- Commit Hash: `08f169803e` (`[grug] Localize automatic JaxPP inputs`).
- Command: matched automatic `std_1f1b`, L4/d2560/e8/top-k1/seq128/b32/m4, four H100x8 stages, ring MoE, two steps, with `JAXPP_CONSERVATIVE_LOOP_CLUSTERING=true`. Parent `/dlwh/iris-run-job-20260724-194656`; child `/dlwh/iris-run-job-20260724-194656/grug-train-jaxpp-auto-localinput-std-conservative-l4-e8-b32-s128-r2-20260724-1248`.
- Results:
  - All workers completed setup. Rank 0 completed both tracing passes and reached JaxPP loop-body clustering before XLA compilation.
  - Conservative clustering failed in `jaxpp/core.py:1149` with `AssertionError: Failed on loop body jaxpr`. The complete unclustered tail was four scalar-zero `broadcast_in_dim` equations producing `f32[8]`, followed by four `add` equations from JaxPP's additive tree reduction.
  - These four leaves are the L4 model's router-bias gradients. `_apply_qb_betas` replaces each trainable router bias with its pending QB-derived value inside the differentiated loss, disconnecting the original leaves and making their gradients exact constant zeros with no pipeline marker ownership.
  - No loss, duration, throughput, or MFU was produced. Parent and child are terminal failed with no running or pending resources.
- Interpretation:
  - With conservative clustering disabled, JaxPP appends the disconnected tail to the last task and later fails on a free variable; with it enabled, the same defect is reported at transformation time. Input placement is no longer the blocker.
  - The scoped fix removes router-bias leaves from the differentiated pipeline tree, reconstructs the fixed QB biases inside the microbatch loss, and restores exact zero router-bias gradients after `treduce`. It must preserve forward values and every non-router gradient exactly.
- Next action:
  - Validate the focused CPU regression test, then launch the smallest matched automatic `std_1f1b` GPU gate. Do not test eager or zero-bubble until standard produces finite training metrics.

### 2026-07-24 13:22 PDT - router-bias fix exposes a separate free stage-0 activation
- Hypothesis: Removing QB-managed router-bias leaves from automatic `treduce` differentiation will clear conservative loop clustering and let the reduced automatic standard schedule compile and execute.
- Commit Hash: `e40085a8b4` (`[grug] Isolate automatic JaxPP router biases`).
- Command: automatic `std_1f1b`, L4/d2560/e8/top-k1/seq128/b32/m4, four H100x8 stages, ring MoE, two steps, conservative clustering enabled, stage-localized explicit input shardings, the const-sharding patch, and `--xla_gpu_shard_autotuning=false`. Parent `/dlwh/iris-run-job-20260724-201419`; child `/dlwh/iris-run-job-20260724-201419/grug-train-jaxpp-auto-routerbias-std-l4-e8-b32-s128-r3-20260724-1314`.
- Results:
  - Local exact-value and gradient contracts passed `3/3`, and changed-file precommit including Pyrefly passed before launch.
  - Both JaxPP tracing passes completed. Conservative loop clustering passed, proving the four disconnected router-bias zero reductions were removed. The `before_loop` output was `4.18 GiB`; all four ranks began compiling their `before_loop_0_<rank>` tasks.
  - Rank 0 then failed compiling `fwd_0` with `KeyError: Var(...):bfloat16[1024@(replica_dcn,data,expert),2560]` from `jaxpp/jax_primitives.py:callable_task`. The shape is the first-stage microbatch hidden activation: `(batch 32 / microbatches 4) * sequence 128 = 1,024` tokens by hidden size `2,560`.
  - The activation is produced by token embedding plus embedding norms before `block_range` and consumed by the first transformer block. JaxPP's generated task Jaxpr does not include it as a task input.
  - No loss, duration, throughput, or MFU was produced. W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-auto-routerbias-std-l4-e8-b32-s128-r3-20260724-1314>.
  - The babysitter stopped only this parent. Parent and child are terminal killed, all four child tasks exited `0`, and no matching Kubernetes resources remain.
- Interpretation:
  - The router-bias change is validated for its scoped ownership defect, but automatic standard scheduling has a second independent JaxPP task-construction bug.
  - Automatic eager and zero-bubble use the same transformed forward tasks, so launching them before this ownership bug is fixed would not test schedule behavior.
  - Explicit GPipe and interleaved GPipe are already measured negatives. NCCL_EP remains the only active mechanism with enough direct routed-MLP gain to plausibly move the `18.2583` target baseline past 20 MFU.
- Next action:
  - Keep automatic schedules blocked and finish the reduced NCCL_EP r11g diagnostic. Scale NCCL_EP to L24 only after a finite reduced training step with bounded route fingerprints and strict loss/gradient checks.

### 2026-07-24 13:38 PDT - r11g localizes NCCL_EP corruption to TE's outer partitioning boundary
- Hypothesis: Disabling XLA's cross-process sharded autotuner will clear the MPMD compilation deadlock and expose whether NCCL_EP receives rank-local routes and token buffers.
- Commit Hash: `249ed44fa4` (`[docs] Record automatic JaxPP activation blocker`).
- Command: `XLA_FLAGS='--xla_gpu_nccl_termination_timeout_seconds=600 --xla_gpu_shard_autotuning=false' experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --cluster cw-rno2a --run-id jaxpp-rno2a-ncclep-routefp-r11g-noshardat-l8-e64k4-b512-s4096-p4m16-20260724-1258 --schedule std_1f1b --implementation explicit_mpmd --physical-stages 4 --logical-stages 4 --stage-layer-counts 2,2,2,2 --microbatches 16 --nodes 4 --gpus-per-replica 8 --expert-axis 8 --layers 8 --experts 64 --top-k 4 --batch 512 --seq-len 4096 --moe-implementation nccl_ep --attention-implementation gpu_fa4_cute --ragged-dot-implementation triton --ragged-dot-block-k 32 --ragged-dot-num-warps 8 --xla-memory-fraction 0.65 --remat save_moe --steps 3 --tracker wandb --jax-init-timeout 7200 --jaxpp-client-timeout-ms 7200000`.
- Results:
  - Parent `/dlwh/iris-run-job-20260724-195722`; child `/dlwh/iris-run-job-20260724-195722/grug-train-jaxpp-rno2a-ncclep-routefp-r11g-noshardat-l8-e64k4-b512-s4096-p4m16-20260724-1258`.
  - All 32 ranks initialized, all four stages compiled, and the r11f sharded-autotuner KV-store deadlock did not recur.
  - Every rank's last pre-`ep_prepare` fingerprint reported routes with local shape `(16384, 4)`, expert IDs spanning `0..63`, destination counts of roughly `7,800..8,600` assignments, and exactly `41,943,040/41,943,040` finite BF16 token values. Each destination-count vector summed to `65,536`, the expected local token count times top-k four.
  - Both Iris attempts failed at the same first causal assertion on task 0 rank 0: `scan_impl_flat(em): padded EM slots 524288 > max_recv_tokens_per_rank 81920`. The later `CUDA error 719: Failed to gpuMemcpyAsync` and coordination socket closures were cascading failures.
  - `524,288` is exactly the global microbatch assignment count: `131,072` global tokens times top-k four. It is not a plausible local destination load. Transformer Engine's outer custom-partitioning primitive retained the global static token extent while its FFI received a physical 16,384-row local buffer, so the scan read beyond the local route buffer.
  - No loss, duration, throughput, MFU, or gradient metric was produced. W&B has no training history: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ncclep-routefp-r11g-noshardat-l8-e64k4-b512-s4096-p4m16-20260724-1258>.
  - Iris began a third automatic attempt after the repeated failure. The babysitter stopped only the parent. Parent, child, and all tasks are terminal killed; no matching pods remain.
- Interpretation:
  - The route generator and JaxPP input sharding are exonerated at the immediate transport boundary. Raising receive capacity to `524,288` would allocate the global assignment count on every rank and mask the out-of-bounds read.
  - Commit `35f766630c` binds Transformer Engine's registered inner prepare, dispatch, combine, and backward FFI primitives inside an explicit expert-axis `shard_map`. Forward and backward abstract-shape probes map global `(131072, ...)` inputs to local `(16384, ...)` FFI operands and reconstruct global outputs. Focused tests and changed-file precommit pass.
  - This fix preserves Transformer Engine's NCCL kernels, 81,920-row receive capacity, and custom gradients. It changes only the custom-partitioning boundary that supplied the wrong static extent.
- Next action:
  - Babysit r11h parent `/dlwh/iris-run-job-20260724-203827`, which runs the unchanged reduced configuration from `35f766630c`. Require finite loss and gradients before launching the L24 target.

### 2026-07-24 13:55 PDT - r11h finds a trace-only backward pytree mismatch
- Hypothesis: Explicit expert-axis `shard_map` calls to Transformer Engine's inner primitives will present rank-local token extents to every NCCL_EP FFI.
- Commit Hash: `35f766630c` (`[levanter] Pin NCCL EP FFI to local shards`).
- Command: matched r11g L8/d2560/e64/top-k4/seq4096/b512/m16 explicit-MPMD `std_1f1b` gate with run ID `jaxpp-rno2a-ncclep-localffi-r11h-l8-e64k4-b512-s4096-p4m16-20260724-1338`. Parent `/dlwh/iris-run-job-20260724-203827`.
- Results:
  - Transformer Engine `4adad4c` built on all four workers, all 32 ranks initialized, and every stage reached JaxPP lowering.
  - Lowering stopped before execution because `EpDispatchBwdPrimitive.inner_primitive` returns a list pytree while `_ep_dispatch_bwd` declared tuple `out_specs`. All four workers reported the same `shard_map` pytree mismatch.
  - No `ep_prepare`, NCCL_EP scan, CUDA FFI, loss, gradient, duration, throughput, or MFU result was reached. W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ncclep-localffi-r11h-l8-e64k4-b512-s4096-p4m16-20260724-1338>.
  - Iris retried once. The babysitter stopped only the parent; parent and child are terminal killed and no matching pods remain.
- Interpretation:
  - This is a deterministic JAX output-tree declaration error in the new adapter, not evidence about local FFI shapes or NCCL transport.
  - Commit `0bcbb00e19` changes the backward `out_specs` to the list pytree returned by the registered TE multi-result primitive. Full-geometry forward/backward abstract tracing with list-valued fake TE primitives, focused tests, Pyrefly, and changed-file precommit pass.
- Next action:
  - Babysit r11i parent `/dlwh/iris-run-job-20260724-205516`; preserve the exact r11g geometry, receive capacity, and XLA flags.

### 2026-07-24 14:19 PDT - r11i proves nested shard_map does not localize TE prepare lowering
- Hypothesis: Fixing the TE backward primitive's output pytree will let the nested expert-axis `shard_map` carry rank-local route extents through execution.
- Commit Hash: `0bcbb00e19` (`[levanter] Match NCCL EP backward output tree`).
- Command: matched r11g L8/d2560/e64/top-k4/seq4096/b512/m16 explicit-MPMD `std_1f1b` gate with run ID `jaxpp-rno2a-ncclep-localffi-r11i-l8-e64k4-b512-s4096-p4m16-20260724-1355`. Parent `/dlwh/iris-run-job-20260724-205516`.
- Results:
  - All four stages lowered, compiled, and entered the first stage-0 forward execution. The TE JIT generated dispatch and combine variants with `maxt16384`, hidden size `2560`, top-k four, and eight local experts.
  - At execution, task 0 rank 0 reported `scan_impl_flat(em): padded EM slots 524288 > max_recv_tokens_per_rank 81920` fourteen times from `ht_scan_flat`, called by `EpPreparePrimitive.inner_primitive`.
  - `524,288` again equals `131,072` global tokens times top-k four. The later XLA `loop_gather_fusion` launch failure, CUDA module-unload errors, and coordination failures were cascading.
  - No loss, gradient, duration, throughput, or MFU result was produced. W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ncclep-localffi-r11i-l8-e64k4-b512-s4096-p4m16-20260724-1355>.
  - The babysitter stopped only the parent. Parent, child, and all tasks are terminal killed; no matching jobs or pods remain.
- Interpretation:
  - A nested `shard_map` can produce local-looking TE JIT specialization metadata while the surrounding JaxPP `auto_axes` transformation still gives the prepare lowering a global route extent.
  - Commit `c1b3087add` removes `auto_axes` from this backend and places the complete NCCL_EP MoE body, including custom VJPs and all inner TE primitives, inside one outer `shard_map`. A public-backend regression test verifies that global EP8 forward and backward inputs reach a fake TE boundary with rank-local shapes and reassemble global gradients. Focused tests pass `7/7`; Pyrefly and changed-file precommit pass.
- Next action:
  - Babysit r11j parent `/dlwh/iris-run-job-20260724-211906`. Require the prepare scan to remain within 65,536 local assignments and produce finite training metrics before L24.

### 2026-07-24 14:43 PDT - r11j and pinned NCCL_EP source correct the receive-capacity diagnosis
- Hypothesis: Placing the complete NCCL_EP MoE body inside one outer expert-axis `shard_map` will make the prepare scan count only rank-local assignments and fit the existing 81,920-row receive buffer.
- Commit Hash: `c1b3087add` (`[levanter] Localize the full NCCL EP MoE body`), documented through `826865c31e`.
- Command: matched L8/d2560/e64/top-k4/seq4096/b512/m16 explicit-MPMD `std_1f1b` gate with run ID `jaxpp-rno2a-ncclep-outershard-r11j-l8-e64k4-b512-s4096-p4m16-20260724-1419`. Parent `/dlwh/iris-run-job-20260724-211906`; child `/dlwh/iris-run-job-20260724-211906/grug-train-jaxpp-rno2a-ncclep-outershard-r11j-l8-e64k4-b512-s4096-p4m16-20260724-1419`.
- Results:
  - All ranks lowered and compiled. Stage 0 entered forward execution and generated TE JIT variants with rank-local `maxt16384`.
  - The earliest runtime failure remained `scan_impl_flat(em): padded EM slots 524288 > max_recv_tokens_per_rank 81920`, from `EpPreparePrimitive`'s `ht_scan_flat` variant. The later XLA `loop_gather_fusion` grid-81,920 launch failure and CUDA unload errors were secondary.
  - No loss, gradient, duration, throughput, or MFU metric was produced. W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ncclep-outershard-r11j-l8-e64k4-b512-s4096-p4m16-20260724-1419>.
  - The babysitter stopped only the parent. Parent, child, and all four tasks are terminal; no retry is live.
- Interpretation:
  - The r11g-r11j claim that 524,288 proved a global-buffer leak was wrong. Transformer Engine/NCCL_EP commit `4adad4c218c115cd9af235fb3d4e13ef4cec55a8` documents that `recv_capacity_per_rank` must be at least `ep_size * max_tokens_per_rank * top_k` to avoid drops. Its own EP test configures `max_recv_tokens_per_rank = num_tokens * n_ranks`.
  - The scan kernel computes the true padded expert-major receive extent, traps when it exceeds the configured maximum, and explicitly directs callers to increase `max_recv_tokens_per_rank` or enable `NCCL_EP_OVERFLOW_DROP`. For this exact geometry the documented bound is `8 * 16,384 * 4 = 524,288`.
  - The branch's capacity-factor formula modeled balanced receive load and underprovisioned the exact flat-layout contract. The scoped fix sizes bootstrap and dispatch to `global_microbatch_tokens * top_k`; focused tests pass `6/6`, and changed-file precommit including Pyrefly passes.
  - The user accepts at most `0.2%` relative error for explicitly approximate QuACK/NCCL_EP/FP8 comparisons. Loss and gradient validation remain strict, exact XLA transport remains bitwise, and no overflow-drop policy is enabled by this fix.
- Next action:
  - Commit and push the exact-capacity correction, then run the unchanged reduced L8 gate with lower XLA preallocation. If it executes finite steps, compare strict loss/gradient behavior before scaling to L24. If the exact 524,288-row flat layout OOMs, investigate an explicitly approximate overflow-drop variant under the accepted `0.2%` ceiling rather than silently changing semantics.

### 2026-07-24 15:14 PDT - r11k clears exact-capacity execution but poisons the next update
- Hypothesis: Provisioning Transformer Engine's documented 524,288-row worst-case receive extent will clear the prepare trap and execute finite reduced-pipeline training without exhausting H100 memory.
- Commit Hash: `210a5c6e8f` (`[levanter] Provision exact NCCL EP receive capacity`).
- Command: matched L8/d2560/e64/top-k4/seq4096/b512/m16 explicit-MPMD `std_1f1b` gate, four H100x8 nodes, 16 microbatches, CuTe FA4, Pallas-Triton `block_k=32`/8 warps, `save_moe`, XLA preallocation `0.55`, three steps, and `--xla_gpu_shard_autotuning=false`. Parent `/dlwh/iris-run-job-20260724-214550`; child `/dlwh/iris-run-job-20260724-214550/grug-train-jaxpp-rno2a-ncclep-exactcap-r11k-l8-e64k4-b512-s4096-p4m16-20260724-1444`.
- Results:
  - All 32 ranks bootstrapped four NCCL_EP groups with `max_tokens_per_rank=16384` and `recv_capacity_per_rank=524288`. All stages compiled forward, backward, accumulation, averaging, and update programs. The prior padded-EM trap did not recur, and no OOM occurred.
  - The first execution produced finite `train/loss=11.79250431060791`. Its compilation-contaminated metrics were `276.2362775s`, `7,591.8776 tokens/s`, and `15.8435726% MFU`; they are not a steady-state performance result.
  - The immediately following execution returned NaN on the training ranks and stopped before a finite gradient norm was captured. W&B finished at global step one: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ncclep-exactcap-r11k-l8-e64k4-b512-s4096-p4m16-20260724-1444>.
  - Iris began an automatic retry after distributed shutdown aborted. The babysitter stopped only this parent. Parent, child, all tasks, and matching pods are terminal.
- Interpretation:
  - Exact flat-layout capacity is memory-feasible for the reduced target geometry and fixes the prepare failure.
  - A finite first forward followed by an immediate NaN is consistent with undefined over-allocation rows entering the backward/update path. Transformer Engine documents that HT dispatch leaves the receive tail uninitialized. The branch masked invalid token inputs before grouped GEMMs but weighted every receive row, so undefined tail weights or cotangents could poison parameter gradients.
  - Commit `f35a6414c2` derives valid rows from `sum(token_counts)` and excludes tail rows plus non-finite/zero routing weights from both the weighting primal and transpose. A focused regression proves NaN tail values produce exact zero outputs and gradients; focused tests pass `7/7`, and changed-file precommit including Pyrefly passes. This preserves exact routed assignments and does not use the `0.2%` approximate-path allowance.
- Next action:
  - Babysit r11l parent `/dlwh/iris-run-job-20260724-221415`, which changes only the receive-tail mask and runs four steps. Require at least two finite post-compilation executions before scaling to the L24 target.

### 2026-07-24 15:35 PDT - r11l tail mask exceeds the 0.55 XLA allocator pool
- Hypothesis: Masking undefined receive-tail rows in the value and transpose will preserve r11k's memory feasibility and prevent the post-step NaN.
- Commit Hash: `f35a6414c2` (`[levanter] Mask unused NCCL EP receive rows`), documented through `afb4e6bf44`.
- Command: r11k's exact L8/d2560/e64/top-k4/seq4096/b512/m16 configuration with four steps and the receive-tail mask. Parent `/dlwh/iris-run-job-20260724-221415`; child `/dlwh/iris-run-job-20260724-221415/grug-train-jaxpp-rno2a-ncclep-tailmask-r11l-l8-e64k4-b512-s4096-p4m16-20260724-1520`.
- Results:
  - Setup, NCCL_EP bootstrap, and forward compilation succeeded. During stage-3 backward executable loading, every rank failed with `RESOURCE_EXHAUSTED` while requesting `24.70-25.24 GiB` under `XLA_PYTHON_CLIENT_MEM_FRACTION=0.55`.
  - No step executed, so this run produced no loss, gradient, duration, throughput, or MFU evidence. W&B remained at step zero: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ncclep-tailmask-r11l-l8-e64k4-b512-s4096-p4m16-20260724-1520>.
  - Parent, child, all four tasks, and matching pods are terminal.
- Interpretation:
  - r11l does not falsify the numerical fix; it failed before execution. The new mask changes backward memory planning, and the 55% XLA pool is too small for its transient 25 GiB allocation plus resident program buffers.
  - r11m parent `/dlwh/iris-run-job-20260724-223512` changes only XLA preallocation to `0.70`. This gives XLA roughly 56 GiB while leaving roughly 24 GiB for NCCL_EP and other non-XLA allocations.
- Next action:
  - Require r11m to load all executables and produce at least two finite post-compilation steps. If non-XLA NCCL_EP allocation then fails, tune the pool between `0.55` and `0.70` or reduce the mask's backward materialization before considering an approximate overflow policy.

### 2026-07-24 15:58 PDT - r11m validates the tail mask and exposes padded expert compute
- Hypothesis: Raising XLA preallocation from `0.55` to `0.70` will load the tail-masked executables and produce finite steady-state training steps.
- Commit Hash: `f35a6414c2` (`[levanter] Mask unused NCCL EP receive rows`), documented through `9652dd5b9a`.
- Command: r11l's exact L8/d2560/e64/top-k4/seq4096/b512/m16 explicit-MPMD `std_1f1b` configuration with four steps and XLA preallocation `0.70`. Parent `/dlwh/iris-run-job-20260724-223512`; child `/dlwh/iris-run-job-20260724-223512/grug-train-jaxpp-rno2a-ncclep-tailmask-r11m-xla070-l8-e64k4-b512-s4096-p4m16-20260724-1535`.
- Results:
  - Parent and child succeeded with exit `0`; all four tasks succeeded, and no Running or Pending worker pod remains.
  - Two steady executions remained finite: step 2 had loss `11.379294395446777`, duration `22.1068141s`, `94,864.5060 tokens/s`, and `1.9797378%` MFU; step 3 had loss `11.25849723815918`, duration `22.0929596s`, `94,923.9958 tokens/s`, and `1.9809793%` MFU.
  - Mean steady-state MFU was `1.9805654%`. W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ncclep-tailmask-r11m-xla070-l8-e64k4-b512-s4096-p4m16-20260724-1535>.
  - Correction to the append-only r11k entry above: W&B's MFU field is already expressed in percentage points. Its recorded numeric value was `0.1584357`, or `0.1584357%` MFU, not `15.8435726%`.
- Interpretation:
  - The receive-tail primal/transpose mask fixes r11k's post-step NaN. Exact flat-layout capacity is numerically viable for multiple updates at the reduced shape.
  - `_local_expert_ffn` extended the final expert's group size across every unused receive row. Both expert GEMMs therefore computed the full `524,288`-row worst-case buffer rather than the approximately `65,536` live rows, explaining the unusable throughput and extra memory.
  - Commit `0486695991` removes that artificial group extension while preserving exact NCCL_EP receive capacity and the existing tail masks. Focused tests pass `7/7`; changed-file precommit including Pyrefly passes.
- Next action:
  - Babysit matched sparse-tail r11n parent `/dlwh/iris-run-job-20260724-225848`. Require finite forward and backward execution plus a material steady-state speedup before considering L24. If the static worst-case buffer still dominates Triton launch overhead, evaluate a bounded `NCCL_EP_OVERFLOW_DROP` variant under the user's explicit `0.2%` relative-error ceiling with separate loss and gradient measurements.

### 2026-07-24 16:44 PDT - sparse exact compute is stable but remains below ring
- Hypothesis: Leaving the exact 524,288-row receive buffer allocated while limiting grouped GEMM work to live expert rows will recover NCCL_EP's direct routed-MLP advantage without reintroducing undefined-tail gradients.
- Commit Hashes:
  - `0486695991` removes the artificial final-expert group extension.
  - `61a0de2c49` masks undefined outputs after both ragged GEMMs in the primal and transpose.
- Commands:
  - r11n parent `/dlwh/iris-run-job-20260724-225848`, child `/dlwh/iris-run-job-20260724-225848/grug-train-jaxpp-rno2a-ncclep-sparsetail-r11n-xla070-l8-e64k4-b512-s4096-p4m16-20260724-1640`.
  - r11o parent `/dlwh/iris-run-job-20260724-231912`, child `/dlwh/iris-run-job-20260724-231912/grug-train-jaxpp-rno2a-ncclep-sparsetailmask-r11o-xla070-l8-e64k4-b512-s4096-p4m16-20260724-1620`.
  - Both used the matched L8/d2560/e64/top-k4/seq4096/b512/m16 explicit-MPMD `std_1f1b` configuration, four H100x8 nodes, exact receive capacity 524,288, XLA preallocation `0.70`, CuTe FA4, and Pallas-Triton `block_k=32`/8 warps.
- Results:
  - r11n produced finite step-0 loss `11.79250431060791` and then NaN at step 1 on every training rank. The first ragged GEMM correctly skipped unused rows but left its output tail undefined; applying SiLU before another mask allowed undefined values to poison the transpose. The babysitter stopped the retry; no resource remains live.
  - r11o succeeded parent `1/1`, child `4/4`, with all ranks exit `0`, finite losses and gradient/watch checks, and no live pod.
  - r11o step 2: loss `11.379260063171387`, `8.7061116s`, `240,882.7391 tokens/s`, `5.0270083%` MFU. Step 3: loss `11.258472442626953`, `8.7148441s`, `240,641.3676 tokens/s`, `5.0219711%` MFU. Mean steady MFU was `5.0244897%`.
  - W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ncclep-sparsetailmask-r11o-xla070-l8-e64k4-b512-s4096-p4m16-20260724-1620>.
- Interpretation:
  - Masking each undefined ragged output tail fixes the sparse-compute gradient path. Focused tests pass `8/8`, and changed-file precommit including Pyrefly passes.
  - Sparse exact compute is `2.5378x` r11m's padded exact `1.9805654%` MFU, but it reaches only `53.35%` of the matched reduced bulk-ring result `9.4180%`. Exact 524,288-row NCCL_EP is therefore not a credible L24 path and must not be scaled.
  - Commits `a724542264`, `9f35f8499f`, and `9284db9761` add an opt-in patched TE overflow policy, a separately named `nccl_ep_drop` backend with aligned capacity `81,920`, and a full-gradient ring-versus-drop parity gate. Exact `nccl_ep` retains pristine TE, trap semantics, and worst-case capacity.
- Next action:
  - Babysit H100x8 parity job `/dlwh/ncclep-h100-overflow-drop-parity-r1-20260724-1646`. Require finite loss and all gradient groups with relative-L2 error at most `0.002`; report output accumulation-order mismatch separately. Only then run reduced JaxPP `nccl_ep_drop`.

### 2026-07-25 13:23 PDT - bounded NCCL_EP is fast but misses the gradient error ceiling
- Hypothesis: Bounded 81,920-row NCCL_EP receive buffers will preserve the direct routed-MLP speedup, and moving weighting plus forward combine to FP32 will reduce every gradient's relative-L2 error below the accepted `0.002` ceiling.
- Commit Hashes:
  - `9284db9761` adds the bounded-overflow full-MLP gate.
  - `4a2eb217ff` adds FP32 weighting/combine.
  - `66a3528cb6` lets the opt-in Transformer Engine patch advertise FP32 payloads and bootstraps the direct gate accordingly.
- Commands:
  - BF16 combine r3: one H100x8 Iris task `/dlwh/ncclep-h100-overflow-drop-parity-r3-20260725-0845`, running `NCCLEP_OVERFLOW_POLICY=drop NCCLEP_COMBINE_DTYPE=bf16 PARITY_MODE=diagnostic bash experiments/ncclep_h100/run_full_mlp_ab.sh`.
  - FP32 bootstrap diagnostic r4: `/dlwh/ncclep-h100-overflow-drop-fp32-parity-r4-20260725-1257`.
  - FP32 combine r5: `uv run --package marin-iris --extra controller iris --config lib/iris/config/cw-rno2a.yaml job run --no-wait --enable-extra-resources --gpu H100x8 --cpu 64 --memory 512GB --disk 256GB --timeout 3600 --max-retries 0 --priority interactive --extra gpu --job-name ncclep-h100-overflow-drop-fp32-parity-r5-20260725-1314 -e NCCLEP_OVERFLOW_POLICY drop -e NCCLEP_COMBINE_DTYPE fp32 -e PARITY_MODE diagnostic -e XLA_PREALLOC_FRACTION 0.65 -- bash experiments/ncclep_h100/run_full_mlp_ab.sh`.
- Results:
  - All successful numerical runs used EP8, e64/top-k4, 16,384 tokens/rank, d2560/i1280, exactly 65,536 assignments per destination, receive capacity 81,920, and zero drops.
  - r3 BF16 combine was finite and `1.4516x` ring (`22.8160ms` versus `15.7178ms`). Loss relative-L2 was `9.94e-06`; token, routing-weight, W13, and W2 gradient relative-L2 errors were `0.004169`, `0.002267`, `0.003061`, and `0.002833`.
  - r4 built the FP32 kernels but failed before parity because bootstrap registered BF16 as the widest payload: `tokens dtype (4) wider than group max_token_dtype (6)`. All ranks exited and no resource remained live. Commit `66a3528cb6` fixes this only for the opt-in patched path.
  - r5 succeeded task `1/1`, all eight ranks exit `0`, and no matching pod remains. It was finite and `1.3679x` ring: medians `22.7535ms` versus `16.6338ms`, or `26.90%` lower latency.
  - r5 loss relative-L2 was `1.4439e-05`. Token, routing-weight, W13, and W2 gradient relative-L2 errors were `0.004238`, `0.002591`, `0.003159`, and `0.002990`. All four exceed `0.002`.
  - The separately accepted output accumulation signature was unchanged: relative-L2 `0.0029623`, mismatch fraction `0.00203554`, and max absolute error `0.0078125`. Every gradient had zero elementwise mismatches under the existing BF16 `rtol=0.1`, `atol=0.0002` check, but that loose check does not override the explicit relative-L2 gate.
- Interpretation:
  - Bounded NCCL_EP has a reproducible `1.37-1.45x` direct full-gradient speed advantage over bulk ring at the target per-microbatch geometry.
  - FP32 forward combine does not reduce the gradient discrepancy. The remaining difference comes from transport/expert accumulation order rather than BF16 route weighting alone.
  - Under the explicit `0.2%` loss-and-gradient relative-L2 ceiling, both BF16 and FP32 bounded paths are numerical negatives. Do not integrate `nccl_ep_drop` into reduced or L24 JaxPP training without a new numerical result or an explicit acceptance-policy change.
- Next action:
  - Keep exact NCCL_EP and bounded NCCL_EP blocked from L24 scaling. If this path is resumed, compare ring and NCCL_EP against an independent higher-precision combine/gradient reference; ring's BF16 scatter/reduction order is not itself a precision oracle.

### 2026-07-25 14:06 PDT - FP32 reference isolates one token-gradient reduction mismatch
- Hypothesis: Comparing bounded NCCL_EP against an FP32 ring scatter/reduction will remove the production ring's BF16 accumulation-order error; using a token-scaled loss and FP32 token dispatch will then bring every gradient below `0.002` relative-L2.
- Commit Hashes:
  - `658325f7ea` adds a diagnostic FP32 ring combine while leaving production ring's default BF16 behavior unchanged.
  - `3be17a42fe` changes the parity loss from a mean over all token-hidden elements to a mean over tokens after summing hidden dimensions.
  - `c2768fff3c` adds FP32 NCCL_EP token dispatch with a BF16 cast before expert GEMMs.
- Commands:
  - r6 FP32 reference: `/dlwh/ncclep-h100-fp32-oracle-r6-20260725-1331`, with FP32 NCCL_EP combine and FP32 ring combine.
  - r7 token-scaled reference: `/dlwh/ncclep-h100-tokenloss-fp32-oracle-r7-20260725-1343`, changing only loss scaling.
  - r8 FP32 dispatch: `/dlwh/ncclep-h100-fp32dispatch-oracle-r8-20260725-1355`, additionally setting `NCCLEP_DISPATCH_DTYPE=fp32`.
- Results:
  - r6 made loss, output, W13 gradients, and W2 gradients bitwise equal to the FP32 ring reference. Routing-gradient relative-L2 was `7.94e-05`. Token-gradient relative-L2 remained `0.002912`, with max absolute error `2.91e-11`.
  - r7's training-representative token-scaled loss changed token-gradient magnitude but not relative error: `0.00290949`, max absolute `1.19209e-7`, mean absolute `3.53064e-9`. Loss, output, W13 gradients, and W2 gradients remained exact; routing-gradient relative-L2 was `8.18e-05`.
  - r8 generated `payloadu32` forward and backward dispatch variants. Its numerical metrics were identical to r7: token-gradient relative-L2 `0.00290949`, routing `8.18e-05`, and exact loss/output/W13/W2. It remained `1.3288x` faster than the FP32 ring control (`23.4127ms` versus `17.6197ms`).
  - All three jobs succeeded with all eight ranks exit `0`, balanced 65,536-assignment destination loads, receive capacity 81,920, zero drops, finite outputs and gradients, and no matching pods after termination.
- Interpretation:
  - The production ring's BF16 forward combine explains the earlier forward, loss, and parameter-gradient discrepancies. Against FP32 ring, bounded NCCL_EP matches all of those exactly.
  - The remaining token-gradient difference is isolated to the dispatch-backward reduction order followed by BF16 input-gradient rounding. FP32 forward/backward payload selection does not change it.
  - Token-gradient relative-L2 is `0.2909%`, above the accepted `0.2%` ceiling despite small absolute error and zero mismatches under the legacy loose BF16 allclose. Bounded NCCL_EP remains blocked from pipeline integration under the current policy.
- Next action:
  - Stop bounded NCCL_EP precision experiments. Resume only if the acceptance policy explicitly changes or a different dispatch-backward algorithm produces token-gradient relative-L2 at most `0.002`. Continue target MFU work on exact ring or another numerically accepted backend.

### 2026-07-25 21:58 PDT - automatic task ownership clears; schedules expose separate runtime blockers
- Hypothesis: Preserving `AbstractMesh` `shard_map` equations during JaxPP mesh binding will remove the free stage activation, let every automatic MPMD transformation validate, and make `std_1f1b`, `eager_1f1b`, or zero-bubble executable on the reduced L4 gate.
- Commit Hashes:
  - `9e58b5a7c2` preserves `AbstractMesh` `shard_map` equations while rebinding concrete stage meshes and adds focused JAXPR validation.
  - `66cab1710c` wraps JaxPP's weak-reference cache in a marker-capable Python function.
  - `99b2aa1b94` adds a one-CPU, one-scalar reproducer. Marin bug #7644 packages the report for upstream review; no NVIDIA issue was filed.
- Commands:
  - All gates used L4/d2560/e8/top-k1/seq128/b32/m4, four H100x8 stages on `cw-rno2a`, ring MoE, XLA loss, two steps, explicit automatic input shardings, the const-sharding patch, conservative loop clustering, and every task-phase validator.
  - r12 parent `/dlwh/iris-run-job-20260725-224250`: automatic `std_1f1b` with `--xla_gpu_shard_autotuning=false`.
  - r13 parent `/dlwh/iris-run-job-20260726-003808`: r12 plus `--xla_gpu_autotune_level=0`.
  - r14 parent `/dlwh/iris-run-job-20260726-010634`: r13 plus `--xla_gpu_enable_cublaslt=false --xla_gpu_enable_triton_gemm=false`.
  - r15 parent `/dlwh/iris-run-job-20260726-012845`: matched r14 `eager_1f1b`.
  - r17 parent `/dlwh/iris-run-job-20260726-014805`: matched r14 zero-bubble.
- Results:
  - The earlier shallow-only r10 control still failed immediately after `bind_meshes` with an undefined BF16 activation. The actual destructive branch was pinned JaxPP `replace_captured_meshes`: `if isinstance(mesh, AbstractMesh): continue` removed the complete `shard_map` equation and left its output undefined.
  - The scalar reproducer transforms one valid equation into zero and `jax.extend.core.check_jaxpr` reports `Variable 'b' not defined`. Preserving the equation while leaving its abstract mesh unchanged returns one valid equation.
  - With the corrected patch, every rank passes `trace_and_place`, intermediate sharding inference, `bind_meshes`, task deduplication, loop unrolling, multidef fixup, common passes, local-JAXPR conversion, and final `mpmdify` validation. The prior free `bf16[1024,2560]` activation does not recur.
  - r12 reached `fwd_0` through `fwd_3` compilation. Rank 3 then spent about 110 minutes in `xla::gpu::AutotunerPass -> CublasLtBackend::GetSupportedConfigs` with all GPUs at 100%; it was stopped as a functional smoke with no metrics.
  - r13 proved `--xla_gpu_autotune_level=0` still enumerates cuBLASLt supported configs in this OpenXLA build. It was stopped after the same active compiler path produced no metric.
  - r14 removed `AutotunerPass`, cuBLASLt config discovery, and Triton GEMM rewriting. Compilation completed and execution began, then automatic `std_1f1b` reached a stable idle transfer deadlock: ranks 0/2/3 waited in `enqueue_nccl_transfer_group`, rank 1 waited in `recv_done_impl`, all GPUs were idle, and no step completed.
  - r15 `eager_1f1b` compiled through `after_loop_0_1`, then failed JaxPP output validation in `jaxpp/array.py:139`: `([P(None, None)], P())` and `([P(None,)], P())`. No step completed.
  - r17 zero-bubble compiled every forward task and entered `bwdA_3`. After roughly 190 minutes it had made no task transition, although nine compiler workers and all eight rank-3 GPUs remained saturated with stable roughly 60.4 GiB memory. It was stopped as operationally unusable for an L4 functional gate.
  - Every stopped parent, child, rank task, and pod is terminal. No Iris cluster was restarted or modified.
- Interpretation:
  - Confidence is high that #7644 is the original task-ownership failure. The runtime patch is validated through compilation and no longer depends on accepting approximate numerics.
  - Automatic schedules now have distinct blockers. Standard is a DIME transfer-order deadlock, eager has a deterministic output-sharding contract mismatch, and zero-bubble has prohibitive backward compile complexity.
  - `eager_1f1b` is the next repair target because its failure is deterministic, post-compilation, and cheaper to reproduce than zero-bubble. The accepted `0.2%` relative-L2 ceiling remains the hard gate for loss and every gradient once a schedule executes.
- Next action:
  - Minimize eager's `MpmdArray` output-sharding mismatch and patch the smallest correct boundary. Require two finite reduced steps before restoring optimized GEMM paths or scaling any automatic schedule.

### 2026-07-25 22:51 PDT - automatic eager 1F1B clears the functional gate
- Hypothesis: Supplying automatic JaxPP with the complete output sharding tree, retaining its MPMD outputs between steps, and materializing only scalar metrics back to SPMD will make eager 1F1B complete multiple updates.
- Commit Hashes:
  - `5cae7372c5` rank-normalizes omitted trailing replicated dimensions in automatic output metadata.
  - `0483010a56` adds a one-CPU reproducer for JaxPP's raw `P()` versus `P(None, ...)` output-spec comparison.
  - `576e5184fc` supplies explicit output shardings and preserves expert axes independently across parameter, optimizer, and EMA leaves.
  - `4754037ff1` adds a host pipeline-step counter and reshards only the MPMD loss leaf back to replicated SPMD.
  - `735c326a3a` prevents forced callbacks from dereferencing MPMD optimizer hyperparameters.
  - `a03b5dff4e` disables unsupported automatic-MPMD checkpoint writes after initial restore and logs every automatic loss exactly.
- Command:
  - `env GRUG_JAXPP_AUTO_EXPLICIT_IN_SHARDINGS=1 GRUG_JAXPP_PATCH_CONST_SHARDINGS=1 GRUG_JAXPP_VALIDATE_TASK_PHASES=1 XLA_FLAGS='--xla_gpu_shard_autotuning=false --xla_gpu_autotune_level=0 --xla_gpu_enable_cublaslt=false --xla_gpu_enable_triton_gemm=false' TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --cluster cw-rno2a --run-id jaxpp-auto-nockpt-legacycublas-eager-l4-e8-b32-s128-r22-20260725-2245 --schedule eager_1f1b --implementation auto --physical-stages 4 --logical-stages 4 --microbatches 4 --nodes 4 --gpus-per-replica 8 --expert-axis 8 --layers 4 --experts 8 --top-k 1 --batch 32 --seq-len 128 --vocab-size 8192 --moe-implementation ring --loss-implementation xla --steps 4 --tracker json_logger --xla-memory-fraction 0.70 --conservative-loop-clustering true --jax-init-timeout 7200 --jaxpp-client-timeout-ms 7200000`
- Results:
  - r18 parent `/dlwh/iris-run-job-20260726-050822` removed the replicated `P(None, ...)/P()` mismatch and exposed a genuine state-output mismatch: actual `P('expert', None, None)` versus replicated target. No step completed.
  - r19 parent `/dlwh/iris-run-job-20260726-051950` preserved expert output metadata and executed the complete forward, backward, and after-loop graph. It then failed because the host loop called `float()` on JaxPP's returned `MpmdArray` loss.
  - r20 parent `/dlwh/iris-run-job-20260726-052748` completed two finite updates and reused returned MPMD state without recompilation. Forced final logging then called `.item()` on an MPMD optimizer hyperparameter.
  - r21 parent `/dlwh/iris-run-job-20260726-053633` completed four finite updates. Its last two distinct samples were loss `9.035545349121094`, duration `0.0936341360s`, `42,084.61` tokens/s, `16.5227%` MFU and loss `9.033437728881836`, duration `0.0591394s`, `69,260.05` tokens/s, `27.1919%` MFU. Final TensorStore serialization rejected the automatic MPMD state.
  - r22 parent `/dlwh/iris-run-job-20260726-054521` succeeded parent `1/1` and child `4/4`, all tasks exit `0`. Exact losses were `9.045680046081543`, `9.050215721130371`, `9.035545349121094`, and `9.033437728881836`. Returned MPMD state was reused for three subsequent calls, loss was resharded to SPMD four times, final callbacks completed, and checkpoint writes were skipped.
  - Every failed predecessor was stopped after Iris entered retry. All parents, children, tasks, and pods are terminal; no cluster was restarted or modified.
  - The standalone output-spec reproducer emits `([P(None,)], P())` and `([P(None, None)], P())` while `NamedSharding.is_equivalent_to(..., ndim)` returns true. Replacing raw spec equality with semantic sharding equivalence passes both cases and JaxPP's five `MpmdArray` tests. Evidence is linked from #7644.
- Interpretation:
  - Automatic eager 1F1B is now functionally executable. Task ownership, output metadata, metric materialization, repeated state ownership, callbacks, and terminal cleanup all pass in one reduced four-step run.
  - The two steady samples straddle `20%` MFU and differ by `1.65x`. This tiny L4/e8/seq128 configuration and the disabled cuBLASLt/Triton rewrite path make them functional evidence only.
  - Automatic checkpoint writes remain unsupported because the returned state is distributed as JaxPP `MpmdArray` objects. The path can restore an initial SPMD checkpoint before compilation but intentionally does not serialize MPMD outputs.
  - No numerical acceptance claim follows from finite losses. Promotion still requires direct-versus-automatic relative-L2 at most `0.002` for loss and every gradient, per the user's accepted ceiling.
- Next action:
  - Build the smallest direct-versus-automatic loss/gradient gate at the same optimizer-free microbatch semantics. If every gradient passes `0.002`, restore optimized GEMM paths and run enough reduced eager steps for stable p50/mean MFU before considering L24.

### 2026-07-25 23:31 PDT - automatic eager fails production mixed-precision parity
- Hypothesis: Automatic eager 1F1B will match an ordinary direct microbatch average within relative-L2 `0.002` for loss and every gradient when both use FP32 master parameters and the production BF16 compute cast.
- Commit Hash: `90b709356f` adds a self-spawned four-rank parity harness and per-leaf acceptance report.
- Commands:
  - FP32 control: `XLA_FLAGS=--xla_force_host_platform_device_count=4 PYTHONPATH=/Users/dlwh/.cache/uv/git-v0/checkouts/b3c26618ca06d656/7091a9b/src uv run python experiments/grug/moe/check_jaxpp_eager_1f1b_parity.py --platform cpu --precision fp32`.
  - Production gate: one RNO2A H100x4 Iris task with `XLA_PYTHON_CLIENT_MEM_FRACTION=.25`, pinned JaxPP `7091a9b5`, patched `jax-tvm-ffi`, and `.venv/bin/python -u experiments/grug/moe/check_jaxpp_eager_1f1b_parity.py --platform gpu --precision production-mixed`. Job `/dlwh/jaxpp-eager-production-parity-20260726-062454`.
- Results:
  - The FP32 schedule-algebra control passed loss and all 72 gradient leaves. Loss was exact; maximum gradient relative-L2 was `0.0018629967`.
  - The production mixed-precision gate passed loss at relative-L2 `3.44356e-7`, but passed only `34/72` gradients. The other 38 exceeded `0.002`.
  - The maximum gradient relative-L2 was `0.113699` on `blocks[3].mlp.router`, whose direct reference norm was only `2.07e-13`. The largest non-router failure was `blocks[0].rms_attn.weight` at `0.0113588`. Attention and norm leaves failed broadly across all four blocks, generally around `0.003-0.011`.
  - The Iris task intentionally exited `1` for the failed gate after `96.42s`. It had no Python traceback, retry, or preemption. Post-report JAX shutdown emitted coordination-service connection-refused warnings after rank 0 exited; the job and H100x4 allocation are terminal with no live resource.
- Interpretation:
  - Automatic eager's microbatch reduction and gradient tree are algebraically correct in FP32, but production BF16 stage-local lowering changes gradients beyond the accepted policy. The broad attention/norm failures show this is not only the near-zero router denominator.
  - The finite four-step r22 loss trajectory is insufficient numerical evidence. Automatic eager is not eligible for optimized-path performance measurement or L24 scaling at the current `0.002` loss-and-every-gradient ceiling.
- Next action:
  - Keep automatic eager as a functional/reproducer path and return performance work to exact explicit MPMD. Resume automatic eager only after identifying a mixed-precision stage-boundary correction that passes every gradient without weakening the threshold.

### 2026-07-25 23:51 PDT - two-stage FSDP topology regresses
- Hypothesis: Mapping the same 4x8 H100 allocation to two 12-layer physical stages will halve inter-stage transfers, while the resulting `data=2` axis FSDP-shards dense and expert hidden dimensions enough to keep the target model within memory.
- Commit Hash: `3c712b28db`.
- Command:
  - `TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --cluster cw-rno2a --run-id jaxpp-rno2a-ring-p2-l24-e64k4-b512-s4096-m16-20260726-063603 --schedule std_1f1b --implementation explicit_mpmd --physical-stages 2 --logical-stages 2 --stage-layer-counts 12,12 --microbatches 16 --nodes 4 --gpus-per-replica 8 --expert-axis 8 --layers 24 --experts 64 --top-k 4 --batch 512 --seq-len 4096 --vocab-size 8192 --attention-implementation gpu_fa4_cute --ragged-dot-implementation triton --ragged-dot-block-k 32 --ragged-dot-num-warps 8 --moe-implementation ring --loss-implementation xla --steps 8 --tracker wandb --xla-memory-fraction 0.65 --remat save_moe`.
- Results:
  - Parent `/dlwh/iris-run-job-20260726-063606` and its child succeeded. All four child tasks exited `0`; there were no retries, preemptions, compile errors, OOMs, or live resources after completion.
  - Mean MFU was `12.9049979`, p50 `13.3542395`, and p90 `13.3597173`. This is `-3.2954904` points or `-20.34%` relative to the matched four-stage `16.2004883` mean-MFU result.
  - The six measured rows alternated between two slower samples at `11.8261` and `11.7322` MFU and four central samples at `13.3537-13.3589`. Central durations were approximately `6.963s` at `301.2k` tokens/s.
  - Loss remained finite from `8.5257807` at step 2 through `7.8227930` at step 7. Peak GPU allocation visible on the rank-0 host was `55.208 GiB` (`69.32%`).
  - W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ring-p2-l24-e64k4-b512-s4096-m16-20260726-063603>.
- Interpretation:
  - Data-axis FSDP makes the 12-layer stage weights fit, so memory is not the blocker. The topology is nevertheless substantially slower; fewer pipeline boundaries do not compensate for data-axis collectives and the larger per-stage forward/backward tasks.
  - The four equal six-layer stage topology remains the exact-path baseline. Do not scale the two-stage point to larger batch or profile it.
- Next action:
  - Keep four physical stages and require a reduced exact-path gain from a new overlap or kernel mechanism before another L24/m256 launch.

### 2026-07-26 00:34 PDT - combine-forward dgrad reproduces NCCL_EP token error
- Hypothesis: Transformer Engine's unweighted FP32 combine-forward transport will provide a different token-gradient reduction order than native dispatch backward, reducing the remaining bounded NCCL_EP token-gradient relative-L2 from `0.2909%` to at most the accepted `0.2%` while preserving a useful fraction of the direct transport speedup.
- Commit Hash: `36c89dfe37` (`[experiments] Add hybrid NCCL EP dgrad gate`).
- Command:
  - `uv run --package marin-iris --extra controller iris --config lib/iris/config/cw-rno2a.yaml job run --no-wait --enable-extra-resources --gpu H100x8 --cpu 64 --memory 512GB --disk 256GB --timeout 3600 --max-retries 0 --priority interactive --extra gpu --job-name ncclep-h100-hybrid-dgrad-r9-20260726-0022 -e NCCLEP_OVERFLOW_POLICY drop -e NCCLEP_COMBINE_DTYPE fp32 -e NCCLEP_DISPATCH_DTYPE bf16 -e NCCLEP_TOKEN_GRADIENT_IMPLEMENTATION hybrid_combine_forward -e RING_COMBINE_DTYPE fp32 -e PARITY_MODE strict -e XLA_PREALLOC_FRACTION 0.65 -- bash experiments/ncclep_h100/run_full_mlp_ab.sh`.
- Results:
  - Job `/dlwh/ncclep-h100-hybrid-dgrad-r9-20260726-0022` succeeded in `9m6.19s`; all eight ranks exited `0`. No resource remains live.
  - Loss, output, W13 gradients, and W2 gradients were bitwise identical to the FP32 ring reference. Routing-weight gradient relative-L2 was `8.1765745e-05`.
  - Token-gradient relative-L2 was `0.002909485039090222` (`0.2909485%`), with max absolute error `1.1920929e-07`, mean absolute error `3.5306382e-09`, and zero elementwise mismatches under the diagnostic BF16 `rtol=0.1`, `atol=2e-4` check.
  - Both arms were finite, routing was balanced at `65,536` assignments per destination, receive capacity was `81,920`, and neither arm dropped assignments.
  - Strict parity stopped before timing, so this run has no p50 or speedup measurement. The earlier native dgrad FP32-reference result was `0.00290949`; the hybrid did not change the numerical signature.
- Interpretation:
  - TE combine-forward and dispatch backward use the same effective reduction order for this token-dgrad. Calling combine-forward in FP32 does not produce the independent reduction algorithm required by the `0.002` gate.
  - Bounded NCCL_EP remains blocked from reduced and L24 JaxPP integration. Do not run a diagnostic timing-only repeat or relax the numerical gate.
- Next action:
  - Keep the four-stage bulk-ring path. The next candidate must change the exposed pipeline/EP overlap mechanism rather than wrapping another TE transport primitive around the same assignment reduction.

### 2026-07-26 01:16 PDT - cross-microbatch exact-ring fusion misses the scaling gate
- Hypothesis: Grouping contiguous microbatches inside one compiled exact-ring task will let XLA overlap independent all-gather, expert-compute, and psum-scatter work enough to recover the remaining `9.5%` MFU gap without changing MoE numerics.
- Commit Hash: `b9f071ddeb` (`[experiments] Add ring microbatch overlap gate`).
- Commands:
  - Group 2: one RNO2A H100x8 task, job `/dlwh/ep-ring-microbatch-overlap-r1-20260726-0110`, running the exact e64/top-k4/d2560/i1280/seq4096/microbatch32 shape for 5 warmups and 30 alternating-order samples.
  - Group 4: matched job `/dlwh/ep-ring-microbatch-overlap-g4-r2-20260726-0120`, changing only `--group-size 4`.
  - Both compared asynchronously queued full-ring calls, one fused full-ring graph, and explicit dispatch-all/expert-all/combine-all phases. Training promotion used fused value-and-grad only.
- Results:
  - Group 2 succeeded in `1m2.48s`. Fused forward improved p50 from `16.9375ms` to `15.0819ms` (`1.1230x`), while explicit phasing regressed to `17.8977ms` (`0.9463x`). Fused value-and-grad improved p50 from `37.2055ms` to `34.1668ms` (`1.088936x`).
  - Group-2 loss, outputs, token gradients, routing-weight gradients, W13 gradients, and W2 gradients were bitwise equal to queued execution. Maximum relative-L2 was `0`.
  - The group-2 training gain missed the `1.11x` promotion threshold. Including the four-stage bubble increase from 256 independent pipeline units to 128 projects the `18.2583` best to about `19.65` MFU.
  - Group 4 succeeded in `49.22s`. Fused forward reached `1.13346x`, explicit phasing regressed to `0.94304x`, and fused value-and-grad reached only `1.081880x`.
  - Group-4 loss, token gradients, and routing-weight gradients were exact, but W13 and W2 gradient relative-L2 were `0.00351787` and `0.00337545`. Both exceed the accepted `0.002` ceiling. The value-and-grad speedup also missed the `1.134x` threshold required after reducing the four-stage pipeline to 64 grouped units.
  - Both jobs exited `0` without retries or preemptions. No H100 resource remains live.
- Interpretation:
  - XLA can remove some launch or scheduling overhead when independent full-ring microbatches share one graph, but the training-step gain saturates below the amount needed to exceed 20 MFU.
  - Explicitly exposing dispatch, expert compute, and combine as separate queued executables is slower. JAX's asynchronous executable queue does not convert that structure into useful cross-microbatch collective/compute overlap.
  - Larger fused groups change the BF16 shared-gradient accumulation order enough to violate the accepted numerical policy and worsen the net pipeline projection.
- Next action:
  - Keep the benchmark and production-preserving ring phase refactor as reproducible evidence, but do not add grouped JaxPP tasks or run an L24 confirmation. Resume with a mechanism that changes stage rendezvous or collective execution rather than graph-level fusion of otherwise identical ring calls.

### 2026-07-26 01:52 PDT - public device ragged all-to-all clears the direct gate
- Hypothesis: The public OpenXLA device-initiated ragged-all-to-all in JAX nightly will remove the private-memory collective bottleneck at the exact EP8 assignment geometry without changing values, then remain usable through JaxPP's explicit MPMD path.
- Commit Hashes:
  - `d5e0cd05f8` adds an opt-in exact JAX CUDA 13 nightly install while preserving the locked JAX 0.10.1 default.
  - `8582f9b122` patches pinned JaxPP to canonicalize its raw `inline` boolean to JAX 0.11's `Inline` enum.
- Commands:
  - Direct H100x8 gate `/dlwh/xla-ragged-a2a-device-nightly-ep8-20260726-0130` used JAX, jaxlib, CUDA 13 plugin, and PJRT `0.11.1.dev20260725`; 65,536 assignments per rank; d2560; 30 measured iterations; symmetric ragged-all-to-all mode; and the experimental device kernel.
  - First reduced integration parent `/dlwh/iris-run-job-20260726-083256` used L8/d2560/e64/top-k4/seq4096/b512/m16, four H100x8 stages, explicit MPMD standard 1F1B, CuTe FA4, Triton grouped GEMM, and the same nightly.
  - The CPU reproducer constructs JaxPP `PjitKwargs`, binds `jax._src.pjit.jit_p` through `apply_task`, and executes one sharded add on JAX `0.11.1.dev20260725`.
- Results:
  - The direct device-kernel gate was bitwise exact with mismatch count `0` and checksum `3,623,878,656`.
  - Device-kernel median latency was `2.2801355ms`, mean `2.2773135ms`, minimum `2.185928ms`, and maximum `2.345059ms`. The matched private-memory baseline was `16.952521ms`, for a `7.4349x` speedup and `86.55%` latency reduction.
  - The requested `ragged_all_to_all_thunk` VLOG selection line did not emit. Symmetric 80 GB virtual address spaces on every rank, successful GIN/RMA setup, absence of fallback errors, exact values, and the latency discontinuity are strong but indirect device-path evidence.
  - The first reduced integration reached JaxPP's first stage task on every rank, then failed before compilation or training with `AttributeError: 'bool' object has no attribute 'value'`. JAX 0.11's `pjit` lowering dereferenced `inline.value`, while pinned JaxPP passed raw `False` directly to the primitive instead of using JAX's public JIT canonicalization.
  - The compatibility patch converts booleans to `jax._src.api.Inline` only on JAX 0.11 or newer. The minimized reproducer now executes with `Inline.AUTO`; the project JAX 0.10.1 path still returns raw booleans. Changed-file precommit passes.
  - The failed parent and all four child tasks are terminal killed with no retries, failures, metrics, or live resources. Reduced retry parent `/dlwh/iris-run-job-20260726-084810` is running from `8582f9b122`.
- Interpretation:
  - The public transport mechanism is the first exact EP8 candidate with enough direct headroom to plausibly close the remaining target MFU gap.
  - The first integrated failure was a narrow, reproducible JaxPP/JAX private-API compatibility break rather than an XLA collective, compiler, memory, or numerical failure.
  - The fixed acceptance policy remains relative-L2 at most `0.002` for loss and every gradient.
- Next action:
  - Require finite reduced retry metrics and a clear gain over the matched `9.4180%` ring reference. If it passes, launch the four-stage L24/b512/m16 target before scaling batch further.

### 2026-07-26 04:18 PDT - NCCL pin removes crashes but exposes two JaxPP integration stalls
- Hypothesis: The reduced JaxPP failures after the direct device-kernel success come from one removable compiler or custom-kernel interaction; isolating attention, expert GEMM, transport, NCCL version, and schedule will identify a functional promotion path.
- Commit Hash: `ddf717e994` pins `nvidia-nccl-cu13==2.30.7` in every opt-in JAX nightly worker setup and prints the installed NCCL version before JaxPP installation.
- Commands:
  - Reduced parents `/dlwh/iris-run-job-20260726-084810`, `/dlwh/iris-run-job-20260726-091919`, and `/dlwh/iris-run-job-20260726-092720` progressively replaced CuTe FA4 with reference attention and Triton expert GEMM with XLA while retaining JAX nightly, ragged-all-to-all, and standard 1F1B.
  - Full-block control `/dlwh/jaxpp-full-block-nightly-bf16-r2-20260726-0230` used the self-contained two-H100x8 JaxPP CuTe FA4 reproducer on the same nightly.
  - Exact-ring control parent `/dlwh/iris-run-job-20260726-093511` used nightly JAX, reference attention, XLA expert GEMM, and otherwise matched reduced pipeline geometry.
  - Pre-pin ragged control parent `/dlwh/iris-run-job-20260726-095230` unset every experimental device and symmetric-memory flag.
  - Post-pin optimized parent `/dlwh/iris-run-job-20260726-100716` restored CuTe FA4, Triton expert GEMM, standard 1F1B, device ragged-all-to-all, and NCCL 2.30.7.
  - Post-pin functional parent `/dlwh/iris-run-job-20260726-104936` used reference attention, XLA expert GEMM, and disabled XLA autotuning, cuBLASLt, and Triton GEMM rewriting.
- Results:
  - The JaxPP `Inline` patch removed the original `bool.value` failure on every rank.
  - With NCCL `2.28.9`, CuTe, reference-attention, and XLA-expert variants all segfaulted compiling the first stage-0 forward in `54-63s`. The pre-pin no-device-flags control failed sooner in `deepCopyDevCommRequirements -> ncclDevCommCreate -> NcclDeviceCommunicator::CreateFrom`, proving the nightly still attempted device-communicator setup.
  - The self-contained nightly JaxPP plus CuTe FA4 full-block gate succeeded `2/2`: lower took `2.083s/2.039s`, compile plus execute `11.914s/100.364s`, and both barriers returned. CuTe FA4 and JaxPP localization alone are not the crash trigger.
  - The matched nightly exact-ring control succeeded `4/4`. First-forward compilation took `14s`; finite loss was `8.797631`; the post-warmup step took `4.434745s` at `7.114594%` MFU and `472,891` tokens/s. W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-nightly-ring-refattn-xlagemm-l8-e64k4-b512-s4096-p4m16-20260726-0300>.
  - Pinning NCCL 2.30.7 removed the device-communicator segfault. The optimized device-ragged run compiled through stage-2 backward, then made no transition for `27m`. Three ranks reported 100% GPU utilization with `57.2-57.6 GiB` allocated, 0% memory throughput, and `115-139W`; one rank was idle. An unchanged native stack showed `cuModuleLoadData -> CustomKernelThunk::Initialize -> GpuProfiler::Profile -> ConfigRunner::ProfileAll`, with NCCL proxy threads spinning.
  - Disabling autotuning and GEMM rewrites cleared `ConfigRunner::ProfileAll` and completed stage-2 backward compilation, then execution deadlocked. Rank 0 blocked launching a JaxPP NCCL transfer group in `ncclGroupEnd`; rank 1 waited for a JaxPP receive; ranks 2-3 remained inside PJRT execution. GPU telemetry was again nonproductive. No loss or MFU was emitted.
  - All failed or stopped parents, children, ranks, and pods are terminal. One intervening submission was setup-invalid because GitHub clone timed out; it produced no experiment evidence and was not counted.
- Interpretation:
  - The direct device collective remains exact and fast, but the current JaxPP integration is not executable. Correct NCCL eliminates the native setup crash but exposes an autotuner/device-communicator stall and, without autotuning, a transfer-order deadlock under standard 1F1B.
  - Attention and expert custom kernels are not required for failure. Exact ring on the same nightly is functional, so JAX 0.11 plus the general JaxPP pipeline is not sufficient.
  - No L24 promotion or MFU claim is valid until a reduced schedule completes finite steps. The fixed relative-L2 ceiling remains `0.002` for loss and every gradient.
- Next action:
  - Test GPipe with the functional no-autotune graph because its non-overlapped transfer ordering may avoid the standard-1F1B deadlock. In parallel, package a self-contained JaxPP transfer plus ragged-all-to-all reproducer and file only a linked Marin issue for upstream review.

### 2026-07-26 04:50 PDT - GPipe and transfer priority retain the device-ragged deadlock
- Hypothesis: GPipe's non-overlapped execution or standard 1F1B's transfer-priority mode will impose a consistent NCCL launch order and avoid the JaxPP transfer plus device-ragged-all-to-all wait.
- Commands:
  - GPipe parent `/dlwh/iris-run-job-20260726-111423`.
  - Standard 1F1B transfer-priority parent `/dlwh/iris-run-job-20260726-113121`.
  - Both used L8/d2560/e64/top-k4/seq4096/b512/m16, four H100x8 stages, JAX `0.11.1.dev20260725`, NCCL `2.30.7`, device ragged-all-to-all, reference attention, XLA expert GEMM, and disabled XLA autotuning/cuBLASLt/Triton GEMM rewriting.
- Results:
  - GPipe made no progress after rank-3 compiler diagnostics. Two unchanged captures showed rank 0 in `enqueue_nccl_transfer_group -> ncclGroupEnd`, ranks 1-2 in `recv_done_impl`, and rank 3 in `backend_compile_and_load`.
  - Transfer priority changed rank placement but not the failure: ranks 0-1 blocked in DIME `enqueue_nccl_transfer_group`, rank 2 remained in `backend_compile_and_load`, and rank 3 remained inside PJRT execution.
  - Both runs held approximately `56.8-57.6 GiB` per active GPU with 0% memory activity and only `114-141W`. Reported 100% core utilization on selected ranks did not correspond to productive work.
  - Neither run emitted loss, duration, throughput, MFU, or a finished W&B summary. Every parent, child, and task is terminal killed with no running descendants.
- Interpretation:
  - The deadlock is not specific to standard 1F1B overlap, GPipe ordering, or accumulation-before-transfer ordering. Explicit JaxPP inter-stage transfers and nightly device ragged-all-to-all do not currently coexist in this four-stage program.
  - Stop L24 promotion and schedule variants. The direct collective's `7.4349x` transport gain cannot be converted into a training result until the reduced distributed regression is fixed.
- Next action:
  - Land the smallest JaxPP transfer plus ragged-all-to-all regression package, file only a Marin issue linked to #7024, and provide upstream-ready evidence without filing externally.

### 2026-07-26 05:34 PDT - device-ragged boundary packaged and numerical gate confirmed
- Hypothesis: The minimum JaxPP transfer plus device-ragged-all-to-all composition will reproduce the L8 transfer deadlock and provide a self-contained upstream boundary.
- Commit Hash: `0cabe5f74b` (`[grug] Add JaxPP ragged all-to-all regression boundary`).
- Command:
  - H100x4 job `/dlwh/jaxpp-jax011-ragged-minimal-r2-20260726-121552` ran direct ragged-all-to-all, JaxPP transfer-only, and JaxPP transfer plus ragged-all-to-all using the command in `experiments/grug/moe/repro_jaxpp_jax011_ragged_all_to_all.README.md`.
- Results:
  - Direct ragged-all-to-all returned in `0.863855s` with zero mismatches and checksum `202`.
  - JaxPP transfer-only returned from `eval_local` in `1.242563s` on rank 0 and `2.206070s` on rank 1 with zero mismatches and checksum `202`.
  - JaxPP plus ragged-all-to-all returned from `eval_local` in `1.425364s` on rank 0 and `2.447640s` on rank 1 with zero mismatches and checksum `202`.
  - All cases completed in one attempt without watchdog output, signals, retries, or nonzero exits. No H100 resource remains live.
  - Filed Marin bug https://github.com/marin-community/marin/issues/7655 with `bug` and `agent-generated`; no NVIDIA issue was filed.
  - Human acceptance decision: relative-L2 error up to `0.2%` is acceptable for this work. The operational gate is `<=0.002` for loss and every gradient leaf; values above `0.002` fail promotion.
- Interpretation:
  - The minimum primitive composition is a passing lower-bound regression package, not a reproducer of the deadlock itself. The failure requires additional context from the four-stage L8 training graph.
  - The direct collective remains the only exact transport candidate with enough measured headroom to close the MFU gap, but it cannot be promoted to L24 until #7655 is fixed.
- Next action:
  - Use #7655 to minimize the additional four-stage condition. Require finite L8 steps and the fixed `0.002` loss/gradient gate before any L24 performance run.

### 2026-07-26 05:54 PDT - repeated four-stage gate passes; test directional communicators
- Hypothesis: Rank count, bidirectional DIME, repeated receive-buffer reuse, or many live stage tasks is sufficient to reproduce the device-ragged training deadlock.
- Commit Hashes:
  - `65bcf217f3` adds the one-microbatch four-stage transfer and ragged cases.
  - `bfbe839d6b` adds configurable repetition and the 16-microbatch gate.
  - `c737829620` forwards JaxPP communicator controls through the Iris launcher.
- Commands:
  - H100x8 job `/dlwh/jaxpp-jax011-ragged-four-stage-r1-20260726-124203` ran one-microbatch four-stage transfer and ragged cases.
  - H100x8 job `/dlwh/jaxpp-jax011-ragged-four-stage-m16-r2-20260726-124630` repeated both cases for 16 microbatches.
  - Parent `/dlwh/iris-run-job-20260726-125321` launches the full L8/d2560/e64/top-k4/seq4096/b512/m16 standard-1F1B device-ragged treatment with `JAXPP_DIRECTIONAL_COMMUNICATORS=true`. It otherwise retains JAX `0.11.1.dev20260725`, NCCL `2.30.7`, reference attention, XLA expert GEMM, disabled autotuning/cuBLASLt/Triton GEMM rewriting, and receive-buffer reuse.
- Results:
  - The one-microbatch transfer and ragged cases passed exactly with checksum `202`. All rank phases returned without watchdog output.
  - The 16-microbatch transfer and ragged cases passed exactly with checksum `3,232`. The treatment exercised 96 logical transfers and 128 stage tasks; all eight rank processes emitted `case_passed` and exited zero.
  - The repeated job succeeded in `60.6s`. Transfer-case rank evaluation took `5.527-5.676s`; ragged-case evaluation took `6.619-6.925s`.
  - No setup, Python, compiler, DIME, NCCL, timeout, teardown, or live-resource failure remained.
- Interpretation:
  - Rank count, bidirectional topology, receive-buffer reuse at tiny payload, 16 microbatches, 96 transfers, and 128 stage tasks are not sufficient to reproduce #7655.
  - The remaining high-probability boundary is full-graph payload/compute compilation interleaved with multiple NCCL communicators. Pinned JaxPP uses one unordered DIME communicator for both directions by default; `JAXPP_DIRECTIONAL_COMMUNICATORS=true` creates separate ordered communicators and is the narrow next treatment.
- Next action:
  - Babysit `/dlwh/iris-run-job-20260726-125321`. If it emits finite steps, run the fixed `0.002` loss/every-gradient parity gate before L24 promotion. If it retains the mixed DIME/PJRT wait, test NCCL implicit launch ordering, prewarming, then disabled receive-buffer reuse as separate controls.

### 2026-07-26 06:13 PDT - directional communicators deadlock during ID creation
- Hypothesis: Giving forward and reverse DIME traffic separate communicators will avoid the full L8 device-ragged launch-order cycle.
- Commit Hashes:
  - `c737829620` forwards `JAXPP_DIRECTIONAL_COMMUNICATORS` and `JAXPP_REUSE_RECV_BUFFERS` through the launcher.
  - `d51de36375` prepares the next isolated `NCCL_LAUNCH_ORDER_IMPLICIT` control.
- Command:
  - Parent `/dlwh/iris-run-job-20260726-125321` ran the matched L8/d2560/e64/top-k4/seq4096/b512/m16 standard-1F1B device-ragged graph with `JAXPP_DIRECTIONAL_COMMUNICATORS=true`, JAX `0.11.1.dev20260725`, NCCL `2.30.7`, reference attention, XLA expert GEMM, disabled autotuning/cuBLASLt/Triton GEMM rewriting, and default receive-buffer reuse.
- Results:
  - A live worker confirmed `JAXPP_DIRECTIONAL_COMMUNICATORS=true` and `JAXPP_REUSE_RECV_BUFFERS` unset.
  - All ranks compiled the initial forward/loss task chain and reached the stage-2 backward boundary. No new log appeared after `12:56:14Z`.
  - Unchanged stack captures at `13:07:10Z` and `13:08:31Z` showed rank 0 in `BlockingKeyValueGet -> dime2.get_nccl_id`, rank 1 in `dime2.recv_done` while creating a CUDA stream, and ranks 2-3 in `PjRtLoadedExecutable::Execute`. Rank 0 and ranks 2-3 reported 100% GPU utilization; rank 1 reported 0%.
  - The native stacks showed no compiler activity despite the final `Compiling` log. No loss, duration, throughput, or MFU was emitted.
  - The parent and four child tasks were stopped after approximately `14m50s`; all are terminal killed with no running descendant.
- Interpretation:
  - Directional communicator reuse is not the fix. It moves rank 0 from the prior `ncclGroupEnd` wait into directional communicator-ID creation, which strengthens the multi-communicator creation/order diagnosis.
  - The next separate control should use NCCL implicit launch ordering with JaxPP directional communicators left at the default.
- Next action:
  - Babysit implicit-order parent `/dlwh/iris-run-job-20260726-130937`. A live worker has confirmed `NCCL_LAUNCH_ORDER_IMPLICIT=1` with directional communicators and receive-buffer controls unset. If it fails, prewarm communicator/executable creation before testing disabled receive-buffer reuse.

### 2026-07-26 06:30 PDT - implicit launch ordering fails; prewarm adjacent DIME links
- Hypothesis: NCCL implicit launch ordering will make JaxPP DIME and device-ragged collective launches consistent across ranks.
- Commit Hashes:
  - `5267c88a53` adds the fixed `0.002` loss and every-gradient parity gate for explicit standard 1F1B with device-ragged MoE.
  - `08b6be5dd3` adds opt-in, globally ordered DIME communicator and stream prewarming.
- Commands:
  - Parent `/dlwh/iris-run-job-20260726-130937` ran the matched L8/d2560/e64/top-k4/seq4096/b512/m16 standard-1F1B device-ragged graph with `NCCL_LAUNCH_ORDER_IMPLICIT=1`. Directional communicators and receive-buffer controls were unset.
  - Parent `/dlwh/iris-run-job-20260726-132849` runs the same graph with only `GRUG_JAXPP_PREWARM_DIME=1`. The prewarm serializes links 0->1, 1->2, and 2->3, creates one communicator and both directional streams for each of eight device lanes, and uses a host coordination barrier after each link.
- Results:
  - The implicit-order run made no progress after `13:12:33Z`, when rank 2 logged compilation of `grug_1f1b_mb0_stage2_backward` and rank 3 logged `grug_1f1b_mb1_accumulate_loss`.
  - Unchanged captures at `13:22:43Z` and `13:24:08Z` showed rank 0 in `ncclGroupEndInternal -> enqueue_nccl_transfer_group`, rank 1 in `cuStreamCreate -> enqueue_wait -> recv_done_impl`, and ranks 2-3 in `PjRtLoadedExecutable::Execute -> apply_task`.
  - Rank 0 and ranks 2-3 held all eight local GPUs at 100% utilization; rank 1's GPUs were at 0%. Per-GPU memory was `57,247-57,593 MiB`. No loss, duration, throughput, MFU, or terminal W&B metric was emitted.
  - The implicit-order parent and child were stopped at approximately `13:24:25Z`. All four child tasks are terminal killed, and no live descendant remains.
  - The prewarm treatment is allocated and in setup. Its monitoring owner is subagent `019f9e9d-a03b-7292-acf9-d9d658971635`.
- Interpretation:
  - `NCCL_LAUNCH_ORDER_IMPLICIT=1` is a hard negative. It does not change the mixed DIME/PJRT execution cycle.
  - Communicator and stream creation still occurs lazily inside different rank-local task phases. The prewarm treatment moves that creation before lowering and imposes one chain-wide order without changing the schedule, collective kernel, or receive-buffer policy.
- Next action:
  - Require all three prewarm-link barriers and finite L8 steps. If the run succeeds, execute the fixed `0.002` parity gate before L24. If it retains the wait, test `JAXPP_REUSE_RECV_BUFFERS=false` as a separate control.

### 2026-07-26 06:35 PDT - early DIME prewarm fragments symmetric memory
- Hypothesis: Prewarming DIME before JaxPP lowering will impose consistent communicator order without changing device-ragged memory capacity.
- Commit Hashes:
  - `08b6be5dd3` adds DIME communicator and stream prewarming before stage-state allocation.
  - `c224c3d442` moves the same prewarm after stage-local parameter and optimizer-state allocation.
- Commands:
  - Parent `/dlwh/iris-run-job-20260726-132849` ran the matched L8 device-ragged graph with the early-prewarm placement.
  - Parent `/dlwh/iris-run-job-20260726-133417` retries the same treatment after stage-state allocation.
- Results:
  - The early treatment prewarmed links 0->1 and 1->2 at `13:31:03Z` and link 2->3 at `13:31:04Z` on all four ranks.
  - Compilation reached `grug_1f1b_mb0_stage2_backward` at `13:31:42Z`, then rank 2 failed at `13:31:46Z` in the DLPack transfer path. NCCL reported no suitable space for `0xcde000000` bytes inside `0x1400000000`: a `51.469 GiB` symmetric-memory allocation inside the 80 GiB device arena.
  - The warning appeared on all eight rank-2 GPUs. Fabric and NVLink checks were healthy. No loss, throughput, or MFU was emitted.
  - Parent and child are terminal killed; all four child ranks exited zero after user termination, and no live descendant remains. W&B initialized but contains only configuration and topology summaries: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ragged-prewarm-l8-e64k4-b512-s4096-p4m16-20260726-1328>.
- Interpretation:
  - Deterministic DIME initialization itself completes, but creating CuPy/NCCL resources before XLA stage state fragments the address space needed by the device-ragged symmetric allocation.
  - The ordinary lazy path creates stage state before DIME and does not hit this allocation failure. `c224c3d442` preserves that allocator order while moving only communicator creation ahead of task lowering.
- Next action:
  - Babysit post-state-prewarm parent `/dlwh/iris-run-job-20260726-133417`. Require all three link barriers, no repeated symmetric-memory failure, and finite L8 steps before the `0.002` parity gate.

### 2026-07-26 06:41 PDT - full DIME prewarm is allocator-incompatible; isolate streams
- Hypothesis: Allocating stage state before all-link DIME prewarming will reserve XLA memory early enough to preserve NCCL's device-ragged symmetric arena.
- Commit Hash: `0cc1f0e539` replaces the boolean prewarm with explicit `streams` and `all` modes.
- Commands:
  - Parent `/dlwh/iris-run-job-20260726-133417` ran full DIME prewarming after stage-local state allocation.
  - Parent `/dlwh/iris-run-job-20260726-133926` runs stream-only prewarming with communicator creation left lazy.
- Results:
  - The post-state full prewarm completed all three link barriers on every rank, then reproduced the exact `0xcde000000`-within-`0x1400000000` NCCL allocation failure at `13:37:11Z` in `jit_grug_1f1b_mb0_stage2_backward`.
  - No finite loss or MFU was emitted. The parent and all four child tasks are terminal killed with no live descendant. W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ragged-poststate-prewarm-l8-e64k4-b512-s4096-p4m16-20260726-1334>.
  - The stream-only treatment is allocated and has initialized W&B. Its monitoring owner is subagent `019f9ea7-1497-73e2-8300-cc78691b00fd`.
- Interpretation:
  - Stage-state allocation is not sufficient. Creating every inter-stage NCCL communicator before each rank executes its first device-ragged MoE fragments the 51.469 GiB symmetric reservation.
  - Stream-only mode targets rank 1's repeated `cuStreamCreate` wait but leaves communicator creation interleaved with each stage's first forward, which is the only observed non-OOM order.
- Next action:
  - Require stream-only link barriers, no communicator logs during prewarm, no symmetric-memory failure, and finite L8 steps. If it fails, isolate receive-buffer reuse next.

### 2026-07-26 06:44 PDT - stream prewarm also fragments symmetric memory
- Hypothesis: Creating only DIME CUDA streams will remove lazy stream initialization from the mixed wait without interfering with device-ragged symmetric memory.
- Command:
  - Parent `/dlwh/iris-run-job-20260726-133926` ran `GRUG_JAXPP_PREWARM_DIME=streams`.
  - Parent `/dlwh/iris-run-job-20260726-134401` runs the matched graph with `JAXPP_REUSE_RECV_BUFFERS=false` and no prewarm.
- Results:
  - Stream-only prewarm logged all three links in `streams` mode on every rank at `13:41:50Z`, with no communicator creation during the prewarm.
  - Ordinary lazy communicator creation followed during forward compilation. At `13:42:25Z`, rank 2 again failed `jit_grug_1f1b_mb0_stage2_backward` because NCCL could not place `0xcde000000` bytes inside `0x1400000000`.
  - No finite loss or MFU was emitted. Parent and child are terminal killed with no live descendant. W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ragged-stream-prewarm-l8-e64k4-b512-s4096-p4m16-20260726-1339>.
  - The no-reuse treatment is submitted. It removes JaxPP's donated receive-buffer reuse fences and allocates private receive buffers per transfer; memory use is an explicit risk.
- Interpretation:
  - Even CUDA stream creation before a stage's first device-ragged MoE fragments the required 51.469 GiB symmetric reservation. DIME prewarming is closed as an initialization-time fix.
  - Receive-buffer reuse is the remaining isolated runtime-lifetime control. It does not change communicator or stream initialization order.
- Next action:
  - Babysit no-reuse parent `/dlwh/iris-run-job-20260726-134401`. Capture HBM/OOM separately from the original mixed DIME/PJRT wait.

### 2026-07-26 06:59 PDT - receive-buffer reuse is negative; warm ragged before DIME
- Hypothesis: JaxPP's donated receive-buffer reuse fences cause the mixed DIME/PJRT wait.
- Commit Hash: `66feb79449` adds an opt-in stage-local device-ragged warmup before DIME initialization.
- Commands:
  - Parent `/dlwh/iris-run-job-20260726-134401` ran `JAXPP_REUSE_RECV_BUFFERS=false` with every prewarm and ordering control unset.
  - Parent `/dlwh/iris-run-job-20260726-135924` first executes a minimal device-ragged collective across each stage's EP8 mesh, then prewarms all adjacent DIME resources in global link order.
- Results:
  - No-reuse made no progress after `13:46:51Z`. Unchanged five- and ten-minute samples showed rank 0 in `transfer_start_impl -> _alloc_zeros -> PjRtLoadedExecutable::Execute`, rank 1 in `recv_done_impl -> enqueue_wait -> DLPackManagedTensorToBuffer -> CreateViewOfDeviceBuffer -> cuStreamCreate`, and ranks 2-3 in `task_impl -> apply_task -> PjRtLoadedExecutable::Execute`.
  - Telemetry was byte-for-byte stable: rank 0 and ranks 2-3 held all GPUs at 100%, rank 1 held all GPUs at 0%, and per-GPU memory was `57,247-57,593 MiB`. No compiler process, symmetric-allocation warning, OOM, loss, or MFU appeared.
  - Parent and all four child ranks are terminal killed with no live descendant. W&B: <https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ragged-no-recv-reuse-l8-e64k4-b512-s4096-p4m16-20260726-1344>.
  - The warm-then-DIME treatment is submitted. Its monitoring owner is subagent `019f9eb9-5b36-76d2-8d1d-e7a4e6c598f8`.
- Interpretation:
  - Receive-buffer reuse is not the deadlock cause. Disabling it changes rank 0 to private-buffer allocation but leaves the rank-1 DLPack stream creation wait and downstream PJRT execution unchanged.
  - All initialization-time prewarm failures occur because CuPy/NCCL resources precede the local stage's first device-ragged symmetric reservation. The new treatment reverses that order, then applies deterministic DIME initialization.
- Next action:
  - Require all four stages to log ragged warmup before any DIME creation, all three DIME link barriers, no 51.469 GiB allocation error, and finite L8 steps.
