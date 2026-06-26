# Grug MoE GPU Work Party Status

Date: 2026-06-25 PT

## TL;DR

We are trying to make Grug MoE viable on NVIDIA GPU nodes without switching the training stack to Megatron/TE. The current active target is the May d2560/L26 Grug MoE shape on CoreWeave H100s, tracked mainly in #4302, #4312, #6493, and #6573.

The most actionable current baseline is SGD forward/backward on H100 with `model_axis=1`, FA4 attention, Pallas CE, ring MoE, and `recompute_all`. B4/device runs; B8/device does not fit in the tested ring-MoE configuration. On N2, the no-flags B4/device run reached 20.63 MFU and 190.8k tokens/s. On N4, the mirror reached 19.93 MFU and 368.6k tokens/s after several clique/rendezvous retries.

The latest evidence splits the work into five lanes:

1. Forward/backward is around 20 MFU in the current runnable config. O1/JAX-Toolbox flags regressed; AutoPGLE without O1 was only a small and noisy final-step win.
2. Single-node scoped profiling showed MoE transport/layout and expert GEMMs are both hot. It is not just GMM. XLA cross entropy and FA4 attention are separate, nontrivial chunks.
3. The primary MoE transport/kernel bet should move away from DeepEP and toward a custom Pallas/Mosaic fused MoE kernel. DeepEP internode smokes proved pieces of the runtime, but the integration path has been fragile enough that it should be treated as background evidence, not the main work-party lane.
4. MuonH is still a separate optimizer problem. MuonH5 adds about 1 second/step versus SGD on the single-node B8 profile. Grouped-bank Muon is not yet good enough: the immediate targets are preventing OOMs, improving real performance, and avoiding grouped-to-FSDP boundary explosions.
5. Profiling is now usable enough for work-party triage. The command-buffer-disable profiling mode preserves name scopes; the May208 roofline/dashboard work (#6573) is the right place to turn profiles into phase tables and speed-of-light rows.

## Source Map

Primary issues:

- #4302: parent H100 x8 MoE MFU tracker and strategy thread.
- #4312: end-to-end H100 Grug MoE forward/backward, DeepEP, and profiling thread.
- #4301: older v4-1024 116B-A16B bring-up thread, now closed. Useful for TPU context and ragged-all-to-all EP32 history, not active H100 work.
- #6493: MuonH GPU speed experiment.
- #6573: May208 roofline dashboard attribution experiment.
- #6557: roofline dashboard implementation request. Closed after triage, but #6573 remains the active experiment/logbook layer.
- #6596: H100 fused cross entropy B-tile performance follow-up.

Recent relevant PRs:

| PR | State | Why it matters |
|---|---|---|
| #6377 | Open | FA4 sliding-window attention support. Required by the older assigned-token DeepEP PR stack. |
| #6251 | Open | Assigned-token DeepEP MoE dispatch, including a DeepEP-backed CUDA path and issue-shape benchmark harness. |
| #6462 | Merged | Profiles are read from the run directory, which changed how profile serving and profile downloads work. |
| #6463 | Open | DeepEP remat policy branch. Replaces DeepEP-specific block call path with checkpoint save-name policy. |
| #6473 | Merged | `scripts/iris/dev_gpu.py`, an Iris-backed CoreWeave H100 dev pod workflow. |
| #6551 | Merged | CoreWeave smoke-test data now writes under a TTL-managed R2 temp prefix. |
| #6572 | Merged | H100 B-tiled fused cross entropy path for large vocab. Bounds logits memory while keeping full-vocab GEMMs. |
| #6577 | Merged | GPU profiling flag guidance, including `--xla_gpu_enable_command_buffer=''` for readable profiles. |
| #6613 | Merged | Routes D128 GQA FA4 backward through native SM90 on H100. Closes #6632. |
| #6633 | Merged | Removes dead experiments off the Grug/MoE critical path. |
| #6637 | Merged | Stages CUDA toolchain for CoreWeave GPU jobs so Pallas/Mosaic lowering can find `ptxas`, `nvlink`, and `libdevice`. |

## Current Performance Baselines

### May d2560 H100 forward/backward lane

Configuration for the latest batch/remat probe:

- Shape: May d2560/L26, seq_len 4096, sliding_window 2048.
- Parallelism: `model_axis=1`, `expert_axis=8`, `replica_axis=2 or 4`.
- Backend: ring MoE, FA4 attention, Pallas CE.
- Optimizer: SGD, so this mostly measures forward/backward.
- Dtype: bf16 params/compute/output.
- Remat: `recompute_all` for the current runnable large-batch probe.

| Run | Nodes | Flags | Batch/device | Remat | Final MFU | Mean MFU | Tokens/s | Duration |
|---|---:|---|---:|---|---:|---:|---:|---:|
| [May402](https://wandb.ai/marin-community/marin_moe/runs/GM2560-MAY-402S4096-W2048-B64-R2-E8M1-PALLASCE-RING-RECOMPUTEALL-FA4SGD-NOFLAGS-N2-cw-20260624-020812) | 2 | none | 4 | `recompute_all` | 20.6288 | 15.5391 | 190,767 | 1.374s |
| [May403](https://wandb.ai/marin-community/marin_moe/runs/GM2560-MAY-403S4096-W2048-B128-R4-E8M1-PALLASCE-RING-RECOMPUTEALL-FA4SGD-NOFLAGS-N4-cw-20260624-021515) | 4 | none | 4 | `recompute_all` | 19.9305 | 14.8978 | 368,619 | 1.422s |
| [May404](https://wandb.ai/marin-community/marin_moe/runs/GM2560-MAY-404S4096-W2048-B64-R2-E8M1-PALLASCE-RING-RECOMPUTEALL-FA4SGD-JAXTOOLBOXO1-N2-cw-20260624-023445) | 2 | O1/toolbox | 4 | `recompute_all` | 18.3573 | 13.7808 | 169,761 | 1.544s |
| [May405](https://wandb.ai/marin-community/marin_moe/runs/GM2560-MAY-405S4096-W2048-B64-R2-E8M1-PALLASCE-RING-RECOMPUTEALL-FA4SGD-PGLE-N2-cw-20260624-032705) | 2 | PGLE, no O1 | 4 | `recompute_all` | 21.3041 | 15.0287 | 197,012 | 1.331s |
| [May406](https://wandb.ai/marin-community/marin_moe/runs/GM2560-MAY-406S4096-W2048-B128-R4-E8M1-PALLASCE-RING-RECOMPUTEALL-FA4SGD-PGLE-N4-cw-20260624-032718) | 4 | PGLE, no O1 | 4 | `recompute_all` | 20.1182 | 14.3038 | 372,090 | 1.409s |

Memory boundary:

- B8/device with `save_moe` OOMed around 120-122 GiB/device, with and without O1/toolbox flags.
- B8/device with `recompute_all` still OOMed on N2 with a 93.15 GiB allocation.
- B4/device with `save_moe` still OOMed on N2 with a 56.39 GiB allocation.
- B4/device with `recompute_all` runs.

Interpretation:

- Use no-flags B4/device + `recompute_all` as the current stable forward/backward baseline.
- Do not carry the O1/JAX-Toolbox flag bundle for this lane. It regressed May402 by about 11% final MFU/tokens and increased duration by about 12%.
- AutoPGLE without O1 is not compelling yet. It gave a small final-step lift but lower mean MFU and emitted `PGLE collected an empty trace`.
- N4 nearly doubles tokens/s over N2 but does not improve MFU, and 32-device startup was flaky.

### Single-node profile attribution

May215 is the key scoped single-node SGD profile in #4312:

- W&B: [May215](https://wandb.ai/marin-community/marin_moe/runs/GM2560-MAY-215S4096-W2048-B8-R1-E8M1-XLACE-RING-REPEMB-REPOUT-FA4SGD-MOESCOPED-PROFILE-N1-cw-20260621-0022)
- Branch: `codex/issue4312-moe-profile-scopes`
- Result: step 4 was 16.0653 MFU, 74,282.8 tokens/s, 0.4411s/step.

Profile split from #4312:

| Family | Aggregate device time | Share |
|---|---:|---:|
| MoE | 3444 ms | 43.04% |
| XLA CE | 1398 ms | 17.47% |
| FA4 attention flash | 802 ms | 10.02% |
| Attention dense | 532 ms | 6.65% |
| Collectives | 407 ms | 5.08% |
| Optimizer apply | 365 ms | 4.57% |

Visible scoped hot paths:

| Phase | Aggregate device time | Share of profile kernel time |
|---|---:|---:|
| `moe_ep_ring/gather_inputs` transport | 814.7 ms | 10.16% |
| `moe_ep_ring/psum_scatter_output` transport | 460.8 ms | 5.75% |
| `moe_ep_ring/dispatch_gather_tokens` | 157.3 ms | 1.96% |
| `moe_ep_ring/combine_scatter_add` | 157.3 ms | 1.96% |
| `moe_expert_mlp/w13_ragged_dot` | 799.5 ms | 9.97% |
| `moe_expert_mlp/w2_ragged_dot` | 109.6 ms | 1.37% |
| XLA CE path | 1187.7 ms | 14.82% |
| FA4 attention | 750.3 ms | 9.36% |

Interpretation:

- The bottleneck is not only expert matmul/GMM. Ring EP transport and layout work are also large.
- `gather_inputs` and `psum_scatter_output` are priority targets if we keep ring EP.
- Cross entropy is large enough to justify the Pallas CE path and the B-tiled H100 work.
- Attention is no longer the old pathological all-gather case, but FA4 remains a visible part of the step.

## Custom Pallas/Mosaic MoE Kernel Lane

DeepEP is no longer the preferred primary workstream. The replacement workstream is a custom Pallas/Mosaic MoE kernel aimed at fusing the hot ring-MoE layout and compute region more directly.

The kernel target should start from the current profile evidence, not from a generic MoE implementation:

- May215 shows ring MoE time split between transport/layout and expert matmul.
- `moe_ep_ring/gather_inputs` and `moe_ep_ring/psum_scatter_output` are large enough to justify replacing the transport pattern.
- `moe_expert_mlp/w13_ragged_dot` is also large, so a kernel that only moves tokens but leaves the expert compute fragmented is probably insufficient.

### Proposed kernel scope

Start with the single-node H100 case. The first useful prototype does not need cross-node remote memory.

The first target is fused dispatch plus the first expert projection:

- consume router assignments / token indices;
- pack or directly read tokens for each expert;
- compute `W_gate/W_up`;
- apply SiLU or the relevant activation;
- write a representation that lets the second projection and combine avoid the current ring gather/scatter pattern where possible.

The backward plan should be part of the design from the start. At minimum, define how the kernel will expose or recompute:

- gradients for `W_gate/W_up`;
- activation backward;
- token-gradient accumulation back to the original sequence layout;
- any expert-local count/prefix metadata needed to avoid a second expensive layout pass.

References:

- Current MoE profile split: #4312 May215.
- Pallas kernel workflow: `.agents/skills/add-pallas-kernel/SKILL.md`.
- SonicMoE kernels are a useful reference for fused MoE compute even though they do not solve EP for us.
- DeepEP remains useful as a reference for packed variable-size dispatch/combine semantics, but not as the primary implementation path.

### Work-party tasks

1. Write the Pallas/Mosaic fused MoE kernel spec against the May215 scoped phases.
2. Build a tiny correctness harness: router assignments plus expert weights in, fused gate/up+activation output out.
3. Add a single-node benchmark harness that compares against current ring `gather_inputs + w13_ragged_dot` for the same shapes.
4. Add backward design before optimizing forward too far.
5. Keep cross-node remote memory or NVSHMEM/Pallas remote refs as a later phase after the single-node kernel proves useful.

## DeepEP Lessons

DeepEP still produced useful evidence:

- Upstream-style process-per-GPU topology matters. `16 x H100x1` can be placed incorrectly; `2 x H100x8` with eight local workers per task matched DeepEP assumptions.
- Runtime, dispatch, combine, and combine-backward smokes can work on CoreWeave under the supervised shape.
- The integration path for full Grug training has been fragile enough that the work party should not spend the main kernel-design slot trying to rescue it.

Keep #6251 and #6463 as references. Do not make them the main day-one plan.

## MuonH Status

MuonH is a separate bottleneck from forward/backward. The clean A/B in #6493 is:

- May140 MuonH5: 4.7499 MFU, 21,962.5 tokens/s, 1.492s/step.
- May141 SGD: 16.1876 MFU, 74,848.2 tokens/s, 0.4378s/step.

This attributes roughly 1.05s/step to Muon/update overhead on that single-node B8 shape.

### What worked in isolation

The standalone Muon update harness now supports production-shaped expert weights and grouped-bank variants. The useful conclusions from #6493 and `.agents/logbooks/grug-moe-muon-gpu.md` are:

- Reducing Newton-Schulz iterations matters. H1 recovered a large fraction of the H5 tail in full-model runs, but that is an algorithm/science question.
- The isolated grouped expert MuonH path can clear roughly 50% nominal speed-of-light when it preserves the grouped representation.
- R4 packed-master/block-group value+grad paths can run around 0.44s with 14 compiled all-gathers under normal XLA settings.
- `--xla_gpu_autotune_level=0` is a bad measurement confound for MuonH harnesses. It changed the compiled program shape and made valid paths look about 2.7x slower.
- The latency-hiding scheduler did not materially hide block-group materialization in the R4 harness.

### What failed or stayed incomplete

- Simple grouped/padded MuonH did not improve full training throughput. May143 grouped/padded MuonH5 finished at 4.6705 MFU and 1.517s/step, essentially flat versus May140.
- Unbounded grouping OOMed during autotune by creating a `bf16[832,2560,2560]` GEMM batch.
- Restoring grouped optimizer updates back to ordinary per-layer FSDP leaves in the hot path causes compiled all-gather/all-to-all patterns or OOMs.
- `jax.custom_partitioning` did not force the desired lower-level grouped-to-FSDP transport on GPU; compiled HLO still inlined back to all-gathers.
- Naive per-layer materialization produced 52 all-gathers and regressed badly.

### Current design direction

The grouped-bank path is not good yet. The promising direction is still to keep routed expert master/momentum in a grouped bank representation and teach the expert consumer to use grouped banks directly, but the immediate engineering target is more basic: make the harness avoid OOM, measure stable performance, and prove it does not add boundary collectives.

Concrete representation from the local design notes:

- `w_gate_up`: `[group, expert, d_model, 2 * intermediate]`
- `w_down`: `[group, expert, intermediate, d_model]`
- group axis sharded over `replica_dcn,data` where possible, padded when needed.

The next integration gate should prove the real `MoEExpertMlp` / `MoEMLP` path can consume grouped expert banks without OOMing and without adding boundary collectives beyond the standalone grouped-MoE baseline. If it does not beat the current MuonH full-run behavior, it should stay a harness result.

## Cross Entropy Status

PR #6572 landed an H100 B-tiled fused cross entropy path for large vocab. It bounds memory while keeping full-vocab GEMMs.

From #6596, at B=32768, H=2560, V=128256:

- XLA reference materializes full logits/dlogits and measured 32.58ms forward, 100.10ms `jax.grad(loss_fn)`, 246,983 tok/s combined.
- Production B-tiled default 8192 measured 33.73ms forward, 123.56ms backward, 208,323 tok/s combined.

Interpretation from #6596:

- The backward gap is mostly the recompute-vs-materialize tradeoff. The B-tiled backward does about 3 full-vocab GEMMs per tile; XLA grad is about 2 GEMMs but pays the logits/dlogits memory cost.
- Default tile 8192 should stay unless a real target shape shows enough memory headroom for 16384. The 16384 timing only buys about 1% over 8192 for about 2x logits memory.
- A fused B-tiled plus V-streamed backward is the possible next kernel path, but it depends on Mosaic GPU lowering support.

Work-party tasks:

1. Check whether the May d2560 forward/backward lane is actually using the H100 B-tiled path where intended.
2. Profile the B-tiled backward in a full train step to verify the recompute GEMM attribution.
3. Keep CE as a secondary target after MoE transport and Muon representation, unless a profile shows CE growing beyond the current ~14-17% range.

## FA4 Attention Status

The earlier FA4 issue was a pathological batch-axis all-gather before the FA4 CuTe FFI. The fix was to put the FA4 CuTe call behind a batch-axis `shard_map` so XLA did not implicitly all-gather the batch argument. That removed a misleading large all-gather around attention.

Recent state:

- #6377 is still open for sliding-window FA4 attention support.
- #6613 merged native SM90 backward routing for D128 GQA on H100.
- #6632 recorded the evidence for #6613: B=8 production backward at 3.142ms, 410 TFLOP/s, 41.47% speed-of-light, and about 95.95% of the built-in-local upstream SM90 path.

Work-party tasks:

1. Verify the merged SM90 path is active in the May d2560 H100 profiles.
2. Keep non-D128/MHA cases on the compatibility path unless they show up in target profiles.
3. Decide whether #6377 should be rebased/closed/continued now that #6613 landed.

## Profiling And Roofline Status

The profile workflow improved during this effort:

- #6462 moved profile lookup to the run directory.
- #6577 documented GPU profiling flags.
- The useful readability flag is `--xla_gpu_enable_command_buffer=''`. It hurts performance and should be used only for attribution profiles.
- #6573 tracks May208 roofline dashboard attribution.

The roofline/dashboard work should answer four questions for each phase:

1. How many FLOPs or bytes should this phase cost?
2. How much device time did the profile observe?
3. Which kernels are still semantically unattributed?
4. Which gaps are compute-bound, bandwidth-bound, collective-latency-bound, or profile-tooling artifacts?

Known buckets from #6573 include:

- `expert_all_to_all`
- `expert_backward_psum`
- MuonH Newton-Schulz rows
- `unaccounted_for`
- `unknown_collective`
- `uncategorized`

Work-party tasks:

1. Land or refresh the roofline dashboard branch from #6573.
2. Seed it with May215, May402/403, and the latest Muon harness profile.
3. Add a phase table template that every new profile can fill: MoE transport, expert GEMM, CE, FA4, dense attention, optimizer, collectives, unknown.
4. Require every optimization PR or experiment to state which phase it is expected to move.

## CoreWeave Operations Status

The operational story is better than it was at the start of the H100 work:

- #6473 added `dev_gpu.py` for Iris-backed CoreWeave H100 dev pods.
- #6637 stages the CUDA toolchain for GPU jobs through setup scripts, fixing Pallas/Mosaic `ptxas`, `nvlink`, and `libdevice` failures.
- #6551 moved CoreWeave smoke-test output to TTL-managed R2 temp prefixes.

Remaining operational risk:

- 32-device rendezvous/clique startup is flaky. May403 eventually completed, but only after multiple clique/rendezvous retries.
- DeepEP process placement matters. `16 x H100x1` can be scheduled as `4+4+4+4` across hosts, which violates DeepEP normal-mode assumptions. The reliable shape so far is `2 x H100x8` with a local supervisor spawning one process per GPU.

## Suggested Work-Party Agenda

### Slide 1: Goal

Make Grug MoE on H100/GB200 viable enough that we do not need to switch the training stack to Megatron/TE for these MoE shapes. The near-term question is where the current 20 MFU H100 path is losing time and which workstream can move it.

### Slide 2: Current runnable baseline

Show May402/May403/May404/May405/May406. The useful headline is B4/device + `recompute_all`: 20.63 MFU on N2, 19.93 MFU on N4, O1 regresses, PGLE is inconclusive.

### Slide 3: Memory boundary

Show the failed B8/device and `save_moe` attempts. B8/device is not the current experiment target for ring MoE unless remat or memory layout changes. B4/device + `recompute_all` is the baseline for flags and transport tests.

### Slide 4: Single-node phase split

Show May215. MoE is 43%, XLA CE is 17%, FA4 flash is 10%, collectives are 5%. Ring MoE hot paths include `gather_inputs`, `psum_scatter_output`, and `w13_ragged_dot`.

### Slide 5: Custom Pallas/Mosaic MoE kernel

Show the May215 MoE split and the target fused region: dispatch/gather plus `W_gate/W_up + SiLU`, with backward planned up front. DeepEP is background evidence for packed dispatch semantics, not the primary path.

### Slide 6: MuonH state

Show May140 vs May141. MuonH5 adds about 1.05s/step. Isolated grouped-bank Newton-Schulz can be fast; restoring to per-layer FSDP leaves is the integration blocker.

### Slide 7: Cross entropy and FA4

Show #6572 and #6613. CE has a memory-bounded H100 path but pays recompute. FA4 D128 GQA backward now routes through native SM90. These are real wins but probably not enough alone to move the whole step.

### Slide 8: Profiling and roofline

Show #6573. Every workstream needs the same phase table and speed-of-light estimate. Attribution is now a first-class task, not an afterthought.

### Slide 9: Workstream assignments

Proposed breakouts:

| Workstream | First task | Owner profile |
|---|---|---|
| Pallas/Mosaic MoE kernel | Spec and prototype fused dispatch plus `W_gate/W_up + SiLU` for single-node May shape | Pallas/Mosaic / MoE kernels |
| Ring MoE transport | Keep as baseline; use May215 phases to define what the custom kernel must replace | MoE systems / collectives |
| MuonH grouped banks | Make grouped-bank harness avoid OOM and show real perf before model integration | optimizer / JAX sharding |
| CE kernel | Verify B-tiled path in full May profile; decide whether V-streamed backward is worth prototyping | Pallas / kernels |
| FA4 | Confirm #6613 active in profiles; resolve #6377 status | attention kernels |
| Roofline dashboard | Turn May215 and May402/403 into a reusable phase table | profiling / tooling |
| CoreWeave ops | Stabilize 32-device rendezvous and document process-per-GPU launch recipes | Iris / cluster ops |

### Slide 10: Near-term success criteria

Use concrete gates:

- Forward/backward lane: repeatable N2/N4 B4/device baseline above 20 MFU, then one optimization that moves a named phase by at least 10%.
- Pallas/Mosaic MoE lane: single-node fused dispatch+`W_gate/W_up+SiLU` harness beats the current ring scoped phase or produces a clear failure reason.
- Muon lane: grouped-bank harness avoids OOM, reports stable performance, and compiles without extra boundary collectives before model integration.
- Profiling lane: every new profile has a phase table with unknown/unaccounted time below an agreed threshold.

## Open Questions

1. What is the exact first fused Pallas/Mosaic kernel boundary: dispatch+`W_gate/W_up+SiLU` only, or dispatch+full expert MLP+combine?
2. Is B4/device acceptable for near-term engineering, or do we need to prioritize memory work to recover B8/device before transport tuning?
3. What is the science tolerance for MuonH3 or fewer Newton-Schulz steps? The systems result says H1/H3 are much cheaper, but training quality is a separate decision.
4. Should grouped expert-bank representation become a real model parameter format, or should we keep it as a harness until it proves both memory and speed?
5. What target should the roofline dashboard use as its first canonical profile: May215 for readable single-node phase split, or May402/403 for current multi-node runnable baseline?

## Recommended First Day Plan

1. Start with May402/May403 as the throughput baseline and May215 as the attribution baseline.
2. Split into Pallas/Mosaic MoE kernel, Muon, CE/FA4, roofline, and ops groups.
3. Each group picks one metric to move or one blocker to clear by the end of the session.
4. Avoid broad flag sweeps unless they are tied to one phase and one baseline. O1/toolbox already regressed the runnable config.
5. Keep `model_axis=1` unless a separate experiment explicitly says otherwise.
