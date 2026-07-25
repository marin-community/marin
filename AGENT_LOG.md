# EP25 Direction 2: Matched Transport Bake-Off

## Background Research Brief

- Effort: medium
- Stop rule: stopped when the supplied ranked direction, the fixed-gather result, the #7421 diagnosis, and the local NGC reference branch converged on the same minimum matrix.
- Date: 2026-07-24

### Evidence

- Issue #7201 comment 5073017396: fixed-capacity gather dispatch measured 20.558% p50 MFU, 318.5K tokens/s, with 20.467/20.662% p10/p90 over 119 samples. Kernel forward/drop results were exact and input/combine/expert gradients matched at `rtol=atol=1e-5`.
- Issue #7279 comment 5074952738: `ring_cute` EP4 won prior e64/e128 backend ladders but has no e256/top-8/EP64 measurement; placement changes results by ±2–4 MFU points, requiring repeated interleaved draws.
- Issue #7421 comment 5064478402: NGC JAX 26.06 does not contain the kernel-cache discriminator fix; a CUDA PJRT plugin containing OpenXLA `4c1b00509e64` is required.
- Read-only worktree `issue-7421-ngc-7279`: commit `499895bd4` contains the guarded NGC overlay; commits `107476c8d` and `f1e96d872` contain the CuTe-backed EP adaptations.
- Current base `fe21ea495`: fixed-A2A exists but uses repeated activation scatter; `ring_cute` is not accepted by the production launcher path.

### Initial Hypotheses

1. `ring_cute` EP4 or EP8 may beat 20.558% at e256/top-8 because it won prior backend ladders, but that evidence is indirect across a different expert shape.
2. Ragged all-to-all with the one-shot kernel disabled may match or beat fixed A2A at parity; no matched production-shape control currently exists.
3. A parity tie favors fixed capacity if the peer pipelining probe is positive, because only fixed peer buckets expose compile-time slices for structural overlap.

## Check-in 2026-07-24 23:15 UTC

Findings so far:

- 20.558% p50 / 318.5K tok/s is the locked fixed+gather reference; 17.552% is the matched scatter control, not the arm to promote.
- 2–4 MFU points of placement variance makes two interleaved draws per arm mandatory.
- 0 production `ring_cute` selector exists on this base; the read-only NGC branch has the minimal prior implementation.
- NGC 26.06 requires an external verified PJRT plugin containing OpenXLA `4c1b00509e64`; the setup code is local, while the artifact URI/SHA still need recovery.
- 2-rack matrix cells remain deferred pending coordinator approval.

Confidence: 6/10 that this direction contributes a significant step toward 25% MFU.

Next: implement and locally verify the gather gate, `ring_cute` selector, and NGC overlay, then launch one-replica smokes.

## Check-in 2026-07-24 23:28 UTC

Findings so far:

- 2/2 focused fixed-A2A tests pass: gather dispatch matches scatter forward output, dropped count, and input/combine/expert gradients at `rtol=atol=1e-5`.
- 1/1 `ring_cute` dependency-contract test passes; the production selector now reaches the CuTe-backed ring implementation, with actual lowering deferred to the GB200 smoke.
- 4/4 NGC dispatch tests pass: NGC JAX/JAXLIB remain protected, `SCALE_*` knobs reach nested workers, and the replacement PJRT plugin is hash-verified before loading.
- The exact tested #7421 fixed plugin was recovered from `/tmp/cubin7421-fix-full.log`: `s3://marin-us-east-02a/tmp/ttl=30d/cubin7421-ngc-xla-plugin-probe-07/fix/xla_cuda_plugin.so`, SHA-256 `e420223a7a3ce7e5a816be50286e3610dacb10971984935ce986b316f47d8194`.
- New coordinator evidence makes drop fraction a primary decision metric: prior 8-bucket production work dropped about 3%, while fixed EP64 uses roughly 16K fine-grained buckets and may be materially worse under sender imbalance.
- New profile evidence strengthens fixed layout's optimization ceiling only conditionally: combine/all-gather backward, stray unstack operations, and an unoverlapped final reduce-scatter are additional costs; the peer agent is probing round-robin pipelining.
- Cluster CLI access is currently blocked by sandbox DNS/auth refresh (`oauth2.googleapis.com` cannot resolve), so no smoke has been submitted from this session yet. The connected browser path was cancelled and no cluster state was changed.

Confidence: 7/10 that this direction contributes a significant step toward 25% MFU.

Next: finish lint and a local reproducibility commit, append exact smoke/rack commands, then retry Iris connectivity; use the stock-toolchain fallback only if NGC setup itself fails after connectivity is restored.

## Check-in 2026-07-24 23:38 UTC

Findings so far:

- 24/24 runnable tests pass and 6 GPU-only tests skip across the complete Grug MoE test file plus the NGC dispatch suite. This includes fixed gather/scatter forward, drop-count, and gradient parity.
- Ruff 0.14.3 passes on all changed Python files. Black 25.9.0 reformatted `experiments/grug/dispatch.py`; the focused tests still pass afterward. The repository wrapper cannot create uv tools under the sandbox's read-only home. Direct Pyrefly 1.0.0 against the complete reference environment reports two unrelated repository gaps (`finelog.telltale` and `scripts.ci.claude_runner`) and no diagnostics in changed files.
- The Iris retry is still blocked before a cluster query. Overlaying current Iris sources onto the peer `ep25-d4-pipelined` environment supplies a compatible `iris_native`, but OAuth refresh fails because `oauth2.googleapis.com` cannot resolve in this sandbox.
- The required local snapshot is also blocked by the managed filesystem: Git cannot create `/home/marin/projects/marin/.git/worktrees/ep25-d2-bakeoff/index.lock`. All intended files remain unstaged in the assigned worktree; no unrelated file was modified or staged.
- No smoke or rack job was submitted. The stock-toolchain fallback clock has not started because the NGC setup has not reached a cluster worker; this is an access failure, not an NGC setup stall.

### Round table

The historical fixed+gather row is a reference, not a draw from this round. Drop count was not reported in the cited result, so it cannot satisfy this round's fidelity gate.

| Arm | Draw | p10 MFU | p50 MFU | p90 MFU | p50 tok/s | Drop count | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| fixed A2A + gather | historical reference | 20.467% | 20.558% | 20.662% | 318.5K | not reported | 119 samples |
| fixed A2A + gather | 1 | pending | pending | pending | pending | pending | cluster access blocked |
| ragged A2A, one-shot off | 1 | pending | pending | pending | pending | pending | cluster access blocked |
| ring_cute EP4 | 1 | pending | pending | pending | pending | pending | cluster access blocked |
| fixed A2A + gather | 2 | pending | pending | pending | pending | pending | cluster access blocked |
| ragged A2A, one-shot off | 2 | pending | pending | pending | pending | pending | cluster access blocked |
| ring_cute EP4 | 2 | pending | pending | pending | pending | pending | cluster access blocked |
| ring_cute EP8 | 1/2 | pending | pending | pending | pending | pending | optional after EP4 proof |

### Provisional ranking and overlap ceiling

1. Fixed A2A + gather remains the throughput leader on direct evidence at 20.558% p50. It is also the only layout compatible with the peer's compile-time bucket pipeline, so a parity result would favor fixed capacity.
2. `ring_cute` EP4 and ragged A2A remain unranked against each other at this shape. `ring_cute` won the e64/e128 ladders, but extrapolating that result to e256/top-8 is weaker evidence than a matched draw. Ragged has no production-shape parity control.
3. Fixed capacity's overlap ceiling is structurally higher but not yet measured. Round-robin pipelining can overlap fixed peer buckets; the present profile still pays combine/all-gather backward, stray unstack, and a final unoverlapped reduce-scatter. A peer pipeline result is required before assigning an MFU ceiling.
4. Drop fraction can overturn the throughput ranking. EP64 creates roughly 16K fixed buckets, while prior production evidence accepted about 3% drop with 8 buckets. Every new draw must report both drop count and fraction.
5. Two-rack cells remain deferred pending coordinator approval.

Confidence: 7/10 that this direction contributes a significant step toward 25% MFU; 2/10 in any transport ranking beyond the historical fixed+gather lead until matched draws run.

Next: preserve a commit as soon as the Git metadata path is writable, restore an Iris client compatible with the current config, then run one-replica EP4 transport smokes followed by fixed-1, ragged-1, ring-1, fixed-2, ragged-2, ring-2 with no overlapping rack jobs.

## Check-in 2026-07-25 05:27 UTC

Findings:

- Relay batch 1 failed before launch for all three arms because `--version ep25d2-ngc2606-v1` is not a valid Iris experiment version. The required form is a calendar version or a label ending in `-dev`.
- Relay batch 2 used `--version ep25d2-ngc2606-dev`. Fixed+gather, ragged, and `ring_cute` all reached the NGC overlay, then failed while importing CUTLASS with `ImportError: cannot import name '_cutlass_ir' from 'cutlass._mlir._mlir_libs'`.
- This checkout already resolves the current CUTLASS DSL 4.6 split: `nvidia-cutlass-dsl`, base, core, cu12, and cu13 are all 4.6.0, with `quack-kernels==0.6.1`. `uv lock` resolved the existing 608-package lock without a change. The main-branch `override-dependencies` exclusion applied only to overlapping 4.5.2 wheels and was removed when main upgraded to 4.6; porting that old exclusion here would remove a required wheel.
- The NGC image path exposed the actual bug. Its system-site-packages overlay protected `nvidia-cutlass-dsl-libs-base` from the GPU sync while allowing the 4.6 cu13 wheel, producing an incomplete CUTLASS package in the venv. The overlay now installs both required 4.6 components and asserts that both base and cu13 dist-info directories exist. A regression test failed on the old generated setup script and passes with this change.
- Round-2 results make fixed layout's prospective overlap advantage smaller: rotation measured -9.46 points, token chunking -1.96 points, and prefetch was null. Rav's leg-batching plus custom-adjoint 120-step run reached 25.39% p50. Fixed capacity still has the only structural path to future fixed-slice overlap, but no measured overlap mechanism currently increases its ceiling.
- Drop fraction remains the fidelity hedge. Until the suspected 65–68% fixed-path metric is reconciled, a slower ragged or ring transport may still be required for usable training.
- `EP25_D2_RELAY_COMMANDS.md` now contains only the corrected smoke3 handoff: unique `*-smoke3-20260725` names and `--version ep25d2-ngc2606-dev`. No job was submitted, stopped, killed, or otherwise mutated from this session.
- The complete local target suite passes with 24 tests passed and 6 GPU-only tests skipped. Ruff 0.14.3, Black 25.9.0, `git diff --check`, and `uv lock --check` pass. The repository pre-commit wrapper cannot fetch PyYAML because sandbox DNS is unavailable.
- The required local commit remains blocked by the managed filesystem. `git add` cannot create `/home/marin/projects/marin/.git/worktrees/ep25-d2-bakeoff/index.lock` because that Git metadata path is read-only. All task changes remain unstaged; no unrelated relay or coordinator artifact was staged.

### Smoke status

| Arm | Draw | p10 MFU | p50 MFU | p90 MFU | p50 tok/s | Drop count | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| fixed A2A + gather | smoke2 | unavailable | unavailable | unavailable | unavailable | unavailable | CUTLASS import failure |
| ragged A2A, one-shot off | smoke2 | unavailable | unavailable | unavailable | unavailable | unavailable | CUTLASS import failure |
| ring_cute EP4 | smoke2 | unavailable | unavailable | unavailable | unavailable | unavailable | CUTLASS import failure |
| fixed A2A + gather | smoke3 | pending | pending | pending | pending | pending | relay command ready |
| ragged A2A, one-shot off | smoke3 | pending | pending | pending | pending | pending | relay command ready |
| ring_cute EP4 | smoke3 | pending | pending | pending | pending | pending | relay command ready |

Provisional ranking: no transport ranking changes without a successful matched draw. Fixed+gather retains the 20.558% historical throughput lead, while ragged and `ring_cute` retain more decision value than their throughput evidence suggests until the fixed-path drop fraction is known.

Confidence: 8/10 that a matched transport result remains significant as a fidelity decision; 2/10 in the transport ranking beyond the historical fixed+gather lead.

Next: stop here for coordinator execution of the three smoke3 commands. If all smokes pass, prepare the interleaved two-draw rack matrix; keep two-rack cells deferred.

## Candidate ranking after round-1 all-gather

1. **1a, lock the adjoint win — 9/10.** The 25.43% grad-only signal already exceeds the locked goal and removes 544 backward scatters; a matched 120-step run with loss and drop fractions is the shortest path to a defensible result.
2. **4, rotation ppermute — 8/10.** CPU EP8 forward/all-gradient parity is complete and a GB200 smoke is in flight; it attacks the now-dominant 22.4% SendRecv region while retaining fixed capacity's overlap structure.
3. **2, matched transport bake-off — 7/10.** Post-adjoint communication remains 26–29.5% of the step, so ring_cute or ragged could contribute at least 1 point; repeated placement-controlled draws and drop fractions are still required.
4. **5, MXFP8 on the winning transport — 7/10.** The measured 1.308x within-EP8 speedup and 37% smaller arena clear the performance bar, but the held-out-loss regression and unmerged dependency make it a follow-on rather than the first locked result.
5. **1c, overlap `reduce-scatter.10` — 6/10.** The final gradient reduce-scatter is visibly unoverlapped and scheduling it under the next layer could recover a point, but no implementation or isolated timing exists yet.
6. **1b, eliminate backward `unstack` — 5/10.** It is a concrete profile lead and likely tractable, but its independent step-time share is not yet large enough to predict a 1-point gain.
7. **4b, token-chunk dispatch/FFN pipelining — 4/10.** The only landed analogue improved FSDP expert chunking from 21.8% to 22.7%, just below one point; applying it to EP64 may still compose with the adjoint.
8. **6, non-EP FA4-LSE or scan gather overlap — 4/10.** FA4-LSE is estimated near one point and scan overlap is plausible, but both are less measured than the communication candidates.
9. **3, TE-at-tip NCCL_EP rerun — 1/10.** The 64-GPU stream-pin crash is deterministic and the unpinned tip is functionally the old approximately 17% path.

The fixed-layout ceiling weighs more heavily after the all-gather: rotation and token-chunk pipelining both require stable fixed-capacity slices. A small ragged parity win would not displace fixed capacity unless its lower drop fraction or a repeated throughput margin covers that lost composition option.

Relay-ready smoke commands are in `EP25_D2_RELAY_COMMANDS.md`. They use EP4 on one four-GPU replica, NGC JAX 26.06, and the hash-verified #7421 plugin. Rack commands will be written after all three smokes reach a terminal state.

## Check-in 2026-07-25 06:37 UTC

All three smoke4 arms completed four training steps on NGC JAX 26.06 with the hash-verified #7421 plugin. The reduced d2048/L4/e256/top-8/b64 shape fit one four-GPU replica. Each log confirms CUTLASS DSL 4.6.0 from the venv, preserved NGC JAX/JAXLIB, the replacement plugin loaded, finite loss, and a tracker finish event.

| Arm | Draw | p10 MFU | p50 MFU | p90 MFU | p50 tok/s | Drop count | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| fixed A2A + gather | smoke4 | 7.072% | 7.072% | 7.151% | 174.3K | not reported | succeeded |
| ragged A2A, one-shot off | smoke4 | 6.591% | 7.019% | 7.019% | 173.0K | not reported | succeeded |
| ring_cute EP4 | smoke4 | 8.864% | 8.873% | 8.873% | 218.7K | not reported | succeeded |

The smoke4 throughput ordering is exploratory because the model, layer count, batch, and expert-axis size differ from the rack operating point. It proves the backend and environment paths only.

Commits `4fbc89152` and `2d4a87395` from `agent/ep25-d1-adjoint` are ported into the working tree without a conflict with the NGC overlay. `SCALE_REPORT_DROPS=1` now requests the backend's dropped-assignment count, sums it across 48 scanned layers, divides by global `batch * sequence * top-k * layers`, and explicitly logs `moe/dropped_assignments` and `moe/drop_fraction`.

The largest supported ring expert axis is EP64. With 16 processes, four local GPUs, one replica axis, and 256 experts, the launcher resolves the mesh to `(replica=1, data=1, expert=64, model=1)` and assigns four experts to each shard. `EP25_D2_RELAY_COMMANDS.md` now contains exactly two 120-step rack commands:

- `ep25d2-rack-ragged-120-20260725`: ragged all-to-all, one-shot disabled, EP64.
- `ep25d2-rack-ring-ep64-120-20260725`: `ring_cute`, EP64.

Both commands use d5120/L48/e256/top-8/b1024/seq4096, MuonH, 16 GB200 replicas, NGC JAX 26.06, the #7421 plugin, `json_logger`, `SCALE_REPORT_DROPS=1`, and `SCALE_DISABLE_CHECKPOINT=1`. The shell syntax checks and the launcher accepts the 64-device mesh. No fixed arm is included.

The comparison bar is d1's matched fixed path: 20.61% p50 with autodiff and 24.04% with the custom adjoint, a 3.43-point gain. Rav's custom-adjoint plus leg-batching run reached 25.39% p50. Ragged or ring must beat 24.04% to displace the fixed adjoint on current throughput. A materially lower drop fraction can still select ragged on fidelity even below that speed.

Local verification:

- 26 passed and 6 GPU-only skipped across `test_model.py`, `test_dispatch.py`, and `test_grugformer_moe.py`.
- 7/7 MoE-relevant Grug variant contracts passed, including the cross-process EP mesh and one-step lowering.
- The new CPU test confirms drop reporting leaves loss and every gradient unchanged.
- Pyrefly 1.0.0, Ruff 0.14.3, Black 25.9.0, `git diff --check`, relay-command `bash -n`, and `uv lock --check` passed.
- The repository pre-commit wrapper could not fetch PyYAML because sandbox DNS is unavailable; the cached direct checks above cover the changed files.
- The broader variant-contract file has two unrelated `base`-variant failures because its debug mesh omits the `expert` axis while shared sharding now names it. The `moe` parameterizations pass.

No job was submitted, stopped, killed, or otherwise mutated from this session.

The sandbox still cannot create `/home/marin/projects/marin/.git/worktrees/ep25-d2-bakeoff/index.lock`; the coordinator must commit `experiments/grug/moe/model.py`, `experiments/grug/moe/train.py`, `experiments/grug/moe/test_model.py`, `EP25_D2_RELAY_COMMANDS.md`, and `AGENT_LOG.md`.

### Rack decision table

| Arm | Draw | p10 MFU | p50 MFU | p90 MFU | p50 tok/s | Drop count/fraction | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| fixed A2A + gather, autodiff | d1 reference | 20.51% | 20.61% | 20.69% | not harvested | pending d1 fidelity run | succeeded |
| fixed A2A + gather, custom adjoint | d1 reference | 23.73% | 24.04% | 24.75% | not harvested | pending d1 fidelity run | succeeded |
| ragged A2A, one-shot off, EP64 | rack draw | pending | pending | pending | pending | pending | relay command ready |
| ring_cute EP64 | rack draw | pending | pending | pending | pending | pending | relay command ready |

Provisional ranking: fixed+adjoint leads on production-shape throughput at 24.04%. Ragged and `ring_cute` remain unranked at the rack shape. The reduced smoke favors `ring_cute`, but it does not narrow the expected ±2–4 point placement spread or establish a production-shape win.

Confidence: 9/10 that the two rack arms settle a transport or fidelity decision; 4/10 that `ring_cute` beats the 24.04% fixed-adjoint bar.

Next: stop for coordinator execution of the two rack commands, one at a time.

## FINAL TRANSPORT VERDICT — written by coordinator 2026-07-25 09:35 UTC
(codex backend 503-circuit-open at wrap-up time; codex may amend on recovery. All
numbers from harvested runs in relay-results/ and peer AGENT_LOGs.)

| transport (operating point, 1 rack, 120 steps) | MFU | drops (QB-off) | note |
|---|---|---|---|
| fixed + gather + custom adjoint (QB-off) | 24.04% p50 | 0.85-0.89 early | d1 matched A/B |
| fixed + gather + custom adjoint (QB-on cf1.0) | 22.60% p50 | 0.90 peak -> 0.083 @119 | d4; honest production config |
| ragged, one-shot kernel disabled | ~12.38% mean | 0.433 end-of-run | single draw; ran clean |
| ring_cute EP64 | DNF | — | OOM 141.79 GiB in jit_train_step |

Direction-2 answer: transport choice does NOT leave >=1pp on the table — fixed-capacity
+ gather dispatch + custom adjoint wins decisively at e256/top-8/EP64. The matched
ragged control #7279 asked for is on record (~12.4%, and its receiver-side capacity
still drops 43% under router collapse — ragged is not a fidelity refuge; QB balancing
is the fidelity lever on every transport). ring_cute's ladder wins at e64/e128 do not
transfer: it cannot fit this shape without memory work (a project, not tuning).
Caveats: ragged/ring arms were QB-off single draws on NGC 26.06 + #7421 plugin; fixed
numbers are stock-toolchain. No further arm is worth a rack; 2-rack cells moot.

## Round 6 R6-4 background research brief

- Effort: high
- Stop rule: stopped when the local #7282 kernel history, production wiring, numerical
  tests, and this worktree's fixed+gather+adjoint call boundary converged on one minimum
  integration.
- Date: 2026-07-25

### Question

Can the proven grouped MXFP8 expert MLP replace only the post-routing w13/w2 compute in
the d5120, e256, top-8, EP64 fixed+gather+custom-adjoint operating point?

### Evidence

- `research/mcwitt/7282-mxfp8-blackwell` commit `42f7d9fa2` measured the forward grouped
  GEMMs at about 2.2 PF/s on GB200. Commit `9e1d8fdc7` measured the fused whole-layer
  pipeline at 1.39x BF16; the earlier unfused producer path only reached break-even.
- Commit `af2110ac5` integrated the winning unit as a stateless whole-expert-MLP custom
  VJP. It pads each expert group to 256 rows after routing, uses fused
  w13+SwiGLU+dual-quantize and grouped w2 forward kernels, and provides MXFP8 dgrad and
  wgrad paths.
- The branch's GB200 numerical ladder checks every dequantized kernel leg below
  1e-3/2e-3 relative-Frobenius error, the black-box output/input/weight gradients below
  0.1 versus an independent BF16-input f32 reference, and exact-zero gradients for
  zero-token experts.
- The 64-GPU matched pair recorded in `c3cb334f8` measured a 1.308x clean MXFP8/BF16
  throughput ratio with full Newton-Schulz. The CuTe activation producer failed
  executable loading at 16 nodes in 3/3 attempts; the XLA producer succeeded and is
  therefore the production default.
- The uniform-MXFP8 branch falsified dense MXFP8 as the next extension: per-tensor FP8
  remained preferable for dense projections, while hybrid grouped-MXFP8 expert GEMMs
  were the only production-viable recipe.
- This worktree's `_ring_local` already exposes exactly the post-routing boundary
  `expert_mlp_fn(x_dispatch, group_sizes)`. Selecting the MXFP8 whole-MLP op there leaves
  gather dispatch, QB routing, capacity clipping, drop accounting, combine, and the
  custom transport adjoint unchanged.
- GitHub issue reads are currently unavailable because `api.github.com` does not
  resolve. The complete local 7282 branch/logbook history supplies the implementation
  and measurement record; no contradictory local evidence changed the design.

### Ranked approaches

1. Port the fused whole-MLP op and its vendored kernels into
   `lib/levanter/src/levanter/grug/_moe/`, then select it only inside the generic fixed
   ring expert callback when `SCALE_MOE_MXFP8=1`. This is the smallest operating-point
   integration and preserves drop parity structurally.
2. Reintroduce 7282's general `MoeExpertMlpOp` injection abstraction across every EP
   backend. This is reusable but expands the conflict and validation surface beyond
   R6-4.
3. Import the old experiment-local module directly. This is mechanically quick but
   leaves production code dependent on a standalone benchmark tree and violates the
   requested Levanter integration boundary.

Confidence: 8/10 that approach 1 can reproduce a material operating-point speedup; 6/10
that CUTLASS DSL 4.6 accepts the vendored 4.5-era kernels without a bounded port.

Next: validate the minimal design, write the test-first implementation plan, then port
the CPU glue tests before any production integration.

## Check-in 2026-07-25 21:03 UTC

Findings:

- The 7282 fused whole-MLP custom VJP and pure-JAX MXFP8 quantizer now live
  under `lib/levanter/src/levanter/grug/_moe/`. The CuTe activation producer
  and the general `MoeExpertMlpOp` abstraction were not ported.
- 29/29 CPU layout contracts pass. They cover skewed and empty expert groups,
  exact pad/unpad behavior, zero-filled padding, host-reference scale-factor
  permutations, gate/up interleave, and the non-sm100 failure boundary.
- The operating point is `_fixed_a2a_core` behind `SCALE_A2A_FIXED=1`, not
  `_ring_local`. The treatment gathers the four local-expert receive buffers,
  concatenates them into the grouped layout after routing, runs one MXFP8
  whole-MLP call, then reuses the existing combine all-to-all.
- This worktree did not contain `c9e30f848` despite the prior round log saying
  the adjoint stack was present. The structured dispatch/combine gather VJPs
  are now ported. Focused tests confirm the custom adjoint requires gather
  dispatch and matches autodiff outputs, drops, and all gradients at
  `rtol=atol=1e-5`.
- The default BF16 per-expert loop is unchanged. `SCALE_MOE_MXFP8=1` is the
  only treatment selector, and the XLA quantization producer is unconditional.
- The eleven vendored fused kernel files syntax-compile locally. Focused Ruff
  passes; vendored NVIDIA kernel bodies carry a file-level Ruff exclusion and
  the Marin adapter remains linted.

Confidence: 8/10 that the integration preserves routing and drop counts; 6/10
that the vendored kernels lower unchanged on the current CUTLASS DSL 4.6
toolchain.

Next: run the complete CPU regression set, check the CUTLASS 4.6 adapter
surface, then prepare the GB200 numerical and EP4 parity jobs.

## Check-in 2026-07-25 21:20 UTC

Findings:

- The final focused CPU suite reports 51 passed and 6 skipped in 37.09s:
  `test_mxfp8_expert_mlp.py`, `test_grugformer_moe.py`, and
  `experiments/grug/moe/test_model.py`. The skips are existing
  accelerator/multi-device conditions.
- The exact task sources pass Ruff lint and format, Python bytecode
  compilation, `git diff --check`, and Pyrefly 0.58 with zero errors. The
  repository pre-commit wrapper reaches and passes Ruff plus all structural
  checks, but its Black subprocess hangs in this sandbox even with
  `BLACK_NUM_WORKERS=1`; it was stopped after repeated no-output waits.
- Iris submission is blocked before controller access because OAuth cannot
  resolve `oauth2.googleapis.com`. No cluster jobs were submitted or mutated.
  `EP25_D2_RELAY_COMMANDS.md` now contains the gated stock-toolchain ladder:
  `ep25d2-mxfp8-numerics-20260725`, matched EP4 BF16/treatment smokes, then
  matched QB-on cf1.0 120-step rack BF16/treatment jobs. All shell blocks pass
  `bash -n`.
- The linked-worktree Git index remains read-only:
  `/home/marin/projects/marin/.git/worktrees/ep25-d2-bakeoff/index.lock`
  cannot be created. The coordinator must commit the exact paths below,
  forcing the ignored `lib/` additions:

```text
AGENT_LOG.md
EP25_D2_RELAY_COMMANDS.md
docs/superpowers/specs/2026-07-25-ep25-mxfp8-expert-gemms-design.md
docs/superpowers/plans/2026-07-25-ep25-mxfp8-expert-gemms.md
experiments/grug/moe/standalone/check_mxfp8_expert_mlp.py
lib/levanter/src/levanter/grug/_moe/ep_ragged_all_to_all.py
lib/levanter/src/levanter/grug/_moe/mxfp8.py
lib/levanter/src/levanter/grug/_moe/mxfp8_kernels/__init__.py
lib/levanter/src/levanter/grug/_moe/mxfp8_kernels/quantize.py
lib/levanter/src/levanter/grug/_moe/mxfp8_kernels/fused/__init__.py
lib/levanter/src/levanter/grug/_moe/mxfp8_kernels/fused/adapter.py
lib/levanter/src/levanter/grug/_moe/mxfp8_kernels/fused/grouped_gemm_dswiglu_quant.py
lib/levanter/src/levanter/grug/_moe/mxfp8_kernels/fused/grouped_gemm_quant.py
lib/levanter/src/levanter/grug/_moe/mxfp8_kernels/fused/grouped_gemm_swiglu_quant.py
lib/levanter/src/levanter/grug/_moe/mxfp8_kernels/fused/moe_blockscaled_grouped_gemm_wgrad.py
lib/levanter/src/levanter/grug/_moe/mxfp8_kernels/fused/moe_kernel_helpers.py
lib/levanter/src/levanter/grug/_moe/mxfp8_kernels/fused/moe_persistent_scheduler.py
lib/levanter/src/levanter/grug/_moe/mxfp8_kernels/fused/moe_sched_extension.py
lib/levanter/src/levanter/grug/_moe/mxfp8_kernels/fused/moe_utils.py
lib/levanter/src/levanter/grug/_moe/mxfp8_kernels/fused/utils.py
lib/levanter/tests/grug/test_grugformer_moe.py
lib/levanter/tests/grug/test_mxfp8_expert_mlp.py
```

Confidence: 9/10 that the CPU-side integration and relay A/B isolation are
correct; 6/10 that the stock GB200 CUTLASS DSL accepts the vendored kernel
unchanged. The first relay job is the explicit decision gate for that remaining
uncertainty.

Next: stop for coordinator execution of the relay ladder. Do not submit the EP4
or rack jobs unless the preceding numerical/drop-parity gate passes.

## Check-in 2026-07-25 22:08 UTC

Findings:

- The relayed numerical job failed before executing its comparisons:
  CUTLASS DSL's libNVVM backend could not compile generated device IR for
  `sm_100a`. The frozen bundle export installed both the CUDA 12 and CUDA 13
  CUTLASS DSL payloads.
- The 7282 branches solve the 4.5.2 form of this problem by forcing
  `nvidia-cutlass-dsl-libs-base` inactive. Main now uses CUTLASS DSL 4.6.0,
  where base/core are shared and the default CUDA payload moved to
  `libs-cu12`. The version-equivalent fix therefore forces
  `nvidia-cutlass-dsl-libs-cu12==4.6.0` inactive and retains base/core/cu13.
- `uv lock --check --offline` resolves 608 packages. A frozen
  `marin-levanter[gpu]` export now contains CUTLASS DSL 4.6.0 plus
  `libs-base`, `libs-core`, and `libs-cu13`; `libs-cu12` is absent.
- The vendored kernel bodies match the final uniform-MXFP8 branch apart from
  file-level Ruff exclusions. The adapter already contains `4876d9670`'s
  generic-to-gmem pointer normalization, so no additional addrspace source
  change was needed.
- The final requested CPU regression reports 51 passed and 6 skipped in 37.47
  seconds.
- `EP25_D2_RELAY_COMMANDS.md` contains the corrected five-job ladder with fresh
  `-v2` names and a fresh `ep25d2-mxfp8-r6-v2-dev` training bundle version.
  No job was submitted from this sandbox.
- The linked-worktree Git index is still read-only:
  `/home/marin/projects/marin/.git/worktrees/ep25-d2-bakeoff/index.lock`
  cannot be created. The coordinator must commit exactly:

```text
AGENT_LOG.md
EP25_D2_RELAY_COMMANDS.md
docs/debug-log-ep25-mxfp8-cutlass-env.md
pyproject.toml
uv.lock
```

Confidence: 9/10 that the shipped environment now selects the CUDA 13 CUTLASS
compiler deterministically; 7/10 that the corrected environment clears the
GB200 compiler gate, pending the `-v2` numerical relay.

Next: stop for coordinator execution of
`ep25d2-mxfp8-numerics-20260725-v2`. Do not submit the EP4 or rack jobs unless
the preceding numerical/drop-parity gate passes.

## Check-in 2026-07-25 22:20 UTC

`ep25d2-mxfp8-numerics-20260725-v2` failed after 12.29 seconds with the same
libNVVM `sm_100a` diagnostic. The 4.6.0 CUDA 12 payload exclusion did not clear
the compiler gate. Because v2 predates the new sentinel, it does not reveal
the loaded `_cutlass_ir` extension or libNVVM path.

The 7282 record identifies `/mwittmann/mxfp8-002-g8` as the first 2.2 PF/s
green job at commit `42f7d9fa2`. It used the stock aarch64 CUTLASS DSL 4.5.2
wheel through a one-GB200 `--extra gpu` submit. The record has no explicit
`CUDA_TOOLKIT_PATH` or wheel-install step. Its configured task image was the
mutable `ghcr.io/marin-community/iris-task:latest`; the pulled digest, expanded
environment, libNVVM path, setup log, and installed wheel hashes are not
retained. GitHub API access to issue comments also failed in this sandbox.

The current Levanter GPU extra pins QuACK 0.6.1 and CUTLASS DSL 4.6.0, so the
known-green 4.5.2 dependency set cannot be copied into the bundled environment
without new isolation work. `EP25_D2_MXFP8_SCOPING.md` records the exact
missing artifacts and estimates 1–2 hours if the archived g8 job metadata is
available, or 4–8 hours plus queue time for a new 4.5.2 isolation or 4.6.0
port.

The numerical script now emits a machine-readable `CUTLASS_ENV_SENTINEL`
before compilation. It reports module and loaded extension paths, CUTLASS
dist-info names and versions, payload ownership from wheel `RECORD` hashes,
`CUDA_TOOLKIT_PATH`, `LD_LIBRARY_PATH`, and the libNVVM path selected by
`cuda.pathfinder`. The CPU regression reports 52 passed and 6 accelerator or
multi-device skips in 37.39 seconds. Targeted Ruff lint and formatting,
`git diff --check`, and `uv lock --check --offline` also pass.

No v3 command was emitted. `EP25_D2_RELAY_COMMANDS.md` now explicitly blocks
the stale v2 EP4/rack ladder and points to the scoping assessment. No cluster
job was submitted or mutated from this sandbox.

The coordinator must commit exactly:

```text
AGENT_LOG.md
EP25_D2_MXFP8_SCOPING.md
EP25_D2_RELAY_COMMANDS.md
docs/debug-log-ep25-mxfp8-cutlass-env.md
experiments/grug/moe/standalone/check_mxfp8_expert_mlp.py
experiments/grug/moe/standalone/test_check_mxfp8_expert_mlp.py
```

Confidence: 9/10 that the retained record is insufficient for a verbatim
green-environment reconstruction; 8/10 that the sentinel will identify the
next environment mismatch before compiler output.

Next: stop under the round-6 fourth-attempt rule. Resume only with the archived
g8 image/job metadata or approval for new 4.5.2-isolation/4.6.0-port work.

## Check-in 2026-07-25 22:41 UTC

The coordinator's known-green worktree comparison superseded the prior
scoping stop. The root UV override now matches
`research/mcwitt/7282-uniform-mxfp8` exactly:
`nvidia-cutlass-dsl-libs-base==4.5.2 ; sys_platform == 'never'`, sourced from
`https://pypi.nvidia.com/`. Both Marin and Levanter GPU extras constrain
`nvidia-cutlass-dsl[cu13]>=4.5.2,<4.6`. The direct QuACK 0.6.1 dependency was
removed because it forces CUTLASS DSL 4.6.0; the green graph resolves
transitive QuACK 0.5.0.

The regenerated lock resolves 606 packages. Its DSL/base/cu13 blocks are
identical to the known-green branch. A frozen Levanter GPU export contains
CUTLASS DSL 4.5.2, `libs-cu13` 4.5.2, and QuACK 0.5.0; it excludes
`libs-base`, `libs-core`, and `libs-cu12` from the Linux GPU environment.
`uv lock --check --offline` and a full offline `uv lock` both pass.

The CPU regression reports 52 passed and 6 accelerator or multi-device skips
in 37.51 seconds. Targeted Ruff lint and formatting, TOML parsing, and
`git diff --check` pass. Importing the vendored adapter against the cached
CUTLASS 4.5.2/cu13 payload also passes. The kernels needed no import changes;
they originated from the 4.5.2 lineage and retain its generic-address-space
fix.

`EP25_D2_RELAY_COMMANDS.md` now contains only
`ep25d2-mxfp8-numerics-20260725-v3`. The numerical script prints the
machine-readable CUTLASS environment sentinel before compilation. The stale
EP4 and rack submissions remain blocked until numerics is green. No cluster
job was submitted or mutated from this sandbox.

The linked-worktree index remains read-only:
`/home/marin/projects/marin/.git/worktrees/ep25-d2-bakeoff/index.lock` cannot
be created. The coordinator must commit exactly:

```text
AGENT_LOG.md
EP25_D2_MXFP8_SCOPING.md
EP25_D2_RELAY_COMMANDS.md
docs/debug-log-ep25-mxfp8-cutlass-env.md
experiments/grug/moe/standalone/test_check_mxfp8_expert_mlp.py
lib/levanter/pyproject.toml
lib/marin/pyproject.toml
pyproject.toml
uv.lock
```

Confidence: 10/10 that the shipped bundle now reproduces the known-green
CUTLASS 4.5.2 dependency resolution; 8/10 that v3 clears the GB200 compiler
gate, pending the relay.

Next: stop for coordinator execution of
`ep25d2-mxfp8-numerics-20260725-v3`. Regardless of its result, operational
friction is a checkpoint escalation and does not close the MXFP8 direction
under the amended round-6 fleet policy.

## Check-in 2026-07-25 22:49 UTC

`/mwittmann/ep25d2-mxfp8-numerics-20260725-v3` succeeded on one GB200 with
exit 0, zero failures, and zero preemptions in 1 minute 2.44 seconds. The
coordinator reports that the numerical checks passed. This clears the
libNVVM `sm_100a` compiler gate for the CUTLASS DSL 4.5.2/cu13 resolution.

Log-server ingestion is degraded for new GB200 jobs. The v3 Iris summary
contains the terminal job state but no task logs, so the
`CUTLASS_ENV_SENTINEL` line is not currently harvestable.

`SCALE_JSON_LOGGER` is passed to `JsonLoggerConfig(logger_name=...)`.
`JsonLoggerTracker` sends its JSON records through `logger.info`; the value is
not a file or S3 path. Levanter has a separate `JsonFileTracker` backed by
`StoragePath`, but `launch_cw_scale.py` does not expose or compose that tracker.
The remaining commands therefore retain stdout JSON metrics. EP4 and rack
metric harvests may need to wait for log-shipping recovery.

`EP25_D2_RELAY_COMMANDS.md` contains the four remaining v3 jobs:

```text
ep25d2-mxfp8-ep4-bf16-20260725-v3
ep25d2-mxfp8-ep4-treatment-20260725-v3
ep25d2-mxfp8-rack-bf16-120-20260725-v3
ep25d2-mxfp8-rack-treatment-120-20260725-v3
```

The EP4 pair uses QB-on, fixed gather dispatch, the custom adjoint, capacity
factor 1.0, and drop reporting. The matched 120-step rack pair uses the same
protocol at d5120, 48 layers, 8-of-256, EP64, batch 1024, and sequence length
4096. Only the treatment legs enable `SCALE_MOE_MXFP8=1`. The rack pair
remains gated on terminal EP4 jobs with harvested drop parity.

No cluster job was submitted or mutated from this sandbox.

The linked-worktree index remains read-only:
`/home/marin/projects/marin/.git/worktrees/ep25-d2-bakeoff/index.lock` cannot
be created. The coordinator must commit exactly:

```text
AGENT_LOG.md
EP25_D2_RELAY_COMMANDS.md
```

Confidence: 10/10 that CUTLASS DSL 4.5.2/cu13 clears the GB200 numerical
compiler gate; 9/10 that the remaining commands reproduce the matched
round-6 protocol.

Next: stop for coordinator submission of the EP4 pair. Submit one rack job at
a time only after EP4 drop parity is confirmed. If metrics are unavailable,
wait for log-shipping recovery before advancing.
