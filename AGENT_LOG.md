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
