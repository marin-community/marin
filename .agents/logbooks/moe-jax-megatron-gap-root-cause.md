# JAX vs Megatron DeepEP Gap Root Cause: Research Logbook

## Scope
- Goal: determine the precise root causes of the remaining throughput gap between JAX DeepEP and Megatron DeepEP on the same fixed-shape H100x8 MoE block regime.
- Primary metric(s): `time_s`, `tokens/s`, and isolated phase timings for routing/layout, transport, local expert compute, and framework/runtime overhead.
- Constraints:
  - use the sealed `#3717` head-to-head matrix as the starting baseline
  - optimize for quick-turnaround experiments that eliminate false hypotheses early
  - only compare apples-to-apples shapes, passes, and token counts
  - update the public issue thread only for major milestones/discoveries
- GitHub issue: https://github.com/marin-community/marin/issues/3752

## Baseline
- Date: 2026-03-16
- Code refs:
  - `lib/levanter/scripts/bench/bench_moe_hillclimb.py`
  - `.agents/scripts/deepep_jax_krt_bench.py`
  - `.agents/scripts/megatron_qwen_krt_bench.py`
  - `.agents/scripts/moe_block_head_to_head.py`
- Baseline numbers:
  - Sealed `#3717` fixed-shape same-global-token head-to-head (`forward_backward`, H100x8, `distribution=random`):

```text
| tokens | topk | jax_current | jax_deepep | megatron_alltoall | megatron_deepep |
| 32768  | 2    |   2,957,986 |  3,312,032 |            537,099|         2,644,307|
| 32768  | 8    |     959,217 |  1,115,553 |            590,120|         1,965,951|
| 65536  | 2    |   3,156,598 |  3,752,354 |            785,280|         2,660,216|
| 65536  | 8    |     961,693 |  1,116,306 |          1,315,403|         3,088,961|
| 131072 | 2    |   3,228,500 |  3,824,437 |          3,454,152|        12,712,443|
| 131072 | 8    |     965,449 |  1,057,153 |          2,489,744|         6,577,851|
| 262144 | 2    |   3,247,008 |  3,821,071 |          6,661,309|        15,830,875|
| 262144 | 8    |         OOM |    985,350 |          6,426,325|         7,652,599|
```

## Experiment Log
### 2026-03-16 17:25 - Kickoff
- Hypothesis:
  - the remaining JAX-vs-Megatron DeepEP gap is attributable to a specific combination of benchmark-path overheads rather than a single unexplained transport failure.
- Command:
  - kickoff/scaffolding only; no benchmark command yet
- Config:
  - refreshed from `origin/main` at `69d30f1c11fcb3ad3349f594f59766b7081370b0`
  - branch: `research/moe-jax-megatron-gap-root-cause`
  - worktree: `/Users/romain/marin-wt/moe-jax-megatron-gap-root-cause`
- Result:
  - created the new research worktree from refreshed `main`
  - initialized the local research logbook
- Interpretation:
  - this thread starts from a clean base and a sealed benchmark table rather than from an unresolved bring-up problem
- Next action:
  - create the experiment issue and write the public baseline/decision scaffolding before defining the first quick-turnaround experiment matrix

### 2026-03-16 16:35 - Shared-expert mismatch check on the large-gap cell
- Hypothesis:
  - one major cross-framework mismatch is that the JAX benchmark includes `shared_expert_dim=2048`, while the Megatron harness does not obviously enable a shared-expert branch; if that is a dominant factor, zeroing the JAX shared-expert path should collapse a large fraction of the JAX-vs-Megatron gap immediately.
- Command:

```bash
KUBECONFIG=/Users/romain/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
  --task-id jax-gap-shared0-t131072-topk2-20260316-163508 \
  --tokens 131072 \
  --shared-expert-dim 0 \
  --kernels current,deepep_transport_capped_prewarmed \
  --topk-list 2 \
  --distributions random \
  --bench-pass forward_backward \
  --ep-list 8 \
  --warmup 1 \
  --iters 3 \
  --per-bench-timeout-seconds 420 \
  --per-bench-kill-after-seconds 20 \
  --build-with-torch-extension \
  --load-as-python-module \
  --skip-smoke
```

- Config:
  - H100x8, isolated CoreWeave lane `iris-3677-jax`
  - `tokens=131072`, `hidden=2048`, `mlp_dim=768`, `experts=128`, `topk=2`
  - `distribution=random`
  - `bench_pass=forward_backward`
  - comparison against sealed `#3717` same-cell baseline with `shared_expert_dim=2048`
- Result:

```text
shared_expert_dim=0:
  current: 0.037423 s, 3,502,411.86 tok/s
  deepep_transport_capped_prewarmed: 0.030084 s, 4,356,856.14 tok/s

sealed shared_expert_dim=2048 baseline:
  current: 0.040598 s, 3,228,500.34 tok/s
  deepep_transport_capped_prewarmed: 0.034272 s, 3,824,436.67 tok/s
```

- Interpretation:
  - removing the shared-expert branch improved JAX materially, but only modestly:
    - `current`: about `1.08x`
    - `deepep`: about `1.14x`
  - that makes the shared-expert mismatch real, but clearly not the main explanation for the much larger JAX-vs-Megatron gap on this cell
- Next action:
  - split the remaining gap into forward vs backward first, because the shared-expert mismatch is not large enough to explain the large-token head-to-head deltas

### 2026-03-16 16:39 - Forward-only split on the same shared-free cell
- Hypothesis:
  - if the remaining JAX-vs-Megatron gap is mostly a backward problem, then JAX forward-only on the shared-free cell should land much closer to the Megatron forward number than the `forward_backward` total does.
- Command:

```bash
KUBECONFIG=/Users/romain/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
  --task-id jax-gap-shared0-fwd-t131072-topk2-20260316-163909 \
  --tokens 131072 \
  --shared-expert-dim 0 \
  --kernels current,deepep_transport_capped_prewarmed \
  --topk-list 2 \
  --distributions random \
  --bench-pass forward \
  --ep-list 8 \
  --warmup 1 \
  --iters 3 \
  --per-bench-timeout-seconds 420 \
  --per-bench-kill-after-seconds 20 \
  --build-with-torch-extension \
  --load-as-python-module \
  --skip-smoke
```

- Config:
  - same H100x8 cell as above, but `bench_pass=forward`
  - comparison against the sealed Megatron forward component on the same token/top-k case (`forward_ms=5.115853 ms` for `megatron_deepep`)
- Result:

```text
shared_expert_dim=0, forward:
  current: 0.013146 s, 9,970,631.70 tok/s
  deepep_transport_capped_prewarmed: 0.010900 s, 12,025,297.24 tok/s

comparison point from sealed Megatron baseline:
  megatron_deepep forward: 0.005116 s, about 25.62M tok/s
```

- Interpretation:
  - the remaining large gap is already present in forward:
    - JAX DeepEP forward is still about `2.13x` slower than Megatron DeepEP forward on this shared-free same-token cell
  - backward is still likely important, but it is not the only remaining source of the gap
  - this shifts the next best experiment toward phase-level forward attribution rather than further shared-expert investigation
- Next action:
  - run the existing staged JAX DeepEP probe kernels on the same cell to isolate transport-only, assignment packing, and ragged-dot compute contributions

### 2026-03-16 16:51 - First staged ladder said the residual gap was post-transport, but the local-shape attribution was still provisional
- Hypothesis:
  - the remaining JAX-vs-Megatron forward gap can be localized by timing progressively fuller DeepEP probe kernels on the same shared-free `tokens=131072, topk=2` cell.
- Config:
  - same authoritative cell as the previous entry:
    - `tokens=131072`
    - `hidden=2048`
    - `mlp_dim=768`
    - `experts=128`
    - `shared_expert_dim=0`
    - `topk=2`
    - `distribution=random`
    - `bench_pass=forward`
    - `ep=8`
- Result:
  - early staged ladder readings:
    - transport-only: about `64.7M tok/s`
    - transport + assignment packing/collapse/combine: about `17.4M tok/s`
    - gate probe: about `2.74M tok/s`
    - first ragged-dot probe: about `2.73M tok/s`
  - Megatron forward comparison point on the same shared-free cell stayed about `25.62M tok/s`
- Interpretation:
  - this was the first strong signal that the remaining gap was not in raw DeepEP transport
  - the provisional read at this moment was that the first local expert GEMM was where the forward path collapsed
- Caveat:
  - these numbers were later found to be quantitatively confounded because the probe kernels were still using inflated uncapped local-assignment shapes
- Next action:
  - make the staged probes share the same exact-cap dispatch and local-assignment limits as the real capped DeepEP path before trusting the phase attribution

### 2026-03-16 17:05 - First exact-cap probe patch landed and was pushed for cluster visibility
- Problem:
  - the staged probes were timing oversized local buffers rather than the real exact-cap shapes used by the capped DeepEP fast path
- Commit:
  - `1403fc03d` — `Cap DeepEP probe kernels to real local shapes`
- Code refs:
  - `lib/levanter/scripts/bench/bench_moe_hillclimb.py`
- Result:
  - the patch threaded `max_recv_tokens` and `max_local_assignments` into:
    - `deepep_transport_identity`
    - `deepep_transport_assignments_identity`
    - `deepep_transport_first_ragged_dot_probe`
    - `deepep_transport_gate_probe`
    - `deepep_transport_second_ragged_dot_probe`
  - the patch was pushed before rerunning because the CoreWeave launcher downloads a GitHub tarball at the branch ref, not local uncommitted files
- Rerun bundle:

```text
POD_NAME=iris-task-e8fc72427ad9
TASK_ID=jax-gap-breakdown-capped-fwd-t131072-topk2-20260316-170732
REPO_REF=1403fc03dd0d3c80819e2f25b09c0d1b1ab63290
```

- Interpretation:
  - from this point onward, exact-cap stage timing was the only attribution path worth trusting
- Next action:
  - run the short breakdown bundle and verify that the corrected probes now report exact-cap timings rather than inflated local-shape costs

### 2026-03-16 17:09 - The first exact-cap patch failed under JIT tracing, so cap computation moved out to the timing layer
- Problem:
  - the first patch still computed exact caps inside the jitted `_forward` path
  - that path eventually called `jax.device_get(...)` on traced values and failed with `TracerArrayConversionError`
- Failure detail:
  - `_deepep_transport_exact_caps(...)` could not safely run inside the traced forward function because its host-side exact-cap computation depends on concrete values of `selected_experts`
- Commit:
  - `03df173ef` — `Time DeepEP probes with precomputed exact caps`
- Code refs:
  - `lib/levanter/scripts/bench/bench_moe_hillclimb.py`
- Result:
  - the fix introduced `_time_deepep_transport_probe_forward(...)`
  - exact caps were computed once in the non-jitted timing layer
  - the probe functions then received `max_recv_tokens` and `max_local_assignments` as explicit arguments
- Rerun bundle:

```text
POD_NAME=iris-task-dd804ef6b0ff
TASK_ID=jax-gap-breakdown-capped2-fwd-t131072-topk2-20260316-171515
REPO_REF=03df173ef7dfb5d31320a7aba9659a7f659fe9bc
```

- Interpretation:
  - this was the final harness state of the frozen session
- Next action:
  - rerun the same short exact-cap ladder from the corrected commit and treat those results as the new authoritative phase breakdown

### 2026-03-16 17:13 - First corrected exact-cap ladder results moved the residual gap past transport and basic packing
- Config:
  - same authoritative shared-free large-gap cell:
    - `tokens=131072`
    - `hidden=2048`
    - `mlp_dim=768`
    - `experts=128`
    - `shared_expert_dim=0`
    - `topk=2`
    - `distribution=random`
    - `bench_pass=forward`
    - `ep=8`
  - repo ref: `03df173ef7dfb5d31320a7aba9659a7f659fe9bc`
- Result:

```text
DEEPEP_EXACT_CAPS max_recv_tokens=30976 max_local_assignments=33024 recv_factor=4.231405 assign_factor=7.937984
RESULT kernel=deepep_transport_assignments_identity ep=8 pass=forward time_s=0.002825 tokens_per_s=46402960.98
RESULT kernel=deepep_transport_first_ragged_dot_probe ep=8 pass=forward time_s=0.008515 tokens_per_s=15392690.83
```

- Interpretation:
  - the earlier uncapped assignment-stage number (`~17.4M tok/s`) was not trustworthy as a final attribution result
  - once timed at exact-cap shapes, transport plus assignment packing/collapse/combine jumped to about `46.4M tok/s`
  - that pushed the remaining JAX-vs-Megatron gap decisively past transport and basic packing
  - the first ragged-dot probe was still substantially slower at about `15.39M tok/s`, so the residual loss had moved into or beyond the first local expert compute stage
  - the earlier uncapped `~2.73M` first-ragged reading was directionally useful but quantitatively too pessimistic
- Operational note:
  - the pod logs still emitted large blocks of XLA/CUDA teardown noise after successful stage results
  - those teardown errors were noisy but did not negate the already printed benchmark outputs
- Next action:
  - finish the exact-cap ladder on the same cell with:
    - `deepep_transport_gate_probe`
    - `deepep_transport_second_ragged_dot_probe`
    - `deepep_transport_capped_prewarmed`
    - `current`

### 2026-03-17 20:21 UTC - Session recovery verified the exact resume point before new runs
- Hypothesis:
  - the frozen session already reached a stable enough attribution point to continue directly from the missing exact-cap ladder stages, without re-running earlier shared-expert or uncapped probe work.
- Command:
  - recovery-only verification; no benchmark launched yet
- Config:
  - requested recovery artifacts read in order:
    - `.agents/projects/codex-session-019ce973-resume-brief-20260317.md`
    - `.agents/logbooks/moe-jax-megatron-gap-root-cause.md`
    - `/tmp/codex-session-019ce973/history-session-019ce973.jsonl`
    - `/tmp/codex-session-019ce973/rollout-2026-03-13T16-06-04-019ce973-283f-7a20-8a01-972be62bf497.jsonl`
  - current branch state:
    - `HEAD=65eb30baee3b5c782c96b9f647d3ab932681a074`
    - branch `research/moe-jax-megatron-gap-root-cause`
    - `origin/research/moe-jax-megatron-gap-root-cause`
  - diff from authoritative harness commit `03df173ef`:
    - only docs/logbook recovery files changed
    - no benchmark harness code drift beyond the sealed exact-cap timing fix
- Result:
  - the current worktree matches the checkpoint docs:
    - clean against origin
    - `65eb30bae` is a docs-only checkpoint commit on top of `03df173ef`
  - the rollout transcript confirms the last trustworthy exact-cap outputs on the authoritative shared-free forward cell:

```text
DEEPEP_EXACT_CAPS max_recv_tokens=30976 max_local_assignments=33024 recv_factor=4.231405 assign_factor=7.937984
RESULT kernel=deepep_transport_assignments_identity ep=8 pass=forward time_s=0.002825 tokens_per_s=46402960.98
RESULT kernel=deepep_transport_first_ragged_dot_probe ep=8 pass=forward time_s=0.008515 tokens_per_s=15392690.83
```

  - later turns did not recover any additional trustworthy conclusions:
    - repeated `task_complete` events had `last_agent_message=null`
    - two later turns were explicitly interrupted
  - the remaining missing exact-cap stages are still:
    - `deepep_transport_gate_probe`
    - `deepep_transport_second_ragged_dot_probe`
    - `deepep_transport_capped_prewarmed`
    - `current`
- Interpretation:
  - the exact-cap transport-plus-packing stage is too fast to explain the remaining JAX-vs-Megatron forward gap
  - the first ragged-dot probe reintroduces a large cost, but the gap between `15.39M tok/s` and the full JAX forward point near `12.03M tok/s` means later exact-cap stages still matter
  - there is no transcript evidence of any hidden post-recovery benchmark result after those two exact-cap numbers
- Next action:
  - resume from the missing exact-cap ladder stages rather than restarting the thread
  - prefer shorter bundles because the recovered pod logs again showed heavy teardown noise after successful stage outputs

### 2026-03-17 20:35 UTC - Resume environment restored, but the lane had to be re-warmed before new exact-cap runs
- Hypothesis:
  - the corrected exact-cap ladder can resume directly once the isolated `#3677` CoreWeave lane is schedulable again; the only meaningful blockers should be operational rather than experimental.
- Command:
  - recovery and lane bring-up only; benchmark pod not launched yet
- Config:
  - kubeconfig restored at `~/.kube/coreweave-iris`
  - ops guide restored at `~/llms/cw_ops_guide.md`
  - exact-cap benchmark still planned for the same authoritative cell:
    - `tokens=131072`
    - `shared_expert_dim=0`
    - `topk=2`
    - `distribution=random`
    - `bench_pass=forward`
    - `ep=8`
- Result:
  - local environment restore succeeded:
    - `kubectl` was installed locally via the official binary into `~/.local/bin`
    - kube context resolved to `208261-marin`
    - namespace `iris-3677-jax` is reachable again
  - the lane was not initially schedulable:
    - both controller pods were pending
    - no `i3677jax-*` nodepools existed
  - `uv run iris --config=lib/iris/examples/coreweave-moe-jax-3677.yaml cluster start` was not usable in this shell because it built the worker image and then failed pushing the pinned GHCR tag with unauthenticated registry access
  - bypassing the CLI image-build/push wrapper and calling the CoreWeave `start_controller` path directly from Python did make progress:
    - `i3677jax-cpu-erapids` was created with `TARGET=1`
    - `i3677jax-h100-8x` was created with `TARGET=0`
    - the controller deployment rolled to a new pod and triggered autoscale-up on the `i3677jax` CPU nodepool
  - current cluster state at this log entry:

```text
nodepools:
  i3677jax-cpu-erapids: TARGET=1 INPROGRESS=1 CURRENT=0
  i3677jax-h100-8x:     TARGET=0 INPROGRESS=0 CURRENT=0

pods:
  iris-controller-5dd9bf995b-kpxll: Pending
```

- Interpretation:
  - the research state is still intact; there is no new benchmark result yet
  - the immediate blocker is now only CoreWeave lane warm-up
  - the direct platform call is a viable workaround for missing GHCR push auth when the cluster resources already exist and only the lane/nodepools need to be recreated
- Next action:
  - wait for `i3677jax-cpu-erapids` to reach `CURRENT=1` and for the controller pod to schedule
  - then launch the short exact-cap benchmark pod for:
    - `deepep_transport_gate_probe`
    - `deepep_transport_second_ragged_dot_probe`

### 2026-03-17 20:46 UTC - Corrected exact-cap gate and second ragged-dot stages ran, but with DeepEP/CUDA failure noise
- Hypothesis:
  - the remaining exact-cap ladder drop after the first ragged-dot stage is more likely in the second ragged-dot work than in the gate itself.
- Command:
  - `KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py --config lib/iris/examples/coreweave-moe-jax-3677.yaml --task-id jax-gap-gate-seconddot-fwd-t131072-topk2-20260317-2038 --tokens 131072 --shared-expert-dim 0 --kernels deepep_transport_gate_probe,deepep_transport_second_ragged_dot_probe --topk-list 2 --distributions random --bench-pass forward --ep-list 8 --warmup 1 --iters 3 --per-bench-timeout-seconds 420 --per-bench-kill-after-seconds 20 --build-with-torch-extension --load-as-python-module --skip-smoke --skip-cleanup`
- Config:
  - authoritative exact-cap cell:
    - `tokens=131072`
    - `shared_expert_dim=0`
    - `topk=2`
    - `distribution=random`
    - `bench_pass=forward`
    - `ep=8`
    - `warmup=1`
    - `iters=3`
  - exact caps reported in-pod:
    - `max_recv_tokens=30976`
    - `max_local_assignments=33024`
    - `recv_factor=4.231405`
    - `assign_factor=7.937984`
- Result:
  - pod `iris-task-0cf9cbceadba` scheduled directly onto cluster node `gd92fe2`, confirming again that the benchmark launcher does not depend on the isolated Iris controller lane
  - the two previously missing mid-ladder stages emitted:

```text
RESULT kernel=deepep_transport_gate_probe ep=8 pass=forward time_s=0.008673 tokens_per_s=15111835.91
RESULT kernel=deepep_transport_second_ragged_dot_probe ep=8 pass=forward time_s=0.010359 tokens_per_s=12653403.81
```

  - the same pod also logged a large amount of failure noise around the gate-probe completion boundary:
    - `DeepEP timeout check failed`
    - repeated CUDA cleanup failures with `CUDA_ERROR_LAUNCH_FAILED: unspecified launch failure`
  - despite that, the harness reported `bench_status=0`, printed `BENCH_END kernel=deepep_transport_gate_probe`, and continued into the second ragged-dot probe, which also emitted a numeric result
  - as of this log entry the pod/session had still not exited cleanly, so the bundle remained operationally noisy even after both results printed
- Interpretation:
  - relative to the earlier trustworthy exact-cap stages:
    - assignments identity: `46.40M tok/s`
    - first ragged-dot probe: `15.39M tok/s`
  - the gate probe at `15.11M tok/s` is close to the first ragged-dot figure, so the gate itself does not currently look like the dominant remaining drop
  - the larger newly observed drop is from gate probe to second ragged-dot probe (`15.11M -> 12.65M tok/s`), which points at the second ragged-dot stage as the next main culprit
  - because the pod also showed DeepEP timeout/CUDA launch-failure cleanup noise, keep the numbers but treat them as slightly lower-confidence than the earlier exact-cap stages
- Next action:
  - note: the pod later reached `Succeeded` with container exit code `0` at `2026-03-17T20:46:44Z`, so the bundle did terminate successfully despite the failure-noise logs
  - after that, finish the ladder with the remaining exact-cap stages:
    - `deepep_transport_capped_prewarmed`
    - `current`

### 2026-03-17 21:04 UTC - Corrected exact-cap ladder completed on the authoritative shared-free forward cell
- Hypothesis:
  - the corrected exact-cap ladder would show whether most remaining loss after the first ragged-dot stage comes from deeper ragged-dot work or from the full `current` path around it.
- Command:
  - `KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py --config lib/iris/examples/coreweave-moe-jax-3677.yaml --task-id jax-gap-finalpair-fwd-t131072-topk2-20260317-2050 --tokens 131072 --shared-expert-dim 0 --kernels deepep_transport_capped_prewarmed,current --topk-list 2 --distributions random --bench-pass forward --ep-list 8 --warmup 1 --iters 3 --per-bench-timeout-seconds 420 --per-bench-kill-after-seconds 20 --build-with-torch-extension --load-as-python-module --skip-smoke --skip-cleanup`
- Config:
  - same authoritative exact-cap cell:
    - `tokens=131072`
    - `shared_expert_dim=0`
    - `topk=2`
    - `distribution=random`
    - `bench_pass=forward`
    - `ep=8`
    - `warmup=1`
    - `iters=3`
  - exact caps reported in-pod:
    - `max_recv_tokens=30976`
    - `max_local_assignments=33024`
    - `recv_factor=4.231405`
    - `assign_factor=7.937984`
- Result:
  - final two missing stages emitted:

```text
RESULT kernel=deepep_transport_capped_prewarmed ep=8 pass=forward time_s=0.010247 tokens_per_s=12791780.70
RESULT kernel=current ep=8 pass=forward time_s=0.013134 tokens_per_s=9979752.99
```

  - the pod again logged `DeepEP timeout check failed` and repeated CUDA `CUDA_ERROR_LAUNCH_FAILED: unspecified launch failure` cleanup errors around the `deepep_transport_capped_prewarmed` completion boundary, but it still advanced into `current` and ended with `EXIT_CODE=0`
  - combined with the earlier recovered and resumed runs, the corrected exact-cap ladder on the authoritative cell is now:
    - `deepep_transport_assignments_identity`: `46,402,960.98 tok/s`
    - `deepep_transport_first_ragged_dot_probe`: `15,392,690.83 tok/s`
    - `deepep_transport_gate_probe`: `15,111,835.91 tok/s`
    - `deepep_transport_second_ragged_dot_probe`: `12,653,403.81 tok/s`
    - `deepep_transport_capped_prewarmed`: `12,791,780.70 tok/s`
    - `current`: `9,979,752.99 tok/s`
- Interpretation:
  - the dominant drop is still the first ragged-dot step (`46.40M -> 15.39M`)
  - the gate adds almost nothing on top of that (`15.39M -> 15.11M`)
  - the second ragged-dot stage adds another material drop (`15.11M -> 12.65M`)
  - the prewarmed capped transport wrapper is essentially flat versus the second ragged-dot probe (`12.65M` vs `12.79M`, within noise)
  - the final `current` path still adds a large residual loss beyond the exact-cap transport/prefetch pieces (`12.79M -> 9.98M`), so the remaining JAX-vs-Megatron gap is not explained by gate setup or capped transport warmup alone
  - the most defensible corrected root-cause readout from this ladder is:
    - biggest culprit: first ragged-dot work
    - secondary culprit: second ragged-dot work
    - additional non-trivial residual in the full `current` path after capped-prewarmed transport
  - the repeated DeepEP timeout / CUDA launch-failure cleanup noise is now reproducible across both resumed bundles, but because both pods completed and emitted consistent stage ordering, the ladder is usable as a research conclusion with that caveat recorded
- Next action:
  - unless a follow-up asks for deeper attribution, the natural next experiment is targeted profiling of the `current` residual versus `deepep_transport_capped_prewarmed`
  - if the cleanup-failure noise becomes a blocker, isolate it separately because it appears orthogonal to the stage-order performance story

### 2026-03-17 21:16 UTC - Public issue thread updated with the corrected exact-cap ladder milestone
- Command:
  - `gh issue comment 3752 --repo marin-community/marin --body-file -`
- Result:
  - posted milestone comment on the public experiment thread:
    - https://github.com/marin-community/marin/issues/3752#issuecomment-4078054492
  - the issue thread now reflects:
    - the completed corrected exact-cap ladder on the authoritative shared-free forward cell
    - the current root-cause readout:
      - first ragged-dot is the dominant drop
      - gate is not the main culprit
      - second ragged-dot is the next material drop
      - `current` still has meaningful residual overhead beyond capped-prewarmed transport
    - the reproducible DeepEP timeout / CUDA cleanup-noise caveat
- Interpretation:
  - the public issue is now aligned with the local research conclusion through the completed ladder milestone

### 2026-03-17 22:36 UTC - Starting the next branch-settling tranche: re-anchor exact-cap JAX vs Megatron before changing the harness
- Hypothesis:
  - before spending code-change effort on profiling the first ragged-dot stage, the thread should first re-establish the actual surviving cross-stack gap on the authoritative exact-cap path, because the local notes currently contain two different Megatron anchors for the `131072, topk=2` regime:
    - sealed `#3717` matrix / issue body: `12.71M tok/s`
    - local forward-only note reused from the earlier Megatron Qwen benchmark: `forward_ms=5.115853 ms`, about `25.62M tok/s`
- Command:
  - start with a four-run single-pod re-anchor matrix only; no harness edits yet
- Config:
  - JAX side:
    - `deepep_transport_capped_prewarmed`
    - `bench_pass=forward`
    - `shared_expert_dim=0`
    - `distribution=random`
    - `topk=2`
    - `ep=8`
    - `warmup=5`
    - `iters=20`
    - token points: `131072`, `262144`
  - Megatron side:
    - dispatcher: `deepep`
    - cases:
      - `marin_3633_topk_2_mb4` -> global tokens `131072`
      - `marin_3633_topk_2_mb8` -> global tokens `262144`
    - `warmup_iters=5`
    - `measure_iters=20`
  - execution policy:
    - one kernel / one dispatcher per pod to avoid the teardown-boundary DeepEP/CUDA noise that appeared in the completed multi-kernel JAX bundles
- Result:
  - no new benchmark numbers yet; this entry records the start of the tranche and the exact run matrix
- Interpretation:
  - this tranche is deliberately narrower than a full new ladder:
    - it settles whether a large exact-cap JAX-vs-Megatron gap still survives on the authoritative path
    - only if that gap is confirmed will the next action be a harness patch for exact-cap probe profiling
- Next action:
  - launch the two JAX `deepep_transport_capped_prewarmed` forward anchors first
  - then launch the two matched Megatron `deepep` cases and compare forward milliseconds directly

### 2026-03-17 23:20 UTC - Re-anchored exact-cap JAX vs Megatron on `131072` and `262144`, with an out-of-band Megatron log capture after the first launcher-level miss
- Commands:
  - JAX `131072` exact-cap anchor:

```bash
KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
  --task-id jax-gap-reanchor-jax-capped-fwd-t131072-topk2-20260317-2237 \
  --tokens 131072 \
  --shared-expert-dim 0 \
  --kernels deepep_transport_capped_prewarmed \
  --topk-list 2 \
  --distributions random \
  --bench-pass forward \
  --ep-list 8 \
  --warmup 5 \
  --iters 20 \
  --per-bench-timeout-seconds 900 \
  --per-bench-kill-after-seconds 20 \
  --build-with-torch-extension \
  --load-as-python-module \
  --skip-smoke \
  --skip-cleanup
```

  - JAX `262144` exact-cap anchor:

```bash
KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
  --task-id jax-gap-reanchor-jax-capped-fwd-t262144-topk2-20260317-2242 \
  --tokens 262144 \
  --shared-expert-dim 0 \
  --kernels deepep_transport_capped_prewarmed \
  --topk-list 2 \
  --distributions random \
  --bench-pass forward \
  --ep-list 8 \
  --warmup 5 \
  --iters 20 \
  --per-bench-timeout-seconds 1200 \
  --per-bench-kill-after-seconds 20 \
  --build-with-torch-extension \
  --load-as-python-module \
  --skip-smoke \
  --skip-cleanup
```

  - first Megatron `131072` launch attempt:

```bash
KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run --package iris python .agents/scripts/megatron_qwen_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
  --worktree /home/ubuntu/dev/marin-wt/moe-jax-megatron-gap-root-cause \
  --task-id jax-gap-reanchor-megatron-deepep-mb4-20260317-2304 \
  --cases marin_3633_topk_2_mb4 \
  --dispatchers deepep \
  --warmup-iters 5 \
  --measure-iters 20
```

  - Megatron `131072` rerun after the launcher-level miss:

```bash
KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run --package iris python .agents/scripts/megatron_qwen_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
  --worktree /home/ubuntu/dev/marin-wt/moe-jax-megatron-gap-root-cause \
  --task-id jax-gap-reanchor-megatron-deepep-mb4-rerun-20260317-2309 \
  --cases marin_3633_topk_2_mb4 \
  --dispatchers deepep \
  --warmup-iters 5 \
  --measure-iters 20
```

  - Megatron `262144` rerun:

```bash
KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run --package iris python .agents/scripts/megatron_qwen_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
  --worktree /home/ubuntu/dev/marin-wt/moe-jax-megatron-gap-root-cause \
  --task-id jax-gap-reanchor-megatron-deepep-mb8-rerun-20260317-2314 \
  --cases marin_3633_topk_2_mb8 \
  --dispatchers deepep \
  --warmup-iters 5 \
  --measure-iters 20
```

- Result:
  - JAX exact-cap forward anchors both completed with `EXIT_CODE=0` and the same reproducible DeepEP/CUDA cleanup-noise footer seen in the earlier resumed ladder:

```text
tokens=131072:
  DEEPEP_EXACT_CAPS max_recv_tokens=30976 max_local_assignments=33024 recv_factor=4.231405 assign_factor=7.937984
  RESULT kernel=deepep_transport_capped_prewarmed ep=8 pass=forward time_s=0.010966 tokens_per_s=11952193.89

tokens=262144:
  DEEPEP_EXACT_CAPS max_recv_tokens=61952 max_local_assignments=65920 recv_factor=4.231405 assign_factor=7.953398
  RESULT kernel=deepep_transport_capped_prewarmed ep=8 pass=forward time_s=0.021691 tokens_per_s=12085257.93
```

  - the first Megatron `mb4` attempt did not finish cleanly at the launcher layer:
    - pod: `iris-task-42e953ec8365`
    - launcher output ended with:

```text
Pod lookup miss for iris-task-42e953ec8365 (1/3) within retry window; treating as transient
Pod lookup miss for iris-task-42e953ec8365 (2/3) within retry window; treating as transient
EXIT_CODE=None
```

    - by the time the pod lookup was retried, the pod had already been cleaned up and no cluster-side logs remained recoverable
  - because the Megatron launcher has no `--skip-cleanup` escape hatch, I reran `mb4` and then `mb8` while tailing the pod logs directly with `kubectl logs -f`, so the benchmark stdout survived independently of the launcher pod-tracking path
  - the rerun outputs were:

```text
tokens=131072 (case=marin_3633_topk_2_mb4):
  CASE_START marin_3633_topk_2_mb4 deepep tokens_per_rank=16384 hidden=2048 moe_ffn_hidden=768 experts=128 topk=2
  RESULT {"case_name": "marin_3633_topk_2_mb4", "dispatcher": "deepep", "forward_ms": 9.174940645694733, "backward_ms": 6.433644783496857, "forward_std_ms": 13.418792570569678, "backward_std_ms": 3.0208743472503228, "world_size": 8, "micro_batch_size": 4, "seq_length": 4096, "router_topk": 2, "hidden_size": 2048, "moe_ffn_hidden_size": 768, "num_experts": 128}
  EXIT_CODE=0

tokens=262144 (case=marin_3633_topk_2_mb8):
  CASE_START marin_3633_topk_2_mb8 deepep tokens_per_rank=32768 hidden=2048 moe_ffn_hidden=768 experts=128 topk=2
  RESULT {"case_name": "marin_3633_topk_2_mb8", "dispatcher": "deepep", "forward_ms": 7.9210944175720215, "backward_ms": 7.583916807174683, "forward_std_ms": 1.9154777214371457, "backward_std_ms": 1.19760510986657, "world_size": 8, "micro_batch_size": 8, "seq_length": 4096, "router_topk": 2, "hidden_size": 2048, "moe_ffn_hidden_size": 768, "num_experts": 128}
  EXIT_CODE=0
```

  - converting the Megatron forward means to global forward token throughput gives:
    - `131072 / 0.009174940645694733 = 14,285,868.98 tok/s`
    - `262144 / 0.0079210944175720215 = 33,094,416.78 tok/s`
  - direct comparison against the new JAX exact-cap anchors:

| tokens | JAX `deepep_transport_capped_prewarmed` | Megatron `deepep` forward component | Megatron / JAX |
| --- | ---: | ---: | ---: |
| 131072 | 11,952,193.89 tok/s | 14,285,868.98 tok/s | 1.20x |
| 262144 | 12,085,257.93 tok/s | 33,094,416.78 tok/s | 2.74x |

- Interpretation:
  - the earlier ambiguity around the Megatron anchor was real; the matched forward-only re-anchor is neither `12.71M` nor `25.62M`
  - on the exact-cap path, the surviving gap is modest at `131072` (`1.20x`), but very large at `262144` (`2.74x`)
  - the new exact-cap JAX numbers are nearly flat between `131072` and `262144` (`11.95M` -> `12.09M`), while the matched Megatron forward component scales up strongly (`14.29M` -> `33.09M`)
  - the `131072` Megatron rerun includes one very large forward timing sample in the emitted measurement list (`66.99 ms`) and the harness still reports the mean as `9.1749 ms`; that outlier is now part of the factual record for this anchor
  - the `262144` Megatron rerun did not show a comparably extreme forward outlier in its emitted measurement list
  - this tranche did settle the branch question:
    - there is still a real exact-cap JAX-vs-Megatron gap
    - the gap is not shape-invariant across the two token points just re-measured
- Next action:
  - update the public experiment issue if this re-anchor is treated as a major milestone
  - otherwise keep it local and use it to choose the next round between:
    - token-scaling-specific follow-up on the exact-cap path
    - or exact-cap first-ragged-dot profiling if the next objective stays phase-local rather than scale-local

### 2026-03-17 23:24 UTC - Public issue thread updated with the exact-cap re-anchor milestone
- Command:
  - `gh issue comment 3752 --repo marin-community/marin --body-file -`
- Result:
  - posted milestone update to the existing experiment thread:
    - https://github.com/marin-community/marin/issues/3752#issuecomment-4078601926
  - the public thread now records:
    - the matched forward-only exact-cap re-anchor commands
    - the new JAX `deepep_transport_capped_prewarmed` anchors at `131072` and `262144`
    - the new Megatron `deepep` forward anchors for `marin_3633_topk_2_mb4` and `marin_3633_topk_2_mb8`
    - the launcher-level `EXIT_CODE=None` caveat on the first Megatron `mb4` attempt and the successful direct-log-capture reruns
- Interpretation:
  - the issue thread is now up to date through the re-anchor milestone and the scale-sensitive exact-cap-gap finding

### 2026-03-17 23:31 UTC - Starting the reduced `262144` exact-cap JAX ladder to localize the non-scaling phase
- Hypothesis:
  - after the re-anchor, the highest-value unresolved question is no longer whether a cross-stack exact-cap gap exists; it does
  - the next discriminator is which JAX phase fails to scale at `tokens=262144`, because:
    - JAX exact-cap `deepep_transport_capped_prewarmed` stayed almost flat from `131072` to `262144` (`11.95M -> 12.09M tok/s`)
    - matched Megatron `deepep` forward scaled up strongly over the same jump (`14.29M -> 33.09M tok/s`)
- Command:
  - run a reduced JAX-only ladder at `tokens=262144`, one kernel per pod
- Config:
  - common settings for every run:
    - `tokens=262144`
    - `shared_expert_dim=0`
    - `topk=2`
    - `distribution=random`
    - `bench_pass=forward`
    - `ep=8`
    - `warmup=5`
    - `iters=20`
    - `--build-with-torch-extension`
    - `--load-as-python-module`
    - `--skip-smoke`
    - `--skip-cleanup`
  - kernels to run:
    - `deepep_transport_assignments_identity`
    - `deepep_transport_first_ragged_dot_probe`
    - `deepep_transport_second_ragged_dot_probe`
    - `deepep_transport_capped_prewarmed`
    - `current`
  - execution policy:
    - one kernel per pod to avoid the already-observed teardown-boundary DeepEP timeout / CUDA cleanup-noise confound
- Result:
  - no new numbers yet; this entry records the approved run matrix before launch
- Interpretation:
  - the decision rule for the tranche is additive in time, not only throughput:
    - if `assignments_identity` itself worsens sharply, the scaling problem is transport/pack/collapse-adjacent
    - if the large new delta appears at `first_ragged_dot_probe`, the scaling problem is the first local expert compute stage
    - if the large new delta appears at `second_ragged_dot_probe`, the second expert compute stage is the main non-scaling phase
  - if the exact-cap stages stay relatively healthy but `current` grows, the remaining scaling problem sits outside the staged exact-cap DeepEP path
- Next action:
  - launch `deepep_transport_assignments_identity` first and walk the reduced ladder forward from there

### 2026-03-17 23:58 UTC - `262144` reduced ladder point 1: `deepep_transport_assignments_identity`
- Command:
  - `KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py --config lib/iris/examples/coreweave-moe-jax-3677.yaml --task-id jax-gap-262144-assign-id-fwd-topk2-20260317-2332 --tokens 262144 --shared-expert-dim 0 --kernels deepep_transport_assignments_identity --topk-list 2 --distributions random --bench-pass forward --ep-list 8 --warmup 5 --iters 20 --per-bench-timeout-seconds 1200 --per-bench-kill-after-seconds 20 --build-with-torch-extension --load-as-python-module --skip-smoke --skip-cleanup`
- Run metadata:
  - pod: `iris-task-63d5f0508b87`
  - node: `gd94886`
  - shape line:
    - `shape tokens=262144 hidden=2048 mlp_dim=768 experts=128 topk=2 shared_expert_dim=0 dtype=bfloat16 distribution=random bench_pass=forward capacity_factor=1.25`
  - exact-cap line:
    - `DEEPEP_EXACT_CAPS max_recv_tokens=61952 max_local_assignments=65920 recv_factor=4.231405 assign_factor=7.953398`
- Result:
  - `RESULT kernel=deepep_transport_assignments_identity ep=8 pass=forward time_s=0.005305 tokens_per_s=49417331.07`
  - launcher terminated cleanly with `EXIT_CODE=0`
- Operational caveat:
  - after the benchmark result, the pod again emitted:
    - `DeepEP timeout check failed`
    - repeated `CUDA_ERROR_LAUNCH_FAILED` cleanup errors during teardown
  - despite that cleanup noise, the harness reported `bench_status=0`, emitted `BENCH_END`, and exited `0`
- Interpretation:
  - this stage did not exhibit the scaling failure seen in the cross-stack re-anchor
  - compared with the authoritative `131072` ladder point (`46,402,960.98 tok/s`), the `262144` identity stage is slightly faster (`49,417,331.07 tok/s`)
  - that keeps the transport/identity-only branch low priority and pushes the reduced ladder directly toward the ragged-dot stages
- Next action:
  - launch `deepep_transport_first_ragged_dot_probe` at `262144` under the same one-kernel-per-pod policy

### 2026-03-18 00:02 UTC - `262144` reduced ladder point 2: `deepep_transport_first_ragged_dot_probe`
- Command:
  - `KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py --config lib/iris/examples/coreweave-moe-jax-3677.yaml --task-id jax-gap-262144-firstdot-fwd-topk2-20260317-2359 --tokens 262144 --shared-expert-dim 0 --kernels deepep_transport_first_ragged_dot_probe --topk-list 2 --distributions random --bench-pass forward --ep-list 8 --warmup 5 --iters 20 --per-bench-timeout-seconds 1200 --per-bench-kill-after-seconds 20 --build-with-torch-extension --load-as-python-module --skip-smoke --skip-cleanup`
- Run metadata:
  - pod: `iris-task-ae95ecc8ea6a`
  - node: `gd94886`
  - shape line:
    - `shape tokens=262144 hidden=2048 mlp_dim=768 experts=128 topk=2 shared_expert_dim=0 dtype=bfloat16 distribution=random bench_pass=forward capacity_factor=1.25`
  - exact-cap line:
    - `DEEPEP_EXACT_CAPS max_recv_tokens=61952 max_local_assignments=65920 recv_factor=4.231405 assign_factor=7.953398`
- Result:
  - `RESULT kernel=deepep_transport_first_ragged_dot_probe ep=8 pass=forward time_s=0.015179 tokens_per_s=17270015.99`
  - pod reached `Succeeded`
  - launcher terminated with `EXIT_CODE=0`
- Operational caveat:
  - teardown again emitted:
    - `DeepEP timeout check failed`
    - repeated `CUDA_ERROR_LAUNCH_FAILED` cleanup errors
  - the pod still completed with exit code `0`
- Interpretation:
  - this stage also does not match the scale-specific collapse implied by the cross-stack re-anchor
  - compared with the authoritative `131072` ladder point (`15,392,690.83 tok/s`), the `262144` first-ragged stage is faster (`17,270,015.99 tok/s`)
  - that keeps the first-ragged branch important for absolute JAX cost, but weakens it as the explanation for the new `262144` exact-cap scaling gap by itself
- Next action:
  - launch `deepep_transport_second_ragged_dot_probe` at `262144`

### 2026-03-18 00:06 UTC - `262144` reduced ladder point 3: `deepep_transport_second_ragged_dot_probe`
- Command:
  - `KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py --config lib/iris/examples/coreweave-moe-jax-3677.yaml --task-id jax-gap-262144-seconddot-fwd-topk2-20260318-0004 --tokens 262144 --shared-expert-dim 0 --kernels deepep_transport_second_ragged_dot_probe --topk-list 2 --distributions random --bench-pass forward --ep-list 8 --warmup 5 --iters 20 --per-bench-timeout-seconds 1200 --per-bench-kill-after-seconds 20 --build-with-torch-extension --load-as-python-module --skip-smoke --skip-cleanup`
- Run metadata:
  - pod: `iris-task-22824ffb86b9`
  - node: `gd94886`
  - shape line:
    - `shape tokens=262144 hidden=2048 mlp_dim=768 experts=128 topk=2 shared_expert_dim=0 dtype=bfloat16 distribution=random bench_pass=forward capacity_factor=1.25`
  - exact-cap line:
    - `DEEPEP_EXACT_CAPS max_recv_tokens=61952 max_local_assignments=65920 recv_factor=4.231405 assign_factor=7.953398`
- Result:
  - `RESULT kernel=deepep_transport_second_ragged_dot_probe ep=8 pass=forward time_s=0.019929 tokens_per_s=13154193.49`
  - launcher terminated with `EXIT_CODE=0`
- Operational caveat:
  - immediately after emitting the result line, this pod stayed `Running` longer than the first two reduced-ladder runs before eventually following the same cleanup path
  - the final trailer still matched the earlier exact-cap pattern:
    - `DeepEP timeout check failed`
    - repeated `CUDA_ERROR_LAUNCH_FAILED` cleanup errors
    - `bench_status=0`
    - `BENCH_END`
    - `EXIT_CODE=0`
- Interpretation:
  - this stage is much flatter than the first-ragged stage across the `131072 -> 262144` jump:
    - `131072`: `12,653,403.81 tok/s`
    - `262144`: `13,154,193.49 tok/s`
  - compared with the `262144` first-ragged point (`17,270,015.99 tok/s`), the second-ragged point gives back most of the small scale gain seen so far
  - that makes the non-scaling branch materially stronger at or after the second ragged-dot stage than at the identity or first-ragged stages
- Next action:
  - reuse the already-measured identical `262144` `deepep_transport_capped_prewarmed` point from the re-anchor tranche if the command/config match holds exactly, then decide whether to run `current` for full-path context

### 2026-03-18 00:06 UTC - Reused the prior identical `262144` `deepep_transport_capped_prewarmed` point instead of rerunning a duplicate pod
- Fact check:
  - the earlier re-anchor command was:
    - `KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py --config lib/iris/examples/coreweave-moe-jax-3677.yaml --task-id jax-gap-reanchor-jax-capped-fwd-t262144-topk2-20260317-2242 --tokens 262144 --shared-expert-dim 0 --kernels deepep_transport_capped_prewarmed --topk-list 2 --distributions random --bench-pass forward --ep-list 8 --warmup 5 --iters 20 --per-bench-timeout-seconds 1200 --per-bench-kill-after-seconds 20 --build-with-torch-extension --load-as-python-module --skip-smoke --skip-cleanup`
  - that matches the reduced-ladder configuration for every substantive setting; only the task id differs
- Reused result:
  - `RESULT kernel=deepep_transport_capped_prewarmed ep=8 pass=forward time_s=0.021691 tokens_per_s=12085257.93`
- Interpretation:
  - this avoids spending another 8-GPU pod on a known duplicate point
  - the `262144` exact-cap ladder is now populated through `capped_prewarmed`:
    - `deepep_transport_assignments_identity`: `49,417,331.07 tok/s`
    - `deepep_transport_first_ragged_dot_probe`: `17,270,015.99 tok/s`
    - `deepep_transport_second_ragged_dot_probe`: `13,154,193.49 tok/s`
    - `deepep_transport_capped_prewarmed`: `12,085,257.93 tok/s`
- Next action:
  - run `current` at `262144` if full-path context is still worth the extra pod before posting a public milestone update

### 2026-03-18 00:08 UTC - `262144` reduced ladder point 5: `current`
- Command:
  - `KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py --config lib/iris/examples/coreweave-moe-jax-3677.yaml --task-id jax-gap-262144-current-fwd-topk2-20260318-0008 --tokens 262144 --shared-expert-dim 0 --kernels current --topk-list 2 --distributions random --bench-pass forward --ep-list 8 --warmup 5 --iters 20 --per-bench-timeout-seconds 1200 --per-bench-kill-after-seconds 20 --build-with-torch-extension --load-as-python-module --skip-smoke --skip-cleanup`
- Run metadata:
  - pod: `iris-task-f9950079665e`
  - node: `gd94886`
  - shape line:
    - `shape tokens=262144 hidden=2048 mlp_dim=768 experts=128 topk=2 shared_expert_dim=0 dtype=bfloat16 distribution=random bench_pass=forward capacity_factor=1.25`
- Result:
  - `RESULT kernel=current ep=8 pass=forward time_s=0.025734 tokens_per_s=10186545.45`
  - pod reached `Succeeded`
  - launcher terminated with `EXIT_CODE=0`
- Operational note:
  - unlike the exact-cap probe runs in this tranche, the `current` run did not emit the familiar `DeepEP timeout check failed` / `CUDA_ERROR_LAUNCH_FAILED` teardown footer in the captured launcher tail
- Interpretation:
  - `current` is also effectively flat across the token doubling:
    - `131072`: `9,979,752.99 tok/s`
    - `262144`: `10,186,545.45 tok/s`
  - the full JAX path therefore does not recover any meaningful scale gain beyond what the reduced exact-cap ladder already showed

### 2026-03-18 00:08 UTC - Reduced `262144` ladder synthesis against the authoritative `131072` ladder
- Comparison caveat:
  - the fresh `262144` reduced-ladder runs used `--warmup 5 --iters 20`
  - the earlier full `131072` exact-cap ladder used `--warmup 1 --iters 3`
  - therefore the stage-by-stage `131072 -> 262144` ratios below are directionally useful but not perfectly matched
  - the one stage with a matched `131072` re-anchor under `--warmup 5 --iters 20` is `deepep_transport_capped_prewarmed` (`11,952,193.89 tok/s`)
- Comparison table:

| stage | `131072` tok/s | `262144` tok/s | `262144 / 131072` |
| --- | ---: | ---: | ---: |
| `deepep_transport_assignments_identity` | `46,402,960.98` | `49,417,331.07` | `1.065x` |
| `deepep_transport_first_ragged_dot_probe` | `15,392,690.83` | `17,270,015.99` | `1.122x` |
| `deepep_transport_second_ragged_dot_probe` | `12,653,403.81` | `13,154,193.49` | `1.040x` |
| `deepep_transport_capped_prewarmed` | `12,791,780.70` | `12,085,257.93` | `0.945x` |
| `current` | `9,979,752.99` | `10,186,545.45` | `1.021x` |

- Derived observations:
  - identity gains a little when tokens double
  - first ragged-dot also gains a little
  - second ragged-dot is much flatter
  - `capped_prewarmed` is actually slightly slower at `262144` than at `131072`
  - `current` is nearly flat
  - the matched exact-cap Megatron/JAX gap from the re-anchor remains:
    - `131072`: `14,285,868.98 / 11,952,193.89 = 1.20x`
    - `262144`: `33,094,416.78 / 12,085,257.93 = 2.74x`
- Interpretation:
  - the `262144` non-scaling behavior is not present in the identity stage and is weak at the first ragged-dot stage
  - the first clear flattening inside the reduced exact-cap ladder appears by the second ragged-dot stage and remains through `capped_prewarmed`
  - because only `capped_prewarmed` has a fully matched `131072` re-anchor under the same measurement settings, the stage-localization claim above is still best treated as directional rather than final
  - the full `current` path stays flat as well, so there is no large compensating scale gain after the exact-cap path
- Next action:
  - decide whether this synthesis is strong enough for a public milestone comment on the existing experiment issue

### 2026-03-18 00:10 UTC - Posted the reduced `262144` ladder milestone to the existing experiment issue
- Command:
  - `gh issue comment 3752 --repo marin-community/marin --body-file /tmp/issue3752_update_20260318_0008.md`
- Result:
  - posted update to the existing thread:
    - https://github.com/marin-community/marin/issues/3752#issuecomment-4078799461
- Public summary scope:
  - recorded the new `262144` reduced-ladder numbers
  - separated the fully matched exact-cap fact from the still-directional stage-localization claim
  - called out the `--warmup 1 --iters 3` vs `--warmup 5 --iters 20` cross-token comparison caveat explicitly

### 2026-03-18 01:12 UTC - Resumed Phase A after the aborted turn and verified there were no partial launches to clean up
- Post-abort verification:
  - branch head still matched the expected checkpoint:
    - `65eb30baee3b5c782c96b9f647d3ab932681a074`
  - current kube context:
    - `208261-marin`
  - namespace check for `iris-3677-jax` showed only the two crashlooping controllers and no live benchmark pods
- Decision:
  - proceed with the approved four-run tranche exactly as planned:
    - `262144` `deepep_transport_gate_probe`
    - `131072` `deepep_transport_first_ragged_dot_probe` at `--warmup 5 --iters 20`
    - `131072` `deepep_transport_gate_probe` at `--warmup 5 --iters 20`
    - `131072` `deepep_transport_second_ragged_dot_probe` at `--warmup 5 --iters 20`

### 2026-03-18 01:18 UTC - Completed the missing `262144` `deepep_transport_gate_probe` rung
- Command:
  - `KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py --config lib/iris/examples/coreweave-moe-jax-3677.yaml --task-id jax-gap-262144-gate-fwd-topk2-20260318-0130 --tokens 262144 --shared-expert-dim 0 --kernels deepep_transport_gate_probe --topk-list 2 --distributions random --bench-pass forward --ep-list 8 --warmup 5 --iters 20 --per-bench-timeout-seconds 1200 --per-bench-kill-after-seconds 20 --build-with-torch-extension --load-as-python-module --skip-smoke --skip-cleanup`
- Run metadata:
  - pod: `iris-task-c9b58797fb32`
  - node: `gd92fe2`
  - shape line:
    - `shape tokens=262144 hidden=2048 mlp_dim=768 experts=128 topk=2 shared_expert_dim=0 dtype=bfloat16 distribution=random bench_pass=forward capacity_factor=1.25`
  - exact-cap line:
    - `DEEPEP_EXACT_CAPS max_recv_tokens=61952 max_local_assignments=65920 recv_factor=4.231405 assign_factor=7.953398`
- Result:
  - `RESULT kernel=deepep_transport_gate_probe ep=8 pass=forward time_s=0.015162 tokens_per_s=17289446.59`
  - launcher terminated with `EXIT_CODE=0`
- Operational notes:
  - the pod initially sat in `ContainerCreating` after a transient scheduler message:
    - `0/4 nodes are available: 4 Insufficient nvidia.com/gpu`
  - it then pulled `pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel`, started normally, and completed
  - the familiar exact-cap teardown noise appeared again:
    - `DeepEP timeout check failed`
    - repeated `CUDA_ERROR_LAUNCH_FAILED` cleanup errors
- Interpretation:
  - the missing `262144` gate rung lands essentially on top of the already-measured `262144` first-ragged rung:
    - `deepep_transport_first_ragged_dot_probe`: `17,270,015.99 tok/s`
    - `deepep_transport_gate_probe`: `17,289,446.59 tok/s`
  - that keeps the gate contribution negligible at `262144` just as it was at `131072`
- Next action:
  - run the matched `131072` late mini-ladder with `--warmup 5 --iters 20`

### 2026-03-18 01:23 UTC - Matched `131072` rerun: `deepep_transport_first_ragged_dot_probe`
- Command:
  - `KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py --config lib/iris/examples/coreweave-moe-jax-3677.yaml --task-id jax-gap-131072-firstdot-fwd-topk2-20260318-0119 --tokens 131072 --shared-expert-dim 0 --kernels deepep_transport_first_ragged_dot_probe --topk-list 2 --distributions random --bench-pass forward --ep-list 8 --warmup 5 --iters 20 --per-bench-timeout-seconds 1200 --per-bench-kill-after-seconds 20 --build-with-torch-extension --load-as-python-module --skip-smoke --skip-cleanup`
- Run metadata:
  - pod: `iris-task-ed3d1f2ac673`
  - node: `gd92fe2`
  - shape line:
    - `shape tokens=131072 hidden=2048 mlp_dim=768 experts=128 topk=2 shared_expert_dim=0 dtype=bfloat16 distribution=random bench_pass=forward capacity_factor=1.25`
  - exact-cap line:
    - `DEEPEP_EXACT_CAPS max_recv_tokens=30976 max_local_assignments=33024 recv_factor=4.231405 assign_factor=7.937984`
- Result:
  - `RESULT kernel=deepep_transport_first_ragged_dot_probe ep=8 pass=forward time_s=0.008636 tokens_per_s=15176572.21`
  - launcher terminated with `EXIT_CODE=0`
- Operational notes:
  - the same exact-cap teardown footer repeated:
    - `DeepEP timeout check failed`
    - repeated `CUDA_ERROR_LAUNCH_FAILED` cleanup errors
- Comparison against the earlier short-run ladder:
  - old `131072` point under `--warmup 1 --iters 3`:
    - `15,392,690.83 tok/s`
  - new matched point under `--warmup 5 --iters 20`:
    - `15,176,572.21 tok/s`
  - ratio:
    - `0.986x`
- Interpretation:
  - the first-ragged `131072` point is very stable across the shorter and longer measurement settings
- Next action:
  - rerun `131072` `deepep_transport_gate_probe` with the same longer settings

### 2026-03-18 01:26 UTC - Matched `131072` rerun: `deepep_transport_gate_probe`
- Command:
  - `KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py --config lib/iris/examples/coreweave-moe-jax-3677.yaml --task-id jax-gap-131072-gate-fwd-topk2-20260318-0123 --tokens 131072 --shared-expert-dim 0 --kernels deepep_transport_gate_probe --topk-list 2 --distributions random --bench-pass forward --ep-list 8 --warmup 5 --iters 20 --per-bench-timeout-seconds 1200 --per-bench-kill-after-seconds 20 --build-with-torch-extension --load-as-python-module --skip-smoke --skip-cleanup`
- Run metadata:
  - pod: `iris-task-e71e216d1758`
  - node: `gd92fe2`
  - shape line:
    - `shape tokens=131072 hidden=2048 mlp_dim=768 experts=128 topk=2 shared_expert_dim=0 dtype=bfloat16 distribution=random bench_pass=forward capacity_factor=1.25`
  - exact-cap line:
    - `DEEPEP_EXACT_CAPS max_recv_tokens=30976 max_local_assignments=33024 recv_factor=4.231405 assign_factor=7.937984`
- Result:
  - `RESULT kernel=deepep_transport_gate_probe ep=8 pass=forward time_s=0.008647 tokens_per_s=15157611.05`
  - launcher terminated with `EXIT_CODE=0`
- Operational notes:
  - the launcher emitted repeated log-parser warnings while streaming `apt` progress lines:
    - `Failed to parse timestamp from kubectl log line: '(Reading database ... 5%'`
  - the benchmark itself still completed normally and ended with the same exact-cap teardown footer:
    - `DeepEP timeout check failed`
    - repeated `CUDA_ERROR_LAUNCH_FAILED` cleanup errors
- Comparison against the earlier short-run ladder:
  - old `131072` point under `--warmup 1 --iters 3`:
    - `15,111,835.91 tok/s`
  - new matched point under `--warmup 5 --iters 20`:
    - `15,157,611.05 tok/s`
  - ratio:
    - `1.003x`
- Interpretation:
  - the longer-run measurement leaves the `131072` gate point essentially unchanged
  - the gate still lands on top of the `131072` first-ragged point, confirming that gate work remains negligible on the matched settings too
- Next action:
  - rerun `131072` `deepep_transport_second_ragged_dot_probe` with the same longer settings

### 2026-03-18 01:29 UTC - Matched `131072` rerun: `deepep_transport_second_ragged_dot_probe`
- Command:
  - `KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py --config lib/iris/examples/coreweave-moe-jax-3677.yaml --task-id jax-gap-131072-seconddot-fwd-topk2-20260318-0126 --tokens 131072 --shared-expert-dim 0 --kernels deepep_transport_second_ragged_dot_probe --topk-list 2 --distributions random --bench-pass forward --ep-list 8 --warmup 5 --iters 20 --per-bench-timeout-seconds 1200 --per-bench-kill-after-seconds 20 --build-with-torch-extension --load-as-python-module --skip-smoke --skip-cleanup`
- Run metadata:
  - pod: `iris-task-e636f378e351`
  - node: `gd92fe2`
  - shape line:
    - `shape tokens=131072 hidden=2048 mlp_dim=768 experts=128 topk=2 shared_expert_dim=0 dtype=bfloat16 distribution=random bench_pass=forward capacity_factor=1.25`
  - exact-cap line:
    - `DEEPEP_EXACT_CAPS max_recv_tokens=30976 max_local_assignments=33024 recv_factor=4.231405 assign_factor=7.937984`
- Result:
  - `RESULT kernel=deepep_transport_second_ragged_dot_probe ep=8 pass=forward time_s=0.010236 tokens_per_s=12805001.33`
  - launcher terminated with `EXIT_CODE=0`
- Operational notes:
  - the same exact-cap teardown footer repeated:
    - `DeepEP timeout check failed`
    - repeated `CUDA_ERROR_LAUNCH_FAILED` cleanup errors
- Comparison against the earlier short-run ladder:
  - old `131072` point under `--warmup 1 --iters 3`:
    - `12,653,403.81 tok/s`
  - new matched point under `--warmup 5 --iters 20`:
    - `12,805,001.33 tok/s`
  - ratio:
    - `1.012x`
- Interpretation:
  - the longer-run measurement also leaves the second-ragged `131072` point essentially unchanged

### 2026-03-18 01:30 UTC - Phase A synthesis: the matched reruns remove the main `131072` settings confound, and the late-stage non-scaling still first appears by `second_ragged_dot_probe`
- New matched late-stage table (`--warmup 5 --iters 20` on both token counts where available):

| stage | `131072` tok/s | `262144` tok/s | `262144 / 131072` |
| --- | ---: | ---: | ---: |
| `deepep_transport_first_ragged_dot_probe` | `15,176,572.21` | `17,270,015.99` | `1.138x` |
| `deepep_transport_gate_probe` | `15,157,611.05` | `17,289,446.59` | `1.141x` |
| `deepep_transport_second_ragged_dot_probe` | `12,805,001.33` | `13,154,193.49` | `1.027x` |
| `deepep_transport_capped_prewarmed` | `11,952,193.89` | `12,085,257.93` | `1.011x` |

- Cross-check against the earlier short-run `131072` ladder:
  - `first_ragged_dot_probe`: `0.986x`
  - `gate_probe`: `1.003x`
  - `second_ragged_dot_probe`: `1.012x`
- Derived observations:
  - the old `131072` ladder was not materially distorted by the shorter `--warmup 1 --iters 3` settings on any of the late-stage points rerun here
  - gate remains negligible at both token counts:
    - `131072`: `15.18M` first-ragged vs `15.16M` gate
    - `262144`: `17.27M` first-ragged vs `17.29M` gate
  - the matched `262144` scale gain is still present through `gate_probe`
  - the first clear flattening on matched settings still appears by `second_ragged_dot_probe`
  - the path stays flat from `second_ragged_dot_probe` into `capped_prewarmed`
  - this now aligns with the already-matched cross-stack exact-cap fact:
    - JAX `deepep_transport_capped_prewarmed`: `11,952,193.89 -> 12,085,257.93 tok/s`
    - Megatron forward deepep: `14,285,868.98 -> 33,094,416.78 tok/s`
- Confidence update:
  - the earlier stage-localization claim is now materially stronger than it was in the `00:08 UTC` synthesis because the largest remaining measurement-settings caveat on the `131072` late ladder has been removed
- Next action:
  - decide whether to post this strengthened milestone to the existing issue thread before any code changes

### 2026-03-18 01:45 UTC - Patched the harness for a late local-compute split and exact-cap forward profiling support
- Motivation:
  - after the matched `131072` reruns, the strongest remaining question was whether the `262144` non-scaling sits in late local expert compute itself or only after local compute, in collapse/combine or nearby glue
- Code change:
  - updated `lib/levanter/scripts/bench/bench_moe_hillclimb.py`
- Functional additions:
  - added a new exact-cap forward probe kernel:
    - `deepep_transport_local_compute_only_probe`
  - this probe reuses the staged DeepEP helpers:
    - `_moe_mlp_deepep_transport_dispatch_pack`
    - `_moe_mlp_deepep_transport_local_compute`
  - it stops after local expert compute and does not run collapse/combine or shared MLP
  - added `--profile-root` support for:
    - the exact-cap forward probe kernels
    - `deepep_transport_capped_prewarmed` forward
- Profiling-label cleanup:
  - added `jax.named_scope` labels around the reusable late-path helpers:
    - dispatch layout
    - intranode dispatch
    - local assignment packing
    - first ragged-dot
    - gate split/activation
    - second ragged-dot
    - collapse of local assignments
    - intranode combine
- Local validation:
  - `python -m py_compile lib/levanter/scripts/bench/bench_moe_hillclimb.py`
  - exited successfully
- Next action:
  - commit and push the benchmark-file change, then run matched `131072` and `262144` `deepep_transport_local_compute_only_probe` pods

### 2026-03-18 01:47 UTC - Committed and pushed the harness patch for cluster use
- Commit:
  - `04c1babb65561efdd0dea5128d5f6d1035a3a12d`
  - subject:
    - `bench: add late DeepEP local-compute probe`
- Push result:
  - pushed `research/moe-jax-megatron-gap-root-cause` from `65eb30baee3b5c782c96b9f647d3ab932681a074` to `04c1babb65561efdd0dea5128d5f6d1035a3a12d`
- Operational note:
  - this was required before launching benchmark pods because the launcher fetches GitHub tarballs by repo ref
- Next action:
  - run matched `131072` and `262144` `deepep_transport_local_compute_only_probe` pods against `REPO_REF=04c1babb65561efdd0dea5128d5f6d1035a3a12d`

### 2026-03-18 02:00 UTC - Attempted the first `local_compute_only` run, but the cluster had no ready GPU nodes
- Command attempted:
  - `KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py --config lib/iris/examples/coreweave-moe-jax-3677.yaml --task-id jax-gap-131072-localcomputeonly-fwd-topk2-20260318-0148 --tokens 131072 --shared-expert-dim 0 --kernels deepep_transport_local_compute_only_probe --topk-list 2 --distributions random --bench-pass forward --ep-list 8 --warmup 5 --iters 20 --per-bench-timeout-seconds 1200 --per-bench-kill-after-seconds 20 --build-with-torch-extension --load-as-python-module --skip-smoke --skip-cleanup`
- Launcher metadata:
  - pod:
    - `iris-task-260911488de3`
  - repo ref:
    - `04c1babb65561efdd0dea5128d5f6d1035a3a12d`
- Observed scheduler state:
  - the pod never bound to a node
  - `kubectl describe pod` showed only scheduling failures, not container startup:
    - `0/3 nodes are available: 3 Insufficient nvidia.com/gpu`
  - autoscaler emitted:
    - `TriggeredScaleUp ... 0->1 (max: 1)`
  - after roughly 11 minutes, the cluster view still contained only three ready nodes:
    - `g505e50`
    - `g7b96e0`
    - `g8fd384`
  - those ready nodes did not advertise any `nvidia.com/gpu` allocatable resource in `kubectl describe node`
- Interpretation:
  - this was a cluster-capacity / node-provisioning blocker, not a harness or benchmark-code failure
  - no benchmark process started on a GPU node, so there is no experimental datapoint to interpret from this attempt
- Cleanup:
  - interrupted the local launcher poll loop with `Ctrl-C`
  - deleted the still-pending pod:
    - `kubectl delete pod iris-task-260911488de3 -n iris-3677-jax --ignore-not-found`
- GH issue policy:
  - did not post an issue update because no new benchmark result or public-facing milestone was produced
- Next action:
  - re-launch the `131072` `deepep_transport_local_compute_only_probe` once ready GPU nodes reappear, then run the matched `262144` point on the same `04c1babb6` harness state

### 2026-03-18 02:20 UTC - Fixed a traced-array mesh-capture bug in the new probe and restarted the AFK loop on the corrected ref
- Trigger:
  - the first real GPU-backed `131072` `deepep_transport_local_compute_only_probe` attempt started after GPU capacity returned, but failed during benchmark execution
- Failing attempt metadata:
  - task id:
    - `jax-gap-131072-localcomputeonly-fwd-topk2-20260318-0212-try1`
  - pod:
    - `iris-task-680fd54709b5`
  - repo ref used by the launcher:
    - `04c1babb65561efdd0dea5128d5f6d1035a3a12d`
- Observed benchmark failure:
  - the pod reached Python/JAX execution and then raised:
    - `AttributeError: The 'sharding' attribute is not available on traced array`
  - the failure occurred inside `_forward_deepep_transport_local_compute_only_probe` where the new helper tried to read `x.sharding.mesh`
- Interpretation:
  - this was a real harness bug in the new probe implementation, not a cluster or launcher problem
  - the bug path was specific to traced/JIT execution of the new local-compute-only forward helper
- Code fix:
  - updated `lib/levanter/scripts/bench/bench_moe_hillclimb.py`
  - changed `_forward_deepep_transport_local_compute_only_probe(...)` to accept an explicit `mesh` argument and fall back to `get_abstract_mesh()` instead of reading `x.sharding.mesh` inside traced code
  - updated `_make_deepep_transport_probe_forward_fn(...)` to pass `mesh=mesh` into the `deepep_transport_local_compute_only_probe` partial
  - updated the direct `_forward(...)` path for `deepep_transport_local_compute_only_probe` to use `mesh = get_abstract_mesh()` and pass that mesh into the helper
- Local validation:
  - `python -m py_compile lib/levanter/scripts/bench/bench_moe_hillclimb.py`
  - exited successfully
- Commit and push:
  - commit:
    - `b225306d7c51ce1bf6839ef1edcc8e1a86aeddcb`
  - subject:
    - `bench: fix local DeepEP probe mesh capture`
  - pushed `research/moe-jax-megatron-gap-root-cause` to origin so the launcher can fetch the corrected tarball
- Operational follow-up:
  - an unattended AFK retry loop had already relaunched once on the old broken ref and printed:
    - `REPO_REF=04c1babb65561efdd0dea5128d5f6d1035a3a12d`
  - interrupted that known-bad retry, deleted its pod:
    - `iris-task-5757c4bc5c35`
  - restarted the AFK loop so the fresh attempt now uses:
    - `REPO_REF=b225306d7c51ce1bf6839ef1edcc8e1a86aeddcb`
- Current state at log time:
  - the restarted `131072` `deepep_transport_local_compute_only_probe` pod is running on node `g1464be`
  - no new benchmark datapoint yet, so there is still no new GH issue milestone to post

### 2026-03-18 02:36 UTC - `local_compute_only` isolates the new non-scaling to post-local-compute work after the second ragged dot
- Motivation:
  - the new probe was intended to answer whether the `262144` flattening lives inside late local expert compute itself or in work that happens immediately after local expert compute
- Probe definition reminder:
  - `deepep_transport_local_compute_only_probe` includes:
    - exact-cap dispatch/layout/pack
    - local expert compute through `w2_ragged_dot`
  - it excludes:
    - `_collapse_deepep_local_assignments`
    - `deepep_combine_intranode`
    - shared MLP / full capped forward glue
- AFK-loop results on commit `b225306d7c51ce1bf6839ef1edcc8e1a86aeddcb`:
  - `131072`:
    - `13,890,515.04 tok/s`
    - `time_s=0.009436`
  - accidental duplicate-loop `131072` rerun:
    - `15,281,865.37 tok/s`
    - `time_s=0.008577`
  - `262144`:
    - `15,631,772.90 tok/s`
    - `time_s=0.016770`
- Operational note:
  - the `15.28M` `131072` point came from an accidental second local shell loop that resumed after an earlier `Ctrl-C` and launched a duplicate pod
  - I killed that extra loop once it was identified so subsequent runs were again single-driver
- Clean manual reruns after removing the duplicate-loop confound:
  - `131072` rerun command:
    - `KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py --config lib/iris/examples/coreweave-moe-jax-3677.yaml --task-id jax-gap-131072-localcomputeonly-fwd-topk2-20260318-0232-rerun1 --tokens 131072 --shared-expert-dim 0 --kernels deepep_transport_local_compute_only_probe --topk-list 2 --distributions random --bench-pass forward --ep-list 8 --warmup 5 --iters 20 --per-bench-timeout-seconds 1200 --per-bench-kill-after-seconds 20 --build-with-torch-extension --load-as-python-module --skip-smoke --skip-cleanup`
  - clean `131072` rerun result:
    - `13,924,357.17 tok/s`
    - `time_s=0.009413`
  - `262144` rerun command:
    - `KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py --config lib/iris/examples/coreweave-moe-jax-3677.yaml --task-id jax-gap-262144-localcomputeonly-fwd-topk2-20260318-0235-rerun1 --tokens 262144 --shared-expert-dim 0 --kernels deepep_transport_local_compute_only_probe --topk-list 2 --distributions random --bench-pass forward --ep-list 8 --warmup 5 --iters 20 --per-bench-timeout-seconds 1200 --per-bench-kill-after-seconds 20 --build-with-torch-extension --load-as-python-module --skip-smoke --skip-cleanup`
  - clean `262144` rerun result:
    - `15,619,829.57 tok/s`
    - `time_s=0.016783`
- Stability readout:
  - clean `131072` points:
    - `13.89M`
    - `13.92M`
  - clean `262144` points:
    - `15.63M`
    - `15.62M`
  - the accidental `15.28M` `131072` duplicate-loop point is inconsistent with the clean reruns and should not be treated as the primary anchor
- Comparison against the matched late-stage ladder:
  - `131072`
    - `gate_probe`: `15,157,611.05 tok/s` -> `8.647 ms`
    - `local_compute_only` clean mean: `13,907,436.11 tok/s` -> `9.425 ms`
    - `second_ragged_dot_probe`: `12,805,001.33 tok/s` -> `10.236 ms`
    - `capped_prewarmed`: `11,952,193.89 tok/s` -> `10.966 ms`
  - `262144`
    - `gate_probe`: `17,289,446.59 tok/s` -> `15.162 ms`
    - `local_compute_only` clean mean: `15,625,801.24 tok/s` -> `16.776 ms`
    - `second_ragged_dot_probe`: `13,154,193.49 tok/s` -> `19.929 ms`
    - `capped_prewarmed`: `12,085,257.93 tok/s` -> `21.691 ms`
- Derived incremental costs from those matched timings:
  - `w2_ragged_dot` plus the remainder of local expert compute after `gate_probe`
    - `131072`: `+0.777 ms`
    - `262144`: `+1.614 ms`
  - post-local-compute collapse/combine work (`second_ragged_dot_probe - local_compute_only`)
    - `131072`: `+0.811 ms`
    - `262144`: `+3.152 ms`
  - remaining full capped path beyond `second_ragged_dot_probe` (`capped_prewarmed - second_ragged_dot_probe`)
    - `131072`: `+0.730 ms`
    - `262144`: `+1.763 ms`
- Factual conclusion from this tranche:
  - the clean `local_compute_only` pair does scale somewhat from `131072 -> 262144` (`13.89/13.92M -> 15.62/15.63M`)
  - the flattening seen at `second_ragged_dot_probe` and `capped_prewarmed` is therefore not explained solely by local expert compute through `w2_ragged_dot`
  - the largest newly isolated non-scaling slice is the post-local-compute path added after `local_compute_only`, especially the collapse/combine work between `local_compute_only` and `second_ragged_dot_probe`
- Operational caveat:
  - all of these runs again emitted the same `DeepEP timeout check failed` and `CUDA_ERROR_LAUNCH_FAILED` teardown noise but still reported `EXIT_CODE=0` on the completed launcher paths
- Next action:
  - capture forward profiles on the exact-cap probes to separate `_collapse_deepep_local_assignments` from `deepep_combine_intranode` on the authoritative `262144` cell

### 2026-03-18 02:54 UTC - Profile diff pins the post-local-compute culprit to DeepEP combine plus a large XLA scatter fusion
- Motivation:
  - after the clean `local_compute_only` reruns, the remaining question was whether the newly isolated `second_ragged_dot_probe - local_compute_only` cost comes mostly from:
    - DeepEP intranode combine
    - JAX/XLA collapse glue around `_collapse_deepep_local_assignments`
    - or some unrelated hidden host/runtime overhead
- Wrapper patch for profile capture:
  - updated `.agents/scripts/deepep_jax_krt_bench.py`
  - added support for:
    - forwarding `--profile-root` into `bench_moe_hillclimb.py`
    - `--post-bench-sleep-seconds` so profile pods can stay alive long enough for `kubectl cp`
  - local validation:
    - `python -m py_compile .agents/scripts/deepep_jax_krt_bench.py`
  - commit:
    - `beea0fb3c0c7db33432117d9f223bfac00ff273b`
  - subject:
    - `bench: keep JAX KRT profile pods alive`
  - pushed to origin before profiling because the launcher fetches GitHub tarballs by repo ref
- Profile run 1:
  - task id:
    - `jax-gap-262144-secondragged-profile-fwd-topk2-20260318-0245`
  - command:
    - `KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py --config lib/iris/examples/coreweave-moe-jax-3677.yaml --task-id jax-gap-262144-secondragged-profile-fwd-topk2-20260318-0245 --tokens 262144 --shared-expert-dim 0 --kernels deepep_transport_second_ragged_dot_probe --topk-list 2 --distributions random --bench-pass forward --ep-list 8 --warmup 2 --iters 3 --profile-root /tmp/jax-profile-secondragged-262144 --post-bench-sleep-seconds 1800 --per-bench-timeout-seconds 1800 --per-bench-kill-after-seconds 20 --build-with-torch-extension --load-as-python-module --skip-smoke --skip-cleanup`
  - first launch attempt failed locally before pod creation:
    - `subprocess.TimeoutExpired: Command '['kubectl', 'apply', '-f', '-']' timed out after 60.0 seconds`
  - retry succeeded and launched pod:
    - `iris-task-63e968c3dcc0`
  - profiled benchmark output:
    - `RESULT kernel=deepep_transport_second_ragged_dot_probe ep=8 pass=forward time_s=0.055278 tokens_per_s=4742257.11`
    - `PROFILE kernel=deepep_transport_second_ragged_dot_probe ep=8 dir=/tmp/jax-profile-secondragged-262144`
  - copied live profile tree to:
    - `scratch/profiles/secondragged-262144`
  - deleted the sleeping pod after copy
- Profile run 2:
  - task id:
    - `jax-gap-262144-localcompute-profile-fwd-topk2-20260318-0248`
  - command:
    - `KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py --config lib/iris/examples/coreweave-moe-jax-3677.yaml --task-id jax-gap-262144-localcompute-profile-fwd-topk2-20260318-0248 --tokens 262144 --shared-expert-dim 0 --kernels deepep_transport_local_compute_only_probe --topk-list 2 --distributions random --bench-pass forward --ep-list 8 --warmup 2 --iters 3 --profile-root /tmp/jax-profile-localcompute-262144 --post-bench-sleep-seconds 1800 --per-bench-timeout-seconds 1800 --per-bench-kill-after-seconds 20 --build-with-torch-extension --load-as-python-module --skip-smoke --skip-cleanup`
  - launched pod:
    - `iris-task-a3a9c0613203`
  - profiled benchmark output:
    - `RESULT kernel=deepep_transport_local_compute_only_probe ep=8 pass=forward time_s=0.049711 tokens_per_s=5273399.48`
    - `PROFILE kernel=deepep_transport_local_compute_only_probe ep=8 dir=/tmp/jax-profile-localcompute-262144`
  - copied live profile tree to:
    - `scratch/profiles/localcompute-262144`
  - deleted the sleeping pod after copy
- Local profile processing:
  - summarized:
    - `scratch/profiles/secondragged-262144-summary.json`
    - `scratch/profiles/localcompute-262144-summary.json`
  - compared:
    - `uv run python lib/marin/tools/profile_summary.py compare --before scratch/profiles/localcompute-262144-summary.json --after scratch/profiles/secondragged-262144-summary.json`
- Top exclusive ops from the `second_ragged_dot_probe` profile:
  - GEMM-like local compute kernels were still the largest entries:
    - `nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NNN`
    - `nvjet_tst_192x192_64x3_1x2_h_bz_coopB_NNN`
  - but the largest non-GEMM additions were:
    - `void deep_ep::intranode::combine<__nv_bfloat16, 8, 768, 4096>(...)`
      - `exclusive_duration=36,815.128`
      - `avg_duration=1,533.964`
    - `void deep_ep::intranode::dispatch<8, 768, 8192>(...)`
      - `exclusive_duration=36,545.110`
    - `input_scatter_fusion`
      - `exclusive_duration=27,619.298`
      - `avg_duration=1,150.804`
    - `void deep_ep::intranode::cached_notify_combine<8>(...)`
      - `exclusive_duration=5,616.440`
      - `avg_duration=234.018`
- Direct raw-trace diff of the key DeepEP kernels:
  - `second_ragged_dot_probe`
    - `combine`: `48` events, total `36,909.604`
    - `dispatch`: `48` events, total `36,675.336`
    - `cached_notify_combine`: `48` events, total `5,732.321`
    - `get_dispatch_layout`: `48` events, total `3,038.697`
    - `notify_dispatch`: `48` events, total `2,602.119`
  - `local_compute_only`
    - `dispatch`: `48` events, total `36,775.008`
    - `get_dispatch_layout`: `24` events, total `2,952.254`
    - `notify_dispatch`: `48` events, total `2,724.820`
    - there is no `combine` or `cached_notify_combine` entry
- Direct raw-trace diff of the key XLA fusion:
  - `second_ragged_dot_probe`
    - `input_scatter_fusion`: `24` events, total `27,619.298`
  - `local_compute_only`
    - `input_scatter_fusion`: `24` events, total `117.024`
- `profile_summary compare` regressed ops (`second_ragged_dot_probe` relative to `local_compute_only`):
  - `void deep_ep::intranode::combine<__nv_bfloat16, 8, 768, 4096>(...)`
    - `delta = +36,815.128`
  - `input_scatter_fusion`
    - `delta = +27,502.274`
  - `void deep_ep::intranode::cached_notify_combine<8>(...)`
    - `delta = +5,616.440`
  - `loop_select_fusion_2`
    - `delta = +5,946.204`
  - `loop_broadcast_fusion`
    - `delta = +3,671.259`
- Interpretation:
  - the large GEMM kernels are essentially unchanged between the two profiles, which matches the timing result that `local_compute_only` already contains the local expert compute
  - the newly added heavy work in `second_ragged_dot_probe` is exactly the post-local-compute tail:
    - DeepEP `combine`
    - DeepEP `cached_notify_combine`
    - a large XLA `input_scatter_fusion`
  - based on the code path difference between the two probes, it is a strong inference that the large `input_scatter_fusion` corresponds to the `_collapse_deepep_local_assignments` segment-sum / scatter reduction step
- Factual conclusion:
  - the post-local-compute non-scaling isolated in the previous timing tranche is now concretely explained by:
    - DeepEP intranode combine
    - a large XLA scatter fusion that appears only once the collapse path is reintroduced
  - this is no longer just “around the second ragged-dot stage”; the heavy added mechanisms have been identified
- Next action:
  - decide whether the remaining thread should stop here with a concrete root-cause writeup or continue one smaller profile to label the `second_ragged_dot_probe -> capped_prewarmed` residual as non-DeepEP shared-MLP/full-forward overhead

## 2026-03-18 13:55:46Z

- Goal:
  - split the already-isolated late return leg into:
    - JAX collapse only
    - DeepEP combine only
- Code change:
  - added two new exact-cap probe kernels in `lib/levanter/scripts/bench/bench_moe_hillclimb.py`:
    - `deepep_transport_collapse_only_probe`
    - `deepep_transport_combine_only_probe`
  - refactored the late helper so the existing collapse+combine path now composes:
    - `_moe_mlp_ep_deepep_transport_collapse_only_local`
    - `_moe_mlp_ep_deepep_transport_combine_only_local`
  - added exact-cap timing/profile support that precomputes dispatch/local-compute inputs outside the measured region for the new pure-stage probes
  - added a cumulative `_forward_deepep_transport_collapse_only_probe` for the generic forward path
- Local validation:
  - `python -m py_compile lib/levanter/scripts/bench/bench_moe_hillclimb.py .agents/scripts/deepep_jax_krt_bench.py`
  - `uv run python lib/levanter/scripts/bench/bench_moe_hillclimb.py --help | rg "collapse_only_probe|combine_only_probe"`
- Immediate next action:
  - commit and push the harness patch before launching any new cluster jobs
  - run matched one-kernel pods for:
    - `deepep_transport_collapse_only_probe`
    - `deepep_transport_combine_only_probe`
    - on `tokens=131072` and `tokens=262144`

## 2026-03-18 14:09:16Z

- Code checkpoint:
  - pushed harness split commit:
    - `722a7432dea129c952a799477dee96864a8b2449`
    - `bench: split DeepEP collapse and combine probes`
- Commands launched:
  - `KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py --config lib/iris/examples/coreweave-moe-jax-3677.yaml --task-id jax-gap-collapseonly-fwd-t131072-topk2-20260318-1401 --tokens 131072 --shared-expert-dim 0 --kernels deepep_transport_collapse_only_probe --topk-list 2 --distributions random --bench-pass forward --ep-list 8 --warmup 5 --iters 20 --per-bench-timeout-seconds 420 --per-bench-kill-after-seconds 20 --build-with-torch-extension --load-as-python-module --skip-smoke --skip-cleanup`
  - `KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py --config lib/iris/examples/coreweave-moe-jax-3677.yaml --task-id jax-gap-collapseonly-fwd-t262144-topk2-20260318-1405 --tokens 262144 --shared-expert-dim 0 --kernels deepep_transport_collapse_only_probe --topk-list 2 --distributions random --bench-pass forward --ep-list 8 --warmup 5 --iters 20 --per-bench-timeout-seconds 420 --per-bench-kill-after-seconds 20 --build-with-torch-extension --load-as-python-module --skip-smoke --skip-cleanup`
  - `KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py --config lib/iris/examples/coreweave-moe-jax-3677.yaml --task-id jax-gap-combineonly-fwd-t131072-topk2-20260318-1411 --tokens 131072 --shared-expert-dim 0 --kernels deepep_transport_combine_only_probe --topk-list 2 --distributions random --bench-pass forward --ep-list 8 --warmup 5 --iters 20 --per-bench-timeout-seconds 420 --per-bench-kill-after-seconds 20 --build-with-torch-extension --load-as-python-module --skip-smoke --skip-cleanup`
  - `KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py --config lib/iris/examples/coreweave-moe-jax-3677.yaml --task-id jax-gap-combineonly-fwd-t262144-topk2-20260318-1414 --tokens 262144 --shared-expert-dim 0 --kernels deepep_transport_combine_only_probe --topk-list 2 --distributions random --bench-pass forward --ep-list 8 --warmup 5 --iters 20 --per-bench-timeout-seconds 420 --per-bench-kill-after-seconds 20 --build-with-torch-extension --load-as-python-module --skip-smoke --skip-cleanup`
- Operational note:
  - all four pods reproduced the same post-`RESULT` linger pattern as earlier exact-cap runs
  - after capturing the `RESULT` line from each pod, I manually deleted the live pod to free the single effective H100x8 lane so the next pending task could start
- New isolated pure-stage results on the authoritative cell (`topk=2`, `forward`, `ep=8`, `random`, shared-free):
  - `deepep_transport_collapse_only_probe`
    - `131072`: `144,646,359.30 tok/s` (`0.906 ms`)
    - `262144`: `153,426,681.78 tok/s` (`1.709 ms`)
  - `deepep_transport_combine_only_probe`
    - `131072`: `119,767,526.46 tok/s` (`1.094 ms`)
    - `262144`: `138,159,193.60 tok/s` (`1.897 ms`)
- Comparison against the existing matched JAX ladder:
  - earlier matched JAX anchors:
    - `local_compute_only`
      - `131072`: `13,907,436.11 tok/s` (`9.425 ms`)
      - `262144`: `15,625,801.24 tok/s` (`16.776 ms`)
    - `second_ragged_dot_probe`
      - `131072`: `12,805,001.33 tok/s` (`10.236 ms`)
      - `262144`: `13,154,193.49 tok/s` (`19.929 ms`)
    - `deepep_transport_capped_prewarmed`
      - `131072`: `11,952,193.89 tok/s` (`10.966 ms`)
      - `262144`: `12,085,257.93 tok/s` (`21.691 ms`)
  - isolated late-tail pure-stage sums:
    - `collapse_only + combine_only`
      - `131072`: `2.001 ms`
      - `262144`: `3.606 ms`
- Cross-stack reference that remains in force:
  - matched Megatron exact-cap forward at `262144`: `33,094,416.78 tok/s` (`7.921 ms`)
- Factual readout:
  - `collapse_only` is individually very fast at both token counts and does not exhibit the catastrophic flat-throughput behavior of the full JAX exact-cap path
  - `combine_only` is also individually very fast at both token counts and similarly does not by itself explain the exact-cap JAX-vs-Megatron gap
  - by `262144`, the JAX `local_compute_only` path is already `16.776 ms`, which is still more than the full matched Megatron exact-cap forward time of `7.921 ms`
  - the isolated late return-leg pieces are real additive overhead, but the new pure-stage timings show they are not by themselves the dominant cross-stack explanation
  - this means the earlier profile-backed late-tail attribution needs narrower wording:
    - the profile correctly identified what work gets added after local compute
    - the new pure-stage timings show that the surviving large JAX-vs-Megatron gap is still already present before those isolated pieces, especially by the time JAX reaches `local_compute_only`
- Immediate next action:
  - post a major milestone update on `#3752` that records the pure-stage split and corrects the current public readout
  - then choose the next branch with the updated fact pattern:
    - isolated `collapse_only` and `combine_only` are not enough
    - the dominant unresolved cross-stack gap still survives by `local_compute_only`

## 2026-03-18 15:04:55 UTC

- Goal:
  - directly split the surviving `local_compute_only` cost into:
    - pure first expert matmul (`w13_ragged_dot`)
    - pure second expert matmul (`w2_ragged_dot`)
  - then capture one clean `w13_only` profile on the authoritative `262144` cell
- Code checkpoint:
  - added two new exact-cap probe kernels in `lib/levanter/scripts/bench/bench_moe_hillclimb.py`:
    - `deepep_transport_w13_only_probe`
    - `deepep_transport_w2_only_probe`
  - implementation details:
    - added pure local helpers:
      - `_moe_mlp_deepep_transport_w13_only(...)`
      - `_moe_mlp_deepep_transport_gate_up_only(...)`
      - `_moe_mlp_deepep_transport_w2_only(...)`
    - wired the new probes through:
      - direct `_forward(...)`
      - exact-cap timing/profile runner `_make_deepep_transport_probe_forward_runner(...)`
      - CLI kernel choices and the shared-MLP bypass set
  - local validation:
    - `python -m py_compile lib/levanter/scripts/bench/bench_moe_hillclimb.py`
    - `uv run python lib/levanter/scripts/bench/bench_moe_hillclimb.py --help | rg 'w13_only_probe|w2_only_probe'`
  - commit:
    - `43c50fae0420137f362ef2a80a45b21bf0e1d9be`
  - subject:
    - `bench: add DeepEP w13 and w2 probes`
  - pushed to origin before launching cluster jobs because the launcher fetches GitHub tarballs by repo ref
- Matched one-kernel timing commands on the authoritative cell (`topk=2`, `forward`, `ep=8`, `random`, shared-free):
  - `KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py --config lib/iris/examples/coreweave-moe-jax-3677.yaml --task-id jax-gap-w13only-fwd-t131072-topk2-20260318-1848 --tokens 131072 --shared-expert-dim 0 --kernels deepep_transport_w13_only_probe --topk-list 2 --distributions random --bench-pass forward --ep-list 8 --warmup 5 --iters 20 --per-bench-timeout-seconds 420 --per-bench-kill-after-seconds 20 --build-with-torch-extension --load-as-python-module --skip-smoke --skip-cleanup`
  - `KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py --config lib/iris/examples/coreweave-moe-jax-3677.yaml --task-id jax-gap-w13only-fwd-t262144-topk2-20260318-1853 --tokens 262144 --shared-expert-dim 0 --kernels deepep_transport_w13_only_probe --topk-list 2 --distributions random --bench-pass forward --ep-list 8 --warmup 5 --iters 20 --per-bench-timeout-seconds 420 --per-bench-kill-after-seconds 20 --build-with-torch-extension --load-as-python-module --skip-smoke --skip-cleanup`
  - `KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py --config lib/iris/examples/coreweave-moe-jax-3677.yaml --task-id jax-gap-w2only-fwd-t131072-topk2-20260318-1858 --tokens 131072 --shared-expert-dim 0 --kernels deepep_transport_w2_only_probe --topk-list 2 --distributions random --bench-pass forward --ep-list 8 --warmup 5 --iters 20 --per-bench-timeout-seconds 420 --per-bench-kill-after-seconds 20 --build-with-torch-extension --load-as-python-module --skip-smoke --skip-cleanup`
  - `KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py --config lib/iris/examples/coreweave-moe-jax-3677.yaml --task-id jax-gap-w2only-fwd-t262144-topk2-20260318-1903 --tokens 262144 --shared-expert-dim 0 --kernels deepep_transport_w2_only_probe --topk-list 2 --distributions random --bench-pass forward --ep-list 8 --warmup 5 --iters 20 --per-bench-timeout-seconds 420 --per-bench-kill-after-seconds 20 --build-with-torch-extension --load-as-python-module --skip-smoke --skip-cleanup`
- Pod placement / operational note:
  - all four timing pods ran on node `g1464be`
  - each pod reproduced the same post-`RESULT` linger pattern as earlier exact-cap runs, so after capturing the numeric `RESULT` line I manually deleted the live pod to keep the single effective H100x8 lane moving:
    - `iris-task-b50498c37c97`
    - `iris-task-cdc7e14b7ae4`
    - `iris-task-050c4e53e8b7`
    - `iris-task-8a59b425bf1f`
- New pure local-compute results:
  - `deepep_transport_w13_only_probe`
    - `131072`: `21,936,683.90 tok/s` (`5.975 ms`)
    - `262144`: `26,051,875.76 tok/s` (`10.062 ms`)
    - throughput scale from `131072 -> 262144`: `1.188x`
  - `deepep_transport_w2_only_probe`
    - `131072`: `53,157,536.67 tok/s` (`2.466 ms`)
    - `262144`: `51,574,551.62 tok/s` (`5.083 ms`)
    - throughput scale from `131072 -> 262144`: `0.970x`
- Comparison against existing anchors:
  - `262144`:
    - `assignments_identity`: `49,417,331.07 tok/s` (`5.305 ms`)
    - `local_compute_only`: `15,625,801.24 tok/s` (`16.776 ms`)
    - matched Megatron exact-cap forward: `33,094,416.78 tok/s` (`7.921 ms`)
  - `w13_only + w2_only` at `262144`:
    - `10.062 ms + 5.083 ms = 15.145 ms`
    - this almost exactly closes the `local_compute_only` budget once the small dispatch-pack slice is added
  - consequence:
    - `w13_only` by itself (`10.062 ms`) is already slower than the full matched Megatron exact-cap forward time (`7.921 ms`)
- Factual readout from the pure local split:
  - the dominant surviving cost inside JAX local expert compute is the first expert matmul path (`w13_only`), not the second expert matmul path
  - `w2_only` is materially smaller than `w13_only` at both token counts
  - the local-compute decomposition is now internally consistent:
    - pure `w13_only`
    - pure `w2_only`
    - earlier small dispatch-pack baseline from `assignments_identity - (collapse_only + combine_only)`
    - together reconstruct the observed `local_compute_only` cost
  - this materially sharpens the branch choice:
    - the dominant surviving exact-cap gap is no longer best described as “late collapse/combine”
    - it is dominated by the first local expert compute block, with the second local expert matmul as a secondary contributor and the isolated return leg as a tertiary one
- Follow-up profile capture for the dominant `w13_only` stage:
  - command:
    - `KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py --config lib/iris/examples/coreweave-moe-jax-3677.yaml --task-id jax-gap-w13only-profile-fwd-t262144-topk2-20260318-1915 --tokens 262144 --shared-expert-dim 0 --kernels deepep_transport_w13_only_probe --topk-list 2 --distributions random --bench-pass forward --ep-list 8 --warmup 2 --iters 3 --profile-root /tmp/jax-profile-w13only-262144 --post-bench-sleep-seconds 1800 --per-bench-timeout-seconds 1800 --per-bench-kill-after-seconds 20 --build-with-torch-extension --load-as-python-module --skip-smoke --skip-cleanup`
  - pod:
    - `iris-task-4e4412ee198d`
  - profiled benchmark output:
    - `RESULT kernel=deepep_transport_w13_only_probe ep=8 pass=forward time_s=0.025045 tokens_per_s=10466783.86`
    - `PROFILE kernel=deepep_transport_w13_only_probe ep=8 dir=/tmp/jax-profile-w13only-262144`
  - copied live profile tree to:
    - `scratch/profiles/w13only-262144`
  - summarized to:
    - `scratch/profiles/w13only-262144-summary.json`
  - deleted the sleeping pod after copy
- `assignments_identity -> w13_only` profile diff at `262144`:
  - command:
    - `uv run python lib/marin/tools/profile_summary.py compare --before scratch/profiles/assignid-262144-summary.json --after scratch/profiles/w13only-262144-summary.json`
  - strongest new regressed ops:
    - `nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NNN`
      - `delta = +164,976.808`
    - `loop_transpose_fusion`
      - `delta = +61,111.415`
    - `wrapped_slice`
      - `delta = +3,290.812`
  - strongest improved ops (removed relative to `assignments_identity`):
    - `deep_ep::intranode::combine`
      - `delta = -36,861.574`
    - `deep_ep::intranode::dispatch`
      - `delta = -36,855.683`
    - `input_scatter_fusion_1`
      - `delta = -27,286.571`
  - `compute` share delta:
    - `+0.2713`
- Top exclusive ops from the pure `w13_only` profile:
  - `nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NNN`
    - `exclusive_duration = 164,976.808`
    - `avg_duration = 6,874.034`
  - `loop_transpose_fusion`
    - `exclusive_duration = 61,111.415`
    - `avg_duration = 2,546.309`
  - `wrapped_slice`
    - `exclusive_duration = 3,290.812`
    - `avg_duration = 137.117`
- Factual conclusion from the profile-backed local split:
  - the dominant remaining exact-cap cost is now directly attributed to the pure first local expert compute stage
  - on the pure `w13_only` profile, the heavy work is almost entirely:
    - one large GEMM-like kernel
    - one large transpose fusion
  - the earlier DeepEP dispatch/combine and collapse-scatter kernels are absent from this pure stage, so they are not needed to explain the dominant `w13_only` cost
  - the surviving exact-cap JAX-vs-Megatron gap is therefore best localized to:
    - JAX `w13_ragged_dot` plus its associated layout/transpose path
    - with `w2_ragged_dot` as a smaller secondary contribution
    - and isolated collapse/combine as smaller tertiary overhead
- Immediate next action:
  - post a major milestone update on `#3752` with the pure local-compute split and the `w13_only` profile-backed readout
