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
