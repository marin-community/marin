# JAX DeepEP Benchmark Reintegration: Research Logbook

## Scope
- Goal: determine whether the working pure-JAX DeepEP transport path from `#3677` materially changes the original fixed-shape JAX benchmark outcome, and localize any remaining JAX/Torch transport delta before broadening the benchmark matrix.
- Primary metric(s):
  - fixed-shape H100x8 forward wall time on the original `#3633`-style benchmark
  - phase-level timing split for layout vs transport vs surrounding bookkeeping
  - matched Torch-vs-JAX transport delta on the authoritative worst-gap cell
- Constraints:
  - Use CoreWeave Iris H100x8 via `~/llms/cw_ops_guide.md`.
  - Reuse the isolated CoreWeave namespace/prefix lane when practical.
  - Keep the next round gated: one-cell attribution first, benchmark reintegration second, broader sweeps only if warranted.
  - Update the GitHub issue only for major milestones.
- Experiment issue: https://github.com/marin-community/marin/issues/3711

## Baseline
- Date: 2026-03-16
- Code refs:
  - sealed prior thread: `research/moe-jax-megatron-root-cause` @ `6baa08edbd8ae9a782d0070a3b7cf0e1f38ba005`
  - key JAX transport files carried forward:
    - `lib/levanter/src/levanter/kernels/deepep/transport_ffi.py`
    - `lib/levanter/src/levanter/kernels/deepep/csrc/deepep_transport_ffi.cu`
    - `lib/levanter/scripts/bench/bench_deepep_dispatch_jax.py`
- Baseline numbers:
  - corrected same-shape transport matrix from `#3677`:
    - `random, topk=2`: Torch `67.44M`, JAX `43.79M`, Torch/JAX `1.54x`
    - `runs, topk=2`: Torch `35.90M`, JAX `29.15M`, Torch/JAX `1.23x`
    - `random, topk=8`: Torch `30.85M`, JAX `25.25M`, Torch/JAX `1.22x`
    - `runs, topk=8`: Torch `25.25M`, JAX `22.94M`, Torch/JAX `1.10x`
  - original fixed-shape negative JAX result (`#3665`) remained layout-only and did not exercise DeepEP transport.

## Initial Hypotheses
- The remaining JAX/Torch transport gap is now small enough that it may mostly be accounted for by non-transport work in the timed region: layout, reductions, wrapper/bookkeeping, or custom-call boundary overhead.
- If the corrected pure-JAX transport is reinserted into the original fixed-shape benchmark, the old negative JAX result should materially improve relative to `deepep_layout_ragged_a2a`.
- `topk=8` drift is likely secondary to the main benchmark-level question and should not gate the first reintegration controls.

## Stop Criteria
- Produce a benchmark-level control on the original fixed-shape JAX path that compares:
  - `current`
  - `ragged_a2a`
  - `deepep_layout_ragged_a2a`
  - full pure-JAX DeepEP transport
- Produce a one-cell attribution split that localizes the remaining JAX/Torch transport gap well enough to decide whether the next step should be transport tuning or broader benchmark reintegration.
- Update the issue body with a concise decision-quality conclusion and a next-step ordering.

## Experiment Log

### 2026-03-15 22:06 PDT - One-cell transport decomposition on the authoritative `random, topk=2` full-shape JAX path
- Context:
  - Before touching the original fixed-shape hillclimb benchmark again, measure how much of the remaining JAX/Torch gap on the authoritative cell is layout vs cached dispatch/combine vs non-transport bookkeeping.
- Commands:
  ```bash
  KUBECONFIG=~/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_transport_krt.py \
    --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
    --worktree /Users/romain/marin-wt/moe-jax-deepep-benchmark-reintegration \
    --task-id deepep-jax-transport-breakdown-20260315-2203 \
    --tokens 32768 \
    --hidden 2048 \
    --experts 128 \
    --topk-list 2 \
    --distributions random \
    --execution-model shard_map \
    --warmup 2 \
    --iters 10 \
    --build-with-torch-extension \
    --timing-breakdown
  ```
  ```bash
  KUBECONFIG=~/.kube/coreweave-iris uv run .agents/scripts/deepep_dispatch_krt_bench.py \
    --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
    --worktree /Users/romain/marin-wt/moe-jax-deepep-benchmark-reintegration \
    --task-id deepep-dispatch-krt-bench-20260315-230650 \
    --tokens 32768 \
    --hidden 2048 \
    --experts 128 \
    --topk-list 2 \
    --distributions random \
    --input-sources torch \
    --warmup 2 \
    --iters 10 \
    --timeout-seconds 7200
  ```
- Result:
  - JAX pure transport decomposition:
    - `CHECK x_max_abs=0.000000e+00 topk_max_abs=0.000000e+00`
    - `BREAKDOWN layout_s=0.000205 dispatch_combine_cached_s=0.000700`
    - `RESULT step_s=0.000725 tokens_per_s=45168819.30`
  - Matching Torch transport control:
    - `RESULT layout_s=0.000025 dispatch_combine_cached_s=0.000450 dispatch_combine_full_s=0.000491 bridge_to_torch_s=0.000000 bridge_to_jax_s=0.000000 tokens_per_s=66774016.98`
- Interpretation:
  - On the authoritative full-shape cell, layout is not the dominant remaining JAX/Torch delta:
    - JAX layout cost is only `0.205 ms`
    - JAX full step is `0.725 ms`
    - JAX full minus cached transport is only `0.025 ms`
  - The remaining gap to Torch on this cell is concentrated in the transport-side timed region, not in layout metadata production.

### 2026-03-15 22:20 PDT - Added a benchmark-level `deepep_transport` kernel to the fixed-shape hillclimb harness
- Context:
  - The original hillclimb benchmark still only had:
    - `ragged_a2a`
    - `deepep_layout_ragged_a2a`
  - It did not yet have a kernel that exercised the working pure-JAX DeepEP dispatch/combine transport path.
- Changes:
  - Added a sibling kernel `deepep_transport` in `lib/levanter/scripts/bench/bench_moe_hillclimb.py`.
  - The new path:
    - keeps the existing local `ragged_dot` expert compute
    - replaces the two EP `jax.lax.ragged_all_to_all` legs with:
      - `deepep_dispatch_intranode(...)`
      - `deepep_combine_intranode(...)`
  - Added helpers to:
    - pack token-level DeepEP receives into the assignment-grouped local-expert layout expected by `ragged_dot`
    - collapse weighted per-assignment expert outputs back to one row per received token before `deepep_combine_intranode(...)`
  - Added launcher support in `.agents/scripts/deepep_jax_krt_bench.py` for `DEEPEP_BUILD_WITH_TORCH_EXTENSION=1`.
- Commits:
  - `3aa2f3353` — `Add DeepEP transport hillclimb kernel`
  - `57c7f178f` — `Update DeepEP hillclimb benchmark defaults`

### 2026-03-15 22:34 PDT - First benchmark-lane smoke failures were packaging/bootstrap issues, not benchmark-kernel failures
- Context:
  - The first single-cell hillclimb smoke targeted:
    - `kernel=deepep_transport`
    - `distribution=random`
    - `topk=2`
    - `bench_pass=forward`
    - `ep_list=8`
- Command:
  ```bash
  KUBECONFIG=~/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
    --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
    --task-id deepep-jax-krt-bench-20260315-232220 \
    --kernels deepep_transport \
    --topk-list 2 \
    --distributions random \
    --bench-pass forward \
    --ep-list 8 \
    --warmup 1 \
    --iters 1 \
    --build-with-torch-extension
  ```
- Result:
  - Attempt 1 failed during `uv sync`:
    - `RuntimeError: Protobuf outputs are missing and npx is not installed.`
  - I first tried staging the generated Iris protobuf outputs into the task bundle.
  - Attempt 2 then failed before pod creation:
    - `metadata.annotations: Too long: may not be more than 262144 bytes`
  - I replaced that approach by installing `nodejs` / `npm` in the pod before `uv sync`, which lets the normal Iris build hook regenerate protobuf outputs in-place.
- Commits:
  - `631b60fcb` — `Stage Iris protobufs in benchmark pods`
  - `546fe311a` — `Install Node in benchmark pods`
- Current status:
  - The rerun `deepep-jax-krt-bench-20260315-233430` is now past the earlier pod-spec and protobuf-generation blockers and is running the normal benchmark bootstrap path.

### 2026-03-15 22:53 PDT - First true hillclimb attempt reached the benchmark body and exposed the wrong DeepEP load mode
- Context:
  - The rerun `deepep-jax-krt-bench-20260315-233430` got through:
    - pod creation
    - Iris build / protobuf generation
    - JAX/Torch preflight
    - the benchmark pod FFI smoke
  - It then entered the actual hillclimb benchmark body with:
    - `kernel=deepep_transport`
    - `distribution=random`
    - `topk=2`
    - `ep_list=8`
- Result:
  - The benchmark pod printed:
    - `RUNNING_BENCH_MATRIX`
    - `BENCH_START kernel=deepep_transport distribution=random topk=2`
    - the full `bench_moe_hillclimb.py` shape header
  - The failure then came from `transport_ffi.py` inside the real benchmark path:
    - `OSError: ... undefined symbol: __cudaRegisterLinkedBinary_a14bd921_23_deepep_transport_ffi_cu_8a7790fa`
  - This happened while using only:
    - `DEEPEP_BUILD_WITH_TORCH_EXTENSION=1`
    - without `DEEPEP_LOAD_AS_PYTHON_MODULE=1`
- Interpretation:
  - This is not the earlier raw-build path.
  - It is the weaker torch-extension load mode.
  - The hillclimb lane needs to use the same stronger build/load combination that the transport harness already proved out.

### 2026-03-16 00:05 PDT - Stronger load mode reached the benchmark body, but benchmark-level `deepep_transport` still failed before the first compiled call
- Context:
  - After switching the hillclimb launcher to the same stronger DeepEP load path used in the working transport harness:
    - `DEEPEP_BUILD_WITH_TORCH_EXTENSION=1`
    - `DEEPEP_LOAD_AS_PYTHON_MODULE=1`
  - I reran the same single-cell benchmark smoke:
    - `kernel=deepep_transport`
    - `distribution=random`
    - `topk=2`
    - `bench_pass=forward`
    - `ep_list=8`
- Result:
  - The benchmark again reached the real hillclimb body and printed the fixed-shape header.
  - The failure then changed from loader-time to runtime:
    - `jax.errors.JaxRuntimeError: INTERNAL: [0] There was an error before calling cuModuleGetFunction (704): cudaErrorPeerAccessAlreadyEnabled : peer access is already enabled`
  - Two preinit variants were then tested outside the timed path:
    - preinit after sharding
    - preinit even earlier before JAX input creation
  - Neither fixed the benchmark:
    - the later preinit variant still failed at the benchmark call
    - the earlier preinit variant failed even earlier at `jax.random.PRNGKey(...)`
- Interpretation:
  - The hillclimb reintegration blocker is no longer the old raw build / loader problem.
  - Explicit host-side runtime preinit is not a viable fix.

### 2026-03-16 00:18 PDT - Shared expert was ruled out, and identity diagnostics narrowed the failure above raw transport
- Context:
  - To separate transport runtime issues from higher-level full-layer logic, I first reran the same single-cell hillclimb smoke with `shared_expert_dim=0`.
  - I then added diagnostic kernels inside `bench_moe_hillclimb.py` that reuse the same hillclimb process, mesh, and sharding setup while cutting away parts of the local expert path.
- Result:
  - `shared_expert_dim=0` still failed with the same peer-access error, so the shared-expert branch is not the trigger.
  - `deepep_transport_identity` succeeded:
    - `RESULT kernel=deepep_transport_identity ep=8 pass=forward time_s=0.000810 tokens_per_s=40435401.28`
  - `deepep_transport_assignments_identity` also succeeded:
    - `RESULT kernel=deepep_transport_assignments_identity ep=8 pass=forward time_s=0.002253 tokens_per_s=14542543.21`
- Interpretation:
  - The same hillclimb script/process/mesh/sharding setup is compatible with:
    - DeepEP dispatch
    - DeepEP combine
    - local assignment pack/collapse bookkeeping
  - The remaining failure domain moved above raw transport and above assignment bookkeeping into the live local expert compute path.

### 2026-03-16 00:34 PDT - First trustworthy consumed-ragged-dot probe failed, moving the frontier to the first live local `ragged_dot`
- Context:
  - The first round of probe kernels was inconclusive because the intermediate tensors were not part of the returned value and could be dead-code-eliminated.
  - I rewrote the probes so each intermediate is adapted to hidden width and then fed through collapse+combine.
  - I also pushed that commit before rerunning so the pod could fetch the exact repo ref from GitHub.
- Commits:
  - `1f94cc153` — `Force DeepEP transport probes through returned outputs`
  - `63862a86b` — `Trim benchmark pod bootstrap installs`
- Command:
  ```bash
  KUBECONFIG=~/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
    --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
    --task-id deepep-jax-krt-bench-20260316-002917 \
    --kernels deepep_transport_first_ragged_dot_probe \
    --topk-list 2 \
    --distributions random \
    --bench-pass forward \
    --ep-list 8 \
    --warmup 1 \
    --iters 1 \
    --build-with-torch-extension \
    --load-as-python-module
  ```
- Result:
  - The pod completed the stronger load/build/bootstrap path and reached the real hillclimb benchmark body.
  - The benchmark then failed again with the same runtime error:
    - `jax.errors.JaxRuntimeError: INTERNAL: [0] There was an error before calling cuModuleGetFunction (704): cudaErrorPeerAccessAlreadyEnabled : peer access is already enabled`
- Interpretation:
  - This is the first trustworthy consumed-intermediate probe, and it fails.
  - Since `deepep_transport_assignments_identity` succeeds but `deepep_transport_first_ragged_dot_probe` fails once the first `ragged_dot(...)` result is live in the returned value, the remaining benchmark-level frontier is now the first live local `ragged_dot` above DeepEP dispatch.
  - This does **not** implicate `ragged_dot` in isolation, because the ring/current hillclimb kernels already use `ragged_dot` successfully. It implicates the same compiled graph that combines live DeepEP transport with the first live local `ragged_dot`.

### 2026-03-16 00:46 PDT - Splitting transport and local compute into separate compiled stages did not fix the failure by itself
- Context:
  - I added a new forward-only benchmark kernel, `deepep_transport_staged`, that:
    1. runs DeepEP dispatch + pack
    2. runs local expert compute in a separate compiled stage
    3. runs collapse + DeepEP combine in a separate compiled stage
    4. adds the shared MLP as a separate compiled stage
  - The goal was to test whether the original failure was specifically a one-big-executable interaction.
- Commit:
  - `8bf4545fb` — `Add staged DeepEP hillclimb forward path`
- Result:
  - The staged path still failed on the same authoritative single-cell case:
    - `tokens=32768, hidden=2048, mlp_dim=768, experts=128, topk=2, distribution=random, EP=8`
  - The failure moved to the separated local-compute stage:
    - `out_dispatch = local_compute(...)`
    - `jax.errors.JaxRuntimeError: INTERNAL: [0] There was an error before calling cuModuleGetFunction (704): cudaErrorPeerAccessAlreadyEnabled : peer access is already enabled`
- Interpretation:
  - Separating transport and local compute into distinct compiled stages is not sufficient on its own.

### 2026-03-16 00:50 PDT - Prewarming `current` in the same pod did not rescue the staged DeepEP path
- Context:
  - To test whether ordinary ragged/XLA kernel initialization order was the missing ingredient, I ran `current` first in the same pod and then `deepep_transport_staged`.
- Command:
  ```bash
  KUBECONFIG=~/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
    --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
    --task-id deepep-jax-krt-bench-20260316-004004 \
    --kernels current,deepep_transport_staged \
    --topk-list 2 \
    --distributions random \
    --bench-pass forward \
    --ep-list 8 \
    --warmup 1 \
    --iters 1 \
    --build-with-torch-extension \
    --load-as-python-module
  ```
- Result:
  - `current` completed first in the same pod:
    - `RESULT kernel=current ep=8 pass=forward time_s=0.003888 tokens_per_s=8427328.95`
  - `deepep_transport_staged` still failed immediately afterward with the same peer-access error at the separated local-compute stage.
- Interpretation:
  - Generic prewarming of the ordinary `current` ragged path is not enough.

### 2026-03-16 00:58 PDT - Exact prewarming of the staged local-compute executable before any DeepEP dispatch unblocked the staged forward path
- Context:
  - I then added a stronger control to `_time_deepep_transport_staged_forward(...)`:
    - prebuild concrete sharded dummy inputs matching the staged local-compute shapes
    - run the exact staged `local_compute(...)` executable once before any DeepEP dispatch
    - also prewarm the separate shared MLP stage
  - One intermediate rerun exposed a bug in my helper:
    - I first used `get_abstract_mesh()` for `NamedSharding(...)`, which failed with:
      - `ValueError: is_fully_addressable is not implemented for jax.sharding.AbstractMesh`
    - I corrected that by using the concrete mesh already attached to the sharded inputs:
      - `mesh = x.sharding.mesh`
- Commits:
  - `b345bb4ae` — `Prewarm staged DeepEP local compute`
  - `da2a6f01a` — `Use concrete mesh for staged prewarm`
- Command:
  ```bash
  KUBECONFIG=~/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
    --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
    --task-id deepep-jax-krt-bench-20260316-004729 \
    --kernels deepep_transport_staged \
    --topk-list 2 \
    --distributions random \
    --bench-pass forward \
    --ep-list 8 \
    --warmup 1 \
    --iters 1 \
    --build-with-torch-extension \
    --load-as-python-module
  ```
- Result:
  - The staged kernel now completed successfully on the same authoritative single-cell case:
    - `RESULT kernel=deepep_transport_staged ep=8 pass=forward time_s=0.016974 tokens_per_s=1930450.75`
- Interpretation:
  - The exact staged local-compute executable can run successfully in the benchmark process if it is prewarmed before any DeepEP dispatch in that process.
  - The stronger control changed the outcome where:
    - plain staged split failed
    - and `current` prewarming failed
  - So the current evidence is now consistent with a more specific executable-initialization / ordering sensitivity than “any ragged kernel after DeepEP is impossible.”

### 2026-03-16 01:07 PDT - The same exact prewarm also unblocked the original monolithic `deepep_transport` kernel, but not its performance deficit
- Context:
  - I reused the same exact local-compute prewarm before timing the original monolithic `deepep_transport` kernel instead of the staged forward path.
  - This introduced a new benchmark kernel name:
    - `deepep_transport_prewarmed`
- Commit:
  - `7fea69e67` — `Add prewarmed DeepEP hillclimb kernel`
- Command:
  ```bash
  KUBECONFIG=~/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
    --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
    --task-id deepep-jax-krt-bench-20260316-005511 \
    --kernels deepep_transport_prewarmed \
    --topk-list 2 \
    --distributions random \
    --bench-pass forward \
    --ep-list 8 \
    --warmup 1 \
    --iters 1 \
    --build-with-torch-extension \
    --load-as-python-module
  ```
- Result:
  - The monolithic path now runs on the same authoritative fixed-shape single-cell case:
    - `RESULT kernel=deepep_transport_prewarmed ep=8 pass=forward time_s=0.017159 tokens_per_s=1909665.39`
- Interpretation:
  - The exact-prewarm trick is not limited to the staged workaround; it is enough to make the original monolithic `deepep_transport` path runnable too.
  - On this small fixed-shape cell, though, the runnable monolithic path is still much slower than `current`.

### 2026-03-16 01:18 PDT - At `32768/device`, prewarmed-monolithic DeepEP remained about `5x` slower than `current`
- Context:
  - User-requested first scale step:
    - `32768/device` on `8` GPUs
    - global `tokens=262144`
  - Measured with the runnable monolithic kernel:
    - `deepep_transport_prewarmed`
- Commands:
  ```bash
  KUBECONFIG=~/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
    --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
    --task-id deepep-jax-krt-bench-20260316-005857 \
    --tokens 262144 \
    --kernels current,deepep_transport_prewarmed \
    --topk-list 2 \
    --distributions random \
    --bench-pass forward \
    --ep-list 8 \
    --warmup 1 \
    --iters 1 \
    --build-with-torch-extension \
    --load-as-python-module
  ```
  ```bash
  KUBECONFIG=~/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
    --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
    --task-id deepep-jax-krt-bench-20260316-010857 \
    --tokens 262144 \
    --kernels current,deepep_transport_prewarmed \
    --topk-list 2 \
    --distributions runs \
    --bench-pass forward \
    --ep-list 8 \
    --warmup 1 \
    --iters 1 \
    --build-with-torch-extension \
    --load-as-python-module
  ```
- Result:
  - `random, topk=2`:
    - `current`: `0.027038 s`, `9695506.06 tok/s`
    - `deepep_transport_prewarmed`: `0.139256 s`, `1882465.68 tok/s`
    - ratio: `current / deepep_transport_prewarmed = 5.15x`
  - `runs, topk=2`:
    - `current`: `0.026870 s`, `9755985.37 tok/s`
    - `deepep_transport_prewarmed`: `0.146935 s`, `1784078.77 tok/s`
    - ratio: `current / deepep_transport_prewarmed = 5.47x`
- Interpretation:
  - On the first larger token point, making the DeepEP path runnable did **not** make it competitive with `current`.
  - The gap stayed large on both `random` and `runs`.

### 2026-03-16 01:24 PDT - Higher scale and higher top-k ran into memory limits before showing any compensating gain
- Context:
  - I then tried:
    - `tokens=262144`, `topk=8`, `distribution in {random, runs}`
    - `tokens=524288`, `topk=2`, `distribution=random`
- Commands:
  ```bash
  KUBECONFIG=~/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
    --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
    --task-id deepep-jax-krt-bench-20260316-010425 \
    --tokens 262144 \
    --kernels current,deepep_transport_prewarmed \
    --topk-list 8 \
    --distributions random,runs \
    --bench-pass forward \
    --ep-list 8 \
    --warmup 1 \
    --iters 1 \
    --build-with-torch-extension \
    --load-as-python-module
  ```
  ```bash
  KUBECONFIG=~/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
    --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
    --task-id deepep-jax-krt-bench-20260316-011302 \
    --tokens 524288 \
    --kernels current,deepep_transport_prewarmed \
    --topk-list 2 \
    --distributions random \
    --bench-pass forward \
    --ep-list 8 \
    --warmup 1 \
    --iters 1 \
    --build-with-torch-extension \
    --load-as-python-module
  ```
- Result:
  - `tokens=262144, topk=8` failed during exact prewarm of the dummy local-compute buffer:
    - `jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 8589934592 bytes.`
  - `tokens=524288, topk=2` let `current` complete:
    - `RESULT kernel=current ep=8 pass=forward time_s=0.059733 tokens_per_s=8777239.74`
  - but `deepep_transport_prewarmed` then failed:
    - `jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 68736253952 bytes.`
- Interpretation:
  - The current runnable prewarmed-monolithic DeepEP path does not scale cleanly to the next requested token / top-k points on H100x8.
  - So the present state is:
    - functionally runnable at the original fixed-shape size and at `32768/device, topk=2`
    - still much slower than `current`
    - and already memory-limited at the next scale points

### 2026-03-16 01:45 PDT - Added a full-mesh layout-stats probe to measure actual receive and local-assignment counts at the larger token points
- Context:
  - The new question after the negative `32768/device` result was whether the large-shape slowdown and OOMs were coming from a grossly pessimistic static capacity bound in the JAX wrapper.
  - I added a new `--layout-stats-only` mode to `lib/levanter/scripts/bench/bench_deepep_dispatch_jax.py` and threaded it through `.agents/scripts/deepep_jax_transport_krt.py`.
  - This mode:
    - builds the full sharded `topk_idx`
    - runs `deepep_get_dispatch_layout(...)` across the whole mesh
    - gathers the resulting source-to-destination token-count matrix and per-expert assignment counts
    - reports:
      - actual `recv_tokens_per_rank`
      - actual `local_assignments_per_rank`
      - the existing wrapper bounds:
        - `recv_capacity_per_rank = local_tokens * world_size`
        - `assignment_rows_per_rank = recv_capacity_per_rank * topk`
- Local validation:
  ```bash
  uv run python -m py_compile \
    lib/levanter/scripts/bench/bench_deepep_dispatch_jax.py \
    .agents/scripts/deepep_jax_transport_krt.py
  ```

### 2026-03-16 01:50 PDT - `32768/device` layout-only stats showed the first strong structural bottleneck
- Commands:
  ```bash
  KUBECONFIG=~/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_transport_krt.py \
    --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
    --worktree /Users/romain/marin-wt/moe-jax-deepep-benchmark-reintegration \
    --task-id deepep-jax-layout-stats-20260316-094520 \
    --tokens 262144 \
    --topk-list 2,8 \
    --distributions random,runs \
    --layout-stats-only \
    --warmup 0 \
    --iters 1 \
    --build-with-torch-extension \
    --load-as-python-module
  ```
- Result:
  - `random, topk=2`:
    - wrapper `recv_capacity_per_rank=262144`
    - actual `max_recv_tokens=61836`
    - wrapper `assignment_rows_per_rank=524288`
    - actual `max_local_assignments=65715`
    - overprovision:
      - recv capacity: `4.24x`
      - assignment rows: `7.98x`
  - `runs, topk=2`:
    - actual `max_recv_tokens=62019`
    - actual `max_local_assignments=65849`
    - overprovision:
      - recv capacity: `4.23x`
      - assignment rows: `7.96x`
  - `random, topk=8`:
    - actual `max_recv_tokens=175129`
    - actual `max_local_assignments=262690`
    - overprovision:
      - recv capacity: `1.50x`
      - assignment rows: `7.98x`
  - `runs, topk=8`:
    - actual `max_recv_tokens=175156`
    - actual `max_local_assignments=262514`
    - overprovision:
      - recv capacity: `1.50x`
      - assignment rows: `7.99x`
- Interpretation:
  - The local expert stage is not merely seeing a modestly padded receive buffer.
  - It is compiling against roughly `8x` more assignment rows per rank than the layout actually produces.
  - This is a much stronger candidate explanation for both:
    - the `~5x` larger-token slowdown
    - the prewarm-time OOMs

### 2026-03-16 01:55 PDT - The same overprovision pattern held at `2x` and `4x` scale
- Commands:
  ```bash
  KUBECONFIG=~/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_transport_krt.py \
    --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
    --worktree /Users/romain/marin-wt/moe-jax-deepep-benchmark-reintegration \
    --task-id deepep-jax-layout-stats-20260316-095012 \
    --tokens 524288 \
    --topk-list 2,8 \
    --distributions random,runs \
    --layout-stats-only \
    --warmup 0 \
    --iters 1 \
    --build-with-torch-extension \
    --load-as-python-module
  ```
  ```bash
  KUBECONFIG=~/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_transport_krt.py \
    --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
    --worktree /Users/romain/marin-wt/moe-jax-deepep-benchmark-reintegration \
    --task-id deepep-jax-layout-stats-20260316-095530 \
    --tokens 1048576 \
    --topk-list 2,8 \
    --distributions random,runs \
    --layout-stats-only \
    --warmup 0 \
    --iters 1 \
    --build-with-torch-extension \
    --load-as-python-module
  ```
- Result:
  - `tokens=524288`:
    - `topk=2`: max recv overprovision stayed at about `4.24x`; assignment-row overprovision stayed at about `7.98x`
    - `topk=8`: max recv overprovision stayed at about `1.49x`; assignment-row overprovision stayed at about `7.98x`
  - `tokens=1048576`:
    - `topk=2`: max recv overprovision stayed at about `4.24x`; assignment-row overprovision stayed at about `7.98x`
    - `topk=8`: max recv overprovision stayed at about `1.50x`; assignment-row overprovision stayed at about `7.99x`
- Interpretation:
  - The bad shape inflation is not a small-batch artifact.
  - It scales cleanly with token count and remains almost constant across:
    - `32768/device`
    - `2x`
    - `4x`
    - `random` vs `runs`
    - `topk=2` vs `topk=8`
  - The most stable factual summary is:
    - the current wrapper chooses `recv_capacity_per_rank = local_tokens * world_size`
    - the local expert path then expands to `recv_capacity_per_rank * topk`
    - but the actual per-rank local assignment count stays very close to `local_tokens * topk`
    - so the local expert stage is effectively paying an extra `world_size` factor in rows

### 2026-03-16 02:09 PDT - Exact-cap reintegration turned the first `32768/device` benchmark cell positive
- Command:
  ```bash
  KUBECONFIG=~/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
    --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
    --task-id deepep-jax-krt-bench-20260316-100530 \
    --tokens 262144 \
    --kernels current,deepep_transport_prewarmed,deepep_transport_capped_prewarmed \
    --topk-list 2 \
    --distributions random,runs \
    --bench-pass forward \
    --ep-list 8 \
    --warmup 1 \
    --iters 1 \
    --build-with-torch-extension \
    --load-as-python-module
  ```
- Result so far:
  - `random, topk=2`:
    - `current`: `0.027047 s`, `9.69M tok/s`
    - `deepep_transport_prewarmed`: `0.144642 s`, `1.81M tok/s`
    - `deepep_transport_capped_prewarmed`: `0.022019 s`, `11.91M tok/s`
  - exact caps chosen for that cell:
    - `max_recv_tokens=61952`
    - `max_local_assignments=65920`
    - `recv_factor=4.23x`
    - `assign_factor=7.95x`
- Interpretation:
  - Tightening both the receive cap and the local-assignment row cap is enough to flip the first larger-token benchmark cell from a large loss into a win over `current`.
  - The positive result is already strong evidence that the oversized wrapper shapes, not the DeepEP transport itself, were the dominant reason the reintegrated path lost so badly at `32768/device`.
  - The uncapped path remains operationally noisy: after returning its result, the same pod spent a long time in the familiar teardown-failure tail before moving on.
  - Because that known uncapped teardown stall wastes the warm H100 lane, the next runs should omit `deepep_transport_prewarmed` and compare `current` directly against the capped path at the requested larger sizes.

### 2026-03-16 02:21 PDT - The capped path stayed positive at `32768/device` for both distributions and at `2x`
- Commands:
  ```bash
  KUBECONFIG=~/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
    --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
    --task-id deepep-jax-krt-bench-20260316-021650 \
    --tokens 524288 \
    --kernels current,deepep_transport_capped_prewarmed \
    --topk-list 2 \
    --distributions random \
    --bench-pass forward \
    --ep-list 8 \
    --warmup 1 \
    --iters 1 \
    --build-with-torch-extension \
    --load-as-python-module
  ```
  ```bash
  KUBECONFIG=~/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
    --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
    --task-id deepep-jax-krt-bench-20260316-022100 \
    --tokens 524288 \
    --kernels current,deepep_transport_capped_prewarmed \
    --topk-list 2 \
    --distributions runs \
    --bench-pass forward \
    --ep-list 8 \
    --warmup 1 \
    --iters 1 \
    --build-with-torch-extension \
    --load-as-python-module
  ```
- Result:
  - `tokens=262144`, `topk=2`:
    - `random`:
      - `current`: `0.027047 s`, `9.69M tok/s`
      - `capped`: `0.022019 s`, `11.91M tok/s`
      - speedup over `current`: `1.23x`
    - `runs`:
      - `current`: `0.027035 s`, `9.70M tok/s`
      - `capped`: `0.022958 s`, `11.42M tok/s`
      - speedup over `current`: `1.18x`
  - `tokens=524288`, `topk=2`:
    - `random`:
      - `current`: `0.055837 s`, `9.39M tok/s`
      - `capped`: `0.045087 s`, `11.63M tok/s`
      - speedup over `current`: `1.24x`
    - `runs`:
      - `current`: `0.056047 s`, `9.35M tok/s`
      - `capped`: `0.045484 s`, `11.53M tok/s`
      - speedup over `current`: `1.21x`
  - exact caps at `tokens=524288` stayed perfectly aligned across both distributions:
    - `max_recv_tokens=123904`
    - `max_local_assignments=131712`
    - `recv_factor=4.23x`
    - `assign_factor=7.96x`
- Interpretation:
  - The positive effect is not confined to one random control cell.
  - It reproduced on:
    - both `random` and `runs`
    - `32768/device`
    - `2x`
  - The speedup remained stable, at about `1.18x` to `1.24x`, which is exactly the kind of pattern you would expect if the benchmark path had been paying a systematic shape-inflation tax rather than hitting a one-off artifact.
  - The operational nuisance remains the same:
    - after the capped timing prints, the pod frequently falls into the familiar XLA / CUDA teardown tail
    - the fastest way to keep the H100 lane productive is to harvest each result and then move on to the next cell with a fresh pod

### 2026-03-16 02:31 PDT - The topk-2 token sweep stayed positive through `4x`
- Commands:
  ```bash
  KUBECONFIG=~/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
    --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
    --task-id deepep-jax-krt-bench-20260316-022420 \
    --tokens 1048576 \
    --kernels current,deepep_transport_capped_prewarmed \
    --topk-list 2 \
    --distributions random \
    --bench-pass forward \
    --ep-list 8 \
    --warmup 1 \
    --iters 1 \
    --build-with-torch-extension \
    --load-as-python-module
  ```
  ```bash
  KUBECONFIG=~/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
    --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
    --task-id deepep-jax-krt-bench-20260316-022820 \
    --tokens 1048576 \
    --kernels current,deepep_transport_capped_prewarmed \
    --topk-list 2 \
    --distributions runs \
    --bench-pass forward \
    --ep-list 8 \
    --warmup 1 \
    --iters 1 \
    --build-with-torch-extension \
    --load-as-python-module
  ```
- Result:
  - `tokens=1048576`, `topk=2`:
    - `random`:
      - `current`: `0.123285 s`, `8.51M tok/s`
      - `capped`: `0.087961 s`, `11.92M tok/s`
      - speedup over `current`: `1.40x`
    - `runs`:
      - `current`: `0.114987 s`, `9.12M tok/s`
      - `capped`: `0.091437 s`, `11.47M tok/s`
      - speedup over `current`: `1.26x`
  - exact caps at `tokens=1048576`:
    - `random`:
      - `max_recv_tokens=247424`
      - `max_local_assignments=262784`
      - `recv_factor=4.24x`
      - `assign_factor=7.98x`
    - `runs`:
      - `max_recv_tokens=246912`
      - `max_local_assignments=262528`
      - `recv_factor=4.25x`
      - `assign_factor=7.99x`
- Combined topk-2 sweep summary:
  - `tokens=262144`:
    - `random`: `1.23x`
    - `runs`: `1.18x`
  - `tokens=524288`:
    - `random`: `1.24x`
    - `runs`: `1.21x`
  - `tokens=1048576`:
    - `random`: `1.40x`
    - `runs`: `1.26x`
- Interpretation:
  - The capped reintegration fix now survives the full three-point token sweep the user asked for.
  - The gain did not wash out at larger token count; if anything, the best cell improved further at `4x`.
  - The next highest-value question is no longer whether the topk-2 fix is real.
  - It is whether the same exact-cap reintegration unlocks the previously blocked `topk=8` regime, where layout-only sweeps showed the same `~8x` local-assignment overprovision even though receive-cap overprovision was much smaller.

### 2026-03-16 02:38 PDT - Exact-cap reintegration also unlocked and won the previously blocked `topk=8` regime
- Commands:
  ```bash
  KUBECONFIG=~/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
    --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
    --task-id deepep-jax-krt-bench-20260316-023240 \
    --tokens 262144 \
    --kernels current,deepep_transport_capped_prewarmed \
    --topk-list 8 \
    --distributions random \
    --bench-pass forward \
    --ep-list 8 \
    --warmup 1 \
    --iters 1 \
    --build-with-torch-extension \
    --load-as-python-module
  ```
  ```bash
  KUBECONFIG=~/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
    --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
    --task-id deepep-jax-krt-bench-20260316-023700 \
    --tokens 262144 \
    --kernels current,deepep_transport_capped_prewarmed \
    --topk-list 8 \
    --distributions runs \
    --bench-pass forward \
    --ep-list 8 \
    --warmup 1 \
    --iters 1 \
    --build-with-torch-extension \
    --load-as-python-module
  ```
- Result:
  - `tokens=262144`, `topk=8`:
    - `random`:
      - `current`: `0.091821 s`, `2.85M tok/s`
      - `capped`: `0.076401 s`, `3.43M tok/s`
      - speedup over `current`: `1.20x`
    - `runs`:
      - `current`: `0.095239 s`, `2.75M tok/s`
      - `capped`: `0.076829 s`, `3.41M tok/s`
      - speedup over `current`: `1.24x`
  - exact caps at `tokens=262144`, `topk=8`:
    - `random`:
      - `max_recv_tokens=175360`
      - `max_local_assignments=262912`
      - `recv_factor=1.49x`
      - `assign_factor=7.98x`
    - `runs`:
      - `max_recv_tokens=175360`
      - `max_local_assignments=263040`
      - `recv_factor=1.49x`
      - `assign_factor=7.97x`
- Interpretation:
  - This regime had previously been blocked by exact-prewarm OOM in the uncapped path.
  - The exact-cap reintegration removed that blocker and produced a win over `current` immediately.
  - The result also sharpens the causal story:
    - at `topk=8`, receive-cap overprovision was only about `1.5x`
    - local-assignment overprovision was still about `8x`
    - and tightening those assignment-row shapes was enough to both restore feasibility and improve throughput

### 2026-03-16 02:47 PDT - The `topk=8` win also held at `2x`
- Commands:
  ```bash
  KUBECONFIG=~/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
    --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
    --task-id deepep-jax-krt-bench-20260316-024150 \
    --tokens 524288 \
    --kernels current,deepep_transport_capped_prewarmed \
    --topk-list 8 \
    --distributions random \
    --bench-pass forward \
    --ep-list 8 \
    --warmup 1 \
    --iters 1 \
    --build-with-torch-extension \
    --load-as-python-module
  ```
  ```bash
  KUBECONFIG=~/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
    --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
    --task-id deepep-jax-krt-bench-20260316-024700 \
    --tokens 524288 \
    --kernels current,deepep_transport_capped_prewarmed \
    --topk-list 8 \
    --distributions runs \
    --bench-pass forward \
    --ep-list 8 \
    --warmup 1 \
    --iters 1 \
    --build-with-torch-extension \
    --load-as-python-module
  ```
- Result:
  - `tokens=524288`, `topk=8`:
    - `random`:
      - `current`: `0.193091 s`, `2.72M tok/s`
      - `capped`: `0.156848 s`, `3.34M tok/s`
      - speedup over `current`: `1.23x`
    - `runs`:
      - `current`: `0.191187 s`, `2.74M tok/s`
      - `capped`: `0.158486 s`, `3.31M tok/s`
      - speedup over `current`: `1.21x`
  - exact caps at `tokens=524288`, `topk=8`:
    - `random`:
      - `max_recv_tokens=350208`
      - `max_local_assignments=524928`
      - `recv_factor=1.50x`
      - `assign_factor=7.99x`
    - `runs`:
      - `max_recv_tokens=350336`
      - `max_local_assignments=524928`
      - `recv_factor=1.50x`
      - `assign_factor=7.99x`
- Interpretation:
  - The `topk=8` story is no longer just “the smaller blocked case now fits.”
  - It now reproduces at both:
    - `32768/device`
    - `2x`
  - The stable numbers again line up with the earlier shape-only diagnosis:
    - modest receive-cap inflation
    - but persistent `~8x` local-assignment inflation
    - and a consistent benchmark win once those assignment shapes are tightened

### 2026-03-16 09:22 PDT - Exact-cap DeepEP also wins on the original fixed-shape small-shape forward quadrant
- Context:
  - The original `#3633` regime is the fixed-shape H100x8 benchmark at:
    - `tokens=32768`
    - `hidden=2048`
    - `mlp_dim=768`
    - `experts=128`
    - `shared_expert_dim=2048`
    - `EP=8`
    - `distribution in {random, runs}`
    - `topk in {2, 8}`
  - This run used the isolated CoreWeave lane and the exact-cap reintegration launcher:
    - pod: `iris-task-349c4b509b10`
    - task: `deepep-jax-krt-bench-20260316-fixedshape-jax-quadrant-kill`
- Command:
  ```bash
  KUBECONFIG=~/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
    --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
    --task-id deepep-jax-krt-bench-20260316-fixedshape-jax-quadrant-kill \
    --kernels current,deepep_transport_capped_prewarmed \
    --topk-list 2,8 \
    --distributions random,runs \
    --bench-pass forward \
    --ep-list 8 \
    --warmup 2 \
    --iters 5 \
    --per-bench-timeout-seconds 150 \
    --per-bench-kill-after-seconds 10 \
    --build-with-torch-extension \
    --load-as-python-module
  ```
- Result:
  - `random, topk=2`:
    - `current`: `0.003876 s`, `8.45M tok/s`
    - `capped`: `0.003325 s`, `9.85M tok/s`
    - speedup over `current`: `1.17x`
    - exact caps: `max_recv_tokens=7808`, `max_local_assignments=8320`, `recv_factor=4.20x`, `assign_factor=7.88x`
  - `random, topk=8`:
    - `current`: `0.011067 s`, `2.96M tok/s`
    - `capped`: `0.009854 s`, `3.33M tok/s`
    - speedup over `current`: `1.12x`
    - exact caps: `max_recv_tokens=22144`, `max_local_assignments=33152`, `recv_factor=1.48x`, `assign_factor=7.91x`
  - `runs, topk=2`:
    - `current`: `0.003867 s`, `8.47M tok/s`
    - `capped`: `0.003785 s`, `8.66M tok/s`
    - speedup over `current`: `1.02x`
    - exact caps: `max_recv_tokens=8064`, `max_local_assignments=8448`, `recv_factor=4.06x`, `assign_factor=7.76x`
  - `runs, topk=8`:
    - `current`: `0.011085 s`, `2.96M tok/s`
    - `capped`: `0.010062 s`, `3.26M tok/s`
    - speedup over `current`: `1.10x`
    - exact caps: `max_recv_tokens=22144`, `max_local_assignments=33152`, `recv_factor=1.48x`, `assign_factor=7.91x`
- Interpretation:
  - This is the first complete four-cell small-shape table for the reintegrated full JAX DeepEP transport path.
  - The earlier “negative on the original shape” story no longer holds in forward mode once the same exact-cap fix is used there too.
  - The win is smaller than at the larger token points, but it is now positive on all four cells.
  - The shape-level diagnostics remain consistent with the larger-token story:
    - `topk=2` still shows heavy receive-cap inflation plus heavy local-assignment inflation
    - `topk=8` shows only modest receive-cap inflation, but still roughly `8x` local-assignment inflation
    - tightening those local assignment shapes is enough to move the original fixed-shape benchmark in the right direction too

### 2026-03-16 09:42 PDT - Fixed-shape Megatron full-layer baseline is now captured on the same `#3633` shape
- Context:
  - The JAX forward quadrant above still left the full-layer Torch baseline column empty.
  - I used the existing Megatron Qwen launcher and same-shape cases already encoded in the benchmark script:
    - `.agents/scripts/megatron_qwen_krt_bench.py`
    - `.agents/scripts/megatron_qwen_moe_perf.py`
  - The benchmark cases were:
    - `marin_3633_topk_2`
    - `marin_3633_topk_8`
  - Dispatchers:
    - `alltoall`
    - `deepep`
  - Task:
    - `megatron-qwen-krt-bench-20260316-marin3633-fixedshape`
  - Pod:
    - `iris-task-afca8f8f1ab9`
- Command:
  ```bash
  KUBECONFIG=~/.kube/coreweave-iris uv run .agents/scripts/megatron_qwen_krt_bench.py \
    --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
    --worktree /Users/romain/marin-wt/moe-jax-deepep-benchmark-reintegration \
    --task-id megatron-qwen-krt-bench-20260316-marin3633-fixedshape \
    --cases marin_3633_topk_2,marin_3633_topk_8 \
    --dispatchers alltoall,deepep \
    --warmup-iters 5 \
    --measure-iters 20
  ```
- Environment bring-up result:
  - The pod successfully built and loaded the full Torch stack:
    - `TRANSFORMER_ENGINE_OK 2.12.0`
    - `DEEPEP_OK /opt/conda/lib/python3.11/site-packages/deep_ep/__init__.py`
    - `MEGATRON_OK /tmp/Megatron-LM/megatron/core/parallel_state.py`
    - `HAVE_DEEP_EP True`
    - `HAVE_HYBRIDEP True`
  - The launcher ended with:
    - `BENCH_DONE`
    - `EXIT_CODE=0`
- Result:
  - `marin_3633_topk_2`, `alltoall`:
    - `forward_ms=14.271280`
    - `backward_ms=19.888771`
    - total step: `34.160051 ms`
    - throughput: `0.96M tok/s`
  - `marin_3633_topk_2`, `deepep`:
    - `forward_ms=4.997616`
    - `backward_ms=6.588206`
    - total step: `11.585822 ms`
    - throughput: `2.83M tok/s`
    - speedup over `alltoall`: `2.95x`
  - `marin_3633_topk_8`, `alltoall`:
    - `forward_ms=12.211341`
    - `backward_ms=22.626626`
    - total step: `34.837966 ms`
    - throughput: `0.94M tok/s`
  - `marin_3633_topk_8`, `deepep`:
    - `forward_ms=5.017590`
    - `backward_ms=6.007670`
    - total step: `11.025261 ms`
    - throughput: `2.97M tok/s`
    - speedup over `alltoall`: `3.16x`
- Interpretation:
  - This fills the same-shape full-layer Torch baseline column for the original `#3633` shape.
  - The Torch baseline is strongly positive on both `topk=2` and `topk=8`.
  - This benchmark is same-shape and full-layer, but it is not distribution-controlled in the same way as the JAX hillclimb harness:
    - the JAX harness can force `distribution in {random, runs}`
    - the Megatron benchmark uses its own router behavior
  - So these rows are matched in shape and pass structure, but not perfectly matched in router/data-generation semantics.

### 2026-03-16 09:49 PDT - The missing JAX `forward_backward` column is now blocked by a concrete transport-AD error, not by an unknown benchmark issue
- Context:
  - The remaining major hole in `#3711` after the fixed-shape JAX forward quadrant and fixed-shape Torch baseline was the full JAX `forward_backward` DeepEP column.
  - The benchmark harness was previously blocking this path with a forward-only guard for the exact-cap kernels.
  - I added the smallest possible backward probe:
    - compute exact caps with `_deepep_transport_exact_caps(...)`
    - prewarm the exact local compute executable
    - run `jax.value_and_grad(...)` over `_forward_deepep_transport_capped(...)`
  - Probe task:
    - `deepep-jax-krt-bench-20260316-fb-probe-v2`
  - Pod:
    - `iris-task-44516cf9a7ac`
- Command:
  ```bash
  KUBECONFIG=~/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
    --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
    --task-id deepep-jax-krt-bench-20260316-fb-probe-v2 \
    --kernels current,deepep_transport_capped_prewarmed \
    --topk-list 2 \
    --distributions random \
    --bench-pass forward_backward \
    --ep-list 8 \
    --warmup 1 \
    --iters 1 \
    --per-bench-timeout-seconds 180 \
    --per-bench-kill-after-seconds 10 \
    --build-with-torch-extension \
    --load-as-python-module
  ```
- Result:
  - Control cell:
    - `current`, `random`, `topk=2`, `EP=8`, `forward_backward`
    - `time_s=0.010978`
    - `tokens_per_s=2.98M`
  - Exact-cap probe:
    - the benchmark reached the exact-cap setup and printed:
      - `DEEPEP_EXACT_CAPS max_recv_tokens=7808 max_local_assignments=8320 recv_factor=4.196721 assign_factor=7.876923`
    - it then failed during JAX differentiation with:
      - `ValueError: The FFI call to levanter_deepep_dispatch_intranode cannot be differentiated. You can use jax.custom_jvp or jax.custom_jvp to add support.`
  - The traceback locates the failure precisely:
    - `loss_fn(...)`
    - `_forward_deepep_transport_capped(...)`
    - `_moe_mlp_deepep_transport(...)`
    - `_moe_mlp_ep_deepep_transport_local(...)`
    - `deepep_dispatch_intranode(...)`
    - `jax.ffi.ffi_call(...)`
    - JAX raises from `ffi_call_jvp`
- Interpretation:
  - The missing `forward_backward` JAX DeepEP column is no longer an ambiguous benchmark gap.
  - The exact-cap path currently has no JAX AD support at the transport boundary.
  - The first failing site is the dispatch custom call itself, before any broader backward benchmarking question can even be measured.
