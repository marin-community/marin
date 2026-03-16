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
