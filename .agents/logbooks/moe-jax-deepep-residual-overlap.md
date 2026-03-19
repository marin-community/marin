# JAX DeepEP Residual Overlap: Research Logbook

## Scope
- Goal: reduce the remaining JAX DeepEP gap after the `w13` fix by attacking ring communication, synchronization, and overlap in the live exact-cap and `current` paths.
- Primary metric(s): `time_s` / `tokens_per_s` for `deepep_transport_capped_prewarmed` and `current`, plus matched fixed-shape `forward_backward` throughput against Megatron DeepEP on the `marin_3633_*` family where relevant.
- Constraints:
  - start from the sealed `w13` optimization result on branch `research/moe-jax-deepep-w13-optimization`
  - treat the `w13` FC1 fix as landed baseline, not an open root-cause branch
  - do not reopen `w2` unless new profiling evidence forces it
  - commit and push any benchmark-code change before launching remote pods
  - post to GitHub only for major milestones / discoveries
- GitHub issue: https://github.com/marin-community/marin/issues/3841
- Prior issue: https://github.com/marin-community/marin/issues/3821
- Prior sealed root-cause issue: https://github.com/marin-community/marin/issues/3752
- Experiment ID prefix: `OVLP-RES`

## Baseline
- Date: 2026-03-19
- Code refs:
  - `lib/levanter/src/levanter/grug/grug_moe.py`
  - `lib/levanter/scripts/bench/bench_moe_hillclimb.py`
  - `.agents/scripts/deepep_jax_krt_bench.py`
  - `.agents/scripts/megatron_qwen_krt_bench.py`
- Fixed baseline case:
  - hardware: H100x8 on CoreWeave
  - `tokens=262144`
  - `topk=2`
  - `bench_pass=forward`
  - `ep=8`
  - `distribution=random`
  - `shared_expert_dim=0`
  - `warmup=5`
  - `iters=20`
- Inherited post-`w13` baseline from `cf16bcc29fbf5cf20d54b21b0cc61c1fa7ab9e83`:
  - `deepep_transport_capped_prewarmed`: `21,763,265.78 tok/s` (`12.045 ms`)
  - `current`: `17,120,042.64 tok/s` (`15.312 ms`)
  - `current - capped_prewarmed` residual: `3.267 ms`
- Trustworthy residual facts:
  - post-fix `w2_only` is effectively flat relative to the sealed baseline, so `w2` is not the next primary branch
  - the post-fix `current` trace is no longer dominated by the old `w13` kernel
  - communication accounts for `25.1%` of exclusive profiled duration
  - pre-op gaps are concentrated before `reduce-scatter` and `all-gather`
  - the live ring path in `grug_moe.py` still performs full `all_gather` dispatch and `psum_scatter` collect
- High-signal post-fix profile facts from `scratch/profiles/current-expertpadded-262144-report.md`:
  - top compute op: `nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT` at `21.9%`
  - collectives:
    - `all-gather`: `67,140.606` exclusive across `48` calls
    - `reduce-scatter`: `66,014.115` exclusive across `24` calls
  - pre-op gaps:
    - before `reduce-scatter`: `229,021.318` total gap (`24` occurrences)
    - before `all-gather`: `175,005.335` total gap (`24` occurrences)
- Reduced same-global-token rerun conclusion:
  - JAX exact-cap DeepEP improved on every rerun row after the `w13` fix
  - JAX still does not broadly keep up with Megatron DeepEP on the high-token `forward_backward` rows from `#3717`

## Experiment Log
### 2026-03-19 01:14 UTC - Kickoff for the residual-overlap thread
- Experiment ID: `OVLP-RES-001`
- Hypothesis:
  - the next meaningful gap closure will come from reducing collective volume and/or hiding collective latency in the live ring path, not from reopening the already-successful `w13` micro-branch.
- Command:
  - admin/scaffolding only; no benchmark launched yet on this thread
- Config:
  - branch: `research/moe-jax-deepep-residual-overlap`
  - starting commit: `cf16bcc29fbf5cf20d54b21b0cc61c1fa7ab9e83`
  - predecessor branch: `research/moe-jax-deepep-w13-optimization`
  - 12-hour window marker: `.agents/projects/afk-start-time-20260319T011420Z.txt`
- Result:
  - forked a fresh residual-overlap branch from the completed `w13` thread
  - recorded the 12-hour working window locally so follow-up sessions can reason about remaining time
  - started new logbook rather than continuing the sealed `w13` branch logbook
  - opened follow-up GitHub issue `#3841`
- Interpretation:
  - the next work should be evidence-driven and residual-specific
  - the first enabling task is to add exact-cap `forward_backward` profiling support so the remaining high-token gap can be localized directly
- Next action:
  - create the new GitHub issue, backfill the issue URL here, then patch profiling support for exact-cap `forward_backward`.

### 2026-03-19 01:18 UTC - Add exact-cap `forward_backward` profiling support
- Experiment ID: `OVLP-RES-002`
- Hypothesis:
  - the fastest way to close the remaining JAX-vs-Megatron gap is to profile the live fixed-shape `forward_backward` path directly instead of inferring from forward-only traces.
- Command:

```bash
git commit -m "bench: add fb profile support for capped deepep"
git push origin research/moe-jax-deepep-residual-overlap
```

- Config:
  - commit: `476ef830a530198c82dcd1bb9993483742e7fedd`
  - files changed:
    - `lib/levanter/scripts/bench/bench_moe_hillclimb.py`
- Result:
  - patched the exact-cap benchmark harness so `deepep_transport_capped_prewarmed` accepts `--profile-root` on the `forward_backward` path.
- Interpretation:
  - the residual thread can now localize the fixed-shape gap directly on the JAX path that matters for `#3717`.
- Next action:
  - add the same profiling support for `current` `forward_backward` with `--w13-expert-padded` so the two live residual paths can be compared apples-to-apples.

### 2026-03-19 01:20 UTC - Add `current` `forward_backward` profile support for the live expert-padded path
- Experiment ID: `OVLP-RES-003`
- Hypothesis:
  - if the exact-cap and `current` residuals are going to be compared fairly, the `current` `forward_backward` path also needs to honor `--w13-expert-padded`.
- Command:

```bash
git commit -m "bench: add current fb profile support for w13 expert padding"
git push origin research/moe-jax-deepep-residual-overlap
```

- Config:
  - commit: `d998b41a4cc06f0f407ad094644f44b83219c3d9`
  - files changed:
    - `lib/levanter/scripts/bench/bench_moe_hillclimb.py`
- Result:
  - added `_time_current_forward_backward_w13_expert_padded` and `_profile_current_forward_backward_w13_expert_padded`, and threaded them through the main kernel dispatcher.
- Interpretation:
  - both live JAX paths can now be profiled under the same expert-padded `forward_backward` regime.
- Next action:
  - run the first matched `forward_backward` profile pair on the fresh H100x8 lane.

### 2026-03-19 01:25 UTC - Exact-cap `forward_backward` profile: host-side waiting dominates more than NCCL
- Experiment ID: `OVLP-RES-004`
- Hypothesis:
  - the remaining fixed-shape JAX gap after the `w13` fix is still mostly collective volume / collective latency.
- Command:

```bash
KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3821.yaml \
  --repo-ref 476ef830a530198c82dcd1bb9993483742e7fedd \
  --task-id ovlpres-exactcap-fb-profile-t131072-topk2-20260319-0125 \
  --tokens 131072 \
  --shared-expert-dim 2048 \
  --kernels deepep_transport_capped_prewarmed \
  --topk-list 2 \
  --distributions random \
  --bench-pass forward_backward \
  --ep-list 8 \
  --warmup 2 \
  --iters 3 \
  --profile-root /tmp/ovlpres-exactcap-fb-131072-topk2 \
  --post-bench-sleep-seconds 1800 \
  --per-bench-timeout-seconds 1800 \
  --per-bench-kill-after-seconds 20 \
  --build-with-torch-extension \
  --load-as-python-module \
  --skip-smoke \
  --skip-cleanup \
  --w13-expert-padded \
  --node-selector iris-i3821jax-scale-group=h100-8x
```

- Config:
  - pod: `iris-task-eaf33a55bea7`
  - node: `g11ed54`
  - copied artifacts:
    - `scratch/profiles/ovlpres-exactcap-fb-131072-topk2`
    - `scratch/profiles/ovlpres-exactcap-fb-131072-topk2-summary.json`
    - `scratch/profiles/ovlpres-exactcap-fb-131072-topk2-report.md`
- Result:
  - timed profile run returned `time_s=0.096506` / `1,358,177.36 tok/s`
  - exclusive time breakdown:
    - compute: `43.31%`
    - communication: `1.53%`
    - host: `55.03%`
  - only collective family with meaningful exclusive time was `all-reduce` (`48` calls, `18,681.861` exclusive)
  - the dominant pre-op gap was before `ncclDevKernel_AllReduce_Sum_f32_RING_LL(...)`: `565,816.590` total across `32` occurrences
- Interpretation:
  - exact-cap `forward_backward` is not dominated by large NCCL transport volume after the `w13` fix
  - the remaining cost is much more consistent with host-side delay / FFI boundary waiting / XLA glue arriving late to relatively small collectives
- Next action:
  - profile the live `current` `forward_backward` path under the same shape to see whether the same host-delay pattern survives there.

### 2026-03-19 01:30 UTC - `current` `forward_backward` profile: ring collectives are visible, but exact-cap still has the larger host problem
- Experiment ID: `OVLP-RES-005`
- Hypothesis:
  - if the live residual is mainly the ring path, `current` `forward_backward` should show larger collective share than exact-cap even on the same fixed-shape regime.
- Command:

```bash
KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3821.yaml \
  --repo-ref d998b41a4cc06f0f407ad094644f44b83219c3d9 \
  --task-id ovlpres-current-fb-profile-t131072-topk2-20260319-0130 \
  --tokens 131072 \
  --shared-expert-dim 2048 \
  --kernels current \
  --topk-list 2 \
  --distributions random \
  --bench-pass forward_backward \
  --ep-list 8 \
  --warmup 2 \
  --iters 3 \
  --profile-root /tmp/ovlpres-current-fb-131072-topk2 \
  --post-bench-sleep-seconds 1800 \
  --per-bench-timeout-seconds 1800 \
  --per-bench-kill-after-seconds 20 \
  --build-with-torch-extension \
  --load-as-python-module \
  --skip-smoke \
  --skip-cleanup \
  --w13-expert-padded \
  --node-selector iris-i3821jax-scale-group=h100-8x
```

- Config:
  - pod: `iris-task-69badeef15d5`
  - node: `g11ed54`
  - copied artifacts:
    - `scratch/profiles/ovlpres-current-fb-131072-topk2`
    - `scratch/profiles/ovlpres-current-fb-131072-topk2-summary.json`
    - `scratch/profiles/ovlpres-current-fb-131072-topk2-report.md`
- Result:
  - timed profile run returned `time_s=0.091547` / `1,431,751.17 tok/s`
  - exclusive time breakdown:
    - compute: `57.15%`
    - communication: `20.84%`
    - host: `21.87%`
  - collectives:
    - `reduce-scatter`: `48` calls / `103,064.284` exclusive
    - `all-gather`: `72` calls / `69,045.884` exclusive
    - `all-reduce`: `48` calls / `15,047.405` exclusive
  - dominant pre-op gaps:
    - before `reduce-scatter`: `270,884.052`
    - before `all-gather`: `226,962.519`
    - before `all-reduce`: `75,778.815`
- Interpretation:
  - `current` still has a real ring-collective residual
  - but the exact-cap fixed-shape path is even more skewed toward host-side waiting, which means the remaining JAX-vs-Megatron gap in `#3717` is not explained by ring collectives alone
- Next action:
  - inspect the DeepEP FFI for host synchronizations on the exact-cap backward path and test a narrow candidate there first.

### 2026-03-19 01:33 UTC - Add an opt-in cached recv-count fast path in the DeepEP FFI
- Experiment ID: `OVLP-RES-006`
- Hypothesis:
  - part of the exact-cap fixed-shape host stall comes from repeatedly doing `cudaMemcpyAsync(... DeviceToHost)` plus `cudaStreamSynchronize(stream)` just to read back `num_recv_tokens` in `DispatchIntranodeCached` and `CombineIntranode`.
- Command:

```bash
git commit -m "deepep: add opt-in runtime recv-count fast path"
git push origin research/moe-jax-deepep-residual-overlap
```

- Config:
  - commit: `a42bae6befabebf0da3fb0f0de4712b03ee355f7`
  - files changed:
    - `lib/levanter/src/levanter/kernels/deepep/csrc/deepep_transport_ffi.cu`
    - `.agents/scripts/deepep_jax_krt_bench.py`
  - new opt-in flag:
    - CLI: `--deepep-trust-runtime-recv-count`
    - env: `LEVANTER_DEEPEP_TRUST_RUNTIME_RECV_COUNT=1`
- Result:
  - when the flag is enabled, `DispatchIntranodeCached` and `CombineIntranode` reuse `runtime.last_num_recv_tokens` if the forward dispatch has already populated it; otherwise they fall back to the old D2H copy + stream sync path.
- Interpretation:
  - this is a low-risk discriminator focused exactly on the host-side D2H readback path seen in the FFI.
- Next action:
  - validate the change in the required order on the authoritative forward cell, then run one directly relevant fixed-shape `forward_backward` row.

### 2026-03-19 01:47 UTC - Cached recv-count reuse is flat on both the required forward ladder and the directly relevant fixed-shape row
- Experiment ID: `OVLP-RES-007`
- Hypothesis:
  - if the repeated cached-path D2H readbacks are materially contributing to the residual, then enabling cached recv-count reuse should move either the exact-cap forward rung or the fixed-shape `forward_backward` row where the JAX-vs-Megatron gap still matters.
- Commands:

```bash
KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3821.yaml \
  --repo-ref a42bae6befabebf0da3fb0f0de4712b03ee355f7 \
  --task-id ovlpres-fastrecv-w13only-t262144-20260319-0132 \
  --tokens 262144 \
  --shared-expert-dim 0 \
  --kernels deepep_transport_w13_only_probe \
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
  --skip-cleanup \
  --w13-expert-padded \
  --deepep-trust-runtime-recv-count \
  --node-selector iris-i3821jax-scale-group=h100-8x

KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3821.yaml \
  --repo-ref a42bae6befabebf0da3fb0f0de4712b03ee355f7 \
  --task-id ovlpres-fastrecv-localcompute-t262144-20260319-0138 \
  --tokens 262144 \
  --shared-expert-dim 0 \
  --kernels deepep_transport_local_compute_only_probe \
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
  --skip-cleanup \
  --w13-expert-padded \
  --deepep-trust-runtime-recv-count \
  --node-selector iris-i3821jax-scale-group=h100-8x

KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3821.yaml \
  --repo-ref a42bae6befabebf0da3fb0f0de4712b03ee355f7 \
  --task-id ovlpres-fastrecv-exactcap-t262144-20260319-0140 \
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
  --skip-cleanup \
  --w13-expert-padded \
  --deepep-trust-runtime-recv-count \
  --node-selector iris-i3821jax-scale-group=h100-8x

KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3821.yaml \
  --repo-ref a42bae6befabebf0da3fb0f0de4712b03ee355f7 \
  --task-id ovlpres-fastrecv-fb131072-topk2-20260319-0146 \
  --tokens 131072 \
  --shared-expert-dim 2048 \
  --kernels deepep_transport_capped_prewarmed \
  --topk-list 2 \
  --distributions random \
  --bench-pass forward_backward \
  --ep-list 8 \
  --warmup 1 \
  --iters 2 \
  --per-bench-timeout-seconds 3600 \
  --per-bench-kill-after-seconds 30 \
  --build-with-torch-extension \
  --load-as-python-module \
  --skip-smoke \
  --skip-cleanup \
  --w13-expert-padded \
  --deepep-trust-runtime-recv-count \
  --node-selector iris-i3821jax-scale-group=h100-8x
```

- Config:
  - all runs pinned to `g11ed54` through `iris-i3821jax-scale-group=h100-8x`
  - forward references from the inherited post-`w13` thread:
    - `w13_only`: `0.002428 s` / `107,976,854.82 tok/s`
    - `local_compute_only`: `0.008898 s` / `29,461,265.86 tok/s`
    - `capped_prewarmed`: `0.012045 s` / `21,763,265.78 tok/s`
  - fixed-shape `forward_backward` reference from the reduced `#3717` rerun:
    - `131072, topk=2`: `0.022549 s` / `5,812,763.83 tok/s`
- Result:
  - `w13_only`: `0.002421 s` / `108,257,578.65 tok/s`
  - `local_compute_only`: `0.008903 s` / `29,443,272.98 tok/s`
  - `capped_prewarmed` forward: `0.012006 s` / `21,834,189.07 tok/s`
  - `capped_prewarmed` `forward_backward` at `131072, topk=2, shared_expert_dim=2048`: `0.022533 s` / `5,816,954.42 tok/s`
- Interpretation:
  - every measurement stayed within noise
  - cached recv-count reuse does not appear to be a meaningful lever for the remaining JAX-vs-Megatron gap
  - the important residual is therefore earlier or broader than the downstream D2H readback sites in `DispatchIntranodeCached` and `CombineIntranode`
- Next action:
  - patch in host-dispatch debug timestamps and rerun a narrow exact-cap `forward_backward` debug row to see where the host-side exact-cap delay is actually accumulating.

### 2026-03-19 02:08 UTC - Named-scope exact-cap forward_backward profile rerun confirms the profile is mostly blocked FFI time plus two shared-looking all-reduce scopes
- Experiment ID: `OVLP-RES-008`
- Hypothesis:
  - if the earlier exact-cap `forward_backward` profile was missing enough scope detail to mislead us, then rerunning it from the current branch tip with named backward scopes should show whether the residual is in DeepEP backward custom calls, host-side D2H paths, or elsewhere.
- Commands:

```bash
KUBECONFIG=/home/ubuntu/.kube/coreweave-iris kubectl -n iris-3821-jax delete pod iris-task-452c0f09db71 --ignore-not-found

TASK_ID=ovlpres-bwdscope-exactcap-profile-t131072-topk2-20260319-0207
KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3821.yaml \
  --repo-ref 20a35a371805b909ab2e81a2d42d618c8a58d258 \
  --task-id "$TASK_ID" \
  --tokens 131072 \
  --shared-expert-dim 2048 \
  --kernels deepep_transport_capped_prewarmed \
  --topk-list 2 \
  --distributions random \
  --bench-pass forward_backward \
  --ep-list 8 \
  --warmup 2 \
  --iters 3 \
  --profile-root /tmp/ovlpres-bwdscope-exactcap-fb-131072-topk2 \
  --post-bench-sleep-seconds 1800 \
  --per-bench-timeout-seconds 1800 \
  --per-bench-kill-after-seconds 20 \
  --build-with-torch-extension \
  --load-as-python-module \
  --skip-smoke \
  --skip-cleanup \
  --w13-expert-padded \
  --node-selector iris-i3821jax-scale-group=h100-8x

KUBECONFIG=/home/ubuntu/.kube/coreweave-iris kubectl -n iris-3821-jax cp \
  iris-task-959114f74f32:/tmp/ovlpres-bwdscope-exactcap-fb-131072-topk2 \
  scratch/profiles/ovlpres-bwdscope-exactcap-fb-131072-topk2

KUBECONFIG=/home/ubuntu/.kube/coreweave-iris kubectl -n iris-3821-jax delete pod iris-task-959114f74f32
```

- Config:
  - pinned fresh H100 lane: `iris-i3821jax-scale-group=h100-8x`
  - repo ref: `20a35a371805b909ab2e81a2d42d618c8a58d258`
  - profile artifacts:
    - `scratch/profiles/ovlpres-bwdscope-exactcap-fb-131072-topk2`
    - `scratch/profiles/ovlpres-bwdscope-exactcap-fb-131072-topk2-summary.json`
    - `scratch/profiles/ovlpres-bwdscope-exactcap-fb-131072-topk2-report.md`
  - raw trace inspected:
    - `scratch/profiles/ovlpres-bwdscope-exactcap-fb-131072-topk2/plugins/profile/2026_03_19_02_07_46/g11ed54.trace.json.gz`
- Result:
  - profiled timing run:
    - `deepep_transport_capped_prewarmed`, `131072/topk=2/shared_expert_dim=2048/forward_backward`: `0.095352 s` / `1,374,616.73 tok/s`
  - top-line profile report:
    - compute: `39.62%`
    - communication: `1.43%`
    - host: `58.83%`
    - largest pre-gap: `565697.633 us` total before `ncclDevKernel_AllReduce_Sum_f32_RING_LL(...)`
  - raw-trace scope aggregation:
    - `jit(loss_fn)/transpose(jvp())/shard_map/dispatch/deepep_dispatch_vjp_bwd_combine/ffi_call`: `27099.697 us` total across `72` device events
    - `jit(loss_fn)/transpose(jvp())/shard_map/combine/deepep_combine_vjp_bwd_cached_dispatch/ffi_call`: `18581.351 us` total across `72` device events
    - `jit(loss_fn)/transpose(jvp(td,dm->tm))/dot_general`: `11231.127 us` total across `24` all-reduce events
    - `jit(loss_fn)/jvp()/reduce_sum`: `7875.708 us` total across `24` all-reduce events
  - host-attributed long regions are mostly blocked FFI envelopes rather than standalone CPU work:
    - host `ffi_call.65`: `276904.597 us`
    - host `ffi_call.63`: `141944.184 us`
    - host `ffi_call.64`: `62492.233 us`
    - host `MemcpyD2H`: `480168.753 us` total, despite the trusted-runtime-recv-count fast path already testing flat in `OVLP-RES-007`
- Interpretation:
  - the profile still looks host-heavy at the report layer, but the raw trace shows most of that host attribution is blocked exact-cap custom-call envelope time, not proof that the CPU is the principal residual
  - the two named DeepEP backward scopes are visible and non-trivial, but they are still smaller than the surrounding blocked FFI envelopes
  - the only named collectives that show up clearly are `jit(loss_fn)/transpose(jvp(td,dm->tm))/dot_general` and `jit(loss_fn)/jvp()/reduce_sum`, which strongly suggests that the remaining exact-cap `forward_backward` residual is no longer mainly the old `w13` path
- Next action:
  - isolate the two exact-cap backward DeepEP VJP legs directly with dedicated benchmark kernels before spending time on another whole-step optimization guess.

### 2026-03-19 02:30 UTC - Exact-cap backward DeepEP VJP probes are both small, pointing the residual away from DeepEP backward custom calls
- Experiment ID: `OVLP-RES-009`
- Hypothesis:
  - if the remaining fixed-shape `forward_backward` gap is dominated by DeepEP backward transport/combine VJPs, then a dedicated pullback-only timing probe should come back large on at least one of those two exact-cap backward legs.
- Commands:

```bash
python -m py_compile lib/levanter/scripts/bench/bench_moe_hillclimb.py
uv run python lib/levanter/scripts/bench/bench_moe_hillclimb.py --help | rg -n 'combine_bwd|dispatch_bwd'
git commit -m "bench: add deepep backward probes"
git push origin research/moe-jax-deepep-residual-overlap

TASK_ID=ovlpres-bwdprobes-fb131072-topk2-20260319-0227
KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3821.yaml \
  --repo-ref eb68b9c0904136256c650245d6c860eac5483cd9 \
  --task-id "$TASK_ID" \
  --tokens 131072 \
  --shared-expert-dim 2048 \
  --kernels deepep_transport_combine_bwd_cached_dispatch_probe,deepep_transport_dispatch_bwd_combine_probe \
  --topk-list 2 \
  --distributions random \
  --bench-pass forward_backward \
  --ep-list 8 \
  --warmup 1 \
  --iters 2 \
  --per-bench-timeout-seconds 3600 \
  --per-bench-kill-after-seconds 30 \
  --build-with-torch-extension \
  --load-as-python-module \
  --skip-smoke \
  --w13-expert-padded \
  --node-selector iris-i3821jax-scale-group=h100-8x

KUBECONFIG=/home/ubuntu/.kube/coreweave-iris kubectl -n iris-3821-jax delete pod iris-task-920d1ac711c3 --ignore-not-found
```

- Config:
  - benchmark harness change committed and pushed at `eb68b9c0904136256c650245d6c860eac5483cd9`
  - changed file:
    - `lib/levanter/scripts/bench/bench_moe_hillclimb.py`
  - new kernels:
    - `deepep_transport_combine_bwd_cached_dispatch_probe`
    - `deepep_transport_dispatch_bwd_combine_probe`
  - pinned fresh H100 lane: `iris-i3821jax-scale-group=h100-8x`
- Result:
  - `deepep_transport_combine_bwd_cached_dispatch_probe`: `0.001174 s` / `111,629,824.28 tok/s`
  - `deepep_transport_dispatch_bwd_combine_probe`: `0.001701 s` / `77,059,700.23 tok/s`
  - the batched pod printed DeepEP timeout-check / CUDA launch-failure noise after the first kernel and then spent a long time before the second result, but both result lines were emitted successfully before pod deletion
- Interpretation:
  - both exact-cap backward DeepEP VJP legs are small relative to the full exact-cap `forward_backward` row (`~22.5 ms` unprofiled) and also small relative to the named blocked FFI envelopes in `OVLP-RES-008`
  - the larger of the two is `dispatch_bwd_combine`, but at `1.701 ms` it is still nowhere near large enough to explain the remaining multi-x JAX-vs-Megatron gap by itself
  - this rules down another tempting DeepEP-local hypothesis: the exact-cap backward transport/combine custom calls are not the main residual
  - combined with the raw trace from `OVLP-RES-008`, the highest-signal remaining suspects are now the non-DeepEP collective / synchronization part of the step, especially the two named all-reduce scopes and whatever upstream work is causing the large pre-all-reduce gaps
- Next action:
  - run the same exact-cap `forward_backward` target row with `shared_expert_dim=0` on the pinned lane to test whether the residual is primarily tied to the shared-MLP backward path rather than the DeepEP routed path.

### 2026-03-19 02:34 UTC - Removing shared experts helps, but only closes a minority of the remaining exact-cap forward_backward gap
- Experiment ID: `OVLP-RES-010`
- Hypothesis:
  - if the two named all-reduce scopes in `OVLP-RES-008` are the main reason exact-cap JAX still trails Megatron on the fixed-shape `forward_backward` rows, then zeroing `shared_expert_dim` on the same target row should remove most of the residual.
- Command:

```bash
TASK_ID=ovlpres-exactcap-fb131072-topk2-shared0-20260319-023158
KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3821.yaml \
  --repo-ref eb68b9c0904136256c650245d6c860eac5483cd9 \
  --task-id "$TASK_ID" \
  --tokens 131072 \
  --shared-expert-dim 0 \
  --kernels deepep_transport_capped_prewarmed \
  --topk-list 2 \
  --distributions random \
  --bench-pass forward_backward \
  --ep-list 8 \
  --warmup 1 \
  --iters 2 \
  --per-bench-timeout-seconds 3600 \
  --per-bench-kill-after-seconds 30 \
  --build-with-torch-extension \
  --load-as-python-module \
  --skip-smoke \
  --skip-cleanup \
  --w13-expert-padded \
  --node-selector iris-i3821jax-scale-group=h100-8x

KUBECONFIG=/home/ubuntu/.kube/coreweave-iris kubectl -n iris-3821-jax delete pod iris-task-f291e0d416ca --ignore-not-found
```

- Config:
  - repo ref: `eb68b9c0904136256c650245d6c860eac5483cd9`
  - target row matched to the still-important reduced `#3717` cell except for `shared_expert_dim=0` instead of `2048`
  - comparison point from the fresh pinned shared-expert run:
    - `deepep_transport_capped_prewarmed`, `131072/topk=2/shared_expert_dim=2048/forward_backward`: `0.022533 s` / `5,816,954.42 tok/s`
- Result:
  - `deepep_transport_capped_prewarmed`, `131072/topk=2/shared_expert_dim=0/forward_backward`: `0.018385 s` / `7,129,291.54 tok/s`
  - delta versus `shared_expert_dim=2048` on the same row:
    - time: `-18.41%`
    - throughput: `+22.57%`
- Interpretation:
  - the shared-expert backward path is real and non-trivial, which is consistent with the two named all-reduce scopes in `OVLP-RES-008`
  - but it is not the main explanation for the remaining JAX-vs-Megatron gap
  - even with `shared_expert_dim=0`, JAX exact-cap `forward_backward` on this row is still far from the Megatron anchor from `#3717`
  - that means the deeper residual still lives in the routed exact-cap path itself, or in surrounding non-shared HLO glue / scheduling around that path
- Next action:
  - profile the same exact-cap `forward_backward` row with `shared_expert_dim=0` so the routed-path residual can be read without the shared-expert all-reduce noise layered on top.

### 2026-03-19 03:10 UTC - Collapse-helper rewrites do not move the exact-cap forward residual
- Experiment ID: `OVLP-RES-011`
- Hypothesis:
  - if the remaining exact-cap forward residual is mainly in `_collapse_deepep_local_assignments`, then swapping the collapse lowering should improve `deepep_transport_capped_prewarmed` while leaving `w13_only` and `local_compute_only` flat.
- Config:
  - harness change from the earlier collapse-impl branch remained active at repo ref `a9fce7e0d5e4f83ccce25f051c806c81240cc17f`
  - pinned lane: `iris-i3821jax-scale-group=h100-8x`
  - authoritative cell: `262144 / topk=2 / forward / ep=8 / shared_expert_dim=0 / --w13-expert-padded`
- Result:
  - `sorted_segment_sum`
    - `deepep_transport_w13_only_probe`: `0.002420 s` / `108,314,467.05 tok/s`
    - `deepep_transport_local_compute_only_probe`: `0.008887 s` / `29,496,827.46 tok/s`
    - `deepep_transport_capped_prewarmed`: `0.016304 s` / `16,078,659.62 tok/s`
  - `lax_scatter`
    - `deepep_transport_w13_only_probe`: `0.002424 s` / `108,163,897.78 tok/s`
    - `deepep_transport_local_compute_only_probe`: `0.008872 s` / `29,547,352.60 tok/s`
    - `deepep_transport_capped_prewarmed`: `0.012013 s` / `21,821,018.72 tok/s`
  - pinned comparison point:
    - `deepep_transport_capped_prewarmed` baseline: `0.012006 s` / `21,834,189.07 tok/s`
- Interpretation:
  - `sorted_segment_sum` is a hard regression on the exact-cap path
  - `lax_scatter` is effectively flat
  - this rules down the collapse-helper-lowering branch as a serious next optimization lever
- Next action:
  - decompose the exact-cap forward residual directly with `collapse_only` and `combine_only` stage probes.

### 2026-03-19 03:13 UTC - Stage probes show the exact-cap forward residual is almost entirely collapse + combine
- Experiment ID: `OVLP-RES-012`
- Hypothesis:
  - if the residual still lives upstream in dispatch/pack, then `collapse_only + combine_only` should reconstruct only a minority of `capped_prewarmed - local_compute_only`.
- Commands:

```bash
TASK_ID=ovlpres-stage-collapseonly-t262144-20260319-030938
TASK_ID=ovlpres-stage-combineonly-t262144-20260319-031208
```

- Result:
  - `deepep_transport_collapse_only_probe`: `0.001713 s` / `153,034,349.77 tok/s`
  - `deepep_transport_combine_only_probe`: `0.001927 s` / `136,047,519.68 tok/s`
  - pinned references:
    - `deepep_transport_local_compute_only_probe`: `0.008903 s`
    - `deepep_transport_capped_prewarmed`: `0.012006 s`
  - residual math:
    - integrated residual over local compute: `12.006 ms - 8.903 ms = 3.103 ms`
    - isolated `collapse_only + combine_only`: `1.713 ms + 1.927 ms = 3.640 ms`
- Interpretation:
  - the exact-cap forward residual is essentially all collapse + combine
  - combine is slightly larger than collapse
  - there is no large leftover residual that still has to be in dispatch/pack
- Next action:
  - profile `combine_only_probe` directly and decide whether the next branch is in the DeepEP combine kernel/config or in glue/synchronization around it.

### 2026-03-19 03:18 UTC - `combine_only` profile points at DeepEP combine runtime config, not another HLO-side collapse rewrite
- Experiment ID: `OVLP-RES-013`
- Command:

```bash
TASK_ID=ovlpres-combineonly-profile-t262144-20260319-031513
KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3821.yaml \
  --repo-ref a9fce7e0d5e4f83ccce25f051c806c81240cc17f \
  --task-id "$TASK_ID" \
  --tokens 262144 \
  --shared-expert-dim 0 \
  --kernels deepep_transport_combine_only_probe \
  --topk-list 2 \
  --distributions random \
  --bench-pass forward \
  --ep-list 8 \
  --warmup 2 \
  --iters 3 \
  --profile-root /tmp/ovlpres-combineonly-262144 \
  --post-bench-sleep-seconds 1800 \
  --per-bench-timeout-seconds 1800 \
  --per-bench-kill-after-seconds 20 \
  --build-with-torch-extension \
  --load-as-python-module \
  --skip-smoke \
  --skip-cleanup \
  --w13-expert-padded \
  --node-selector iris-i3821jax-scale-group=h100-8x
```

- Artifacts:
  - raw trace: `scratch/profiles/ovlpres-combineonly-262144`
  - summary: `scratch/profiles/ovlpres-combineonly-262144-summary.json`
  - report: `scratch/profiles/ovlpres-combineonly-262144-report.md`
- Result:
  - profiled timing: `0.030496 s` / `8,596,126.28 tok/s`
  - top GPU ops from the profile report:
    - `deep_ep::intranode::combine<__nv_bfloat16, 8, 768, 4096>`: `38,280.413 us` exclusive across `24` launches
    - `deep_ep::intranode::cached_notify_combine<8>`: `2,327.546 us` exclusive across `24` launches
  - largest pre-op gap bucket:
    - `MemcpyD2D`: `10,536.077 us` total pre-gap across `16` occurrences
- Interpretation:
  - the dominant op is now the DeepEP intranode combine kernel itself, not a JAX-side collapse helper
  - the 8-rank combine runtime config in the JAX bridge remains the small hard-coded default:
    - `num_sms=20`
    - `num_max_send_tokens=4`
    - `num_max_recv_tokens=256`
  - the next cheapest high-signal branch is therefore runtime-config tuning for the exact-cap combine path
- Next action:
  - add override flags for dispatch/combine runtime config to the main MoE harness and launcher, commit/push, then test a first combine-config candidate on the pinned lane.

### 2026-03-19 03:33 UTC - Added process-wide DeepEP dispatch/combine override flags for exact-cap tuning
- Experiment ID: `OVLP-RES-014`
- Hypothesis:
  - if the remaining exact-cap residual is sensitive to DeepEP runtime config, then the main MoE benchmark needs a clean way to override dispatch/combine defaults without rewriting every callsite.
- Commands:

```bash
python -m py_compile \
  lib/levanter/src/levanter/kernels/deepep/transport_ffi.py \
  lib/levanter/src/levanter/kernels/deepep/__init__.py \
  lib/levanter/scripts/bench/bench_moe_hillclimb.py \
  .agents/scripts/deepep_jax_krt_bench.py
uv run python lib/levanter/scripts/bench/bench_moe_hillclimb.py --help | rg 'deepep-(dispatch|combine)-num'
uv run python .agents/scripts/deepep_jax_krt_bench.py --help | rg 'deepep-(dispatch|combine)-num'
uv run pytest lib/haliax/tests/test_ragged_dot_dispatch.py
git commit -m "bench: add deepep transport config overrides"
git push origin research/moe-jax-deepep-residual-overlap
```

- Config:
  - committed and pushed repo ref: `0a3e69ac166fcb583d5e21e2b380c7e3918e74ba`
  - changed files:
    - `.agents/scripts/deepep_jax_krt_bench.py`
    - `lib/levanter/scripts/bench/bench_moe_hillclimb.py`
    - `lib/levanter/src/levanter/kernels/deepep/__init__.py`
    - `lib/levanter/src/levanter/kernels/deepep/transport_ffi.py`
- Result:
  - the JAX bridge now supports process-wide default overrides for `dispatch_config` and `combine_config`
  - the MoE harness and cluster launcher now expose:
    - `--deepep-dispatch-num-sms`
    - `--deepep-dispatch-num-max-send-tokens`
    - `--deepep-dispatch-num-max-recv-tokens`
    - `--deepep-combine-num-sms`
    - `--deepep-combine-num-max-send-tokens`
    - `--deepep-combine-num-max-recv-tokens`
  - local verification passed:
    - `py_compile` on changed files
    - parser help for both scripts
    - `pytest lib/haliax/tests/test_ragged_dot_dispatch.py`
- Interpretation:
  - the next experiment can now tune the exact-cap combine runtime without another code change
  - the first concrete candidate should be a larger 8-rank combine send cap, starting with `--deepep-combine-num-max-send-tokens 8`
- Next action:
  - run the required forward validation ladder on the first combine-config candidate:
    1. `deepep_transport_w13_only_probe`
    2. `deepep_transport_local_compute_only_probe`
    3. `deepep_transport_capped_prewarmed`
    4. `current` only if warranted

### 2026-03-19 03:39 UTC - First exact-cap combine send-cap candidate (`combine_num_max_send_tokens=8`) produces a small real win
- Experiment ID: `OVLP-RES-015`
- Hypothesis:
  - if the exact-cap residual is partly caused by the 8-rank combine path using a too-small send cap (`4`), then raising only `combine_num_max_send_tokens` should leave the `w13_only` and local-compute guardrails flat while improving `deepep_transport_capped_prewarmed`.
- Commands:

```bash
KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3821.yaml \
  --repo-ref 0a3e69ac166fcb583d5e21e2b380c7e3918e74ba \
  --task-id ovlpres-cfg-combsend8-w13only-t262144-20260319-0337 \
  --tokens 262144 \
  --shared-expert-dim 0 \
  --kernels deepep_transport_w13_only_probe \
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
  --w13-expert-padded \
  --deepep-combine-num-max-send-tokens 8 \
  --node-selector iris-i3821jax-scale-group=h100-8x
KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3821.yaml \
  --repo-ref 0a3e69ac166fcb583d5e21e2b380c7e3918e74ba \
  --task-id ovlpres-cfg-combsend8-localcompute-t262144-20260319-0331 \
  --tokens 262144 \
  --shared-expert-dim 0 \
  --kernels deepep_transport_local_compute_only_probe \
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
  --w13-expert-padded \
  --deepep-combine-num-max-send-tokens 8 \
  --node-selector iris-i3821jax-scale-group=h100-8x
KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3821.yaml \
  --repo-ref 0a3e69ac166fcb583d5e21e2b380c7e3918e74ba \
  --task-id ovlpres-cfg-combsend8-exactcap-t262144-20260319-0335 \
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
  --w13-expert-padded \
  --deepep-combine-num-max-send-tokens 8 \
  --node-selector iris-i3821jax-scale-group=h100-8x
```

- Result:
  - `deepep_transport_w13_only_probe`: `0.002425 s`, `108,120,102.47 tok/s`
  - `deepep_transport_local_compute_only_probe`: `0.008867 s`, `29,562,545.06 tok/s`
  - `deepep_transport_capped_prewarmed`: `0.011843 s`, `22,135,269.86 tok/s`
  - reference ladder on the same pinned lane:
    - `w13_only`: `0.002421 s`, `108,257,578.65 tok/s`
    - `local_compute_only`: `0.008903 s`, `29,443,272.98 tok/s`
    - `capped_prewarmed`: `0.012006 s`, `21,834,189.07 tok/s`
- Interpretation:
  - the first config-only exact-cap tuning branch is real but small:
    - `w13_only` stayed flat
    - `local_compute_only` stayed flat-to-slightly-better
    - `capped_prewarmed` improved by about `1.36%` time / `1.38%` throughput
  - the gain is too small to justify a `current` rung yet, but it establishes `combine_num_max_send_tokens=8` as the best current exact-cap config among the settings tried so far
- Next action:
  - try a stronger send-cap candidate in a single three-kernel pod to see whether the exact-cap gain scales or peaks near `8`.

### 2026-03-19 03:48 UTC - Larger combine send cap (`16`) stays valid but regresses versus the `8`-token candidate
- Experiment ID: `OVLP-RES-016`
- Hypothesis:
  - if the `send=8` win came from reducing combine chunking overhead, then pushing further to `combine_num_max_send_tokens=16` might continue improving exact-cap.
- Command:

```bash
KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3821.yaml \
  --repo-ref 0a3e69ac166fcb583d5e21e2b380c7e3918e74ba \
  --task-id ovlpres-cfg-combsend16-ladder-t262144-20260319-034044 \
  --tokens 262144 \
  --shared-expert-dim 0 \
  --kernels deepep_transport_w13_only_probe,deepep_transport_local_compute_only_probe,deepep_transport_capped_prewarmed \
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
  --w13-expert-padded \
  --deepep-combine-num-max-send-tokens 16 \
  --node-selector iris-i3821jax-scale-group=h100-8x
```

- Result:
  - `deepep_transport_w13_only_probe`: `0.002438 s`, `107,539,887.84 tok/s`
  - `deepep_transport_local_compute_only_probe`: `0.008859 s`, `29,590,271.35 tok/s`
  - `deepep_transport_capped_prewarmed`: `0.011886 s`, `22,054,333.62 tok/s`
- Interpretation:
  - this is still a mild improvement over the pinned exact-cap baseline, but it is worse than the `send=8` candidate on the only rung that matters
  - the likely conclusion is that the useful region for this knob is near `8`, not “bigger is always better”
- Next action:
  - stop increasing the combine send cap and try a different combine-side knob.

### 2026-03-19 03:57 UTC - Independent combine-SM tuning is invalid on the exact-cap path
- Experiment ID: `OVLP-RES-017`
- Hypothesis:
  - if the exact-cap combine kernel is under-occupying the H100s even after the send-cap tweak, then keeping `combine_num_max_send_tokens=8` while increasing `combine_num_sms` from `20` to `32` could improve the exact-cap rung.
- Command:

```bash
KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3821.yaml \
  --repo-ref 0a3e69ac166fcb583d5e21e2b380c7e3918e74ba \
  --task-id ovlpres-cfg-comb32-send8-ladder-t262144-20260319-035000 \
  --tokens 262144 \
  --shared-expert-dim 0 \
  --kernels deepep_transport_w13_only_probe,deepep_transport_local_compute_only_probe,deepep_transport_capped_prewarmed \
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
  --w13-expert-padded \
  --deepep-combine-num-sms 32 \
  --deepep-combine-num-max-send-tokens 8 \
  --node-selector iris-i3821jax-scale-group=h100-8x
```

- Result:
  - `deepep_transport_w13_only_probe`: `0.002422 s`, `108,219,863.45 tok/s`
  - `deepep_transport_local_compute_only_probe`: `0.008878 s`, `29,527,580.56 tok/s`
  - `deepep_transport_capped_prewarmed`: failed with
    - `INVALID_ARGUMENT: DeepEP intranode combine handle tensor shapes are invalid`
  - pod had to be deleted after the exact-cap failure because the wrapper never observed a clean container exit:
    - `iris-task-d71c08803265`
- Interpretation:
  - this is not just a “bad tuning point”; it exposed a real exact-cap constraint in the JAX FFI path
  - the shape contract is currently inconsistent for independent combine-SM tuning:
    - `lib/levanter/src/levanter/kernels/deepep/transport_ffi.py` sizes the dispatch handle tensors from `resolved_dispatch_config.num_sms // 2`
    - `lib/levanter/src/levanter/kernels/deepep/csrc/deepep_transport_ffi.cu` validates combine handle tensors against `runtime.combine_num_channels()`
  - in other words, the exact-cap path currently assumes the combine handle layout matches the dispatch channel count; changing `combine_num_sms` alone makes the handle tensor shapes invalid before the combine kernel launches
- Next action:
  - treat `combine_num_sms != dispatch_num_sms` as unsupported for the exact-cap path until the handle-layout contract is fixed, and continue with shape-safe combine tuning such as `num_max_recv_tokens`.

### 2026-03-19 04:19 UTC - Add a local-compute backward probe for the exact-cap fixed-shape row
- Experiment ID: `OVLP-RES-018`
- Hypothesis:
  - if the remaining `131072/topk=2/shared_expert_dim=0/forward_backward` exact-cap residual is now mostly in the routed local compute backward rather than in DeepEP transport/combine VJPs, then a dedicated `local_compute` pullback-only probe should come back large and should show the same hot GEMM/scatter/transpose families as the shared-0 exact-cap trace.
- Command:

```bash
git commit -m "bench: add local compute backward probe"
git push origin research/moe-jax-deepep-residual-overlap
git commit -m "bench: fix local compute backward probe staging"
git push origin research/moe-jax-deepep-residual-overlap
```

- Config:
  - commits:
    - `726b8b7f1e6b1a55c27a90e8881c6f4709613adf`
    - `66c91a2fac9c491eaca45cf5f9e49f4a395d9a80`
  - files changed:
    - `lib/levanter/scripts/bench/bench_moe_hillclimb.py`
  - local verification:
    - `python -m py_compile lib/levanter/scripts/bench/bench_moe_hillclimb.py`
    - `uv run python lib/levanter/scripts/bench/bench_moe_hillclimb.py --help | rg local_compute_bwd_probe`
- Result:
  - added `deepep_transport_local_compute_bwd_probe`.
  - the first pinned launch from `726b8b7f1e6b1a55c27a90e8881c6f4709613adf` failed immediately with
    - `UnboundLocalError: cannot access local variable 'x_dispatch' where it is not associated with a value`
    - pod: `iris-task-7554dddedf1e`
  - the rerun from `66c91a2fac9c491eaca45cf5f9e49f4a395d9a80` completed far enough to emit:
    - `RESULT kernel=deepep_transport_local_compute_bwd_probe ep=8 pass=forward_backward time_s=0.592834 tokens_per_s=221093.80`
    - pod: `iris-task-6fd04116bbaa`
  - the profiled rerun also hit:
    - `A large amount of constants were captured during lowering (4.73GB total)`
    - `xla.HloProto exceeded maximum protobuf size of 2GB: 3643875661`
    - post-run cleanup emitted repeated `DeepEP timeout check failed` and `CUDA_ERROR_LAUNCH_FAILED` noise even though the wrapper observed `bench_status=0`.
  - copied artifacts:
    - `scratch/profiles/ovlpres-localcomputebwd-fb-131072-topk2-shared0`
    - `scratch/profiles/ovlpres-localcomputebwd-fb-131072-topk2-shared0-summary.json`
    - `scratch/profiles/ovlpres-localcomputebwd-fb-131072-topk2-shared0-report.md`
  - summary/report highlights from the copied trace:
    - time breakdown: compute `7.90%`, host `92.10%`, communication `0.00%`
    - top ops: `input_scatter_fusion`, `nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT`, `input_transpose_fusion_1`
- Interpretation:
  - the profiled timing is not directly comparable to steady-state because the profile itself blew through the 2GB HLO proto limit and forced a pathological slow path.
  - the structural signal is still useful: the local-compute backward slice is large enough to trigger enormous captured constants and the same scatter/GEMM/transpose families that dominate the shared-0 exact-cap compute trace.
- Next action:
  - rerun the same probe unprofiled on the pinned H100 and then split the local-compute backward slice into narrower staged backward probes.

### 2026-03-19 04:33 UTC - Split the exact-cap local-compute backward slice into staged `w13` and `w2` probes
- Experiment ID: `OVLP-RES-019`
- Hypothesis:
  - if the shared-0 exact-cap residual is now concentrated in routed local compute backward, then splitting that slice into `w13`-only versus `w2`-only backward probes should tell us whether the next code change should attack the first ragged dot backward or the second ragged dot backward.
- Command:

```bash
git commit -m "bench: add staged exact-cap backward probes"
git push origin research/moe-jax-deepep-residual-overlap
```

- Config:
  - commit: `3bb12f8a61e8a37c529b2791230d26b7cb903c8f`
  - files changed:
    - `lib/levanter/scripts/bench/bench_moe_hillclimb.py`
  - local verification:
    - `python -m py_compile lib/levanter/scripts/bench/bench_moe_hillclimb.py`
    - `uv run python lib/levanter/scripts/bench/bench_moe_hillclimb.py --help | rg 'w13_only_bwd_probe|w2_only_bwd_probe'`
- Result:
  - added:
    - `deepep_transport_w13_only_bwd_probe`
    - `deepep_transport_w2_only_bwd_probe`
  - both are wired through the main forward/backward probe dispatcher and use the exact-cap staged inputs on the pinned lane.
- Interpretation:
  - the next benchmark cycle can attribute the routed backward residual without changing the live kernel path yet.
- Next action:
  - run `w13_only_bwd_probe`, `w2_only_bwd_probe`, and `local_compute_bwd_probe` together on the pinned shared-0 row.

### 2026-03-19 04:36 UTC - Repair the KRT launcher after removing a required `npx` dependency
- Experiment ID: `OVLP-RES-020`
- Hypothesis:
  - the launcher cleanup that removed in-pod package installation was too aggressive because the Iris editable build regenerates protobuf outputs during `uv sync` when `npx` is available and required.
- Command:

```bash
git commit -m "tools: restore npx for krt bench pods"
git push origin research/moe-jax-deepep-residual-overlap
```

- Config:
  - bad launcher commit: `1bce285ec12066463a8462d0b327c2436465cb86`
  - fix commit: `26377202e3dba9059edc5cfe54e02fdedf2a08ea`
  - files changed:
    - `.agents/scripts/deepep_jax_krt_bench.py`
  - local verification:
    - `python -m py_compile .agents/scripts/deepep_jax_krt_bench.py`
- Result:
  - the first unprofiled rerun attempt from `1bce285ec12066463a8462d0b327c2436465cb86` failed during `uv sync` with:
    - `RuntimeError: Protobuf outputs are missing and npx is not installed.`
    - pod: `iris-task-b629471fce25`
  - the launcher now conditionally runs `apt-get update && apt-get install -y nodejs npm` only when `npx` is absent.
- Interpretation:
  - the pod bootstrap got slower again, but the launcher is functionally correct and no longer fails before the benchmark starts.
- Next action:
  - relaunch the staged backward attribution run from the fixed launcher commit.

### 2026-03-19 04:37 UTC - Staged backward attribution rerun is in flight on the pinned H100 lane
- Experiment ID: `OVLP-RES-021`
- Hypothesis:
  - if one routed backward stage is the dominant remaining exact-cap residual, then a single pinned shared-0 run over `w13_only_bwd_probe`, `w2_only_bwd_probe`, and `local_compute_bwd_probe` should identify it cleanly.
- Command:

```bash
KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3821.yaml \
  --repo-ref 26377202e3dba9059edc5cfe54e02fdedf2a08ea \
  --task-id ovlpres-stagedbwd-fb131072-topk2-shared0-rerun-20260319-043734 \
  --tokens 131072 \
  --shared-expert-dim 0 \
  --kernels deepep_transport_w13_only_bwd_probe,deepep_transport_w2_only_bwd_probe,deepep_transport_local_compute_bwd_probe \
  --topk-list 2 \
  --distributions random \
  --bench-pass forward_backward \
  --ep-list 8 \
  --warmup 2 \
  --iters 5 \
  --per-bench-timeout-seconds 1800 \
  --per-bench-kill-after-seconds 20 \
  --build-with-torch-extension \
  --load-as-python-module \
  --skip-smoke \
  --skip-cleanup \
  --w13-expert-padded \
  --node-selector iris-i3821jax-scale-group=h100-8x
```

- Config:
  - pod: `iris-task-9180096f0166`
  - node: `g11ed54`
- Interim status:
  - pod is running on the pinned node.
  - `BENCH_START kernel=deepep_transport_w13_only_bwd_probe distribution=random topk=2` has appeared.
  - lowering again reports a very large captured-constants warning:
    - `A large amount of constants were captured during lowering (2.02GB total)`
- Next action:
  - wait for `RESULT` lines, then choose the first real code candidate from the dominant staged backward slice.

### 2026-03-19 04:52 UTC - Staged backward attribution closes on a split residual rather than a chained-overhead spike
- Experiment ID: `OVLP-RES-021`
- Result:
  - `deepep_transport_w13_only_bwd_probe`: `0.003433 s` / `38,179,646.02 tok/s`
  - `deepep_transport_w2_only_bwd_probe`: `0.003251 s` / `40,321,856.23 tok/s`
  - `deepep_transport_local_compute_bwd_probe`: `0.006558 s` / `19,987,362.12 tok/s`
  - the pod completed successfully and was deleted to free the pinned lane:
    - pod: `iris-task-9180096f0166`
- Interpretation:
  - the routed backward residual is split across both local-compute stages:
    - `w13_only_bwd + w2_only_bwd = 6.684 ms`
    - `local_compute_bwd = 6.558 ms`
  - that is close enough to rule out a large extra steady-state chaining penalty in the exact-cap backward local-compute path.
  - the earlier profiled `0.592834 s` outlier was therefore a profiling/lowering pathology, not the representative runtime of the chained backward compute.
  - the next candidate should target one or both matmul stages directly rather than a generic “chained local compute” hypothesis.
- Next action:
  - implement the narrowest opt-in candidate that can move the exact-cap local-compute rung, starting with `w2` expert padding because `w13` expert padding is already landed.

### 2026-03-19 05:05 UTC - Add an opt-in `w2` expert-padded exact-cap path and launch the forward validation ladder
- Experiment ID: `OVLP-RES-022`
- Hypothesis:
  - if the post-`w13` exact-cap local-compute residual is now materially split across `w13` and `w2`, then applying the same expert-padded batched-dot strategy to `w2` should improve `deepep_transport_local_compute_only_probe` and propagate some of that win into `deepep_transport_capped_prewarmed`.
- Command:

```bash
git commit -m "bench: add w2 expert-padded exact-cap path"
git push origin research/moe-jax-deepep-residual-overlap

KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3821.yaml \
  --repo-ref b63556b2993164133eb7aebd2d1269f578c4ed09 \
  --task-id ovlpres-w2pad-ladder-t262144-r2-20260319-0511 \
  --tokens 262144 \
  --shared-expert-dim 0 \
  --kernels deepep_transport_w13_only_probe,deepep_transport_local_compute_only_probe,deepep_transport_capped_prewarmed \
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
  --skip-cleanup \
  --w13-expert-padded \
  --w2-expert-padded \
  --deepep-combine-num-max-send-tokens 8 \
  --node-selector iris-i3821jax-scale-group=h100-8x
```

- Config:
  - commit: `b63556b2993164133eb7aebd2d1269f578c4ed09`
  - files changed:
    - `lib/levanter/scripts/bench/bench_moe_hillclimb.py`
    - `.agents/scripts/deepep_jax_krt_bench.py`
  - local verification:
    - `python -m py_compile lib/levanter/scripts/bench/bench_moe_hillclimb.py .agents/scripts/deepep_jax_krt_bench.py`
    - `uv run python lib/levanter/scripts/bench/bench_moe_hillclimb.py --help | rg 'w2-expert-padded|w13-expert-padded'`
    - `uv run python .agents/scripts/deepep_jax_krt_bench.py --help | rg 'w2-expert-padded|w13-expert-padded'`
  - validation baseline on the pinned lane with `combine_num_max_send_tokens=8`:
    - `deepep_transport_w13_only_probe`: `0.002425 s`
    - `deepep_transport_local_compute_only_probe`: `0.008867 s`
    - `deepep_transport_capped_prewarmed`: `0.011843 s`
- Result:
  - the first ladder launch used a mistyped full SHA and failed before checkout with `HTTP Error 404: Not Found`; I deleted pod `iris-task-aef7ba3c4b7a` and relaunched from the correct commit.
  - corrected ladder:
    - pod: `iris-task-ebc5dab9ba00`
    - node: `g11ed54`
    - `deepep_transport_w13_only_probe`: `0.002426 s` / `108,047,212.04 tok/s`
    - `deepep_transport_local_compute_only_probe`: `0.005050 s` / `51,911,860.65 tok/s`
    - `deepep_transport_capped_prewarmed`: `0.007973 s` / `32,880,566.00 tok/s`
    - pod completed with `EXIT_CODE=0` and was deleted after capture to free the pinned lane
- Interpretation:
  - the candidate is narrow, opt-in, and directly targeted at the exact-cap local-compute rung rather than transport or combine.
  - it passes the required forward ladder cleanly:
    - `w13_only` stayed flat versus the `send=8` baseline (`0.002425 s -> 0.002426 s`)
    - `local_compute_only` improved by about `43.0%` time / `76.0%` throughput (`0.008867 s -> 0.005050 s`)
    - `capped_prewarmed` improved by about `32.7%` time / `48.5%` throughput (`0.011843 s -> 0.007973 s`)
  - this is large enough to skip the optional `current` rung and go directly to the reduced shared-2048 `forward_backward` reruns from `#3717`.
- Next action:
  - rerun the large-token JAX DeepEP shared-2048 `forward_backward` rows with `2` measurement iterations to measure how much of the forward exact-cap win survives on the actual `#3717` regime.

### 2026-03-19 05:23 UTC - Shared-2048 `forward_backward` rerun at `131072` shows another large partial fix, not parity
- Experiment ID: `OVLP-RES-023`
- Command:

```bash
KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3821.yaml \
  --repo-ref b63556b2993164133eb7aebd2d1269f578c4ed09 \
  --task-id ovlpres-3717jax-fb-t131072-20260319-0513 \
  --tokens 131072 \
  --shared-expert-dim 2048 \
  --kernels deepep_transport_capped_prewarmed \
  --topk-list 2,8 \
  --distributions random \
  --bench-pass forward_backward \
  --ep-list 8 \
  --warmup 1 \
  --iters 2 \
  --per-bench-timeout-seconds 3600 \
  --per-bench-kill-after-seconds 30 \
  --build-with-torch-extension \
  --load-as-python-module \
  --skip-smoke \
  --skip-cleanup \
  --w13-expert-padded \
  --w2-expert-padded \
  --deepep-combine-num-max-send-tokens 8 \
  --node-selector iris-i3821jax-scale-group=h100-8x
```

- Config:
  - pod: `iris-task-bc97f56ccd62`
  - node: `g11ed54`
  - code ref: `b63556b2993164133eb7aebd2d1269f578c4ed09`
  - reference JAX values from the reduced `#3717` rerun on the post-`w13` branch:
    - `131072, topk=2`: `5,816,954.42 tok/s`
    - `131072, topk=8`: `1,905,892.83 tok/s`
  - sealed Megatron anchors from `#3717`:
    - `131072, topk=2`: `12,706,733.38 tok/s`
    - `131072, topk=8`: `6,580,540.53 tok/s`
- Result:
  - `131072, topk=2`: `7,426,381.18 tok/s` (`17.650 ms`)
    - vs prior JAX rerun: `+27.7%`
    - vs sealed Megatron anchor: still about `1.71x` slower
  - `131072, topk=8`: `2,698,342.41 tok/s` (`48.575 ms`)
    - vs prior JAX rerun: `+41.6%`
    - vs sealed Megatron anchor: still about `2.44x` slower
  - the pod completed successfully and was deleted after capture.
- Interpretation:
  - the `w2` expert-padded exact-cap candidate carries over to the actual shared-2048 `forward_backward` regime from `#3717`.
  - the improvement is meaningful on both `topk=2` and `topk=8`, but it is still not enough to close the JAX-vs-Megatron gap at `131072`.
- Next action:
  - run the matching `262144` shared-2048 rerun on the same settings to see whether the large-token row moves by a similar fraction.

### 2026-03-19 05:35 UTC - Shared-2048 `forward_backward` rerun at `262144` confirms the same story on the largest row
- Experiment ID: `OVLP-RES-024`
- Command:

```bash
KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3821.yaml \
  --repo-ref b63556b2993164133eb7aebd2d1269f578c4ed09 \
  --task-id ovlpres-3717jax-fb-t262144-20260319-0524 \
  --tokens 262144 \
  --shared-expert-dim 2048 \
  --kernels deepep_transport_capped_prewarmed \
  --topk-list 2,8 \
  --distributions random \
  --bench-pass forward_backward \
  --ep-list 8 \
  --warmup 1 \
  --iters 2 \
  --per-bench-timeout-seconds 3600 \
  --per-bench-kill-after-seconds 30 \
  --build-with-torch-extension \
  --load-as-python-module \
  --skip-smoke \
  --skip-cleanup \
  --w13-expert-padded \
  --w2-expert-padded \
  --deepep-combine-num-max-send-tokens 8 \
  --node-selector iris-i3821jax-scale-group=h100-8x
```

- Config:
  - pod: `iris-task-3d86d92a9813`
  - node: `g11ed54`
  - code ref: `b63556b2993164133eb7aebd2d1269f578c4ed09`
  - reference JAX values from the reduced `#3717` rerun on the post-`w13` branch:
    - `262144, topk=2`: `5,902,299.01 tok/s`
    - `262144, topk=8`: `1,790,803.36 tok/s`
  - sealed Megatron anchors from `#3717`:
    - `262144, topk=2`: `15,830,875 tok/s`
    - `262144, topk=8`: `7,652,599 tok/s`
- Result:
  - `262144, topk=2`: `7,766,980.51 tok/s` (`33.751 ms`)
    - vs prior JAX rerun: `+31.6%`
    - vs sealed Megatron anchor: still about `2.04x` slower
  - `262144, topk=8`: `2,736,691.72 tok/s` (`95.789 ms`)
    - vs prior JAX rerun: `+52.8%`
    - vs sealed Megatron anchor: still about `2.80x` slower
  - the pod auto-cleaned before post-run inspection; the wrapper had already captured the result lines.
- Interpretation:
  - the `w2` expert-padded exact-cap candidate is another substantial improvement across the large-token shared-2048 regime.
  - even after combining the earlier `w13` fix with this `w2` fix, JAX DeepEP still does not keep up with Megatron DeepEP on the hardest high-token rows from `#3717`.
  - the improvement is larger on `topk=8` than on `topk=2`, but the absolute parity gap there remains especially large.
- Next action:
  - post a milestone update to `#3841`, then start the next candidate branch. The most likely next local-compute candidate is a `w2` layout experiment (`out_first`) or another narrowing step that targets the remaining non-parity gap without reopening the already-flat `w13` guardrail.

### 2026-03-19 05:51 UTC - `w2` `out_first` is a negative result on the exact-cap forward ladder
- Experiment ID: `OVLP-RES-025`
- Hypothesis:
  - if the remaining exact-cap local-compute gap after `w2` expert padding is partly caused by the stored `w2` weight layout, then flipping `w2` to `out_first` should improve `local_compute_only` and maybe `capped_prewarmed` while leaving the `w13_only` guardrail flat.
- Command:

```bash
git commit -m "bench: add w2 out-first benchmark flag"
git push origin research/moe-jax-deepep-residual-overlap

KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3821.yaml \
  --repo-ref 923184132e6c87462de6204cb3204491725d8b60 \
  --task-id ovlpres-w2of-ladder-t262144-r2-20260319-0544 \
  --tokens 262144 \
  --shared-expert-dim 0 \
  --kernels deepep_transport_w13_only_probe,deepep_transport_local_compute_only_probe,deepep_transport_capped_prewarmed \
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
  --skip-cleanup \
  --w13-expert-padded \
  --w2-expert-padded \
  --w2-out-first \
  --deepep-combine-num-max-send-tokens 8 \
  --node-selector iris-i3821jax-scale-group=h100-8x
```

- Config:
  - commit: `923184132e6c87462de6204cb3204491725d8b60`
  - files changed:
    - `lib/levanter/scripts/bench/bench_moe_hillclimb.py`
    - `.agents/scripts/deepep_jax_krt_bench.py`
  - local verification:
    - `python -m py_compile lib/levanter/scripts/bench/bench_moe_hillclimb.py .agents/scripts/deepep_jax_krt_bench.py`
    - help output for both scripts shows `--w2-out-first`
  - reference ladder from the `w2` expert-padded candidate:
    - `w13_only`: `0.002426 s`
    - `local_compute_only`: `0.005050 s`
    - `capped_prewarmed`: `0.007973 s`
- Result:
  - `w13_only`: `0.002423 s` / `108,199,823.29 tok/s`
  - `local_compute_only`: `0.005046 s` / `51,950,466.08 tok/s`
  - `capped_prewarmed`: `0.007978 s` / `32,859,716.19 tok/s`
  - the first launch used a mistyped full SHA and was deleted before it could consume the lane; the corrected launch ran on pod `iris-task-4ee10de390e1`
- Interpretation:
  - this is flat within noise on every rung that matters.
  - `w2 out_first` is therefore not the next path to closing the JAX-vs-Megatron gap, at least when stacked on top of `w13` and `w2` expert padding in the exact-cap forward ladder.
- Next action:
  - stop spending rows on `w2 out_first` and re-profile the post-`w2` exact-cap `forward_backward` path to identify the next residual.

### 2026-03-19 05:51 UTC - Post-`w2` exact-cap `forward_backward` profile is in flight on the shared-2048 row
- Experiment ID: `OVLP-RES-026`
- Command:

```bash
KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3821.yaml \
  --repo-ref 923184132e6c87462de6204cb3204491725d8b60 \
  --task-id ovlpres-exactcap-fb-profile-w2pad-t131072-20260319-0555 \
  --tokens 131072 \
  --shared-expert-dim 2048 \
  --kernels deepep_transport_capped_prewarmed \
  --topk-list 2 \
  --distributions random \
  --bench-pass forward_backward \
  --ep-list 8 \
  --warmup 2 \
  --iters 3 \
  --profile-root /tmp/ovlpres-exactcap-fb-131072-topk2-w2pad \
  --post-bench-sleep-seconds 1800 \
  --per-bench-timeout-seconds 1800 \
  --per-bench-kill-after-seconds 20 \
  --build-with-torch-extension \
  --load-as-python-module \
  --skip-smoke \
  --skip-cleanup \
  --w13-expert-padded \
  --w2-expert-padded \
  --deepep-combine-num-max-send-tokens 8 \
  --node-selector iris-i3821jax-scale-group=h100-8x
```

- Config:
  - pod: `iris-task-1c98330fed6f`
  - node selector: `iris-i3821jax-scale-group=h100-8x`
- Result:
  - timed result: `0.089530 s` / `1,464,006.24 tok/s`
  - profile artifacts copied locally:
    - `scratch/profiles/ovlpres-exactcap-fb-131072-topk2-w2pad`
    - `scratch/profiles/ovlpres-exactcap-fb-131072-topk2-w2pad-summary.json`
    - `scratch/profiles/ovlpres-exactcap-fb-131072-topk2-w2pad-report.md`
  - direct profile deltas vs the pre-`w2` exact-cap shared-2048 profile (`scratch/profiles/ovlpres-exactcap-fb-131072-topk2-report.md`):
    - compute share: `43.31% -> 42.58%`
    - communication share: `1.53% -> 1.34%`
    - host share: `55.03% -> 55.90%`
    - all-reduce exclusive: `18,681.861 -> 12,872.026`
    - pre-all-reduce idle gap: `565,816.590 -> 458,099.668`
  - top-op mix changed materially:
    - the old large `nvjet_tst_*` GEMM hotspots no longer dominate
    - the new leaders are `input_scatter_fusion_4` (`63,757.272`), `input_scatter_fusion` (`60,565.861`), DeepEP dispatch (`37,006.390`), DeepEP combine (`34,550.040`), and `cached_notify_combine` (`32,619.284`)
- Interpretation:
  - the `w2` expert-padded change did remove a real local-compute bottleneck, but it did not convert the residual into a simple next GEMM-layout problem.
  - the remaining shared-2048 `forward_backward` gap is now even more clearly dominated by host/glue overhead, pre-collective waiting, and dispatch/combine-side work.
  - this makes another narrow local layout tweak low priority relative to attacking XLA glue or transport/combine synchronization.
- Next action:
  - delete the sleeping profile pod, then inspect the exact-cap `forward_backward` code around dispatch/combine and the HLO-driving glue to pick the next candidate branch.

### 2026-03-19 06:15 UTC - Gather-based expert-padded pack is a strong positive result on the exact-cap forward ladder
- Experiment ID: `OVLP-RES-027`
- Hypothesis:
  - the remaining top exact-cap local-compute hotspots (`input_scatter_fusion*`) come from the scatter-based expert-padded pack path; replacing that with a gather-based pack that exploits the already-grouped expert layout should materially improve `w13_only`, `local_compute_only`, and `capped_prewarmed`.
- Code:
  - commit: `52fb00bf02ba7375467ed4e58194b3a076ea47c6`
  - commit message: `moe: gather-pack expert-padded ragged dots`
  - files changed:
    - `lib/levanter/src/levanter/grug/grug_moe.py`
    - `lib/levanter/scripts/bench/bench_moe_hillclimb.py`
  - local verification:
    - `python -m py_compile lib/levanter/src/levanter/grug/grug_moe.py lib/levanter/scripts/bench/bench_moe_hillclimb.py`
    - `uv run python` numerical equivalence check against the old scatter-based reference on multiple small cases; max abs diff `0.0`
- Command:

```bash
git commit -m "moe: gather-pack expert-padded ragged dots"
git push origin research/moe-jax-deepep-residual-overlap

KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3821.yaml \
  --repo-ref 52fb00bf02ba7375467ed4e58194b3a076ea47c6 \
  --task-id ovlpres-gatherpack-ladder-t262144-r2-20260319-0613 \
  --tokens 262144 \
  --shared-expert-dim 0 \
  --kernels deepep_transport_w13_only_probe,deepep_transport_local_compute_only_probe,deepep_transport_capped_prewarmed \
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
  --skip-cleanup \
  --w13-expert-padded \
  --w2-expert-padded \
  --deepep-combine-num-max-send-tokens 8 \
  --node-selector iris-i3821jax-scale-group=h100-8x
```

- Config:
  - pinned lane: `iris-i3821jax-scale-group=h100-8x`
  - pod: `iris-task-2a8726118710`
  - authoritative pre-change reference from the previous `w2` expert-padded baseline:
    - `w13_only`: `0.002426 s` / `108,047,212.04 tok/s`
    - `local_compute_only`: `0.005050 s` / `51,911,860.65 tok/s`
    - `capped_prewarmed`: `0.007973 s` / `32,880,566.00 tok/s`
- Result:
  - `w13_only`: `0.001080 s` / `242,704,374.67 tok/s`
    - vs prior: `-55.5%` time / `+124.6%` throughput
  - `local_compute_only`: `0.003675 s` / `71,327,106.63 tok/s`
    - vs prior: `-27.2%` time / `+37.4%` throughput
  - `capped_prewarmed`: `0.006578 s` / `39,852,597.50 tok/s`
    - vs prior: `-17.5%` time / `+21.2%` throughput
  - each kernel still ended with the known DeepEP/XLA teardown noise (`DeepEP timeout check failed`, `CUDA_ERROR_LAUNCH_FAILED`) but the wrapper completed successfully with `EXIT_CODE=0`
- Interpretation:
  - this confirms the expert-padded scatter pack itself was a real residual bottleneck after the earlier `w13` and `w2` layout fixes.
  - the change is not just a local-compute micro-win: it moves the decision rung `capped_prewarmed` substantially on the authoritative exact-cap forward cell.
  - the very large `w13_only` improvement means the old scatter-based pack was dominating more of the first ragged-dot path than the earlier profile decomposition made obvious.
- Next action:
  - rerun the reduced shared-2048 `forward_backward` JAX exact-cap rows (`131072` and `262144`, `topk=2,8`) on this commit to measure how much of the `#3717` JAX-vs-Megatron gap this closes.

### 2026-03-19 06:22 UTC - Shared-2048 `forward_backward` rerun shows only modest uplift at `131072`
- Experiment ID: `OVLP-RES-028`
- Hypothesis:
  - if the gather-based expert-padded pack fix closes a meaningful chunk of the remaining `#3717` gap, then the reduced shared-2048 `forward_backward` JAX rows should move materially, not just the exact-cap forward ladder.
- Commands:

```bash
KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3821.yaml \
  --repo-ref 52fb00bf02ba7375467ed4e58194b3a076ea47c6 \
  --task-id ovlpres-3717jax-gatherpack-fb-t131072-20260319-0616 \
  --tokens 131072 \
  --shared-expert-dim 2048 \
  --kernels deepep_transport_capped_prewarmed \
  --topk-list 2,8 \
  --distributions random \
  --bench-pass forward_backward \
  --ep-list 8 \
  --warmup 1 \
  --iters 2 \
  --per-bench-timeout-seconds 3600 \
  --per-bench-kill-after-seconds 30 \
  --build-with-torch-extension \
  --load-as-python-module \
  --skip-smoke \
  --skip-cleanup \
  --w13-expert-padded \
  --w2-expert-padded \
  --deepep-combine-num-max-send-tokens 8 \
  --node-selector iris-i3821jax-scale-group=h100-8x

KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3821.yaml \
  --repo-ref 52fb00bf02ba7375467ed4e58194b3a076ea47c6 \
  --task-id ovlpres-3717jax-gatherpack-fb-t262144-20260319-0621 \
  --tokens 262144 \
  --shared-expert-dim 2048 \
  --kernels deepep_transport_capped_prewarmed \
  --topk-list 2,8 \
  --distributions random \
  --bench-pass forward_backward \
  --ep-list 8 \
  --warmup 1 \
  --iters 2 \
  --per-bench-timeout-seconds 3600 \
  --per-bench-kill-after-seconds 30 \
  --build-with-torch-extension \
  --load-as-python-module \
  --skip-smoke \
  --skip-cleanup \
  --w13-expert-padded \
  --w2-expert-padded \
  --deepep-combine-num-max-send-tokens 8 \
  --node-selector iris-i3821jax-scale-group=h100-8x
```

- Config:
  - code ref: `52fb00bf02ba7375467ed4e58194b3a076ea47c6`
  - `131072` pod: `iris-task-d87568544c07`
  - `262144` pod in flight: `iris-task-dcc994e5d236`
  - prior shared-2048 reference from the post-`w2` branch:
    - `131072, topk=2`: `7,426,381.18 tok/s`
    - `131072, topk=8`: `2,698,342.41 tok/s`
    - `262144, topk=2`: `7,766,980.51 tok/s`
    - `262144, topk=8`: `2,736,691.72 tok/s`
- Result so far:
  - `131072, topk=2`: `7,561,909.56 tok/s` (`17.333 ms`)
    - vs prior JAX rerun: `+1.8%`
    - vs sealed Megatron anchor `12,712,443`: still about `1.68x` slower
  - `131072, topk=8`: `2,779,335.11 tok/s` (`47.159 ms`)
    - vs prior JAX rerun: `+3.0%`
    - vs sealed Megatron anchor `6,577,851`: still about `2.37x` slower
  - `262144` is still running as of this checkpoint.
- Interpretation:
  - the gather-pack fix is real and valuable on the exact-cap forward ladder, but its benefit only partly carries into the shared-2048 `forward_backward` regime.
  - this suggests the dominant residual for the `#3717` gap is now elsewhere again, likely in backward/shared-expert work, synchronization, or other transport/combine glue that the exact-cap forward ladder does not stress as directly.
- Next action:
  - wait for the `262144` companion row, then decide whether this branch is worth posting as a milestone or whether the next optimization thread should pivot back to the broader backward/shared-2048 residual.

### 2026-03-19 15:36 UTC - Complete the gather-pack shared-2048 rerun and confirm it is only a partial win
- Experiment ID: `OVLP-RES-029`
- Hypothesis:
  - if the gather-pack fix materially closes the broader `#3717` gap, the missing `262144` shared-2048 `forward_backward` row should show the same kind of step change as the exact-cap forward ladder rather than another marginal move.
- Command:

```bash
KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3821.yaml \
  --repo-ref 52fb00bf02ba7375467ed4e58194b3a076ea47c6 \
  --task-id ovlpres-3717jax-gatherpack-fb-t262144-20260319-1503 \
  --tokens 262144 \
  --shared-expert-dim 2048 \
  --kernels deepep_transport_capped_prewarmed \
  --topk-list 2,8 \
  --distributions random \
  --bench-pass forward_backward \
  --ep-list 8 \
  --warmup 1 \
  --iters 2 \
  --per-bench-timeout-seconds 3600 \
  --per-bench-kill-after-seconds 30 \
  --build-with-torch-extension \
  --load-as-python-module \
  --skip-smoke \
  --skip-cleanup \
  --w13-expert-padded \
  --w2-expert-padded \
  --deepep-combine-num-max-send-tokens 8 \
  --node-selector iris-i3821jax-scale-group=h100-8x
```

- Config:
  - code ref: `52fb00bf02ba7375467ed4e58194b3a076ea47c6`
  - pod: `iris-task-c685fabfad01`
  - prior shared-2048 reference from the post-`w2` branch:
    - `262144, topk=2`: `7,766,980.51 tok/s`
    - `262144, topk=8`: `2,736,691.72 tok/s`
- Result:
  - `262144, topk=2`: `7,905,937.49 tok/s` (`33.158 ms`)
    - vs prior JAX rerun: `+1.8%`
    - vs sealed Megatron anchor `15,830,593`: still about `2.00x` slower
  - `262144, topk=8`: `2,820,152.95 tok/s` (`92.954 ms`)
    - vs prior JAX rerun: `+3.0%`
    - vs sealed Megatron anchor `7,649,303`: still about `2.71x` slower
- Interpretation:
  - the gather-pack fix is now fully measured across the reduced shared-2048 rerun and the answer is consistent: it is a real exact-cap win, but only a modest `forward_backward` win in the regime that matters for the remaining JAX-vs-Megatron gap.
  - the next residual branch should focus on backward/shared-expert work rather than extending the forward gather-pack idea further.
- Next action:
  - implement one bounded backward-side candidate and validate it on the exact-cap forward guardrails before rerunning the shared-2048 row.

### 2026-03-19 15:36 UTC - Add an explicit backward for expert-padded ragged dot
- Experiment ID: `OVLP-RES-030`
- Hypothesis:
  - the gather-pack forward fix removed the pack bottleneck on the exact-cap forward ladder, but the remaining shared-2048 gap is now likely dominated by backward-side scatter/transposes and generic autodiff glue in the expert-padded batched-dot path.
  - replacing that generic backward with an explicit custom VJP should reduce the isolated exact-cap backward probes enough to justify another shared-2048 rerun.
- Commands:

```bash
python -m py_compile \
  lib/levanter/src/levanter/grug/grug_moe.py \
  lib/levanter/scripts/bench/bench_moe_hillclimb.py \
  lib/levanter/tests/test_grug_moe.py

uv run pytest lib/levanter/tests/test_grug_moe.py -q
uv run pytest lib/haliax/tests/test_ragged_dot_dispatch.py -q

git commit -m "moe: add explicit backward for expert-padded ragged dot"
git push origin research/moe-jax-deepep-residual-overlap

KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3821.yaml \
  --repo-ref 25deb3ac940fde89bc1732acf76cfbb4ab30f2e9 \
  --task-id ovlpres-bwdpad-ladder-t262144-20260319-1524 \
  --tokens 262144 \
  --shared-expert-dim 0 \
  --kernels deepep_transport_w13_only_probe,deepep_transport_local_compute_only_probe,deepep_transport_capped_prewarmed \
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
  --skip-cleanup \
  --w13-expert-padded \
  --w2-expert-padded \
  --deepep-combine-num-max-send-tokens 8 \
  --node-selector iris-i3821jax-scale-group=h100-8x
```

- Config:
  - commit: `25deb3ac940fde89bc1732acf76cfbb4ab30f2e9`
  - files changed:
    - `lib/levanter/src/levanter/grug/grug_moe.py`
    - `lib/levanter/scripts/bench/bench_moe_hillclimb.py`
    - `lib/levanter/tests/test_grug_moe.py`
  - prior forward reference on the gather-pack branch:
    - `w13_only`: `242,704,374.67 tok/s`
    - `local_compute_only`: `71,327,106.63 tok/s`
    - `capped_prewarmed`: `39,852,597.50 tok/s`
- Result:
  - local verification passed:
    - `py_compile`
    - `lib/levanter/tests/test_grug_moe.py`
    - `lib/haliax/tests/test_ragged_dot_dispatch.py`
  - forward validation ladder stayed clean:
    - `w13_only`: `242,678,805.79 tok/s`
    - `local_compute_only`: `71,669,032.68 tok/s`
    - `capped_prewarmed`: `40,048,270.15 tok/s`
- Interpretation:
  - the explicit backward candidate does not regress the forward guardrails and is safe to promote to backward-focused validation.
  - because the forward numbers are effectively flat to slightly better, any downstream improvement on the shared-2048 `forward_backward` row is likely attributable to the backward change rather than a new forward-side confounder.
- Next action:
  - run isolated exact-cap backward probes on the same commit to see whether the backward-side residual actually moved.

### 2026-03-19 15:36 UTC - Isolated exact-cap backward probes show a large move on `w13` and `w2`
- Experiment ID: `OVLP-RES-031`
- Hypothesis:
  - if the new explicit backward is addressing the right residual, the shared-0 exact-cap backward probes should move sharply on the `w13` and `w2` stages before we spend another shared-2048 rerun.
- Commands:

```bash
KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3821.yaml \
  --repo-ref 25deb3ac940fde89bc1732acf76cfbb4ab30f2e9 \
  --task-id ovlpres-bwdpad-probes-fb131072-shared0-20260319-1528 \
  --tokens 131072 \
  --shared-expert-dim 0 \
  --kernels deepep_transport_w13_only_bwd_probe,deepep_transport_w2_only_bwd_probe,deepep_transport_local_compute_bwd_probe \
  --topk-list 2 \
  --distributions random \
  --bench-pass forward_backward \
  --ep-list 8 \
  --warmup 1 \
  --iters 2 \
  --per-bench-timeout-seconds 3600 \
  --per-bench-kill-after-seconds 30 \
  --build-with-torch-extension \
  --load-as-python-module \
  --skip-smoke \
  --skip-cleanup \
  --w13-expert-padded \
  --w2-expert-padded \
  --deepep-combine-num-max-send-tokens 8 \
  --node-selector iris-i3821jax-scale-group=h100-8x

KUBECONFIG=/home/ubuntu/.kube/coreweave-iris kubectl delete pod -n iris-3821-jax iris-task-a81021d98d1a --wait=true

KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3821.yaml \
  --repo-ref 25deb3ac940fde89bc1732acf76cfbb4ab30f2e9 \
  --task-id ovlpres-bwdpad-3717jax-fb-t131072-20260319-1541 \
  --tokens 131072 \
  --shared-expert-dim 2048 \
  --kernels deepep_transport_capped_prewarmed \
  --topk-list 2,8 \
  --distributions random \
  --bench-pass forward_backward \
  --ep-list 8 \
  --warmup 1 \
  --iters 2 \
  --per-bench-timeout-seconds 3600 \
  --per-bench-kill-after-seconds 30 \
  --build-with-torch-extension \
  --load-as-python-module \
  --skip-smoke \
  --skip-cleanup \
  --w13-expert-padded \
  --w2-expert-padded \
  --deepep-combine-num-max-send-tokens 8 \
  --node-selector iris-i3821jax-scale-group=h100-8x
```

- Config:
  - code ref: `25deb3ac940fde89bc1732acf76cfbb4ab30f2e9`
  - backward probe pod: `iris-task-a81021d98d1a`
  - new shared-2048 pod: `iris-task-d382380b074a`
  - prior exact-cap backward probe reference:
    - `w13_only_bwd_probe`: `38,179,646.02 tok/s`
    - `w2_only_bwd_probe`: `40,321,856.23 tok/s`
    - `local_compute_bwd_probe`: `19,987,362.12 tok/s`
- Result so far:
  - `w13_only_bwd_probe`: `195,195,904.08 tok/s`
    - vs prior: about `5.1x` faster
  - `w2_only_bwd_probe`: `249,467,080.75 tok/s`
    - vs prior: about `6.2x` faster
  - both isolated stages still ended with the known DeepEP/XLA teardown noise (`CUDA_ERROR_LAUNCH_FAILED`) but returned `bench_status=0`
  - `local_compute_bwd_probe` was still compiling/running after more than two minutes on the pinned H100 lane, so I deleted the pod rather than blocking the shared-2048 promotion test behind a lower-signal diagnostic.
- Interpretation:
  - the isolated backward probes are strong enough to justify promotion: the new custom backward appears to remove a large exact-cap residual on both routed backward stages.
  - the unresolved question is not whether the micro-probes moved, but whether that move survives into the actual shared-2048 `forward_backward` row that defines the JAX-vs-Megatron branch decision.
- Next action:
  - wait for `ovlpres-bwdpad-3717jax-fb-t131072-20260319-1541`, then compare it against the gather-pack shared-2048 reference before deciding whether to post a milestone update.

### 2026-03-19 15:42 UTC - The custom backward carries into the first shared-2048 row
- Experiment ID: `OVLP-RES-032`
- Hypothesis:
  - if the explicit backward is fixing a branch-level residual rather than just probe noise, the reduced shared-2048 `forward_backward` row should improve materially on both `topk=2` and `topk=8`.
- Commands:

```bash
KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3821.yaml \
  --repo-ref 25deb3ac940fde89bc1732acf76cfbb4ab30f2e9 \
  --task-id ovlpres-bwdpad-3717jax-fb-t131072-20260319-1541 \
  --tokens 131072 \
  --shared-expert-dim 2048 \
  --kernels deepep_transport_capped_prewarmed \
  --topk-list 2,8 \
  --distributions random \
  --bench-pass forward_backward \
  --ep-list 8 \
  --warmup 1 \
  --iters 2 \
  --per-bench-timeout-seconds 3600 \
  --per-bench-kill-after-seconds 30 \
  --build-with-torch-extension \
  --load-as-python-module \
  --skip-smoke \
  --skip-cleanup \
  --w13-expert-padded \
  --w2-expert-padded \
  --deepep-combine-num-max-send-tokens 8 \
  --node-selector iris-i3821jax-scale-group=h100-8x

KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3821.yaml \
  --repo-ref 25deb3ac940fde89bc1732acf76cfbb4ab30f2e9 \
  --task-id ovlpres-bwdpad-3717jax-fb-t262144-20260319-1542 \
  --tokens 262144 \
  --shared-expert-dim 2048 \
  --kernels deepep_transport_capped_prewarmed \
  --topk-list 2,8 \
  --distributions random \
  --bench-pass forward_backward \
  --ep-list 8 \
  --warmup 1 \
  --iters 2 \
  --per-bench-timeout-seconds 3600 \
  --per-bench-kill-after-seconds 30 \
  --build-with-torch-extension \
  --load-as-python-module \
  --skip-smoke \
  --skip-cleanup \
  --w13-expert-padded \
  --w2-expert-padded \
  --deepep-combine-num-max-send-tokens 8 \
  --node-selector iris-i3821jax-scale-group=h100-8x
```

- Config:
  - code ref: `25deb3ac940fde89bc1732acf76cfbb4ab30f2e9`
  - completed `131072` pod: `iris-task-d382380b074a`
  - in-flight `262144` pod: `iris-task-089991961ff8`
  - prior gather-pack shared-2048 reference:
    - `131072, topk=2`: `7,561,909.56 tok/s`
    - `131072, topk=8`: `2,779,335.11 tok/s`
- Result so far:
  - `131072, topk=2`: `11,239,581.66 tok/s` (`11.662 ms`)
    - vs gather-pack branch: `+48.6%`
    - vs sealed Megatron anchor `12,712,443`: now only about `13.1%` slower
  - `131072, topk=8`: `5,313,729.98 tok/s` (`24.667 ms`)
    - vs gather-pack branch: `+91.2%`
    - vs sealed Megatron anchor `6,577,851`: now only about `23.8%` slower
  - `262144, topk=2`: `12,042,936.22 tok/s` (`21.767 ms`)
    - vs gather-pack branch: `+52.4%`
    - vs sealed Megatron anchor `15,830,593`: now about `1.31x` slower
  - `262144, topk=8`: `5,402,725.41 tok/s` (`48.521 ms`)
    - vs gather-pack branch: `+91.6%`
    - vs sealed Megatron anchor `7,649,303`: now about `1.42x` slower
- Interpretation:
  - this is the first shared-2048 evidence that the residual-overlap branch is actually closing the JAX-vs-Megatron gap instead of just moving exact-cap forward microbenchmarks.
  - the improvement now holds across all four rows in the reduced rerun, with the strongest relative gains on `topk=8`.
  - the gap is no longer “much much slower” in the same way it was earlier in the night; JAX still trails Megatron on the large rows, but it is now within roughly `13%` to `42%` on these sealed anchors rather than `2x` to `4x`.
- Next action:
  - run one matched Megatron DeepEP rerun on the same fresh lane to validate the comparison against a current same-cluster reference before posting a milestone update.

### 2026-03-19 16:02 UTC - Same-lane Megatron reruns are too noisy to replace the sealed anchors
- Experiment ID: `OVLP-RES-033`
- Hypothesis:
  - if the remaining JAX-vs-Megatron comparison is mostly a stale-reference problem, a same-lane Megatron rerun should reproduce the older Megatron column closely enough to replace the sealed anchors.
- Commands:

```bash
KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/megatron_deepep_qwen_runner.py \
  --config lib/iris/examples/coreweave-moe-jax-3821.yaml \
  --repo-ref 25deb3ac940fde89bc1732acf76cfbb4ab30f2e9 \
  --task-id ovlpres-megatron-deepep-reduced-20260319-1550 \
  --cases marin_3633_topk_2_mb4,marin_3633_topk_8_mb4,marin_3633_topk_2_mb8,marin_3633_topk_8_mb8 \
  --dispatchers deepep \
  --warmup-iters 1 \
  --measure-iters 2 \
  --dummy-gemm-size 8192 \
  --node-selector iris-i3821jax-scale-group=h100-8x

KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/megatron_deepep_qwen_runner.py \
  --config lib/iris/examples/coreweave-moe-jax-3821.yaml \
  --repo-ref 25deb3ac940fde89bc1732acf76cfbb4ab30f2e9 \
  --task-id ovlpres-megatron-topk2-stability-20260319-1600 \
  --cases marin_3633_topk_2_mb4,marin_3633_topk_2_mb8 \
  --dispatchers deepep \
  --warmup-iters 1 \
  --measure-iters 5 \
  --dummy-gemm-size 8192 \
  --node-selector iris-i3821jax-scale-group=h100-8x
```

- Config:
  - code ref: `25deb3ac940fde89bc1732acf76cfbb4ab30f2e9`
  - 4-case reduced rerun log: `scratch/3717-rerun/ovlpres-megatron-deepep-reduced-20260319-1550.log`
  - topk=2 stability rerun log: `scratch/3717-rerun/ovlpres-megatron-topk2-stability-20260319-1600.log`
  - sealed Megatron anchors still used for comparison:
    - `131072, topk=2`: `12,712,443 tok/s`
    - `131072, topk=8`: `6,577,851 tok/s`
    - `262144, topk=2`: `15,830,593 tok/s`
    - `262144, topk=8`: `7,649,303 tok/s`
- Result:
  - 4-case rerun (`measure-iters=2`):
    - `131072, topk=2` (`marin_3633_topk_2_mb4`): `956,158.77 tok/s`, `forward_std_ms=82.974`, `backward_std_ms=23.458`
    - `131072, topk=8` (`marin_3633_topk_8_mb4`): `4,380,088.15 tok/s`, `forward_std_ms=0.927`, `backward_std_ms=0.051`
    - `262144, topk=2` (`marin_3633_topk_2_mb8`): `9,336,756.43 tok/s`, `forward_std_ms=2.632`, `backward_std_ms=7.304`
    - `262144, topk=8` (`marin_3633_topk_8_mb8`): `6,906,325.29 tok/s`, `forward_std_ms=0.332`, `backward_std_ms=0.581`
  - topk=2 stability rerun (`measure-iters=5`):
    - `131072, topk=2`: `1,554,727.41 tok/s`, `forward_std_ms=67.135`, `backward_std_ms=23.392`
    - `262144, topk=2`: `8,937,870.59 tok/s`, `forward_std_ms=8.112`, `backward_std_ms=4.871`
- Interpretation:
  - the same-lane Megatron reruns are not stable enough to replace the earlier sealed anchors, especially on `topk=2` where the variance is obviously pathological.
  - the branch-level JAX conclusion does not depend on these noisy reruns: relative to the sealed Megatron anchors, the new explicit-backward branch has pulled JAX to within roughly `13%` to `42%` on the reduced high-token table.
  - operationally, this means the right next step is to trust the sealed Megatron anchors as the comparison baseline and profile the remaining JAX residual, not to keep spending H100 time on more noisy Megatron reruns.
- Next action:
  - snapshot this milestone on the branch and start a fresh JAX `forward_backward` profile on the strongest remaining residual row (`262144`, shared `2048`, `topk=8`, exact-cap path).

### 2026-03-19 18:01 UTC - Shared-MLP explicit backward clears forward guardrails; decisive rerun is boot-blocked
- Experiment ID: `OVLP-RES-034`
- Hypothesis:
  - the worst remaining shared-2048 residual may now sit in the replicated shared-MLP backward rather than the routed exact-cap path, and an explicit backward for the shared MLP could reduce the pre-`all-reduce` wait seen in the `262144/topk=8` profile.
- Commands:

```bash
KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3821.yaml \
  --repo-ref research/moe-jax-deepep-residual-overlap \
  --task-id ovlpres-sharedbwd-ladder-t262144-r2-20260319-1636 \
  --tokens 262144 \
  --shared-expert-dim 0 \
  --kernels deepep_transport_w13_only_probe,deepep_transport_local_compute_only_probe,deepep_transport_capped_prewarmed \
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
  --skip-cleanup \
  --w13-expert-padded \
  --w2-expert-padded \
  --deepep-combine-num-max-send-tokens 8 \
  --shared-mlp-explicit-bwd

KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3821.yaml \
  --repo-ref research/moe-jax-deepep-residual-overlap \
  --task-id ovlpres-sharedbwd-3717jax-fb-t262144-topk8-20260319-175102 \
  --tokens 262144 \
  --shared-expert-dim 2048 \
  --kernels deepep_transport_capped_prewarmed \
  --topk-list 8 \
  --distributions random \
  --bench-pass forward_backward \
  --ep-list 8 \
  --warmup 1 \
  --iters 2 \
  --per-bench-timeout-seconds 3600 \
  --per-bench-kill-after-seconds 30 \
  --build-with-torch-extension \
  --load-as-python-module \
  --skip-smoke \
  --skip-cleanup \
  --w13-expert-padded \
  --w2-expert-padded \
  --deepep-combine-num-max-send-tokens 8 \
  --shared-mlp-explicit-bwd \
  --node-selector iris-i3821jax-scale-group=h100-8x
```

- Config:
  - code ref: `6630030e44d9495062b440e4bf1228fe72e4d20d`
  - ladder log: `scratch/3717-rerun/ovlpres-sharedbwd-ladder-t262144-r2-20260319-1636.log`
  - decisive rerun log: `scratch/3717-rerun/ovlpres-sharedbwd-3717jax-fb-t262144-topk8-20260319-175102.log`
  - prior exact-cap forward reference on the explicit-bwd-expert-padded branch:
    - `w13_only`: `242,678,805.79 tok/s`
    - `local_compute_only`: `71,669,032.68 tok/s`
    - `capped_prewarmed`: `40,048,270.15 tok/s`
- Result so far:
  - the required forward guardrail ladder stayed effectively flat:
    - `w13_only`: `242,557,393.44 tok/s`
    - `local_compute_only`: `71,628,043.93 tok/s`
    - `capped_prewarmed`: `39,810,103.51 tok/s`
  - the new shared-2048 rerun is launched on pod `iris-task-2739f114f804` and is currently still pending because the fresh H100 nodepool is booting.
  - infra state on the pinned lane at checkpoint time:
    - `i3821jax-h100-8x`: `target=1`, `inprogress=1`, `current=0`
    - CoreWeave assigned node `g1464be` to the nodepool and reported that it may take up to `20 minutes` to boot.
- Interpretation:
  - the shared-MLP explicit backward is safe enough to keep exploring because it does not introduce a forward regression on the exact-cap guardrail ladder.
  - the branch-decision result is still pending; at this checkpoint the blocker is infra bring-up, not a benchmark or code failure.
- Next action:
  - wait for the fresh H100 node to register, finish the `262144/topk=8/shared2048/forward_backward` rerun, and decide whether the shared-MLP backward is a real branch-level win or whether the next pivot should move to collective scheduling/overlap.
