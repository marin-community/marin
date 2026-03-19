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
