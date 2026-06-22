# DeepEP Remat Profile Variants

## Scope

This note records the current DeepEP remat comparison for the d2560/L26 Grug
MoE CoreWeave experiments. The goal is to find the fastest practical MoE block
boundary for EP8 or EP16 while avoiding full block remat around effectful
DeepEP FFI calls.

Branch checkpoint:

```text
branch=codex/deepep-remat-profile-variants
guard_commit=31599501c [grug] Guard DeepEP remat launcher path
```

The guard commit skips outer block-level checkpointing when the MoE
implementation uses effectful DeepEP FFI calls. Without the guard, JAX fails
before training with:

```text
NotImplementedError: Effects not supported in partial-eval of checkpoint/remat: [FfiEffect()]
```

Inner DeepEP remat modes still apply inside `ep_deepep.py`; this comparison is
about those inner modes.

## EP8 Intranode Result

All EP8 runs below used one H100x8 node, global batch 8, sequence length 4096,
sliding window 2048, L26, `attention=gpu_fa4_cute`, `ce=pallas_gpu`,
`moe=deepep`, `optimizer=sgd`, bf16 params/compute/output, synthetic data, and
no checkpoints.

| Run | Remat mode | Profiler | Steady MFU | Steady step duration | Notes |
| --- | --- | --- | ---: | ---: | --- |
| MAY340 | `offload_moe_expert` | readable profile | 17.46 | 0.407s | Profile dominated by host copies: H2D+D2H about 2.52s in profiled window. |
| MAY342 | `save_moe` | readable profile | 16.76 | 0.436s | Host copies gone; profiler overhead/readability flags depress throughput. |
| MAY343 | `save_moe` | disabled | 20.20 | 0.3509s | Throughput comparison without command-buffer readability penalty. |
| MAY344 | `none` | disabled | 20.56 | 0.3447s | Best measured EP8 variant so far. |

Interpretation:

- Host offload is not useful for this EP8 shape; it replaces remat pressure with
  large H2D/D2H copies.
- `save_moe` is slightly slower than `none` when measured without profiler
  flags.
- Current best EP8 DeepEP remat setting is `--remat none`.

## EP16 Internode Direction

The working EP16 path is `moe=deepep_internode` in process-per-GPU topology:
two H100x8 tasks, `DEEPEP_RANKS_PER_NODE=8`, and
`MAY_DEEPEP_PROCESSES_PER_TASK=8`. Earlier notes in
`docs/debug-log-deepep-internode-jax.md` show:

- May292 proved a full L26 two-node EP16 run with fused dispatch-backward:
  `13.96` steady MFU at global batch 64.
- May293/May294 fused-combine removed a visible local collapse scatter but
  regressed wall time.
- May295 explicit local-collapse FFI followed by normal DeepEP combine recovered
  the fused-combine regression: about `13.86` late-step MFU at global batch 64.

Next comparison:

```text
script=scratch/launch_may345_deepep_ep16_noremat_l26_b128_n2_throughput.sh
shape=2 H100x8 nodes, EP16, global batch 128, 8 sequences/device
moe=deepep_internode
remat=none
profiler=disabled
collapse=ffi
assignment_gradient=fused
```

This run tests whether the EP8 conclusion (`remat=none` wins) also holds for
the current best EP16 internode DeepEP boundary.

Update:

- May345 reached train-step dispatch but failed before metrics with
  `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 102.90GiB`.
- The failed May345 parent/child were stopped to avoid retry churn.
- The next lower-memory comparison is the same EP16 no-remat config at global
  batch 64:

```text
script=scratch/launch_may346_deepep_ep16_noremat_l26_b64_n2_throughput.sh
shape=2 H100x8 nodes, EP16, global batch 64, 4 sequences/device
```

May346 reached train step 0 but failed before metrics with:

```text
jax.errors.JaxRuntimeError: INVALID_ARGUMENT: DeepEP local assignment collapse destination map shape is invalid
```

This is not the May345 B128 allocation failure. It isolates a separate issue in
the explicit local-collapse FFI when combined with the no-remat B64 path.

Root cause: this branch was missing the internode dispatch output projection
fix. The FFI dispatch result tuple stores:

```text
20 = local_group_cursors
21 = recv_assignment_indices
22 = assignment_destinations
```

but `_dispatch_internode_impl` returned `DeepEPInternodeDispatch(*results[:21])`,
which exposed `local_group_cursors` as `assignment_destinations`. The FFI
collapse correctly rejected that length-`local_experts` vector because the
destination map must have length `max_recv_tokens * topk`.

May347 was a temporary fallback isolation that kept EP16 B64 no-remat and fused
assignment-gradient, but switched only the local collapse mode back to the JAX
scatter path:

```text
script=scratch/launch_may347_deepep_ep16_noremat_l26_b64_scatter_n2_throughput.sh
shape=2 H100x8 nodes, EP16, global batch 64, 4 sequences/device
moe=deepep_internode
remat=none
profiler=disabled
collapse=scatter
assignment_gradient=fused
```

May347 was submitted as:

```text
parent=/dlwh/iris-run-job-20260622-182310
child=/dlwh/iris-run-job-20260622-182310/grug-train-MAY347-DEEPEP-EP16-NOREMAT-FUSEDASSIGN-SCATTERCOLLAPSE-L26-B64-N2-20260622-1823
wandb=marin-community/marin_moe/MAY347-DEEPEP-EP16-NOREMAT-FUSEDASSIGN-SCATTERCOLLAPSE-L26-B64-N2-20260622-1823
```

At launch time the child pods were `SchedulingGated` by Kueue admission/topology
because the H100 nodepool was fully occupied by one admitted 32-pod workload.
The run has not yet tested the model/runtime path.

After fixing the projection and adding test coverage for the exposed
`assignment_destinations` shape, the next preferred run is the original FFI
collapse comparison:

```text
script=scratch/launch_may348_deepep_ep16_noremat_l26_b64_fixedffi_n2_throughput.sh
shape=2 H100x8 nodes, EP16, global batch 64, 4 sequences/device
moe=deepep_internode
remat=none
profiler=disabled
collapse=ffi
assignment_gradient=fused
```

May348 was submitted as:

```text
parent=/dlwh/iris-run-job-20260622-184105
child=/dlwh/iris-run-job-20260622-184105/grug-train-MAY348-DEEPEP-EP16-NOREMAT-FUSEDASSIGN-FIXEDFFICOLLAPSE-L26-B64-N2-20260622-1841
wandb=marin-community/marin_moe/MAY348-DEEPEP-EP16-NOREMAT-FUSEDASSIGN-FIXEDFFICOLLAPSE-L26-B64-N2-20260622-1841
```

Result:

- May348 got past the May346 projection bug. The run built DeepEP, initialized
  all 16 process-per-GPU ranks, and confirmed:

```text
Grug compact mesh shape: {'replica_dcn': 1, 'data': 1, 'expert': 16, 'model': 1}; batch_shards=16
```

- It compiled and dispatched train step 0, then failed while waiting for
  `train/loss` with:

```text
jax.errors.JaxRuntimeError: INTERNAL: DeepEP internode JAX dispatch timed out waiting for recv counters
```

- The FFI printed timeout diagnostics for ranks on the second node, for example:

```text
DEEPEP_INTERNODE_COUNTER_TIMEOUT {"rank":11,"call_sequence":-1,"num_recv_tokens":-1,"num_rdma_recv_tokens":-1,"num_local_experts":16,"expert_counters":[-1,...,-1]}
```

- Iris retried and reproduced the same failure before any W&B metric rows. The
  run was stopped to avoid retry churn.

Interpretation: the exposed `assignment_destinations` shape bug is fixed, but
the EP16 no-remat FFI-collapse throughput run is still blocked by an internode
DeepEP dispatch/counter timeout before metrics. This is not yet evidence about
the remat overhead of EP16 `none` versus `save_moe`.

Follow-up control:

```text
script=scratch/launch_may349_deepep_ep16_savemoe_l26_b64_fixedffi_n2_throughput.sh
shape=2 H100x8 nodes, EP16, global batch 64, 4 sequences/device
moe=deepep_internode
remat=save_moe
profiler=disabled
collapse=ffi
assignment_gradient=fused
```

May349 keeps the May348 code path and switches only the inner MoE remat mode
back to `save_moe`. It was submitted as:

```text
parent=/dlwh/iris-run-job-20260622-192045
child=/dlwh/iris-run-job-20260622-192045/grug-train-MAY349-DEEPEP-EP16-SAVEMOE-FUSEDASSIGN-FIXEDFFICOLLAPSE-L26-B64-N2-20260622-1920
wandb=marin-community/marin_moe/MAY349-DEEPEP-EP16-SAVEMOE-FUSEDASSIGN-FIXEDFFICOLLAPSE-L26-B64-N2-20260622-1920
```

At submission time May349 was `SchedulingGated` by Kueue behind the admitted
32-task banked-MuonH workload `/dlwh/iris-run-job-20260622-175816`. The Kueue
condition said the two-pod set could not fit because topology `"infiniband"`
excluded all 32 nodes for CPU availability.

May349 later admitted on nodes `g73b7ae` and `g7406a6`. It built DeepEP,
initialized all 16 process-per-GPU ranks, confirmed the EP16 mesh, and reached
train step 0 dispatch. It failed before any W&B metric rows with:

```text
jax.errors.JaxRuntimeError: INTERNAL: cudaMallocAsync(fused bwd recv_x): out of memory
```

The run retried and reproduced the same failure mode, then was stopped to avoid
retry churn. This points at the fused dispatch-backward assignment-gradient path
rather than the inner `save_moe` remat policy itself.

If May349 starts and reproduces the May348 recv-counter timeout, the next two
controls are queued as scripts but should not be submitted while May349 is still
waiting for the same two-node topology slot:

```text
script=scratch/launch_may350_deepep_ep16_savemoe_l26_b64_scatter_n2_throughput.sh
shape=2 H100x8 nodes, EP16, global batch 64, 4 sequences/device
moe=deepep_internode
remat=save_moe
profiler=disabled
collapse=scatter
assignment_gradient=fused
purpose=separate FFI local-collapse from the recv-counter timeout
```

```text
script=scratch/launch_may351_deepep_ep16_savemoe_l26_b64_jaxassign_n2_throughput.sh
shape=2 H100x8 nodes, EP16, global batch 64, 4 sequences/device
moe=deepep_internode
remat=save_moe
profiler=disabled
collapse=ffi
assignment_gradient=jax
purpose=separate fused dispatch-backward assignment-gradient from the recv-counter timeout
```

Because May349 failed in `fused bwd recv_x` allocation rather than with the
recv-counter timeout, the preferred next control is May351: keep FFI local
collapse, but switch assignment gradients back to the JAX path.

May351 was submitted as:

```text
parent=/dlwh/iris-run-job-20260622-195438
wandb=marin-community/marin_moe/MAY351-DEEPEP-EP16-SAVEMOE-JAXASSIGN-FIXEDFFICOLLAPSE-L26-B64-N2-20260622-1954
```

At submission time the parent was pending scheduler feedback and had not yet
created a child job or W&B run.
