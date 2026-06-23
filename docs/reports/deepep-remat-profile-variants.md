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
archive_bootstrap_commit=99ba0edcf [grug] Bootstrap DeepEP from source archive
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
| MAY357 | `offload_moe_hidden` | disabled | 17.30 | 0.4097s | Works, but costs about 3.3 MFU versus `none`; narrow hidden offload is still too expensive. |
| MAY358 | `offload_moe_output` | disabled | 20.60 | 0.3440s | Matches `none` within run noise while preserving output offload semantics. |
| MAY359 | `none` | disabled | 24.60 | 0.5762s | B16 control; increasing batch from 8 to 16 improves throughput substantially. |
| MAY360 | `offload_moe_output` | disabled | 24.82 | 0.5711s | B16 output-offload comparison; still matches `none` within run noise. |
| MAY361 | `save_moe` | disabled | 24.94 | 0.5682s | B16 save control; also matches `none`, slightly faster in this run. |

Interpretation:

- Broad/expert host offload is not useful for this EP8 shape; it replaces remat
  pressure with large H2D/D2H copies. Narrow offload is different:
  `offload_moe_hidden` is still expensive, but `offload_moe_output` is
  effectively throughput-neutral at B8.
- `save_moe` is at worst a small tax for this guarded DeepEP path. At B8 it
  measured slightly slower than `none`; at B16 it measured slightly faster, so
  the practical conclusion is that its effect is in run-noise range for EP8.
- Current best EP8 DeepEP remat setting for pure throughput is `--remat none`,
  with `--remat offload_moe_output` as the first memory-headroom knob to try
  because it matched `none` in both B8 and B16 throughput runs.

Why the `save_moe` delta is small:

- For effectful DeepEP backends the model skips the outer block checkpoint
  entirely:

```text
_should_checkpoint_block(remat_mode, uses_effectful_moe=True) == False
```

  That avoids rematting a block containing DeepEP FFI effects, so EP8
  `save_moe` is not paying for a full block remat versus `none`.
- The remaining difference is the inner DeepEP expert MLP checkpoint in
  `ep_deepep.py`. `save_moe` keeps the tagged ragged-dot residuals
  (`dispatch_input`, `expert_hidden`, `dispatch_output`, `moe_output`) while
  allowing the rest of the small expert up/down wrapper to remat. That is much
  narrower than "save every block activation", which is why the measured loss
  is only about 1.8% versus `none`.

Narrow offload follow-up:

```text
MAY357:
  parent=/dlwh/iris-run-job-20260622-215727
  wandb=https://wandb.ai/marin-community/marin_moe/runs/MAY357-DEEPEP-EP8-OFFLOAD-HIDDEN-L26-B8-N1-20260622-2157
  remat=offload_moe_hidden
  steady_mfu=17.30
  steady_step_duration=0.4097s

MAY358:
  parent=/dlwh/iris-run-job-20260622-220907
  wandb=https://wandb.ai/marin-community/marin_moe/runs/MAY358-DEEPEP-EP8-OFFLOAD-OUTPUT-L26-B8-N1-20260622-2209
  remat=offload_moe_output
  steady_mfu=20.60
  steady_step_duration=0.3440s

MAY359:
  parent=/dlwh/iris-run-job-20260622-222336
  wandb=https://wandb.ai/marin-community/marin_moe/runs/MAY359-DEEPEP-EP8-NOREMAT-L26-B16-N1-20260622-2223
  remat=none
  global_batch=16
  steady_mfu=24.60
  steady_step_duration=0.5762s

MAY360:
  parent=/dlwh/iris-run-job-20260622-224147
  wandb=https://wandb.ai/marin-community/marin_moe/runs/MAY360-DEEPEP-EP8-OFFLOAD-OUTPUT-L26-B16-N1-20260622-2241
  remat=offload_moe_output
  global_batch=16
  steady_mfu=24.82
  steady_step_duration=0.5711s

MAY361:
  parent=/dlwh/iris-run-job-20260622-230807
  wandb=https://wandb.ai/marin-community/marin_moe/runs/MAY361-DEEPEP-EP8-SAVEMOE-L26-B16-N1-20260622-2308
  remat=save_moe
  global_batch=16
  steady_mfu=24.94
  steady_step_duration=0.5682s
```

Interpretation: the output tensor is a narrow enough boundary that host offload
does not show up materially in the EP8 throughput runs. The hidden tensor is not
narrow enough; it behaves more like the earlier broad offload result and gives
back most of the `save_moe` versus `none` win. B16 is the current best
single-node EP8 DeepEP throughput point in this report. The B16 result also
shows that the `save_moe` versus `none` difference is not a stable throughput
tax for the guarded DeepEP path; it is small enough to be dominated by run
noise.

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
child=/dlwh/iris-run-job-20260622-195438/grug-train-MAY351-DEEPEP-EP16-SAVEMOE-JAXASSIGN-FIXEDFFICOLLAPSE-L26-B64-N2-20260622-1954
wandb=marin-community/marin_moe/MAY351-DEEPEP-EP16-SAVEMOE-JAXASSIGN-FIXEDFFICOLLAPSE-L26-B64-N2-20260622-1954
```

Result:

- May351 admitted on two H100x8 tasks, built DeepEP, initialized all 16
  process-per-GPU ranks, and confirmed:

```text
Grug compact mesh shape: {'replica_dcn': 1, 'data': 1, 'expert': 16, 'model': 1}; batch_shards=16
```

- The run compiled `jit_train_step` and dispatched step 0, but failed before
  any W&B metric rows while waiting for `train/loss`:

```text
jax.errors.JaxRuntimeError: INTERNAL: NCCL operation ncclGroupEnd() failed:
unhandled cuda error (run with NCCL_DEBUG=INFO for details). Last NCCL
warning(error) log entry (may be unrelated) 'Cuda failure 2 'out of memory''.
```

- The child began retrying and was stopped after the first reproduced run
  failure signal to avoid retry churn.

Interpretation: switching `assignment_gradient` from `fused` to `jax` avoided
the named May349 `cudaMallocAsync(fused bwd recv_x)` allocation, but EP16 B64
still does not fit this path. The next useful control is the same
`save_moe`/FFI-collapse/JAX-assignment path at global batch 32.

May352 ran that B32 control:

```text
script=scratch/launch_may352_deepep_ep16_savemoe_l26_b32_jaxassign_n2_throughput.sh
parent=/dlwh/iris-run-job-20260622-201346
child=/dlwh/iris-run-job-20260622-201346/grug-train-MAY352-DEEPEP-EP16-SAVEMOE-JAXASSIGN-FIXEDFFICOLLAPSE-L26-B32-N2-20260622-2013
wandb=marin-community/marin_moe/MAY352-DEEPEP-EP16-SAVEMOE-JAXASSIGN-FIXEDFFICOLLAPSE-L26-B32-N2-20260622-2013
```

Result:

- May352 admitted on two H100x8 tasks, built DeepEP, initialized all 16 ranks,
  confirmed the EP16 mesh, compiled, and dispatched step 0.
- It failed before any W&B metric rows with the same NCCL/CUDA OOM class as
  May351:

```text
jax.errors.JaxRuntimeError: INTERNAL: NCCL operation ncclGroupEnd() failed:
unhandled cuda error (run with NCCL_DEBUG=INFO for details). Last NCCL
warning(error) log entry (may be unrelated) 'Cuda failure 2 'out of memory''.
```

- The retrying child was stopped to avoid repeating the same failure.

Interpretation: EP16 `save_moe` with FFI local collapse and JAX assignment
gradient still does not fit at global batch 32, so the failure is not merely
B64 activation pressure. The next EP16 control, if needed, should be a B16
minimum-batch sanity probe or a different assignment/collapse implementation;
it should not be treated as an expected-throughput run.

May353 attempted the next different assignment-gradient implementation:

```text
script=scratch/launch_may353_deepep_ep16_savemoe_l26_b32_ffiassign_n2_throughput.sh
parent=/dlwh/iris-run-job-20260622-203041
child=/dlwh/iris-run-job-20260622-203041/grug-train-MAY353-DEEPEP-EP16-SAVEMOE-FFIASSIGN-FIXEDFFICOLLAPSE-L26-B32-N2-20260622-2030
shape=2 H100x8 nodes, EP16, global batch 32, 2 sequences/device
moe=deepep_internode
remat=save_moe
profiler=disabled
collapse=ffi
assignment_gradient=ffi
```

Result:

- May353 did not reach W&B or model execution. It failed during DeepEP source
  bootstrap because a worker could not clone upstream DeepEP from GitHub:

```text
fatal: unable to access 'https://github.com/deepseek-ai/DeepEP.git/':
Failed to connect to github.com port 443 after 133097 ms: Couldn't connect to server
```

- Iris retried twice; the child was stopped to avoid burning the two-node slot
  on a network/bootstrap failure.

Interpretation: May353 is not evidence for or against FFI assignment gradients.
Before retrying this control, remove the live GitHub clone dependency by using
an existing `DEEPEP_SRC_ROOT`, a pre-staged DeepEP source tarball/prefix, or an
image with the validated DeepEP checkout baked in.

May354 retried that FFI-assignment control with a pre-staged source archive:

```text
script=scratch/launch_may354_deepep_ep16_savemoe_l26_b32_ffiassign_archive_n2_throughput.sh
source_archive=s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/deepep-source/DeepEP-7febc6e25660af0f54d95dd781ecdcd62265ecca.tgz
parent=/dlwh/iris-run-job-20260622-204910
child=/dlwh/iris-run-job-20260622-204910/grug-train-MAY354-DEEPEP-EP16-SAVEMOE-FFIASSIGN-ARCHIVE-FIXEDFFICOLLAPSE-L26-B32-N2-20260622-2049
wandb=marin-community/marin_moe/MAY354-DEEPEP-EP16-SAVEMOE-FFIASSIGN-ARCHIVE-FIXEDFFICOLLAPSE-L26-B32-N2-20260622-2049
shape=2 H100x8 nodes, EP16, global batch 32, 2 sequences/device
moe=deepep_internode
remat=save_moe
profiler=disabled
collapse=ffi
assignment_gradient=ffi
```

Result:

- The archive bootstrap worked on both tasks:

```text
Using DeepEP source archive s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/deepep-source/DeepEP-7febc6e25660af0f54d95dd781ecdcd62265ecca.tgz for /tmp/marin-deepep/DeepEP
```

- May354 reached W&B, initialized all 16 process-per-GPU ranks, confirmed the
  EP16 mesh, and dispatched step 0.
- It produced no W&B metric rows. The first attempt died with JAX coordination
  connection-refused noise after one worker group exited. The automatic retry
  reached the model path again, then failed while waiting for `train/loss` with:

```text
DEEPEP_INTERNODE_COUNTER_TIMEOUT {"rank":4,"call_sequence":-1,"num_recv_tokens":-1,"num_rdma_recv_tokens":-1,"num_local_experts":16,"expert_counters":[-1,...,-1]}
jax.errors.JaxRuntimeError: INTERNAL: DeepEP internode JAX dispatch timed out waiting for recv counters
```

- The retrying child and parent were stopped after the reproduced model-path
  failure to avoid retry churn.

Interpretation: removing the live GitHub dependency was necessary and is now
working, but FFI assignment-gradient does not rescue the EP16 B32
`save_moe`/FFI-collapse path. The failure class is back to the internode DeepEP
recv-counter timeout seen in May348, not a source-bootstrap failure and not the
JAX-assignment NCCL/CUDA OOM from May351/May352.

May355 tested the other branch of the isolation: keep `save_moe` and fused
assignment-gradient, but use the JAX scatter collapse path instead of FFI
collapse, with the same source archive bootstrap:

```text
script=scratch/launch_may355_deepep_ep16_savemoe_l26_b64_scatter_archive_n2_throughput.sh
parent=/dlwh/iris-run-job-20260622-212253
child=/dlwh/iris-run-job-20260622-212253/grug-train-MAY355-DEEPEP-EP16-SAVEMOE-FUSEDASSIGN-SCATTERCOLLAPSE-ARCHIVE-L26-B64-N2-20260622-2122
wandb=marin-community/marin_moe/MAY355-DEEPEP-EP16-SAVEMOE-FUSEDASSIGN-SCATTERCOLLAPSE-ARCHIVE-L26-B64-N2-20260622-2122
shape=2 H100x8 nodes, EP16, global batch 64, 4 sequences/device
moe=deepep_internode
remat=save_moe
profiler=disabled
collapse=scatter
assignment_gradient=fused
```

Result:

- May355 used the source archive, reached W&B, initialized all 16 ranks, and
  dispatched step 0.
- It failed before any metric rows with the same fused assignment-gradient
  allocation class seen in May349:

```text
jax.errors.JaxRuntimeError: INTERNAL: cudaMallocAsync(fused bwd recv_x): out of memory
```

- The retrying child and parent were stopped to avoid repeating the same
  failure.

Interpretation: scatter collapse avoids the May354 FFI recv-counter timeout
surface, but at B64 the fused assignment-gradient backward still does not fit.
The next useful control is the same `save_moe`/scatter/fused path at B32; if
that fits, it can give an EP16 remat-throughput point, and if it still OOMs the
fused assignment-gradient path is not viable for this comparison without memory
work.

May356 ran that B32 control:

```text
script=scratch/launch_may356_deepep_ep16_savemoe_l26_b32_scatter_archive_n2_throughput.sh
parent=/dlwh/iris-run-job-20260622-214151
child=/dlwh/iris-run-job-20260622-214151/grug-train-MAY356-DEEPEP-EP16-SAVEMOE-FUSEDASSIGN-SCATTERCOLLAPSE-ARCHIVE-L26-B32-N2-20260622-2141
wandb=marin-community/marin_moe/MAY356-DEEPEP-EP16-SAVEMOE-FUSEDASSIGN-SCATTERCOLLAPSE-ARCHIVE-L26-B32-N2-20260622-2141
shape=2 H100x8 nodes, EP16, global batch 32, 2 sequences/device
moe=deepep_internode
remat=save_moe
profiler=disabled
collapse=scatter
assignment_gradient=fused
```

Result:

- May356 used the source archive, reached W&B, initialized all 16 DeepEP ranks,
  confirmed the EP16 mesh, and dispatched step 0.
- It produced no W&B metric rows. After dispatch, `train/loss` failed while
  blocking on an NCCL group:

```text
jax.errors.JaxRuntimeError: INTERNAL: NCCL operation ncclGroupEnd() failed: unhandled cuda error
Last NCCL warning(error) log entry (may be unrelated) 'Cuda failure 2 'out of memory''.
```

- The process-per-GPU workers then exited, Iris began retrying, and the retrying
  child and parent were stopped.

Interpretation: lowering the scatter/fused EP16 `save_moe` run from B64 to B32
does not produce a usable throughput point. It changes the visible error from a
direct fused `recv_x` allocation failure to an NCCL group CUDA OOM during
`train/loss` blocking, but the fused assignment-gradient path still does not fit
cleanly enough for the remat comparison.

May362 ran the minimum practical batch control for the same scatter/fused path:

```text
script=scratch/launch_may362_deepep_ep16_savemoe_l26_b16_scatter_archive_n2_throughput.sh
parent=/dlwh/iris-run-job-20260622-232840
child=/dlwh/iris-run-job-20260622-232840/grug-train-MAY362-DEEPEP-EP16-SAVEMOE-FUSEDASSIGN-SCATTERCOLLAPSE-ARCHIVE-L26-B16-N2-20260622-2328
wandb=marin-community/marin_moe/MAY362-DEEPEP-EP16-SAVEMOE-FUSEDASSIGN-SCATTERCOLLAPSE-ARCHIVE-L26-B16-N2-20260622-2328
shape=2 H100x8 nodes, EP16, global batch 16, 1 sequence/device
moe=deepep_internode
remat=save_moe
profiler=disabled
collapse=scatter
assignment_gradient=fused
```

Result:

- May362 used the source archive, reached W&B, initialized all 16 DeepEP ranks,
  confirmed the EP16 mesh, and dispatched step 0.
- It produced no W&B metric rows. The W&B run ended `failed` with no
  `_timestamp`, `global_step`, `train/loss`, or throughput metrics.
- After dispatch, `train/loss` failed with the same internode recv-counter
  timeout class seen in May348/May354:

```text
jax.errors.JaxRuntimeError: INTERNAL: DeepEP internode JAX dispatch timed out waiting for recv counters
DEEPEP_INTERNODE_COUNTER_TIMEOUT {"rank":5,"call_sequence":-1,"num_recv_tokens":-1,"num_rdma_recv_tokens":-1,"num_local_experts":16,"expert_counters":[-1,...,-1]}
```

- Iris began retrying the same child after one failed attempt. The retrying
  child and parent were stopped to avoid burning the two-node slot on the same
  failing configuration.

Interpretation: reducing the scatter/fused EP16 `save_moe` control to B16
removes the visible CUDA OOM/NCCL OOM class from B32/B64, but it still does not
produce a usable throughput point. At minimum batch the blocker becomes the
DeepEP internode recv-counter synchronization failure, not remat overhead. EP16
therefore remains blocked for this remat comparison; the usable throughput
evidence in this report is the EP8 path, where B16 with `none`,
`save_moe`, and `offload_moe_output` all cluster around 24.6-24.9 MFU.
