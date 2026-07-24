# Debugging log for per-long-layer KV conditional execution

Test whether heterogeneous local/global KV-head counts can run at d6144 on 64
GB200s without the nested-scan CUBIN loader failure.

## Initial status

The prototype stores local and global layers in separate stacks because their
K/V projections have different shapes. It executes an outer scan over groups
and an inner scan over local layers. The model is correct on CPU and 4/8 GPUs,
but the first d6144 training step on 64 GPUs fails while loading the compiled
`jit_train_step` CUBIN. The uniform single-scan model succeeds at that scale.

## Hypothesis 1

A single flat layer scan with a `lax.cond` around the complete local or global
block will compile each FA4 call in its statically shaped branch while avoiding
the nested scan. Both branches return the same hidden and router-stat shapes.
FA4 metadata remains precomputed and explicitly batch-sharded before either
branch, so no conditional metadata output feeds the CUTLASS callback.

## Changes to make

- Replace the nested heterogeneous scan with a flat scan over layer indices.
- Dynamically select the next block from the appropriate local/global stack
  inside the corresponding `lax.cond` branch.
- Keep the local and global FA4 calls, masks, and KV shapes branch-local.
- Compare the conditional scan against a small unrolled reference on CPU.

## Results

The first CPU parity test found that the prototype attached local FA4 bounds to
the full-causal `long_mask`. FA4 used the explicit bounds correctly, but the
reference backend ignored `fa4_bounds` and therefore evaluated local layers as
full causal. The local branch now attaches the bounds to `short_mask`, preserving
sliding-window semantics for both backends.

The flat conditional scan matches a static Python-unrolled reference for hidden
states and every per-layer router metric at 1e-5 tolerance. It also executes on a
four-device explicit CPU mesh with the output batch-sharded across the `data`
and `expert` axes.

The d6144, 48-layer, local-16/global-8 KV run compiled on 64 GB200s and completed
all 30 training steps. The previous in-memory CUBIN load failure and the
device-0 metadata/rematerialization failure did not recur. W&B steps 10–28 had a
median 227,755 tokens/s and 20.578% MFU; the final logged loss was 6.0746.

The Iris job is technically failed because task 1 exited 137 while writing the
final step-30 checkpoint to local TensorStore. The other ranks then aborted
after the coordination service closed. This occurred after the training
objective completed and is separate from the attention execution path.

## Future work

- [ ] Inspect compiled sharding for device-0 maximal placements if the scale run
      still wedges.
- [ ] Disable the final local checkpoint for future throughput-only probes, or
      separately diagnose the task-1 checkpoint exit 137.

## Hypothesis 2

The remaining throughput gap comes from placing the entire local/global block,
including MoE, behind `lax.cond`. A single homogeneous block stack can instead
store K/V at the larger local width and slice those weights before the global
projection. Only attention then branches between static 16-KV and 8-KV FA4
calls; Q and the rest of the block remain shared.

## Changes to make

- Replace the two local/global block stacks with one uniform stack.
- Store K/V weights at `max(local_kv_heads, global_kv_heads)`.
- Slice K/V to the selected static width inside the attention branch.
- Keep FA4 bounds precomputed and batch-sharded outside the scan.
- Verify that padded K/V columns cannot affect global attention output.

## Results

The homogeneous-stack implementation matches a static layer unroll for hidden
states and router metrics. Perturbing every padded K/V column leaves global
attention unchanged while changing local attention, confirming that the global
branch projects only the sliced width. The model also executes on a four-device
explicit CPU mesh with the expected batch sharding.

The first 64-GB200 homogeneous-stack attempt failed before step 0 with a CUDA
OOM, not the prior CUBIN/device-placement failure. XLA estimated 158.82 GiB
before rematerialization and 154.69 GiB afterward against a 133.93 GiB target.
The conditional still enclosed K/V projection, XSA, gating, and output
projection, so it duplicated too much live state across its branches.

## Hypothesis 3

Project padded 16-head K/V once outside the conditional, and keep only the
static K/V slice, FA4 call, and V-head alignment inside each branch. Move XSA,
gating, and output projection back to the shared path. The branch results are
explicitly resharded to one common Q-head layout before leaving the conditional.
This should retain separate static 16-head and 8-head FA4 calls without the
branch-sized memory pressure from the first homogeneous attempt.

The narrower conditional reduced XLA's post-rematerialization estimate from
154.69 GiB to 151.28 GiB, but its first execution requested one contiguous
105.53 GiB allocation per GPU and failed. The working two-stack flat
conditional had a much higher 198.21 GiB post-rematerialization estimate, so
the aggregate estimate does not explain the failure. Padding added only
100,663,296 parameters (0.028%).

## Isolation control

Run the same homogeneous 48-layer stack with a uniform 16 KV heads and no
heterogeneous conditional for one training step. If it succeeds, the failure is
specific to the attention conditional inside the layer rematerialization
boundary. If it fails with the same allocation, inspect the single-stack
optimizer/collective layout instead.

The uniform control completed its update on all 16 ranks with the exact same
359,942,670,336 parameters as the failing heterogeneous run. Its XLA estimate
was similar (149.28 GiB after rematerialization), but it did not request the
105.53 GiB contiguous buffer. This isolates the allocation to the conditional
inside the rematerialized attention rather than the padded stack or optimizer.

## Hypothesis 4

Move the heterogeneous conditional outside the layer rematerialization
boundary. Both branches use the same homogeneous layer weights, but pass a
static global/local choice so K/V slicing and FA4 shapes remain static. The
conditional now returns the normal block hidden state and router metrics, not
backend head tensors with branch-dependent layout inference.

This variant also failed before step 0 in the MoE all-to-all. The uniform
control therefore rules out the homogeneous stack, while both placements of a
shape-changing conditional fail when differentiated through the scanned
parameters.

## Hypothesis 5

Eliminate the shape-changing conditional. For a logical 8-head global layer,
slice the first eight projected K/V heads and repeat each one twice into the
static 16-head kernel layout. With 48 Q heads this is exactly the same grouping:
each logical KV head serves six consecutive Q heads. Local layers use the 16
projected heads directly, and a same-shaped scalar selection chooses the
activation. A focused reference test confirms that the repeated 16-head global
path matches a true uniform 8-head attention module.

This variant completed all 30 steps on 64 GB200s, and both the training child
and wrapper exited successfully. Its compile estimate (149.53 GiB after
rematerialization) nearly matched the successful uniform-16 control
(149.28 GiB), and the large conditional allocation did not recur. W&B steps
10–28 had a median 238,547 tokens/s and 21.553% MFU; final loss was 6.0386.
Compared with the working two-stack conditional baseline, throughput improved
by 4.7% and MFU by 0.975 percentage points.

The final implementation preserves 8-head GQA semantics on global layers, but
it projects and runs FA4 at the physical 16-head width. It does not recover the
parameter, projection, or kernel savings of storing global layers at width 8.
