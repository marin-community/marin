# Debugging log for DeepEP MoE remat OOM

Goal: reduce Grug MoE remat overhead at EP8/EP16 using DeepEP internode transport
or the fastest available backend, potentially with JAX host offload.

## Initial status

The working DeepEP internode path reaches step 0 on a 2-node EP16 Grug MoE run,
but B64 and B32 fail before first metrics with an NCCL `ncclGroupEnd` CUDA OOM
while waiting for `train/loss`.

Baseline failing shape:

- 2 H100 nodes, EP16, model axis 1, batch 32 or 64.
- `moe_implementation=deepep_internode`.
- `remat=offload_moe_hidden`.
- `LEVANTER_DEEPEP_INTERNODE_COLLAPSE_MODE=ffi`.
- `LEVANTER_DEEPEP_INTERNODE_COMBINE_X_ONLY=true`.
- `LEVANTER_DEEPEP_INTERNODE_ASSIGNMENT_GRADIENT_MODE=jax`.

## Hypothesis 1: receive capacity is too conservative

The internode path sized `max_recv_tokens` as the worst-case
`local_tokens * topk * num_rdma_ranks`. I added
`LEVANTER_DEEPEP_INTERNODE_RECV_CAPACITY_MODE=local_assignment` as a diagnostic
knob to use the local assignment capacity instead.

## Results

MAY323 used the local-assignment capacity mode and failed the same way as the
worst-case-capacity run:

- W&B: `marin-community/marin_moe/MAY323-HIDDENOFFLOAD-B64-XONLY-JAXBWD-LOCALCAP-1017`
- Failure: NCCL `ncclGroupEnd` CUDA OOM after step 0 dispatch.
- No history rows.

Capacity is not the only issue.

## Hypothesis 2: the outer block remat policy was accidentally disabled

`Transformer.__call__` built a policy for effectful MoE but skipped
`eqx.filter_checkpoint` for all `uses_effectful_moe` blocks. I enabled the
policy for effectful blocks and launched MAY325.

## Results

MAY325 failed immediately with:

```text
NotImplementedError: Effects not supported in partial-eval of `checkpoint`/`remat`: [FfiEffect()]
```

That explains the existing bypass. JAX cannot checkpoint/remat through these
effectful DeepEP FFI calls today. The patch was reverted in commit
`341ff14bf`.

## Hypothesis 3: valid inner expert offload is sufficient

`offload_moe_expert` offloads both expert hidden and dispatch output inside the
expert MLP checkpoint without wrapping the effectful transport boundary. I
launched MAY326 with the same B32 x-only/JAX-backward DeepEP shape.

## Results

MAY326 reproduced the same failure as MAY324:

- W&B: `marin-community/marin_moe/MAY326-OFFLOADEXPERT-B32-XONLY-JAXBWD-1057`
- Failure point: `Finished Grug train_step dispatch for step 0; waiting for train/loss`
- Error: NCCL `ncclGroupEnd` CUDA OOM.
- No first metrics or profile.

Inner expert offload is not enough.

## Current hypothesis

The high-memory boundary is the DeepEP combine/collapse transport boundary, not
ordinary expert MLP remat.

In `deepep_combine_internode_x_only_with_local_collapse`, the forward C++ FFI
still allocates a full recv-capacity `collapsed_recv` buffer before calling
DeepEP combine. Its Python custom VJP also reconstructs a full `grad_recv_out`
with `_dispatch_internode_cached_impl`, then runs collapse backward to produce
packed expert-output gradients. That defeats the intended x-only/local-collapse
memory savings.

## Changes to make

The next useful patch is a lower-level fused backward/transport primitive for
the x-only local-collapse path. It should transport `grad_combined_x` directly
to packed expert-output gradients and assignment-weight gradients, avoiding the
materialized recv-capacity `grad_recv_out` intermediate.

The forward path may also need a true no-`collapsed_recv` implementation, but
the repeated step-0 backward OOM makes the backward primitive the priority.

## Future work

- [ ] Add a C++/FFI primitive for x-only internode combine-with-local-collapse
      backward that returns `grad_out_dispatch` and `grad_assignment_weights`
      directly.
- [ ] Wire the Python custom VJP to call that primitive instead of
      `_dispatch_internode_cached_impl` followed by
      `_collapse_local_assignments_internode_bwd_impl`.
- [ ] Re-run B32 EP16 DeepEP internode with `offload_moe_hidden` first.
- [ ] If B32 passes, retry B64 and profile remat overhead versus ring/all-to-all.
