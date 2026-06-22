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

## Hypothesis 4: hide the backward recv buffer inside one FFI

Commit `d1fc8e172` adds
`levanter_deepep_combine_internode_x_only_with_local_collapse_bwd_fused`. The
new primitive keeps the cached backward DeepEP dispatch and local-collapse
backward in one C++ FFI call, with the recv-capacity `grad_recv_out` allocated
as a scoped CUDA temp instead of an XLA-visible array.

This is a staging primitive, not the final no-temp solution: it tests whether
XLA liveness was the dominant problem before writing a direct packed-output
transport kernel.

## Results

MAY327 validated that the fused backward target compiles and loads on
CoreWeave:

- W&B: `marin-community/marin_moe/MAY327-FUSEDBWD-B32-XONLY-JAXBWD-1117`
- Parent Iris job: `/dlwh/iris-run-job-20260622-111711`
- Child Iris job:
  `/dlwh/iris-run-job-20260622-111711/grug-train-MAY327-FUSEDBWD-B32-XONLY-JAXBWD-1117`
- Runtime reached 16-rank DeepEP internode init and step-0 train dispatch.
- Failure: DeepEP recv-counter timeout, not NCCL OOM:
  `INTERNAL: DeepEP internode JAX dispatch timed out waiting for recv counters`.
- Node 0 ranks reached `Finished Grug train_step dispatch for step 0; waiting
  for train/loss`; node 1 ranks started step-0 dispatch but did not reach the
  same completion line before node 0 timed out.
- The run was stopped after the fatal training-loop errors to avoid wasting
  GPUs.

This did not prove the fused backward memory behavior because the run appears
to fail before the fused backward path is exercised.

## Current hypothesis

The x-only/local-collapse backward change is compile-valid, but the B32 EP16
validation is currently blocked by DeepEP internode rendezvous/counter behavior
on the forward dispatch. The next debugging step is to make the counter wait
diagnostic enough to distinguish:

- one node stuck before `notify_dispatch`;
- one node stuck inside `notify_dispatch`;
- the recv counters being written somewhere other than the mapped host-visible
  locations;
- a timeout that is simply too short for this launch mode.

The forward path may also still need a true no-`collapsed_recv` implementation,
but the current failure must be cleared before memory results are meaningful.

## Hypothesis 5: the recv-counter timeout was too short

Commit `d19d24de7` adds `LEVANTER_DEEPEP_COUNTER_TIMEOUT_SECONDS` and logs
recv-counter timeout state (`ready_expert_counters`, `first_pending_expert`,
and `timeout_seconds`). I launched MAY328 with a 900s timeout and host-dispatch
stage diagnostics.

## Results

MAY328 showed that the original 180s timeout was too short for at least some
attempts, but it did not yet produce a clean training result:

- W&B: `marin-community/marin_moe/MAY328-COUNTERDBG-B32-XONLY-JAXBWD-1133`
- Parent Iris job: `/dlwh/iris-run-job-20260622-113322`
- Child Iris job:
  `/dlwh/iris-run-job-20260622-113322/grug-train-MAY328-COUNTERDBG-B32-XONLY-JAXBWD-1133`
- The run reached W&B and 16-rank DeepEP runtime initialization.
- With the longer timeout, host-dispatch diagnostics showed ranks getting past
  `internode_jax_after_wait_recv_counts` and into dispatch/combine calls.
- The job bounced twice under Iris atomic re-scheduling before producing
  metrics. The first bounce did not surface a Python traceback, OOM, NCCL
  error, or counter-timeout line in the filtered logs.
- The third attempt was still running when this note was written; W&B only had
  static FLOP summary fields and no `train/loss` or throughput rows yet.

The useful signal is that the counter timeout is no longer the only known
blocker. The next missing diagnostic is which local process-per-GPU worker dies
first when an Iris task bounces.

## Hypothesis 6: expose process-per-GPU worker exits

Commit `4258fe300` changes the DeepEP process-per-GPU supervisor to:

- run child workers unbuffered;
- poll local workers and log each `local_rank`, global process index, and return
  code as soon as it exits;
- terminate sibling workers immediately after the first nonzero exit, with a
  kill fallback if a sibling ignores `terminate()`.

This is an observability change only. It should make the next MAY328-style run
report the first failed local rank instead of a generic Iris task `Error`.

## Future work

- [x] Add a C++/FFI primitive for x-only internode combine-with-local-collapse
      backward that returns `grad_out_dispatch` and `grad_assignment_weights`
      without exposing `grad_recv_out` to XLA.
- [x] Wire the Python custom VJP to call that primitive instead of
      `_dispatch_internode_cached_impl` followed by
      `_collapse_local_assignments_internode_bwd_impl`.
- [x] Add narrower DeepEP recv-counter diagnostics for the EP16 timeout.
- [ ] Re-run B32 EP16 DeepEP internode with `offload_moe_hidden` from
      `4258fe300` or later if MAY328 bounces again without metrics.
- [ ] If B32 passes, retry B64 and profile remat overhead versus ring/all-to-all.
- [ ] If B32 reaches `cudaMallocAsync(x-only fused local-collapse bwd recv_out)`
      OOM, replace the staging temp with a true direct packed-output backward
      transport kernel.
