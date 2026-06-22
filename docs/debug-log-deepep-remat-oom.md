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

## Hypothesis 7: the apparent dispatch hang is actually first-step compilation

MAY328 was stopped after its third attempt ran for more than 20 minutes without
metrics, counter-timeout diagnostics, or worker-exit logs. I launched MAY329
from commit `381b88233`, which includes the worker-exit diagnostics and omits
the very noisy host-dispatch debug logging.

MAY329:

- W&B: `marin-community/marin_moe/MAY329-WORKEREXIT-B32-XONLY-JAXBWD-1209`
- Parent Iris job: `/dlwh/iris-run-job-20260622-120913`
- Child Iris job:
  `/dlwh/iris-run-job-20260622-120913/grug-train-MAY329-WORKEREXIT-B32-XONLY-JAXBWD-1209`
- All 16 local worker processes reached DeepEP internode runtime post-init.
- All 16 ranks logged `Starting Grug train_step dispatch for step 0`.
- After ~17 minutes with no metrics, process inspection showed all local worker
  main threads inside JAX compilation:
  `backend_compile_and_load` -> `_compile_and_write_cache` ->
  `compile_or_get_cached` -> `pxla.compile` -> `_pjit_call_impl_python`.

This means the "dispatch" log line is not proof that DeepEP transport is
executing. It is emitted immediately before the first compiled train-step call.
For MAY329, the current active state is first-step compilation, not a proven
DeepEP forward-dispatch hang.

## Results

MAY329 eventually got past the first-step compilation window and failed in the
DeepEP internode forward-dispatch counter wait:

- W&B state: failed, with only static FLOP summary fields and no train metrics.
- Task 0 ranks 0-7 timed out in
  `WaitForInternodeRecvCounts`.
- The timeout diagnostics reported `ready_expert_counters=0`,
  `num_recv_tokens=-1`, `num_rdma_recv_tokens=-1`, and all expert counters as
  `-1`.
- No matching filtered task-1 Python traceback, OOM, NCCL error, or counter
  timeout was visible before the run was stopped.
- The child and parent Iris jobs were stopped after W&B had already marked the
  run failed.

This changes the interpretation again: the long silent phase was compilation,
but after compilation the EP16 x-only/JAX-backward DeepEP path still has an
internode forward-dispatch synchronization problem. The untouched counters on
task 0 mean the peer-side notify/write path did not become visible to task 0.
The next run needs low-noise host-dispatch stage logging that distinguishes
`before_notify`, `after_notify`, `before_wait`, and `after_wait` by rank without
dumping buffers.

## Hypothesis 8: stage-only host-dispatch logging can expose the asymmetry

The existing `LEVANTER_DEEPEP_HOST_DISPATCH_DEBUG` flag enables both stage logs
and buffer summaries. That is too noisy for a 16-rank training launch and also
means `call_sequence` stays `-1` unless the heavyweight debug mode is enabled.

The fresh `codex/deepep-stage-debug` branch adds
`LEVANTER_DEEPEP_HOST_DISPATCH_STAGE_DEBUG`, which enables:

- `HOST_DISPATCH_STAGE` lines at notify/wait boundaries;
- per-runtime `call_sequence` IDs in those lines;
- matching `call_sequence` IDs in recv-counter timeout diagnostics.

It intentionally does not enable `HOST_DISPATCH_BUFFER` summaries. The next B32
EP16 run should use this stage-only flag with the 900s counter timeout to
determine whether task 1 reaches notify/wait and whether task 0 is waiting
before its peer has notified.

## Results

MAY330 used `LEVANTER_DEEPEP_HOST_DISPATCH_STAGE_DEBUG=1` from
`codex/deepep-stage-debug`:

- W&B:
  `marin-community/marin_moe/MAY330-STAGEDBG-B32-XONLY-JAXBWD-1238`
- Parent Iris job: `/dlwh/iris-run-job-20260622-123834`
- Child Iris job:
  `/dlwh/iris-run-job-20260622-123834/grug-train-MAY330-STAGEDBG-B32-XONLY-JAXBWD-1238`
- All ranks reached DeepEP runtime post-init and first `train_step` dispatch.
- Stage logs showed forward dispatch/combine execution with real `call_sequence`
  IDs. Call sequence 1 reached `after_wait_recv_counts` on both nodes with
  non-negative recv counts, so this attempt did not reproduce the untouched
  recv-counter timeout from MAY329.
- The run failed with a DeepEP generated-kernel assertion:
  `pcie.cu:219, condition: num_warps >= num_channels`.
- Stage logs showed `aux0=12`, where `aux0` is the resolved channel count. This
  comes from the internode dispatch default `num_sms=24`, i.e. 12 channels.
- The child and parent jobs were stopped while Iris was retrying.

This is progress: the stage-only diagnostics work, and the next blocker is an
invalid internode dispatch channel count rather than the previous silent
counter wait. The branch now caps the default internode dispatch configuration
to `num_sms=16` (8 channels) and fails fast if an override asks for more.

## Hypothesis 9: dispatch-local assignment packing is the silent bounce point

MAY331 reran the same B32 EP16 stage-debug shape after capping internode
`num_sms` to 16:

- W&B:
  `marin-community/marin_moe/MAY331-SMS16-B32-XONLY-JAXBWD-1247`
- Parent Iris job: `/dlwh/iris-run-job-20260622-124757`
- Child Iris job:
  `/dlwh/iris-run-job-20260622-124757/grug-train-MAY331-SMS16-B32-XONLY-JAXBWD-1247`
- Stage logs showed `aux0=8`, confirming the channel cap took effect.
- Forward dispatch reached `after_wait_recv_counts` and `before_dispatch_launch`
  on task-0 ranks with non-negative recv counts.
- The run still failed before first train metrics; W&B marked the run failed and
  Iris retried after task 0 reported a generic `Error`.
- The filtered logs did not show a Python traceback, DeepEP counter timeout,
  CUDA OOM, NCCL error, or local-worker supervisor line for the first attempt.
- Later `pcie.cu:219` assertion text appeared only as compile warning text from
  the retry, not as a proven runtime trap in the first attempt.

The suspicious gap is immediately after `before_dispatch_launch`.
`DispatchInternode` launched `LaunchPackLocalAssignmentsFromCounts` after the
DeepEP dispatch but did not check `cudaGetLastError()` before returning success.
The next branch state adds a CUDA error check, debug stream sync, and
`internode_jax_after_assignment_pack` stage line after that launch. If the pack
kernel is the silent failure, the next run should report it directly.

## Future work

- [x] Add a C++/FFI primitive for x-only internode combine-with-local-collapse
      backward that returns `grad_out_dispatch` and `grad_assignment_weights`
      without exposing `grad_recv_out` to XLA.
- [x] Wire the Python custom VJP to call that primitive instead of
      `_dispatch_internode_cached_impl` followed by
      `_collapse_local_assignments_internode_bwd_impl`.
- [x] Add narrower DeepEP recv-counter diagnostics for the EP16 timeout.
- [x] Re-run B32 EP16 DeepEP internode with worker-exit diagnostics from
      `4258fe300` or later.
- [x] Let MAY329 compile long enough to distinguish compile latency from
      runtime DeepEP transport failure.
- [x] Add low-noise DeepEP host-dispatch stage diagnostics for notify/wait
      asymmetry without enabling full buffer summaries.
- [x] Re-run B32 EP16 DeepEP internode with
      `LEVANTER_DEEPEP_HOST_DISPATCH_STAGE_DEBUG=1`.
- [x] Re-run B32 EP16 DeepEP internode with internode `num_sms=16`.
- [ ] Re-run B32 EP16 with the dispatch local-assignment pack CUDA check.
- [ ] If B32 passes, retry B64 and profile remat overhead versus ring/all-to-all.
- [ ] If B32 reaches `cudaMallocAsync(x-only fused local-collapse bwd recv_out)`
      OOM, replace the staging temp with a true direct packed-output backward
      transport kernel.
