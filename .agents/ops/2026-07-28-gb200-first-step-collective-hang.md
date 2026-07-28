# GB200 training gangs wedged in first-step distributed execution

## Summary

On 2026-07-28, both 32-GPU NEST-BURN-001 r5 arms stopped making progress
inside their first compiled training step. Compilation completed normally and
all eight tasks in each arm entered the same PJRT dispatch, but neither arm
published update 0. The E256 arm's last log was at 00:12:50 UTC after an
extended input-prefetch warning on task 1. The fixed25 arm exhibited the same
steady state.

The original jobs and bounded diagnostic reproductions were stopped after live
evidence was collected. No task, node, or Iris cluster was restarted.

## Impact

Sixty-four GB200 GPUs are allocated without making training progress. The
failure precedes the first optimizer update, so it contributes no evidence to
the nested-MoE comparison. Both architecture arms fail in the same phase,
which points to a shared distributed runtime or launch issue rather than the
fixed-nesting treatment.

## Evidence

- Every task's Python main thread is parked in
  `pxla.py:413 __call__ -> _pjit_call_impl_python -> train.py:780`.
- Every sampled data-loader thread is blocked in `queue.put`; the prefetch
  queues are full rather than starved.
- All GPUs report 100% utilization at only 190--235 W of a 1,200 W cap. They
  remain in P0 at 1,950 MHz and hold approximately 157.7 GiB each.
- Three-second InfiniBand counter deltas contain only tiny, identical
  keep-alive traffic: 216 transmitted data units, 4,872 received data units,
  three transmitted packets, and 51 received packets per active interface.
  There is no bulk collective traffic.
- NVLink fabric status is `Success` on the sampled node. NVML reports no
  requested recovery action, remapped-row errors, or route recovery.
- Pods and nodes remain live with no relevant Kubernetes event or restart.

NCCL 2.28.9 RAS initially returned all 32 ranks in each arm as
`RUNNING` with no error:

- one 32-rank communicator spanning eight nodes;
- four eight-rank communicators with one rank per node.

Later global `STATUS` and `VERBOSE STATUS` collection attempts accepted a
local IPv6 connection on `::1:28028` but did not return a result, including
with a 35-second client timeout. RAS therefore proves that the host-side
control thread was initially alive and that no rank had left a communicator;
it does not rule out a GPU-side kernel hang. Per-rank JSON collective counters
could not be recovered from the wedged state.

## Runtime inconsistency

Both task environments contain JAX 0.11.0 and both CUDA 12 and CUDA 13 NVIDIA
wheel families, including `nvidia-nccl-cu12==2.28.9` and
`nvidia-nccl-cu13==2.28.9`. The active E256 communicator reports NCCL compiled
with CUDA 13.0 and CUDA runtime 13.0. The active fixed25 communicator reports
NCCL compiled with CUDA 12.9 and CUDA runtime 12.9. Their Python package
metadata is otherwise identical.

The two NCCL wheels install the same `nvidia/nccl` import and library path.
The selected library therefore varies between matched arms. This is an
independent reproducibility defect and a plausible contributor to the
first-step failure, although the CUDA-13 E256 arm also hangs and the current
evidence does not establish it as the sole cause.

## Next diagnostic run

A fresh short reproduction should pin one CUDA/NCCL family and set diagnostics
before communicator initialization:

- `NCCL_DEBUG=INFO`;
- `NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH,TUNING,RAS,COLL`;
- a unique `NCCL_DEBUG_FILE` per host and process, copied or uploaded before
  task teardown;
- explicit RAS enablement and periodic JSON `STATUS` snapshots;
- the existing Python thread, GPU power/utilization, and InfiniBand delta
  snapshots when the training-step heartbeat stops advancing.

The reproduction should first run only through compilation and several
optimizer updates. Once the runtime family and collective sequence are
verified, the matched full burn can use lower-volume NCCL logging.

## Diagnostic reproduction

Iris GPU setup now restores both CUDA-13 cuDNN and NCCL precedence after the
mixed CUDA wheel set is installed. Focused behavior tests execute the real
setup script against both shared-library paths.

An initial eight-step diagnostic used the shortened training length as the
Datakit simulated-epoch planning horizon. It failed before JAX dispatch because
one finite mixture component became empty. The corrected r7 smoke retains the
full 16,840-step data horizon while requesting eight optimizer updates.

During r7 compilation, the Python main thread contains active FA4 forward and
backward lowering frames. GPUs report zero utilization. NCCL RAS JSON returns
in 1 ms with no timeout, no missing rank, and identical `AllReduce=1` counts
across four eight-rank communicators. NCCL reports CUDA runtime 13.0. These
signals provide a positive control for compilation and distinguish it from the
r5 state: r5 showed no lowering frames, 100% GPU utilization, and no bulk
network progress after dispatch.

At 01:41:42 UTC, r7 entered its first compiled executable. Every sampled GPU
immediately settled at 100% utilization and approximately 202--236 W. RAS
reported the same stationary state on every rank: a healthy 32-rank
communicator with `AllGather=147` and `AllReduce=6`, plus the four healthy
eight-rank communicators. Four samples over 52 seconds were bit-for-bit
unchanged. All sampled hosts had the same XLA execution-worker and RDMA event
reader thread populations. No optimizer update or loss was emitted.

NCCL INFO logs show that the first executable submitted all 147 all-gathers
and six all-reduces at 01:41:42. Small operations selected the NVLS SIMPLE
path, while large all-gathers selected RING with LL128 or SIMPLE across the
P2P/MNNVL topology. This narrows the failure from generic compilation or
rank-entry skew to a GPU-side collective or transport interaction. r7 was
stopped after the state remained stationary.

The failure is not confined to the rack implicated by the resolved
`GpuRackTraysBelowMinimum` alert. The r5 E256 arm ran on rack 393; fixed25
attempts ran on racks 392 and 394; and deterministic-CUDA-13 r7 reproduced on
rack 392. All sampled nodes remained Ready. The rack-137 alert is therefore
unrelated to this incident.

Two matched two-step r8 isolates disable one NCCL path at a time:

- `NCCL_NVLS_ENABLE=0` on rack 393;
- `NCCL_MNNVL_ENABLE=0` on rack 392.

Both preserve the full Datakit planning horizon and deterministic CUDA-13
runtime. A successful first optimizer update in one arm will identify the
transport setting for a confirmation smoke and the matched scientific
relaunch.

Both one-factor isolates reproduced the failure. The NVLS-off arm entered its
first executable at 02:04:05 UTC and the MNNVL-off arm at 02:04:53. Each
settled at the same `AllGather=147`, `AllReduce=6` RAS counts on every rank,
with no skew or error. Both showed the shallow PJRT stack, 100% GPU
utilization, and low power, and neither emitted update 0. Disabling NVLS alone
changed small collectives from NVLS to RING; disabling MNNVL alone reduced the
selected RING channel count and changed small collectives to TREE. Neither
change restored progress.

The r8 isolates were stopped after the stationary state was captured. A
two-step r9 isolate now disables both NVLS and MNNVL, forcing NCCL away from
both GB200-specific paths while preserving all other runtime and model inputs.

r9 also stopped at exactly `AllGather=147`, `AllReduce=6` on every rank and
did not emit update 0. Therefore neither NVLS, MNNVL, nor their combination is
the sole cause. Two follow-up smokes target the remaining high-probability
dimensions:

- r10 disables NVLS and MNNVL and forces RING with SIMPLE protocol;
- r11 retains the default transport selection but enables
  `NCCL_LAUNCH_ORDER_IMPLICIT=1` and `NCCL_LAUNCH_RACE_FATAL=1`.

The second arm is motivated by the executable submitting 153 collectives
across multiple XLA execution streams in one burst. NCCL 2.28.9 contains both
controls, as verified against the loaded CUDA-13 library.

r10 also stopped at `AllGather=147`, `AllReduce=6` with no update. Forcing
RING/SIMPLE in addition to disabling NVLS and MNNVL therefore does not restore
progress. The selected collective algorithm, protocol, NVLS, and MNNVL paths
are ruled out as individual causes. r11 remains the active launch-order
diagnostic.

r11 reproduced the same state with implicit launch ordering enabled. The
launch-race fatal control remained silent. This rules out NCCL's optional
multi-stream launch-order mechanism as a sufficient fix.

The burn-in launcher had explicitly set `replica_axis_size=1`, producing a
single 32-GPU FSDP group across all eight nodes. That explains why the first
executable submits 147 parameter all-gathers over the global communicator.
Grug's standard small-model mesh instead sets the replica axis to the process
count: eight data replicas, each with four-GPU node-local FSDP. This preserves
the same global batch, parameters, optimizer, and architecture while replacing
cross-node parameter all-gathers with node-local all-gathers and cross-replica
gradient reductions. r12 tests this standard mesh.

A one-node, four-GPU FSDP positive control reproduced the failure with
`AllGather=147`, `AllReduce=3`. This proves that multi-node transport is not
required and localizes the common trigger to the FSDP parameter-all-gather
sequence on this GB200/JAX/NCCL runtime. r12 was stopped before completing its
redundant compile.

r14 uses a `(replica_dcn=32, data=1)` mesh. The model and optimizer are fully
replicated, while the global batch remains sharded one sequence per GPU and
gradients are reduced across the replica axis. The approximately 1.8B
parameter model fits within each 180 GB GPU, so this removes the implicated
parameter all-gathers without changing the scientific comparison.

The attempted replicated variants did not remove all parameter sharding from
the actual Grug parameter specs. A one-node r15 reproduction stopped at
`AllGather=66`, `ReduceScatter=1`, and `AllReduce=5` with the same shallow PJRT
stack and low-power GPU busy-poll state. r14 and r15 were stopped after this
made them redundant.

The earlier fixed-subset experiment provides a known-good topology: expert
axis 16 and data axis 4 on 64 GB200 GPUs completed 8,192 updates and 4.295B
tokens. The current global batch of 32 limits each arm to 32 devices, so the
recovery smoke uses `(replica=1, data=2, expert=16, model=1)` on eight
four-GPU nodes. This preserves the compute-optimal model, global batch, data,
optimizer, and fixed-subset treatment while returning expert storage and
dispatch to the proven EP=16 layout.

The runtime defect is tracked in
[GitHub issue #7694](https://github.com/marin-community/marin/issues/7694).
Two matched two-update EP=16 smokes were submitted at batch priority:

- `/power/nest-burn-001-e256-c4p14e18-diag-ep16-r16-coord`;
- `/power/nest-burn-001-fixed25-c4p14e18-diag-ep16-r16-coord`.

The `(replica=1, data=2, expert=16)` smokes compiled and initialized their
expert communicators, then reproduced the first-dispatch hang. RAS showed no
rank skew or error. Both arms stopped with `AllGather=83` on every size-two
data communicator; the E256 and fixed25 global communicators stopped at
`AllGather=64, AllReduce=6` and `AllGather=55, AllReduce=5`. Each expert
communicator reached `AllGather=28, ReduceScatter=10`.

This demonstrates that EP=16 is not itself the trigger: the remaining
two-device FSDP shard is sufficient. The next recovery removes the data axis
entirely. Each matched arm uses 16 GPUs with
`(replica=1, data=1, expert=16, model=1)`, so experts are sharded across every
device and all dense parameters are replicated:

- `/power/nest-burn-001-e256-c4p14e18-diag-eponly-r17-coord`;
- `/power/nest-burn-001-fixed25-c4p14e18-diag-eponly-r17-coord`.

The pure-EP isolate also reproduced. E256's 16-rank communicator stopped at
`AllGather=92, ReduceScatter=10, AllReduce=8`; four size-four communicators
stopped at `AllReduce=1`. All ranks were aligned, and fixed25 showed the same
shallow PJRT stack and low-power GPU busy-poll state. FSDP is therefore not a
necessary trigger.

The `grug/embedding-gather-shard-map` branch documents a separate first-step
rendezvous caused by the token embedding lookup. Its production fix fully
replicates the embedding table and performs each batch shard's lookup under
`shard_map`; it also restricts the LM-head contraction shard to the data axis.
Those changes are now scoped to the MoE model in this worktree. Matched r18
two-step smokes retain the pure EP=16 mesh:

- `/power/nest-burn-001-e256-c4p14e18-diag-embedfix-r18-coord`;
- `/power/nest-burn-001-fixed25-c4p14e18-diag-embedfix-r18-coord`.

E256 r18 reproduced with the same pure-EP communicator counts as r17, so the
embedding rendezvous fixed on `grug/embedding-gather-shard-map` is not the
direct cause of this incident.

The device-zero rematerialization warning names `s32[32,2]`, matching the
FA4-THD packed-sequence metadata built from the global batch and Datakit's two
packing segments. The THD path explicitly replicates those lengths before its
custom call. The FA4 CuTe path performs attention under a batch-axis
`shard_map`, so r19 changes only the attention implementation while retaining
the pure EP=16 topology:

- `/power/nest-burn-001-e256-c4p14e18-diag-cute-r19-coord`;
- `/power/nest-burn-001-fixed25-c4p14e18-diag-cute-r19-coord`.

r19 did not reach executable dispatch. CUTLASS DSL 4.6 rejected four
pre-4.6 `cute.make_fragment` calls in the segmented backward kernel. Ported
the supplied branch's `5833e329ea99` migration to `cute.make_rmem_tensor` and
submitted a single-arm API smoke before spending another matched pair:

- `/power/nest-burn-001-e256-c4p14e18-diag-cuteapi-r20-coord`.

r20 passed CuTe kernel compilation but emitted eight per-layer SPMD warnings
for `s32[32,8192]` conditional outputs pinned to device zero and then scattered
across 16 expert ranks. An on-demand thread dump showed the main thread in
`backend_compile_and_load`; this was compilation, not the earlier PJRT
collective wait.

Ported the supplied branch's precomputed FA4 bounds path to the unrolled MoE.
Long and sliding-window bounds are now computed once outside the layer loop,
batch-sharded, and attached to each layer mask. The replacement isolate is:

- `/power/nest-burn-001-e256-c4p14e18-diag-cutebounds-r21-coord`.

r21 compiled without the device-zero bounds warnings and executed both
train-step variants. The first forward loss was finite (11.7966), but CuTe's
segmented backward returned NaN/Inf gradients before the optimizer update.
The second step then detected non-finite loss from poisoned parameters. This
separates the runtime recovery from an independent CuTe backward correctness
defect.

The THD backend's metadata conversion contains two compiled `eqx.error_if`
conditionals. The first wraps the exact `[32,2]` segment-length tensor named by
the original SPMD warning. Removed these redundant checks—the metadata is
constructed from validated packed segment IDs—and submitted:

- `/power/nest-burn-001-e256-c4p14e18-diag-thdvalidate-r22-coord`.

r22 compiled and reproduced the first-dispatch freeze. Live NCCL RAS was
queried twice, 20 seconds apart. Every rank in the 16-device communicator was
stationary at `AllGather=134`, `ReduceScatter=16`, and `AllReduce=9`; four
size-four communicators were stationary at `AllReduce=1`. RAS reported no
missing rank, count skew, async error, or timeout. Removing the THD metadata
conditionals therefore removed a device-zero lowering hazard but did not clear
the runtime failure.

The known-good r47 cell used the same JAX 0.11, CUDA 13, NCCL 2.28.9, and
CUTLASS 4.6 stack at sequence length 2,048. The failing burn cells use sequence
length 8,192. The next recovery bypasses THD with the supplied branch's cuDNN
fused-attention path, applied identically to a matched two-arm smoke:

- `/power/nest-burn-001-e256-c4p14e18-diag-cudnnprefix-r23-coord`;
- `/power/nest-burn-001-fixed25-c4p14e18-diag-cudnnprefix-r23-coord`.

The E256 r23 gang later exposed a distinct compiler-future wedge before model
initialization completed. All ranks slept in `backend_compile_and_load`, and
rank 0 accumulated only seven process CPU ticks over 20 seconds (about 0.35
cores). The concurrently compiling fixed25 rank accumulated 2,446 ticks
(about 122 cores). No executable dispatch or optimizer step occurred. A
queued gang kick did not apply after two controller ticks, so only the E256
r23 coordinator was stopped and the identical control was resubmitted as:

- `/power/nest-burn-001-e256-c4p14e18-diag-cudnnprefix-r24-coord`.

The fixed25 cuDNN arm subsequently reached executable creation and failed on
all ranks with `cudnn_frontend: No valid execution plans built`. The
coordinator connection failures were secondary gang teardown. The cuDNN
fallback therefore cannot execute this sequence-8,192 graph. Both cuDNN
coordinators were stopped, and the next matched smoke uses the existing
reference attention backend:

- `/power/nest-burn-001-e256-c4p14e18-diag-reference-r25-coord`;
- `/power/nest-burn-001-fixed25-c4p14e18-diag-reference-r25-coord`.
