# Event Tensor GPU coverage for routed work and attention

This audit separates task edges that execute through generic Event Tensor
lowering from structural plans and opaque backend boundaries. It does not treat
an expert kernel's internal semaphore constants as compiler-derived evidence.

## Attention

The high-throughput readiness derivation now consumes a generic
`StreamingContractFoldDescriptor`. The descriptor contains only two Contract
task names, one Fold-update task name, one finalizer task name, logical
extents, physical tile sizes, payload widths, and pipeline depth. The adapter
from the current normalized-attention program erases the semantic operation
before `derive_streaming_physical_event_schedule` constructs any task relation
or Event Tensor. The Event Tensor module no longer imports an attention
program, score-axis enum, mask, or normalized-exponential implementation.

The generated correctness-oriented streaming body executes these edges on a
GB200 through JAX typed FFI:

| Edge | Realization |
| --- | --- |
| K/V stage -> QK/PV Contract | CTA acquire barrier |
| QK Contract -> normalized-exp Fold | Program order in one row owner |
| Fold update -> PV Contract | Program order in one row owner |
| Last PV -> finalization | Partition-loop completion |
| Last K/V consumer -> next stage generation | CTA release barrier and generation advance |

This body consumes real Q/K/V tensors and runs exact online normalized-exp
state, but it is an FP32 scalar/reference payload. Its approximately 0.075 ms
latency is execution evidence, not an attention performance result.

The H100 SM90 TMA/WGMMA skeleton has a separate high-throughput attachment.
Shuttle derives Q/K/V full events, K/V/Q empty and reuse events, the ordered
Fold handoff, pipeline capacities, generations, barrier storage, transaction
bytes, and worker participants. CuTe allocates physical mbarriers and phase
bits. The matched H100 replay measured 0.080272 ms before the attachment and
0.080352 ms with it, a 1.001x ratio, with identical deterministic output hashes.

The remaining attention gaps are routed/sparse scheduling and backend breadth.
The high-throughput attachment covers the generic dense streaming
Contract/Fold schedule. The KV-major routed-attention relation and inverse
partial-state movement do not yet execute through an equivalent tensor-core
Event Tensor attachment.

## MoE and segmented work

The generated RelationPlan/SegmentedContract body executes this edge on a
GB200:

```text
RelationPlan CSR edge readiness -> segmented Contract task
```

Relation indegrees and offsets determine the task domain. The first generated
CUDA body gathers the source row and runs the segment Contract in one CTA, so
the edge event erases to program order. This is real tensor/CSR execution and
mutates with the relation, but it is not distributed expert-parallel transport
or a high-throughput grouped GEMM.

The natural differentiated Grug transform now has a clean whole-value
collective attachment:

```text
generated group-batched weight Contract
    -> existing post-SPMD all-reduce
    -> recovered CollectiveFoldPlan
    -> system-visible Event Tensor completion
```

The generated Contract remains the direct producer of the ordinary XLA
all-reduce. Replica groups derive Event Tensor counts. Shuttle does not replace
the collective, register a custom adjoint, or claim transport ownership. JAX
and XLA retain AD and physical communication selection.

The generic boundary was replayed on two H100 GPUs with BF16 inputs. Full-group
sum, a grouped-maximum reducer mutation, and JAX-owned reverse execution all
matched their references exactly and were bitwise deterministic. The forward
StableHLO modules contain one ordinary all-reduce each, the differentiated
module contains two, and none contains a semantic custom call. This establishes
real multi-GPU execution of the recovered collective/Event Tensor contract; it
is not a communication-performance claim.

The performance-bearing grouped-GEMM primitive remains the high-throughput MoE
gap. Its load/output semaphore storage is visible, but producer ownership,
consumer ownership, arrival cardinalities, and per-stage release points remain
internal. Runtime relation indegrees establish when a segmented Contract may
start; they do not prove the primitive's internal TMA/WGMMA semaphore counts.
Copying those literal counts would encode the oracle schedule.

DeepEP or another ragged transport can remain a generic payload transport. Its
asynchronous completion has not yet been connected to the segmented-Contract
Event Tensor on GPU. The current distributed MoE result therefore does not
claim a fully generated transport-to-GMM readiness pipeline.

## Exact evidence boundary

The current GPU evidence establishes:

- generic runtime-relation readiness and a segmented payload body;
- generic streaming Contract/Fold readiness with bounded shared-memory reuse;
- an Event-derived synchronization descriptor attached to an SM90 tensor-core
  attention skeleton;
- whole-value collective semantics attached to natural Grug weight Contracts,
  with physical transport still delegated to JAX/XLA.

It does not establish:

- Event-derived internal synchronization for the high-throughput grouped GEMM;
- Event-driven overlap between asynchronous ragged transport and expert
  Contracts;
- a high-throughput routed sparse-attention Event Tensor backend;
- Shuttle-owned all-reduce, reduce-scatter, or communication AD.

The next high-performance MoE step is to connect runtime RelationPlan
eligibility to the existing grouped-Contract synchronization descriptor in a
Torch-free executable path. The next routed-attention step is to attach the
existing RelationPlan orientation and partial-state merge tasks to the
tensor-core streaming skeleton without adding a sparse-attention workload
switch.

## Smallest next GPU experiment

Do not spend the next allocation retiming the reference payloads. The smallest
missing execution proof is a Torch-free JAX typed-FFI chain on one physical
GB200:

```text
runtime RelationPlan tables
  -> generated grouping/padding Contract preparation
  -> device-visible EventTensorPlan completion
  -> generic SM100 grouped Contract
```

The first bounded version should realize the outer readiness edge by verified
same-stream order. This legally coarsens the per-edge relation without claiming
transport overlap. It must use uneven runtime segment counts with at least one
empty segment, then mutate the relation while keeping the event construction
and grouped-Contract body unchanged. The result must compare against the same
segmented Contract reference, be deterministic, record handler invocations,
and preserve the generated outer Event Tensor and inner synchronization
fingerprints. JAX owns the entry point and any differentiation. Torch may be
used in a separate build/oracle process, but the measured runtime must import
neither Torch nor a complete MoE kernel.

This experiment would prove relation-to-grouped-Contract GPU linkage. It would
not prove fine-grained asynchronous transport overlap, and it would not close
the internal grouped-Contract ownership gap: the external primitive would
still place its mbarrier arrive/wait operations. Those claims must remain
separate.

The CPU-side schedule boundary is prepared. A
`SegmentedGroupedContractEventSchedule` composes runtime segment readiness with
the independently derived grouped-Contract pipeline, realizes the outer edge
by verified device-stream order, and keeps program and runtime fingerprints
separate. A relation mutation changes only runtime tables and their fingerprint;
a cluster-worker mutation changes only the inner physical schedule and program
fingerprint. Empty segments are represented by zero-count ready events. This
has not been replayed through the SM100 typed-FFI path yet.

For attention, another dense H100 replay is not presently informative: the
real TMA/WGMMA attachment is already correct, deterministic, and 1.001x the
matched pre-attachment body. The next attention allocation should instead be a
routed relation mutation whose generic right-resource staging feeds the same
streaming descriptor and whose partial Fold states return to a generated merge.
