# Event Tensor attachment to high-throughput GPU skeletons

Status: the SM90 streaming Contract/Fold path has a structural attachment. An
H100 replay is blocked by the same pre-existing normalized-exponential CuTe
verification failure in both the canonical and Event Tensor sources. The
grouped-GEMM path remains blocked at an opaque primitive synchronization
contract.

## Boundary

Event Tensors remain schedule objects. Tensor semantics determine the QK
Contract, normalized-exponential Fold, PV Contract, and final Map. Task
decomposition then induces exact producer/consumer relations. The attachment
derives bounded-buffer lifetimes and synchronization parameters from those
relations before the CuTe backend allocates concrete barriers.

The current attachment does not synthesize CUDA barrier instructions. It gives
the existing generic tensor-core skeleton a verified synchronization contract:

```text
Contract / Fold program
  -> task relations
  -> separate Q, K, and V BoundedBufferPlans
  -> EventTensorPlans and realization audit
  -> pipeline depths, barrier storage, worker counts, transaction bytes
  -> CuTe full/empty mbarrier allocation
```

Barrier identifiers, phase-bit encoding, and the CuTe pipeline primitive are
backend allocation choices. The generated payload body is unchanged.

## SM90 streaming Contract/Fold inventory

The high-throughput implementation is
`backends/h100/cute_streaming_sm90.py`. It uses TMA and WGMMA through CuTe.

| Quantity or edge | Previous source | Derivation | Attachment status |
| --- | --- | --- | --- |
| K/V pipeline depth | `StreamingTileSchedule.pipeline_depth` passed as `num_stages` | Capacity of separate K and V `BoundedBufferPlan`s | Derived and checked |
| Q pipeline depth | Literal one stage | Q remains resident for one scheduled work tile; next tile is a new generation | Derived and checked |
| Barrier storage per stage | Literal two entries | One full and one empty event realization per bounded-buffer slot | Derived and used by shared-storage construction |
| TMA producer participants | Literal one cooperative warp | Selected transfer-worker assignment | Derived and checked |
| Matrix consumer participants | Tiled-MMA size | `query_tile / 64` matrix warpgroups, four warps each | Derived independently and checked against the tiled MMA |
| CTA threads | Recomputed in the skeleton | One transfer warpgroup plus derived matrix warpgroups | Derived and checked |
| TMA transaction counts | CuTe layout byte size | Tile extent, head/value dimension, BF16 width | Derived independently and checked against CuTe |
| K full event | Pipeline construction | `key_stage -> QK Contract` | Physical Event Tensor realization |
| V full event | Pipeline construction | `value_stage -> PV Contract` | Physical Event Tensor realization |
| Q full event | Pipeline construction | `query_stage -> every QK partition` | Physical Event Tensor realization |
| K empty/reuse event | `consumer_release` after QK | Last consumer of each K item is QK | Derived `QK -> next key_stage` phased dependence |
| V empty/reuse event | `consumer_release` after PV | Last consumer of each V item is PV | Derived `PV -> next value_stage` phased dependence |
| Q empty/reuse event | Q release in the matrix body | Last Q consumer is the final QK partition | Derived `final QK -> next query_stage` phased dependence |
| Ordered Fold handoff | Named scheduler barrier | `PV(partition) -> QK(partition + 1)` under the selected overlapping worker schedule | Physical with multiple matrix warpgroups; erased with one |
| Scheduler arrival participants | Literal `2 * 128` threads | Pairwise handoff between adjacent 128-thread matrix warpgroups | Derived and checked |
| Finalization readiness | Loop/control order | Last PV partition reaches finalization through the ordered recurrence | Event erased by proven program order |
| Q/K/V phase seeds | Literal pipeline-state conventions | Need for distinct generations follows from buffer reuse; zero/one phase encoding belongs to CuTe | Backend-owned encoding |
| Named barrier identifiers | FlashAttention-derived enum | Allocation among backend-reserved barrier IDs | Backend-owned allocation |
| Cluster initialization | Literal `(1, 1)` cluster shape | Current skeleton does not distribute one tile across a cluster | Physical schedule choice, not yet searched |
| V-only producer tail | Handwritten load order | V is the last asynchronous producer in the chosen K-then-V transport order | Audited schedule fact; not yet emitted from Event Tensor IR |

Q, K, and V are intentionally separate. Combining K and V into one buffer
would delay K reuse until PV finishes and would not describe the physical
pipeline accurately.

The derivation uses two logical query-tile generations to expose Q-buffer
reuse. It represents every K/V partition for each generation. Pipeline-depth
and worker-decomposition mutations regenerate buffer assignments, generations,
event realizations, barrier counts, and the audit fingerprint without a
workload dispatch key.

## SM100 grouped-GEMM inventory

The performance-bearing grouped Contract wrapper is
`backends/sm100/mok_gmm_probe/mok_gmm_probe.cu`. It calls a generic segmented
contraction primitive extracted from Mixture-of-Kittens, not the complete MoE
kernel.

| Quantity | Current source | Candidate derivation | Status |
| --- | --- | --- | --- |
| Load-pipeline stage count | External `config::MLP_LOAD_PIPE_DEPTH` | Bounded input/weight tile buffers selected for the Contract | Derivable in principle |
| `inputs_arrived[stage]` count | Literal one | Number of elected input-tile producers per stage | Primitive ownership is opaque |
| `scales_arrived[stage]` count | Literal one | Number of scale-tile producers per stage | Primitive ownership is opaque |
| `inputs_finished[stage]` count | Literal one | Number of logical consumers elected to release the input slot | Primitive ownership is opaque |
| `scales_finished[stage]` count | Literal one | Number of logical consumers elected to release the scale slot | Primitive ownership is opaque |
| `outputs_arrived` count | Literal one | Completion of the matrix owner producing the output tile | Primitive ownership is opaque |
| `outputs_finished` count | External `config::CLUSTER_SIZE` | Number of cluster CTAs consuming or acknowledging the output tile | Primitive ownership is opaque |
| Cluster-wide sync before/after call | Handwritten wrapper | Cluster visibility around tensor-memory allocation and teardown | Backend primitive contract |
| Per-stage generations | Internal primitive implementation | Repeated K-tile traversal and pipeline-slot reuse | Not exposed by the primitive interface |

The correct next step is to expose a generic grouped-Contract synchronization
descriptor from the primitive: task owners, producer/consumer cardinalities,
buffer stages, and release points. Generating a header that repeats the current
literal counts without that information would reverse-engineer and bless an
opaque contract, not derive synchronization from Shuttle task relations. This
checkpoint therefore stops at the explicit interface gap.

Runtime `RelationPlan` indegrees still derive when a segmented Contract becomes
eligible. They do not determine the internal TMA/WGMMA semaphore counts once a
Contract tile has been launched. Those are distinct event domains.

## Validation and remaining work

CPU tests currently establish:

- Q, K, and V last-consumer derivation;
- independent bounded-buffer generations;
- pipeline-depth mutation;
- one- versus two-matrix-warpgroup mutation;
- rejection when backend synchronization constants drift from the Event Tensor
  schedule.

An H100 replay was attempted with the pinned helper stack. Both the canonical
comparison source and the Event Tensor branch fail at the same CuTe IR
verification in the identical Shuttle-owned normalized-exponential helper:
the row-sum register tensor does not dominate a generated layout use. The
failure occurs before benchmark samples are emitted, so the structural
checkpoint still has no performance claim. Commands, environment, source
hashes, and sanitized stderr are preserved under
`benchmarks/artifacts/event_tensor_sm90_compile_blocker_h100_v0/`.

Before this becomes a performance-bearing checkpoint, repair and independently
validate that canonical Fold helper, then replay the unchanged CuTe payload on
an actual H100 and preserve generated source, event-plan fingerprint, hardware
identity, correctness, determinism, and latency. A B200 run would be a separate
portability result. A GB200 run would require an SM100 backend attachment and
must not be inferred from either result.
