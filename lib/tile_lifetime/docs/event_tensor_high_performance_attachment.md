# Event Tensor attachment to high-throughput GPU skeletons

Status: the SM90 streaming Contract/Fold attachment compiles and executes on
H100. Its matched Event/pre-Event latency ratio is 1.001x at the tested shape.
The SM100 grouped-Contract wrapper now consumes a generated synchronization ABI
whose ownership, logical counts, buffer stages, transaction bytes, and release
points are derived from a generic task graph. That wrapper compiled and passed
correctness on a physical NVIDIA GB200. The external grouped-GEMM primitive
still owns the placement of its barrier arrival and wait instructions.

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

| Quantity | Previous source | Derivation | Attachment status |
| --- | --- | --- | --- |
| Load-pipeline stage count | External `config::MLP_LOAD_PIPE_DEPTH` | Capacity of the bounded operand-tile buffer selected for the Contract | Derived, generated, and statically checked |
| Operand producer ownership | Implicit in primitive control flow | One cooperative transfer task per cluster CTA for each K partition | Exposed by generic worker assignment and mutated in tests |
| Logical operand-ready count | Not represented separately from TMA completion | Notify indegree from cooperative transfer tasks to one operand-stage event | Derived as two for the selected two-CTA cluster |
| Operand TMA completion | Literal transaction-enabled barrier plus expected bytes | Physical realization of the operand-ready event from tile byte extents and cluster cardinality | Derived as 65,536 bytes; kept distinct from logical indegree |
| Operand-release count | Literal one | One matrix owner consumes the cluster-wide operand stage | Derived as one |
| Operand release point | Internal matrix loop | Last consumption of the staged A/B operands | Named `matrix_operand_consumed` by the generated ABI |
| Output-ready count | Literal one | One matrix owner completes the accumulator tile | Derived as one |
| Output-release count | External `config::CLUSTER_SIZE` | One epilogue task per cluster CTA reads its output half | Derived as two and mutated with cluster cardinality |
| Output release point | Internal epilogue | Accumulator-to-register read completes | Named `accumulator_read_complete` by the generated ABI |
| Per-stage generations | Internal phase loop | Bounded-buffer slot reuse across K partitions | Derived by the task graph; stage mutation changes slots, generations, and fingerprint |
| Cluster-wide sync before/after call | Handwritten wrapper | Cluster visibility around tensor-memory allocation and teardown | Backend primitive contract |

The descriptor is generic: it describes cluster participants, task owners,
bounded stages, logical producer and consumer cardinalities, and release
points. `derive_grouped_contract_physical_event_schedule` builds the exact task
relations and mechanically produces the Event Tensor plans. The SM100 codegen
then emits a fingerprinted include consumed by the wrapper. Static assertions
reject drift between the descriptor and the selected grouped-Contract tile
configuration.

The TMA realization deserves special care. The primitive initializes the
operand-ready barrier with transaction completion enabled and supplies expected
bytes for both cluster CTAs. The physical barrier's arrival-count argument is
therefore one, but the logical Event Tensor indegree is two cooperative transfer
tasks. Shuttle represents both facts instead of mistaking the CUDA barrier
encoding for the task-dependence cardinality.

Runtime `RelationPlan` indegrees still derive when a segmented Contract becomes
eligible. They do not determine the internal TMA/WGMMA semaphore counts once a
Contract tile has been launched. Those are distinct event domains.

Shuttle now owns the synchronization ABI and counts at the grouped-Contract
wrapper boundary. The remaining ownership gap is narrower: the external
primitive still contains the actual mbarrier arrival/wait operations, phase-bit
advancement, TMA issue, and accumulator read/release sites. Closing it requires
the generic primitive to accept the generated descriptor at those release
sites, or extracting a generic grouped-Contract template whose synchronization
operations are emitted from that descriptor. The current result does not claim
that Shuttle generates those internal instructions. The MXFP8 scale pipeline
also remains unaudited; this checkpoint covers the BF16 operand pipeline only.

## Validation and remaining work

CPU tests currently establish:

- Q, K, and V last-consumer derivation;
- independent bounded-buffer generations;
- pipeline-depth mutation;
- one- versus two-matrix-warpgroup mutation;
- rejection when backend synchronization constants drift from the Event Tensor
  schedule.

The grouped-Contract tests additionally establish:

- mechanical derivation of worker owners, logical cardinalities, and release
  points;
- cluster-cardinality mutation from two to four CTAs;
- pipeline-stage mutation from two to three slots;
- separation of logical producer indegree from byte-counted TMA completion;
- rejection of backend-parameter and generated-include drift.

The original H100 replay exposed a CuTe SSA dominance failure in Shuttle's
normalized-exponential helper. A finalize-only alias change did not repair it.
Carrying the register state through local SSA values during every Fold update
did. On one H100, two counterbalanced 10-sample captures measured 0.080272 ms
for the repaired pre-Event source and 0.080352 ms for the Event Tensor source,
a 1.000997x ratio. Both paths were correct, bitwise deterministic, and produced
the same output hash. The failed and successful replays are preserved under
`benchmarks/artifacts/event_tensor_sm90_fold_alias_replay_h100_v1/` and
`benchmarks/artifacts/event_tensor_sm90_fold_state_replay_h100_v1/`.

The SM100 wrapper was built from Shuttle revision `30c0ba6bfc` on a physical
NVIDIA GB200 reported by `nvidia-smi`, driver 595.71.05. The build used PyTorch
2.10.0+cu130, NVCC 13.0.88, pinned MoK `3e1cf43ab9`, and pinned ThunderKittens
`1c3920d993`. Its generated synchronization fingerprint matched at runtime.
The W2 correctness probe passed with maximum absolute error 0.0148849, mean
absolute error 0.00112154, and no NaNs or infinities. PTXAS reported 255
registers, five barriers, 224 bytes of static shared memory, and no spills for
the grouped-GEMM kernel. This validates the generated wrapper ABI; it is not a
claim that Shuttle yet emits the primitive's internal barrier instructions.

The exact five-sample result and provenance are preserved under
`benchmarks/artifacts/event_tensor_grouped_contract_sm100_gb200_v0/`. B200 is a
separate hardware target and must not be labeled as GB200 evidence.
