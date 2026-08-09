# Event Tensor background research

## Background Research Brief

- Effort: medium
- Stop rule: stopped after the primary paper, paper source and slides, official TVM follow-up, public-code search, related in-repository schedule structures, and an adversarial limitations pass converged on the same design boundary.
- Date: 2026-08-08

### Question

Should Shuttle represent Event Tensors, and if so, should they be semantic tensor operations, explicitly authored task-graph edges, or derived schedule objects?

### Current Marin Context

Shuttle already has three partial schedule vocabularies:

- `ReadinessEvent`, `PersistentTaskRole`, and `BoundedBuffer` in `plan.py` describe named physical schedule records.
- `TileFlowEdge`, `BufferLifetime`, and `ReadinessGranularity` in `expert_parallel_plan.py` describe MoE tile flow and storage.
- `routed_attention_plan.py` manually constructs counted readiness for query-major, KV-major, and slot-wave candidates.

These records are inspectable, but they do not retain an exact indexed producer-to-consumer relation. Arrival counts are constructed by workload planners rather than derived through one generic pass.

### External Prior Art

The primary source is Jin et al., [*Event Tensor: A Unified Abstraction for Compiling Dynamic Megakernel*](https://arxiv.org/abs/2604.13327), MLSys 2026. The paper represents a task graph as tiled device-function calls connected through tensor-shaped event counters. Coordinate relations map producer tiles to event coordinates and event coordinates to consumer tiles. The same graph can lower to static per-SM queues with notify/wait or a dynamic on-GPU ready queue with push/pop. Runtime-dependent mappings such as token-to-expert routing determine both notifications and triggered task ranges.

The [MLSys 2026 slides](https://mlsys.org/media/mlsys-2026/Slides/3815.pdf) make the compiler contract especially explicit: the input already contains tiled task grids and `in_edges`/`out_edges` Event Tensor annotations. The paper's conclusion identifies automatic generation of Event Tensor task graphs from standard computational graphs as future work. Shuttle's proposed derivation from a post-decomposition exact dependence relation therefore extends, rather than reproduces, the paper's frontend boundary.

The official TVM [TIRx announcement](https://tvm.apache.org/2026/06/22/tirx) says Event Tensor was built on a TVM-based tile DSL and that integration into TIRx is planned for follow-up releases. As of this research date, searches of the paper, arXiv metadata/source, conference slides, Apache TVM, MLC GitHub, and the authors' linked public pages found no official ETC repository or artifact. A third-party repository named RuntimeCompGen contains files that use the phrase “event tensor,” but it is not authored or cited by the ETC team and is not evidence about the paper's implementation.

The physical-lowering boundary in the plan agrees with NVIDIA's programming model. The CUDA guide describes [asynchronous barriers](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-programming-guide/04-special-topics/async-barriers.html) as separate arrival and wait phases, with hardware support at block and cluster scopes, and describes [Programmatic Dependent Launch](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html) as opportunistic overlap that still requires an explicit dependency synchronization before consuming producer data. The [PTX memory model and `mbarrier` reference](https://docs.nvidia.com/cuda/archive/12.1.1/parallel-thread-execution/index.html) distinguish CTA, cluster, GPU, and system scopes and attach visibility semantics to acquire/release synchronization. These sources support keeping readiness and memory visibility explicit while deferring primitive selection.

### Negative / Failed Leads

- No official ETC code or artifact was found. The paper provides compiler pseudocode but not an executable artifact link.
- The paper does not derive event relations from ordinary tensor semantics; it assumes pre-tiled operators with event edges already annotated.
- The paper does not present a formal dependency-coverage verifier, deadlock proof, event-generation protocol, or numerical-determinism analysis.
- The static scheduler uses round-robin queues and conservatively collapses data-dependent edges to a single event. The dynamic scheduler uses a centralized global-memory queue. These are useful lowering examples, not evidence that either policy is universally profitable.
- Event counters alone do not state the memory visibility contract. A counter reaching zero is insufficient unless the selected primitive makes producer writes visible to the consumer.

### Evidence Map

#### Claim: event coordinates compactly factor tiled task dependencies

- Support:
  - Primary paper, Sections 2.1–2.2: event elements have wait counts; producer and consumer coordinate mappings define notify and trigger edges.
  - MLSys slides, pages 13–14: `ij -> i` and `i -> i` encode split-reduction dependencies.
- Contradictions:
  - The paper starts from explicit event annotations, so compact representation does not imply automatic derivation.
- Directness to Marin: exact; Shuttle already decomposes Folds, relations, and communication-style tile flows.
- Confidence: high.
- Action: retain exact task dependence as source of truth and mechanically derive a producer-event-consumer factorization.

#### Claim: one event plan can support static and dynamic schedules

- Support:
  - Primary paper, Sections 3.1–3.2 and Appendix A: static notify/wait and dynamic trigger/push/pop are transformations of the same Event Tensor graph.
- Contradictions:
  - Paper ablations show static and dynamic performance trade-offs vary by workload; dynamic queues are not free.
- Directness to Marin: high for a backend-neutral reference interpreter; unproven for production SM100 schedules.
- Confidence: high for semantics, exploratory for performance.
- Action: implement both as reference policies and stop before GPU runtime work.

#### Claim: Event Tensor should be below Shuttle semantics

- Support:
  - The paper operates on already tiled device functions.
  - Shuttle's exact dependency relation appears only after semantic rewrite, materialization, and task decomposition choices.
- Contradictions:
  - ETC calls Event Tensor a first-class compiler IR object and includes it in graph functions.
- Directness to Marin: exact; putting Event Tensor in `TensorProgram` would make synchronization part of mathematical meaning.
- Confidence: high.
- Action: place `EventTensorPlan` in a schedule-only module and leave `TensorProgram` unchanged.

### Recommended Next Experiments

#### 1. Derive events from three unrelated exact dependency relations

- Minimum experiment: split Fold, runtime relation segmentation, and tiled Contract-to-placement-change graph.
- Baseline/control: direct reference computation and the exact dependency relation before eventization.
- Expected signal: one derivation/verifier/interpreter implementation handles all three without workload dispatch.
- Falsifier: any case needs named workload logic in event construction.
- Cost/risk: low; CPU-only bounded prototype.
- Sources: Event Tensor paper Sections 2–3; Shuttle prototype plan.

#### 2. Treat event granularity as schedule search

- Minimum experiment: compare one event per consumer with a projection that groups consumers.
- Baseline/control: exact required dependency relation.
- Expected signal: both execute correctly and the coarser plan's composed relation is a strict superset.
- Falsifier: the representation cannot distinguish required from scheduled false dependencies.
- Cost/risk: low.
- Sources: paper's conservative single-event static fallback; Shuttle prototype plan.

#### 3. Defer SM100 legalization but preserve its required inputs

- Minimum experiment: record scope, release/acquire visibility, generation, count, placement, and scheduling mode in the logical plan.
- Baseline/control: CUDA/PTX requirements for barriers, device atomics, and PDL.
- Expected signal: an emitter can choose no-op/program order, `mbarrier`, scoped semaphore, queue trigger, PDL, or kernel boundary without changing dependency semantics.
- Falsifier: target selection needs semantic workload identity rather than schedule metadata.
- Cost/risk: documentation only in this phase.
- Sources: CUDA 13.2 programming guide and PTX ISA.

### Hypothesis Queue Update

- Add: exact dependence is the durable schedule-semantic object; Event Tensor is a derived factorization selected for implementation.
- Add: event coarsening can be represented as a quotient/projection of consumer coordinates.
- Revise: existing scalar `ReadinessEvent` remains a plan-dump/physical record, not the source for indexed count derivation.
- Falsify / stop: do not put Event Tensor into semantic `TensorProgram`.
- Promote: memory visibility and generation must survive into physical lowering even in the CPU prototype.

### Source Ledger

| Source | Type | Location | Claim used for | Confidence | Notes |
|---|---|---|---|---|---|
| Jin et al., Event Tensor | paper | https://arxiv.org/abs/2604.13327 | Event representation, dynamic relations, static/dynamic transformations, current frontend boundary | high | Primary paper and downloaded source read in full |
| MLSys 2026 Event Tensor slides | conference slides | https://mlsys.org/media/mlsys-2026/Slides/3815.pdf | Concrete IR syntax and evaluation summary | high | Author presentation |
| TIRx announcement | official project post | https://tvm.apache.org/2026/06/22/tirx | TVM-based implementation lineage and future public integration | medium-high | Does not release ETC itself |
| CUDA asynchronous barriers | official docs | https://docs.nvidia.com/cuda/archive/13.2.0/cuda-programming-guide/04-special-topics/async-barriers.html | Physical scope and arrival/wait candidates | high | CUDA 13.2 |
| CUDA PDL | official docs | https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html | Kernel-boundary lowering and visibility caveat | high | CUDA 13.1 |
| PTX ISA memory model and mbarrier | official docs | https://docs.nvidia.com/cuda/archive/12.1.1/parallel-thread-execution/index.html | CTA/cluster/GPU/system scopes, phases, acquire/release | high | Versioned PTX reference |
| Shuttle schedule code | Marin code | `lib/tile_lifetime/src/tile_lifetime/{plan,expert_parallel_plan,routed_attention_plan}.py` | Existing readiness and tile-flow abstractions | high | Base commit `e4e40e8c6d` |

### Handoff

- Suggested logbook entry: Event Tensor should enter Shuttle as a derived schedule view over exact indexed task dependencies, not as tensor semantics. The primary ETC paper assumes explicit tiled event annotations; Shuttle's experiment tests automatic derivation, coverage verification, coarsening, and target-neutral lowering metadata.
- Open questions: symbolic rather than concretely enumerated relations; circular-buffer capacity/backpressure; multi-generation persistent execution; profitable SM100 primitive selection.
- Stop reason: additional searches no longer changed the bounded prototype or ranked experiments.
