# Event Tensor SM100 lowering attachment

This note identifies the target inputs needed for an eventual SM100 emitter. It does not implement GPU synchronization.

## Selection inputs

An SM100 lowering should consume:

- exact and scheduled task relations;
- producer and consumer placement, including warpgroup, CTA, cluster, device, or separate kernel;
- event count and whether it is static or runtime-derived;
- event generation policy and buffer generation;
- event memory scope;
- release/acquire visibility requirements;
- static or dynamic scheduling mode;
- event fan-in/fan-out and coarsening metadata;
- whether the dependency tracks task completion, asynchronous transaction completion, or transport completion;
- buffer address space and reuse condition;
- boundedness and deadlock proof obligations.

## Candidate legalization

The emitter can choose the weakest sufficient mechanism:

| Schedule condition | Candidate lowering |
|---|---|
| Same sequential task | Program order; erase event |
| Same warp with guaranteed converged order | Warp ordering or warp synchronization |
| Producer/consumer within one CTA | Shared-memory asynchronous barrier or CTA barrier |
| Cross-CTA within a cluster | Cluster-scope barrier or cluster-visible shared state |
| TMA completion feeding consumers | Transaction-counted `mbarrier` phase |
| Device-wide persistent task graph | Scoped global semaphore; optionally queue on zero transition |
| Separate dependent kernels | Stream/kernel boundary or Programmatic Dependent Launch plus explicit dependency synchronization |
| Remote/system-visible transport | Transport completion plus system-scope release/acquire |

NVIDIA's [CUDA asynchronous barrier documentation](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-programming-guide/04-special-topics/async-barriers.html) distinguishes block, cluster, device, and system scopes and transaction-tracking barriers. The [PTX `mbarrier` documentation](https://docs.nvidia.com/cuda/archive/12.1.1/parallel-thread-execution/index.html) includes phases, expected arrivals, pending transactions, and acquire/release rules. [Programmatic Dependent Launch](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html) can overlap dependent kernels but does not by itself guarantee that producer data is visible when a secondary block begins.

## Required follow-up verification

Before any physical lowering is accepted, verify:

- selected scope covers all producer and consumer placements;
- notification has the required release and consumption has the required acquire;
- expected arrival and transaction counts match the logical event count;
- phase/generation reuse cannot observe stale arrivals;
- event storage outlives every producer and consumer in its generation;
- static waits cannot occupy all resources needed by unfinished producers;
- dynamic queues cannot overflow and do not enqueue a task twice;
- buffer reuse is coupled to consumer completion, not merely producer completion.

The current prototype provides the dependency, count, scope, visibility, generation, and scheduling-mode inputs. It intentionally does not yet model worker capacity, circular-buffer backpressure, asynchronous transaction byte counts, or a hardware primitive registry.
