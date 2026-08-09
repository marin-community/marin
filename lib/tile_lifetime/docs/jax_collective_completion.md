# JAX-owned collective completion

Shuttle can now execute a recovered `CollectiveFoldPlan` without owning model
automatic differentiation or a communication runtime. The bounded prototype
connects four existing objects:

```text
post-SPMD HLO all-reduce
    -> CollectiveFoldPlan + PlacementTransitionPlan
    -> CollectiveCompletionTaskDataflow
    -> system-visible Event Tensor completion
    -> JAX named-axis collective
    -> complete value consumed downstream
```

The physical boundary is intentionally generic. It knows the Fold reducer,
dtype, numerical policy, replica groups, and device placement. It does not know
about Grug, MoE, attention, gradients, or parameter names.

## Ownership boundary

Shuttle owns:

- recovery of the generic Fold and placement transition;
- explicit `allow_rounding_reorder` policy;
- mapping recovered global device IDs onto JAX logical axis indices;
- Event Tensor notification counts and system-scoped release/acquire contract;
- rejection of unsupported reducers or mismatched device domains.

JAX/XLA owns:

- differentiation of the collective;
- selection and lowering of the physical collective implementation;
- stream ordering and communication launch;
- data-result readiness for downstream consumers.

No custom VJP, FFI communication kernel, semantic combine, Torch runtime, or
model-specific dispatch is present. The Event Tensor does not become a global
counter. Its completion is erased into the all-reduce result dependency because
the downstream consumer cannot execute until that result exists. This is a
device data-dependence guarantee, not a host-observed completion event.

## Current physical choice

This checkpoint uses one whole-value collective. It therefore coarsens any
potential per-tile completion into a single completion point. That may add false
dependencies, but it cannot omit a required dependency. Fine-grained collective
streaming remains a later scheduling candidate.

The current adapter accepts only replica groups expressed as global device IDs
and requires an explicit global-device-ID to JAX-axis-index mapping. It rejects
local replica-ID semantics because their process-local interpretation has not
yet been proven.

The current JAX version provides direct named-axis primitives for sum, maximum,
and minimum. Product is rejected rather than silently lowered to a different
algorithm. A bitwise-fixed reduction tree is also rejected because XLA may
choose or reorder the reduction tree.

## Evidence

The four-device virtual-CPU replay is preserved under
`benchmarks/artifacts/jax_collective_completion_cpu_v0`.

- full-group sum: zero forward error;
- two-group maximum mutation: zero forward error;
- JAX-generated sum gradient: zero error;
- deterministic repeated output;
- one forward StableHLO all-reduce and no custom call;
- two all-reduces in the differentiated StableHLO;
- Event Tensor count four and system-scoped release/acquire visibility.

This proves executable semantic and AD linkage, not communication performance.
GPU and multi-host execution have not been validated. The next useful step is a
minimal multi-GPU replay of the same API, followed by wiring the boundary into a
natural differentiated Grug weight-gradient region while leaving XLA transport
selection intact.
