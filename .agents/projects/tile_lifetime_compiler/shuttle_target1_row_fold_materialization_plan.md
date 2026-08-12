# Target 1 row-Fold materialization-plan boundary

Status: local compiler slice. This is not executable code generation and is not
an acceptance claim.

## Boundary

The native pipeline currently ends at semantic Shuttle algebra and lowers that
algebra back to StableHLO for XLA. It has no target layout, allocation offset or
alignment model, address spaces, worker-capacity model, EventTensor schedule,
device placement, GPU/NVVM lowering, executable replacement boundary, or
runtime ABI. Lowering directly to NVVM would therefore be an orphan backend:
there is no lossless contract that connects the generated code to XLA buffer
assignment or execution.

This slice adds an opt-in, backend-neutral `shuttle.materialization_plan` after
algebra conversion. Despite describing work that a later physical planner may
consume, it freezes only:

- exact Map/Fold task dependencies in source SSA order;
- explicit logical tensor materializations, including rank-zero tensors;
- live-in, live-out, producer, consumer, and task-ordinal lifetime intervals;
- static domains, reduction dimensions, accumulator policy, element types, and
  rename-invariant structural source bindings.

The plan does not choose layout, offsets, alignment, address space, target,
device, worker assignment, parallel schedule, or EventTensor coordinates. A
later derived scheduling view may consume these exact dependencies without
changing their meaning.

Both planning and verification passes are opt-in and are absent from the
observed production StableHLO pipeline. They do not change pipeline ABI 5,
option parsing, cache identity, observer output, or exported StableHLO.

## Closed first family

`shuttle-plan-row-fold-materialization` accepts exactly one eligible
`shuttle.region` in a module. The region must contain only a connected task
graph of Maps and at least one Fold. Each Fold contract is:

- one positive static rank-two FP32 input and one rank-zero FP32 initializer;
- one reduction dimension, either `[0]` or `[1]`, with the corresponding
  rank-one FP32 result and an FP32 accumulator;
- `order_free=true` and an exact scalar FP32 add combiner;
- every task result is consumed or returned by the region.

Maps retain their verified affine indexing semantics. Their result indexing
map derives the static task domain. A rank-zero Map is an explicit scalar task
with an empty domain and a rank-zero materialized output; scalar constants and
Fold initializers are never silently dropped. Repeated SSA operands remain
repeated task inputs, while dependency and consumer lists remain unique.

The pass rejects zero/dynamic extents, other reduction dimensions, multiple
candidate regions, disconnected or dead tasks, non-Map/Fold operations, and
unsupported types. Task and buffer identifiers are contiguous structural
integers, never symbols or workload identifiers.

## Verification and fingerprint

The dialect verifier recomputes producer/consumer ownership, dependencies,
lifetimes, storage class, row-Fold legality, and the plan fingerprint. The
source-binding verifier independently derives a fresh plan from the bound
algebra and compares:

- exact task order and input/output buffer ordinal vectors;
- exact buffer types, producers, consumers, and live-out uses;
- Map domains from affine maps and Fold domains/reduction policy;
- per-task semantic SHA-256 digests from operation types, attributes, scalar
  bodies, and operand/result structure;
- unique structural source references and exact one-region coverage.

Source operation names are used only while recomputing typed Map/Fold
semantics; they are not serialized in the plan or used as a replay key. Module,
function, target, and workload names are excluded. The plan fingerprint covers
the closed schema and changes with policy, shapes, task semantics, dependency
edges, materialization, or lifetime state.

## Behavior gates

The bounded local gates are:

- both frozen forward shapes (`2048x4096` and `7x13`) produce exactly 19 tasks
  and 21 buffers, including rank-zero scalar Maps; the frozen `7x13` backward
  and composed graphs produce 48 tasks/51 buffers and 51 tasks/54 buffers,
  respectively, including five axis-zero-or-one Folds apiece;
- symbol renaming preserves the plan fingerprint;
- changing the already-converted Region policy to `fast` is structurally valid
  and changes the plan fingerprint. This tests policy binding only; it is not a
  full fast-policy conversion claim;
- deleting or reordering tasks, self-consistently reordering independent tasks,
  replaying or duplicating source references, swapping same-typed SSA edges
  while recomputing dependencies/lifetimes, changing scalar/tensor domains,
  adding unknown attributes, or changing the Fold scalar body is rejected;
- other Fold axes, unsupported types, and multiple candidate regions are
  rejected;
- the existing CPU StableHLO round-trip parity suite remains the numerical
  gate because this opt-in analysis does not execute or replace the graph.

Static GPU-code inspection is not yet meaningful: there is no concrete target
layout, launch geometry, device ABI, or XLA buffer binding to inspect. The next
independently testable stage is a target-specific schedule/layout contract that
consumes this plan, followed by static device IR verification. Rebuilt jaxlib
and real GPU execution are required only after that consumer enters the
production transform boundary.
