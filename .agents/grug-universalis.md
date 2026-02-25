# Grug Universalis Plan (De-Haliaxify Levanter)

## Goal
Move Levanter from Haliax-centered abstractions (`NamedArray`, `Axis`, `axis_mapping`, `hax.nn.*`) to an explicit JAX + Equinox model:
- Equinox modules with `init(...)` / `__call__(...)` as the dominant pattern.
- Plain `jax.Array` tensors with jaxtyping annotations.
- Explicit mesh + explicit partitioning (`PartitionSpec` trees or leaf `.sharding` introspection).
- No dependence on global axis-mapping context managers in hot paths.

This is a follow-on to `/Users/dlwh/src/marin/.agents/grug-native.md`: grug-native proves the style in one training path; this plan generalizes it across Levanter.

## Scope
In scope:
- Training, eval, inference, optimizer, checkpoint, and callback paths under `lib/levanter/src/levanter`.
- Converting model APIs and utilities away from `NamedArray` contracts.
- Replacing Haliax-specific Linear assumptions in LoRA/Muon code.
- Replacing/rewriting scan layer infrastructure so explicit-axis code remains compile/perf safe.
- HF export/import parity for migrated model families.

Out of scope (for now):
- Full rewrite of every historical experiment script.
- Big-bang migration; we want incremental, shippable PR slices.

## Desired End State
1. New/active code paths accept and return plain `jax.Array` pytrees.
2. Model classes are `eqx.Module` and expose:
   - `@staticmethod init(cfg, *, key, mesh, pspec)`
   - `def __call__(self, ...) -> ...`
3. Sharding is explicit at module boundaries:
   - passed as `PartitionSpec` trees, or
   - inferred from incoming array `.sharding` when safe.
4. Most code type-annotates tensor shape/dtype via jaxtyping.
5. Haliax usage is either deleted or isolated in compatibility adapters.

## Migration Principles
- Prefer direct `jax.Array` operations (`einsum`, `take`, `where`, `vmap`, `scan`) over wrappers.
- Prefer explicit sharding metadata over implicit axis registries.
- Keep module APIs boring and copy-pastable.
- No compatibility shims unless required for a short transition window; update call sites.
- Preserve behavior first, then simplify internals.

## Coupling Map (What Must Change)
Primary Haliax coupling zones today:
1. `lib/levanter/src/levanter/models/*`
   - `LmHeadModel` and model implementations with `NamedArray` signatures.
2. `lib/levanter/src/levanter/trainer.py` and runtime helpers
   - `axis_mapping`, `named_jit`, NamedArray-aware axis discovery.
3. `lib/levanter/src/levanter/eval.py`
   - part-migrated; still has named-interop and axis mapping hooks.
4. `lib/levanter/src/levanter/optim/*`
   - Muon/AdamH/AdamMini masks and transforms key off `haliax.nn.Linear` and scan-aware utilities.
5. `lib/levanter/src/levanter/inference/*`
   - heavy `NamedArray` + `haxtyping` usage in scheduler/page-table paths.
6. Config/serialization glue
   - config codecs and utilities reference Haliax enums/types.

## Architecture Contracts to Introduce Early

### 1) Module Contract (Eqx First)
```python
import equinox as eqx
import jax
from jaxtyping import Array, Float, Int

class CausalLM(eqx.Module):
    token_embed: Float[Array, "vocab hidden"]
    blocks: tuple["TransformerBlock", ...]
    lm_head: Float[Array, "hidden vocab"]

    @staticmethod
    def init(cfg: "ModelConfig", *, key: jax.Array, init_pspec: "PspecTree") -> "CausalLM":
        ...

    def __call__(
        self,
        token_ids: Int[Array, "batch seq"],
        *,
        attn_mask: Int[Array, "batch seq"] | None,
    ) -> Float[Array, "batch seq vocab"]:
        ...
```

### 2) Sharding Contract (No Global Axis Mapping)
```python
from jax.sharding import PartitionSpec as P

@dataclass(frozen=True)
class TrainShardings:
    params: "PspecTree"
    opt_state: "PspecTree"
    batch: P
    logits: P
```

- `train_step(...)` receives shardings explicitly, or derives leaf constraints from inputs.
- Any previous `axis_mapping` lookup moves to config/build time, not per-step logic.

### 3) Tensor Contract (jaxtyping + Plain Arrays)
- Batches/loss fns/eval fns return plain arrays.
- Use conventions (`batch`, `seq`, `hidden`, `vocab`) instead of named axes for most code.

## Specific Problem Areas You Called Out

### Equinox modules and `init/__call__`
- Standardize all active model families on Eqx module classes.
- Move free-floating param dataclasses into Eqx modules where composition benefits clarity.
- Keep simple pure helper functions for math kernels.

### Explicit meshes + jaxtyping
- Mesh setup lives in one runtime utility.
- Every compiled boundary receives shardings/config explicitly.
- `jaxtyping` annotations become required for public model/loss/train entrypoints.

### Replace axis_mapping magic
Replacement order:
1. Convention-based defaults (e.g. `batch -> data`, `hidden/vocab -> model` where applicable).
2. Explicit `PartitionSpec` trees passed through config/runtime.
3. Opportunistic `.sharding` introspection only when inputs are authoritative.

### LoRA and Muon (currently Linear-specific)
Introduce an explicit marker base class so code does not depend on `haliax.nn.Linear` concrete type.

```python
import equinox as eqx
import jax
from typing import ClassVar, Literal

class LinearLikeModule(eqx.Module):
    __levanter_linear_like__: ClassVar[Literal[True]] = True
    weight: jax.Array
    bias: jax.Array | None

def is_linear_like(x: object) -> bool:
    return isinstance(x, LinearLikeModule)
```

Then:
- LoRA: adapters wrap `LinearLikeModule` leaves or explicit param paths.
- Muon/AdamH masks: classify by `is_linear_like(...)` + path metadata, not raw `weight/bias` structure or Haliax class identity.
- Keep one compatibility adapter for legacy Haliax models during transition.

### Scan layers vs stacks
Safer plan: ship an explicit-axis-friendly stack abstraction before replacing all scan usage.
- Add `ModuleStack` (tuple/list of blocks, optional `lax.scan` execution mode).
- Keep two execution modes behind one API:
  - eager-for loop (debug simplicity)
  - `lax.scan` path (compile/perf)
- This avoids forcing immediate loop rewrites in each model.

## What Else Is Missing (Added Workstreams)
1. Inference migration
   - `inference/jit_scheduler.py` and `inference/page_table.py` are strongly NamedArray-centric.
2. Optimizer utility migration
   - `optim/util.py` scan-aware flatten/unflatten helpers assume Haliax linear/scanned layers.
3. Config codec cleanup
   - remove Haliax-specific decode/encode registrations and enums in config paths.
4. Checkpoint/state-tree schema
   - define stable plain-array trees for params/opt/fp8/ema so resume stays deterministic.
5. Callback boundaries
   - ensure `watch.py`, trainer hooks, and metric collectors operate on generic pytrees/arrays.
6. Type and lint policy
   - add pyrefly/ruff expectations for jaxtyping usage in new de-haliaxified modules.
7. Performance guardrails
   - add microbenchmarks or smoke throughput checks so refactor doesn’t silently regress compile/runtime.
8. Distributed ergonomics
   - formalize when we use explicit sharding constraints vs relying on input shardings.
9. Future FP8 path
   - keep room for ref-based mutable FP8 state (JAX array refs) but do not block initial migration.
10. HF export/import compatibility
   - replace Haliax-dependent export surfaces with array-native/Eqx-aware converters and parity tests.

## Phased Plan

## Phase 0: Foundation Contracts (Precursor-Friendly)
Deliverables:
1. Shared types module for `PspecTree`, `ShardingTree`, and tensor aliases.
2. `LinearLikeModule` marker class + path-based parameter classification helper.
3. `ModuleStack` interface with eager and scan execution backends.

Acceptance:
- New utilities land without behavior changes in existing trainer paths.
- Unit tests for classification and stack semantics.

## Phase 1: Model Interface Cutover
Deliverables:
1. Eqx `init/__call__` implementations for active LM models.
2. Public forward/loss APIs return plain arrays.
3. Legacy Haliax model wrappers become compatibility-only.

Acceptance:
- Model forward/loss parity tests pass on identical checkpoints/inputs.
- New code paths have no `NamedArray` in signatures.

## Phase 2: Trainer/Eval Runtime Cutover
Deliverables:
1. Explicit-mesh runtime utilities used by trainer and grug-native paths.
2. `train_step`/`eval_step` contracts use explicit `PartitionSpec` inputs.
3. `eval.py` completes migration: core accum/einsum paths array-native, no axis mapping in batch accumulation.

Acceptance:
- E2E train/eval loop works with array-native model and checkpoints.
- No `hax.axis_mapping(...)` in hot training/eval loop.

## Phase 3: Optimizer + PEFT Cutover
Deliverables:
1. Muon/AdamH/AdamMini masks rewritten to avoid Haliax class checks.
2. LoRA adapters target `LinearLikeModule`/path metadata.
3. Scan-aware optimizer helpers updated for `ModuleStack` representation.

Acceptance:
- Muon + LoRA smoke tests pass on at least one migrated model.
- No dependency on `haliax.nn.Linear` for active optimizer/adapter paths.

## Phase 4: Inference Cutover
Deliverables:
1. `jit_scheduler` and `page_table` rewritten to plain arrays + jaxtyping.
2. Decode state and queue ops run without NamedArray contracts.

Acceptance:
- Inference tests/smokes pass with migrated model.
- No new Haliax imports under `levanter/inference`.

## Phase 5: Cleanup + Deletion
Deliverables:
1. Remove dead Haliax-only adapters and fallback code.
2. Tighten docs/recipes to Eqx + explicit-sharding default guidance.
3. Keep optional compatibility layer only where still needed for external users.

Acceptance:
- Haliax usage is either gone or isolated to explicit compatibility modules.
- CI coverage includes migrated train/eval/inference/optimizer paths.

## Phase 6: HF Export/Import Cutover
Deliverables:
1. Array-native HF serialization/deserialization path for migrated Eqx modules.
2. Config/tokenizer/checkpoint metadata mapping documented and tested.
3. Conversion utilities no longer require Haliax model wrappers.

Acceptance:
- Round-trip test passes: Levanter -> HF -> Levanter for at least one migrated LM family.
- HF load + generation smoke test passes with exported artifacts.
- Export path works from both trainer checkpoints and standalone model weights.

## Precursor PR Slices (Recommended)
1. `ModuleStack` + tests.
2. `LinearLikeModule` marker class + optimizer mask helper migration.
3. Explicit sharding runtime helper module (mesh + pspec tree plumbing).
4. Eqx model API conversion for one target model + parity tests.
5. Eval accumulation cleanup (finish array-native tags/loss contracts).
6. Inference conversion slice.
7. HF export/import slice for the first migrated model family.

These can land independently and reduce risk before the full trainer consolidation.

## Open Design Decisions to Set Early
1. Parameter metadata strategy for optimizer/LoRA routing:
   - path regexes vs typed wrapper leaves vs explicit tags.
2. Default stack execution mode:
   - eager loop for readability vs `lax.scan` for compile-time from day 1.
3. Sharding source of truth:
   - config pspec tree vs inferred-from-array vs hybrid.
4. FP8 state placement:
   - params/module-attached state vs external trainer state tree, with future option for refs.

## Suggested First Milestone After This Plan
Land Phase 0 fully (shared contracts + `ModuleStack` + `LinearLikeModule`) and switch one model family end-to-end through train+eval with no `NamedArray` in public signatures. That gives a concrete template for the rest of Levanter and makes later deletions mostly mechanical.
