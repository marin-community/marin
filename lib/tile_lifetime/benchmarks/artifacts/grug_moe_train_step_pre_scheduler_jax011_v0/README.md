# Grug train-step PRE_SCHEDULER capture

This artifact records the exact `HloModuleProto` received by a no-op JAX HLO
module transformation while compiling the natural one-layer Grug MoE train
step. The frontend uses reference tensor-algebra attention, scatter MoE, XLA
ragged contractions, ordinary `value_and_grad`, and an SGD update.

## Result

JAX invoked the `PRE_SCHEDULER` callback once for `jit_train_step`. The captured
module contains 82 `Dot`, 67 `Reduce`, 8 `Scatter`, and 4 `Sort` instructions.
The frontend StableHLO contains no custom calls. The captured HLO contains one
`TopK` custom call introduced by XLA; it is a generic XLA lowering, not an
expert attention, MoE, or recurrent kernel.

The callback returned `None`, so this run only proves the inspection/no-op
insertion point. It does not rewrite the module.

A subsequent metadata-independent structural pass inlines ordinary fusion
bodies and recovers two shared-input Contract pairs feeding scalar Maps and a
downstream Contract. The pass sees 82 Contracts across 2,913 inlined logical
nodes. The recovered Maps retain every BF16 conversion boundary; this report
does not claim that collapsing those conversions is source-order legal.

The routed-program pass also recovers the main sparse forward/backward boundary
without consulting HLO metadata or source names:

- two equivalent runtime Relations (the executed path and its rematerialized
  backward path), each with 8 source rows, 2 slots, 16 edges, 4 destination
  segments, a stable destination permutation, destination counts, and prefix
  offsets;
- one executed and one rematerialized segmented Contract -> pair-Map ->
  segmented Contract forward chain;
- one segmented input-gradient Contract -> pointwise Map-adjoint -> segmented
  Contract chain;
- two source-keyed additive scatter Folds, including the forward weighted
  contribution and the reducer's BF16 round trip; and
- two group-batched weight-gradient Contracts with the following all-reduce
  left as an explicit placement boundary.

This identifies what a Shuttle-owned training region must replace while
allowing communication to remain external. It is not yet an executable GPU
replacement. In particular, the scalar Map is preserved as an opcode and cast
program; importing it into the shared scalar AST generator remains the next
code-generation step. The captured XLA implementation pads 16 logical edges to
a physical 512-row Contract domain. That 32-times amplification is an artifact
of this tiny CPU fixture, but it makes the required segmented-GMM replacement
boundary concrete. The HLO scatter does not establish a source-ordered GPU
merge; Shuttle must select and generate that numerical policy explicitly.

## Pinned environment

- JAX: `0.11.0`
- JAXLIB: `0.11.0`
- JAX release revision: `a1521744c6dc074443fe549f19f48d7197abf759`
- Backend: single-device CPU
- Compilation cache: disabled
- Marin source revision before this probe: `6d34fbab6f6b3c1d31c9d4fe672b143fbebf1b59`

`jax011_probe_uv.toml` overrides Marin's JAX constraints only in the isolated
probe environment. Marin's main lock remains on JAX 0.10.1.

## Reproduction

From the repository root:

```bash
uv run \
  --config-file lib/tile_lifetime/benchmarks/jax011_probe_uv.toml \
  --isolated \
  --package marin-core \
  --extra cpu \
  --with 'jax==0.11.0' \
  --with 'jaxlib==0.11.0' \
  python lib/tile_lifetime/benchmarks/grug_moe_train_step_hlo.py \
  --stablehlo-output \
    lib/tile_lifetime/benchmarks/artifacts/grug_moe_train_step_pre_scheduler_jax011_v0/frontend-stablehlo.mlir \
  --summary-output \
    lib/tile_lifetime/benchmarks/artifacts/grug_moe_train_step_pre_scheduler_jax011_v0/frontend-summary.json \
  --pre-scheduler-artifact-directory \
    lib/tile_lifetime/benchmarks/artifacts/grug_moe_train_step_pre_scheduler_jax011_v0
```

The smaller self-contained API smoke uses the PEP 723 pins embedded in the
probe script:

```bash
uv run --script lib/tile_lifetime/benchmarks/xla_pre_scheduler_probe.py \
  --artifact-directory /tmp/shuttle-xla-smoke
```

## Files

- `frontend-stablehlo.mlir`: natural JAX 0.11.0 train-step StableHLO.
- `frontend-summary.json`: frontend operation and custom-call census.
- `pre-scheduler-hlo.pb`: serialized callback input.
- `pre-scheduler-hlo.txt.gz`: deterministic gzip of the callback input text.
- `pair-map-recovery.json`: generic Contract/Map recovery report from the frozen
  callback HLO.
- `relation-program-recovery.json`: generic RelationPlan, segmented Contract,
  Map, Fold, and Contract-adjoint ownership report from the same frozen HLO.
- `summary.json`: callback stage, versions, hashes, and HLO census.

Regenerate the recovery report with:

```bash
uv run --frozen --package marin-tile-lifetime \
  python lib/tile_lifetime/benchmarks/analyze_xla_pair_map_hlo.py \
  lib/tile_lifetime/benchmarks/artifacts/grug_moe_train_step_pre_scheduler_jax011_v0/pre-scheduler-hlo.txt.gz \
  --output \
    lib/tile_lifetime/benchmarks/artifacts/grug_moe_train_step_pre_scheduler_jax011_v0/pair-map-recovery.json
```

Regenerate the routed-program report with:

```bash
uv run --frozen --package marin-tile-lifetime \
  python lib/tile_lifetime/benchmarks/analyze_xla_relation_program_hlo.py \
  lib/tile_lifetime/benchmarks/artifacts/grug_moe_train_step_pre_scheduler_jax011_v0/pre-scheduler-hlo.txt.gz \
  --output \
    lib/tile_lifetime/benchmarks/artifacts/grug_moe_train_step_pre_scheduler_jax011_v0/relation-program-recovery.json
```

## Limits

This CPU run has `num_partitions=1`. The API places the callback at the
pre-scheduler insertion point, but this artifact does not exercise a nontrivial
SPMD partition. A multi-GPU capture must verify that sharding preserves the
relation and contraction structure needed by Shuttle.

The raw XLA client cannot compile the frozen donated StableHLO with default
compile options because its alias layouts are unspecified. This harness captures
during the real `Lowered.compile()` call, which supplies JAX's layouts and alias
configuration. Replaying a frozen module will need the matching compile options
or a fixture without donation aliases.

The JAX HLO transformation API is experimental. The probe uses JAXLIB's
`_hlo.HloModule` binding for inspection and serialization; a durable plugin
should isolate that version-sensitive boundary. Inserting a generic Shuttle FFI
call and compiling on H100 remain untested.
