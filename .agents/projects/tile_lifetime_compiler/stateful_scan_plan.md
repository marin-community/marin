# StatefulScan Prototype Plan

Date: 2026-08-07

## Checkpoint

The routed sparse-attention prototype is frozen at commit
`fae336fd48143fb70a9be3257ac45223a710d675`. StatefulScan work begins on
`research/shuttle-stateful-scan`.

## Phase 0: Account for the sparse-attention gap

Explain the 16K Seer query-major baseline versus generated KV-major slot waves:

```text
Seer query-major:       2.388 ms
Shuttle KV-major:      4.017 ms
ratio:                 1.682x
```

Use source inspection and a focused profile to separate:

- online state residency versus FP32 HBM state round trips;
- fused kernel versus one launch per selected slot;
- K/V staging and reuse;
- Q and relation-metadata traffic;
- useful QK/PV work versus padding/masked work;
- occupancy and synchronization effects.

Do not perform another tile-size search. Only pursue another sparse-attention
implementation if it tests non-monotone relations plus real cluster/shared-KV
staging.

## Phase 1: Recover the ordered recurrence

Add dependency-free semantic records for:

- persistent state shape and dtype;
- ordered scan axis;
- state initialization;
- named update and read functions;
- numerical contract;
- optional chunk algebra.

Build a GDN program record without adding `GatedDeltaNetOp` to the semantic IR.
Validate dimensions and the relationship among Q, K, V, decay, update gate,
state, and output.

Exit condition: a readable dump describes the source recurrence and rejects
incompatible shapes or precision contracts.

Status: implemented for scalar-decay GDN and per-channel-decay KDA. Both use
the same generic `StatefulScan`; neither adds an architecture-specific semantic
node.

Frontend finding: ordinary JAX `lax.scan` exports as `stablehlo.while` plus
private called recurrence bodies. Shuttle's current flat StableHLO importer
does not support those regions. Add narrow structured-control-flow import after
the backend experiment rather than hiding the scan behind static unrolling.

## Phase 2: Execute the recurrent form

Implement a simple NumPy source-order executor for the GDN update. Test it
against an independent direct loop and, at a backend boundary, the existing
Levanter/Hugging Face recurrent implementation.

Required cases:

- zero and nonzero initial state;
- one and multiple tokens;
- rectangular key/value state;
- continuation across a prefix/suffix boundary;
- explicit FP32 update state with BF16 boundary simulation;
- extreme gates.

Exit condition: recurrent output and final state agree with independent
references, and the plan identifies this candidate as the decode form.

Status: dependency-free NumPy recurrent references pass source-order tests for
both GDN and KDA, including nonzero persistent state.

## Phase 3: Represent chunk algebra

Start with the weakest useful generic contract:

```text
summary = summarize(chunk)
new_state = apply(summary, incoming_state)
outputs = emit_outputs(summary, incoming_state)
```

Represent an optional associative `compose` only if a compact closed summary is
demonstrated. The initial GDN chunkwise algorithm may use parallel in-chunk
triangular work followed by an ordered scan over chunk transforms.

Exit condition: the chunkwise candidate is visibly a lowering of the same
ordered scan and matches recurrent execution for several chunk sizes.

Status: exact full-affine `(P,H)` summaries and ordered composition are
implemented. GDN and KDA chunk executions match their recurrent references for
tail chunks and several chunk sizes. Expert factored backends remain to be
measured.

## Phase 4: Physical backend and measurements

Pin the official GDN/KDA reference source and use the smallest backend adapter
that exposes recurrent and chunkwise entry points. Benchmark:

- one-token recurrent decode over representative batch/head/state shapes;
- long prefill with a bounded chunk-size candidate set;
- compilation and warm steady-state execution separately;
- output/final-state deviation and determinism;
- state, summary, and temporary bytes.

The candidate set is deliberately small and must be saved before selection.

Exit condition: the report separates compiler-plan overhead and materialization
choices from backend-kernel quality.

Status: oracle measurement complete; recurrent Shuttle backend executes. Pinned FLA
recurrent and chunkwise kernels execute on H100, both match the independent
recurrence, and matched measurements find recurrent faster through length 256
but chunkwise 7.72 times faster at length 2048. These calls define targets and
do not count as generated implementations. FlashQLA's API installs, but its JIT
is blocked by the holder's incomplete split CUDA toolkit; the exact failure is
preserved. Generic affine recovery now instantiates a compiler-owned recurrent
Triton skeleton that executes scalar/per-key diagonal and rank-1/rank-2 updates
without oracle packages installed. The ordered factored-chunk backend remains
incomplete.

## Phase 5: KDA fit test

Map the Kimi Delta Attention recurrence and chunk algorithm onto the same
records. Prefer a paper/source-level fit analysis plus one small semantic
fixture before adding another optimized backend.

Record whether KDA requires:

- only a different structured update/read function;
- a richer chunk summary;
- a factored or low-rank state transition;
- multiple coupled scans; or
- a genuinely new generic semantic primitive.

Exit condition: there is an evidence-backed accept/reject decision on the
current `StatefulScan` abstraction.

Status: accepted for the matrix-state core. Per-channel KDA uses the same
semantic record, exact affine composition, executor, and physical candidate
types. It requires diagonal-plus-low-rank physical factors, but no new semantic
primitive. Full-layer short-convolution caches and a compiler-owned factored
KDA backend remain outside this slice.

## Conclusion

The prototype validates the semantic abstraction, one generated recurrent
physical skeleton, and a shape-dependent execution-form choice, with three
explicit boundaries:

- structured StableHLO control-flow import is not implemented;
- generic state-affinity/factor recovery is not yet connected to the importer;
- Shuttle has generated the recurrent core but not the ordered factored-chunk
  kernel or the complete producer/finalizer region.

Before FSDP, Shuttle must generate recurrent and ordered factored-scan kernels
from the generic recovered transition. The mutation suite must demonstrate
that scalar/per-channel diagonal choices, alternative gate formulas, and
bounded update ranks reuse the same skeleton.
