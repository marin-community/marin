# Datakit single-document packing exhausted host memory before training

## Summary

On 2026-07-27, both NEST-BURN-001 r2 arms stalled before model
initialization while constructing the canonical 10.37T-token Datakit mix.
Each worker exceeded 61 GB RSS because `pack=1` eagerly materialized one
Python `range` per source document. Both job trees were stopped before host
OOM; neither produced a training step.

## Impact

The matched d768 E256 and fixed25 runs were delayed. The failure happened
before any model update, so it produced no experimental evidence and did not
affect the comparison.

## Diagnosis

A live stack sample placed every worker in `pack_documents`.
`GreedyPrepackedDataset` loaded complete offset and length arrays and built an
index entry for every document even though `max_segments_per_example=1`
cannot combine documents. The canonical mix amplifies that unnecessary
metadata into tens of gigabytes of Python objects.

The top-level `with_pack(data, 1)` transformation also left children of
`ConcatDatasetComponent` unchanged. Those tail components would eventually
have supplied unpacked examples to THD attention.

## Resolution

`GreedyPrepackedDataset` now uses a lazy single-document index when
`max_segments_per_example=1`. Batch reads fetch only selected documents and
apply the original left/right truncation and padding semantics. `with_pack`
now applies the requested pack count to concatenated children as well as
direct components.

A regression fake fails if offsets are read and records every requested
document index. Dataset configuration coverage verifies direct and
concatenated components. All 29 focused tests and the relevant pre-commit
checks pass.

A first 16 GB CPU preflight instantiated all 168 training components but did
not perform an underlying row read. The first r3 sample then found that the
production `_ShardedJaggedArrayStore` lacked the `get_batch` method available
on a materialized `JaggedArrayStore`. Both matched jobs were stopped after
their first zero-step failure.

The sharded view now delegates selected-row reads to its owning `TreeCache`.
A production-shaped regression test covers noncontiguous ordering. A second
16 GB CPU preflight authenticated native GCS, instantiated all 168 components,
fetched a real sharded row, and emitted packed 8,192-token leaves:
`/power/nest-burn-001-datakit-sample-probe-r4b`.

## Follow-up

The matched experiment was relaunched under fresh r4 identities. Monitor host
RSS through first model initialization and keep the general multi-document
packing path unchanged.
