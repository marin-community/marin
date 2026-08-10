# Debugging log for Shuttle JAX selected-region membership

Determine whether the `jaxacceptance3` selected-region mismatch came from a
stale acceptance oracle or from Shuttle changing the ordinary JAX program.
The investigation is source-only and does not authorize another Iris job.

## Initial status

The CPU preflight from canonical
`ba31d47e354746543bdc179277071de25c48eaed` passed every native, lit, XLA,
wheel, and install gate. Invocation 2 in the cache-disabled acceptance worker
then reported three successful observer phases under `source_ordered` policy.
The selected-region value had length 373 and SHA-256
`cc8a395018a9866e14fa882137009717ca604c13ce48ccc6044bac5d5a154df3`.
Both audited contracts rejected that value. The VJP oracle also has length 373,
but its SHA-256 is
`f40e841a76f6654fafba97fce274ee1eb6abda9b244b171c46adc6cf4619abec`.

## Hypothesis 1

The acceptance oracle describes the raw ordinary-JAX StableHLO export, while
the XLA hook runs after XLA's StableHLO preprocessing. A preprocessing pass may
change source-operation order without changing program semantics.

The observed digest has one exact preimage among ordered three-group partitions
of 11 selected source ordinals from the 14-operation VJP fixture:

```text
[[1, 2], [5, 6], [7, 8, 9, 10, 11, 12, 13]]
```

The checked-in oracle is:

```text
[[0, 1], [5, 6], [7, 8, 9, 10, 11, 12, 13]]
```

Running the pinned jaxlib StableHLO complex-math expander on the audited VJP
fixture moves its scalar constant from source ordinal 2 to ordinal 0. The dot
and tanh become ordinals 1 and 2; all operations from ordinal 5 onward remain
in place. This produces the exact selected groups recovered from the native
diagnostic. The pass is in `MlirToXlaComputation` before the Shuttle registry
callback.

## Changes to make

Audit the full hook-boundary preprocessing sequence against both fixtures.
Extend the fixture audit to derive membership, coverage, unsupported-island,
and final-normalized expectations from that boundary instead of the raw export.
The acceptance contract must keep exact values and reject any structural drift.

## Results

The mismatch is stale acceptance evidence, not Shuttle semantic drift. Applying
the pinned complex-math expander gives selected regions
`[[1, 2], [5, 6], [7, 8, 9, 10, 11, 12, 13]]`. Building the expected manifest
from those hook-boundary ordinals reproduces every bounded native diagnostic:

- selected-region SHA-256
  `cc8a395018a9866e14fa882137009717ca604c13ce48ccc6044bac5d5a154df3`;
- coverage-manifest SHA-256
  `a626b0c7df55bf83154a5697de85eb3c99989bff0cc77b8f3bf469310db0238d`;
- unsupported-island SHA-256
  `1a9aad82650111cbc134fcc17d1afcb051f9ae729f6cdfd48105d1e8dc210201`;
- final normalized module SHA-256
  `d4dad86c0c4abf2f4a98bdd19879cbfb789c8d6cba8b18fa56decc4589a8ddb5`.

The forward fixture is unchanged. The fixture audit now rederives the
hook-boundary digest through pinned JAX/jaxlib 0.10.1, and the acceptance oracle
uses the exact hook-boundary VJP structure. It still rejects old raw-export
ordinals and any field-level structural drift.

The generator emits hook-boundary metadata only for the forward and VJP
fixtures consumed by this acceptance contract. Its default no-write audit still
checks all six ordinary-JAX fixtures. The regression builds the complete old
VJP contract, including its selected ordinals, excluded dependency references,
unsupported fingerprint, and final raw-export fingerprint, and proves that the
hook-boundary validator rejects it.

A local exact-pin Bazel compile probe could not validate the updated native
path because the pinned Darwin toolchain requests unavailable SDK
`macosx10.11`. No Iris rerun was launched.

## Future work

- [ ] Run the full CPU acceptance job after independent source review and
      canonical integration.
