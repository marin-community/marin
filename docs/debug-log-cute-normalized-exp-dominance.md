# Debugging log for CuTe normalized-exponential dominance

Repair the compiler verification failure blocking the SM90 Event Tensor
streaming-attention replay while preserving generic Fold semantics.

## Initial status

Both the canonical and Event Tensor H100 sources fail before code generation in
`NormalizedExpFoldState.finalize`. CuTe reports that the register tensor used by
`cute.get_layout` does not dominate the use created by
`self.row_sum.store(utils.warp_reduce(self.row_sum.load(), ...))`.

## Hypothesis 1

Repeated field access through the `ParamsBase` dataclass causes CuTe to rematerialize
the register tensor in a child region. The upstream generic softmax helper first
binds `row_sum`, `row_max`, and `scale_log2` to local SSA values and then performs
the reduction and loop through those aliases.

## Changes to make

Match that backend-neutral SSA discipline in Shuttle's generic normalized-exp
Fold helper without changing its state, update, merge, or finalization algebra.
Add a static regression test that preserves the local-alias boundary, then replay
the existing H100 compile command before making any performance claim.

## Results

The local alias change preserves all normalized-exp and Event Tensor semantic
tests: 21 targeted tests pass. An actual H100 replay is in progress; no compile
or performance claim is made yet.

## Future work

- [ ] Determine whether other `ParamsBase` helpers directly access tensor fields
  across CuTe child regions.
