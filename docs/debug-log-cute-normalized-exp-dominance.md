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
tests: 21 targeted tests pass. The first H100 replay falsified the finalize-only
version: the same child-region dominance failure remained.

## Hypothesis 2

The non-dominating value is produced earlier by `update`, which wrote register
state through repeated dataclass-field lookup inside its row child region. The
upstream helper also binds state to locals before the update loop. Apply the
same explicit SSA discipline to both update and finalize, then replay.

The second H100 replay confirms this hypothesis. Commit `ad1a0c3192` compiles
and runs on an H100 after carrying the state returned by each update iteration
into the next iteration and finalization. Two counterbalanced 10-sample
captures measured 0.080272 ms for the pre-Event source and 0.080352 ms with
the derived Event Tensor attachment, a ratio of 1.000997. Both paths produced
the same deterministic output hash and maximum sampled error of 0.015625.

The failed finalize-only replay and the successful update/finalize replay are
preserved under `lib/tile_lifetime/benchmarks/artifacts/` as
`event_tensor_sm90_fold_alias_replay_h100_v1` and
`event_tensor_sm90_fold_state_replay_h100_v1`.

## Future work

- [ ] Determine whether other `ParamsBase` helpers directly access tensor fields
  across CuTe child regions.
