PASS_AFTER_BLOCKERS_RESOLVED

# CC review: saved-checkpoint gradient plot completion v3

Review date: 2026-08-22

Reviewer: `claude-opus-5[1m]` at maximum effort, invoked read-only through the
`plambdafour@proton.me` Claude subscription with `ANTHROPIC_API_KEY` removed.

## Verdict

The saved-checkpoint recovery is approved for staged execution. The review
confirmed that no training is launched, endpoint outcomes are not read, output
identities are release-bound, and every launch stage is blocked on exact audits
of all preceding stages.

## Blockers found and resolved

1. Iris excludes `.agents` by default, while the remote parent hash-checks this
   review. The exact review file must therefore be explicitly re-included in
   every launch command. Re-including the file path, rather than `.agents`, adds
   only this hash-pinned artifact.
2. The v10 full mechanism manifest was initially grouped with eight large
   visualization-only CSV inputs. It is now an execution-reference input and is
   hash-verified in every mode, including remote launch.
3. Authorization now independently verifies all eight materialization inputs
   before creating its sidecar. This remains true for direct programmatic calls,
   not only the CLI path.

## Verified trust boundary

- Remote launch skips hash checks only for six frozen v10 result tables, the
  all-state source-geometry table, and the multiplicity audit. No launch or TPU
  worker reads these eight visualization-only inputs.
- Release identity, implementation hashes, execution manifests, analysis
  contract, parent v6 release and design, v10 release and full mechanism
  manifest, coverage audit, and this review remain mandatory in every mode.
- Materialization re-verifies all eight visualization inputs before merging the
  recovered outputs and rendering plots.
- The exact historical `trainer.checkpointer.write` normalization is fail-closed
  and changes only the post-v6 throughput-only field covered by the full train
  configuration hash.
- v3 has a distinct result root, schema version, artifact version, release hash,
  authorization sidecar, table directory, and plot directory; it cannot collide
  with v10 or superseded completion releases.

## Residual observations

The authorization sidecar records that materialization inputs were verified,
but an operator with write access could manually reproduce that deterministic
payload. This does not permit unverified plot evidence into the final artifact:
remote execution does not consume those inputs and materialization verifies
them again. Launch commands must also explicitly exclude the visualization
directories; repository defaults alone do not enforce the intended small
workspace.

No remaining blocker was identified.
