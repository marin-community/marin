# Debugging log for Shuttle seven-call Fold audit

Run the compiler-generated routed, attention-backward, and row-Fold replacements from a natural Grug HLO module on a verified H100, while preserving fail-closed structural auditing.

## Initial status

The exact canonical revision `ce24d55ca5` compiled all routed and row-Fold handlers and successfully generated the compiler-owned Triton attention AOT cache. Before JAX executable compilation or handler execution, the structural audit failed in `audit_routed_training_attention_and_axis_fold_replacement` because replaying the replacements from the original HLO did not compare equal to the transformed round-tripped HLO module. No correctness or timing result was produced.

The failure was isolated to reconstruction/audit behavior. The H100 allocation was explicitly released and verified inactive before offline diagnosis.

## Hypothesis 1

The audit compares complete parsed HLO modules after an XLA round trip, including computations or canonicalized attributes outside the replaced entry-region dataflow. The replacement may still cover the exact required consumers even when whole-module dataclass equality fails.

## Changes to make

Preserve the original, transformed, and reconstructed HLO modules and produce a structured difference. Determine whether the mismatch is:

- a harmless round-trip canonicalization outside the replacement boundary;
- name or attribute normalization on the generated custom calls;
- an uncovered consumer or stale instruction in either row-Fold region.

Replace whole-module equality only if a narrower audit can prove exact entry-region coverage, generated-call signatures, root liveness, and absence of the replaced instructions without weakening the fail-closed boundary.

## Results

Exact revision `ce24d55ca5` reached successful compiler-owned Triton AOT generation for forward, dQ, and dK/dV before the audit stopped executable compilation. Offline reproduction found no coverage or liveness gap. XLA had:

- reordered computations;
- renumbered volatile `stack_frame_id` metadata;
- canonicalized `-0` to `0`;
- inserted operand-index comments;
- removed one dead transpose.

The audit now checks each actual post-roundtrip Fold call directly: unique target, output/operand/layout/API signature, adapter provenance, old-region liveness, and exact external-consumer rewiring. Copy and transpose counts may decrease through dead-code elimination but may not increase. Regression coverage includes the observed canonicalizations and fail-closed target, signature, consumer, liveness, and added-layout-operation cases.

The offline fix is commit `d25dbcd0a3` on the canonical branch. Forty-two focused axis-Fold, streaming-attention-backward, and shared-Contract/multi-Map tests pass after integration.

## Future work

- [x] Add regression coverage for the exact round-tripped HLO mismatch.
- [ ] Rerun the bounded H100 replay only after the offline audit test passes.
