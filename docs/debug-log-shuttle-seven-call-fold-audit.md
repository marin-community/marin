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
- [x] Rerun the bounded H100 replay only after the offline audit test passes.

## Hypothesis 2

The exact `6c718d8d4b` H100 replay passed the post-roundtrip audit and executed all seven handlers 35 times. Its final evidence guard then reported target occurrences `(1, 1, 1, 1, 1, 17, 17)`. The two axis-Fold targets are embedded in generated adapter and instruction names, so a substring count does not measure the number of custom calls.

The H100 was released and verified inactive. The run produced no accepted correctness or timing artifact because the terminal guard failed.

## Changes to make

Count exact parsed `custom_call_target` attributes before warmup and timing. Keep handler-count validation after execution. Reject missing and duplicate target attributes, and preserve partial evidence if a post-execution guard fails.

## Results

The offline fix is commit `49c572a6ec` on the canonical branch. It parses every
post-roundtrip custom call, requires exactly one `custom_call_target` attribute,
and verifies that each selected target occurs exactly once before any correctness,
warmup, or timed execution. Post-execution failures now preserve raw samples,
hashes, comparisons, and handler counts in an explicitly unaccepted artifact.

The regression reproduces the axis-Fold target fragments in adapter names while
proving their exact attribute count is one; missing and duplicate target
attributes fail closed. Forty-two focused streaming-attention-backward,
routed-training, and shared-Contract/multi-Map tests pass after integration.

The previous H100 run remains unaccepted because its raw timing samples existed
only in process memory. One bounded H100 replay is still required.

## Hypothesis 3

The exact `9798ebd794` replay passed structural auditing and ordered-FP
correctness, but three of 30 generated executions produced a second output hash.
The baseline hash was stable in all 30 executions. The current whole-tree hash
does not identify which of the 53 output leaves changed.

The H100 was released and verified inactive. The replay measured `0.528433 ms`
for XLA and `0.603667 ms` for the generated path, or `1.142374x`, but this is an
unaccepted diagnostic result because the determinism gate failed.

## Changes to make

Record a path, dtype, shape, and hash for every output leaf after each execution.
Preserve those hashes in both successful and unaccepted artifacts. Use the first
varying leaf to select the next component-level audit; do not rerun all components
independently before locating the affected output.

## Results

The replay artifact is preserved under
`lib/tile_lifetime/benchmarks/artifacts/xla_grug_shared_map_h100_unaccepted_v0/`.
It contains all 60 raw counterbalanced samples, original and transformed HLO,
generated CUDA, numerical comparison, and checksums. The benchmark now records
per-leaf hashes. A CPU regression verifies that mutating one leaf changes only
that leaf's hash.
