# EP64 latest-main integration

Integrate the sealed BF16 EP64 handoff with the current JAX and dependency
floors before launching full-rack experiments.

## Initial status

The handoff tag was based 104 commits past an older merge base. A squash onto
`origin/main` produced only dependency-lock conflicts, but the first
latest-main compatibility run failed six of 221 targeted tests.

## Hypothesis 1

The failures are semantic drift exposed by the latest explicit-sharding rules
and a stale optimizer-mask expectation, rather than failures in the
receiver-ECHO kernels.

## Changes to make

- Keep the current exact CUTLASS DSL pin and regenerate the lock with QuACK
  0.6.1.
- Give the base Grug debug mesh the standard size-one expert axis required by
  shared Grug shardings.
- Construct next-token labels without concatenating differently sharded
  operands.
- Pass the MoE model configuration and expert-axis size through lowering
  contract tests.
- Keep routers and normalization scales on AdamW; reserve Muon for matrix
  weights.

## Results

- Initial targeted run: 167 passed, 48 skipped, and 6 failed.
- Corrected targeted run: 173 passed and 48 skipped. The two warnings are the
  expected Pallas TPU fallback probe and the CPU test mesh lacking a resource
  mapping.

## Future work

- [x] Pass the targeted compatibility suite.
- [ ] Run the exact four-GB200 Sonic clone-gradient parity probe.
- [ ] Reproduce the 64-GPU v153 control before interpreting treatment MFU.
