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
- [x] Run the exact four-GB200 Sonic clone-gradient parity probe.
- [ ] Reproduce the 64-GPU v153 control before interpreting treatment MFU.

## First rack control failure

The first latest-main control,
`ep30-cx-main-control-20260727-0534`, initialized 64 processes and compiled the
train step, then failed before step 0. The first crashing rank segfaulted in
NCCL `ncclDevCommCreate` through XLA `NcclDeviceCommunicator`; the remaining
ranks were terminated by the coordinated failure. Iris recorded zero retries,
and W&B recorded no history rows.

The successful v165 control used JAX and JAXlib 0.10.1. Latest main resolves
GPU jobs to 0.11.0, whose experimental ragged-all-to-all NCCL barrier path is
enabled by default and requests the device communicator seen in the stack.
Keep native ragged-all-to-all semantics while disabling that experimental
barrier path explicitly:

```text
--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false
```

The flag parses under the local JAX 0.11.0 runtime. A rack retry is still
required to prove that it avoids the communicator crash.
