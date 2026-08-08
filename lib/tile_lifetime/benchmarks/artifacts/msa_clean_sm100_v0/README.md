# Clean MSA synthesis checkpoint

This artifact freezes the first clean MiniMax Sparse Attention (MSA) synthesis
comparison on one NVIDIA GB200. The generated path starts from BF16 hidden
inputs and uses generic Shuttle `Contract`, maximum `Fold`, `Selection`,
`RelationPlan`, normalized-exponential `Fold`, and output `Contract` machinery.
It does not call the public MSA score, attention, or combine entry points.

The physical score and selected-attention templates retain low-level CuTe
layout, copy, MMA, and pipeline machinery extracted from pinned MSA. Shuttle
owns the semantic score body, causal domain restriction, online state,
relation scheduling, and deterministic state merge. Candidate choice was kept
deliberately small: block size, query tile, partial-state dtype, and one of two
generic row-merge schedules. The rejected BF16x2 merge is preserved because it
regressed rather than being hidden.

## Result

Configuration: `Q=K=16384`, `Hq/Hkv=64/4`, `D=128`, block 128, top-k 16,
causal, BF16 payload, FP32 index projections and accumulation.

| Matched boundary | Shuttle | MSA oracle | Ratio |
|---|---:|---:|---:|
| Score Contract + block-max Fold + top-k | 0.637888 ms | 0.707600 ms | 0.9015x |
| Natural index projections + selection | 0.785760 ms | 0.837360 ms | 0.9384x |
| Natural projection + selection + selected payload | 4.431920 ms | 3.234160 ms | 1.37035x |

The full result does not meet the written `1.20x` completion gate. It is kept
as partial progress because closing the remaining gap with an MSA-specific
combine would weaken the clean-synthesis result. With the exact official
relation, the generic payload is 1.39992x the oracle; 1.83155 ms of its
3.70227-ms median is the generic deterministic merge.

The generated and isolated-oracle selectors produce the identical route hash
`5d669570...` and repeat bitwise. Both differ from a materialized Torch
reference in 61,446 slots across 7,681 rows. Of those rows, 7,680 are early
causal rows with fewer than top-k finite competitors; the remaining row has a
zero cutoff margin. This is recorded under the declared
`real_algebra_equivalent` routing policy. Consequently, the natural-program
output differs from the materialized reference by maximum `0.0536499` and mean
`6.8702e-5`, exceeding the existing `0.01` maximum-error threshold. The
exact-relation payload independently agrees with official MSA to maximum
`0.0009765625` and mean `3.75e-9` and is deterministic.

`invalid-pre-causal-fix.json` records a run that accidentally omitted the
causal `DomainRestriction`; it is invalid evidence and is retained to make the
correction auditable.

## Pins

- MSA: `80434d7f67877c6570ca19cac444b84bc9855dac`
- CUTLASS gitlink: `eb61c911471867a5fd2466bfd8f29306cea6ebf8`
- CUDA/NVCC: `13.0.88`
- Torch: `2.10.0+cu130`
- CUTLASS DSL: `4.4.1`
- QuACK: `0.2.10`
- Driver: `595.71.05`
- GPU: NVIDIA GB200, 1200-W policy, clocks unpinned

The generated and oracle runners execute in separate processes because loading
the direct instantiation and the identical private MSA variant in one process
causes CUDA error 719 in the second runner. The isolated route hashes establish
that both processes used the same seeded fixture and selected relation.

## Files

- `generated-natural-16k.json`: all generated-path raw samples and audits.
- `oracle-natural-16k.json`: isolated oracle raw samples and route diagnostics.
- `generated-route-diagnostic-16k.json`: isolated generated route hash.
- `exact-relation-payload.json`: paired payload correctness and timing.
- `rejected-bf16x2-merge.json`: negative generic merge candidate.
- `invalid-pre-causal-fix.json`: invalidated pre-fix run.
- `oracle_only_natural_16k_capture.py`: exact isolated oracle harness.
- `*-command.txt`: reproduction commands.

