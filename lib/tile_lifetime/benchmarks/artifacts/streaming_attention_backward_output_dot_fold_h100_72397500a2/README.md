# Query-owner Fold attachment on H100

This artifact measures Shuttle revision `72397500a24ae788215ca17f365cb7a3e063401c` on one batch-priority NVIDIA H100 80GB HBM3. The semantic input is ordinary JAX causal GQA differentiated with `jax.vjp`, serialized as StableHLO, and recovered into Shuttle's generic streaming reverse program.

The compiler represents `sum(output * output_cotangent, feature)` as an FP32 Map followed by a Fold. Generic owner-tile placement attaches that Fold to query-gradient preparation after proving both inputs are complete along the 128-element reduction axis. The dQ owner computes and stores the row scalar once; the dK/dV owner consumes it. There is no standalone output-dot launch.

For sequence length 2,048, 32 query heads, 8 KV heads, head dimension 128, 32x32 blocks, 8 warps, and 3 stages:

```text
generated median:                     0.527968 ms
Torch flash-SDPA median:              0.462000 ms
generated/oracle:                     1.142788x
fused dQ plus output-dot median:      0.173603 ms
dK/dV median:                         0.356864 ms
prior generated median:               0.549139 ms
prior output-dot plus dQ medians:     0.202422 ms
```

Relative to the sealed standalone-Fold result at revision `9cac1dd40b92f7c46c24b54a658078398438b890`, full latency improves 3.86% and the affected component boundary improves 14.24%. The oracle median is effectively unchanged.

Correctness passed with maximum absolute errors 0.015625 for dQ, 0.03125 for dK, and 0.0625 for dV. The deterministic output hash is `31a453c90265d128ddb0d98e03ada15b8fc291649b5be79cab24f7cca4b94006`.

The H100 used driver 595.71.05, a 700 W power limit, CUDA 12.8, Torch 2.11.0+cu128, Triton 3.6.0, and JAX 0.11.0. Clocks were not pinned. The run used 5 warmups, 30 counterbalanced repeats, 5 iterations per sample, and component profiling. `raw/shuttle-fold-h100-result.json` preserves all raw samples, correctness data, the selected schedule, and completeness evidence.

