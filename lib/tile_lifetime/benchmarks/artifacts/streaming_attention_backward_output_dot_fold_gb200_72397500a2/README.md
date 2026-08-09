# Query-owner Fold attachment on GB200

This artifact repeats the H100 protocol at Shuttle revision `72397500a24ae788215ca17f365cb7a3e063401c` on an actual NVIDIA GB200 with compute capability 10.0. The device was admitted from `cw-us-east-08a`; it was not substituted with a B200 or an H100.

The semantic input and generated plan match the H100 run. Shuttle recovers an FP32 Map/Fold for `sum(output * output_cotangent, feature)`, proves the complete feature axis is resident in the query owner, and attaches the Fold to dQ preparation. dK/dV consumes the materialized row scalar. There is no standalone output-dot launch.

For sequence length 2,048, 32 query heads, 8 KV heads, head dimension 128, 32x32 blocks, 8 warps, and 3 stages:

```text
generated median:                     0.455238 ms
Torch flash-SDPA median:              0.407830 ms
generated/oracle:                     1.116244x
fused dQ plus output-dot median:      0.140496 ms
dK/dV median:                         0.314048 ms
prior generated median:               0.484029 ms
prior output-dot plus dQ medians:     0.176759 ms
```

Relative to the sealed standalone-Fold result at revision `9cac1dd40b92f7c46c24b54a658078398438b890`, full latency improves 5.95% and the affected component boundary improves 20.52%. dK/dV remains effectively unchanged.

Correctness passed with maximum absolute errors 0.015625 for dQ, 0.03125 for dK, and 0.03125 for dV. The deterministic output hash is `38cef00445e9738c1a14675e6a3f62db0709bd04c7ab12c11789a86924985bf3`.

The GB200 UUID was `GPU-78bf2ae1-552e-d7f3-cccd-3e522bcb9887`. It used driver 595.71.05, a 1200 W power limit, CUDA 12.8, Torch 2.11.0+cu128, Triton 3.6.0, and JAX 0.11.0. Clocks were not pinned. The run used 5 warmups, 30 counterbalanced repeats, 5 iterations per sample, and component profiling. `raw/shuttle-fold-gb200-result.json` preserves all raw samples, correctness data, the selected schedule, and completeness evidence.

