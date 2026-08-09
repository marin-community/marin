# Recovered streaming-attention backward H100 replay

This artifact replays Shuttle revision `9cac1dd40b92f7c46c24b54a658078398438b890` on one batch-priority NVIDIA H100 80GB HBM3. The measured causal BF16 GQA workload has sequence length 2,048, 32 query heads, 8 KV heads, head dimension 128, 32x32 blocks, 8 warps, and 3 pipeline stages.

The generated backward measures 0.549139 ms median over 30 counterbalanced samples. Torch flash SDPA measures 0.462077 ms, for a 1.188415x ratio. The generated path passes the 1.2x acceptance threshold. Component medians are 0.044742 ms for the output-dot Fold, 0.157680 ms for dQ, and 0.356966 ms for dK/dV.

The source boundary is ordinary JAX causal GQA differentiated with `jax.vjp`, serialized as StableHLO, and recovered into Shuttle's streaming backward program. The recovered score-map VJP guard ran before physical lowering. The output is deterministic across repeated execution with SHA256 `6957ec539e0a1d6270a1a8d1efa0531a5bec5a624d885dbd8bc51120d723f58d`.

The physical execution harness is transitional Torch/Triton. This result measures the recovered semantic boundary and compiler-owned streaming backward schedule. It is not direct JAX/XLA backend integration.

Torch 2.11 first selected cuDNN SDPA for the matched oracle. cuDNN failed with `No valid execution plans built` for this GQA shape; `cudnn_oracle_failure.log` preserves that failure. The recorded run disables only cuDNN SDPA before invoking the unchanged benchmark, which lets Torch select flash SDPA. The generated path, recovery path, physical schedule, and measurement configuration are unchanged.

The H100 ran with driver 595.71.05, CUDA 12.8, Torch 2.11.0+cu128, Triton 3.6.0, and JAX 0.11.0. The device power limit was 700 W. Clocks were not pinned; the post-run sample recorded 1,830 MHz SM and 2,619 MHz memory clocks.

The benchmark used 5 warmups, 30 counterbalanced repeats, and 5 iterations per sample with component profiling enabled. Its arguments were:

```text
--semantic-source jax_vjp_hlo_recovery
--sequence 2048
--mutation causal
--block-m 32
--block-n 32
--num-warps 8
--num-stages 3
--warmups 5
--repeats 30
--iterations 5
--profile-components
--shuttle-revision 9cac1dd40b92f7c46c24b54a658078398438b890
```

`result.json` contains the raw counterbalanced samples, component samples, correctness errors, semantic recovery provenance, schedule, and toolchain. `stdout.log` is the exact JSON printed by the benchmark. `source_vjp_stablehlo.mlir.bc` freezes the exported JAX VJP input. `postrun_telemetry.csv` records the exact GPU UUID, driver, power limit, and sampled clocks.
