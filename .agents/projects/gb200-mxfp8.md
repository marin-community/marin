# GB200 MXFP8 for the lc1 recipe (speed first, quality later)

Goal: measure the step-time win of MXFP8 (e4m3 data, e8m0 scale per 32 elements) on the lc1 GB200 d6144
recipe, one piece at a time, then check quality on the pieces that pay. Parameters, optimizer state and
Muon momentum stay fp32; only GEMM operands are quantized.

## Measured so far (one GB200, expert shapes M=114688, E=6)

- QuACK MXFP8 grouped GEMM alone: 1.68x (up 3072->4608), 1.93x (down 4608->3072) over bf16.
- Full ReLU^2 expert MLP fwd+bwd with every quantization a separate torch pass: 0.83x (15.36 -> 18.48 ms).
  Quantization must ride in producers (XLA fusions or GEMM epilogues), not be standalone passes.
- XLA scaled_dot_general MXFP8: ~1.0x, not usable.

## Stages

1. Routed experts, activation path (up, down, dact, dx): row-wise (feature-dim) quantization only.
   - JAX quantizer -> e4m3 + e8m0; SFA in QuACK's blocked (L, rm, rk, 32, 4, 4) layout with per-expert
     tile padding (expert e's scale rows start at tile cu[e]//128 + e).
   - x_dispatch quantized before the dispatch all-to-all (rows travel as fp8 + e8m0, half the bytes).
   - post / d_pre quantized in the producing GEMM's epilogue (QuACK quantize_out SFD).
   - dy quantized inside the XLA fusion that produces it (combine backward).
   - Weights quantized per step from the bf16 compute copy (small).
2. Weight gradients (varlen-k, token-dim scales): needs 32-token blocks relative to each expert segment;
   likely 128-align expert segments in the receive buffer so the blocks are global.
3. Dense projections (attention / shared expert): cuBLASLt MXFP8 via XLA if it fuses, else QuACK dense.

After each stage: one rack A/B against the current recipe (`gb200_recipe` memory), interactive priority.

## Status

- [ ] Probe: JAX quantizer + blocked SF + blockscaled varlen-m GEMM through cutlass.jax (`scratch_lc1/mxfp8_jax_probe.py`).
