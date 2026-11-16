# Gated DeltaNet-2 TPU kernel

The experimental single-device Pallas implementation exposes
`gdn2_pallas_forward_trainable` in
`levanter.kernels.pallas.gdn2.candidate.gdn2_pipeline`.
Model code does not call it yet.

Pass `q`, `k`, `v`, `w`, `b`, and `g` with shape `[B, L, H, 128]`
and FP32 or BF16 dtype. Log-decays `g` must be finite and nonpositive.
Pass the output scale as a static argument, for example `scale=128**-0.5`.
The optional FP32 initial state
has shape `[B, H, 128, 128]` and defaults to zeros. Keep all inputs on
one device. The kernel returns FP32 output and final state. Its
gradients cover the six token inputs and the initial state.

The sequence length must be divisible by the chunk length `bt`.
The solve microblock `mb` must divide `bt / 2`. Select the score layout
with `ScoreLayout.for_device_kind(device.device_kind)` when constructing
the TPU configuration. TPU v4 uses the feature-first layout.

The [GDN-2 research notebook](https://github.com/marin-community/marin-logbooks/blob/bc377bb875485b5170d8eb222b2ee5d228fb0933/logbooks/42-gdn2-tpu/notebook.md)
links the runnable benchmark, the v5e correctness check, and the
September measurement snapshots. Representative-shape performance and
v4 memory fit for the refactored kernel remain unverified.
