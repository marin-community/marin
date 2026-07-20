# Issue #7279: latent MoE expert-parallel throughput

Append-only research log for the single-rack GB200 baseline and latent-MoE
matrix requested in [#7279](https://github.com/marin-community/marin/issues/7279).

## 2026-07-20 15:15 PDT - Matrix selected

- Reference: [#7201 comment 5016389653](https://github.com/marin-community/marin/issues/7201#issuecomment-5016389653).
- Fixed architecture: 64 GB200 GPUs, replica axis 2, batch 1,024, sequence
  length 4,096, D=6,144, 48 layers, 128 routed experts, top-4, routed
  intermediate 3,072, shared intermediate 6,144, 48 query heads, 8 KV heads,
  sliding window 512, and one global layer every six layers.
- Baselines: `sonic_cute` EP1 diagnostic, `ring_cute` EP4, `ring_cute` EP8,
  and `ragged_all_to_all_cute` EP8.
- Matched-work latent treatment: L=3,072, routed intermediate 6,144, latent
  RMSNorm, router and shared expert unchanged at D=6,144. This halves the
  dispatched activation width while preserving routed-expert parameters and
  active routed-expert GEMM work.
- Deferred efficient arm: L=3,072, 256 experts, top-4, intermediate 3,072 on
  the fastest backend after the matched-work matrix.

## 2026-07-20 16:05 PDT - Standalone harness ready for accelerator smoke

- Added explicit query/KV head, expert/shared intermediate, sliding-window,
  global-layer-frequency, and latent-MoE flags.
- Ported the validated shared D-to-L projection, latent RMSNorm, routed expert
  path, and L-to-D projection. The no-latent RNG split remains unchanged.
- Analytic FLOPs now replace the baseline routed-expert term with the latent
  expert term and add both projections. Each metrics row reports GB200 MFU using
  2.5 PFLOP/s/GPU and retains the historical B200-convention field.
- Removed two unused optimizer-choice registrations from the standalone module.
  They collided with production optimizer registration when the harness and
  Grug contract tests shared a process; the standalone constructs its optimizer
  directly and does not deserialize through the registry.
- Verification: 19 passed, 1 GPU-only skipped, 1 unrelated existing TensorStore
  checkpoint test deselected. That test reproducibly times out at 60 seconds in
  `test_grug_variant_contracts.py` alone while awaiting async checkpoint
  serialization.
- Required lint wrapper: Markdown, AST, conflict, whitespace, license, Levanter
  Ruff, and formatting checks pass after autoformat. Branch-pre-existing failures
  remain: 72 Ruff findings in the generated/inlined standalone source and seven
  Pyrefly missing-import errors for Blackwell-only `cutlass`/`quack` packages.

## Attempts

| Arm | Iris job | Placement | Commit | State | Median tok/s | Median step | GB200 MFU | Notes |
|---|---|---|---|---|---:|---:|---:|---|
