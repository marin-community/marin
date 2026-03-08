# JPEG Tokenizer Phase 0

This report captures the local feasibility spike for the JPEG tokenizer project.

## Run

- Command:
  `uv run python scripts/jpeg_tokenizer/inspect_representations.py --max-examples 9469 --log-every 1000 --output-dir artifacts/jpeg_tokenizer/phase0/imagenette_train_320px_v0`
- Corpus:
  `frgfm/imagenette`, config `320px`, split `train`
- Examples:
  `9469`
- Canonical image recipe:
  `256x256`, bicubic center-crop, luma-only (`Y`), JPEG quality `95`, non-progressive, metadata stripped
- Artifact directory:
  `/Users/dlwh/.codex/worktrees/1bd2/marin/artifacts/jpeg_tokenizer/phase0/imagenette_train_320px_v0`
- Reconstruction artifact directory:
  `/Users/dlwh/.codex/worktrees/1bd2/marin/artifacts/jpeg_tokenizer/phase0/coeff_reconstruction_imagenette_train_320px_v0`

## Findings

- Deterministic canonicalization passed for all `9469/9469` examples.
- Byte-token sequences are long: mean `25466.86`, p95 `38317.80`, max `54543`, vocab `256`.
- Symbol-token sequences are even longer: mean `32068.28`, p95 `46867.20`, max `57997`.
- The packed symbol encoding reduced the configured symbol vocabulary to `36833`, with `8088` symbols actually observed on this corpus.
- Coefficient-token sequences are perfectly regular at this setting: exactly `4096` tokens per image for every example.
- The coefficient vocabulary looks operationally reasonable for a first pass: configured `4095`, observed `1796`.
- Coefficient-only reconstruction is intentionally lossy at low `K`, and the loss is now quantified with SSIM as a perceptual proxy:
  - `K=4`: mean SSIM `0.7164`, p95 DSSIM `0.2485`, mean PSNR `24.65 dB`
  - `K=8`: mean SSIM `0.8139`, p95 DSSIM `0.1772`, mean PSNR `27.03 dB`
  - `K=16`: mean SSIM `0.9064`, p95 DSSIM `0.1013`, mean PSNR `30.68 dB`
  - `K=64`: mean SSIM `0.9937`, p95 DSSIM `0.0050`, mean PSNR `45.83 dB`

## Decisions

- Keep `Imagenette 320px` as the local Phase 0 corpus.
  It is small enough to iterate quickly and large enough to expose stable sequence-length and vocabulary statistics.
- Keep the canonical image size at `256x256`.
  At `K=4` zigzag coefficients per `8x8` block, this lands exactly on `4096` coefficient tokens, which matches the initial model budget cleanly.
- Keep `Y` rather than `RGB` for V0.
  The safest baseline is the coefficient stream, and moving to three channels would roughly triple the coefficient sequence length to about `12288` tokens before any additional JPEG chroma complications.
- Start Phase 1 with the coefficient baseline only.
  It is the only tokenizer family here that naturally fits the initial `4096` sequence-length target without additional windowing.
- Keep `K=4` as the first coefficient baseline, but treat it as a deliberately lossy baseline rather than a near-lossless one.
  The perceptual results show that `K=4` preserves a meaningful amount of structure while still discarding substantial information.
- Use `K=8` or `K=16` as the first follow-up coefficient ablations once the training path is stable.
  They offer materially better perceptual fidelity, but they increase the sequence length to `8192` and `16384` tokens respectively at `256x256`.
- Treat byte and symbol baselines as longer-sequence families that need an explicit plan before training.
  For the current `256x256` recipe, they do not fit the initial `4096` token budget; they will need windowed training, a smaller canonical resolution, or a different matched-compute setup.

## Implications For Phase 1

- The first end-to-end training path should use coefficient tokens, passthrough tokenization, and explicit vocab size `4095`.
- The first reconstruction-aware training comparison should probably be `K=4` versus `K=8`, because that is the smallest ablation that meaningfully changes perceptual fidelity.
- Byte and symbol variants should remain in the tokenizer ladder, but they should not block the first training implementation.
- The next implementation task is a deterministic precomputed token store plus a thin random-access dataset adapter for coefficient tokens.
