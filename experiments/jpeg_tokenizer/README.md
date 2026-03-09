# JPEG Tokenizer Experiments

This experiment tree is a controlled testbed for a broader question:

> which tokenization properties make autoregressive next-token prediction work in non-symbolic domains?

JPEG is the first probe because it exposes a useful ladder over the same image content:

- canonicalized file bytes,
- pre-Huffman syntax symbols,
- low-frequency quantized coefficients.

The canonical template lives in `/experiments/jpeg_tokenizer/base/`. Future variants should start as explicit copies of `base/` and keep tokenizer-specific logic local until the comparison stabilizes.

The local Phase 0 run summary is captured in [docs/reports/jpeg-tokenizer-phase0.md](/Users/dlwh/.codex/worktrees/1bd2/marin/docs/reports/jpeg-tokenizer-phase0.md).

## Local Artifacts

The current local `K=4` coefficient token store lives at:

- `/Users/dlwh/.codex/worktrees/1bd2/marin/artifacts/jpeg_tokenizer/token_store/imagenette_coeff_k4_v0`
- `gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_coeff_k4_v0`

It was built with:

```bash
uv run python scripts/jpeg_tokenizer/build_coeff_token_store.py \
  --log-every 1000 \
  --output-dir artifacts/jpeg_tokenizer/token_store/imagenette_coeff_k4_v0
```

and mirrored to GCS with:

```bash
gsutil -m rsync -r \
  artifacts/jpeg_tokenizer/token_store/imagenette_coeff_k4_v0 \
  gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_coeff_k4_v0
```

The store contains:

- train: `9469` examples
- validation: `3925` examples
- sequence length: `4096`
- vocab size: `4095`

The current byte-window baseline store lives at:

- `/Users/dlwh/.codex/worktrees/1bd2/marin/artifacts/jpeg_tokenizer/token_store/imagenette_bytes_w8192_v0`
- `gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_bytes_w8192_v0`

It was built with:

```bash
uv run python scripts/jpeg_tokenizer/build_byte_token_store.py \
  --log-every 1000 \
  --output-dir artifacts/jpeg_tokenizer/token_store/imagenette_bytes_w8192_v0
```

and mirrored to GCS with:

```bash
gsutil -m rsync -r \
  artifacts/jpeg_tokenizer/token_store/imagenette_bytes_w8192_v0 \
  gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_bytes_w8192_v0
```

The byte-window store contains:

- train: `34191` windows
- validation: `14264` windows
- sequence length: `8192`
- vocab size: `257`

## V0 Decisions

This section freezes the initial assumptions for the Phase 0 feasibility spike so later scripts do not drift.

- Corpus: start with `frgfm/imagenette` (`320px` config) from Hugging Face datasets for the first 1k-10k image inspection pass.
- Robustness corpus: defer WebVision to the first stress-test pass, using a small stratified subset after the clean-corpus token stats and training path are working.
- Resolution: `256x256`.
- Color mode: luma-only (`Y`) for the initial pass.
- Resize/crop policy: deterministic bicubic center-crop to `256x256`.
- Canonical JPEG encoding: deterministic re-encode, quality `95`, non-progressive, no metadata, fixed behavior versioned in code.
- Coefficient baseline: first `K=4` zigzag coefficients per `8x8` block.
- Variant names: `base` (coefficient baseline), `bytes`, `symbols`.
- Initial model size: Grug-style decoder-only transformer with `hidden_dim=512`, `intermediate_dim=1792`, `num_layers=6`, `num_heads=8`, `num_kv_heads=8`, and `max_seq_len=4096`.
- Initial metrics: sequence-length distribution, approximate vocab size, deterministic canonicalization checks, train/eval NLL, normalized likelihood (for example bits-per-pixel), and rollout decode-validity.

These are V0 defaults for the feasibility spike, not permanent project commitments. If Phase 0 statistics invalidate one of them, update this document and version the change explicitly.

### Why Not WebVision First

WebVision is useful for later robustness checks because it captures real web-image messiness, but it is not the best first corpus for this project. The initial pass should isolate tokenizer behavior with the smallest possible amount of ingestion noise, download friction, and label-quality baggage. Once the coefficient, byte, and symbol paths are working on the clean corpus, add a small WebVision subset to test whether the conclusions survive more heterogeneous JPEG sources.

## Layout

`/experiments/jpeg_tokenizer/base/` is split to mirror the repo's template-first experiment style:

- `jpeg_codecs.py`: canonical JPEG decisions and token encoders.
- `data.py`: thin random-access dataset adapters and `LmDataConfig` wiring.
- `model.py`: local model surface, initially reusing the grug base transformer.
- `train.py`: local training surface, initially reusing the grug base trainer.
- `launch.py`: last-mile run configuration for JPEG tokenizer trials.
- `eval.py`: small evaluation/statistics helpers for Phase 0 and early comparisons.

## Immediate Next Steps

1. Launch `tokexplore/jpeg-tokenizer-k4-smoke` on `v6e-8` against the regional `gs://marin-eu-west4/...` token store.
2. If the smoke run is healthy, launch `tokexplore/jpeg-tokenizer-k4-trial` as the first real Phase 1 baseline.
3. Run `tokexplore/jpeg-tokenizer-bytes-w8192-smoke`, then promote it to a matched-budget byte trial if startup and the first eval look healthy.
