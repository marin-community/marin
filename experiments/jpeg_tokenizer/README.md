# JPEG Tokenizer Experiments

This experiment tree is a controlled testbed for a broader question:

> which tokenization properties make autoregressive next-token prediction work in non-symbolic domains?

JPEG is the first probe because it exposes a useful ladder over the same image content:

- canonicalized file bytes,
- pre-Huffman syntax symbols,
- low-frequency quantized coefficients.

The canonical template lives in `/experiments/jpeg_tokenizer/base/`. Future variants should start as explicit copies of `base/` and keep tokenizer-specific logic local until the comparison stabilizes.

The local Phase 0 run summary is captured in [docs/reports/jpeg-tokenizer-phase0.md](/Users/dlwh/.codex/worktrees/1bd2/marin/docs/reports/jpeg-tokenizer-phase0.md).
The gzip reset follow-up note is captured in [docs/reports/jpeg-tokenizer-gzip-reset-design.md](/Users/dlwh/.codex/worktrees/1bd2/marin/docs/reports/jpeg-tokenizer-gzip-reset-design.md).

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

The codebase now also has a whole-image byte-store builder:

```bash
uv run python scripts/jpeg_tokenizer/build_whole_image_byte_token_store.py \
  --log-every 1000 \
  --output-dir artifacts/jpeg_tokenizer/token_store/imagenette_bytes_whole_v0
```

That produces one padded sequence per canonical JPEG image with distinct `EOS=256` and `PAD=257`, and the token-store
metadata tells the LM path to mask loss on the pad tail. On full Imagenette, the whole-image byte lengths are:

- examples: `13394`
- mean length: `25524.86`
- max length: `54544`

That whole-image path is now exercised under `SWA=4096`, so the byte comparison is no longer hypothetical.

The exact libjpeg-backed `K=8` coefficient store lives at:

- `/Users/dlwh/.codex/worktrees/1bd2/marin/artifacts/jpeg_tokenizer/token_store/imagenette_coeff_k8_libjpeg_v0`
- `gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_coeff_k8_libjpeg_v0`

It was built with:

```bash
uv run python scripts/jpeg_tokenizer/build_coeff_token_store.py \
  --source libjpeg \
  --k 8 \
  --log-every 1000 \
  --output-dir artifacts/jpeg_tokenizer/token_store/imagenette_coeff_k8_libjpeg_v0
```

and mirrored to GCS with:

```bash
gsutil -m rsync -r \
  artifacts/jpeg_tokenizer/token_store/imagenette_coeff_k8_libjpeg_v0 \
  gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_coeff_k8_libjpeg_v0
```

The exact `K=8` store contains:

- train: `9469` examples
- validation: `3925` examples
- sequence length: `8192`
- vocab size: `4095`

The exact libjpeg-backed whole-image symbol store lives at:

- `/Users/dlwh/.codex/worktrees/1bd2/marin/artifacts/jpeg_tokenizer/token_store/imagenette_symbols_whole_libjpeg_v0`
- `gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_symbols_whole_libjpeg_v0`

It was built with:

```bash
uv run python scripts/jpeg_tokenizer/build_whole_image_symbol_token_store.py \
  --pad-to-multiple 128 \
  --log-every 2000 \
  --output-dir artifacts/jpeg_tokenizer/token_store/imagenette_symbols_whole_libjpeg_v0
```

and mirrored to GCS with:

```bash
gsutil -m rsync -r \
  artifacts/jpeg_tokenizer/token_store/imagenette_symbols_whole_libjpeg_v0 \
  gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_symbols_whole_libjpeg_v0
```

The exact symbol store contains:

- train: `9469` examples
- validation: `3925` examples
- padded sequence length: `58240`
- vocab size: `36835`

Two additional middle-ground baselines are now built and mirrored in `eu-west4`:

- `scan_payload_bytes`
  - builder:
    `scripts/jpeg_tokenizer/build_whole_image_scan_byte_token_store.py`
  - representation:
    only the entropy-coded scan payload bytes, excluding JPEG container headers/tables/markers
  - vocab:
    `258` (`0..255` plus `EOS` and `PAD`)
- `huffman_events`
  - builder:
    `scripts/jpeg_tokenizer/build_whole_image_huffman_event_token_store.py`
  - representation:
    decoded JPEG entropy events with event ids and amplitude payloads split into separate tokens
  - vocab:
    `2224`

The full Imagenette stores resolved to:

- `scan_payload_bytes`: `seq_len=53760`
- `huffman_events`: `seq_len=115840`

The first `SWA=4096` smokes came back as:

- `scan_payload_bytes`: eval loss `5.518` at step `32`
- `huffman_events`: eval loss `2.276` at step `32`

So `scan_payload_bytes` does not seem to buy much over whole-image bytes, while `huffman_events` is strong enough to
justify a full trial even with the longer sequence.

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
  - `bytes` means the full canonical JPEG byte stream after deterministic re-encoding, not just entropy payload bytes.
- `data.py`: thin random-access dataset adapters and `LmDataConfig` wiring.
- `model.py`: local model surface, initially reusing the grug base transformer.
- `train.py`: local training surface, initially reusing the grug base trainer.
- `launch.py`: last-mile run configuration for JPEG tokenizer trials.
- `eval.py`: small evaluation/statistics helpers for Phase 0 and early comparisons.

## Immediate Next Steps

The first clean `SWA=4096` whole-image comparison is now complete:

- exact JPEG symbols:
  eval loss `2.8858`, mean bits/image `145094.24`
- exact `K=8` coeffs:
  eval loss `3.262`, mean bits/image `44928.74`
- canonical JPEG bytes:
  eval loss `4.211`, mean bits/image `159685.81`

The exact sequence-level evaluator can be rerun with:

```bash
uv run python scripts/jpeg_tokenizer/evaluate_representation_head2head.py \
  --run-spec name=coeff_k8_exact,checkpoint=gs://marin-eu-west4/tokexplore/jpeg-tokenizer-k8-libjpeg-swa4096-trial-392707/checkpoints/step-2000,store=gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_coeff_k8_libjpeg_v0,sliding_window=4096,unit_name=block,unit_count=1024 \
  --run-spec name=bytes_whole,checkpoint=gs://marin-eu-west4/tokexplore/jpeg-tokenizer-bytes-whole-swa4096-trial-7cc718/checkpoints/step-2000,store=gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_bytes_whole_v0,sliding_window=4096 \
  --run-spec name=symbols_whole_exact,checkpoint=gs://marin-eu-west4/tokexplore/jpeg-tokenizer-symbols-whole-libjpeg-swa4096-trial-a844e3/checkpoints/step-2000,store=gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_symbols_whole_libjpeg_v0,sliding_window=4096 \
  --output-dir gs://marin-eu-west4/tokexplore/jpeg-tokenizer-representation-eval-manual
```

The next experiment thread should still move to the gzip reset mechanism test; JPEG is now strong enough to serve as
the codec-structured baseline. If we come back to JPEG before that, `scan_payload_bytes` is the next clean baseline to
run, not `huffman_events`.
