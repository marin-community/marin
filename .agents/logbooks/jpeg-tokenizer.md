# JPEG Tokenizer: Research Logbook

## Scope

- Goal: determine whether simple autoregressive modeling over canonical JPEG-derived token streams is viable, starting with a bounded `K=4` coefficient baseline.
- Primary metric(s): train/eval NLL, sequence-length feasibility, tokenizer determinism, and basic reconstruction quality for lossy coefficient streams.
- Constraints: keep storage localized to `eu-west4`, prefer small reproducible artifacts, and use `marin-community/tokexplore` for W&B.

## Baseline

- Date: 2026-03-08
- Code refs:
  - `/Users/dlwh/.codex/worktrees/1bd2/marin/experiments/jpeg_tokenizer/base/jpeg_codecs.py`
  - `/Users/dlwh/.codex/worktrees/1bd2/marin/experiments/jpeg_tokenizer/base/data.py`
  - `/Users/dlwh/.codex/worktrees/1bd2/marin/experiments/jpeg_tokenizer/base/launch.py`
- Baseline numbers:
  - Imagenette `320px` train split: `9469` examples
  - K=4 coefficient store: `4096` tokens/example, configured vocab `4095`, observed vocab `1796`
  - K=4 reconstruction: mean SSIM `0.7164`, mean PSNR `24.65 dB`

## Experiment Log

### 2026-03-08 20:00 - Phase 0 finalized locally

- Hypothesis: a fixed-length K=4 coefficient stream is a clean first training target because it avoids the sequence-length variance of bytes and symbols.
- Command:
  - `uv run python scripts/jpeg_tokenizer/inspect_representations.py --output-dir ...`
  - `uv run python scripts/jpeg_tokenizer/evaluate_coeff_reconstruction.py --output-dir ...`
- Config: Imagenette `320px`, canonical `256x256` luma JPEG, quality `95`, `K=4`.
- Result: deterministic preprocessing passed on all examples; coefficient sequences are exactly length `4096`; K=4 is lossy but structurally coherent.
- Interpretation: Phase 1 should begin with coefficient-only training, while bytes and symbols need explicit windowing or a different compute budget.
- Next action: materialize a reusable token store and move the first TPU run to `eu-west4`.

### 2026-03-08 21:10 - Token store built and mirrored

- Hypothesis: a precomputed store is the lowest-risk input path for the first TPU-backed run.
- Command:
  - `uv run python scripts/jpeg_tokenizer/build_coeff_token_store.py --output-dir artifacts/jpeg_tokenizer/token_store/imagenette_coeff_k4_v0`
  - `gsutil -m rsync -r artifacts/jpeg_tokenizer/token_store/imagenette_coeff_k4_v0 gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_coeff_k4_v0`
- Config: local build plus regional GCS mirror.
- Result: train/validation matrices and manifests were written locally and mirrored to `gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_coeff_k4_v0`.
- Interpretation: the first cluster run can stream a small fixed artifact instead of rebuilding data remotely.
- Next action: wire the launch step directly to the mirrored store.

### 2026-03-08 21:35 - Launch path hardening

- Hypothesis: the first Ray launch should use a small smoke step on `v6e-8` before attempting the longer baseline.
- Command:
  - `uv run python experiments/jpeg_tokenizer/base/launch.py --help`
- Config: `tokexplore/jpeg-tokenizer-k4-smoke` and `tokexplore/jpeg-tokenizer-k4-trial`, both reading the regional GCS store.
- Result: discovered and fixed a script-entrypoint import hazard where local `tokenizers.py` shadowed the third-party `tokenizers` package; renamed the module to `jpeg_codecs.py`.
- Interpretation: the launch surface is now safe to execute as a standalone script under Ray runtime packaging.
- Next action: validate tests/pre-commit, commit the launch milestone, then submit the smoke run to `marin-eu-west4-a`.
