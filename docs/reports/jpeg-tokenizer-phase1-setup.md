# JPEG Tokenizer Phase 1 Setup

This note records the first runnable training setup for the `K=4` coefficient baseline.

## Prepared Artifacts

- Token store:
  `/Users/dlwh/.codex/worktrees/1bd2/marin/artifacts/jpeg_tokenizer/token_store/imagenette_coeff_k4_v0`
- GCS mirror:
  `gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_coeff_k4_v0`
- Store contents:
  - train: `9469` examples
  - validation: `3925` examples
  - sequence length: `4096`
  - vocab size: `4095`

## Code Paths

- Store reader and `LmDataConfig` wiring:
  `/Users/dlwh/.codex/worktrees/1bd2/marin/experiments/jpeg_tokenizer/base/data.py`
- Store builder:
  `/Users/dlwh/.codex/worktrees/1bd2/marin/scripts/jpeg_tokenizer/build_coeff_token_store.py`
- Runnable launch step:
  `/Users/dlwh/.codex/worktrees/1bd2/marin/experiments/jpeg_tokenizer/base/launch.py`

## Launch Conventions

- Executor step name:
  `tokexplore/jpeg-tokenizer-k4-trial`
- W&B target:
  `marin-community/tokexplore`
- W&B group:
  `tokexplore-jpeg-tokenizer-k4`
- Default token store path:
  `gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_coeff_k4_v0`

## Validation

- The file-backed store round-trips through the Levanter causal LM path in tests.
- A one-step CPU smoke test runs successfully from the file-backed store.
- The launch module now resolves the token store at runtime rather than import time, so the code remains importable in clean checkouts.

## Next Step

Use the `tokexplore/jpeg-tokenizer-k4-trial` step as the first TPU-backed end-to-end run, starting conservatively with the existing short `2000`-step config.
