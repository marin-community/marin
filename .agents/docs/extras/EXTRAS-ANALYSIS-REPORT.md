# Marin Extras and Dependency Groups Analysis Report

**Date:** 2025-12-15
**Analyzed packages:** marin, levanter, fray, dupekit, zephyr, haliax

---

## Changes Applied

The following cleanup was performed on `lib/marin/pyproject.toml`:

| Change | Details |
|--------|---------|
| ✅ Removed `gcp` extra | All deps were duplicates of main deps |
| ✅ Cleaned `download_transform` | Removed 8 unused deps: chardet, datasets, fastparquet, google-cloud-storage-transfer, htmlmin, markdownify, readabilipy, boto3 |
| ✅ Cleaned `quality_dedup_consolidate` | Removed 3 unused deps: cattrs, nltk, datasets |
| ✅ Removed `tokenize_train` extra | All deps were duplicates |
| ✅ Removed `post_training` extra | Dead code (explicitly marked unused) |
| ✅ Cleaned `eval` extra | Removed langdetect, immutabledict |

---

## Executive Summary

This report analyzes all extras and dependency groups across the Marin workspace to identify:
- Dead/unused extras
- Redundant/duplicate dependencies
- Opportunities for consolidation

### Categorization Principle

**Extras should only be used for mutually exclusive/conflicting dependencies.** Non-conflicting optional features should be:
- Main dependencies (if production code needs them)
- Dependency groups (if dev/test only)

### Remaining Work

| Category | Count | Action |
|----------|-------|--------|
| **CONFLICTING extras (keep)** | 5 | `cuda12`, `tpu`, `cpu`, levanter `gpu`/`tpu` |
| **NON-CONFLICTING extras (reorganize)** | 8 | Consider moving to main deps or groups |

---

## lib/marin Extras Analysis

### `gcp` Extra - ✅ **REMOVED**

This extra was removed entirely. All useful dependencies were already in main deps.

---

### `cuda12`, `tpu`, `cpu` Extras - **ESSENTIAL (KEEP)**

These hardware extras are **critical infrastructure** for:
- JAX hardware-specific builds (`jax[cuda12]`, `jax[tpu]`, base `jax`)
- PyTorch index selection (CUDA vs CPU wheels)
- Automatic hardware detection in Fray cluster management

**Recommendation:** Keep unchanged. These serve essential hardware isolation.

---

### `download_transform` Extra - ✅ **CLEANED**

Removed 8 unused/duplicate deps. Current state:

| Dependency | Notes |
|------------|-------|
| html2text==2024.2.26 | HTML to text conversion |
| lxml[html_clean] | HTML parsing |
| readability-lxml | Article extraction |
| resiliparse | Fast HTML parsing |
| trafilatura>=2.0 | Web scraping |
| warcio | WARC file handling |

**Next step:** Consider moving these to main deps if they're production requirements.

---

### `quality_dedup_consolidate` Extra - ✅ **CLEANED**

Removed 3 unused/duplicate deps (cattrs, nltk, datasets). Current state:

| Dependency | Notes |
|------------|-------|
| ddsketch | Statistics sketching |
| dupekit | Deduplication tools |
| fasttext | Text classification |
| huggingface_hub | Model hub access |
| transformers | Model inference |
| zstandard>=0.18.0 | Compression |

**Next step:** Consider moving these to main deps if they're production requirements.

---

### `tokenize_train` Extra - ✅ **REMOVED**

This extra was removed entirely. All dependencies were duplicates:
- `multiprocess` - in main deps
- `haliax` - in main deps
- `lm-eval` - in `eval` extra
- `tblib` - in levanter main deps

---

### `rl` Extra - **ACTIVE (KEEP)**

| Dependency | Status | Notes |
|------------|--------|-------|
| prime | USED | For RL environments |
| sympy | USED | For math evaluation |
| verifiers==0.1.5 | USED | For verification |

**Recommendation:** Keep. Referenced in RL experiment code.

---

### `post_training` Extra - ✅ **REMOVED**

This extra was removed entirely. It was dead code (explicitly marked "No longer used" in pyproject.toml excludes).

---

### `eval` Extra - ✅ **CLEANED**

Removed unused `langdetect` and `immutabledict`. Current state:

| Dependency | Notes |
|------------|-------|
| lm-eval[math] | Evaluation harness |

**Next step:** Consider moving to main deps if evaluations are core functionality.

---

## lib/levanter Extras Analysis

### `gpu`, `tpu` Extras - **ESSENTIAL (KEEP)**

Same as marin hardware extras - essential for JAX hardware isolation.

### `torch_test` Extra - **ACTIVE (KEEP)**

Used for PyTorch interop testing. Contains torch + peft.

### `profiling` Extra - **ACTIVE (KEEP)**

Contains xprof, tensorboard, tensorboardX for performance profiling.

### `serve` Extra - **ESSENTIAL (KEEP)**

| Notes |
|-------|
| marin depends on `levanter[serve]` |
| Contains fastapi, uvicorn, openai |
| Used for inference serving |

### `lm_eval` Extra - **CONSIDER CONSOLIDATING**

Overlaps with marin's `eval` extra. Consider using only one.

### `gcp` Extra - **KEEP (DIFFERENT FROM MARIN)**

Levanter's gcp extra includes `google-auth` not in marin. Keep separate.

---

## lib/fray Extras Analysis

### `ray` Extra - **ESSENTIAL (KEEP)**

| Notes |
|-------|
| zephyr depends on `fray[ray]` |
| Optional Ray backend for fray |
| Allows fray to work without Ray |

---

## lib/dupekit Extras Analysis

### `benchmark` Extra - **ACTIVE (KEEP)**

| Notes |
|-------|
| 5 benchmark test files |
| pytest-benchmark + pytest-memray |
| Well-documented in README |
| Intentionally separated from CI |

---

## Summary Tables

### Marin Extras - Current State

| Extra | Status | Remaining Deps |
|-------|--------|----------------|
| gcp | ✅ REMOVED | - |
| cuda12 | KEEP (conflicting) | jax[cuda12], torch |
| tpu | KEEP (conflicting) | jax[tpu], torch |
| cpu | KEEP (conflicting) | jax, torch |
| download_transform | ✅ CLEANED | 6 deps |
| quality_dedup_consolidate | ✅ CLEANED | 6 deps |
| tokenize_train | ✅ REMOVED | - |
| rl | NON-CONFLICTING | 3 deps |
| post_training | ✅ REMOVED | - |
| eval | ✅ CLEANED | 1 dep |

### Remaining Non-Conflicting Extras (Consider Reorganizing)

| Package | Extra | Action |
|---------|-------|--------|
| marin | download_transform | Move to main deps |
| marin | quality_dedup_consolidate | Move to main deps |
| marin | rl | Move to main deps |
| marin | eval | Move to main deps |
| levanter | torch_test | Move to `test` group |
| levanter | profiling | Move to main or `dev` group |
| levanter | serve | Move to main deps |
| levanter | lm_eval | Consolidate with marin eval |
| levanter | gcp | Move to main deps |
| fray | ray | Move to main deps |
| dupekit | benchmark | Move to `test` group |

---

## Individual Analysis Files

Detailed analysis for each extra is available at:
- `.agents/docs/extras/marin-gcp-analysis.md`
- `.agents/docs/extras/marin-hardware-extras-analysis.md`
- `.agents/docs/extras/marin-download-transform-analysis.md`
- `.agents/docs/extras/marin-quality-dedup-analysis.md`
- `.agents/docs/extras/marin-tokenize-train-analysis.md`
- `.agents/docs/extras/marin-rl-analysis.md`
- `.agents/docs/extras/marin-post-training-analysis.md`
- `.agents/docs/extras/marin-eval-analysis.md`
- `.agents/docs/extras/levanter-hardware-extras-analysis.md`
- `.agents/docs/extras/levanter-*-analysis.md`
- `.agents/docs/extras/fray-ray-analysis.md`
- `.agents/docs/extras/dupekit-benchmark-analysis.md`
- `.agents/docs/extras/dependency-groups-analysis.md`
