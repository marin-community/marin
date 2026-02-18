# GDN Pallas TPU Hill-Climb Log

Append-only running log for `lib/levanter/src/levanter/layers/gated_deltanet.py` TPU optimization work.

## Goal
Increase training MFU for the Gated DeltaNet TPU implementation without changing model semantics.

## Loop Contract
Each iteration should include:
1. one optimization hypothesis,
2. one code change set,
3. TPU correctness validation,
4. one profiled training run,
5. one commit if validated.

## Known Constraints (as of 2026-01-06)
- Strict lower-triangular inversion is a TPU hotspot.
- Pallas TPU kernels do not support dynamic slice indexing in-kernel, requiring static indexing/segmentation.

## Entry Template

```markdown
### Iteration <N> - <short title>
- Date: <UTC timestamp>
- Commit: <sha>
- Hypothesis:
- Change summary:
- Correctness checks:
  - Command:
  - Result:
- Profile run:
  - Command:
  - Job ID:
  - Trace location:
- Hotspots observed:
- MFU/throughput delta:
- Next hypothesis:
```

## Iterations

### Iteration 0 - Infra bootstrap
- Date: 2026-02-18
- Commit: (pending)
- Hypothesis: Standardized scripts/docs and lightweight profile entrypoint reduce iteration overhead for future optimization passes.
- Change summary: Added `scripts/gdn/gdnctl.py`, tiny profile experiment, recipe/docs, and unattended Codex loop harness.
- Correctness checks:
  - Command: N/A (infra-only change)
  - Result: N/A
- Profile run:
  - Command: N/A
  - Job ID: N/A
  - Trace location: N/A
- Hotspots observed: N/A
- MFU/throughput delta: N/A
- Next hypothesis: Use new loop to target one kernel bottleneck per commit.

### Iteration 1 - Loop hardening + trace validation
- Date: 2026-02-18
- Commit: (pending)
- Hypothesis: The loop must run reliably under TPU queue contention; adding safe tiny-profile defaults and a first-class dev TPU profile path will make each iteration deterministic.
- Change summary:
  - Fixed `ray-test`/`ray-profile` command and submission-id parsing issues in `scripts/gdn/gdnctl.py`.
  - Defaulted unattended Codex loop to `gpt-5.3-codex` + `model_reasoning_effort=xhigh`.
  - Added safe tiny-profile defaults for v5p-8 (`batch_size=8`, shorter profile window) in `experiments/speedrun/hackable_transformer_gdn/tiny_profile.py` and CLI defaults.
  - Added `dev-tpu-profile` subcommand in `scripts/gdn/gdnctl.py` to bypass Ray queueing.
- Correctness checks:
  - Command: `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-central1 --tpu-name calvin-gdn-loop --tests both --no-sync`
  - Result: `87 passed, 2 skipped`.
- Profile run:
  - Command: `uv run scripts/ray/dev_tpu.py --cluster us-central1 --tpu-name calvin-gdn-loop execute --no-sync -e EQX_ON_ERROR=nan -e WANDB_MODE=online -e MARIN_PREFIX=gs://marin-us-central1 -e GDN_PROFILE_SIZE=130m -e GDN_PROFILE_NUM_STEPS=8 -e GDN_PROFILE_PROFILE_START_STEP=2 -e GDN_PROFILE_PROFILE_NUM_STEPS=3 -e GDN_PROFILE_BATCH_SIZE=8 -e GDN_PROFILE_RUN_NAME_PREFIX=gdn_loopcheck -- "uv pip uninstall --python .venv/bin/python torchvision || true && .venv/bin/python -m experiments.speedrun.hackable_transformer_gdn.tiny_profile --force_run_failed true"`
  - Job/Run: W&B run `gdn_loopcheck_130m_ch128_seg16_8steps-5ecaf5`
  - Trace location: `.profiles/wandb/gdn_loopcheck_130m_ch128_seg16_8steps-5ecaf5-profiler-v0/plugins/profile/2026_02_18_12_05_06/perfetto_trace.json.gz`
- Hotspots observed (TPU:0 XLA Ops aggregate):
  - `while`: `218.985 ms` total; major loops mapped to `lib/levanter/src/levanter/layers/gated_deltanet.py:1861` and `lib/levanter/src/levanter/layers/gated_deltanet.py:2361`.
  - `custom-call`: `182.564 ms` total; dominant entries are `shard_map.1068-1072` from `lib/levanter/src/levanter/layers/gated_deltanet.py:2361` and `shard_map.1063-1067` from `lib/levanter/src/levanter/layers/gated_deltanet.py:1315`.
  - Large non-GDN training cost remains in logits path (`fusion.321`, source in Equinox/JAX jit; `long_name` includes `bf16[2,4096,128256]` dot-general outputs).
- MFU/throughput delta: N/A (infra-validation iteration; no kernel math change yet).
- Next hypothesis: reduce GDN segment scan overhead by fusing segment boundaries/state handoff so line-2361 and line-1861 while/custom-call blocks execute fewer large-loop iterations per step.
