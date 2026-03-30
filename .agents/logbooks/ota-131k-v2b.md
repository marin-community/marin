# OT-Agent 131K v2b: Research Logbook

## Scope
- Goal: Find a `v5p-32` Levanter recipe for 131K SFT that can run stably to at least the halfway point without OOM.
- Primary metric(s): no host-RAM or HBM OOM through long-run training; improved effective throughput by removing avoidable checkpoint overhead.
- Constraints: keep `max_seq_len=131072`; do not change away from the Levanter training path; prefer `v5p-32` over `v5p-256`.

## Baseline
- Date: 2026-03-30
- Issue: https://github.com/marin-community/marin/issues/3897
- Code refs:
  - `experiments/exp3897_sft_ota_131k_qwen3_8b.py`
  - `experiments/exp3897v3_sft_ota_131k_qwen3_8b_no_offload.py`
- Baseline findings:
  - Offload-based `v2` on `v5p-32` failed repeatedly with exit 137 during JAX checkpoint serialization.
  - No-offload `v3` on `v5p-32` was manually killed after 1h29m, not OOM-killed.
  - `v3` reached step 13 and survived multiple time-based checkpoints at 256 GB host RAM.

## EXP3897-V2B-001
- Date: 2026-03-30 UTC
- Hypothesis: `v5p-32` instability came from `ScanCheckpointPolicy(save_carries="offload")` plus Marin's 10-minute temporary checkpoint cadence, not from 131K training itself.
- Evidence gathered:
  - `uv run iris --config lib/iris/examples/marin.yaml job bug-report /kevin/iris-run-exp3897_sft_ota_131k_qwen3_8b-20260325-061745/train_lm --tail 120`
  - `uv run iris --config lib/iris/examples/marin.yaml job bug-report /kevin/iris-run-exp3897v3_sft_ota_131k_qwen3_8b_no_offload-20260326-055401 --tail 80`
  - `uv run iris --config lib/iris/examples/marin.yaml job logs /kevin/iris-run-exp3897_sft_ota_131k_qwen3_8b-20260325-061745/train_lm --level info | tail -n 220`
- Results:
  - Offload `v2` saved a temporary checkpoint every ~10 minutes, which at ~224s/step meant roughly every 3 steps.
  - The fatal `v2` OOM at step 410 occurred inside `jax.experimental.array_serialization.serialization` while committing checkpoint data.
  - No-offload `v3` was explicitly `Terminated by user`, not OOM, after 1h29m. It had already completed several checkpoint commits successfully.
- Decision:
  - Build `v2b` from the no-offload recipe.
  - Disable time-based temporary checkpoints.
  - Disable interim HF exports.
  - Use a half-run stability target by default.

## Next
- Launch `exp3897v2b` on `v5p-32` with `num_train_steps=989`, `save_interval=None`, and no HF exports.
- Monitor for:
  - successful JIT and first train steps,
  - absence of checkpoint-related OOM,
  - absence of host-offload/XLA regressions.
