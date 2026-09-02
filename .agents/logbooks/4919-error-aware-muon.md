---
topic: error-aware Muon
issue: https://github.com/marin-community/marin/issues/4919
description: Reproduce, stabilize, and evaluate error-aware Muon on Qwen3 130M and 300M.
author: kaiyuew
---

# Error-aware Muon: Task Logbook

## Scope

- Goal: determine whether blend or Hessian-corrected Muon improves held-out C4-en BPB.
- Primary metrics: Paloma C4-en BPB and native speedrun training time.
- Constraints: single-seed matched learning-rate comparisons; keep 300M data and checkpoints in `us-central1`.
- Coordinating issue/PR: [#4919](https://github.com/marin-community/marin/issues/4919), [#7118](https://github.com/marin-community/marin/pull/7118).

## Current TL;DR

The stabilized 300M Hesscorr implementation completed all 15 cells. Gains `0.1` and `0.3` beat paired Muon at all five learning rates, but all Hesscorr variants lost to blend `0.05` at all five learning rates and cost about 90% more native training time than Muon. The best Hesscorr cell reached `1.057701` BPB at gain `0.3`, learning rate `0.012`; paired Muon reached `1.058626`, while blend `0.05` reached `1.056606`. This single-seed result establishes numerical stability and a small improvement over Muon, but not a favorable quality-per-compute tradeoff.

## Baseline

- Date: 2026-07-24
- Code ref: artifact version `2026.07.23.4`
- Muon at learning rate `0.012`: `1.058626` C4-en BPB, 17,696.8 seconds.
- Blend `0.05` at learning rate `0.012`: `1.056606` C4-en BPB, 17,680.2 seconds.

## Negative Results Index

- The first 300M Hesscorr sweep failed all 15 cells at global step 0 with non-finite loss.
- A 50-step warmup retry failed on the first corrected update, localizing the instability to the Sylvester path.
- A v4-8 copy in `us-central2` failed before W&B initialization because its workers rejected the `us-central1` training cache. It was not retried and is not optimizer evidence.
- Stabilized Hesscorr did not beat blend `0.05` at any of 15 paired gain/LR comparisons.

## Entry Log

### 2026-09-02 09:30 - EAM-300-001 stabilized Hesscorr result

- Hypothesis: warm up the momentum state and harden the Sylvester solve to preserve a useful Hessian correction at 300M without non-finite updates.
- Launch base commit: `3bcd661840b28da7e06995975ccceccf3eaac5f8`; the Iris workspace bundle included the uncommitted stabilization source now published on `codex/error-aware-muon-speedrun`.
- Command: `iris --cluster=marin job run --no-wait --job-name muon-error-feedback-300m-hesscorr-stable-20260828-222720 ... -- python -m experiments.speedrun.prism_berkeley_qwen3_scaling.muon_error_feedback_sweep --size 300m --version 2026.08.28.3 --max-concurrent 8 --variant-group hesscorr`.
- Config: Qwen3 300M, FineWeb-Edu 10B, v5p-8 in `us-central1`, 11,444 steps, batch 128, sequence length 4,096, momentum `0.98`, 50-update correction warmup, 15 cubic steps, 400 Sylvester steps, 60 inverse steps, gains `{0.1, 0.3, 1.0}`, learning rates `{0.004, 0.006, 0.008, 0.010, 0.012}`.
- Result: 15/15 cells finished at global step 11,443 with final checkpoint metadata; zero failures and 124 preemptions. Best Hesscorr BPB was `1.057701`. Gains `0.1` and `0.3` beat Muon 5/5, but every Hesscorr gain lost to blend `0.05` 5/5. Mean native training-time overhead versus Muon was 90.03% to 90.05%.
- Interpretation: the stabilization is effective, but the observed quality gain over Muon does not compensate for the correction's cost or beat the simple blend baseline. Confidence is exploratory because every cell uses seed 0.
- Artifacts: [`muon_error_feedback_results.json`](../../experiments/speedrun/prism_berkeley_qwen3_scaling/muon_error_feedback_results.json), [`error-aware-muon-speedrun.md`](../../docs/reports/error-aware-muon-speedrun.md), [best Hesscorr W&B run](https://wandb.ai/understanding-sam/marin/runs/qwen3_300m_error_aware_muon_hesscorr-g0p3_lr0p012-2026.08.28.3).
- Next action: stop scaling this implementation by default; revisit only with a cheaper correction or a multi-seed quality-per-compute objective that the blend baseline does not satisfy.
