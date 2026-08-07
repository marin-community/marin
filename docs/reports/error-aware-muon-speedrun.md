# Error-aware Muon on Qwen3 130M

## TL;DR

A single-seed, 40-run sweep found that Hessian-corrected Muon with gain `0.1` improved C4-en BPB at four of five learning rates relative to a fresh Muon control. The best run used learning rate `0.020` and reached `1.164666` BPB, compared with `1.165484` for Muon at the same learning rate. The mean paired improvement was `0.000587` BPB. Native speedrun training time increased by 4.33% on average. These results justify a multi-seed replication; they do not establish a stable improvement yet.

## Method

The optimizer stores the normalized momentum exponential moving average

\[
M_t = \beta M_{t-1} + (1 - \beta) G_t.
\]

The `blend` policy applies the standard five-step Muon quintic iteration to `M_t + gamma * (G_t - M_t)`. The `hesscorr` policy adds `gamma * D sign(M_t)[G_t - M_t]` to the standard Muon direction. We approximate the derivative with `jax.jvp` through a separate convergent cubic Newton--Schulz iteration and clip its Frobenius norm to `sqrt(rank)`. The sweep used 30 cubic steps because a 15-step float32 probe had `0.620` relative JVP error, while 30 steps reduced the error to `1.4e-6`.

The implementation uses five constant-coefficient quintic steps for the base Muon direction. Matrix operations and momentum state use float32. Embeddings, the language-model head, biases, norms, and small matrices use AdamW.

## Experimental setup

| Field | Value |
|---|---:|
| Model | Qwen3, 154,147,328 parameters |
| Sequence length | 4,096 |
| Batch size | 128 |
| Steps | 4,959 |
| Tokens per run | 2,599,944,192 |
| Data | FineWeb-Edu 10B cache |
| Hardware | one v5p-8 slice |
| Seed | 0 |
| Muon learning rates | 0.008, 0.012, 0.016, 0.020, 0.024 |
| Adam learning rate | 0.2 times the Muon learning rate |
| Momentum | 0.95 |
| Weight decay | 0.1 with AdamC scheduling |
| Cubic steps | 30 |

The eight variants were one fresh Muon control, blend gains `{0.05, 0.15, 0.3, 0.5}`, and Hessian-correction gains `{0.1, 0.3, 1.0}`. All 40 training runs and all 40 result steps succeeded.

## Results

The metric is final Paloma C4-en bits per byte. Lower is better.

| Variant | LR 0.008 | LR 0.012 | LR 0.016 | LR 0.020 | LR 0.024 |
|---|---:|---:|---:|---:|---:|
| Muon | 1.175648 | 1.169083 | 1.168223 | 1.165484 | 1.165807 |
| Blend 0.05 | 1.175938 | 1.169460 | 1.167328 | 1.165508 | 1.165188 |
| Blend 0.15 | 1.180755 | 1.173499 | 1.169831 | 1.169152 | 1.169576 |
| Blend 0.3 | 1.189138 | 1.179468 | 1.176004 | 1.174829 | 1.174163 |
| Blend 0.5 | 1.199996 | 1.188054 | 1.183696 | 1.181777 | 1.181414 |
| Hesscorr 0.1 | 1.175588 | 1.169243 | 1.166605 | **1.164666** | 1.165209 |
| Hesscorr 0.3 | 1.175735 | 1.170340 | 1.167521 | 1.165753 | 1.164843 |
| Hesscorr 1.0 | 1.181352 | 1.174660 | 1.173288 | 1.173181 | 1.172279 |

Paired differences below subtract the fresh Muon result at the same learning rate. Negative values favor the feedback policy.

| Variant | LR 0.008 | LR 0.012 | LR 0.016 | LR 0.020 | LR 0.024 |
|---|---:|---:|---:|---:|---:|
| Blend 0.05 | +0.000290 | +0.000376 | -0.000895 | +0.000023 | -0.000619 |
| Blend 0.15 | +0.005108 | +0.004416 | +0.001608 | +0.003667 | +0.003769 |
| Blend 0.3 | +0.013490 | +0.010384 | +0.007781 | +0.009345 | +0.008356 |
| Blend 0.5 | +0.024348 | +0.018971 | +0.015473 | +0.016292 | +0.015607 |
| Hesscorr 0.1 | -0.000060 | +0.000160 | -0.001618 | -0.000818 | -0.000598 |
| Hesscorr 0.3 | +0.000087 | +0.001257 | -0.000702 | +0.000269 | -0.000964 |
| Hesscorr 1.0 | +0.005705 | +0.005577 | +0.005065 | +0.007697 | +0.006472 |

Hesscorr `0.1` won four of five paired comparisons, with mean delta `-0.000587` and median delta `-0.000598` BPB. Hesscorr `0.3` won two comparisons and was neutral on average (`-0.000011`). Blend `0.05` also won two comparisons, with mean delta `-0.000165`. Blend gains at or above `0.15` and Hesscorr gain `1.0` regressed at every learning rate.

Native speedrun `training_time` for Hesscorr `0.1` was 4.33% above fresh Muon on average across the five paired learning rates and 4.37% above Muon at learning rate `0.020`. This metric sums logged step durations, so it excludes queue time but includes any repeated training work logged after recovery. Blend overhead was indistinguishable from run-to-run noise.

## Baselines and limits

The historical Qwen3 130M Muon speedrun reached `1.166289` BPB at learning rate `0.016`. The new best result is `0.001624` lower, but the historical run used Nesterov momentum, constant decoupled weight decay, Muon epsilon `1e-5`, a different cache instance, and v5p-32 hardware. At the historical learning rate, Hesscorr `0.1` reached `1.166605`, which is `0.000315` worse. The fresh in-sweep Muon control is the appropriate attribution baseline.

The archived PRISM-Berkeley result from PR #4933 reached `1.170267` BPB at learning rate `0.016`. Fresh Muon reached `1.168223` at that same learning rate, so the `0.005601` best-run gap between Hesscorr and PRISM is not attributable to error feedback alone.

This sweep used one seed and selected a winner from 40 runs. A clean follow-up should pair Hesscorr `0.1` with fresh Muon at learning rate `0.020` across multiple seeds, then add an exact historical-Muon configuration on the same v5p-8 hardware.

## 300M spectral/Sylvester follow-up

The 300M continuation completed 23 of 40 cells. The best completed cell was blend gain `0.05` at learning rate `0.012`, reaching `1.056606` C4-en BPB. The paired Muon control reached `1.058626`, a delta of `-0.002020` BPB. This is a single completed comparison, not evidence for the Hessian correction at 300M: all 15 Hesscorr cells failed at global step 0 with `Loss is NaN`.

The 300M sweep estimates the spectral norm with five power iterations, divides the cubic Newton--Schulz input by `1.1` times that estimate, and uses 15 cubic steps. The correction uses the SVD-free polar/Sylvester identity: a 400-step damped fixed-point solve of `H S + S H = C` plus a 60-step Newton--Hotelling inverse. It does not use JVP.

The model is Qwen3 ~300M, trained for 11,444 steps with batch size 128 and sequence length 4,096 on one v5p-8 slice. Each completed cell processed 5,999,951,872 tokens. The learning rates are `0.004`, `0.006`, `0.008`, `0.010`, and `0.012`; momentum is `0.98`.

Final Paloma C4-en BPB; lower is better. A dash means the cell did not produce a final result.

| Variant | LR 0.004 | LR 0.006 | LR 0.008 | LR 0.010 | LR 0.012 |
|---|---:|---:|---:|---:|---:|
| Muon | 1.067245 | 1.062251 | 1.060028 | 1.059351 | 1.058626 |
| Blend 0.05 | 1.066771 | 1.061070 | 1.059204 | 1.057656 | **1.056606** |
| Blend 0.15 | 1.071553 | 1.064368 | — | 1.059578 | 1.058150 |
| Blend 0.3 | 1.080248 | 1.070832 | 1.065348 | 1.064243 | 1.063211 |
| Blend 0.5 | 1.091561 | 1.077783 | — | 1.069972 | 1.068790 |
| Hesscorr 0.1 | — | — | — | — | — |
| Hesscorr 0.3 | — | — | — | — | — |
| Hesscorr 1.0 | — | — | — | — | — |

All five Muon cells and 18 of 20 blend cells reached global step 11,443. Blend `0.5` at learning rate `0.008` became NaN at step 5,393. Blend `0.15` at learning rate `0.008` did not produce a usable final W&B summary. The 15 Hesscorr cells all failed before their first logged training step, so this sweep does not establish whether the 130M Hesscorr `0.1` result transfers to 300M.

## Artifacts

- [Completed experiment](https://marin.community/data-browser/experiment?path=gs%3A//marin-us-central1/experiments/muon_error_feedback_sweep-d76bb7.json)
- [Machine-readable results](https://github.com/marin-community/marin/blob/main/experiments/speedrun/prism_berkeley_qwen3_scaling/muon_error_feedback_results.json)
- [Best Hesscorr run](https://wandb.ai/understanding-sam/marin/runs/qwen3_130m_error_aware_muon_hesscorr-g0p1_lr0p02-72a859)
- [Paired Muon control](https://wandb.ai/understanding-sam/marin/runs/qwen3_130m_error_aware_muon_muon_lr0p02-5c7b32)
- [Historical Muon baseline](https://wandb.ai/marin-community/marin/runs/qwen3_130m_muon_4096-04770b)
- [PRISM-Berkeley baseline](https://wandb.ai/understanding-sam/marin/runs/qwen3_130m_prism_berkeley_o5_4096_lrx1-2fd229)
- [Best completed 300M blend run](https://wandb.ai/understanding-sam/marin/runs/qwen3_300m_error_aware_muon_blend-g0p05_lr0p012-2026.07.23.4)
- [Paired 300M Muon control](https://wandb.ai/understanding-sam/marin/runs/qwen3_300m_error_aware_muon_muon_lr0p012-2026.07.23.4)
