# Error-aware Muon on Qwen3 130M and 300M

## TL;DR

A single-seed, 40-run 130M sweep found that Hessian-corrected Muon with gain `0.1` improved C4-en BPB at four of five learning rates relative to a fresh Muon control. A stabilized 15-cell 300M follow-up then completed every Hesscorr cell. Gains `0.1` and `0.3` each beat paired Muon at all five learning rates, with mean improvements of `0.000449` and `0.000589` BPB. However, neither beat the simpler blend gain `0.05` at any learning rate, and Hesscorr increased native training time by about 90%. The best Hesscorr result was `1.057701` BPB at gain `0.3`, learning rate `0.012`; the paired Muon and blend results were `1.058626` and `1.056606`. The evidence supports a small, consistent improvement over plain Muon, but not the correction's added complexity and cost under this single-seed 300M setup.

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

## 300M stabilized spectral/Sylvester follow-up

### Stabilization and source

The original 300M spectral/Sylvester launch failed all 15 Hesscorr cells at global step 0. A first warmup-only retry stayed finite through step 50 and failed on step 51, the first update with an enabled correction. This localized the numerical failure to the Sylvester correction rather than the ordinary Muon path.

The stabilized implementation makes four changes:

- use ordinary quintic Muon for the first `round(1 / (1 - momentum)) = 50` optimizer updates;
- run the polar-Hessian, Sylvester, and inverse matrix products with explicit highest matmul precision;
- form the polar Hessian from one product and its transpose so it is exactly symmetric after rounding;
- use a scale-safe Frobenius norm and return a zero correction when the Sylvester solve or its norm is non-finite.

The implementation is in [`muon_error_feedback_optimizer.py`](../../experiments/speedrun/prism_berkeley_qwen3_scaling/muon_error_feedback_optimizer.py), the exact 100-step TPU gate is in [`muon_error_feedback_stability_gate.py`](../../experiments/speedrun/prism_berkeley_qwen3_scaling/muon_error_feedback_stability_gate.py), and the 15-cell launcher is in [`muon_error_feedback_sweep.py`](../../experiments/speedrun/prism_berkeley_qwen3_scaling/muon_error_feedback_sweep.py). The v5p launch used commit `3bcd661` plus the optimizer and gate changes now published in those files. The launcher's default v5p path is unchanged; a later optional TPU-variant argument also records the unsuccessful v4 experiment.

The stabilized path passed the 100-step TPU gate, including the first corrected update, then completed all 15 cells. Each run trained Qwen3 ~300M for 11,444 steps with batch size 128 and sequence length 4,096 on one preemptible v5p-8 slice in `us-central1`, processing 5,999,951,872 tokens. The grid crossed learning rates `{0.004, 0.006, 0.008, 0.010, 0.012}` with correction gains `{0.1, 0.3, 1.0}` and momentum `0.98`. Iris recorded 124 aggregate preemptions but no failed cells; every run reached global step 11,443 with finite final loss and final checkpoint metadata.

### Held-out C4-en quality

Final Paloma C4-en BPB is materialized from each cell's `speedrun_results.json`; lower is better. Muon and blend `0.05` are the completed, same-geometry `2026.07.23.4` references.

| Variant | LR 0.004 | LR 0.006 | LR 0.008 | LR 0.010 | LR 0.012 |
|---|---:|---:|---:|---:|---:|
| Muon | 1.067245 | 1.062251 | 1.060028 | 1.059351 | 1.058626 |
| Blend 0.05 | 1.066771 | 1.061070 | 1.059204 | 1.057656 | **1.056606** |
| Hesscorr 0.1 | 1.066853 | 1.062055 | 1.059515 | 1.058882 | 1.057950 |
| Hesscorr 0.3 | 1.067066 | 1.061869 | 1.059530 | 1.058388 | **1.057701** |
| Hesscorr 1.0 | 1.067755 | 1.062778 | 1.060028 | 1.058927 | 1.058036 |

Paired differences below subtract the reference at the same learning rate. Negative values favor Hesscorr.

| Variant | Reference | LR 0.004 | LR 0.006 | LR 0.008 | LR 0.010 | LR 0.012 | Mean | Wins |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Hesscorr 0.1 | Muon | -0.000392 | -0.000196 | -0.000513 | -0.000469 | -0.000676 | -0.000449 | 5/5 |
| Hesscorr 0.3 | Muon | -0.000179 | -0.000383 | -0.000497 | -0.000963 | -0.000925 | -0.000589 | 5/5 |
| Hesscorr 1.0 | Muon | +0.000511 | +0.000527 | +0.000001 | -0.000423 | -0.000590 | +0.000005 | 2/5 |
| Hesscorr 0.1 | Blend 0.05 | +0.000082 | +0.000986 | +0.000310 | +0.001226 | +0.001344 | +0.000790 | 0/5 |
| Hesscorr 0.3 | Blend 0.05 | +0.000295 | +0.000799 | +0.000326 | +0.000732 | +0.001095 | +0.000649 | 0/5 |
| Hesscorr 1.0 | Blend 0.05 | +0.000984 | +0.001709 | +0.000824 | +0.001272 | +0.001430 | +0.001244 | 0/5 |

Gains `0.1` and `0.3` therefore transfer the small directionally consistent improvement over plain Muon to 300M. Gain `1.0` is neutral on average. None of the Hesscorr settings improves on blend `0.05`, including the best Hesscorr cell: at learning rate `0.012`, Hesscorr `0.3` is `0.000925` better than Muon but `0.001095` worse than blend.

### Cost and decision

Native speedrun `training_time` averaged about 33,582 seconds for Hesscorr versus 17,671 seconds for paired Muon. Mean overhead was 90.03% for gain `0.1`, 90.05% for gain `0.3`, and 90.05% for gain `1.0`. Queue time is excluded, while repeated logged work after a preemption is included.

This is a single-seed grid, so the sub-millibit differences should not be treated as a definitive optimizer ranking. Even so, the observed tradeoff is unfavorable: the stabilized Hesscorr path roughly doubles native training time and adds a fragile iterative solve, while the cheaper blend `0.05` is better at every paired learning rate. Further Hesscorr scaling is not justified by this result alone. If the method is revisited, the next useful experiment is a multi-seed comparison focused on Hesscorr `0.3`, Muon, and blend `0.05` at learning rates `0.010` and `0.012`, with an optimized Sylvester implementation or an explicit quality-per-compute target.

## Artifacts

- [Completed experiment](https://marin.community/data-browser/experiment?path=gs%3A//marin-us-central1/experiments/muon_error_feedback_sweep-d76bb7.json)
- [Machine-readable results](../../experiments/speedrun/prism_berkeley_qwen3_scaling/muon_error_feedback_results.json)
- [Best Hesscorr run](https://wandb.ai/understanding-sam/marin/runs/qwen3_130m_error_aware_muon_hesscorr-g0p1_lr0p02-72a859)
- [Paired Muon control](https://wandb.ai/understanding-sam/marin/runs/qwen3_130m_error_aware_muon_muon_lr0p02-5c7b32)
- [Historical Muon baseline](https://wandb.ai/marin-community/marin/runs/qwen3_130m_muon_4096-04770b)
- [PRISM-Berkeley baseline](https://wandb.ai/understanding-sam/marin/runs/qwen3_130m_prism_berkeley_o5_4096_lrx1-2fd229)
- [Best completed 300M blend run](https://wandb.ai/understanding-sam/marin/runs/qwen3_300m_error_aware_muon_blend-g0p05_lr0p012-2026.07.23.4)
- [Paired 300M Muon control](https://wandb.ai/understanding-sam/marin/runs/qwen3_300m_error_aware_muon_muon_lr0p012-2026.07.23.4)
- [Best stabilized 300M Hesscorr run](https://wandb.ai/understanding-sam/marin/runs/qwen3_300m_error_aware_muon_hesscorr-g0p3_lr0p012-2026.08.28.3)
