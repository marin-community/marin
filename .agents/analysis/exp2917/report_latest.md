# exp2917 Updated Report (final full/era/block runs)

Runs:
- `full_shuffle`: https://wandb.ai/marin-community/marin/runs/exp2917_shuffle_block_150m_2026-02-20-full_shuffle-e638fd
- `era_shuffle`: https://wandb.ai/marin-community/marin/runs/exp2917_shuffle_block_150m_2026-02-20-era_shuffle-d4b30e
- `block_shuffle`: https://wandb.ai/marin-community/marin/runs/exp2917_shuffle_block_150m_2026-02-20-block_shuffle-5dab59

## Main findings
- Block shuffle keeps throughput close to full shuffle (`-0.85%` tokens/s) while cutting backend bytes by `99.58%`.
- Era shuffle is strongest on backend locality (`99.58%` fewer GCS bytes than full), but quality regresses heavily on uncheatable eval (`+14.53%` macro loss).
- Block shuffle has much smaller quality regression than era (`+2.28%` uncheatable macro vs full).

## Final metrics (step 2061)

| run | train_loss | eval_loss | uncheatable_macro | uncheatable_micro | tokens/s | gcs_reads_total | gcs_bytes_total | cache_hit_rate_total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full_shuffle | 3.6928 | 3.8418 | 3.8546 | 3.9532 | 882216 | 1932354 | 621914747700 | 0.2211 |
| era_shuffle | 3.6962 | 4.1593 | 4.4147 | 4.5372 | 869611 | 4899 | 2608161636 | 0.9829 |
| block_shuffle | 3.7034 | 3.9114 | 3.9424 | 4.0531 | 874715 | 9137 | 2642644048 | 0.2978 |

## Relative vs full_shuffle (%)

| run | tokens/s | gcs_reads_total | gcs_bytes_total | train_loss | eval_loss | uncheatable_macro | uncheatable_micro |
|---|---:|---:|---:|---:|---:|---:|---:|
| era_shuffle | -1.43% | -99.75% | -99.58% | +0.09% | +8.26% | +14.53% | +14.77% |
| block_shuffle | -0.85% | -99.53% | -99.58% | +0.29% | +1.81% | +2.28% | +2.53% |

## Train-loss stability (tail window step >= 1000)

| run | loss_mean | loss_std | jitter_std_diff | jitter_mean_abs_diff |
|---|---:|---:|---:|---:|
| full_shuffle | 3.8609 | 0.0862 | 0.034429 | 0.027142 |
| era_shuffle | 3.8121 | 0.0821 | 0.032220 | 0.025868 |
| block_shuffle | 3.8561 | 0.0890 | 0.036955 | 0.028833 |

## Notes
- `throughput/loading_time` is a hook-time gauge and not a direct measure of backend IO volume. Use TensorStore GCS totals for backend cost/load comparisons.
- All three runs are finished at step 2061 (same horizon).

## Artifacts
- `metrics_summary_latest.json`
- `loss_loglog_vs_tokens_latest.png`
- `throughput_tokens_per_s_by_step_latest.png`
- `loading_time_by_step_latest.png`
- `tensorstore_gcs_reads_total_by_step_latest.png`
- `uncheatable_eval_macro_loss_by_step_latest.png`
- `uncheatable_eval_micro_loss_by_step_latest.png`
