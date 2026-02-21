# exp2917 Block-Window Report (w8/w16/w32)

Runs:
- `block_w8`: https://wandb.ai/marin-community/marin/runs/exp2917_shuffle_block_150m_20260220-190441-exp2-block_shuffle_w8-1d74bd
- `block_w16`: https://wandb.ai/marin-community/marin/runs/exp2917_shuffle_block_150m_20260220-190441-exp-block_shuffle_w16-c50e53
- `block_w32`: https://wandb.ai/marin-community/marin/runs/exp2917_shuffle_block_150m_20260220-190441-exp-block_shuffle_w32-01502f

## Final metrics (step 2061)

| run | train_loss | eval_loss | uncheatable_macro | uncheatable_micro | tokens/s | tensorstore_reads_total |
|---|---:|---:|---:|---:|---:|---:|
| block_w8 | 3.6543 | 3.9034 | 3.9164 | 4.0272 | 871570 | 4465 |
| block_w16 | 3.6524 | 3.8860 | 3.8931 | 4.0068 | 868988 | 3089 |
| block_w32 | 3.5904 | 3.8841 | 3.8801 | 3.9859 | 873077 | 4577 |

## Relative to w16 (%)

| run | tokens/s | eval_loss | uncheatable_macro | uncheatable_micro | tensorstore_reads_total |
|---|---:|---:|---:|---:|---:|
| block_w8 | +0.30% | +0.45% | +0.60% | +0.51% | +44.55% |
| block_w32 | +0.47% | -0.05% | -0.33% | -0.52% | +48.17% |

## Tail loss stability (step >= 1000)

| run | loss_std | jitter_std_diff |
|---|---:|---:|
| block_w8 | 0.110942 | 0.045568 |
| block_w16 | 0.112688 | 0.045551 |
| block_w32 | 0.125579 | 0.036995 |

## Takeaways
- `w16` minimizes read count in this sweep (`3089` total reads).
- `w32` has best final quality metrics (lowest eval and uncheatable losses) and slightly best throughput, but read count is highest (`4577`).
- `w8` is in-between on throughput/quality, with read count above `w16`.

## Artifacts
- `block_window_summary_latest.json`
- `block_window_report_latest.md`
- `block_window_loss_loglog_vs_tokens_latest.png`
- `block_window_throughput_tokens_per_s_by_step_latest.png`
- `block_window_tensorstore_reads_total_by_step_latest.png`
- `block_window_uncheatable_macro_by_step_latest.png`
- `block_window_uncheatable_micro_by_step_latest.png`
