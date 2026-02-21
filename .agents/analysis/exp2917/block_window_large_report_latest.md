# exp2917 Large-Window Report (w64/w128/w256/w512/w1024)

## Baselines
- Full shuffle: https://wandb.ai/marin-community/marin/runs/exp2917_shuffle_block_150m_2026-02-20-full_shuffle-e638fd
- Legacy block (small-window): https://wandb.ai/marin-community/marin/runs/exp2917_shuffle_block_150m_2026-02-20-block_shuffle-5dab59

## Runs
- w64: https://wandb.ai/marin-community/marin/runs/exp2917_shuffle_block_150m_2026-02-21-block_shuffle_w64-27229c
- w128: https://wandb.ai/marin-community/marin/runs/exp2917_shuffle_block_150m_2026-02-21-block_shuffle_w128-53ab01
- w256: https://wandb.ai/marin-community/marin/runs/exp2917_shuffle_block_150m_2026-02-21-block_shuffle_w256-daeeaa
- w512: https://wandb.ai/marin-community/marin/runs/exp2917_shuffle_block_150m_2026-02-21-block_shuffle_w512-b8c967
- w1024: https://wandb.ai/marin-community/marin/runs/exp2917_shuffle_block_150m_2026-02-21-block_shuffle-8ba290

## Final metrics (step 2061)

| window | eval_loss | uncheatable_macro | uncheatable_micro | eval_loading_time_s | eval_total_time_s |
|---|---:|---:|---:|---:|---:|
| w64 | 3.876324 | 3.862838 | 3.975510 | 24.28 | 36.69 |
| w128 | 3.868693 | 3.862938 | 3.976528 | 23.06 | 35.45 |
| w256 | 3.865820 | 3.875195 | 3.983520 | 17.39 | 29.49 |
| w512 | 3.850497 | 3.853288 | 3.962978 | 17.22 | 38.59 |
| w1024 | 3.846557 | 3.860804 | 3.966571 | 16.29 | 28.71 |

## Relative to full shuffle (%)

| window | eval_loss | uncheatable_macro | uncheatable_micro |
|---|---:|---:|---:|
| w64 | +0.899% | +0.213% | +0.564% |
| w128 | +0.700% | +0.216% | +0.590% |
| w256 | +0.625% | +0.534% | +0.767% |
| w512 | +0.226% | -0.034% | +0.247% |
| w1024 | +0.124% | +0.160% | +0.338% |

## Relative to legacy small-window block (%)

| window | eval_loss | uncheatable_macro | uncheatable_micro |
|---|---:|---:|---:|
| w64 | -0.897% | -2.018% | -1.914% |
| w128 | -1.092% | -2.015% | -1.889% |
| w256 | -1.166% | -1.704% | -1.716% |
| w512 | -1.557% | -2.260% | -2.223% |
| w1024 | -1.658% | -2.069% | -2.135% |

## Interpretation

- `w512` already closes almost all of the full-shuffle gap.
- `w1024` is similarly close to full; on this run it improves eval loss slightly vs `w512` but is slightly worse on uncheatable macro/micro.
- The `w512` vs `w1024` difference is small enough that longer-run confirmation is still appropriate.

## Artifacts

- `block_window_large_summary_latest.json`
- `block_window_large_report_latest.md`
- `block_window_large_eval_gap_vs_full_latest.png`
