# exp2917 Uncheatable Eval (Macro/Micro)

Crash-safe treatment used for plotting:
- dedupe by `global_step` (last write wins)
- comparisons aligned at shared eval horizon (`global_step <= 1000`)
- no cumulative-counter math used in these plots

At `global_step=1000`:

| run | macro_loss | micro_loss | macro vs full | micro vs full |
|---|---:|---:|---:|---:|
| full_shuffle | 4.2613 | 4.3572 | baseline | baseline |
| era_shuffle | 4.7481 | 4.8700 | +11.43% | +11.77% |
| block_shuffle_w8 | 4.3364 | 4.4457 | +1.76% | +2.03% |
| block_shuffle_w16 | 4.3007 | 4.4114 | +0.92% | +1.25% |
| block_shuffle_w32 | 4.3167 | 4.4258 | +1.30% | +1.58% |

Files:
- `uncheatable_eval_macro_loss_by_step.png`
- `uncheatable_eval_micro_loss_by_step.png`
- `uncheatable_eval_summary.json`
