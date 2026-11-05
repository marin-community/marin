# Training–inference mismatch diagnostics

Training–inference mismatch (TIM) compares the trainer's and sampler's token
probabilities. Ordinary PPO with tokenwise truncated importance sampling (TIS)
logs TIM using chosen-token probabilities already available during training.

Check counts before comparing means. Empty buckets omit means; absence does
not establish agreement.

| Metrics | Interpretation |
| --- | --- |
| `policy/mismatch/pooled/` | Old trainer minus behavior sampler log probabilities, before this update. Check `selected_tokens`, `finite_tokens`, `missing_behavior`, `log_ratio_abs_mean`, `log_ratio_abs_p99` and `frac_outside_0_5_2`. |
| `policy/mismatch/staleness0/` | Tokens stamped with this learner snapshot's version. Disagreement at matching weights can indicate engine mismatch. |
| `policy/mismatch/staleness1/` through `staleness8+/` | Older generating versions; disagreement includes weight drift. Age buckets are 1, 2, 3, 4–7 and 8+. |
| `policy/mismatch/unknown/` | No generating-policy stamp. A rollout lease bounds age but does not establish matching weights. Check `async/staleness_tokens_stamped` for stamp coverage. |
| `policy/correction/truncated_fraction` | Loss tokens whose uncapped TIS weight exceeded the cap. |
| `policy/ppo_clip_ratio*`, `policy/log_ratio_*` | PPO clipping and current-versus-old trainer changes during optimization. |

Each mismatch bucket also reports `pos_first256/`, `pos_last256/` and
`response_ge2048/` counts and absolute mean log ratios. Positions include masked
response tokens. First and last windows overlap on short responses. Inspect
long-response counts before interpreting their means. Selected and finite counts
remain separate.
