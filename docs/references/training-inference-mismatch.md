# Generating-policy ages in TIM diagnostics

Training–inference mismatch (TIM) compares trainer and sampler token
probabilities. Ordinary PPO with tokenwise truncated importance sampling (TIS)
uses the chosen-token probabilities already available during training.

A token is fresh only when its generating-policy stamp matches the learner's
published policy version. Missing stamps have unknown age, represented by −1.
The rollout buffer's lease age limits admission but cannot identify the weights
that generated an unstamped token. Batch construction preserves this distinction
for missing stamp channels and partially stamped responses.

| Metric prefix | Generating-policy age |
| --- | --- |
| `policy/mismatch/pooled/` | All selected tokens. |
| `policy/mismatch/staleness0/` | The generating version matches this learner snapshot. |
| `policy/mismatch/staleness1/`, `staleness2/`, `staleness3/`, `staleness4-7/`, `staleness8+/` | One, two, three, four–seven, or at least eight published policy steps old. |
| `policy/mismatch/unknown/` | No generating-policy stamp. |

Selected tokens have a positive loss mask. `selected_tokens` counts that
population; `finite_tokens` counts its finite trainer-minus-sampler log
probabilities. Means, percentiles and fractions use finite tokens. Empty buckets
omit means. Check `async/staleness_tokens_stamped` for stamp coverage before
interpreting fresh-policy mismatch.
