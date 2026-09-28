# Grug-MoE v4 Path Response Analysis

This analysis uses the one-dimensional path \(w(t)=(1-t)p+t w_{v4}\), with task deltas oriented so positive means better than proportional at the same scale.

The representability hypothesis predicts three broad empirical regimes: aligned tasks improve along the controllable path, coverage-sensitive tasks worsen as the path moves away from proportional, and weakly controllable or noisy tasks show flat or inconsistent response.

## Coverage

- Headline strict-common analysis uses hidden dimensions: `512, 768, 1024, 1280, 1536`.
- Headline task count: `20` after excluding incomplete non-verb MMLU-SL aliases.
- Complete task/scale paths in the filtered input: `100` out of `100` task-scale cells.
- No missing task/scale paths remain in the filtered input.

## Headline Classification

- Endpoint improves: `4` tasks.
- Interior peak: `6` tasks.
- Worsens with t: `4` tasks.
- Mixed or flat: `6` tasks.

## Strongest Positive t-Response

- `logprob_gsm8k_5shot`: Pearson `0.754`, endpoint delta `0.1106`, best t `1.00`.
- `logprob_humaneval_10shot`: Pearson `0.708`, endpoint delta `0.08194`, best t `1.00`.
- `piqa_5shot`: Pearson `0.687`, endpoint delta `0.0008126`, best t `0.75`.
- `arc_easy_5shot`: Pearson `0.521`, endpoint delta `0.002039`, best t `0.75`.
- `arc_challenge_5shot`: Pearson `0.336`, endpoint delta `0.001008`, best t `0.75`.
- `medmcqa_5shot`: Pearson `0.262`, endpoint delta `0.01807`, best t `1.00`.
- `boolq_10shot`: Pearson `0.244`, endpoint delta `0.008685`, best t `1.00`.
- `truthfulqa_mc1_0shot`: Pearson `0.136`, endpoint delta `0.004896`, best t `1.00`.

## Strongest Negative t-Response

- `hellaswag_0shot`: Pearson `-0.824`, endpoint delta `-0.0005721`, best t `0.00`.
- `hellaswag_5shot`: Pearson `-0.798`, endpoint delta `-0.0005605`, best t `0.00`.
- `openbookqa_0shot`: Pearson `-0.573`, endpoint delta `-0.001942`, best t `0.25`.
- `copa_0shot`: Pearson `-0.332`, endpoint delta `-0.002728`, best t `0.00`.
- `mmlu_sl_verb_0shot`: Pearson `-0.282`, endpoint delta `-0.0002842`, best t `0.25`.
- `boolq_sl_verb_10shot`: Pearson `-0.280`, endpoint delta `-0.02633`, best t `0.00`.
- `mmlu_sl_verb_5shot`: Pearson `-0.271`, endpoint delta `-0.000291`, best t `0.00`.
- `wsc273_0shot`: Pearson `-0.271`, endpoint delta `-0.0002816`, best t `0.00`.

## Standardized Effect-Size View

- Standardization divides each task's oriented delta by the empirical standard deviation of that task's oriented metric values across the Grug-MoE dashboard/path cells. This is a native-unit effect-size diagnostic, not a repeated-seed noise standard deviation.
- At t=1, `11` tasks are positive and `9` tasks are negative in standardized units.
- Mean positive endpoint standardized delta: `0.409`; mean absolute negative endpoint standardized delta: `0.293`.

### Largest Standardized Endpoint Gains

- `medmcqa_5shot`: endpoint z-delta `0.846`, best t `1.00`.
- `medmcqa_sl_verb_5shot`: endpoint z-delta `0.811`, best t `0.75`.
- `boolq_10shot`: endpoint z-delta `0.677`, best t `1.00`.
- `truthfulqa_mc1_0shot`: endpoint z-delta `0.510`, best t `1.00`.
- `logprob_humaneval_10shot`: endpoint z-delta `0.501`, best t `1.00`.
- `logprob_gsm8k_5shot`: endpoint z-delta `0.407`, best t `1.00`.

### Largest Standardized Endpoint Deteriorations

- `boolq_sl_verb_10shot`: endpoint z-delta `-1.161`, best t `0.00`.
- `mmlu_sl_verb_0shot`: endpoint z-delta `-0.385`, best t `0.25`.
- `mmlu_sl_verb_5shot`: endpoint z-delta `-0.334`, best t `0.00`.
- `copa_0shot`: endpoint z-delta `-0.276`, best t `0.00`.
- `openbookqa_0shot`: endpoint z-delta `-0.183`, best t `0.25`.
- `wsc273_0shot`: endpoint z-delta `-0.166`, best t `0.00`.

## Interpretation Notes

- Correlation is useful for sign and monotonicity, but not effect size across tasks because task metrics have different native units.
- This is not a repeated-seed uncertainty analysis. Flat or mixed response should be treated as weak evidence until paired with SNR/noise estimates.
- A positive endpoint and consistent positive slope supports controllability along the v4 direction. A negative endpoint and consistent negative slope supports a real path tradeoff against proportional coverage. An interior best t suggests a trust-region interpolation may dominate the endpoint.
