# Grug-MoE v4 Path Response Analysis

This analysis uses the one-dimensional path \(w(t)=(1-t)p+t w_{v4}\), with task deltas oriented so positive means better than proportional at the same scale.

The representability hypothesis predicts three broad empirical regimes: aligned tasks improve along the controllable path, coverage-sensitive tasks worsen as the path moves away from proportional, and weakly controllable or noisy tasks show flat or inconsistent response.

## Coverage

- Headline strict-common analysis uses hidden dimensions: `512, 768, 1024`.
- Headline task count: `20` after excluding incomplete non-verb MMLU-SL aliases.
- Complete task/scale paths in the filtered input: `62` out of `98` task-scale cells.
- The current strict-common cut intentionally excludes d1280 for most tasks and d1536 for all intermediate path points; rerun this script after pending eval/training completion.

## Headline Classification

- Endpoint improves: `4` tasks.
- Interior peak: `9` tasks.
- Worsens with t: `6` tasks.
- Mixed or flat: `1` tasks.

## Strongest Positive t-Response

- `logprob_humaneval_10shot`: Pearson `0.917`, endpoint delta `0.1091`, best t `1.00`.
- `logprob_gsm8k_5shot`: Pearson `0.828`, endpoint delta `0.1195`, best t `1.00`.
- `piqa_5shot`: Pearson `0.647`, endpoint delta `0.0007869`, best t `0.75`.
- `arc_easy_5shot`: Pearson `0.619`, endpoint delta `0.001676`, best t `0.75`.
- `medmcqa_5shot`: Pearson `0.273`, endpoint delta `0.01163`, best t `0.75`.
- `boolq_10shot`: Pearson `0.253`, endpoint delta `0.00978`, best t `1.00`.
- `truthfulqa_mc1_0shot`: Pearson `0.160`, endpoint delta `0.005712`, best t `1.00`.
- `medmcqa_sl_verb_5shot`: Pearson `0.070`, endpoint delta `0.0009566`, best t `0.75`.

## Strongest Negative t-Response

- `hellaswag_0shot`: Pearson `-0.813`, endpoint delta `-0.0005048`, best t `0.00`.
- `hellaswag_5shot`: Pearson `-0.708`, endpoint delta `-0.0005049`, best t `0.00`.
- `openbookqa_0shot`: Pearson `-0.638`, endpoint delta `-0.00214`, best t `0.25`.
- `wsc273_0shot`: Pearson `-0.487`, endpoint delta `-0.0003601`, best t `0.00`.
- `boolq_sl_verb_10shot`: Pearson `-0.442`, endpoint delta `-0.04161`, best t `0.00`.
- `mmlu_sl_verb_0shot`: Pearson `-0.411`, endpoint delta `-0.0006343`, best t `0.25`.
- `csqa_sl_verb_5shot`: Pearson `-0.310`, endpoint delta `-0.001991`, best t `0.25`.
- `mmlu_sl_verb_5shot`: Pearson `-0.302`, endpoint delta `-0.0003886`, best t `0.00`.

## Interpretation Notes

- Correlation is useful for sign and monotonicity, but not effect size across tasks because task metrics have different native units.
- This is not a repeated-seed uncertainty analysis. Flat or mixed response should be treated as weak evidence until paired with SNR/noise estimates.
- A positive endpoint and consistent positive slope supports controllability along the v4 direction. A negative endpoint and consistent negative slope supports a real path tradeoff against proportional coverage. An interior best t suggests a trust-region interpolation may dominate the endpoint.
