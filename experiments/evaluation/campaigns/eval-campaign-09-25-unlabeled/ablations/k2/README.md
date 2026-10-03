# K2 Horizon screening ablation

These runs are unlabeled diagnostics, not policy-conformant benchmark scores. The three Evalchemy configs select the first 20 tasks under one pinned Evalchemy revision and retain the campaign graders. Compare task IDs and first responses across arms before interpreting aggregate scores.

The first pair holds the current compiled TP2/DP1 serve path and 73,728-token server limit fixed. It changes only the K2 reasoning setting on thinking-off tasks (`low` versus `high`); MATH500 remains `high` in both arms. The `high` arm intentionally departs from the draft thinking-off mapping. Neither arm enables eager execution or HF position overrides.

```bash
uv run python -m experiments.evaluation.cli launch \
  --model-config experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/ablations/k2/model-tp2-dp1-low.yaml \
  --evalchemy-config experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/ablations/k2/humanevalplus-20.yaml \
  --evalchemy-config experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/ablations/k2/mbppplus-20.yaml \
  --evalchemy-config experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/ablations/k2/math500-20.yaml \
  --federated_cluster cw-rno2a --priority interactive --no-wait \
  --description 'Unlabeled K2 screen: compiled TP2/DP1, low reasoning on thinking-off tasks'
```

Change only the model-config basename and description to run the other three arms (`tp2-dp1-high`, `tp2-dp4-low`, and `tp2-dp4-high`). The TP2/DP4 configs otherwise match the TP2/DP1 pair, so they isolate the MoE topology change. Do not use `--version` or a policy label.
