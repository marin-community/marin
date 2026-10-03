# K2 Horizon screening ablation

These runs are unlabeled diagnostics, not policy-conformant benchmark scores. Use the canonical full Evalchemy configs for scored comparisons. The 20-item configs were retired: `--limit 20` capped requests but left the chat benchmark graders with the full example set, so none of the three subset runs produced a valid score. Compare task IDs and first responses across arms before interpreting aggregate scores.

The first pair holds the current compiled TP2/DP1 serve path and 73,728-token server limit fixed. It changes only the K2 reasoning setting on thinking-off tasks (`low` versus `high`); MATH500 remains `high` in both arms. The `high` arm intentionally departs from the draft thinking-off mapping. Neither arm enables eager execution or HF position overrides.

```bash
uv run python -m experiments.evaluation.cli launch \
  --model-config experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/ablations/k2/model-tp2-dp1-low.yaml \
  --evalchemy-config experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/evalchemy-configs/humanevalplus.yaml \
  --evalchemy-config experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/evalchemy-configs/math500.yaml \
  --federated_cluster cw-rno2a --priority interactive --no-wait \
  --description 'Unlabeled K2 screen: compiled TP2/DP1, low reasoning on thinking-off tasks'
```

Change only the model-config basename and description to run the other three arms (`tp2-dp1-high`, `tp2-dp4-low`, and `tp2-dp4-high`). The TP2/DP4 configs otherwise match the TP2/DP1 pair, so they isolate the MoE topology change. Do not use `--version` or a policy label.

After a serving arm recovers on the full cheap benchmarks, screen it against
`tb2-ten-old-wins.yaml`. The filter is the first ten task names in sorted order
among the 30 tasks K2 solved in the September 24 Terminal-Bench archive; all
ten are present in the new dataset at ref 6. It uses one trial per task to
limit diagnostic cost. The release benchmark remains the unfiltered,
three-trial `harbor-configs/tb2-recovery.yaml`; do not report the diagnostic
subset as a policy score.

The second Terminal-Bench isolation pair uses those same ten tasks. Run
`model-tp2-dp1-eager-low.yaml` with `tb2-ten-old-wins.yaml` to change only
compiled versus eager execution. Run `model-tp2-dp1-low.yaml` with
`tb2-ten-old-wins-output8k.yaml` to change only the per-turn output cap from
16,384 to 8,192. The eager and 8k arms are diagnostic deviations, not release
settings. Keep their run descriptions explicit and omit a policy version.

The `model-tp2-dp1-high-65k-*` trio isolates the archived single-turn serving
changes on full HumanEval+ under one current grader. All three hold TP2/DP1,
high reasoning, compiled execution, and the 65,536-token server window fixed.
`high-65k-16k` is the no-HF-override control; `high-65k-16k-hf-overrides`
adds only the archived HF position overrides; `high-65k-8k` instead changes
only the generation cap. These are unlabeled diagnostics, not policy scores.

The `model-tp2-dp4-high-65k-16k` pair repeats the HF-override comparison at
the degraded TP2/DP4 topology. The two configs differ only in
`serve.hf_overrides`; run each with the canonical full HumanEval+ config and
the same launch options as the first pair. This tests whether position
overrides interact with DP4/EP8 to account for the residual historical gap.
