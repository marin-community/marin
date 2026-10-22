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

The exact six-arm Terminal-Bench grid uses the same ten-task filter and one
trial per task. Pair each model config with the indicated Harbor config; omit
`--version` when launching:

| Arm | Model config | Harbor config |
| --- | --- | --- |
| A | `model-grid-a-eager-dp1-65k-8k.yaml` | `tb2-ten-old-wins-output8k.yaml` |
| B | `model-grid-b-eager-dp4-65k-8k.yaml` | `tb2-ten-old-wins-output8k.yaml` |
| C | `model-tp2-dp1-high-65k-8k.yaml` | `tb2-ten-old-wins-output8k.yaml` |
| D | `model-grid-d-eager-dp1-65k-8k-hf-overrides.yaml` | `tb2-ten-old-wins-output8k.yaml` |
| E | `model-grid-e-eager-dp1-65k-16k.yaml` | `tb2-ten-old-wins.yaml` |
| F | `model-tp2-dp4-high-65k-16k-hf-overrides.yaml` | `tb2-ten-old-wins.yaml` |

These are unlabeled diagnostics. The eager arms require Marin's explicit
slow-serving acknowledgement, which is pinned in their model configs.

For example, launch A from the Marin checkout with:

```bash
uv run python -m experiments.evaluation.cli launch \
  --model-config experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/ablations/k2/model-grid-a-eager-dp1-65k-8k.yaml \
  --harbor-config experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/ablations/k2/tb2-ten-old-wins-output8k.yaml \
  --federated_cluster cw-rno2a --priority interactive --no-wait \
  --description 'Unlabeled K2 ten-task grid arm A'
```

Replace only the two config paths and arm letter for B–F as mapped above.

## Selected policy-compatible candidate

The matched grid closed at A 7/10, B 0/10, C 8/10, D 9/10, E 5/10, and
F 0/10. E's last trial resolved to a verifier-backed zero after one retry.
B and F each had ten agent
timeouts with substantial model output and no recorded transport or sandbox
failure in the ten audited trajectories. B differs from A only in topology,
so TP2/DP4/EP8 is sufficient to reproduce the collapse; eager execution does
not rescue it. The paired full HumanEval+ TP2/DP4 arms scored 76/164 without
HF position overrides and 75/164 with them, so those overrides are not the
main cause on the current grader.

An additional compiled TP2/DP1 arm with native HF positions, 65,536-token
serving context, and the draft policy's 16,384-token output cap scored **9/10**
on the same ten tasks. Its only zero was a scoreable 30-minute agent timeout
on `build-pov-ray`. The result is at
`s3://marin-us-east-02a/marin/evals/20261003-164550-K2-Horizon-screen-tp2-dp1-high-65k-16k-tb2-ten-old-wins-e79a/results/harbor_jobs/harbor_terminal-bench_terminal-bench-2-_e217262a908e/result.json`.
This single-trial subset is not a release score, but it does not justify a
K2-specific 8k output-cap exception.

The campaign's default K2 config uses the same tested TP2/DP1 serving setup
with 73,728 served tokens for the full single-turn Evalchemy budgets. The
`campaign.yaml` override selects `model-recommended-tb2.yaml` with the
canonical three-trial `harbor-configs/tb2-recovery.yaml`. The TB2 config
differs from the scored 9/10 diagnostic
model config only in its `thinking_off_template_kwargs` (`low` rather than
`high`), which Terminal-Bench does not request; its effective agent reasoning
remains `high`. It uses compiled TP2/DP1, native HF position settings, and the
same 65k/16k context and output limits as the release policy. Do not assign a
policy label until the full benchmark has actually run and passed conformance.
