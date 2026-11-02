# Qwen3.6 τ³ context ablation

The two existing full-benchmark points differ in more than Pi's context limit.
They establish the observed regression but are not a controlled context pair.
The October arms below use the current pinned Harbor runtime, `pi-acp@0.0.33`,
`tau3-bench@1.0.1`, the same tasks, verifier, timeouts, retry rules, model
revision, thinking mode, TP2/DP2 topology, and scoring. They are unlabeled
diagnostics, not release-policy results.

| Run | Served context | Pi context | Pi output cap | Score |
| --- | ---: | ---: | ---: | ---: |
| September 17 campaign | 73,728 | 73,728 | 8,192 | [0.252](s3://marin-us-east-02a/marin/evals/20260923-220513-Qwen-Qwen3.6-35B-A3B-tau3-pi-1515/results) |
| Current campaign | 73,728 | 32,768 | 8,192 | [0.068](s3://marin-us-east-02a/marin/evals/20260930-143615-Qwen-Qwen3.6-35B-A3B-tau3-pi-d608/results) |
| 65k/16k | 65,536 | 65,536 | 16,384 | [0.202; 372/375 scored](s3://marin-us-east-02a/marin/evals/20261004-001237-Qwen-Qwen3.6-35B-A3B-tau3-65k-16k-tau3-65k-16k-f0e0/results) |
| 65k/32k | 65,536 | 65,536 | 32,768 | [Early-stopped; 217/375 scored](s3://marin-us-east-02a/marin/evals/20261004-090453-Qwen-Qwen3.6-35B-A3B-tau3-65k-32k-tau3-65k-32k-5e2a/results) |
| 131k/32k | 131,072 | 131,072 | 32,768 | [0.410; 373/375 scored](s3://marin-us-east-02a/marin/evals/20261004-001235-Qwen-Qwen3.6-35B-A3B-tau3-131k-32k-tau3-131k-32k-a80a/results) |
| 131k/65k | 131,072 | 131,072 | 65,536 | pending |
| 262k/65k | 262,144 | 262,144 | 65,536 | pending |

The first launch of all three arms failed because Harbor resolved the pinned
ACP registry shorthand during each trial setup. Its registry cache was written
but not read, and repeated GitHub requests returned HTTP 403 rate limits. The
replacement configs embed the exact `pi-acp@0.0.33` registry record from
`agentclientprotocol/registry@993fab130b3475e9c1c745ca6820185219698f33`.
This keeps the installed agent version fixed and removes registry network
requests from trial setup. The failed attempts are not benchmark scores.

The old score used Harbor `6543ab6cf5562e690203ddffb40efac74ef4f45b`;
the current score used `2666d6526477ae3e46030a8dc4f3f2c68fd7a84f`.
The diagnostic arms use the current campaign Harbor pin. Their model and Harbor YAMLs are
paired by suffix; each file is an immutable launch input. The 65k/16k to
65k/32k pair isolates the per-turn output cap, but the 65k/32k run stopped
with 217 scoreable trials and 49 unscored outcomes after repeated τ³ runtime
service-unavailable errors. Its partial mean of 0.195 is not a benchmark score.
The 131k/32k to 131k/65k pair isolates the output cap; the 131k/65k to
262k/65k pair isolates total context. Comparison with the existing 32k/8k point is
suggestive because its served window remained 73,728 tokens and its Harbor
pin differs.

Run each arm from the Marin worktree with `--no-wait` and no policy version:

```bash
for arm in 131k-65k 262k-65k; do
  uv run python -m experiments.evaluation.cli launch \
    --model-config "experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/ablations/tau3_context/model-${arm}.yaml" \
    --harbor-config "experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/ablations/tau3_context/tau3-${arm}.yaml" \
    --federated_cluster cw-rno2a --priority interactive --no-wait \
    --description "Unlabeled Qwen3.6 tau3 context ablation ${arm}"
done
```

Interpret a plateau only after comparing scored-trial coverage, retry and
timeout counts, and the effective config snapshot for every arm. If a 32k
output cap prevents any arm from serving, record that as a feasibility limit
instead of a zero benchmark score.
