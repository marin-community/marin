# Fixed-grader replay of sealed campaign answers

The Evalchemy revision in `config/external/evalchemy/pyproject.toml` rejects empty math answers before an LLM judge and rejects vacuous symbolic matches between equations without free symbols. This replay holds the original model responses and extracted answers fixed. `sources.yaml` freezes the sealed source for each of the 21 tracker models. FinanceBench is listed in that inventory but uses a separate direct LLM-judge path and is not regraded here.

MATH500 (500 trials per model) and AIME24 (30 problems × 10 trials) call their native deterministic benchmark graders. OlympiadBench (30 × 10) calls its native deterministic grader and the 131,072-token MiniMax equivalence judge for unresolved answers. An exact cached MiniMax verdict is reused only when the question, reference answers, and candidate answer are unchanged from the earlier judge ablation. Newly unresolved answers go to the live judge. Failed judge requests abort a model result. Every per-model result contains all paired trial verdicts, and writes fail if the destination object already exists.

Run the following from the Marin checkout with CoreWeave S3 credentials resolved by `common.sh`. The OlympiadBench command needs one `H100x8` Iris task in `cw-rno2a` with 64 CPUs, 512 GB RAM, and 600 GB disk; the script starts the local judge server using the checked-in YAML. The observed job was `/benfeuer/olympiadbench-fixed-grader-minimax-131k-20261009`. The MATH500 and AIME24 commands require no GPU.

```bash
CAMPAIGN=experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled
MODULE=experiments.evaluation.campaigns.eval-campaign-09-25-unlabeled.ablations.shared_grader_replay
SOURCES="$CAMPAIGN/ablations/shared_grader_replay/sources.yaml"
OUTPUT=s3://marin-us-east-02a/marin/evals/ablations/shared-grader-fixes-20261009-958cdb80
PRIOR=s3://marin-us-east-02a/marin/evals/ablations/olympiadbench-minimax-m3-judge-20261009-guided-choice-thinking
replay_python() {
  uv run --project config/external/evalchemy --with-editable lib/iris --with-editable lib/marin python "$@"
}

source "$CAMPAIGN/common.sh"
resolve_coreweave_credentials
for BENCHMARK in math500 aime24; do
  replay_python -m "$MODULE.regrade_deterministic" \
    --sources "$SOURCES" --benchmark "$BENCHMARK" --output-prefix "$OUTPUT"
done
replay_python -m "$MODULE.regrade_olympiadbench" \
  --sources "$SOURCES" --prior-prefix "$PRIOR" \
  --judge-config "$CAMPAIGN/ablations/olympiadbench_judge/MiniMaxAI-MiniMax-M3-MXFP8-131k.yaml" \
  --output-prefix "$OUTPUT"
replay_python -m "$MODULE.update_tracker" \
  --tracker "$TRACKER" --sources "$SOURCES" --output-prefix "$OUTPUT"
```

Set `TRACKER` to the tracker to update before running the final command. The three `provenance-*.json` objects under `OUTPUT` include the source manifest and its hash. The OlympiadBench provenance also includes the exact judge YAML, request constraint, and prior-ablation location. The new per-trial objects are under `results/<benchmark>/<model>.json`. The tracker updater verifies source identity, complete trial counts, no OlympiadBench judge failures, and each current score/link before replacing a cell.
