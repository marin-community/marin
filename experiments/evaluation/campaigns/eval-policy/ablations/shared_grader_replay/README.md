# Fixed-grader replay of sealed campaign answers

These historical replays require the campaign's pre-vendoring Marin checkout,
such as `6885cabd456787794551a2e41fbff99188fedd37`. They call the fork's benchmark
`evaluate_responses` APIs, which the vendored evaluator replaced with shared
graders. The commands below use the isolated project retained in that historical
checkout; run them there to preserve the recorded grader revision.

The Evalchemy revision in `config/external/evalchemy/pyproject.toml` rejects empty math answers before an LLM judge and rejects vacuous symbolic matches between equations without free symbols. These replays hold the original model responses and extracted answers fixed. `sources.yaml` freezes the original 21-model cohort. `sources_current_20261015.yaml` freezes the 34-model FinanceBench and OlympiadBench cohort.

MATH500 (500 trials per model) and AIME24 (30 problems × 10 trials) call their native deterministic benchmark graders. OlympiadBench (30 × 10) calls its native deterministic grader and the 131,072-token MiniMax equivalence judge for unresolved answers. An exact cached MiniMax verdict is reused only when the question, reference answers, and candidate answer are unchanged from the earlier judge ablation. Newly unresolved answers go to the live judge. Failed judge requests abort a model result. Every per-model result contains all paired trial verdicts, and writes fail if the destination object already exists.

`regrade_minimax.py` serves one pinned MiniMax instance for both judged benchmarks. FinanceBench sends all 50 sealed responses to MiniMax. OlympiadBench applies its deterministic grader first and sends only unresolved answers. The provenance object reads the Evalchemy and Harbor revisions from their canonical isolated-project pins. Replayed tracker links are resolved back to the original FineStore samples while preserving the current per-trial verdicts as the comparison baseline.

Run the following from the Marin checkout with CoreWeave S3 credentials resolved by `common.sh`. The OlympiadBench command needs one `H100x8` Iris task in `cw-rno2a` with 64 CPUs, 512 GB RAM, and 600 GB disk; the script starts the local judge server using the checked-in YAML. The observed job was `/benfeuer/olympiadbench-fixed-grader-minimax-131k-20261009`. The MATH500 and AIME24 commands require no GPU.

```bash
CAMPAIGN=experiments/evaluation/campaigns/eval-policy
MODULE=experiments.evaluation.campaigns.eval-policy.ablations.shared_grader_replay
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

For newly evaluated models, freeze their sealed OlympiadBench cells in a new source manifest using the same `models` / `benchmarks` schema. Only the selected benchmark is required. Omit `--prior-prefix` when those sources have no previous MiniMax verdicts; the native grader sends all unresolved, nonempty answers to the live judge. Update those cells with `update_tracker --benchmark olympiadbench`, preserving other models and benchmarks. Keep the final analysis pending until these replay outputs are promoted. When refill is active, hold its tracker lock while calling the updater so concurrent state writes cannot replace newly published scores.

To replay the current judged cohort, freeze the tracker, validate every source, and submit the combined replay from the pinned Marin checkout. The combined replay uses a 262,144-token judge window. Although policy generation is capped at 73,664 tokens, a response tokenized by its source model can expand substantially under MiniMax's tokenizer; the larger judge window keeps the evidence intact and leaves room for the judge label.

```bash
CURRENT_SOURCES="$CAMPAIGN/ablations/shared_grader_replay/sources_current_20261015.yaml"
CURRENT_OUTPUT=s3://marin-us-east-02a/marin/evals/ablations/minimax-m3-judge-regrade-20261015-016dd78b

uv run python -m "$MODULE.freeze_tracker_sources" \
  --tracker "$TRACKER" --output "$CURRENT_SOURCES" \
  --benchmark financebench --benchmark olympiadbench

replay_python -m "$MODULE.regrade_minimax" \
  --sources "$CURRENT_SOURCES" \
  --judge-config "$CAMPAIGN/ablations/olympiadbench_judge/MiniMaxAI-MiniMax-M3-MXFP8-262k.yaml" \
  --output-prefix "$CURRENT_OUTPUT"

replay_python -m "$MODULE.update_tracker" \
  --tracker "$TRACKER" --sources "$CURRENT_SOURCES" --output-prefix "$CURRENT_OUTPUT" \
  --benchmark financebench --benchmark olympiadbench
```
