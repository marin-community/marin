# NUPA Loose grader replay

Run from a Marin checkout with `vendor/evalchemy`. The replay uses the NUPA Loose
scorer imported by that checkout's evaluator environment. Each result and the
run's `provenance.json` record `scorer_sha256`, the SHA-256 digest of the imported
scorer's Python source. The replay refuses to resume an output prefix whose
recorded scorer differs. Use a new output prefix for each scorer revision.

The grader accepts numeric answers after sentence-ending punctuation, including
`1046.1046` when the same number precedes the period. It also accepts final
numbers in `$...$` and `**...**`. It does not repair concatenated repeats such as
`1818` for target `18` or `486.612486.612` for target `486.612`. See the
[grader contract](../../../../../../vendor/evalchemy/eval/chat_benchmarks/NUPA-Loose/README.md).

The original completion archive remains the source of each trial, including
when the tracker points to an earlier replay. Regrading cannot restore thinking
markers removed during the original generation. New local generation requests
retain special tokens by default so the scorer can separate reasoning from the
final answer.

Upload a tracker snapshot, then run the replay on an Iris CPU task in `cw-rno2a`:

```bash
REPLAY_OUTPUT=s3://marin-us-east-02a/marin/evals/ablations/nupa-loose-20261103
uv run fsutil cp /absolute/path/TRACKER.md "$REPLAY_OUTPUT/input/TRACKER.md"
uv run iris --config lib/iris/config/marin.yaml job run --no-wait \
  --enable-extra-resources --target-cluster cw-rno2a --priority interactive \
  --cpu 8 --memory 32GB --disk 64GB --timeout 21600 \
  --job-name nupa-loose-replay-20261103 \
  -- uv run --project vendor/evalchemy --frozen --extra nupa5k-loose \
  python -m experiments.evaluation.campaigns.eval-campaign-09-25-unlabeled.ablations.nupa_loose_replay.regrade \
  --tracker-uri "$REPLAY_OUTPUT/input/TRACKER.md" --output-prefix "$REPLAY_OUTPUT" --workers 4
```

After the job succeeds, run `update_tracker` locally while the campaign refill
lock is held. Each replacement cell links to its per-trial replay JSON. The JSON
retains corrected binary trial rewards for uncertainty calculations and links
back to the immutable raw-response archive.
