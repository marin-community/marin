# NUPA Loose grader replay

Run from a Marin checkout with `vendor/evalchemy`. The replay uses the NUPA Loose
scorer imported by that checkout's evaluator environment. Each result and the
run's `provenance.json` record `scorer_sha256`, the SHA-256 digest of the imported
scorer's Python source. The replay refuses to resume an output prefix whose
recorded scorer differs. Use a new output prefix for each scorer revision.

The grader accepts final numbers in `$...$` and `**...**`. It does not split
period-separated numeric text such as `1046.1046` into an answer, or repair
concatenated repeats such as `1818` for target `18` or `486.612486.612` for
target `486.612`. See the
[grader contract](../../../../../../vendor/evalchemy/eval/chat_benchmarks/NUPA-Loose/README.md).

The original completion archive remains the source of each trial, including
when the tracker points to an earlier replay. Each result records
`thinking_marker_trials`, the number of stored responses containing a Grug
thinking marker. This count describes the stored text, not the server's
original decoding settings.

Regrading cannot restore thinking markers removed during generation. For
Grug/Snowball NUPA cells, require fresh generations when the stored run config
does not explicitly establish `skip_special_tokens: "false"`; the marker count
alone cannot establish that boundaries were preserved. Do not promote replay
scores for these cells. The updated campaign model configs explicitly set `generation.extra_gen_kwargs.skip_special_tokens` to
`"false"`, which is retained in the evaluation run record. Publish the fresh
results before comparing those cells with other runs that retain boundaries.

For other cells, upload a tracker snapshot and select the model to replay.
Run the replay on an Iris CPU task in `cw-rno2a`:

```bash
REPLAY_MODEL=org/model
REPLAY_OUTPUT=s3://marin-us-east-02a/marin/evals/ablations/nupa-loose-20261103
uv run fsutil cp /absolute/path/TRACKER.md "$REPLAY_OUTPUT/input/TRACKER.md"
uv run iris --config lib/iris/config/marin.yaml job run --no-wait \
  --enable-extra-resources --target-cluster cw-rno2a --priority interactive \
  --cpu 8 --memory 32GB --disk 64GB --timeout 21600 \
  --job-name nupa-loose-replay-20261103 \
  -- uv run --project vendor/evalchemy --frozen --extra nupa5k-loose \
  python -m experiments.evaluation.campaigns.eval-campaign-09-25-unlabeled.ablations.nupa_loose_replay.regrade \
  --tracker-uri "$REPLAY_OUTPUT/input/TRACKER.md" --output-prefix "$REPLAY_OUTPUT" --workers 4 \
  --model "$REPLAY_MODEL"
```

Before promotion, the output prefix must contain a replay result for every
sealed NUPA cell in the tracker. Repeat the model-filtered command as needed.
The updater rereads each Grug/Snowball source run's `record.json` and leaves its
cell unchanged unless `model.config.generation.extra_gen_kwargs.skip_special_tokens`
is explicitly `"false"`. Missing records or settings require fresh generations.
Run [update_tracker](update_tracker.py) locally with `--tracker`, `--output-prefix`,
`--summary`, and `--lock` set to the campaign's `artifacts/.auto-refill.lock`;
the updater acquires the [refill tracker lock](../../analysis/harbor_verifier_audit/README.md).
The summary records both replaced cells in `updates` and models requiring fresh
generations in `regenerate`.
Each replacement cell links to its per-trial replay JSON, retaining binary
rewards and the immutable raw-response source.
