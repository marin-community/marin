# BFCL native preference caches

The BFCL offline DPO entrypoint consumes one `RecoveryPreferenceCache` artifact.
Combine independently completed native caches with `preference_union` when a
training run should sample rows from several student collections. The union
copies token IDs and assistant masks without retokenization. The trainer's
existing single-component shuffle applies to the combined rows; it does not
assign equal weight to each input cache.

## Build a union

Pass each cache's artifact name and immutable version with `--cache`. Names are
relative to the current user's namespace; omit `users/<username>/`. Choose one
completed curated cache version per original student run. Expanded snapshots overlap their
predecessors and cannot both be included.

```bash
uv run python -m experiments.post_training.bfcl_rl.preference_union \
  --version <new-calendar-version> \
  --cache data/bfcl-rl-native-preferences-seed-<seed-a> <cache-a-version> \
  --cache data/bfcl-rl-native-preferences-seed-<seed-b> <cache-b-version>
```

The command prints the artifact graph. Add `--run` to submit the CPU stage to
Iris. Its output name is `data/bfcl-rl-native-preference-union`, under the same
user namespace. Inputs and the audited complement are explicit dependencies.

Each input must have a finished ledger, success marker, matching artifact and
selection row counts, and a selection hash bound into the ledger. Inputs must
agree on the student model, tokenizer, context limit, scoring, repeated-tool-call
policy and collection-conditions digest. Collection seed and traversal order
are outside that digest, so independent seed replicas can be combined.

Every pair must distinguish a correct branch from an incorrect branch for the
same complement task, harness and repetition. Parity tasks are rejected. An
original student run/task/harness/repetition identity may occur only once,
including when different snapshots use different archive paths. Independently
generated runs may contain the same task.

The output retains source selection-manifest locations and SHA-256 hashes,
concatenated pair provenance, and row/token totals. Row order follows the input
cache order and then each cache's original order. A failed build is not a usable
artifact; use a new version for another attempt.

## Train from an explicit cache

`experiments.post_training.bfcl_rl.native_optimize` requires `--preference-name`
and `--preference-version`. The former `--seed` option is removed. For a single-seed cache,
pass `data/bfcl-rl-native-preferences-seed-<seed>`; for a union, pass
`data/bfcl-rl-native-preference-union`.

Keep the policy checkpoint and optimizer settings explicit. A run with batch 16
and 13 updates presents 208 pairs, regardless of cache size. A larger cache reduces
repetition and may leave some pairs unseen during that run. This entrypoint does
not configure inline evaluation.

After changing collection, curation or training code, validate the complete
collection→curation→training pipeline before using it for a scaled candidate.
Local cache tests do not establish live training success or holdout improvement.
