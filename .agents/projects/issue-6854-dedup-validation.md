# Issue 6854 dedup validation

## TL;DR

Compare the pinned baseline (`8f1ba5363`) and treatment (`3605aa714`) on the
same 103,716,988-document testbed. Do not accept capped connected components,
sampled false-positive review, or job success alone as evidence. Publish a
verdict after every dropped pair has a full-text label and the score, label,
report, and finelog accounting gates pass.

## Fixed inputs

- Corpus:
  `s3://marin-us-east-02a/marin/datakit/sample_100b_8ae7a94f`
- Inventory:
  `s3://marin-us-east-02a/marin/user/rav/datakit/dedup-ab/issue6854-100b-20260724-v1/inventory.json`
- Corpus size: 115 sources, 768 shards, 103,716,988 documents, and
  256,440,051,494 compressed bytes.
- MinHash parameters: 286 permutations, 26 bands, 5-grams, seed 42, and a
  500,000-character input cap.
- Baseline uses character 5-grams. Treatment uses word 5-grams and
  `canonical_star_v1` marker emission.

## Execution gates

1. Require identical MinHash input counters across arms: 768 shards,
   103,716,988 documents, 2,696,641,688 buckets, and 31,351 text truncations.
2. Require a zero-change connected-components iteration in each arm. If the
   baseline reaches round 50 with changes remaining, snapshot all cap-50 marker
   shards, resume from `metadata/cc/it_50`, and retain both outputs.
3. Require exact marker accounting. For treatment:

   ```text
   singletons_skipped + cluster_members + transitive_members_kept
       == input_documents
   duplicates_to_drop == cluster_members - canonicals
   ```

4. Validate each report artifact against the dedup counters and the JSON
   embedded in `report.html`. Check parameters, source coverage, sampled-member
   histograms, placeholders, and finite rates.

## Exhaustive false-positive audit

Run `experiments/datakit/scripts/dedup_ab_audit.py` after both primary arms
converge:

```bash
uv run python experiments/datakit/scripts/dedup_ab_audit.py \
  --baseline-dedup "$OUTPUT/baseline/dedup" \
  --treatment-dedup "$OUTPUT/treatment/dedup" \
  --baseline-minhash "$OUTPUT/baseline/minhash-combined" \
  --treatment-minhash "$OUTPUT/treatment/minhash-combined" \
  --output "$OUTPUT_AUDIT" \
  --max-workers 128
```

The audit must read all marker shards and all baseline CC iterations. Each
dropped member is joined to its canonical and scored with raw SHA-256,
character and word 5-gram Jaccard, directional containment, length,
cross-source status, bucket collisions, MinHash truncation, and baseline graph
propagation distance.

Materialize both complete raw texts for every drop:

```bash
uv run python experiments/datakit/scripts/dedup_ab_materialize.py \
  --scores-dir "$OUTPUT_AUDIT/scores" \
  --output "$OUTPUT_REVIEW" \
  --max-workers 128
```

The materializer must verify the member and canonical SHA-256 values and emit
exactly two texts and one pair per drop. Byte-identical pairs are confirmed
duplicates. Very low bidirectional word overlap is a confirmed false positive.
Every remaining pair receives a full-text semantic label. Threshold changes
must be derived from the full score distribution and checked against the
manually labeled smoke pairs before use.

`dedup_ab_review.py` is the final coverage gate. It rejects missing, extra, or
duplicate labels and requires every marker occurrence to be either a labeled
drop or the reviewed canonical of a label.

## Performance comparison

Recover every `zephyr.stage` row from the finelog archive. Sum MinHash, initial
graph construction, every executed CC iteration, and marker emission.
`cpu_time_total` is the primary cost metric. Report item-normalized CPU when
stage item counts differ. Report peak aggregate worker memory and wall time as
secondary context. A run with missing execution IDs or mismatched items is not
comparable.

## Verdict gate

The treatment is acceptable only if it reduces false-positive drops without an
unexplained loss of confirmed duplicate removals, all counters and reports are
exact, and its total worker cost and memory remain operationally acceptable.
Run the 500B testbed only if the 100B score distribution, source composition,
large clusters, or truncation cases leave a scale-dependent uncertainty.
