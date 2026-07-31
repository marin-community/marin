# Verify Fuzzy-Duplicate Clusters

## TL;DR

Marin will keep its current MinHash LSH and connected-components job as candidate discovery. A second Zephyr job will verify each proposed removal against normalized text.

The second job will use saved LSH buckets to select a bounded set of local representatives. Only a direct text match will produce `dup_doc=True`.

## Background

The current cluster artifact has no text, but the store treats each noncanonical cluster member as a duplicate.

Issues [#6851](https://github.com/marin-community/marin/issues/6851) and [#6854](https://github.com/marin-community/marin/issues/6854) show that this rule causes large false deletions. The [research brief](research.md) contains the rated-pair results, cluster measurements, resource results, code references, and prior art.

## Challenges

The candidate job removes text before its graph shuffle. Verification must read the normalized text column again.

The text join can use co-partitioned shards, but verification must then shuffle candidate text by `dup_cluster_id`. The current 100B candidate artifact has a 104,490-member cluster.

Jaccard threshold relations are not transitive. Each removed document must pass a direct comparison against a retained representative.

The representative must also agree with global exact deduplication. A different
decision for equal content IDs can cause the two filters to remove all copies
of identical text.

## Costs / Risks

- The map-side join scans normalized text again. Only non-singleton cluster members enter the cluster shuffle.
- A hot cluster still routes to one reducer. The reference configuration permits two direct comparisons per member.
- The reducer keeps at most 64 representatives and 2,000,000 local representative characters per cluster.
- The strict local token and character gates limit added recall.
- The new persisted attribute type requires updates to the store, ferries, reports, validators, and consolidation configs.
- The representative limits can miss a subgroup late in a large cluster.
- The final 100B run added only 24 local matches above 27,179 canonical
  matches. This small recall gain is an explicit cost of strict and bounded
  local decisions.

## Design

The current [`compute_fuzzy_dups_attrs`](../../../lib/marin/src/marin/processing/classification/deduplication/fuzzy_dups.py)
will stay candidate-only. Its `FuzzyDupsAttrData` output will keep all non-singleton cluster members.

A new `verify_fuzzy_dups` module will consume normalized data, saved MinHash attributes, and `FuzzyDupsAttrData`. It will produce a separate `VerifiedFuzzyDupsAttrData` artifact.

The job will validate source keys, expected basenames, sorted IDs, and unique IDs. A missing sparse candidate file will act as an empty shard.

The map stage will use a streaming three-way merge. It will join each sorted
normalized shard, candidate shard, and MinHash shard. A missing candidate file
will produce no candidate rows.

The join will read `id` and full `text` from each normalized shard. It will
read `id`, `dup_cluster_id`, and `is_cluster_canonical` from the matching
candidate shard. It will read `id` and `buckets` from the MinHash shard.

The joined records will keep `file_idx`, `source_key`, `is_cluster_canonical`,
and unique sorted buckets. A sentinel record from each input shard will make
sure that every output shard exists.

The job will assign file indices from sorted source names. It will group real
records by `dup_cluster_id`.

It will put the existing connected-components canonical first. It will fail if
a cluster has zero or more than one canonical.

The candidate canonical will be the first retained representative. The reducer
will compare every non-exact member to this canonical first. Thus, all prior
canonical accept decisions stay unchanged.

If another cluster member has the same content ID, the verifier will check that
its text is equal to the representative text and write no fuzzy marker. The
global exact job will remove all but its first copy. This rule prevents exact
and fuzzy deduplication from removing all copies.

The final store will apply the global exact marker to a fuzzy representative
when that representative is not the first copy of its content ID. The old store
kept such representatives. This behavior change is intentional because the
global exact job always keeps the first copy.

After a canonical rejection, the reducer will select retained local representatives that share LSH buckets with the member.

The reducer will rank local representatives by shared bucket count. Retention
order will resolve a tie. The reducer will compare at most one local
representative after the canonical.

A local comparison must pass the full-text subset rule. It must also have
token-3-gram Jaccard of at least 0.98. If the case-folded whitespace token
sequences differ, it must also have character-13 Jaccard of at least 0.98.
Equal token sequences bypass the character rule so that whitespace-only
changes can match. These extra rules apply only to a local representative.

If no representative matches, the reducer will retain the member. The count,
document-size, and cluster-text limits control retention.

The reducer will not do all-pairs comparison or connected-component closure.
Its comparison cost is at most two times the number of noncanonical members.

The verifier will use the pure scorer from closed PR #7591. The reference pipeline will pass an explicit, hashed `FuzzyVerificationParams` value.

The base verification parameter set is:

- `ngram_size=3`.
- `minimum_member_containment=1.0`.
- `maximum_member_unique_ngrams=0`.
- `maximum_chars_per_token=10.0`.
- `under_tokenized_char_ngram_size=5`.
- `under_tokenized_minimum_char_jaccard=0.90`.

The reference local representative parameter set is:

- `maximum_comparisons_per_document=2`.
- `maximum_representatives_per_cluster=64`.
- `maximum_local_representative_chars=500000`.
- `maximum_local_representative_chars_per_cluster=2000000`.
- `minimum_local_token_ngram_jaccard=0.98`.
- `local_char_ngram_size=13`.
- `minimum_local_char_jaccard=0.98`.

The normal rule uses case-folded whitespace token 3-gram sets. Every member 3-gram must occur in the representative.

The scorer will calculate exact token-3-gram Jaccard for every comparison. Jaccard is
an audit score for canonical comparisons and a deletion rule for local
comparisons. The local character rule rejects high token-set scores from short
template additions or low-vocabulary n-gram saturation.

When either text has above ten characters per whitespace token, the scorer will also calculate full-text character-5 Jaccard. This guard protects compressed or poorly tokenized text.

The verifier will accept a member only when the member is no longer than the representative. The containment rule and applicable character guard must also pass.

The cluster reducer will stream members. It will keep the canonical and the
bounded local representative set in memory.

Accepted members will move through a second `group_by(file_idx)` operation. Each file group will sort rows by `id` and write the matching normalized basename.

The co-partitioned output will contain rows only for accepted duplicates. Each row will contain these fields:

- `id`.
- `dup_doc=True`.
- `dup_cluster_id`.
- `dup_representative_id`.
- `dup_representative_source_key`.
- `dup_representative_kind`.
- `dup_shared_lsh_buckets`.
- `dup_comparisons`.
- `dup_member_containment`.
- `dup_jaccard`.
- `dup_under_tokenized`.
- `dup_char_jaccard`.
- `dup_local_token_sequence_equal`.
- `dup_local_char_jaccard`.

The artifact stores `FuzzyVerificationParams` and `LocalRepresentativeParams`.
Each per-source entry stores its attribute directory and explicit source tag.
The output rows do not repeat these artifact-level values.

Retained candidates will not enter the attribute files. Fixed-bin counters will
record comparison results, document decisions, scores, cluster size,
representative limits, and comparisons per document.

The store and consolidation jobs will accept only `VerifiedFuzzyDupsAttrData` for fuzzy deletion. The candidate artifact cannot satisfy that type contract.

The store will replace `is_cluster_canonical` logic with a sparse `dup_doc` set. Consolidation will use `REMOVE_DOC`, `name="dup_doc"`, and `keep_if_missing=True`.

The dedup report will show candidate members, canonical matches, local matches,
retained candidates, representative limits, score histograms, and comparisons
per document. A rerun can change verifier parameters without a MinHash rerun.

The reference DAG will become:

`normalize → minhash → fuzzy clusters → fuzzy verification → store`

Other normalized-text attributes can still run in parallel. The verification
step will depend on normalized sources, MinHash attributes, and the candidate
artifact.

## Testing

Pure scorer tests cover exact copies, strict subsets, template changes, long
tokens, and configurable thresholds.

A local Zephyr test creates one cluster with a true duplicate and a template
false positive. Only the true duplicate receives `dup_doc=True`. Other tests
cover token n-gram saturation, whitespace-only changes, comparison and memory
limits, repeated content IDs, and stable selection.

The test includes a shard with no accepted duplicate. That shard still gets an empty Parquet file with the specified schema.

A filter-composition test combines global exact and fuzzy markers. At least
one copy of identical text remains. Exact copies do not get fuzzy markers. A
source-order test makes source names and source keys sort in different orders.

Determinism tests will change worker counts and input dictionary order. The
local representative and output rows must stay the same.

The single-canonical rated-pair, 0.1B, and 100B gates are complete. They rejected
longest-first selection and confirmed the candidate-canonical direction.

The local representative version kept all canonical accepts and added 24
reviewed local matches. It used two or fewer comparisons per member and
finished the 100B hot clusters without a worker failure. The final 0.1B
end-to-end replay also passed.

The [research brief](research.md) is the single record for measured counts,
source checks, precision, recall, task time, and memory.

## Open Questions

- Must the first version store bounded rejected-decision samples, or are aggregate counters sufficient?
- What reducer time or shuffle-byte limit will require a sharded large-cluster path before production use?
- Do the 64-representative and 2,000,000-character limits miss a material number of safe local matches?
