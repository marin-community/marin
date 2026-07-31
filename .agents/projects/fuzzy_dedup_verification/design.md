# Verify Fuzzy-Duplicate Clusters

## TL;DR

Marin will keep its current MinHash LSH and connected-components job as candidate discovery. A second Zephyr job will verify each proposed removal against full normalized text.

Only a verified match will produce `dup_doc=True`. This change prevents a false cluster from deleting all but one of its documents.

## Background

The current cluster artifact has no text, but the store treats each noncanonical cluster member as a duplicate.

Issues [#6851](https://github.com/marin-community/marin/issues/6851) and [#6854](https://github.com/marin-community/marin/issues/6854) show that this rule causes large false deletions. The [research brief](research.md) contains the rated-pair results, cluster measurements, resource results, code references, and prior art.

## Challenges

The candidate job removes text before its graph shuffle. Verification must read the normalized text column again.

The text join can use co-partitioned shards, but verification must then shuffle candidate text by `dup_cluster_id`. The current 100B candidate artifact has a cluster with more than 100,000 members.

Jaccard threshold relations are not transitive. Each removed document must pass a direct comparison against a retained representative.

The representative must also agree with global exact deduplication. A different
decision for equal content IDs can cause the two filters to remove all copies
of identical text.

## Costs / Risks

- The map-side join scans normalized text again. Only non-singleton cluster members enter the cluster shuffle.
- A hot cluster still routes to one reducer. The first version uses one representative and linear comparison cost.
- The strict rule favors precision and has low measured recall.
- The new persisted attribute type requires updates to the store, ferries, reports, validators, and consolidation configs.
- One candidate canonical can miss duplicate subgroups that do not match it.

## Design

The current [`compute_fuzzy_dups_attrs`](../../../lib/marin/src/marin/processing/classification/deduplication/fuzzy_dups.py)
will stay candidate-only. Its `FuzzyDupsAttrData` output will keep all non-singleton cluster members.

A new `verify_fuzzy_dups` module will consume `dict[str, NormalizedData]` and `FuzzyDupsAttrData`. It will produce a separate `VerifiedFuzzyDupsAttrData` artifact.

The job will validate source keys, expected basenames, sorted IDs, and unique IDs. A missing sparse candidate file will act as an empty shard.

The map stage will use a streaming two-way merge for an inner join of each
sorted normalized shard and its sorted candidate shard. A missing candidate
file will produce no candidate rows.

The join will read `id` and full `text` from each normalized shard. It will
read `id`, `dup_cluster_id`, and `is_cluster_canonical` from the matching
candidate shard.

The joined records will keep `file_idx`, `source_key`, and
`is_cluster_canonical`. A sentinel record from each input shard will make sure
that every output shard exists.

The job will assign file indices from sorted source names. It will group real
records by `dup_cluster_id`.

It will put the existing connected-components canonical first. It will fail if
a cluster has zero or more than one canonical.

The candidate canonical will be the retained representative. This direction is
the direction that the rated baseline audit measured.

If another cluster member has the same content ID, the verifier will check that
its text is equal to the representative text and write no fuzzy marker. The
global exact job will remove all but its first copy. This rule prevents exact
and fuzzy deduplication from removing all copies.

The final store will apply the global exact marker to a fuzzy representative
when that representative is not the first copy of its content ID. The old store
kept such representatives. This behavior change is intentional because the
global exact job always keeps the first copy.

The reducer will compare each remaining member directly against the representative. It will not do all-pairs comparison or connected-component closure.

The verifier will restore the pure scorer from closed PR #7591. The reference pipeline will pass an explicit, hashed `FuzzyVerificationParams` value.

The initial parameter set is:

- `ngram_size=3`.
- `minimum_member_containment=1.0`.
- `maximum_member_unique_ngrams=0`.
- `maximum_chars_per_token=10.0`.
- `under_tokenized_char_ngram_size=5`.
- `under_tokenized_minimum_char_jaccard=0.90`.

The normal rule uses case-folded whitespace token 3-gram sets. Every member 3-gram must occur in the representative.

The scorer will calculate exact token Jaccard for every comparison. Jaccard is an audit score and does not set the initial deletion decision.

When either text has above ten characters per whitespace token, the scorer will also calculate full-text character-5 Jaccard. This guard protects compressed or poorly tokenized text.

The verifier will accept a member only when the member is no longer than the representative. The containment rule and applicable character guard must also pass.

The cluster reducer will stream members and keep only the representative shingle sets in memory. Comparison cost is linear in cluster members.

Accepted members will move through a second `group_by(file_idx)` operation. Each file group will sort rows by `id` and write the matching normalized basename.

The co-partitioned output will contain rows only for accepted duplicates. Each row will contain these fields:

- `id`.
- `dup_doc=True`.
- `dup_cluster_id`.
- `dup_representative_id`.
- `dup_representative_source_key`.
- `dup_member_containment`.
- `dup_jaccard`.
- `dup_under_tokenized`.
- `dup_char_jaccard`.

The artifact stores `FuzzyVerificationParams`, including its rule version. The
output rows do not repeat this artifact-level value.

Rejected candidates will not enter the attribute files. Fixed-bin counters will record rejection reasons, containment, token Jaccard, character Jaccard, cluster size, and source.

The store and consolidation jobs will accept only `VerifiedFuzzyDupsAttrData` for fuzzy deletion. The candidate artifact cannot satisfy that type contract.

The store will replace `is_cluster_canonical` logic with a sparse `dup_doc` set. Consolidation will use `REMOVE_DOC`, `name="dup_doc"`, and `keep_if_missing=True`.

The dedup report will show candidate members, verified duplicates, acceptance rate, rejection reasons, and exact-score histograms. Reruns can change verifier parameters without a MinHash rerun.

The reference DAG will become:

`normalize → minhash → fuzzy clusters → fuzzy verification → store`

Other normalized-text attributes can still run in parallel. The verification step will depend on all normalized sources and the candidate artifact.

## Testing

Pure scorer tests will cover exact copies, strict subsets, template changes,
long tokens, and configurable thresholds.

A local Zephyr test will create one cluster with a true duplicate and a
template false positive. Only the true duplicate must receive `dup_doc=True`.
The test will set the representative through `is_cluster_canonical`.

The test will include a shard with no accepted duplicate. That shard must still get an empty Parquet file with the specified schema.

A filter-composition test will combine global exact and fuzzy markers. At least
one copy of identical text must remain. Exact copies will not get fuzzy
markers. A source-order test will make source names and source keys sort in
different orders.

Determinism tests will change worker counts and input dictionary order. The
representative and output rows must stay the same.

The rated-pair, 0.1B, and 100B gates are complete. They rejected longest-first
selection, confirmed the candidate-canonical direction, and showed that the
streaming reducer can process the current largest clusters. The
[research brief](research.md) is the single record for the measured counts,
source checks, precision, recall, task time, and memory.

## Open Questions

- Must the first version store bounded rejected-decision samples, or are aggregate counters sufficient?
- What reducer time or shuffle-byte limit will require a sharded large-cluster path before production use?
- Can a later policy improve recall without changing the safe candidate-canonical direction?
