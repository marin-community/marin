# Background Research Brief

- Effort: Low.
- Stop rule: Stop when the current code and completed 100B audit give a clear job boundary.
- Date: 2026-07-31.

## Question

Can Marin keep the current MinHash LSH and connected-components job, then verify candidate cluster members in a second Zephyr job?

## Current Marin Context

The current job writes every non-singleton member to a sparse, co-partitioned cluster artifact. It does not read normalized text during cluster formation.

[`fuzzy_dups.py`](../../../lib/marin/src/marin/processing/classification/deduplication/fuzzy_dups.py)
builds `FuzzyDupsAttrData`. Each row contains `id`, `dup_cluster_id`, and `is_cluster_canonical`.

[`fuzzy_dups.py`](../../../lib/marin/src/marin/processing/classification/deduplication/fuzzy_dups.py)
uses all MinHash collisions as graph edges. Connected components then select the minimum normalized ID as the canonical.

Normalized and candidate attribute shards both use sorted IDs. A two-way merge
can join them without a shuffle.

[`build_copartitioned_shards`](../../../lib/marin/src/marin/datakit/copartitioned.py)
gives each normalized shard a stable source key, file index, basename, and output path.

The current store treats `is_cluster_canonical=False` as a deletion decision. [`datakit_store.py`](../../../experiments/datakit/store/datakit_store.py)
therefore removes candidates without an exact text comparison.

The reference pipeline sends `FuzzyDupsAttrData` directly to the store. [`reference_pipeline.py`](../../../experiments/datakit/reference_pipeline.py)
has a clear location for a verification step between cluster formation and the store.

Global exact deduplication selects the first source, shard, and row for each content ID. [`global_exact_dedup.py`](../../../experiments/datakit/global_exact_dedup.py)
sorts sources by source name and uses file index for retained-copy priority.

## Internal Prior Work

[Issue #6854](https://github.com/marin-community/marin/issues/6854) records whole-source deletion from very large false clusters. The largest measured cluster had 859,091 members.

[Issue #6851](https://github.com/marin-community/marin/issues/6851) records 743,779 distinct template-based documents in one cluster. Shared template text dominated the MinHash signature.

[Closed PR #7591](https://github.com/marin-community/marin/pull/7591) changed candidate formation and added exact verification in the same job. The new design keeps candidate formation unchanged.

The 100B audit covered 103,716,988 documents from 115 sources. The current character-shingle baseline had a measured semantic false-positive rate of 63.617%.

The word-shingle treatment in PR #7591 reduced that rate to 52.149%. Candidate parameter changes alone did not make deletion safe.

The selected full-text rule accepted 689 true duplicates and one label disagreement from 5,410 resolved treatment pairs. Measured precision was 99.855%, and measured recall was 27.527%.

The same rule accepted 23,362 of 155,212 materialized treatment candidates. It accepted no measured candidates from the three primary wipeout sources.

The exact scorer processed 1.55 billion pair characters in 353.53 worker CPU-seconds. Peak worker memory was 705 MB.

The closed PR already contains a pure `fuzzy_verification.py` scorer. The scorer computes token containment, token Jaccard, and a character-Jaccard guard.

## External Prior Art

[NearDup](https://arxiv.org/abs/2107.06499) uses MinHash for candidate search and token edit similarity for edge verification.

[OLMo 3](https://arxiv.org/abs/2512.13961) uses token 5-gram MinHash candidates. It computes exact token 3-gram Jaccard before document removal.

[Duplodocus](https://github.com/allenai/duplodocus/blob/02cf2f74334d4d87cdec8bec24238cc8fdbc3d95/src/true_jaccard.rs)
rebuilds token n-gram sets and computes exact Jaccard for candidate pairs.

These systems separate candidate search from exact verification. Marin can use the same boundary without a change to its current cluster job.

## Negative / Failed Leads

Higher LSH thresholds reduce low-similarity candidates, but they do not reject template-based documents with high shared-token similarity.

Word 5-grams reduce false candidates, but the measured treatment still had a 52.149% semantic false-positive rate.

Connected-component closure cannot be a deletion rule. Jaccard threshold relations are not transitive.

The minimum-ID canonical can be shorter than its members. The prior audit rejected 77,972 candidates because the proposed member was longer.

All-pairs verification has quadratic comparison cost. The 859,091-member cluster makes this policy unsuitable as the first production version.

## Evidence Map

### Claim: Candidate clusters and deletion decisions need different artifacts

- Support:
  - Issues #6851 and #6854 show that cluster membership has poor deletion precision.
  - The strict full-text rule measured 99.855% precision on resolved treatment pairs.
- Contradictions:
  - The strict rule measured only 27.527% recall.
- Directness to Marin: The measurements use Marin candidate pairs and normalized text.
- Confidence: High for precision, and medium for recall.
- Action: Keep `FuzzyDupsAttrData` as candidate data and add `VerifiedFuzzyDupsAttrData`.

### Claim: One longest representative gives a safe first implementation

- Support:
  - The strict rule rejects a member that is longer than its representative.
  - Longest-first selection removes this avoidable rejection reason.
  - One comparison per member gives linear comparison cost.
- Contradictions:
  - One representative can miss a duplicate subgroup inside a false cluster.
- Directness to Marin: The current minimum-ID canonical caused 77,972 length rejections.
- Confidence: Medium until a new fixed-sample comparison is complete.
- Action: Use longest-first selection and measure recall before a multiple-representative policy.

### Claim: Existing Zephyr and Datakit contracts support the second job

- Support:
  - Sorted IDs support a streaming text-to-attribute merge.
  - `group_by` supports a cluster shuffle and a second file-index shuffle.
  - `build_copartitioned_shards` supplies stable output paths.
- Contradictions:
  - A very large cluster still routes to one reducer.
- Directness to Marin: These are the current APIs at commit `656d77bff`.
- Confidence: High for correctness, and medium for production task time.
- Action: Add a streaming reducer and measure the largest cluster before production use.

## Recommended Next Experiments

### 1. Compare minimum-ID and longest-first representatives

- Minimum experiment: Re-score the fixed 5,410 resolved treatment pairs by candidate cluster.
- Baseline/control: The minimum-ID canonical from PR #7591.
- Expected signal: Longest-first selection keeps precision and increases accepted true duplicates.
- Falsifier: Precision falls below 99% or recall does not increase.
- Cost/risk: Low because the materialized pair corpus already contains full text.
- Sources: PR #7591 and the 100B audit artifacts.

### 2. Measure the second job on known hot clusters

- Minimum experiment: Run the join and cluster reducer on the 100B testbed.
- Baseline/control: The prior exact scorer resource totals.
- Expected signal: Memory stays bounded, and task time scales linearly with cluster members.
- Falsifier: One hot reducer exceeds the Zephyr task limit or its local disk limit.
- Cost/risk: Medium because the job reads normalized text and shuffles candidate text.
- Sources: Issues #6851 and #6854.

## Hypothesis Queue Update

- Add: Longest-first verification improves recall without a precision decrease.
- Add: A streaming one-representative reducer is sufficient for the first production run.
- Revise: MinHash parameters control retrieval cost and recall, but they do not control final deletion precision.
- Falsify / stop: Do not use connected-component membership as a direct deletion decision.
- Promote: Require a direct full-text verification result for each removed document.

## Source Ledger

| Source | Type | Location | Claim used for | Confidence | Notes |
|---|---|---|---|---|---|
| Marin code | Code | `fuzzy_dups.py` at `656d77bff` | Candidate artifact contract | High | Current `origin/main` |
| Marin code | Code | `copartitioned.py` at `656d77bff` | Map-side join contract | High | Current `origin/main` |
| Marin code | Code | `datakit_store.py` at `656d77bff` | Current deletion behavior | High | Current `origin/main` |
| #6854 | GitHub issue | Whole-source wipeouts | Cluster failure sizes | High | Production artifact analysis |
| #6851 | GitHub issue | Template over-merge | Template failure mode | High | Production artifact analysis |
| #7591 | Pull request | Closed replacement | Exact-rule measurements | Medium | Model-assisted labels |
| NearDup | Paper | arXiv 2107.06499 | Candidate verification | Medium | Different corpus |
| OLMo 3 | Paper | arXiv 2512.13961 | Exact Jaccard stage | Medium | Different pipeline |
| Duplodocus | External code | Pinned GitHub source | Exact Jaccard implementation | Medium | Different runtime |

## Handoff

- Suggested issue `Prior work` block: Candidate parameter changes reduced errors but did not make cluster membership a safe deletion decision.
- Suggested logbook entry: Add a second Zephyr job that verifies full text and writes sparse `dup_doc=True` attributes.
- Open questions: Large-cluster task limits, rejected-decision retention, and a future multiple-representative policy.
- Stop reason: The current code and completed audit support one clear design.
