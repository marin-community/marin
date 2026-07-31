# Background Research Brief

## TL;DR

Keep the current candidate job, but do not use cluster membership as a deletion
decision. Use the current candidate canonical as the retained document, and
require a direct full-text check for each fuzzy removal.

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

[Issue #6854](https://github.com/marin-community/marin/issues/6854) records whole-source deletion from very large false clusters. An exact scan of the current 100B candidate artifact found a largest cluster of 104,490 members.

[Issue #6851](https://github.com/marin-community/marin/issues/6851) records 743,779 distinct template-based documents in one cluster. Shared template text dominated the MinHash signature.

[Closed PR #7591](https://github.com/marin-community/marin/pull/7591) changed candidate formation and added exact verification in the same job. The new design keeps candidate formation unchanged.

The 100B audit covered 103,716,988 documents from 115 sources. The current character-shingle baseline had a measured semantic false-positive rate of 63.617%.

The word-shingle treatment in PR #7591 reduced that rate to 52.149%. Candidate parameter changes alone did not make deletion safe.

On 7,508 resolved baseline pairs from the current candidate rule, the selected
full-text rule accepted 233 true duplicates and no false positives. Its
observed precision was 100%, with a 98.38% Wilson lower bound. Its recall was
9.05%.

The treatment candidates from PR #7591 are a secondary result. On 5,410
resolved treatment pairs, the same rule accepted 690 true duplicates and two
false positives. Its precision was 99.71%, and its recall was 27.57%.

The same rule accepted 23,362 of 155,212 materialized treatment candidates. It accepted no measured candidates from the three primary wipeout sources.

The exact scorer processed 1.55 billion pair characters in 353.53 worker CPU-seconds. Peak worker memory was 705 MB.

The closed PR already contains a pure `fuzzy_verification.py` scorer. The scorer computes token containment, token Jaccard, and a character-Jaccard guard.

The corrected 0.1B run reused all 13 current candidate members. It rejected all
eight noncanonical members in 68.51 worker CPU-seconds, with 474 MB peak worker
memory. Direct review confirmed that seven comparisons were clear template or
topic false matches. The remaining notebook pair had 97.09% containment but
five unique member 3-grams, which shows the expected recall cost of the strict
rule.

The 100B run reused 1,513,510 current candidate members in 505,876 clusters.
It made 1,007,455 direct text comparisons and accepted 27,179, or 2.70%. It
deferred 179 equal-ID copies to global exact deduplication. The new job wrote
97.30% fewer fuzzy removal markers than the candidate-only rule.

The three largest current candidate clusters had 104,490, 34,060, and 15,106
members. Earlier issue measurements include whole-source removal totals and
must not be read as current cluster sizes.

The verifier accepted no direct comparison from `massive_function_calling`,
`starcoder2/ir_cpp`, or `starcoder2/ir_python`. These sources include the
template and generated-code failure modes from the issue review.

The 512-worker 100B run completed in 20 minutes and 14 seconds. It processed
31.75 billion candidate-text characters, used 6,587.96 worker CPU-seconds, and
reached 9.67 GB peak worker memory. The hot reducer completed in 13 minutes
with Zephyr external sort.

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

The minimum-ID canonical can be shorter than its members. This lowers recall,
but the rated audit shows that changing to the longest member is unsafe. A
two-document longest-first proxy on baseline pairs had only 51.91% precision.
It often retained a synthetic expansion and removed the source document.

All-pairs verification has quadratic comparison cost. The 104,490-member
cluster makes this policy unsuitable as the first production version.

## Evidence Map

### Claim: Candidate clusters and deletion decisions need different artifacts

- Support:
  - Issues #6851 and #6854 show that cluster membership has poor deletion precision.
  - The strict full-text rule accepted no false positives on 7,508 resolved baseline pairs.
- Contradictions:
  - The strict rule measured only 9.05% recall on these baseline pairs.
- Directness to Marin: The measurements use Marin candidate pairs and normalized text.
- Confidence: High for precision, and medium for recall.
- Action: Keep `FuzzyDupsAttrData` as candidate data and add `VerifiedFuzzyDupsAttrData`.

### Claim: The existing candidate canonical gives a safe first implementation

- Support:
  - The rated baseline audit used this comparison direction.
  - The strict rule accepted no rated false positives in this direction.
  - One comparison per member gives linear comparison cost.
- Contradictions:
  - One representative can miss a duplicate subgroup inside a false cluster.
- Directness to Marin: The current minimum-ID canonical caused many length
  rejections, but longest-first selection had only 51.91% proxy precision.
- Confidence: High for the selected direction, and low for future
  representative changes.
- Action: Use the existing candidate canonical. Do not use longest-first selection.

### Claim: Existing Zephyr and Datakit contracts support the second job

- Support:
  - Sorted IDs support a streaming text-to-attribute merge.
  - `group_by` supports a cluster shuffle and a second file-index shuffle.
  - `build_copartitioned_shards` supplies stable output paths.
- Contradictions:
  - A very large cluster still routes to one reducer.
- Directness to Marin: These are the current APIs at commit `9ca32c43e`.
- Confidence: High for correctness and initial production feasibility.
- Action: Keep the streaming reducer and monitor the first production run for a hot task.

## Recommended Next Experiments

### 1. Reject longest-first representative selection

- Minimum experiment: Re-score the fixed 7,508 resolved baseline pairs with the
  longer document as the pair representative.
- Baseline/control: The existing connected-components canonical.
- Result: Proxy precision fell from 100% observed to 51.91%.
- Cause: Synthetic expansions often contain the source document but are not the
  same semantic document.
- Decision: Keep the existing connected-components canonical.
- Sources: The 100B audit artifacts.

### 2. Measure the second job on known hot clusters

- Minimum experiment: Run the join and cluster reducer on the 100B testbed.
- Baseline/control: The prior exact scorer resource totals.
- Result: The job finished in 20 minutes and 14 seconds with no failure or
  preemption. The hot reducer finished in 13 minutes with Zephyr external sort,
  and peak worker memory was 9.67 GB.
- Decision: The streaming one-representative reducer is feasible for the first
  production version. Monitor the hot task before a later sharding design.
- Cost/risk: Medium because the job reads normalized text and shuffles candidate text.
- Sources: Issues #6851 and #6854.

## Hypothesis Queue Update

- Falsify / stop: Longest-first verification is safe.
- Add: A streaming one-representative reducer is sufficient for the first production run.
- Revise: MinHash parameters control retrieval cost and recall, but they do not control final deletion precision.
- Falsify / stop: Do not use connected-component membership as a direct deletion decision.
- Promote: Require a direct full-text verification result for each removed document.

## Source Ledger

| Source | Type | Location | Claim used for | Confidence | Notes |
|---|---|---|---|---|---|
| Marin code | Code | `fuzzy_dups.py` at `9ca32c43e` | Candidate artifact contract | High | Current `origin/main` |
| Marin code | Code | `copartitioned.py` at `9ca32c43e` | Map-side join contract | High | Current `origin/main` |
| Marin code | Code | `datakit_store.py` at `9ca32c43e` | Current deletion behavior | High | Current `origin/main` |
| #6854 | GitHub issue | Whole-source wipeouts | Cluster failure sizes | High | Production artifact analysis |
| #6851 | GitHub issue | Template over-merge | Template failure mode | High | Production artifact analysis |
| #7591 | Pull request | Closed replacement | Exact-rule measurements | Medium | Model-assisted labels |
| NearDup | Paper | arXiv 2107.06499 | Candidate verification | Medium | Different corpus |
| OLMo 3 | Paper | arXiv 2512.13961 | Exact Jaccard stage | Medium | Different pipeline |
| Duplodocus | External code | Pinned GitHub source | Exact Jaccard implementation | Medium | Different runtime |

## Peer Review Result

Two independent full-diff reviews found no error in the direct-comparison rule,
candidate-canonical selection, co-partitioned join, sentinel path, or
determinism.

The reviews found and fixed an implicit file-index lookup, report accounting for
equal-ID copies, unused artifact fields, and counters that the report did not
show. The writer now uses an explicit file-index map. The report separates
direct comparisons from copies sent to global exact deduplication, derives both
from verifier decisions, and shows per-source decisions.

The independent testbed merge remains separate from the production join. This
is intentional because the testbed checks persisted inputs and outputs without
calling the production merge.

## Handoff

- Suggested issue `Prior work` block: Candidate parameter changes reduced errors but did not make cluster membership a safe deletion decision.
- Suggested logbook entry: Add a second Zephyr job that verifies full text and writes sparse `dup_doc=True` attributes.
- Open questions: Large-cluster task limits, rejected-decision retention, and a future multiple-representative policy.
- Stop reason: The current code and completed audit support one clear design.
