# Issue #6854 fuzzy-dedup false-positive validation

Status: running

Coordinating issue: https://github.com/marin-community/marin/issues/6854

## Objective

Compare the pre-fix fuzzy-dedup implementation with the proposed word-shingle,
direct-canonical-neighbor implementation on the full 115-source 100B Datakit
testbed. Classify every emitted duplicate and every disagreement for false
positives, preserve all intermediate artifacts, and use per-stage finelog
statistics for performance comparisons.

## Pinned arms

- Baseline: `8f1ba5363` (`origin/main` on 2026-07-24), MinHash artifact v2,
  character 5-grams, all connected-component members emitted.
- Treatment: `3605aa714`, MinHash artifact v3, word 5-grams, only the canonical
  and direct canonical neighbors emitted.
- Shared input:
  `s3://marin-us-east-02a/marin/datakit/sample_100b_8ae7a94f`.
- Shared parameters: 286 permutations, 26 bands, n-gram size 5, 500,000
  character cap, seed 42, 50 CC iterations with resume enabled.
- Artifact root:
  `s3://marin-us-east-02a/marin/user/rav/datakit/dedup-ab/issue6854-100b-20260724-v1`.

## Evaluation contract

1. Read every baseline and treatment duplicate marker.
2. Join every marker back to its full normalized source document.
3. Compute exact character- and word-5-gram similarity, containment, length
   ratio, source crossing, canonical relation, and graph distance.
4. Classify every marker with objective evidence. Semantically inspect every
   baseline/treatment disagreement and every case that objective checks leave
   ambiguous.
5. Verify marker/source cardinality, canonical invariants, CC convergence,
   report statistics, and report HTML data.
6. Compare matched Zephyr stages with finelog `cpu_time_total`,
   `mem_peak_bytes_max`, processed items, and processed bytes. Wall time is
   operational context only.

## Experiment log

### 2026-07-24T06:56:03Z — research isolation and initial inventory

- Created treatment worktree/branch
  `research/rav/6854-dedup-validation` at `3605aa714`.
- Created baseline worktree at `8f1ba5363`; a dedicated baseline research
  branch will carry only the launcher.
- Confirmed the testbed contains 115 normalized source artifacts with
  `/rav/datakit-6854-100b-inspect-20260724`.
- `/rav/datakit-6854-100b-inventory-20260724` failed before reading any
  parquet rows because `pq.ParquetFile` received an unentered
  `fsspec.OpenFile`. The replacement launcher enters the storage context and
  reads parquet footers only.
- The fix PR worktree remains separate and dirty only with pre-existing shared
  CI-selector review refinements. No validation harness is being added to PR
  #7591.

### 2026-07-24T07:03:10Z — exact 100B input inventory

- `/rav/datakit-6854-100b-inventory-v2-20260724` succeeded.
- Exact input size: 115 sources, 768 parquet shards, 103,716,988 documents,
  and 256,440,051,494 compressed parquet bytes.
- Inventory artifact:
  `s3://marin-us-east-02a/marin/user/rav/datakit/dedup-ab/issue6854-100b-20260724-v1/inventory.json`.
- The launcher is pinned in treatment commit `97e1c719a` and baseline commit
  `e463145bd`. Its artifact-version guard rejects a treatment invocation on
  v2 code or a baseline invocation on v3 code.
- The treatment requires DupeKit source-build mode because the published
  native wheel does not yet export `NgramKind`. The canonical
  `scripts/rust_mode.py dev` edits remain uncommitted and will be bundled only
  into treatment jobs. The local host lacks `cc`; the CoreWeave task image
  built the same source successfully in the earlier smoke.

### 2026-07-24T07:48:30Z — pinned native wheel and 0.1B smoke

- Built a Linux x86-64 CPython 3.12 ABI3 DupeKit wheel from treatment commit
  `3605aa714` with Zig. SHA-256:
  `dcb15e7c524af67078096481d916fc9feb990d60bb0e92b58682e9da6e1501d8`.
  `/rav/datakit-6854-treatment-wheel-probe-20260724` loaded
  `NgramKind.WORD` and independently reproduced the checksum in CoreWeave.
  The wheel and its temporary lock overrides remain uncommitted and must not
  enter PR #7591.
- Baseline combined-layout smoke
  `/rav/datakit-6854-ab-baseline-combined-smoke-20260724` succeeded over all
  115 sources and 103,848 documents. MinHash produced 2,700,048 buckets and
  truncated 32 documents. Fuzzy dedup emitted 13 members in 5 clusters:
  8 non-canonicals to drop and 103,835 singletons.
- Baseline MinHash execution `20260724-072214-e24a6615` reported 102.35 worker
  CPU-seconds and 628,441,088 peak worker bytes. Baseline fuzzy-dedup final
  stage execution `20260724-073448-a8a81dcc` reported 32.53 worker CPU-seconds
  and 369,868,800 peak worker bytes. These inline counters are preliminary;
  the A/B verdict will use matched finelog stage aggregates.
- Treatment wheel MinHash processed the identical 103,848 documents,
  2,700,048 buckets, and 32 truncations with 41.65 worker CPU-seconds and
  640,974,848 peak worker bytes. Its connected-components execution is
  `20260724-074353-8dc09291`.
- The first wheel smoke failed before reading data because the launcher was
  given the literal prefix `0.1b`, which contains no artifacts, instead of
  the immutable S3 testbed path. The corrected v3 smoke is
  `/rav/datakit-6854-ab-treatment-wheel-combined-smoke-v3-20260724`.
- Added an exhaustive audit pipeline that joins every marker to its normalized
  text and both arms' MinHash buckets, streams cluster members against their
  canonical, computes exact character- and word-5-gram Jaccard and
  containment, and emits an occurrence-level A/B comparison. It preserves
  source, shard, and ID references for full-text semantic review without
  copying the corpus.
- Added a structural report validator. Its first run correctly exposed a
  historical schema difference: the baseline report predates
  `transitive_members_kept` and omits that key. The validator now requires the
  historical baseline schema and the expanded treatment schema while enforcing
  exact document accounting for both.

### 2026-07-24T08:18:00Z — exhaustive 0.1B audit and semantic adjudication

- Treatment combined-layout smoke
  `/rav/datakit-6854-ab-treatment-wheel-combined-smoke-v3-20260724`
  succeeded over the same 115 sources and 103,848 documents. It emitted four
  members in two clusters: two canonicals, two documents to drop, one
  transitive member kept, and 103,843 singletons.
- `/rav/datakit-6854-treatment-report-validate-20260724` validated the full
  treatment report payload and HTML accounting: 115 sources, 103,848
  documents, four sampled members, two clusters, two drops, and one transitive
  member kept. The baseline report validator had already validated 13 sampled
  members, five clusters, and eight drops over the same input.
- The first audit attempt failed before emitting classifications because sparse
  sources use valid zero-row Parquet marker stubs with no columns. The corrected
  reader treats only zero-row files as empty and still requires `id` and
  `attributes` on non-empty files.
- `/rav/datakit-6854-dedup-audit-smoke-v2-20260724` succeeded and covered every
  marker: 13 baseline rows, four treatment rows, and 15 distinct occurrences.
  It found five baseline canonicals, eight baseline drops, two treatment
  canonicals, two treatment drops, seven baseline-drop/treatment-keep
  disagreements, one treatment-drop/baseline-keep disagreement, one shared
  drop, and six canonical-only occurrences. Baseline graph coverage was exact:
  five distance-0 canonicals, six direct distance-1 neighbors, and two
  distance-2 transitive members.
- Full-text semantic adjudication read every dropped document and its canonical:
  - All four baseline MASSIVE drops are false positives. Their shared payload
    is a function-schema catalog, but they contain different tool subsets,
    languages, requests, function names, arguments, and call IDs. Treatment
    keeps three but still drops the Spanish calendar example against a Latvian
    weather example; that shared treatment drop is also a false positive.
  - Both baseline Nemotron-code drops are false positives. One pairs distinct
    PHP manual pages (`MongoDB...getServer` versus `UI...isReadOnly`); the other
    pairs different PHP builds/environments. Treatment keeps all four.
  - The baseline StarCoder2 drop is a false positive: it removes a 1,876-line
    C++/LLVM document against a different 929-line program, including 938
    unique exact lines and a different source solution. Treatment keeps it.
  - The baseline Nemotron high-quality drop is a true positive: the 1,101-byte
    member is a literal truncated prefix of the 1,872-byte canonical. Treatment
    misses this pair.
  - The treatment-only Nemotron medium-quality drop is a true positive:
    both documents are the same low-quality template with entity slots
    substituted. Baseline misses this pair.
- On the union of smoke candidates, baseline therefore made seven false-positive
  drops and one true-positive drop; treatment made one false-positive and one
  true-positive drop. This is not a corpus-wide recall estimate because neither
  arm enumerates non-candidate pairs.
- Matched finelog statistics use exact root-job execution IDs. MinHash processed
  115 identical shard entries and 21,160 bytes: worker CPU fell from 102.35s to
  41.65s (-59.3%), while peak memory rose from 628,441,088 to 640,974,848 bytes.
  Summing the initial graph build, three CC iterations, and marker emission
  (deduplicating one identical repeated finelog END row) gives 240.46 baseline
  CPU-seconds versus 233.93 treatment CPU-seconds (-2.7%). Total MinHash plus
  dedup CPU was 342.81s versus 275.58s (-19.6%). Graph-stage item counts differ
  slightly because the graph semantics differ; only MinHash has identical work.

### 2026-07-24T08:31:00Z — full 100B A/B launched and MinHash complete

- Launched both non-preemptible full arms against the pinned 103,716,988-document
  inventory with identical resources and only the intended implementation
  difference:
  - baseline `/rav/datakit-6854-ab-baseline-100b-20260724`, submitted
    2026-07-24T08:18:53Z;
  - treatment `/rav/datakit-6854-ab-treatment-100b-20260724`, submitted within
    23 seconds of the baseline.
- Exact command shape for both arms:
  `uv run python experiments/datakit/scripts/dedup_ab_run.py run --variant
  <arm> --code-ref <pinned-sha> --max-workers 256 --dedup-parallelism 512
  --max-concurrent-sources 4 --layout combined`.
- Both MinHash stages completed all 768 input shards with identical accounting:
  103,716,988 documents, 2,696,641,688 buckets, 31,351 capped texts, and
  141,312 input-list bytes. Inline worker CPU was 84,007.55 seconds for
  baseline character 5-grams and 24,982.35 seconds for treatment word 5-grams
  (-70.3%). Peak aggregate worker memory was 3,859,361,792 versus
  3,819,208,704 bytes. Final performance claims will use the matched finelog
  records after all stages finish.
- The frequent best-effort endpoint-unregister and finelog send warnings during
  the 256-worker shutdown burst did not fail or preempt either job and did not
  change the exact output counters.
- Baseline combined and per-source smoke layouts are exactly equivalent:
  115 sources, 115 shards, 13 marker rows, and five canonical rows. Artifact:
  `s3://marin-us-east-02a/marin/user/rav/datakit/dedup-ab/issue6854-layout-verification-20260724-v1/baseline.json`.
- The corrected audit invariant run additionally rejects duplicate marker IDs
  and requires strictly sorted normalized IDs. Its score and graph passes
  reproduced the prior smoke counts, including exact graph distances 0/1/2 =
  5/6/2. `/rav/datakit-6854-dedup-audit-smoke-v3-20260724` then completed
  successfully with all 15 distinct occurrences accounted for and the same
  seven baseline-only drops, one treatment-only drop, one shared drop, and six
  canonical-only occurrences.

### 2026-07-24T09:18:00Z — directional audit correction

- Rechecked the machine adjudication against every manual smoke label before
  scaling it to the full corpus. Shorter-side containment was unsound for a
  dropped member longer than its canonical: it would have called the
  1,876-line StarCoder member redundant even though it has 938 unique exact
  lines.
- The audit now records raw-text SHA-256 identity, canonical and member
  containment separately, which side is longer, cross-source pairs, and
  MinHash truncation on either side. Only byte-identical raw text is
  machine-confirmed as a duplicate. Very low bidirectional overlap is
  machine-confirmed as a false positive; all intermediate and normalized-only
  matches remain ambiguous for full-text semantic review.
- Six focused tests cover directionality, longer dropped members,
  normalized-text containment, raw identity, conservative false positives,
  and low-Jaccard one-sided containment. The corrected smoke score pass
  classified all eight baseline and both treatment drops as ambiguous,
  matching the need for the existing full-text labels and eliminating the
  prior false automatic duplicate label.
- `/rav/datakit-6854-dedup-audit-smoke-v5-20260724` completed successfully
  after Iris retried a comparison coordinator whose first attempt wrote all
  15 records but timed out during log-service shutdown. The successful retry
  reproduced the exact comparison counters.
- The treatment per-source smoke and exact layout verifier also completed.
  Combined and per-source treatment artifacts match record-for-record across
  115 sources and 115 shards: four marker rows and two canonical rows.
  Artifact:
  `s3://marin-us-east-02a/marin/user/rav/datakit/dedup-ab/issue6854-layout-verification-20260724-v1/treatment.json`.

### 2026-07-24T09:49:00Z — full connected-components convergence

- The full graph requires more convergence rounds than the smoke, so p4 and
  later `zephyr-fuzzy-dups` executions are still connected-components
  iterations, not report aggregation. Both arms are running iteration 6.
- Exact change counts through iteration 5:
  - baseline: 922,366; 249,346; 206,516; 142,637; 114,607;
  - treatment: 182,128; 18,892; 14,942; 7,342; 6,503.
- This is already a material structural difference in the candidate graphs,
  but it is not yet a final runtime claim. Continue both arms to actual
  convergence, include every executed iteration in matched finelog totals,
  and do not confuse stage number with report work.

### 2026-07-24T10:35:00Z — exhaustive review coverage and later convergence

- Added an exact semantic-label coverage gate. It rejects missing, extra, or
  duplicate drop labels; requires raw-identity and low-overlap machine labels
  to match their computed evidence; and requires every marker occurrence to
  be covered as either a labeled member or its reviewed canonical.
- Added a distributed full-text materializer. It groups requests by normalized
  shard, reads each requested shard once, verifies member and canonical raw
  text against SHA-256 values recorded by the audit, and requires exactly two
  texts and one output pair per drop. Fourteen focused audit, coverage, and
  materialization tests pass.
- Connected-components change counts now extend through:
  - baseline iteration 8: 922,366; 249,346; 206,516; 142,637; 114,607;
    85,577; 70,279; 62,095;
  - treatment iteration 9: 182,128; 18,892; 14,942; 7,342; 6,503; 4,407;
    3,363; 3,136; 2,462.
  Neither graph has converged.
- Retained live finelog rows cover baseline iterations 2–7 and treatment
  iterations 3–9. After deduplicating identical END rows, the baseline rounds
  average 5,817 CPU-seconds over 210,424,922 items; treatment rounds average
  5,527 CPU-seconds over 207,984,482 items. Normalized CPU per item is about
  3.9% lower for treatment. Peak shard RSS is broadly flat (baseline
  623–628 MB, treatment 621–634 MB). These are interim steady-round numbers,
  not the final total; recover missing early rows from archived finelog and
  include all iterations plus marker emission before reporting.

### 2026-07-24T11:36:00Z — archive recovery and convergence through round 15

- Connected-components changes now extend through baseline iteration 13 and
  treatment iteration 15:
  - baseline: 922,366; 249,346; 206,516; 142,637; 114,607; 85,577;
    70,279; 62,095; 55,204; 49,510; 44,698; 38,120; 32,140;
  - treatment: 182,128; 18,892; 14,942; 7,342; 6,503; 4,407; 3,363;
    3,136; 2,462; 2,146; 1,521; 932; 564; 328; 192.
  Neither arm has yet recorded a zero-change convergence round.
- Latest per-round worker CPU and peak shard RSS remain stable: baseline
  iteration 13 used 5,849.23 CPU-seconds and 619,982,848 bytes; treatment
  iteration 15 used 5,518.38 CPU-seconds and 621,023,232 bytes. All completed
  graph stages still have zero failures and zero preemptions.
- A batch-priority in-cluster archive query recovered early `zephyr.stage`
  rows that had aged out of the live finelog cache. Exact Iris child submission
  times map treatment's initial graph build to
  `20260724-082900-2c45bc51` and baseline's to
  `20260724-083236-74cc35d9`. Their two-stage totals are 58,553.08 and
  57,196.64 worker CPU-seconds respectively. Treatment processed fewer graph
  items, so these raw initial-build totals are not a normalized speed claim.
- The first archive helper never ran because its workspace package scope used
  the CLI name rather than `marin-finelog`. The second reached the query but
  exposed a finelog archive-CLI limitation: CoreWeave S3 segment metadata does
  not populate the GCS-style creation-time field, causing the time-window
  prefilter to drop every file. The successful third query omitted that broken
  metadata prefilter and retained the exact execution-ID SQL predicate. These
  helper failures did not read corpus data or affect either A/B arm.
