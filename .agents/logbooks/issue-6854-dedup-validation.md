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

### 2026-07-24T13:52:00Z — treatment convergence and report validation

- The treatment graph converged exactly after 24 connected-components
  iterations. Changes in iterations 16–24 were 94; 97; 52; 65; 30; 32; 5; 4;
  0. One iteration-16 worker exited, but Zephyr recovered all 512 tasks without
  a failed or retried stage and the final counters are exact.
- Treatment marker emission accounted for every input document:
  103,388,382 singletons, 297,446 emitted cluster members, and 31,160
  transitive members kept sum to 103,716,988 documents. The markers contain
  142,234 canonicals and 155,212 drops.
- The treatment report completed, and an independent validator read the
  manifest and report data rather than trusting job success alone. It verified
  the 103,716,988-document total, all six dedup counters, the WORD/5-gram
  MinHash parameters, a 29,780-member report sample, and a renderable HTML
  report. Artifact:
  `s3://marin-us-east-02a/marin/user/rav/datakit/dedup-ab/issue6854-100b-20260724-v1/validation/treatment-report.json`.
- Baseline changes through iteration 24 are 922,366; 249,346; 206,516;
  142,637; 114,607; 85,577; 70,279; 62,095; 55,204; 49,510; 44,698;
  38,120; 32,140; 26,468; 21,103; 16,774; 14,257; 13,627; 12,502;
  11,735; 11,671; 13,915; 14,924; 16,472. The late increase is a propagation
  wave, not convergence. Iteration 25 is running; the baseline must reach a
  true zero-change round before its drop set is used for false-positive
  adjudication.

### 2026-07-24T14:23:00Z — baseline rounds 25–28 and audit hard gates

- Baseline changes continued downward after the propagation peak: 15,708;
  12,690; 9,673; 7,196 in iterations 25–28. Iteration 29 is queued. Every
  completed CC stage remains successful with no failed stage attempt.
- Added and tested a cap-preserving continuation path on the pinned baseline
  branch. It copies every marker shard to a self-contained cap-50 artifact,
  verifies each server-side copy by byte size, and then resumes the existing
  `metadata/cc` state. MinHash does not need to run again if the 50-round cap is
  reached.
- The full audit now discovers all contiguous CC iterations instead of stopping
  at 50. It rejects a final shard containing any `changed=True` node, so a
  nonconverged artifact cannot enter the primary false-positive comparison.
- Score coverage must equal each arm's exact `cluster_members` and
  `cluster_members - canonicals` counters. Comparison categories must partition
  every drop in both arms. Eighteen audit, materialization, and semantic-label
  coverage tests pass.
- The acceptance gates and reproducible commands now live in
  `.agents/projects/issue-6854-dedup-validation.md`.

### 2026-07-24T15:45:00Z — second baseline wave and exhaustive census smoke

- Baseline changes through iteration 38 are now 922,366; 249,346; 206,516;
  142,637; 114,607; 85,577; 70,279; 62,095; 55,204; 49,510; 44,698;
  38,120; 32,140; 26,468; 21,103; 16,774; 14,257; 13,627; 12,502;
  11,735; 11,671; 13,915; 14,924; 16,472; 15,708; 12,690; 9,673; 7,196;
  5,566; 4,807; 4,319; 3,047; 2,141; 1,848; 2,406; 2,963; 2,138;
  1,222. The rise in rounds 35–36 is a second propagation wave. Round 39 is
  running, and the primary comparison remains gated on a zero-change round.
- The current-schema smoke audit and its exhaustive census both completed.
  The census accounted for all 17 score rows and all 15 comparison rows. Its
  categories exactly partition the occurrences: seven baseline-only drops,
  one shared drop, one treatment-only drop, and six canonical-only rows. The
  baseline has five canonicals and eight drops; treatment has two of each.
- The audit can now inspect an explicitly pinned nonconverged baseline
  iteration while reading markers from a separately preserved cap artifact.
  Primary audits still reject any final shard with changed nodes. This keeps a
  possible old cap-50 result distinct from the true-convergence comparison.
- Full-text review labels now require the score and pair artifacts, verify both
  persisted texts against the audited SHA-256 values, and cover exactly every
  drop and every marker. The audit, materialization, census, and review suite
  passes all 25 focused tests.

### 2026-07-24T16:42:00Z — persisted full-text smoke coverage

- Materialized the current-schema smoke review corpus from its exact score
  artifact. The distributed job accounted for all 10 drop requests, both
  complete texts for every request, and all 10 output pairs.
- Bound the historical manual decisions to the current member and canonical
  locations, then rehashed the persisted complete texts. The independent
  validator covered all 17 marker occurrences and all 10 drops with no
  missing, extra, or duplicate labels.
- The smoke result remains seven false positives and one true duplicate among
  eight baseline drops, versus one false positive and one true duplicate among
  two treatment drops. All labels use full-text semantic review; this smoke
  result validates the workflow but is not evidence substituted for the full
  100B adjudication.
- The focused audit, census, materialization, label-binding, and coverage suite
  now passes all 30 tests. Baseline connected-components changes in iterations
  39–44 were 682, 480, 251, 224, 287, and 149. Iteration 45 is running; the
  full audit remains gated on a zero-change iteration.

### 2026-07-25T00:08:00Z — baseline true convergence and full audit launch

- The default baseline run reached its 50-iteration ceiling with four changed
  rows still present. Before resuming it, copied and size-verified all 768
  marker shards into the immutable `dedup-cap50` artifact. The snapshot is
  91,994,385 bytes.
- Resumed from `metadata/cc/it_50` without recomputing MinHash. Iterations
  51–53 had one, one, and zero changes, so the baseline converged exactly at
  iteration 53.
- The converged artifact has 1,513,510 cluster members, 505,876 canonicals, and
  1,007,634 drops. An independent report validator again accounts for all
  103,716,988 input documents and verifies the report parameters, sampled
  histograms, embedded JSON, and HTML structure.
- Compared every capped and converged marker across all 768 shards. Both
  artifacts have 1,513,510 markers; 1,513,508 are byte-for-byte equal at the
  attribute level. Exactly two noncanonical `swe-zero-12m` documents change
  cluster ID, with no marker additions, removals, or canonical-role changes.
  Both old and new member/canonical relationships remain subject to full-text
  semantic review.
- Launched the primary full audit against the converged baseline and treatment.
  Its score pass reads all 768 co-partitioned normalized, marker, and MinHash
  shard sets. The 34 focused audit, census, full-text, coverage, cap-diff, and
  archived-metrics tests pass.

### 2026-07-25T00:28:00Z — exhaustive adjudication routing and final gate

- The hash-verifying machine pass completed on the current-schema smoke review
  corpus. It accounted for all ten pairs and conservatively routed all ten to
  semantic review, matching the existing full-text judgments. It did not
  auto-label any truncated or ambiguous pair.
- Added a distributed final-adjudication gate. It rehashes every persisted
  member and canonical text before shuffling compact evidence, requires one
  exact machine decision per pair, requires a bound semantic decision exactly
  when the machine pass requested one, and rejects tampered identities,
  hashes, labels, methods, or evidence.
- A second distributed join independently requires every scored drop to occur
  once as a labeled member and every scored canonical to be referenced by at
  least one labeled member. Global and per-arm marker, drop, pair, semantic,
  and canonical-reference counters must match the immutable audit artifacts.
  The 35 focused audit, materialization, routing, review, and finalization tests
  pass. Snapshot: `66b30796b`.
- The full score pass is at 127/128 reduce shards. The remaining shard is a
  genuine data-skew case, not a stalled worker: a thread profile repeatedly
  shows the subprocess active with the GIL in exact shingle construction.
  CPU usage increased from 1,027 to 1,378 seconds during monitoring, while
  current memory stayed near 15.9 GB and peak memory stayed at 19.7 GB under
  the 24 GiB allocation. The shard remains in the audit rather than being
  sampled or skipped.

### 2026-07-25T00:50:00Z — complete score, distance, and A/B audit

- The full audit completed successfully. It scored all 1,810,956 marker rows,
  including the skewed shard, and computed a baseline graph distance for all
  1,513,510 baseline markers. The largest finite propagation distance is 52,
  consistent with convergence in round 53.
- The occurrence comparison accounts for all 1,534,372 rows: 863,859 are
  baseline-drop/treatment-keep, 11,437 are treatment-drop/baseline-keep,
  143,775 are dropped by both, and 515,301 are canonical-only. These categories
  reproduce the exact drop totals of 1,007,634 and 155,212.
- The baseline-only attribution partition is exact: 475,960 direct word-ngram
  changes, 387,718 combined direct and transitive changes, 77 transitive-only
  changes, 104 canonical or graph changes, and 670,513 rows where a
  baseline-only attribution is not applicable.
- The exhaustive census confirms 414,700 strong false-positive candidates,
  592,755 ambiguous drops, and 179 byte-identical drops in the baseline. The
  treatment has 155,033 ambiguous drops, 179 byte-identical drops, and no
  strong false-positive candidates. Exact clean text occurs in 3,181 baseline
  drops and 3,241 treatment drops; it is not treated as raw identity.
- The current semantic-batch loader revalidated every pair reference in the
  ten-row smoke artifact against its persisted Parquet row. All ten references
  are unique and reproduce the expected machine decision from the complete
  texts. All 50 focused A/B validation tests pass.
- Launched exhaustive materialization of 1,162,846 dropped pairs. It must
  retrieve and hash-check 2,325,692 complete texts before machine labeling can
  start. No pair is sampled or omitted.

### 2026-07-25T01:05:00Z — complete materialization and adjudication routing

- Full-text materialization completed with exact coverage: 1,162,846 drop
  requests produced 2,325,692 hash-verified texts and 1,162,846 pairs. The
  corpus contains 57,913,756,011 raw characters. The worker stage used
  3,601.53 CPU-seconds and peaked at 7,786,008,576 bytes on its largest shard.
- The machine-label pass rehashed every persisted text and reproduced all
  1,162,846 pair identities. It confirmed 358 byte-identical pairs as true
  duplicates and 407,207 complete, nontruncated baseline pairs as strong false
  positives. It routed the remaining 755,281 pairs to semantic adjudication:
  600,248 baseline and 155,033 treatment.
- Retrieved the exact converged baseline and treatment report HTML artifacts
  and verified their SHA-256 values against an in-cluster read. Both render
  successfully in headless Chromium. Headline cards, MinHash parameters,
  per-source tables, and sampled cluster-size histograms are populated and
  internally consistent; no blank, malformed, overlapping, or non-finite
  fields are visible.

### 2026-07-25T07:01:00Z — semantic calibration label correction

- A two-H100 `Qwen/Qwen3.5-35B-A3B` calibration served all 20 structured
  judgments for the ten manually reviewed smoke pairs, but the v6 model-facing
  label enum matched only 2/10 pairs. All ten pairs were unanimous. Several
  explanations identified distinct payloads while the serialized label still
  said `true_duplicate`, so the 755,281-pair production launch remained gated.
  Artifact:
  `s3://marin-us-east-02a/marin/user/rav/datakit/dedup-ab/issue6854-semantic-calibration-qwen35-35b-a3b-20260725-v6/calibration.json`.
- Commit `79198807a` replaced the model-facing audit labels with the directional
  boolean `deletion_loses_substantive_content`. Evidence is generated before
  the boolean and the audit label is mapped deterministically in code. The same
  calibration then reached 9/10 correct, 10/10 unanimous, and 20/20 valid
  judgments. Artifact:
  `s3://marin-us-east-02a/marin/user/rav/datakit/dedup-ab/issue6854-semantic-calibration-qwen35-35b-a3b-20260725-v7/calibration.json`.
- The sole v7 miss was the treatment's known low-value college/career spam
  template. The 875- and 763-character documents preserve the same sentences
  while substituting institutions, locations, jobs, and programs; word 5-gram
  Jaccard is 0.5033 and member containment is 0.6333. Both model passes treated
  the nonsensical slot values as facts. Commit `8c2d28445` makes this template
  boundary explicit while retaining different function-call examples, source
  programs, and API methods as distinct content.
- Calibration command:
  `uv run iris --config lib/iris/config/cw-rno2a.yaml job run --no-wait --job-name datakit-6854-semantic-calibration-qwen35-35b-a3b-v7 --enable-extra-resources --cpu 2 --memory 8g --disk 20g --priority batch --extra marin-core:cpu -- python experiments/datakit/scripts/dedup_ab_semantic_judge.py --machine-labels s3://marin-us-east-02a/marin/user/rav/datakit/dedup-ab/issue6854-machine-labels-smoke-0.1b-20260725-v2/machine-labels.json --manual-labels .agents/logbooks/issue-6854-dedup-smoke-labels.json --model Qwen/Qwen3.5-35B-A3B --output s3://marin-us-east-02a/marin/user/rav/datakit/dedup-ab/issue6854-semantic-calibration-qwen35-35b-a3b-20260725-v7/calibration.json`.
- Nine focused semantic-judge tests pass and the two touched files pass the
  repository pre-commit checks. Next action: rerun the full ten-pair gate after
  the template-policy clarification; do not launch production unless it reaches
  10/10 correct and unanimous.

### 2026-07-25T07:08:00Z — direct semantic calibration gate passed

- Calibration v8 succeeded with all ten manually reviewed pairs correct and
  unanimous across two independently framed judgments. All 20 judgments were
  schema-valid on their first attempt and none remained unresolved. The
  previously missed college/career spam pair was classified as a true
  duplicate in both passes using the documented low-value-template boundary.
- The root Iris job
  `/rav/datakit-6854-semantic-calibration-qwen35-35b-a3b-v8` completed
  successfully. The inference broker and worker were cleaned up after the
  root job finished. Artifact:
  `s3://marin-us-east-02a/marin/user/rav/datakit/dedup-ab/issue6854-semantic-calibration-qwen35-35b-a3b-20260725-v8/calibration.json`.
- Calibration command:
  `uv run iris --config lib/iris/config/cw-rno2a.yaml job run --no-wait --job-name datakit-6854-semantic-calibration-qwen35-35b-a3b-v8 --enable-extra-resources --cpu 2 --memory 8g --disk 20g --priority batch --extra marin-core:cpu -- python experiments/datakit/scripts/dedup_ab_semantic_judge.py --machine-labels s3://marin-us-east-02a/marin/user/rav/datakit/dedup-ab/issue6854-machine-labels-smoke-0.1b-20260725-v2/machine-labels.json --manual-labels .agents/logbooks/issue-6854-dedup-smoke-labels.json --model Qwen/Qwen3.5-35B-A3B --output s3://marin-us-east-02a/marin/user/rav/datakit/dedup-ab/issue6854-semantic-calibration-qwen35-35b-a3b-20260725-v8/calibration.json`.
- This passes the direct-pair semantic gate only. The 755,281-pair production
  review remains unlaunched until its restart-safe output shards, exact
  coverage checks, and oversized-document path are tested and calibrated.

### 2026-07-25T07:50:00Z — restart-safe and chunked semantic gates passed

- Commit `2c9e78132` adds deterministic decision-range batches, outcome
  Parquet files, completion markers written last, and strict resume
  validation. A completed batch is accepted only after rechecking its model,
  input, range, case-key hash, configuration hash, byte size, SHA-256,
  evidence-derived outcomes, identity, and coverage. Direct pairs and
  oversized pairs share the same outcome schema.
- Oversized pairs are reviewed by overlapping 24,000-character member chunks.
  Every member character is covered. The canonical is completely indexed in
  24,000-character chunks, and exact five-gram retrieval plus positional
  fallbacks select four candidate chunks for each member chunk. A local
  pathological-pair probe indexed 77 canonical chunks and matched 77 member
  chunks for 1,753,339 versus 1,767,339 characters in 4.414 seconds, with
  21.0 MiB peak traced memory.
- Chunk calibration v1 resolved and classified 7/10 manual pairs correctly.
  The three unresolved expected false positives all contained one or more
  independent, unanimous false-positive chunks. Commit `f1c7f26d6` corrected
  aggregation so any unanimously distinct member chunk proves deletion loss;
  a true-duplicate label still requires every member chunk to resolve as
  represented. Artifact:
  `s3://marin-us-east-02a/marin/user/rav/datakit/dedup-ab/issue6854-semantic-chunk-calibration-qwen35-35b-a3b-20260725-v1/calibration.json`.
- Chunk calibration v2 then reached 9/10 correct and resolved. Its sole
  unresolved pair was the known low-value college/career template: the
  loss-oriented pass returned a high-confidence true duplicate, while the
  duplication-oriented pass returned a low-confidence false positive despite
  describing the slot substitutions as nonsubstantive. Commit `097ad75b3`
  adds exactly one independently framed tiebreak only for unresolved units;
  consensus still requires two non-low-confidence votes with the same label.
  Artifact:
  `s3://marin-us-east-02a/marin/user/rav/datakit/dedup-ab/issue6854-semantic-chunk-calibration-qwen35-35b-a3b-20260725-v2/calibration.json`.
- Chunk calibration v3 passed all ten manually labeled pairs: 10/10 correct,
  10/10 resolved, and complete coverage of 24 member chunks. It made 51 model
  requests: 48 initial independent judgments and three targeted tiebreaks.
  The root Iris job
  `/rav/datakit-6854-semantic-chunk-calibration-qwen35-35b-a3b-v3`
  succeeded. Artifact:
  `s3://marin-us-east-02a/marin/user/rav/datakit/dedup-ab/issue6854-semantic-chunk-calibration-qwen35-35b-a3b-20260725-v3/calibration.json`.
- The four semantic test modules pass all 29 focused tests. The touched files
  pass `./infra/pre-commit.py`, and `uv run pyrefly` reports zero errors. The
  remaining launch gate is a live production-runner smoke that writes and
  revalidates its Parquet/checkpoint contract on object storage.

### 2026-07-25T08:01:00Z — production checkpoint and resume gate passed

- The production runner processed all ten semantic smoke pairs from all 16
  machine-decision files. It wrote one deterministic Parquet outcome shard and
  one completion marker per nonempty semantic batch. The root Iris job
  `/rav/datakit-6854-semantic-production-smoke-qwen35-35b-a3b-v1`
  succeeded with 10 expected pairs, 10 completed pairs, 10 resolved pairs, and
  zero unresolved pairs. Output:
  `s3://marin-us-east-02a/marin/user/rav/datakit/dedup-ab/issue6854-semantic-production-smoke-qwen35-35b-a3b-20260725-v1`.
- Commit `83a4f2e8a` adds a validation-only execution path that never starts an
  inference service. The separate CPU job
  `/rav/datakit-6854-semantic-production-smoke-validate-v1` reread all source
  decisions, referenced pair rows, completion markers, and Parquet outcomes.
  It verified byte counts and SHA-256 hashes, rebuilt each decision from its
  persisted model evidence, and reproduced 10 expected/completed/resolved with
  zero unresolved. Validation summary:
  `s3://marin-us-east-02a/marin/user/rav/datakit/dedup-ab/issue6854-semantic-production-smoke-qwen35-35b-a3b-20260725-v1/semantic-review-validation.json`.
- The focused semantic suite now passes 30 tests. Repository pre-commit checks
  pass for both changed files, and the production script reports zero pyrefly
  errors. The checkpoint/resume gate is clear for the full 755,281-pair
  semantic pass.

### 2026-07-25T09:03:00Z — eight-H100 semantic launch and cap-impact audit

- Launched four disjoint semantic-review partitions with one two-H100
  `Qwen/Qwen3.5-35B-A3B` worker each. All four roots, brokers, and workers use
  Iris batch priority. The first decision files contain 5,862, 5,992, 5,969,
  and 5,995 pairs; every partition is actively completing inference requests.
  Output:
  `s3://marin-us-east-02a/marin/user/rav/datakit/dedup-ab/issue6854-semantic-review-100b-qwen35-35b-a3b-20260725-v1`.
- The first launch revealed that the outer root priority did not propagate to
  inference child jobs. Commit `46ce6ddc9` adds an explicit semantic-runner
  priority and passes it through `IrisConfig`; the four replacement roots and
  all eight H100s were verified at `PRIORITY_BAND_BATCH`.
- The exhaustive cap-50 versus converged marker audit completed with two
  differences and zero DataKit keep/drop changes. Both changed
  `swe-zero-12m` documents remain noncanonical in both artifacts; only
  `dup_cluster_id` changes. The capped label
  `100237470490578047684007976797182131665` has no canonical marker, while
  converged label `100230046777701736319278129735665406779` resolves to
  `9a0afd4f471c03f07001868f1350f330`. This is a nonconvergence metadata defect,
  not a document-output difference. Artifact:
  `s3://marin-us-east-02a/marin/user/rav/datakit/dedup-ab/issue6854-cap50-relations-100b-20260725-v2/relation-review.json`.

### 2026-07-25T09:16:00Z — resource audit recovered with one explicit telemetry gap

- The resource audit recovered all 56 baseline executions and 26 of 27
  treatment executions from archived stage rows or exact coordinator final
  counters. Baseline cap-50 consumed 442,633.00 observed CPU-seconds; full
  convergence consumed 459,946.73. Treatment consumed 221,222.15 observed
  CPU-seconds, excluding treatment connected-components iteration 19
  (`20260724-122800-b77f4f34`).
- That missing execution succeeded and its worker logs prove that all shards
  ran, but both workers and the coordinator logged Finelog timeouts during the
  run. No stage, worker, or final-counter resource row exists in live or
  archived stats. The audit reports this execution as unavailable rather than
  treating it as zero. Artifact:
  `s3://marin-us-east-02a/marin/user/rav/datakit/dedup-ab/issue6854-finelog-100b-20260725-v10.json`.
- The identical-work MinHash comparison is complete: item and byte totals
  match exactly. Baseline used 84,007.55 CPU-seconds and treatment used
  24,982.35, a reduction of 59,025.20 CPU-seconds (70.26%). Peak worker memory
  was 3,859,361,792 bytes versus 3,819,208,704 bytes.
- All eight H100s remain active at batch priority. The four workers completed
  605, 141, 118, and 241 inference responses in the latest ten-minute window.
  The first 128-pair batches include oversized chunked documents, so no
  completion marker has been published yet; the roots show no runtime error.

### 2026-07-25T09:28:00Z — exact semantic request census

- The distributed workload census scanned all 128 decision shards and
  succeeded as Iris job
  `/rav/datakit-6854-semantic-workload-100b-v3`. The 755,281 semantic pairs
  expand to 1,101,833 complete-text review units: 746,797 pairs are direct and
  8,484 require chunked coverage.
- Two independent initial judgments per unit require exactly 2,203,666 model
  requests. At most 3,305,499 requests are needed if every unit invokes the
  targeted tiebreak. Baseline contributes 939,026 review units and treatment
  contributes 162,807. The reviewed semantic text totals 33,190,693,023
  characters; the largest pair totals 19,412,892 characters.
- The four two-H100 semantic jobs remain active at batch priority. At 09:27Z,
  their workers had returned 1,660, 754, 725, and 1,079 successful response
  batches with no runtime error. Artifact:
  `s3://marin-us-east-02a/marin/user/rav/datakit/dedup-ab/issue6854-semantic-workload-100b-20260725-v2/workload.json`.

### 2026-07-25T09:52:00Z — first full-run semantic checkpoint verified

- Partition p3 completed semantic range 0:128 for decision file 96 and
  immediately advanced to range 128:256. All 128 pairs resolved; none remained
  unresolved. The batch contained 107 direct and 21 chunked pairs and used
  1,664 model-response attempts.
- A separate process in the root task reread the completion marker and Parquet
  bytes from object storage. It independently verified the 433,265-byte object
  SHA-256, 128-row count, ordered review-key hash, status and mode counters,
  and single semantic-configuration hash. Outcome SHA-256:
  `d3a270e72108ab8e88ae876c5cf4f3d5a167cf236cb298633bf99ef45f0d1cc8`.
  Marker:
  `s3://marin-us-east-02a/marin/user/rav/datakit/dedup-ab/issue6854-semantic-review-100b-qwen35-35b-a3b-20260725-v1/batches/decision-00096/semantic-00000000.json`.
- The Iris CLI controller port-forward timed out once during a scheduled
  monitor pass. Direct Kubernetes inspection showed all 12 root, broker, and
  worker pods Ready with zero restarts. Four GPU pods each request two H100s
  on node `gb976f0`; inference continued throughout. Monitoring now uses the
  Kubernetes API as the primary liveness path and Iris logs as a secondary
  signal.

### 2026-07-25T10:18:00Z — second checkpoint verified

- Partition p0 completed decision file 0 semantic range 0:128 and advanced to
  range 128:256. All 128 baseline pairs resolved: 119 false positives and 9
  true duplicates. The batch contained 102 direct pairs and 26 chunked pairs,
  expanding to 1,296 review units.
- The batch produced 2,592 initial judgments and 300 targeted tiebreaks. All
  2,892 judgments were valid on their first response; there were no retries or
  invalid structured outputs. A separate root-task process verified the
  686,085-byte Parquet object, ordered identity hash, all manifest counters,
  and configuration hash. Outcome SHA-256:
  `d35eaa5ca70ffe28af64d8673a79724628470b698c188e4ac8266dd3394c2ba6`.
- The earlier p3 checkpoint contains 111 baseline false positives and 17 true
  duplicates. Its 86 requests beyond the 1,578 initial judgments were all
  targeted tiebreaks, with zero retries and zero invalid outputs. Across the
  first 256 durable pairs, semantic review has classified 230 false positives
  and 26 true duplicates, with no unresolved pair.

### 2026-07-25T11:25:00Z — third checkpoint verified

- Partition p0 completed decision file 0 semantic range 128:256 and advanced
  to range 256:384. All 128 baseline pairs resolved: 85 false positives and 43
  true duplicates. The batch contained 118 direct pairs and 10 chunked pairs,
  expanding to 931 review units.
- A separate root-task process verified the 500,684-byte Parquet object, its
  ordered identity hash, manifest counters, and configuration hash. All 1,921
  model judgments were valid first responses: 1,862 initial judgments and 59
  targeted tiebreaks, with zero retries or invalid outputs. Outcome SHA-256:
  `b4a0018df44e8a2f85f7eea3b5e802af12b57dece203dde54d6d31bc6c7b9bf9`.
- Across three durable checkpoints, all 384 baseline pairs resolved: 315 false
  positives and 69 true duplicates. The sample remains far too small for the
  final comparative rate; its purpose is to verify the production protocol and
  preserve auditable incremental evidence.

### 2026-07-25T11:45:00Z — three additional checkpoints verified

- Partition p0 completed decision file 0 semantic ranges 256:384, 384:512, and
  512:640, then advanced to 640:768. Separate object-store rereads verified all
  three Parquet byte hashes, ordered identity hashes, manifest counters, and
  configuration hashes.
- The three batches contain 384 resolved baseline pairs: 262 false positives
  and 122 true duplicates. They used 1,891 valid first-response judgments with
  25 targeted tiebreaks, zero retries, and zero invalid outputs. Outcome
  SHA-256 values:
  `816901804cb85032d23c0cc2770e1d8bee2e205e903489a1ccee97ae1a4c7bfa`,
  `16b82e0e89f4d1899637c8675c596ea07e1f111eb46a11d674c652c8b96c94a7`,
  and `148b9ff725b909ce424437ed4d248f2b9167997893d814be4fdd5a675bfc606a`.
- Across six durable checkpoints, all 768 baseline pairs resolved: 577 false
  positives and 191 true duplicates. All 12 run pods remain Ready with zero
  restarts; the four two-H100 workers continue at batch priority.

### 2026-07-25T11:55:00Z — semantic review reaches 1,664 verified pairs

- Partition p0 completed six additional ranges through 1280:1408 and advanced
  to 1408:1536. Partition p3 completed range 128:256 and advanced to 256:384.
  Separate object-store rereads verified the seven new Parquet byte hashes,
  ordered identity hashes, manifest counters, and configuration hashes.
- The seven checkpoints contain 896 resolved baseline pairs: 700 false
  positives and 196 true duplicates. They used 4,806 valid first-response
  judgments, with zero retries, zero invalid outputs, and zero unresolved
  outcomes. The six short p0 batches were nearly all direct review; p3's batch
  contained 24 chunked pairs and accounted for 3,187 judgments.
- Across 13 durable checkpoints, all 1,664 baseline pairs resolved: 1,277 false
  positives and 387 true duplicates. All 12 run pods remain Ready with zero
  restarts. The artifact prefix remains
  `s3://marin-us-east-02a/marin/user/rav/datakit/dedup-ab/issue6854-semantic-review-100b-qwen35-35b-a3b-20260725-v1/batches/`.

### 2026-07-25T12:04:00Z — semantic review reaches 2,432 verified pairs

- Partition p0 completed five additional ranges through 1920:2048 and
  advanced to 2048:2176. Partition p3 completed range 256:384 and advanced to
  384:512. Independent object-store audits passed for all six new shards.
- The new 768 baseline pairs contain 447 false positives and 321 true
  duplicates. Their 2,645 model judgments were all valid first responses, with
  zero retries, zero invalid outputs, and zero unresolved outcomes. The five
  p0 batches were nearly all direct review; p3's batch contained 14 chunked
  pairs and accounted for 1,204 judgments.
- Across 19 durable checkpoints, all 2,432 baseline pairs resolved: 1,724 false
  positives and 708 true duplicates. All run components remain Ready with zero
  restarts.

### 2026-07-25T12:15:00Z — semantic review reaches 3,712 verified pairs

- Partition p0 completed five additional ranges through 2560:2688, while
  partition p3 completed five ranges through 896:1024. Independent object-store
  audits passed for all 10 new shards, including byte hashes, ordered review
  keys, manifest counters, complete-text coverage, and evidence-derived labels.
- The new 1,280 baseline pairs contain 707 false positives and 573 true
  duplicates. Their 2,879 model judgments were all valid first responses, with
  zero retries, zero invalid outputs, and zero unresolved outcomes. Only four
  pairs required chunked review; 113 review units required a tiebreak.
- Across 29 durable checkpoints, all 3,712 baseline pairs resolved: 2,431 false
  positives and 1,281 true duplicates. All 12 run pods remain Ready with zero
  restarts.
