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

### 2026-07-25T12:26:00Z — first unresolved cases manually adjudicated

- Independent audits passed for 13 additional checkpoints: 1,664 baseline
  pairs, 1,022 false positives, 640 true duplicates, and two unresolved model
  outcomes. Their 3,618 judgments had zero retries or invalid outputs. Across
  42 checkpoints, the immutable semantic artifacts now cover 5,376 pairs:
  3,453 false positives, 1,921 true duplicates, and two unresolved.
- Full-text inspection resolves both ambiguous pairs as true duplicates under
  the protocol's low-value-template boundary. The Speechelo pair repeats
  `Amazon Polly vs Ibm Watson` versus `Naturalreader Software Ultimate Crack`
  as SEO title slots; all 62 member sentences were checked against all 58
  canonical sentences, and the apparent extra voiceover categories restate
  canonical marketing, e-learning, and media sections. Pair location:
  `part-00000-of-00128.parquet:5154`; member/canonical text SHA-256:
  `244080df4f6cc9272163db407a5e0be1cb6778144904282b602c9609db283aaa` /
  `2d726138bafade4d2dc6821804dff5386dfe8a54ba9339e376ca21f241cfcffc`.
- The electronic-cigarette pair has exactly 58 sentences on each side, with
  every member sentence aligned to its canonical counterpart. Its differences
  are synonym rewrites plus opposing keyword injections: `JUUL Pods` in the
  member and `podsmall.com` in the canonical. Pair location:
  `part-00096-of-00128.parquet:2325`; member/canonical text SHA-256:
  `a031a1bb7e742078835f8e8e0f4b93d5ac304f2314526c0aa17b404c502dcb74` /
  `8f169363974838670ef0c2f2aad9e70b194fc361ece865c8229254b7af9a96f7`.
- Manual-overridden totals are therefore 3,453 false positives and 1,923 true
  duplicates across all 5,376 pairs. The immutable model shards remain
  unchanged; the manual decisions will be carried in a separate override
  artifact for finalization.

### 2026-07-25T12:35:00Z — semantic review reaches 6,912 verified pairs

- Twelve additional checkpoints passed independent validation: 1,536 baseline
  pairs, 946 false positives, 588 true duplicates, and two unresolved model
  outcomes. They used 3,467 request attempts. One structured response failed
  all three JSON parses, accounting for three invalid attempts and two retries;
  all other responses were valid on the first attempt.
- Complete character-level comparison resolves the first ambiguous pair as a
  true duplicate. Across 5,597/5,604 characters, its only differences are the
  canonical's `\text{` and matching `}` around the same final boxed `B`.
  Pair location: `part-00000-of-00128.parquet:7314`; member/canonical text
  SHA-256:
  `d605b4dc820a6502cec72f4e48d2f5a36ad3612d807e36f46cca93070edb035e` /
  `9a4b6be06b14382d4fbbcec64182307a8ce6da536b82989da4245dea4088c554`.
- Complete character-level comparison resolves the second ambiguous pair as a
  true duplicate. The shared Naturepedic founding paragraph is unchanged
  except for `The` versus `This` and an incomplete member suffix, `was the
  first to create`, with no object. Its other difference is repeated
  `Dust Mite Mattress Cover` versus `Organic Foldable` keyword stuffing. Pair
  location: `part-00096-of-00128.parquet:3422`; member/canonical text SHA-256:
  `952db158f9896de072a2e277302eb2b7439b69c205de9c2107f5a046bbee1b7e` /
  `1df7ca826f22a2cdd3d2508e91053af274f4bda2339a7f7223c20b0ad44f107e`.
- Across 54 immutable checkpoints, raw model totals are 4,399 false positives,
  2,509 true duplicates, and four unresolved. Applying the four separately
  reviewed overrides yields 4,399 false positives and 2,513 true duplicates
  across all 6,912 pairs.

### 2026-07-25T12:41:00Z — manual overrides made machine-verifiable

- Commit `c542630ef` extends the finalizer with a separate manual-decision input.
  An override is accepted only for an unresolved semantic record, must match
  every pair identity and full-text hash, and must bind the exact persisted
  `judgments_json` SHA-256. A manual record cannot replace a resolved semantic
  decision. Twelve focused finalizer tests pass; lint and implementation
  type-check pass.
- The four reviewed decisions are now immutable one-row Parquet shards with
  JSON completion markers under
  `s3://marin-us-east-02a/marin/user/rav/datakit/dedup-ab/issue6854-semantic-manual-overrides-100b-qwen35-35b-a3b-20260725-v1/decisions/`.
  Each marker binds the review key, label, semantic-evidence hash, Parquet byte
  length, and Parquet SHA-256. Writes were reread and checked for exact record
  equality before the markers were published.

### 2026-07-25T12:44:00Z — semantic review reaches 7,936 verified pairs

- Eight additional checkpoints passed independent validation: 1,024 baseline
  pairs, 603 false positives, 418 true duplicates, and three unresolved model
  outcomes. They used 2,229 request attempts; two judgments each exhausted
  three invalid JSON responses, accounting for six invalid attempts and four
  retries. All other judgments were valid on the first attempt.
- Two complete SFT pairs resolve as true duplicates. Character-level comparison
  found only the insertion or deletion of `\text{` and `}` around the same
  boxed `A` or `E`, with similarity ratios 0.999602 and 0.999796. Pair
  locations and member/canonical text SHA-256 values:
  `part-00000-of-00128.parquet:7338`,
  `0d330d3efa4532ee6a511b64fdaea60422c122efde810aa847766ff4d89ba639` /
  `399108bcfd36069b7e9fd5905f6f17eff5cdae9a5b4986984137281740035376`;
  and `part-00000-of-00128.parquet:7599`,
  `adfb9a4cf00e34d9d32eb55cedc8a89f5f7805d46d98367b4adb03a1e93bef0c` /
  `cb03db77ddc3f9892821ce78a9d62b18ea8f07b1b3a45ec30c87c817fa570e2b`.
- The third case resolves conservatively as a false positive. Although many
  changes are college, location, and program template slots, complete-text
  comparison found a member-only admissions job-duty sentence about managing
  public relations and prospective-student calls, plus a distinct
  business-practices clause. Pair location:
  `part-00096-of-00128.parquet:4878`; member/canonical text SHA-256:
  `1a563a858a5c5f0558ffa2a3283d912c6d6dff7ae117437741aed002748e0511` /
  `4893ce3e5b496530e4b22e31e43f7505e575b1b5c377c62acb8ae20ccb5c1c4e`.
- The three new override shards were reread and hash-verified. Their initial
  metadata carried a timestamp one minute ahead of the actual review; all three
  exact targets were corrected to `2026-07-25T12:44:00Z` and their completion
  markers were republished with the corrected Parquet hashes before any
  finalizer consumed them. Labels and evidence were unchanged.
- Across 62 immutable semantic checkpoints, raw model totals are 5,002 false
  positives, 2,927 true duplicates, and seven unresolved. Applying all seven
  manual records yields 5,003 false positives and 2,933 true duplicates across
  all 7,936 pairs.

### 2026-07-25T12:56:00Z — treatment review begins; 10,112 pairs verified

- Thirteen additional checkpoints first brought audited coverage to 9,600
  pairs. The new 1,664-pair block contains 1,032 false positives and 632 true
  duplicates, with zero unresolved outcomes, retries, or invalid responses.
  Partition p1's first 757,436-byte shard passed all byte, row, key, evidence,
  and manifest checks; its 128 pairs included 25 chunked reviews, 3,196
  judgments, and 258 tiebreaks.
- A global reread caught four more shards that completed during the audit.
  All four then passed the same independent checks: 512 pairs, 296 false
  positives, 216 true duplicates, zero unresolved, 1,098 first-attempt-valid
  judgments, and 62 tiebreaks. They split evenly between baseline and
  treatment.
- Exact threshold reconstruction confirms that the first 62 checkpoint entries
  were correctly described as baseline-only. Treatment cases first appear in
  the 63rd–75th checkpoint block. At 75 checkpoints, coverage was 8,851
  baseline pairs and 749 treatment pairs.
- The stable 79-checkpoint snapshot contains 10,112 semantic pairs. Raw
  baseline counts are 5,855 false positives, 3,245 true duplicates, and seven
  unresolved; raw treatment counts are 475 false positives and 530 true
  duplicates, with none unresolved. Applying the seven hash-bound manual
  decisions produces:

  - baseline: 9,107 pairs, 5,856 false positives, 3,251 true duplicates;
  - treatment: 1,005 pairs, 475 false positives, 530 true duplicates;
  - combined: 10,112 pairs, 6,331 false positives, 3,781 true duplicates.

- The global pass reread all 79 semantic shards and all seven manual shards. It
  verified every manual record's review key, member/canonical SHA-256 values,
  and semantic `judgments_json` SHA-256, leaving no unresolved record without
  exactly one manual decision.

### 2026-07-25T13:05:00Z — 10,726 pairs verified; treatment ambiguities resolved

- Five new checkpoints passed independent validation: 614 pairs, 450 false
  positives, 158 true duplicates, and six unresolved model outcomes. The
  1,287 request attempts include 39 invalid JSON responses and 27 retries.
  Invalid responses were concentrated in LaTeX-heavy SFT pairs whose generated
  JSON strings contained unescaped control sequences.
- Complete character-level comparison resolves five SFT pairs as true
  duplicates. Each full document is otherwise identical; the only two changed
  spans insert or delete `\text{` and its closing `}` around the same boxed
  answer. The three treatment pair locations, member/canonical text SHA-256,
  and character-similarity ratios are:

  - `part-00000-of-00128.parquet:9024`,
    `350033bc27ec34d106098fc791e62caadc3f7d97c43808467e2792b39cffd035` /
    `e2fd0621676f407acb75c4e4fdd10eafb5b33e8af63f1220eaca895c4a3f1d94`,
    0.999772;
  - `part-00000-of-00128.parquet:9025`,
    `aeb80c298bd417cecd04a85a467612f5dd871c39b06f508da08ba564c948edc7` /
    `b66a89567986730f7598ef6f2bef5cb36e8be997cd56b2d00b7bfb3c2e5e9597`,
    0.999842;
  - `part-00000-of-00128.parquet:9026`,
    `b6781fc424cdbf7ff4e887b2c788631bbf84eaa787b6d3e4128041727b3e5379` /
    `79a5fecb12722e6f10c788c3944060e7749d8759486af856be50a91f7c9cc1d5`,
    0.999787.

- The two equivalent baseline SFT pairs are:

  - `part-00096-of-00128.parquet:7694`,
    `d66deae3d36d2fd69867afcd841a89d2a7f8bc04dc5e50dd765c9a081bae31a1` /
    `633fc5123e50232ba8386df0d4ab5394490b555a0e934f8e876908367dccaf5c`,
    0.999634;
  - `part-00096-of-00128.parquet:7697`,
    `e68c5138db7797698a0374f71db57969f95493d9ada8e13d1460e45ff03866e9` /
    `7a29a8629ac14a81447d3f6efbeebf3d5ae9d241b03f286156382b1b613a5aaf`,
    0.999207.

- The sixth pair is also a true duplicate after complete-text inspection. Both
  sides contain the same Mia/Noah/Olivia book problem, operands, derivation,
  and answer 1,800. Its five spans differ only in equivalent `1/5 × 75` versus
  `75/5` notation and sentence phrasing. Pair location:
  `part-00096-of-00128.parquet:7620`; member/canonical text SHA-256:
  `e39d51e6f7067fdd513979e1628900aa501e6fa7385849efab75f7a962049e2d` /
  `74ca511acacf7115ef9debeb883d83b2813cfbde46cbca014977207892857866`.
- Six new one-row manual Parquet shards and their completion markers were
  written, reread, and hash-verified. The stable 84-shard global snapshot
  contains 10,726 pairs. After all 13 evidence-bound manual decisions:

  - baseline: 9,491 pairs, 6,145 false positives, 3,346 true duplicates;
  - treatment: 1,235 pairs, 636 false positives, 599 true duplicates;
  - combined: 10,726 pairs, 6,781 false positives, 3,945 true duplicates.

- A global reread verified every semantic shard and all 13 manual records,
  including review-key, full-text-hash, and semantic-evidence-hash bindings.
  No unresolved semantic record in the snapshot lacks a manual decision.

### 2026-07-25T13:13:00Z — semantic review reaches 11,622 verified pairs

- Seven additional partition-p3 checkpoints passed independent validation:
  896 pairs, 483 false positives, 413 true duplicates, zero unresolved
  outcomes, 1,895 first-attempt-valid judgments, and 57 tiebreaks. One pair
  required chunked review; the remaining 895 were direct.
- The first checkpoint straddled the variant boundary: its 39 baseline pairs
  contain 36 false positives and three true duplicates, while its 89 treatment
  pairs contain 48 false positives and 41 true duplicates. The remaining 768
  pairs are treatment, with 399 false positives and 369 true duplicates.
- Across the stable 91-checkpoint snapshot, the 13 prior manual records leave:

  - baseline: 9,530 pairs, 6,181 false positives, 3,349 true duplicates;
  - treatment: 2,092 pairs, 1,083 false positives, 1,009 true duplicates;
  - combined: 11,622 pairs, 7,264 false positives, 4,358 true duplicates.

- All 12 root, broker, and worker pods remain Ready with zero restarts. Partition
  p0 is processing the chunk-heavy first batch of decision file 1; partition
  p1 is processing the second batch of decision file 32; partition p2 remains
  active on the largest initial batch.

### 2026-07-25T13:24:00Z — treatment decision file 96 completes

- The final three checkpoints in treatment decision file 96 passed independent
  validation: 363 pairs, 201 false positives, 162 true duplicates, and zero
  unresolved outcomes. All 754 model judgments were valid on their first
  attempt; 49 pairs required a tiebreak.
- The audit reread each completion marker and Parquet shard, verified byte
  length and SHA-256, reconstructed the ordered case-key SHA-256, validated all
  row identities and configuration hashes, and recomputed every outcome from
  its complete persisted evidence.
- Across the stable 94-checkpoint snapshot, the 13 prior manual records leave:

  - baseline: 9,530 pairs, 6,181 false positives, 3,349 true duplicates;
  - treatment: 2,455 pairs, 1,284 false positives, 1,171 true duplicates;
  - combined: 11,985 pairs, 7,465 false positives, 4,520 true duplicates.

- Partition p3 advanced to decision file 97. All 12 root, broker, and worker
  pods remain Ready with zero restarts, retaining the requested eight H100s at
  batch priority.

### 2026-07-25T13:42:00Z — partition 2 completes its first large batch

- The first checkpoint in baseline decision file 64 passed independent
  validation: 128 pairs, 120 false positives, eight true duplicates, and zero
  unresolved outcomes.
- This batch contained 31 chunked pairs and 97 direct pairs. All 4,014 model
  judgments were valid on their first attempt. The audit verified the
  completion marker, outcome bytes and SHA-256, ordered case-key SHA-256, row
  identities, configuration hash, counters, and deterministic outcome
  reconstruction from every persisted judgment.
- Across the stable 95-checkpoint snapshot, the 13 prior manual records leave:

  - baseline: 9,658 pairs, 6,301 false positives, 3,357 true duplicates;
  - treatment: 2,455 pairs, 1,284 false positives, 1,171 true duplicates;
  - combined: 12,113 pairs, 7,585 false positives, 4,528 true duplicates.

- Partition p2 moved to semantic range 128:256. All four GPU workers continue
  serving requests with all 12 pods Ready and zero restarts.

### 2026-07-25T13:51:00Z — partition 1 completes its second large batch

- The second checkpoint in baseline decision file 32 passed independent
  validation: 128 pairs, 75 false positives, 53 true duplicates, and zero
  unresolved outcomes.
- This batch contained 11 chunked pairs and 117 direct pairs. All 1,062 model
  judgments were valid on their first attempt. The audit verified the marker,
  outcome bytes and SHA-256, ordered case-key SHA-256, identities,
  configuration hash, counters, and deterministic outcomes.
- Across the stable 96-checkpoint snapshot, the 13 prior manual records leave:

  - baseline: 9,786 pairs, 6,376 false positives, 3,410 true duplicates;
  - treatment: 2,455 pairs, 1,284 false positives, 1,171 true duplicates;
  - combined: 12,241 pairs, 7,660 false positives, 4,581 true duplicates.

- Partition p1 moved to semantic range 256:384. All 12 pods remain Ready with
  zero restarts.

### 2026-07-25T14:40:00Z — partition 0 completes its first large batch

- The first checkpoint in baseline decision file 1 passed independent
  validation: 128 pairs, 116 false positives, 12 true duplicates, and zero
  unresolved outcomes.
- This batch contained 31 chunked pairs and 97 direct pairs. All 2,459 model
  judgments were valid on their first attempt. The audit verified the marker,
  outcome bytes and SHA-256, ordered case-key SHA-256, identities,
  configuration hash, counters, and deterministic outcomes.
- Across the stable 97-checkpoint snapshot, the 13 prior manual records leave:

  - baseline: 9,914 pairs, 6,492 false positives, 3,422 true duplicates;
  - treatment: 2,455 pairs, 1,284 false positives, 1,171 true duplicates;
  - combined: 12,369 pairs, 7,776 false positives, 4,593 true duplicates.

- Partition p0 moved to semantic range 128:256. All four GPU workers continue
  serving requests with all 12 pods Ready and zero restarts.

### 2026-07-25T14:58:00Z — partition 1 completes its third large batch

- The third checkpoint in baseline decision file 32 passed independent
  validation: 128 pairs, 88 false positives, 40 true duplicates, and zero
  unresolved outcomes.
- This batch contained 17 chunked pairs and 111 direct pairs. One malformed
  response was retried successfully; all 1,670 accepted judgments validate.
  The audit verified the marker, outcome bytes and SHA-256, ordered case-key
  SHA-256, identities, configuration hash, counters, and deterministic
  outcomes.
- Across the stable 98-checkpoint snapshot, the 13 prior manual records leave:

  - baseline: 10,042 pairs, 6,580 false positives, 3,462 true duplicates;
  - treatment: 2,455 pairs, 1,284 false positives, 1,171 true duplicates;
  - combined: 12,497 pairs, 7,864 false positives, 4,633 true duplicates.

- Partition p1 moved to semantic range 384:512. All 12 pods remain Ready with
  zero restarts.

### 2026-07-25T15:13:00Z — 13,137 pairs verified; Apache ambiguity resolved

- Five new baseline checkpoints passed independent validation: 640 pairs, 463
  false positives, 176 true duplicates, and one unresolved model outcome.
  Their 5,295 accepted judgments cover 48 chunked and 592 direct pairs; all
  responses were valid on their first attempt.
- Complete-text inspection resolves the ambiguous Apache-on-Windows pair as a
  false positive. A reproducible comparison covered all 6,668 member
  characters and 7,554 canonical characters, split into 120 and 136 normalized
  sentence or command units. Deleting the member loses five technical details:
  the custom service-config installation command using `-f`, the named-service
  `shutdown` command, forward-slash path guidance, parent/child process
  behavior, and `Control-Break` console restart behavior. Pair location:
  `part-00032-of-00128.parquet:1107`; member/canonical text SHA-256:
  `02c9fa5b3d10ab13db55ea2d360a8b19dc3d9746b459b145a66caecb85e7ba67` /
  `d89320ae00be44eb9ca4c88d90d36cc2b81d6b5e6e1775253d6f82099c5eb3f5`.
- The one-row manual Parquet shard and its completion marker were written,
  reread, and hash-verified. The record binds the review identity, both
  full-text hashes, semantic shard hash, and exact `judgments_json` SHA-256.
- A global reread verified all 103 semantic markers, 13,137 unique outcome
  records, and 14 manual markers. Every unresolved semantic record has exactly
  one matching manual decision; none remains unresolved. Adjusted totals are:

  - baseline: 10,682 pairs, 7,044 false positives, 3,638 true duplicates;
  - treatment: 2,455 pairs, 1,284 false positives, 1,171 true duplicates;
  - combined: 13,137 pairs, 8,328 false positives, 4,809 true duplicates.

- All 12 pods remained Ready with zero restarts at the latest health sample.

### 2026-07-25T15:28:00Z — semantic review reaches 14,289 verified pairs

- Nine additional baseline checkpoints passed independent validation: 1,152
  pairs, 878 false positives, 274 true duplicates, and zero unresolved
  outcomes. They used 4,621 valid first-attempt judgments with no invalid
  responses or retries; 26 pairs were chunked and 1,126 were direct.
- Every checkpoint passed marker and Parquet byte-hash validation, ordered
  case-key reconstruction, identity and configuration checks, manifest-counter
  reconstruction, complete member-coverage checks for chunked cases, and
  deterministic outcome reconstruction from persisted evidence.
- Across the stable 112-checkpoint snapshot, all 14 manual records remain
  exactly bound and leave:

  - baseline: 11,834 pairs, 7,922 false positives, 3,912 true duplicates;
  - treatment: 2,455 pairs, 1,284 false positives, 1,171 true duplicates;
  - combined: 14,289 pairs, 9,206 false positives, 5,083 true duplicates.

- At the latest health sample, all 12 pods were Ready with zero restarts.
  Partition p0 was processing decision-file-1 offset 256, p1 offset 1,536, p2
  decision-file-64 offset 512, and p3 decision-file-97 offset 128.

### 2026-07-25T15:39:00Z — 15,697 pairs verified; SQL ambiguity resolved

- Eleven additional baseline checkpoints passed independent validation: 1,408
  pairs, 842 false positives after manual review, 566 true duplicates after
  manual review, and zero remaining unresolved outcomes. They used 3,903 valid
  first-attempt judgments with no invalid responses or retries; 17 pairs were
  chunked and 1,391 were direct.
- Complete-text inspection resolves the one model ambiguity as a true
  duplicate. Both sides have exactly 10 SQL blocks, and normalized comparison
  confirms that every block is identical and appears in the same order. Every
  exercise and table row is represented by the canonical. The member-only
  units are the `Using Null` title, `From SQLZOO` attribution, structural
  headings, `CAPTION` prefixes, and a repeated NULL-propagation explanation
  already present in the canonical. Pair location:
  `part-00064-of-00128.parquet:1392`; member/canonical text SHA-256:
  `42e74e0c1362413d992f406ed319506e5fb532345eac47f5e71d36126e45044f` /
  `f95f687a1cf6953e59919222475fd0d0e26787434988b80c886658478b82095b`.
- The new manual Parquet shard and marker were written, reread, and
  hash-verified against the exact semantic evidence and source identities.
- Across the stable 123-checkpoint snapshot, all 15 manual records leave:

  - baseline: 13,242 pairs, 8,764 false positives, 4,478 true duplicates;
  - treatment: 2,455 pairs, 1,284 false positives, 1,171 true duplicates;
  - combined: 15,697 pairs, 10,048 false positives, 5,649 true duplicates.

- All 12 pods remain Ready with zero restarts, and all four GPU workers
  continue serving requests.

### 2026-07-25T15:48:00Z — semantic review reaches 16,849 verified pairs

- Nine additional baseline checkpoints passed independent validation: 1,152
  pairs, 781 false positives, 371 true duplicates, and zero unresolved
  outcomes. They used 2,541 valid first-attempt judgments with no invalid
  responses or retries; three pairs were chunked and 1,149 were direct.
- Across the stable 132-checkpoint snapshot, all 15 manual records leave:

  - baseline: 14,394 pairs, 9,545 false positives, 4,849 true duplicates;
  - treatment: 2,455 pairs, 1,284 false positives, 1,171 true duplicates;
  - combined: 16,849 pairs, 10,829 false positives, 6,020 true duplicates.

- A public heartbeat records the full-run setup, resource result, current
  semantic coverage, and the order-bias caveat:
  https://github.com/marin-community/marin/issues/6854#issuecomment-5079122624.
- At the latest health sample all 12 pods were Ready with zero restarts. The
  four GPU workers served 5,612 successful responses over the prior 15
  minutes.

### 2026-07-25T15:57:00Z — 19,025 pairs verified; SEO ambiguity resolved

- Seventeen additional baseline checkpoints passed independent validation:
  2,176 pairs, 1,377 false positives, 798 model true duplicates, and one
  unresolved model outcome. They used 5,844 valid first-attempt judgments with
  no invalid responses or retries; 18 pairs were chunked and 2,158 were direct.
- Complete-text inspection resolves the model ambiguity as a true duplicate
  under the explicit low-value SEO-template boundary. All 44 member sentence
  or heading units were compared against all 47 canonical units. Every body
  sentence has a direct paraphrase; the differences are SEO headings and the
  `Roseville CA 95678` versus `Cherry Valley MA 01611` location slot. Pair
  location: `part-00032-of-00128.parquet:4761`; member/canonical text SHA-256:
  `25ba9708a9fee2794ac1066abcb97d2b4613f0a656f1fe03412b30ecccbcdca7` /
  `5e0dd4605140ed851a5d8b3d0d72e203325500ce2db75613d4dec5e0bbe74fdd`.
- The new manual Parquet shard and marker were written, reread, and
  hash-verified against the exact semantic evidence and source identities.
- Across the stable 149-checkpoint snapshot, all 16 manual records leave:

  - baseline: 16,570 pairs, 10,922 false positives, 5,648 true duplicates;
  - treatment: 2,455 pairs, 1,284 false positives, 1,171 true duplicates;
  - combined: 19,025 pairs, 12,206 false positives, 6,819 true duplicates.

- All 12 pods remain Ready with zero restarts. Transient Finelog send timeouts
  affected telemetry only; Kubernetes and 7,571 successful model responses in
  the prior 15 minutes confirmed reviewer liveness.

### 2026-07-25T16:06:00Z — semantic review reaches 20,945 verified pairs

- Fifteen additional baseline checkpoints passed independent validation:
  1,920 pairs, 971 false positives, 949 true duplicates, and zero unresolved
  outcomes. They used 5,318 valid first-attempt judgments with no invalid
  responses or retries; 15 pairs were chunked and 1,905 were direct.
- Across the stable 164-checkpoint snapshot, all 16 manual records leave:

  - baseline: 18,490 pairs, 11,893 false positives, 6,597 true duplicates;
  - treatment: 2,455 pairs, 1,284 false positives, 1,171 true duplicates;
  - combined: 20,945 pairs, 13,177 false positives, 7,768 true duplicates.

- All 12 pods remain Ready with zero restarts. The four GPU workers served
  8,012 successful responses over the prior 15 minutes.

### 2026-07-25T16:17:00Z — 23,377 pairs verified; three ambiguities resolved

- Nineteen additional baseline checkpoints passed independent validation:
  2,432 pairs, 1,395 model false positives, 1,034 model true duplicates, and
  three unresolved model outcomes. They used 5,674 valid first-attempt
  judgments with no invalid responses or retries; 13 pairs were chunked and
  2,419 were direct.
- Complete sentence alignment resolves the surname-history pair as a true
  duplicate. All 28 member units are represented by the same 50-unit
  genealogy SEO template; differences are the surname slot, headings,
  illustrative occupation or religious-phrase substitutions, and an
  incomplete `Famous People` heading. Pair location:
  `part-00001-of-00128.parquet:4724`; member/canonical text SHA-256:
  `a538c95d274280bc1336a815cce478229dc9e66d1e68786a8933ac97e829a0a0` /
  `601a3dde5681f75a466ffb662db60199e2247144c9597fa901cf865f914fa2ec`.
- Complete sentence alignment resolves the college/career pair as a false
  positive. The 10-unit member contains four substantive sentences absent
  from the six-unit canonical: an online-course recruiting advantage,
  registration deadlines and program requirements, an admissions-information
  request, and a bachelor's degree requirement listing five disciplines.
  Pair location: `part-00032-of-00128.parquet:5882`;
  member/canonical text SHA-256:
  `6787e628a041043e2fbf65358652d4b49ff351c808edbd355e2817143b339a4c` /
  `4893ce3e5b496530e4b22e31e43f7505e575b1b5c377c62acb8ae20ccb5c1c4e`.
- Complete sentence alignment resolves the template-advice article as a true
  duplicate. All 29 member units align to the 29-unit canonical; differences
  are the incongruous medical-assistant versus Bootstrap title, synonym
  substitutions, and two garbled numeric insertions. Pair location:
  `part-00064-of-00128.parquet:4680`; member/canonical text SHA-256:
  `440a3a99f79b6c9196b989e717ed63f7728cb62ef927416c34faf208e6f8b284` /
  `11ad38e8986f596f9d60954a49db12024f47beb555aa4318c3adf096db31ebd5`.
- Three manual Parquet shards and markers were written, reread, and
  hash-verified. Across the stable 183-checkpoint snapshot, all 19 manual
  records leave:

  - baseline: 20,922 pairs, 13,289 false positives, 7,633 true duplicates;
  - treatment: 2,455 pairs, 1,284 false positives, 1,171 true duplicates;
  - combined: 23,377 pairs, 14,573 false positives, 8,804 true duplicates.

- All 12 pods remain Ready with zero restarts. Partition p1's latest
  15-minute response-log count was truncated despite four advancing
  checkpoints, so it is excluded from throughput aggregation.

### 2026-07-25T16:27:02Z — semantic review reaches 25,809 verified pairs

- Nineteen additional baseline checkpoints passed independent validation:
  2,432 pairs, 1,548 false positives, 884 true duplicates, and zero unresolved
  outcomes. They used 5,464 valid first-attempt judgments with no invalid
  responses or retries; seven pairs were chunked and 2,425 were direct.
- Across the stable 202-checkpoint snapshot, all 19 manual records leave:

  - baseline: 23,354 pairs, 14,837 false positives, 8,517 true duplicates;
  - treatment: 2,455 pairs, 1,284 false positives, 1,171 true duplicates;
  - combined: 25,809 pairs, 16,121 false positives, 9,688 true duplicates.

- All 12 pods remain Ready with zero restarts. The four GPU workers served
  7,836 successful responses over the prior 15 minutes.

### 2026-07-25T16:41:33Z — source-code context overflow isolated

- Partitions 0 and 1 stopped on unfinished batches after the model rejected
  direct source-code prompts over its 131,072-token context. The affected
  prompts contained 214,156, 158,066, and 165,243 input tokens despite fitting
  the 300,000-character direct-review cutoff. Partition 2 stopped independently
  after its inference endpoint returned HTTP 502. No failed batch wrote a
  completion marker.
- Exact tokenizer inspection confirms that the existing exhaustive chunk path
  keeps the affected pairs within context: their largest chunk prompts contain
  91,028, 116,973, and 90,483 input tokens.
- Automatic direct review now falls back to exhaustive chunk review only for a
  model context-limit response. Explicitly forced direct review and other bad
  requests still fail. The regression failed before the fix and passes after
  it; all 11 semantic-review tests pass.
- Partition 3 remains healthy. The failed partitions can resume from their
  immutable, hash-verified checkpoint frontiers without repeating completed
  work.

### 2026-07-25T16:45:27Z — eight H100s restored; 28,241 pairs verified

- Partitions 0, 1, and 2 were resubmitted as `v3` batch-priority jobs from the
  tested context-overflow fix. All four 2-H100 workers and their coordinators
  are Ready with zero Kubernetes restarts.
- Nineteen additional checkpoints passed independent validation: 2,432 pairs,
  1,438 false positives, 994 true duplicates, and zero unresolved outcomes.
  Their 5,545 judgments were valid on the first attempt, with no invalid
  responses or retries; ten pairs were chunked and 2,422 were direct.
- The new set contains 2,430 baseline pairs and two treatment pairs. Both
  treatment pairs are false positives. Across the stable 221-checkpoint
  snapshot, all 19 manual records leave:

  - baseline: 25,784 pairs, 16,273 false positives, 9,511 true duplicates;
  - treatment: 2,457 pairs, 1,286 false positives, 1,171 true duplicates;
  - combined: 28,241 pairs, 17,559 false positives, 10,682 true duplicates.

### 2026-07-25T17:00:32Z — context fallback proven; 30,161 pairs verified

- Fifteen additional checkpoints passed independent validation: 1,920 pairs,
  1,350 model false positives, 568 model true duplicates, and two unresolved
  outcomes. They contain 4,403 valid judgments across 4,408 attempts; five
  invalid JSON responses were retried successfully. Nine pairs were chunked
  and 1,911 were direct.
- The recovered partition-0 checkpoint contains both source-code pairs that
  exceeded the model context. The recovered partition-1 checkpoint contains
  the third context-overflow pair plus two multi-million-character documents.
  All three overflow pairs completed through exhaustive chunk review, and both
  jobs continued into later checkpoints.
- Complete sentence alignment resolves the first ambiguity as a true duplicate.
  All 11 member units align to the same 11-unit chair SEO scaffold. Differences
  are product and date slots, synonyms, and a truncated computer-job sentence.
  Pair location: `part-00097-of-00128.parquet:5448`;
  member/canonical text SHA-256:
  `f27fcf602c3e9d9456757bc8b0f2e0d65c303f825fca5cd9ec4f986576e776e5` /
  `93250706850ba0d21cf4eaafa4529a5fa0eeb16a859a86d0fdacdb275b69d894`.
- Complete sentence alignment resolves the second ambiguity as a false
  positive. Its member-only sentence says that pursuing a business internship
  develops career prospects and work experience. The other seven member units
  are college-spam scaffolds or entity slots, but that additional advice claim
  is absent from all six canonical units. Pair location:
  `part-00097-of-00128.parquet:5508`; member/canonical text SHA-256:
  `e8c1cce36443c20ff50b52047739be40f151c0243f8221e382e357b756ebbd18` /
  `4893ce3e5b496530e4b22e31e43f7505e575b1b5c377c62acb8ae20ccb5c1c4e`.
- Both manual artifacts were written and then reread in a separate process.
  The verification bound their exact source identities, full-text hashes,
  semantic outcome hash, judgment-evidence hash, sentence evidence, Parquet
  bytes, and completion markers.
- Across the stable 236-checkpoint snapshot, all 21 manual records leave:

  - baseline: 27,431 pairs, 17,478 false positives, 9,953 true duplicates;
  - treatment: 2,730 pairs, 1,432 false positives, 1,298 true duplicates;
  - combined: 30,161 pairs, 18,910 false positives, 11,251 true duplicates.

- All 12 pods remain Ready with zero restarts. The four GPU workers served
  6,956 successful responses over the prior 15 minutes.

### 2026-07-25T17:11:05Z — 31,441 pairs verified; fourth partition patched

- Ten additional checkpoints passed independent validation: 1,280 pairs,
  868 model false positives, 409 model true duplicates, and three unresolved
  outcomes. All pairs used complete-text review. They contain 2,608 valid
  judgments across 2,621 attempts; 13 invalid JSON responses caused ten
  retries.
- Complete mathematical comparison resolves the sum-of-cubes pair as a true
  duplicate. Both texts use the same problem, nonzero constraint, identity,
  derivation, and result. The member's explicit intermediate identity is
  represented by the canonical derivation; the remaining differences are
  wording and LaTeX formatting. Pair location:
  `part-00097-of-00128.parquet:7550`; member/canonical text SHA-256:
  `3703edfb9d0b6868604bd1d2c296d6db99c33f1cc5372aa3ac001e10612f2d1e` /
  `a70343994792dbed5345824d17c4573da2a1d3669a13f201679bde9c2ac5d021`.
- Complete mathematical comparison resolves the inequality pair as a true
  duplicate. All 39 member units align to the canonical's 38 units: the same
  inequality, squared expansion, sum-of-squares proof, and equality condition.
  Only paraphrase and final-display emphasis differ. Pair location:
  `part-00097-of-00128.parquet:7600`; member/canonical text SHA-256:
  `4b4a91bd3ad33fe21cb6ff364796f83ba85f5e12dc33bb3881b7a0ca0f275212` /
  `9b5fb368950d663ea1a20ce1e67cd7db215128f526bcd6cf952394c7781d6e79`.
- Complete unit and character comparison resolves the cybersecurity SFT pair
  as a true duplicate. All 121 question, option, reasoning, and answer units
  match. The only two changed spans add `\text{` and `}` around the same boxed
  answer, giving character similarity 0.999587. Pair location:
  `part-00097-of-00128.parquet:7687`; member/canonical text SHA-256:
  `9cccefe0c17b19cdaf280d1310aacc058adb1a0d2d0593c386bce97b0470fcb5` /
  `ca080587a0bbfa0de00fe392974fca8a3fa7b5fd50e37b0e6f84a2932980ec6a`.
- A batch-priority Iris job published the three immutable manual Parquet
  records and completion markers. A separate Iris process reread the source
  cases, semantic checkpoint, judgment hashes, exact manual rows, Parquet
  hashes, and markers; all checks passed.
- Across the stable 246-checkpoint snapshot, all 24 manual records leave:

  - baseline: 27,943 pairs, 17,773 false positives, 10,170 true duplicates;
  - treatment: 3,498 pairs, 2,005 false positives, 1,493 true duplicates;
  - combined: 31,441 pairs, 19,778 false positives, 11,663 true duplicates.

- Partition 3's older `v2` job later reached the same context-limit condition
  while its current batch was unfinished. It was replaced by the patched `v3`
  job and is revalidating prior immutable checkpoints before resuming. All
  four batch-priority 2-H100 workers and their coordinators are Ready with zero
  Kubernetes restarts.

### 2026-07-25T17:34:22Z — 34,143 pairs verified

- Twenty-two additional checkpoints passed independent validation: 2,702
  pairs, 1,319 model false positives, 1,376 model true duplicates, and seven
  unresolved outcomes. They contain 5,776 valid judgments across 5,820
  attempts; 44 invalid responses caused 31 retries. Three pairs were chunked
  and 2,699 were direct.
- Complete character comparison resolves six ambiguities as true duplicates.
  In each case, every character of one document occurs unchanged in the other;
  the only difference is a seven-character LaTeX answer wrapper. The source
  locations and member/canonical text SHA-256 prefixes are:

  - `part-00001-of-00128.parquet:9157`, `5c32379b` / `86301086`;
  - `part-00001-of-00128.parquet:9180`, `0cd19260` / `fe5076b5`;
  - `part-00032-of-00128.parquet:9269`, `626a3e26` / `8e62a831`;
  - `part-00064-of-00128.parquet:7794`, `422c74b2` / `741014b7`;
  - `part-00064-of-00128.parquet:7797`, `52720e61` / `a7568be9`;
  - `part-00064-of-00128.parquet:7806`, `c4e43a5d` / `018231f1`.

- Complete code and image comparison resolves the seventh ambiguity as a
  false positive. The source code is byte-identical after replacing its
  embedded PNG, but the asserted output images differ: both are 360×360 RGBA,
  while 17,121 pixels differ over a `[0, 0, 360, 133]` bounding box and the
  mean absolute channel difference is 14.693. The image byte hashes are
  `53798660` / `be5955ab`, and decoded pixel hashes are `c4bc7e4d` /
  `2ea4d5ea`. Pair location: `part-00064-of-00128.parquet:8040`;
  member/canonical text SHA-256 prefixes: `c294f6bf` / `bc50a49f`.
- A batch-priority Iris job published the seven immutable manual Parquet
  records and completion markers. A separate Iris job reread all seven source
  cases, semantic checkpoints, manual rows, judgment hashes, Parquet hashes,
  and markers; all exact checks passed.
- Across the stable 268-checkpoint snapshot, all 31 manual records leave:

  - baseline: 28,230 pairs, 18,041 false positives, 10,189 true duplicates;
  - treatment: 5,913 pairs, 3,057 false positives, 2,856 true duplicates;
  - combined: 34,143 pairs, 21,098 false positives, 13,045 true duplicates.

- Partition 3 recovered through its immutable frontier and resumed new work.
  All four batch-priority 2-H100 workers, coordinators, and brokers remain
  Ready with zero Kubernetes restarts.

### 2026-07-25T17:57:39Z — 35,120 pairs verified

- Eight additional checkpoints passed independent validation: 977 pairs, 595
  model false positives, 379 model true duplicates, and three unresolved
  outcomes. They contain 2,038 valid judgments across 2,051 request attempts;
  13 invalid responses affected five retried judgments. Two pairs were
  chunked and 975 were direct.
- Complete character comparison resolves two treatment SFT ambiguities as true
  duplicates. Every character of the shorter document is unchanged in the
  longer; the only two changed spans add or remove `\text{` and `}` around the
  same boxed answer:

  - `part-00064-of-00128.parquet:9204`, 5,794 / 5,801 characters,
    similarity 0.999396; member/canonical text SHA-256
    `05ce37ad035da9d5ed0489685db6785f74bbbf252d5da3de8e6ae7cbd2ecebc6` /
    `85c46f4f373506fea675e64ebb4706fe7d54bf596fffa9ce51ba4d4f9d88a6e6`;
  - `part-00064-of-00128.parquet:9225`, 5,593 / 5,586 characters,
    similarity 0.999374; member/canonical text SHA-256
    `0f25896f8140c7d1f0990de1bff63f477ca8b05a0d235ddec1ad8d8178f71697` /
    `e13974fbfbbd1eefb03f786a2029db968d4e4a9234fb6edfef6349e0b8e300c9`.

- Complete forum-post alignment resolves the third treatment ambiguity as a
  false positive. The member has 20 posts and the canonical has 19. Member
  Post #11 by `andywheels` is a 388-character account of NVflash/CWM
  installation concerns, a plan to try the android-hilfe CWM build, and a
  question about the main-ROM installation hack. It is absent from every
  canonical post; its best lexical match is only 0.299465. The other 19 member
  posts align one-to-one with canonical posts at similarities from 0.832 to
  1.000. Pair location: `part-00097-of-00128.parquet:8074`;
  member/canonical text SHA-256:
  `4fd3b67d39e87ea1797a6b7ff2f75d6b205e6df40d2a22f44e4ebca64dcdb721` /
  `11ccbe82d7f4c9e9914199144a089440c7dd1bbe93565f3736824c1e173660d6`;
  missing-post SHA-256:
  `c10ed782b8a1f6032a15cda102be55742c1557e270dad0bbd1a82252e34d8566`.
- The three manual Parquet records have SHA-256
  `d0656255dc5d672a7e5c09788fd6d7158023eea9c1f6cea78f972f9f17f256a6`,
  `14d864523c4c10cb26cc648e01e014e1795db0eaabf2f0b4af53b0e44836d318`,
  and `dafca05f188e7ce7beea2da7fafdd1c32f508682de93694a80adee4bb1734577`.
  A separate batch-priority Iris job reread and exactly checked the source
  cases, semantic checkpoints, judgment hashes, manual rows, Parquet bytes,
  and markers.
- Across the stable 276-checkpoint snapshot, all 34 manual records leave:

  - baseline: 28,281 pairs, 18,089 false positives, 10,192 true duplicates;
  - treatment: 6,839 pairs, 3,605 false positives, 3,234 true duplicates;
  - combined: 35,120 pairs, 21,694 false positives, 13,426 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T02:12:35Z — 77,736 pairs verified

- Six additional baseline checkpoints passed independent validation: 768
  pairs, 514 model false positives, 253 model true duplicates, and one
  unresolved outcome. All 2,489 judgments were valid on the first attempt,
  with no retries. Fourteen pairs were chunked and 754 were direct.
- Complete inspection resolves the ambiguity as a false positive. The
  479-line member and 319-line canonical are separate renderings of the same
  academic CV and publication page, but the member includes 40 prose
  summaries of individual publications that do not occur anywhere in the
  canonical. The summaries were enumerated and inspected in full; their
  joined SHA-256 is
  `6fd821437b6a0eccaec55f2f285a4249981c0e0814ce55140c53e0a121999510`.
  Character similarity is 0.632220 and line similarity is 0.263158. Pair
  location: `part-00034-of-00128.parquet:1280`; member/canonical text
  SHA-256:
  `01d4854c750a9a89d69fd13b9ea19bfe13fa179688fa2452d1b869e60bf76449`
  /
  `e9a8793028e7c7f30b06090e4d56d8d0ca79227420ad331a07132c97554de71f`.
- The loss pass correctly identified distinct member content. The duplication
  pass incorrectly asserted containment, and the low-confidence tiebreak was
  truncated and self-contradictory. The semantic-judgment SHA-256 is
  `ea18fb964a66b89428c6d92cf731bf78f83d77ae9ba6c4e5e56659be2905750b`.
  The hash-bound manual record has Parquet SHA-256
  `645d3ce98831fa9ff4b883129dc1313131f36ef7cb8802e24b2fda2738611fbc`.
  A separate read-only batch-priority Iris job exactly reread the source
  pair, all 40 summaries, semantic checkpoint, manual row, Parquet bytes, and
  completion marker.
- The six outcome Parquet SHA-256 values for decision-file 34 offsets 256
  through 896 are
  `92f1bb2f4aa5d878f2ea928aeb4f3b3457c02eb7c54d90a8404dbe5a9c7b2b96`,
  `5d931ea65ad5888fcac2cee0539bb222c603438db922057137a766e17c60482d`,
  `753ae21c2e5bb5ace7db45509241d37cb475fed8fd2eea61f1992dcf62db98b6`,
  `a5f8bb9ccbb864f7d75f5df71ee6bc4bf36399970b569923e2f8e4786de1bc80`,
  `17e79dea370f9b33dadfbbc1693a1e34f3a930c17d163dfa884d775f6b9b851b`,
  and
  `3353adac38553fb01a3eff11d3abc83dc0b547768146a339e143b71f36ce39f2`.
- Across the stable 611-checkpoint snapshot, all 84 manual records leave:

  - baseline: 62,100 pairs, 39,562 false positives, 22,538 true duplicates;
  - treatment: 15,636 pairs, 8,109 false positives, 7,527 true duplicates;
  - combined: 77,736 pairs, 47,671 false positives, 30,065 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T01:41:23Z — 76,968 pairs verified

- One additional baseline checkpoint passed independent validation: 128
  pairs, 91 false positives and 37 true duplicates, with no unresolved
  outcomes. All 1,961 judgments were valid on the first attempt. Nineteen
  pairs were chunked and 109 were direct.
- The outcome Parquet SHA-256 is
  `f795b32ead2a2b5a7e04c78e0624ae1c40f89660993c8b1ef138665331b5a07c`.
- Across the stable 605-checkpoint snapshot, all 83 manual records leave:

  - baseline: 61,332 pairs, 39,047 false positives, 22,285 true duplicates;
  - treatment: 15,636 pairs, 8,109 false positives, 7,527 true duplicates;
  - combined: 76,968 pairs, 47,156 false positives, 29,812 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T01:30:07Z — 76,840 pairs verified

- Six additional treatment checkpoints passed independent validation: 709
  pairs, 263 model false positives, 444 model true duplicates, and two
  unresolved outcomes. They contain 1,461 valid responses across 1,476
  attempts; 15 invalid JSON responses affected five retried judgments. All
  709 pairs used direct review.
- Exhaustive character comparison resolves both ambiguities as true
  duplicates:

  - `part-00099-of-00128.parquet:9034` has 15,235 / 15,242 characters
    across 316 lines. The first 15,233 characters match, followed only by
    insertion of `\text{` and `}` around the same boxed answer B. Character
    similarity is 0.999770 and line similarity is 0.996835. Member/canonical
    SHA-256 values are
    `17331a08a1b2b1af915d387c46a7cb16b3fb53b00e90140fa9bb35b2a288f126`
    /
    `0e6714bcaf3ead61b7ebc20c4353953918ec070ad17332ef46e805aebc0dcaa0`.
  - `part-00099-of-00128.parquet:9037` has 15,153 / 15,160 characters
    across 321 lines. The first 15,151 characters match, followed only by
    insertion of `\text{` and `}` around the same boxed answer I. Character
    similarity is 0.999769 and line similarity is 0.996885. Member/canonical
    SHA-256 values are
    `eb1cdaa015c77a738912ac788296664e11410c7b6489250bbcc16ca0256f6e32`
    /
    `17f444058ad87c8cad14a4f191615e102d7fd673ba43c8ca8c5d939236903ca8`.

- The model ambiguity was procedural. JSON parsing exhausted the initial
  passes for both pairs; the first pair's valid tiebreak also called it a true
  duplicate. The exact semantic-judgment SHA-256 values are
  `020790bdf2870e8bd9bdc454dccfa29fe4269403bc18512ae1422f3d65b98f3b`
  and
  `a52d8e4cc6b745babd24289f84ad7539374a65c03eeb3bb967b880b3fb03edd8`.
- The hash-bound manual-record Parquet SHA-256 values are
  `4a99b209c57d4c57978fbec9b8ce2184c27b80e52bbf452fb08175a5675330b1`
  and
  `12860a8e1e18223144511cf718363a583678a0fde3c7a93b303e14bfec04d9fc`.
  A separate read-only batch-priority Iris job reread and exactly checked
  both source cases, all-character diffs, semantic evidence, manual rows,
  Parquet bytes, and completion markers.
- The six outcome Parquet SHA-256 values for p3 decision-file 99 offsets
  5120 through 5760 are:
  `de16d694720408be1a5133ffb2bdfd3d18ca3c23b0aca31a78e103df84257509`,
  `53414d6f997550c95480329ab0a5483cd436204918367559a66fd665e57481a5`,
  `97b4d37226d6da01bc4f48de572be5dc3fd9ebcb598ee50d0dbfd0d87415d65b`,
  `ba068b0857d99a67507ecbf66b1c349a67385e619931f4b5dec8ee8a68dce0d3`,
  `7b8f8b85da55d17558c1b01a20ab7c24932371d5915df52372426eb8a9597540`,
  and
  `0702b71e47c16e42b81699d188fa773a802f90d84ea3141f30597f479228c5ba`.
- Across the stable 604-checkpoint snapshot, all 83 manual records leave:

  - baseline: 61,204 pairs, 38,956 false positives, 22,248 true duplicates;
  - treatment: 15,636 pairs, 8,109 false positives, 7,527 true duplicates;
  - combined: 76,840 pairs, 47,065 false positives, 29,775 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T01:22:13Z — 76,131 pairs verified

- Fifteen additional checkpoints passed independent validation: 1,805 pairs,
  995 model false positives, 803 model true duplicates, and seven unresolved
  outcomes. They contain 3,720 valid responses across 3,760 attempts; 40
  invalid JSON responses affected 14 retried judgments. All 1,805 pairs used
  direct review. This block contains 285 baseline pairs and 1,520 treatment
  pairs.
- Complete character comparison resolves six ambiguities as true duplicates.
  In each pair, every line and character is identical except for `\text{`
  and `}` surrounding the same boxed answer:

  - treatment `part-00003-of-00128.parquet:9093`, answer A, 158 lines;
  - treatment `part-00066-of-00128.parquet:8989`, answer G, 268 lines;
  - treatment `part-00066-of-00128.parquet:9015`, answer C, 234 lines;
  - treatment `part-00066-of-00128.parquet:9016`, answer C, 128 lines;
  - treatment `part-00066-of-00128.parquet:9017`, answer D, 297 lines;
  - baseline `part-00099-of-00128.parquet:7641`, answer H, 252 lines.

- For treatment row 9,015, one otherwise valid model response incorrectly
  claimed that the boxed span contained the full CBT answer. The exhaustive
  character diff disproves that claim: the only difference is the seven-byte
  LaTeX wrapper around the same answer C.
- Complete line review resolves baseline
  `part-00099-of-00128.parquet:7543` as a true duplicate. Both texts contain
  the same positive-real AM-GM request, `(a+b+c)^3 >= 27abc`, AM-GM proof,
  cubing and multiplication steps, equality condition `a=b=c`, and boxed
  conclusion. The differences are prose, spacing, and punctuation. The texts
  contain 1,397 / 1,297 characters and 47 / 43 lines; character similarity is
  0.913883 and line similarity is 0.666667.
- The seven exact semantic-judgment SHA-256 values are:

  - `67ff635616e2bba05f9163655f2b13dba5d0365aefb11467ada897fbb51f3c3a`;
  - `1aa5e83046312e7ae7256172a5869c002525f587bdc158d8b666d497409d8c21`;
  - `5eb4abffdeb538ba800b75f1ec0ba67606c79f4e818ce6d706469c09cd0cc02f`;
  - `646d22d96d83f30cc2509aa6c046cd6e2d9a2c7b3d84ef8a816609c5bc568204`;
  - `06da6a323fc916a2a739d811256923ab26ea0c4adcb3065f30d883ae02eba4d0`;
  - `c4ea231ecff4bb5e06ce8a6421478621e2625954366ae62d3f46773c9d76b4e5`;
  - `96e5c9861a831008f995cf80adc79a5d160e843af30c071423563adbbceba6d6`.

- The corresponding hash-bound manual-record Parquet SHA-256 values are:

  - `0c92a019fb1fe420f1957f90d6d6c32745b2f8f1ab8b3890fd2701ed841e8b2a`;
  - `6a9eee022b097a498db1cd080055725108266128c9c5783a7ce4d68b481a2157`;
  - `b5026e62e93e241149251dac375dd242fd1e580f7bfac5579817dd972bb5e7b6`;
  - `b089ecb2cae03a58c3fb74b2d146628de0e494d3155767b63d8a37ee1936ead6`;
  - `0a78fcb79a7f102c99c7b363b984d030c8a971ec10cc060f7937cf39de296498`;
  - `50e60bc21e652daad2233984e3a40c807c578991e586961f68ee149c853ff6bb`;
  - `14c7a6efddccc2e8efee10d3447af08109d8f44ceac3a91de46270f066796b9c`.

- A separate read-only batch-priority Iris job reread and exactly checked all
  seven source cases, semantic checkpoints, manual rows, Parquet bytes, and
  completion markers.
- The 15 outcome Parquet SHA-256 values are:

  - p0 decision-file 3 offsets 5504 through 5888:
    `b31f8194852104a8cef73d65f28970d22c68f889c7ef5c9a18be2588a5a8c911`,
    `de44b83a7248aa0acfd34fc7731b07f9bde0ac4439387d6b36331715dcfbf130`,
    `80b420b9fb87ac36c5a5398c316d9b99be240b6e20a7c536cc85cd4748cf6a5f`,
    and
    `a38df9b7e8efac85f0a750a6ca80d5c2c3523f63b2e3912a1a0ffabd86d98fca`;
  - p2 decision-file 66 offsets 5248 through 5760:
    `6b4100f1f4571e200f53f61961c375a471eddbe37e2607b68974b949daca7dc0`,
    `6a086461bfcde815bb44c2614a689f31119d89647954ac3b80832f630d862b41`,
    `6d21c9c6a474376919321743732badac46966187ea55c6edceae6221a44ceb10`,
    `2233094a51e040ffb0285da3b6ec494cc42116dd995560c3e9e4f26b7e50ffb2`,
    and
    `9bb65bdfd957874f631d7c3d2fc164dfde840770ddede3393f26b1466cfaec95`;
  - p3 decision-file 99 offsets 4352 through 4992:
    `a8476425adc2554c8d126a59e876aa885e5f8a91198af2eb005940ae1711dc45`,
    `369c60347241af267f8bbe1f6a116c4985b548cac9b04a67094e72fc5f8ef7b9`,
    `2902e40fea151b4cd2ff8f89ed3593e88132470239b7b02eea4d456f716c31f2`,
    `202845322cfb91b73a30a2d1c71b9ee5b07540b1d031ca97c9ccefffcf95753b`,
    `18b2fa678f3c4a11eb6635acd8a554ab10fb3f60724be2717b649c62e290e02f`,
    and
    `0a62082193d517d18af8f403cf6bd5e4119d9d0079b99f2c85a67186f359e569`.

- Across the stable 598-checkpoint snapshot, all 81 manual records leave:

  - baseline: 61,204 pairs, 38,956 false positives, 22,248 true duplicates;
  - treatment: 14,927 pairs, 7,846 false positives, 7,081 true duplicates;
  - combined: 76,131 pairs, 46,802 false positives, 29,329 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T01:09:21Z — 74,326 pairs verified

- Eleven additional checkpoints passed independent validation: 1,408 pairs,
  872 model false positives, 533 model true duplicates, and three unresolved
  outcomes. They contain 6,341 valid responses across 6,356 attempts; 15
  invalid JSON responses affected seven retried judgments. Twenty-nine pairs
  were chunked and 1,379 were direct. This block includes 512 baseline pairs
  and 896 treatment pairs.
- Complete-text review resolves the 8,054-character versus 8,002-character
  treatment wiki/code pair as a false positive. The member alone contains the
  executable `find . -type f -name '*.lua' -print0 | xargs -0 sed -i
  '/^[ \t]*--/d'` command for stripping Lua comments; the canonical ends the
  corresponding defaultconfig advice without it. Character similarity is
  0.961136 and line similarity is 0.702929. Pair location:
  `part-00066-of-00128.parquet:7940`; member/canonical text SHA-256:
  `9335f10db4cddb0a056306773f48890a335d6252338913c13dbf787744ce59a4` /
  `3a0d069d9c4ce3ef8f04c22f48a4ce8ad46af3c2aaba172e3a1b6c5864ed853a`.
- Exact character comparison resolves the 163-line cross-source SFT pair as a
  true duplicate. All 10,633 member characters equal the corresponding
  canonical characters; the canonical's only seven additional characters are
  `\text{` and `}` around the same boxed answer A. Pair location:
  `part-00099-of-00128.parquet:7421`; member/canonical text SHA-256:
  `d9de3b0c40d335ef504f2d1405f4f81ee3738b4715f1d043af9c0a310a59a361` /
  `5b4e0665d494bf2541300bda8049c60e07452a302f7d09079a37ef471a4404f7`.
- Complete-text comparison resolves the 1,427-character versus
  1,312-character inequality SFT pair as a true duplicate. Both contain the
  same positive-real constraint, target inequality, sum-of-cubes identity,
  substitution, three-term sum-of-squares argument, and boxed conclusion.
  The changes are prose and LaTeX formatting, not mathematical content. Pair
  location: `part-00099-of-00128.parquet:7474`; member/canonical text SHA-256:
  `290a41e3723cbc65f25e338873bc15c4fa99547913c3d468b7485e0865a6b4b0` /
  `2c23f6524f14bec5e342831b464e8212d46411119fcb6d1636a4eff09f6c4017`.
- The exact semantic judgments have SHA-256 values
  `78cd5a820a89294377ddd22c35edc001516df4699fdf6e0603f5e29af3e90190`,
  `b4c31ecbe58bb9673368b0a777875054c843d27c55418949a69f18ee13c4034a`,
  and
  `59e2a52277dc38a787bf5e9974047710ad6cad4e40dfa9d08f38a00b79bd38b7`.
  The three hash-bound manual records have Parquet SHA-256 values
  `1c7ac544abbfa24d9bc46a5c873c8c43caeb7a3f94ffba06d6dd08d27012f4af`,
  `07586cfbb92130cef604b88a4f50bf56d386c162f1e2d769ecaf1083cc1e7e89`,
  and
  `88d74f152894cfcee3ca624a725b792674d3b3883ccc3a2d4d4f4b701ff16083`.
  A separate read-only batch-priority Iris process reread and exactly checked
  all three source cases, semantic checkpoints, manual rows, Parquet bytes,
  and completion markers.
- The 11 outcome Parquet SHA-256 values are:

  - p0 decision-file 3 offsets 5120, 5248, and 5376:
    `5f5fa8ffc7ec489ac1fd0d1ef151fbc131056029913dea2a3695b80922dc115b`,
    `a655cb4d9eb48fc83280b6b1e47e29bd36e6cb939d30600d6a7d9d5ffdc6c430`,
    and
    `6ccda91a6d409f87b1d4b832a3738c7e0fa649804f9ca254e88886dd663a1652`;
  - p1 decision-file 34 offset 0:
    `0023a0cefae9bc1edfe1d2b675ebfc233e027d333dbca3425223a7b6e0bf1f59`;
  - p2 decision-file 66 offsets 4736 through 5120:
    `6e503143f7093b286271d6e43d89e65cb0019ebe3bf247fa76d6ec85cccba9cb`,
    `1b35cc00233069e1936e3120a045b805235bcce7f8ad7f0e0828b4afc7d4c403`,
    `691d81b97d1c99407481e925f1e1893e2279dea3031404f4fb1424d89e76e116`,
    and
    `e96a2f5c238ffce531d112fda15b53a0be830d2b6a18175fd5695adaa77035c4`;
  - p3 decision-file 99 offsets 3968, 4096, and 4224:
    `72841ba4f78f7e0c80a3ab70ab1e94e8925f92dea873e4dfbb87e03f08873aae`,
    `b474dd0f6c38af08af21b7b748fb12ccc4e12d2cc755ae3ed945d37de894e1d8`,
    and
    `f6ddfacfbd18ee41e8c095ab1a7659c5ac98c6436911c6c987afa559a66120d1`.

- Across the stable 583-checkpoint snapshot, all 74 manual records leave:

  - baseline: 60,919 pairs, 38,740 false positives, 22,179 true duplicates;
  - treatment: 13,407 pairs, 7,067 false positives, 6,340 true duplicates;
  - combined: 74,326 pairs, 45,807 false positives, 28,519 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T00:52:01Z — 72,918 pairs verified

- Nine additional checkpoints passed independent validation: 1,152 pairs,
  825 false positives and 327 true duplicates, with no unresolved outcomes.
  All 3,339 judgments were valid on the first attempt. Thirteen pairs were
  chunked and 1,139 were direct.
- This block contains 687 baseline pairs (499 false positives and 188 true
  duplicates) and 465 treatment pairs (326 false positives and 139 true
  duplicates).
- The nine outcome Parquet SHA-256 values are:

  - p0 decision-file 3 offsets 4736, 4864, and 4992:
    `f7bfb08124ea3b2888f440771d4bf7ca6012897dd370f872a2f8343ac298f65a`,
    `cbfcc24ea86d0c58f3d2ee863a720917ea2f294192691c6c540572f93f5f4d78`,
    and
    `b3cd4e19c9b30c835c165a1ddba83aec80f17abf589cf84dcaa5da87484b96be`;
  - p2 decision-file 66 offset 4608:
    `a1f21b81fd8125ed1d5ac586441bb3cb938a4115970ea784523c3d4920eae343`;
  - p3 decision-file 99 offsets 3328 through 3840:
    `446babf3985115a75f40deb179b6df7d4b7128a5ce8704c4e6baa9b666beb1a7`,
    `722962725c1fc9f7f1708821e1fc30efaa8f20b88b0b29729ec4ef7223629ca1`,
    `6b3ad2040178978a915c25ba07d0c93a7b42752cda980ab8f12753c68c4b77ef`,
    `031d157e31f10efe083f56a14d84546c437a03a9a59cf155cf814b645e38b1af`,
    and
    `739ff7f4356a6c57672f19c33f8012ce907262a4618c7d36b18cafa11bc57b0e`.

- Across the stable 572-checkpoint snapshot, all 71 manual records leave:

  - baseline: 60,407 pairs, 38,326 false positives, 22,081 true duplicates;
  - treatment: 12,511 pairs, 6,608 false positives, 5,903 true duplicates;
  - combined: 72,918 pairs, 44,934 false positives, 27,984 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T00:49:00Z — 71,766 pairs verified

- Ten additional checkpoints passed independent validation: 1,280 pairs,
  787 model false positives, 490 model true duplicates, and three unresolved
  outcomes. They contain 2,733 valid responses across 2,742 attempts; nine
  invalid JSON responses were the three exhausted attempts for each of three
  judgments. One pair was chunked and 1,279 were direct. This block includes
  1,271 baseline pairs and the next nine treatment pairs.
- Complete issue and trajectory diff resolves the 17,585-character versus
  32,589-character same-repository SWE pair as a false positive. The member
  asks for an `addEsMetadataFields` flag on `SearchResult.getHits`; the
  canonical asks to parse and expose Elasticsearch `matched_queries`. The
  issue specifications, APIs, and exploration trajectories differ despite
  the shared harness and repository snapshot. Pair location:
  `part-00003-of-00128.parquet:7880`; member/canonical text SHA-256:
  `b74a2725c2c1c341b23e90539d1f0ccdd3755c22e74c577342d7ef93cbfffe79` /
  `28da14a7a0ab3d177f2b2532a7df63264303bc3aef0054660b28db61b6777429`.
- Complete line-by-line comparison resolves the cross-source Thirty Years'
  War SFT pair as a true duplicate. All 226 lines contain the same question,
  choices, reasoning, and boxed answer B. The only changed spans add `\text{`
  and its closing `}` around B. Character similarity is 0.999619 and line
  similarity is 0.995575. Pair location:
  `part-00066-of-00128.parquet:7555`; member/canonical text SHA-256:
  `77da9a4e147b049bfdf18fa5967b95772c7b092bd5c7a3e8e147db9b9e59d744` /
  `8f6c457b9b616272d5ad6d7b19f1e3dd8fc38b385285bbab152af3f1632f76ec`.
- Complete paragraph comparison resolves the 2,845-character versus
  3,151-character escape-room SEO pair as a true duplicate under the explicit
  low-value-template boundary. The five member body paragraphs are direct
  word-spun counterparts of the canonical. The canonical has an extra
  introduction; the member's extra title and backlink headline add no
  substantive information. Pair location:
  `part-00099-of-00128.parquet:5089`; member/canonical text SHA-256:
  `860356f684edcb1f8e2610c35658658c6b50ebf7d0e5458cb8878374cbb2ddf6` /
  `67d499f9b4fdc95f8ae9739e6505abcbd3b52e68fbfc55e3277499ec08a8d146`.
- The exact semantic judgments have SHA-256 values
  `67a0da567c09e7a55a4a80be4307e557968c0783390f9c65a6b03809594f54cd`,
  `412f18121aebb4cbfa70bb229ff1b6d39738832a52ab22839125ecc8c53e7381`,
  and
  `134fd68e67baf92aa833901fffed65b5834d340088c055addf5128033db3d79f`.
  The three hash-bound manual records have Parquet SHA-256 values
  `c9bb2e7dce9114b21dab5e95d45442261380f6a2e9e6d9d35efed28df4df148e`,
  `44ee775f65116b69660514bbe5a86104cc78e069234238d4461f676ade17f3e5`,
  and
  `07204b08f5868154cca604d3f54b6fa0d286fd3297a79e63a5e05c5cc6a5b34b`.
  A separate read-only batch-priority Iris process reread and exactly checked
  all three source cases, semantic checkpoints, manual rows, Parquet bytes,
  and completion markers.
- The ten outcome Parquet SHA-256 values are:

  - p0 decision-file 3 offsets 4352, 4480, and 4608:
    `4774e1c37f60bcb350055710d16e7b05e80c4ee7dc835aa90175ccafd5f3f6c9`,
    `5d0086a873ffda5c549c2d881a55a8ed6544f4b1c8984428f5a58866047943ce`,
    and
    `7ceeba7a4be99dafa3ad468c505667669b806b532602c101c63573ebefabeb73`;
  - p2 decision-file 66 offsets 4224, 4352, and 4480:
    `92c1fc06aef69b9f94869bd6d5c78896ab19c3934713bb795e06f912f6bbc142`,
    `6ce25ab14615c6f88ee9cb3b974bd0a76783bbee958c54dc8f4042f10d2abc0d`,
    and
    `54f8611dcb327a5e5d54093ccbc99e8d7b20eda43556fb3a2911da19bcb6cdcc`;
  - p3 decision-file 99 offsets 2816 through 3200:
    `64b6f14f7d11617afd973fe95418cdfc7a31cda73db4f04e85aec5c58e1dfd26`,
    `a8ba0506150ff001c64f4cac0f23b219478ee7e6379b90e7bb6a982b80d2d3fb`,
    `e884d323f37b40d54c15a4f67241d067711c67e788070d380338953fdf0cc82e`,
    and
    `73a4bdbd083ef75fc482288c5e20f503d7f028e2627acb1689035e1d110183ae`.

- Across the stable 563-checkpoint snapshot, all 71 manual records leave:

  - baseline: 59,720 pairs, 37,827 false positives, 21,893 true duplicates;
  - treatment: 12,046 pairs, 6,282 false positives, 5,764 true duplicates;
  - combined: 71,766 pairs, 44,109 false positives, 27,657 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T00:39:35Z — 70,486 pairs verified

- Eighteen additional baseline checkpoints passed independent validation:
  2,304 pairs, 1,528 model false positives, 774 model true duplicates, and two
  unresolved outcomes. They contain 5,107 valid responses across 5,113
  attempts; six invalid JSON responses were the three exhausted attempts for
  each of two judgments. Six pairs were chunked and 2,298 were direct.
- Complete line-by-line comparison resolves the cross-source Aristotle SFT
  pair as a true duplicate. All 174 lines contain the same question, choices,
  reasoning, and boxed answer B. The only changed spans remove `\text{` and its
  closing `}` around B. Character similarity is 0.999575 and line similarity
  is 0.994253. Pair location: `part-00066-of-00128.parquet:7279`;
  member/canonical text SHA-256:
  `abde68acc267def23217145d26e2b38981fdbda529aefd40e61a21e6fd8fc7f1` /
  `6a2c7053436ddf2c7b4377fe87047bae130a2d6e3063222b8aa4df5629b3dcfa`.
- Complete-text comparison resolves the career-spam pair as a false positive.
  Within the shared scaffold, the 553-character member uniquely states that
  medical engineers require a bioengineering or biology bachelor's degree
  with surgical-technology and nursing electives. The 827-character canonical
  instead gives a generic description of biomedicine and surgery, so deleting
  the member loses a distinct career fact. Character similarity is 0.508696
  and line similarity is zero. Pair location:
  `part-00099-of-00128.parquet:4085`; member/canonical text SHA-256:
  `a86cafc57f3d6344effe02650d5577215c04590431dc60285353ad8bf706a249` /
  `4893ce3e5b496530e4b22e31e43f7505e575b1b5c377c62acb8ae20ccb5c1c4e`.
- The exact semantic judgments have SHA-256 values
  `4d6e5b9467ccc7e9e10750bc78546e59ea8f12b4ff6602d5684d27fc8e177073`
  and
  `a5d5e954983a64d8484ac2669ce2b7e618ebdbffeead77a1367a53f51ad50846`.
  The two hash-bound manual records have Parquet SHA-256 values
  `a8d46510336255d0e5dc98d04fd9d2167b7109c18f121ca8d69556431d3b5cdf`
  and
  `3377e0748bf64b4703b4d6e883a97814e7682b84f0a589ab46b6a2459fefc291`.
  A separate read-only batch-priority Iris process reread and exactly checked
  both source cases, semantic checkpoints, manual rows, Parquet bytes, and
  completion markers.
- The 18 outcome Parquet SHA-256 values are:

  - p0 decision-file 3 offsets 3584 through 4224:
    `1a7aa7ff6036dd62f5cdfe782a270e4bc6a6eb6bae43c1a460d4889da9630dc9`,
    `bd52e49a8c084966fc597850478855f47380d582d7b4492dc7b4226c9e2525bc`,
    `ea4f7e1cc2929184ce267d2297e39342b5bfb5388f81778469bfc45727546aa0`,
    `665f17826c08d44e8ec8b74bb1c23a026806cb827b0c69ceec78ff30d77804fd`,
    `d3752cbd128e41b7902a285ff5ffc8463a4407fae7ecd6fc582eddd4fcbbd303`,
    and
    `58eb6e505fc499040aea5cf6972f9d3595f066cedc18379de7fd07553e23f5a8`;
  - p2 decision-file 66 offsets 3456 through 4096:
    `f986b65ce4c9c559e21686431c83f881471306c5cfb26d864f170f662787dac5`,
    `c50218afc78c524ab0a8e711aa716791822eef77f899730381ecfb8ee6047f99`,
    `8677140d53af1bdcb86acc155d2a8fb43de6e790284bf6077ad9adbd4ba56b61`,
    `c4beb4debc6553057d2e07d051ff3c7ec9ece6de2783f665bccef66878ed9216`,
    `8c6df0bdb6d53ac3ea23a083ed93d5e169b16ac092d5f21348a9b9033e7f6061`,
    and
    `618976b69a3e23d0d536de54e7634e61d480c09dca713f6f27932b592ffed78b`;
  - p3 decision-file 99 offsets 2048 through 2688:
    `08dfd47a64a042e2d6c3728be2b1de5be07bc913a6bb8f9ac4a94239dc918e67`,
    `1dce01cc2d6357fbeece487f44309a72fa785c15c948003e7ba8c832296637d7`,
    `d5effda7a2115f5465e97117181c3ccbaf474f5d130450d670309a65250b0f42`,
    `35909820dc1f726712b5d065c2b1bbe77ea2b20e3da6129db5bbe0c8097beb71`,
    `88635a9200ad69d5f23d2dea731925e068e686b7f7607b0374b21a423eaf660b`,
    and
    `690aa7f68752b52bdd21b5e578978819ac633be66008b7605fa5caf8aa7731c7`.

- Across the stable 553-checkpoint snapshot, all 68 manual records leave:

  - baseline: 58,449 pairs, 37,047 false positives, 21,402 true duplicates;
  - treatment: 12,037 pairs, 6,274 false positives, 5,763 true duplicates;
  - combined: 70,486 pairs, 43,321 false positives, 27,165 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T00:31:28Z — 68,182 pairs verified

- Sixteen additional baseline checkpoints passed independent validation:
  2,048 pairs, 1,094 model false positives, 952 model true duplicates, and two
  unresolved outcomes. They contain 4,652 valid judgments across 4,652 request
  attempts, with no invalid responses or retries. Six pairs were chunked and
  2,042 were direct.
- Complete-text comparison resolves the word-spun gambling-history pair
  conservatively as a false positive. The 4,463-character member and
  4,468-character canonical align paragraph for paragraph and share the same
  timeline, but the member says that the Mississippi River was a major trade
  route where merchants brought money. The canonical replaces that subject
  with Korean SEO spam and therefore does not retain the factual statement.
  Deleting the member loses the cleaner fact. Character similarity is
  0.512373 and line similarity is 0.629630. Pair location:
  `part-00003-of-00128.parquet:5750`; member/canonical text SHA-256:
  `930521e30e07f84f923d406866d1c0c6253ab4e516ddbeb360ad1d9ad07205ec` /
  `e0610df9cd62d4883811daa22a00ba85f8bcaf9143d9918621afe9b207130111`.
- Complete-text comparison resolves the career-spam pair as a false positive.
  Before the shared scaffold, the 1,005-character member uniquely contains a
  `Development Engineer Moog` job title and sentence about control-hardware
  analysis, design, development, and testing, plus an
  architecture/sustainable-agriculture study sentence. The 827-character
  canonical contains neither. Character similarity is 0.710699 and line
  similarity is 0.222222. Pair location:
  `part-00066-of-00128.parquet:4902`; member/canonical text SHA-256:
  `cba19a9e39ff27f9f84c2581e2969a0f3651b31811ec059391428373d73dcb99` /
  `4893ce3e5b496530e4b22e31e43f7505e575b1b5c377c62acb8ae20ccb5c1c4e`.
- The exact semantic judgments have SHA-256 values
  `595a910203cf7a6d1ee72f3fca44f22dc222ba87c582d48422237b80446e2030`
  and
  `8a48d78bd6a132a5c738bad4898c2ca78df815e713445ddbda8f8edb89f9a397`.
  The two hash-bound manual records have Parquet SHA-256 values
  `f61295bd1bdc006c4730d05365dfc9d7eebba59bc9c05ca385876b790790c936`
  and
  `69d9dc5a066e1412d1e8a139f89f682ff225095bf51e5166f867f06a6d4305b1`.
  A separate read-only batch-priority Iris process reread and exactly checked
  both source cases, semantic checkpoints, manual rows, Parquet bytes, and
  completion markers.
- Across the stable 535-checkpoint snapshot, all 66 manual records leave:

  - baseline: 56,145 pairs, 35,518 false positives, 20,627 true duplicates;
  - treatment: 12,037 pairs, 6,274 false positives, 5,763 true duplicates;
  - combined: 68,182 pairs, 41,792 false positives, 26,390 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T00:20:14Z — 66,134 pairs verified

- Seven additional baseline checkpoints passed independent validation: 896
  pairs, 527 model false positives, 368 model true duplicates, and one
  unresolved outcome. They contain 1,985 valid judgments across 1,985 request
  attempts, with no invalid responses or retries. Two pairs were chunked and
  894 were direct.
- Complete-text comparison resolves the ambiguity as a true duplicate under
  the audit's explicit low-value-template rule. The 767-character member and
  681-character canonical are the same ForSaleByOwner location-marketing
  template. Removing the member's generic 92-character map/navigation prefix
  and substituting `Englewood` with `Coopertown` plus nine numeric fields makes
  the member character-for-character identical to the canonical. The numeric
  fields are population, commute time and mode share, income and change,
  college and graduate education shares, residency duration, and home value.
  Character similarity is 0.824586 and line similarity is 0.230769. Pair
  location: `part-00066-of-00128.parquet:4298`; member/canonical text SHA-256:
  `69858ade7f503286f7cbb1fbdac39eb8779d691110b120e559ac1c6263a5bdcc` /
  `44d13f015e513e39f09083cacf59fb27f606dca23cb8c9ba3cf81fff6e5a2f97`.
- The exact semantic judgments have SHA-256
  `f8a4492d7009de00868516be733e0dc29ccc7cd775c6fdafdaff61768196d05c`.
  The hash-bound manual record has Parquet SHA-256
  `dcff98e29578993a4b2a6e4351d27c9aa02c13f14dabc954b3aea5dc3bc18ad1`.
  A separate read-only batch-priority Iris process reread and exactly checked
  the source cases, semantic checkpoint, manual row, Parquet bytes, and
  completion marker.
- The seven outcome Parquet SHA-256 values are:

  - p0 decision-file 3 offsets 2688 and 2816:
    `a7c2df9b09094ec952f16191f2cb86264841cc0f139e666e6ec5ba04cd822c43`
    and
    `fbd45d0b24398d594fb39a8fc73343093ef02a6c1ca7e3229ef8b27394048de4`;
  - p2 decision-file 66 offsets 2432, 2560, and 2688:
    `fe85010547f250944c37d1a863930c12c1bd06a2c0bea8d537c4042b2f45ed39`,
    `be26da5457053f3316fea2dd06f504f4e1d68acb73c5993495d007b66c26546c`,
    and
    `87e3ee643add478b07c13c07c19c56a174d11aa4e9eebf3c5a7b21062b30f838`;
  - p3 decision-file 99 offsets 1024 and 1152:
    `e8043e6d3118ce40f7cb3de42d4b89db25d160ea5965ec37298b38a53d8a9622`
    and
    `50ab6f0d17d552f62a1eba0f1808ae6b41f9f9ccadff021e523e8e6c36f69bcc`.

- Across the stable 519-checkpoint snapshot, all 64 manual records leave:

  - baseline: 54,097 pairs, 34,422 false positives, 19,675 true duplicates;
  - treatment: 12,037 pairs, 6,274 false positives, 5,763 true duplicates;
  - combined: 66,134 pairs, 40,696 false positives, 25,438 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T00:11:20Z — 65,238 pairs verified

- Six additional baseline checkpoints passed independent validation: 768
  pairs, 398 false positives, 370 true duplicates, and no unresolved outcomes.
  They contain 1,689 valid judgments across 1,689 request attempts, with no
  invalid responses or retries. Two pairs were chunked and 766 were direct.
- The outcome Parquet SHA-256 values are:

  - p0 decision-file 3 offsets 2432 and 2560:
    `22f1764cfbad1d1c87f0d1dc6aa98cb3906b56be5317f93941a7fff9fbf21e13`
    and
    `fa7f486ee2cbf9230eb7c3b63f5244d4c117aa4cb0518fd802eb782e6e42243e`;
  - p2 decision-file 66 offsets 2176 and 2304:
    `2cf140c1d1f9390d05f9c1080aa19ad706775b23b29c9cd0f83b097355652409`
    and
    `6a5b91f9175d695a08936c689558adcb4759e77276c79bab7613eed64ba92372`;
  - p3 decision-file 99 offsets 768 and 896:
    `a0444c49bc61135685cb678e65f9f2f2765bbb955c013af21bd622dd71ecab4a`
    and
    `cc90f84ab55235044d3d3b0a6e22584f90a440ab3ad98c88ef85c22f6675cbba`.

- Across the stable 512-checkpoint snapshot, all 63 manual records leave:

  - baseline: 53,201 pairs, 33,895 false positives, 19,306 true duplicates;
  - treatment: 12,037 pairs, 6,274 false positives, 5,763 true duplicates;
  - combined: 65,238 pairs, 40,169 false positives, 25,069 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T00:07:35Z — 64,470 pairs verified

- Five additional baseline checkpoints passed independent validation: 640
  pairs, 328 false positives, 312 true duplicates, and no unresolved outcomes.
  They contain 1,600 valid judgments across 1,600 request attempts, with no
  invalid responses or retries. Six pairs were chunked and 634 were direct.
- The outcome Parquet SHA-256 values are:

  - p0 decision-file 3 offsets 2176 and 2304:
    `2614ae9e765116affad01f50bae8708bcc9ff1bd16a70f90457f47d7fff96f8b`
    and
    `abb7687644869c71d7a6209f2ca2d25b7070c96ab08d7c448398c4846f330aa3`;
  - p2 decision-file 66 offset 2048:
    `090cd370e898bdab5100e02d9dd99ea2f983be073129d56443609bf6ba9902a5`;
  - p3 decision-file 99 offsets 512 and 640:
    `61a9e8ed702444f69e7e820f87990f3fd372c0e00fba42bb793d668257ef3c8d`
    and
    `3320a21c442f6b5fa4821e7b2bb1d7c78377b84d8f687df2060f5875f4067506`.

- Across the stable 506-checkpoint snapshot, all 63 manual records leave:

  - baseline: 52,433 pairs, 33,497 false positives, 18,936 true duplicates;
  - treatment: 12,037 pairs, 6,274 false positives, 5,763 true duplicates;
  - combined: 64,470 pairs, 39,771 false positives, 24,699 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T00:04:51Z — 63,830 pairs verified

- Four additional baseline checkpoints passed independent validation: 512
  pairs, 275 false positives, 237 true duplicates, and no unresolved outcomes.
  They contain 1,180 valid judgments across 1,180 request attempts, with no
  invalid responses or retries. Three pairs were chunked and 509 were direct.
- The outcome Parquet SHA-256 values are:

  - p0 decision-file 3 offset 2048:
    `dd55291536f78c6b8aaebf442d61950e5f006ff121612ac0774c4a570c146c38`;
  - p2 decision-file 66 offsets 1792 and 1920:
    `7b134f9d1716809ce517de26b4789821d1a679068618bf2890296c3d8207dcf2`
    and
    `0586fcd73b080f8ed7d61271c44fc475150da7d4f36573f71f931b3f207e1bdc`;
  - p3 decision-file 99 offset 384:
    `e43bbd94385f9fc8c37072f7acd0934dc6249e964080d059748e8d9d522b31b6`.

- Across the stable 501-checkpoint snapshot, all 63 manual records leave:

  - baseline: 51,793 pairs, 33,169 false positives, 18,624 true duplicates;
  - treatment: 12,037 pairs, 6,274 false positives, 5,763 true duplicates;
  - combined: 63,830 pairs, 39,443 false positives, 24,387 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T00:01:53Z — 63,318 pairs verified

- Thirteen additional baseline checkpoints passed independent validation:
  1,664 pairs, 1,108 false positives, 556 true duplicates, and no unresolved
  outcomes. They contain 4,652 valid judgments across 4,652 request attempts,
  with no invalid responses or retries. Fourteen pairs were chunked and 1,650
  were direct.
- The outcome Parquet SHA-256 values are:

  - p0 decision-file 3 offsets 1280 through 1920:
    `80ab1fe7805fcf2784b0ca74847d00a2ac1c51355f8171cf2904695bff209c1f`,
    `465ca20211924deef8cff7f80871251dbde7e43bf6bb927bef32ae86c53a82ba`,
    `fa9c73895e1019b2ce02629bb569c74884337b2bcb45b3b4692cf268c59b9fd0`,
    `b0ed1d1f76035bc041a993667f08d431a9702708644a00c17ab72127ee678965`,
    `f4b00a49d9af34a12855c459e53f73322c6dca9137bac44e35c5d230f3c3f454`,
    and
    `b974fb913fa8f9e9ef03efd6f16e3397fb35983469abc04cd30805caf2b37076`;
  - p2 decision-file 66 offsets 1024 through 1664:
    `49d5facb4a8b8ebf7eddfca81a51014a139c04ba91d50ae19def6a66c8455b78`,
    `211e3e8598bebf85bb09218f922fe2903d411b5c181291a6e48075bd481ec4ec`,
    `306d2f0e5675b88cdfe57a1b94f6d0ee85a1dd851d3d1483ecde556cf05b76d9`,
    `3cf00b25e48618a1e5fb4e8b4b0f3e886e9ae0b13c78475f41cd47ee458b7aba`,
    `9be7d438148c3a22b8a799a765c54065c96f8dfd86367ecd8d2712fe832bd363`,
    and
    `b1dd01f0207cdc82b675d1d53cd0ae2a1bdaa15e5163b69cc9726092844f3f7e`;
  - p3 decision-file 99 offset 256:
    `80bd8b635a38672dbe01cdc0679c3f1bc25cb66c638dd18177189cb5c7bccbc1`.

- Across the stable 497-checkpoint snapshot, all 63 manual records leave:

  - baseline: 51,281 pairs, 32,894 false positives, 18,387 true duplicates;
  - treatment: 12,037 pairs, 6,274 false positives, 5,763 true duplicates;
  - combined: 63,318 pairs, 39,168 false positives, 24,150 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-25T23:59:31Z — 61,654 pairs verified

- Four additional baseline checkpoints passed independent validation: 512
  pairs, 392 model false positives, 119 model true duplicates, and one
  unresolved outcome. They contain 1,077 valid judgments across 1,080 request
  attempts; one judgment exhausted three invalid responses. One pair was
  chunked and 511 were direct.
- Complete-text comparison resolves the ambiguity as a false positive. Both
  records cover the same MathOverflow thread, but the 3,262-character member
  contains Gerald Edgar's distinct request for examples of two locally compact,
  non-discrete groups that are homeomorphic but not topologically isomorphic,
  including his question about which two of compact, connected, and abelian may
  hold. The 3,023-character canonical retains replies to that request but omits
  the request itself. Deleting the member therefore loses a distinct training
  example under the audit rubric. Character similarity is 0.743357 and line
  similarity is 0.395349. Pair location:
  `part-00066-of-00128.parquet:1549`; member/canonical text SHA-256:
  `1e3a17b9c53d7b4c18c81d052522b578dfbfc37e5b89dccd257b8a719836d4d0` /
  `56094a10e874c18bf12a962f1f4249ec1e1e692179dd8ba99f59cb1024984bf2`.
- The exact semantic judgments have SHA-256
  `7f647222798305a693c421dc23a2c9351647f9628b8d8069300afed4b0c84325`.
  The hash-bound manual record has Parquet SHA-256
  `766d344b33cd0688db8107c89779391eaeb3962a3fffba4bbc32660bf704cb1c`.
  A separate read-only batch-priority Iris process reread and exactly checked
  the source cases, semantic checkpoint, manual row, Parquet bytes, and
  completion marker.
- The four outcome Parquet SHA-256 values are:

  - p0 decision-file 3 offsets 1024 and 1152:
    `b8f17f138e7909a9b997f49ac55528fd2efd992b2ee588e1b97f109828b022ee`
    and
    `77637c8c8bedee64b75698ba8d98c4d1d940ce9e1a520e6ce540aff8fcae6cd2`;
  - p2 decision-file 66 offsets 768 and 896:
    `2d7cd94a37fc16bb6762b2bef73eb0158d80cffa34dc748b555142cf0c5ee894`
    and
    `a038da0243b3bb88c794507c23fe098bb816d48da8b59aa1e90e264a2acd1dd1`.

- Across the stable 484-checkpoint snapshot, all 63 manual records leave:

  - baseline: 49,617 pairs, 31,786 false positives, 17,831 true duplicates;
  - treatment: 12,037 pairs, 6,274 false positives, 5,763 true duplicates;
  - combined: 61,654 pairs, 38,060 false positives, 23,594 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-25T23:50:06Z — 61,142 pairs verified

- Three additional baseline checkpoints passed independent validation: 384
  pairs, 249 false positives, 135 true duplicates, and no unresolved outcomes.
  They contain 822 valid judgments across 822 request attempts, with no invalid
  responses or retries. One pair was chunked and 383 were direct.
- The outcome Parquet SHA-256 values are:

  - p0 decision-file 3 offsets 768 and 896:
    `f4fb509c4b4e47c94030247e4722f41180cf8216dea9c4910e7d2c148402429a`
    and
    `e278d423f22ccaabe85f8cb31b5adbaa5c0ce5f1077b75c1c1ae3f3e6ed81ac8`;
  - p2 decision-file 66 offset 640:
    `046c563d61acd502930bc810cec5dac10306ef0ffc3d69eaa0f36e275a3b3af7`.

- Across the stable 480-checkpoint snapshot, all 62 manual records leave:

  - baseline: 49,105 pairs, 31,393 false positives, 17,712 true duplicates;
  - treatment: 12,037 pairs, 6,274 false positives, 5,763 true duplicates;
  - combined: 61,142 pairs, 37,667 false positives, 23,475 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-25T23:48:13Z — 60,758 pairs verified

- Seven additional baseline checkpoints passed independent validation: 896
  pairs, 610 false positives, 286 true duplicates, and no unresolved outcomes.
  They contain 4,423 valid judgments across 4,423 request attempts, with no
  invalid responses or retries. Thirty-nine pairs were chunked and 857 were
  direct.
- The outcome Parquet SHA-256 values are:

  - p0 decision-file 3 offsets 384, 512, and 640:
    `d2b6a5b3f42b169aa477bfe4799ab43b469f07f0bb7c57a9e2d5086df9a23c4d`,
    `e5a91ea9d15d166938bb715e4f77895f2796c1f42b05a62e5965ded0df010aee`,
    and
    `8a91ac4601eb1ffb8847f910eb29ce68b8528133f3574e958f64735843d0ce11`;
  - p2 decision-file 66 offsets 256, 384, and 512:
    `1d57fa26bf3800eafdfb0e44f929b17ee99ba00712da5d866d1f5a7b7a4144ae`,
    `7c7df1f77ab44757966aa0de380d53d7a0ade2ed26cbb0e99e2ecb7eda1eddeb`,
    and
    `f2c87f49334675b6a8912ca0afc0ea88c84c6a3d980d2fab33dd1a77fabfea63`;
  - p3 decision-file 99 offset 128:
    `5d53f26340cf1b39825006d2ed86df369b4aea505a8845c46440903f86801d1c`.

- Across the stable 477-checkpoint snapshot, all 62 manual records leave:

  - baseline: 48,721 pairs, 31,144 false positives, 17,577 true duplicates;
  - treatment: 12,037 pairs, 6,274 false positives, 5,763 true duplicates;
  - combined: 60,758 pairs, 37,418 false positives, 23,340 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-25T23:40:47Z — 59,862 pairs verified

- Partition p0's next baseline checkpoint passed independent validation: 128
  pairs, 91 false positives, 37 true duplicates, and no unresolved outcomes.
  It contains 1,180 valid judgments across 1,180 request attempts, with no
  invalid responses or retries. Eight pairs were chunked and 120 were direct.
  The outcome Parquet SHA-256 is
  `d0856e793e16acd3302229b93337cf80bb0894b0748f329f27835e15a52b981c`.
- Across the stable 470-checkpoint snapshot, all 62 manual records leave:

  - baseline: 47,825 pairs, 30,534 false positives, 17,291 true duplicates;
  - treatment: 12,037 pairs, 6,274 false positives, 5,763 true duplicates;
  - combined: 59,862 pairs, 36,808 false positives, 23,054 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-25T23:32:33Z — 59,734 pairs verified

- Two additional baseline checkpoints passed independent validation: 256
  pairs, 159 false positives, 97 true duplicates, and no unresolved outcomes.
  They contain 2,495 valid judgments across 2,495 request attempts, with no
  invalid responses or retries. Twenty-five pairs were chunked and 231 were
  direct.
- Partition p0's decision-file 3 checkpoint at semantic offset 128 contains
  82 false positives and 46 true duplicates. Its outcome Parquet SHA-256 is
  `c8aa8413513ad757d397ce74aecfa8ce62cb9705efc61506bfd4e5cbd0f9e870`.
  Partition p2's decision-file 66 checkpoint at semantic offset 128 contains
  77 false positives and 51 true duplicates. Its outcome Parquet SHA-256 is
  `4aded5ca7399b9debbdb5111a7e6090f339b1ded9edeb2031fb3c34ad3fd9180`.
- Across the stable 469-checkpoint snapshot, all 62 manual records leave:

  - baseline: 47,697 pairs, 30,443 false positives, 17,254 true duplicates;
  - treatment: 12,037 pairs, 6,274 false positives, 5,763 true duplicates;
  - combined: 59,734 pairs, 36,717 false positives, 23,017 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-25T23:11:51Z — 59,478 pairs verified

- Partition p0's next baseline checkpoint passed independent validation: 128
  pairs, 122 false positives, six true duplicates, and no unresolved outcomes.
  It contains 2,799 valid judgments across 2,799 request attempts, with no
  invalid responses or retries. Thirty-two pairs were chunked and 96 were
  direct. The outcome Parquet SHA-256 is
  `403da56e1a5845263c890f2c509e661c4707cedf8bba9b2642df47ea71129a1b`.
- Across the stable 467-checkpoint snapshot, all 62 manual records leave:

  - baseline: 47,441 pairs, 30,284 false positives, 17,157 true duplicates;
  - treatment: 12,037 pairs, 6,274 false positives, 5,763 true duplicates;
  - combined: 59,478 pairs, 36,558 false positives, 22,920 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-25T22:55:34Z — 59,350 pairs verified

- Partition p2's first baseline checkpoint passed independent validation: 128
  pairs, 118 false positives, ten true duplicates, and no unresolved outcomes.
  It contains 2,325 valid judgments across 2,325 request attempts, with no
  invalid responses or retries. Twenty-three pairs were chunked and 105 were
  direct. The outcome Parquet SHA-256 is
  `dcdc244d7aca5a0b35376d2c8a743391673afe0a9edd3b2536f3609780d6e3df`.
- Across the stable 466-checkpoint snapshot, all 62 manual records leave:

  - baseline: 47,313 pairs, 30,162 false positives, 17,151 true duplicates;
  - treatment: 12,037 pairs, 6,274 false positives, 5,763 true duplicates;
  - combined: 59,350 pairs, 36,436 false positives, 22,914 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-25T22:36:11Z — 59,222 pairs verified

- Partition p3's first baseline checkpoint passed independent validation: 128
  pairs, 119 false positives, nine true duplicates, and no unresolved
  outcomes. It contains 1,987 valid judgments across 1,987 request attempts,
  with no invalid responses or retries. Seventeen pairs were chunked and 111
  were direct. The outcome Parquet SHA-256 is
  `f4cad04e8f9530673a1a8b8da8544dc3d053daa8e77a882337bc6625329761ae`.
- Across the stable 465-checkpoint snapshot, all 62 manual records leave:

  - baseline: 47,185 pairs, 30,044 false positives, 17,141 true duplicates;
  - treatment: 12,037 pairs, 6,274 false positives, 5,763 true duplicates;
  - combined: 59,222 pairs, 36,318 false positives, 22,904 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-25T22:26:01Z — 59,094 pairs verified

- The final three treatment checkpoints in decision file 33 passed independent
  validation: 361 pairs, 195 false positives, 166 true duplicates, and no
  unresolved outcomes. They contain 739 judgments and 741 request attempts:
  739 valid responses and two invalid responses in one retried judgment. All
  361 pairs used direct review.
- Across the stable 464-checkpoint snapshot, all 62 manual records leave:

  - baseline: 47,057 pairs, 29,925 false positives, 17,132 true duplicates;
  - treatment: 12,037 pairs, 6,274 false positives, 5,763 true duplicates;
  - combined: 59,094 pairs, 36,199 false positives, 22,895 true duplicates.

- Partition p1 advanced to decision file 34. All four batch-priority 2-H100
  workers continue serving requests, and their 12 root, broker, and GPU pods
  remain Ready with zero Kubernetes restarts.

### 2026-07-25T22:20:58Z — 58,733 pairs verified

- Four additional treatment checkpoints passed independent validation: 512
  pairs, 184 false positives, 328 true duplicates, and no unresolved outcomes.
  They contain 1,053 valid judgments across 1,053 request attempts, with no
  invalid responses or retries. All 512 pairs used direct review.
- Across the stable 461-checkpoint snapshot, all 62 manual records leave:

  - baseline: 47,057 pairs, 29,925 false positives, 17,132 true duplicates;
  - treatment: 11,676 pairs, 6,079 false positives, 5,597 true duplicates;
  - combined: 58,733 pairs, 36,004 false positives, 22,729 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-25T22:15:44Z — 58,221 pairs verified

- Three additional checkpoints passed independent validation: 384 pairs, 270
  false positives, 114 true duplicates, and no unresolved outcomes. They
  contain 1,134 valid judgments across 1,134 request attempts, with no invalid
  responses or retries. Two pairs were chunked and 382 were direct.
- The block contains 68 baseline pairs, with 62 false positives and six true
  duplicates, and 316 treatment pairs, with 208 false positives and 108 true
  duplicates.
- Across the stable 457-checkpoint snapshot, all 62 manual records leave:

  - baseline: 47,057 pairs, 29,925 false positives, 17,132 true duplicates;
  - treatment: 11,164 pairs, 5,895 false positives, 5,269 true duplicates;
  - combined: 58,221 pairs, 35,820 false positives, 22,401 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-25T22:08:35Z — 57,837 pairs verified

- Two additional baseline checkpoints passed independent validation: 256
  pairs, 172 model false positives, 83 model true duplicates, and one
  unresolved outcome. They contain 527 judgments and 532 request attempts:
  526 valid responses and six invalid responses across three retried
  judgments. All 256 pairs used direct review.
- Complete character comparison resolves the ambiguity as a true duplicate.
  The first 9,174 characters are identical; the member only adds `\text{` and
  its matching `}` around the same final boxed `D`. The member/canonical
  lengths are 9,183 / 9,176 characters and their SHA-256 values are
  `072d4c3f4b24440df2925af66ea94584656d0d29dce5177a9dc8cf44fc3526f1`
  and
  `84d556ca7e793dfe87106a8129e4b841d58e4fdadb9588ee0d01a383ba5c059d`.
  Pair location: `part-00033-of-00128.parquet:7500`.
- The hash-bound manual record has Parquet SHA-256
  `1af569b729c7e27231938b94ef0eb248eceb81558a1eb7ae3f0d1ee7739f4217`
  and semantic-judgments SHA-256
  `0eff1cdeec9146cc59d82fc33f4beb8ec0d9e85f4d03add6185ad2ce80af6da0`.
  A separate batch-priority process reread the full texts, semantic evidence,
  manual record, Parquet bytes, and completion marker exactly.
- Across the stable 454-checkpoint snapshot, all 62 manual records leave:

  - baseline: 46,989 pairs, 29,863 false positives, 17,126 true duplicates;
  - treatment: 10,848 pairs, 5,687 false positives, 5,161 true duplicates;
  - combined: 57,837 pairs, 35,550 false positives, 22,287 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-25T21:58:28Z — 57,581 pairs verified

- Two additional baseline checkpoints passed independent validation: 256
  pairs, 200 false positives, 56 true duplicates, and no unresolved outcomes.
  They contain 529 valid judgments across 529 request attempts, with no invalid
  responses or retries. All 256 pairs used direct review.
- Across the stable 452-checkpoint snapshot, all 61 manual records leave:

  - baseline: 46,733 pairs, 29,691 false positives, 17,042 true duplicates;
  - treatment: 10,848 pairs, 5,687 false positives, 5,161 true duplicates;
  - combined: 57,581 pairs, 35,378 false positives, 22,203 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-25T21:53:42Z — 57,325 pairs verified

- Three additional baseline checkpoints passed independent validation: 384
  pairs, 291 false positives, 93 true duplicates, and no unresolved outcomes.
  They contain 840 valid judgments across 840 request attempts, with no invalid
  responses or retries. One pair was chunked and 383 were direct.
- Across the stable 450-checkpoint snapshot, all 61 manual records leave:

  - baseline: 46,477 pairs, 29,491 false positives, 16,986 true duplicates;
  - treatment: 10,848 pairs, 5,687 false positives, 5,161 true duplicates;
  - combined: 57,325 pairs, 35,178 false positives, 22,147 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-25T21:49:44Z — 56,941 pairs verified

- One additional baseline checkpoint passed independent validation: 128
  pairs, 93 false positives, 35 true duplicates, and no unresolved outcomes.
  It contains 266 valid judgments across 266 request attempts, with no invalid
  responses or retries. All 128 pairs used direct review. The outcome Parquet
  SHA-256 is
  `3beddbd980fa164d5d91f002a6c0720b09a7419acbeac3be7fe19f3660c9fdb4`.
- Across the stable 447-checkpoint snapshot, all 61 manual records leave:

  - baseline: 46,093 pairs, 29,200 false positives, 16,893 true duplicates;
  - treatment: 10,848 pairs, 5,687 false positives, 5,161 true duplicates;
  - combined: 56,941 pairs, 34,887 false positives, 22,054 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-25T21:05:56Z — 49,425 pairs verified

- Eighteen additional baseline checkpoints passed independent validation:
  2,304 pairs, 1,729 model false positives, 569 model true duplicates, and six
  unresolved outcomes. They contain 6,795 valid judgments across 6,828
  attempts; 33 invalid responses affected 13 retried judgments. Twenty pairs
  were chunked and 2,284 were direct.
- Complete full-text comparison resolves all six ambiguities as true
  duplicates. One pair is the same local water-heater service template with
  only the city and state changed. The other five are cross-source SFT pairs
  whose complete answers differ only by `\boxed{B}` versus
  `\boxed{\text{B}}`, or the equivalent formatting around `H` and `F`.
- The six hash-bound manual Parquet records have SHA-256
  `729293dbcb3a24cebeef640a517aecc9c706ea1c28c3db430061616617a2150e`,
  `f2749566b7ef6591052c10d34a78fa9d3358f90641eba135737f1d31fbf12772`,
  `694ddb21c5177471f9fd0f80ef9a4d4ff12c2534294cd33c43a3792e4db7cb31`,
  `16d774183614ad09aea80d29492a64559e786636d69b4e4785406dca51938abe`,
  `2a492bb9e375f0e717fccd5268aacdb9dc0dc9964de8025ba28c2bb51f1f2e95`,
  and
  `c2ec0e86d451966698792a5147481f6180c8056966df919c4a58810cec7f1206`.
  A separate batch-priority Iris process exactly reread all six source pairs,
  semantic checkpoints, manual records, Parquet hashes, and completion
  markers.
- Across the stable 388-checkpoint snapshot, all 49 manual records leave:

  - baseline: 42,105 pairs, 26,765 false positives, 15,340 true duplicates;
  - treatment: 7,320 pairs, 3,826 false positives, 3,494 true duplicates;
  - combined: 49,425 pairs, 30,591 false positives, 18,834 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-25T21:12:44Z — 50,833 pairs verified

- Eleven additional checkpoints passed independent validation: 1,408 pairs,
  979 model false positives, 428 model true duplicates, and one unresolved
  outcome. The block contains 3,980 valid judgments across 3,986 attempts; six
  invalid responses affected three retried judgments. Thirteen pairs were
  chunked and 1,395 were direct.
- Complete character comparison resolves the ambiguity as a true duplicate.
  The cross-source SFT pair contains the same question, options, full
  reasoning, and answer. Across 7,572 / 7,565 characters, the only differences
  are `\text{` and its closing `}` around the final `B`: the member ends in
  `\boxed{\text{B}}` and the canonical ends in `\boxed{B}`. Character
  similarity is 0.999538 and line similarity is 0.992647. The member and
  canonical text SHA-256 values are
  `41342ac9ceb96862cbe21815c560718d266d4cfe3d337dcf1bd47a3d647632a9`
  and
  `857f23088c623ac5d4a2f6744274ab2d7c69b8967e36acafbe2706dfcefe7c41`.
- The hash-bound manual Parquet record has SHA-256
  `ad7aacb32700cd0a11132402729e64cbd5c3c3ebf9bc2880d063b31cb2fe8e52`
  and semantic-judgments SHA-256
  `ea636e33480a2b861c2a645f72b1d7db676d532ab8e94129e3f3cdd0929be526`.
  A separate batch-priority Iris process exactly reread the source pair,
  semantic checkpoint, manual record, and Parquet bytes.
- Across the stable 399-checkpoint snapshot, all 50 manual records leave:

  - baseline: 42,997 pairs, 27,420 false positives, 15,577 true duplicates;
  - treatment: 7,836 pairs, 4,150 false positives, 3,686 true duplicates;
  - combined: 50,833 pairs, 31,570 false positives, 19,263 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-25T21:14:36Z — 52,497 pairs verified

- Thirteen additional checkpoints passed independent validation: 1,664 pairs,
  862 false positives, 802 true duplicates, and no unresolved outcomes. The
  block contains 4,270 valid judgments across 4,270 attempts, with no invalid
  responses or retries. Ten pairs were chunked and 1,654 were direct.
- Across the stable 412-checkpoint snapshot, all 50 manual records leave:

  - baseline: 43,277 pairs, 27,596 false positives, 15,681 true duplicates;
  - treatment: 9,220 pairs, 4,836 false positives, 4,384 true duplicates;
  - combined: 52,497 pairs, 32,432 false positives, 20,065 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-25T21:21:38Z — 53,137 pairs verified

- Five additional checkpoints passed independent validation: 640 pairs, 430
  model false positives, 206 model true duplicates, and four unresolved
  outcomes. The block contains 1,330 valid judgments across 1,351 attempts; 21
  invalid responses affected eight retried judgments. One pair was chunked and
  639 were direct.
- Complete character comparison resolves all four ambiguities as true
  duplicates. Every pair is a cross-source SFT example with identical question,
  options, reasoning, and answer. The only complete-text differences add or
  remove `\text{` and its closing `}` around the same final answer letter. Pair
  lengths are 10,015 / 10,022, 18,211 / 18,218, 9,991 / 9,984, and 11,983 /
  11,976 characters. Character similarity ranges from 0.999650 to 0.999808.
- The four hash-bound manual Parquet records have SHA-256
  `c67e6b06f4df0b29cd9326e7fe8c920f764a7a0d10271cd5a838e91ff190960e`,
  `73973e01eaa8d62160ff12783a8d9e170a4a68adeb73ab124007067a68bf85b2`,
  `6e41f8294d73727fba996cb665ad8f2deb0dc691ca9749a4c7e1c074baeb6ff6`,
  and
  `80542f90385dd5c90d72f9c7b41463c6ade0157b84b630b867d4618434a844e4`.
  A separate batch-priority Iris process exactly reread all four source pairs,
  semantic evidence, manual records, and Parquet bytes.
- Across the stable 417-checkpoint snapshot, all 54 manual records leave:

  - baseline: 43,533 pairs, 27,771 false positives, 15,762 true duplicates;
  - treatment: 9,604 pairs, 5,091 false positives, 4,513 true duplicates;
  - combined: 53,137 pairs, 32,862 false positives, 20,275 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-25T21:31:46Z — 54,927 pairs verified

- Fourteen additional checkpoints passed independent validation: 1,790 pairs,
  1,028 model false positives, 760 model true duplicates, and two unresolved
  outcomes. The block contains 3,689 valid judgments across 3,695 attempts;
  six invalid responses affected four retried judgments. All 1,790 pairs used
  direct review.
- Complete-text review resolves one same-source Masonic SEO pair as a false
  positive. The member uniquely asks about the highest degree, Doctor of
  Philosophy as a non-degree, the three fundamental degrees, and
  darkness-to-light orientation; the canonical has a different Q&A payload.
  Deleting the member loses distinct training examples despite the shared
  template. Character similarity is 0.703975 and line similarity is 0.388060.
  Pair location: `part-00033-of-00128.parquet:2046`; member/canonical text
  SHA-256 values are
  `4fe2c2ce314b573119d9cf3891bcbd26f988841749fc71bdc8b32b496b1cec8c` /
  `09b3a89aab08de4ed660673e3807d643a8d6af9a2e3bb2c741bd5654873c8305`.
- Complete character comparison resolves the second ambiguity as a true
  duplicate. Both cross-source SFT documents have the same marketing A/B test
  question, choices, reasoning, and answer; the only changes delete `\text{`
  and its closing `}` around the same boxed answer I. Character similarity is
  0.999619. Pair location: `part-00098-of-00128.parquet:9066`;
  member/canonical text SHA-256 values are
  `32b5e81e308c598f8771797fb3d0deb4f93de1e6154d5bb62043a114e40508a9` /
  `a690ae4ea88bc2d78c80e858cdf89fbb2588d5b8eba8532797628503928a01b1`.
- The two hash-bound manual Parquet records have SHA-256
  `ec0c83eeafe69bf25f2ca4115b35d07fbfbe88785e486152284f6e25d17c7138`
  and
  `7856db14bf5e2077a1bfcc49d1444a2f03253ad41ddf13c08a3e49bd3150b3de`.
  A separate batch-priority Iris process exactly reread both source pairs,
  semantic checkpoints, manual records, Parquet bytes, and completion markers.
- Across the stable 431-checkpoint snapshot, all 56 manual records leave:

  - baseline: 44,173 pairs, 28,259 false positives, 15,914 true duplicates;
  - treatment: 10,754 pairs, 5,632 false positives, 5,122 true duplicates;
  - combined: 54,927 pairs, 33,891 false positives, 21,036 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-25T21:39:11Z — 55,789 pairs verified

- Seven additional checkpoints passed independent validation: 862 pairs, 399
  model false positives, 459 model true duplicates, and four unresolved
  outcomes. The block contains 1,952 valid judgments across 1,971 attempts; 19
  invalid responses affected seven retried judgments. Two pairs were chunked
  and 860 were direct.
- Complete-text review resolves the baseline ambiguity as a true duplicate.
  Both single-paragraph documents are the same low-value synthetic error
  response: the provided input is incomplete, has no meaningful information,
  and cannot be rewritten. `Anderson` versus `N2AB` is a placeholder slot with
  no standalone fact or distinct training example. Character similarity is
  0.725067. Pair location: `part-00033-of-00128.parquet:3065`;
  member/canonical text SHA-256 values are
  `9527dcf7aac9ac2b782d97c5d8b83aec60aa57a960a99bc3740018a641d1df50` /
  `402d69e85a0c15dd4ec2709b943fb04fc1c9f296a8478fdce1e3cfce9657a53d`.
- Complete character comparison resolves all three treatment ambiguities as
  true duplicates. Each cross-source SFT pair has identical questions,
  choices, reasoning, and answer. Its only two changed spans add or remove
  `\text{` and its closing `}` around the same boxed answer. The pairs cover
  renewable-energy answer D, soft-power answer B, and ribosome answer E;
  character similarities are 0.999646, 0.999503, and 0.999540.
- The four hash-bound manual Parquet records have SHA-256
  `07dd82ad161b68fc875aff23309cf988618dbd1c79938759c8f12999d7452542`,
  `fed59af27ffcf0423472683af7620e27f5c1885f006538415987eda34436cfc9`,
  `09f14760817a69520d60da73f682f18d0d514cc4ad8e6d4b4ce5c39552171fde`,
  and
  `de90010461fa91c2b66058abe1c0cd20e14cfab88c225880a757302d778c1169`.
  A separate batch-priority Iris process exactly reread all four source pairs,
  semantic checkpoints, manual records, Parquet bytes, and completion markers.
- Across the stable 438-checkpoint snapshot, all 60 manual records leave:

  - baseline: 44,941 pairs, 28,603 false positives, 16,338 true duplicates;
  - treatment: 10,848 pairs, 5,687 false positives, 5,161 true duplicates;
  - combined: 55,789 pairs, 34,290 false positives, 21,499 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-25T21:43:59Z — 56,429 pairs verified

- Five additional baseline checkpoints passed independent validation: 640
  pairs, 311 model false positives, 328 model true duplicates, and one
  unresolved outcome. The block contains 1,379 valid judgments across 1,379
  attempts, with no invalid responses or retries. All 640 pairs used direct
  review.
- Complete-text review resolves the ambiguity as a false positive. Beyond the
  college and program slots, the member uniquely gives biomedical-engineering
  degree requirements and corporate-internship career guidance; the canonical
  instead gives targeted-job-board guidance and a different biomedicine
  description. Character similarity is 0.460645 and word-5-gram Jaccard is
  0.131980. Pair location: `part-00033-of-00128.parquet:3942`;
  member/canonical text SHA-256 values are
  `21a5e0a8f766ef47b9e7ec935727b480d78b139a9c8d9648773edb99bd4b5e7b` /
  `4893ce3e5b496530e4b22e31e43f7505e575b1b5c377c62acb8ae20ccb5c1c4e`.
- The hash-bound manual Parquet record has SHA-256
  `998d6a562906e4cbbf0a7577b53f0349b403449aac3b3a24633a2d3cd01655af`
  and semantic-judgments SHA-256
  `6212c5d6b430a4091c3e6aa9c5f15ad7a64d9daec1145264394d1329970e1451`.
  A separate batch-priority Iris process exactly reread the source pair,
  semantic checkpoint, manual record, Parquet bytes, and completion marker.
- Across the stable 443-checkpoint snapshot, all 61 manual records leave:

  - baseline: 45,581 pairs, 28,915 false positives, 16,666 true duplicates;
  - treatment: 10,848 pairs, 5,687 false positives, 5,161 true duplicates;
  - combined: 56,429 pairs, 34,602 false positives, 21,827 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-25T21:45:30Z — 56,685 pairs verified

- Two additional baseline checkpoints passed independent validation: 256
  pairs, 125 false positives, 131 true duplicates, and no unresolved outcomes.
  The block contains 630 valid judgments across 630 attempts, with no invalid
  responses or retries. Two pairs were chunked and 254 were direct.
- Across the stable 445-checkpoint snapshot, all 61 manual records leave:

  - baseline: 45,837 pairs, 29,040 false positives, 16,797 true duplicates;
  - treatment: 10,848 pairs, 5,687 false positives, 5,161 true duplicates;
  - combined: 56,685 pairs, 34,727 false positives, 21,958 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-25T21:46:34Z — 56,813 pairs verified

- One additional baseline checkpoint passed independent validation: 128 pairs,
  67 false positives, 61 true duplicates, and no unresolved outcomes. It
  contains 330 valid judgments across 330 attempts, with no invalid responses
  or retries. One pair was chunked and 127 were direct. The outcome Parquet
  SHA-256 is
  `3cafd74969f4ce1d67692f7516778d71a3bc74c4b92a4b3282957583510e79d2`.
- Across the stable 446-checkpoint snapshot, all 61 manual records leave:

  - baseline: 45,965 pairs, 29,107 false positives, 16,858 true duplicates;
  - treatment: 10,848 pairs, 5,687 false positives, 5,161 true duplicates;
  - combined: 56,813 pairs, 34,794 false positives, 22,019 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-25T20:56:02Z — 47,121 pairs verified

- Eighteen additional baseline checkpoints passed independent validation:
  2,304 pairs, 1,286 model false positives, 1,017 model true duplicates,
  and one unresolved outcome. They contain 5,365 valid judgments across
  5,365 attempts, with no invalid responses or retries. Seven pairs were
  chunked and 2,297 were direct.
- Complete-text review resolves the ambiguity as a false positive. Both
  documents are incoherent college SEO pages, but the member contains complete
  advanced-degree permanence and work-hard guidance absent from the canonical;
  the canonical instead contains early-decision and certification guidance.
  The difference exceeds the template's institution, sport, and program slots.
  Character similarity is 0.564896. The member/canonical text SHA-256 values
  are `dfc0cedac52819f05ad258ad9a053ee42560db812e1dccfbc92c708546c82255` /
  `aa1d8fee3f01e7edece44c88166d773c2ec0b834b92b62194193de5e388ff286`.
- The hash-bound manual record has Parquet SHA-256
  `2e57d2b636e1e550f9fd079e60a466cb0b81c63907b6ac080ef14010c6c20085`
  and semantic-judgments SHA-256
  `7e3fb0ff427b91136896778be83f60a1cf44f8b1db98ae3a0c86c30d22192830`.
  A separate batch-priority Iris process exactly reread the source pair,
  semantic checkpoint, manual record, Parquet bytes, and completion marker.
- Across the stable 370-checkpoint snapshot, all 43 manual records leave:

  - baseline: 39,801 pairs, 25,036 false positives, 14,765 true duplicates;
  - treatment: 7,320 pairs, 3,826 false positives, 3,494 true duplicates;
  - combined: 47,121 pairs, 28,862 false positives, 18,259 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-25T20:49:22Z — 44,817 pairs verified

- Eighteen additional baseline checkpoints passed independent validation:
  2,304 pairs, 1,075 model false positives, 1,225 model true duplicates,
  and four unresolved outcomes. They contain 5,414 valid judgments across
  5,414 attempts, with no invalid responses or retries. Ten pairs were
  chunked and 2,294 were direct.
- Complete-text review resolves three ambiguities as false positives:

  - a college/career SEO page retains member-only ACT browsing instructions,
    medical-engineering degree details, and corporate-internship advice;
  - a shared article has member-only PaySprint, Fratelli Wines, Tinna Trade,
    and Abler Nordic news payloads;
  - a Melio SEO page retains a complete credit-card remittance explanation
    absent from the corrupted canonical.

- The fourth ambiguity is a true duplicate: the complete kitchen-renovation
  pages contain the same builder-selection advice, with only location and
  postcode slots, synonym spinning, and paragraph corruption changed.
- The four hash-bound manual records have Parquet SHA-256 values
  `82fd63ea465bc7e36f5fbf9a7b318a82d06bdc7681c8ce14935844201538efef`,
  `a88081d4b4668597968f28b28e360d5eb0e25378c6612c13eb24f4558c8f379a`,
  `77237d1de7601ea6d8d1cb748d1560abdb29ffbc44be456673922a529b1eb340`,
  and `0221af855232707c72e15f20258e6217465a631d892f665845c65db5cbc602fb`.
  A separate batch-priority Iris process exactly reread the source cases,
  semantic checkpoints, manual records, Parquet bytes, and completion markers.
- Across the stable 352-checkpoint snapshot, all 42 manual records leave:

  - baseline: 37,497 pairs, 23,749 false positives, 13,748 true duplicates;
  - treatment: 7,320 pairs, 3,826 false positives, 3,494 true duplicates;
  - combined: 44,817 pairs, 27,575 false positives, 17,242 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-25T20:09:01Z — 38,545 pairs verified

- Eighteen additional baseline checkpoints passed independent validation:
  2,304 pairs, 1,590 model false positives, 713 model true duplicates, and one
  unresolved outcome. They contain 13,772 valid judgments across 13,772
  request attempts, with no invalid responses or retries. Eighty-three pairs
  were chunked and 2,221 were direct.
- Complete-text review resolves the ambiguity as a true duplicate. Both
  four-line records are the same generic wiki welcome template. Every
  difference is a recipient username, signer, timestamp, punctuation, or the
  superficial phrase `recognize you as` versus `recognize`. The member and
  canonical are 230 and 227 characters with similarity 0.844639; their
  SHA-256 values are
  `428a7bcad94d721c250299a9ace507e54033d9b521351d74427f7c67b38efbd6` and
  `b95af225d148376dd31bc65afece5332219c1b623b789aa5abaf463e4463a47f`.
  Pair location: `part-00098-of-00128.parquet:346`.
- The immutable manual Parquet record has SHA-256
  `a760123a7ea1cd92f5b2f9ea6d59e7e7ce2dadc748d1f647bafb37907f0e97bd`
  and semantic-judgments SHA-256
  `f4a54f05f487ae2f94b244117195a38fd2fdc65b146a79663dfc0a0c0d4aed18`.
  It records all 15 changed spans and binds semantic outcome
  `3b1f29089d2337a72ec5886f83b79cd24b6beab7bc1362d259da82eee9660a6e`.
  A separate batch-priority Iris process exactly reread the complete source
  texts, semantic checkpoint, manual record, Parquet bytes, and completion
  marker.
- Across the stable 303-checkpoint snapshot, all 38 manual records leave:

  - baseline: 31,225 pairs, 20,192 false positives, 11,033 true duplicates;
  - treatment: 7,320 pairs, 3,826 false positives, 3,494 true duplicates;
  - combined: 38,545 pairs, 24,018 false positives, 14,527 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-25T20:12:38Z — 39,441 pairs verified

- Seven additional baseline checkpoints passed independent validation: 896
  pairs, 627 false positives, 269 true duplicates, and no unresolved outcomes.
  They contain 1,904 valid judgments across 1,904 request attempts, with no
  invalid responses or retries. Three pairs were chunked and 893 were direct.
- Across the stable 310-checkpoint snapshot, all 38 manual records leave:

  - baseline: 32,121 pairs, 20,819 false positives, 11,302 true duplicates;
  - treatment: 7,320 pairs, 3,826 false positives, 3,494 true duplicates;
  - combined: 39,441 pairs, 24,645 false positives, 14,796 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-25T20:19:34Z — 41,233 pairs verified

- Fourteen additional baseline checkpoints passed independent validation:
  1,792 pairs, 1,269 false positives, 523 true duplicates, and no unresolved
  outcomes. They contain 3,681 valid judgments across 3,681 request attempts,
  with no invalid responses or retries. All 1,792 pairs used direct review.
- Across the stable 324-checkpoint snapshot, all 38 manual records leave:

  - baseline: 33,913 pairs, 22,088 false positives, 11,825 true duplicates;
  - treatment: 7,320 pairs, 3,826 false positives, 3,494 true duplicates;
  - combined: 41,233 pairs, 25,914 false positives, 15,319 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-25T20:25:15Z — 42,513 pairs verified

- Ten additional baseline checkpoints passed independent validation: 1,280
  pairs, 583 false positives, 697 true duplicates, and no unresolved outcomes.
  They contain 2,678 valid judgments across 2,678 request attempts, with no
  invalid responses or retries. All 1,280 pairs used direct review.
- Across the stable 334-checkpoint snapshot, all 38 manual records leave:

  - baseline: 35,193 pairs, 22,671 false positives, 12,522 true duplicates;
  - treatment: 7,320 pairs, 3,826 false positives, 3,494 true duplicates;
  - combined: 42,513 pairs, 26,497 false positives, 16,016 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-25T19:51:24Z — 36,241 pairs verified

- One additional baseline checkpoint passed independent validation: 128
  pairs, 84 false positives, 44 true duplicates, and no unresolved outcomes.
  It contains 1,287 valid judgments across 1,287 request attempts, with no
  invalid responses or retries. Eleven pairs were chunked and 117 were
  direct. The outcome Parquet SHA-256 is
  `90c4589a69a7fa82786caf6b2439904d8c2b355715ea3f9d118c7cb7d925f8ed`.
- Across the stable 285-checkpoint snapshot, all 37 manual records leave:

  - baseline: 28,921 pairs, 18,602 false positives, 10,319 true duplicates;
  - treatment: 7,320 pairs, 3,826 false positives, 3,494 true duplicates;
  - combined: 36,241 pairs, 22,428 false positives, 13,813 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-25T18:36:26Z — 35,729 pairs verified

- One additional baseline checkpoint passed independent validation: 128
  pairs, 117 false positives, 11 true duplicates, and no unresolved outcomes.
  It contains 1,956 valid judgments across 1,956 request attempts, with no
  invalid responses or retries. Twenty-nine pairs were chunked and 99 were
  direct. The outcome Parquet SHA-256 is
  `ab1910f7e8e6711ba3a01280ae4ae0f324720e3f7a35e0f276626163848b92d4`.
- Across the stable 281-checkpoint snapshot, all 36 manual records leave:

  - baseline: 28,409 pairs, 18,206 false positives, 10,203 true duplicates;
  - treatment: 7,320 pairs, 3,826 false positives, 3,494 true duplicates;
  - combined: 35,729 pairs, 22,032 false positives, 13,697 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-25T18:53:41Z — 35,857 pairs verified

- One additional baseline checkpoint passed independent validation: 128
  pairs, 111 model false positives, 16 model true duplicates, and one
  unresolved outcome. It contains 2,309 valid judgments across 2,309 request
  attempts, with no invalid responses or retries. Twenty-two pairs were
  chunked and 106 were direct. The outcome Parquet SHA-256 is
  `c7fa87d38879e0d9ff82bb9852ae7055e3886a15f6d58fe5f274e39ed9111aa5`.
- Complete line and character comparison resolves the ambiguity as a false
  positive. The member's complete eight-line numeric sequence is
  `6,1,2,2,0,1,0,0`; the canonical's complete eleven-line sequence is
  `9,2,2,2,0,1,0,0,0,1,1`. The first two aligned values differ and the
  canonical has the distinct suffix `0,1,1`; neither text contains the other.
  Character similarity is 0.736842 and line similarity is 0.631579. The
  member/canonical text SHA-256 values are
  `8cf7b251d1ee84a118a3cd40a444c263013f10a00cf6d27e20d246831999a56c` /
  `2c3ef9c9a5a079ae9ce7eeb9d62a54706a9c1b11d766ff380832155adca2fede`.
  Pair location: `part-00065-of-00128.parquet:112`.
- The loss pass incorrectly called the member a represented prefix despite
  identifying the leading-value mismatch. The duplication pass returned a
  low-confidence duplicate verdict, and the high-confidence tiebreak correctly
  identified a distinct numeric example. The hash-bound manual false-positive
  record has Parquet SHA-256
  `d6865fb3a49e9f2e88f10e19e93633a9354c78e56a7fe78443918c14db6d0478`
  and semantic-judgments SHA-256
  `0238ffe0f51b48fb9543fa6fe91cf8832ce26f5f5b15541f131440e0596a6b56`.
  A separate batch-priority Iris process exactly reread the source cases,
  semantic checkpoint, manual record, Parquet bytes, and completion marker.
- Across the stable 282-checkpoint snapshot, all 37 manual records leave:

  - baseline: 28,537 pairs, 18,318 false positives, 10,219 true duplicates;
  - treatment: 7,320 pairs, 3,826 false positives, 3,494 true duplicates;
  - combined: 35,857 pairs, 22,144 false positives, 13,713 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-25T19:06:53Z — 35,985 pairs verified

- One additional baseline checkpoint passed independent validation: 128
  pairs, 118 false positives, 10 true duplicates, and no unresolved outcomes.
  It contains 2,759 valid judgments across 2,759 request attempts, with no
  invalid responses or retries. Twenty-three pairs were chunked and 105 were
  direct. The outcome Parquet SHA-256 is
  `82e4a27f37094fb9c5228c017a5e046209a91a1270df285a487f29cd39d4b99f`.
- Across the stable 283-checkpoint snapshot, all 37 manual records leave:

  - baseline: 28,665 pairs, 18,436 false positives, 10,229 true duplicates;
  - treatment: 7,320 pairs, 3,826 false positives, 3,494 true duplicates;
  - combined: 35,985 pairs, 22,262 false positives, 13,723 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-25T19:46:12Z — 36,113 pairs verified

- One additional baseline checkpoint passed independent validation: 128
  pairs, 82 false positives, 46 true duplicates, and no unresolved outcomes.
  It contains 1,642 valid judgments across 1,642 request attempts, with no
  invalid responses or retries. Sixteen pairs were chunked and 112 were
  direct. The outcome Parquet SHA-256 is
  `190f36c2534305999f4c82f40d6e85bc9a9e73eed13325c95f8b196bfbf1a85b`.
- Across the stable 284-checkpoint snapshot, all 37 manual records leave:

  - baseline: 28,793 pairs, 18,518 false positives, 10,275 true duplicates;
  - treatment: 7,320 pairs, 3,826 false positives, 3,494 true duplicates;
  - combined: 36,113 pairs, 22,344 false positives, 13,769 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-25T18:06:35Z — 35,601 pairs verified

- Four additional treatment checkpoints passed independent validation: 481
  pairs, 221 model false positives, 258 model true duplicates, and two
  unresolved outcomes. They contain 1,026 valid judgments across 1,044
  attempts; 18 invalid responses affected seven retried judgments. One pair
  was chunked and 480 were direct.
- Complete character comparison resolves both ambiguities as true duplicates.
  Every canonical character occurs unchanged in the member; the only two
  changed spans add `\text{` and `}` around the same boxed answer:

  - `part-00097-of-00128.parquet:9115`, 7,245 / 7,238 characters,
    similarity 0.999517; member/canonical text SHA-256
    `647f4c81be7dcc0bcaed08ae4f6f64c93df5e3430fa8bdb0dbd748663ec17aec` /
    `68e53aa3683085d10e452a97f65daa8482d5dca19da2efe65d237b393935edba`;
  - `part-00097-of-00128.parquet:9116`, 17,165 / 17,158 characters,
    similarity 0.999796; member/canonical text SHA-256
    `adfb9a4cf00e34d9d32eb55cedc8a89f5f7805d46d98367b4adb03a1e93bef0c` /
    `cb03db77ddc3f9892821ce78a9d62b18ea8f07b1b3a45ec30c87c817fa570e2b`.

- The manual Parquet records have SHA-256
  `3c54cbb018d2e9da1f0c100f19a2f3b2202d1f7877b99bc3f970d366dcb68782`
  and `08af4a4e85213121780ead4e9e5c7ae3e23c09fe90d61ab3fa411b514d27e259`.
  A separate batch-priority Iris process ran in read-only verification mode
  and exactly reread the source cases, semantic checkpoint, manual records,
  Parquet hashes, and completion markers.
- Across the stable 280-checkpoint snapshot, all 36 manual records leave:

  - baseline: 28,281 pairs, 18,089 false positives, 10,192 true duplicates;
  - treatment: 7,320 pairs, 3,826 false positives, 3,494 true duplicates;
  - combined: 35,601 pairs, 21,915 false positives, 13,686 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.
