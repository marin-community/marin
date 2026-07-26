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

### 2026-07-26T15:29:54Z — 151,202 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1545-v399` independently
  revalidated p0 decision-file 6 semantic offset 3,328. Its 128 baseline pairs
  contain 69 false positives and 59 true duplicates, with no unresolved
  outcomes. All pairs used direct review, and all 283 judgments were valid on
  their first attempts. The outcome Parquet SHA-256 is
  `31661ae781bd83be3ae30e07b17043f415fe83ac1b273a73d6cddd048bcf797a`.

- Across the stable 1,190-checkpoint snapshot, all 166 unresolved model
  outcomes remain covered by 128 true-duplicate and 38 false-positive manual
  records. The adjusted totals are:

  - baseline: 121,603 pairs, 77,169 false positives, 44,434 true duplicates;
  - treatment: 29,599 pairs, 15,235 false positives, 14,364 true duplicates;
  - combined: 151,202 pairs, 92,404 false positives, 58,798 true duplicates.

- The next audit frontiers are p0 `(6, 3,456)`, p1 `(38, 128)`,
  p2 `(70, 256)`, and p3 `(103, 0)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T15:27:44Z — 151,074 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1541-v398` independently
  revalidated p0 decision-file 6 semantic offset 3,200. Its 128 baseline pairs
  contain 71 false positives and 57 true duplicates, with no unresolved
  outcomes. All pairs used direct review, and all 283 judgments were valid on
  their first attempts. The outcome Parquet SHA-256 is
  `0cd6be710977d777f0b13849a31e2a77c1673fbcf1a17be2f92f072bcc0c58a4`.

- Across the stable 1,189-checkpoint snapshot, all 166 unresolved model
  outcomes remain covered by 128 true-duplicate and 38 false-positive manual
  records. The adjusted totals are:

  - baseline: 121,475 pairs, 77,100 false positives, 44,375 true duplicates;
  - treatment: 29,599 pairs, 15,235 false positives, 14,364 true duplicates;
  - combined: 151,074 pairs, 92,335 false positives, 58,739 true duplicates.

- The next audit frontiers are p0 `(6, 3,328)`, p1 `(38, 128)`,
  p2 `(70, 256)`, and p3 `(103, 0)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T15:26:23Z — 150,946 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1539-v397` independently
  revalidated p0 decision-file 6 semantic offset 3,072. Its 128 baseline pairs
  contain 59 false positives and 69 true duplicates, with no unresolved
  outcomes. All pairs used direct review, and all 276 judgments were valid on
  their first attempts. The outcome Parquet SHA-256 is
  `22b82a64d7d2b6a73c7dcf6149c793368b33e080dd18f8ab8ef2d840b0e84903`.

- Across the stable 1,188-checkpoint snapshot, all 166 unresolved model
  outcomes remain covered by 128 true-duplicate and 38 false-positive manual
  records. The adjusted totals are:

  - baseline: 121,347 pairs, 77,029 false positives, 44,318 true duplicates;
  - treatment: 29,599 pairs, 15,235 false positives, 14,364 true duplicates;
  - combined: 150,946 pairs, 92,264 false positives, 58,682 true duplicates.

- The next audit frontiers are p0 `(6, 3,200)`, p1 `(38, 128)`,
  p2 `(70, 256)`, and p3 `(103, 0)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T15:25:07Z — 150,818 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1537-v396` independently
  revalidated p0 decision-file 6 semantic offset 2,944. Its 128 baseline pairs
  contain 64 false positives and 64 true duplicates, with no unresolved
  outcomes. All pairs used direct review, and all 268 judgments were valid on
  their first attempts. The outcome Parquet SHA-256 is
  `65482be7d2934e46cd7eb5bb193fb881a3c5c56adf0375f2e9ab3c98254b933e`.

- Across the stable 1,187-checkpoint snapshot, all 166 unresolved model
  outcomes remain covered by 128 true-duplicate and 38 false-positive manual
  records. The adjusted totals are:

  - baseline: 121,219 pairs, 76,970 false positives, 44,249 true duplicates;
  - treatment: 29,599 pairs, 15,235 false positives, 14,364 true duplicates;
  - combined: 150,818 pairs, 92,205 false positives, 58,613 true duplicates.

- The next audit frontiers are p0 `(6, 3,072)`, p1 `(38, 128)`,
  p2 `(70, 256)`, and p3 `(103, 0)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T15:23:36Z — 150,690 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1535-v395` independently
  revalidated p0 decision-file 6 semantic offset 2,816, p1 decision-file 38
  semantic offset 0, and p2 decision-file 70 semantic offset 128. Their 384
  baseline pairs contain 268 false positives and 116 true duplicates, with no
  unresolved outcomes. Thirty-eight pairs were chunked and 346 were direct.
  All 3,982 judgments were valid on their first attempts. The outcome Parquet
  SHA-256 values are
  `416a1614e5ead0b6c0d99a4673336797766e1ff85ae5eb0d410b647785e01016`,
  `0040816ad1d3e1799c536269fb00e115131bc38e5ab3ed93cdaaff985eed6710`,
  and `5b93fdde909a35cb79cf31f0f45c9b07ec859bec189e611ba5c0a1f414f7a620`.

- Across the stable 1,186-checkpoint snapshot, all 166 unresolved model
  outcomes remain covered by 128 true-duplicate and 38 false-positive manual
  records. The adjusted totals are:

  - baseline: 121,091 pairs, 76,906 false positives, 44,185 true duplicates;
  - treatment: 29,599 pairs, 15,235 false positives, 14,364 true duplicates;
  - combined: 150,690 pairs, 92,141 false positives, 58,549 true duplicates.

- The next audit frontiers are p0 `(6, 2,944)`, p1 `(38, 128)`,
  p2 `(70, 256)`, and p3 `(103, 0)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T15:21:22Z — 150,306 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1533-v394` independently
  revalidated p0 decision-file 6 semantic offset 2,688. Its 128 baseline pairs
  contain 59 false positives and 69 true duplicates, with no unresolved
  outcomes. One pair was chunked and 127 were direct. All 287 judgments were
  valid on their first attempts. The outcome Parquet SHA-256 is
  `0287cae3ddcf475a9b1032632ee60466ad4365282670969ec80035f2103860e9`.

- Across the stable 1,183-checkpoint snapshot, all 166 unresolved model
  outcomes remain covered by 128 true-duplicate and 38 false-positive manual
  records. The adjusted totals are:

  - baseline: 120,707 pairs, 76,638 false positives, 44,069 true duplicates;
  - treatment: 29,599 pairs, 15,235 false positives, 14,364 true duplicates;
  - combined: 150,306 pairs, 91,873 false positives, 58,433 true duplicates.

- The next audit frontiers are p0 `(6, 2,816)`, p1 `(38, 0)`,
  p2 `(70, 128)`, and p3 `(103, 0)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T15:19:57Z — 150,178 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1531-v393` independently
  revalidated p0 decision-file 6 semantic offset 2,560. Its 128 baseline pairs
  contain 64 false positives and 64 true duplicates, with no unresolved
  outcomes. All pairs used direct review, and all 275 judgments were valid on
  their first attempts. The outcome Parquet SHA-256 is
  `e2e39c2fe573c1fd0ef30e169d4f1ba27d89f0d50412b6ffb9e07460a98299b6`.

- Across the stable 1,182-checkpoint snapshot, all 166 unresolved model
  outcomes remain covered by 128 true-duplicate and 38 false-positive manual
  records. The adjusted totals are:

  - baseline: 120,579 pairs, 76,579 false positives, 44,000 true duplicates;
  - treatment: 29,599 pairs, 15,235 false positives, 14,364 true duplicates;
  - combined: 150,178 pairs, 91,814 false positives, 58,364 true duplicates.

- The next audit frontiers are p0 `(6, 2,688)`, p1 `(38, 0)`,
  p2 `(70, 128)`, and p3 `(103, 0)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T15:18:24Z — 150,050 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1529-v392` independently
  revalidated p0 decision-file 6 semantic offset 2,432. Its 128 baseline pairs
  contain 65 false positives and 63 true duplicates, with no unresolved
  outcomes. One pair was chunked and 127 were direct. All 322 judgments were
  valid on their first attempts. The outcome Parquet SHA-256 is
  `9eeabc889422bce149e190b3130ba74f402bf6713f4cd0a659c31dfb1fb0f03c`.

- Across the stable 1,181-checkpoint snapshot, all 166 unresolved model
  outcomes remain covered by 128 true-duplicate and 38 false-positive manual
  records. The adjusted totals are:

  - baseline: 120,451 pairs, 76,515 false positives, 43,936 true duplicates;
  - treatment: 29,599 pairs, 15,235 false positives, 14,364 true duplicates;
  - combined: 150,050 pairs, 91,750 false positives, 58,300 true duplicates.

- The next audit frontiers are p0 `(6, 2,560)`, p1 `(38, 0)`,
  p2 `(70, 128)`, and p3 `(103, 0)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T15:17:06Z — 149,922 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1527-v391` independently
  revalidated p0 decision-file 6 semantic offset 2,304. Its 128 baseline pairs
  contain 58 false positives and 70 true duplicates, with no unresolved
  outcomes. One pair was chunked and 127 were direct. All 274 judgments were
  valid on their first attempts. The outcome Parquet SHA-256 is
  `686aee12a0e4601c65a97fd2c928d96e224041e4037dc2e9b5bba06f2e7cae0d`.

- Across the stable 1,180-checkpoint snapshot, all 166 unresolved model
  outcomes remain covered by 128 true-duplicate and 38 false-positive manual
  records. The adjusted totals are:

  - baseline: 120,323 pairs, 76,450 false positives, 43,873 true duplicates;
  - treatment: 29,599 pairs, 15,235 false positives, 14,364 true duplicates;
  - combined: 149,922 pairs, 91,685 false positives, 58,237 true duplicates.

- The next audit frontiers are p0 `(6, 2,432)`, p1 `(38, 0)`,
  p2 `(70, 128)`, and p3 `(103, 0)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T15:14:01Z — 149,794 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1523-v389` independently
  revalidated p0 decision-file 6 semantic offset 2,176. Its 128 baseline pairs
  contain 55 false positives and 73 true duplicates, with no unresolved
  outcomes. Two pairs were chunked and 126 were direct. All 378 judgments were
  valid on their first attempts. The outcome Parquet SHA-256 is
  `f753bf93f4c588c2ee373df1c79d142a6498b2419e0b817962f766ffee5c80f6`.

- Across the stable 1,179-checkpoint snapshot, all 166 unresolved model
  outcomes remain covered by 128 true-duplicate and 38 false-positive manual
  records. The adjusted totals are:

  - baseline: 120,195 pairs, 76,392 false positives, 43,803 true duplicates;
  - treatment: 29,599 pairs, 15,235 false positives, 14,364 true duplicates;
  - combined: 149,794 pairs, 91,627 false positives, 58,167 true duplicates.

- The next audit frontiers are p0 `(6, 2,304)`, p1 `(38, 0)`,
  p2 `(70, 128)`, and p3 `(103, 0)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T15:12:41Z — 149,666 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1521-v388` independently
  revalidated p0 decision-file 6 semantic offsets 1,792, 1,920, and 2,048.
  Their 384 baseline pairs contain 199 false positives and 185 true
  duplicates, with no unresolved outcomes. All pairs used direct review, and
  all 796 judgments were valid on their first attempts. The outcome Parquet
  SHA-256 values are
  `547ea67cd5fdd82f5e0b9b5f8aef229f2ed3652aff22e92e2d6e8e3392c32822`,
  `d5ed4d75e27d1cdc7d249312fb10498a190735efd451c3ba80be3105a5320ff3`,
  and `4eaac69b5175e24fbc4dc269d61958522901c8a6dfef156fc6689f59fb2c1892`.

- Across the stable 1,178-checkpoint snapshot, all 166 unresolved model
  outcomes remain covered by 128 true-duplicate and 38 false-positive manual
  records. The adjusted totals are:

  - baseline: 120,067 pairs, 76,337 false positives, 43,730 true duplicates;
  - treatment: 29,599 pairs, 15,235 false positives, 14,364 true duplicates;
  - combined: 149,666 pairs, 91,572 false positives, 58,094 true duplicates.

- The next audit frontiers are p0 `(6, 2,176)`, p1 `(38, 0)`,
  p2 `(70, 128)`, and p3 `(103, 0)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T15:10:56Z — 149,282 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1516-v384` independently
  revalidated p0 decision-file 6 semantic offset 1,664 and the final p3
  checkpoint in decision-file 102 at offset 5,760. Their 219 pairs contain
  104 model false positives, 114 model true duplicates, and one unresolved
  outcome. All pairs used direct review. Four invalid responses affected two
  retried judgments; the batch contains 455 valid attempts across 459 request
  attempts. The outcome Parquet SHA-256 values are
  `4ade856f1f5853f960538221a35c94b4928c424711651f36f2fd6676ff409c48`
  and `626e19da9f31b039cc821fcf42fab92c9de66b6b091cc90ad4f8eece1034c4e1`.

- Complete-text inspection resolves the ambiguity as a true duplicate.
  `part-00102-of-00128.parquet:9065` compares 235-line SFT records with
  identical questions, answer choices, reasoning, citations, conclusions, and
  answers. Their only changed line is the final-answer markup:
  `\boxed{B}` versus `\boxed{\text{B}}`. Member/canonical character counts
  are 13,023/13,030; character, line, and word-sequence similarities are
  0.999731, 0.995745, and 0.999765. Text SHA-256 values are
  `66fe8a6a20fc6a8d57538432aca85d7e3d96d790d667455f89344ab763e146bc`
  and
  `22c795ac883b4d5d54b4dae118f82afb7638bea448a84be9f33b09e47c5e75f1`;
  inspection SHA-256 is
  `e27e55b674ab445f6afeb7f7c00dd9493a4b34770a08562779fc0cd220c96728`.

- `/rav/datakit-6854-publish-row9065-1518-v386` wrote the immutable
  true-duplicate record. The separate verify-only job
  `/rav/datakit-6854-verify-row9065-1519-v387` reread the complete source
  texts, semantic evidence, inspection artifact, deterministic Parquet bytes,
  and completion marker. The semantic-evidence, manual-Parquet, and marker
  SHA-256 values are
  `3a187664ca880fe0a09acad4cb67a0c09e8c292c4ad5de2257e301641f99cfa8`,
  `cedfac4d59cf2f7c8f02ff7745dcfd1daf7d98b9de4686ebc7891643d0e514a1`,
  and `071fc949542ac9d2181292549329ae996f26e82287a31ff85796ee62e41c5cc6`.

- Across the stable 1,175-checkpoint snapshot, all 166 unresolved model
  outcomes are covered by 128 true-duplicate and 38 false-positive manual
  records. The adjusted totals are:

  - baseline: 119,683 pairs, 76,138 false positives, 43,545 true duplicates;
  - treatment: 29,599 pairs, 15,235 false positives, 14,364 true duplicates;
  - combined: 149,282 pairs, 91,373 false positives, 57,909 true duplicates.

- The next audit frontiers are p0 `(6, 1,792)`, p1 `(38, 0)`,
  p2 `(70, 128)`, and p3 `(103, 0)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T15:06:31Z — 149,063 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1514-v383` independently
  revalidated p0 decision-file 6 semantic offset 1,536 and p3 decision-file
  102 semantic offset 5,632. Their 256 pairs contain 175 false positives and
  81 true duplicates, with no unresolved outcomes. All pairs used direct
  review, and all 522 judgments were valid on their first attempts. The
  outcome Parquet SHA-256 values are
  `0b6b643817c1c4dccb056be6c06c491b1fd37a6156ab7e6b86b8d6e879eeba64`
  and `60bcac955e320e451f43f7fe206626514eb262c3ba154c8d017dde087ecba61e`.
  The p0 checkpoint contains 128 baseline pairs; the p3 checkpoint contains
  128 treatment pairs.

- Across the stable 1,173-checkpoint snapshot, all 165 unresolved model
  outcomes remain covered by 127 true-duplicate and 38 false-positive manual
  records. The adjusted totals are:

  - baseline: 119,555 pairs, 76,082 false positives, 43,473 true duplicates;
  - treatment: 29,508 pairs, 15,187 false positives, 14,321 true duplicates;
  - combined: 149,063 pairs, 91,269 false positives, 57,794 true duplicates.

- The next audit frontiers are p0 `(6, 1,664)`, p1 `(38, 0)`,
  p2 `(70, 128)`, and p3 `(102, 5,760)`. All four batch-priority 2-H100
  workers continue serving requests. Their 12 root, broker, and GPU pods
  remain Ready with zero Kubernetes restarts.

### 2026-07-26T15:05:15Z — 148,807 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1512-v382` independently
  revalidated p0 decision-file 6 semantic offset 1,408 and p3 decision-file
  102 semantic offset 5,504. Their 256 pairs contain 159 false positives and
  97 true duplicates, with no unresolved outcomes. All pairs used direct
  review, and all 528 judgments were valid on their first attempts. The
  outcome Parquet SHA-256 values are
  `6a6cb5d9e2d659c5d5cf329b3518a26927251ff0c27d17d76a251097e4fae488`
  and `00980805628856603b5825f6da91fc8a04719cbcac1663e379471011e1abd01b`.
  The p0 checkpoint contains 128 baseline pairs; the p3 checkpoint contains
  128 treatment pairs.

- Across the stable 1,171-checkpoint snapshot, all 165 unresolved model
  outcomes remain covered by 127 true-duplicate and 38 false-positive manual
  records. The adjusted totals are:

  - baseline: 119,427 pairs, 76,017 false positives, 43,410 true duplicates;
  - treatment: 29,380 pairs, 15,077 false positives, 14,303 true duplicates;
  - combined: 148,807 pairs, 91,094 false positives, 57,713 true duplicates.

- The next audit frontiers are p0 `(6, 1,536)`, p1 `(38, 0)`,
  p2 `(70, 128)`, and p3 `(102, 5,632)`. All four batch-priority 2-H100
  workers continue serving requests. Their 12 root, broker, and GPU pods
  remain Ready with zero Kubernetes restarts.

### 2026-07-26T15:04:01Z — 148,551 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1510-v381` independently
  revalidated p0 decision-file 6 semantic offset 1,280 and p3 decision-file
  102 semantic offset 5,376. Their 256 pairs contain 164 false positives and
  92 true duplicates, with no unresolved outcomes. All pairs used direct
  review, and all 530 judgments were valid on their first attempts. The
  outcome Parquet SHA-256 values are
  `eb16b1e8b991ec7de508d3a9523ec9dcb97c8ee13b4ab441cf3890d7525aad8b`
  and `6873e6a63aea6c2bece4fcab1d85140dd4e8f4022f7bff291e660656093cd8d0`.
  The p0 checkpoint contains 128 baseline pairs; the p3 checkpoint contains
  128 treatment pairs.

- Across the stable 1,169-checkpoint snapshot, all 165 unresolved model
  outcomes remain covered by 127 true-duplicate and 38 false-positive manual
  records. The adjusted totals are:

  - baseline: 119,299 pairs, 75,903 false positives, 43,396 true duplicates;
  - treatment: 29,252 pairs, 15,032 false positives, 14,220 true duplicates;
  - combined: 148,551 pairs, 90,935 false positives, 57,616 true duplicates.

- The next audit frontiers are p0 `(6, 1,408)`, p1 `(38, 0)`,
  p2 `(70, 128)`, and p3 `(102, 5,504)`. All four batch-priority 2-H100
  workers continue serving requests. Their 12 root, broker, and GPU pods
  remain Ready with zero Kubernetes restarts.

### 2026-07-26T15:02:45Z — 148,295 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1508-v380` independently
  revalidated p0 decision-file 6 semantic offset 1,152 and p3 decision-file
  102 semantic offset 5,248. Their 256 pairs contain 132 false positives and
  124 true duplicates, with no unresolved outcomes. All pairs used direct
  review, and all 529 judgments were valid on their first attempts. The
  outcome Parquet SHA-256 values are
  `7bb3405ab95080633c530617de9d98ef5bf62ce1e9926def4acce75c0270c56d`
  and `8b687bc51d1f4b8cca248e6af737614d476354bcabc95bd692389153c24e8a40`.
  The p0 checkpoint contains 128 baseline pairs; the p3 checkpoint contains
  128 treatment pairs.

- Across the stable 1,167-checkpoint snapshot, all 165 unresolved model
  outcomes remain covered by 127 true-duplicate and 38 false-positive manual
  records. The adjusted totals are:

  - baseline: 119,171 pairs, 75,785 false positives, 43,386 true duplicates;
  - treatment: 29,124 pairs, 14,986 false positives, 14,138 true duplicates;
  - combined: 148,295 pairs, 90,771 false positives, 57,524 true duplicates.

- The next audit frontiers are p0 `(6, 1,280)`, p1 `(38, 0)`,
  p2 `(70, 128)`, and p3 `(102, 5,376)`. All four batch-priority 2-H100
  workers continue serving requests. Their 12 root, broker, and GPU pods
  remain Ready with zero Kubernetes restarts.

### 2026-07-26T15:01:31Z — 148,039 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1506-v379` independently
  revalidated p0 decision-file 6 semantic offset 1,024 and p3 decision-file
  102 semantic offset 5,120. Their 256 pairs contain 159 false positives and
  97 true duplicates, with no unresolved outcomes. All pairs used direct
  review, and all 527 judgments were valid on their first attempts. The
  outcome Parquet SHA-256 values are
  `2293d4c6ee03ed918e077607990787b740c6da9a47a8de25c0463746d7e91231`
  and `c8fa2806e061b6f2fa869d4749ff473b5f96dcd752ffcf7e42226e18dff3a60b`.
  The p0 checkpoint contains 128 baseline pairs; the p3 checkpoint contains
  128 treatment pairs.

- Across the stable 1,165-checkpoint snapshot, all 165 unresolved model
  outcomes remain covered by 127 true-duplicate and 38 false-positive manual
  records. The adjusted totals are:

  - baseline: 119,043 pairs, 75,672 false positives, 43,371 true duplicates;
  - treatment: 28,996 pairs, 14,967 false positives, 14,029 true duplicates;
  - combined: 148,039 pairs, 90,639 false positives, 57,400 true duplicates.

- The next audit frontiers are p0 `(6, 1,152)`, p1 `(38, 0)`,
  p2 `(70, 128)`, and p3 `(102, 5,248)`. All four batch-priority 2-H100
  workers continue serving requests. Their 12 root, broker, and GPU pods
  remain Ready with zero Kubernetes restarts.

### 2026-07-26T14:59:17Z — 147,783 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1504-v378` independently
  revalidated p0 decision-file 6 semantic offset 896 and p3 decision-file 102
  semantic offsets 4,864 and 4,992. Their 384 pairs contain 230 false
  positives and 154 true duplicates, with no unresolved outcomes. All pairs
  used direct review, and all 791 judgments were valid on their first
  attempts. The outcome Parquet SHA-256 values are
  `170b0bc657b5fae4632a5e7fa788c1d02b7b1bf755e76eddb305d5408d67779c`,
  `dc57a3f203e5e0baac4ce6b244ceabe1b175f93ea059a334ad28649747351d76`,
  and `c224ef045e2bc4ecf43a2de3e542f9a871c0a280f8c7d0d8cefe93a5f22039d7`.
  The p0 checkpoint contains 128 baseline pairs; the p3 checkpoints contain
  256 treatment pairs.

- Across the stable 1,163-checkpoint snapshot, all 165 unresolved model
  outcomes remain covered by 127 true-duplicate and 38 false-positive manual
  records. The adjusted totals are:

  - baseline: 118,915 pairs, 75,553 false positives, 43,362 true duplicates;
  - treatment: 28,868 pairs, 14,927 false positives, 13,941 true duplicates;
  - combined: 147,783 pairs, 90,480 false positives, 57,303 true duplicates.

- The next audit frontiers are p0 `(6, 1,024)`, p1 `(38, 0)`,
  p2 `(70, 128)`, and p3 `(102, 5,120)`. All four batch-priority 2-H100
  workers continue serving requests. Their 12 root, broker, and GPU pods
  remain Ready with zero Kubernetes restarts.

### 2026-07-26T14:57:18Z — 147,399 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1501-v377` independently
  revalidated p0 decision-file 6 semantic offset 768 and p3 decision-file 102
  semantic offset 4,736. Their 256 pairs contain 176 false positives and 80
  true duplicates, with no unresolved outcomes. All pairs used direct review,
  and all 532 judgments were valid on their first attempts. The outcome
  Parquet SHA-256 values are
  `97433f8e5f128a3c50534c44b22f3a329a4ae5e08afcb1d419c6cda3a78a9777`
  and `5677e0236bf7340dda4a0e3c1ae786765d925af1e932fb4966f4f84e95ba1422`.
  The p0 checkpoint contains 128 baseline pairs; the p3 checkpoint contains
  128 treatment pairs.

- Across the stable 1,160-checkpoint snapshot, all 165 unresolved model
  outcomes remain covered by 127 true-duplicate and 38 false-positive manual
  records. The adjusted totals are:

  - baseline: 118,787 pairs, 75,486 false positives, 43,301 true duplicates;
  - treatment: 28,612 pairs, 14,764 false positives, 13,848 true duplicates;
  - combined: 147,399 pairs, 90,250 false positives, 57,149 true duplicates.

- The next audit frontiers are p0 `(6, 896)`, p1 `(38, 0)`,
  p2 `(70, 128)`, and p3 `(102, 4,864)`. All four batch-priority 2-H100
  workers continue serving requests. Their 12 root, broker, and GPU pods
  remain Ready with zero Kubernetes restarts.

### 2026-07-26T14:56:01Z — 147,143 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1459-v376` independently
  revalidated p0 decision-file 6 semantic offset 640 and p3 decision-file 102
  semantic offset 4,608. Their 256 pairs contain 129 false positives and 127
  true duplicates, with no unresolved outcomes. One pair was chunked and 255
  were direct. All 537 judgments were valid on their first attempts. The
  outcome Parquet SHA-256 values are
  `6e628a8d09690e636bba1f38929af203550a61ca7656fa9d5d558f6220d840a4`
  and `0b5ab0dff4b615f897387fb2588a380f04a868e3479fdbd7d033438930510438`.
  The p0 checkpoint contains 128 baseline pairs; the p3 checkpoint contains
  128 treatment pairs.

- Across the stable 1,158-checkpoint snapshot, all 165 unresolved model
  outcomes remain covered by 127 true-duplicate and 38 false-positive manual
  records. The adjusted totals are:

  - baseline: 118,659 pairs, 75,409 false positives, 43,250 true duplicates;
  - treatment: 28,484 pairs, 14,665 false positives, 13,819 true duplicates;
  - combined: 147,143 pairs, 90,074 false positives, 57,069 true duplicates.

- The next audit frontiers are p0 `(6, 768)`, p1 `(38, 0)`,
  p2 `(70, 128)`, and p3 `(102, 4,736)`. All four batch-priority 2-H100
  workers continue serving requests. Their 12 root, broker, and GPU pods
  remain Ready with zero Kubernetes restarts.

### 2026-07-26T14:52:58Z — 146,887 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1454-v374` independently
  revalidated p0 decision-file 6 semantic offset 512 and p3 decision-file 102
  semantic offset 4,480. Their 256 pairs contain 193 false positives and 63
  true duplicates, with no unresolved outcomes. All pairs used direct review,
  and all 537 judgments were valid on their first attempts. The outcome
  Parquet SHA-256 values are
  `1d0d833e28d3a53c364b48d75f283bc49c704ef1f0bbec7140f413938d867d1b`
  and `4a6595533d8c26693758e5168dafbfb55032c6c62e95cf800a918f1f26e8f482`.
  The p3 checkpoint crosses the arm boundary and contains 110 baseline and 18
  treatment pairs.

- Across the stable 1,156-checkpoint snapshot, all 165 unresolved model
  outcomes remain covered by 127 true-duplicate and 38 false-positive manual
  records. The adjusted totals are:

  - baseline: 118,531 pairs, 75,339 false positives, 43,192 true duplicates;
  - treatment: 28,356 pairs, 14,606 false positives, 13,750 true duplicates;
  - combined: 146,887 pairs, 89,945 false positives, 56,942 true duplicates.

- The next audit frontiers are p0 `(6, 640)`, p1 `(38, 0)`,
  p2 `(70, 128)`, and p3 `(102, 4,608)`. All four batch-priority 2-H100
  workers continue serving requests. Their 12 root, broker, and GPU pods
  remain Ready with zero Kubernetes restarts.

### 2026-07-26T14:51:34Z — 146,631 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1452-v373` independently
  revalidated p0 decision-file 6 semantic offset 384. Its 128 baseline pairs
  contain 100 false positives and 28 true duplicates, with no unresolved
  outcomes. All pairs used direct review, and all 260 judgments were valid on
  their first attempts. The outcome Parquet SHA-256 is
  `ad97dfe50564d51c43732ad772d678d7e81a977098bfe5a4b9750621bad14033`.

- Across the stable 1,154-checkpoint snapshot, all 165 unresolved model
  outcomes remain covered by 127 true-duplicate and 38 false-positive manual
  records. The adjusted totals are:

  - baseline: 118,293 pairs, 75,161 false positives, 43,132 true duplicates;
  - treatment: 28,338 pairs, 14,591 false positives, 13,747 true duplicates;
  - combined: 146,631 pairs, 89,752 false positives, 56,879 true duplicates.

- The next audit frontiers are p0 `(6, 512)`, p1 `(38, 0)`,
  p2 `(70, 128)`, and p3 `(102, 4,480)`. All four batch-priority 2-H100
  workers continue serving requests. Their 12 root, broker, and GPU pods
  remain Ready with zero Kubernetes restarts.

### 2026-07-26T14:50:14Z — 146,503 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1449-v372` independently
  revalidated three baseline checkpoints: p0 decision-file 6 offset 256 and
  p3 decision-file 102 offsets 4,224 and 4,352. Their 384 pairs contain 233
  false positives and 151 true duplicates, with no unresolved outcomes. Ten
  pairs were chunked and 374 were direct. One invalid response affected one
  retried judgment; all 1,464 judgments have valid final evidence across 1,465
  request attempts. The outcome Parquet SHA-256 values are
  `eed92a67eab49295ab543de5fd8c922210c94ea6393e3e6988cb7fe059329aea`,
  `16bba1550074293eb53bac07e0b3fb96a377191c2eb0e7990e4bfe00bb78d308`,
  and `078bc104d8e6c3282cbd54c36bef2a4faf6177e760b898dc94102a0da1c79a88`.

- Across the stable 1,153-checkpoint snapshot, all 165 unresolved model
  outcomes remain covered by 127 true-duplicate and 38 false-positive manual
  records. The adjusted totals are:

  - baseline: 118,165 pairs, 75,061 false positives, 43,104 true duplicates;
  - treatment: 28,338 pairs, 14,591 false positives, 13,747 true duplicates;
  - combined: 146,503 pairs, 89,652 false positives, 56,851 true duplicates.

- The next audit frontiers are p0 `(6, 384)`, p1 `(38, 0)`,
  p2 `(70, 128)`, and p3 `(102, 4,480)`. All four batch-priority 2-H100
  workers continue serving requests. Their 12 root, broker, and GPU pods
  remain Ready with zero Kubernetes restarts.

### 2026-07-26T14:48:45Z — 146,119 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1444-v371` independently
  revalidated eight baseline checkpoints: p0 decision-file 6 offset 128, p2
  decision-file 70 offset 0, and p3 decision-file 102 offsets 3,456 through
  4,096. Their 1,024 pairs contain 769 false positives and 255 true
  duplicates, with no unresolved outcomes. Forty-three pairs were chunked and
  981 were direct. All 5,985 judgments were valid on their first attempts.
  The outcome Parquet SHA-256 values are
  `971c7e32f996b39570bcfe506f3ac84c8f04b283b5ef9d1226300aa1050e3ab0`,
  `9fc8c8a9a9215eeabf460dded27ad2a2a07596687dadd2dbab5db36504ee736b`,
  `74d285532e9eeea83a117cc348177ae6fa0e7b44e6b8b5cbc4bdb7348775c497`,
  `e2fe9b310a7506b40457db78d3c9c914fc757c27c93e440b44f4650b33bd2eb3`,
  `e02a4a1a3b98f28a8570780be40d8780f170af2e8a79c0b3ca4f33c687d23a10`,
  `c0a9c365a7bbeb717bf1c59133f1e83ba9c2f7e388f83f815fee80c5d95365e8`,
  `7f2c0975ee810d179be971f8a9efb98da63332c48c9781d356ae107e3171732b`,
  and `2d9889b337fb7bb19c4b98818a05642334fa39f4bef127cff86eb9bd975abc8f`.

- Across the stable 1,150-checkpoint snapshot, all 165 unresolved model
  outcomes remain covered by 127 true-duplicate and 38 false-positive manual
  records. The adjusted totals are:

  - baseline: 117,781 pairs, 74,828 false positives, 42,953 true duplicates;
  - treatment: 28,338 pairs, 14,591 false positives, 13,747 true duplicates;
  - combined: 146,119 pairs, 89,419 false positives, 56,700 true duplicates.

- The next audit frontiers are p0 `(6, 256)`, p1 `(38, 0)`,
  p2 `(70, 128)`, and p3 `(102, 4,224)`. All four batch-priority 2-H100
  workers continue serving requests. Their 12 root, broker, and GPU pods
  remain Ready with zero Kubernetes restarts.

### 2026-07-26T14:44:38Z — 145,095 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1439-v367` independently
  revalidated p3 decision-file 102 semantic offsets 3,072, 3,200, and 3,328.
  Their 384 baseline pairs contain 190 model false positives, 193 model true
  duplicates, and one unresolved outcome. Two pairs were chunked and 382 were
  direct. All 888 judgments were valid on their first attempts. The three
  outcome Parquet SHA-256 values are
  `5bc3ad38379ddf64d2afea59cb626e599b62ddb3cc727d90237264e95b840322`,
  `444137bdf88f6814cbd30e0342c6a30f20fb5e39d0be0a8bc9cda6efe8625a11`,
  and `a60e423187fedd412904ce7da6a1aef375c62049bab61de8ccca9f952e5d4bae`.

- Complete-text inspection resolves the ambiguity as a false positive.
  `part-00102-of-00128.parquet:5226` shares the same college-SEO scaffold as
  its canonical, but the member adds coherent advice that a corporate
  internship develops career prospects and work experience, plus certificate
  and adult-education content. Those payloads are absent from the complete
  canonical, so deletion loses content. Member/canonical character counts are
  1,005/827; character, line, and word-sequence similarities are 0.647380,
  0.333333, and 0.577778. Text SHA-256 values are
  `ae6e47b417a51ff1bfc8453f82be7332f40180c03a0c671568dec63a0d0ae7e7`
  and
  `4893ce3e5b496530e4b22e31e43f7505e575b1b5c377c62acb8ae20ccb5c1c4e`;
  inspection SHA-256 is
  `5275d4dd81d4a9fd4ed731f0f1913c2fa2fed1fd91abe6aa92fdec23b10aae45`.

- `/rav/datakit-6854-publish-row5226-1442-v369` wrote the immutable
  false-positive record. The separate verify-only job
  `/rav/datakit-6854-verify-row5226-1443-v370` reread the complete source
  texts, semantic evidence, inspection artifact, deterministic Parquet bytes,
  and completion marker. The semantic-evidence, manual-Parquet, and marker
  SHA-256 values are
  `9502ee62e3e9b8e23e083e15b976d159753bd524066b5603fd50044126e1205f`,
  `b0c62d5d15f3b7a107f0032afec9a906e91d00c664c2255a2c09fbd5617955ee`,
  and `38ace0fb55a5940f5d9d215b4e2091488c0dc0ba96c69814893648a333a39c69`.

- Across the stable 1,142-checkpoint snapshot, all 165 unresolved model
  outcomes are covered by 127 true-duplicate and 38 false-positive manual
  records. The adjusted totals are:

  - baseline: 116,757 pairs, 74,059 false positives, 42,698 true duplicates;
  - treatment: 28,338 pairs, 14,591 false positives, 13,747 true duplicates;
  - combined: 145,095 pairs, 88,650 false positives, 56,445 true duplicates.

- The next audit frontiers are p0 `(6, 128)`, p1 `(38, 0)`,
  p2 `(70, 0)`, and p3 `(102, 3,456)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T14:36:51Z — 144,711 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1432-v363` independently
  revalidated p3 decision-file 102 semantic offset 2,944. Its 128 baseline
  pairs contain 57 model false positives, 70 model true duplicates, and one
  unresolved outcome. All pairs used direct review, and all 276 judgments
  were valid on their first attempts. The outcome Parquet SHA-256 is
  `08fe692fe099394803271578398ae920b21da7fbb695a580d759506569bebb6f`.

- Complete-text inspection resolves the ambiguity as a true duplicate under
  the low-value-template boundary. `part-00102-of-00128.parquet:5046`
  contains the same incoherent college-SEO sentence scaffold as its
  canonical. Institutions, locations, programs, job claims, and superficial
  connective wording are template slots; neither page contains a coherent
  distinct factual or instructional payload outside the template.
  Member/canonical character counts are 761/827; character, line, and
  word-sequence similarities are 0.341310, 0.333333, and 0.210970. Text
  SHA-256 values are
  `25750f00d75bfa364a5ff3a4a0f02e2ef95346e902ebd4bdeb866ae101a26f62`
  and
  `4893ce3e5b496530e4b22e31e43f7505e575b1b5c377c62acb8ae20ccb5c1c4e`;
  inspection SHA-256 is
  `181012a1be79d5987fa1f453049f8dba619dc1b5bcfdff2b55653f46ce6e8c5b`.

- `/rav/datakit-6854-publish-row5046-1437-v365` wrote the immutable
  true-duplicate record. The separate verify-only job
  `/rav/datakit-6854-verify-row5046-1438-v366` reread the complete source
  texts, semantic evidence, inspection artifact, deterministic Parquet bytes,
  and completion marker. The semantic-evidence, manual-Parquet, and marker
  SHA-256 values are
  `39af0fc7686da4b39066dd4804c8235c34fd8e2a94fe90ef678441b2a2770259`,
  `efae13d3db7709790d829e21d9ca28c02b48743dcf77a0f67454c55759701301`,
  and `df9f8fd57fe2d09cd2e961493ba18133411053014674f2d31019bd16935d5121`.

- Across the stable 1,139-checkpoint snapshot, all 164 unresolved model
  outcomes are covered by 127 true-duplicate and 37 false-positive manual
  records. The adjusted totals are:

  - baseline: 116,373 pairs, 73,868 false positives, 42,505 true duplicates;
  - treatment: 28,338 pairs, 14,591 false positives, 13,747 true duplicates;
  - combined: 144,711 pairs, 88,459 false positives, 56,252 true duplicates.

- The next audit frontiers are p0 `(6, 128)`, p1 `(38, 0)`,
  p2 `(70, 0)`, and p3 `(102, 3,072)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T14:30:17Z — 144,583 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1429-v362` independently
  revalidated p3 decision-file 102 semantic offsets 2,688 and 2,816. Their 256
  baseline pairs contain 113 false positives and 143 true duplicates, with no
  unresolved outcomes. One pair was chunked and 255 were direct. All 639
  judgments were valid on their first attempts. The outcome Parquet SHA-256
  values are
  `9b782ad364dd40cdc11b7cc36da6a0d12867ef998fc7d2743b1dcfba42e4ada0`
  and `4ed3415aae0570d470892e3aae378fcce38b5eebbfd8a9a992a4226e3993a99c`.

- Across the stable 1,138-checkpoint snapshot, all 163 unresolved model
  outcomes remain covered by 126 true-duplicate and 37 false-positive manual
  records. The adjusted totals are:

  - baseline: 116,245 pairs, 73,811 false positives, 42,434 true duplicates;
  - treatment: 28,338 pairs, 14,591 false positives, 13,747 true duplicates;
  - combined: 144,583 pairs, 88,402 false positives, 56,181 true duplicates.

- The next audit frontiers are p0 `(6, 128)`, p1 `(38, 0)`,
  p2 `(70, 0)`, and p3 `(102, 2,944)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T14:27:29Z — 144,327 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1427-v361` independently
  revalidated p3 decision-file 102 semantic offsets 2,432 and 2,560. Their 256
  baseline pairs contain 119 false positives and 137 true duplicates, with no
  unresolved outcomes. All pairs used direct review, and all 542 judgments
  were valid on their first attempts. The outcome Parquet SHA-256 values are
  `c74ace9440ae88a5e57ddba5c898979cb51e1baad9e65961a95cebb2c550c106`
  and `c0b4e8b0611c94af5046480b3474ea88276a114939d755fc74f59d600062bd14`.

- Across the stable 1,136-checkpoint snapshot, all 163 unresolved model
  outcomes remain covered by 126 true-duplicate and 37 false-positive manual
  records. The adjusted totals are:

  - baseline: 115,989 pairs, 73,698 false positives, 42,291 true duplicates;
  - treatment: 28,338 pairs, 14,591 false positives, 13,747 true duplicates;
  - combined: 144,327 pairs, 88,289 false positives, 56,038 true duplicates.

- The next audit frontiers are p0 `(6, 128)`, p1 `(38, 0)`,
  p2 `(70, 0)`, and p3 `(102, 2,688)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T14:23:53Z — 144,071 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1423-v360` independently
  revalidated p3 decision-file 102 semantic offset 2,304. Its 128 baseline
  pairs contain 61 false positives and 67 true duplicates, with no unresolved
  outcomes. All pairs used direct review, and all 281 judgments were valid on
  their first attempts. The outcome Parquet SHA-256 is
  `23e577206c3f613dc882fb7e6194ff986bdc4131b26d82cb7ac569a2897f1617`.

- Across the stable 1,134-checkpoint snapshot, all 163 unresolved model
  outcomes remain covered by 126 true-duplicate and 37 false-positive manual
  records. The adjusted totals are:

  - baseline: 115,733 pairs, 73,579 false positives, 42,154 true duplicates;
  - treatment: 28,338 pairs, 14,591 false positives, 13,747 true duplicates;
  - combined: 144,071 pairs, 88,170 false positives, 55,901 true duplicates.

- The next audit frontiers are p0 `(6, 128)`, p1 `(38, 0)`,
  p2 `(70, 0)`, and p3 `(102, 2,432)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T14:22:19Z — 143,943 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1422-v359` independently
  revalidated four p3 decision-file 102 checkpoints at semantic offsets 1,792
  through 2,176. Their 512 baseline pairs contain 233 false positives and 279
  true duplicates, with no unresolved outcomes. Two pairs were chunked and
  510 were direct. All 1,188 judgments were valid on their first attempts. The
  outcome Parquet SHA-256 values, in frontier order, are:

  - `b148f4792df93b1517d881af8869e8277c9d65aed1386ce107578d281700ea3a`;
  - `338b762f24f045093e6c1d7d0a30d828f278b4dbc56981ca046f17b12ea8b194`;
  - `66935ec407c852abcce63e1b2827ab0e42619cbd3c41c768c35b04f22754720a`;
  - `a40cacd6f0394ba66482cfe351aff5c6c77b20e469dba96323761650db3777b4`.

- Across the stable 1,133-checkpoint snapshot, all 163 unresolved model
  outcomes remain covered by 126 true-duplicate and 37 false-positive manual
  records. The adjusted totals are:

  - baseline: 115,605 pairs, 73,518 false positives, 42,087 true duplicates;
  - treatment: 28,338 pairs, 14,591 false positives, 13,747 true duplicates;
  - combined: 143,943 pairs, 88,109 false positives, 55,834 true duplicates.

- The next audit frontiers are p0 `(6, 128)`, p1 `(38, 0)`,
  p2 `(70, 0)`, and p3 `(102, 2,304)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T14:20:01Z — 143,431 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1415-v355` independently
  revalidated p3 decision-file 102 semantic offset 1,664. Its 128 baseline
  pairs contain 50 model false positives, 77 model true duplicates, and one
  unresolved outcome. All pairs used direct review, and all 266 judgments
  were valid on their first attempts. The outcome Parquet SHA-256 is
  `8ece4a96ab8074ed58855a937e82d85bba38621c32e065ba97f411da2add34e5`.

- Complete-text inspection resolves the ambiguity as a true duplicate under
  the low-value-template boundary. `part-00102-of-00128.parquet:2527` compares
  two incoherent booking-listing fragments built from repeated `Objekttyp`,
  `Unterkunft für`, and `Schlafzimmer` fields. `Alberto2016`, `also of`, and
  the different duplicated field fragments are superficial slots or
  corruption, not distinct facts. Member/canonical character counts are
  128/126; character, line, and word-sequence similarities are 0.708661,
  0.666667, and 0.571429. Text SHA-256 values are
  `295dc6bc23caf8d25feb4833786b2fe58bf1d59d388fa0f598eb151f559430a7`
  and
  `c1bebf6a8b76e9c8204efc3845d66e4f269c765e2f3616bc4c8833d074f7c27e`;
  inspection SHA-256 is
  `814e2b0a632b332bb3d8da7bdd3301957500d7bb850d29c31a89f652a39ab3d6`.

- `/rav/datakit-6854-publish-row2527-1420-v357` wrote the immutable
  true-duplicate record. The separate verify-only job
  `/rav/datakit-6854-verify-row2527-1421-v358` reread the complete source
  texts, semantic evidence, inspection artifact, deterministic Parquet bytes,
  and completion marker. The semantic-evidence, manual-Parquet, and marker
  SHA-256 values are
  `5ec4d4d15f896dbf6c4481d1a516921c64cb612b2271e2eafef4add0d93bb729`,
  `7a73b298c641cd7c056617502d61d90568274ecd3ca915a383a904a28fea32e4`,
  and `7a1da455d64ca36b4483675dbbf8f3d9379b820443917ce276598a0f00d30fbf`.

- Across the stable 1,129-checkpoint snapshot, all 163 unresolved model
  outcomes are covered by 126 true-duplicate and 37 false-positive manual
  records. The adjusted totals are:

  - baseline: 115,093 pairs, 73,285 false positives, 41,808 true duplicates;
  - treatment: 28,338 pairs, 14,591 false positives, 13,747 true duplicates;
  - combined: 143,431 pairs, 87,876 false positives, 55,555 true duplicates.

- The next audit frontiers are p0 `(6, 128)`, p1 `(38, 0)`,
  p2 `(70, 0)`, and p3 `(102, 1,792)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T14:14:09Z — 143,303 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1414-v354` independently
  revalidated p3 decision-file 102 semantic offset 1,536. Its 128 baseline
  pairs contain 67 false positives and 61 true duplicates, with no unresolved
  outcomes. One pair was chunked and 127 were direct. All 334 judgments were
  valid on their first attempts. The outcome Parquet SHA-256 is
  `5d16846dfd67b5f7b9d0cd11a6ba87b306453ee7e81949b0ea370c2203a0a235`.

- Across the stable 1,128-checkpoint snapshot, all 162 unresolved model
  outcomes remain covered by 125 true-duplicate and 37 false-positive manual
  records. The adjusted totals are:

  - baseline: 114,965 pairs, 73,235 false positives, 41,730 true duplicates;
  - treatment: 28,338 pairs, 14,591 false positives, 13,747 true duplicates;
  - combined: 143,303 pairs, 87,826 false positives, 55,477 true duplicates.

- The next audit frontiers are p0 `(6, 128)`, p1 `(38, 0)`,
  p2 `(70, 0)`, and p3 `(102, 1,664)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T14:12:25Z — 143,175 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1412-v353` independently
  revalidated four p3 decision-file 102 checkpoints at semantic offsets 1,024
  through 1,408. Their 512 baseline pairs contain 420 false positives and 92
  true duplicates, with no unresolved outcomes. All pairs used direct review,
  and all 1,054 judgments were valid on their first attempts. The outcome
  Parquet SHA-256 values, in frontier order, are:

  - `c26228f50c6dc5959b912b2cb39b9a0a8b08268cb375242dde18446232fa2473`;
  - `4aacafa37f6962077633b9de6808ba1d0fb7acec2fcab475119a27bf22262a8e`;
  - `bd2c25f9df1ec390c16b5684528851134a91058070f55be8465da6ddad764d77`;
  - `5d87b21a55555c7c99d959ada71834f95194dbfb0f2b8907a9c275d8950467a7`.

- Across the stable 1,127-checkpoint snapshot, all 162 unresolved model
  outcomes remain covered by 125 true-duplicate and 37 false-positive manual
  records. The adjusted totals are:

  - baseline: 114,837 pairs, 73,168 false positives, 41,669 true duplicates;
  - treatment: 28,338 pairs, 14,591 false positives, 13,747 true duplicates;
  - combined: 143,175 pairs, 87,759 false positives, 55,416 true duplicates.

- The next audit frontiers are p0 `(6, 128)`, p1 `(38, 0)`,
  p2 `(70, 0)`, and p3 `(102, 1,536)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T14:10:15Z — 142,663 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1405-v349` independently
  revalidated the final 74-pair checkpoint in p1 decision-file 37 at semantic
  offset 5,888 and the 128-pair p3 decision-file 102 checkpoint at offset 896.
  Their 202 pairs contain 123 model false positives, 78 model true duplicates,
  and one unresolved outcome. One pair was chunked and 201 were direct. The
  445 judgments required 452 attempts: 442 valid responses and ten invalid
  responses across four retried judgments. The outcome Parquet SHA-256 values
  are:

  - `d20a6bbb5eeadd4244583ccc9fccf5d3ca026ade0f63faae62221e22a3920a4d`;
  - `3b91178cea4e010d4eee3e3127c3cc2d096a7481feb5ad339c17522dca9528e9`.

- Complete-text inspection resolves the treatment ambiguity as a true
  duplicate. `part-00037-of-00128.parquet:9027` contains the identical
  294-line fitness-app question, choices, reasoning, conclusion, and answer.
  Only `\boxed{H}` versus `\boxed{\text{H}}` differs. Member/canonical
  character counts are 13,604/13,611; character, line, and word-sequence
  similarities are 0.999743, 0.996599, and 0.999764. Text SHA-256 values are
  `393c19cef2248dae724ca7dd09878a18684df9249ca198e4bd5e938587547e0d`
  and
  `447847a4937ff578ac7b887b539fc1b195193e57f7a419382d65cd1181d36136`;
  inspection SHA-256 is
  `be1f10ddbceb155a738dc70603a7e8676c47a3032b00e7cddc465406495b7f06`.

- `/rav/datakit-6854-publish-row9027-1410-v351` wrote the immutable
  true-duplicate record. The separate verify-only job
  `/rav/datakit-6854-verify-row9027-1411-v352` reread the complete source
  texts, semantic evidence, inspection artifact, deterministic Parquet bytes,
  and completion marker. The semantic-evidence, manual-Parquet, and marker
  SHA-256 values are
  `045683d6b6fd14f42bb67d64e777ae579bdfd78acf1adbefaa4b84a865a52703`,
  `38d31b6cec509db8da6d320b352a4f9b5ece2d983dc829b137248073902bd854`,
  and `3500bccfc2238f47aca7894d20a50a3c0db7db98f151f05a1e56f278d75c59b4`.

- Across the stable 1,123-checkpoint snapshot, all 162 unresolved model
  outcomes are covered by 125 true-duplicate and 37 false-positive manual
  records. The adjusted totals are:

  - baseline: 114,325 pairs, 72,748 false positives, 41,577 true duplicates;
  - treatment: 28,338 pairs, 14,591 false positives, 13,747 true duplicates;
  - combined: 142,663 pairs, 87,339 false positives, 55,324 true duplicates.

- The next audit frontiers are p0 `(6, 128)`, p1 `(38, 0)`,
  p2 `(70, 0)`, and p3 `(102, 1,024)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T14:04:26Z — 142,461 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1402-v348` independently
  revalidated p0 decision-file 6 semantic offset 0, four p1 decision-file 37
  checkpoints at offsets 5,376 through 5,760, and p3 decision-file 102 offset
  768. Their 768 pairs contain 409 false positives and 359 true duplicates,
  with no unresolved outcomes. Thirty-two pairs were chunked and 736 were
  direct. All 4,945 judgments were valid on their first attempts. The outcome
  Parquet SHA-256 values, in p0, p1, then p3 frontier order, are:

  - `0efb0d3c5b0cb2aadaa0703947dad477c198de96d14562c7cd1a49c7a243caa0`;
  - `7ed1a39afb232ed66d79b1439a623e822060ff3014e4ae75db592737c91a9f6e`;
  - `edb72cf0a906cb163ce82e75ef22aa5cb796acd16ce731e57d0bf479a88f3e52`;
  - `4d33c7b7ae39763ab6314ad202e39aee6e278c0267ba462e71a115ab6f60a863`;
  - `bfc3b2b5722c6c7cf5bfe7400cd00d3cd895aabf1ca3aee881dccf600fdbeb02`;
  - `006b1b4b28221c998c109fd712118c813293884fe08008c99fc237bbf052db3a`.

- The added labels split by arm as follows:

  - baseline: 256 pairs, 210 false positives, 46 true duplicates;
  - treatment: 512 pairs, 199 false positives, 313 true duplicates.

- Across the stable 1,121-checkpoint snapshot, all 161 unresolved model
  outcomes remain covered by 124 true-duplicate and 37 false-positive manual
  records. The adjusted totals are:

  - baseline: 114,197 pairs, 72,661 false positives, 41,536 true duplicates;
  - treatment: 28,264 pairs, 14,555 false positives, 13,709 true duplicates;
  - combined: 142,461 pairs, 87,216 false positives, 55,245 true duplicates.

- The next audit frontiers are p0 `(6, 128)`, p1 `(37, 5,888)`,
  p2 `(70, 0)`, and p3 `(102, 896)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T14:01:26Z — 141,693 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1400-v347` independently
  revalidated six p1 decision-file 37 checkpoints at semantic offsets 4,608
  through 5,248 and four p3 decision-file 102 checkpoints at offsets 256
  through 640. Their 1,280 pairs contain 861 false positives and 419 true
  duplicates, with no unresolved outcomes. Twelve pairs were chunked and
  1,268 were direct. All 3,443 judgments were valid on their first attempts.
  The outcome Parquet SHA-256 values, in p1 then p3 frontier order, are:

  - `1dee0afa6edb56dc9ed24b7cf48ba4d97f8eef7ecf2590f2b6fdce9e0801cbf9`;
  - `341962a1fc57b6008b248dcf3dbef88632c3cec27a3bb3d8f8a9601b4ce0ca2b`;
  - `0c65f43abf14497b3b849e5c81a2bcd8e41bb27a3aea17e6ca425d601eb771ac`;
  - `d1381364a25fb28b8a2b46d516e93f07057c7c179125a47d66ee36db1bf565f6`;
  - `0287aea7a1700b13d8590a0737d273bf018164d78273b11bfb4dadc1b761c33e`;
  - `cabdc14086f398137500a9760aa655f27cfb9fc9c525070b112544f024f59703`;
  - `70b56c3065c5bf84b43ac6094db1d3a0ed14772810cc23387de3b9a1ea1bc1fe`;
  - `ea73e507aba47a84fddb1d1bb440cb00a4bec1a4e7f504e91a873be6f1d99cfe`;
  - `1329eac6790a8c82e81a0305e8f0f88a29a0f71f0765d47b6728020f9e0c36b1`;
  - `2f343b3e584180da424d99f246b614e3df52af842f5a0c6a252c9ad52fe7a751`.

- The added labels split by arm as follows:

  - baseline: 673 pairs, 491 false positives, 182 true duplicates;
  - treatment: 607 pairs, 370 false positives, 237 true duplicates.

- Across the stable 1,115-checkpoint snapshot, all 161 unresolved model
  outcomes remain covered by 124 true-duplicate and 37 false-positive manual
  records. The adjusted totals are:

  - baseline: 113,941 pairs, 72,451 false positives, 41,490 true duplicates;
  - treatment: 27,752 pairs, 14,356 false positives, 13,396 true duplicates;
  - combined: 141,693 pairs, 86,807 false positives, 54,886 true duplicates.

- The next audit frontiers are p0 `(6, 0)`, p1 `(37, 5,376)`,
  p2 `(70, 0)`, and p3 `(102, 768)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T13:58:24Z — 140,413 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1346-v342` independently
  revalidated p1 decision-file 37 checkpoints at semantic offsets 4,352 and
  4,480 plus p3 decision-file 102 offset 128. Their 384 baseline pairs contain
  244 model false positives, 137 model true duplicates, and three unresolved
  outcomes. Nineteen pairs were chunked and 365 were direct. The 2,405
  judgments required 2,417 attempts: 2,399 valid responses and 18 invalid
  responses across six retried judgments. The outcome Parquet SHA-256 values
  are:

  - `2e9c7e4833b927f04ae45a371192717f6c54af9979a2d01ef5b30c90ac5ea8c2`;
  - `9c0c8084d50959f53a11895446ce1823795a9302db6e7467295f6dc43c32bce5`;
  - `10b518bbcda835dfbadcbc099ed1ec3aa9f99b26a00274dc2b0e33e760919acf`.

- Complete-text inspection resolves all three ambiguities as true duplicates:

  - `part-00037-of-00128.parquet:7403` contains the same math question,
    choices, factorization, approximation, and answer. One sentence restates
    the same approximation in equivalent wording. Member/canonical character
    counts are 529/519; character, line, and word-sequence similarities are
    0.940840, 0.882353, and 0.931937. Text SHA-256 values are
    `82a9b80826c6cabf6de8a8929e3f0fc59cc7d9e74e33ed3265ce458c625aa811`
    and
    `de105429ebf9b34c839e94ad7f9ef34f939257f60e4b2e60727ebd1f3902c5e3`;
    inspection SHA-256 is
    `e1915b63f9b3f933729d5c9bfd984b896b3aadc8561c84a269fa9570b087a7d0`.
  - `part-00037-of-00128.parquet:7426` contains the identical 392-line
    electrical-grid question, choices, reasoning, conclusion, and answer.
    Only `\boxed{D}` versus `\boxed{\text{D}}` differs. Member/canonical
    character counts are 19,038/19,045; character, line, and word-sequence
    similarities are 0.999816, 0.997449, and 0.999832. Text SHA-256 values are
    `080ee0819406e5cdfe4f71cb2a87adbbdcc449e9b3c66e3b1a597bdef135da86`
    and
    `c27fac1589857850d03f979d4bd16b9fa913af45bfc0439b6de82d18e2d2d99d`;
    inspection SHA-256 is
    `58aa21d81c76ddd01a6ab267019b975764fa8605afbc2ddd0d051b19150c9214`.
  - `part-00037-of-00128.parquet:7627` contains the identical 84-line
    différance question, choices, reasoning, conclusion, and answer. Only
    `\boxed{\text{D}}` versus `\boxed{D}` differs. Member/canonical character
    counts are 3,809/3,802; character, line, and word-sequence similarities
    are 0.999080, 0.988095, and 0.999136. Text SHA-256 values are
    `473edfb001e334470d708fe815e7025c16159554be71239a312d65fcdc57d391`
    and
    `f37d69207e741503a3652204537f7d7e461de886f947b3555611b80096415ce1`;
    inspection SHA-256 is
    `11385900ab0c1e6761a5f8f0f42fd3417f9ebf9e137acef8ae70724dbe47f54a`.

- `/rav/datakit-6854-publish-manual-three-1355-v345` wrote the three immutable
  true-duplicate records. The separate verify-only job
  `/rav/datakit-6854-verify-manual-three-1357-v346` reread the complete source
  texts, semantic evidence, inspection artifacts, deterministic Parquet
  bytes, and completion markers. In pair order, semantic-evidence,
  manual-Parquet, and marker SHA-256 values are:

  - `192f43db3e23dca50490f3b073d202428fc010032ff1880e959afe62896c51e5`,
    `fd11598cf207f5033dc626da056a80975d57bc1e89e157722f06c4d26d676f8b`,
    and `6210719d67a3c81796af31b426ee1daedbd449746ca4897904874e6c6916f707`;
  - `01f815ca15e4f974bbff4f4c33f521a07e3e914ceddd0cca356bdc9735cac559`,
    `b465620684c54edae619725237412c8b9f29a875e9826498faafac3ecc9034bc`,
    and `fcabc126ab14b2b0bf1c7f8b57f702836c354acd2e85eb531fe9c2f16a7c3f79`;
  - `9ccc903e59e4731adb0b20005e2e5e89f57293f41d1d55f488f0af9cf64e05e8`,
    `f1cb7a74f77a1db301f7a7070d7c021777af3b2d93ae00b8b6a80705eb351fb4`,
    and `db5f0d407b3f9ee7a4e2831a0b54dff32629d8115164d349a1a52fa18a5c57ab`.

- Across the stable 1,105-checkpoint snapshot, all 161 unresolved model
  outcomes are covered by 124 true-duplicate and 37 false-positive manual
  records. The adjusted totals are:

  - baseline: 113,268 pairs, 71,960 false positives, 41,308 true duplicates;
  - treatment: 27,145 pairs, 13,986 false positives, 13,159 true duplicates;
  - combined: 140,413 pairs, 85,946 false positives, 54,467 true duplicates.

- The next audit frontiers are p0 `(6, 0)`, p1 `(37, 4,608)`,
  p2 `(70, 0)`, and p3 `(102, 256)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T13:44:20Z — 140,029 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1343-v341` independently
  revalidated three p1 decision-file 37 checkpoints at semantic offsets 3,968
  through 4,224. Their 384 baseline pairs contain 296 false positives and 88
  true duplicates, with no unresolved outcomes. All pairs used direct review,
  and all 795 judgments were valid on their first attempts. The outcome
  Parquet SHA-256 values, in frontier order, are:

  - `11d30414b85a21b0bc0d00ad398dd0d25d7ebccf6a61607569cc195051661118`;
  - `6ed4b626f66f38d43eb6e3c870acad284c3345d412806fe2c4332b222aa63335`;
  - `040ebd5ae1c85b9da0a54c21ec88bce5fad70ee7cc0e5b6139c8c344e00cb91d`.

- Across the stable 1,102-checkpoint snapshot, all 158 unresolved model
  outcomes remain covered by 121 true-duplicate and 37 false-positive manual
  records. The adjusted totals are:

  - baseline: 112,884 pairs, 71,716 false positives, 41,168 true duplicates;
  - treatment: 27,145 pairs, 13,986 false positives, 13,159 true duplicates;
  - combined: 140,029 pairs, 85,702 false positives, 54,327 true duplicates.

- The next audit frontiers are p0 `(6, 0)`, p1 `(37, 4,352)`,
  p2 `(70, 0)`, and p3 `(102, 128)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T13:40:30Z — 139,645 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1338-v340` independently
  revalidated p1 decision-file 37 semantic offset 3,840. Its 128 baseline
  pairs contain 99 false positives and 29 true duplicates, with no unresolved
  outcomes. All pairs used direct review, and all 264 judgments were valid on
  their first attempts. The outcome Parquet SHA-256 is
  `1c5b226ff1b70f478fd45e464c57328c30e01b7cf33d395cf1aeff260c3efefb`.
- Across the stable 1,099-checkpoint snapshot, all 158 unresolved model
  outcomes remain covered by 121 true-duplicate and 37 false-positive manual
  records. The adjusted totals are:

  - baseline: 112,500 pairs, 71,420 false positives, 41,080 true duplicates;
  - treatment: 27,145 pairs, 13,986 false positives, 13,159 true duplicates;
  - combined: 139,645 pairs, 85,406 false positives, 54,239 true duplicates.

- The next audit frontiers are p0 `(6, 0)`, p1 `(37, 3,968)`,
  p2 `(70, 0)`, and p3 `(102, 128)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T13:37:50Z — 139,517 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1336-v339` independently
  revalidated two p1 decision-file 37 checkpoints at semantic offsets 3,584
  and 3,712. Their 256 baseline pairs contain 190 false positives and 66 true
  duplicates, with no unresolved outcomes. Two pairs were chunked and 254 were
  direct. All 691 judgments were valid on their first attempts. The outcome
  Parquet SHA-256 values, in frontier order, are
  `2f9a83e2b2a7acfdc5631c571be6d9794c3d0047c13f1df643d129327ad14e53`
  and
  `a370c837adde0269c9d93277a4b6ce0c42e8a7d0b3f437d00d15fd5d94258e12`.
- Across the stable 1,098-checkpoint snapshot, all 158 unresolved model
  outcomes remain covered by 121 true-duplicate and 37 false-positive manual
  records. The adjusted totals are:

  - baseline: 112,372 pairs, 71,321 false positives, 41,051 true duplicates;
  - treatment: 27,145 pairs, 13,986 false positives, 13,159 true duplicates;
  - combined: 139,517 pairs, 85,307 false positives, 54,210 true duplicates.

- The next audit frontiers are p0 `(6, 0)`, p1 `(37, 3,840)`,
  p2 `(70, 0)`, and p3 `(102, 128)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T13:35:10Z — 139,261 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1334-v338` independently
  revalidated p1 decision-file 37 semantic offset 3,456. Its 128 baseline
  pairs contain 70 false positives and 58 true duplicates, with no unresolved
  outcomes. All pairs used direct review, and all 267 judgments were valid on
  their first attempts. The outcome Parquet SHA-256 is
  `33a841b23177dd9e517a5b83eacec1b3c51b647291664c14efdf14af11b1fd02`.
- Across the stable 1,096-checkpoint snapshot, all 158 unresolved model
  outcomes remain covered by 121 true-duplicate and 37 false-positive manual
  records. The adjusted totals are:

  - baseline: 112,116 pairs, 71,131 false positives, 40,985 true duplicates;
  - treatment: 27,145 pairs, 13,986 false positives, 13,159 true duplicates;
  - combined: 139,261 pairs, 85,117 false positives, 54,144 true duplicates.

- The next audit frontiers are p0 `(6, 0)`, p1 `(37, 3,584)`,
  p2 `(70, 0)`, and p3 `(102, 128)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T13:32:40Z — 139,133 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1331-v337` independently
  revalidated five p1 decision-file 37 checkpoints at semantic offsets 2,816
  through 3,328. Their 640 baseline pairs contain 308 false positives and 332
  true duplicates, with no unresolved outcomes. Three pairs were chunked and
  637 were direct. All 1,542 judgments were valid on their first attempts.
  The outcome Parquet SHA-256 values, in frontier order, are:

  - `9552042c3eb2b66f1b167e32f4dadeadddd1b84bb00a7015f2821c1a811bcf7f`;
  - `7e2da42e69ae24ff3ebc421ac553ade904334df5c130702d64d74630b882726f`;
  - `430813da9f6c3c5df4fdd75b51bca194db8f9c46aa6b33d1ca033b6e83ff580a`;
  - `754c439912d83b1a5398e64648023868311e07912995044b3c3e35ff28537f34`;
  - `9927cdd57da0f796e2cb0f9d267f32efac82486ae529156e82403ee18ea5b377`.

- Across the stable 1,095-checkpoint snapshot, all 158 unresolved model
  outcomes remain covered by 121 true-duplicate and 37 false-positive manual
  records. The adjusted totals are:

  - baseline: 111,988 pairs, 71,061 false positives, 40,927 true duplicates;
  - treatment: 27,145 pairs, 13,986 false positives, 13,159 true duplicates;
  - combined: 139,133 pairs, 85,047 false positives, 54,086 true duplicates.

- The next audit frontiers are p0 `(6, 0)`, p1 `(37, 3,456)`,
  p2 `(70, 0)`, and p3 `(102, 128)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T13:30:10Z — 138,493 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1329-v336` independently
  revalidated six p1 decision-file 37 checkpoints at semantic offsets 2,048
  through 2,688. Their 768 baseline pairs contain 332 false positives and 436
  true duplicates, with no unresolved outcomes. All pairs used direct review,
  and all 1,615 judgments were valid on their first attempts. The outcome
  Parquet SHA-256 values, in frontier order, are:

  - `a52df519336f63dc2863d1fbd9e3bab944bc83c7fb364c04dc850a85fe8b646d`;
  - `07758dab2668216470f8887e563ed2dd1c01149bd049ca3f69b6de8dd1f1abdf`;
  - `ff1fef5546d794fd9857bf57aec1690efcccde565bdc2597395dfd90f884c2cf`;
  - `85810eeed21bef494d2696884145fb9b23eba61db7ef4ffd27cd368728419d86`;
  - `b9c393a032f28c6130055aa86f40b4ae601a965bc40fe1fc2031666d607f48b2`;
  - `dd497d41fce0dc152adcc39237aee0b3fda12339526db0ade664836f98465589`.

- Across the stable 1,090-checkpoint snapshot, all 158 unresolved model
  outcomes remain covered by 121 true-duplicate and 37 false-positive manual
  records. The adjusted totals are:

  - baseline: 111,348 pairs, 70,753 false positives, 40,595 true duplicates;
  - treatment: 27,145 pairs, 13,986 false positives, 13,159 true duplicates;
  - combined: 138,493 pairs, 84,739 false positives, 53,754 true duplicates.

- The next audit frontiers are p0 `(6, 0)`, p1 `(37, 2,816)`,
  p2 `(70, 0)`, and p3 `(102, 128)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T13:27:20Z — 137,725 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1317-v332` independently
  revalidated six p1 decision-file 37 checkpoints at semantic offsets 1,280
  through 1,920 and the final five p2 decision-file 69 checkpoints at offsets
  5,376 through 5,888. Their 1,379 pairs contain 695 model false positives,
  680 model true duplicates, and four unresolved outcomes. Two pairs were
  chunked and 1,377 were direct. The audit checked 2,927 judgments across
  2,941 attempts: 2,921 valid and 20 invalid responses affecting eight retried
  judgments. The outcome Parquet SHA-256 values, in p1 then p2 frontier order,
  are:

  - `3c5fe357726fd4ec9cf6f55ff112bcad4fce9b20d1c7c92f33fb60681d821b22`;
  - `cdc76e329840afe0f303d043b5cc6dd217e81f9ad2cf20dc04ca0385dffba7bc`;
  - `c981bdafdb167dec71f3dd70ab42fc51baa4e226ffb127533058b57e8e82e365`;
  - `afccfc97ea5a29e420c1f4dc064f38a68e62e7304b3c3ca88cd16452f6c6d763`;
  - `e79f6393fdf54cba226e50215e98c4d7768c346c6a3df3025701d33febfd1f02`;
  - `c806cb94698c805b9b8e35d9fd39ea76903d89d83a5d5ac023f25edcad6603b3`;
  - `63c0fe0577d589bd4fbf2862f4a871c9efd1b4e40a04880afd3b5ec66731959c`;
  - `05e33dba212e184dd44fc7735606eb3d85025767870ed9e932eabfe67792dec7`;
  - `b7f86c7cf2918ac88101fcd07294ebc48d42891400b5af130ac9464355ab375d`;
  - `e8d9ad9370c87d63b5c59c3792441a5792a58f855dadc6bad2b78d5074efb89d`;
  - `95e0ff39484dce4768882a1cb3c2e0f49a0fd6fd1c06281b35cc0e999a2416e9`.

- Complete-text inspection resolves all four treatment ambiguities as true
  duplicates:

  - `part-00069-of-00128.parquet:8716` contains the same complete 195-line
    Fifth Fleet opinion article. The sole changed line is an already-corrupted
    ending: the member ends with `the latter questions` and the canonical
    with `ion`. Neither fragment adds a coherent proposition. Character,
    line, and word-sequence similarity are 0.999163, 0.994872, and 0.998879.
    Member/canonical text SHA-256 values are
    `4b6e2e1f834c9b781d43710c14859a02b069b72c3f4d0b41e8f4e6849436bff5`
    and
    `b64b5b7fa2420c4352a1b001c459ab3812734fb94582e653d4ed7e5b8a2d87d4`;
    inspection SHA-256 is
    `f2de3129fd5e43d591dc873bb660dfeb68c9d96b1de820a4e8bbbd83a83336d9`.
  - `part-00069-of-00128.parquet:9053` contains the identical satellite
    thermal-control question, choices, reasoning, conclusion, and answer.
    Only `\boxed{D}` versus `\boxed{\text{D}}` differs. The texts contain
    16,952/16,959 characters; character, line, and word-sequence similarity
    are 0.999794, 0.996599, and 0.999816. Member/canonical text SHA-256 values
    are `fc1ea8fec964279b24c71c3cdaa19f835076922d8f24e80c740e1206bf23fdaf`
    and
    `8a372502e17c922d0f03725a7ddcccaece28ac45a28166ed9c1ba07351c0c736`;
    inspection SHA-256 is
    `1e30ed2ab3b1db1e2fcff1f9153fbb26c31de6bb99d5f4fafd8d712893ea38af`.
  - `part-00069-of-00128.parquet:9062` contains the identical Triangle
    Shirtwaist Factory question, choices, reasoning, conclusion, and answer.
    Only `\boxed{B}` versus `\boxed{\text{B}}` differs. The texts contain
    13,340/13,347 characters; character, line, and word-sequence similarity
    are 0.999738, 0.996774, and 0.999779. Member/canonical text SHA-256 values
    are `e3b9c0b283298eb1c2d2e6b920ace4b973dd2e0cbc3c3c5c31b3b2cc84f12815`
    and
    `a44416e1e7c9981c6e1a10eec90da4d9417b74616baabbbeb65cca3c38fb2f05`;
    inspection SHA-256 is
    `51378ca2f018f23ac530cff2eafc049c7c51f2a6772a0febf5d036803a29e409`.
  - `part-00069-of-00128.parquet:9085` contains the identical work-related
    eye-strain question, choices, reasoning, conclusion, and answer. Only
    `\boxed{C}` versus `\boxed{\text{C}}` differs. The texts contain
    9,085/9,078 characters; character, line, and word-sequence similarity are
    0.999615, 0.994444, and 0.999676. Member/canonical text SHA-256 values are
    `bba7bcff63263fc15654074e6c12f2e6d7ea84e2cfa3314241878f9aaa6df94e`
    and
    `d430c733bdd33aa9460058bd8c2f712cbce424d17b73a234d8a9f060b7adb498`;
    inspection SHA-256 is
    `2cc3e18f37258809d9a6f06f76d0328add608f4354dd9f5adf77154f600db1c3`.

- `/rav/datakit-6854-publish-manual-four-1326-v334` wrote the four immutable
  manual records. `/rav/datakit-6854-verify-manual-four-1326-v335`
  separately reread and exactly checked all complete source pairs, semantic
  checkpoints and evidence, inspection artifacts, deterministic Parquet
  bytes, and completion markers. In pair order, semantic-evidence,
  manual-Parquet, and marker SHA-256 values are:

  - `4e783f3e9f9e12b6f1c61218d3336228ac06526ae163c143b5a10fbd4fa344f5`,
    `b9a5830dc5b0ff19435f4e4da2a18ce3799b5b77a4a465a6ebba25aacc872d97`,
    and `0e11cfe9561a40de3db5cef1e95be17609b5a0df1e20d62b04e2fbdfe0810584`;
  - `7b13a194ca82f810a7569b890836a85582e51b1195a0d948bc247b748c377d3a`,
    `3bd0af5f0f4aa5e361c20bfe671ffd81e82801550e153ec4c461ce454c9a45f7`,
    and `b111ff2e0ed5bf5b9084f037a59f1e558f81ac3699ce789739ff55e15bfaa7f2`;
  - `1fbd62141b7e70d6d29c56e51843ff2ed446cd9ebaecf971caf8b00ca765e711`,
    `47b3491c25055e220ba76ebe2052d0a99f86cc8e48890447209289f909ca9028`,
    and `47bb9f67c29ac48c88963afaf5e7b8a14c753f3d6e0f9d9d72e748ad405ced4f`;
  - `dc853e649e272fb4b127971f3747da2c9720e3d51c64d159d8708243d744eea6`,
    `cd1f38d1d3772947f6b3e851f0e7441f01bd02832b6cf70aa94d38a8d6155eb7`,
    and `c6ed32ca69f986008f9c37bcfa373ddc89096804b71227c39b65fbea7d6f1cc2`.

- Across the stable 1,084-checkpoint snapshot, all 158 unresolved model
  outcomes are covered by 121 true-duplicate and 37 false-positive manual
  records. The adjusted totals are:

  - baseline: 110,580 pairs, 70,421 false positives, 40,159 true duplicates;
  - treatment: 27,145 pairs, 13,986 false positives, 13,159 true duplicates;
  - combined: 137,725 pairs, 84,407 false positives, 53,318 true duplicates.

- The next audit frontiers are p0 `(6, 0)`, p1 `(37, 2,048)`,
  p2 `(70, 0)`, and p3 `(102, 128)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T13:16:10Z — 136,346 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1305-v326` independently
  revalidated six p1 decision-file 37 checkpoints at semantic offsets 512
  through 1,152 and six p2 decision-file 69 checkpoints at offsets 4,608
  through 5,248. Their 1,536 pairs contain 1,040 model false positives, 495
  model true duplicates, and one unresolved outcome. Four pairs were chunked
  and 1,532 were direct. All 3,305 judgments were valid on their first
  attempts. The outcome Parquet SHA-256 values, in p1 then p2 frontier order,
  are:

  - `8775bc4ade60935fd74dad9a0f7a39bcdcb71f39a0369d3a33c63bfded336842`;
  - `996219006e79a8c75c55e8b29843edde45c9dc5a3e16aeb910d577645e7dd9d8`;
  - `7b93e5ede5985029eb025f5edc59912affa984a89ac249b1935600ab12688401`;
  - `7e06649b5067e7ce7aec1c1f3c7be7e6293e523d979662fb8b4b9b6644b91b53`;
  - `ea771717b084d8580214989ac047aa5b2c02c65c52445cb4990b81e69c5ac729`;
  - `836d4fe3a7dede79ac5ebe3f5cf7d726fe65ed54eb8a1eac950b2eaa6768addf`;
  - `de77fb645bb45172d4ebc6085e65558bc066b6e7e866b5935d88b52bd3e83fdc`;
  - `f9de4153a0ba020ac4885756da637f9c264e5ba742af5c934dc60a76c563a63a`;
  - `e7b56ba2b6306272a6718f47501159ebc1e6b3ffaa6fa7933c50db05bfe29431`;
  - `d0d8c7414cb523962314ebc81f9bb1604160cefd0247edd36a164cb9dcf46a50`;
  - `60fe1a6ff3383943804963d7ccb0bdada27be533ff5a296a7ff3d991f1acd4c8`;
  - `5ab9e26802ec8a7167514213e8a1dc8a123844589ad1cfeafff5f73a196d4c0f`.

- Complete-text inspection resolves baseline
  `part-00037-of-00128.parquet:1736` as a false positive. The 990-character
  member contains the Eastep surname census record, including different race
  and ethnicity statistics and two Eastep-specific questions and answers.
  The 942-character canonical contains Landgraf statistics and different
  questions. This is structured factual content, not the narrow incoherent
  SEO, college, or career slot-template exception. Character, line, and word
  sequence similarity are 0.783644, 0.551181, and 0.604230.
  Member/canonical text SHA-256 values are
  `762efedf8c3dd856d562e028f31f2d7a03f159c497002df37b31101423539bcd`
  and
  `0e11f48ff12fd12b1932031eb24d0bb69f55e54cf4d8ef7e715bfddbed8e5e9e`;
  inspection SHA-256 is
  `5c3ea8ddfe2870d93fd6a1cfb126ebe7068da46d266dffe895000532720da06d`.
- `/rav/datakit-6854-publish-manual-row1736-1315-v330` wrote the immutable
  manual record.
  `/rav/datakit-6854-verify-manual-row1736-1315-v331` separately reread and
  exactly checked the complete source pair, semantic checkpoint and evidence,
  inspection artifact, deterministic Parquet bytes, and completion marker.
  The semantic-evidence, manual-Parquet, and marker SHA-256 values are
  `ce4b218f64e61806bc21c89cffc68f0e167f73c077ae85deeb6bf194d613d359`,
  `6e0ab5da6d192942a845a7c92823a560141491e0b58346fa0a4ee34f2ba16a03`,
  and `364b55736395c3f9c79dff8435b4f2340302e69f8a62dacb8d320f375fd77b42`.
- Across the stable 1,073-checkpoint snapshot, all 154 unresolved model
  outcomes are covered by 117 true-duplicate and 37 false-positive manual
  records. The adjusted totals are:

  - baseline: 109,812 pairs, 69,990 false positives, 39,822 true duplicates;
  - treatment: 26,534 pairs, 13,722 false positives, 12,812 true duplicates;
  - combined: 136,346 pairs, 83,712 false positives, 52,634 true duplicates.

- The next audit frontiers are p0 `(6, 0)`, p1 `(37, 1,280)`,
  p2 `(69, 5,376)`, and p3 `(102, 128)`. All four batch-priority 2-H100
  workers continue serving requests. Their 12 root, broker, and GPU pods
  remain Ready with zero Kubernetes restarts.

### 2026-07-26T13:00:41Z — 134,810 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1251-v322` independently
  revalidated two p1 decision-file 37 checkpoints at semantic offsets 256 and
  384 and three p2 decision-file 69 checkpoints at offsets 4,224 through
  4,480. Their 640 baseline pairs contain 438 model false positives, 197 model
  true duplicates, and five unresolved outcomes. Twenty pairs were chunked and
  620 were direct. The audit checked 2,786 judgments across 2,798 attempts:
  2,781 valid and 17 invalid responses affecting seven retried judgments. The
  outcome Parquet SHA-256 values, in p1 then p2 frontier order, are:

  - `cea666fb97bb33cc370d3f90892331f7f9d8089a755bb140b7eaf3aa801dead6`;
  - `ded60c0791469e578da4b1e65b86aa97002d7df2bcdfde6f3f59354b58953b58`;
  - `3ac055a6ab490e25cce162aa85d78d926abb334cc1345e05ddc61f74c1d15854`;
  - `dd4cc36bd508ba887590ee803bcebcf2a6a3832f70f6d44f511066ea05ba90cc`;
  - `d4b0c3df9b5e6cca792f3ab709071312fef9096b256dc887a0dba92e4f9b8646`.

- Complete-text inspection resolves one ambiguity as a false positive and four
  as true duplicates:

  - `part-00037-of-00128.parquet:984` is a true duplicate. Both documents
    contain the same WordPress Roots Sage question, PHP snippets, failure,
    `$sage_includes` solution, and documentation-gap explanation. The member
    adds headings, author/date metadata, and summary wording; the canonical
    also contains an extra image-path correction. Member/canonical character
    counts are 1,614/1,322 and character, line, and word-sequence similarity
    are 0.372616, 0.312500, and 0.632603. Text and inspection SHA-256 values
    are `b890cb5acd95ce9bce3aec548bca81af6c6a2714d69b4a1048bf95265c7bb587`,
    `1993307b9a30c2c0b476e6b5aec4391fbb7188acfd5c27816af50f9a3a3b9606`,
    and `a4aef66167e21e054501d23b2b707ca9958a4999004eab70fd24bec9e47452a3`.
  - `part-00069-of-00128.parquet:7225` is a false positive. The member
    explicitly requires an empty input list to return an empty list; the
    canonical request does not state that behavior. This is a distinct
    edge-case instruction even though both implementations use set
    conversion. Member/canonical character counts are 381/346 and character,
    line, and word-sequence similarity are 0.423659, 0.250000, and 0.701754.
    Text and inspection SHA-256 values are
    `cd72c8836e016094ff8c70fdb49e899029d5f27c77d1c7505f327255a3e9545c`,
    `761f8b367e9dc83795ccc0b7558639ed10fd3617a1a0e18c4fc7ef9169379427`,
    and `563fd1a3efc42ac4756ee022517ba8e5f89a5e2a6ac8232a91aa8d3851033956`.
  - `part-00069-of-00128.parquet:7293` is a true duplicate. Both complete
    74-line texts contain the same supply-chain question, options, reasoning,
    conclusion, and answer; only `\boxed{C}` versus `\boxed{\text{C}}`
    differs. Member/canonical character counts are 5,541/5,548 and character,
    line, and word-sequence similarity are 0.999369, 0.986486, and 0.998744.
    Text and inspection SHA-256 values are
    `92664422e367a973caf31589e0f6a72722c32b321b936d50cb5e435e4821b820`,
    `c1cc009879192e16714cbc263b4cfc207a4394888953e5f4ca895f2f453fa8c9`,
    and `50af05129aa093aaead08474f5084fb874efa9f57079ec4c68618e5d9518360c`.
  - `part-00069-of-00128.parquet:7371` is a true duplicate. Both complete
    66-line texts contain the same surveys-versus-interviews question,
    options, reasoning, conclusion, and answer; only `\boxed{B}` versus
    `\boxed{\text{B}}` differs. Member/canonical character counts are
    7,482/7,489 and character, line, and word-sequence similarity are
    0.999532, 0.984848, and 0.999109. Text and inspection SHA-256 values are
    `04886e78b7b54a4832541320868557377f4300a2efc41f1adea56b31170bb668`,
    `8642626cc582c9a1765dee143d744246a82bd2f67a2c0a2f6d7fe4687484b8a4`,
    and `bc2b345207f58b53a7ce9b0984e1393c51b053a5243e9753e220f2fe19633f6d`.
  - `part-00069-of-00128.parquet:7396` is a true duplicate. Both texts ask
    the same exponent-comparison question and give the same logarithmic proof,
    inequalities, conclusion, and boxed answer. Prompt formatting and minor
    connective rephrasing add no new requirement. Member/canonical character
    counts are 1,232/1,191 and character, line, and word-sequence similarity
    are 0.844408, 0.882353, and 0.903382. Text and inspection SHA-256 values
    are `39c9055559e564e9f57ad94c99c78ba55804c3e39d2edf90c5cda6557219b90e`,
    `3567a4614d59c44f5c2061b8c149ed8937d8c8ef36b342ea5d42a45fba8cc841`,
    and `bfcacc443e261ac605b72db9d17e83a3b6a73ef0ceda20b7f0474157b6f80b6a`.

- `/rav/datakit-6854-publish-manual-five-rows-1300-v324` wrote the five
  immutable manual records.
  `/rav/datakit-6854-verify-manual-five-rows-1300-v325` separately reread all
  source pairs, semantic checkpoints and evidence, inspection artifacts,
  deterministic Parquet bytes, and completion markers. In pair order,
  semantic-evidence, manual-Parquet, and marker SHA-256 values are:

  - `b666d77d9e87e992d4c945d0a08a03692218d38899b361777e5e340d01e0aa81`,
    `8fceb7f02eb039d87ddf5b5a04c70824f8fe3bfb9c17a3558e297fdaa52eb18b`,
    and `ed41b2221e78e72f8ea4cf13e66d82636161c0820e9be3461a5d4a03e0c8076e`;
  - `b241a594f6a83eb8f88c61acc865f9a0c83c2a4dabd262f323a63d6d88b5aad1`,
    `744b9ab09b50179ce10b9f89eb2aecf3f50c19e4c3450a3003b5d093f2c79da8`,
    and `f13bec97685c27402957ffb17befef47c82f0baa786ed168a3365b05661e1005`;
  - `6945f2a366d7fe22a8c524302c0566bcc86e291578f2c71c299461c498f69c99`,
    `f98eb70631b634b4ea8ebc79c0bc54ce1b7640dd00eb222ed6daf3f5eccb3803`,
    and `ac924a59719c54d89d80632fbdb94c2a2e4071b114a45f76ffc6392261ef523f`;
  - `d91af3183abb28f9482f9f27131d7a4828a9a4a5d66ca9b3f9c41b37105a6bcd`,
    `c5a9b7bf56eb0b102364529adc098a466a0057fe7c222ed74c30f661009cdaa5`,
    and `3826eb2b5efdd9958f5cc886ce03413cb6948629ab6c3f535511940133fc744b`;
  - `6b12e426b3d7a458e5d17c2f982007cb223ce5e3ae8f4d5098ca14d5a30a653a`,
    `f065c437a7f01b23f4add89e577cd4a839af134900bfce3b187f4c1f9d0502ab`,
    and `c80fcb24e0258046d9abfbab7d70ea26a07a35530e7a190e070a7f181e7db694`.

- Across the stable 1,061-checkpoint snapshot, all 153 unresolved model
  outcomes are covered by 117 true-duplicate and 36 false-positive manual
  records. The adjusted totals are:

  - baseline: 108,954 pairs, 69,358 false positives, 39,596 true duplicates;
  - treatment: 25,856 pairs, 13,313 false positives, 12,543 true duplicates;
  - combined: 134,810 pairs, 82,671 false positives, 52,139 true duplicates.

- The next audit frontiers are p0 `(6, 0)`, p1 `(37, 512)`,
  p2 `(69, 4,608)`, and p3 `(102, 128)`. All four batch-priority 2-H100
  workers continue serving requests. Their 12 root, broker, and GPU pods
  remain Ready with zero Kubernetes restarts.

### 2026-07-26T12:47:58Z — 134,170 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1246-v321` independently
  revalidated six p2 decision-file 69 checkpoints at semantic offsets 3,456
  through 4,096 and p3 decision-file 102 semantic offset 0. Their 896 baseline
  pairs contain 677 false positives and 219 true duplicates, with no
  unresolved outcomes. Twenty-one pairs were chunked and 875 were direct. All
  3,275 judgments were valid on their first attempts. The outcome Parquet
  SHA-256 values, in p2 then p3 frontier order, are:

  - `14d557d37f90195995301a8fbc0461a9dd81525449c555e86adb3ea5a9736508`;
  - `5373a85e09e40b3aee4005ca8d89920ab4456bda7673b87dd89601f220ae6fd7`;
  - `c321f0c4c1868dbaf4394fa466c73b078197b231f2fb6484c40d8558fac3b4b1`;
  - `2780f4754c70b20d3c1088570d258e30580877ba4e41bf0899d632763e564cb5`;
  - `ac6cf2b3574373d64bda5d4021822b07df6c7a49025275fde0998259a68c0f11`;
  - `337477fdebab3974821b77c9ef4ac78f9381a89bcc9a30546e1fd71e94fb40b2`;
  - `0550c2a89a151eb0cd945966b2b958e851358d449adbb62e53d04f5e5805782d`.

- Across the stable 1,056-checkpoint snapshot, all 148 unresolved model
  outcomes remain covered by 113 true-duplicate and 35 false-positive manual
  records. The adjusted totals are:

  - baseline: 108,314 pairs, 68,919 false positives, 39,395 true duplicates;
  - treatment: 25,856 pairs, 13,313 false positives, 12,543 true duplicates;
  - combined: 134,170 pairs, 82,232 false positives, 51,938 true duplicates.

- The next audit frontiers are p0 `(6, 0)`, p1 `(37, 256)`,
  p2 `(69, 4,224)`, and p3 `(102, 128)`. All four batch-priority 2-H100
  workers continue serving requests. Their 12 root, broker, and GPU pods
  remain Ready with zero Kubernetes restarts.

### 2026-07-26T12:44:32Z — 133,274 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1238-v317` independently
  revalidated one p1 decision-file 37 checkpoint at semantic offset 128 and
  six p2 decision-file 69 checkpoints at offsets 2,688 through 3,328. Their
  896 baseline pairs contain 426 model false positives, 468 model true
  duplicates, and two unresolved outcomes. Twenty-five pairs were chunked and
  871 were direct. All 3,181 judgments were valid on their first attempts. The
  outcome Parquet SHA-256 values, in p1 then p2 frontier order, are:

  - `730f40a1e2991214304ec0dac0f0adf5341f49ab5d05d5602df188e9d0bc5074`;
  - `fc886c5e843ab9a8c9023d3049c4cd1e336ac9a578f022bf01af8ad38adec338`;
  - `eb90cf0a5c88982919d9bfaafbc3d15fc8a3a184aa89a60a8d4a3e391aa976b3`;
  - `367176fc02a7cd1f0cd792fdfc3c539fc9efde35a8003592b5364e4231b68a42`;
  - `afbc8f6eb85f116f3f948d6ec6bb3109861ec8dd8a9fdd4959a9c33113aa708b`;
  - `703d714307fa76d340100ba330e5e0c3f5b3d6d4af0f070fa60d3fd150986dd2`;
  - `b72cea38fc4a8a494b8a1606fe10556272e95231623039b2e0c5efc885ec0dee`.

- Complete-text inspection resolves both baseline ambiguities as true
  duplicates under the explicit low-value-template boundary:

  - `part-00069-of-00128.parquet:4675` contains the same incoherent
    college-SEO sentence scaffold in both sources. Differences are
    institution, location, program, job, and superficial wording slots.
    Member/canonical character counts are 849/827 and character, line, and
    word-sequence similarity are 0.471360, 0.333333, and 0.669355.
    Member/canonical text SHA-256 values are
    `40a9aa99eb206feb0fe8885acf8cd46c0f0deab1423b4f236726e64e4a0417e5` and
    `4893ce3e5b496530e4b22e31e43f7505e575b1b5c377c62acb8ae20ccb5c1c4e`;
    inspection SHA-256 is
    `f6c70f3df9965d65afea9bdc8b2d34cc6f8834e8f29c1d4527d37a31daef408e`.
  - `part-00069-of-00128.parquet:5292` contains the same low-value CertBus
    exam-dump sales template. Vendor, exam, and testimonial fields are
    superficial slots; the member's PDF, support, and sales claims are
    represented by corresponding canonical sections. Member/canonical
    character counts are 5,810/6,291 and character, line, and word-sequence
    similarity are 0.481117, 0.491525, and 0.779426.
    Member/canonical text SHA-256 values are
    `3467975defe29aece0ed932c3fe4920b3ef343fb45bf29d44dd0323ca95ed9e8` and
    `147d1563a232411a57ff87485cdfe9098584bc883a4a180a7014f383a28f255b`;
    inspection SHA-256 is
    `1000ad7960db0b8bc721ae1fcfa47fde6bc7a50613960a4f7980bd3d4c2b9e77`.

- `/rav/datakit-6854-publish-manual-row4675-1236-v315` and
  `/rav/datakit-6854-publish-manual-row5292-1243-v319` wrote the immutable
  true-duplicate records. Separate verify-only jobs
  `/rav/datakit-6854-verify-manual-row4675-1237-v316` and
  `/rav/datakit-6854-verify-manual-row5292-1244-v320` reread the complete
  sources, semantic evidence, inspection artifacts, deterministic Parquet
  bytes, and completion markers. In pair order, semantic-evidence,
  manual-Parquet, and marker SHA-256 values are:

  - `fe2a2e073975245e4e21046977c31bfff5b383604d388883fe3c0e858ba0b12c`,
    `899db9c2d4d3ecb5883c5680f003b2c544252ec61a65f98382bad8b15b5395d5`,
    and `067ae2fb47e8892940b8db0a36cfe9e95242c5eef4c4feb9ab3b5b06fc989b46`;
  - `34311baf2efa170acf311bb0481d8fe508a702110ee205321ef2f568182b817a`,
    `e4014b70789cf11d91e2dd8c80b0300606186871a1f41c288e7f0a225f6588a3`,
    and `1daa5407f4ba0d05ceb52bff8ea9f7c6db16597b56ec537db09d8e7f8dbcd097`.

- Across the stable 1,049-checkpoint snapshot, all 148 unresolved model
  outcomes are covered by 113 true-duplicate and 35 false-positive manual
  records. The adjusted totals are:

  - baseline: 107,418 pairs, 68,242 false positives, 39,176 true duplicates;
  - treatment: 25,856 pairs, 13,313 false positives, 12,543 true duplicates;
  - combined: 133,274 pairs, 81,555 false positives, 51,719 true duplicates.

- The next audit frontiers are p0 `(6, 0)`, p1 `(37, 256)`,
  p2 `(69, 3,456)`, and p3 `(102, 0)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T12:25:04Z — 132,378 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1224-v312` independently
  revalidated two p2 decision-file 69 checkpoints at semantic offsets 2,432
  and 2,560. Their 256 baseline pairs contain 113 false positives and 143 true
  duplicates, with no unresolved outcomes. Three pairs were chunked and 253
  were direct. All 684 judgments were valid on their first attempts. The
  outcome Parquet SHA-256 values are:

  - `07a0bb3e76a67cad53b84129733eedfa8cde3584230c02ac7464a38f5af2654d`;
  - `38ac4fc86222c3c0df95114e4bda686cdb17991bc82f6edfe45e3e19ebdc0558`.

- Across the stable 1,042-checkpoint snapshot, all 146 unresolved model
  outcomes remain covered by 111 true-duplicate and 35 false-positive manual
  records. The adjusted totals are:

  - baseline: 106,522 pairs, 67,816 false positives, 38,706 true duplicates;
  - treatment: 25,856 pairs, 13,313 false positives, 12,543 true duplicates;
  - combined: 132,378 pairs, 81,129 false positives, 51,249 true duplicates.

- The next audit frontiers are p0 `(6, 0)`, p1 `(37, 128)`,
  p2 `(69, 2,688)`, and p3 `(102, 0)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T12:21:08Z — 132,122 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1220-v311` independently
  revalidated two p2 decision-file 69 checkpoints at semantic offsets 2,176
  and 2,304. Their 256 baseline pairs contain 129 false positives and 127 true
  duplicates, with no unresolved outcomes. One pair was chunked and 255 were
  direct. All 569 judgments were valid on their first attempts. The outcome
  Parquet SHA-256 values are:

  - `d063adbd4e98c75b082125f7afa36d95429f710947a1436a23103c442fdea612`;
  - `2991969fad5ac4cb58aca8e618f095736675c36d9570a081d3a51612996de57b`.

- Across the stable 1,040-checkpoint snapshot, all 146 unresolved model
  outcomes remain covered by 111 true-duplicate and 35 false-positive manual
  records. The adjusted totals are:

  - baseline: 106,266 pairs, 67,703 false positives, 38,563 true duplicates;
  - treatment: 25,856 pairs, 13,313 false positives, 12,543 true duplicates;
  - combined: 132,122 pairs, 81,016 false positives, 51,106 true duplicates.

- The next audit frontiers are p0 `(6, 0)`, p1 `(37, 128)`,
  p2 `(69, 2,432)`, and p3 `(102, 0)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T12:17:07Z — 131,866 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1216-v310` independently
  revalidated three p2 decision-file 69 checkpoints at semantic offsets 1,792
  through 2,048. Their 384 baseline pairs contain 182 false positives and 202
  true duplicates, with no unresolved outcomes. All pairs used direct review,
  and all 807 judgments were valid on their first attempts. The outcome
  Parquet SHA-256 values are:

  - `4cd65a9dc496f8df70720cdf11386d7bca4220f89202e5e0f034df924979e09f`;
  - `7ac5d1d03becdd64455c996ddd9d484fd037d479d53302d6a8e2da07d8bc1cfe`;
  - `da24fd7bc0ce5e65beadabdf2e6d97438c6f5da32c956d7fcdf7a88b460eceb2`.

- Across the stable 1,038-checkpoint snapshot, all 146 unresolved model
  outcomes remain covered by 111 true-duplicate and 35 false-positive manual
  records. The adjusted totals are:

  - baseline: 106,010 pairs, 67,574 false positives, 38,436 true duplicates;
  - treatment: 25,856 pairs, 13,313 false positives, 12,543 true duplicates;
  - combined: 131,866 pairs, 80,887 false positives, 50,979 true duplicates.

- The next audit frontiers are p0 `(6, 0)`, p1 `(37, 128)`,
  p2 `(69, 2,176)`, and p3 `(102, 0)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T12:13:10Z — 131,482 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1211-v309` independently
  revalidated three p2 decision-file 69 checkpoints at semantic offsets 1,408
  through 1,664. Their 384 baseline pairs contain 213 false positives and 171
  true duplicates, with no unresolved outcomes. Two pairs were chunked and
  382 were direct. All 898 judgments were valid on their first attempts. The
  outcome Parquet SHA-256 values are:

  - `2e850e080980417dda02144f947424256aec63b8dd8e369236d0aca10d806f3d`;
  - `10263374772f32fbd2ecb278ec161ddefddd002ba2132fd4b7517bd5c92e258d`;
  - `a8f42d5760ad21e19dfed29511d625974737f7c3005598edf6915d35da29c04a`.

- Across the stable 1,035-checkpoint snapshot, all 146 unresolved model
  outcomes remain covered by 111 true-duplicate and 35 false-positive manual
  records. The adjusted totals are:

  - baseline: 105,626 pairs, 67,392 false positives, 38,234 true duplicates;
  - treatment: 25,856 pairs, 13,313 false positives, 12,543 true duplicates;
  - combined: 131,482 pairs, 80,705 false positives, 50,777 true duplicates.

- The next audit frontiers are p0 `(6, 0)`, p1 `(37, 128)`,
  p2 `(69, 1,792)`, and p3 `(102, 0)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T12:07:07Z — 131,098 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1206-v308` independently
  revalidated three p2 decision-file 69 checkpoints at semantic offsets 1,024
  through 1,280. Their 384 baseline pairs contain 350 false positives and 34
  true duplicates, with no unresolved outcomes. All pairs used direct review,
  and all 786 judgments were valid on their first attempts. The outcome
  Parquet SHA-256 values are:

  - `d42857e54d8e8e46106dc86b9039277bb574b902ee91791ff8c4397021f48e2a`;
  - `9844b89989399e512af329d6d91e45066dc8d9e52536825fae8d60b24c5b40c7`;
  - `0b24c559e8c59831d147d91ee5fbcfc47fd32448d00dbaab36fd1b4fefafcbe2`.

- Across the stable 1,032-checkpoint snapshot, all 146 unresolved model
  outcomes remain covered by 111 true-duplicate and 35 false-positive manual
  records. The adjusted totals are:

  - baseline: 105,242 pairs, 67,179 false positives, 38,063 true duplicates;
  - treatment: 25,856 pairs, 13,313 false positives, 12,543 true duplicates;
  - combined: 131,098 pairs, 80,492 false positives, 50,606 true duplicates.

- The next audit frontiers are p0 `(6, 0)`, p1 `(37, 128)`,
  p2 `(69, 1,408)`, and p3 `(102, 0)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T12:03:41Z — 130,714 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1202-v307` independently
  revalidated p2 decision-file 69 semantic offset 896. Its 128 baseline pairs
  contain 90 false positives and 38 true duplicates, with no unresolved
  outcomes. One pair was chunked and 127 were direct. All 321 judgments were
  valid on their first attempts. The outcome Parquet SHA-256 is
  `8a352fb8edaa1f6c197885afd80956f76c040a1387fee5b1b325327febbcbabc`.
- Across the stable 1,029-checkpoint snapshot, all 146 unresolved model
  outcomes remain covered by 111 true-duplicate and 35 false-positive manual
  records. The adjusted totals are:

  - baseline: 104,858 pairs, 66,829 false positives, 38,029 true duplicates;
  - treatment: 25,856 pairs, 13,313 false positives, 12,543 true duplicates;
  - combined: 130,714 pairs, 80,142 false positives, 50,572 true duplicates.

- The next audit frontiers are p0 `(6, 0)`, p1 `(37, 128)`,
  p2 `(69, 1,024)`, and p3 `(102, 0)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T11:59:43Z — 130,586 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1158-v306` independently
  revalidated p1 decision-file 37 semantic offset 0 and four p2 decision-file
  69 checkpoints at offsets 384 through 768. Their 640 baseline pairs contain
  462 false positives and 178 true duplicates, with no unresolved outcomes.
  Thirty pairs were chunked and 610 were direct. All 4,542 judgments were
  valid on their first attempts. The outcome Parquet SHA-256 values, in p1
  then p2 frontier order, are:

  - `c06182192065a9bda6c5b933c6cc55fe5394b5841ec0376d58314cb9fe90dd4a`;
  - `8e1440c3f0da33215d74ceaca5f80e4c08c8e7deae807526ea65a6ceee5240ac`;
  - `038210e393651941ee2b99de46004ac5ca980f4472c9b45ccb5a093b74fcc9ab`;
  - `788bf0344313fc34672445063a56eda57ab773a3d56caf43eebd056e79b0d1cb`;
  - `a9cc08450c54b889d6156ea5f980b546e4dd94e5de2ddc23d5c1f9e5b0d6ca66`.

- Across the stable 1,028-checkpoint snapshot, all 146 unresolved model
  outcomes remain covered by 111 true-duplicate and 35 false-positive manual
  records. The adjusted totals are:

  - baseline: 104,730 pairs, 66,739 false positives, 37,991 true duplicates;
  - treatment: 25,856 pairs, 13,313 false positives, 12,543 true duplicates;
  - combined: 130,586 pairs, 80,052 false positives, 50,534 true duplicates.

- The next audit frontiers are p0 `(6, 0)`, p1 `(37, 128)`, p2 `(69, 896)`,
  and p3 `(102, 0)`. All four batch-priority 2-H100 workers continue serving
  requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-26T11:56:38Z — 129,946 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1150-v302` independently
  revalidated p2 decision-file 69 semantic offset 256 and the final three p3
  decision-file 101 checkpoints at offsets 5,632, 5,760, and 5,888. Their 476
  pairs contain 273 model false positives, 201 model true duplicates, and two
  unresolved outcomes. Nine pairs were chunked and 467 were direct. The audit
  checked 1,741 judgments across 1,753 attempts: 1,735 valid and 18 invalid
  responses affecting six retried judgments. The outcome Parquet SHA-256
  values, in p2 then p3 frontier order, are:

  - `cd3c274b3b753af22fb4545e35696fb133f480fa9b70209a274d728e056fcdec`;
  - `7510fc71325b091124d10c84d4cb4cb3bbc9b65364189feeacc12ed7094f7e36`;
  - `6f8114211076f5a47b856d819e65e3ee06765f5fcaa12494938fd2ee469aeb50`;
  - `06f87c0cfa3a39710988c67696deb93d3de65c44c4a446143c7e6bb340b5f143`.

- Complete-text review resolves both treatment ambiguities as true duplicates:

  - `part-00101-of-00128.parquet:9127` contains the same 96-line
    computational-mechanics question, ten methods, full method-by-method
    analysis, conclusion, and answer. Its only changed span formats the final
    answer as `\boxed{\text{B}}` instead of `\boxed{B}`. Member/canonical
    character counts are 6,455/6,448 and character, line, and word-sequence
    similarity are 0.999457, 0.989583, and 0.998974. The member/canonical text
    and inspection SHA-256 values are
    `891843a878e6a64ad081d9111542e3835c14d833c3f44d08f5ceed742891b3c4`,
    `59f635555f10a4b29629b8376dde0d913549b3c2c60c1765fe7c2ac44eb1874b`,
    and `13f4f4a1dec6078ed306521a591b72d17f1410330813a41464994f37a1497931`.
  - `part-00101-of-00128.parquet:9128` contains the same 76-line
    customer-loyalty question, ten strategies, full strategy-by-strategy
    analysis, conclusion, and answer. Its only changed span formats the final
    answer as `\boxed{\text{C}}` instead of `\boxed{C}`. Member/canonical
    character counts are 5,712/5,705 and character, line, and word-sequence
    similarity are 0.999387, 0.986842, and 0.998761. The member/canonical text
    and inspection SHA-256 values are
    `061f4f2164a2ccf82fa361a77727c8708d50a566157374a3d442f2c778bbf307`,
    `124f1d5603ae142c41c19533240c29208e477491598464a8192b396e06ad179e`,
    and `ae11bc3473d028d2a854793a6510d3e5308b733e2080a49dcc20fe0a8dac5750`.

- All nine semantic attempts for each pair were invalid because the model put
  unescaped control characters into JSON strings. The complete persisted
  source diff for each pair contains exactly one changed line.
  `/rav/datakit-6854-publish-manual-rows9127-9128-1155-v304` wrote the
  immutable true-duplicate records, and
  `/rav/datakit-6854-verify-manual-rows9127-9128-1156-v305` separately reread
  both source pairs, semantic checkpoint, judgment hashes, inspection
  artifacts, records, deterministic Parquet bytes, and completion markers.
  In pair order, semantic-evidence, manual-Parquet, and marker SHA-256 values
  are:

  - `3490ba0b4e4d20e1df9f1eb53e6351acee1c13615615929079a5f614ecbedb6f`,
    `886736e5ccada4a75599abf4cfd7b146761d5f2988b6431c52ae021f9d4a3bb0`,
    and `df5c4addfe962f6aab5b24a95cdfeab170c029cedfbee212e4d499047d3211b7`;
  - `3eeb70a1ce03cd83dd1959f0245e69aba3116e824bd32429703a31d4a550b60a`,
    `b022f321c5788e63a14827b7de6ee874aa461a320d94ea44f0f9bff9a506b587`,
    and `fc9a0e165f66ef01b31d181c6c607a1468b0adaad9c09ca6a4e4dcf4133fb8b1`.

- Across the stable 1,023-checkpoint snapshot, all 146 unresolved model
  outcomes are covered by 111 true-duplicate and 35 false-positive manual
  records. The adjusted totals are:

  - baseline: 104,090 pairs, 66,277 false positives, 37,813 true duplicates;
  - treatment: 25,856 pairs, 13,313 false positives, 12,543 true duplicates;
  - combined: 129,946 pairs, 79,590 false positives, 50,356 true duplicates.

- The next audit frontiers are p0 `(6, 0)`, p1 `(37, 0)`, p2 `(69, 384)`,
  and p3 `(102, 0)`. All four batch-priority 2-H100 workers continue serving
  requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-26T11:47:11Z — 129,470 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1147-v301` independently
  revalidated p3 decision-file 101 semantic offset 5,504. Its 128 treatment
  pairs contain 41 false positives and 87 true duplicates, with no unresolved
  outcomes. All 267 judgments were valid on their first attempt and all pairs
  used direct review. The outcome Parquet SHA-256 is
  `7d266a83320c72c2754b2206d9f098937e400fd7fb69b830835fed0570866c19`.
- Across the stable 1,019-checkpoint snapshot, all 144 unresolved model
  outcomes remain covered by 109 true-duplicate and 35 false-positive manual
  records. The adjusted totals are:

  - baseline: 103,962 pairs, 66,192 false positives, 37,770 true duplicates;
  - treatment: 25,508 pairs, 13,125 false positives, 12,383 true duplicates;
  - combined: 129,470 pairs, 79,317 false positives, 50,153 true duplicates.

- The next audit frontiers are p0 `(6, 0)`, p1 `(37, 0)`, p2 `(69, 256)`,
  and p3 `(101, 5,632)`. All four batch-priority 2-H100 workers continue
  serving requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-26T11:45:37Z — 129,342 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1144-v300` independently
  revalidated the final three p0 decision-file 5 checkpoints at semantic
  offsets 5,632, 5,760, and 5,888, one p2 decision-file 69 checkpoint at
  offset 128, and five p3 decision-file 101 checkpoints at offsets 4,864
  through 5,376. Their 1,079 pairs contain 618 false positives and 461 true
  duplicates, with no unresolved outcomes. Fourteen pairs were chunked and
  1,065 were direct. The audit checked 3,125 judgments across 3,127 attempts:
  3,124 valid and three invalid responses, all on one retried judgment. The
  outcome Parquet SHA-256 values, in p0, p2, then p3 frontier order, are:

  - `287fee453746d3e4f49d3e0f9b2517679f4fdf7c784162a930494be6a36ca4a3`;
  - `08ca8f6ed1729d34700f20f37151507857884a17dce36f470d24c4f8c152eb27`;
  - `f363bc269941430088920aa9060c94cf34614cf966f75acf29a80f0b29896067`;
  - `e9d6cf8d99a6027faba29b7dd9190ab33e20c0874673d53d3d98b1bee36630f2`;
  - `9647ad8091540efbb4e06163c673adecd0d34f08fd474c2f0068f323ae008c7f`;
  - `14f586b3aae8191be0fbf11f831421f815b2b2176e165b4498ac1d2ae0bddbe9`;
  - `71a2e9f64927768e332ac3d844997187b8d86773f195ab014c64ec3ce22d674b`;
  - `f4c447b2828cb5629fc6d336660c52a6070fbe594118ad51983a4dff532c7cba`;
  - `b91a8b9325aaf667b735f98ccd8c0f7d70926371a846fc358c0f3aef09b05dbe`.

- Across the stable 1,018-checkpoint snapshot, all 144 unresolved model
  outcomes remain covered by 109 true-duplicate and 35 false-positive manual
  records. The adjusted totals are:

  - baseline: 103,962 pairs, 66,192 false positives, 37,770 true duplicates;
  - treatment: 25,380 pairs, 13,084 false positives, 12,296 true duplicates;
  - combined: 129,342 pairs, 79,276 false positives, 50,066 true duplicates.

- The next audit frontiers are p0 `(6, 0)`, p1 `(37, 0)`, p2 `(69, 256)`,
  and p3 `(101, 5,504)`. All four batch-priority 2-H100 workers continue
  serving requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-26T11:43:13Z — 128,263 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1136-v296` independently
  revalidated six p0 decision-file 5 checkpoints at semantic offsets 4,864
  through 5,504, one p2 decision-file 69 checkpoint at offset 0, and three p3
  decision-file 101 checkpoints at offsets 4,480 through 4,736. Their 1,280
  pairs contain 767 model false positives, 512 model true duplicates, and one
  unresolved outcome. Twenty-six pairs were chunked and 1,254 were direct.
  The audit checked 6,920 judgments across 6,922 attempts: 6,919 valid and
  three invalid responses, all on one retried judgment. The outcome Parquet
  SHA-256 values, in p0, p2, then p3 frontier order, are:

  - `9551265b61ffd2b502a13e422c143b97817468c6536a39707e97141ed850588c`;
  - `fa08a98ed3f8b40ef3ebb585205c297d14530cdab9c0a935ebae8a3e68056ed4`;
  - `e1e1326a534319dae67b1ff07e7e4071b3bd072ae5072348bcad554755e73494`;
  - `3ba3b94f7e752adf28af26f60b889ca0bd3954f81225834bc476c234bed0e918`;
  - `96d026e7b2ee065781878ae284c20f1241d0fd18e3f7c2b09d99c51a456d3d7e`;
  - `04d12aadabf5743f44e52572cc72ab309c2408f514799e609c7c9ebdd3efbd8a`;
  - `2c3c4600589d03649190cd03eddb35f45d5bb44ed898dee95f43da94d7cad9c1`;
  - `1d22b2a61c6ed84ca4640e8fd050b4f2a6a85322a9a37f714f0e137c0ea8a4e7`;
  - `bc3d400ee3252a0ffc4b2edd130ec5ab2c2569507b3775cbded6e3339fbe3d97`;
  - `59b845ad8e12ceeab0305708745f41638f401eaddf5ef4fe4f060eedff961370`.

- Complete comparison resolves `part-00101-of-00128.parquet:7676` as a true
  duplicate. Both 226-line texts contain the same project-management-system
  question, ten options, full option-by-option analysis, conclusion, and
  answer. The only changed span formats the final answer as
  `\boxed{\text{J}}` instead of `\boxed{J}`. Member/canonical character
  counts are 11,366/11,359 and character, line, and word-sequence similarity
  are 0.999692, 0.995575, and 0.999416. The member/canonical text and
  inspection SHA-256 values are
  `db7e2e978cfcff0fa7bcdce8c0fc17a066a09024c792cd0632b188c85ed374e9`,
  `569649d617e6fd8698a868e60971e8222e33a174afd508107dab30e7084be983`,
  and `198bf2ade4282c6f8f1e33c26130f1fd110cb3edaf1ca3385e4630018da8f74b`.
  The semantic loss and tiebreak passes independently returned
  high-confidence true-duplicate verdicts; the pair remained unresolved only
  because all three duplication-pass attempts contained invalid JSON control
  characters.
- `/rav/datakit-6854-publish-manual-row7676-1142-v298` wrote the immutable
  true-duplicate record, and
  `/rav/datakit-6854-verify-manual-row7676-1143-v299` separately reread the
  source pair, semantic checkpoint, judgment hash, inspection artifact,
  record, deterministic Parquet bytes, and completion marker. The
  semantic-evidence, manual-Parquet, and marker SHA-256 values are
  `b48f7ee29e24b5fe643175599a67481631e30976538e5c638a754ad6d1990129`,
  `9168b9ed34e7070e43eea0b7212683a237bf6c5d61849bb44ad865f884d24b68`,
  and `0603d6d9b32a43ded7e743efa59b22a44cf9e3d0fc8f956f9de0bce90e91518b`.
- Across the stable 1,009-checkpoint snapshot, all 144 unresolved model
  outcomes are covered by 109 true-duplicate and 35 false-positive manual
  records. The adjusted totals are:

  - baseline: 103,834 pairs, 66,112 false positives, 37,722 true duplicates;
  - treatment: 24,429 pairs, 12,546 false positives, 11,883 true duplicates;
  - combined: 128,263 pairs, 78,658 false positives, 49,605 true duplicates.

- The next audit frontiers are p0 `(5, 5,632)`, p1 `(37, 0)`,
  p2 `(69, 128)`, and p3 `(101, 4,864)`. All four batch-priority 2-H100
  workers continue serving requests. Their 12 root, broker, and GPU pods
  remain Ready with zero Kubernetes restarts.

### 2026-07-26T11:34:13Z — 126,983 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1127-v292` independently
  revalidated four p0 decision-file 5 checkpoints at semantic offsets 4,352
  through 4,736 and five p3 decision-file 101 checkpoints at offsets 3,840
  through 4,352. Their 1,152 pairs contain 806 model false positives, 344
  model true duplicates, and two unresolved outcomes. Three pairs were
  chunked and 1,149 were direct. The audit checked 2,597 judgments across
  2,604 attempts: 2,594 valid and 10 invalid responses, with four judgments
  requiring retries. The outcome Parquet SHA-256 values, in p0 then p3
  frontier order, are:

  - `b5816e1942a9d67a39ac33ee864d5cd6e6cd4952b091463c3bdb8ce0880739c1`;
  - `d68a30d9b55010c51db004237f1c412769a5fcdb61aeffba2e427435d4675448`;
  - `51cc8863c635638cf9090b2ff4bf5d1fef8688794dcddb127add0e84281977fd`;
  - `85bf4a88b85149cfe445346bf2417b44a54e26b6fbd9c4b1d776465e84241b84`;
  - `756c4b4fa6f262266fa8daf40f305d8b40f8bde1a95907e31830837c46440a08`;
  - `20e8fd7e4088f4fe378030c14b7266b0457021ee4e3dfb66aa0608915b67cc2d`;
  - `4ac1f880099821172933e07b6955d3d2aa22b1c235518b9e050cb49d8f28966a`;
  - `72a11d5dd40fbbc51d0dd9773ba8cc4307aef201d2eba38d438dd699ebc811cc`;
  - `fbf06a17aa103ba9b90cee3ba94360f8e4997b05f8d80167165d0d503edcdef8`.

- Complete-text review resolves both ambiguities as true duplicates:

  - `part-00005-of-00128.parquet:7732` contains the same 66-line
    drug-efficacy question, option-by-option reasoning, conclusion, and answer
    in both sources. The only changed span formats the final answer as
    `\boxed{\text{D}}` instead of `\boxed{D}`. Member/canonical character
    counts are 6,071/6,064 and character, line, and word-sequence similarity
    are 0.999423, 0.984848, and 0.998970. The member/canonical text and
    inspection SHA-256 values are
    `df5208521157372b0ff739d2c9a67ede1d110ff635de658b6184a75638fbbadf`,
    `312af2d39f1135a75a16986c4a472fa8bcf3fdf3a4e8bd6769cc78935081e0dd`,
    and `0b69e3a04becf99c362ad39b95eb5a9eed491243549c8cd48a7307b24eb8beeb`.
  - `part-00101-of-00128.parquet:7463` contains the same library-table
    word problem, three-stage derivation, arithmetic, and answer. Differences
    are limited to headings, whitespace, and final-answer formatting.
    Member/canonical character counts are 1,118/1,114 and character, line,
    and word-sequence similarity are 0.957885, 0.790698, and 0.962085. The
    member/canonical text and inspection SHA-256 values are
    `59c1865b4031f193c2c6c2ac743ac76d45f21475c74b01618972849b06a75643`,
    `a1af579f2d824a4d3e900a22eb77b8831a08be0e4b3274eae281f2a3b2f7db68`,
    and `c864b5b92e0cb5b6715d4c6658f091b9412510f404bb6abf6739d4e245cb13ad`.

- `/rav/datakit-6854-publish-manual-rows7732-7463-1133-v294` wrote the
  immutable true-duplicate records, and
  `/rav/datakit-6854-verify-manual-rows7732-7463-1134-v295` separately reread
  both source pairs, semantic checkpoints, judgment hashes, inspection
  artifacts, records, deterministic Parquet bytes, and completion markers.
  In pair order, semantic-evidence, manual-Parquet, and marker SHA-256 values
  are:

  - `352fb29c60e64c672c49c3c19b78f78f95c50edb384154b32d0b20c7eaa389de`,
    `d49fd81261fb53dbf9981c21ce3499e5a1dbcd940e9da67df577095e53a53c59`,
    and `221f74a87ac171788ddc0f1f8cf2dfa215bddb9d576519a27f6c12f1f7df3de8`;
  - `445ab51708e8017819271fa2c60e6403c924fa75fa1fc8edc4bea5e0e04e2af2`,
    `49218c7cd5834e89610eb35b84e4e883e1cb23029a6e901679876233fc905506`,
    and `e58e24a6d687d2c572e893f633c52db1d1dd0bfaa29477dd231a5f19ef790a98`.

- Across the stable 999-checkpoint snapshot, all 143 unresolved model outcomes
  are covered by 108 true-duplicate and 35 false-positive manual records. The
  adjusted totals are:

  - baseline: 103,430 pairs, 65,784 false positives, 37,646 true duplicates;
  - treatment: 23,553 pairs, 12,107 false positives, 11,446 true duplicates;
  - combined: 126,983 pairs, 77,891 false positives, 49,092 true duplicates.

- The next audit frontiers are p0 `(5, 4,864)`, p1 `(37, 0)`, p2 `(69, 0)`,
  and p3 `(101, 4,480)`. All four batch-priority 2-H100 workers continue
  serving requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-26T11:23:05Z — 125,831 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1118-v288` independently
  revalidated six p0 decision-file 5 checkpoints at semantic offsets 3,584
  through 4,224 and four p3 decision-file 101 checkpoints at offsets 3,328
  through 3,712. Their 1,280 baseline pairs contain 916 model false positives,
  362 model true duplicates, and two unresolved outcomes. Four pairs were
  chunked and 1,276 were direct. The audit checked 2,854 judgments across
  2,858 request attempts. Six invalid responses affected two retried
  judgments; every accepted judgment validates. The outcome Parquet SHA-256
  values, in p0 then p3 frontier order, are:

  - `8506b4c09d9f89085ece129437f838b337b6560b55dd67cdcb16f121949bfe08`;
  - `fa8d9c86fd58ad71850cd38c0e2f44f76ae4996211d60f9a1e11461a351219b1`;
  - `e39c9aa1af331c69e14abb4cdabebe911dacec7d6219213f4894405da992507c`;
  - `731ee44876a096cec2c15a6499baf353154697c33e40ed55df08debda573dbe8`;
  - `7bab2ba20d1aadea24ccbe35b81a16fe022efad1288f2cff5d268d8531470b35`;
  - `94f939f91819494c24be4eca5b5dabc360f7eb3f00c49c3f6b73df112ee68188`;
  - `3af829262e2d34bfc5f1c61657af221b825f74ff3a9c99695267230e7a5dc82b`;
  - `d9e7ac4a4a053a4d77d051026c31d3b68e5249b7e994b4cae5041bc4126a974e`;
  - `bbd50e94a14efe92d1d63b111a3d6ac11e9a487985217510309e3a13ec402163`;
  - `4de132e5976d4c7ccf10568fabbfd159a7318c6b753262cdc431297227615037`.

- Complete-text review resolves the two ambiguities:

  - `part-00005-of-00128.parquet:6383` is a false positive. The
    Billingshurst and Swinmore location-specific brothel-guide variants have
    different Q&A training examples. The member uniquely asks about two
    selection factors, the target audience, and online research, and contains
    a complete definition section missing from the canonical. Across all
    6,582/7,013 member/canonical characters, character, line, and word-sequence
    similarity are 0.378227, 0.438356, and 0.609976. The member/canonical text
    and inspection-artifact SHA-256 values are
    `a8280e08dd1cc8043f794cf01376c46ea8527131e8ab819200043ccebafaad25`,
    `c1218b02a35c8f7601b8998be8049b6f2000734a44ba8bd64140be6ac4749710`,
    and `dc91577e36d8d23688464702ad8acb979fe9c84f9ea80109f78b6de2a6255a4e`.
  - `part-00005-of-00128.parquet:7445` is a true duplicate. Every one of
    its 240 lines contains the same software-bug question, ten options,
    reasoning, conclusion, and answer. Replacing canonical
    `\boxed{\text{J}}` with member `\boxed{J}` makes the complete texts byte
    identical. Their character, line, and word-sequence similarity are
    0.999712, 0.995833, and 0.999490. The member/canonical text and
    inspection-artifact SHA-256 values are
    `0454156d7f426811580e2418bb1c8dd3b808a2dce29aab11d9ab52eb1b4ac675`,
    `0afdb59578c3ccc65df3d61b41c9bf6a9884b5de5ceb55b4701b40984921f136`,
    and `f697b5fc3ae9d9eb8bba7d97310d47183791fb0405cb378583a63462faf4f2ce`.

- `/rav/datakit-6854-publish-rows6383-7445-1124-v290` wrote the immutable
  false-positive and true-duplicate records, and
  `/rav/datakit-6854-verify-rows6383-7445-1124-v291` separately reread both
  source pairs, semantic checkpoints, inspection artifacts, records,
  deterministic Parquet bytes, and completion markers. In pair order, the
  semantic-evidence, manual-record, and manual-Parquet SHA-256 values are:

  - `57b66dc7afc322e122fe4ecc4b85cabb409f6e25614d33fe690a831cb809dd1f`,
    `3790f94c9c72c31471e6508dd66cbd30177ec0a02add1536289cea3afc98e2c2`,
    and `0f80bfb35eacdbff1e036e956a4dd5d98fbb54c79847238a0d68efcd00df10c0`;
  - `e261afe76f9542e818c85983fb70ac4dde485942fd2927fbdb4b504ae90a7e30`,
    `d895875291548fbc182e5d3777ccf88410470ceec8b9d8071be1e5f498ed2bd1`,
    and `58315c4e8b499daca210a541694f042291e12bbce19a94c8fca74b4acfa10062`.

- Across the stable 990-checkpoint snapshot, all 141 unresolved model outcomes
  are covered by 106 true-duplicate and 35 false-positive manual records. The
  adjusted totals are:

  - baseline: 102,418 pairs, 65,053 false positives, 37,365 true duplicates;
  - treatment: 23,413 pairs, 12,032 false positives, 11,381 true duplicates;
  - combined: 125,831 pairs, 77,085 false positives, 48,746 true duplicates.

- The next audit frontiers are p0 `(5, 4,352)`, p1 `(37, 0)`, p2 `(69, 0)`,
  and p3 `(101, 3,840)`. All four batch-priority 2-H100 workers continue
  serving requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-26T11:15:10Z — 124,551 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1110-v284` independently
  revalidated six p0 decision-file 5 checkpoints at semantic offsets 2,816
  through 3,456 and six p3 decision-file 101 checkpoints at offsets 2,560
  through 3,200. Their 1,536 baseline pairs contain 791 model false positives,
  743 model true duplicates, and two unresolved outcomes. Five pairs were
  chunked and 1,531 were direct. All 3,627 judgments were valid on their first
  attempts. The outcome Parquet SHA-256 values, in p0 then p3 frontier order,
  are:

  - `3d288519beb0b9b638aa873b8eca91eafe670a0e0fc93493f5cd41e6993a1cd3`;
  - `a4c5acb3c0339cc988dc33e8508e798179b707fb11916c56787a2d8937c718ae`;
  - `f97e20eb5df28e03cf8679f40c32368ea836e4c5565f8677283a2bab897622e9`;
  - `59441b64ed5f170da5712990553fab464180ba2371bc08915c079572387f70aa`;
  - `a9447b7ccecc96f376cf8415c6c6ee519c8601a4ba4535f7e5dcb7d5b9d3292a`;
  - `cb7c772dd7b92b59f4924fea2dea85d74b4085accf54c4e088995d54e80b9563`;
  - `4d794431cdf9084b2d4f09ce5d9cd69b9fb71d9768b203c2e4596acf4df90144`;
  - `9c2c8063ca6f1d2d9b01b32dbc1a75eaa029eb16cd3b3134eb0835113d6c9ef8`;
  - `6ea2662613cc9d687f1b64b71aefec1ba159269f52ac86a09e47632a3b6d5b2d`;
  - `881d791e885cdad13b3cbc7c2c83896bfc66948bfde1e63e16cd228fb9cf00f7`;
  - `8d1111751a8e671c5b5da8b3b94217aaf28186f839925267f0eeb9ac9c17c271`;
  - `6d139334a7015b9acad78e79fced98e1d534c6b6cbcb642e36e70a4ee82ea022`.

- Complete-text review resolves both ambiguities as false positives:

  - `part-00101-of-00128.parquet:4159` contains two variants of a George
    Soros chain letter. Across all 12,986 member and 14,257 canonical
    characters, the member has substantive body passages absent or truncated
    in the canonical: the Soros birth and Esperanto paragraph, the
    malignant-narcissist sentence, the full shadow-government passage, and a
    distinct Patrick Henry closing. Character, line, and word-sequence
    similarity are 0.833755, 0.522523, and 0.836657. The member/canonical text
    and inspection-artifact SHA-256 values are
    `27c0d3c911373fe54cc4f716d75ae0bf36566b899c93555fe9ac58424e4d5591`,
    `fa2df97057756a52ab960cb52e04cbd4c6fe264587665a56793b56d1517ff62d`,
    and `e60d8b62127107e7cbb52aacda907f1a7bbcf5e918c9b8b58289d01e900846c7`.
  - `part-00101-of-00128.parquet:4380` is a 1,410-character college-template
    member against an 827-character canonical. The member uniquely states
    biomedical-engineering degree requirements, mechanical and surgical
    electives, and business-internship benefits, beyond its different school
    names. Character, line, and word-sequence similarity are 0.389808,
    0.250000, and 0.451807. The member/canonical text and inspection-artifact
    SHA-256 values are
    `8f03fb2ebd36c0cc88dd47d86ed741c2abd09188fc6ea5fbdc21d724313b1aa5`,
    `4893ce3e5b496530e4b22e31e43f7505e575b1b5c377c62acb8ae20ccb5c1c4e`,
    and `7698f7738c79fb0fd88cd5efb755ec62dd81b9956ba068ffc379713341351920`.

- `/rav/datakit-6854-publish-rows4159-4380-1117-v286` wrote both immutable
  false-positive records, and
  `/rav/datakit-6854-verify-rows4159-4380-1117-v287` separately reread both
  source pairs, semantic checkpoints, inspection artifacts, records,
  deterministic Parquet bytes, and completion markers. In pair order, the
  semantic-evidence, manual-record, and manual-Parquet SHA-256 values are:

  - `e2f6bcfc5d7026198e67aa5f7f5eb0620ca32f60edf23f84f4fb7c933b830aea`,
    `e96a69023b9d2bba55e9a369b8cb7aba6aa18a07469e4a1ce07786ae6b2db8e6`,
    and `f509a24ab27768df60ec33804e4c8cdc4263dd50ce53e04ebdf7edfeadc8aae3`;
  - `8f53a21331863f70630f054264e9ca7ece97821d0d7dd0e66242929a0f08712a`,
    `63534d8ee366f55b5d17393a5bbb1a0ded14c56547ae270823ddf1135518ae3b`,
    and `cd75b5af195450573df5018d5de910bec800400366673c65ae80ca0e5213c1cc`.

- Across the stable 980-checkpoint snapshot, all 139 unresolved model outcomes
  are covered by 105 true-duplicate and 34 false-positive manual records. The
  adjusted totals are:

  - baseline: 101,138 pairs, 64,136 false positives, 37,002 true duplicates;
  - treatment: 23,413 pairs, 12,032 false positives, 11,381 true duplicates;
  - combined: 124,551 pairs, 76,168 false positives, 48,383 true duplicates.

- The next audit frontiers are p0 `(5, 3,584)`, p1 `(37, 0)`, p2 `(69, 0)`,
  and p3 `(101, 3,328)`. All four batch-priority 2-H100 workers continue
  serving requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-26T11:07:10Z — 123,015 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1101-v280` independently
  revalidated six p0 decision-file 5 checkpoints at semantic offsets 2,048
  through 2,688 and six p3 decision-file 101 checkpoints at offsets 1,792
  through 2,432. Their 1,536 baseline pairs contain 678 model false positives,
  857 model true duplicates, and one unresolved outcome. One pair was chunked
  and 1,535 were direct. All 3,284 judgments were valid on their first
  attempts. The outcome Parquet SHA-256 values, in p0 then p3 frontier order,
  are:

  - `041fbb17a0686e2b7553b584366ada3c6b318c324e1f58698b0f62e0bfa3dfec`;
  - `91da3b5242991b77734d3130141cbe5674eee73324bbfce92ba878ba4b1f582f`;
  - `2a1c90496d35159c1abe129e28e783ace878886594c715f8f21b934d24e4f89d`;
  - `b5c458c2f0ed11aae78a43d9621b262adb4128c4cd31d89f180aa25ed909719e`;
  - `b1c4ce6c61e176edb6f5e388aad7b4e651d0d8bd3fbcb4142970d76c8095768a`;
  - `a316aaeb3dbd28820190c75d80437b0b06d4477575e130e3878d9a47d282f8a1`;
  - `df5dcf43295e26d8d13466f65e266cac2b9cdb9970c60841707994d5dc0e2f6f`;
  - `30d23edf446549d906af3fb9e4cc8d624454609a67625db91e41c5017e943e17`;
  - `780eab7442195c3641e245d5ad251820d4d321f5ef1469db51960492ca0cd013`;
  - `a7c2dea89eeefa5ab8c4a40e72c37905b3c8563a1158de6c5bd20da5c3431f70`;
  - `a100198dc3eca15a9b0345cda065b31bf18b88233afa30acba8677180675000d`;
  - `819d935ff75df4f6391e615513e423b15cd3e5d49ee1d2470eeb6a0004ec6b5b`.

- Complete-text review resolves `part-00005-of-00128.parquet:4583` as a false
  positive. The 2,092-character, 19-line member and 827-character, three-line
  canonical share only two admissions/job-board boilerplate fragments. The
  member additionally contains a Carrollton nursing-application title, West
  Georgia admissions instructions, a college-grades paragraph, nursing-degree
  guidance, and clinical-internship and nursing-career content. Character,
  line, and word-sequence similarity are 0.199383, 0.090909, and 0.249412;
  neither text contains the other. The member/canonical text SHA-256 values are
  `951e9953b9801de5b85b29bf447b1b7a357e66a4850b041e9cc88312529b1137` /
  `4893ce3e5b496530e4b22e31e43f7505e575b1b5c377c62acb8ae20ccb5c1c4e`;
  inspection-artifact SHA-256 is
  `b2bb4517a03d918a2a0f54746e7c0ca450e6789875154f0acffce00fd0860daa`.
- The loss pass classified the pair as a high-confidence false positive. The
  duplication pass and low-confidence tiebreak both identified the member-only
  institutions and nursing topic but treated them as low-value template slots.
  `/rav/datakit-6854-publish-row4583-1107-v282` wrote the immutable
  false-positive record, and `/rav/datakit-6854-verify-row4583-1107-v283`
  separately reread the source pair, semantic checkpoint, inspection artifact,
  record, deterministic Parquet bytes, and completion marker.
  Semantic-evidence, manual-record, and manual-Parquet SHA-256 values are
  `3f06e57aeed2d51f0265c6f4068736734d36f6d370b9ca940a149ff8c932e6ba`,
  `f89edc79e293b0967ef15b3e16e3c51cb5e495eaa06cab80c0c025248022b469`,
  and `c83be81a1ca8143930f1fe8897aec646c58ade5a71f6499bd1021e3eeba3e83d`.
- Across the stable 968-checkpoint snapshot, all 137 unresolved model outcomes
  are covered by 105 true-duplicate and 32 false-positive manual records. The
  adjusted totals are:

  - baseline: 99,602 pairs, 63,343 false positives, 36,259 true duplicates;
  - treatment: 23,413 pairs, 12,032 false positives, 11,381 true duplicates;
  - combined: 123,015 pairs, 75,375 false positives, 47,640 true duplicates.

- The next audit frontiers are p0 `(5, 2,816)`, p1 `(37, 0)`, p2 `(69, 0)`,
  and p3 `(101, 2,560)`. All four batch-priority 2-H100 workers continue
  serving requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-26T10:58:50Z — 121,479 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1047-v272` independently
  revalidated p0 decision-file 5 semantic offsets 1,792 and 1,920 plus p3
  decision-file 101 semantic offsets 1,536 and 1,664. Their 512 baseline pairs
  contain 233 model false positives, 278 model true duplicates, and one
  unresolved outcome. All 1,073 judgments were valid on their first attempts,
  and every pair used direct review. The outcome Parquet SHA-256 values are
  `d771e94cb9d93e3ecc0018e47aa31819b6d5044d9ff0a8f9b8ba52ba52cf06e4`,
  `0121d99c81db3364c477ce7b79c6787281d386be60d70f67259c36e4cd6a26ac`,
  `11934cf32404b28503b5ca566becd5ffe778fd54b6f944398bb9cdaaf21b64fa`,
  and
  `2b96161619cbf3128d17987cb335f6618df1f5b5a895736959042e2ef09311fc`.
- Complete-text review resolves `part-00101-of-00128.parquet:2298` as a false
  positive. The member is a Jelenich surname page reporting 51 family trees,
  17 people in 1940, 41% female, Frank as the most common male name, 25%
  renters, and a three-person household. The canonical is a Northrop page
  reporting 85,172 trees, 1,996 people, salesman as the common occupation, 44
  work hours, $1,245 income, and eight vacation weeks. Neither text contains
  the other. Character, line, and word-sequence similarity are 0.317678,
  0.524590, and 0.664653. The member/canonical text SHA-256 values are
  `ed0daf573572003b4fe0a3d6c1315e6c789d4337edce991117b36bfc30a5d3c3` /
  `d8862607bb758393fc473ddd6695e035597efd95b96b4dec5d4976926bbbb35e`;
  inspection-artifact SHA-256 is
  `f1d24089d109afe6db8bfe93508bd779484c272d1590ee024e2aaca907cceeae`.
- The loss pass classified the pair as a high-confidence false positive. The
  duplication pass treated the distinct surname statistics as template slots
  and classified it as a high-confidence duplicate. The low-confidence
  tiebreak again identified the Jelenich facts as unique, so the model left the
  pair unresolved. `/rav/datakit-6854-publish-row2298-1059-v278` wrote the
  immutable false-positive record, and
  `/rav/datakit-6854-verify-row2298-1059-v279` separately reread the source
  pair, semantic checkpoint, inspection artifact, record, deterministic
  Parquet bytes, and completion marker. Semantic-evidence, manual-record, and
  manual-Parquet SHA-256 values are
  `2ffbfec33949372425d7b43efbea5e2ed50b097e9e4aec1ff84c124961c53f17`,
  `7119620f315877cc4f7af0e7670723658147ae0fe30f4ad88f2e308a4990c604`,
  and `3e1c5c13a980f09f1c83552acc549e5b013e76e56258fa9cf83120881c63dfd2`.
- Across the stable 956-checkpoint snapshot, all 136 unresolved model outcomes
  are covered by 105 true-duplicate and 31 false-positive manual records. The
  adjusted totals are:

  - baseline: 98,066 pairs, 62,664 false positives, 35,402 true duplicates;
  - treatment: 23,413 pairs, 12,032 false positives, 11,381 true duplicates;
  - combined: 121,479 pairs, 74,696 false positives, 46,783 true duplicates.

- The next audit frontiers are p0 `(5, 2,048)`, p1 `(37, 0)`, p2 `(69, 0)`,
  and p3 `(101, 1,792)`. All four batch-priority 2-H100 workers continue
  serving requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-26T10:44:28Z — 120,967 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1043-v271` independently
  revalidated p0 decision-file 5 semantic offsets 1,536 and 1,664 plus p3
  decision-file 101 semantic offsets 1,280 and 1,408. Their 512 baseline pairs
  contain 352 false positives, 160 true duplicates, and no unresolved
  outcomes. Of the pairs, 511 were direct and one was chunked. All 1,062
  judgments were valid on their first attempts. The outcome Parquet SHA-256
  values are
  `a015055a5cc2900c0b146a3cb0360d920b71e7bda180d09fe0e9289ba5c83f33`,
  `9a8a614608cff09ad11db8e26c6f9f5fe2fafa52d7e94586f1ea7cee62e88755`,
  `46ebcf9172cfba81cd8b5a169ae45b8640f25986c32ce97df331b83aa2a0019d`,
  and
  `908d83a4f54a0c70f9f2b90bc702b0cc1af5cfd265ef9d7075ba1a517306bb8b`.
- Across the stable 952-checkpoint snapshot, all 135 unresolved model outcomes
  remain covered by 105 true-duplicate and 30 false-positive manual records.
  The adjusted totals are:

  - baseline: 97,554 pairs, 62,430 false positives, 35,124 true duplicates;
  - treatment: 23,413 pairs, 12,032 false positives, 11,381 true duplicates;
  - combined: 120,967 pairs, 74,462 false positives, 46,505 true duplicates.

- The next audit frontiers are p0 `(5, 1,792)`, p1 `(37, 0)`, p2 `(69, 0)`,
  and p3 `(101, 1,536)`. All four batch-priority 2-H100 workers continue
  serving requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-26T10:41:33Z — 120,455 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1041-v270` independently
  revalidated p0 decision-file 5 semantic offsets 1,280 and 1,408 plus p3
  decision-file 101 semantic offsets 1,024 and 1,152. Their 512 baseline pairs
  contain 454 false positives, 58 true duplicates, and no unresolved outcomes.
  All 512 pairs were direct, and all 1,047 judgments were valid on their first
  attempts. The outcome Parquet SHA-256 values are
  `d534b8fc990617ce9369ee1400d528d7f687daad7b95cdae04cd16002fdcad9d`,
  `c1a2e7fd16efa73460c3dfeabfa03a9b29ebdbb853d10af3384cb0e29256fef4`,
  `0330a14712a933465bd17dc6ba4bc12235f35ae8874b8009b79d74e5a8331a07`,
  and
  `317ae5d2819ef179e6d4932cd5a9966b7a5e94d8a1933620742cd1d1f3ba0895`.
- Across the stable 948-checkpoint snapshot, all 135 unresolved model outcomes
  remain covered by 105 true-duplicate and 30 false-positive manual records.
  The adjusted totals are:

  - baseline: 97,042 pairs, 62,078 false positives, 34,964 true duplicates;
  - treatment: 23,413 pairs, 12,032 false positives, 11,381 true duplicates;
  - combined: 120,455 pairs, 74,110 false positives, 46,345 true duplicates.

- The next audit frontiers are p0 `(5, 1,536)`, p1 `(37, 0)`, p2 `(69, 0)`,
  and p3 `(101, 1,280)`. All four batch-priority 2-H100 workers continue
  serving requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-26T10:37:54Z — 119,943 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1037-v269` independently
  revalidated p0 decision-file 5 semantic offsets 896, 1,024, and 1,152 plus
  p3 decision-file 101 semantic offsets 768 and 896. Their 640 baseline pairs
  contain 441 false positives, 199 true duplicates, and no unresolved
  outcomes. All 640 pairs were direct, and all 1,324 judgments were valid on
  their first attempts. The outcome Parquet SHA-256 values are
  `c5ed9704ef1a15b66098ec880d36a64b7984664cb7a14b860cf93a9ecb995bae`,
  `d788b31722fb493d9386d3fe0278365a167c18709fc1bea91e8dc7a5e2cebe48`,
  `c1204eada1b799f3ac067e5553b13778800c2aae0325142315bd5c2293fd18f5`,
  `b20764d7aa40a365b0edc501395b410e174e2cac061cb252949bc84180cbfea2`,
  and
  `880ccb633d905e20f83e61559dcdaa90ed405dce05f8f1bff28611608e162576`.
- Across the stable 944-checkpoint snapshot, all 135 unresolved model outcomes
  remain covered by 105 true-duplicate and 30 false-positive manual records.
  The adjusted totals are:

  - baseline: 96,530 pairs, 61,624 false positives, 34,906 true duplicates;
  - treatment: 23,413 pairs, 12,032 false positives, 11,381 true duplicates;
  - combined: 119,943 pairs, 73,656 false positives, 46,287 true duplicates.

- The next audit frontiers are p0 `(5, 1,280)`, p1 `(37, 0)`, p2 `(69, 0)`,
  and p3 `(101, 1,024)`. All four batch-priority 2-H100 workers continue
  serving requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-26T10:34:26Z — 119,303 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1034-v268` independently
  revalidated p0 decision-file 5 semantic offset 768 and p3 decision-file 101
  semantic offsets 512 and 640. Their 384 baseline pairs contain 248 false
  positives, 136 true duplicates, and no unresolved outcomes. Of the pairs,
  383 were direct and one was chunked. All 843 judgments were valid on their
  first attempts. The outcome Parquet SHA-256 values are
  `3b12b954b3b9159c2bf8630cc4fe54b2e41a8c8926a3121c1d436f5994a67278`,
  `3c02e053d4e6b0a7a756299b5161b953e6cc8afc2c62d74de8a1fcc528f42298`,
  and
  `a7e1e7a9c682cf33141296988b218f40c5e5938b0571e664bc72161f5eddef73`.
- Across the stable 939-checkpoint snapshot, all 135 unresolved model outcomes
  remain covered by 105 true-duplicate and 30 false-positive manual records.
  The adjusted totals are:

  - baseline: 95,890 pairs, 61,183 false positives, 34,707 true duplicates;
  - treatment: 23,413 pairs, 12,032 false positives, 11,381 true duplicates;
  - combined: 119,303 pairs, 73,215 false positives, 46,088 true duplicates.

- The next audit frontiers are p0 `(5, 896)`, p1 `(37, 0)`, p2 `(69, 0)`,
  and p3 `(101, 768)`. All four batch-priority 2-H100 workers continue serving
  requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-26T10:31:44Z — 118,919 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1031-v267` independently
  revalidated p0 decision-file 5 semantic offset 640 and p3 decision-file 101
  semantic offset 384. Their 256 baseline pairs contain 185 false positives,
  71 true duplicates, and no unresolved outcomes. Of the pairs, 243 were
  direct and 13 were chunked. All 1,878 judgments were valid on their first
  attempts. The outcome Parquet SHA-256 values are
  `44938285292476d498cd1c28a171f416a405e1a85cd060a4e3e6352bbf76699b`
  and
  `b085b759190e15311ce2f5b347a0c5afa39df9931921fd3a293c1f4d8625c1d0`.
- Across the stable 936-checkpoint snapshot, all 135 unresolved model outcomes
  remain covered by 105 true-duplicate and 30 false-positive manual records.
  The adjusted totals are:

  - baseline: 95,506 pairs, 60,935 false positives, 34,571 true duplicates;
  - treatment: 23,413 pairs, 12,032 false positives, 11,381 true duplicates;
  - combined: 118,919 pairs, 72,967 false positives, 45,952 true duplicates.

- The next audit frontiers are p0 `(5, 768)`, p1 `(37, 0)`, p2 `(69, 0)`,
  and p3 `(101, 512)`. All four batch-priority 2-H100 workers continue serving
  requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-26T10:29:12Z — 118,663 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1028-v266` independently
  revalidated p0 decision-file 5 semantic offsets 384 and 512. Their 256
  baseline pairs contain 181 false positives, 75 true duplicates, and no
  unresolved outcomes. Of the pairs, 250 were direct and six were chunked.
  All 1,216 judgments were valid on their first attempts. The outcome Parquet
  SHA-256 values are
  `4fd5da118d8552f2ab7b6a4c730ab9b2abc8f560890aed19ca38c611f285fcc3`
  and
  `17192789312aa622cb989dcc53d16cbb0110c13944bccccf9f5411c5ee061d47`.
- Across the stable 934-checkpoint snapshot, all 135 unresolved model outcomes
  remain covered by 105 true-duplicate and 30 false-positive manual records.
  The adjusted totals are:

  - baseline: 95,250 pairs, 60,750 false positives, 34,500 true duplicates;
  - treatment: 23,413 pairs, 12,032 false positives, 11,381 true duplicates;
  - combined: 118,663 pairs, 72,782 false positives, 45,881 true duplicates.

- The next audit frontiers are p0 `(5, 640)`, p1 `(37, 0)`, p2 `(69, 0)`,
  and p3 `(101, 384)`. All four batch-priority 2-H100 workers continue serving
  requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-26T10:23:21Z — 118,407 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1023-v265` independently
  revalidated p0 decision-file 5 semantic offset 256. Its 128 baseline pairs
  contain 76 false positives, 52 true duplicates, and no unresolved outcomes.
  Of the pairs, 118 were direct and ten were chunked. All 847 judgments were
  valid on their first attempts. The outcome Parquet SHA-256 is
  `8a329d85132851a9a3e8018c0a3ebf98189336a5119a74f1eb510793ebdfaecf`.
- Across the stable 932-checkpoint snapshot, all 135 unresolved model outcomes
  remain covered by 105 true-duplicate and 30 false-positive manual records.
  The adjusted totals are:

  - baseline: 94,994 pairs, 60,569 false positives, 34,425 true duplicates;
  - treatment: 23,413 pairs, 12,032 false positives, 11,381 true duplicates;
  - combined: 118,407 pairs, 72,601 false positives, 45,806 true duplicates.

- The next audit frontiers are p0 `(5, 384)`, p1 `(37, 0)`, p2 `(69, 0)`,
  and p3 `(101, 384)`. All four batch-priority 2-H100 workers continue serving
  requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-26T10:14:36Z — 118,279 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1014-v264` independently
  revalidated p3 decision-file 101 semantic offset 256. Its 128 baseline pairs
  contain 81 false positives, 47 true duplicates, and no unresolved outcomes.
  Of the pairs, 120 were direct and eight were chunked. All 767 judgments were
  valid on their first attempts. The outcome Parquet SHA-256 is
  `f3232b244f34721a60f1b48cbfaeee3c8456eb595c9a1c984cb7ebd9d2b3237e`.
- Across the stable 931-checkpoint snapshot, all 135 unresolved model outcomes
  remain covered by 105 true-duplicate and 30 false-positive manual records.
  The adjusted totals are:

  - baseline: 94,866 pairs, 60,493 false positives, 34,373 true duplicates;
  - treatment: 23,413 pairs, 12,032 false positives, 11,381 true duplicates;
  - combined: 118,279 pairs, 72,525 false positives, 45,754 true duplicates.

- The next audit frontiers are p0 `(5, 256)`, p1 `(37, 0)`, p2 `(69, 0)`,
  and p3 `(101, 384)`. All four batch-priority 2-H100 workers continue serving
  requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-26T10:11:21Z — 118,151 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1009-v263` independently
  revalidated two baseline checkpoints with 256 pairs and no unresolved
  outcomes:

  - p0 decision-file 5 semantic offset 128 contains 89 false positives and 39
    true duplicates; outcome Parquet SHA-256 is
    `724d486e60a087369ccbb103d87df9c9e5d90f8046b7273dfcc21a0db09eab3e`;
  - p3 decision-file 101 semantic offset 128 contains 85 false positives and
    43 true duplicates; outcome Parquet SHA-256 is
    `05f82189cd233bb8c46fc49eb5ea30faccac37f8b8ad9f674c5905d8410f1cbf`.

- Of the pairs, 225 were direct and 31 were chunked. All 3,486 judgments were
  valid on their first attempts.
- Across the stable 930-checkpoint snapshot, all 135 unresolved model outcomes
  remain covered by 105 true-duplicate and 30 false-positive manual records.
  The adjusted totals are:

  - baseline: 94,738 pairs, 60,412 false positives, 34,326 true duplicates;
  - treatment: 23,413 pairs, 12,032 false positives, 11,381 true duplicates;
  - combined: 118,151 pairs, 72,444 false positives, 45,707 true duplicates.

- The next audit frontiers are p0 `(5, 256)`, p1 `(37, 0)`, p2 `(69, 0)`,
  and p3 `(101, 256)`. All four batch-priority 2-H100 workers continue serving
  requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-26T10:01:20Z — 117,895 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0951-v255` independently
  revalidated the final two checkpoints in p1 decision-file 36. Semantic
  offsets 5,632 and 5,760 contain 219 treatment pairs: 140 model false
  positives, 78 model true duplicates, and one unresolved outcome. The first
  checkpoint contains 128 pairs and has outcome Parquet SHA-256
  `edf0ab60504dc6103ad5f8bf5e6e10fd969b1e9f527755242441885a127dca69`;
  the final partial checkpoint contains 91 pairs and has SHA-256
  `631d04064d16c16a7bed3a605c2fd005bf659f734dfbcb2c5c2e63fc05b79ca9`.
- Of the pairs, 218 were direct and one was chunked. The audit checked 457
  judgments across 461 request attempts. Four hundred fifty-six judgments
  produced valid responses; five invalid attempts affected two retried
  judgments and caused the one unresolved outcome.
- Complete-text comparison resolves `part-00036-of-00128.parquet:8957` as a
  true duplicate. Both SFT records contain the same protein-secondary-structure
  question, ten answer choices, 73-line reasoning, conclusion, and answer.
  Their only difference is equivalent final-answer LaTeX:
  `\boxed{B}` versus `\boxed{\text{B}}`. The member/canonical texts have
  5,668/5,675 characters and character, line, and word-sequence similarity
  0.999383, 0.986301, and 0.998788. Their SHA-256 values are
  `28c2e76d3640417b72ebc39150013010c423a40438a75bcc6e848f5f148a2a39` /
  `3c2ccaab237a099e1f2c840d93c18583a588c3295deb407c20db2161f7d8c25f`;
  inspection-artifact SHA-256 is
  `f41ffa5624a48d57e0031fc59f6662a11bcdd9b06dfa732444ed2c9da6067b45`.
- `/rav/datakit-6854-inspect-row8957-0955-v257` persisted the complete source
  texts, semantic evidence, and diff.
  `/rav/datakit-6854-publish-row8957-0959-v260` wrote the immutable manual
  record, and `/rav/datakit-6854-verify-row8957-1000-v261` separately reread
  the source pair, semantic checkpoint, inspection artifact, record,
  deterministic Parquet bytes, and completion marker. Semantic-evidence,
  manual-record, and manual-Parquet SHA-256 values are
  `d078b97e5138f60e6ad52f1462a204e99bd368153c4c862228c9db053bb64b38`,
  `cfa5d08215e1f08321303569d4b984d064a977c1012afdcd218c43ee8dc7826c`,
  and `87b1678b0f65c78b444066ae1d6f187596c2283da383d44babed423c873ac946`.
- Across the stable 928-checkpoint snapshot, all 135 unresolved model outcomes
  are covered by 105 true-duplicate and 30 false-positive manual records. The
  adjusted totals are:

  - baseline: 94,482 pairs, 60,238 false positives, 34,244 true duplicates;
  - treatment: 23,413 pairs, 12,032 false positives, 11,381 true duplicates;
  - combined: 117,895 pairs, 72,270 false positives, 45,625 true duplicates.

- The next audit frontiers are p0 `(5, 128)`, p1 `(37, 0)`, p2 `(69, 0)`,
  and p3 `(101, 128)`. All four batch-priority 2-H100 workers continue serving
  requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-26T09:49:16Z — 117,676 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0948-v254` independently
  revalidated five checkpoints with 640 pairs and no unresolved outcomes:

  - p1 decision-file 36 semantic offsets 5,120 through 5,504 contain 512
    treatment pairs, 132 false positives, and 380 true duplicates;
  - p3 decision-file 101 semantic offset 0 contains 128 baseline pairs, 120
    false positives, and eight true duplicates.

- Of the pairs, 611 were direct and 29 were chunked. All 5,308 judgments were
  valid on their first attempts. Outcome Parquet SHA-256 values, in p1 then p3
  frontier order, are:

  - `69292bb18c63b540e3292d06aeb8a456bb8e9e43665f453c790528ab6b2e63b7`;
  - `270dd60edcbff59cd2a4ff7cc307f055579fc0aa5b3f3ed9ed21ad080f0af5f5`;
  - `ed7129eb9d24fec2e72fe1998104a8f3fc5ac589cb418b6f455c0be4fc2dd7ec`;
  - `b255208c14eab2c58e0a7403f7243b310295217fedc2f0a561294e9227c350b9`;
  - `ee62358168774168eba936fd1686765102c0c0e33aa93c563743b1e32fd5326f`.

- Across the stable 926-checkpoint snapshot, all 134 unresolved model outcomes
  remain covered by 104 true-duplicate and 30 false-positive manual records.
  The adjusted totals are:

  - baseline: 94,482 pairs, 60,238 false positives, 34,244 true duplicates;
  - treatment: 23,194 pairs, 11,892 false positives, 11,302 true duplicates;
  - combined: 117,676 pairs, 72,130 false positives, 45,546 true duplicates.

- The next audit frontiers are p0 `(5, 128)`, p1 `(36, 5632)`, p2 `(69, 0)`,
  and p3 `(101, 128)`. All four batch-priority 2-H100 workers continue serving
  requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-26T09:46:25Z — 117,036 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0942-v250` independently
  revalidated p1 decision-file 36 semantic offsets 4,480 through 4,992. Their
  640 pairs contain 467 model false positives, 172 model true duplicates, and
  one unresolved outcome. The block comprises 152 baseline pairs (145 false
  positives and seven true duplicates) and 488 treatment pairs (322 false
  positives, 165 true duplicates, and one unresolved). Of the pairs, 639 were
  direct and one was chunked. All 1,418 judgments were valid on their first
  attempts. Outcome Parquet SHA-256 values, in offset order, are:

  - `454f17ef117cf3e3f3714cc10ef39113d67495380b193adab03e1de0105d79b3`;
  - `0b5ed97c9568466d99603b7e564c1f75d949470811251c4cc9394a2d65ba87dd`;
  - `596b81956eaff2051bbf8747054efaaaff69093c3e0547117477abe9bbda2bbd`;
  - `6dea2fcd5faf4cf5625fbbc1209f65645426895f75e85fdcadc114d702d91ef6`;
  - `b5723b699d2ba7ad1b664c2368d44f5928699d471107de363e9810e51f53d897`.

- Complete-text inspection resolves the treatment ambiguity as a true
  duplicate. `part-00036-of-00128.parquet:7810` contains two instances of the
  same explicitly automated Wikiteam welcome-message boilerplate. Their
  substantive body is identical. The differences are a username and timestamp
  in the thread header, the generic member-only salutation `Welcome`, and
  trailing whitespace. These are source metadata, template boilerplate, and
  formatting rather than a distinct request, fact, instruction, or training
  example. The member/canonical texts have 266/255 characters, five/four
  lines, and character, line, and word-sequence similarity 0.921305, 0.444444,
  and 0.963855. Their SHA-256 values are
  `52cec00fda64dabbc5a0e74241e07d498af04bf9b71ec1cea0ccae8655c0adc7` /
  `19953c25d01df09648766416deb366bc2a5c6bdf812a060e6f92b1bfa4a5d707`;
  inspection-artifact SHA-256 is
  `c3a446a2d79f0e286fe93ffb5150a1f7ebb07a257eeaedd760f3e00333cc5ac7`.

- `/rav/datakit-6854-inspect-row7810-0944-v251` persisted the complete source
  texts, semantic evidence, and diff.
  `/rav/datakit-6854-publish-row7810-0946-v252` wrote the immutable manual
  record, and `/rav/datakit-6854-verify-row7810-0946-v253` separately reread
  the source pair, semantic checkpoint, inspection artifact, record,
  deterministic Parquet bytes, and completion marker. Semantic-evidence,
  manual-record, and manual-Parquet SHA-256 values are
  `f8b728cb97bc5bb2002b5b81ea850a07476b4deb1e090ed152f3e5aa8ca27226`,
  `aacca10408b292950a7907a87eb36ae210773cdfa688aa524485ca7b704ec57d`,
  and `24bb054a9a2ec6fff3092f597fe27ee2ff38ab7cb5847260d6ae045005fc5578`.

- Across the stable 921-checkpoint snapshot, all 134 unresolved model outcomes
  are covered by 104 true-duplicate and 30 false-positive manual records. The
  adjusted totals are:

  - baseline: 94,354 pairs, 60,118 false positives, 34,236 true duplicates;
  - treatment: 22,682 pairs, 11,760 false positives, 10,922 true duplicates;
  - combined: 117,036 pairs, 71,878 false positives, 45,158 true duplicates.

- The next audit frontiers are p0 `(5, 128)`, p1 `(36, 5120)`, p2 `(69, 0)`,
  and p3 `(101, 0)`. All four batch-priority 2-H100 workers continue serving
  requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-26T09:40:06Z — 116,396 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0932-v243` independently
  revalidated p1 decision-file 36 semantic offset 4,352. Its 128 baseline
  pairs contain 85 model false positives, 41 model true duplicates, and two
  unresolved outcomes. All pairs were direct. The audit checked 263 judgments
  across 271 request attempts; 11 invalid attempts affected five retried
  judgments. The outcome Parquet SHA-256 is
  `9fca7c2e527f4962df6be943ef5558f5b13c24eb4455bceb0694411cb6f3e604`.

- Complete-text comparison resolves both ambiguous pairs as true duplicates:

  - `part-00036-of-00128.parquet:7517` contains identical 85-line astronomy
    SFT texts: the question, choices, reasoning, option analysis, conclusion,
    and answer all match. The sole difference is final
    `\boxed{\text{J}}` versus `\boxed{J}` formatting. The texts have
    7,689/7,682 characters and character, line, and word-sequence similarity
    0.999545, 0.988235, and 0.999180. Member/canonical text SHA-256 values are
    `9b87f5cbc13998f1152d3c68495975b1c90e0734078d3ab94172b6d84f22589e` /
    `fcf2b2ece5a3d05c1c733e1d4b7af3f85d352b532463ff1c5afa0348b3bea6f0`;
    inspection-artifact SHA-256 is
    `26832cf0fe832d257f3560d7faaeeafccbb48e12fcb9c581f064d224cad61486`.
  - `part-00036-of-00128.parquet:7523` contains identical 276-line consumer-law
    SFT texts. The sole difference is final `\boxed{\text{A}}` versus
    `\boxed{A}` formatting. The texts have 13,164/13,157 characters and
    character, line, and word-sequence similarity 0.999734, 0.996377, and
    0.999525. Member/canonical text SHA-256 values are
    `34d33c3e231c011454d192e2197d9d69c96aed9d3977945327d17fad32c7e050` /
    `9f3bd191706207927c0b4bf1cb9eec731bc495344f108b292fb3e6882b2bc648`;
    inspection-artifact SHA-256 is
    `42364787b942801203167d8cc5663fb40b9dc312b6e706fc95d5871e38a14fb9`.

- Inspection jobs `/rav/datakit-6854-inspect-row7517-0934-v244` and
  `/rav/datakit-6854-inspect-row7523-0934-v245` persisted the full source
  texts, semantic evidence, and complete diffs. Publisher jobs
  `/rav/datakit-6854-publish-row7517-0940-v246` and
  `/rav/datakit-6854-publish-row7523-0940-v247` wrote immutable manual records.
  Separate verifier jobs `/rav/datakit-6854-verify-row7517-0940-v248` and
  `/rav/datakit-6854-verify-row7523-0940-v249` reread the source pairs,
  semantic checkpoint, inspection artifacts, records, deterministic Parquet
  bytes, and completion markers. In pair order, semantic-evidence SHA-256
  values are
  `30957fb7de32f234b8decd3bd7876017f3a758ff2f49a6bf714a79a1ca32dc14`
  and
  `97f04647377fba89248c4decdeccfe4cdfd47485ef79bef989861e6be51c2d06`;
  manual-record SHA-256 values are
  `82bb62ca28483ad7270b9df30d304e6546118edcaa85f64b54044e6a8cff9929`
  and
  `d67beeadedc3d5910f06993d311401c87655ad72b8e607c63e892b3e0c99f659`;
  manual-Parquet SHA-256 values are
  `162b2476c17098016c4a4a3d855cda17dd59324919ba01d91c4f1cddca97bd5f`
  and
  `ce4cdd21eb698569e8fb817ea53b513cd8fa3544588118ecb5cd1992fd854cef`.

- Across the stable 916-checkpoint snapshot, all 133 unresolved model outcomes
  are covered by 103 true-duplicate and 30 false-positive manual records. The
  adjusted totals are:

  - baseline: 94,202 pairs, 59,973 false positives, 34,229 true duplicates;
  - treatment: 22,194 pairs, 11,438 false positives, 10,756 true duplicates;
  - combined: 116,396 pairs, 71,411 false positives, 44,985 true duplicates.

- The next audit frontiers are p0 `(5, 128)`, p1 `(36, 4480)`, p2 `(69, 0)`,
  and p3 `(101, 0)`. All four batch-priority 2-H100 workers continue serving
  requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-26T09:31:19Z — 116,268 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0931-v242` independently
  revalidated p1 decision-file 36 semantic offset 4,224. Its 128 baseline
  pairs contain 77 false positives, 51 true duplicates, and no unresolved
  outcomes. All pairs were direct. All 265 judgments were valid on their first
  attempts. The outcome Parquet SHA-256 is
  `7295eac4cf4a599ab1ae9feb05396dc7554032c564709e2d440eb26447f0bb9c`.

- Across the stable 915-checkpoint snapshot, all 131 unresolved model outcomes
  remain covered by 101 true-duplicate and 30 false-positive manual records.
  The adjusted totals are:

  - baseline: 94,074 pairs, 59,888 false positives, 34,186 true duplicates;
  - treatment: 22,194 pairs, 11,438 false positives, 10,756 true duplicates;
  - combined: 116,268 pairs, 71,326 false positives, 44,942 true duplicates.

- The next audit frontiers are p0 `(5, 128)`, p1 `(36, 4352)`, p2 `(69, 0)`,
  and p3 `(101, 0)`. All four batch-priority 2-H100 workers continue serving
  requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-26T09:30:02Z — 116,140 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0930-v241` independently
  revalidated p1 decision-file 36 semantic offset 4,096. Its 128 baseline
  pairs contain 85 false positives, 43 true duplicates, and no unresolved
  outcomes. All pairs were direct. All 265 judgments were valid on their first
  attempts. The outcome Parquet SHA-256 is
  `78d91c787e76a55986ba6f4955fad5c1889abcc2ce30645589b50a10dc02a0b6`.

- Across the stable 914-checkpoint snapshot, all 131 unresolved model outcomes
  remain covered by 101 true-duplicate and 30 false-positive manual records.
  The adjusted totals are:

  - baseline: 93,946 pairs, 59,811 false positives, 34,135 true duplicates;
  - treatment: 22,194 pairs, 11,438 false positives, 10,756 true duplicates;
  - combined: 116,140 pairs, 71,249 false positives, 44,891 true duplicates.

- The next audit frontiers are p0 `(5, 128)`, p1 `(36, 4224)`, p2 `(69, 0)`,
  and p3 `(101, 0)`. All four batch-priority 2-H100 workers continue serving
  requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-26T09:27:32Z — 116,012 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0927-v239` independently
  revalidated p1 decision-file 36 semantic offset 3,968. Its 128 baseline
  pairs contain 106 false positives, 22 true duplicates, and no unresolved
  outcomes. All pairs were direct. All 265 judgments were valid on their first
  attempts. The outcome Parquet SHA-256 is
  `2690b5e06dddefd21108b80c9c099d689ee538d2413597c2cbeaed4cf8b90ee2`.

- Across the stable 913-checkpoint snapshot, all 131 unresolved model outcomes
  remain covered by 101 true-duplicate and 30 false-positive manual records.
  The adjusted totals are:

  - baseline: 93,818 pairs, 59,726 false positives, 34,092 true duplicates;
  - treatment: 22,194 pairs, 11,438 false positives, 10,756 true duplicates;
  - combined: 116,012 pairs, 71,164 false positives, 44,848 true duplicates.

- The next audit frontiers are p0 `(5, 128)`, p1 `(36, 4096)`, p2 `(69, 0)`,
  and p3 `(101, 0)`. All four batch-priority 2-H100 workers continue serving
  requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-26T09:26:19Z — 115,884 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0926-v238` independently
  revalidated p1 decision-file 36 semantic offset 3,840. Its 128 baseline
  pairs contain 96 false positives, 32 true duplicates, and no unresolved
  outcomes. All pairs were direct. All 260 judgments were valid on their first
  attempts. The outcome Parquet SHA-256 is
  `b9ee18e622062ce8a6cb4285f503ea15c44893bb3183be750f5c6d963f1f4f1b`.

- Across the stable 912-checkpoint snapshot, all 131 unresolved model outcomes
  remain covered by 101 true-duplicate and 30 false-positive manual records.
  The adjusted totals are:

  - baseline: 93,690 pairs, 59,620 false positives, 34,070 true duplicates;
  - treatment: 22,194 pairs, 11,438 false positives, 10,756 true duplicates;
  - combined: 115,884 pairs, 71,058 false positives, 44,826 true duplicates.

- The next audit frontiers are p0 `(5, 128)`, p1 `(36, 3968)`, p2 `(69, 0)`,
  and p3 `(101, 0)`. All four batch-priority 2-H100 workers continue serving
  requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-26T09:25:10Z — 115,756 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0925-v237` independently
  revalidated p1 decision-file 36 semantic offset 3,712. Its 128 baseline
  pairs contain 100 false positives, 28 true duplicates, and no unresolved
  outcomes. All pairs were direct. All 261 judgments were valid on their first
  attempts. The outcome Parquet SHA-256 is
  `ca7a95c6dc5118636d348f9fe35863f9691ab487a7663cc9ae81d34cc11b8918`.

- Across the stable 911-checkpoint snapshot, all 131 unresolved model outcomes
  remain covered by 101 true-duplicate and 30 false-positive manual records.
  The adjusted totals are:

  - baseline: 93,562 pairs, 59,524 false positives, 34,038 true duplicates;
  - treatment: 22,194 pairs, 11,438 false positives, 10,756 true duplicates;
  - combined: 115,756 pairs, 70,962 false positives, 44,794 true duplicates.

- The next audit frontiers are p0 `(5, 128)`, p1 `(36, 3840)`, p2 `(69, 0)`,
  and p3 `(101, 0)`. All four batch-priority 2-H100 workers continue serving
  requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-26T09:23:52Z — 115,628 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0923-v236` independently
  revalidated p1 decision-file 36 semantic offset 3,584. Its 128 baseline
  pairs contain 94 false positives, 34 true duplicates, and no unresolved
  outcomes. Of the pairs, 127 were direct and one was chunked. All 320
  judgments were valid on their first attempts. The outcome Parquet SHA-256 is
  `c37f1497d3ff983532f763203668862e8b61e3f5eb5f8e496e0323274a06db0e`.

- Across the stable 910-checkpoint snapshot, all 131 unresolved model outcomes
  remain covered by 101 true-duplicate and 30 false-positive manual records.
  The adjusted totals are:

  - baseline: 93,434 pairs, 59,424 false positives, 34,010 true duplicates;
  - treatment: 22,194 pairs, 11,438 false positives, 10,756 true duplicates;
  - combined: 115,628 pairs, 70,862 false positives, 44,766 true duplicates.

- The next audit frontiers are p0 `(5, 128)`, p1 `(36, 3712)`, p2 `(69, 0)`,
  and p3 `(101, 0)`. All four batch-priority 2-H100 workers continue serving
  requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-26T09:21:09Z — 115,500 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0921-v234` independently
  revalidated p1 decision-file 36 semantic offsets 3,328 and 3,456. Their 256
  baseline pairs contain 156 false positives, 100 true duplicates, and no
  unresolved outcomes. All pairs were direct. Of 530 request attempts, two
  invalid attempts for one judgment were retried; all 528 judgments have valid
  outcomes. The outcome Parquet SHA-256 values are:

  - `01aead1afc5451a77e172692db957f0581574e2721d9c763526712d5ecfffd5b`;
  - `ec080aaec5610688711a5a495e692f1c8b9ac7b060b87f1126e202086d70f264`.

- Across the stable 909-checkpoint snapshot, all 131 unresolved model outcomes
  remain covered by 101 true-duplicate and 30 false-positive manual records.
  The adjusted totals are:

  - baseline: 93,306 pairs, 59,330 false positives, 33,976 true duplicates;
  - treatment: 22,194 pairs, 11,438 false positives, 10,756 true duplicates;
  - combined: 115,500 pairs, 70,768 false positives, 44,732 true duplicates.

- The next audit frontiers are p0 `(5, 128)`, p1 `(36, 3584)`, p2 `(69, 0)`,
  and p3 `(101, 0)`. All four batch-priority 2-H100 workers continue serving
  requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-26T09:17:50Z — 115,244 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0917-v232` independently
  revalidated p1 decision-file 36 semantic offsets 3,072 and 3,200. Their 256
  baseline pairs contain 118 false positives, 138 true duplicates, and no
  unresolved outcomes. Of the pairs, 255 were direct and one was chunked. All
  621 judgments were valid on their first attempts. The outcome Parquet
  SHA-256 values are:

  - `0ff21af1aa1c340192f21f9dd94fe81718161d4fa620fdcbc3cab7120db8b767`;
  - `e98288318c4e203482b06ea609b19742d4c665185a953614afab8781e338e831`.

- Across the stable 907-checkpoint snapshot, all 131 unresolved model outcomes
  remain covered by 101 true-duplicate and 30 false-positive manual records.
  The adjusted totals are:

  - baseline: 93,050 pairs, 59,174 false positives, 33,876 true duplicates;
  - treatment: 22,194 pairs, 11,438 false positives, 10,756 true duplicates;
  - combined: 115,244 pairs, 70,612 false positives, 44,632 true duplicates.

- The next audit frontiers are p0 `(5, 128)`, p1 `(36, 3328)`, p2 `(69, 0)`,
  and p3 `(101, 0)`. All four batch-priority 2-H100 workers continue serving
  requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-26T09:16:05Z — 114,988 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0916-v231` independently
  revalidated p1 decision-file 36 semantic offsets 2,816 and 2,944. Their 256
  baseline pairs contain 130 false positives, 126 true duplicates, and no
  unresolved outcomes. All pairs were direct. All 552 judgments were valid on
  their first attempts. The outcome Parquet SHA-256 values are:

  - `569c63536ffdacd3ed678d0bd61fedf1f5956b14af0d87d83e10ba89f0538539`;
  - `e083b227e03928e27bd9913602c5878b969bbf59b38ccca5e38d309c219b037e`.

- Across the stable 905-checkpoint snapshot, all 131 unresolved model outcomes
  remain covered by 101 true-duplicate and 30 false-positive manual records.
  The adjusted totals are:

  - baseline: 92,794 pairs, 59,056 false positives, 33,738 true duplicates;
  - treatment: 22,194 pairs, 11,438 false positives, 10,756 true duplicates;
  - combined: 114,988 pairs, 70,494 false positives, 44,494 true duplicates.

- The next audit frontiers are p0 `(5, 128)`, p1 `(36, 3072)`, p2 `(69, 0)`,
  and p3 `(101, 0)`. All four batch-priority 2-H100 workers continue serving
  requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-26T09:12:39Z — 114,732 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0912-v230` independently
  revalidated p1 decision-file 36 semantic offsets 2,432, 2,560, and 2,688.
  Their 384 baseline pairs contain 179 false positives, 205 true duplicates,
  and no unresolved outcomes. Of the pairs, 383 were direct and one was
  chunked. All 956 judgments were valid on their first attempts. The outcome
  Parquet SHA-256 values are:

  - `2941ebb2949634cad9bd44a150d582a49722e629385514014ad815dc3b03856a`;
  - `bfe0f38b279b3bc9a682796629cb0b5f128c32b2f858450267e4ea4f2095e1c4`;
  - `ddb60707d93555eb5aad57cc2da58e5555cca23db5f61d65d2c7354ddb317d2e`.

- Across the stable 903-checkpoint snapshot, all 131 unresolved model outcomes
  remain covered by 101 true-duplicate and 30 false-positive manual records.
  The adjusted totals are:

  - baseline: 92,538 pairs, 58,926 false positives, 33,612 true duplicates;
  - treatment: 22,194 pairs, 11,438 false positives, 10,756 true duplicates;
  - combined: 114,732 pairs, 70,364 false positives, 44,368 true duplicates.

- The next audit frontiers are p0 `(5, 128)`, p1 `(36, 2816)`, p2 `(69, 0)`,
  and p3 `(101, 0)`. All four batch-priority 2-H100 workers continue serving
  requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-26T09:10:18Z — 114,348 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0905-v226` independently
  revalidated p1 decision-file 36 semantic offset 2,304. Its 128 baseline pairs
  contain 70 model false positives, 57 model true duplicates, and one
  unresolved outcome. Of the pairs, 127 were direct and one was chunked. All
  315 judgments were valid on their first attempts. The outcome Parquet
  SHA-256 is
  `9a01882c44813f8609cb96ee06a931c5b3e34785528f33a7ab426cebb2fe2407`.
- Complete-text inspection resolves the ambiguity as a true duplicate. The
  4,166-character, 35-line member is the same fragile-X scientific article
  contained in the 6,664-character, 63-line canonical. The canonical adds
  article-derived Q&A; article differences are capitalization, dash style, and
  page chrome. The member-only `Forum for Science, Industry and Business`
  header and `10.06.2011` date are source metadata, not a distinct example,
  fact payload, or instruction. Character, line, and word-sequence similarity
  are 0.757710, 0.591837, and 0.761739 after the canonical additions.
  Pair location is `part-00036-of-00128.parquet:3691`; member/canonical text
  SHA-256 values are
  `c4c9cb7d94e6b3c8137414e2e1dc6ec14a3c29671458b9bcc76aaf0583ba1132` /
  `4f1bee92afaeeff03d199d77ca1403afbd08537520944dea9a19edd90d287e69`;
  inspection-artifact SHA-256 is
  `243421de3d2747b00385cedd37e9fd843bda8a47a03ec31809b7a5fd11f3531e`.
- `/rav/datakit-6854-publish-row3691-0909-v228` wrote the immutable hash-bound
  manual record. `/rav/datakit-6854-verify-row3691-0910-v229` separately reread
  the source pair, semantic checkpoint, complete inspection artifact, manual
  record, deterministic Parquet bytes, and completion marker. The
  semantic-evidence, manual-record, and manual-Parquet SHA-256 values are
  `8f2949f2dacede6f56d4a7fcc16ad11ac5b2ca17029286f59bf6e15770367660`,
  `326abae037fe3f8530df9adbe1da95660a0e86804ff29b6d15946fd46f62fc59`,
  and `4be3ba0986d33dfd582fd63d4bf2ad9916d3cff260ccaeafa96157372dd63219`.
- Across the stable 900-checkpoint snapshot, all 131 unresolved model outcomes
  are covered by 101 true-duplicate and 30 false-positive manual records. The
  adjusted totals are:

  - baseline: 92,154 pairs, 58,747 false positives, 33,407 true duplicates;
  - treatment: 22,194 pairs, 11,438 false positives, 10,756 true duplicates;
  - combined: 114,348 pairs, 70,185 false positives, 44,163 true duplicates.

- The next audit frontiers are p0 `(5, 128)`, p1 `(36, 2432)`, p2 `(69, 0)`,
  and p3 `(101, 0)`. All four batch-priority 2-H100 workers continue serving
  requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-26T09:04:01Z — 114,220 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0903-v225` independently
  revalidated p1 decision-file 36 semantic offset 2,176. Its 128 baseline pairs
  contain 59 false positives, 69 true duplicates, and no unresolved outcomes.
  Of the pairs, 127 were direct and one was chunked. All 304 judgments were
  valid on their first attempts. The outcome Parquet SHA-256 is
  `71f0aec4688dd64fe6193c7ed991010cb2132b279e668944863fb4cd1bc604e6`.
- Across the stable 899-checkpoint snapshot, all 130 unresolved model outcomes
  remain covered by 100 true-duplicate and 30 false-positive manual records.
  The adjusted totals are:

  - baseline: 92,026 pairs, 58,677 false positives, 33,349 true duplicates;
  - treatment: 22,194 pairs, 11,438 false positives, 10,756 true duplicates;
  - combined: 114,220 pairs, 70,115 false positives, 44,105 true duplicates.

- The next audit frontiers are p0 `(5, 128)`, p1 `(36, 2304)`, p2 `(69, 0)`,
  and p3 `(101, 0)`. All four batch-priority 2-H100 workers continue serving
  requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-26T09:02:22Z — 114,092 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0902-v224` independently
  revalidated p1 decision-file 36 semantic offset 2,048. Its 128 direct
  baseline pairs contain 70 false positives, 58 true duplicates, and no
  unresolved outcomes. All 267 judgments were valid on their first attempts.
  The outcome Parquet SHA-256 is
  `068db42d7cd5ec97ff6c16480b6d310c345d361b44f40f27c34e715cf03ab4cb`.
- Across the stable 898-checkpoint snapshot, all 130 unresolved model outcomes
  remain covered by 100 true-duplicate and 30 false-positive manual records.
  The adjusted totals are:

  - baseline: 91,898 pairs, 58,618 false positives, 33,280 true duplicates;
  - treatment: 22,194 pairs, 11,438 false positives, 10,756 true duplicates;
  - combined: 114,092 pairs, 70,056 false positives, 44,036 true duplicates.

- The next audit frontiers are p0 `(5, 128)`, p1 `(36, 2176)`, p2 `(69, 0)`,
  and p3 `(101, 0)`. All four batch-priority 2-H100 workers continue serving
  requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-26T09:00:32Z — 113,964 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0900-v223` independently
  revalidated p1 decision-file 36 semantic offset 1,920. Its 128 direct
  baseline pairs contain 50 false positives, 78 true duplicates, and no
  unresolved outcomes. All 266 judgments were valid on their first attempts.
  The outcome Parquet SHA-256 is
  `9b2fe6e33b02db06ea9f8fc19cf980983a323aea8d38e84e8a67d627e0758054`.
- Across the stable 897-checkpoint snapshot, all 130 unresolved model outcomes
  remain covered by 100 true-duplicate and 30 false-positive manual records.
  The adjusted totals are:

  - baseline: 91,770 pairs, 58,548 false positives, 33,222 true duplicates;
  - treatment: 22,194 pairs, 11,438 false positives, 10,756 true duplicates;
  - combined: 113,964 pairs, 69,986 false positives, 43,978 true duplicates.

- The next audit frontiers are p0 `(5, 128)`, p1 `(36, 2048)`, p2 `(69, 0)`,
  and p3 `(101, 0)`. All four batch-priority 2-H100 workers continue serving
  requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-26T08:58:57Z — 113,836 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0858-v222` independently
  revalidated p1 decision-file 36 semantic offset 1,792. Its 128 direct
  baseline pairs contain 59 false positives, 69 true duplicates, and no
  unresolved outcomes. All 271 judgments were valid on their first attempts.
  The outcome Parquet SHA-256 is
  `4f8776aa17be90b79bfc62b37389d8ffcfdbbdd8f1525cd9da0a149ac1879478`.
- Across the stable 896-checkpoint snapshot, all 130 unresolved model outcomes
  remain covered by 100 true-duplicate and 30 false-positive manual records.
  The adjusted totals are:

  - baseline: 91,642 pairs, 58,498 false positives, 33,144 true duplicates;
  - treatment: 22,194 pairs, 11,438 false positives, 10,756 true duplicates;
  - combined: 113,836 pairs, 69,936 false positives, 43,900 true duplicates.

- The next audit frontiers are p0 `(5, 128)`, p1 `(36, 1920)`, p2 `(69, 0)`,
  and p3 `(101, 0)`. All four batch-priority 2-H100 workers continue serving
  requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-26T08:57:18Z — 113,708 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0856-v221` independently
  revalidated p1 decision-file 36 semantic offsets 1,536 and 1,664. Their 256
  direct baseline pairs contain 110 false positives, 146 true duplicates, and
  no unresolved outcomes. All 527 judgments were valid on their first
  attempts. The outcome Parquet SHA-256 values are
  `fc06f347956caaa94c2913f2f4528c293ef9ac2542f433efebbec33ba93300e9` and
  `c5b1e960b2df3d9fc47ba72c294e67dc33384849301e2976ee9a5ebf93de5152`.
- Across the stable 895-checkpoint snapshot, all 130 unresolved model outcomes
  remain covered by 100 true-duplicate and 30 false-positive manual records.
  The adjusted totals are:

  - baseline: 91,514 pairs, 58,439 false positives, 33,075 true duplicates;
  - treatment: 22,194 pairs, 11,438 false positives, 10,756 true duplicates;
  - combined: 113,708 pairs, 69,877 false positives, 43,831 true duplicates.

- The next audit frontiers are p0 `(5, 128)`, p1 `(36, 1792)`, p2 `(69, 0)`,
  and p3 `(101, 0)`. All four batch-priority 2-H100 workers continue serving
  requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-26T08:55:23Z — 113,452 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0855-v220` independently
  revalidated p1 decision-file 36 semantic offset 1,408. Its 128 direct
  baseline pairs contain 116 false positives, 12 true duplicates, and no
  unresolved outcomes. All 265 judgments were valid on their first attempts.
  The outcome Parquet SHA-256 is
  `299f98f5f46b2528892b50902481a0cb5a5a5db910364595980ec348bb7b8eec`.
- Across the stable 893-checkpoint snapshot, all 130 unresolved model outcomes
  remain covered by 100 true-duplicate and 30 false-positive manual records.
  The adjusted totals are:

  - baseline: 91,258 pairs, 58,329 false positives, 32,929 true duplicates;
  - treatment: 22,194 pairs, 11,438 false positives, 10,756 true duplicates;
  - combined: 113,452 pairs, 69,767 false positives, 43,685 true duplicates.

- The next audit frontiers are p0 `(5, 128)`, p1 `(36, 1536)`, p2 `(69, 0)`,
  and p3 `(101, 0)`. All four batch-priority 2-H100 workers continue serving
  requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-26T08:53:54Z — 113,324 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0853-v219` independently
  revalidated p1 decision-file 36 semantic offset 1,280. Its 128 direct
  baseline pairs contain 124 false positives, four true duplicates, and no
  unresolved outcomes. All 261 judgments were valid on their first attempts.
  The outcome Parquet SHA-256 is
  `f658a95ee3c87692da7ff6b6162e57a98935bb7651b906c7fcf84600fcc4aa2f`.
- Across the stable 892-checkpoint snapshot, all 130 unresolved model outcomes
  remain covered by 100 true-duplicate and 30 false-positive manual records.
  The adjusted totals are:

  - baseline: 91,130 pairs, 58,213 false positives, 32,917 true duplicates;
  - treatment: 22,194 pairs, 11,438 false positives, 10,756 true duplicates;
  - combined: 113,324 pairs, 69,651 false positives, 43,673 true duplicates.

- The next audit frontiers are p0 `(5, 128)`, p1 `(36, 1408)`, p2 `(69, 0)`,
  and p3 `(101, 0)`. All four batch-priority 2-H100 workers continue serving
  requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-26T08:52:19Z — 113,196 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0851-v218` independently
  revalidated p1 decision-file 36 semantic offset 1,152. Its 128 direct
  baseline pairs contain 120 false positives, eight true duplicates, and no
  unresolved outcomes. All 260 judgments were valid on their first attempts.
  The outcome Parquet SHA-256 is
  `1cf8224a2dd2d1d84d6f64b63bb125ce8fa330a107ef0fa5d34043d04d3f6227`.
- Across the stable 891-checkpoint snapshot, all 130 unresolved model outcomes
  remain covered by 100 true-duplicate and 30 false-positive manual records.
  The adjusted totals are:

  - baseline: 91,002 pairs, 58,089 false positives, 32,913 true duplicates;
  - treatment: 22,194 pairs, 11,438 false positives, 10,756 true duplicates;
  - combined: 113,196 pairs, 69,527 false positives, 43,669 true duplicates.

- The next audit frontiers are p0 `(5, 128)`, p1 `(36, 1280)`, p2 `(69, 0)`,
  and p3 `(101, 0)`. All four batch-priority 2-H100 workers continue serving
  requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-26T08:50:23Z — 113,068 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0849-v217` independently
  revalidated p1 decision-file 36 semantic offsets 896 and 1,024. Their 256
  baseline pairs contain 194 false positives, 62 true duplicates, and no
  unresolved outcomes. Of the pairs, 254 were direct and two were chunked.
  All 556 judgments were valid on their first attempts. The outcome Parquet
  SHA-256 values are
  `4c68f1b84c1faf2cd1551dde30b228e807396720af2dbb3fcfb994b40524ca11` and
  `d8bd1c657aa78abf0d83f67a64c00a7b05d652add61a44a73e86fdfbfea24084`.
- Across the stable 890-checkpoint snapshot, all 130 unresolved model outcomes
  remain covered by 100 true-duplicate and 30 false-positive manual records.
  The adjusted totals are:

  - baseline: 90,874 pairs, 57,969 false positives, 32,905 true duplicates;
  - treatment: 22,194 pairs, 11,438 false positives, 10,756 true duplicates;
  - combined: 113,068 pairs, 69,407 false positives, 43,661 true duplicates.

- The next audit frontiers are p0 `(5, 128)`, p1 `(36, 1152)`, p2 `(69, 0)`,
  and p3 `(101, 0)`. All four batch-priority 2-H100 workers continue serving
  requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-26T08:46:48Z — 112,812 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0846-v215` independently
  revalidated p1 decision-file 36 semantic offset 768. Its 128 baseline pairs
  contain 93 false positives, 35 true duplicates, and no unresolved outcomes.
  Of the pairs, 127 were direct and one was chunked. All 288 judgments were
  valid on their first attempts. The outcome Parquet SHA-256 is
  `795f05d4f8ee788baf35120e7f08c092bc65baa46b11d37daa1ba681d6c5b2a9`.
- Across the stable 888-checkpoint snapshot, all 130 unresolved model outcomes
  remain covered by 100 true-duplicate and 30 false-positive manual records.
  The adjusted totals are:

  - baseline: 90,618 pairs, 57,775 false positives, 32,843 true duplicates;
  - treatment: 22,194 pairs, 11,438 false positives, 10,756 true duplicates;
  - combined: 112,812 pairs, 69,213 false positives, 43,599 true duplicates.

- The next audit frontiers are p0 `(5, 128)`, p1 `(36, 896)`, p2 `(69, 0)`,
  and p3 `(101, 0)`. All four batch-priority 2-H100 workers continue serving
  requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-26T08:45:07Z — 112,684 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0844-v214` independently
  revalidated p1 decision-file 36 semantic offset 640. Its 128 direct baseline
  pairs contain 93 false positives, 35 true duplicates, and no unresolved
  outcomes. All 269 judgments were valid on their first attempts. The outcome
  Parquet SHA-256 is
  `d56841bbe9b0fcd71b02a1eef7cb2f1836c7482abf29d0bf56bed9ad1ae0d91b`.
- Across the stable 887-checkpoint snapshot, all 130 unresolved model outcomes
  remain covered by 100 true-duplicate and 30 false-positive manual records.
  The adjusted totals are:

  - baseline: 90,490 pairs, 57,682 false positives, 32,808 true duplicates;
  - treatment: 22,194 pairs, 11,438 false positives, 10,756 true duplicates;
  - combined: 112,684 pairs, 69,120 false positives, 43,564 true duplicates.

- The next audit frontiers are p0 `(5, 128)`, p1 `(36, 768)`, p2 `(69, 0)`,
  and p3 `(101, 0)`. All four batch-priority 2-H100 workers continue serving
  requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-26T08:42:58Z — 112,556 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0841-v213` independently
  revalidated three p1 decision-file 36 checkpoints at semantic offsets 256,
  384, and 512. Their 384 baseline pairs contain 261 false positives, 123 true
  duplicates, and no unresolved outcomes. Of the pairs, 368 were direct and 16
  were chunked. All 1,593 judgments were valid on their first attempts. The
  outcome Parquet SHA-256 values are:

  - `c7327b73dbcb8010a294b95b7aa7f59b38620be1f3b7d8e3ddf84c40d0f74091`;
  - `2b49b33b8cebb8e598863438ce2e7d936c49386d2a85d07eca39782277ab333e`;
  - `576119df1239cf580f60c2e613ae7977b2282481fa994fedf7d3bbd407350de0`.

- Across the stable 886-checkpoint snapshot, all 130 unresolved model outcomes
  remain covered by 100 true-duplicate and 30 false-positive manual records.
  The adjusted totals are:

  - baseline: 90,362 pairs, 57,589 false positives, 32,773 true duplicates;
  - treatment: 22,194 pairs, 11,438 false positives, 10,756 true duplicates;
  - combined: 112,556 pairs, 69,027 false positives, 43,529 true duplicates.

- The next audit frontiers are p0 `(5, 128)`, p1 `(36, 640)`, p2 `(69, 0)`,
  and p3 `(101, 0)`. All four batch-priority 2-H100 workers continue serving
  requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-26T08:40:27Z — 112,172 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0830-v206` independently
  revalidated eight checkpoints: p0 decision-file 5 offset 0; p1 decision-file
  36 offset 128; and p2 decision-file 68 offsets 4,992 through 5,632. Their
  955 pairs contain 470 model false positives, 483 model true duplicates, and
  two unresolved outcomes. Of the pairs, 916 were direct and 39 were chunked.
  The audit validated 6,701 judgments across 6,706 request attempts; five
  invalid responses affected two retried judgments. Checkpoint outcome
  SHA-256 values, in frontier order, are:

  - `9b18330e2f8008cd68818a0c62b0fbc961d15f50c6948b45ba3688dfaa50db55`;
  - `29947044a71f98410e2f25015e06591f470dbb96283f466c32fc2a904aa2b9e7`;
  - `46572e84ab245d99cca1c7728270270a70a86be425dc5180709de2158dd9fd62`;
  - `44e87d86f460f2f41949bc985373cda195ea64da842dfa6999e9ec20493b8e28`;
  - `73743726fc79c05130346c16a74809a6f34dec9c94e9dc684e0700f29783d45d`;
  - `7331d3e7f4d88cadc97dd9da3541d65519d0210a16e9353362da6e6608b792c0`;
  - `4ad54b615e87b08cf37cd3360eea6b0eafd8b9bc5b2c553e6e4e893f036eadc8`;
  - `dfd05513b0c6ab9f6cf59fd09d970299f2177f055e2586aebf75462dc20c9cec`.

- Complete-text inspection resolves both ambiguous treatment pairs as true
  duplicates:

  - `part-00068-of-00128.parquet:8756` contains the identical German question
    and answer in both 4-line texts. Only the isolated first-line
    source-generation fragments differ: `dass` versus
    `beibehalten einschließlich`. They are not distinct questions, answers,
    facts, or instructions. Member/canonical text SHA-256 values are
    `8039a3ff24e7a20be56e4056edb244d4a23131a9b7954549e922642b24ba42d3` /
    `205a885fce13ed7d01b2b3f3bfe5a5a93f19197e7b2b57555a6166c6e4f02730`;
    inspection-artifact SHA-256 is
    `8421e5e1352294e00103536aebdf8421c9efc5f956ba4f6b0ea700af163bed97`.
  - `part-00068-of-00128.parquet:8930` has identical 96-line question,
    choices, reasoning, and answer texts. The only difference is
    `\boxed{\text{A}}` versus `\boxed{A}`. Member/canonical text SHA-256
    values are
    `a1762b74ab59c21fd70a73ac9ac1db59c914788b4ae945ecfa6f7985dc5bf471` /
    `5ab434132eb6b26252c2f08c0ee78631ad34719ed9d144af41eb19858518ee18`;
    inspection-artifact SHA-256 is
    `5b59c4dc9f2d1a61bf94f2b0380a5d94862e995d0dd0af2460a94e41acb37ef9`.

- Publisher jobs `/rav/datakit-6854-publish-row8756-0839-v209` and
  `/rav/datakit-6854-publish-row8930-0839-v210` wrote the immutable hash-bound
  manual records. Separate jobs
  `/rav/datakit-6854-verify-row8756-0840-v211` and
  `/rav/datakit-6854-verify-row8930-0840-v212` reread the source pairs,
  semantic checkpoints, complete inspection artifacts, manual records,
  deterministic Parquet bytes, and completion markers. In pair order, their
  semantic-evidence SHA-256 values are
  `0b1cd2a23be7e3d6bc054cf2158a1186ce650bebcddac19fb9c015915a7cfed7` and
  `16c21edc30dab562e2b7501a671140ef15ff3138096fbc358e770ccf7353a929`;
  manual-record SHA-256 values are
  `e2c2da50c6d074b8db12df8df3380b420d286360a3ac6a5eaafa75d5feca1580` and
  `194d54a765c80731deeaa85c35f6d0defc61c0f6a241b5b1e42fc0dd71ca6a7f`;
  manual-Parquet SHA-256 values are
  `c1ed5d45d7ff25255785ffd97efb84a4eabd7aa811831b1549fab4287e3bb452` and
  `7e030a8384178fc5f73801fd3aeb232b50c63856f0341edbff2bcc633bc0d4d7`.

- Across the stable 883-checkpoint snapshot, all 130 unresolved model outcomes
  are covered by 100 true-duplicate and 30 false-positive manual records. The
  adjusted totals are:

  - baseline: 89,978 pairs, 57,328 false positives, 32,650 true duplicates;
  - treatment: 22,194 pairs, 11,438 false positives, 10,756 true duplicates;
  - combined: 112,172 pairs, 68,766 false positives, 43,406 true duplicates.

- The next audit frontiers are p0 `(5, 128)`, p1 `(36, 256)`, p2 `(69, 0)`,
  and p3 `(101, 0)`. All four batch-priority 2-H100 workers continue serving
  requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-26T08:28:02Z — 111,217 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0819-v196` independently
  revalidated six p2 decision-file 68 checkpoints at semantic offsets 4,224
  through 4,864. Their 768 pairs contain 543 model false positives, 222 model
  true duplicates, and three unresolved outcomes. All 767 direct and one
  chunked pair were covered by 1,603 judgments and 1,611 requests; 1,599
  attempts were valid and 12 invalid attempts affected four retried judgments.
  The six outcome Parquet SHA-256 values, in offset order, are
  `b4f5b8342139da6da47e020da5d13e05e5966f6bb86406f8b22011b49d912744`,
  `c2468d02d9a60af552e207ec2904e5811bcd81b5c03cad680ad021df4e3a81aa`,
  `bc876927ab91b49b16cac9ce7bd52d29db1058ba1cb221c97393847dd9cb6ba1`,
  `61b300b910df34055eba89da2397eadc9d685ccbe5c055b0b4c962e38db55318`,
  `3a28923806f93a170a76621e9f9ab7c87348b8348d9125664fe8185b110acecc`,
  and
  `a4e97cab76b61b2a895ee1074931ba7a715a89c8bc72d269032a8775baca758f`.
- Three parallel batch-priority jobs read every character and judgment for the
  ambiguous pairs:

  - `/rav/datakit-6854-inspect-row7480-0822-v197` compared the complete
    1,224- and 1,315-character math SFT records. Both ask the same limit
    problem and give the same Riemann-sum derivation: `x^2` on `[0,1]`,
    endpoints `k/n`, width `1/n`, the integral of `x^2` from 0 to 1,
    antiderivative `x^3/3`, and result `1/3`. Their different LaTeX layout and
    explanatory phrasing add no distinct problem, method, fact, or answer, so
    the pair is a true duplicate. Pair location
    `part-00068-of-00128.parquet:7480`; member/canonical text SHA-256
    `d4e44dec91d5b9a5b6a4b36341f2fd5f630e47b99ac754238ab040827aebc915` /
    `0b8ac053f1f77c7085bf293c9bb30cdaa9d6ddc10cb4b1eb32e2b4e905e45f53`;
    inspection-artifact SHA-256
    `09959f2d5adb1bd8d7f80483948e49d0d58a806516e531ddbf2b2b70fdec471d`.
  - `/rav/datakit-6854-inspect-row7538-0822-v198` proved that the complete
    9,777- and 9,770-character customer-service SFT texts are identical except
    for final `\boxed{\text{G}}` versus `\boxed{G}` formatting, so the pair is
    a true duplicate. Pair location `part-00068-of-00128.parquet:7538`;
    member/canonical text SHA-256
    `7efec9dfd355b3f18e844383034f33742ef2a67ea17e1e38b86118c6050d86fd` /
    `0ee4257462dea2888aaac7b8e610d2ddd51184d40f8b40fecc117638f06997c4`;
    inspection-artifact SHA-256
    `498c780c2794de47d0c695be451bf2cba75a3f796a195a68027a8cf518983201`.
  - `/rav/datakit-6854-inspect-row7541-0822-v199` proved that the complete
    7,745- and 7,738-character personality-traits SFT texts are identical
    except for final `\boxed{\text{B}}` versus `\boxed{B}` formatting, so the
    pair is a true duplicate. Pair location
    `part-00068-of-00128.parquet:7541`; member/canonical text SHA-256
    `f19a36e0de03b3189a260e99044de3ec79a59c299f63f9c7d52bfe1b5b13b792` /
    `726131c9c410ac09789baee5d15176a85cd24bf848f276751eecabd51d2b10cd`;
    inspection-artifact SHA-256
    `49efe240cae224cf139a2342b5d7d56f8691869d761316911ca0977443a2aeb3`.

- Three publisher jobs wrote the immutable hash-bound manual records. Separate
  jobs `/rav/datakit-6854-verify-row7480-0827-v203`,
  `/rav/datakit-6854-verify-row7538-0827-v204`, and
  `/rav/datakit-6854-verify-row7541-0827-v205` reread the source pairs,
  semantic checkpoints, inspection artifacts, records, deterministic Parquet
  bytes, and completion markers. In pair order, their semantic-evidence
  SHA-256 values are
  `7d90df23055ad118d5f9f2258e711ae1b865d6f3ffa3839f5b7e79d0946c7ec7`,
  `28f4b027cc77678968ff5b4477a7bf4b25d7ba092a39978a662b7a08c059d2cf`,
  and
  `175129e26059cbb0244e8bea7ba336f068fd73a0f40150ce951b2abe62f7f070`;
  manual-record SHA-256 values are
  `ec4299ff1c0c6b8e5a6e1464bc07843489878612fd4fd50bc31dc09b7e47bc60`,
  `53d957b3b080f88cbac61d3b5c70a01377a4f40c49994a1b68aaea0b98aceeab`,
  and
  `e8f06db509471865d40170449c046d07e8c72412a4eb67bd50c78893ac6eab69`;
  manual-Parquet SHA-256 values are
  `cfae4d4ca97e9f7c7e8f241b80121c7effbb7b07ffc022c96dc6cbdcd8b2cb90`,
  `da49cd52644d5971462fa63017734233e0b620db4f9efdfd53792ad105e5038f`,
  and
  `4ab021d47ababb2bb6fa3eba62472088ab9c139a5794efbf5d1a0b9c72d80e7c`.
- Across the stable 875-checkpoint snapshot, all 128 unresolved model outcomes
  are covered by 98 true-duplicate and 30 false-positive manual records. The
  adjusted totals are:

  - baseline: 89,722 pairs, 57,115 false positives, 32,607 true duplicates;
  - treatment: 21,495 pairs, 11,181 false positives, 10,314 true duplicates;
  - combined: 111,217 pairs, 68,296 false positives, 42,921 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T08:18:00Z — 110,449 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0806-v190` independently
  revalidated p2 decision-file 68 baseline semantic offset 4,096. Its 128
  direct-review pairs contain 79 model false positives, 48 model true
  duplicates, and one unresolved outcome. The checkpoint contains 263
  judgments across 267 requests: 262 valid attempts and five invalid attempts
  affecting two retried judgments. The outcome Parquet SHA-256 is
  `1510f7ed0b1ba3d546e4620cf05715fa97976b11dbd2a724a33b5e13fb74d854`.
- `/rav/datakit-6854-inspect-row7309-0809-v191` read all 7,597 member
  characters, all 7,604 canonical characters, and every persisted judgment for
  pair `part-00068-of-00128.parquet:7309`. The 88-line SFT records have
  character, line, and word sequence ratios of 0.999540, 0.988636, and
  0.999576. Every question, option, reasoning paragraph, fact, and answer is
  character-for-character identical; the only difference is the final
  `\boxed{D}` versus `\boxed{\text{D}}` wrapper. The member/canonical SHA-256
  values are
  `3c5a65eea891d29211f296a02baad98a5d422ee83b041c48ec4daa9d7f880855`
  and
  `d89d6db8085aabc3f8ae30231080db4e3f3c6750dc6ffe1ebc35b2dfc137510c`.
  The immutable full-text inspection artifact has SHA-256
  `5e58949ba354f0bcb0fcf98ab7f156f02b20864065321ff1f9e66b6b4cab17c2`.
  The pair is a true duplicate.
- `/rav/datakit-6854-publish-row7309-0816-v194` published the hash-bound manual
  record. `/rav/datakit-6854-verify-row7309-0817-v195` independently reread the
  source pair, semantic checkpoint, inspection artifact, manual record,
  deterministic Parquet bytes, and completion marker. The semantic-evidence,
  manual-record, and manual-Parquet SHA-256 values are
  `c5fcbb6b4da2e331c1b31c3e3065ab00ca9e829efb0df85fb5f833564d9c026f`,
  `8d4b91037429f51d06a11088509f8c3cc836ae324a14b0fb5508ed988217475f`,
  and
  `1ae054b8686c59436c31c66166bba5258cf72b8fc1e5c889de34e9480f45617e`.
- Across the stable 869-checkpoint snapshot, all 125 unresolved model outcomes
  are covered by 95 true-duplicate and 30 false-positive manual records. The
  adjusted totals are:

  - baseline: 89,433 pairs, 56,904 false positives, 32,529 true duplicates;
  - treatment: 21,016 pairs, 10,849 false positives, 10,167 true duplicates;
  - combined: 110,449 pairs, 67,753 false positives, 42,696 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T08:06:10Z — 110,321 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0804-v189` independently
  revalidated three p2 decision-file 68 baseline checkpoints at semantic
  offsets 3,712, 3,840, and 3,968. All 384 direct-review outcomes resolved as
  297 false positives and 87 true duplicates. All 795 judgments and requests
  were valid on their first attempt. The outcome Parquet SHA-256 values are
  `a216eb31f94b9143bd2ddac54570f1e1da46cfd94761a6170cd7861946989973`,
  `9a15f87ba329f72a30698bf41c58b45b0844b79fff8c3a5924353df16389e24a`,
  and
  `0de60618f41c03bc4d7c42d9ffbe125a85ef2c195f5bc24cc80ef115474219f7`.
- Across the stable 868-checkpoint snapshot, all 124 unresolved model outcomes
  remain covered by 94 true-duplicate and 30 false-positive manual records.
  The adjusted totals are:

  - baseline: 89,305 pairs, 56,825 false positives, 32,480 true duplicates;
  - treatment: 21,016 pairs, 10,849 false positives, 10,167 true duplicates;
  - combined: 110,321 pairs, 67,674 false positives, 42,647 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T08:03:33Z — 109,937 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0759-v188` independently
  revalidated six p2 decision-file 68 baseline checkpoints at semantic offsets
  2,944 through 3,584. All 768 outcomes resolved as 471 false positives and
  297 true duplicates. All 2,014 judgments and requests were valid on their
  first attempt; 761 pairs used direct review and seven used chunked review.
  The six outcome Parquet SHA-256 values, in offset order, are
  `66871d8f0686a37773051b15bf31916cd9779167a782b6238e57d1bb4124d2a0`,
  `e1d31b65ac1373bd6b8c017cbb8a410560e4ee234f2fcfd4f5baedf684830911`,
  `9208f2d1bc2f81a13ebd4a40808dd47f0cbb4f5257415ed1f9f1df2ffbb5579f`,
  `0a2834aa3600d8cecb448bf4da5d8fdf4688da194fdf6fb06f02d941a3fc48fb`,
  `28d5ad747fff53fc57f9fd82480a7aa896fbcf526971ead0b03de29deae00da7`,
  and
  `3926370bd71cf7f5a00499d85d7a3e7771fd7dff140f78cec79718a472121766`.
- Across the stable 865-checkpoint snapshot, all 124 unresolved model outcomes
  remain covered by 94 true-duplicate and 30 false-positive manual records.
  The adjusted totals are:

  - baseline: 88,921 pairs, 56,528 false positives, 32,393 true duplicates;
  - treatment: 21,016 pairs, 10,849 false positives, 10,167 true duplicates;
  - combined: 109,937 pairs, 67,377 false positives, 42,560 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T07:58:30Z — 109,169 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0748-v182` independently
  revalidated p2 decision-file 68 baseline semantic offset 2,816. Its 128
  direct-review pairs contain 61 model false positives, 66 model true
  duplicates, and one unresolved outcome. All 277 judgments and requests were
  valid on their first attempt. The outcome Parquet SHA-256 is
  `8a0b13d7070d4b1247b43109ecd6f9e73383ea9968ecf3f416556796f59e0f4c`.
- `/rav/datakit-6854-inspect-row4861-0749-v183` read both complete same-source
  college-SEO texts and all three model judgments. The 852-character member
  and 827-character canonical have SHA-256
  `ada82d4d5907b36a978b78c61c0d9b9bc5a51e06edabe64957c8c995947e10a3`
  and
  `4893ce3e5b496530e4b22e31e43f7505e575b1b5c377c62acb8ae20ccb5c1c4e`.
  Their complete character and line sequence ratios are 0.582490 and 0. The
  texts share admissions and course-selection boilerplate, but the member
  uniquely contains career-path advice, Allen degree-online and Garrett/Wilson
  program references, a biomedical-engineering bachelor's-degree requirement
  with mechanical-design and surgical-practice electives, and a Rhode Island
  plant-pathology reference. The canonical instead names Tarleton State,
  College of New Rochelle, Wilbur Wright, and Southwestern CNA and contains
  distinct employment advice. Neither contains the other, so deletion loses
  distinct propositions and the pair is a false positive. This agrees with
  the prior full-text adjudication of row 5,848 against the same canonical.
- `/rav/datakit-6854-publish-row4861-0755-v186` published the hash-bound manual
  record. `/rav/datakit-6854-verify-row4861-0757-v187` separately reread the
  source pair, semantic checkpoint, manual row, deterministic Parquet bytes,
  and completion marker. The semantic-evidence, manual-record, and
  manual-Parquet SHA-256 values are
  `07c71f5dee17460872fc0f5c18bbe5318ce04f16cedcbf42f1fbdeca18479497`,
  `3562b6ce7c7d17111ccd4bfb33c5a9010417031290076568468e455e4342cb82`,
  and
  `72af6cf0e65def84b10e5506708122131f453aab4cee7a7a13322efc8e3fb158`.
- Across the stable 859-checkpoint snapshot, all 124 unresolved model outcomes
  remain covered by 94 true-duplicate and 30 false-positive manual records.
  The adjusted totals are:

  - baseline: 88,153 pairs, 56,057 false positives, 32,096 true duplicates;
  - treatment: 21,016 pairs, 10,849 false positives, 10,167 true duplicates;
  - combined: 109,169 pairs, 66,906 false positives, 42,263 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T07:46:32Z — 109,041 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0745-v181` independently
  revalidated p2 decision-file 68 baseline semantic offset 2,688. All 128
  pairs resolved as 66 false positives and 62 true duplicates. All 332
  judgments and requests were valid on their first attempt; 127 pairs used
  direct review and one used chunked review. The outcome Parquet SHA-256 is
  `bf69163f41efabe225dda481a5e620d4d19f9f400d47b558bba4e76d05a686b3`.
- Across the stable 858-checkpoint snapshot, all 123 unresolved model outcomes
  remain covered by 94 true-duplicate and 29 false-positive manual records.
  The adjusted totals are:

  - baseline: 88,025 pairs, 55,995 false positives, 32,030 true duplicates;
  - treatment: 21,016 pairs, 10,849 false positives, 10,167 true duplicates;
  - combined: 109,041 pairs, 66,844 false positives, 42,197 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T07:44:42Z — 108,913 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0743-v180` independently
  revalidated p2 decision-file 68 baseline semantic offset 2,560. All 128
  direct-review pairs resolved as 62 false positives and 66 true duplicates.
  All 281 judgments and requests were valid on their first attempt. The
  outcome Parquet SHA-256 is
  `5dcdf2e6cc58be8f37a00e8471de28a9f298b16e34c5c0b8aeb27c577396a349`.
- Across the stable 857-checkpoint snapshot, all 123 unresolved model outcomes
  remain covered by 94 true-duplicate and 29 false-positive manual records.
  The adjusted totals are:

  - baseline: 87,897 pairs, 55,929 false positives, 31,968 true duplicates;
  - treatment: 21,016 pairs, 10,849 false positives, 10,167 true duplicates;
  - combined: 108,913 pairs, 66,778 false positives, 42,135 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T07:42:46Z — 108,785 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0741-v179` independently
  revalidated p2 decision-file 68 baseline semantic offset 2,432. All 128
  direct-review pairs resolved as 57 false positives and 71 true duplicates.
  All 276 judgments and requests were valid on their first attempt. The
  outcome Parquet SHA-256 is
  `4b9f569c8f070611a85f2b30dd1749cf8bc4ed628fee3870e7afd5e99f2aa86c`.
- Across the stable 856-checkpoint snapshot, all 123 unresolved model outcomes
  remain covered by 94 true-duplicate and 29 false-positive manual records.
  The adjusted totals are:

  - baseline: 87,769 pairs, 55,867 false positives, 31,902 true duplicates;
  - treatment: 21,016 pairs, 10,849 false positives, 10,167 true duplicates;
  - combined: 108,785 pairs, 66,716 false positives, 42,069 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T07:40:59Z — 108,657 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0740-v178` independently
  revalidated p2 decision-file 68 baseline semantic offset 2,304. All 128
  pairs resolved as 61 false positives and 67 true duplicates. All 383
  judgments and requests were valid on their first attempt; 126 pairs used
  direct review and two used chunked review. The outcome Parquet SHA-256 is
  `73c2536c14f6d8777938a471cd491b89aefc687b23203422b1c00f1efa3e88f4`.
- Across the stable 855-checkpoint snapshot, all 123 unresolved model outcomes
  remain covered by 94 true-duplicate and 29 false-positive manual records.
  The adjusted totals are:

  - baseline: 87,641 pairs, 55,810 false positives, 31,831 true duplicates;
  - treatment: 21,016 pairs, 10,849 false positives, 10,167 true duplicates;
  - combined: 108,657 pairs, 66,659 false positives, 41,998 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T07:39:15Z — 108,529 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0738-v177` independently
  revalidated p2 decision-file 68 baseline semantic offset 2,176. All 128
  direct-review pairs resolved as 49 false positives and 79 true duplicates.
  All 268 judgments and requests were valid on their first attempt. The
  outcome Parquet SHA-256 is
  `513579aff363147bd7334e7be94d37d57ffea62e87d70b6bdac077ff84c5cb66`.
- Across the stable 854-checkpoint snapshot, all 123 unresolved model outcomes
  remain covered by 94 true-duplicate and 29 false-positive manual records.
  The adjusted totals are:

  - baseline: 87,513 pairs, 55,749 false positives, 31,764 true duplicates;
  - treatment: 21,016 pairs, 10,849 false positives, 10,167 true duplicates;
  - combined: 108,529 pairs, 66,598 false positives, 41,931 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T07:37:33Z — 108,401 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0736-v176` independently
  revalidated p2 decision-file 68 baseline semantic offset 2,048. All 128
  direct-review pairs resolved as 63 false positives and 65 true duplicates.
  All 267 judgments and requests were valid on their first attempt. The
  outcome Parquet SHA-256 is
  `f93922af365ed27e2822f3060f6ad40ff5ca8909dfed23e460894c1d4376b8f6`.
- Across the stable 853-checkpoint snapshot, all 123 unresolved model outcomes
  remain covered by 94 true-duplicate and 29 false-positive manual records.
  The adjusted totals are:

  - baseline: 87,385 pairs, 55,700 false positives, 31,685 true duplicates;
  - treatment: 21,016 pairs, 10,849 false positives, 10,167 true duplicates;
  - combined: 108,401 pairs, 66,549 false positives, 41,852 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T07:35:41Z — 108,273 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0733-v175` independently
  revalidated three baseline checkpoints:

  - p1 decision-file 36 offset 0: 128 pairs, 122 false positives and six true
    duplicates, outcome SHA-256
    `36a949703aca3553bd017897973be83f7e67422cbdf427c0c115cb27da90d65e`;
  - p2 decision-file 68 offset 1,792: 128 pairs, outcome SHA-256
    `300b37cea8abfa33dc60540de3cd71fa891ecba3557b625bf2f4e3de2f375d7b`;
  - p2 decision-file 68 offset 1,920: 128 pairs, outcome SHA-256
    `9d19d7e98b007877b6f7fa81eba1ca8ff9def383631c32e488c9a65e13c07fbd`.

- The two p2 checkpoints contain 107 false positives and 149 true duplicates.
  Across all three checkpoints, all 384 pairs resolved as 229 false positives
  and 155 true duplicates. All 3,179 judgments and requests were valid on their
  first attempt; 351 pairs used direct review and 33 used chunked review.
- Across the stable 852-checkpoint snapshot, all 123 unresolved model outcomes
  remain covered by 94 true-duplicate and 29 false-positive manual records.
  The adjusted totals are:

  - baseline: 87,257 pairs, 55,637 false positives, 31,620 true duplicates;
  - treatment: 21,016 pairs, 10,849 false positives, 10,167 true duplicates;
  - combined: 108,273 pairs, 66,486 false positives, 41,787 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T07:32:35Z — 107,889 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0731-v174` independently
  revalidated two p2 decision-file 68 baseline checkpoints at semantic offsets
  1,536 and 1,664. All 256 direct-review pairs resolved as 134 false positives
  and 122 true duplicates. All 528 judgments and requests were valid on their
  first attempt. The outcome Parquet SHA-256 values are
  `bd91f5e9c57f01cea8f3eb9639657663426e4ab5b7847bacca528488e339ec7a`
  and
  `70c619d1345205ad8b0ced4ceefe4f6e00ed0cc3a8d5f3fc70311cb97cf621a8`.
- Across the stable 849-checkpoint snapshot, all 123 unresolved model outcomes
  remain covered by 94 true-duplicate and 29 false-positive manual records.
  The adjusted totals are:

  - baseline: 86,873 pairs, 55,408 false positives, 31,465 true duplicates;
  - treatment: 21,016 pairs, 10,849 false positives, 10,167 true duplicates;
  - combined: 107,889 pairs, 66,257 false positives, 41,632 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T07:30:52Z — 107,633 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0729-v173` independently
  revalidated p2 decision-file 68 baseline semantic offset 1,408. All 128
  direct-review pairs resolved as 79 false positives and 49 true duplicates.
  All 265 judgments and requests were valid on their first attempt. The
  outcome Parquet SHA-256 is
  `08f2533d3094ba0a4cf8ba68efbfef30c3c7dc8ca99c70183429fdf1ac037ff2`.
- Across the stable 847-checkpoint snapshot, all 123 unresolved model outcomes
  remain covered by 94 true-duplicate and 29 false-positive manual records.
  The adjusted totals are:

  - baseline: 86,617 pairs, 55,274 false positives, 31,343 true duplicates;
  - treatment: 21,016 pairs, 10,849 false positives, 10,167 true duplicates;
  - combined: 107,633 pairs, 66,123 false positives, 41,510 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T07:28:49Z — 107,505 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0727-v172` independently
  revalidated p2 decision-file 68 baseline semantic offset 1,280. All 128
  direct-review pairs resolved as 116 false positives and 12 true duplicates.
  All 266 judgments and requests were valid on their first attempt. The
  outcome Parquet SHA-256 is
  `b29d3d95834a8230980fdee1324d24614f7b83cd41bdb1ae2190f334e1d395dc`.
- Across the stable 846-checkpoint snapshot, all 123 unresolved model outcomes
  remain covered by 94 true-duplicate and 29 false-positive manual records.
  The adjusted totals are:

  - baseline: 86,489 pairs, 55,195 false positives, 31,294 true duplicates;
  - treatment: 21,016 pairs, 10,849 false positives, 10,167 true duplicates;
  - combined: 107,505 pairs, 66,044 false positives, 41,461 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T07:26:56Z — 107,377 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0725-v171` independently
  revalidated two p2 decision-file 68 baseline checkpoints at semantic offsets
  1,024 and 1,152. All 256 direct-review pairs resolved as 236 false positives
  and 20 true duplicates. All 530 judgments and requests were valid on their
  first attempt. The outcome Parquet SHA-256 values are
  `a078de8d08aac60be677eb7e9b3a892ed2426c3053689f36d26fc7760b7974fa`
  and
  `e4e60a08d4feedae1d5752b706d4ca06205912f144f95ca2c6fe5cdd4cbb9ab8`.
- Across the stable 845-checkpoint snapshot, all 123 unresolved model outcomes
  remain covered by 94 true-duplicate and 29 false-positive manual records.
  The adjusted totals are:

  - baseline: 86,361 pairs, 55,079 false positives, 31,282 true duplicates;
  - treatment: 21,016 pairs, 10,849 false positives, 10,167 true duplicates;
  - combined: 107,377 pairs, 65,928 false positives, 41,449 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T07:25:02Z — 107,121 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0723-v170` independently
  revalidated p2 decision-file 68 baseline semantic offset 896. All 128
  direct-review pairs resolved as 83 false positives and 45 true duplicates.
  All 262 judgments and requests were valid on their first attempt. The
  outcome Parquet SHA-256 is
  `776bd21a6cf2e381f37387fc86eaa622c439f4ac6b033257c297a15c0a67055e`.
- Across the stable 843-checkpoint snapshot, all 123 unresolved model outcomes
  remain covered by 94 true-duplicate and 29 false-positive manual records.
  The adjusted totals are:

  - baseline: 86,105 pairs, 54,843 false positives, 31,262 true duplicates;
  - treatment: 21,016 pairs, 10,849 false positives, 10,167 true duplicates;
  - combined: 107,121 pairs, 65,692 false positives, 41,429 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T07:23:07Z — 106,993 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0721-v169` independently
  revalidated p2 decision-file 68 baseline semantic offset 768. All 128
  direct-review pairs resolved as 91 false positives and 37 true duplicates.
  All 266 judgments and requests were valid on their first attempt. The
  outcome Parquet SHA-256 is
  `de040bdd9353e17204ab3b33e4f69d49bc56b196a9b911b733d9e3e49e68e4f1`.
- Across the stable 842-checkpoint snapshot, all 123 unresolved model outcomes
  remain covered by 94 true-duplicate and 29 false-positive manual records.
  The adjusted totals are:

  - baseline: 85,977 pairs, 54,760 false positives, 31,217 true duplicates;
  - treatment: 21,016 pairs, 10,849 false positives, 10,167 true duplicates;
  - combined: 106,993 pairs, 65,609 false positives, 41,384 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T07:21:03Z — 106,865 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0719-v168` independently
  revalidated four p2 decision-file 68 baseline checkpoints at semantic
  offsets 256 through 640. All 512 pairs resolved as 337 false positives and
  175 true duplicates. All 2,274 judgments and requests were valid on their
  first attempt; 500 pairs used direct review and 12 used chunked review. The
  outcome Parquet SHA-256 values are
  `7d2d4977e479566969246eab0d895d9a84c56ef7ea0bf0553aa0a31dd1d54996`,
  `a014807c17f2bf3ec274025f77b5195a070655dd96b1d439b65dc1d4a2881e4e`,
  `fe86a32232a48a902f36ff8700879a8dd7955c2d649d78ecfd8709f7df7b9f1f`,
  and
  `02c3b185d809fb7ffac4eb41c5ad080333a45596ce3085adeb059c3eee32e21d`.
- Across the stable 841-checkpoint snapshot, all 123 unresolved model outcomes
  remain covered by 94 true-duplicate and 29 false-positive manual records.
  The adjusted totals are:

  - baseline: 85,849 pairs, 54,669 false positives, 31,180 true duplicates;
  - treatment: 21,016 pairs, 10,849 false positives, 10,167 true duplicates;
  - combined: 106,865 pairs, 65,518 false positives, 41,347 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T07:18:37Z — 106,353 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0710-v164` independently
  revalidated the final p3 decision-file 100 treatment checkpoint at semantic
  offset 5,888. Its 75 pairs contain 34 model false positives, 39 model true
  duplicates, and two unresolved outcomes. The checkpoint contains 167 valid
  responses across 176 requests; nine invalid responses affected three
  retried judgments. Seventy-four pairs used direct review and one used
  chunked review. The outcome Parquet SHA-256 is
  `7776601afd4be4a73b4a5e3bc0f158d2ae3d38b9c56012573760595a580e1b14`.
- Complete source-document comparison resolves both ambiguities as true
  duplicates:

  - row 9,138 has 10,941 / 10,948 characters and 206 / 206 lines. Its only
    changed span is `\boxed{\text{D}}` versus `\boxed{D}`; character
    similarity is 0.999680 and line similarity is 0.995146. Its
    semantic-evidence SHA-256 is
    `dcb691138d622559abbf991925a70419b6b0a597832c96006f47969582a28950`.
  - row 9,167 has 4,762 / 4,755 characters and 78 / 78 lines. Its only changed
    span is `\boxed{B}` versus `\boxed{\text{B}}`; character similarity is
    0.999264 and line similarity is 0.987179. Its semantic-evidence SHA-256 is
    `f118a75e07e0ac29113437ae72355c7e23cafeacaaa270a8f0ad3114fec434f0`.

- The hash-bound manual records have Parquet SHA-256 values
  `44599f78e21ccdadab00f0357a56efd1a737b4efcd93fddbf63af102fa6bdae1`
  and
  `c51620cc257c963dacae18d824b9c7867cdcbbac4824855c0763cab9d4f4428a`.
  `/rav/datakit-6854-verify-manual-0715-v167` separately reread and exactly
  checked both source cases, the semantic checkpoint, judgment hashes, manual
  rows, Parquet bytes, and completion markers.
- Across the stable 837-checkpoint snapshot, all 123 unresolved model outcomes
  remain covered by 94 true-duplicate and 29 false-positive manual records.
  The adjusted totals are:

  - baseline: 85,337 pairs, 54,332 false positives, 31,005 true duplicates;
  - treatment: 21,016 pairs, 10,849 false positives, 10,167 true duplicates;
  - combined: 106,353 pairs, 65,181 false positives, 41,172 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T07:09:45Z — 106,278 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0708-v163` independently
  revalidated two p3 decision-file 100 treatment checkpoints at semantic
  offsets 5,632 and 5,760. All 256 direct-review pairs resolved: 157 false
  positives and 99 true duplicates. All 525 judgments and requests were valid
  on their first attempt, with no ambiguous outcomes. The outcome Parquet
  SHA-256 values are
  `06dcc623e7b8ba7eade50332ea78ff94b3c62887a33c34edd0bb5ce673740f2d`
  and
  `62c50c2b6e2bd9f7920bcfda2197078c59a3fee0f65236b844ae62725aaeb68e`.
- Across the stable 836-checkpoint snapshot, all 121 unresolved model outcomes
  remain covered by 92 true-duplicate and 29 false-positive manual records.
  The adjusted totals are:

  - baseline: 85,337 pairs, 54,332 false positives, 31,005 true duplicates;
  - treatment: 20,941 pairs, 10,815 false positives, 10,126 true duplicates;
  - combined: 106,278 pairs, 65,147 false positives, 41,131 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T07:07:26Z — 106,022 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0704-v162` independently
  revalidated seven checkpoints containing 896 pairs: p2 decision-file 68
  offset 128 and p3 decision-file 100 offsets 4,864 through 5,504. All pairs
  resolved as 456 false positives and 440 true duplicates, with no ambiguous
  outcomes. All 2,913 judgments and requests were valid on their first attempt;
  879 pairs used direct review and 17 used chunked review.
- The baseline checkpoint contains 128 pairs: 86 false positives and 42 true
  duplicates. The six treatment checkpoints contain 768 pairs: 370 false
  positives and 398 true duplicates. The outcome Parquet SHA-256 values are
  `d3b5d79c7b9e0d1b835ed467c81bac75fce5be016e5889f681aa818bec7d6471`,
  `f6ffbdd004683b716bdd87922aa6cbd745fbba6839bf73fef504b3365c72b92a`,
  `170e63b58c61dec5fb2b859b3437d4d061efc1bb79f066a291128b60418a2fe3`,
  `d1add2ff06d7ac19df757e623a8874e6cd9add3e0bff34cde76feb10c44cb72e`,
  `dea72a902f8c68f6b2490e01c31ad3d75f0f10d7816bc9fbd7c6cdae74288537`,
  `2756c699082b49be5b16155911e42bc46a93ac69c9123b663fd8af162157e681`,
  and
  `405f52d14bf0dfba3cc9d31760d6270a10b66fec2093c57efe70afe0aeeb59d5`.
- Across the stable 834-checkpoint snapshot, all 121 unresolved model outcomes
  remain covered by 92 true-duplicate and 29 false-positive manual records.
  The adjusted totals are:

  - baseline: 85,337 pairs, 54,332 false positives, 31,005 true duplicates;
  - treatment: 20,685 pairs, 10,658 false positives, 10,027 true duplicates;
  - combined: 106,022 pairs, 64,990 false positives, 41,032 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T07:03:53Z — 105,126 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0657-v158` independently
  revalidated three p3 decision-file 100 checkpoints at semantic offsets 4,480
  through 4,736. Their 384 pairs contain 288 false positives, 94 true
  duplicates, and two unresolved outcomes. The 831 judgments used 839
  requests: 828 valid and 11 invalid attempts, with five judgments retried.
  There were 383 direct pairs and one chunked pair. The batch includes 255
  baseline and 129 treatment pairs; both unresolved outcomes are baseline. The
  outcome Parquet SHA-256 values are
  `cd305e4db6e590c3a3d3fb03eb6b9ddb609f0278bdb4e8d4559940e5615fa7c1`,
  `2c1d8c3a756a14ae1438613fb986f7c8b1e266cf8ee667bf9eac29ef16e4cc8d`,
  and
  `78664e686e13250d6a8f062f387886de2ce2d2474c1a40239cb7194815e3153d`.
- `/rav/datakit-6854-inspect-rows7691-7692-0659-v159` read both complete
  cross-source SFT pairs and every persisted model attempt. Exhaustive diffs
  across all 232 lines found one seven-character LaTeX formatting difference
  in each pair and no other changed content:

  - row 7,691 has 10,122- and 10,115-character texts with member/canonical
    SHA-256
    `ba9f41d5984393a28b6a684e9bc71a47b0a2f848bbee00a14a0d93fe6d5561e5` /
    `4537ecf91a9dd859e66ff0f86011a2a92b968f74cec321b4a5b28520e277d3a8`.
    The final answer changes only from `\boxed{C}` to `\boxed{\text{C}}`;
    character and line sequence ratios are 0.999654 and 0.993333.
  - row 7,692 has 6,115- and 6,108-character texts with member/canonical
    SHA-256
    `05ee4cc26e2ebbc6302bd8c61ea02d6955c4f3c969a08887d0ae80c3e8e88ef0` /
    `ba441d6e7e6874ad81a38484bb8a7e3921171b3443203fcb1b2d7d202af87ee4`.
    The final answer changes only from `\boxed{H}` to `\boxed{\text{H}}`;
    character and line sequence ratios are 0.999427 and 0.987805.

  Both are true duplicates because every question, choice, reasoning passage,
  factual claim, and answer is otherwise represented.
- `/rav/datakit-6854-publish-manual-0702-v160` published both hash-bound
  true-duplicate records, and `/rav/datakit-6854-verify-manual-0703-v161`
  separately reread the source pairs, semantic checkpoint, manual rows,
  Parquet bytes, and completion markers. Their semantic-evidence SHA-256
  values are
  `97d99071c3d9210b927009f5cf0c9a12037b4a23155bd2c73cb6c5ab8495d1fd`
  and
  `d2dbca74e26472d105ae24317cb610a99bb2903d9bd511f383ecd7eb85357e62`;
  their manual-record SHA-256 values are
  `284528bd55b8ce091320caa8734b1484f37f40995433e7640c6a7819aa3555a0`
  and
  `931500140897fd7d5be97bbe0280503c06f36cda4d4bd6f70293d4cc06847349`;
  their manual-Parquet SHA-256 values are
  `0b8a8936f8ae81f70fc7065fc4940b14524fb28b9a69a2e0b516e591228b7e49`
  and
  `c3f261f86e722c5ce99a223841b61634e7c73775dc3db8324e9a053818a7ed64`.
- Across the stable 827-checkpoint snapshot, all 121 unresolved model outcomes
  are covered by 92 true-duplicate and 29 false-positive manual records. The
  adjusted totals are:

  - baseline: 85,209 pairs, 54,246 false positives, 30,963 true duplicates;
  - treatment: 19,917 pairs, 10,288 false positives, 9,629 true duplicates;
  - combined: 105,126 pairs, 64,534 false positives, 40,592 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T06:56:32Z — 104,742 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0650-v154` independently
  revalidated five p3 decision-file 100 checkpoints at semantic offsets 3,840
  through 4,352. Their 640 baseline pairs contain 462 false positives, 176
  true duplicates, and two unresolved outcomes. The 1,374 judgments used 1,382
  requests: 1,371 valid and 11 invalid attempts, with four judgments retried.
  There were 638 direct pairs and two chunked pairs. The outcome Parquet
  SHA-256 values are
  `3464431b49f0f7708cafeb58c767f61233d40e2012d7833d60ea71881c1914d7`,
  `4b7e4de4863c05b04ad51feeb870fb5c1debc98876f9ef34efdb8857dfcdca32`,
  `0c8ba34a2f6bfbdae8b7ded43422923b7037934ecbbdd2e7adcb2e94e3775906`,
  `ee55308333afab9c96c3d10dea1e000bfff5cc0cce19f4e75057a56f23ea1083`,
  and
  `2e8742150df1be3339f0c950cfa84ec96535becac6259f1a2ea0e0e4eb1ab77d`.
- `/rav/datakit-6854-inspect-rows7452-7502-0652-v155` read both complete
  ambiguous SFT pairs and every persisted model attempt:

  - Cross-source row 7,452 has 15,053- and 15,060-character texts with
    member/canonical SHA-256
    `1b0fea5fcdd2f02b0084aeb06722ea31497c89743e1e065378b9d181dd931df6` /
    `2a3a107b86a63fc451c0ec742d7c822a4c4b8bbe6bc4aec437a5d73226c6332c`.
    Exhaustive unified diff found one seven-character formatting difference:
    the final answer is `\boxed{C}` instead of `\boxed{\text{C}}`. All 352
    lines of the question, choices, reasoning, factual claims, and answer are
    otherwise represented, so this is a true duplicate. Character and line
    sequence ratios are 0.999768 and 0.997159.
  - Same-source row 7,502 has 424- and 485-character texts with
    member/canonical SHA-256
    `c0703338ee4140d288c9a51a50ace2d01246dc31555e72af67ad0a41f027ce59` /
    `987b60cb2ac229ddf85800ae1271c587ef6e72ae2d9c9078fd832a86c4c23c24`.
    The common text is only the response-format instruction. The member asks
    to simplify square roots and answers -21; the canonical asks for 54 times
    46 and derives 2484. These are distinct training examples, so the pair is
    a false positive.

- `/rav/datakit-6854-publish-manual-0655-v156` published both hash-bound
  manual records, and `/rav/datakit-6854-verify-manual-0656-v157` separately
  reread the source pairs, semantic checkpoint, manual rows, Parquet bytes,
  and completion markers. Their semantic-evidence SHA-256 values are
  `1bfc2f6fc03ccc2ef78491520b1bc0e126d4fcbce92350c67b05e77c5f508e2d`
  and
  `e88f82c1f67209059c48cf49aecfc60dd6ceba27b8fadf76faf34e89b377d0e4`;
  their manual-record SHA-256 values are
  `d717236c01f2631b2586d17fb83a3b9b7558554745f5f2978da94f12809563a3`
  and
  `d7c271429cc564b6c533c4e3950c76e03fd933320e2a019d535cfe75da22ebe4`;
  their manual-Parquet SHA-256 values are
  `c2f150c58263946bd830b5ff49760a8eec066af1808fcc1725f77ab6833b4f7d`
  and
  `bfd2df4a410432d6030eaa3a89ba5c993f8c092a8607bb950de7ada3ed986419`.
- Across the stable 824-checkpoint snapshot, all 119 unresolved model outcomes
  are covered by 90 true-duplicate and 29 false-positive manual records. The
  adjusted totals are:

  - baseline: 84,954 pairs, 54,030 false positives, 30,924 true duplicates;
  - treatment: 19,788 pairs, 10,216 false positives, 9,572 true duplicates;
  - combined: 104,742 pairs, 64,246 false positives, 40,496 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T06:49:25Z — 104,102 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0642-v150` independently
  revalidated three p3 decision-file 100 checkpoints at semantic offsets 3,456
  through 3,712. Their 384 baseline pairs contain 269 false positives, 114 true
  duplicates, and one unresolved outcome. All 937 judgments and requests were
  valid on their first attempt; 382 pairs used direct review and two used
  chunked review. The outcome Parquet SHA-256 values are
  `b736146772891e5418c3f889374e1305225d78ebcce224709b277c00f25b9eff`,
  `816c15a99eace15b8317e84321eb1e11e20823d8f2ca205b0295137490afe417`,
  and
  `b4ec9986d58a8259df9f64cd637443b21be0b1a70392f847855da377ce12622b`.
- `/rav/datakit-6854-inspect-row5848-0644-v151` read both complete
  same-source college-SEO texts and all three model judgments. The
  1,345-character member and 827-character canonical have SHA-256
  `f2644d391643dd08210c76c2332402af01a2cc7578d913577036d9550d586437`
  and
  `4893ce3e5b496530e4b22e31e43f7505e575b1b5c377c62acb8ae20ccb5c1c4e`,
  with character and line sequence ratios of 0.580110 and 0.250000.
  Institution, location, and program names are low-value slot substitutions,
  but the member also adds distinct propositions absent from the canonical:
  a biomedical-engineering degree requirement involving biotechnology or
  biology and mechanical-design and surgical-practice electives, plus
  separate corporate-internship advice. Deleting those claims loses content,
  so the pair is a false positive.
- `/rav/datakit-6854-publish-manual-0647-v152` published the hash-bound manual
  record, and `/rav/datakit-6854-verify-manual-0648-v153` separately reread
  the source pair, semantic checkpoint, manual row, Parquet bytes, and
  completion marker. The semantic-evidence, manual-record, and manual-Parquet
  SHA-256 values are
  `6210b11bceee642342436712cce88b3482abac5a50754b1083d9fcf2b787f957`,
  `3acd3cbc58328a803aad195d014f979ee1a5ebd9be8063ac048de4f300982ee7`,
  and
  `9ae2ad4313870f2598d120ebe58f022506f720cbc037b4ad28d7360b371797ac`.
- Across the stable 819-checkpoint snapshot, all 117 unresolved model outcomes
  are covered by 89 true-duplicate and 28 false-positive manual records. The
  adjusted totals are:

  - baseline: 84,314 pairs, 53,567 false positives, 30,747 true duplicates;
  - treatment: 19,788 pairs, 10,216 false positives, 9,572 true duplicates;
  - combined: 104,102 pairs, 63,783 false positives, 40,319 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T06:41:20Z — 103,718 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0640-v149` independently
  revalidated six p3 decision-file 100 checkpoints at semantic offsets 2,688
  through 3,328. All 768 baseline pairs resolved: 385 false positives and 383
  true duplicates. All 1,687 judgments and requests were valid on their first
  attempt; 767 pairs used direct review and one used chunked review.
- The outcome Parquet SHA-256 values are
  `5c35a882793cf060c3af16e3c764c61accd61a51b9cc7b34774294f8d8dab858`,
  `552f6c5db08ba4ed7f2f9c1571339b47856172a1f82845b4ec1b51a128d2d2e0`,
  `644343a5b9df4b922bb767d76f0f673f4611e80f9d8ec3000fc7d30898297711`,
  `deb383d4018c2995c27427a0f6290938a8a0329379777c5e8b762354dd420814`,
  `0d3bb5b67c1c91660cccdd861bc80033fb0a14e32c10bdfc2d456f21c8c91e71`,
  and
  `0a0d7f6dfbc359fbb7b9319ec7cbe75a36a6e567f570391023a2bd81cb126d13`.
- Across the stable 816-checkpoint snapshot, all 116 unresolved model outcomes
  remain covered by 89 true-duplicate and 27 false-positive manual records.
  The adjusted totals are:

  - baseline: 83,930 pairs, 53,297 false positives, 30,633 true duplicates;
  - treatment: 19,788 pairs, 10,216 false positives, 9,572 true duplicates;
  - combined: 103,718 pairs, 63,513 false positives, 40,205 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T06:37:41Z — 102,950 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0630-v145` independently
  revalidated seven additional baseline checkpoints: p2 decision-file 68
  offset 0 and p3 decision-file 100 offsets 1,920 through 2,560. Their 896
  pairs contain 488 false positives, 407 true duplicates, and one unresolved
  outcome. All 4,442 judgments and requests were valid on their first attempt;
  863 pairs used direct review and 33 used chunked review.
- The outcome Parquet SHA-256 values are
  `7ac05bec11162e464249c52258a5abcdbb33eb42e5099952a3263cfdfd3296ce`,
  `25811092f6b6f4c073c5be6ceb33b292597eac8fefff22bc90e13c05817a13c4`,
  `84f2ea570062e9dc52204b5abc765aad6b7ee77921920553e754cbc1603aa040`,
  `a3e58e61ed64dd8a18c6d0c225dab72db95dece046abd354e8382cfb72d1e1c9`,
  `6ae5e081edb4fefa71c8ddae50ec34715b9b95833e190c5c462f22a4d4fa3f9e`,
  `58f0f31c9a78ad492df033b12773572066d3e35f1067c1e2206ce7eb7d1195cc`,
  and
  `a82e987b2a24b0cfeb40d54a84cf79508bc93c36731e74520abf23efde557ee9`.
- `/rav/datakit-6854-inspect-row3107-0634-v146` compared every character of
  the unresolved same-source synthetic wiring-diagram pair. The
  2,297-character member and 2,137-character canonical have SHA-256
  `b6609c07b5f79d98bd3a1cd57328d08690e5c7240dedba1d8f3dd91fc5b11f1e`
  and
  `81f59fa98b48c305a03dc58eb42847dc845ca70192e8f680e76a010b261c1eab`,
  with a 0.698241 sequence ratio. Beyond the repeated synthetic scaffold and
  Rib Relay/Chopper title substitution, the member alone says diagram
  component order is relative rather than logical and places the negative
  supply symbol below the line. Deleting those circuit-drawing instructions
  loses substantive content, so the pair is a false positive.
- `/rav/datakit-6854-publish-manual-0635-v147` published the hash-bound manual
  record, and `/rav/datakit-6854-verify-manual-0637-v148` separately reread the
  source pair, semantic checkpoint, manual row, Parquet bytes, and completion
  marker. The semantic-evidence, manual-record, and manual-Parquet SHA-256
  values are
  `21cd2e521b4d91088e8fe49b1ace377afdcf1c7135077fdf8649c60228e565fb`,
  `19a02ef6d0cc2073c6f97552c0a4afd1d2839b4726c3bda90fe5a7df1efc10ee`,
  and
  `387b45e70d88a4a2a72c74f5a511182a93f1c3548dabaae1dccc403cc9a7b6ad`.
- Across the stable 810-checkpoint snapshot, all 116 unresolved model outcomes
  are covered by 89 true-duplicate and 27 false-positive manual records. The
  adjusted totals are:

  - baseline: 83,162 pairs, 52,912 false positives, 30,250 true duplicates;
  - treatment: 19,788 pairs, 10,216 false positives, 9,572 true duplicates;
  - combined: 102,950 pairs, 63,128 false positives, 39,822 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T06:30:01Z — 102,054 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0616-v137` independently
  revalidated two p3 decision-file 100 checkpoints at semantic offsets 1,664
  and 1,792. The 256 baseline pairs contain 122 false positives, 132 true
  duplicates, and two unresolved outcomes. All 534 direct-review requests were
  valid on their first attempt. The outcome Parquet SHA-256 values are
  `8288321d5f57770c4bae0a3151566ddcb0c1425ba2236e4e7a5d76e86ded9b7d`
  and
  `6298b834a28c7640187888befdbd0808479e5adc108db26ff6a8dd36cb775ecb`.
- `/rav/datakit-6854-inspect-row2772-0622-v140` compared every character in
  the first unresolved cross-source gardening pair. The 2,883-character member
  and 3,381-character canonical have SHA-256
  `d28d9108bdc017220bf0b3ccbe70c6ed4440f2671e3b03771d9b10e18c907b41`
  and
  `254364d1470b663c667e6462a702ce96b67e817943653341c94737ac33ac7f56`,
  with a 0.884100 sequence ratio. In addition to its title, the member alone
  instructs readers to dilute worm-compost tea five-to-one. That substantive
  gardening guidance is absent from the canonical, so this is a false
  positive.
- `/rav/datakit-6854-inspect-row2787-0624-v141` compared every character in
  the second unresolved same-source SEO pair. The 2,289-character member and
  2,249-character canonical have SHA-256
  `960968ff987aa2e3939003b93967055e218e6058279cc67cb19b6c143fc6cd63`
  and
  `c36eab9dec05adae252d5b44c915216dfc5f9419f25f1dd8d0c2bc0beac1b127`,
  with a 0.706479 sequence ratio. URL and keyword substitutions are
  non-substantive under the low-value-template boundary, but the member alone
  states that the FDA estimates 50% of generic-drug production is by
  companies. Deleting that fact loses substantive content, so this is also a
  false positive.
- `/rav/datakit-6854-publish-manual-0627-v143` published both hash-bound manual
  records, and `/rav/datakit-6854-verify-manual-0629-v144` separately reread
  the source pairs, semantic checkpoint, manual rows, Parquet bytes, and
  completion markers. The semantic-evidence SHA-256 values are
  `04d6b663cb9fdc88996bd45e693b446eae040af556f831e4c11675186e95f6d7`
  and
  `9452ecca57fd22b73ed238bca0380016f4cef14222a26768617830e1dca9c039`.
  The manual Parquet SHA-256 values are
  `8f5d401b22c6a3aef9c14d17ebdb4fb239da23d094d091786e1918e46c163cbb`
  and
  `ea91f5aa1691c761e634a52d83a37fc8a8d6da4aa34f9ba435dcfc14049a4101`.
- Across the stable 803-checkpoint snapshot, all 115 unresolved model outcomes
  are covered by 89 true-duplicate and 26 false-positive manual records. The
  adjusted totals are:

  - baseline: 82,266 pairs, 52,423 false positives, 29,843 true duplicates;
  - treatment: 19,788 pairs, 10,216 false positives, 9,572 true duplicates;
  - combined: 102,054 pairs, 62,639 false positives, 39,415 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T06:13:47Z — 101,798 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0612-v136` independently
  revalidated two p3 decision-file 100 checkpoints at semantic offsets 1,408
  and 1,536. All 256 baseline pairs resolved: 189 false positives and 67 true
  duplicates. All 527 direct-review requests were valid on their first attempt.
  The outcome Parquet SHA-256 values are
  `36255c7f6678977676c95b6bc26769220aebd0de94f877f271a53bca2af5bf35`
  and
  `110656c1a541fa2f8eda8615bdb8044118ed0c3b158874fe10a2f0a5a6f05f90`.
- Across the stable 801-checkpoint snapshot, all 113 unresolved model outcomes
  remain covered by 89 true-duplicate and 24 false-positive manual records.
  The adjusted totals are:

  - baseline: 82,010 pairs, 52,299 false positives, 29,711 true duplicates;
  - treatment: 19,788 pairs, 10,216 false positives, 9,572 true duplicates;
  - combined: 101,798 pairs, 62,515 false positives, 39,283 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T06:11:54Z — 101,542 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0611-v135` independently
  revalidated p3 decision-file 100 semantic offset 1,280. All 128 baseline pairs
  resolved: 116 false positives and 12 true duplicates. All 261 direct-review
  requests were valid on their first attempt. The outcome Parquet SHA-256 is
  `b8fe447b9cb6dc095428dcbf3b8149fcecb5d4d5bdd66c4f37310ec331156ab0`.
- Across the stable 799-checkpoint snapshot, all 113 unresolved model outcomes
  remain covered by 89 true-duplicate and 24 false-positive manual records.
  The adjusted totals are:

  - baseline: 81,754 pairs, 52,110 false positives, 29,644 true duplicates;
  - treatment: 19,788 pairs, 10,216 false positives, 9,572 true duplicates;
  - combined: 101,542 pairs, 62,326 false positives, 39,216 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T06:10:09Z — 101,414 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0608-v134` independently
  revalidated four additional p3 decision-file 100 checkpoints at semantic
  offsets 768 through 1,152. All 512 baseline pairs resolved: 380 false
  positives and 132 true duplicates. The 1,120 judgments were direct for 510
  pairs and chunked for two, and all 1,120 requests were valid on their first
  attempt.
- The outcome Parquet SHA-256 values are
  `09deffa40aa9e8659ad55db1350ba5043a1814066d94f331d541e0003901b6f3`,
  `a140fae32e5415b75af6d0661e9e6211e8e3c2571a0e112b3cfc398b5e1d0aaf`,
  `8a98589476c003f43322f6ece4521d251e8abe724a165b9d032152749c8a7352`,
  and
  `1614ed31c7ec49ff2e16da7b429d1b6c374877cf30d2795db4b240d79b36167e`.
- Across the stable 798-checkpoint snapshot, all 113 unresolved model outcomes
  remain covered by 89 true-duplicate and 24 false-positive manual records.
  The adjusted totals are:

  - baseline: 81,626 pairs, 51,994 false positives, 29,632 true duplicates;
  - treatment: 19,788 pairs, 10,216 false positives, 9,572 true duplicates;
  - combined: 101,414 pairs, 62,210 false positives, 39,204 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T06:07:30Z — 100,902 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0601-v130` independently
  revalidated seven additional checkpoints containing 746 pairs: 384 baseline
  and 362 treatment. Raw labels are 480 false positives, 261 true duplicates,
  and five unresolved. The 2,259 judgments used 2,285 requests: 2,249 valid
  and 36 invalid attempts, with 15 judgments retried. Eight pairs were chunked
  and 738 were direct.
- The p0 outcome Parquet SHA-256 values are
  `1d623b02dfe1275dc15e0a3bfd8b07508bc2113044e3d294d370bdb395269232`
  and
  `c327da68a2c325ca1080495019a99e6211b43d6414384e811ba8963ea940c6ec`
  for decision-file 4 offsets 5,504 and 5,632, followed by
  `4a1ca91cdaa1aaa8ff3d9090fd8ef4c9b99cb752c59ce8052b6f58a1839412fe`
  for its final 45-pair offset 5,760. The p1 decision-file 35 final 61-pair
  offset 5,760 is
  `fda095c8a7f9a7134e9254cfefdba29ed790e29ebf108527df95dae93bac5a12`.
  The p3 decision-file 100 values for offsets 384, 512, and 640 are
  `fd81fb9e352657077471885d88b2ec565b3e27a793233e2c4fba48a124fe9fa6`,
  `9c90a20feea9d8eefa3c23fa0951feb69eb6ded6771228e84257f5877f63781b`,
  and
  `6a2d4f61d43681482546f18a4010ccc474bb9b112e010daedef70b4df7c8e13c`.
- `/rav/datakit-6854-inspect-unresolved-0604-v131` hashed and compared every
  character of all ten documents in the five unresolved cross-source SFT
  pairs. Every comparison has exactly two edit spans totaling seven
  characters: insertion or deletion of `\text{` and the matching `}` around
  the same final boxed answer. All five therefore resolve as true duplicates:

  - row 9,016: 17,796/17,803 characters across 470 lines, ratio 0.999803,
    member/canonical SHA-256
    `e21e8880ead4d637c4c06640f168214d4f15803ddc68a32f5b273b52d133ae60`
    /
    `7c69a2c509b03c85a834c3fcba56a77226b96fff73e164fb26d9ec3e3268bf86`;
  - row 9,040: 6,896/6,889 characters across 99 lines, ratio 0.999492,
    `6dd24105a5e72f5947ad63893389aa774727ba8233a2ee5d23b12a4d9cd57ed0`
    /
    `0e8e38d02e119df203133eb1e1cb0e02432bbeac0560f78b27c91672e278fcab`;
  - row 9,043: 12,825/12,818 characters across 258 lines, ratio 0.999727,
    `f0d1bf4f6f3e4977caa2f7ff3e97c00835184ec432d792f4532a8fea836ad87f`
    /
    `79eaf75685cd818c11831f696e64a3fb6328bb12e88e8e5ec74d0489363904c2`;
  - row 9,044: 5,408/5,401 characters across 64 lines, ratio 0.999352,
    `0bfb8c79f0875bfd69a3509a7463142693055fc5d0225f82a9ee2a4701b4ad3a`
    /
    `9d6350e224689518024c26a8fb3bfcddc404633a18f4fe1ad92ad70c57e5db04`;
  - row 9,047: 6,471/6,464 characters across 68 lines, ratio 0.999459,
    `ca36448949bb1cdce13ff7827866b1f42716e075e655bc4bd7d50d5b9e536d13`
    /
    `7d3df671af58301d5f4f864c5b341b64dd0eaabcde19ce8f6b5d99a825e206f6`.

- `/rav/datakit-6854-publish-manual-0606-v132` published the five hash-bound
  true-duplicate decisions, and the separate
  `/rav/datakit-6854-verify-manual-0607-v133` job reread the exact pairs,
  semantic checkpoint, records, Parquet bytes, and completion markers. Their
  semantic-evidence SHA-256 values are
  `7c1fddb98c1b8725a11854dc5e33333c407ea0a0427bda1aa109a59836ef017b`,
  `4d80255aa4cd60741f495f7a35c6d313cc0c90402d205e388a3a950dcaef6369`,
  `c90a42bb60567bf6e87f494497c66b6e2e14417e1cfda519c2c660c6b962208b`,
  `484192a33b069551fc928157f50a41fdb8cefa1ea1cb3e8c614c8b7eeb59228c`,
  and
  `3bcfb2ca76132480a54959e1f38c1ebdbd827ddf9979c0ee7209e6f22d421220`.
  Their Parquet SHA-256 values are
  `9367658b7ea7e03d03ca205d9e8a0cea2fb554a7f80dc351f9d82bfa5bd21602`,
  `5fbd0e976ba6908edf5edaf9c9fe52e2355cf32e4d71c133bc6f2e0ea19368bd`,
  `5ec81302c48da4538553aeca22056d71824ec8356c2735c0e2e53430042a7f7a`,
  `cdba7cdba792c4172c6a6213513fef57b7603621f01b2bb8961f75ee19734827`,
  and
  `5a404c341fd81da3872b26e4616beba3152c6ff8b7b47990eeb61ec8e73a2149`.
  Their record SHA-256 values are
  `2fa826e4f7bc7d740e331bab4cae5a685fa3f239c2d76c33a26b695b71209678`,
  `ef49199d2e7a738b24574f711d77a29468c1b2073ad53ea33ffff0c6b7af98eb`,
  `34aa7f70981c40a54829c2c85777c797fcf27d9478d7c2b0d3f8d7b860f9f2f8`,
  `09811219129ecd9b9b8c42260644acba35f3ef12585fe2f6cf95ed84fcff2797`,
  and
  `50de0ed36a0396e88167aa9d8e9ead737df9542984a11f6b87d296dd2fdb0474`.
- Across the stable 794-checkpoint snapshot, all 113 unresolved model outcomes
  are covered by 89 true-duplicate and 24 false-positive manual records. The
  adjusted totals are:

  - baseline: 81,114 pairs, 51,614 false positives, 29,500 true duplicates;
  - treatment: 19,788 pairs, 10,216 false positives, 9,572 true duplicates;
  - combined: 100,902 pairs, 61,830 false positives, 39,072 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T05:59:51Z — 100,156 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0549-v124` independently
  revalidated 13 additional checkpoints containing 1,664 pairs: 128 baseline
  and 1,536 treatment. Raw labels are 796 false positives, 867 true
  duplicates, and one unresolved. The 4,104 judgments used 4,107 requests:
  4,103 valid and four invalid attempts, with two judgments retried. Nine pairs
  were chunked and 1,655 were direct.
- The p0 decision-file 4 outcome Parquet SHA-256 values for offsets 4,736
  through 5,376 are
  `47c04f431891ed10ca9388f18f1020fda24b22fae7d71f2821dd6c1aa433b4e9`,
  `291c4a1f43905f7ef32219cbee69a956cc5fa51409f1784fc13bcd74cf2fefaf`,
  `a2089662738361b9edda06c0036c48c999bdfb1e77609841b35a193bff98c2bc`,
  `9b89a91d8b7337670c41610292755cbcf5fa4c38314c83205402241165aaf793`,
  `633ea30502b8500906696f1f4b364c3ed80e7dd4c4932708d09b7b8879d8e348`,
  and
  `af5632eb9d580b85b3b18dcf6ebcb58f409c56149a161a53b474be6d0ec1872a`.
  The p1 decision-file 35 values for offsets 4,992 through 5,632 are
  `6221dbbc4f943928cbaabfe5ac8ee7db700e350cd968c51e804c6e384730774a`,
  `ed28c500b9736d69c00f2d0c4252ccd876ebdaaf7862e62a421b3650dc3ff0bb`,
  `d5b83ebc72353983b188ca6d1e01b80e01b45d2efbbbb795ef852869de1a81d2`,
  `7ca4e571defa540091ccdddaf87fbdc3ad503b860e3169c44db378a1e539a113`,
  `2da486497846a8d959138160c6ef78176d12b2cd36532a69e353a7c3fcf030b1`,
  and
  `cc9497bd19b55c5e9a5e46ec917540f811577b1195e3ae0bc822a39b0574f8dc`.
  The p3 decision-file 100 value for offset 256 is
  `23b22c1b88bd4c0dfbd4bf61a24f8e29cccae4272c92bbf7506e8450a242fcc2`.
- `/rav/datakit-6854-inspect-unresolved-0551-v125` and the compact rerun
  `/rav/datakit-6854-inspect-unresolved-0554-v126` hashed and compared every
  character of the unresolved same-source math-forum pair. The member and
  canonical contain 4,135/4,195 characters across 55/69 lines, with 3,867
  matching characters and a 0.928451 sequence ratio. Their SHA-256 values are
  `e0c3c89fffbe47bd04db1ffb69c1bd39e302ac48830b57c9c8faa3fb0d3df30f`
  and
  `76ac33a4861acd0e9566aaa2d35ea8bccd8cd94e4e4d553ec7b8c72a8af3b28e`.
  The 101 edit spans are mostly Markdown, expanded dates, punctuation, and
  emoticons, but two are substantive mathematical corruption:
  `\theta_2 = \beta + \alpha_2` becomes `\theta_2 = B + A_2`, and a later
  `\theta_2 - 90` loses its subscript. Deleting the member therefore loses the
  correct formulas, so the pair is a false positive.
- `/rav/datakit-6854-publish-manual-0558-v128` published that hash-bound
  false-positive decision, and the separate
  `/rav/datakit-6854-verify-manual-0558-v129` job reread the exact pair,
  semantic checkpoint, record, Parquet bytes, and completion marker. The
  Parquet and record JSON SHA-256 values are
  `d34920d094143547e557fae271e5e95cd8dfccacd26bd7ce24aae18b265b1493`
  and
  `4fbdb707fcdbde6c1fba04d8963d1a7c3ede6de4578ab90fe77c8cb5179551d4`.
- Across the stable 787-checkpoint snapshot, all 108 unresolved model outcomes
  are covered by 84 true-duplicate and 24 false-positive manual records. The
  adjusted totals are:

  - baseline: 80,730 pairs, 51,351 false positives, 29,379 true duplicates;
  - treatment: 19,426 pairs, 9,999 false positives, 9,427 true duplicates;
  - combined: 100,156 pairs, 61,350 false positives, 38,806 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T05:46:34Z — 98,492 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0538-v120` independently
  revalidated 11 additional checkpoints containing 1,408 pairs: 908 baseline
  and 500 treatment. Raw labels are 1,008 false positives, 395 true
  duplicates, and five unresolved. The 6,156 judgments used 6,172 requests:
  6,148 valid and 24 invalid attempts, with eight judgments retried. Twenty-
  eight pairs were chunked and 1,380 were direct.
- The p0 decision-file 4 outcome Parquet SHA-256 values for offsets 4,096
  through 4,608 are
  `1a50429275d5e3d036ef7b1a852a41a78b1227a06ebf958f0d72bb5747e50f46`,
  `916f7c7c138de3760e8e38e7b190f23027cb21e49726a96ce9c99b09171b7957`,
  `0b14f494ef8656162bfe959881b95371ec00a64b11e4dd93cbe7ab80eaf260c5`,
  `36b2d19d7a58a0028b2ecfed2b99ed439dabe30a1503572fc7f31203c0f45e97`,
  and
  `3c7dc8809457598cbd87182f7b5cdae26a0dc3e1be8960d1d5200033d3fe0637`.
  The p1 decision-file 35 values for offsets 4,352 through 4,864 are
  `fc8a7bc59096d801a1860125ced8c32957ba9e18cee07033c862d30cce0dcd15`,
  `0d95bbb552f94db7a26789691c5c1678918604086cdd83fc48ef5267a49e7a47`,
  `8897b143957dd02acdce75aa2ad5689807f0028301d6b32d16707e02e95a6fae`,
  `d77a3a15658ab2629cab4e976db59bc0799911216bebce386fdfda074432d071`,
  and
  `e36a3af8d97e59287cd2a4d0d093e81e9966482b5ce122466a5dadd131b5bf52`.
  The p3 decision-file 100 value for offset 128 is
  `21c4606c7ed87fdd1ba438d25073cc76f674ffab9ffeecd451eef11e5e7a1cd9`.
- `/rav/datakit-6854-inspect-unresolved-0541-v121` hashed and compared every
  character of all ten documents in the five unresolved cross-source SFT
  pairs. Each complete unified diff contains exactly one change:
  `\boxed{X}` versus `\boxed{\text{X}}`. All five therefore resolve as true
  duplicates:

  - row 7,233, answer B, 5,446/5,453 characters, Jaccard 0.997349,
    member/canonical SHA-256
    `0a7a7ed4a240dd631a272b80b4d4e07edb7d3914f99a9e6a0fd2a3d0536b76ac`
    /
    `4c60ac94f43c746e5ad7cadaa222fe8f5e3801ecdad9a3819fc8bc41f06d1646`;
  - row 7,246, answer A, 4,959/4,966 characters, Jaccard 0.997241,
    member/canonical SHA-256
    `0f1b5f6c3c5e905ed186611db5be437e2c3d81aadd70b65b8d4ea6cd2ce44961`
    /
    `59ad7dc689ae9d27f79bc508f94880238578b265495d251ad5f1df9f8d15ff72`;
  - row 7,284, answer A, 16,045/16,052 characters, Jaccard 0.998788,
    member/canonical SHA-256
    `a0cb685bc913f1135030143edd23f22a59add9c794e64e549c8fc3c27b4f0968`
    /
    `dd318abb9474561a59b1256b022fe9739a0029ae4fd16e2458a0412cdf7c172a`;
  - row 7,302, answer G, 16,080/16,087 characters, Jaccard 0.998784,
    member/canonical SHA-256
    `5ba23c544dcf265c345938381b84c5eef735a802b7004c0117fd44cf3af18bc5`
    /
    `e0a2e1276ce057fa321cd9b638ef96ff764284a40f73b1bd20a394507054db8c`;
  - row 7,496, answer C, 4,247/4,240 characters, Jaccard 0.996931,
    member/canonical SHA-256
    `6b460727f5c516a93b04e086efa3f19c4dc8f6cdac3f6773b85d024403546606`
    /
    `a3c67d868c4633bb3a01d5e8e429b274b3204654dc94e751a90a0edb60cd5ec2`.

- `/rav/datakit-6854-publish-manual-0544-v122` published the five hash-bound
  true-duplicate shards, and the separate
  `/rav/datakit-6854-verify-manual-0545-v123` job reread and verified them.
  Their semantic-judgment SHA-256 values are
  `caf1338b4b7de220150e346e1a19f014038d11436d170fc8a1ad9fcbfb5e1291`,
  `ac50a731b707ba61564e8eadb4943461d3f301a18927f4be6b9a4572a97abbcc`,
  `7b42e23af81c4c7d1df43716af907656d3c211980ba3c6bb75b97074911ffd74`,
  `423d5d03369d62013c4bae8967947106a3439bb633e09ddcef048bffe3fe08db`,
  and
  `9417471438077c1ae2c1f7141132cc84cb13a550b0df23fc6c475f1ccef35c4c`.
  Their Parquet SHA-256 values are
  `f9a64e6c3e0d1842e38dbd429723244bb366d6cf88b541e0c091da1302a3288f`,
  `b52428807f0d7d122e41bad2dbda218e9131902e0077958f0c7fe0c53b67c468`,
  `b83d1ddd82c508ac857df74e09e5d7e2037661fe36d6a3e572fa787eb037084e`,
  `f052e1aeef1da0dc35c4e5102177fd604a51e93cd453bc81a2db733bc4d75984`,
  and
  `2de7e440b4d99cd5a68390afbc48eaa4eaff82fb8e53934f99b7a1a2ecc0590e`.
  Their record SHA-256 values are
  `cd42153b46f1baec275c0b0eced3713b39a8b2a2377b3acc1dd645e2bb5cfe4a`,
  `118104e68219086f8a1abe4d962d8f24e384b03c0a75da48719270c943577369`,
  `3e9a1033e1e43b5151a283aebc0b1d1ece6e3639b1ad79a288ced7c6a04ce216`,
  `b836949a0f0ebf7980b664871eee8a6f91cd6aedf81c22c2fab104c86e5b64cc`,
  and
  `2011d797ff7212b9ccdb0578f897f0180ded217b58948252e9e996418f43816c`.
- Across the stable 774-checkpoint snapshot, all 107 unresolved model outcomes
  are covered by 84 true-duplicate and 23 false-positive manual records. The
  adjusted totals are:

  - baseline: 80,602 pairs, 51,272 false positives, 29,330 true duplicates;
  - treatment: 17,890 pairs, 9,281 false positives, 8,609 true duplicates;
  - combined: 98,492 pairs, 60,553 false positives, 37,939 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T05:37:44Z — 97,084 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0528-v115` independently
  revalidated ten additional baseline checkpoints. Their 1,280 pairs contain
  951 false positives, 326 true duplicates, and three unresolved outcomes.
  The 2,855 judgments used 2,864 requests: 2,851 valid and 13 invalid attempts,
  with five judgments retried. Four pairs were chunked and 1,276 were direct.
- The p0 decision-file 4 outcome Parquet SHA-256 values for offsets 3,456
  through 3,968 are
  `f4040f33ed806dc9f455a1fd4084b5b0d9f77f46e2a457ec6ed8d0ae9c0b03fb`,
  `da6a0318b1cd0daf25dabee46625bdeb9ef48f3418fd1b34bf33f9a3c40d0330`,
  `94e1de86510cab7d1544fa9d24f083d02369cba139d58bc0895897b292f07f5c`,
  `0eb041526b604f092b02ad6f96658f56122bc6b0e9e1900dad9439364fc5b9a3`,
  and
  `738264771c7920798dcd3523c64aff4c1ae8720dc78ff7a0addd38c969f7e8b7`.
  The p1 decision-file 35 values for offsets 3,712 through 4,224 are
  `93ebbc73b8073ffaaffe43feef1f0ff7e4b1e4535f066ee93b2ed85a191fba5a`,
  `b4187965d94b036afa4b0d43162d0aae63c1d40e8e0ac3ba4cc0b94db25ad986`,
  `343abf1907ea76c896696818c3e836f3ac8b97cc43d7cdabe8e80a0ac6c746c2`,
  `dd723bd37abaedc2674ba6ef8a8c5b69bea4c7d31c8a608afb388d74db362929`,
  and
  `f8fa3c3994a6186cb01ab02bf1168ab99c6105b7bdc757c9a9e9dabbcd2cace6`.
- Full-text inspection resolves all three ambiguous cross-source SFT pairs as
  true duplicates. Each pair has the same question, options, complete
  reasoning, and answer; the only character-level difference is a final
  answer rendered as `\boxed{X}` versus `\boxed{\text{X}}`. The cases are:

  - renewable-energy answer D, 9,896/9,903 characters, character 5-gram
    Jaccard 0.998324, row 7,349, member/canonical SHA-256
    `8966e6aee8db0e9f02687c4901506ddb6d107878ae510bdd9c3c0c5d62de83ad`
    /
    `738af429cb10cc01ef6b8a93ed002b6ae3270af717191478211016d42f03ea0e`;
  - DNA-sequencing answer F, 11,883/11,890 characters, character 5-gram
    Jaccard 0.998311, row 7,381, member/canonical SHA-256
    `5f1d6a5e65ff860ee3c875b7baa62478671a7cd3b1cead7be08a5687d8a2a936`
    /
    `1740a741dfa43a4f4c77a50286350271645aeb9aa32e74d01497b6960c9e44fe`;
  - French-Revolution answer B, 12,140/12,147 characters, character 5-gram
    Jaccard 0.998433, row 7,409, member/canonical SHA-256
    `83ef15e9ceff2e82b0f52b2840019c0fefbce849bd22b6398ba17c3bdbd542f8`
    /
    `1e48dbf1373f2441690ee9f7390d29deed121246796fd0112ed85e20e790d7ff`.

- `/rav/datakit-6854-inspect-unresolved-0531-v116` failed before source
  inspection because a copied member ID omitted its final two characters.
  `/rav/datakit-6854-inspect-unresolved-0532-v117` selected by hash-verified
  pair-row identity instead and inspected all six complete documents.
- `/rav/datakit-6854-publish-manual-0536-v118` published the three hash-bound
  true-duplicate shards, and the separate
  `/rav/datakit-6854-verify-manual-0536-v119` job reread and verified them.
  Their semantic-judgment SHA-256 values are
  `f0c31197005900727abf1f69e7afb75b018c12c903728880600245cc89a4145d`,
  `1a6739c6111438469a405712af91f3088624beeec91f8ef0135572c926274155`,
  and
  `c48e6a33f47c0b0aa4382ae5cb2b5d8cba85e4398cef38965388191b632b8b93`.
  Their Parquet SHA-256 values are
  `1894b6ee5c63902adf89920fe9b473786404801927dee60ba837a6bbed846405`,
  `7d6058cb7a2ab402ff8f49d1b71213e6318b8a2793d50335714d2d97b6bb51a8`,
  and
  `747c9638b20c285d12fcc5d5bdaf311dd0faee25f6411a79bdcd1b1062213896`.
  Their record SHA-256 values are
  `f1670958290162c434496c83caa3fa46d3c2a4008ccfc7f282f24cee9eeb9bc2`,
  `1f2e28bcf245e04529b2cc2fea66fdb92c3e7cbc9de864de5c8390555488031a`,
  and
  `fe31e2f7f9a73e7caaf53922c390fbf2c57e0a511726b59bd2729848213c4355`.
- Across the stable 763-checkpoint snapshot, all 102 unresolved model outcomes
  are covered by 79 true-duplicate and 23 false-positive manual records. The
  adjusted totals are:

  - baseline: 79,694 pairs, 50,590 false positives, 29,104 true duplicates;
  - treatment: 17,390 pairs, 8,955 false positives, 8,435 true duplicates;
  - combined: 97,084 pairs, 59,545 false positives, 37,539 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T05:27:22Z — 95,804 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0518-v111` independently
  revalidated five additional baseline checkpoints. Their 640 pairs contain
  393 false positives, 246 true duplicates, and one unresolved outcome. All
  1,556 judgments were valid on the first request; four pairs were chunked and
  636 were direct.
- The p0 decision-file 4 outcome Parquet SHA-256 values for offsets 3,200 and
  3,328 are
  `346cfa488f9e3e53786cbed6dd418408cae4216997797c61156fa18bc37519ec`
  and
  `ff42a62d83db6ba9c0ee5552627bf034ca6d366d184c61b1f65420132c1d52f2`.
  The p1 decision-file 35 values for offsets 3,328, 3,456, and 3,584 are
  `562f21705ca3ef35f61f50b62f3eca6db1ab934033bc7c7a66a9774e9843df7a`,
  `be069281968d8d461b3cdf5a08bc2943399cbbd72d6179a7053a6bfef7851d3d`,
  and
  `b88e939d2f07656e6d041ae0c340df0a737fb86568c80a745288c8aa773eb5a3`.
- Complete-text review resolves the new ambiguous baseline pair as a false
  positive. Both documents are low-value college SEO pages with some shared
  sentence scaffolding, but the 911-character member and 827-character
  canonical name different institutions and programs. The member also has
  admissions, internship, and wildlife-ecology claims absent from the
  canonical. Their character 5-gram Jaccard value is 0.195294 and word
  5-gram Jaccard value is 0.086207, so dropping the member would remove
  substantive unique content. Pair location:
  `part-00035-of-00128.parquet:5706`; member/canonical text SHA-256 values:
  `4e5ea623b527e9b59b9eaa3d06efbb6d62e125b9cf4d7fcf9e676207027ed39c`
  /
  `4893ce3e5b496530e4b22e31e43f7505e575b1b5c377c62acb8ae20ccb5c1c4e`.
- The semantic-judgment SHA-256 value is
  `660ccd09c7064fef5dbda40299bc0086236502809df572fb95877732a5787902`.
  `/rav/datakit-6854-publish-manual-0524-v113` published its hash-bound manual
  shard, and the separate
  `/rav/datakit-6854-verify-manual-0525-v114` job reread and verified it. The
  Parquet and record SHA-256 values are
  `674985382c96ff96a4cc15200dafdbc61daa5e8e0ce8e23aa50bb239d866da6a`
  and
  `cccec5a3be82bd5171dfc2df3897f89ee9679c573a7f54e9266457538fe52669`.
- Across the stable 753-checkpoint snapshot, all 99 unresolved model outcomes
  are covered by 76 true-duplicate and 23 false-positive manual records. The
  adjusted totals are:

  - baseline: 78,414 pairs, 49,639 false positives, 28,775 true duplicates;
  - treatment: 17,390 pairs, 8,955 false positives, 8,435 true duplicates;
  - combined: 95,804 pairs, 58,594 false positives, 37,210 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T05:17:26Z — 95,164 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0516-v110` independently
  revalidated two additional baseline checkpoints. Their 256 pairs contain
  121 false positives and 135 true duplicates, with no unresolved outcomes.
  All 646 judgments were valid on the first request; two pairs were chunked
  and 254 were direct.
- The p0 decision-file 4 outcome Parquet SHA-256 value for offset 3,072 is
  `7b99e2d54b493841b77354ced6333236e917eaa109b89a2d03a0e151d9fc2bb9`.
  The p1 decision-file 35 value for offset 3,200 is
  `368cc59a2a07979c42663aac6c07d5f3a5b572582995bc350e8da72acce517cf`.
- Across the stable 748-checkpoint snapshot, all 98 unresolved model outcomes
  remain covered by 76 true-duplicate and 22 false-positive manual records.
  The adjusted totals are:

  - baseline: 77,774 pairs, 49,245 false positives, 28,529 true duplicates;
  - treatment: 17,390 pairs, 8,955 false positives, 8,435 true duplicates;
  - combined: 95,164 pairs, 58,200 false positives, 36,964 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T05:15:10Z — 94,908 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0514-v109` independently
  revalidated two additional baseline checkpoints. Their 256 direct-review
  pairs contain 119 false positives and 137 true duplicates, with no
  unresolved outcomes. All 557 judgments were valid on the first request.
- The p0 decision-file 4 outcome Parquet SHA-256 value for offset 2,944 is
  `c9de2d176995306c42e381b3dcbd9e6a70f817cd465d6af7ccd03d0a346c7043`.
  The p1 decision-file 35 value for offset 3,072 is
  `72cae58c8f60853fc3726f34bb3ab2acf235d6d54927be5220f5d5a15afeca99`.
- Across the stable 746-checkpoint snapshot, all 98 unresolved model outcomes
  remain covered by 76 true-duplicate and 22 false-positive manual records.
  The adjusted totals are:

  - baseline: 77,518 pairs, 49,124 false positives, 28,394 true duplicates;
  - treatment: 17,390 pairs, 8,955 false positives, 8,435 true duplicates;
  - combined: 94,908 pairs, 58,079 false positives, 36,829 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T05:12:34Z — 94,652 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0511-v108` independently
  revalidated ten additional baseline checkpoints. Their 1,280 pairs contain
  615 false positives and 665 true duplicates, with no unresolved outcomes.
  All 2,935 judgments were valid on the first request. Four pairs were chunked
  and 1,276 were direct.
- The p0 decision-file 4 outcome Parquet SHA-256 values for offsets 2,304,
  2,432, 2,560, 2,688, and 2,816 are
  `3c0051128815e25a7cd7b3c989af0f52ba1bc018002f8532bf47fb4f8dccd606`,
  `d15eecc81f99f379d919fa874178a558ff94b19930fe01f465b320206ed97bcb`,
  `925a35ee2d9f544b77a83cebb724a9df7fb7f3f7378bfae55261c86955586746`,
  `ea8302d227fae458b7b90ba1ae1309f3cd6ddce6659e34ad869a60035c93a23b`,
  and
  `d2b4ebb41198845619c1cfbbcb78f241091ed21d4c0c8c2f9696326d32f59331`.
  The p1 decision-file 35 values for offsets 2,432, 2,560, 2,688, 2,816, and
  2,944 are
  `8cd7e62ddf21c5ecf18e0bcde9facfa40b943d929939a39297faebc94df81216`,
  `750d46d4ad7bce5bd0ee7305eb5f9a207e6553e2bd5073649baadc93b652bb8e`,
  `0b7e8c3580436e891426d8fbf1aec03b750f1a1a32b435b79790146e8164951d`,
  `a7f63dc5d4cdb36fb2619ca366f75dce0bb800a04349d44f199e5d5bad0fa993`,
  and
  `ca9fd373690e7165d0acc536bd3e08d51670cbd4d4ed7dbaeface00872d9f85a`.
- Across the stable 744-checkpoint snapshot, all 98 unresolved model outcomes
  remain covered by 76 true-duplicate and 22 false-positive manual records.
  The adjusted totals are:

  - baseline: 77,262 pairs, 49,005 false positives, 28,257 true duplicates;
  - treatment: 17,390 pairs, 8,955 false positives, 8,435 true duplicates;
  - combined: 94,652 pairs, 57,960 false positives, 36,692 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T05:09:39Z — 93,372 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0502-v104` independently
  revalidated ten additional baseline checkpoints. Their 1,280 pairs contain
  583 model false positives, 695 model true duplicates, and two unresolved
  outcomes. All 2,970 judgments were valid on the first request. Five pairs
  were chunked and 1,275 were direct.
- The p0 decision-file 4 outcome Parquet SHA-256 values for offsets 1,664,
  1,792, 1,920, 2,048, and 2,176 are
  `27ee2503e2bb7c76c3db6386e0c68432f616b0e2924e4494c64914438db6a4ef`,
  `8275a24e597709f17a9817d28ef1cf8495ab090d721076d650efe0ddeb4f2258`,
  `0f47c2085a0d1dbfe684fc3135ea8232201c6c39dace716bbc7e12017f081b87`,
  `f4c020174f635c05f87367eb8172ed1ae6e1b4b4658af5d9a7ede5cc8196cc3b`,
  and
  `3346ec6d323a7ba001b0ff4af9dd6a96b5d787511fb7bc23a0ab1bd81e17d44a`.
  The p1 decision-file 35 values for offsets 1,792, 1,920, 2,048, 2,176, and
  2,304 are
  `c74f75e250042a89c568aae1f51a17656a48aeb819d2367f15aef7fbc532465e`,
  `479cd70f0d972fe9a549f8efe8f2624feb9ba2a7b661effbee3f7df33755dde3`,
  `35a5f2b0bd0349c1d18b4d5781c129b4d30fee741e7a2a1578139c5aab2762c2`,
  `65550ba0c7ef6650f2e85a00279bbece0486f3f56e085fedc87e22ee87275349`,
  and
  `57314c700ff6c61d81541ff827af29e27f487714be460c757eea95a4917c8a32`.
- Complete-text inspection resolves `part-00004-of-00128.parquet:2633` as a
  false positive. The 2,905-character member and 4,239-character canonical
  share fragments of a BetterHelp article but have character similarity only
  0.477044 and line similarity 0.400000. The member alone includes claims
  about a free seven-day trial, fast weekend responses, editable messages, and
  a Talkspace comparison. Member/canonical text SHA-256:
  `200d35ba0c45f842e0ede8cf478ebed2fe0d614350049fd6aeca3d5997119b2d` /
  `e756fcd33cf056fce930638d25a896c7519c2693c7f1faaf3eaa561a7fa9866a`.
- Complete-text inspection resolves `part-00004-of-00128.parquet:3376` as a
  true duplicate under the explicit low-value-template boundary. Both rows
  contain the same substantive sentence about teaching colors. The member's
  Pooh Bear title and publication date versus the canonical's playground-swing
  title are superficial slots unsupported by the shared Dora body. The texts
  contain 279 / 241 characters with 0.861538 character similarity.
  Member/canonical text SHA-256:
  `bb966ef213a7afeab6d2958ba42f868acccc34996716cec632ab7378c73b2775` /
  `3f0a88a725970caefc23c8df46ec4c743f862b09cc4b1d669bfe212707813543`.
- The false-positive and true-duplicate manual records have Parquet SHA-256
  `385f196eb03d9a912f1a51e4eac2905b39c497f99bd2c3afb57e820d9b1269e9`
  and
  `c3b214234412bb717ed6612f1962b5a8d54dd81de7f4a7a9c31c856046a458ee`.
  Their semantic-judgments SHA-256 values are
  `5290ca22ef081d0c0a9064099945eef15d032f89d24c61e1098463d75f5359c6`
  and
  `f852c93f8aec953540de6f4a18f732aba317234275590958a88d7f06ef19c177`.
  `/rav/datakit-6854-verify-manual-0509-v107` independently reread both source
  pairs, semantic checkpoints, manual rows, Parquet bytes, record hashes, and
  completion markers with exit 0 and no failures.
- Across the stable 734-checkpoint snapshot, all 98 unresolved model outcomes
  have manual records: 76 true duplicates and 22 false positives. The adjusted
  totals are:

  - baseline: 75,982 pairs, 48,390 false positives, 27,592 true duplicates;
  - treatment: 17,390 pairs, 8,955 false positives, 8,435 true duplicates;
  - combined: 93,372 pairs, 57,345 false positives, 36,027 true duplicates.

- Published and verified the
  [92,092-pair heartbeat](https://github.com/marin-community/marin/issues/6854#issuecomment-5082130212)
  on the coordinating issue.
- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T05:01:30Z — 92,092 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0454-v100` independently
  revalidated three additional baseline checkpoints. Their 384 pairs contain
  193 model false positives, 190 model true duplicates, and one unresolved
  outcome. All 922 judgments were valid on the first request. Two pairs were
  chunked and 382 were direct.
- The p0 decision-file 4 outcome Parquet SHA-256 values for offsets 1,408 and
  1,536 are
  `b5656a6691096b0ead89cd63b4647166cc5107cb1f3a042bd3e51f3358fa03d6`
  and
  `ca37575d13d30bd1f873b5995eb7379370fe282c69a5ab97a19df11b10b79120`.
  The p1 decision-file 35 value for offset 1,664 is
  `5e81fb7f656ec68226629b03284ebef0b41fb7bd1fdc014e85c4cd6a3a53a99d`.
- Complete-text inspection resolves the ambiguous baseline pair as a false
  positive. The canonical's 869 characters occur unchanged at the start of
  the 994-character member. The member then adds a complete question asking
  who can help set up Firefox Sync and an answer identifying the linked setup
  article. This is a distinct Q&A training example even though the shared
  forum answer contains its source material. Character similarity is 0.932904
  and line similarity is 0.909091. Pair location:
  `part-00004-of-00128.parquet:2097`; member/canonical text SHA-256:
  `37006d524a66f442870ff216a46e4565fe5532b821b2691b12940f9a81e212dd` /
  `7296d7a5db7226c781f93a46051ccdaaf2ac4f8cebcc9960090c72b2f2b578c8`.
- The hash-bound manual record has Parquet SHA-256
  `e20ee4dcf8063d3a6ebcbb9b28e0067516872f44a1b87a5c45205fff55582e54`
  and semantic-judgments SHA-256
  `5e5ba86efcd9610eda95f2c06f97f3bdb0e904b3f32b825cd4bad40e0181db1c`.
  `/rav/datakit-6854-verify-manual-0500-v103` independently reread the source
  pair, semantic checkpoint, manual row, Parquet bytes, record hash, and
  completion marker.
- Across the stable 724-checkpoint snapshot, all 96 unresolved model outcomes
  have manual records: 75 true duplicates and 21 false positives. The adjusted
  totals are:

  - baseline: 74,702 pairs, 47,806 false positives, 26,896 true duplicates;
  - treatment: 17,390 pairs, 8,955 false positives, 8,435 true duplicates;
  - combined: 92,092 pairs, 56,761 false positives, 35,331 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T04:53:15Z — 91,708 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0452-v99` independently
  revalidated five additional baseline checkpoints from the immutable semantic
  output. Their 640 direct-review pairs contain 507 false positives and 133
  true duplicates, with no unresolved outcomes. All 1,319 judgments were valid
  on their first request.
- The p0 decision-file 4 outcome Parquet SHA-256 values for offsets 1,152 and
  1,280 are
  `ce02ffbe15e1b83c93b485781138e61360e764f9d02a8b3bac4f2597449fc82f`
  and
  `b44e9ccad5130fac1b2e94a51676aaac0d1518a502f57afd460ab19af650b03b`.
  The p1 decision-file 35 values for offsets 1,280, 1,408, and 1,536 are
  `4d171aec850471c2813bc3df918b493e3ec6dbe391ccfb26a009f9cb1fd44622`,
  `33aecbab7c703ef6bdfc440b542b52874e84d7e02f9b61e661872eaba2700e3f`,
  and
  `8fb656da6d1f431537c2e5c2eecce75fd0418060040f7c06700ade91f7bdcd1d`.
- Across the stable 721-checkpoint snapshot, all 95 unresolved model outcomes
  remain covered by 75 true-duplicate and 20 false-positive manual records.
  The adjusted totals are:

  - baseline: 74,318 pairs, 47,612 false positives, 26,706 true duplicates;
  - treatment: 17,390 pairs, 8,955 false positives, 8,435 true duplicates;
  - combined: 91,708 pairs, 56,567 false positives, 35,141 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T04:50:02Z — 91,068 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0448-v98` independently
  revalidated ten additional baseline checkpoints from the immutable semantic
  output. Their 1,280 pairs contain 954 false positives and 326 true
  duplicates, with no unresolved outcomes. All 2,738 judgments were valid on
  their first request. Four pairs were chunked and 1,276 were direct.
- The p0 decision-file 4 outcome Parquet SHA-256 values for offsets 512, 640,
  768, 896, and 1,024 are
  `ba572605190aacd341577f454d2f40d8a7db777ae9b7d987fab6521d052ecc6e`,
  `a6c6249faf86aea088e262f2101c1407ace1a7c1f714cb696ee4b34411facbb3`,
  `d36b74f32c01ceec3b881acd282dac96f10cb1b8ce3ca36899a67b029d2ce69f`,
  `8753937f76e8d66ad9a78b552b56ceb536fb9ead64ce17985befb17e8229bdde`,
  and
  `7b954b1579689a0ca79e885a2226a52f2aa62873da209a9956544c9cdd09458c`.
  The p1 decision-file 35 values for offsets 640, 768, 896, 1,024, and 1,152
  are
  `1724dac9655a110197d6afdf127c3bf41442d40bf5199d59ad37d0dd14081b2c`,
  `ea55dbadc554a8f7dd27e22dc3e7cb66b2e2e7195c9b62efcaaed41b146618b2`,
  `06ef79ebf117b1c6e910f034f8d272560a1c6f253ac60851b62d394560626189`,
  `84dc336bf3ba945011b6375438d10db6a4e829304234f55346908bea7c1bb3da`,
  and
  `7087b6f661add9ba44ce6fd3af8b048ca2c0844a36b48bb828a3717dd70ab3b0`.
- Across the stable 716-checkpoint snapshot, all 95 unresolved model outcomes
  remain covered by 75 true-duplicate and 20 false-positive manual records.
  The adjusted totals are:

  - baseline: 73,678 pairs, 47,105 false positives, 26,573 true duplicates;
  - treatment: 17,390 pairs, 8,955 false positives, 8,435 true duplicates;
  - combined: 91,068 pairs, 56,060 false positives, 35,008 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T04:44:29Z — 89,788 pairs verified

- Seven additional checkpoints passed independent validation: two p0 and
  three p1 baseline checkpoints plus two p2 treatment checkpoints. Their 827
  pairs contain 589 model false positives, 235 model true duplicates, and
  three unresolved outcomes. The 4,157 judgments used 4,171 requests: 4,151
  valid and 20 invalid responses across seven retried judgments. Twenty-five
  pairs were chunked and 802 were direct.
- The p0 decision-file 4 outcome Parquet SHA-256 values for offsets 256 and
  384 are
  `aeb561ba245e095775e56068eeb0bcaaccbd462fe23d387d0437da903fcd251a`
  and
  `107a5ecb58d74836506547c6936175f1e7ebbdf13ea05f0c40027363ecb84ed8`.
  The p1 decision-file 35 values for offsets 256, 384, and 512 are
  `51d09401aaadb1ba97be38b7f8a0af1ad32d23d1dbef1af2b718377fe159ff5c`,
  `7af2c7c77766f4d6afd15fde07438934579953a4caffc976ab7f65c3dbbe3faf`,
  and
  `edd90d79e26aa8fd0becee02a9135530cdd65fcadbafadc0408d775dc2b8cef7`.
  The p2 decision-file 67 values for offsets 5760 and 5888 are
  `89f0cddbc3edc3f05780943e1053b58eb8a475edc7a2e51232c7c2a1166698d6`
  and
  `071b634c0d8fd81162fcc26e5fcd86fb40c3e54d73ae755d07fa91c30e2b4a33`.
- Complete line and character diffs resolve all three ambiguities as true
  duplicates. Each member and canonical SFT example is identical except for a
  seven-character `\text{...}` wrapper around the same final boxed answer:

  - `part-00067-of-00128.parquet:9034`, 15,665 / 15,672 characters,
    similarity 0.999777; member/canonical SHA-256
    `80637ed3184835ad3f1055a13a61458149065786a1ef87ac0a346ea3cc93cd5a` /
    `20fca21367a1faaf7fece908268fe7d1eb3ed0f6c74e18af325f27706b6d3140`;
  - `part-00067-of-00128.parquet:9048`, 7,517 / 7,510 characters,
    similarity 0.999534; member/canonical SHA-256
    `3f549f8a4cab6bf126a0fd61b70697d9ae46521eca1a4b29a5ccff96fa136106` /
    `1cdbdf0b62292cf9de7b2cc755faecca00c0fddd9440569db5156c25a1a18b90`;
  - `part-00067-of-00128.parquet:9050`, 5,623 / 5,616 characters,
    similarity 0.999377; member/canonical SHA-256
    `6aded64e77eee455d034749f9e0810180798b9fea5f3d96e52fb0d6f0c01b7d9` /
    `b2c59dc22b34e32ff2b5e2e8b72148b78a5d9e9d633604fd33eff553415872b3`.

- The three manual Parquet records have SHA-256
  `3b96e0b39b6360c5e1615a76447676c40125c53372fbe1f81dcdc764f1f6af25`,
  `d655c7110e725ed25030165dab008883117a67d5afc378c9d95bcec7adeb40b7`,
  and
  `e5212f74c103232852ffde70c9c8f02644fe536d59ecfafa84177b331793927f`.
  A separate batch-priority Iris process reread and exactly checked the source
  pairs, semantic checkpoint, judgment hashes, manual rows, Parquet bytes, and
  completion markers.
- Across the stable 706-checkpoint snapshot, all 95 unresolved model outcomes
  have manual records: 75 true duplicates and 20 false positives. The adjusted
  totals are:

  - baseline: 72,398 pairs, 46,151 false positives, 26,247 true duplicates;
  - treatment: 17,390 pairs, 8,955 false positives, 8,435 true duplicates;
  - combined: 89,788 pairs, 55,106 false positives, 34,682 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T04:37:05Z — 88,961 pairs verified

- Three additional treatment checkpoints passed independent validation. Their
  384 pairs contain 100 model false positives, 283 model true duplicates, and
  one unresolved outcome. All 797 judgments were valid on the first attempt;
  every pair used direct review. Decision-file 67 outcome Parquet SHA-256
  values for offsets 5376, 5504, and 5632 are
  `ff052404a7efbc1c8730e1540a93a8a76b713ec9727c79bdd171d303fdf8cc5a`,
  `2f24f59bff204d087f46baaed07c56de568c914bd2a93cf403af316827e5a327`,
  and
  `700f919a33581e5ee1b258ec2e75917dc3cc29bf8f4025bd8df03a0d35e11cc6`.
- Complete-text review resolves the ambiguity as a true duplicate. Both
  documents contain the same substantive paragraph verbatim. The member's
  only unique text is a weekly-instagram date-range heading and a two-comment
  count; the canonical retains the full title and adds two Q&A examples.
  Character similarity is 0.853157. Pair location:
  `part-00067-of-00128.parquet:8547`; member/canonical text SHA-256:
  `ca25d3f183b75dd09ed51b4c2c23e80675acb04be591917291d20d69d5b1cf83` /
  `153c38439185e187b0fd262e6f3994f3e98d8f603d6c1068b493bc5a59ea3dc4`.
- The hash-bound manual record has Parquet SHA-256
  `f4bbb2e3d05727330132abaa4dc2835288e5bceacb605f11d14468dac1329c10`
  and semantic-judgments SHA-256
  `33b6a0244c953beab62d20945c24dcc41d764325395a8d9e838fd3675f2a9ea4`.
  A separate batch-priority Iris process reread and exactly checked the source
  pair, semantic checkpoint, judgment hash, manual row, Parquet bytes, and
  completion marker.
- Across the stable 699-checkpoint snapshot, all 92 unresolved model outcomes
  have manual records: 72 true duplicates and 20 false positives. The adjusted
  totals are:

  - baseline: 71,758 pairs, 45,706 false positives, 26,052 true duplicates;
  - treatment: 17,203 pairs, 8,811 false positives, 8,392 true duplicates;
  - combined: 88,961 pairs, 54,517 false positives, 34,444 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T04:26:56Z — 88,577 pairs verified

- One p0 and three p2 baseline checkpoints passed independent validation. Their
  512 pairs contain 285 false positives and 227 true duplicates, with no
  unresolved outcomes. All 2,298 judgments were valid on the first attempt.
  Fourteen pairs were chunked and 498 were direct.
- The p0 decision-file 4 offset-128 outcome Parquet SHA-256 is
  `36c4f38ebec8884de47cae938878f86d8e4aa33cc9c107ecb7c429d3b62a4497`.
  The p2 decision-file 67 values for offsets 4992, 5120, and 5248 are
  `e27fdb458fe980b2ee6c6f4137db35d3831c58cfa7230983e08692fb81ab2dea`,
  `bfd5492f5f683026c879ef909070c7317b84cc0918e463e771739f1172eb7a6b`,
  and
  `62387c06888c3c7240a3d03df41c750e7d475584a65e8e93da8701bcc8d53391`.
- Across the stable 696-checkpoint snapshot, all 91 unresolved model outcomes
  have manual records: 71 true duplicates and 20 false positives. The adjusted
  totals are:

  - baseline: 71,758 pairs, 45,706 false positives, 26,052 true duplicates;
  - treatment: 16,819 pairs, 8,711 false positives, 8,108 true duplicates;
  - combined: 88,577 pairs, 54,417 false positives, 34,160 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T04:23:01Z — 88,065 pairs verified

- One p1 and three p2 baseline checkpoints passed independent validation. Their
  512 pairs contain 371 false positives and 141 true duplicates, with no
  unresolved outcomes. All 1,586 judgments were valid on the first attempt.
  Thirteen pairs were chunked and 499 were direct.
- The p1 decision-file 35 offset-128 outcome Parquet SHA-256 is
  `4c0d4dd30ebdc7d569903969135fe334a1bc4984f3887202492484febd69ff29`.
  The p2 decision-file 67 values for offsets 4608, 4736, and 4864 are
  `015d29ce412133b50f052c18256b32d1d5e29be917c1c3b6f22dd22a3f57f8ea`,
  `85dbf86c99870cae6568dc732fa19b63c50abacf7abf6f57ec054c6c5ecc9425`,
  and
  `44ee76326e4b9500712680331a0450dc243f8892859e812b1ee425be6cbae347`.
- Across the stable 692-checkpoint snapshot, all 91 unresolved model outcomes
  have manual records: 71 true duplicates and 20 false positives. The adjusted
  totals are:

  - baseline: 71,246 pairs, 45,421 false positives, 25,825 true duplicates;
  - treatment: 16,819 pairs, 8,711 false positives, 8,108 true duplicates;
  - combined: 88,065 pairs, 54,132 false positives, 33,933 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T04:20:34Z — 87,553 pairs verified

- Two additional p2 baseline checkpoints passed independent validation. Their
  256 pairs contain 167 model false positives, 88 model true duplicates, and
  one unresolved outcome. All pairs were direct. The 523 judgments used 525
  attempts: 522 valid responses and three invalid JSON responses confined to
  one judgment. The decision-file 67 outcome Parquet SHA-256 values for offsets
  4352 and 4480 are
  `71b9a3f139b8b812ce1e9486d613239dca813008f20e9ad312fcbb71eeb3c629`
  and
  `5d14c5be361057d502539d280bd3e368d82e77d2fecc318fe4cf4e8d43020be6`.
- Complete character and line comparison resolves pair-file row 7601 as a true
  duplicate. The 13,368-character member and 13,361-character canonical contain
  the identical World War II multiple-choice question, ten options, full
  reasoning trace, historical explanation, and answer A. The only changed line
  is the final answer formatting: member `\boxed{\text{A}}` versus canonical
  `\boxed{A}`. Sequence similarity is 0.999738.
- The member/canonical text SHA-256 values are
  `57d6c8b2bdf35c7b2ec43677de968278748ae22cd42ec33baf692748edc75f7d`
  and
  `5ab29e339ffb25bdb62ad902b73b2e4aaf11556d9bf25c71470f653874a3385b`.
  Both arms share all 26 MinHash buckets for the pair.
- The manual decision binds semantic-judgment SHA-256
  `9a670a8f551b359f6171fa31417f8a616b8c697ff499826d8171fc5dc9d7a83d`.
  Its Parquet SHA-256 is
  `719c2adaa608c008cdd1fce888e4890bc33597cc21b30005251ed0fe2a249292`.
  A separate batch-priority Iris process exactly reread the source pair,
  semantic outcome, manual row, Parquet bytes, and completion marker.
- Across the stable 688-checkpoint snapshot, all 91 unresolved model outcomes
  have manual records: 71 true duplicates and 20 false positives. The adjusted
  totals are:

  - baseline: 70,734 pairs, 45,050 false positives, 25,684 true duplicates;
  - treatment: 16,819 pairs, 8,711 false positives, 8,108 true duplicates;
  - combined: 87,553 pairs, 53,761 false positives, 33,792 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T04:12:52Z — 87,297 pairs verified

- Three additional p2 baseline checkpoints passed independent validation.
  Their 384 pairs contain 297 false positives and 87 true duplicates, with no
  unresolved outcomes. All 831 judgments were valid on the first attempt. One
  pair was chunked and 383 were direct.
- The decision-file 67 outcome Parquet SHA-256 values for offsets 3968, 4096,
  and 4224 are
  `4e97fb34c8f9adc02933b31089847b6fc9ae2548f1e1bd13a9239fe3ba11f8bf`,
  `04ff4d7f0ccee9e223a666c056f812c6a067a0f7cf7c1be31d58683ec81091f2`,
  and
  `d22d8dc82307a3766185075b36b45e81de0051481245456702e89788d1296b38`.
- Across the stable 686-checkpoint snapshot, all 90 unresolved model outcomes
  have manual records: 70 true duplicates and 20 false positives. The adjusted
  totals are:

  - baseline: 70,478 pairs, 44,883 false positives, 25,595 true duplicates;
  - treatment: 16,819 pairs, 8,711 false positives, 8,108 true duplicates;
  - combined: 87,297 pairs, 53,594 false positives, 33,703 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T04:07:27Z — 86,913 pairs verified

- Two additional p2 baseline checkpoints passed independent validation. Their
  256 pairs contain 195 false positives and 61 true duplicates, with no
  unresolved outcomes. All 582 judgments were valid on the first attempt. One
  pair was chunked and 255 were direct.
- The decision-file 67 outcome Parquet SHA-256 values for offsets 3712 and
  3840 are
  `b0fd99d466e7e46549925ab496206fdd206104724b68bffcbed5b86d45dd7052`
  and
  `27670e50a1241fe923bdacbde3f73133151b282c2b7bc29dc4c69a3a67c5e185`.
- Across the stable 683-checkpoint snapshot, all 90 unresolved model outcomes
  have manual records: 70 true duplicates and 20 false positives. The adjusted
  totals are:

  - baseline: 70,094 pairs, 44,586 false positives, 25,508 true duplicates;
  - treatment: 16,819 pairs, 8,711 false positives, 8,108 true duplicates;
  - combined: 86,913 pairs, 53,297 false positives, 33,616 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T04:03:12Z — 86,657 pairs verified

- One p1 and six p2 baseline checkpoints passed independent validation. Their
  896 pairs contain 536 false positives and 360 true duplicates, with no
  unresolved outcomes. All 3,439 judgments were valid on the first attempt.
  Twenty-seven pairs were chunked and 869 were direct.
- The p1 decision-file 35 offset-0 outcome Parquet SHA-256 is
  `f454a0f5a83cd72bd9a656056b5ab3a59e8df9880dcd454f09ee45b34989e75a`.
  The p2 decision-file 67 values for offsets 2944 through 3584 are:

  - 2944:
    `83b9e24cf8a133ef35396e355e1895cff2d7ca95841baf2db6937ca3ea8981e5`;
  - 3072:
    `2eb665822d52cd46a69fb030d79057eefdbd95d69792e0362af6020907ceab59`;
  - 3200:
    `60423446ea206c63de439974b8425697db16e691e7124c9fab5ac4c05ca858b1`;
  - 3328:
    `3190e8f8f5f42dc7aa72aabdb52e30aacc358a4ed5477384307b652079499e3d`;
  - 3456:
    `e15e8e0ad61d3af90f2371be5f9c78562a4520f2b2a0a2d2f82f4b34129671ef`;
  - 3584:
    `b7418982b994fbbf318fe2c67302118419e150477cce107414fd7191b6348bf3`.

- Across the stable 681-checkpoint snapshot, all 90 unresolved model outcomes
  have manual records: 70 true duplicates and 20 false positives. The adjusted
  totals are:

  - baseline: 69,838 pairs, 44,391 false positives, 25,447 true duplicates;
  - treatment: 16,819 pairs, 8,711 false positives, 8,108 true duplicates;
  - combined: 86,657 pairs, 53,102 false positives, 33,555 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T04:00:50Z — 85,761 pairs verified

- Two additional p2 baseline checkpoints passed independent validation. Their
  256 pairs contain 120 model false positives, 135 model true duplicates, and
  one unresolved outcome. All 550 judgments were valid on the first attempt,
  and all 256 pairs were direct. The decision-file 67 outcome Parquet SHA-256
  values for offsets 2688 and 2816 are
  `5ee9ac61049be2ed72d72070cc9b65ea53efd375c13d240e14c77a1164fca76c`
  and
  `71d019fc4218a5c2bda4c701f68d285254e7f1b28302a27ead400127022a0279`.
- Complete-text review resolves pair-file row 4604 as a true duplicate. Both
  documents are thesaurus-spun Hemant Enterprises court-marriage SEO pages
  with the same legal claims and workflow. The member-only material is the
  Kanjurmarg West/Maharashtra location slot, vendor framing, and the incomplete
  heading `Required Documents (Witness) for` with no document list. The audit
  contract treats locations substituted into the same low-value SEO scaffold
  as superficial fields.
- The member/canonical texts contain 3,259 / 3,140 characters, have sequence
  similarity 0.765120, character-5-gram Jaccard 0.477076, and word-5-gram
  Jaccard 0.127485. Their SHA-256 values are
  `cca2efac3fe2cf0a181c918cee7dce1454a0fe0cdb74de03e67c682bfb762b8f`
  and
  `a8af5c41186b0879c642b1f095802df9a086dba039831271a82ce3a158f9c84c`.
  They share one baseline bucket and no treatment bucket.
- The manual decision binds semantic-judgment SHA-256
  `adde39f23f867799a8673e7852350388959c782503cf3bb83996debe301d5283`.
  Its Parquet SHA-256 is
  `37769d017a4b7ad77e850e677999c4901bd64b3d3c0df828b6af7eed4e7f0c14`.
  A separate batch-priority Iris process exactly reread the source pair,
  semantic outcome, manual row, Parquet bytes, and completion marker.
- Across the stable 674-checkpoint snapshot, all 90 unresolved model outcomes
  have manual records: 70 true duplicates and 20 false positives. The adjusted
  totals are:

  - baseline: 68,942 pairs, 43,855 false positives, 25,087 true duplicates;
  - treatment: 16,819 pairs, 8,711 false positives, 8,108 true duplicates;
  - combined: 85,761 pairs, 52,566 false positives, 33,195 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T03:49:27Z — 85,505 pairs verified

- Three additional p2 baseline checkpoints passed independent validation.
  Their 384 pairs contain 181 false positives and 203 true duplicates, with no
  unresolved outcomes. All 888 judgments were valid on the first attempt. One
  pair was chunked and 383 were direct.
- The decision-file 67 outcome Parquet SHA-256 values for offsets 2304, 2432,
  and 2560 are
  `8e18e7cbe5369df48cb03789c298da62a57f72529221a0291b3a981c78ac8429`,
  `ccbd21bc573f7d39ad75e49c0064299f6b16113118a3f2c9e3015e526d3898f8`,
  and
  `e327c3a15072fff688506f49e89387540090966c44137cb9b26df82520d2ddfe`.
- Across the stable 672-checkpoint snapshot, all 89 unresolved model outcomes
  have manual records: 69 true duplicates and 20 false positives. The adjusted
  totals are:

  - baseline: 68,686 pairs, 43,735 false positives, 24,951 true duplicates;
  - treatment: 16,819 pairs, 8,711 false positives, 8,108 true duplicates;
  - combined: 85,505 pairs, 52,446 false positives, 33,059 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T03:43:57Z — 85,121 pairs verified

- Two additional p2 baseline checkpoints passed independent validation. Their
  256 pairs contain 129 false positives and 127 true duplicates, with no
  unresolved outcomes. All 747 judgments were valid on the first attempt.
  Four pairs were chunked and 252 were direct.
- The decision-file 67 outcome Parquet SHA-256 values for offsets 2048 and
  2176 are
  `78fd2c720de10dfefe390b8febb548edcb85feeffc71348db8a95db7abcefbef`
  and
  `ae5408aaf8b2b27dcc9e73d584e92008e558b507594ce855d121c83dde3a8ed9`.
- Across the stable 669-checkpoint snapshot, all 89 unresolved model outcomes
  have manual records: 69 true duplicates and 20 false positives. The adjusted
  totals are:

  - baseline: 68,302 pairs, 43,554 false positives, 24,748 true duplicates;
  - treatment: 16,819 pairs, 8,711 false positives, 8,108 true duplicates;
  - combined: 85,121 pairs, 52,265 false positives, 32,856 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T03:39:04Z — 84,865 pairs verified

- One p0 and three p2 baseline checkpoints passed independent validation.
  Their 512 pairs contain 286 false positives and 226 true duplicates, with no
  unresolved outcomes. All 4,540 judgments were valid on the first attempt.
  Twenty-six pairs were chunked and 486 were direct.
- The p0 decision-file 4 offset-0 outcome Parquet SHA-256 is
  `472a28160174c4f80a65655be008f2c147a6de1f4770d10e1805bc40566460ba`.
  The p2 decision-file 67 values for offsets 1664, 1792, and 1920 are
  `5c64a1926bc9e49c4e93066ef24e11769ff0cbe8bf58f9e519259caf51be1562`,
  `bd0e208bb6e98176123dfd269e0f98b27cdc1337efab84e9ebf15778b57fc203`,
  and
  `933efe4879b77d9c70c5a916447d4ddae01a312ecc0feaadc162c096fd0c62e1`.
- Across the stable 667-checkpoint snapshot, all 89 unresolved model outcomes
  have manual records: 69 true duplicates and 20 false positives. The adjusted
  totals are:

  - baseline: 68,046 pairs, 43,425 false positives, 24,621 true duplicates;
  - treatment: 16,819 pairs, 8,711 false positives, 8,108 true duplicates;
  - combined: 84,865 pairs, 52,136 false positives, 32,729 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T03:35:58Z — 84,353 pairs verified

- Four additional p2 baseline checkpoints passed independent validation.
  Their 512 pairs contain 410 false positives and 102 true duplicates, with no
  unresolved outcomes. All 1,055 judgments were valid on the first attempt,
  and all 512 pairs were direct.
- The decision-file 67 outcome Parquet SHA-256 values for offsets 1152, 1280,
  1408, and 1536 are
  `30839dd71559ea6cbad0443e81bbc3c52ae59827e85809f50f1ee6da6e74e730`,
  `2e1aae555e9f892f1198adc317d3da170602dae2cad5651077e893a93b885e51`,
  `da4d6a5836b3b7ef707f89e8557e6ce82fa5fb8edc349f41926f470f3fef00c9`,
  and
  `c643ad16cf8a81a7a3e82ee823b9825d563b65778119175a2b156a6dc6a71833`.
- Across the stable 663-checkpoint snapshot, all 89 unresolved model outcomes
  have manual records: 69 true duplicates and 20 false positives. The adjusted
  totals are:

  - baseline: 67,534 pairs, 43,139 false positives, 24,395 true duplicates;
  - treatment: 16,819 pairs, 8,711 false positives, 8,108 true duplicates;
  - combined: 84,353 pairs, 51,850 false positives, 32,503 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T03:29:43Z — 83,841 pairs verified

- Three additional p2 baseline checkpoints passed independent validation.
  Their 384 pairs contain 255 false positives and 129 true duplicates, with no
  unresolved outcomes. All 842 judgments were valid on the first attempt. Two
  pairs were chunked and 382 were direct.
- The decision-file 67 outcome Parquet SHA-256 values for offsets 768, 896,
  and 1024 are
  `154fd7c0b3772121e4b7baa4d5492ad3f730e3666690e2c43ad7caad623f52e1`,
  `148ad2d4c50beb15e93fede4e8a75fc9e92281152c8927a20f2d421e7d7fa600`,
  and
  `5bc13491bbd768b0ea4e7c2141876d1deee40722441984c893c244fecaec4e29`.
- Across the stable 659-checkpoint snapshot, all 89 manual records leave:

  - baseline: 67,022 pairs, 42,729 false positives, 24,293 true duplicates;
  - treatment: 16,819 pairs, 8,711 false positives, 8,108 true duplicates;
  - combined: 83,841 pairs, 51,440 false positives, 32,401 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T03:22:55Z — 83,457 pairs verified

- Three additional p2 baseline checkpoints passed independent validation.
  Their 384 pairs contain 268 false positives and 116 true duplicates, with no
  unresolved outcomes. All 945 judgments were valid on the first attempt.
  Five pairs were chunked and 379 were direct.
- The decision-file 67 outcome Parquet SHA-256 values for offsets 384, 512,
  and 640 are
  `ec5d2a14bb4d2ffce1b40edb0fb51fbda2a2ff96f658b416a24887c9d63c66cb`,
  `8686111082235673d2984ba1680759343b662dda6fc8fc9c5c38c427792a3ccf`,
  and
  `c62ec834fcebf34b4fef3a241da1f54f06c0a1d078fc080762fb4c12c319a3b1`.
- Across the stable 656-checkpoint snapshot, all 89 manual records leave:

  - baseline: 66,638 pairs, 42,474 false positives, 24,164 true duplicates;
  - treatment: 16,819 pairs, 8,711 false positives, 8,108 true duplicates;
  - combined: 83,457 pairs, 51,185 false positives, 32,272 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T03:16:52Z — 83,073 pairs verified

- Two additional baseline checkpoints passed independent validation: one from
  p2 and one from p3. Their 256 pairs contain 203 false positives and 53 true
  duplicates, with no unresolved outcomes. All 4,664 judgments were valid on
  the first attempt. Forty-nine pairs were chunked and 207 were direct.
- The p2 decision-file 67 offset-256 outcome Parquet SHA-256 is
  `34ebb47ae4a68075049ca7e54d888431242e8da69a37e75ba79806394d72a8fb`.
  The p3 decision-file 100 offset-0 SHA-256 is
  `8fa3f3e9a3de338f356f2eb891af5db09c71db5eee6dd6e2b9f7e3ce814f434d`.
- Across the stable 653-checkpoint snapshot, all 89 manual records leave:

  - baseline: 66,254 pairs, 42,206 false positives, 24,048 true duplicates;
  - treatment: 16,819 pairs, 8,711 false positives, 8,108 true duplicates;
  - combined: 83,073 pairs, 50,917 false positives, 32,156 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T03:12:17Z — 82,817 pairs verified

- Two additional treatment checkpoints passed independent validation. Their
  217 pairs contain 149 model false positives, 66 model true duplicates, and
  two unresolved outcomes. The 447 judgments used 460 attempts: 441 valid
  responses and 19 invalid responses affecting seven retried judgments. All
  pairs were direct. The outcome Parquet SHA-256 values for decision-file 34
  offsets 5632 and 5760 are
  `f767004cd947c6740b06d1be3ec0249800c2bc06a1856a236b79030ccfb15e50`
  and
  `3206848347369b7bfce8041273ce0c20f924a396d624c12b24917b3ed4c38d36`.
- Complete character comparison resolves both ambiguities as true duplicates.
  In both pairs, every canonical character occurs unchanged and in order in
  the member. The member differs only by adding `\text{` and `}` around the
  same final answer:

  - `part-00034-of-00128.parquet:8939`, 11,402 / 11,395 characters;
    member/canonical text SHA-256
    `80e868bf4bb40bcb7c0c134a3b3d3c698745c0df0d4c36d7f966a4dbed2eec8d` /
    `3de6dd4d001621269654614bb8150c313222674d0219f571fb1e58a2e666b0c9`;
  - `part-00034-of-00128.parquet:8942`, 8,488 / 8,481 characters;
    member/canonical text SHA-256
    `9cccefe0c17b19cdaf280d1310aacc058adb1a0d2d0593c386bce97b0470fcb5` /
    `ca080587a0bbfa0de00fe392974fca8a3fa7b5fd50e37b0e6f84a2932980ec6a`.

- The manual decisions bind semantic-judgment SHA-256 values
  `e4fae28e7c5ac06a5ae48feecd43566ecc853ec406a3b78a85a6cff47fbcf5b5`
  and
  `66b404615c69c0affc1c0c952eff81d2567cd497f68cf9e8893bacf11a5228d5`.
  Their Parquet SHA-256 values are
  `87d5733a3794a75ff7ab941a1b3929021ecc3136ac266119befd27f59cf6129b`
  and
  `35e2a11bac9ff193de953a89500609fa4b30109d3d8806d4717a1fd1aef511a2`.
  A separate batch-priority Iris process exactly reread the source pairs,
  semantic evidence, manual records, Parquet bytes, and completion markers.
- Across the stable 651-checkpoint snapshot, all 89 manual records leave:

  - baseline: 65,998 pairs, 42,003 false positives, 23,995 true duplicates;
  - treatment: 16,819 pairs, 8,711 false positives, 8,108 true duplicates;
  - combined: 82,817 pairs, 50,714 false positives, 32,103 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T03:03:06Z — 82,600 pairs verified

- Two additional treatment checkpoints passed independent validation. Their
  256 pairs contain 85 false positives and 171 true duplicates, with no
  unresolved outcomes. All 538 judgments were valid on the first attempt,
  and all 256 pairs were direct.
- The outcome Parquet SHA-256 values for decision-file 34 offsets 5376 and
  5504 are
  `ae6e1e4309b8f50e20614d6ffe0943f62c86b8fa90a2ed7aac33e8ab35b04742`
  and
  `9727268983e502521fe4a2ddd19fb0451caff31b1ab04995546c38f22ef5a7ca`.
- Across the stable 649-checkpoint snapshot, all 87 manual records leave:

  - baseline: 65,998 pairs, 42,003 false positives, 23,995 true duplicates;
  - treatment: 16,602 pairs, 8,562 false positives, 8,040 true duplicates;
  - combined: 82,600 pairs, 50,565 false positives, 32,035 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T02:59:37Z — 82,344 pairs verified

- Seven additional checkpoints passed independent validation: six from p1
  and one from p2. Their 896 pairs contain 501 false positives and 395 true
  duplicates, with no unresolved outcomes. All 2,395 judgments were valid on
  the first attempt. Thirteen pairs were chunked and 883 were direct.
- The p1 block crossed from the baseline into treatment decisions: 58
  baseline and 710 treatment pairs. P2 contributed 128 baseline pairs. The p1
  outcome Parquet SHA-256 values for decision-file 34 offsets 4608 through
  5248 are
  `fafc3b7262ffa8e98ec17c6c38061df38c87fda4aa5b353681f557602071bf6d`,
  `25a6d46ba64dd88ef8167ba11878fe9553d9294040091d2b8b36ca91486556f3`,
  `ae1d7eb77f172cfb5cd90dff9e43b4949288f84fca570e5f616f8c58521de8cb`,
  `0218f6fc3ee739f28ac7e3fdf1a1cabc846e3bb2c0637e578bc06511d83ab138`,
  `e75173b241e257c848167a783029f43c8e378481f2d1f7779dfea588877b8e3c`,
  and
  `645b7e5e13d962161ad103829bca7958c991dcc66d99b12878a18b91d6e0ca52`.
  The p2 decision-file 67 offset-128 SHA-256 is
  `687554265d8fdee250b62b97453641d53d5fa71963f76ee6d03a98377dfa8325`.
- Across the stable 647-checkpoint snapshot, all 87 manual records leave:

  - baseline: 65,998 pairs, 42,003 false positives, 23,995 true duplicates;
  - treatment: 16,346 pairs, 8,477 false positives, 7,869 true duplicates;
  - combined: 82,344 pairs, 50,480 false positives, 31,864 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T02:57:10Z — 81,448 pairs verified

- One additional baseline checkpoint passed independent validation: 128
  pairs, 115 model false positives, 10 model true duplicates, and three
  unresolved outcomes. Its 274 valid judgments used 298 attempts; eight
  judgments exhausted three invalid JSON responses each. One pair was chunked
  and 127 were direct. The outcome Parquet SHA-256 is
  `8f83e858322070b732847d1ed85263796d900f5cd72505776111ed8f45b97e9b`.
- Full-text and character-level comparison resolves all three ambiguities as
  true duplicates. Each pair has an exactly identical prefix followed only by
  `\boxed{\text{X}}` in the member versus `\boxed{X}` in the canonical:

  - row 7520: answer J, 9,341 identical preceding characters, prefix SHA-256
    `751c8a713b01fabc200abcff5b74f8ade8697f7b30bdfd2086a2d44b28f1508e`;
  - row 7523: answer E, 12,667 identical preceding characters, prefix SHA-256
    `df378e73fe02bac3931db90b70a4f08aec95e2a180ce7b290fe70913c108e9f9`;
  - row 7536: answer B, 5,930 identical preceding characters, prefix SHA-256
    `d19784b55db21caa07740aab02ffe06225b9f1627aee8b31f7ca862963cd1cbf`.

- Member/canonical SHA-256, semantic-judgment SHA-256, and manual-record
  Parquet SHA-256 values are:

  - row 7520:
    `3aabdcc3b5065abb07a5f55f4826e565d0f07684f20f02479f1b87964e4cb392`
    /
    `ad391063d7bc56f3130299aedaf469214b107b3bf3c85edcec078543b7e7d752`,
    `2e6eecba87d25f64dfe9590c8c4d4d5aa7cb4a7d10421607296e275facab158e`,
    `d4072b826882e9ac98eea5caf7f1dda001648b0d88c2f7fd74ba681c5b0a8a3e`;
  - row 7523:
    `273d875fddfabf00be68ee047b85f0a545d553b1addb659e8e3f800c6806f4c2`
    /
    `acd3b7e4d16184594dd87944fe8dd875ff297adc52eb65070a845212f75c4324`,
    `c76018056e48114cd122d8cfd2f4ffcd4d449d3aebc5c88e9c70d527935b56d8`,
    `64ef9613c40cf0ec80e8c01e7cca0d1b3ae9bad2c778e789e6817625a797fccf`;
  - row 7536:
    `cc4d04eedd9c344a57c36a70770dfe65fa28d5954561b41bbcdab77d8f6584d6`
    /
    `3ec1bc2b87cd551d3eb51a6af90a4f2c13b32414a2879f5a7f7618d758f10925`,
    `242c90068126d1a487c601f9125cf94fd0d0216eb43c1a9e4a898e74f38fd856`,
    `4ebf3a80041a1a233e68fa23e8b2567126bda6d7866474d28fd7c441e8a863f1`.

- A separate read-only batch-priority Iris job exactly reread all three full
  source pairs, shared-prefix proofs, semantic evidence, manual rows, Parquet
  bytes, and completion markers.
- Across the stable 640-checkpoint snapshot, all 87 manual records leave:

  - baseline: 65,812 pairs, 41,870 false positives, 23,942 true duplicates;
  - treatment: 15,636 pairs, 8,109 false positives, 7,527 true duplicates;
  - combined: 81,448 pairs, 49,979 false positives, 31,469 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T02:46:41Z — 81,320 pairs verified

- Three additional baseline checkpoints passed independent validation: 384
  pairs, 263 false positives and 121 true duplicates, with no unresolved
  outcomes. Their 796 valid judgments used 797 attempts; one invalid response
  was retried once. Every pair used complete-text direct review.
- The outcome Parquet SHA-256 values for decision-file 34 offsets 4096 through
  4352 are
  `50f61f5b273859e292ce8a1c2a5c5ccebf0ba8148b9856850f77f07e7026757d`,
  `79685a7158535bdeca7d39a2afa2be5f65acc595d6c794df29824d26020f5314`,
  and
  `d27a4f3e86e09bb98a51949176bd9e0c4516e6b3df440a020ba1aa5c6a2281f9`.
- Across the stable 639-checkpoint snapshot, all 84 manual records leave:

  - baseline: 65,684 pairs, 41,755 false positives, 23,929 true duplicates;
  - treatment: 15,636 pairs, 8,109 false positives, 7,527 true duplicates;
  - combined: 81,320 pairs, 49,864 false positives, 31,456 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T02:42:32Z — 80,936 pairs verified

- Two additional baseline checkpoints passed independent validation: 256
  pairs, 196 false positives and 60 true duplicates, with no unresolved
  outcomes. All 569 judgments were valid on the first attempt. One pair was
  chunked and 255 were direct.
- The outcome Parquet SHA-256 values for decision-file 34 offsets 3840 and
  3968 are
  `8dd3da517319f1cea4902a8239bf764eacc26f9a74a88e8d93ca605616818369`
  and
  `52ed2037d8f2b072d1bb5bf7eba8a101a2e77158ad80178758a49d3e1eaf96f0`.
- Across the stable 636-checkpoint snapshot, all 84 manual records leave:

  - baseline: 65,300 pairs, 41,492 false positives, 23,808 true duplicates;
  - treatment: 15,636 pairs, 8,109 false positives, 7,527 true duplicates;
  - combined: 80,936 pairs, 49,601 false positives, 31,335 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T02:40:19Z — 80,680 pairs verified

- Two additional baseline checkpoints passed independent validation: 256
  pairs, 192 false positives and 64 true duplicates, with no unresolved
  outcomes. All 526 judgments were valid on the first attempt. Every pair used
  complete-text direct review.
- The outcome Parquet SHA-256 values for decision-file 34 offsets 3584 and
  3712 are
  `f26badec85e130e45324b49e7dea21c6841c0af37759d81bd044c4acb2dd2e28`
  and
  `2b9e23976c5d80fa40cb76d3634b4dde5a76649bde476f8a6212adc15408a797`.
- Across the stable 634-checkpoint snapshot, all 84 manual records leave:

  - baseline: 65,044 pairs, 41,296 false positives, 23,748 true duplicates;
  - treatment: 15,636 pairs, 8,109 false positives, 7,527 true duplicates;
  - combined: 80,680 pairs, 49,405 false positives, 31,275 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T02:36:04Z — 80,424 pairs verified

- Three additional baseline checkpoints passed independent validation: two
  from partition p1 and one from p2. Their 384 pairs contain 250 false
  positives and 134 true duplicates, with no unresolved outcomes. All 2,784
  judgments were valid on the first attempt. Twenty-six pairs were chunked and
  358 were direct.
- The p1 outcome Parquet SHA-256 values for decision-file 34 offsets 3328 and
  3456 are
  `2bf6a6ba761269883715b5ec0b66a442a80a79298dd70fc8857bed810f30f632`
  and
  `fd79f07fd88493bde1d8026abe9954dc0e9a75e3a1e1cb83eb8509f1f02168fb`.
  The p2 decision-file 67 offset-0 SHA-256 is
  `a696b3896e78c1ea85b0c8fb37cf04f6ac7d80fc9a00e6e450d5449cd477b9cc`.
- Across the stable 632-checkpoint snapshot, all 84 manual records leave:

  - baseline: 64,788 pairs, 41,104 false positives, 23,684 true duplicates;
  - treatment: 15,636 pairs, 8,109 false positives, 7,527 true duplicates;
  - combined: 80,424 pairs, 49,213 false positives, 31,211 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T02:33:22Z — 80,040 pairs verified

- Two additional baseline checkpoints passed independent validation: 256
  pairs, 123 false positives and 133 true duplicates, with no unresolved
  outcomes. All 658 judgments were valid on the first attempt. Two pairs were
  chunked and 254 were direct.
- The outcome Parquet SHA-256 values for decision-file 34 offsets 3072 and
  3200 are
  `12938a1d2282acbbb82e36decf10f7422e6cd43f12c332be125e20f81450eaa2`
  and
  `71ea68113da020992e3a1c09b93afa14f2dd6799f89c04bd546f2b468829bf6e`.
- Across the stable 629-checkpoint snapshot, all 84 manual records leave:

  - baseline: 64,404 pairs, 40,854 false positives, 23,550 true duplicates;
  - treatment: 15,636 pairs, 8,109 false positives, 7,527 true duplicates;
  - combined: 80,040 pairs, 48,963 false positives, 31,077 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T02:29:03Z — 79,784 pairs verified

- Two additional baseline checkpoints passed independent validation: 256
  pairs, 107 false positives and 149 true duplicates, with no unresolved
  outcomes. All 647 judgments were valid on the first attempt. Two pairs were
  chunked and 254 were direct.
- The outcome Parquet SHA-256 values for decision-file 34 offsets 2816 and
  2944 are
  `8dde65ed084df254eea6109e48fa5c89ed35fd306fbdb99b2cfc573d70754263`
  and
  `b97ab61550f6c492cb47a2051c1fd56c18f980e4799679de9a537fd3db88dd8e`.
- Across the stable 627-checkpoint snapshot, all 84 manual records leave:

  - baseline: 64,148 pairs, 40,731 false positives, 23,417 true duplicates;
  - treatment: 15,636 pairs, 8,109 false positives, 7,527 true duplicates;
  - combined: 79,784 pairs, 48,840 false positives, 30,944 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T02:24:51Z — 79,528 pairs verified

- Three additional baseline checkpoints passed independent validation: 384
  pairs, 191 false positives and 193 true duplicates, with no unresolved
  outcomes. All 817 judgments were valid on the first attempt. Every pair used
  complete-text direct review.
- The outcome Parquet SHA-256 values for decision-file 34 offsets 2432 through
  2688 are
  `3bcf6a5c09eda1bc3cd6e4d5e2864621a155e9fe19d003281dd0facb334d05ca`,
  `d7f7a3543b08594d8c0cbca341b0537c041f3d41d40ad9d2cfaf46dc16914c11`,
  and
  `164f4a2d1b3478d55ab9ff8769f3476c2440a725d8e5b2a3e69012124a4fe695`.
- Across the stable 625-checkpoint snapshot, all 84 manual records leave:

  - baseline: 63,892 pairs, 40,624 false positives, 23,268 true duplicates;
  - treatment: 15,636 pairs, 8,109 false positives, 7,527 true duplicates;
  - combined: 79,528 pairs, 48,733 false positives, 30,795 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T02:20:08Z — 79,144 pairs verified

- One additional baseline checkpoint passed independent validation: 128
  pairs, 56 false positives and 72 true duplicates, with no unresolved
  outcomes. All 271 judgments were valid on the first attempt. Every pair used
  complete-text direct review. The outcome Parquet SHA-256 is
  `b0c4755b20e80fa108326570924528ca16256fb93f6d2f714da5261b05c3d23e`.
- Across the stable 622-checkpoint snapshot, all 84 manual records leave:

  - baseline: 63,508 pairs, 40,433 false positives, 23,075 true duplicates;
  - treatment: 15,636 pairs, 8,109 false positives, 7,527 true duplicates;
  - combined: 79,144 pairs, 48,542 false positives, 30,602 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T02:18:01Z — 79,016 pairs verified

- Four additional baseline checkpoints passed independent validation: 512
  pairs, 240 false positives and 272 true duplicates, with no unresolved
  outcomes. All 1,222 judgments were valid on the first attempt. Three pairs
  were chunked and 509 were direct.
- The outcome Parquet SHA-256 values for decision-file 34 offsets 1792 through
  2176 are
  `c052bfa5057e214c6630a6f3b10b5ffba9448ec25e3ff2b4715cdad34cb0df90`,
  `65128157730a740e07869c317fff2b75089e4e5aa134c462cbd4e2eb6afad8ea`,
  `647e533053d56f3bec5dc00d88556be4691477bdc4b130bd323727cf5cf16c19`,
  and
  `a8b41cb1bc1f3bb6dba7aba1f777a431650921fd2295feb4b90f65353f94dd9c`.
- Across the stable 621-checkpoint snapshot, all 84 manual records leave:

  - baseline: 63,380 pairs, 40,377 false positives, 23,003 true duplicates;
  - treatment: 15,636 pairs, 8,109 false positives, 7,527 true duplicates;
  - combined: 79,016 pairs, 48,486 false positives, 30,530 true duplicates.

- All four batch-priority 2-H100 workers continue serving requests. Their 12
  root, broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T02:15:50Z — 78,504 pairs verified

- Six additional baseline checkpoints passed independent validation: 768
  pairs, 575 false positives and 193 true duplicates, with no unresolved
  outcomes. All 1,591 judgments were valid on the first attempt. Every pair
  used complete-text direct review.
- The outcome Parquet SHA-256 values for decision-file 34 offsets 1024 through
  1664 are
  `069b7e5378792abc66bc70a49ed40c22c0f59df9f98b7a8b660e0d202a14878a`,
  `7d808c4f28a8b5557e0af11abbd19458cc5dd14c747212eee94e18d637ce6baf`,
  `6511e8fc137dab33034c922c3457421b3a2263fbce48e9e5acd9e0aff48d0ccc`,
  `f7473e854de00963cf17344e51ae51a1e7e977ad5c9f47b38cb2443b3f7bb8e1`,
  `a26547dc63d95c90d541cce82a1c39cce6cb1bb0debacd9e3e10b9b423dc7d15`,
  and
  `0c14e62a518633b81a5f0bb2a7dc3746e339c106d85906ce92badcbf6bbc96cc`.
- Across the stable 617-checkpoint snapshot, all 84 manual records leave:

  - baseline: 62,868 pairs, 40,137 false positives, 22,731 true duplicates;
  - treatment: 15,636 pairs, 8,109 false positives, 7,527 true duplicates;
  - combined: 78,504 pairs, 48,246 false positives, 30,258 true duplicates.

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
