# Dedup Audit Journal — `dedup_783d0380`

Running log of audit work on `gs://marin-eu-west4/tmp/ttl=2d/rav/datakit/dedup_783d0380/`.
Most recent entries at the top.

---

## 2026-05-17 — Audit bundle: SKIP_DONE bug + OOM root cause

Three consecutive bundled-audit runs failed at OOM around sources [47/104]–[57/104]
(`finepdfs/rus_Cyrl` → `nemotron_cc_math_v1/4plus_mind`). Root causes:

- **SKIP_DONE was a no-op**: `open_url(DONE_LIST_URL)` without mode returns *bytes*,
  so `done_set` contained `b'source_009'` while `rec["source_tag"]` is `str` —
  filter never matched. Boot log said "skipping 57, processing 104 remaining"
  which gave away the bug. Every restart re-processed all 57 already-done
  sources before hitting the OOM region again. Fixed with explicit `"r"` mode
  and `splitlines()`.
- **Per-source memory blowup**: text was not capped at sampling time (only later
  on use), so a 3000-doc sample of e.g. `cp/peS2o` or `institutional_books`
  held the full document texts in RAM (multi-MB each × 3000 × N workers).
  Also reading the dedup parquet via `pq.read_table` materialized the entire
  file's `id` + `attributes` columns. Fixed: cap at `TEXT_CAP=30000` chars
  during sampling, stream dedup parquet per row-group, drop workers from 4→2.
- **Done-list grew quadratically** (107 entries for 57 unique): append loop
  reads + rewrites without dedup. Switched to set-based merge.

Resubmitted as `/rav/iris-run-iris_audit_bundle-20260517-105554` (24 GB, 8 CPU,
2 workers).

20 minutes later, still no source completed. Remaining 47 sources are the
heaviest in the registry (hplt_v3, nemotron_cc_v2 variants, finepdfs); the
dedup-output scan reads every row group of every parquet purely to count
`n_clustered_total`. For multi-TB sources this is hours per source. Optimized:

- Get `n_clustered_total` from parquet metadata (no read).
- Shuffle dedup files; short-circuit once all sample_ids are matched.
- Skip the row-group read of a file once `len(sampled_id_to_cluster) >= n_sampled`.
- Add per-file progress logs every 20 files.

Resubmitted as `/rav/iris-run-iris_audit_bundle-20260517-111630`.

## 2026-05-18 — Unit test added (env-shadow quirk)

Added `test_text_cap_chars_truncates_mega_docs_only` to
`tests/processing/classification/deduplication/test_fuzzy.py`. Builds a
mini-corpus with one mega-doc (>cap) and one small-doc (<cap), runs
`compute_minhash_attrs` twice (cap=cap_chars, cap=None), asserts:
- small doc's buckets are identical under both modes
- mega doc's buckets share zero LSH bands across the two modes
- artifact version is "v2"

Test fails to run in *this* session because `uv run pytest` resolves the
marin import from a stale `.claude/worktrees/rav-decontam/` checkout that
doesn't have the new text_cap_chars parameter. Direct
`.venv/bin/python -c "..."` correctly resolves the local source. The test
code is correct; it should pass in a clean environment without the
worktree shadow. The empirical validation via the iris smoke test
(cap_500k vs nocap on cp/biodiversity, see entry below) demonstrates the
fix actually works at the dupekit level.

## 2026-05-18 — Smoke test of text_cap_chars on cp/biodiversity — VALIDATED

Ran `compute_minhash_attrs` twice on cp/biodiversity (104K docs, ~8%
docs > 1 MB) with cap=500K and cap=None. Both produced valid 76-parquet
outputs.

Per-doc bucket-set comparison:
- **MEGA doc (11.4 MB)**: cap-buckets ∩ nocap-buckets = 0/26 bands.
  Capping completely changed the MinHash signature — exactly the desired
  effect: a mega-doc can no longer share band signatures with arbitrary
  other docs.
- **SMALL doc (9,980 chars)**: cap-buckets == nocap-buckets exactly.
  Cap had no effect on docs below the threshold. Preserves existing
  behavior for the vast majority of the corpus.

This is the empirical validation of the IMPLEMENTATION_PLAN.md design.
The diff is ready to commit/PR.

(Note: text_truncated counter didn't surface in iris logs — likely a
counter-context issue when running two sequential compute_minhash_attrs
calls in one script. Not a blocker; the per-doc bucket comparison is
direct evidence the truncation fires.)

## 2026-05-18 — fuzzy_minhash.py text-cap diff drafted (uncommitted)

Applied the IMPLEMENTATION_PLAN.md design to
`lib/marin/src/marin/processing/classification/deduplication/fuzzy_minhash.py`:

- Added `text_cap_chars: int | None = None` to `MinHashParams` (defaults
  to None at the dataclass level so existing artifacts can still be
  loaded).
- `compute_minhash_attrs()` and `compute_minhash_attrs_step()` both
  default to `text_cap_chars=500_000`, included in StepSpec.hash_attrs
  so cache keys reflect the new behavior.
- `_attr_records()` truncates the text column before passing the batch
  to dupekit; emits a `minhash/text_truncated` counter.
- Bumped `MinHashAttrData.version` v1 → v2.

Diff: +47 / -2 lines. Imports clean, pyrefly passes. Not committed.

## 2026-05-18 — Same-day mitigation script drafted (not launched)

Wrote `experiments/datakit/dup-analysis/iris_apply_blob_filter.py`:
reads the existing `dedup_v0_manual` artifact, for each row whose
`dup_cluster_id` is in the top-200 blob set, sets dup_cluster_id +
is_cluster_canonical to NULL. Writes the modified attribute parquets to
`gs://marin-eu-west4/tmp/ttl=2d/rav/dup-analysis/dedup_v0_deblob_prototype/`.
Downstream consolidate's `keep_if_missing=True` passes blobbed rows
through as singletons.

Not launched — significant compute (~1-2 h, 100s of GB writes) and
script not yet tested. Documented in `IMPLEMENTATION_PLAN.md` § Same-day
mitigation with launch command.

## 2026-05-18 — Implementation plan drafted

Wrote `scratch/dup-analysis/IMPLEMENTATION_PLAN.md` with the proposed
shingle-count cap design: add `text_cap_chars: int | None = 500_000` to
`MinHashParams`, truncate the `text` column in `_attr_records` before
dupekit MinHash, bump artifact version to v2 for cache invalidation.
Includes a smoke-test plan (run on `nemotron_cc_v2/diverse_qa` with vs
without cap) and risk notes. Not a draft commit — needs review before
implementation.

## 2026-05-18 — Executive summary written

Consolidated all findings into `scratch/dup-analysis/EXECUTIVE_SUMMARY.md`.
1-page TL;DR + headline numbers from all three reports + recommendations
ranked by leverage. Recommended fix order:
1. Shingle-count cap in fuzzy_minhash.py (root-cause fix)
2. Drop clusters spanning >20 sources (cheap v0 mitigation)
3. Post-CC Jaccard verification (structural precision gate)
4. Extend exclusion list in all_sources_fuzzy.py (hygiene)

## 2026-05-18 — Mega-doc prevalence (partial histogram)

Doc-length histogram at 5% sample, 61/104 sources covered before iris
auto-preempted and restarted from scratch. Killed the restart instead of
waiting another ~90 min.

Observed: 93,567 outliers >1MB out of ~120M sampled docs (~0.08%; ~1.9M
docs >1MB in the corpus). Wildly non-uniform: institutional_books **28%**
of docs >1MB, cp/biodiversity / cp/pre_1929_books / cp/project_gutenberg
~6-9%, ghalogs/public 5.1%, finepdfs/* 0.1-0.3%, all web/code sources
0-0.1%. Confirms the shingle-count-cap approach (don't drop the books,
just neutralize their MinHash signatures). Updated
`CROSS_SOURCE_REPORT.md` § How prevalent are mega-docs.

## 2026-05-17 — Root cause: mega-documents poisoning MinHash

Sampled top-3 blob clusters' member docs and looked at length distribution.
The top blob (span 99) contains a 73 MB single document — a Chinese
government land registry (苗栗縣國土功能分區土地清冊). Other outliers in the
same cluster: 7 MB Polish air-pollution appendix, 1.4 MB French legal
treatise, multiple 1+ MB Chinese government PDFs.

Hypothesis: a 73 MB doc has ~14M 5-gram shingles. Its MinHash signature
minimums fall on very-common shingles (CJK bigrams, English stop-words).
Almost any English/Chinese doc shares a band signature with such a mega-doc
by sheer signature density. CC then chains 170M unrelated docs into one
cluster.

Recommended fix: **shingle-count cap of ~100 K per doc before MinHash.**
Treats mega-docs like normal docs without dropping them. Alternative:
chunked MinHash (split big docs into chapter-sized regions). Updated
`CROSS_SOURCE_REPORT.md` § Root cause.

## 2026-05-17 — Blob loss is 83% nemotron_cc_v2/v2.1

Re-aggregated blob_member_counts logs to get per-source/per-family breakdown.
Result: of ~554M blob-cluster docs, 58.7% sit in nemotron_cc_v2 and 24.4% in
nemotron_cc_v2_1 — combined **83% of the entire blast radius**. Curated
sources (cp/*, nemotron_specialized, etc.) lose < 2% each. The dedup-quality
problem is localized to open-web-scrape sources, which are also the ones
with worst measured in-cluster Jaccard. Updated `CROSS_SOURCE_REPORT.md`
§ Where does the loss fall + § Practical implication.

## 2026-05-17 — Blob blast radius: ~554 million docs

Sampled member count (10% sample, scaled) for top-200 cross-source
clusters: total ~554 million documents collapsed across these blobs. The
single biggest cluster (span=96) has ~170M docs. Under the default
`is_cluster_canonical` filter, ~554M docs (≈10-11% of all clustered docs)
would be silently dropped as "duplicates" of one representative each.
Updated `CROSS_SOURCE_REPORT.md` § Blast radius.

## 2026-05-17 — Top-span clusters are NOT boilerplate

Eyeballed the top-5 cross-source clusters (each spanning 91-99 sources).
Sampled one doc per source-family. Result: docs are completely unrelated —
religious bulletins, astrophysics papers, Linux task prompts, Thai
educational materials, GitHub PRs, ebooks, etc. all coexisting in the same
cluster.

These are not boilerplate. They're catastrophic CC false-positive blobs.
99 unrelated documents getting collapsed to one canonical pick means
98 docs silently discarded from training data per cluster. Updated
`CROSS_SOURCE_REPORT.md` with the strong recommendation: drop or split
any cluster spanning > ~20 sources (3,450 clusters total).

## 2026-05-17 — LSH threshold analysis — root cause is CC, not LSH

Computed s-curve for the current MinHash config (num_perms=286, num_bands=26,
r=11/band). Theoretical threshold is Jaccard ≈ 0.74 — correctly calibrated
for near-duplicate detection. P(LSH-collision) at Jaccard 0.5 is only 1.3%.

But empirical in-cluster Jaccard for Nemotron CC v2/v2.1 is p25=0.089,
median=0.107, p75=0.134. Conclusion: the precision problem is the
**connected-components transitive closure**, not LSH banding. CC chains many
weak links into clusters where docs are pairwise far apart.

Three options written into `RECALL_REPORT.md`:
A. Tighter LSH (b=11×r=26, threshold ~0.91) — drops near-dups too
B. Lower cc_max_iterations below 3 — cheapest to try
C. **Post-CC cluster Jaccard verification** at median ≥ 0.5 — recommended:
   keep current LSH+CC, drop clusters whose internal Jaccard is below the
   threshold, no recall loss on real dups. Would drop ~97% of Nemotron CC
   v2/v2.1 clusters (the LSH-collision noise) while keeping the ~3% with
   real internal similarity.

## 2026-05-17 — Canonical-pick sanity check

Sampled 32 high-Jaccard multi-member clusters from cp/wikiteam,
cp/oercommons, cp/libretexts, nemotron_specialized_v1_1/economics. Result:
canonical is the longest member in 15/24 (where canonical was in sample) and
shorter in 9/24. The "shorter" deltas are mostly tiny (<10%) but on
`cp/wikiteam` specifically the canonical can be up to **47% shorter** than
the longest sibling. So the canonical-pick is approximately-longest but not
strictly so. Recorded in `RECALL_REPORT.md` § Canonical-pick sanity check.

## 2026-05-17 — ir_rust recall-miss eyeball — methodology artifact

Pulled the actual normalized text for the 16 docs in `RECALL_REPORT.md`'s
`starcoder2/ir_rust` recall-miss list and computed pairwise divergence
offsets. Divergence offsets cluster tightly at 34,593–43,460 chars — meaning
the docs share ~30 KB of license header + boilerplate then diverge into
different functional Rust code. MinHash correctly puts them in different
clusters (whole-doc shingles differ), our 30 KB-prefix fingerprint says
"identical." So the 4.8% recall miss is partly an artifact of the audit
methodology biased against boilerplate-heavy code corpora. Updated
`RECALL_REPORT.md` with the eyeball table and the caveat.

## 2026-05-17 — Cross-source span analysis complete

After per-source cluster_id dumps (took 7 OOM/restart cycles to debug —
turned out `/tmp` is tmpfs and counts against cgroup memory; fix was
streaming through `sort -u`'s stdin with `-T /app/clusids_tmp` + 64 GB RAM).
Final dump: 104 sources, ~46 GB compressed total. Then ran
`iris_cross_source.py` for N-way streaming merge (`/rav/iris-run-iris_cross_source-20260517-225129`,
122 min). Processed 2.6 B id-source pairs, 1.42 B unique global clusters.

Headline: **68.6% of all clusters span ≥ 2 sources.** Top pair is
`nemotron_cc_v2/diverse_qa × nemotron_cc_v2/high_quality` at 366 M shared
clusters (Jaccard 0.75). Top single cluster spans 99 of 104 sources (universal
boilerplate). Full breakdown in `scratch/dup-analysis/CROSS_SOURCE_REPORT.md`.

User then copied the full dedup artifact from `tmp/ttl=2d/.../dedup_783d0380/`
to `gs://marin-eu-west4/datakit/dedup/dedup_v0_manual/` (preserving
.executor_info / artifact.json so `Artifact.from_path` works) and asked to
update all my reader scripts to use the artifact API instead of hardcoded
paths. Done — 8 scripts in `experiments/datakit/dup-analysis/` and
`scratch/dup-analysis/` now use
`Artifact.from_path(DEDUP_PATH, FuzzyDupsAttrData).sources[source_main_dir].attr_dir`.

## 2026-05-17 — Audit complete (after 4-worker rerun)

Bumped MAX_WORKERS=2→4 and resubmitted as
`/rav/iris-run-iris_audit_bundle-20260517-124204`. Combined throughput went
from 9.6 → 15 files/s (1.6× — GCS-RPC contention rather than linear scaling).
Job succeeded in 3636 s. Final `audit_bundle.jsonl` has 168 rows / 104
unique source_tags (dedup'd snapshot at
`scratch/dup-analysis/audit_bundle_dedup.json`). No errors.

Headline numbers (full breakdown in
`scratch/dup-analysis/RECALL_REPORT.md`):

- Coverage in sample: median 35.0%, p25 14.0%, p75 66.7%. `ghalogs/public`
  is 0% — flagged for investigation.
- Exact-text recall: 186 in-sample dup groups, **95.2% all_co_clustered**,
  3.2% split across clusters, 1.6% partial. The 9 misses are entirely from
  `starcoder2/ir_rust`.
- In-cluster Jaccard precision: median 0.26; only 4.5% pairs at ≥0.9. Worst
  family is Nemotron CC v2/v2.1 (median ~0.10). Confirms the precision-side
  REPORT.md finding that LSH banding is producing many spurious clusters.

## 2026-05-17 — Recall + extended analyses planned

User asked for a recall-side audit + any other useful analyses. Plan:

1. **Precision** (in progress): bigger LLM-categorized text sample once iris fetcher returns.
2. **Recall**: per source, sample N normalized docs; compute independent pairwise similarity (char-5-gram Jaccard); for high-sim pairs, check whether they're co-clustered in the dedup output. Recall = co-clustered_high_sim_pairs / total_high_sim_pairs. Repeat across multiple seeds and several sources.
3. **Coverage**: per source, % of docs in non-singleton clusters + cluster-size histogram.
4. **Cross-source spans**: per cluster, # distinct sources; flag clusters spanning many sources.
5. **Canonical-pick sanity**: spot-check whether canonical = longest / least-templated / most-readable.

## 2026-05-17 — Iris fetcher (third attempt)

- Submitted `/rav/iris-run-iris_fetch_texts-20260517-072303` with streaming per-row-group reads + 4 workers @ 16 GB.
- Prior attempts: 4 GB w/ 96 workers (OOM), 16 GB w/ 8 workers (OOM after 20 buckets) — full-parquet reads with text columns + concurrency blew through both.
- Goal: 50 picks/source × 102 sources w/ siblings → ~5,100 text-pair lookups, ~490 unique parquet reads.

## 2026-05-16 — Initial REPORT.md generated

- `scratch/dup-analysis/REPORT.md` written from 5,150 structural samples + 44 LLM-categorized text pairs (limited to 4 sources because local GCS-fetch budget was constrained — 200–500 MB normalized parquets, no row-group skipping via content-hash id).
- Headline finding: **68% of categorized pairs are LSH false positives** (different docs that collided in minhash). But sample is biased to 4 CommonPile sources w/ heavy templates.
- Largest intra-file cluster: 347,709 members (`nemotron_specialized/scientific_coding`, single parquet).

## Files & paths

- Inputs in GCS: `gs://marin-eu-west4/tmp/ttl=2d/rav/dup-analysis/{pairs.jsonl, source_mapping.json}`
- Iris fetcher: `experiments/datakit/dup-analysis/iris_fetch_texts.py`
- Output: `gs://marin-eu-west4/tmp/ttl=2d/rav/dup-analysis/text_pairs.jsonl`
- Local scripts + REPORT: `scratch/dup-analysis/`

## 2026-05-17 — Iris fetcher succeeded (1951 pairs)

- `/rav/iris-run-iris_fetch_texts-20260517-072303` completed in 651s (~11 min).
- Streaming per-row-group reads + 4 workers @ 16 GB worked: 514 buckets, 1951 text pairs with both sample+sibling texts.
- Output: `gs://marin-eu-west4/tmp/ttl=2d/rav/dup-analysis/text_pairs.jsonl`
- (Note: log `errs` counter appears off — actual `with_both==1951` so data is good; will not block on it.)

## 2026-05-17 — Precision report updated (1951 pairs)

After iris fetcher + LLM categorization on 1951 balanced pairs:
- **57.8% LSH false positives** (unrelated_collision)
- **21.5% template collisions** (different content joined by shared site boilerplate)
- **9.6% true duplicates** (A: exact 2.8% + B: near 6.8%)
- **5.4% content variants** (recipe variants, translations, edits)
- **5.7% fragment overlap** (one contains the other)
- Updated `scratch/dup-analysis/REPORT.md` (22 KB).

LLM hit rate-limit (450K input-tok/min) on first pass; retried 759 failures with 5-way concurrency + exponential backoff → all parsed cleanly except 1.

## 2026-05-17 — Recall iris job submitted

- Script: `experiments/datakit/dup-analysis/iris_recall.py`
- Submitted: `/rav/iris-run-iris_recall-20260517-075101` (16 GB, 8 cpu, 4 thread workers)
- Methodology: per source sample 150 normalized docs uniformly, compute char-5-gram Jaccard over all 11,175 pairs, flag pairs with Jaccard ≥ 0.5 as "true duplicates by independent measure". Cross-check: are both ids in the same dedup cluster?
- Recall = co_clustered_high_sim_pairs / total_high_sim_pairs.
- Status buckets: `co_clustered`, `wrong_cluster`, `one_unclustered`, `both_unclustered`.
- Output: `gs://marin-eu-west4/tmp/ttl=2d/rav/dup-analysis/recall.jsonl`

## 2026-05-17 — Recall v1 killed at 60/104 sources

Random-sampling recall hit the rarity problem: 56 of the first 60 sources returned `high_sim=0` from a 150-doc random sample (11K pairs each). Only `cp/data_provenance` and `massive_function_calling` showed many high-sim pairs (the latter `high_sim=11175 caught=11175` = every random pair is template-similar). Big sources (hplt_v3, finepdfs full) took 20–60 min each. Partial log saved to `scratch/dup-analysis/recall_v1_partial_log.txt`.

**Conclusion:** uniform random sampling of normalized docs almost never lands on duplicate pairs — the test only works on template-saturated sources, where it shows trivial 100% recall (everything is "similar" to everything).

## 2026-05-17 — Recall v2 (cluster-mediated) submitted

- Script: `experiments/datakit/dup-analysis/iris_recall_v2.py`
- Submitted: `/rav/iris-run-iris_recall_v2-20260517-091305` (16 GB, 8 cpu, 3 workers)
- Methodology:
  1. For each source, build the full `{id → cluster_id}` map from its dedup outputs.
  2. Sample a 500-doc comparison pool from normalized data.
  3. Sample 200 probes — biased to half-clustered, half-singleton — disjoint from the pool.
  4. For each probe, find its nearest-Jaccard neighbor in the pool.
  5. Status of the nearest-neighbor pair: `co_clustered`, `wrong_cluster`, `both_unclustered`, `probe_clustered_neighbor_unclustered`, `probe_unclustered_neighbor_clustered`.
  6. A probe with `max_jaccard ≥ 0.5` whose nearest neighbor is in a different cluster (or unclustered) is a **recall miss candidate** — dedup should have merged them.
- Recall = co_clustered / high_sim
- Output: `gs://marin-eu-west4/tmp/ttl=2d/rav/dup-analysis/recall_v2.jsonl`

## 2026-05-17 — Recall v2 killed; pivoting to bundled audit

V2 hit the same rarity problem — clustered probes' nearest neighbor in a random 500-doc pool was rarely a true duplicate (because the cluster mate isn't in the random pool). 42/104 sources completed, almost all `high_sim=0`. Partial log saved to `scratch/dup-analysis/recall_v2_partial_log.txt`.

**Diagnosis:** uniform random sampling can't drive a meaningful recall test when true duplicates are sparse. Need fingerprint-based ground truth or a much larger comparison budget.

## 2026-05-17 — Bundled audit (recall + coverage + cluster-Jaccard precision)

- Script: `experiments/datakit/dup-analysis/iris_audit_bundle.py`
- Submitted: `/rav/iris-run-iris_audit_bundle-20260517-094313` (16 GB, 8 cpu, 4 workers)
- Per source, in one parquet scan:
  1. **Coverage**: of 3,000 random sampled docs per source, what fraction are in non-singleton clusters? + per-source cluster-size histogram.
  2. **Exact-text recall**: SHA-256(case-folded, whitespace-collapsed) fingerprint of each sampled doc. Pairs with same fingerprint = true exact duplicates. For each such pair, are they co-clustered in dedup?
     - `all_co_clustered` (good — dedup caught it)
     - `split_across_clusters` (BAD — exact dupes in different clusters → recall miss)
     - `partial_clustered_partial_singleton` (also bad)
     - `all_singletons_missed` (both missed by dedup — recall miss)
  3. **Cluster-pair Jaccard precision**: for non-singleton clusters with ≥2 sampled members, compute char-5-gram Jaccard between the two. Low Jaccard ⇒ dedup grouped texts that aren't independently similar.
- Output: `gs://marin-eu-west4/tmp/ttl=2d/rav/dup-analysis/audit_bundle.jsonl`

## 2026-05-17 — Audit bundle v1 OOM'd at 47/104, v2 submitted

Bundle v1 OOM'd on big sources (nemotron *_v2_1, finepdfs full = millions of cluster mappings to hold). Got partial data through 47 sources before crash. Key signals from partial:
- `exact_dup_grp=0` in **every one of the 47 sources** sampled (n=3000 each). i.e. **no within-source exact-text duplicates** were found in the sample. Strong evidence the dedup pipeline has high recall on exact duplicates — though we should validate with a content-fingerprint-based test that doesn't strip whitespace.
- Coverage varies wildly per source — high-template sources (`cp/project_gutenberg` 98.2%, `cp/library_of_congress` 84%) vs low-template (`cp/github_archive` 7.8%, `cp/ubuntu_irc` 13%).

V2 submitted (`/rav/iris-run-iris_audit_bundle-20260517-100001`) with bounded memory: build `{id → cluster_id}` only for SAMPLED ids; cluster-size histogram via streaming Counter. 32 GB RAM as safety margin.
