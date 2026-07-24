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
