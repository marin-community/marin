---
topic: echo-pre-seed
description: Persist Echo searches, grade observed queries, and pre-seed durable answers
author: russell.power
---

# Echo Pre-Seed Search Quality: Research Logbook

## Scope

- Goal: Preserve Echo search traffic, grade a fixed observed-query baseline, and stage supported wiki and core-document improvements for bulk review.
- Primary metrics: Query-level 0–10 answer quality, direct-answer rank, held-out paraphrase rank, and existing benchmark regressions.
- Constraints: Keep query history indefinitely; omit network identity; stage all content before bulk upload; reserve OPS.md, AGENTS.md, and skills for frequent workflows.
- Coordinating issue: https://github.com/marin-community/marin/issues/8203
- Experiment prefix: `ECHO-SEED`.

## Current TL;DR

- Cloud Logging retains two days of Echo request URLs. The preserved slice contains 355 federated searches with 338 unique normalized queries and 94 grep requests with 88 unique patterns.
- Echo stores 51 feedback submissions and 138 per-result grades. These judgments were agent-authored and will not calibrate the new grader.
- The initial federated discovery and regression pool contains 413 unique queries: 338 retained searches, one feedback-only query, and 74 existing benchmark cases.
- Search execution persistence is live on Echo revision `echo-api-00024-vtc`. The first verified execution stored the exact query, authenticated author, selected domains, repository commit, duration, and five ranked results.
- The 413-query replay is complete. Luna/medium has graded every case with a fresh rubric; Sol/medium is adjudicating low, middling, and documentation-routed cases before content drafting.

## Current Baseline

- Date: 2026-08-12.
- Code: `288d91aa23461d1fbc996fef98ac45757ac8910b`.
- Observed federated traffic: 355 executions, 338 normalized unique queries, all HTTP 200.
- Existing feedback: 51 queries, 138 grades, 45 graded submissions, six explanation-only submissions.
- Existing source-grounded benchmark: 74 queries.

## Hypothesis Queue

### Active

- `ECHO-SEED-001`: Persisting ranked search executions will make feedback and before/after comparisons replayable. Next test: API persistence tests for successful, empty, and domain-filtered searches.
- `ECHO-SEED-002`: A fresh Luna/medium rubric can separate content gaps from retrieval failures over the 413-query pool. Next test: blind double-grade an initial batch and inspect disagreement.
- `ECHO-SEED-003`: Intent-clustered wiki and core-document additions will improve direct-answer rank on original queries and held-out paraphrases without regressing the 74-case benchmark. Next test: first staged wave after grading.

### Blocked

- None.

### Falsified / Dead End

- None.

### Promoted

- None.

## Decision Log

- 2026-08-12: Retain query history indefinitely. Storage cost is negligible at current volume.
- 2026-08-12: Use a fresh grading rubric; existing feedback was agent-authored.
- 2026-08-12: Fetch full result sources lazily during grading and cache by stable source identity or content hash.
- 2026-08-12: Put frequent, everyday workflows in OPS.md, AGENTS.md, or skills. Use wiki notes for less frequent events; overlap that expands core guidance is acceptable.
- 2026-08-12: Stage every wiki and core-document proposal, summarize each wave, then bulk upload after review.
- 2026-08-12: Core-document improvements are in scope.
- 2026-08-12: Do not add a historical backfill endpoint or import result-less log rows. Replay each unique preserved query through normal search so the durable record contains its current ranked results.

## Negative Results Index

- Cloud Logging cannot recover the complete history: the live `_Default` bucket retains two days.

## Entry Log

### 2026-08-12 19:40 UTC - ECHO-SEED-001 traffic and feedback canvass

- Hypothesis: Existing infrastructure contains enough query history to bootstrap a useful discovery set.
- Commit Hash: `288d91aa23461d1fbc996fef98ac45757ac8910b` before implementation.
- Commands: Read-only `gcloud logging read` over `echo-api`; `_Default` bucket inspection; aggregate SQL over `search_feedback` and `search_feedback_grades`; benchmark manifest inspection.
- Config: `hai-gcp-models`, Cloud Run service `echo-api`, Cloud SQL database `context`.
- Result: Preserved 355 federated executions and 94 grep executions from the two-day window. Found 338 unique federated queries, 51 feedback queries with one query absent from the log slice, and 74 non-overlapping benchmark queries.
- Interpretation: The 413-query federated pool supports a first grading pass. Future searches require database persistence because request logs expire.
- Next action: Add search execution and ranked-result persistence, then import the sanitized retained slice.

### 2026-08-12 21:25 UTC - ECHO-SEED-001 permanent search retention deployed

- Hypothesis: Every Echo search can retain enough context to reproduce and grade the result set later.
- Commit Hash: Uncommitted branch deployment from `weaver/echo-seed-articles`; schema migration based on `288d91aa23461d1fbc996fef98ac45757ac8910b`.
- Commands: Applied `m0011_search_history` with `infra/echo/migrate.py`, previewed the `marin-echo` stack, deployed with Pulumi, ran one live federated search, and exported its execution.
- Config: Echo revision `echo-api-00024-vtc`; indexed repository commit `26b338faddffbab89cc497d3fd0c64570842ffa1`.
- Result: Execution 1 stored `how do I deploy Iris?`, caller attribution, four selected domains, five ordered result snapshots, and an 18.6-second server duration. The response exposed execution ID 1 for feedback linkage.
- Interpretation: ECHO-SEED-001 is supported. Future search and feedback data is replayable without relying on expiring Cloud Logging.
- Next action: Replay the deduplicated 413-query manifest at four workers, then grade the persisted executions.

### 2026-08-12 23:10 UTC - ECHO-SEED-002 baseline replay and Luna grading

- Hypothesis: The retained traffic and benchmark pool can be replayed through the ordinary search contract and graded without a separate history-import API.
- Commit Hash: Uncommitted branch deployment from `weaver/echo-seed-articles`; indexed repository commit `26b338faddffbab89cc497d3fd0c64570842ffa1`.
- Commands: Replayed `echo-baseline-manifest.jsonl` through `GET /api/federated-search`; reconciled responses against `GET /api/search-executions`; graded the joined snapshots in three Luna/medium batches plus a tail batch.
- Config: 413 unique queries, result limit 10, at most four concurrent requests, fresh query-level 0–10 rubric.
- Result: All 413 cases have durable execution IDs and ranked snapshots. The raw Luna pass produced 198 keep, 118 ranking/index, 35 wiki, two core-doc, 20 review, and 40 ephemeral/no-answer dispositions. The 106 low, middling, review, and documentation-routed cases require Sol adjudication before drafting.
- Interpretation: Normal search plus durable reconciliation is sufficient for historical replay. The first raw content count is intentionally an upper bound because snippet-only grading confuses retrieval defects and ephemeral requests with documentation gaps.
- Next action: Complete Sol adjudication, cluster supported gaps, then stage evidence-backed wiki and core-document proposals in Weaver.
