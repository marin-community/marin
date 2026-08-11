---
topic: harrier-fuzzy-duplicates
issue: https://github.com/marin-community/marin/issues/8162
description: Produce Harrier embeddings for noncanonical fuzzy-duplicate documents.
author: Rafal Wojdyla
---

# Harrier Fuzzy-Duplicate Embeddings: Run Logbook

## Run Contract

- DRI: Rafal Wojdyla.
- Goal: Produce co-partitioned Harrier embeddings for noncanonical fuzzy-duplicate documents in all 292 Datakit sources.
- Stop or escalation criteria: Escalate repeated application failures, output conflicts, or no worker allocation after 30 minutes.
- Issue: https://github.com/marin-community/marin/issues/8162
- Source: `48ef14ea8e62d15fee2d1db101feae216ca04fbc` on `origin/rav/harrier-all`.
- Output root: `s3://marin-us-east-02a/marin/datakit/embed/harrier-fuzzy-duplicates/`.
- W&B: Not used by this embedding job.
- Checkpoints: Not used. Zephyr keeps completed Parquet shards and skips them after a restart.
- Completion target: 292 source artifacts across two deterministic partitions.
- Monitor owner: Codex, with a 15-minute check cadence after the initial checks.

## Entry Log

### 2026-08-11 19:54 UTC - Launch plan

- Commit hash: `48ef14ea8e62d15fee2d1db101feae216ca04fbc`.
- Dirty tree: No.
- Runtime source: Iris repository sync from `origin/rav/harrier-all`.
- Hardware: 96 H100 TEI workers in `cw-us-east-02a` and 96 H100 TEI workers in `cw-rno2a`.
- Source concurrency: One source in each region.
- East command: `uv run iris --cluster marin job run --no-wait --target-cluster cw-us-east-02a --job-name harrier-fuzzy-dups-east-p0-20260811 --priority interactive --cpu 2 --memory 8GB --disk 16GB --enable-extra-resources --extra datakit -e MARIN_PREFIX s3://marin-us-east-02a/marin -- python -m experiments.datakit.embeddings.harrier.run --document-set fuzzy_duplicates --partition-index 0 --partition-count 2 --tei-instances 96 --max-concurrent 1`.
- RNO command: `uv run iris --cluster marin job run --no-wait --target-cluster cw-rno2a --job-name harrier-fuzzy-dups-rno-p1-20260811 --priority interactive --cpu 2 --memory 8GB --disk 16GB --enable-extra-resources --extra datakit -e MARIN_PREFIX s3://marin-us-east-02a/marin -- python -m experiments.datakit.embeddings.harrier.run --document-set fuzzy_duplicates --partition-index 1 --partition-count 2 --tei-instances 96 --max-concurrent 1`.
- Result: The launch is not started.
- Next action: Commit and push this run contract, then submit both Iris jobs.
