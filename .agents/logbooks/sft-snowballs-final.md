---
topic: sft-snowballs-final
issue: https://github.com/marin-community/marin/issues/8977
description: Five-way Snowball LCE SFT and evaluation campaign derived from issue 8225.
author: benfeuer
---

# Snowball LCE Final: Task Logbook

## Scope

- Goal: publish five Snowball LCE bases, run the full Chat/Thinking/OpenCode/Nemotron-Terminal SFT graph for each on RNO2A, and evaluate every checkpoint under the fixed policy.
- Primary metrics: 17-task non-agentic results for all 25 checkpoints and matching-harness agentic reward for ten final checkpoints.
- Constraints: RNO2A interactive priority; preserve per-base architecture; no unapproved cross-region transfer over 10 GB; reproducible pinned inputs and durable artifacts.
- Coordinating issues: [#8977](https://github.com/marin-community/marin/issues/8977), [#8225](https://github.com/marin-community/marin/issues/8225), [#7958](https://github.com/marin-community/marin/issues/7958).

## Current TL;DR

- Fixed policy written. Five native sources identified. Two base HF uploads are validated and three `/held` uploaders remain active. Two RNO2A smoke attempts failed before model load; fixes and exact agentic-stage cache wiring are locally validated. WIP implementation commit `dfd8a1518f` is pushed and its full safe test suite is green.

## Entry Log

### 2026-09-08 20:31 EDT - Policy reconstruction

- Hypothesis: first-class Snowball HF initialization can avoid copying five roughly 1 TB native GCS checkpoints to the RNO2A region while preserving each model's `qk_mult`.
- Commit Hash: `1234cbe4bfbcdc24c1bcfb20741f557c1eec567e`
- Command: read-only inspection of issues #8977/#8225/#7958, PR #8172 and current training/evaluation launchers.
- Config: five step-157000 262K-context bases; Base -> Chat -> Thinking -> two independent 1,888-step agentic branches; evaluation policy recorded in the external experiment directory.
- Result: `POLICY.md` and `STATE.md` created; two in-progress `/held` uploads observed; no training submitted.
- Interpretation: validate the HF path before GPU fan-out; never race another owner's upload or perform an unapproved native cross-region mirror.
- Next action: monitor uploads, validate the Snowball HF SFT path, and resolve the remaining three base exports.

### 2026-09-08 20:50 EDT - Packed-mask blocker and RNO2A smoke launch

- Hypothesis: the first-class Snowball adapter can train the published HF checkpoint at the historical 64-H100 topology after preserving packed segment IDs and expressing an intra-node `expert=8`, cross-node `data=8` mesh.
- Commit Hash: uncommitted launch bundle based on `1234cbe4bfbcdc24c1bcfb20741f557c1eec567e`; WIP commit pending checks.
- Command: `iris --config lib/iris/config/marin.yaml job run --target-cluster cw-rno2a --job-name snowball-final-qk157-smoke-coord ... -- python -m experiments.sft.configs.snowball_lce_final --base qk157 --stage smoke --version 2026.09.08.1 --run`
- Config: `qk157@2b1f526273b8968b307a0098c08fb4321bb91e35`, one update, seq 32768, global batch 64, 8 nodes x 8 H100, ring expert parallelism 8, AdamH/Adam at 5e-5, unique JAX port 19301, RNO2A interactive.
- Result: job `/benfeuer/snowball-final-qk157-smoke-coord` submitted. Local packed-boundary regression reproduced: changing segment 0 changed all 128 checked hidden values in segment 1 (maximum absolute change 0.01901957). The fix makes segment 1 bit-identical. Mesh and launcher unit tests pass.
- Interpretation: the previous adapter could report a healthy run while violating packed-example isolation. Full training remains gated on the smoke reaching a finite update and producing a reloadable checkpoint.
- Next action: monitor the exact job handle; validate output and record its immutable code commit.

### 2026-09-08 20:52 EDT - Smoke attempt 1 failed before model load

- Hypothesis: Levanter tokenizer staging accepts the same `repo@revision` syntax as its HF checkpoint converter.
- Commit Hash: uncommitted launch bundle based on `1234cbe4bfbcdc24c1bcfb20741f557c1eec567e`.
- Command: `/benfeuer/snowball-final-qk157-smoke-coord`.
- Config: qk157 smoke policy above; tokenizer path was `open-athena/snowball-67b-a2b-base-262k-qk157@2b1f...`.
- Result: falsified before model loading. `snapshot_download` rejected the `@revision` suffix as an invalid repo ID on all eight ranks. No update ran and no checkpoint was written.
- Interpretation: model checkpoint loading supports `RepoRef` strings; tokenizer staging does not. For the next smoke, use the validated repository's current tokenizer path while retaining the immutable model revision. Add revision-aware tokenizer staging before full training.
- Next action: relaunch with a new artifact version and job handle; retain this failed attempt in the record.

### 2026-09-08 21:05 EDT - Smoke attempt 2 initialized on RNO2A

- Hypothesis: the pinned qk157 HF weights can be loaded into the first-class Snowball model and trained at the target 64-H100 topology when the tokenizer is read from the already-validated repository root.
- Commit Hash: submitted bundle based on the changes later committed as `dfd8a1518f`; the bundle predates the tokenizer-revision patch included in that commit.
- Command: `/benfeuer/snowball-final-qk157-smoke2-coord`, version `2026.09.08.2`, JAX port `19302`.
- Config: immutable model revision `2b1f526273b8968b307a0098c08fb4321bb91e35`; repository-root tokenizer; one update; seq 32768; batch 64; 8 nodes x 8 H100; RNO2A interactive.
- Result: active replacement child `/benfeuer/snowball-final-qk157-smoke2-coord/run_levanter_train_lm-5a295d2a` staged the tokenizer and joined all eight JAX ranks. The coordinator retains one failure from an earlier child attempt. Weight-load/update/save evidence is pending.
- Interpretation: scheduling and multi-host bootstrap are healthy. Do not infer success until a finite optimizer update and reloadable checkpoint are observed.
- Next action: continue polling this exact handle, then run a third smoke from committed code with both model and tokenizer revisions pinned.

### 2026-09-08 21:05 EDT - OpenCode cache lineage check

- Hypothesis: the token caches in the pinned `open-athena/grug-67b-a2b-agentic-sft-training-data` release are the exact input to the 1,888-step Stage 3 run.
- Commit Hash: `dfd8a1518f`.
- Command: inspect Hub revision `a9805934c9c98908c611236bbfc87799f1ff6fe5` manifest and compare it with issue #8225's final record.
- Result: falsified. The release has 29 components and 797,783,562 raw cached tokens, yielding 1,903 naïve five-epoch steps at seq32768/batch64. Issue #8225 explicitly identifies these as the original 1,903-step lineage and says the 1,888-step model used a corrected consolidated cache.
- Interpretation: use the immutable release for provenance and converted source data, but use the corrected consolidated fixed-EOT cache for the reproduction. Hard-coding 1,888 over the superseded cache would not reproduce the run.
- Next action: port the corrected-cache dependency from the historical launcher, validate its token accounting and RNO2A accessibility, and keep Nemotron Terminal as an independent sibling of the Thinking checkpoint.

### 2026-09-08 21:20 EDT - Smoke attempt 2 cancelled after stale coordinator endpoint

- Hypothesis: the replacement child was quietly downloading or loading the large HF checkpoint after all ranks entered JAX initialization.
- Commit Hash: submitted bundle based on changes later committed as `dfd8a1518f`.
- Command: inspect both child handles and their task-level logs, then cancel `/benfeuer/snowball-final-qk157-smoke2-coord`.
- Result: falsified. First child task 0 failed binding `[::]:19302` because the address was already in use. In the replacement child, nonzero ranks connected to the stale first-child endpoint at `10.168.195.25:19302`, while replacement task 0 started its coordinator at `10.168.194.209:19302`. All eight replacement tasks stayed blocked for 23 minutes with no post-connect log. The owned smoke was cancelled to release 64 H100s; no model load or update ran.
- Interpretation: this was an Iris child-retry endpoint collision, not a Snowball or HF-loader result. A distinct root job identity and port avoids reusing the stale endpoint name.
- Next action: complete local validation, commit the exact bundle, and launch attempt 3 under a fresh root job and JAX port.

### 2026-09-08 21:25 EDT - Exact agentic caches and tokenizer compatibility

- Hypothesis: the corrected OpenCode and historical Nemotron-Terminal caches can be consumed without rebuilding them, while preserving their original packing and loss-mask semantics.
- Commit Hash: uncommitted follow-up on `dfd8a1518f`.
- Command: inspect historical cache declarations; add materialized-config and behavioral dataset tests; compare the qk157 base `tokenizer.json` with the current pinned Marin tokenizer artifact.
- Result: the OpenCode stage adopts the corrected consolidated `2026.08.05` fixed-EOT cache, right-slices overlength records, shifts the assistant mask for next-token loss, and clears loss at packed segment boundaries. Nemotron adopts the exact `2026.07.17` chat cache and left-slices. Both are independent 1,888-step children of Thinking. The qk157 and current Marin `tokenizer.json` files have identical SHA-256 `881c9c36...`.
- Interpretation: cache token IDs are compatible with the base tokenizer; the base repository's thinner tokenizer metadata does not imply a different vocabulary. The exact caches still need an RNO2A read/accounting check before full training.
- Next action: run the complete safe test suite, commit and push, then launch smoke attempt 3 with a new root identity and port.

### 2026-09-08 21:34 EDT - Five uploads validated and smoke attempt 3 launched

- Hypothesis: each remaining uploader completed the same local-to-Hub hash validation and immutable tagging as qk157 and qk175.
- Commit Hash: `8fc736c4dd` for smoke attempt 3; revision-pin follow-up uncommitted.
- Command: inspect `/held/snowball-8977-athena-{skew2,skew4,skew8}` terminal state and filtered `44/44`/`VERIFIED` logs; submit `/benfeuer/snowball-final-qk157-smoke3-coord` with JAX port 19403.
- Result: all three upload jobs succeeded with 44/44 files and tags `ce41c24df0afc10079210521ea7e231115ad5a92`, `5052e68c4d88c9e0de87f7595a25ee4005aef1cf`, and `058ecaf27b9e4f37219df221a51e7d490d58ec3d`. All five base uploads now pass the publication gate, so the external `TRACKER.md` was created. Smoke attempt 3 is active with zero coordinator failures at submission.
- Interpretation: every training base can now be referenced immutably; no training job will resolve a moving Hub `main`.
- Next action: observe smoke 3 through load/update/save/reload, then materialize all five configs and fan out the dependency-ordered chains.

### 2026-09-08 21:42 EDT - Cache preflight attempt 1 failed before remote read

- Hypothesis: the cache preflight imports the same public filesystem helper locally and in the Iris bundle.
- Commit Hash: `1d649a017f`.
- Command: `/benfeuer/snowball-final-cache-preflight` on RNO2A.
- Result: falsified before S3 access. The bundled `rigging.filesystem` package does not re-export `prefix_join`; Iris exhausted three identical attempts with `ImportError`. No cache content or accounting result was produced.
- Interpretation: import `prefix_join` from its defining `rigging.filesystem.storage_path` module and cover the preflight import in the focused campaign test.
- Next action: commit the import fix and run a new preflight identity.

### 2026-09-08 21:46 EDT - Agentic cache preflight passed on RNO2A

- Hypothesis: both adopted caches are complete, readable through Levanter's training cache reader, tokenizer-compatible, and reproduce the historical OpenCode step count.
- Commit Hash: `099a4c977c`.
- Command: `/benfeuer/snowball-final-cache-preflight2` on RNO2A.
- Result: passed on the first attempt. OpenCode has 76,928 rows and 791,560,603 tokens; five epochs at seq32768/global-batch64 resolve to exactly 1,888 updates. Nemotron-Terminal has 366,154 rows and 6,068,571,206 tokens. For both caches, the ledger agrees with `.stats.json`, the first and last records load, input and mask lengths match, tokens stay inside the 128,256 vocabulary, and masks are binary.
- Interpretation: the corrected OpenCode cache is the required 1,888-step lineage, unlike the superseded 797,783,562-token publication. Both prebuilt inputs are ready for training once the HF smoke gate passes.
- Next action: continue smoke 3 through its finite update and checkpoint validation.

### 2026-09-08 21:56 EDT - Prefix caches absent in RNO2A; rebuild gated

- Hypothesis: the historical WildChat and Nemotron-Science chat caches already exist under the RNO2A S3 prefix and can be audited before the full DAG.
- Commit Hash: `f834a56079`.
- Command: `/benfeuer/snowball-final-cache-preflight3`.
- Result: falsified before any content read: `tokenized/wildchat_386k-chat-25177e/2026.07.17/train/.stats.json` does not exist in `s3://marin-us-east-02a/marin`. Iris repeated the deterministic missing-file failure three times. The two agentic caches remain validated by attempt 2.
- Interpretation: do not copy the large GCP cache across regions. The DAG's pinned tokenization dependencies will build the prefix caches on RNO2A before allocating their training child. Add an execution-time expected-step invariant so a rebuilt cache cannot silently drift from 257/630.
- Next action: validate the expected-step gate locally and retain the smoke gate before launching the five full DAGs.
