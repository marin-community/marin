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

### 2026-09-08 22:10 EDT - Tokenizer identity unified; cache fan-out race removed

- Hypothesis: all five published bases preserve the same tokenizer bytes, so one immutable tokenizer identity can safely back every stage and both rebuilt prefix caches.
- Commit Hash: uncommitted follow-up on `9c40c7b495`.
- Command: download only `tokenizer.json` at each of the five pinned base revisions and at `marin-community/marin-tokenizer@a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2`; compare SHA-256; add and test a data-only prefix-cache gate.
- Result: all six files have SHA-256 `881c9c36c359e1617afef6f7583403567931b7b4f43f6552d2b2155a131650a2`. The campaign now keeps each pinned model revision for weights/config while using the one pinned byte-identical tokenizer. A `data` stage builds the WildChat and Nemotron-Science caches once and refuses counts other than 257/630 updates before five GPU roots launch. Focused campaign tests pass. Smoke 3 remains failure-free; several ranks have completed shard 39/39 and the slowest are finishing the final shards.
- Interpretation: parent tokenizer behavior is unchanged, while cache artifact identity is now genuinely shared instead of being forked by five equivalent repository strings. Prebuilding once prevents cross-coordinator writes to the same cache path.
- Next action: finish smoke load/update/save/loadability validation, commit the cache gate, and run the data stage on RNO2A before fan-out.

### 2026-09-08 22:15 EDT - Smoke 3 found fused-loss sharding mismatch

- Hypothesis: after loading the full checkpoint, the first-class Snowball model can enter the generic Levanter fused-loss path with the campaign's eight-node data mesh.
- Commit Hash: smoke ran `8fc736c4dd`; fix uncommitted on `6fe198511e`.
- Command: inspect `/benfeuer/snowball-final-qk157-smoke3-coord/run_levanter_train_lm-b559324d` after all eight ranks completed shard 39/39; reproduce its `shard_map` mismatch in a fresh eight-CPU-device process; compile forward and gradient after the boundary fix.
- Result: falsified before any update. Stored LM head sharding was `P(('replica_dcn', 'data'), 'model')`, but generic fused loss inferred `P(None, None)` for logical `(Embed, Vocab)` while example activations were data-sharded. The coordinator began an automatic retry; it was cancelled before repeating the full weight load. Snowball now reshards only `get_lm_head()` to `P(None, None)` at the generic loss boundary. The fresh eight-device regression reproduces the old exception and now compiles a finite loss and finite LM-head gradient. The broader Snowball/parity/SFT-packing/campaign suite passes (39 passed, 1 skipped).
- Interpretation: HF conversion and every source shard are loadable on RNO2A. The blocker is isolated to the adapter between the raw Grug parameter layout and generic loss, not the checkpoint. Replication at the loss boundary is correct for the target mesh (`model=1`) and gradients return to the stored layout.
- Next action: commit the fix and launch a fresh one-update smoke identity/port; continue to require update/save/loadability before campaign fan-out.

### 2026-09-08 22:35 EDT - Shared prefix caches passed exact accounting gates

- Hypothesis: one pinned tokenizer identity can materialize both shared prefix datasets on RNO2A with the fixed one-epoch counts before any five-model GPU fan-out.
- Commit Hash: `6fe198511e`.
- Command: `/benfeuer/snowball-final-prefix-caches-coord`, followed by `/benfeuer/snowball-final-prefix-manifest-read`.
- Result: both jobs succeeded. WildChat produced 538,877,811 tokens, which resolves to exactly 257 updates at seq32768/global-batch64. Nemotron-Science produced 1,321,079,881 tokens, which resolves to exactly 630 updates. The manifests live under `wildchat_386k-chat-866d4c/2026.07.17` and `nemotron_science_think-chat-866d4c/2026.07.17` in the RNO2A S3 prefix.
- Interpretation: the dependency gate has eliminated shared-cache races and exact epoch drift. This does not replace the GPU smoke gate.
- Next action: finish `/benfeuer/snowball-final-qk157-smoke4-coord` through update/save/reload, then launch the five full chains.

### 2026-09-08 22:35 EDT - Evaluation campaign surface validated locally

- Hypothesis: the existing evaluation framework can represent all 25 checkpoints, three explicit seeds, and the branch-matching Nemotron-Terminal harness without weakening checked-in task policy.
- Commit Hash: uncommitted follow-up on `c5ef700ef1`.
- Command: focused evaluation tests, explicit harness integration test, and an H100x8 RNO2A dry-run launch for `snowball-final-qk157-base` at seed 42.
- Result: 29 default-marker tests passed, the explicit integration test passed, and the dry run resolved the interactive RNO2A accelerator. The catalog contains all five bases across Base/Chat/Thinking/OpenCode/Nemotron-Terminal, and the launcher records seed overrides only for Evalchemy definitions.
- Interpretation: non-agentic launches can be split by suite and seed while preserving task-local limits and few-shot settings; OpenCode and Nemotron-Terminal retain distinct harnesses.
- Next action: review, commit, and push the evaluation surface while smoke 4 continues loading.

### 2026-09-08 22:39 EDT - Native reload gate made executable

- Hypothesis: materializing the smoke artifact is insufficient unless the exact native checkpoint initialization used by Chat -> Thinking -> final branches also succeeds on the target topology.
- Commit Hash: uncommitted follow-up on `1c01bff36d`.
- Command: add a `smoke-reload` stage that depends on the one-update HF smoke, initializes weights strictly from its native checkpoint with a fresh optimizer, and performs one further update and HF/native save.
- Result: the campaign wiring test confirms native rather than HF initialization, the exact parent checkpoint path, the pinned Snowball config/tokenizer, and a one-update run. All 11 campaign tests and the full pre-commit gate pass.
- Interpretation: after smoke 4 succeeds, a second RNO2A job can validate the same native boundary every downstream stage uses without starting the five-model campaign.
- Next action: push this gate, then launch it against smoke version `2026.09.08.5` after the parent finishes.

### 2026-09-08 22:44 EDT - First non-agentic base evaluation submitted

- Hypothesis: one qk157 Base/NLP launch can validate the distributed Snowball serving/evaluation path while the 64-H100 training smoke continues, and contributes directly to the required matrix.
- Commit Hash: `265ff8e495` (evaluation surface introduced in `1c01bff36d`).
- Command: launch `snowball-final-qk157-base` with suite `nlp`, seed 42, H100x8, federated cluster `cw-rno2a`, and interactive priority.
- Result: submitted group `20260909-024426-snowball-final-qk157-base-3d38` with all 14 NLP evaluations sharing one serve. Task-local generation caps overrode the catalog maximum as intended.
- Interpretation: hold the other 29 base suite/seed launches until this canary proves model serving and durable record completion.
- Next action: monitor both the qk157 HF training smoke and this evaluation group; fan out only after their respective gates pass.

### 2026-09-08 22:53 EDT - Evaluation canary exposed undersized serve disk

- Hypothesis: the generic evaluation worker resources are sufficient to stage the 39-shard Snowball checkpoint and start vLLM.
- Commit Hash: canary ran `265ff8e495`; fix uncommitted.
- Command: inspect the failed group and inference child `/benfeuer/eval-20260909-024426-snowball-final-qk157-base-3d38/inference-11485f945b2b44b081c903e5922d7c44`.
- Result: falsified before any benchmark request. The child began streaming 47 HF files, emitted no model/runtime exception, and exited 137 after six minutes. The catalog had overridden memory to 512 GB but inherited `DEFAULT_SERVE_DISK = "100g"`, which cannot hold this approximately 134 GB checkpoint plus staging overhead.
- Interpretation: this is a resource declaration bug in the Snowball campaign catalog, not a score/model failure. Give every one of the 25 entries an explicit 512 GB disk and cover it in the catalog test.
- Next action: run checks, push the resource fix, and repeat only the failed qk157 Base/NLP seed-42 canary.

### 2026-09-08 23:15 EDT - Evaluation retry exposed local-file URI boundary

- Hypothesis: after increasing the serve disk to 512 GB, the pinned HF snapshot returned by the regional cache is directly loadable by vLLM.
- Commit Hash: retry ran `16bd28c4ae`; fix uncommitted.
- Command: inspect retry group `20260909-025605-snowball-final-qk157-base-6886` and inference child `/benfeuer/eval-20260909-025605-snowball-final-qk157-base-6886/inference-3a8bcfe6cf1d45469e4c04734e91a143`.
- Result: staging completed and vLLM started, falsifying any remaining disk diagnosis. The cache resolver supplied `file:///Users/.../quick-serve-models/...`; Transformers passed that string through Hugging Face repository validation and raised `HFValidationError` before loading config or weights. No benchmark ran. `resolve_model_path` now converts local `file://` cache URIs to decoded filesystem paths while leaving Hub IDs and object-store URIs unchanged. A parameterized, model-agnostic regression covers ordinary, percent-encoded, and localhost file URIs; the focused resolver tests pass (7 passed).
- Interpretation: this failure is independent of Qwen/Snowball architecture and can affect any model mirrored into a process-local cache. The correction belongs at Marin's cache-to-model-loader boundary, not in the Snowball catalog or vLLM model configuration.
- Next action: run the full pre-commit and inference tests, commit the fix, then repeat only this canary.

### 2026-09-08 23:28 EDT - Smoke 4 reached the update and exposed global rematerialization

- Hypothesis: the LM-head loss-boundary reshard from smoke 3 is sufficient for the full 64-H100 train step.
- Commit Hash: `c5ef700ef1`.
- Command: inspect `/benfeuer/snowball-final-qk157-smoke4-coord/run_levanter_train_lm-356928e0` after all ranks loaded shard 39/39 and compiled `jit__train_step`.
- Result: falsified before the first update. The GPU path selected `batched_xla`, but XLA planned 283.12 GiB per device and attempted one 288,666,052,208-byte (268.84 GiB) allocation. Its partitioner reported involuntary full rematerialization while converting a locally sharded `[1, 32768, 2560]` activation to an incompatible layout. All devices held only 14.32 GiB at failure, so this is a compiled intermediate, not resident checkpoint or optimizer-state exhaustion. Iris started an identical retry; the owned coordinator was cancelled before another 39-shard load.
- Interpretation: the small eight-device loss/gradient test caught the prior legality error but cannot catch memory scaling at the real sequence, vocabulary, and nested-sharding geometry. The next regression must inspect the compiled sharding/memory shape, and the correction must preserve the bounded loss path.
- Next action: reproduce the nested generic-loss/raw-Grug sharding transition locally, fix the boundary, and compile a memory-shape regression before smoke 5.

### 2026-09-08 23:28 EDT - Local-file eval fix passed and canary 3 submitted

- Hypothesis: normalizing the staged `file://` URI at `resolve_model_path` lets vLLM treat the cache as a local directory for every model architecture.
- Commit Hash: `19df94ff62`.
- Command: focused resolver tests, full inference tests, affected safe-test selection, full pre-commit, then launch qk157 Base/NLP seed 42 on H100x8.
- Result: focused resolver tests passed (7). Inference tests passed 107 plus one skip after two unrelated host-lock cases passed serially. The affected-test runner passed all 1,600 selected tests with platform skips, and pre-commit passed. Canary 3 is group `20260909-032756-snowball-final-qk157-base-49f7` with the same 14 NLP evaluations.
- Interpretation: the regression is architecture-neutral (`org/model` in the test) and exercises the public resolved loader path. Hold the other matrix launches until this canary serves and writes benchmark records.
- Next action: monitor canary 3 through endpoint readiness and durable evaluation records.

### 2026-09-08 23:44 EDT - Local-token loss boundary validated; eval canary exposed stale loader option

- Hypothesis: Snowball can use the production Grug loss boundary for the default training reduction while retaining the generic Levanter path for non-default evaluation reductions, and the repaired evaluation canary will progress beyond its former URI failure.
- Commit Hash: uncommitted follow-up on `19df94ff62`.
- Command: compare the generic and raw Grug loss shard maps; add an eight-CPU-device local-token regression and a full-logits numerical oracle; inspect canary 3 through its vLLM worker failure.
- Result: the training adapter now hands each device only its local flattened token batch to the bounded fused kernel and matches a full-logits cross-entropy oracle. The focused suite passed 53 tests, the affected runner passed 1,429 tests with 172 platform skips, and pre-commit passed. Canary 3 successfully normalized the cached URI and initialized the Grug architecture, then vLLM rejected `{"distributed": true}` because the staged HF directory uses load format `auto`; that option belongs only to the RunAI streaming loader.
- Interpretation: keep non-default reductions on the generic model API, but use the explicit raw-array shard map for the default SFT mean. Remove the RunAI-only loader option only from the staged final-campaign models; retain it for direct object-store Snowball exports.
- Next action: commit and push both corrections, launch smoke 5 with a fresh job identity/port, and repeat only the qk157 Base/NLP seed-42 canary.

### 2026-09-08 23:51 EDT - Smoke 5 and evaluation canary 4 launched

- Hypothesis: commit `d08ef276c5` keeps the full-shape training loss device-local and allows the staged HF base to start vLLM without changing direct object-store streaming behavior.
- Commit Hash: `d08ef276c5`.
- Command: submit `/benfeuer/snowball-final-qk157-smoke5-coord` on RNO2A with JAX port 19406 and version `2026.09.08.5`; launch qk157 Base/NLP seed 42 as evaluation group `20260909-035119-snowball-final-qk157-base-4835`.
- Result: both bounded gates were accepted at interactive priority. Smoke 5 owns 64 H100s only after its child materializes; canary 4 contains the same 14 NLP tasks and one shared H100x8 serve.
- Interpretation: do not fan out either campaign until the corresponding gate has durable success evidence.
- Next action: monitor smoke 5 through load/compile/update/save and canary 4 through endpoint registration plus result persistence.

### 2026-09-09 00:14 EDT - Eval canary 4 exposed insufficient runtime HBM headroom

- Hypothesis: successful endpoint registration means the default vLLM memory reservation is safe for the actual NLP workload.
- Commit Hash: canary ran `d08ef276c5`; fix uncommitted.
- Command: inspect inference child `/benfeuer/eval-20260909-035119-snowball-final-qk157-base-4835/inference-d160db4fa02d494cb30860acbbea64fa` through startup, cache sizing, and its first MMLU requests.
- Result: falsified after serving began. All eight ranks loaded and registered, and `/v1/models` returned 200. vLLM's 0.92 utilization target allocated a 51.2 GiB KV cache per 79.18 GiB GPU; during the workload, ranks had only 510 MiB free and failed a 1.46 GiB allocation. The benchmark's subsequent 404s were consequences of the dead inference endpoint, not an endpoint-name transport bug. The owned group was cancelled to stop futile retries. The campaign catalog now sets utilization to 0.85, retaining about 5.5 GiB more headroom per GPU, and the exact catalog invariant is covered by the existing 25-model regression test (29 tests passed).
- Interpretation: staging, local URI normalization, architecture resolution, weight loading, endpoint registration, and request routing are all validated. One fresh canary must demonstrate that the lower KV reservation survives real requests and persists results before fan-out.
- Next action: run repository gates, commit and push the headroom correction, then launch only qk157 Base/NLP seed 42 as canary 5.

### 2026-09-09 00:22 EDT - Eval canary 5 submitted with explicit HBM headroom

- Hypothesis: reducing the Snowball campaign's vLLM utilization from 0.92 to 0.85 leaves enough activation headroom for the real NLP workload while retaining ample KV capacity.
- Commit Hash: `994053ca51`.
- Command: launch `snowball-final-qk157-base` with suite `nlp`, seed 42, H100x8, federated cluster `cw-rno2a`, and interactive priority.
- Result: submitted group `/benfeuer/eval-20260909-042148-snowball-final-qk157-base-4b74` with the same 14 NLP evaluations and one shared serve. Mechanical validation passed: pre-commit, 135 evaluation tests, and 1,429 affected safe tests with 172 skips. The branch-wide advisory review found no defect in the headroom change.
- Interpretation: retain the single-canary gate until logs confirm 0.85 allocation, inference survives MMLU traffic, and all results are durable.
- Next action: monitor canary 5 and smoke 5; do not fan out either campaign yet.

### 2026-09-09 00:38 EDT - Smoke 5 falsified the local-kernel-only fix

- Hypothesis: handing only device-local tokens to the fused CE kernel prevents the 64-way train-step transpose from materializing the global batch-by-vocabulary surface.
- Commit Hash: `d08ef276c5`.
- Command: run `/benfeuer/snowball-final-qk157-smoke5-coord` through all 39 HF shards and the first `jit__train_step` compile on 64 H100s.
- Result: falsified before update. All eight ranks loaded 39/39 and selected `batched_xla`, but XLA again requested exactly 268.84 GiB per device. The compiler showed the local `[1,32768,2560]` activation crossing from the 64-way batch sharding to a replicated layout during transpose. The owned coordinator retry was cancelled immediately. The remaining conflict is the stored LM head: its hidden dimension uses the same physical axes as the token batch, so differentiating through its loss-boundary replication makes XLA reconstruct the global activation/logit surface.
- Interpretation: keep `output_proj` replicated in the first-class adapter at initialization and HF/native load. The head is about 0.7 GiB in bf16, so replication is bounded; its data-parallel gradient can all-reduce without an incompatible post-loss reduce-scatter. The eight-device loss test now asserts replicated head storage and device-local kernel input. Focused Snowball/SFT/load tests pass (36), and full pre-commit passes.
- Next action: complete affected tests and review, then launch smoke 6 with a fresh identity and port; continue to require finite update, save, and native reload.

### 2026-09-09 00:47 EDT - Smoke 6 launched with replicated LM head

- Hypothesis: storing `output_proj` replicated removes the batch/head layout conflict that produced the 268.84 GiB train-step allocation.
- Commit Hash: `2da2e46812`.
- Command: submit `/benfeuer/snowball-final-qk157-smoke6-coord` on RNO2A with JAX port 19407 and version `2026.09.08.6`.
- Result: coordinator accepted at interactive priority. Focused tests, full pre-commit, the 1,429-test affected suite, and advisory review passed without a head-placement finding. Evaluation canary 5 independently confirmed a 67.3 GiB target and 45.66 GiB KV cache at 0.85 utilization.
- Interpretation: smoke 6 must still demonstrate the production 64-H100 compile, finite update, save, and native reload. Canary 5 must survive actual benchmark traffic.
- Next action: monitor both gates; launch no campaign fan-out yet.

### 2026-09-09 01:36 EDT - Replicated head falsified; base evaluations released

- Result: smoke 6 reproduced the 268.69 GiB `jit__train_step` allocation after all ranks entered `batched_xla`; the owned retry was cancelled. MMLU canary 5 independently persisted 14,042 samples across 57 subtasks with `status=succeeded`, and its server advanced to ARC Challenge.
- Interpretation: head storage alone is not causal. The compiler is still transposing between the full `data × expert` batch layout and an 8-way layout at the loss boundary. The evaluation serving correction is validated under benchmark traffic.
- Next action: compare one-node/batch-8 and two-node/batch-16 probes with one sequence per GPU, while the remaining 29 base suite/seed evaluation groups run in parallel.

### 2026-09-09 01:38 EDT - Topology probes relaunched with valid versions

- Result: the initial probe coordinators rejected `topology1`/`topology2` as invalid immutable version labels before allocating GPUs.
- Interpretation: this was a launch-only validation error and yielded no topology evidence.
- Next action: monitor `/benfeuer/snowball-final-qk157-topology1b-coord` (`2026.09.09.1`) and `/benfeuer/snowball-final-qk157-topology2b-coord` (`2026.09.09.2`).

### 2026-09-09 01:44 EDT - Full batch-axis shard map rejected on GPU

- Result: smoke 7 from `d68ae9d104` failed before model loading. XLA GPU rejected the canonical `data × expert` shard map because its device assignment was non-IOTA (`0,8,16,…,1,9,…`). The owned retry was cancelled.
- Interpretation: the existing one-axis shard map is a GPU layout constraint, not an accidental loss of `expert`. The speculative code and test change are reverted while this evidence remains in the log.
- Next action: use the prior-commit one- and two-node controls to determine whether the 268.69 GiB transpose requires cross-node `data`; inspect a device-local loss boundary that does not introduce a nested global shard map.

### 2026-09-09 02:31 EDT - Restore production rematerialization and preserve physical loss sharding

- Hypothesis: the first-class Snowball training graph diverges from the working production Grug recipe in two memory-critical ways: its transformer scan omits per-block recompute-all checkpointing, and its reduced fused-loss shard map reconstructs logical axis names instead of preserving the input tensors' physical device order.
- Command: compare `SnowballTransformer` with `experiments/grug/moe/model.py`, exercise a two-by-two-by-two explicit CPU mesh, and differentiate weighted `none`, `sum`, and `mean` losses against a full-logits oracle.
- Result: the production model checkpoints every block, while Snowball only scanned them. The pending correction checkpoints the scan body, restores the historically sharded `Plm_head` storage, infers reduced-loss input specs from the physical arrays, and reduces over the shard-map value's actually varying manual axes. Twenty-five Snowball/parity tests pass, including nonzero LM-head gradients for all reduction modes; full pre-commit passes. The standalone `uv run ty check` entry point is absent from this environment, while the configured Pyrefly gate passes.
- Interpretation: the replicated-head hypothesis remains falsified. The new regression covers the missing remat primitive, local fused-kernel batch size, multi-axis physical layout, head-gradient storage, and exact weighted reduction gradients. A full-shape H100 smoke is still required because the CPU compiler cannot validate the production memory schedule.
- Next action: commit and push the correction, then run one fresh eight-node RNO2A smoke through finite update, save, and native reload before releasing training chains.

### 2026-09-09 02:47 EDT - Generic mesh axis order differs from production Grug

- Hypothesis: smoke 8's non-IOTA assignment comes from the generic trainer placing all ICI axes before DCN axes, while Snowball's raw batch specs assume the production Grug order `(replica_dcn, data, expert)`.
- Command: compare `TrainerConfig.device_mesh`, `create_mesh_from_axis_specs`, `compact_grug_mesh`, and the Snowball campaign topology; reproduce the emitted `0,8,16,...,1,9,...` permutation from the logged batch-axis order.
- Result: confirmed. The generic campaign mesh resolves as ICI `(replica, model, expert)` followed by DCN `(replica_dcn, data)`, whereas the working Grug trainer constructs `(replica_dcn, data, expert, model)` directly. A new opt-in `MeshConfig.axis_order` preserves existing defaults and lets Snowball request the historical physical order. Forty-two focused mesh, campaign, and Snowball tests pass; full pre-commit, including Pyrefly, passes.
- Interpretation: fix the mesh at its construction boundary instead of adding custom gradient rules or changing the meaning of batch axes. This is reusable for any model with raw physical PartitionSpecs, and existing non-Snowball users are unchanged because the new field defaults to `None`.
- Next action: commit and push, then launch one full-topology smoke with a fresh port and require finite update, save, and native reload.

### 2026-09-09 02:52 EDT - Base evaluation fan-out exceeded vLLM's internal load timeout

- Hypothesis: the 24 failed non-qk157 Base evaluation groups are checkpoint failures or HBM failures.
- Command: enumerate top-level Iris states and inspect the qk175-skew2 NLP seed-42 inference timeline through its first exception.
- Result: falsified. Every qk157 group is progressing; all 24 later groups failed during inference startup. The representative worker remained healthy and streamed 502/502 tensors, but object-store load took 10m40s, just beyond vLLM's fixed 600-second engine-ready timeout. The frontend killed the engine at 600 seconds even though loading completed moments later. Marin's outer readiness budget is 2,400 seconds.
- Interpretation: set vLLM's internal engine-ready default to 1,500 seconds, below Marin's outer budget, while preserving an explicit environment override. This is a general large-checkpoint startup correction, not Snowball model logic.
- Next action: run gates, commit and push, then relaunch the 24 failed Base suite/seed groups with new identities; retain the five progressing qk157 groups.

### 2026-09-09 03:02 EDT - Old-mesh control ended; corrected-mesh smoke remains healthy

- Result: smoke 8's first attempt lost the rank-0 coordination service after the long checkpoint load. Iris marked task 0 failed and the seven peers `cosched_failed`; logs contain no train-step compilation, resource exhaustion, optimizer update, or save. Its automatic identical retry was cancelled because the old-mesh control had already supplied its intended non-IOTA comparison. Smoke 9 remains healthy with no non-IOTA warning and reached shard 26/39 on its fastest rank. The representative qk175-skew2 evaluation retry reached vLLM engine initialization without failure.
- Interpretation: smoke 8 contributes topology evidence only and cannot adjudicate the memory fix. Smoke 9 remains the sole finite-update/save gate. The evaluation timeout retry has started correctly but must remain alive beyond the former 600-second internal boundary before the fix is considered runtime-validated.
- Next action: monitor smoke 9 through compile/update/save and the representative evaluation past its old timeout boundary; then run native reload and release the five training chains only after both gates are durable.

### 2026-09-09 03:20 EDT - Smoke 9 exposes generic fused-loss manual-axis carry bug

- Hypothesis: preserving production physical mesh order is sufficient for the first train-step compile.
- Result: partially confirmed and then falsified. All eight ranks loaded 39/39 shards with no non-IOTA warnings, proving the mesh-order correction. The first train-step trace then failed in the generic XLA fused CE backward: its replicated zero LM-head-gradient carry became varying across `(replica_dcn, data, expert)` after the first batch-block update, violating JAX VMA loop invariants. No executable compiled, update ran, or checkpoint saved; the deterministic retry was cancelled. A one-device explicit `shard_map` reproduces the exact error without Snowball. The pending fix gives all batch-derived loop initializers the input's manual-axis type and psums the shared-weight gradient back to its replicated primal type.
- Tests: direct VMA/numerical tests pass for both XLA backward implementations. The existing eight-device Snowball physical-layout test now restores its fake instrumentation kernel before its numerical section, so it exercises the real fused kernel and matches the full-logits oracle for `none`, `sum`, and `mean`. The full fused-kernel file passed 85 tests with 15 platform skips; all 20 Snowball tests passed; the diff-driven suite passed 1,563 tests with 192 skips; full pre-commit/Pyrefly passed.
- Interpretation: this is a general explicit-mesh fused-kernel correctness gap, not a Qwen- or Snowball-specific exception. Default non-manual callers take the helper's no-op path; full existing kernel and repository gates remain required before another smoke.
- Next action: complete full kernel/pre-commit gates, commit and push, then launch a new full-topology smoke with a fresh identity, port, and version.

### 2026-09-09 03:48 EDT - Fused-loss correction and regression suite complete

- Result: the generic manual-axis correction is committed as `c26a2b3c00`. Follow-up review replaced the Snowball rematerialization string search with a structural assertion that a `remat2` primitive is directly inside the transformer scan body, removed assertions that only restated constants or import provenance, corrected stale campaign documentation, and made evaluation checkpoint path joining storage-backend-safe. A first path-join attempt exposed the two-argument `prefix_join` contract during collection; the corrected call now has a catalog regression covering every base, stage, version, and step path.
- Tests: focused campaign, catalog, and structural-remat tests passed (13); full pre-commit/Pyrefly passed; the corrected diff-driven suite passed 1,562 tests with 192 platform skips. The earlier fused-kernel file (85 passed, 15 skipped), complete Snowball tests (20), and campaign tests (12) remain green on the same implementation.
- Interpretation: coverage now directly exercises the model-independent fused-loss failure, its fast and slow backward variants, the production Snowball physical layout, real numerical gradients, and the placement of rematerialization inside the scan. Existing non-manual callers and default mesh ordering remain unchanged.
- Next action: commit and push the review follow-ups, then launch smoke 10 on RNO2A and require a finite update, checkpoint save, and native reload before campaign fan-out.

### 2026-09-09 04:16 EDT - Corrected kernel crosses the full-topology lowering gate

- Result: smoke 10 (`/benfeuer/snowball-final-qk157-smoke10-coord`, `9e35a01320`) coordinated all eight RNO2A workers, loaded all 39 pinned qk157 shards on every rank, selected a fused-CE kernel, and lowered the full `train_step` on the production 64-H100 topology. The autotuner's attempt to inspect an auxiliary kernel jaxpr still reports a manual-axis warning for its loss/LSE carry, but that warning is nonfatal: candidate compilation completed and lowering continued. There is no traceback or `RESOURCE_EXHAUSTED` evidence; executable compilation, a finite update, and save remain the active gates.
- Evaluation: the first quota repair (`/benfeuer/eval-20260909-075152-snowball-final-qk175-skew2-base-f0f2`, seed 43) cleared dataset materialization and is serving successful MMLU requests. Seed 44 was submitted more than 20 minutes later as group `20260909-081602-snowball-final-qk175-skew2-base-7fbd`, record `20260909-081602-snowball-final-qk175-skew2-base-mmlu-d580`. The remaining six quota repairs stay staggered.
- Interpretation: the generic backward-carry correction fixes smoke 9's fatal trace failure and is independent of Snowball model code. Full runtime success is not claimed until update/save/reload complete.
- Next action: monitor smoke 10 through update/save, run the native reload gate, and advance one MMLU repair only after the preceding repair has safely crossed dataset loading.

### 2026-09-09 05:04 EDT - Smoke compile remains coordinated; fourth quota repair released

- Result: smoke 10 remains on attempt 0 with all eight ranks running and no failures or preemptions. It has spent approximately 50 minutes after all ranks logged `Lowered train_step`; periodic NCCL RAS probes receive every peer response and report zero missing responses. There is still no traceback, resource exhaustion, restart, update, or save evidence. The qk175-skew2 seed-43 repair now has a durable succeeded record. Repairs for qk175-skew2 seed 44 and qk175-skew4 seeds 42/43 crossed dataset materialization, so qk175-skew4 seed 44 was submitted as group `20260909-090323-snowball-final-qk175-skew4-base-150c`, record `20260909-090323-snowball-final-qk175-skew4-base-mmlu-b010`.
- Interpretation: the smoke has decisively passed the one-minute post-lowering OOM signature from smokes 5/6, but runtime success is not yet established. The all-peer RAS probes do not establish a deadlock. Staggering remains necessary to avoid another Hugging Face quota burst.
- Next action: continue monitoring without cancelling a healthy attempt; release skew8 seed 42 only after the current repair crosses dataset materialization.

### 2026-09-09 05:34 EDT - Base evaluation repair path is durable

- Result: all six qk157 Base NLP/Chat seed groups are succeeded. The qk175-skew2 MMLU seed-43 and seed-44 repairs have durable succeeded records. The qk175-skew4 seed-42 repair compacted 14,042 samples and wrote `status=succeeded`; seeds 43/44 are executing. The first skew8 repair was released as `20260909-091628-snowball-final-qk175-skew8-base-38b8` only after skew4 seed 44 crossed dataset loading. The isolated qk175 LAMBADA seed-44 remote disconnect was reproduced as transient: targeted group `20260909-090830-snowball-final-qk175-base-663f` completed and wrote a succeeded record.
- Interpretation: the targeted reruns preserve the original suite policy and seeds while replacing only missing cells. No new model-serving or scoring failure appears in the replacement inventory.
- Next action: complete the two remaining skew8 MMLU repairs, aggregate durable Base records into the canonical table, and continue the unchanged smoke 10 update/save gate.

### 2026-09-09 05:40 EDT - First canonical result matrix and skew8 seed 43

- Hypothesis: the completed qk157 Base record set is sufficient to establish the canonical table format, while releasing one additional MMLU repair only after its predecessor enters inference keeps dataset-materialization traffic below the observed Hub quota.
- Command: run read-only RNO2A record snapshot `/benfeuer/snowball-final-qk157-results-snapshot`; reduce its durable records by policy headline metric; inspect qk175-skew8 seed-42 traffic; submit only seed 43 with the checked-in `mmlu` definition, H100x8, RNO2A, interactive priority.
- Result: the snapshot succeeded with exactly 51 `status=succeeded` records (17 tasks times seeds 42/43/44). Their per-seed values, means, and sample standard deviations are recorded in external `RESULTS.md`. Seed 42 was actively serving successful MMLU completions before seed 43 group `20260909-093910-snowball-final-qk175-skew8-base-b4d0` was submitted; its record is `20260909-093910-snowball-final-qk175-skew8-base-mmlu-4da6`.
- Interpretation: qk157 Base evaluation is complete and auditable. The final quota-repair seed remains gated on seed 43 crossing dataset materialization. Smoke 10 remains live with all eight workers and no reported failures, but has not yet supplied finite-update/save evidence.
- Next action: wait for durable completion of active MMLU repairs, release skew8 seed 44 after the materialization gate, and continue the smoke through update/save/native reload before five-chain fan-out.

### 2026-09-09 06:04 EDT - Bug ledger and reproducible-path decision

- Result: confirmed campaign defects are filed as Marin bugs #9029 through #9037, covering packed attention boundaries, missing Snowball rematerialization, physical mesh order, fused-XLA manual-axis gradients, pinned tokenizer staging, cached file-URI normalization, backend-specific loader options, large-model engine readiness, and concurrent MMLU Hub quota bursts. Each issue has `bug` and `agent-generated` labels and a concrete reproduction; no issue comments were posted.
- Evaluation: qk175-skew4 MMLU seeds 43 and 44 succeeded. Qk175-skew8 seed 43 entered completion traffic, so the final seed-44 repair was released as group `20260909-100246-snowball-final-qk175-skew8-base-cff6`, record `20260909-100246-snowball-final-qk175-skew8-base-mmlu-343a`.
- Reproducibility decision: the #8977 native checkpoints remain in `gs://marin-us-central2` and have no documented RNO2A mirror. The historical #8172 path accepts only its vendored native Grug pytree and statically reconstructs one architecture. Continue the current smoke as one bounded qualification because the alternatives require a roughly 5 TB regional copy or a new HF-to-vendored-Grug bridge. Make no further training-path changes without deterministic smoke evidence.
- Result collection: the first fleet snapshot failed before data reads because the reducer attempted to compare `StoragePath` objects. Corrected CPU-only snapshot `/benfeuer/snowball-final-base-results-snapshot-all2` uses `key=str` and is active; no evaluation artifacts were mutated.
- Next action: require smoke update/save/reload, finish the last staggered repair, and use the corrected durable snapshot to populate the remaining Base tables.

### 2026-09-09 07:20 EDT - Five-base evaluation matrix complete

- Result: the three staggered qk175-skew8 MMLU repairs all succeeded. Seeds 43 and 44 each wrote 14,042 samples across 57 subtasks and durable succeeded records. The apparent qk175-skew4 OlympiadBench seed-43 gap was an object-listing race: its original evaluation completed after the earlier snapshot enumerated the prefix, so no duplicate was launched.
- Audit: read-only RNO2A snapshot `/benfeuer/snowball-final-base-results-snapshot-final` selected exactly 255 succeeded records: five Base checkpoints times 17 tasks times seeds 42, 43, and 44. Every Base has 51 headline cells. All five per-seed, mean, and sample-standard-deviation tables are recorded in external `RESULTS.md`; superseded failures are excluded.
- Training: smoke 10 remains on attempt 0 with all eight workers running, zero failures/preemptions, and periodic collective probes reporting zero missing peers. It has not emitted a finite update or save, so the full chains remain gated.
- Next action: require smoke 10 finite update/save and a native reload before releasing the five dependency-ordered training chains.

### 2026-09-09 07:30 EDT - Distributed fused-CE autotuning deadlock isolated

- Hypothesis: smoke 10 is still performing a long whole-step compilation.
- Result: falsified. Native thread snapshots of all eight ranks found every main thread blocked in `CoordinationServiceAgent::GetKeyValue` under XLA `ConfigAssigner::AssignConfigs` and `AutotunerPass`. Fused-CE logs immediately before compilation show that the manual-axis auxiliary jaxpr could not be hashed, after which ranks independently selected different block-size winners (rank 0 used vocab block 256; rank 6 used 64). The resulting HLOs diverged, so no rank could satisfy distributed config assignment. The owned smoke was cancelled after 3h42m to release 64 H100s; it produced no update or checkpoint.
- Issue: filed [#9038](https://github.com/marin-community/marin/issues/9038) for the distinct distributed deadlock when fused-CE autotuning lacks a safe shared identity. No issue comment or fix PR was added.
- Disposition: use the existing supported `LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS=0` launch setting. That branch returns the deterministic inferred block sizes and does not introduce a new training implementation. Smoke 11 is `/benfeuer/snowball-final-qk157-smoke11-coord`, version `2026.09.09.6`, unique JAX port 19414.
- Next action: require smoke 11 finite update/save, then launch the native reload gate before any campaign fan-out.

### 2026-09-09 08:44 EDT - Deterministic fused-CE setting did not cure HLO divergence

- Smoke 11 loaded all 39 pinned HF shards on all eight ranks and lowered `train_step` in 52–56 seconds per rank with `LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS=0` confirmed inside every task. No rank logged a Levanter fused-CE autotune miss or selected winner.
- Result: the executable still could not compile. Repeated native profiles found every rank's main thread idle in XLA `ConfigAssigner::AssignConfigs` under distributed `GetKeyValue`. The pinned OpenXLA implementation first autotunes and publishes the current rank's shard under a key containing the full HLO-module fingerprint, then waits for every other rank's fingerprint-derived key. Since all eight ranks had reached the wait loop but none could read its peers' keys, their module fingerprints differed and the 24-hour waits could not resolve. The owned job was cancelled; no update or checkpoint was written.
- Publication: created and verified the public Hugging Face collection [Snowball LCE Issue 8977](https://huggingface.co/collections/open-athena/snowball-lce-issue-8977-6aa1523f60786ed95561b4f1) with exactly the five immutable model repositories.
- Decision: stop iterating on the poorly qualified first-class Snowball training path. Re-evaluate the historical #8172 plus corrective-commit route and use only a source/weight conversion whose equivalence can be independently validated on RNO2A; do not copy the roughly 5 TB GCP-native checkpoint set across regions without explicit approval.

### 2026-09-09 09:16 EDT - Historical Grug path entered qualification

- Bug: the live Grug model exposes `hf_checkpoint_converter()` and canonical HF export keys but has no inverse mapping; importing its own state dict falls back to internal Equinox names and fails with `KeyError: token_embed`. Filed [#9039](https://github.com/marin-community/marin/issues/9039) with `bug` and `agent-generated`; no comment or fix PR was posted.
- Bridge: commit `1d556ded10` adds a cacheable CPU-only conversion from an immutable Snowball HF revision to the vendored stacked Grug training pytree. It uses the existing tested Snowball canonical-key loader, validates all architecture fields including base-specific `qk_mult`, stacks into the historical trainer layout, and reconstructs `pending_qb_betas` from the effective exported router bias. A tiny-model regression verifies every dynamic leaf and logits after the trainer's first QB application. Thirteen focused tests and full pre-commit/Pyrefly pass.
- Workspace: `/Users/benfeuer/Documents/marin` is clean. Further edits and launches use `/Users/benfeuer/Documents/marin-worktrees/sft-snowballs-final` on `benfeuer/sft-snowballs-final-run`.
- Run: submitted `/benfeuer/snowball-final-qk157-grug-smoke12-coord` from that worktree at version `2026.09.09.7`, unique JAX port 19415. Its single-node conversion child began streaming the pinned qk157 HF revision; the eight-node historical training smoke remains dependency-gated.

### 2026-09-09 09:21 EDT - Conversion allocation bounded before training

- Audit: the smoke 12 conversion would retain the roughly 134 GB BF16 HF state dict while allocating both unstacked and stacked random FP32 model templates, roughly 268 GB each. That deterministic approximately 670 GB live set exceeded the conversion pod's 512 GB memory limit before any training allocation.
- Action: cancelled exactly `/benfeuer/snowball-final-qk157-grug-smoke12-coord`; it had not completed conversion, allocated training GPUs, updated weights, or saved a checkpoint. Replaced both concrete FP32 templates with Equinox shape-only templates and bumped the conversion artifact version so no partial artifact can be reused.
- Validation: the tiny end-to-end import regression passes with shape-only leaves and continues to compare every dynamic block leaf plus logits after the first historical QB application.
- Next action: run focused and repository gates, commit and push from the isolated worktree, then launch smoke 13 with a fresh root identity, artifact version, and JAX port.

### 2026-09-09 09:24 EDT - Historical Grug smoke 13 submitted

- Validation: thirteen focused tests and the complete repository pre-commit gate passed after formatting. Commit `7a3c1be553` is pushed on `benfeuer/sft-snowballs-final-run`.
- Run: submitted `/benfeuer/snowball-final-qk157-grug-smoke13-coord` from `/Users/benfeuer/Documents/marin-worktrees/sft-snowballs-final`, stage version `2026.09.09.8`, conversion artifact version `2026.09.09.2`, and unique JAX port 19416.
- Next action: monitor conversion memory and artifact completion, then require the dependency-gated eight-node smoke to produce a finite update and loadable checkpoint before campaign fan-out.

### 2026-09-09 10:33 EDT - Shape-only import passed; checkpoint path rejected

- Result: smoke 13 loaded all 39 pinned HF shards and completed the full-scale shape-only import/stack phase on attempt 0 without exceeding the 512 GB pod limit. It then raised before checkpoint serialization because the new bridge passed three arguments to the two-argument `prefix_join` helper.
- Action: cancelled exactly `/benfeuer/snowball-final-qk157-grug-smoke13-coord` before its deterministic retry could repeat the 134 GB download. No training GPU was allocated and no update or checkpoint save occurred.
- Classification: this is a call-site error in the unmerged bridge, not an existing Marin defect, so no public issue was filed. The correction composes two joins, adds an object-store URI regression, and bumps the conversion artifact version to exclude the incomplete output.
- Next action: run focused and repository gates, push the correction from the isolated worktree, and launch smoke 14 with a fresh identity and JAX port.

### 2026-09-09 10:35 EDT - Historical Grug smoke 14 submitted

- Validation: fourteen focused tests and the complete repository pre-commit gate passed. Commit `05967167c2` is pushed on the isolated run branch.
- Run: submitted `/benfeuer/snowball-final-qk157-grug-smoke14-coord` from the worktree, stage version `2026.09.09.9`, conversion artifact version `2026.09.09.3`, and unique JAX port 19417.
- Next action: require conversion save success, then one finite eight-node update and a loadable checkpoint before any campaign fan-out.

### 2026-09-09 12:31 EDT - Native conversion committed; tokenizer boundary corrected

- Result: smoke 14 loaded all 39 pinned qk157 HF shards, imported and stacked the full model, and durably committed all 28 writes of the 124.94 GiB native checkpoint at conversion version `2026.09.09.3`. The conversion completed on attempt 0. All eight H100 workers then coordinated on port 19417 but failed before weight loading because the new historical wrapper supplied the conversion artifact's `s3://` URI as a tokenizer identifier; Levanter correctly rejected it as an invalid Hub repo. The deterministic retry was cancelled.
- Correction: restore the previously validated common tokenizer `marin-community/marin-tokenizer@a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2` as both the build-time tokenizer and construction-time cache key. Tests assert this exact immutable ref in the smoke, reload, OpenCode, and Nemotron configs and across all five bases, and reject storage-scheme tokenizer values.
- Validation: fourteen focused tests and full pre-commit/Pyrefly pass. The standalone `ty` executable remains absent from the environment.
- Run: commit `630157b418` is pushed. Submitted `/benfeuer/snowball-final-qk157-grug-smoke15-coord` from the isolated worktree, stage version `2026.09.09.10`, cached conversion version `2026.09.09.3`, and unique JAX port 19418.
- Next action: require smoke 15 to load the native checkpoint, produce one finite update, and save; then launch the separate native reload gate before campaign fan-out.

### 2026-09-09 12:41 EDT - Historical path reaches loss; batched-XLA VMA gap corrected

- Result: smoke 15 reused the durable native conversion, coordinated all eight H100 ranks on port 19418, accepted the pinned tokenizer, loaded the historical Grug checkpoint, and reached the real training loss. The H100 full-vocabulary batch-tiled `batched_xla` forward then rejected its `fori_loop` carry: replicated `loss`/`lse` zeros became varying across `(replica_dcn, data, expert)` after the first batch tile. No update or save ran; the deterministic retry was cancelled.
- Interpretation: the exact historical #8172 recipe selects `batched_xla`, so changing loss implementations would reduce path fidelity. The failure is a generic explicit-mesh bug in the current Levanter implementation, not a Qwen/Snowball model exception, and falls under public bug #9032's forward/backward loop-carry scope.
- Correction: preserve the input's manual-axis type on the H100 tiled forward and both custom-backward loop initializers, then sum batch-introduced manual axes before returning a shared-weight gradient to a replicated primal. One-device explicit `shard_map(check_vma=True)` regressions exercise the H100 tiled forward plus tiled and streaming backward paths and compare values/gradients with the dense reference.
- Validation: the complete fused-loss file passed 88 tests with 15 platform skips; fourteen Snowball/historical-bridge tests passed; full pre-commit, formatting, and Pyrefly checks passed.
- Next action: commit and push the correction, then launch smoke 16 from the isolated worktree with a fresh identity, version, and JAX port. Require one finite update, save, and separate native reload before fan-out.

### 2026-09-09 12:43 EDT - Historical Grug smoke 16 submitted

- Validation: the VMA correction and regressions are pushed in `33be7b0b56`; the complete fused-loss file passed 88 tests with 15 platform skips, fourteen Snowball/historical-bridge tests passed, and full pre-commit/Pyrefly passed.
- Run: submitted `/benfeuer/snowball-final-qk157-grug-smoke16-coord` from the isolated worktree, stage version `2026.09.09.11`, cached conversion version `2026.09.09.3`, and unique JAX port 19419. The root pruned the completed conversion dependency and allocated the eight historical-training workers on attempt 0.
- Next action: monitor this exact handle through native load, one finite optimizer update, and checkpoint save; then launch the separate native reload gate before campaign fan-out.

### 2026-09-09 13:40 EDT - Smoke 17 isolates rank-divergent collectives and historical wrapper drift

- Correction: smoke 16's coordinator requested `LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS=0`, but the historical dispatcher did not forward any `LEVANTER_` variables. The absence of fused-CE search logs did not prove the setting was active. Bug [#9044](https://github.com/marin-community/marin/issues/9044) tracks the propagation failure.
- Result: smoke 17 used `--xla_gpu_autotune_level=0`, bypassed the prior `ConfigAssigner` wait, compiled, and entered the first train-step executable. It then stalled with approximately 62.5 GiB allocated per H100, 100% utilization at only 116–126 W, and all ranks alive. NCCL RAS returned in 1 ms: 21 communicators were valid, no rank was missing or unresponsive, and the 64-rank communicator had `collective_mismatch=true` with AllReduce counts from 16 to 17. The exact owned root was cancelled without an update/save.
- Interpretation: autotune level 0 moved the same rank divergence from compile-time assignment to runtime collective execution and is removed from campaign policy. Comparing `8274751d290607` with the launched tree found that both the first-class and historical trainers import the campaign-modified shared Grug loss wrapper. The successful #8225 wrapper uses explicit input specs and fixed batch reduction axes; `908e81c73e` changed the reduced path to inferred input specs and trace-local varying axes. This is the next falsifiable cause.
- Next action: restore the explicit #8225 wrapper, forward and verify the Levanter control, run local numerical/gradient regressions, and launch smoke 18 with normal XLA autotuning and a fresh JAX port.

### 2026-09-09 13:50 EDT - Exact loss-wrapper discriminator launched

- Correction: commit `f722cdadd0` restores explicit loss shard-map inputs and fixed logical batch reductions, and forwards `LEVANTER_` settings through the historical dispatcher. The environment regression failed on the old allowlist and passes after the fix.
- Validation: twenty Snowball tests, nineteen historical import/campaign tests, eighty-eight fused-kernel tests with fifteen platform skips, and full pre-commit/Pyrefly pass. The advisory review found no issue in this commit. The standalone `ty` executable is absent.
- Run: submitted `/benfeuer/snowball-final-qk157-grug-smoke18-coord` at interactive priority from the isolated worktree, stage version `2026.09.09.13`, cached conversion version `2026.09.09.3`, and unique port 19421. It uses normal XLA autotuning and requests `LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS=0`.
- Next action: verify the child worker environment contains the Levanter control, then require native load, one finite optimizer update, and checkpoint save before the separate reload gate.

### 2026-09-09 13:56 EDT - Explicit historical loss boundary passes full-topology smoke

- Result: ranks 0 and 1 both exposed `LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS=0` and port 19421 in `/proc/1/environ`. All eight attempt-0 ranks completed the one-step loop in approximately 78.5 seconds and committed the 624.72 GiB step-1 checkpoint at `s3://marin-us-east-02a/marin/snowball-final/qk157/hf-smoke/2026.09.09.13/checkpoints/step-1`. Root and child succeeded with zero failures or preemptions.
- Interpretation: restoring the explicit #8225 loss shard map changes the failed outcome to a finite update/save under the same model, converted weights, mesh, kernel correction, runtime, and cluster. This strongly supports `908e81c73e`'s trace-inferred reduced-loss wrapper as the collective-divergence cause; the historical Grug trainer is not causal in this discriminator.
- Run: submitted native reload gate `/benfeuer/snowball-final-qk157-grug-smoke18-reload-coord` at interactive priority with stage version `2026.09.09.13` and unique port 19422.
- Next action: require weights-only load from smoke 18, a fresh step counter/optimizer, one finite update, and an independent checkpoint commit before releasing five Chat roots.

### 2026-09-09 14:04 EDT - Native reload gate passes

- Result: reload ranks 0 and 1 confirmed the Levanter setting and unique port 19422. The stage discovered smoke 18's step-1 checkpoint, found no stage-local resume checkpoint, loaded the parent weights into a fresh step-0 state, completed one update in approximately 76.7 seconds, and committed `s3://marin-us-east-02a/marin/snowball-final/qk157/hf-smoke-reload/2026.09.09.13/checkpoints/step-1`.
- Audit: root and all eight attempt-0 workers succeeded with zero failures or preemptions. The full HF conversion to native load, update, save, weights-only reload, update, and independent-save gate now passes.
- Next action: release the five Chat roots in parallel with distinct JAX ports, then evaluate each completed Chat checkpoint while advancing its chain to Thinking.

### 2026-09-09 14:05 EDT - Five Chat roots released

- Run: submitted qk157, qk175, qk175-skew2, qk175-skew4, and qk175-skew8 Chat roots at interactive priority with version `2026.09.09.13` and distinct ports 19423 through 19427. All five roots entered running.
- Scheduling: each stage is a separate root so sibling and cross-base gangs never inherit the same JAX port. Qk157 reuses its validated conversion; the other four roots first build their immutable HF-to-native conversion dependencies.
- Next action: monitor conversions and Chat training independently; as each Chat checkpoint commits, launch its non-agentic evaluation and its Thinking child with new unique ports.

### 2026-09-10 11:01 EDT - AIME policy matrix completed and repaired evaluators pinned

- Code refs: campaign branch `93251781092fa281077dff0a9c7b455fd6cc9677`; Evalchemy `f7c85686d4b8d1957135a7a18391c2023b670e8d`; Harbor `ce2a55949f99f2d3baa48e5b523e0703f81280c0`.
- Command: `uv run python config/update-external.py evalchemy harbor`, followed by `uv run python config/update-external.py --check evalchemy harbor` in `/Users/benfeuer/Documents/marin-worktrees/sft-snowballs-final`.
- Durable audit: RNO2A CPU jobs `/benfeuer/snowball-final-record-audit-20260910` and `/benfeuer/snowball-final-aime-summary-20260910` scanned `s3://marin-us-east-02a/marin/evals` from inside the credentialed cluster boundary.
- Result: all 13 previously missing AIME seed/model cells are `status=succeeded`, contain `accuracy_avg`, and attempted all 30 questions. `RESULTS.md` now reports seeds 42–51 for all five Base models instead of leaving seeds 49–51 pending.
- Next action: exercise the repaired Evalchemy task routes with bounded canaries, repair Marin's verifier-secret forwarding boundary for SimpleQA, and rerun the incomplete targeted benchmarks before full five-model fan-out.

### 2026-09-10 16:30 EDT - Custom benchmark cap gates passed; full recovery released

- Code refs: Evalchemy #116 at `d6c02d2606709087a70de782857c8061649f490f` centralizes unique-source sample caps and adds the future-custom-benchmark contract; Evalchemy #117 at `1d6943d88062f7d87976adf59792dd4f35a3983a` serializes CruxEval directional samples; campaign pin `dd4131163d75bd06651fc40eb4c325ca776373fb`.
- Runtime gates: group `/benfeuer/eval-20260910-194301-snowball-final-qk157-base-abde` wrote one capped MMLU-Pro result and one capped MRCR result. Fresh Crux gate `/benfeuer/eval-20260910-201418-snowball-final-qk157-base-26e2` succeeded on #117 and compacted exactly two directional records for one source.
- Runs: released qk157 CruxEval/MRCR group `…-c592`; released MMLU-Pro/CruxEval/MRCR groups qk175 `…-7f85`, skew2 `…-8d90`, skew4 `…-bfea`, and skew8 `…-9289`. The older qk157 full MMLU-Pro run remains `…-2429` and is not duplicated.
- FinanceBench: skew2 `…-6394` completed at 44% (22 correct, 20 incorrect, 8 not attempted; 50/50) and skew8 `…-5f04` at 42% (21/19/10; 50/50). Qk175 and skew4 failed on transient Together HTTP 503 responses and were resubmitted as `…-3362` and `…-40cc` without changing policy.
- Observation: typed parsing of the one-item MMLU-Pro canary rejects its `accuracy_std_err: null`; this does not block the full run, whose sample count should make the uncertainty numeric. Campaign collector now reports one malformed record without aborting the rest of a batch.
- Next action: monitor the full custom, FinanceBench retry, SimpleQA, DS-1000, and hardened agentic roots; reduce only durable successful records meeting full-coverage gates into external `RESULTS.md`.

### 2026-09-10 16:42 EDT - FinanceBench five-base matrix complete

- Evidence: read-only RNO2A snapshot `/benfeuer/snowball-final-finance-summary-20260910` parsed the two retry records against campaign schema. Qk175 `…-3362` scored 40% (20 correct, 19 incorrect, 11 not attempted); skew4 `…-40cc` scored 44% (22/22/6). Both have 50 attempted and 50 scored, empty error maps, Evalchemy runtime `1d6943d…`, and Marin provenance `dd4131163d…`.
- Result: combined with qk157 44% (`…-9dc4`), skew2 44% (`…-6394`), and skew8 42% (`…-5f04`), FinanceBench is complete for all five Base checkpoints under the unchanged Together `openai/gpt-oss-120b` judge policy.
- Next action: continue monitoring the remaining custom, SimpleQA, DS-1000, and agentic recoveries; snapshot and report each only after its full coverage gate passes.

## 2026-09-10 16:47 EDT — Full recovery progress denominators

- Verified immutable suite sizes: DS-1000 1,000, SimpleQA 4,326, SWE-bench random recovery 100, Terminal-Bench 2.0 89, OT-TBLite 100, MMLU-Pro test 12,032, CruxEval 799 sources/two directions, and MRCR 1,200 selected sources at the served 65,536-token context limit.
- Persisted progress counters were DS-1000 239–260, SimpleQA 125–469, and the first SWE stage 34–43 per Base. These are operational counters only, not coverage claims, because retries may be represented.
- All full custom evaluators remained running. One MMLU-Pro evaluator logged a transient HTTP 502 and entered its configured retry path; no new blocking Harbor or Evalchemy defect was established.

## 2026-09-10 17:00 EDT — Recovery progress remains live

- DS-1000 advanced to 275–307 persisted trial events per Base and SimpleQA to 140–493, materially above the 16:47 snapshot.
- Qk157 CruxEval remained controller-running with zero failures or preemptions after 851 successful generations. Its current 16-request decode batch had produced no completion for roughly 14 minutes, which remains inside Marin's configured 30-minute Evalchemy request timeout.
- No restart was launched; continue observing the exact handle and classify only after it advances, times out into the typed retry/failure path, or becomes terminal.

## 2026-09-10 17:13 EDT — DS-1000 progress checkpoint

- All five DS-1000 roots remained live and reached 318–355 persisted trial events out of 1,000 per Base, up from 275–307 at 17:00.
- No newly failed targeted root appeared. Qk157 CruxEval resumed incremental successful responses after its long decode batch, so no restart or framework repair was justified.

## 2026-09-10 17:20 EDT — DS-1000 timeout recovery replaced

- Evidence: qk157 alone logged 132 distinct `AgentTimeoutError` trials before task ordinal 400; the other Bases logged 146–150. Representative trial `390__jeuTrAX` spent 904 seconds between agent start and verification, exactly matching the task's 900-second agent timeout, then persisted as failed. Every first-attempt full run was therefore already unable to meet the 95% coverage gate.
- Policy: external `eval-configs/ds-1000.yaml` now sets `agent_timeout_multiplier: 8.0`. Harbor's isolated preflight resolved this as an agent-only multiplier with the global multiplier at 1.0 and no verifier multiplier, yielding a 7,200-second agent ceiling without changing scoring.
- Lifecycle: cancelled only the five invalid DS roots (`…-3e51`, `…-0606`, `…-7692`, `…-1fbf`, `…-5e6d`). Submitted fresh full roots qk157 `…-2df9`, qk175 `…-5cfa`, skew2 `…-991f`, skew4 `…-d225`, and skew8 `…-1aaa` from campaign worktree commit `dd4131163d` at interactive priority on `cw-rno2a`.
- Initial state: qk157 root entered running with its serve child pending on the peer; the other four roots briefly queued while cancelled serve pods released. By 17:21 EDT, all five roots and all five serve children were running on RNO2A. The first submission attempt created no jobs because the fresh shell lacked ADC; retry loaded `/Users/benfeuer/Documents/secrets.env` so the existing environment-backed Daytona secret resolved. No secret values were printed or recorded.

## 2026-09-10 17:25 EDT — SimpleQA timeout recovery replaced

- Evidence: trial `simpleqa-3111__XaapTrV` started its agent at 20:25:56 UTC, entered verification at 21:16:00, completed verification successfully, and persisted as `AgentTimeoutError`. The 3,004-second agent interval matches the task's 3,000-second ceiling. Across the five roots, distinct timeout counts were 27/543, 12/163, 53/431, 53/391, and 51/431 persisted trial events; four were already below 95% among completed events and qk157 was at the edge.
- Policy: external `eval-configs/simpleqa.yaml` now sets `agent_timeout_multiplier: 2.4`. Harbor's isolated preflight resolves this as an agent-only multiplier with the global multiplier at 1.0 and no verifier multiplier, yielding a 7,200-second agent ceiling without changing judge or verifier behavior.
- Lifecycle: cancelled only the five invalid SimpleQA roots (`…-6154`, `…-193e`, `…-d1fd`, `…-00e0`, `…-6619`). Submitted fresh full roots qk157 `…-86de`, qk175 `…-0f33`, skew2 `…-0d04`, skew4 `…-c792`, and skew8 `…-337c` from campaign worktree commit `dd4131163d` at interactive priority on `cw-rno2a`, using the established Together judge environment from `/Users/benfeuer/Documents/secrets.env`.
- Initial state: qk157 and qk175 roots were running; the three skew roots were pending. All five reported zero failures and zero preemptions.
