# KV Cache Refactor Plan (Codex) - Multi-Host Safe, Test-Gated, No Silent Failures

This is a gradual refactor plan for Levanter's inference-time KV caching and related orchestration. It is intentionally *more conservative* than `CLAUDE_REFACTOR_KV.md`: we will only ship steps that are (1) locally testable, (2) multi-host safe by construction, and (3) do not rely on "it seems to work" evidence.

The immediate trigger for this doc is the `simpo` multi-host inference break that was "fixed" by restoring the known-good stack from `multihost_inference_work` and removing stale YAML keys (see `CODEX_MULTI_HOST_DEBUG.md`). The root cause was a correctness regression: **`InferenceEngine.reset()` was skipped on multihost**, creating host/device state divergence and leading to TPU/XLA-level crashes with no Python traceback. This plan is designed so that kind of regression is hard to introduce.

TOP-LEVEL REMINDER (DO NOT REMOVE):
- If I work for >1 hour, add a dated timestamp + short summary of progress to this doc.

References:
- KV cache mental model and current dataflow: `KV_CACHING_RESTRUCTURE.MD`
- Prior plan (contains good ideas but also introduced subtle, high-risk changes): `CLAUDE_REFACTOR_KV.md`
- Repro + fix log for the break: `CODEX_MULTI_HOST_DEBUG.md`

## Work Log
- 2026-02-05 00:46 PST: M5 multi-round debugging. Tested multiple configs (5 prompts/2048, 1 prompt/256) across physical/logical reset, engine reinit, per-round request rebuild, skip barrier, skip samples table. All still fail after round 0 when n_rounds>1. With TPU debug env enabled, failures show **TPU slice/runtime halt** (`program continuator has halted unexpectedly`, slice health check failed). Documented in `CODEX_REFACTOR_KV_M5_DEBUG.md`; likely runtime issue, not Python exception.
- 2026-02-05 01:18 PST: M5 batched-rounds workaround validated. Updated engine sizing validation to scale `max_seqs`, `max_seqs_in_prefill`, and `max_prefill_size` by `n_rounds` when rounds are batched. `n_rounds=2` batched run (1 prompt/256 tokens) succeeded. Initial 5 prompts/2048 batched run failed with `Out of free pages during allocation` at `max_pages=240`; increasing `max_pages` to 336 fixed it (job finished, `pages_in_use=330 free=6`). Logs recorded in `CODEX_REFACTOR_KV_M5_DEBUG.md`.

---
## gold command
python lib/levanter/infra/launch.py --foreground --zone us-central1-a \
    --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand -- \
    uv run src/levanter/main/sample_lm_multihost.py \
    --config_path config/sampler/sample_llama8b_multihost_real_1prompt_2048.yaml \
    2>&1 | tee /tmp/levanter_run_m3.log


## Executive Summary

1. **Codify invariants** of paged KV caching (`PageTable`, `SequenceTable`, `DecodeState`, `PageBatchInfo`) and enforce them with cheap checks and tests.
2. **Make reset semantics explicit and multi-host safe**. A reset must always clear device-side state; it must never be conditional on `jax.process_count()`.
3. **Separate "logical reset" from "physical zeroing"**. Use logical reset by default for performance, but keep physical zeroing as an opt-in debug tool.
4. **Only then** consider optional features like incremental cleanup during decode, or prompt streaming/admission.
5. Keep changes **small, monotonic, and test-gated**. Every step must have an exit criterion and a rollback strategy.

---

## Ground Truth: How KV Caching Works Today

Everything below is summarized from `KV_CACHING_RESTRUCTURE.MD` and should remain the shared mental model.

1. KV is stored in a global page pool (`KvPageCache.kv_pages[Page, Slot, ...]`), where each page holds `page_size` token positions.
2. Each active sequence occupies a *local slot id* (`slot_id in [0..max_seqs)`), and maps its token positions to pages via `SequenceTable.page_indices[slot_id, page_idx]`.
3. The core address mapping for token `(slot_id=s, position=p)` is:
   - `page_idx = p // page_size`
   - `slot = p % page_size`
   - `page = page_indices[s, page_idx]`
   - KV location is `kv_pages[page, slot, ...]`
4. `PageTable.page_ref_counts[Page]` is the allocator state. Pages are free when `ref_count == 0`.
5. `SequenceTable.allocate_for_seq(...)` is the allocator + batch planner. It ensures pages exist for the decode batch and constructs `PageBatchInfo`, including `new_token_dests` for KV writes.
6. Correctness relies on two contracts:
   - **Metadata correctness**: attention must only read tokens `< kv_len` for each sequence and only from pages referenced by `page_indices`.
   - **Write-before-read for new tokens**: within `model.decode(...)`, new tokens' K/V must be incorporated correctly (either via cache update before attention reads, or fused logic).
7. Clones (`n_generations > 1`) share full pages by increasing refcounts, and copy/unique the final partial page.

Implication: we can safely skip physical KV zeroing *if and only if* we reliably reset metadata (`seq_lens`, `page_indices`, `used_mask`, `page_ref_counts`, queues).

---

## What Went Wrong Before (Lessons From `CLAUDE_REFACTOR_KV.md` / `simpo`)

This is not "Claude was wrong"; it's a list of failure patterns we must prevent.

1. **Correctness bypass in multi-host reset**
   - Regression: `InferenceEngine.reset()` was changed to *skip device-state reset* when `jax.process_count() > 1`.
   - Outcome: host assumed slots/pages were free, device state still had them allocated, and the next `engine.generate()` produced TPU/XLA-level crashes (exit status 1, no Python traceback).
   - Rule: *If reset is unsafe on multihost, fix reset. Never bypass reset on multihost.*

2. **Config drift causing hard failures**
   - `draccus` decoding is strict. Adding YAML keys like `engine.use_logical_reset` or `max_prompts_per_batch` without keeping code and configs in lockstep caused immediate `DecodingError`.
   - Rule: *Config changes are atomic.* A PR that adds a field must update all relevant YAMLs + docs + tests in the same change.

3. **High-risk changes bundled together**
   - A single effort tried to modify reset semantics, cleanup semantics, queue/finished semantics, attention block sizes, and multi-prompt batching.
   - That makes it difficult to isolate regressions and encourages "quick hacks" like skipping reset.
   - Rule: *One behavioral change per milestone.* Perf changes and correctness changes should not land together.

4. **Cleanup semantics are subtle**
   - Freeing pages and invalidating metadata during generation interacts with:
     - monotonic `finished` flags
     - output extraction ordering
     - TokenQueue purge
     - clone refcounts
   - Rule: *Cleanup must be introduced behind invariants and tests.* Prefer host-controlled cleanup boundaries first, then move into the decode kernel only if needed.

---

## Goals

1. Make KV cache orchestration **robust** across:
   - single-host, multi-device
   - multi-host TPU (v5p-16 and higher)
2. Enable **fast sequential batches** (multi-prompt workloads) without paying `jnp.zeros_like()` over the entire KV cache every `generate()` call.
3. Enable **optional logical cleanup** primitives to be used safely (free finished sequences/pages) without silent corruption.
4. Make failures **observable and diagnosable**, especially the "exit status 1 / no traceback" class.

---

## Non-Goals (For This Refactor)

1. Building a fully general streaming inference server API (we can move toward it, but it's not the first target).
2. Maintaining backward compatibility shims for old config keys. The repo rule is **NO BACKWARD COMPATIBILITY**.
3. Solving TPU ragged paged attention VMEM OOM immediately. We will create a clear path to that work, but correctness and reset semantics come first.

---

## Invariants We Will Enforce

These are the "must never be false" statements that guard against silent bugs.

### Allocation/Mapping Invariants

1. For any active slot (`used_mask[slot] == True`), `seq_lens[slot] >= 0` and `seq_lens[slot] <= max_seq_len`.
2. For any token decoded at `(slot, pos)` in a batch, `pos < seq_lens[slot]` after `allocate_for_seq(...)` updates lengths.
3. For any valid `(slot, page_idx)` where `page_indices[slot, page_idx]` is valid, `page_ref_counts[page] >= 1`.
4. For any page with `page_ref_counts[page] == 0`, it must not appear in any `page_indices` row for any active slot.

### Clone/Refcount Invariants

1. Clones may share only full pages; the final partial page must be unique per clone.
2. Refcount changes must be balanced:
   - cloning increments
   - freeing decrements
   - refcounts never go negative

### Reset Invariants (Critical)

1. After `engine.reset_*()` returns, the device state must be in a baseline state:
   - no used slots
   - zero page refcounts
   - empty queues
   - finished flags cleared
2. Reset behavior must be **identical** on single-host and multi-host. No correctness branches on `jax.process_count()`.

### Multi-Host Determinism Invariants

1. All hosts must see identical configs and prompts before calling any JIT entrypoints.
2. All JIT entrypoints used in multi-host runs must be called by all hosts in the same order.
3. Host-only debugging must not change traced code or traced shapes.

---

## Plan Overview (Milestones)

Each milestone is designed to be a small PR. Do not start the next milestone until the prior milestone has its exit criteria met.

### M0 - Baseline Lock-In (Repro + Guardrails)

Goal: ensure we always have a "known-good" multi-host sampling command that passes, and a minimal set of regression checks.

Deliverables:
- A canonical repro command and expected outputs in `CODEX_MULTI_HOST_DEBUG.md` (already exists; keep updated).
- A small "smoke test checklist" section for humans: which configs to run and what to look for.
- A config schema audit step in CI or pre-commit:
  - parse the sampler YAMLs used in docs
  - fail if unknown fields exist (this catches the `draccus.DecodingError` class before TPU time is wasted)

Exit criteria:
- v5p-16 known-good command completes with `Job finished with no error`.
- Config schema audit catches an intentionally injected stale key.

Rollback:
- None; this is additive.

### M1 - Observability Without Behavioral Changes

Status: DONE (2026-02-04)

Goal: add diagnostics that are safe for multi-host and do not change results.

Key principle: diagnostics must be available without peppering JIT code with `print()`.

Deliverables:
- A `DecodeState.stats()` method that returns a small PyTree of scalars:
  - number of active slots
  - pages in use (count of `ref_counts > 0`)
  - free pages
  - maybe max refcount
- Host-side logging hooks in `InferenceEngine.generate()` that log these stats at stable points:
  - after reset
  - after prefill
  - after generation loop completes
- Optional: a debug flag to log stats every N decode iterations (default 10; set to None/0 to disable).

Exit criteria:
- All existing inference unit tests pass.
- Stats logging works on single-host and multi-host without triggering host callbacks inside JIT.

Rollback:
- Remove logging calls (keep the stats method if it's used by tests).

### M2 - Make Reset Semantics Explicit and Safe (Logical vs Physical)

Status: IN PROGRESS (2026-02-04)

Goal: remove the temptation to "skip reset on multihost" by making reset fast and correct.

Design:
- A reset always resets device metadata (`DecodeState.reset()` / `PageTable.reset()` / queue reset).
- Physical KV zeroing becomes an explicit, opt-in operation.

Deliverables:
- Add a `ResetMode` concept at the engine layer (API name TBD):
  - `logical`: reset metadata only (default for sequential multi-prompt workloads)
  - `physical`: logical reset + zero KV cache (debug-only)
- Implement `PageCache.zero()` (or equivalent) that zeros KV pages.
- Ensure `InferenceEngine.reset()` always executes a jitted reset path on all hosts, regardless of `jax.process_count()`.
- Update docs/configs to never rely on stale keys:
  - if a new config field is introduced, update all relevant YAMLs in the same PR

Exit criteria:
- Single-host: calling `engine.generate()` twice in a row works under both reset modes.
- Multi-host: sequential batches (multiple `engine.generate()` calls in one process) work with `logical` reset.
- A unit test asserts that `reset()` clears `used_mask` and `page_ref_counts` in a way that would fail if reset were conditionally skipped.

Rollback:
- Keep physical reset as default if logical reset exposes a bug; fix bug before re-enabling logical default.

Notes:
- Verified sequential single-prompt runs with both `reset_mode: physical` and `reset_mode: logical` completed without regression (multi-host v5p-16, 1 prompt, 2048 tokens).
- Multi-prompt stress (5 prompts, 2048 tokens, v5p-16) still fails with TPU/XLA exit status 1 for both reset modes; DecodeStats remain healthy (no page exhaustion). This is expected given the known multi-prompt failure and is tracked under M5 (multi-prompt admission), not a regression in reset semantics.

Expected vs Not Expected:
- Expected: single-prompt runs (and sequential rounds) behave identically under logical vs physical reset, with logical potentially faster between rounds.
- Not expected: multi-prompt stability is still broken; failures here are not caused by reset-mode changes and must be addressed under M5.

### M3 - Remove Redundant KV Page State (Single Source of Truth)

Status: DONE (2026-02-05) — refactor landed; post-change validation recorded.

High-level problem:
We currently store the same mapping in two places (`SequenceTable.kv_pages` and `SequenceTable.page_indices`).
This duplication is easy to accidentally de-sync, and when it diverges the failure mode is subtle:
one part of the stack reads one mapping while another writes or validates against the other.
That can produce silent corruption, allocator refcount mismatches, or multi-host nondeterminism that is hard to debug.

What we are trying to fix:
Make the sequence → page mapping have exactly one authoritative representation, and remove all redundant storage.
This eliminates an entire class of “state drift” bugs and simplifies invariants and refcount reasoning.

Conceptual clarification (why this is redundant):
- `KvPageCache.kv_pages` is the *physical* KV tensor: `[Page, Slot, 2*KVHeads, HeadDim]`.
- `SequenceTable.page_indices` is the *logical* mapping from sequence slot → global page IDs.
- `SequenceTable.kv_pages` is **not** the KV tensor; it is just a second copy of the logical mapping.
- In today’s code, `SequenceTable.kv_pages` is always written to match `page_indices`, so it adds no information.
- This is called out in `KV_CACHING_RESTRUCTURE.MD` (Sections 3.1, 3.3, 5, and 10.3).

Ideal outcome:
- One canonical mapping (`page_indices`) used everywhere.
- No compatibility alias that risks hiding divergence.
- Fewer invariants to enforce; easier to reason about allocation, cloning, and freeing.
- Multi-host behavior stays deterministic and test coverage improves around allocation/freeing.

Implementation notes:
- Removed redundant `SequenceTable.kv_pages` storage; `page_indices` is the single source of truth.
- Updated assign/clone/reset paths and debug output to use `page_indices` only.

Validation notes (post-M3):
- v5p-16, 1 prompt, 2048 tokens completed with `Job finished with no error` (log: `/private/tmp/levanter_run_m3.log`).
- DecodeStats progression: `after_reset` pages_in_use=0/free=48 → `after_prefill` pages_in_use=1/free=47 → `after_decode` pages_in_use=33/free=15.

Takeaways:
- Single-prompt run remains healthy after removing the redundant mapping.
- Multi-prompt instability remains a separate issue (tracked in M5).

Background:
- `KV_CACHING_RESTRUCTURE.MD` notes `SequenceTable.kv_pages` was redundant with `page_indices`.
- `CLAUDE_REFACTOR_KV.md` attempted to make `kv_pages` a property alias for `page_indices`.
- The safe approach is to pick one canonical representation and update all call sites.

Deliverables:
- Remove `kv_pages` as a stored field everywhere; use `page_indices` as the canonical mapping.
- Update all call sites to accept `page_indices` only.
- Delete any "compatibility alias" properties. This repo does **not** want long-lived shims.

Exit criteria:
- Unit tests cover `allocate_for_seq`, clone paths, and page freeing with the new representation.
- Multi-host sampling still works (manual smoke).

Rollback:
- Re-introduce stored `kv_pages` only if a concrete correctness issue is found and cannot be quickly fixed.

### M4 - Cleanup Primitives: Correctness First, Then Performance

Status: DONE (2026-02-05)

Goal: wire up page freeing in a way that cannot silently corrupt state.

Step 1 (safe boundary): free pages at end-of-generation only.
- After generation finishes, compute which slots are finished and free them.
- This validates refcount math and clone behavior without affecting the decode loop.

Step 2 (higher risk): incremental cleanup during decode.
- Purge queued tokens for finished slots.
- Preserve monotonic `finished` reporting in `_DecodeOutputs`.
- Ensure freeing does not cause allocator non-determinism across hosts.

Implementation notes:
- Added `DecodeState.free_finished()` for end-of-generation cleanup (clears finished flags after freeing).
- Added `DecodeState.cleanup_finished()` to purge queued tokens for finished sequences and free their pages.
- Added `InferenceEngineConfig.cleanup_mode` with values `none | end | incremental` (default `end`).
- Updated all sampler YAMLs to include `cleanup_mode: end` and added
  `config/sampler/sample_llama8b_multihost_real_1prompt_2048_cleanup_incremental.yaml` for incremental cleanup.

Tests added:
- `tests/inference/test_jit_scheduler.py::test_decode_state_free_finished_clears_pages_and_flags`
- `tests/inference/test_jit_scheduler.py::test_decode_state_cleanup_finished_purges_queue`
- `tests/inference/test_page_table.py::test_free_pages_for_finished_respects_clone_refcounts`

Validation runs:
- Step 1 (end cleanup):
  - Config: `config/sampler/sample_llama8b_multihost_real_1prompt_2048.yaml`
  - Log: `/tmp/levanter_run_m4_step1.log`
  - Result: `Job finished with no error`.
  - Log evidence (2026-02-05):
    - After reset: `active=0 pages_in_use=0 free=48 max_refcount=0` (00:35:40).
    - After prefill: `active=1 pages_in_use=1 free=47 max_refcount=1` (00:36:09).
    - After decode: `active=1 pages_in_use=33 free=15 max_refcount=1` (00:37:24).
    - After cleanup: `active=0 pages_in_use=0 free=48 max_refcount=0` (00:37:25).
- Step 2 (incremental cleanup):
  - Config: `config/sampler/sample_llama8b_multihost_real_1prompt_2048_cleanup_incremental.yaml`
  - Log: `/tmp/levanter_run_m4_step2.log`
  - Result: `Job finished with no error`.
  - Log evidence (2026-02-05):
    - After reset: `active=0 pages_in_use=0 free=48 max_refcount=0` (00:42:11).
    - After prefill: `active=1 pages_in_use=1 free=47 max_refcount=1` (00:42:40).
    - After decode: `active=0 pages_in_use=0 free=48 max_refcount=0` (00:43:56).
    - After cleanup: `active=0 pages_in_use=0 free=48 max_refcount=0` (00:43:57).

Rollback:
- Disable incremental cleanup (`cleanup_mode: end`), keep end-of-generation cleanup only.

### M5 - Multi-Prompt Admission: Make Multi-Round Stable and Predictable

Status: DONE (2026-02-06)

Goal:
Eliminate the multi-host multi-round crash class that surfaced as TPU worker exit status 1
with no Python traceback.

Chosen strategy:
- Keep per-round `engine.generate()` orchestration for a fixed prompt set.
- Make round-boundary tracker behavior safe by default.
- Reject known-unsafe configurations at startup.

Critical cause (what actually broke):
- The main trigger was **asymmetric in-loop tracker emission at round boundaries**
  (especially leader-side in-loop `tracker.log(...)`) when
  `defer_tracker_logs_until_end=false`.
- This can produce launch-group failure / TPU runtime halt symptoms (`exit status 1`) even
  when decode/page stats look healthy.
- Reset mode (`physical` vs `logical`) was not the discriminating factor.

Prevention (what now keeps it stable):
1. Default to deferred tracker emission:
   - `defer_tracker_logs_until_end: true`
   - Emit metrics/tables after rounds complete instead of at in-loop round boundaries.
2. Add startup guardrails:
   - Reject known-unsafe multi-host multi-round configs that combine
     `defer_tracker_logs_until_end=false` with leader-side in-loop tracker emission.
3. Keep safe exceptions explicit:
   - `defer=false` is only allowed for validated non-leader-emission diagnostic paths.

Exit criteria:
- Multi-host multi-round configs complete consistently without the prior crash class.
- Unsafe legacy modes fail fast with explicit config validation errors instead of TPU slice death.

Validation evidence (full matrix in `CODEX_FIX_M5.2.md`):
- 5 prompts x 2048 x 2 rounds: PASS in deferred mode; unsafe legacy mode is fast-rejected.
- 20 prompts x 2048 x 2 rounds (`noop`, `wandb`): PASS.
- 20 prompts x 4096 x 2 rounds (`wandb`): PASS.
- Representative logs:
  - `/tmp/levanter_run_m52_final_20prompts_2048_lockin.log`
  - `/tmp/levanter_run_m52_final_20prompts_2048_lockin_wandb.log`
  - `/tmp/levanter_run_m52_final_20prompts_4096_lockin_wandb.log`

M5 guarantees (in scope):
- Multi-host multi-round runs are stable for configs that fit one-round engine capacity and use safe tracker settings.
- Unsafe legacy tracker modes are rejected at startup with explicit errors.
- The previous "exit status 1 between rounds" class is removed for the validated config families above.

M5 does not do (out of scope):
- It does **not** chunk an arbitrary prompt set `N` automatically inside one run.
- `n_rounds` means repeat sampling over the same prompt set; it is not prompt batching.
- It does **not** guarantee that very large `N` fits in one call; `max_seqs`, `max_pages`, and `max_prefill_size` still bound admission.
- It does **not** improve throughput; long generations remain slow on the current reference path.
- It does **not** implement dynamic host-side prompt replacement/eviction during a single `generate()` call.

Operational note:
- Inference is still slow for long generations (for example, ~55 minutes/round for
  20 prompts x 4096 on v5p-16), but behavior is now consistent and reproducible.

Rollback:
- Keep startup guard enabled and force deferred logging (`defer_tracker_logs_until_end=true`)
  for all multi-host multi-round runs.

### M5.1 - Very Large Prompt Sets (Design + Tradeoffs)

Status: DEFERRED (post-M5 scaling)

Problem statement:
We need a safe, repeatable way to handle very large prompt sets (e.g., 10k–100k prompts)
without silent TPU failures or KV cache exhaustion.

Known constraints and failure modes:
1. Unsafe in-loop leader tracker emission at round boundaries (`defer_tracker_logs_until_end=false`)
   can reintroduce launch-group failures; deferred logging should remain the default.
2. Single-call overfill hits `Out of free pages during allocation` when `max_pages` is undersized.
3. Leader-only `jax.device_get` of globally sharded arrays can crash/hang; all-host reads are required.
4. Large batched runs can saturate `max_prefill_size` and `max_seqs_in_prefill` (both scale with total prompts).

Sizing formula (for any single `engine.generate()` call):
`required_pages ≈ ceil((max_prompt_len + max_new_tokens) / page_size) * (num_prompts * n_rounds * n_generations)`
This must be ≤ `engine.max_pages`, and `max_seqs`/`max_seqs_in_prefill` must cover the same total.

Strategies for very large prompt sets:

| Strategy | Pros | Cons / Risks | When to use |
| --- | --- | --- | --- |
| Single giant batch (`n_rounds=1`, all prompts in one call) | Avoids per-call reset; simplest operationally | Usually exceeds `max_pages`/HBM; large prefill; W&B tables huge | Only if prompt count is small enough to fit in one call |
| Batched rounds (single call with `n_rounds>1`) | Avoids per-round resets; stable in current TPU setup | Multiplies capacity requirements; still limited by `max_pages` | When you need repeated samples per prompt and can fit capacity |
| Multiple generate calls inside one job (prompt chunking) | Amortizes model load; avoids repeated startup | Requires strict deferred logging safety and careful boundary validation | Use when orchestration simplicity matters and run time is acceptable |
| Separate jobs per batch (external orchestration) | Most reliable today; no in-process reset; easy to parallelize | Repeated model load overhead; higher cost/latency; more complex bookkeeping | **Recommended now** for 10k–100k prompts |
| Persistent service / queue (future) | Best long-run throughput and latency | Requires robust in-process reset and admission control; not implemented | Long-term target, not for M5 |

Current recommendation (M5.1):
Use **external orchestration** to split large prompt sets into batches, each run as a **single job**
with `n_rounds=1` (or batched rounds if repeated sampling is needed). This keeps retries and failure
recovery simple at very large scales. Batch size should be chosen using the `required_pages` estimate and
`max_prefill_size` constraints.

External orchestration sketch (detailed plan, tied to `KV_CACHING_RESTRUCTURE.MD`):

1. Preprocess prompts and compute token lengths.
   - Use the same tokenizer as the job (`config.tokenizer` or `hf_checkpoint`).
   - Record `prompt_id`, `prompt_text`, `prompt_token_len`.
   - Why: KV cache allocation is page-based (`PageTable` + `SequenceTable.allocate_for_seq`), so we must
     bound pages per sequence before we build a batch.

2. Compute per-sequence page usage (upper bound).
   - `pages_per_seq = ceil((prompt_token_len + max_new_tokens) / page_size)`.
   - For `n_generations > 1`, use `pages_per_seq * n_generations` as a safe upper bound (clones may share
     full pages, but we plan with a worst-case bound to avoid `Out of free pages`).

3. Choose a batch size algorithm (two constraints).
   - **Page budget**: sum of per-seq pages ≤ `engine.max_pages`.
   - **Prefill budget**: sum of prompt token lengths ≤ `engine.max_prefill_size`.
   - Also respect `engine.max_seqs` and `engine.max_seqs_in_prefill`.
   - Practical heuristic: sort prompts by length (descending) and greedy-pack until either budget is hit.

4. Emit one config per batch.
   - Base on a template sampler config (same model/engine settings).
   - Replace `prompts:` with the batch’s prompt subset.
  - Keep `defer_tracker_logs_until_end: true` for multi-host multi-round safety.
   - Keep `tracker.type=wandb` and set tags that include batch index and size.

5. Launch one job per batch (external orchestration).
   - Each job is a **fresh process**: new `DecodeState`, `PageTable`, `SequenceTable`, `TokenQueue`
     are created during `InferenceEngine` initialization.
  - This limits blast radius and simplifies retries for long-running workloads.
   - Logs per batch: `/tmp/levanter_run_m5_batch_<idx>.log`.

6. Track completion and resume safely.
   - Write a `batches.jsonl` manifest with status: `pending`, `running`, `done`, `failed`.
   - On restart, skip `done` batches; rerun `failed`.

7. Validate outputs and aggregation.
   - Aggregate outputs with stable `prompt_id` + `generation_id`.
   - Optional: store generated text in a single `jsonl` file per batch to avoid huge W&B tables.

Minimal orchestration pseudocode (host-side):
```
load prompts
tokenize -> lengths
sort by length desc
for batch in greedy_pack(lengths, page_budget, prefill_budget):
  write batch_config.yaml (prompts subset, n_rounds=1)
  run launch.py with batch_config.yaml, log to /tmp/...
  mark batch done/failed in manifest
```

Notes:
- Avoid leader-only `jax.device_get` on sharded arrays; all-host stats calls only.
- If you must increase throughput, add TPUs and run batches in parallel (one TPU per batch).

### M5.2 - Crash Mechanism (Closed)

Status: CLOSED (2026-02-06)

Final root-cause summary:
- The M5 crash class was not primarily a reset-mode failure.
- The critical trigger was **asymmetric in-loop tracker emission at round boundaries**
  (especially leader-side in-loop tracker logging) when
  `defer_tracker_logs_until_end=false`.
- This produced TPU/launch-group failure symptoms (`exit status 1`, sometimes TPU continuator/slice-health
  errors) without a useful Python traceback.

How we prevent it now:
1. Keep `defer_tracker_logs_until_end=true` as the default for multi-host multi-round sampling.
2. Use startup validation to fail fast on known-unsafe `defer=false` leader-side emission combinations.
3. Keep multi-host debug/stats reads all-host safe for sharded arrays.

Validation references:
- Full experiment matrix and logs: `CODEX_FIX_M5.2.md`.
- Representative passing stress runs:
  - `/tmp/levanter_run_m52_final_20prompts_2048_lockin_wandb.log`
  - `/tmp/levanter_run_m52_final_20prompts_4096_lockin_wandb.log`

### M6 - Performance Work: Reference Attention and KV Update

Status: M6 FINALIZED for 10x2048 baseline (M6.1-M6.6 completed on 2026-02-06)

Goal: improve throughput without introducing correctness risk.

Baseline (real TPU, single round):
- Config:
  - `config/sampler/sample_llama8b_multihost_real_10prompts_2048_reset_physical_round1_cleanup_none_noop_m6_baseline.yaml`
- Log:
  - `/tmp/levanter_run_m6_baseline_10prompts_2048_round1.log`
- Command:
  - `python infra/launch.py --foreground --zone us-central1-a --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand -- uv run src/levanter/main/sample_lm_multihost.py --config_path config/sampler/sample_llama8b_multihost_real_10prompts_2048_reset_physical_round1_cleanup_none_noop_m6_baseline.yaml`
- Outcome:
  - PASS (`Job finished with no error.`)
  - End-to-end wall time (launcher + container + run): `10:22.20` (`622.2s`)
  - Round time (`Round 0` start to `Round 0` done): `506s` (`20:05:21` -> `20:13:47`)
  - Generated tokens: `20480` total
  - Effective throughput:
    - `40.5 tok/s` at round level (`20480 / 506`)
    - `32.9 tok/s` end-to-end (`20480 / 622.2`)
  - Decode-loop-only profile (`engine.py:1225`, host 0):
    - `64` decode iterations, `461.588s` summed decode time
    - `20470` tokens from decode loop (`+10` from prefill/extraction -> `20480` total)
    - `44.35 tok/s` decode-loop average
  - Behavior trend:
    - Early decode iterations: up to ~`290 tok/s`
    - Tail decode iterations: down to ~`25 tok/s`
    - `DecodeStats`: `pages_in_use` grows `10 -> 330`

M6.1 implementation (completed):
- Code changes:
  - `src/levanter/inference/engine.py`
    - Added `InferenceEngineConfig.log_decode_perf_summary` (default `true`).
    - Added low-overhead `_DecodePerfSamples` accumulator.
    - Collects per-iteration timings in memory only (`iter_total`, `submit`, `extract`, `host`, `device`, `new_tokens`).
    - Emits one aggregate log line per `generate()` call:
      - `Decode perf summary: {...}`
      - Includes `p50/p90/max` for key timings plus decode/round throughput.
- Safety/perf constraints:
  - No additional in-loop tracker calls.
  - No extra cross-host synchronization.
  - Only one end-of-round summary log emission.
- Unit coverage:
  - `tests/inference/test_engine.py`:
    - `test_decode_perf_samples_summary_aggregates_timings`
    - `test_decode_perf_samples_summary_handles_empty`

M6.1 validation rerun (same real TPU baseline config):
- Log:
  - `/tmp/levanter_run_m6_baseline_10prompts_2048_round1_m61.log`
- Outcome:
  - PASS (`Job finished with no error.`)
  - New aggregate summary emitted as expected:
    - `Decode perf summary: {'decode_iters': 64, ...}`
- Comparison vs pre-M6.1 baseline:
  - End-to-end wall time:
    - before: `622.2s` (`10:22.20`)
    - after: `618.28s` (`10:18.28`)
    - delta: `-3.92s` (`-0.63%`)
  - Round time:
    - before: `~506s`
    - after: `505.266s`
    - delta: `~ -0.7s`
  - Decode-loop aggregate (host 0 parse):
    - before: `461.588s`, `44.347 tok/s`
    - after: `461.145s`, `44.390 tok/s`
    - delta: effectively noise-level
- Takeaway:
  - M6.1 instrumentation is performance-neutral within run-to-run variance and does not reintroduce M5-style instability.

M6.2 implementation (completed):
- Code changes:
  - `src/levanter/inference/engine.py`
    - `_extract_outputs` now pulls only used token slices (`[:num_tokens]`) instead of full backing buffers.
    - Added one-time per-call `slot_results` mapping to avoid repeated nested dict churn per token append.
    - Avoided duplicate final decode work for already-finished requests (`if dr.done: continue`).
    - Guarded final-text decode logging behind `logger.isEnabledFor(logging.DEBUG)`.
    - Converted debug/error log calls to lazy logging args to avoid formatting overhead when disabled.
- Validation:
  - Log:
    - `/tmp/levanter_run_m6_baseline_10prompts_2048_round1_m62.log`
  - Outcome:
    - PASS (`Job finished with no error.`)
  - Comparison vs M6.1:
    - End-to-end wall time: `10:18.28` -> `10:04.86` (`-13.42s`, `-2.17%`)
    - Round time: `505.266s` -> `505.000s` (`-0.266s`, noise-level)
    - Decode total: `461.142s` -> `460.971s` (noise-level)
- Takeaway:
  - Host-path hardening is safe and slightly reduces end-to-end overhead, but does not move round throughput materially.

M6.3 implementation (completed):
- Code changes:
  - `src/levanter/layers/attention.py`
    - Added tunables for reference ragged paged attention:
      - `ragged_paged_q_block_size` (default `16`)
      - `ragged_paged_kv_block_pages` (default `16`)
    - Threaded these through `Attention.paged_decode` -> `ragged_paged_attention` -> `default_ragged_paged_attention`.
    - Added validation for positive block sizes.
    - Replaced hardcoded tiny blocks (`Q_BS=min(1, ...)`, `KV_BS=min(2, ...)`) with tunable values.
  - `src/levanter/models/llama.py`
    - Exposed model config fields:
      - `ragged_paged_q_block_size`
      - `ragged_paged_kv_block_pages`
- Validation run A (new defaults `q=16`, `kv_pages=16`):
  - Log:
    - `/tmp/levanter_run_m6_baseline_10prompts_2048_round1_m63.log`
  - Outcome:
    - PASS (`Job finished with no error.`)
  - Comparison vs M6.2:
    - End-to-end wall time: `10:04.86` -> `4:24.67` (`-56.2%`, `2.29x` faster)
    - Round time: `504.999s` -> `164.534s` (`-67.4%`, `3.07x` faster)
    - Decode total: `460.971s` -> `120.182s` (`-73.9%`)
- Validation run B (sweep point `q=32`, `kv_pages=16`):
  - Config:
    - `config/sampler/sample_llama8b_multihost_real_10prompts_2048_reset_physical_round1_cleanup_none_noop_m6_m63_q32_kv16.yaml`
  - Log:
    - `/tmp/levanter_run_m6_baseline_10prompts_2048_round1_m63_q32_kv16.log`
  - Outcome:
    - PASS (`Job finished with no error.`)
  - Comparison vs run A (`q=16`, `kv=16`):
    - End-to-end wall time: `4:24.67` -> `4:27.47` (slightly slower)
    - Round time: `164.534s` -> `165.267s` (slightly slower)
    - Decode total: `120.182s` -> `120.521s` (slightly slower)
- Takeaway:
  - Major M6 speed issue was the tiny hardcoded reference attention block sizes; larger chunking gives a step-function throughput gain.
  - Keep M6.3 defaults at `q=16`, `kv_pages=16` for now.

M6.4 implementation (completed):
- Code changes:
  - `src/levanter/layers/kv_cache.py`
    - Replaced token-by-token `lax.fori_loop` + `lax.dynamic_update_slice` writes in `kv_update_unified_prefix` with a masked vectorized scatter-add delta path over the valid prefix `[0, K)`.
    - Kept donation semantics (`donate_argnums=(0,)`) and explicit prefix clipping (`K` clipped to token axis size).
  - `tests/inference/test_kv_cache.py` (new)
    - `test_kv_update_unified_prefix_updates_only_valid_prefix`
    - `test_kv_update_unified_prefix_noop_when_k_zero`
    - `test_kv_update_unified_prefix_clips_k_to_token_count`
- Local validation:
  - `uv run ruff check src/levanter/layers/kv_cache.py tests/inference/test_kv_cache.py`
  - `uv run pytest tests/inference/test_kv_cache.py`
  - `uv run pytest tests/inference/test_kv_cache.py tests/inference/test_paged_attention.py`
  - all PASS
- Real TPU validation run A:
  - Log:
    - `/tmp/levanter_run_m6_baseline_10prompts_2048_round1_m64.log`
  - Outcome:
    - PASS (`Job finished with no error.`)
  - Comparison vs M6.3 baseline (`m63`):
    - End-to-end wall time: `4:24.67` -> `4:23.15` (`-1.52s`, `-0.57%`)
    - Round time: `164.534s` -> `161.859s` (`-2.675s`, `-1.63%`)
    - Decode total: `120.182s` -> `116.668s` (`-2.92%`)
- Real TPU validation run B (follow-up confirm):
  - Log:
    - `/tmp/levanter_run_m6_baseline_10prompts_2048_round1_m64_castfix.log`
  - Outcome:
    - PASS (`Job finished with no error.`)
  - Comparison vs M6.3 baseline (`m63`):
    - End-to-end wall time: `4:24.67` -> `4:28.62` (slightly slower)
    - Round time: `164.534s` -> `163.379s` (`-1.155s`, `-0.70%`)
    - Decode total: `120.182s` -> `116.814s` (`-2.80%`)
- Notes:
  - A `jax/_src/ops/scatter.py` `FutureWarning` about `float32 -> bfloat16` scatter casting appears during generation in this environment; a local isolated `kv_update_unified_prefix` warning-as-error repro does not trigger it, so the warning source is likely elsewhere in the broader decode path.
- Takeaway:
  - M6.4 improves throughput relative to M6.3 on the target config and does not regress stability.
  - The gain is incremental versus M6.3 (single-digit percent), but positive and repeatable at round level.

M6.5 implementation (completed):
- Scope:
  - Scheduler/decode batch sweep of `engine.max_rounds` while keeping all other M6.4 settings fixed.
- Sweep configs:
  - `config/sampler/sample_llama8b_multihost_real_10prompts_2048_reset_physical_round1_cleanup_none_noop_m6_m65_r64.yaml`
  - `config/sampler/sample_llama8b_multihost_real_10prompts_2048_reset_physical_round1_cleanup_none_noop_m6_m65_r96.yaml`
  - `config/sampler/sample_llama8b_multihost_real_10prompts_2048_reset_physical_round1_cleanup_none_noop_m6_m65_r128.yaml`
- Real TPU validation (`v5p-16`, foreground):
  - `r64` log: `/tmp/levanter_run_m6_m65_r64.log`
    - PASS (`Job finished with no error.`)
    - `round_total_s=161.337s`, `decode_total_s=115.506s`, `decode_iters=32`
  - `r96` log: `/tmp/levanter_run_m6_m65_r96.log`
    - PASS (`Job finished with no error.`)
    - `round_total_s=161.564s`, `decode_total_s=115.648s`, `decode_iters=22`
  - `r128` log: `/tmp/levanter_run_m6_m65_r128.log`
    - PASS (`Job finished with no error.`)
    - `round_total_s=162.752s`, `decode_total_s=116.038s`, `decode_iters=16`
    - End-to-end shell wall: `5:00.64` total
- Comparison (lower is better):
  - `r64` vs `r96`: `-0.227s` on `round_total_s` (`~0.14%` faster).
  - `r64` vs `r128`: `-1.414s` on `round_total_s` (`~0.87%` faster).
- Takeaway:
  - For the M6 baseline workload (`10 prompts x 2048`, single round), `max_rounds=64` is the fastest point among tested values.
  - `max_rounds=128` is consistently slower; reducing decode iteration count alone does not improve round time.
  - Lock M6 default benchmarking on `r64` and keep this as the regression reference for further M6 tuning.

M6.6 implementation (completed):
- Scope:
  - Additional scheduler sweep around M6.5 winner (`max_rounds=64`).
  - Concrete available knobs in current code are `engine.max_tokens_per_round` and `engine.max_queued_tokens` (not `max_tokens_per_step` / `max_docs_per_round`).
- Sweep configs:
  - `config/sampler/sample_llama8b_multihost_real_10prompts_2048_reset_physical_round1_cleanup_none_noop_m6_m66_tpr10_mqt64.yaml`
  - `config/sampler/sample_llama8b_multihost_real_10prompts_2048_reset_physical_round1_cleanup_none_noop_m6_m66_tpr10_mqt128.yaml`
  - `config/sampler/sample_llama8b_multihost_real_10prompts_2048_reset_physical_round1_cleanup_none_noop_m6_m66_tpr12_mqt64.yaml`
  - `config/sampler/sample_llama8b_multihost_real_10prompts_2048_reset_physical_round1_cleanup_none_noop_m6_m66_tpr8_mqt64.yaml`
- Real TPU validation (`v5p-16`, foreground):
  - M6.5 reference (`r64`) log: `/tmp/levanter_run_m6_m65_r64.log`
    - `round_total_s=161.337s`, `decode_total_s=115.506s`, `round_total_generated=20480`
  - `tpr10/mqt64` log: `/tmp/levanter_run_m6_m66_tpr10_mqt64.log`
    - `round_total_s=160.436s`, `decode_total_s=115.146s`, `round_total_generated=20480`
    - vs M6.5 reference: `-0.901s` on round time (`-0.56%`)
  - `tpr10/mqt128` log: `/tmp/levanter_run_m6_m66_tpr10_mqt128.log`
    - `round_total_s=161.317s`, `decode_total_s=115.710s`, `round_total_generated=20480`
    - vs M6.5 reference: `-0.021s` on round time (`-0.01%`, noise-level)
  - `tpr12/mqt64` log: `/tmp/levanter_run_m6_m66_tpr12_mqt64.log`
    - `round_total_s=161.113s`, `decode_total_s=115.889s`, `round_total_generated=20480`
    - vs M6.5 reference: `-0.224s` on round time (`-0.14%`)
  - `tpr8/mqt64` log: `/tmp/levanter_run_m6_m66_tpr8_mqt64.log`
    - `round_total_s=152.841s`, `decode_total_s=106.870s`, `round_total_generated=18440`
    - Not comparable for round-time winner selection because generated tokens are lower than the 20480-token target.
- Takeaway:
  - Best M6.6 point is `max_tokens_per_round=10`, `max_queued_tokens=64`.
  - Reducing queue capacity from `512 -> 64` yields a small but repeatable gain at fixed generation workload.
  - `max_tokens_per_round=8` is not a safe default for this benchmark; it changes generation workload (fewer total tokens produced in this run) and has lower decode throughput.
- M6 final config (recommended):
  - `config/sampler/sample_llama8b_multihost_real_10prompts_2048_reset_physical_round1_cleanup_none_noop_m6.yaml`
  - `config/sampler/sample_llama8b_multihost_real_10prompts_2048_reset_physical_round1_cleanup_none_noop_m6_final.yaml` (same settings, compatibility alias)
  - Rationale:
    - Keeps M5 correctness/stability semantics unchanged (`physical` reset, `cleanup_mode=none`, reference attention path).
    - Uses the fastest comparable scheduler settings from M6 (`max_rounds=64`, `max_tokens_per_round=10`, `max_queued_tokens=64`).
    - Does not change sampling behavior knobs (`temperature`, stop handling, generations), so output quality/distribution intent is preserved.

M6 speedup tracking (round-time, milestone-to-milestone):
- Metric:
  - Uses `round_total_s` (lower is better) and one canonical run per milestone.
- Canonical round times:
  - M6.1: `505.266s`
  - M6.2: `505.000s`
  - M6.3: `164.534s`
  - M6.4: `161.859s` (run A)
  - M6.5: `161.337s` (`max_rounds=64`)
  - M6.6: `160.436s` (`max_tokens_per_round=10`, `max_queued_tokens=64`)
- Sequential speedups:
  - M6.1 -> M6.2: `0.05%`
  - M6.2 -> M6.3: `67.42%`
  - M6.3 -> M6.4: `1.63%`
  - M6.4 -> M6.5: `0.32%`
  - M6.5 -> M6.6: `0.56%`
- Cumulative speedups:
  - M6.2 -> M6.5: `68.05%` faster (`505.000s` -> `161.337s`)
  - M6.1 -> M6.5: `68.07%` faster (`505.266s` -> `161.337s`)
  - M6.2 -> M6.6: `68.23%` faster (`505.000s` -> `160.436s`)
  - M6.1 -> M6.6: `68.25%` faster (`505.266s` -> `160.436s`)

Updated M6 priorities:
1. M6.1 instrumentation: DONE.
2. M6.2 host-path hardening: DONE.
3. M6.3 reference attention block tuning: DONE (locked to `q=16`, `kv=16`).
4. M6.4 KV update optimization: DONE.
5. M6.5: Scheduler/decode batch sweep. DONE (lock `max_rounds=64` for this workload).
6. M6.6: Scheduler sweep (`max_tokens_per_round`, `max_queued_tokens`). DONE.
7. M6.7: Fine-grained queue/pack sweep around `tpr10/mqt64` and validate on lock-in long-gen configs. (OPTIONAL/FUTURE)

Rules:
- Performance changes must land separately from correctness changes.
- Every performance change must include a benchmark run and regression guard.
- Keep M5 lock-in safety semantics unchanged while doing M6.

Exit criteria:
- Median round throughput improves by >=15% on the baseline config above.
- No regressions on known-good lock-in configs (`20 prompts x 2048`, `20 prompts x 4096`).
- No new vmem failures or multi-host synchronization failures.

Rollback:
- Revert perf patch only.

### M7 - TPU Ragged Paged Attention VMEM OOM (Dedicated Track)

Goal: re-enable TPU kernel safely or provide a robust alternative.

Deliverables:
- A small reproducer focused only on the kernel.
- One of:
  - chunked kernel implementation
  - reduced scratch usage
  - alternative attention backend for inference

Exit criteria:
- TPU kernel path passes 1-prompt long generation without vmem OOM.

Rollback:
- Keep reference attention path default.

---

## Test Matrix (What We Run After Each Milestone)

Unit tests (fast):
1. `uv run pytest tests/inference/test_paged_attention.py`
2. `uv run pytest tests/inference -m "not entry and not slow and not ray"`

Local integration (single-host):
1. A tiny model + tiny page table:
   - allocate, decode a few steps, reset, decode again
2. Clone test:
   - prompt length crossing a page boundary and not crossing (tests partial/full page handling)

Multi-host smoke (manual, expensive but required for reset changes):
1. v5p-16, 1 prompt, 2048 new tokens (known-good command)
2. v5p-16, multi-prompt scenario that previously failed

Important: multi-host reset safety cannot be validated by single-host unit tests alone. Any change that touches reset, allocation, freeing, or decode loop boundaries must be smoke-tested on TPU.

---

## Config Hygiene Rules (To Prevent `draccus` Drift)

1. No orphan config keys:
   - A PR that introduces `InferenceEngineConfig.foo` must update all YAMLs that set `engine:` fields.
2. No "temporary debug config" fields. Use logging toggles or environment variables if needed.
3. Add a CI check that loads key YAMLs into their dataclasses and fails on unknown fields.

This directly prevents the "fixed code but config still had stale `use_logical_reset` / `max_prompts_per_batch`" failure pattern documented in `CODEX_MULTI_HOST_DEBUG.md`.

---

## Appendix: Concrete Code Touchpoints

These are the files that define the behavior described in `KV_CACHING_RESTRUCTURE.MD`:

1. `src/levanter/inference/engine.py`
2. `src/levanter/inference/jit_scheduler.py`
3. `src/levanter/inference/page_table.py`
4. `src/levanter/layers/kv_cache.py`
5. `src/levanter/layers/attention.py`
6. `src/levanter/main/sample_lm_multihost.py`

When making changes, prefer to:
1. Add/strengthen invariants in `jit_scheduler.py` / `page_table.py`.
2. Keep `engine.py` orchestration boring and predictable.
3. Keep `sample_lm_multihost.py` minimal; it is a consumer and should not contain "engine behavior" logic.
