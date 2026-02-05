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

### M5 - Multi-Prompt Admission: Pick One Strategy and Make It Boring

Status: IN PROGRESS (2026-02-05)

Goal: eliminate the "multi-prompt fails with exit status 1" class by reducing complexity in orchestration.

Preferred strategy:
- Build one `requests` list and call `engine.generate(requests)` once per round.
- Avoid batching prompts at the Python level unless we have a proven reason.

If batching is required (memory/time), we must ensure:
- `engine.generate()` boundaries are safe:
  - reset runs
  - slots/pages are truly free
  - no host/device divergence

Deliverables:
- Simplify `sample_lm_multihost.py` to one of:
  - single `engine.generate(...)` per round, or
  - explicit batch loop with mandatory reset between batches and explicit assertions after reset
- Add a "batch boundary assertion" helper that checks:
  - `page_ref_counts` are all zero after reset
  - `used_mask` is all false after reset

Exit criteria:
- v5p-16 multi-prompt config runs to completion with diagnostics that identify which batch is active.

Rollback:
- Revert to known-good single-request mode; keep diagnostics.

Progress so far:
- Multi-prompt (5 prompts, 2048 tokens, reset_mode=physical, cleanup_mode=end) with `n_rounds: 5`
  - Config: `config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical.yaml`
  - Log: `/tmp/levanter_run_m5_5prompts_reset_physical.log`
  - Log evidence (2026-02-05):
    - After reset: `active=0 pages_in_use=0 free=240 max_refcount=0` (00:50:44).
    - After prefill: `active=5 pages_in_use=5 free=235 max_refcount=1` (00:51:22).
    - After decode: `active=5 pages_in_use=165 free=75 max_refcount=1` (00:55:29).
    - After cleanup: `active=0 pages_in_use=0 free=240 max_refcount=0` (00:55:30).
    - `launch.py` reports `Command execution on worker {0,1} failed with exit status 1` immediately after cleanup.
- One-round diagnostic run to isolate multi-round behavior:
  - Config: `config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round1.yaml`
  - Log: `/tmp/levanter_run_m5_5prompts_round1.log`
  - Log evidence (2026-02-05):
    - After reset: `active=0 pages_in_use=0 free=240 max_refcount=0` (01:03:36).
    - After prefill: `active=5 pages_in_use=5 free=235 max_refcount=1` (01:04:14).
    - After decode: `active=5 pages_in_use=165 free=75 max_refcount=1` (01:08:21).
    - After cleanup: `active=0 pages_in_use=0 free=240 max_refcount=0` (01:08:22).
    - `Job finished with no error`.
- Two-round diagnostic run:
  - Config: `config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2.yaml`
  - Log: `/tmp/levanter_run_m5_5prompts_round2.log`
  - Log evidence (2026-02-05):
    - After reset: `active=0 pages_in_use=0 free=240 max_refcount=0` (01:25:35).
    - After prefill: `active=5 pages_in_use=5 free=235 max_refcount=1` (01:26:16).
    - After decode: `active=5 pages_in_use=165 free=75 max_refcount=1` (01:30:23).
    - After cleanup: `active=0 pages_in_use=0 free=240 max_refcount=0` (01:30:24).
    - `launch.py` reports `Command execution on worker {0,1} failed with exit status 1` immediately after cleanup.
- Two-round run with round-boundary diagnostics:
  - Config: `config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2.yaml`
  - Log: `/tmp/levanter_run_m5_5prompts_round2_with_roundstats.log`
  - Result: failed before any `RoundStats[...]` logs. The last visible activity is model shard reads
    (`model-00003-of-00004.safetensors`), followed by `Command execution on worker {0,1} failed with exit status 1`.
- RoundStats instrumentation fix (all-host stats call, leader-only logging):
  - Change: `_log_decode_stats` now calls `jax.device_get(engine.gen_state.decode_state.stats())` on **all hosts**
    and only the leader logs the values.
  - Log: `/tmp/levanter_run_m3_after_only_allhosts_run2.log`
  - Run id: `7rkia52d`
  - W&B run name: `sparkling-bush-218`
  - Result: **success**, `Job finished with no error.` This removes the leader-only stats call crash/hang.
- Two-round run after fixing RoundStats logging:
  - Config: `config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2.yaml`
  - Log: `/tmp/levanter_run_m5_5prompts_round2_roundstats_allhosts.log`
  - Run id: `rc47ssbz`
  - W&B run name: `deep-hill-219`
  - Log evidence (2026-02-05):
    - After reset: `active=0 pages_in_use=0 free=240 max_refcount=0` (06:36:33).
    - After prefill: `active=5 pages_in_use=5 free=235 max_refcount=1` (06:37:11).
    - After decode: `active=5 pages_in_use=165 free=75 max_refcount=1` (06:41:18).
    - After cleanup: `active=0 pages_in_use=0 free=240 max_refcount=0` (06:41:19).
    - **Immediate failure after cleanup**: `Command execution on worker {0,1} failed with exit status 1`.
- Multi-round failure is **not reset-mode dependent** (physical/logical) and happens even with 1 prompt / 256 tokens:
  - 5 prompts, physical reset: `/tmp/levanter_run_m5_5prompts_round2_cleanup_none_noop_zero_mul.log`
  - 5 prompts, logical reset: `/tmp/levanter_run_m5_5prompts_round2_cleanup_none_noop_reset_logical.log`
  - 1 prompt, 256 tokens: `/tmp/levanter_run_m5_1prompt_256_round2.log` (fast repro)
  - All cases: **Round 0 completes, process dies before Round 1** (no Python traceback).
- TPU debug logs (extra env `TPU_STDERR_LOG_LEVEL=0`, `TPU_MIN_LOG_LEVEL=0`, `JAX_ASYNC_ERROR_CHECKING=1`)
  show a TPU runtime failure:
  - Log: `/tmp/levanter_run_m5_1prompt_256_round2_debuglogs.log`
  - Errors include: `program continuator has halted unexpectedly`, slice health check failed, slice failure/core dump.
  - This points to a TPU runtime/slice failure rather than a Python-level bug.

Guardrail: Multi-host logging safety (from `CODEX_REFACTOR_KV_M5_DEBUG.md`)
- Never call `jax.device_get(...)` on a **globally sharded** array from a single host only.
  - Failure modes observed: **immediate exit status 1** or **hang after cleanup**.
  - Safe pattern: run the `device_get` on **all hosts**, and log only on the leader.
  - If you need leader-only stats, use an all-host gather (e.g., `multihost_utils.process_allgather`) or
    compute locally per-host and reduce, but do not single-host `device_get` a sharded array.

Latest update (2026-02-05):
- Batched-rounds workaround is now the **current stable path** for `n_rounds > 1`:
  - `sample_lm_multihost.py` batches all rounds into a single `engine.generate()` when `n_rounds > 1`.
  - `_validate_engine_config` now scales `max_seqs`, `max_seqs_in_prefill`, and `max_prefill_size`
    by `n_rounds` for the batched path.
  - 1 prompt / 256 tokens, `n_rounds=2`: **success**.
    - Config: `config/sampler/sample_llama8b_multihost_real_1prompt_256_round2.yaml`
    - Log: `/tmp/levanter_run_m5_1prompt_256_round2_batched3.log`
  - 5 prompts / 2048 tokens, `n_rounds=2`: **success after increasing page budget**.
    - First attempt failed with `Out of free pages during allocation` at `max_pages=240`.
      - Log: `/tmp/levanter_run_m5_5prompts_2048_round2_batched1.log`
    - Fixed by `max_pages=336` (required pages ≈ 330 for 10 sequences @ `page_size=64`).
      - Log: `/tmp/levanter_run_m5_5prompts_2048_round2_batched2.log`
      - `DecodeStats[after_decode]: pages_in_use=330 free=6`.
- Multi-round **per-round reset** path is still failing (TPU slice/runtime halt). For now, treat
  batched rounds as the supported multi-round workflow; deeper reset fixes remain a follow-up.
- We are **forgoing multi-round per-call resets** for now because they reliably trigger TPU slice/runtime
  failures between rounds on v5p-16 (no Python traceback). The only supported options are:
  - Batched rounds in a single `engine.generate()` call (with correct `max_pages` sizing).
  - Separate jobs per round if batched capacity is too large.
- Clarification (2026-02-05):
  - `n_rounds > 1` **only works** when all rounds are **batched into a single** `engine.generate()` call.
    The per-round reset loop (multiple generate calls in one job) is still unstable on TPU.
  - If you only want **one response per prompt**, use `n_rounds=1`.
  - Very large prompt sets (e.g., 100k prompts) **cannot** fit in a single call; they require batching.
    Today the safe option is **separate jobs per batch** until the reset path is fixed.

Note: We are committing **partial M5 progress** with the batched-rounds workaround + hard `max_pages`
validation; the per-round reset instability remains an open follow-up.

M5 failure modes and strategies (clarified):
1. **Multi-round reset failure (TPU slice halt).**
   - Symptom: `n_rounds > 1` in a single job crashes between rounds with TPU runtime errors
     (no Python traceback; slice health check failure).
   - Strategy A: **Per-round reset** (legacy loop). Currently unstable on TPU.
   - Strategy B: **Batched rounds** (current workaround). Combine all rounds into a single
     `engine.generate()` call and avoid repeated resets inside one job.
   - Strategy C: **External orchestration**. Run one round per job (no multi-rounds inside a job).

2. **Out-of-free-pages failure (KV capacity).**
   - Symptom: `Out of free pages during allocation` during batched rounds.
   - Cause: `engine.max_pages` too small for the total sequences and tokens.
   - Strategy A: **Increase `max_pages`** using the estimate (now enforced as a hard error):
     `required_pages ≈ ceil((max_prompt_len + max_new_tokens) / page_size) * (num_prompts * n_rounds * n_generations)`.
   - Strategy B: **Reduce load** by lowering `n_rounds`, `num_prompts`, or `max_new_tokens`,
     or by splitting into multiple jobs.

Next steps for M5:
- Decide whether we want to keep the batched-rounds mode as the **official** path for `n_rounds > 1`,
  or keep investigating the per-round reset TPU slice failure as a deeper runtime issue.
- Consider adding a helper that recommends a safe **batch size** given `max_pages` + prompt lengths,
  so large prompt sets can be chunked without trial-and-error.

### M5.1 - Very Large Prompt Sets (Design + Tradeoffs)

Status: DESIGN NOTES (2026-02-05)

Problem statement:
We need a safe, repeatable way to handle very large prompt sets (e.g., 10k–100k prompts)
without silent TPU failures or KV cache exhaustion.

Known constraints and failure modes (from `CODEX_REFACTOR_KV_M5_DEBUG.md`):
1. Multi-round/per-batch resets in a **single job** are unstable on v5p-16 and can trigger TPU slice failures.
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
| Multiple generate calls inside one job (prompt chunking) | Amortizes model load; best throughput if reset were stable | **Currently unstable** due to TPU slice failure on reset | Future option once reset path is fixed |
| Separate jobs per batch (external orchestration) | Most reliable today; no in-process reset; easy to parallelize | Repeated model load overhead; higher cost/latency; more complex bookkeeping | **Recommended now** for 10k–100k prompts |
| Persistent service / queue (future) | Best long-run throughput and latency | Requires robust in-process reset and admission control; not implemented | Long-term target, not for M5 |

Current recommendation (M5.1):
Use **external orchestration** to split large prompt sets into batches, each run as a **single job**
with `n_rounds=1` (or batched rounds if repeated sampling is needed). This avoids the TPU reset failure
until we fix multi-call stability. Batch size should be chosen using the `required_pages` estimate and
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
   - Keep `n_rounds=1` (or `n_rounds>1` batched) to avoid multi-call reset instability.
   - Keep `tracker.type=wandb` and set tags that include batch index and size.

5. Launch one job per batch (external orchestration).
   - Each job is a **fresh process**: new `DecodeState`, `PageTable`, `SequenceTable`, `TokenQueue`
     are created during `InferenceEngine` initialization.
   - This avoids the in-process reset path that triggers TPU slice failures.
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

### M5.2 - Per-Round Reset Failure (Root-Cause Sketch + Visibility Plan)

Status: INVESTIGATION PLAN (2026-02-05)

Problem statement:
Multi-round **per-call** reset (multiple `engine.generate()` calls inside one job) fails on TPU
with a runtime slice failure between rounds. We need to understand **why reset fails** and
add visibility so the failure is observable before the TPU halts.

Observed failure signature:
- `launch.py` reports: `Command execution on worker {0,1} failed with exit status 1`.
- With TPU debug logs (see `CODEX_REFACTOR_KV_M5_DEBUG.md`):
  - `FAILED_PRECONDITION: The program continuator has halted unexpectedly.`
  - `INTERNAL: *** Halt is unexpected, all pending programs will fail.`
  - `Slice health check failed ... Worker connection ... failed.`
  - `Detected slice failure, core dump will begin.`
- No Python traceback; indicates TPU runtime instability or invalid program state between rounds.

Configs that reliably trigger the failure (per-round reset path):
- `config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop.yaml`
- `config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_logical_round2_cleanup_none_noop.yaml`
- `config/sampler/sample_llama8b_multihost_real_1prompt_256_round2.yaml` (fast repro)
All fail after Round 0 completes and before Round 1 starts.

What does **not** fix it:
- Physical vs logical reset mode.
- Engine recreation between rounds.
- Per-round request rebuild (new keys, new `SeqDecodingParams`).
- Skipping round barrier (`skip_round_barrier: true`).
- Skipping samples table (`skip_samples_table: true`).
- KV cache zeroing change (multiply-by-zero to encourage reuse).

What is **not** the cause:
- Leader-only stats read: fixed separately (leader-only `device_get` of sharded arrays was a **different** failure).
- Model load and engine creation: both succeed and logs show the first round running normally.

Hypotheses (ordered by likelihood):
1. **Reset path triggers invalid device state** after a full decode run.
   - `InferenceEngine.reset()` → `GenState.reset_*()` → `PageCache.zero()`.
   - Potentially invalid donation / aliasing across JAX buffers on TPU in multi-host.
2. **Dangling in-flight work / program state** at reset boundary.
   - TPU may still have pending programs at round boundary; reset might clobber buffers.
3. **Cross-host synchronization mismatch** around reset.
   - If one host reaches reset earlier, a sharded operation could leave TPU in inconsistent state.

Visibility plan (increase signal before the TPU halts):
1. Add **explicit reset boundary logging** on all hosts:
   - `Generate reset: start` / `Generate reset: done` already exists.
   - Add log markers inside `GenState.reset_*` (pre/post) on all hosts (not just leader).
2. Add **JAX async error checking** in a dedicated debug config:
   - `JAX_ASYNC_ERROR_CHECKING=1`
   - `TPU_STDERR_LOG_LEVEL=0`, `TPU_MIN_LOG_LEVEL=0`
3. Add **post-round invariant checks** before reset:
   - `DecodeState.stats()` (all-host).
   - Validate `used_mask` and `page_ref_counts` are zeroed after cleanup.
4. Add **explicit device sync** before reset:
   - `jax.block_until_ready` on relevant state (tokens/logprobs or decode state).
   - This may surface the failure earlier with a clearer error.
5. Minimal reproducer:
   - Use `sample_llama8b_multihost_real_1prompt_256_round2.yaml`.
   - Reduce logs to just reset markers + async error checking to keep signal clear.
   - Command (with tee logging):
     `python lib/levanter/infra/launch.py --foreground --zone us-central1-a --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand -- -e TPU_STDERR_LOG_LEVEL 0 -e TPU_MIN_LOG_LEVEL 0 -e JAX_ASYNC_ERROR_CHECKING 1 -- uv run src/levanter/main/sample_lm_multihost.py --config_path config/sampler/sample_llama8b_multihost_real_1prompt_256_round2.yaml 2>&1 | tee /tmp/levanter_run_m5_1prompt_256_round2_debuglogs.log`

Plan to isolate the root cause:
1. Verify the failure is **deterministic** with the fast repro config.
2. Add reset boundary markers inside `GenState.reset_*` to locate exact crash point.
3. Try `jax.block_until_ready` **before** `reset_*` to see if the failure moves earlier.
4. If failure persists, bisect: disable `PageCache.zero()` and only reset metadata
   (if safe) to see whether physical KV zeroing is the trigger.
5. If physical zeroing is the trigger, attempt a **donation-safe** reset path
   (new buffers vs in-place) to avoid TPU aliasing.

Baseline per-round failure command (non-debug, tee):
`python lib/levanter/infra/launch.py --foreground --zone us-central1-a --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand -- uv run src/levanter/main/sample_lm_multihost.py --config_path config/sampler/sample_llama8b_multihost_real_1prompt_256_round2.yaml 2>&1 | tee /tmp/levanter_run_m5_1prompt_256_round2.log`

Exit criteria for M5.2:
- We can run `n_rounds=2` with **per-round reset** without TPU slice failure.
- Or, we can provide a clear minimal reproducer and a narrowed root cause
  that can be handed off to TPU runtime / JAX maintainers.

### M6 - Performance Work: Reference Attention and KV Update

Goal: improve throughput without introducing correctness risk.

Work items (in priority order):
1. Reference attention block size tuning (`default_ragged_paged_attention`).
2. Reduce KV update overhead (vectorize writes if possible, or improve `PageBatchInfo` layout).

Rules:
- Performance changes must land separately from correctness changes.
- Every performance change must include a benchmark script and a regression guard (even if approximate).

Exit criteria:
- Tok/s improves or stays flat across the known-good configs.
- No new vmem failures on TPU.

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
