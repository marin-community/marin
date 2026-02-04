# KV Cache Refactor Plan (Codex) - Multi-Host Safe, Test-Gated, No Silent Failures

This is a gradual refactor plan for Levanter's inference-time KV caching and related orchestration. It is intentionally *more conservative* than `CLAUDE_REFACTOR_KV.md`: we will only ship steps that are (1) locally testable, (2) multi-host safe by construction, and (3) do not rely on "it seems to work" evidence.

The immediate trigger for this doc is the `simpo` multi-host inference break that was "fixed" by restoring the known-good stack from `multihost_inference_work` and removing stale YAML keys (see `CODEX_MULTI_HOST_DEBUG.md`). The root cause was a correctness regression: **`InferenceEngine.reset()` was skipped on multihost**, creating host/device state divergence and leading to TPU/XLA-level crashes with no Python traceback. This plan is designed so that kind of regression is hard to introduce.

References:
- KV cache mental model and current dataflow: `KV_CACHING_RESTRUCTURE.MD`
- Prior plan (contains good ideas but also introduced subtle, high-risk changes): `CLAUDE_REFACTOR_KV.md`
- Repro + fix log for the break: `CODEX_MULTI_HOST_DEBUG.md`

---

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

### M3 - Remove Redundant KV Page State (Single Source of Truth)

Goal: remove duplicated state that can diverge silently.

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

Goal: wire up page freeing in a way that cannot silently corrupt state.

We will not start by freeing pages inside the main decode kernel. Instead, we take two safer steps.

Step 1 (safe boundary): free pages at end-of-generation only.
- After generation finishes, compute which slots are finished and free them, then reset for the next call.
- This validates refcount math and clone behavior without affecting the decode loop.

Step 2 (optional, higher risk): incremental cleanup during decode.
- Only after Step 1 is solid, we can consider freeing pages mid-loop.
- If we do this, we must also:
  - purge queued tokens for finished slots
  - preserve monotonic `finished` reporting in `_DecodeOutputs` (do not regress output assembly)
  - ensure freeing does not cause allocator non-determinism across hosts

Deliverables:
- A `free_finished()` operation with a clear contract:
  - input: finished mask
  - output: updated `DecodeState` + updated `PageTable`
  - invariants asserted (no negative refcounts, no freed page referenced)
- Tests for clone + free interactions.
- Optional: `engine.cleanup_mode` config, but only if we have multiple supported policies.

Exit criteria:
- Refcount invariants pass under unit tests, including clone scenarios.
- Multi-host 1-prompt sampling passes.
- Multi-prompt sequential workloads do not regress.

Rollback:
- Disable incremental cleanup; keep end-of-generation cleanup only.

### M5 - Multi-Prompt Admission: Pick One Strategy and Make It Boring

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
