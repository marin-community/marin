# Autodoc Experiments Summary

Six rounds of experiments to find the optimal documentation format for AI coding agents.
Task: haiku writes a fuzzy dedup script from generated docs; scored on 10 criteria (import path, function name, 4 defaults, max_parallelism, keyword args, runnable, no hallucination).

## R1: Baseline (sonnet coder, sonnet reviewer)

Tested 7 context sizes (2KB-53KB) with hand-crafted docs. All scored 100% except compressed formats (62-75%). Proved sonnet is too capable to stress-test docs — switched to haiku.

## R2: Haiku coder (2 variations)

Full signatures scored 90%; 1-sentence descriptions scored 50%. Haiku needs explicit signatures, not summaries.

## R3: Cross-domain examples (5 variations, haiku coder)

V3 (conceptual doc + exact dedup example) scored 100%. But using same-domain examples was "cheating" — docs can't anticipate the user's task.

## R4: No dedup examples (5 variations, haiku coder)

V1-V4 all scored 100% with conceptual + calling convention format. **Key finding: examples are unnecessary.** The conceptual explanation + full signatures is sufficient.

## R5: Module-level docs from actual pipeline (5 variations, haiku coder)

First test of the real doc generator. Best: V3 (heavy prose) at 80%. V4 (code blocks) got imports right but missed defaults. Neither format alone hit 100%. **Key finding: module-level docs are too broad** — `marin.processing` (40+ functions, 15-30KB docs) overwhelms haiku.

## R6: Package-level docs (production system)

Combined V3 prose (explains WHY defaults matter) with V4 code blocks (exact import paths). Sub-package granularity (`marin.processing.classification.deduplication` instead of `marin.processing`). Scored **10/10 on 4/4 runs**. Review rubric embeds ground truth to prevent hallucinated API judgments.

## Key Decisions

| Decision | Evidence |
|----------|----------|
| Haiku for testing, sonnet for generation | R1: sonnet too capable to reveal doc quality issues |
| No examples tier | R4: conceptual + signatures sufficient without examples |
| Sub-package granularity | R5: module-level too broad for weaker models |
| Combined prose + code blocks | R5: prose fixes defaults, code blocks fix imports |
| Haiku for file summaries, sonnet for aggregation | File summaries are structured extraction; aggregation needs judgment |
| Ground truth in review rubric | Sonnet reviewer hallucinated API judgments without it |

---

# Experiment v2: budgeted two-axis taxonomy, 10-probe suite

Replaces the single fuzzy-dedup probe with 10 probes mined from real `~/.claude`
usage (4 iris, 3 finelog, 2 marin, 1 levanter; use vs understand). Docs are a
budgeted two-axis taxonomy: per sub-project `overview`/`ops`/`architecture` +
top-level `MAP`, each ≤1000 tokens, generated from a signatures-only tree-sitter
digest. Coder = haiku with NO tools (docs only); judge = sonnet, scores anchors
(0/1) + holistic quality (1-5) + high-precision hallucination.

## R7: First end-to-end run — harness bugs dominate (mean rubric 0.09, q 1.3)

Two harness bugs, not doc signal:
- **Digest omitted dataclass fields.** Dataclasses have no explicit `__init__`,
  so the digest rendered them as `Foo()`; the generator then invented fields
  (`ResourceSpec(cpu_millicores=...)` vs real `cpu/memory/device`). Fixed: digest
  extracts class-body `name: type = default` fields.
- **Judge treated anchors as an allowlist.** Any API not among the ~5 anchors
  (even the real `IrisClient.remote`) was flagged as a hallucination → quality
  pinned at 1, nothing passes. Fixed: hallucination = forbidden-list terms or
  direct contradictions of an anchor only.
- Also discovered the merge to main had refactored several probed APIs; all 48
  anchors re-grounded against current source (notably P8: dedup is now a two-stage
  `compute_minhash_attrs` → `compute_fuzzy_dups_attrs` pipeline).

## R8: After harness fixes — real doc-capacity ceiling (mean rubric 0.07, q 1.3)

Harness now validated (P9's coder honestly refused: "the documentation does not
contain the distributed configuration class" — exactly the right behavior; the
judge correctly caught P7 fabricating `ExecutorStep.__init__`). Scores stay near
zero because the docs genuinely lack the needed APIs:
- **Use probes:** a 1000-token whole-sub-project `ops.md` covers only the 2-3 most
  prominent entry points. `marin/ops.md` documents `datakit`, omits fuzzy dedup
  (P8); `levanter/ops.md` covers training-config basics, omits `DistributedConfig`
  (P9). The budget can't hold a 48-package sub-project's API surface.
- **Understand probes:** several anchors are internal symbols (`_reconcile_tick`,
  `seg_filename`, `LogClient._invalidate`) that a doc legitimately would not name,
  so those probes under-measure doc quality and need re-scoping to public concepts.

Echoes the v1 R5 finding (module-level too broad) and R6 win (sub-package
granularity). Suggests the per-sub-project budget is orientation-grade, not
task-completion-grade, for narrow deep-API tasks.

## R9: Breadth-first ops doc (in progress)

Lever test within the 1000-token budget: replace `ops.md`'s "1-3 entry points +
happy path" with an "entry-point index" of 12-20 of the most important callables
across the whole sub-project. Tests whether breadth (vs depth) recovers the use
probes under the fixed budget.
