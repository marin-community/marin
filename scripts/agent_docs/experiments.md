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
