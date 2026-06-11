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

**R9 result: breadth-first ops worked — mean rubric 0.07 → 0.23 (3x).** The
breadth-first `ops.md` surfaced the previously-missing APIs (`marin/ops.md` now
lists `compute_minhash_attrs_step(num_perms=286, num_bands=26, ngram_size=5)`;
`levanter/ops.md` lists `levanter.initialize`). Per-probe gains: P1 0→3/5, P3
0→2/5, P7 0→2/4, P9 0→1/4. So the per-sub-project 1000-token budget CAN carry
useful API coverage when prompted for breadth over depth.

Trajectory (mean rubric / mean quality):
- R7 (harness-buggy):   0.09 / 1.3
- R8 (harness fixed):   0.07 / 1.3
- R9 (breadth-first):   0.23 / 1.3

Remaining limiters (next levers):
- **Anchor precision.** P8 surfaced `compute_minhash_attrs_step` (a real StepSpec
  builder) but the anchor is `compute_minhash_attrs`; the judge flagged the
  variant as a hallucination. Anchors should accept `<name>` and `<name>_step`.
- **Quality scale is flat (~1.3 every round).** The judge rates any incomplete
  script low, so PASS (quality ≥ 4) is ~unreachable and doesn't discriminate.
  Either recalibrate the 1-5 rubric with explicit anchors, or make rubric
  accuracy the primary metric and treat quality as secondary.
- **Internal-symbol understand-probes.** P4/P5/P10 anchor on private internals
  (`_reconcile_tick`, Rust store fns, `_invalidate`) that a doc would not name;
  they under-measure doc quality and should be re-scoped to public concepts.

## R10: Judge fairness fixes (re-eval of R9 docs)

Same R9 docs, fixed judge: intent-aware anchor scoring (USE = used in code;
UNDERSTAND = named/referenced/explained), `foo`/`foo_step` builder-variant
equivalence, recalibrated 1-5 quality rubric, PASS = rubric_acc ≥ 0.8 AND no
hallucination (quality demoted to a secondary signal). Result: **0.23 → 0.26**
rubric, quality 1.3 → 1.70. The key finding was qualitative: the judge is
**high-precision, not over-flagging** — every hallucination flag was a real
fabrication the (then haiku, tool-less) coder invented because the doc never
surfaced the right symbol (`ControllerServiceClientSync`→`ControllerClient`;
`DistributedConfig` fields jammed onto `TrainerConfig`). Root cause = doc
content, not judging.

## R11: Architecture docs name internal edit-site seams

A separate architecture-only digest that also surfaces private methods on public
classes + private module-level functions (marked `internal`), plus an
ARCHITECTURE_PROMPT that must name the exact edit-site symbol. **0.26 → 0.39**
rubric, quality 2.10, first PASS (P8). Big understand-probe gains (P7 2/4→4/4,
P10 0/4→3/4). Validity caveat discovered: UNDERSTAND scores are partly inflated
by **prompt-vocabulary leakage** — a coder can echo the prompt's words
("resolver", "invalidation") into comments and the intent-aware judge credits
them; the *non-obvious* anchors (`LogServiceProxy`) are the real discriminators.
This is the last round of the old "haiku, no tools" coder regime.

## Coder regime change (per request, R12+)

The coder became a realistic agent: **sonnet that may open ~3 files** (read-only
Read/Grep/Glob, no execution) under a tight $0.30 cap (no `--max-turns` flag
exists; the dollar cap bounds exploration). This reframes the metric from
"docs in isolation" to "how well docs ROUTE a capable, budget-limited agent."
A new metric, **mean coder spend**, is tracked since rubric accuracy can saturate
when the agent reads source.

## The 2×2 + ablation (fixed sonnet model, R12–R17)

These six rounds vary only doc-method and coder file-access:

| docs | +files coder | docs-only coder |
|------|:---:|:---:|
| mechanical (R11) | R12: **0.66** | R15: **0.43** |
| agentic ($4.71 gen) | R13: **0.64** | R17: **0.26** |
| none | R14: **0.72** | R16: **0.12** (floor) |

Key reads:
- **Docs' value is conditional on file access.** Docs-only: mechanical docs give
  **+0.31** (0.12 → 0.43). File-reading: docs give **~0** (0.66 vs 0.72, within
  noise) — the agent recovers the symbols by reading code.
- **Mechanical beats agentic.** Tied with files (0.66 vs 0.64) but mechanical is
  clearly better docs-only (**0.43 vs 0.26**): the signature-dense mechanical
  docs carry exact import paths + defaults a doc-only coder needs, while the
  agentic docs read as narrative. Agentic generation cost $4.71/round for no win.
- **Floor (R16 = 0.12):** with neither docs nor file access, only P9 (levanter,
  JAX/distributed priors) and P8 scored — Marin's internal APIs aren't in
  parametric knowledge, so docs/source are genuinely necessary.
- **P4 (iris reconcile loop) is the hard case:** 0/4 with any doc, but 2/4 for
  the no-docs file-reading coder (R14) which read `controller.py` directly.

## Conclusions

1. **Ship the cheap mechanical generator.** Agentic doc generation is grounded
   and reads well but never wins: tied with files, *worse* doc-only (0.26 vs
   0.43), at ~$4.71/round vs ~free.
2. **Budgeted 1000-token two-axis docs work**, and the highest-value levers are
   breadth-first `ops.md` (entry-point index) and an `architecture.md` that names
   concrete edit-site symbols including internal seams. Keep docs signature-dense.
3. **Docs primarily serve doc-only / weak / orientation use** (+0.31 there).
   For agents that read source under a budget, docs are roughly neutral on
   correctness; their value is speed/orientation, which this eval does not
   measure. The realistic agent reads code regardless.
4. **Eval caveats** for any follow-up: single-run variance (~±0.05–0.08);
   UNDERSTAND prompts leak some anchor vocabulary; the eval rewards recovering
   source symbols, which structurally favors file-reading agents; Rust non-`pub`
   internals (P5) aren't in the digest (parser only captures `pub`).

---

# v3: hybrid split-docs (agentic narrative + mechanical reference)

A third pass (R18–R27) testing a new doc architecture: each doc gets a **1500-token
split budget** — an agentic NARRATIVE half (sonnet with read-only Read/Grep/Glob
explores source and writes concepts/data-flow/why) followed by a deterministic
MECHANICAL reference half (tree-sitter, no LLM: important files / entry-point
signatures / internal edit seams, per axis). The narrative half is cached per
(project, axis), so iterating the free mechanical half costs $0. **sonnet
generates, opus judges** (v2 judged with sonnet, so the absolute numbers shift —
hence the opus-judged re-baselines below). All on the subscription plan.

## The opus-judged frame (docs-only coder)

| round | docs | rubric | qual | pass | hallucinated probes |
|---|---|:---:|:---:|:---:|:---:|
| R20 | none (floor) | 0.28 | 1.80 | 2/10 | — |
| R19 | agentic | 0.38 | 2.10 | 3/10 | — |
| R18 | mechanical (v2 winner) | 0.53 | 2.60 | 3/10 | 1 |
| R21 | hybrid v1 (raw dump) | 0.39 | 2.30 | 2/10 | — |
| R22 | hybrid v2 (ranked reference) | **0.56** | 2.80 | 4/10 | — |
| R23 | hybrid v3 (ranked + sharpened narrative) | **0.58** | 3.00 | 3/10 | 4 |
| R27 | deterministic reference only ($0, no LLM) | 0.56 | 2.70 | 1/10 | 6 |

Opus judges more generously than the v2 sonnet judge (mechanical 0.53 vs 0.43)
but preserves the ordering: mechanical > agentic > floor.

### What moved
- **R21 → R22 (0.39 → 0.56): selection, not density.** The raw hybrid dump
  *underperformed* mechanical (0.39 < 0.53): a breadth-first/source-order dump
  buried the load-bearing symbols (it surfaced `Controller._run_scheduling_loop`
  but dropped the real reconcile loop `_run_polling_loop`) and handed the
  docs-only coder near-misses it grabbed wrong. Ranking the *deterministic*
  reference — control-flow/lifecycle method names for the architecture edit
  sites, an importance heuristic (submit/run/connect verbs, Config/Client
  suffixes, shallow paths) for the ops index — took it to **0.56**, past the
  mechanical bar. Biggest gains: P8 marin-use 3/7→7/7, P9 levanter-use 1/4→4/4.
- **R22 → R23 (0.56 → 0.58): sharpened narrative.** Making the narrative *name
  the exact* load-bearing symbol (the reconcile loop `_reconcile_tick`, the right
  entry point per task) added a noisy +0.02. The honest read: rubric is within
  noise of R22, and the same change that lifted iris/finelog understand probes
  *regressed* one marin use probe 7/7→2/7 — narrative content is high-variance.

### The $0-doc ablation (R27)
The ranked deterministic reference **alone** (no narrative, no LLM at all) scores
**0.56 rubric** docs-only — matching the full hybrid and beating the v2 mechanical
LLM doc (0.53). But it is **hallucination-prone**: 6/10 probes flagged (vs hybrid
4, mechanical 1), so only **1/10 pass**. A bare signature list gives a docs-only
coder rope to fabricate plausible near-misses; the narrative contextualizes which
symbol to use, and the mechanical LLM doc's curation simply lists fewer.

## The practical (file-reading coder) frame

A coder that may read ~3 files under a tight budget, opus judge:

| round | docs | rubric | pass |
|---|---|:---:|:---:|
| R26 | none (floor) | 0.74 | 5/10 |
| R25 | mechanical | 0.70 | 5/10 |
| R24 | hybrid (best) | **0.82** | 6/10 |

**This reverses v2's "docs are neutral once the agent can read source."** Under
the opus judge, the hybrid clearly helps a file-reading agent (**0.82 vs 0.74
no-docs**), while a plain signature doc does not (mechanical 0.70 ≈ no-docs). The
differentiator is *routing*: the narrative orients and the ranked reference points
at exact files, so the budget-limited coder reads the *right* source. The win
concentrates on the finelog probes (P5 0.0→1.0, P6 0.5→1.0). Caveat unchanged from
v2: on the hardest probe (P4, iris reconcile loop) docs still mislead — mechanical
0.00, hybrid 0.25, but no-docs 0.75 (the coder read `controller.py` directly).

## Conclusions (v3)

1. **Selection is the lever, and it is free.** Ranking a deterministic tree-sitter
   reference (control-flow seams + importance-ranked entry points) beats both the
   v2 mechanical LLM doc (0.56 vs 0.53 docs-only) and the unranked dump (0.56 vs
   0.39) — at $0 and zero hallucination risk *in the generator*. Density without
   ranking is actively harmful.
2. **The narrative earns its keep on orientation and hallucination control, not
   raw rubric.** It adds only ~+0.02 docs-only rubric over the bare reference, but
   it raises the pass rate (fewer fabricated near-misses) and is *decisive* in the
   realistic file-reading regime (0.82 vs 0.70 mechanical, 0.74 no-docs).
3. **The hybrid is the best doc in both regimes** — docs-only 0.58 and
   file-reading 0.82 — and updates v2's "ship mechanical, skip agentic": the
   agentic narrative *does* help when paired with a ranked mechanical reference.
4. **Cost.** Reference half: $0 (deterministic), regenerates instantly. Narrative
   half: ~$0.3/doc agentic, cached → free to re-render; a 12-doc set is ~$2.4–3.9
   to generate once (v2 agentic generation total was $6.2 here). Ship the hybrid:
   the free ranked reference carries the rubric, the cached narrative buys the
   orientation that matters to a realistic file-reading agent.
5. **Harness hardening this pass:** the file-reading coder's spend cap (an
   *intended* bound) was crashing rounds as a hard CLI error — now its partial
   result envelope is used; and a transient judge 529 is retried instead of
   aborting a 50-minute round.

### Caveats
- Single-run variance ~±0.05–0.08; R22↔R23 (0.56↔0.58) and the file-reading
  mechanical↔no-docs (0.70↔0.74) gaps are within it. The robust signals are the
  ranked-vs-dump jump (0.39→0.56), the hybrid file-reading win (0.82 vs ~0.72),
  and the detonly hallucination rate (6/10).
- The narrative is high-variance (a re-roll swung one probe 7/7→2/7); the ranked
  reference is deterministic and stable.
- Heuristic name lists (`_OPS_VERB_HINTS`, `_CONTROL_FLOW_HINTS`) are tuned to
  this monorepo's conventions; they generalize as priors, not guarantees.
