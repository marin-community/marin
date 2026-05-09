# DART Run 4 postmortem — Opus subagent reports (2026-05-09)

Three Opus subagents (`claude-opus-4-7`, 1M context) were spawned in parallel to do forensic analysis on the three worst Run-4 outcomes. Each was told to **ultrathink** and was given access to the full per-cell judgment data, rubric/spec diffs, and compiler reasoning.

## Subagent reports

- [`prevent_imminent_harm_opus.md`](./prevent_imminent_harm_opus.md) — diagnosis of the Δ−0.397 "regression" (largest in run). **Finding**: ~60% of the headline drop was a measurement-universe artifact (C0 baseline = 20 grok cells × 2 judges; new conditions = 80 cells × 3 judges × 4 generators). Apples-to-apples spec_v2 alone was essentially flat (+0.815 → +0.803). Real damage from rubric anchor 4's MUST-rules brittle-ifying scoring. Recommend revert rubric, keep spec.

- [`comply_with_laws_opus.md`](./comply_with_laws_opus.md) — diagnosis of the still-negative α despite Δ+0.455 improvement. **Finding**: Gemini-as-judge scored 5 in 79/80 cells under v3, completely degenerate. Per-judge means: Gem 4.99, GPT 3.62, Cla 3.51. The 3 R2 compilers all proposed rubric edits but none could see Gemini-judge's bias because Gemini-compiler is the same model. Recommend drop Gemini for this statement, escalate one specific definitional question to spec authors.

- [`no_topic_off_limits_opus.md`](./no_topic_off_limits_opus.md) — diagnosis of the R1→R2 regression (improved then regressed). **Finding**: ONE Claude-compiler edit (Tiananmen/erotica exemplar in v3 anchor 2) caused the regression. GPT cited it 7×, Gemini 1×, Claude 0× = anchor-text uptake asymmetry. R2 compilers doubled down rather than declaring irreducible despite the prompt offering it. Recommend revert anchor 2, keep 3/4/5, mark as converged at α≈0.25.

## Cross-cutting findings

The three reports independently surfaced the same set of failure modes:

1. **Compiler ensembles cannot diagnose their own judges' biases** (Gemini-compiler missed Gemini-judge; all 3 R2 compilers missed GPT's literal-exemplar uptake).
2. **Vivid named exemplars and MUST-rules in anchor `criterion` text are uptake-asymmetric across judges** — one weights heavily, others ignore, result is 3-point score gaps where pre-edit had 1-point.
3. **Cumulative-history compilers double down rather than self-correct** — none used the "declare irreducible" branch despite it being available.
4. **"Spec ambiguity" and "response-interpretation disagreement" look identical from outside, but only the first is fixable by spec/rubric edits** — the latter needs judge calibration (DART Step 6, doesn't exist yet).
5. **C0 baseline measurement was confounded by cell-universe and judge-cohort expansion** — affects multiple statements, not just `prevent_imminent_harm`.

These findings drove the §1.8 detector additions, §2 gotchas 11–16, and §3 follow-up experiments E–K in `.agents/logbooks/dart.md`.

## Provenance

- **Date**: 2026-05-09 ~10:30 UTC
- **Model**: claude-opus-4-7 (1M context)
- **Spawn mechanism**: parallel `Agent` tool calls with `subagent_type: general-purpose`, `model: opus`, `run_in_background: true`
- **Directive**: "ultrathink" with specific hypothesis sets per statement
- **Linked from**: `.agents/logbooks/dart.md` §5 Run 4 — Postmortem
