# Autonomous shift plan — 2026-04-30 night

**Authorized by Ahmed at ~2026-04-30 08:30 UTC** before sleeping. OpenAI cap $200; Together + Gemini treated as free. Hard abort at $80 cumulative spend. Full plan including stretch goals.

## North star: by morning, deliver

1. Per-pair rubrics for all 65 target pairs (currently judges work without explicit rubrics, just `predicted_relation` guidance)
2. Phase 4 re-judged with grounded rubrics; before/after κ comparison
3. Within-judge reproducibility floor (3 reps × 3 judges × 30 scenarios)
4. RepairProposal records for the 29 problem cases — review-ready, not applied
5. Edit-impact simulation per proposal (post-edit rubric hypothesized + same response re-judged under it; no spec fork)
6. Calibration probe v2 — chosen+rejected per pair × ensemble judges
7. Diagnostics: cross-judge κ matrix, anchoring, verbatim audit, control re-sample, top-k=10 H2 retro
8. Stretch goals: per-pair rubric stability check, generator behavioral noise floor, Phase 1B K=10 retry
9. `MORNING_HANDOFF.md` pinned with "Decisions for Ahmed" + cumulative spend + pointers

## Hard rules

- Every action → logbook entry, Codex template (Q/I/M/O/R/Interp/Next).
- Before any expensive call (>$1): log `ABOUT TO SPEND ~$X on Y; running total $Z`.
- After every artifact: log path + row count + parse status.
- After every failure: error verbatim + recovery.
- Hard abort at $80 cumulative spend. Log the abort.
- **DO NOT** apply edits to spec/rubric files (Codex Gate H5). DO NOT fork the spec. DO NOT auto-apply even `add_example` edits — those need Ahmed's morning sign-off.
- DO NOT run DPO, generate preference shards, or do anything that touches Codex Gate H7.
- DO NOT push to git, open PRs, comment on issues. No external writes.
- DO NOT use weak-oracle generators or re-add atlas seeds — Ahmed closed both threads.

## Shift schedule

| shift | what | est cost | wall |
|---|---|--:|---|
| 1 | Build pair-rubric writer; generate rubrics for 65 target pairs | $3 | 10 min |
| 2 | Re-judge all 578 oracle responses with grounded rubrics | $10 | 15 min |
| 3 | Within-judge reproducibility (30 scenarios × 3 judges × 3 reps) | $0.50 | 10 min |
| 4 | propose_spec_repairs.py + 29 RepairProposals | $0.50 | 10 min |
| 5 | Edit-impact simulation per proposal | $5 | 15 min |
| 6 | Calibration probe v2 | $5 | 15 min |
| 7 | Diagnostics: κ matrix + anchoring + verbatim + controls + top-k=10 retro | $2 | 20 min |
| 8 | Stretch: rubric stability, behavioral noise floor, K=10 retry | $5 | 30 min |
| 9 | MORNING_HANDOFF.md, lint, logbook closeout | $0 | 15 min |
| **total** | | **~$31** | ~140 min |

## Memories that bind this shift

- `project_lm_compiler_is_gpt51.md` — compiler is GPT-5.1 reasoning_effort=none
- `project_judge_ensemble_required.md` — 3-judge ensemble required, single-judge fallback NOT allowed
- `project_h1_hierarchy_disagreements_known.md` — 4 known multi-model role disagreements; do not re-litigate
- `feedback_logbook_discipline_autonomous.md` — the logbook protocol for this shift
- `feedback_no_reasoning_for_alignment_project.md` — never use reasoning, lowest tier on every call
- `feedback_source_env_first.md` — `source .env` before any API call
- `feedback_never_save_generations_in_tmp.md` — outputs go to project dirs, not `/tmp/`
