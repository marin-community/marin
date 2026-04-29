# LM judge rubric edits

Per-judge subdirectories holding 22 cross-tier rubric files each, plus
empty `proposed_spec_edits` arrays where per-judge spec edit suggestions
can be added.

## Subdirectories

- `flash/` — `gemini-3-flash-preview`, `thinking_budget=0`
- `gpt51/` — `gpt-5.1`, `reasoning_effort=none`
- `pro/`   — `gemini-3-pro-preview`, `thinking_budget=128` (API minimum;
  zero is not allowed for Pro)
- `glm51/` — `zai-org/GLM-5.1` via Together API, no reasoning toggle

## File schema (per rubric)

```json
{
  "pair_id": "<dominant>__<subordinate>",
  "tension_point_idx": 0,
  "tension_name": "<short label from atlas>",
  "dominant_id": "<spec statement id>",
  "subordinate_id": "<spec statement id>",
  "topic_targeted": false,
  "writer_model": "...",
  "writer_settings": "...",
  "rubric": {
    "rationale": {
      "spec_clauses_anchored_on": [...],
      "interpretive_choices_made": "...",
      "alternative_readings_rejected": "..."
    },
    "dominant_rubric":   {"GOOD": "...", "BAD": "...", "KEY_TENSION": "..."},
    "non_leakage_rubric": {"GOOD": "...", "BAD": "...", "KEY_TENSION": "..."},
    "worked_example": {
      "spec_compliant": "...",
      "subordinate_over_satisfaction": "...",
      "pathological_refusal": "..."
    }
  },
  "proposed_spec_edits": []
}
```

`topic_targeted` flags the 6 pairs that were pre-identified as
likely-affected by the original Codex topic-specific REQUIREMENTS
(dogwhistles tp=3, self-harm tp=0/3, political×{be_kind,do_not_lie} tp=0,
formatting tp=4).

## Per-judge pathologies (from chat session 2026-04-26)

See `.agents/logbooks/executable_specs_claude.md` →
"Substantive per-pair review and pathology findings" section for the
full per-model pathology analysis.

Headlines:
- **flash**: "I am programmed" boilerplate worked examples on safety
  cases; misses laundering reading on political manipulation. Highest
  verbatim audit (99%).
- **gpt51**: best content, warmest worked examples, but **88% verbatim
  audit** — paraphrases the spec rather than quoting verbatim.
- **pro**: pathologically terse worked examples; dominant.GOOD vs
  non_leakage.BAD internal contradiction on dogwhistles; can't satisfy
  the project's no-reasoning rule. Probably not a viable writer.
- **glm51**: warmest user-care voice, biggest interpretive_choices_made;
  leans toward strict-refusal on dogwhistles (same internal contradiction
  as pro, milder); slow per-call (35-55s via Together).
