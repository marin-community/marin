# Single-phase Observatory, round 4: literature mechanisms (2026-09-03)

Fieldbook `exp_01m1ge7ye6hz2epd0mjkbkrvt8`. Local compute only (no cluster). Review of the six papers:
`.agents/handoffs/single_phase_related_work_review_20260903.md`. Same protocol as rounds 1-3: Screen against the
successor, then the bank (frozen-style heldout stage, selection scoring with paired bootstrap, leave-one-source-out
regimes), split-half by source wherever a choice is made, Codex and DeepSeek reviews before naming anything.

## Entries (all nested on `weibull_softplus_unscaled`; registry `ROUND4_ENTRIES`)

| Entry | Mechanism | Source |
|---|---|---|
| `@share_penalty` | nonnegative linear share penalty per bucket | Sedova gamma h |
| `@onset_inventory` | harm onset threshold + slope x centred log inventory (4 slopes on the grid) | Finetuner's Fallacy, Repetition Mismatch |
| `@onset_quality` | harm onset threshold + slope x centred quality rank | Scaling Domain Data Repetition |
| `@harm_hierarchical` | shared harm column + ridge-shrunk signed per-bucket deviations (3 shrinks) | InfoLaw shared lambda |
| `@interaction_total_hub`, `@interaction_cc_hub` | signed products of the total or CC benefit signal with each bucket | Scheffe hub pairs |
| `@unique_benefit` | Weibull benefit in unique tokens, harm in epochs | Repetition Mismatch, Finetuner's Fallacy |
| `pooled_effective_data` | concave pooled power law in effective data + share penalties (new model class) | Sedova |

Plus the cap-policy analysis (`single_phase_round4_cap_policies_20260903.py`): per-type epoch caps against the bank.

## Steps

1. Code, tests, pin refresh (57 sampled shards reproduce bit-for-bit). Done.
2. Screen tier for the seven nested entries; heldout stage and selection scoring on the canonical registry. Running.
3. Pooled law: Screen, heldout, scoring (separate chain after step 2 to avoid CPU contention).
4. Leave-one-source-out and split-half for anything that beats the successor on the bank.
5. Reviews, report, deck, Fieldbook checkpoint, commit.
