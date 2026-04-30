# Rubric stability across two runs

Compared 65 pairs. Run 1 = `target_pair_rubrics.jsonl`, run 2 = `target_pair_rubrics_run2.jsonl`. Both use `gpt-5.1` reasoning_effort=none, temperature=0.2.

## Mean Jaccard across runs

| field | mean Jaccard | meaning |
|---|--:|---|
| good_criterion (token) | 0.336 | how similar the GOOD criterion phrasing is across runs |
| bad_criterion (token) | 0.312 | how similar the BAD criterion phrasing is across runs |
| key_tension (token) | 0.356 | how similar the explanatory paragraph is |
| spec_clauses_anchored_on (verbatim set) | 0.843 | fraction of anchored verbatim clauses shared between runs |

## Top 10 pairs with LOWEST verbatim-clause overlap (least stable)

| pair | verbatim Jaccard | good Jaccard | bad Jaccard | n clauses (run1, run2) |
|---|--:|--:|--:|---|
| `avoid_overstepping` × `letter_and_spirit` | 0.25 | 0.19 | 0.32 | (5, 5) |
| `do_not_facilitate_illicit_behavior` × `ignore_untrusted_data` | 0.33 | 0.38 | 0.41 | (4, 4) |
| `avoid_abuse` × `avoid_extremist_content` | 0.50 | 0.32 | 0.40 | (4, 5) |
| `do_not_lie` × `follow_all_applicable_instructions` | 0.50 | 0.28 | 0.25 | (6, 6) |
| `follow_all_applicable_instructions` × `protect_privileged_messages` | 0.50 | 0.34 | 0.29 | (6, 6) |
| `letter_and_spirit` × `protect_privileged_messages` | 0.50 | 0.26 | 0.29 | (6, 6) |
| `present_perspectives` × `refusal_style` | 0.50 | 0.24 | 0.26 | (5, 4) |
| `assume_best_intentions` × `respect_creators` | 0.57 | 0.28 | 0.26 | (5, 6) |
| `ask_clarifying_questions` × `avoid_overstepping` | 0.60 | 0.21 | 0.23 | (4, 4) |
| `avoid_abuse` × `respect_creators` | 0.60 | 0.25 | 0.23 | (4, 4) |

