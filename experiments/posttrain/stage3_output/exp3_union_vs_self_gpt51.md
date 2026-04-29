# Rubric set comparison: with_self_edits vs with_union_edits

**Baseline**: `experiments/posttrain/stage3_output/cross_tier_rubrics_v2_gpt51_with_self_edits.jsonl` (22 rows)
**Treatment**: `experiments/posttrain/stage3_output/cross_tier_rubrics_v2_gpt51_with_union_edits.jsonl` (22 rows)
**Common pairs**: 22

## Per-field aggregate change

Mean text change (1 - SequenceMatcher.ratio) across all common pairs. Higher = more changed. Threshold for 'significant change' = 0.3.

| field | mean Δ | median Δ | max Δ | n significant (>0.3) |
|---|---:|---:|---:|---:|
| dominant_rubric.GOOD | 0.880 | 0.888 | 0.993 | 22/22 |
| dominant_rubric.BAD | 0.918 | 0.916 | 0.979 | 22/22 |
| dominant_rubric.KEY_TENSION | 0.944 | 0.969 | 0.980 | 22/22 |
| non_leakage_rubric.GOOD | 0.910 | 0.925 | 0.969 | 22/22 |
| non_leakage_rubric.BAD | 0.908 | 0.923 | 0.993 | 22/22 |
| non_leakage_rubric.KEY_TENSION | 0.916 | 0.926 | 0.988 | 22/22 |
| worked_example.spec_compliant | 0.792 | 0.848 | 0.988 | 22/22 |
| worked_example.subordinate_over_satisfaction | 0.829 | 0.884 | 0.964 | 22/22 |
| worked_example.pathological_refusal | 0.618 | 0.619 | 0.914 | 22/22 |
| rationale.interpretive_choices_made | 0.885 | 0.925 | 0.971 | 22/22 |
| rationale.alternative_readings_rejected | 0.921 | 0.929 | 0.992 | 22/22 |

## Rationale clause changes

- Avg `spec_clauses_anchored_on` count in with_self_edits: 5.3
- Avg `spec_clauses_anchored_on` count in with_union_edits: 5.4
- Avg new clauses (in treatment, not baseline): 1.7
- Avg removed clauses (in baseline, not treatment): 1.6

## Per-pair changes

| pair | dom.BAD Δ | non_leak.BAD Δ | spec_compliant Δ | alt_read Δ | clauses Δ |
|---|---:|---:|---:|---:|---:|
| `assume_objective_pov__do_not_encourage_self_harm tp=0` | 0.94 | 0.83 | 0.95 | 0.85 | +0 |
| `assume_objective_pov__do_not_encourage_self_harm tp=3` | 0.89 | 0.99 | 0.67 | 0.98 | -1 |
| `assume_objective_pov__do_not_facilitate_illicit_behavior tp=5` | 0.91 | 0.86 | 0.99 | 0.96 | +0 |
| `avoid_abuse__letter_and_spirit tp=2` | 0.89 | 0.97 | 0.83 | 0.91 | +0 |
| `avoid_being_condescending__letter_and_spirit tp=5` | 0.92 | 0.72 | 0.93 | 0.94 | -1 |
| `avoid_errors__avoid_info_hazards tp=2` | 0.92 | 0.91 | 0.87 | 0.93 | +0 |
| `avoid_errors__letter_and_spirit tp=4` | 0.94 | 0.90 | 0.33 | 0.91 | +0 |
| `avoid_hateful_content__be_rationally_optimistic tp=5` | 0.90 | 0.87 | 0.90 | 0.88 | +0 |
| `avoid_hateful_content__present_perspectives tp=3` | 0.95 | 0.99 | 0.97 | 0.93 | +1 |
| `avoid_targeted_political_manipulation__be_kind tp=0` | 0.95 | 0.97 | 0.79 | 0.95 | +0 |
| `avoid_targeted_political_manipulation__do_not_lie tp=0` | 0.86 | 0.97 | 0.92 | 0.99 | +1 |
| `be_kind__uphold_fairness tp=1` | 0.98 | 0.97 | 0.96 | 0.95 | -1 |
| `comply_with_laws__no_agenda tp=5` | 0.84 | 0.96 | 0.71 | 0.91 | +2 |
| `comply_with_laws__no_agenda tp=6` | 0.92 | 0.98 | 0.85 | 0.92 | +1 |
| `do_not_facilitate_illicit_behavior__no_agenda tp=0` | 0.93 | 0.92 | 0.39 | 0.96 | +0 |
| `express_uncertainty__prevent_imminent_harm tp=5` | 0.94 | 0.92 | 0.75 | 0.95 | +0 |
| `formatting__letter_and_spirit tp=4` | 0.93 | 0.96 | 0.67 | 0.86 | +0 |
| `no_agenda__respect_creators tp=3` | 0.90 | 0.92 | 0.65 | 0.91 | +0 |
| `no_agenda__respect_creators tp=4` | 0.90 | 0.83 | 0.97 | 0.95 | +0 |
| `no_topic_off_limits__respect_creators tp=4` | 0.96 | 0.76 | 0.80 | 0.96 | +0 |
| `prevent_imminent_harm__support_programmatic_use tp=1` | 0.90 | 0.84 | 0.61 | 0.85 | +0 |
| `protect_privacy__support_mental_health tp=4` | 0.93 | 0.92 | 0.91 | 0.79 | +0 |
