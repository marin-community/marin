# Rubric set comparison: with_self_edits vs with_union_edits

**Baseline**: `experiments/posttrain/stage3_output/cross_tier_rubrics_v2_glm51_with_self_edits.jsonl` (22 rows)
**Treatment**: `experiments/posttrain/stage3_output/cross_tier_rubrics_v2_glm51_with_union_edits.jsonl` (22 rows)
**Common pairs**: 22

## Per-field aggregate change

Mean text change (1 - SequenceMatcher.ratio) across all common pairs. Higher = more changed. Threshold for 'significant change' = 0.3.

| field | mean Δ | median Δ | max Δ | n significant (>0.3) |
|---|---:|---:|---:|---:|
| dominant_rubric.GOOD | 0.817 | 0.868 | 0.991 | 22/22 |
| dominant_rubric.BAD | 0.831 | 0.852 | 0.959 | 22/22 |
| dominant_rubric.KEY_TENSION | 0.913 | 0.924 | 0.975 | 22/22 |
| non_leakage_rubric.GOOD | 0.868 | 0.902 | 0.955 | 22/22 |
| non_leakage_rubric.BAD | 0.750 | 0.767 | 0.862 | 22/22 |
| non_leakage_rubric.KEY_TENSION | 0.869 | 0.896 | 0.968 | 22/22 |
| worked_example.spec_compliant | 0.822 | 0.915 | 0.987 | 22/22 |
| worked_example.subordinate_over_satisfaction | 0.708 | 0.831 | 0.967 | 19/22 |
| worked_example.pathological_refusal | 0.599 | 0.676 | 0.968 | 21/22 |
| rationale.interpretive_choices_made | 0.920 | 0.941 | 0.989 | 22/22 |
| rationale.alternative_readings_rejected | 0.898 | 0.928 | 0.982 | 22/22 |

## Rationale clause changes

- Avg `spec_clauses_anchored_on` count in with_self_edits: 4.6
- Avg `spec_clauses_anchored_on` count in with_union_edits: 4.8
- Avg new clauses (in treatment, not baseline): 2.9
- Avg removed clauses (in baseline, not treatment): 2.6

## Per-pair changes

| pair | dom.BAD Δ | non_leak.BAD Δ | spec_compliant Δ | alt_read Δ | clauses Δ |
|---|---:|---:|---:|---:|---:|
| `assume_objective_pov__do_not_encourage_self_harm tp=0` | 0.94 | 0.82 | 0.60 | 0.89 | +1 |
| `assume_objective_pov__do_not_encourage_self_harm tp=3` | 0.74 | 0.72 | 0.92 | 0.92 | +0 |
| `assume_objective_pov__do_not_facilitate_illicit_behavior tp=5` | 0.92 | 0.82 | 0.76 | 0.72 | +1 |
| `avoid_abuse__letter_and_spirit tp=2` | 0.94 | 0.77 | 0.95 | 0.98 | +1 |
| `avoid_being_condescending__letter_and_spirit tp=5` | 0.85 | 0.70 | 0.86 | 0.93 | +0 |
| `avoid_errors__avoid_info_hazards tp=2` | 0.59 | 0.78 | 0.94 | 0.72 | +0 |
| `avoid_errors__letter_and_spirit tp=4` | 0.96 | 0.86 | 0.92 | 0.96 | +0 |
| `avoid_hateful_content__be_rationally_optimistic tp=5` | 0.61 | 0.77 | 0.97 | 0.93 | +0 |
| `avoid_hateful_content__present_perspectives tp=3` | 0.77 | 0.77 | 0.94 | 0.91 | +1 |
| `avoid_targeted_political_manipulation__be_kind tp=0` | 0.82 | 0.81 | 0.96 | 0.98 | +1 |
| `avoid_targeted_political_manipulation__do_not_lie tp=0` | 0.88 | 0.78 | 0.92 | 0.96 | +0 |
| `be_kind__uphold_fairness tp=1` | 0.94 | 0.74 | 0.31 | 0.97 | -1 |
| `comply_with_laws__no_agenda tp=5` | 0.86 | 0.82 | 0.99 | 0.92 | +1 |
| `comply_with_laws__no_agenda tp=6` | 0.84 | 0.71 | 0.94 | 0.79 | +1 |
| `do_not_facilitate_illicit_behavior__no_agenda tp=0` | 0.70 | 0.80 | 0.95 | 0.80 | +0 |
| `express_uncertainty__prevent_imminent_harm tp=5` | 0.88 | 0.69 | 0.85 | 0.96 | +1 |
| `formatting__letter_and_spirit tp=4` | 0.85 | 0.72 | 0.73 | 0.83 | -1 |
| `no_agenda__respect_creators tp=3` | 0.77 | 0.74 | 0.68 | 0.84 | -1 |
| `no_agenda__respect_creators tp=4` | 0.80 | 0.69 | 0.72 | 0.94 | -1 |
| `no_topic_off_limits__respect_creators tp=4` | 0.85 | 0.61 | 0.68 | 0.98 | +1 |
| `prevent_imminent_harm__support_programmatic_use tp=1` | 0.88 | 0.73 | 0.89 | 0.85 | -1 |
| `protect_privacy__support_mental_health tp=4` | 0.91 | 0.64 | 0.62 | 0.98 | +1 |
