# Rubric set comparison: with_self_edits vs with_r1r2_edits

**Baseline**: `experiments/posttrain/stage3_output/cross_tier_rubrics_v2_gpt51_with_self_edits.jsonl` (22 rows)
**Treatment**: `experiments/posttrain/stage3_output/cross_tier_rubrics_v2_gpt51_with_r1r2_edits.jsonl` (22 rows)
**Common pairs**: 22

## Per-field aggregate change

Mean text change (1 - SequenceMatcher.ratio) across all common pairs. Higher = more changed. Threshold for 'significant change' = 0.3.

| field | mean Δ | median Δ | max Δ | n significant (>0.3) |
|---|---:|---:|---:|---:|
| dominant_rubric.GOOD | 0.870 | 0.918 | 0.957 | 22/22 |
| dominant_rubric.BAD | 0.924 | 0.936 | 0.973 | 22/22 |
| dominant_rubric.KEY_TENSION | 0.934 | 0.947 | 0.983 | 22/22 |
| non_leakage_rubric.GOOD | 0.887 | 0.918 | 0.958 | 22/22 |
| non_leakage_rubric.BAD | 0.881 | 0.918 | 0.995 | 22/22 |
| non_leakage_rubric.KEY_TENSION | 0.920 | 0.936 | 0.988 | 22/22 |
| worked_example.spec_compliant | 0.785 | 0.847 | 0.975 | 21/22 |
| worked_example.subordinate_over_satisfaction | 0.871 | 0.920 | 0.975 | 22/22 |
| worked_example.pathological_refusal | 0.595 | 0.603 | 0.932 | 19/22 |
| rationale.interpretive_choices_made | 0.902 | 0.925 | 0.982 | 22/22 |
| rationale.alternative_readings_rejected | 0.926 | 0.933 | 0.982 | 22/22 |

## Rationale clause changes

- Avg `spec_clauses_anchored_on` count in with_self_edits: 5.3
- Avg `spec_clauses_anchored_on` count in with_r1r2_edits: 5.2
- Avg new clauses (in treatment, not baseline): 1.4
- Avg removed clauses (in baseline, not treatment): 1.5

## Per-pair changes

| pair | dom.BAD Δ | non_leak.BAD Δ | spec_compliant Δ | alt_read Δ | clauses Δ |
|---|---:|---:|---:|---:|---:|
| `assume_objective_pov__do_not_encourage_self_harm tp=0` | 0.96 | 0.78 | 0.87 | 0.88 | -1 |
| `assume_objective_pov__do_not_encourage_self_harm tp=3` | 0.94 | 0.95 | 0.83 | 0.93 | +0 |
| `assume_objective_pov__do_not_facilitate_illicit_behavior tp=5` | 0.94 | 0.82 | 0.82 | 0.93 | +0 |
| `avoid_abuse__letter_and_spirit tp=2` | 0.93 | 0.88 | 0.81 | 0.85 | +0 |
| `avoid_being_condescending__letter_and_spirit tp=5` | 0.94 | 0.94 | 0.97 | 0.95 | -1 |
| `avoid_errors__avoid_info_hazards tp=2` | 0.94 | 0.97 | 0.90 | 0.98 | +0 |
| `avoid_errors__letter_and_spirit tp=4` | 0.94 | 0.98 | 0.28 | 0.93 | +0 |
| `avoid_hateful_content__be_rationally_optimistic tp=5` | 0.94 | 0.93 | 0.89 | 0.94 | -1 |
| `avoid_hateful_content__present_perspectives tp=3` | 0.89 | 0.75 | 0.97 | 0.93 | +1 |
| `avoid_targeted_political_manipulation__be_kind tp=0` | 0.91 | 0.74 | 0.67 | 0.95 | +0 |
| `avoid_targeted_political_manipulation__do_not_lie tp=0` | 0.88 | 0.95 | 0.44 | 0.98 | +0 |
| `be_kind__uphold_fairness tp=1` | 0.97 | 0.92 | 0.88 | 0.86 | +1 |
| `comply_with_laws__no_agenda tp=5` | 0.92 | 0.94 | 0.85 | 0.95 | +1 |
| `comply_with_laws__no_agenda tp=6` | 0.95 | 0.88 | 0.88 | 0.93 | -1 |
| `do_not_facilitate_illicit_behavior__no_agenda tp=0` | 0.93 | 0.96 | 0.71 | 0.94 | +0 |
| `express_uncertainty__prevent_imminent_harm tp=5` | 0.89 | 0.85 | 0.78 | 0.95 | +0 |
| `formatting__letter_and_spirit tp=4` | 0.90 | 0.83 | 0.47 | 0.89 | +0 |
| `no_agenda__respect_creators tp=3` | 0.85 | 0.80 | 0.72 | 0.91 | +0 |
| `no_agenda__respect_creators tp=4` | 0.94 | 0.70 | 0.90 | 0.97 | -1 |
| `no_topic_off_limits__respect_creators tp=4` | 0.90 | 0.90 | 0.96 | 0.94 | -1 |
| `prevent_imminent_harm__support_programmatic_use tp=1` | 0.93 | 0.94 | 0.69 | 0.90 | +0 |
| `protect_privacy__support_mental_health tp=4` | 0.94 | 0.99 | 0.96 | 0.90 | +0 |