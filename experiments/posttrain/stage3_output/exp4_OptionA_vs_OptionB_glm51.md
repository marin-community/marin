# Rubric set comparison: OptionA_self_edits vs OptionB_alwayson

**Baseline**: `experiments/posttrain/stage3_output/cross_tier_rubrics_v2_glm51_with_self_edits.jsonl` (22 rows)
**Treatment**: `experiments/posttrain/stage3_output/cross_tier_rubrics_v3_alwayson_glm51.jsonl` (22 rows)
**Common pairs**: 22

## Per-field aggregate change

Mean text change (1 - SequenceMatcher.ratio) across all common pairs. Higher = more changed. Threshold for 'significant change' = 0.3.

| field | mean Δ | median Δ | max Δ | n significant (>0.3) |
|---|---:|---:|---:|---:|
| dominant_rubric.GOOD | 0.815 | 0.852 | 0.982 | 22/22 |
| dominant_rubric.BAD | 0.878 | 0.904 | 0.958 | 22/22 |
| dominant_rubric.KEY_TENSION | 0.919 | 0.933 | 0.993 | 22/22 |
| non_leakage_rubric.GOOD | 0.881 | 0.900 | 0.974 | 22/22 |
| non_leakage_rubric.BAD | 0.784 | 0.811 | 0.890 | 22/22 |
| non_leakage_rubric.KEY_TENSION | 0.897 | 0.904 | 0.978 | 22/22 |
| worked_example.spec_compliant | 0.847 | 0.878 | 0.989 | 22/22 |
| worked_example.subordinate_over_satisfaction | 0.810 | 0.905 | 0.993 | 21/22 |
| worked_example.pathological_refusal | 0.629 | 0.653 | 0.956 | 21/22 |
| rationale.interpretive_choices_made | 0.897 | 0.922 | 0.977 | 22/22 |
| rationale.alternative_readings_rejected | 0.902 | 0.923 | 0.990 | 22/22 |

## Rationale clause changes

- Avg `spec_clauses_anchored_on` count in OptionA_self_edits: 4.6
- Avg `spec_clauses_anchored_on` count in OptionB_alwayson: 4.9
- Avg new clauses (in treatment, not baseline): 3.0
- Avg removed clauses (in baseline, not treatment): 2.7

## Per-pair changes

| pair | dom.BAD Δ | non_leak.BAD Δ | spec_compliant Δ | alt_read Δ | clauses Δ |
|---|---:|---:|---:|---:|---:|
| `assume_objective_pov__do_not_encourage_self_harm tp=0` | 0.95 | 0.86 | 0.76 | 0.90 | -1 |
| `assume_objective_pov__do_not_encourage_self_harm tp=3` | 0.89 | 0.74 | 0.61 | 0.94 | +0 |
| `assume_objective_pov__do_not_facilitate_illicit_behavior tp=5` | 0.81 | 0.70 | 0.88 | 0.93 | +0 |
| `avoid_abuse__letter_and_spirit tp=2` | 0.92 | 0.89 | 0.96 | 0.96 | +1 |
| `avoid_being_condescending__letter_and_spirit tp=5` | 0.92 | 0.74 | 0.88 | 0.74 | +0 |
| `avoid_errors__avoid_info_hazards tp=2` | 0.94 | 0.71 | 0.96 | 0.96 | +0 |
| `avoid_errors__letter_and_spirit tp=4` | 0.95 | 0.81 | 0.85 | 0.99 | +1 |
| `avoid_hateful_content__be_rationally_optimistic tp=5` | 0.80 | 0.68 | 0.73 | 0.94 | -1 |
| `avoid_hateful_content__present_perspectives tp=3` | 0.96 | 0.81 | 0.81 | 0.93 | +3 |
| `avoid_targeted_political_manipulation__be_kind tp=0` | 0.78 | 0.85 | 0.55 | 0.94 | +1 |
| `avoid_targeted_political_manipulation__do_not_lie tp=0` | 0.87 | 0.84 | 0.91 | 0.88 | +0 |
| `be_kind__uphold_fairness tp=1` | 0.95 | 0.80 | 0.99 | 0.92 | -2 |
| `comply_with_laws__no_agenda tp=5` | 0.71 | 0.85 | 0.77 | 0.95 | +2 |
| `comply_with_laws__no_agenda tp=6` | 0.94 | 0.84 | 0.98 | 0.84 | +3 |
| `do_not_facilitate_illicit_behavior__no_agenda tp=0` | 0.94 | 0.85 | 0.95 | 0.71 | +1 |
| `express_uncertainty__prevent_imminent_harm tp=5` | 0.77 | 0.72 | 0.97 | 0.85 | +2 |
| `formatting__letter_and_spirit tp=4` | 0.84 | 0.63 | 0.82 | 0.82 | -2 |
| `no_agenda__respect_creators tp=3` | 0.84 | 0.70 | 0.75 | 0.89 | -1 |
| `no_agenda__respect_creators tp=4` | 0.93 | 0.83 | 0.97 | 0.92 | +0 |
| `no_topic_off_limits__respect_creators tp=4` | 0.87 | 0.74 | 0.78 | 0.92 | +0 |
| `prevent_imminent_harm__support_programmatic_use tp=1` | 0.90 | 0.81 | 0.89 | 0.92 | -1 |
| `protect_privacy__support_mental_health tp=4` | 0.83 | 0.84 | 0.88 | 0.98 | +1 |
