# Rubric set comparison: with_self_edits vs with_r1r2_edits

**Baseline**: `experiments/posttrain/stage3_output/cross_tier_rubrics_v2_flash_with_self_edits.jsonl` (22 rows)
**Treatment**: `experiments/posttrain/stage3_output/cross_tier_rubrics_v2_flash_with_r1r2_edits.jsonl` (22 rows)
**Common pairs**: 22

## Per-field aggregate change

Mean text change (1 - SequenceMatcher.ratio) across all common pairs. Higher = more changed. Threshold for 'significant change' = 0.3.

| field | mean Δ | median Δ | max Δ | n significant (>0.3) |
|---|---:|---:|---:|---:|
| dominant_rubric.GOOD | 0.711 | 0.740 | 0.938 | 22/22 |
| dominant_rubric.BAD | 0.813 | 0.852 | 0.942 | 22/22 |
| dominant_rubric.KEY_TENSION | 0.890 | 0.934 | 0.974 | 22/22 |
| non_leakage_rubric.GOOD | 0.829 | 0.859 | 0.971 | 22/22 |
| non_leakage_rubric.BAD | 0.766 | 0.757 | 0.941 | 22/22 |
| non_leakage_rubric.KEY_TENSION | 0.867 | 0.902 | 0.985 | 22/22 |
| worked_example.spec_compliant | 0.713 | 0.842 | 0.982 | 19/22 |
| worked_example.subordinate_over_satisfaction | 0.718 | 0.806 | 0.972 | 21/22 |
| worked_example.pathological_refusal | 0.554 | 0.600 | 0.954 | 19/22 |
| rationale.interpretive_choices_made | 0.755 | 0.843 | 0.979 | 21/22 |
| rationale.alternative_readings_rejected | 0.804 | 0.815 | 0.900 | 22/22 |

## Rationale clause changes

- Avg `spec_clauses_anchored_on` count in with_self_edits: 4.6
- Avg `spec_clauses_anchored_on` count in with_r1r2_edits: 4.5
- Avg new clauses (in treatment, not baseline): 1.1
- Avg removed clauses (in baseline, not treatment): 1.2

## Per-pair changes

| pair | dom.BAD Δ | non_leak.BAD Δ | spec_compliant Δ | alt_read Δ | clauses Δ |
|---|---:|---:|---:|---:|---:|
| `assume_objective_pov__do_not_encourage_self_harm tp=0` | 0.79 | 0.73 | 0.80 | 0.83 | +0 |
| `assume_objective_pov__do_not_encourage_self_harm tp=3` | 0.89 | 0.74 | 0.51 | 0.86 | +1 |
| `assume_objective_pov__do_not_facilitate_illicit_behavior tp=5` | 0.78 | 0.77 | 0.81 | 0.67 | +0 |
| `avoid_abuse__letter_and_spirit tp=2` | 0.92 | 0.88 | 0.97 | 0.87 | -1 |
| `avoid_being_condescending__letter_and_spirit tp=5` | 0.94 | 0.94 | 0.89 | 0.88 | -1 |
| `avoid_errors__avoid_info_hazards tp=2` | 0.91 | 0.70 | 0.98 | 0.81 | +0 |
| `avoid_errors__letter_and_spirit tp=4` | 0.81 | 0.73 | 0.90 | 0.90 | +0 |
| `avoid_hateful_content__be_rationally_optimistic tp=5` | 0.80 | 0.73 | 0.84 | 0.81 | +0 |
| `avoid_hateful_content__present_perspectives tp=3` | 0.90 | 0.76 | 0.97 | 0.77 | +0 |
| `avoid_targeted_political_manipulation__be_kind tp=0` | 0.91 | 0.93 | 0.26 | 0.79 | -1 |
| `avoid_targeted_political_manipulation__do_not_lie tp=0` | 0.79 | 0.90 | 0.90 | 0.75 | +0 |
| `be_kind__uphold_fairness tp=1` | 0.92 | 0.89 | 0.66 | 0.88 | +1 |
| `comply_with_laws__no_agenda tp=5` | 0.85 | 0.89 | 0.67 | 0.77 | +0 |
| `comply_with_laws__no_agenda tp=6` | 0.52 | 0.78 | 0.87 | 0.81 | +0 |
| `do_not_facilitate_illicit_behavior__no_agenda tp=0` | 0.87 | 0.57 | 0.54 | 0.59 | +0 |
| `express_uncertainty__prevent_imminent_harm tp=5` | 0.68 | 0.73 | 0.95 | 0.64 | -1 |
| `formatting__letter_and_spirit tp=4` | 0.89 | 0.71 | 0.02 | 0.85 | +0 |
| `no_agenda__respect_creators tp=3` | 0.60 | 0.78 | 0.20 | 0.86 | +0 |
| `no_agenda__respect_creators tp=4` | 0.71 | 0.79 | 0.44 | 0.87 | -1 |
| `no_topic_off_limits__respect_creators tp=4` | 0.70 | 0.72 | 0.84 | 0.81 | +0 |
| `prevent_imminent_harm__support_programmatic_use tp=1` | 0.82 | 0.51 | 0.85 | 0.85 | +1 |
| `protect_privacy__support_mental_health tp=4` | 0.89 | 0.67 | 0.78 | 0.81 | +0 |
