# Rubric set comparison: OptionA_self_edits vs OptionB_alwayson

**Baseline**: `experiments/posttrain/stage3_output/cross_tier_rubrics_v2_gpt51_with_self_edits.jsonl` (22 rows)
**Treatment**: `experiments/posttrain/stage3_output/cross_tier_rubrics_v3_alwayson_gpt51.jsonl` (22 rows)
**Common pairs**: 22

## Per-field aggregate change

Mean text change (1 - SequenceMatcher.ratio) across all common pairs. Higher = more changed. Threshold for 'significant change' = 0.3.

| field | mean Δ | median Δ | max Δ | n significant (>0.3) |
|---|---:|---:|---:|---:|
| dominant_rubric.GOOD | 0.886 | 0.894 | 0.975 | 22/22 |
| dominant_rubric.BAD | 0.914 | 0.922 | 0.978 | 22/22 |
| dominant_rubric.KEY_TENSION | 0.932 | 0.954 | 0.992 | 22/22 |
| non_leakage_rubric.GOOD | 0.923 | 0.923 | 0.978 | 22/22 |
| non_leakage_rubric.BAD | 0.873 | 0.874 | 0.993 | 22/22 |
| non_leakage_rubric.KEY_TENSION | 0.909 | 0.930 | 0.993 | 22/22 |
| worked_example.spec_compliant | 0.874 | 0.923 | 0.982 | 22/22 |
| worked_example.subordinate_over_satisfaction | 0.829 | 0.861 | 0.988 | 22/22 |
| worked_example.pathological_refusal | 0.654 | 0.653 | 0.834 | 22/22 |
| rationale.interpretive_choices_made | 0.890 | 0.935 | 0.969 | 22/22 |
| rationale.alternative_readings_rejected | 0.925 | 0.933 | 0.988 | 22/22 |

## Rationale clause changes

- Avg `spec_clauses_anchored_on` count in OptionA_self_edits: 5.3
- Avg `spec_clauses_anchored_on` count in OptionB_alwayson: 5.8
- Avg new clauses (in treatment, not baseline): 2.8
- Avg removed clauses (in baseline, not treatment): 2.3

## Per-pair changes

| pair | dom.BAD Δ | non_leak.BAD Δ | spec_compliant Δ | alt_read Δ | clauses Δ |
|---|---:|---:|---:|---:|---:|
| `assume_objective_pov__do_not_encourage_self_harm tp=0` | 0.97 | 0.83 | 0.86 | 0.96 | +0 |
| `assume_objective_pov__do_not_encourage_self_harm tp=3` | 0.98 | 0.98 | 0.92 | 0.88 | +0 |
| `assume_objective_pov__do_not_facilitate_illicit_behavior tp=5` | 0.91 | 0.99 | 0.95 | 0.93 | +0 |
| `avoid_abuse__letter_and_spirit tp=2` | 0.96 | 0.87 | 0.95 | 0.96 | +0 |
| `avoid_being_condescending__letter_and_spirit tp=5` | 0.95 | 0.72 | 0.87 | 0.97 | -1 |
| `avoid_errors__avoid_info_hazards tp=2` | 0.93 | 0.83 | 0.96 | 0.97 | +0 |
| `avoid_errors__letter_and_spirit tp=4` | 0.88 | 0.82 | 0.61 | 0.91 | +0 |
| `avoid_hateful_content__be_rationally_optimistic tp=5` | 0.90 | 0.98 | 0.90 | 0.95 | +0 |
| `avoid_hateful_content__present_perspectives tp=3` | 0.95 | 0.79 | 0.97 | 0.96 | +0 |
| `avoid_targeted_political_manipulation__be_kind tp=0` | 0.87 | 0.81 | 0.98 | 0.94 | +0 |
| `avoid_targeted_political_manipulation__do_not_lie tp=0` | 0.91 | 0.91 | 0.98 | 0.88 | +1 |
| `be_kind__uphold_fairness tp=1` | 0.96 | 0.87 | 0.97 | 0.94 | +0 |
| `comply_with_laws__no_agenda tp=5` | 0.92 | 0.94 | 0.81 | 0.99 | +1 |
| `comply_with_laws__no_agenda tp=6` | 0.84 | 0.95 | 0.88 | 0.92 | +3 |
| `do_not_facilitate_illicit_behavior__no_agenda tp=0` | 0.96 | 0.92 | 0.98 | 0.97 | +0 |
| `express_uncertainty__prevent_imminent_harm tp=5` | 0.91 | 0.77 | 0.88 | 0.86 | +0 |
| `formatting__letter_and_spirit tp=4` | 0.81 | 0.90 | 0.96 | 0.93 | +0 |
| `no_agenda__respect_creators tp=3` | 0.90 | 0.85 | 0.52 | 0.91 | +3 |
| `no_agenda__respect_creators tp=4` | 0.86 | 0.86 | 0.83 | 0.90 | +2 |
| `no_topic_off_limits__respect_creators tp=4` | 0.92 | 0.87 | 0.96 | 0.85 | +1 |
| `prevent_imminent_harm__support_programmatic_use tp=1` | 0.96 | 0.93 | 0.61 | 0.86 | +1 |
| `protect_privacy__support_mental_health tp=4` | 0.84 | 0.79 | 0.87 | 0.91 | +0 |