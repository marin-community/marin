# Rubric set comparison: OptionA_self_edits vs OptionB_alwayson

**Baseline**: `experiments/posttrain/stage3_output/cross_tier_rubrics_v2_pro_with_self_edits.jsonl` (22 rows)
**Treatment**: `experiments/posttrain/stage3_output/cross_tier_rubrics_v3_alwayson_pro.jsonl` (22 rows)
**Common pairs**: 22

## Per-field aggregate change

Mean text change (1 - SequenceMatcher.ratio) across all common pairs. Higher = more changed. Threshold for 'significant change' = 0.3.

| field | mean Δ | median Δ | max Δ | n significant (>0.3) |
|---|---:|---:|---:|---:|
| dominant_rubric.GOOD | 0.570 | 0.568 | 0.935 | 20/22 |
| dominant_rubric.BAD | 0.625 | 0.603 | 0.933 | 21/22 |
| dominant_rubric.KEY_TENSION | 0.903 | 0.940 | 0.978 | 22/22 |
| non_leakage_rubric.GOOD | 0.684 | 0.745 | 0.917 | 21/22 |
| non_leakage_rubric.BAD | 0.780 | 0.891 | 0.976 | 21/22 |
| non_leakage_rubric.KEY_TENSION | 0.859 | 0.919 | 0.982 | 22/22 |
| worked_example.spec_compliant | 0.640 | 0.730 | 0.969 | 19/22 |
| worked_example.subordinate_over_satisfaction | 0.637 | 0.728 | 0.982 | 18/22 |
| worked_example.pathological_refusal | 0.561 | 0.571 | 0.934 | 20/22 |
| rationale.interpretive_choices_made | 0.813 | 0.891 | 0.965 | 22/22 |
| rationale.alternative_readings_rejected | 0.904 | 0.909 | 1.000 | 22/22 |

## Rationale clause changes

- Avg `spec_clauses_anchored_on` count in OptionA_self_edits: 3.2
- Avg `spec_clauses_anchored_on` count in OptionB_alwayson: 3.3
- Avg new clauses (in treatment, not baseline): 1.5
- Avg removed clauses (in baseline, not treatment): 1.5

## Per-pair changes

| pair | dom.BAD Δ | non_leak.BAD Δ | spec_compliant Δ | alt_read Δ | clauses Δ |
|---|---:|---:|---:|---:|---:|
| `assume_objective_pov__do_not_encourage_self_harm tp=0` | 0.60 | 0.97 | 0.53 | 0.83 | -1 |
| `assume_objective_pov__do_not_encourage_self_harm tp=3` | 0.90 | 0.96 | 0.73 | 0.91 | -1 |
| `assume_objective_pov__do_not_facilitate_illicit_behavior tp=5` | 0.47 | 0.86 | 0.95 | 0.84 | +0 |
| `avoid_abuse__letter_and_spirit tp=2` | 0.60 | 0.52 | 0.32 | 0.88 | +1 |
| `avoid_being_condescending__letter_and_spirit tp=5` | 0.81 | 0.98 | 0.85 | 0.98 | -1 |
| `avoid_errors__avoid_info_hazards tp=2` | 0.90 | 0.69 | 0.52 | 0.99 | -1 |
| `avoid_errors__letter_and_spirit tp=4` | 0.43 | 0.72 | 0.76 | 0.79 | +0 |
| `avoid_hateful_content__be_rationally_optimistic tp=5` | 0.78 | 0.64 | 0.86 | 0.93 | +1 |
| `avoid_hateful_content__present_perspectives tp=3` | 0.58 | 0.91 | 0.92 | 0.99 | +0 |
| `avoid_targeted_political_manipulation__be_kind tp=0` | 0.41 | 0.89 | 0.83 | 0.91 | +0 |
| `avoid_targeted_political_manipulation__do_not_lie tp=0` | 0.56 | 0.92 | 0.55 | 0.94 | +0 |
| `be_kind__uphold_fairness tp=1` | 0.77 | 0.90 | 0.38 | 0.86 | -1 |
| `comply_with_laws__no_agenda tp=5` | 0.59 | 0.55 | 0.29 | 0.78 | +1 |
| `comply_with_laws__no_agenda tp=6` | 0.69 | 0.90 | 0.90 | 0.91 | +1 |
| `do_not_facilitate_illicit_behavior__no_agenda tp=0` | 0.61 | 0.97 | 0.84 | 0.89 | +0 |
| `express_uncertainty__prevent_imminent_harm tp=5` | 0.56 | 0.90 | 0.97 | 0.87 | +1 |
| `formatting__letter_and_spirit tp=4` | 0.62 | 0.55 | 0.55 | 0.77 | +0 |
| `no_agenda__respect_creators tp=3` | 0.45 | 0.64 | 0.42 | 0.98 | +1 |
| `no_agenda__respect_creators tp=4` | 0.45 | 0.81 | 0.26 | 0.94 | +0 |
| `no_topic_off_limits__respect_creators tp=4` | 0.29 | 0.27 | 0.30 | 0.96 | +0 |
| `prevent_imminent_harm__support_programmatic_use tp=1` | 0.93 | 0.91 | 0.55 | 0.94 | +0 |
| `protect_privacy__support_mental_health tp=4` | 0.75 | 0.72 | 0.81 | 1.00 | +0 |