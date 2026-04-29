# Rubric set comparison: with_self_edits vs with_union_edits

**Baseline**: `experiments/posttrain/stage3_output/cross_tier_rubrics_v2_pro_with_self_edits.jsonl` (22 rows)
**Treatment**: `experiments/posttrain/stage3_output/cross_tier_rubrics_v2_pro_with_union_edits.jsonl` (22 rows)
**Common pairs**: 22

## Per-field aggregate change

Mean text change (1 - SequenceMatcher.ratio) across all common pairs. Higher = more changed. Threshold for 'significant change' = 0.3.

| field | mean Δ | median Δ | max Δ | n significant (>0.3) |
|---|---:|---:|---:|---:|
| dominant_rubric.GOOD | 0.587 | 0.600 | 0.874 | 20/22 |
| dominant_rubric.BAD | 0.520 | 0.498 | 0.926 | 17/22 |
| dominant_rubric.KEY_TENSION | 0.895 | 0.913 | 0.968 | 22/22 |
| non_leakage_rubric.GOOD | 0.723 | 0.791 | 0.952 | 22/22 |
| non_leakage_rubric.BAD | 0.715 | 0.811 | 0.940 | 20/22 |
| non_leakage_rubric.KEY_TENSION | 0.829 | 0.915 | 0.980 | 22/22 |
| worked_example.spec_compliant | 0.637 | 0.733 | 0.968 | 19/22 |
| worked_example.subordinate_over_satisfaction | 0.683 | 0.667 | 0.990 | 22/22 |
| worked_example.pathological_refusal | 0.545 | 0.592 | 0.977 | 19/22 |
| rationale.interpretive_choices_made | 0.805 | 0.894 | 0.986 | 21/22 |
| rationale.alternative_readings_rejected | 0.899 | 0.921 | 1.000 | 22/22 |

## Rationale clause changes

- Avg `spec_clauses_anchored_on` count in with_self_edits: 3.2
- Avg `spec_clauses_anchored_on` count in with_union_edits: 2.9
- Avg new clauses (in treatment, not baseline): 1.3
- Avg removed clauses (in baseline, not treatment): 1.6

## Per-pair changes

| pair | dom.BAD Δ | non_leak.BAD Δ | spec_compliant Δ | alt_read Δ | clauses Δ |
|---|---:|---:|---:|---:|---:|
| `assume_objective_pov__do_not_encourage_self_harm tp=0` | 0.48 | 0.34 | 0.70 | 0.92 | -1 |
| `assume_objective_pov__do_not_encourage_self_harm tp=3` | 0.91 | 0.44 | 0.73 | 0.67 | -1 |
| `assume_objective_pov__do_not_facilitate_illicit_behavior tp=5` | 0.53 | 0.76 | 0.94 | 0.98 | -1 |
| `avoid_abuse__letter_and_spirit tp=2` | 0.28 | 0.81 | 0.82 | 0.92 | +1 |
| `avoid_being_condescending__letter_and_spirit tp=5` | 0.90 | 0.94 | 0.79 | 0.97 | -2 |
| `avoid_errors__avoid_info_hazards tp=2` | 0.29 | 0.26 | 0.38 | 0.93 | +0 |
| `avoid_errors__letter_and_spirit tp=4` | 0.50 | 0.85 | 0.55 | 0.77 | +0 |
| `avoid_hateful_content__be_rationally_optimistic tp=5` | 0.46 | 0.85 | 0.87 | 0.91 | +0 |
| `avoid_hateful_content__present_perspectives tp=3` | 0.79 | 0.78 | 0.94 | 0.95 | +0 |
| `avoid_targeted_political_manipulation__be_kind tp=0` | 0.47 | 0.90 | 0.58 | 0.77 | +0 |
| `avoid_targeted_political_manipulation__do_not_lie tp=0` | 0.14 | 0.84 | 0.97 | 0.90 | +0 |
| `be_kind__uphold_fairness tp=1` | 0.93 | 0.78 | 0.19 | 0.90 | -1 |
| `comply_with_laws__no_agenda tp=5` | 0.55 | 0.90 | 0.91 | 0.92 | +1 |
| `comply_with_laws__no_agenda tp=6` | 0.38 | 0.55 | 0.96 | 0.99 | +0 |
| `do_not_facilitate_illicit_behavior__no_agenda tp=0` | 0.73 | 0.90 | 0.88 | 0.73 | +0 |
| `express_uncertainty__prevent_imminent_harm tp=5` | 0.54 | 0.62 | 0.39 | 0.92 | +0 |
| `formatting__letter_and_spirit tp=4` | 0.48 | 0.84 | 0.56 | 0.94 | -2 |
| `no_agenda__respect_creators tp=3` | 0.26 | 0.53 | 0.39 | 0.94 | +0 |
| `no_agenda__respect_creators tp=4` | 0.53 | 0.76 | 0.05 | 0.94 | +0 |
| `no_topic_off_limits__respect_creators tp=4` | 0.28 | 0.30 | 0.24 | 0.89 | +0 |
| `prevent_imminent_harm__support_programmatic_use tp=1` | 0.67 | 0.89 | 0.41 | 0.93 | +0 |
| `protect_privacy__support_mental_health tp=4` | 0.35 | 0.88 | 0.73 | 1.00 | -1 |