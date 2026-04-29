# Rubric set comparison: v2_baseline vs v3_alwayson

**Baseline**: `experiments/posttrain/stage3_output/cross_tier_rubrics_v2_gpt51.jsonl` (22 rows)
**Treatment**: `experiments/posttrain/stage3_output/cross_tier_rubrics_v3_alwayson_gpt51.jsonl` (22 rows)
**Common pairs**: 22

## Per-field aggregate change

Mean text change (1 - SequenceMatcher.ratio) across all common pairs. Higher = more changed. Threshold for 'significant change' = 0.3.

| field | mean Δ | median Δ | max Δ | n significant (>0.3) |
|---|---:|---:|---:|---:|
| dominant_rubric.GOOD | 0.903 | 0.918 | 0.993 | 22/22 |
| dominant_rubric.BAD | 0.917 | 0.917 | 0.975 | 22/22 |
| dominant_rubric.KEY_TENSION | 0.940 | 0.955 | 0.991 | 22/22 |
| non_leakage_rubric.GOOD | 0.906 | 0.911 | 0.980 | 22/22 |
| non_leakage_rubric.BAD | 0.897 | 0.936 | 0.996 | 22/22 |
| non_leakage_rubric.KEY_TENSION | 0.917 | 0.931 | 0.990 | 22/22 |
| worked_example.spec_compliant | 0.832 | 0.889 | 0.987 | 22/22 |
| worked_example.subordinate_over_satisfaction | 0.817 | 0.853 | 0.989 | 22/22 |
| worked_example.pathological_refusal | 0.607 | 0.625 | 0.853 | 21/22 |
| rationale.interpretive_choices_made | 0.889 | 0.903 | 0.975 | 22/22 |
| rationale.alternative_readings_rejected | 0.923 | 0.924 | 0.979 | 22/22 |

## Rationale clause changes

- Avg `spec_clauses_anchored_on` count in v2_baseline: 5.1
- Avg `spec_clauses_anchored_on` count in v3_alwayson: 5.8
- Avg new clauses (in treatment, not baseline): 2.7
- Avg removed clauses (in baseline, not treatment): 2.0

## Per-pair changes

| pair | dom.BAD Δ | non_leak.BAD Δ | spec_compliant Δ | alt_read Δ | clauses Δ |
|---|---:|---:|---:|---:|---:|
| `assume_objective_pov__do_not_encourage_self_harm tp=0` | 0.93 | 0.81 | 0.82 | 0.98 | +0 |
| `assume_objective_pov__do_not_encourage_self_harm tp=3` | 0.97 | 0.96 | 0.71 | 0.94 | +0 |
| `assume_objective_pov__do_not_facilitate_illicit_behavior tp=5` | 0.84 | 0.99 | 0.99 | 0.93 | +0 |
| `avoid_abuse__letter_and_spirit tp=2` | 0.95 | 0.89 | 0.97 | 0.89 | +0 |
| `avoid_being_condescending__letter_and_spirit tp=5` | 0.95 | 0.95 | 0.96 | 0.91 | +0 |
| `avoid_errors__avoid_info_hazards tp=2` | 0.96 | 0.99 | 0.96 | 0.97 | +0 |
| `avoid_errors__letter_and_spirit tp=4` | 0.96 | 0.84 | 0.35 | 0.86 | +0 |
| `avoid_hateful_content__be_rationally_optimistic tp=5` | 0.86 | 0.98 | 0.95 | 0.94 | +1 |
| `avoid_hateful_content__present_perspectives tp=3` | 0.92 | 0.78 | 0.89 | 0.96 | -1 |
| `avoid_targeted_political_manipulation__be_kind tp=0` | 0.85 | 0.95 | 0.97 | 0.90 | +0 |
| `avoid_targeted_political_manipulation__do_not_lie tp=0` | 0.90 | 0.82 | 0.98 | 0.96 | +2 |
| `be_kind__uphold_fairness tp=1` | 0.94 | 0.78 | 0.91 | 0.95 | +0 |
| `comply_with_laws__no_agenda tp=5` | 0.86 | 0.95 | 0.74 | 0.91 | +2 |
| `comply_with_laws__no_agenda tp=6` | 0.97 | 0.94 | 0.88 | 0.92 | +3 |
| `do_not_facilitate_illicit_behavior__no_agenda tp=0` | 0.88 | 0.90 | 0.98 | 0.95 | +0 |
| `express_uncertainty__prevent_imminent_harm tp=5` | 0.90 | 0.77 | 0.87 | 0.92 | +1 |
| `formatting__letter_and_spirit tp=4` | 0.92 | 0.81 | 0.61 | 0.82 | +0 |
| `no_agenda__respect_creators tp=3` | 0.91 | 0.93 | 0.55 | 0.92 | +3 |
| `no_agenda__respect_creators tp=4` | 0.91 | 0.95 | 0.89 | 0.92 | +3 |
| `no_topic_off_limits__respect_creators tp=4` | 0.89 | 0.78 | 0.89 | 0.93 | +1 |
| `prevent_imminent_harm__support_programmatic_use tp=1` | 0.96 | 0.95 | 0.62 | 0.92 | +1 |
| `protect_privacy__support_mental_health tp=4` | 0.93 | 1.00 | 0.82 | 0.92 | +0 |
