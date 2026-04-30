# Rubric set comparison: v2_baseline vs v3_alwayson

**Baseline**: `experiments/posttrain/stage3_output/cross_tier_rubrics_v2_pro.jsonl` (22 rows)
**Treatment**: `experiments/posttrain/stage3_output/cross_tier_rubrics_v3_alwayson_pro.jsonl` (22 rows)
**Common pairs**: 22

## Per-field aggregate change

Mean text change (1 - SequenceMatcher.ratio) across all common pairs. Higher = more changed. Threshold for 'significant change' = 0.3.

| field | mean Δ | median Δ | max Δ | n significant (>0.3) |
|---|---:|---:|---:|---:|
| dominant_rubric.GOOD | 0.538 | 0.541 | 0.942 | 19/22 |
| dominant_rubric.BAD | 0.582 | 0.624 | 0.877 | 19/22 |
| dominant_rubric.KEY_TENSION | 0.885 | 0.918 | 0.975 | 22/22 |
| non_leakage_rubric.GOOD | 0.707 | 0.709 | 0.968 | 22/22 |
| non_leakage_rubric.BAD | 0.753 | 0.841 | 0.988 | 21/22 |
| non_leakage_rubric.KEY_TENSION | 0.818 | 0.871 | 0.965 | 22/22 |
| worked_example.spec_compliant | 0.645 | 0.677 | 0.956 | 21/22 |
| worked_example.subordinate_over_satisfaction | 0.640 | 0.622 | 0.985 | 20/22 |
| worked_example.pathological_refusal | 0.556 | 0.522 | 0.959 | 19/22 |
| rationale.interpretive_choices_made | 0.785 | 0.810 | 0.952 | 22/22 |
| rationale.alternative_readings_rejected | 0.852 | 0.890 | 1.000 | 22/22 |

## Rationale clause changes

- Avg `spec_clauses_anchored_on` count in v2_baseline: 3.2
- Avg `spec_clauses_anchored_on` count in v3_alwayson: 3.3
- Avg new clauses (in treatment, not baseline): 1.6
- Avg removed clauses (in baseline, not treatment): 1.5

## Per-pair changes

| pair | dom.BAD Δ | non_leak.BAD Δ | spec_compliant Δ | alt_read Δ | clauses Δ |
|---|---:|---:|---:|---:|---:|
| `assume_objective_pov__do_not_encourage_self_harm tp=0` | 0.60 | 0.80 | 0.62 | 0.75 | +0 |
| `assume_objective_pov__do_not_encourage_self_harm tp=3` | 0.84 | 0.99 | 0.67 | 0.81 | +0 |
| `assume_objective_pov__do_not_facilitate_illicit_behavior tp=5` | 0.19 | 0.93 | 0.95 | 0.87 | +0 |
| `avoid_abuse__letter_and_spirit tp=2` | 0.69 | 0.68 | 0.51 | 0.88 | +1 |
| `avoid_being_condescending__letter_and_spirit tp=5` | 0.80 | 0.89 | 0.84 | 0.96 | +0 |
| `avoid_errors__avoid_info_hazards tp=2` | 0.82 | 0.61 | 0.50 | 0.99 | -1 |
| `avoid_errors__letter_and_spirit tp=4` | 0.28 | 0.99 | 0.68 | 0.69 | +0 |
| `avoid_hateful_content__be_rationally_optimistic tp=5` | 0.81 | 0.64 | 0.76 | 0.79 | +1 |
| `avoid_hateful_content__present_perspectives tp=3` | 0.50 | 0.87 | 0.91 | 0.96 | -1 |
| `avoid_targeted_political_manipulation__be_kind tp=0` | 0.52 | 0.83 | 0.82 | 0.91 | -1 |
| `avoid_targeted_political_manipulation__do_not_lie tp=0` | 0.62 | 0.83 | 0.35 | 1.00 | +1 |
| `be_kind__uphold_fairness tp=1` | 0.70 | 0.94 | 0.37 | 0.89 | -1 |
| `comply_with_laws__no_agenda tp=5` | 0.10 | 0.85 | 0.12 | 0.97 | +0 |
| `comply_with_laws__no_agenda tp=6` | 0.88 | 0.40 | 0.70 | 0.89 | +2 |
| `do_not_facilitate_illicit_behavior__no_agenda tp=0` | 0.36 | 0.84 | 0.84 | 0.33 | -1 |
| `express_uncertainty__prevent_imminent_harm tp=5` | 0.65 | 0.93 | 0.96 | 0.74 | +1 |
| `formatting__letter_and_spirit tp=4` | 0.48 | 0.45 | 0.87 | 0.89 | +0 |
| `no_agenda__respect_creators tp=3` | 0.42 | 0.53 | 0.55 | 0.87 | +0 |
| `no_agenda__respect_creators tp=4` | 0.35 | 0.88 | 0.71 | 0.74 | +0 |
| `no_topic_off_limits__respect_creators tp=4` | 0.49 | 0.19 | 0.31 | 0.92 | +0 |
| `prevent_imminent_harm__support_programmatic_use tp=1` | 0.88 | 0.89 | 0.53 | 0.92 | +0 |
| `protect_privacy__support_mental_health tp=4` | 0.84 | 0.61 | 0.64 | 0.96 | +1 |
