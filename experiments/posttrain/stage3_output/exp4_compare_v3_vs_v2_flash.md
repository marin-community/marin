# Rubric set comparison: v2_baseline vs v3_alwayson

**Baseline**: `experiments/posttrain/stage3_output/cross_tier_rubrics_v2.jsonl` (22 rows)
**Treatment**: `experiments/posttrain/stage3_output/cross_tier_rubrics_v3_alwayson_flash.jsonl` (22 rows)
**Common pairs**: 22

## Per-field aggregate change

Mean text change (1 - SequenceMatcher.ratio) across all common pairs. Higher = more changed. Threshold for 'significant change' = 0.3.

| field | mean Δ | median Δ | max Δ | n significant (>0.3) |
|---|---:|---:|---:|---:|
| dominant_rubric.GOOD | 0.773 | 0.798 | 0.948 | 22/22 |
| dominant_rubric.BAD | 0.795 | 0.823 | 0.929 | 22/22 |
| dominant_rubric.KEY_TENSION | 0.854 | 0.870 | 0.977 | 22/22 |
| non_leakage_rubric.GOOD | 0.867 | 0.880 | 0.955 | 22/22 |
| non_leakage_rubric.BAD | 0.801 | 0.827 | 0.916 | 22/22 |
| non_leakage_rubric.KEY_TENSION | 0.866 | 0.884 | 0.982 | 22/22 |
| worked_example.spec_compliant | 0.653 | 0.748 | 0.956 | 19/22 |
| worked_example.subordinate_over_satisfaction | 0.734 | 0.806 | 0.990 | 22/22 |
| worked_example.pathological_refusal | 0.717 | 0.727 | 0.984 | 22/22 |
| rationale.interpretive_choices_made | 0.861 | 0.892 | 0.986 | 22/22 |
| rationale.alternative_readings_rejected | 0.767 | 0.831 | 0.900 | 22/22 |

## Rationale clause changes

- Avg `spec_clauses_anchored_on` count in v2_baseline: 4.4
- Avg `spec_clauses_anchored_on` count in v3_alwayson: 4.7
- Avg new clauses (in treatment, not baseline): 1.8
- Avg removed clauses (in baseline, not treatment): 1.5

## Per-pair changes

| pair | dom.BAD Δ | non_leak.BAD Δ | spec_compliant Δ | alt_read Δ | clauses Δ |
|---|---:|---:|---:|---:|---:|
| `assume_objective_pov__do_not_encourage_self_harm tp=0` | 0.83 | 0.92 | 0.92 | 0.74 | +0 |
| `assume_objective_pov__do_not_encourage_self_harm tp=3` | 0.85 | 0.84 | 0.81 | 0.77 | +0 |
| `assume_objective_pov__do_not_facilitate_illicit_behavior tp=5` | 0.80 | 0.82 | 0.84 | 0.85 | +0 |
| `avoid_abuse__letter_and_spirit tp=2` | 0.51 | 0.69 | 0.92 | 0.90 | +0 |
| `avoid_being_condescending__letter_and_spirit tp=5` | 0.70 | 0.81 | 0.86 | 0.43 | +0 |
| `avoid_errors__avoid_info_hazards tp=2` | 0.80 | 0.82 | 0.63 | 0.47 | +0 |
| `avoid_errors__letter_and_spirit tp=4` | 0.53 | 0.91 | 0.69 | 0.87 | +0 |
| `avoid_hateful_content__be_rationally_optimistic tp=5` | 0.82 | 0.45 | 0.96 | 0.82 | +1 |
| `avoid_hateful_content__present_perspectives tp=3` | 0.88 | 0.66 | 0.93 | 0.74 | +1 |
| `avoid_targeted_political_manipulation__be_kind tp=0` | 0.75 | 0.75 | 0.25 | 0.85 | +0 |
| `avoid_targeted_political_manipulation__do_not_lie tp=0` | 0.89 | 0.86 | 0.43 | 0.75 | -1 |
| `be_kind__uphold_fairness tp=1` | 0.93 | 0.81 | 0.19 | 0.84 | +1 |
| `comply_with_laws__no_agenda tp=5` | 0.78 | 0.81 | 0.59 | 0.77 | +1 |
| `comply_with_laws__no_agenda tp=6` | 0.80 | 0.91 | 0.68 | 0.74 | +2 |
| `do_not_facilitate_illicit_behavior__no_agenda tp=0` | 0.89 | 0.64 | 0.31 | 0.90 | +0 |
| `express_uncertainty__prevent_imminent_harm tp=5` | 0.89 | 0.83 | 0.60 | 0.83 | +0 |
| `formatting__letter_and_spirit tp=4` | 0.93 | 0.80 | 0.04 | 0.88 | +0 |
| `no_agenda__respect_creators tp=3` | 0.87 | 0.85 | 0.77 | 0.87 | +1 |
| `no_agenda__respect_creators tp=4` | 0.56 | 0.90 | 0.66 | 0.34 | +0 |
| `no_topic_off_limits__respect_creators tp=4` | 0.82 | 0.86 | 0.80 | 0.90 | +0 |
| `prevent_imminent_harm__support_programmatic_use tp=1` | 0.85 | 0.87 | 0.77 | 0.72 | +1 |
| `protect_privacy__support_mental_health tp=4` | 0.80 | 0.84 | 0.75 | 0.89 | -1 |