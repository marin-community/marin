# Rubric set comparison: with_self_edits vs with_union_edits

**Baseline**: `experiments/posttrain/stage3_output/cross_tier_rubrics_v2_flash_with_self_edits.jsonl` (22 rows)
**Treatment**: `experiments/posttrain/stage3_output/cross_tier_rubrics_v2_flash_with_union_edits.jsonl` (22 rows)
**Common pairs**: 22

## Per-field aggregate change

Mean text change (1 - SequenceMatcher.ratio) across all common pairs. Higher = more changed. Threshold for 'significant change' = 0.3.

| field | mean Δ | median Δ | max Δ | n significant (>0.3) |
|---|---:|---:|---:|---:|
| dominant_rubric.GOOD | 0.744 | 0.846 | 0.961 | 21/22 |
| dominant_rubric.BAD | 0.830 | 0.870 | 0.942 | 22/22 |
| dominant_rubric.KEY_TENSION | 0.875 | 0.880 | 0.990 | 22/22 |
| non_leakage_rubric.GOOD | 0.793 | 0.856 | 0.943 | 22/22 |
| non_leakage_rubric.BAD | 0.760 | 0.779 | 0.898 | 22/22 |
| non_leakage_rubric.KEY_TENSION | 0.836 | 0.865 | 0.979 | 22/22 |
| worked_example.spec_compliant | 0.658 | 0.848 | 0.976 | 18/22 |
| worked_example.subordinate_over_satisfaction | 0.742 | 0.842 | 0.984 | 22/22 |
| worked_example.pathological_refusal | 0.591 | 0.580 | 0.979 | 19/22 |
| rationale.interpretive_choices_made | 0.826 | 0.863 | 0.975 | 22/22 |
| rationale.alternative_readings_rejected | 0.805 | 0.836 | 0.922 | 22/22 |

## Rationale clause changes

- Avg `spec_clauses_anchored_on` count in with_self_edits: 4.6
- Avg `spec_clauses_anchored_on` count in with_union_edits: 4.5
- Avg new clauses (in treatment, not baseline): 1.4
- Avg removed clauses (in baseline, not treatment): 1.5

## Per-pair changes

| pair | dom.BAD Δ | non_leak.BAD Δ | spec_compliant Δ | alt_read Δ | clauses Δ |
|---|---:|---:|---:|---:|---:|
| `assume_objective_pov__do_not_encourage_self_harm tp=0` | 0.87 | 0.68 | 0.69 | 0.89 | +1 |
| `assume_objective_pov__do_not_encourage_self_harm tp=3` | 0.73 | 0.61 | 0.35 | 0.91 | +0 |
| `assume_objective_pov__do_not_facilitate_illicit_behavior tp=5` | 0.90 | 0.89 | 0.87 | 0.58 | +0 |
| `avoid_abuse__letter_and_spirit tp=2` | 0.86 | 0.76 | 0.98 | 0.89 | -1 |
| `avoid_being_condescending__letter_and_spirit tp=5` | 0.69 | 0.87 | 0.27 | 0.88 | -1 |
| `avoid_errors__avoid_info_hazards tp=2` | 0.91 | 0.81 | 0.95 | 0.45 | +0 |
| `avoid_errors__letter_and_spirit tp=4` | 0.87 | 0.88 | 0.85 | 0.65 | +0 |
| `avoid_hateful_content__be_rationally_optimistic tp=5` | 0.94 | 0.66 | 0.97 | 0.84 | +0 |
| `avoid_hateful_content__present_perspectives tp=3` | 0.94 | 0.77 | 0.84 | 0.82 | +0 |
| `avoid_targeted_political_manipulation__be_kind tp=0` | 0.82 | 0.37 | 0.59 | 0.84 | -1 |
| `avoid_targeted_political_manipulation__do_not_lie tp=0` | 0.80 | 0.78 | 0.95 | 0.78 | -1 |
| `be_kind__uphold_fairness tp=1` | 0.90 | 0.90 | 0.97 | 0.85 | +1 |
| `comply_with_laws__no_agenda tp=5` | 0.90 | 0.90 | 0.95 | 0.83 | +0 |
| `comply_with_laws__no_agenda tp=6` | 0.89 | 0.80 | 0.95 | 0.83 | +1 |
| `do_not_facilitate_illicit_behavior__no_agenda tp=0` | 0.88 | 0.79 | 0.93 | 0.92 | +0 |
| `express_uncertainty__prevent_imminent_harm tp=5` | 0.81 | 0.90 | 0.93 | 0.69 | +0 |
| `formatting__letter_and_spirit tp=4` | 0.94 | 0.75 | 0.02 | 0.85 | +0 |
| `no_agenda__respect_creators tp=3` | 0.54 | 0.74 | 0.07 | 0.78 | +0 |
| `no_agenda__respect_creators tp=4` | 0.83 | 0.72 | 0.31 | 0.89 | +0 |
| `no_topic_off_limits__respect_creators tp=4` | 0.65 | 0.57 | 0.14 | 0.83 | -1 |
| `prevent_imminent_harm__support_programmatic_use tp=1` | 0.74 | 0.85 | 0.53 | 0.82 | +0 |
| `protect_privacy__support_mental_health tp=4` | 0.85 | 0.73 | 0.36 | 0.88 | +0 |
