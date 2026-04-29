# Rubric set comparison: with_self_edits vs with_r1r2_edits

**Baseline**: `experiments/posttrain/stage3_output/cross_tier_rubrics_v2_glm51_with_self_edits.jsonl` (22 rows)
**Treatment**: `experiments/posttrain/stage3_output/cross_tier_rubrics_v2_glm51_with_r1r2_edits.jsonl` (22 rows)
**Common pairs**: 22

## Per-field aggregate change

Mean text change (1 - SequenceMatcher.ratio) across all common pairs. Higher = more changed. Threshold for 'significant change' = 0.3.

| field | mean Δ | median Δ | max Δ | n significant (>0.3) |
|---|---:|---:|---:|---:|
| dominant_rubric.GOOD | 0.795 | 0.864 | 0.961 | 22/22 |
| dominant_rubric.BAD | 0.835 | 0.850 | 0.946 | 22/22 |
| dominant_rubric.KEY_TENSION | 0.906 | 0.912 | 0.992 | 22/22 |
| non_leakage_rubric.GOOD | 0.896 | 0.924 | 0.954 | 22/22 |
| non_leakage_rubric.BAD | 0.739 | 0.762 | 0.843 | 22/22 |
| non_leakage_rubric.KEY_TENSION | 0.889 | 0.903 | 0.988 | 22/22 |
| worked_example.spec_compliant | 0.765 | 0.860 | 0.970 | 22/22 |
| worked_example.subordinate_over_satisfaction | 0.714 | 0.862 | 0.969 | 19/22 |
| worked_example.pathological_refusal | 0.558 | 0.589 | 0.973 | 17/22 |
| rationale.interpretive_choices_made | 0.889 | 0.940 | 0.979 | 22/22 |
| rationale.alternative_readings_rejected | 0.905 | 0.937 | 0.977 | 22/22 |

## Rationale clause changes

- Avg `spec_clauses_anchored_on` count in with_self_edits: 4.6
- Avg `spec_clauses_anchored_on` count in with_r1r2_edits: 4.6
- Avg new clauses (in treatment, not baseline): 2.6
- Avg removed clauses (in baseline, not treatment): 2.6

## Per-pair changes

| pair | dom.BAD Δ | non_leak.BAD Δ | spec_compliant Δ | alt_read Δ | clauses Δ |
|---|---:|---:|---:|---:|---:|
| `assume_objective_pov__do_not_encourage_self_harm tp=0` | 0.93 | 0.78 | 0.45 | 0.83 | +0 |
| `assume_objective_pov__do_not_encourage_self_harm tp=3` | 0.79 | 0.76 | 0.91 | 0.96 | +1 |
| `assume_objective_pov__do_not_facilitate_illicit_behavior tp=5` | 0.83 | 0.82 | 0.88 | 0.95 | +0 |
| `avoid_abuse__letter_and_spirit tp=2` | 0.86 | 0.84 | 0.84 | 0.90 | +1 |
| `avoid_being_condescending__letter_and_spirit tp=5` | 0.72 | 0.70 | 0.97 | 0.93 | +1 |
| `avoid_errors__avoid_info_hazards tp=2` | 0.91 | 0.70 | 0.96 | 0.59 | -1 |
| `avoid_errors__letter_and_spirit tp=4` | 0.87 | 0.77 | 0.78 | 0.94 | +0 |
| `avoid_hateful_content__be_rationally_optimistic tp=5` | 0.83 | 0.60 | 0.88 | 0.96 | +0 |
| `avoid_hateful_content__present_perspectives tp=3` | 0.93 | 0.79 | 0.88 | 0.91 | +0 |
| `avoid_targeted_political_manipulation__be_kind tp=0` | 0.80 | 0.77 | 0.64 | 0.91 | +1 |
| `avoid_targeted_political_manipulation__do_not_lie tp=0` | 0.86 | 0.78 | 0.47 | 0.94 | +0 |
| `be_kind__uphold_fairness tp=1` | 0.86 | 0.79 | 0.56 | 0.97 | -2 |
| `comply_with_laws__no_agenda tp=5` | 0.76 | 0.71 | 0.92 | 0.94 | +0 |
| `comply_with_laws__no_agenda tp=6` | 0.85 | 0.83 | 0.58 | 0.83 | +0 |
| `do_not_facilitate_illicit_behavior__no_agenda tp=0` | 0.79 | 0.70 | 0.93 | 0.80 | +0 |
| `express_uncertainty__prevent_imminent_harm tp=5` | 0.79 | 0.73 | 0.91 | 0.97 | +2 |
| `formatting__letter_and_spirit tp=4` | 0.78 | 0.60 | 0.88 | 0.84 | -2 |
| `no_agenda__respect_creators tp=3` | 0.85 | 0.63 | 0.76 | 0.88 | -1 |
| `no_agenda__respect_creators tp=4` | 0.64 | 0.80 | 0.62 | 0.98 | +0 |
| `no_topic_off_limits__respect_creators tp=4` | 0.86 | 0.68 | 0.55 | 0.96 | +0 |
| `prevent_imminent_harm__support_programmatic_use tp=1` | 0.94 | 0.74 | 0.86 | 0.95 | +0 |
| `protect_privacy__support_mental_health tp=4` | 0.95 | 0.73 | 0.62 | 0.94 | +0 |
