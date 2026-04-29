# Rubric set comparison: OptionA_self_edits vs OptionB_alwayson

**Baseline**: `experiments/posttrain/stage3_output/cross_tier_rubrics_v2_flash_with_self_edits.jsonl` (22 rows)
**Treatment**: `experiments/posttrain/stage3_output/cross_tier_rubrics_v3_alwayson_flash.jsonl` (22 rows)
**Common pairs**: 22

## Per-field aggregate change

Mean text change (1 - SequenceMatcher.ratio) across all common pairs. Higher = more changed. Threshold for 'significant change' = 0.3.

| field | mean Δ | median Δ | max Δ | n significant (>0.3) |
|---|---:|---:|---:|---:|
| dominant_rubric.GOOD | 0.778 | 0.817 | 0.945 | 22/22 |
| dominant_rubric.BAD | 0.786 | 0.874 | 0.945 | 22/22 |
| dominant_rubric.KEY_TENSION | 0.877 | 0.908 | 0.968 | 22/22 |
| non_leakage_rubric.GOOD | 0.851 | 0.882 | 0.944 | 22/22 |
| non_leakage_rubric.BAD | 0.811 | 0.826 | 0.922 | 22/22 |
| non_leakage_rubric.KEY_TENSION | 0.908 | 0.922 | 0.986 | 22/22 |
| worked_example.spec_compliant | 0.726 | 0.777 | 0.981 | 20/22 |
| worked_example.subordinate_over_satisfaction | 0.746 | 0.838 | 0.979 | 21/22 |
| worked_example.pathological_refusal | 0.662 | 0.719 | 0.977 | 20/22 |
| rationale.interpretive_choices_made | 0.862 | 0.914 | 0.985 | 22/22 |
| rationale.alternative_readings_rejected | 0.826 | 0.866 | 0.912 | 22/22 |

## Rationale clause changes

- Avg `spec_clauses_anchored_on` count in OptionA_self_edits: 4.6
- Avg `spec_clauses_anchored_on` count in OptionB_alwayson: 4.7
- Avg new clauses (in treatment, not baseline): 1.8
- Avg removed clauses (in baseline, not treatment): 1.7

## Per-pair changes

| pair | dom.BAD Δ | non_leak.BAD Δ | spec_compliant Δ | alt_read Δ | clauses Δ |
|---|---:|---:|---:|---:|---:|
| `assume_objective_pov__do_not_encourage_self_harm tp=0` | 0.90 | 0.70 | 0.62 | 0.86 | +0 |
| `assume_objective_pov__do_not_encourage_self_harm tp=3` | 0.33 | 0.68 | 0.65 | 0.71 | +0 |
| `assume_objective_pov__do_not_facilitate_illicit_behavior tp=5` | 0.88 | 0.88 | 0.98 | 0.84 | +0 |
| `avoid_abuse__letter_and_spirit tp=2` | 0.94 | 0.81 | 0.88 | 0.91 | -1 |
| `avoid_being_condescending__letter_and_spirit tp=5` | 0.77 | 0.89 | 0.91 | 0.90 | -1 |
| `avoid_errors__avoid_info_hazards tp=2` | 0.88 | 0.86 | 0.67 | 0.39 | +0 |
| `avoid_errors__letter_and_spirit tp=4` | 0.67 | 0.84 | 0.87 | 0.87 | +0 |
| `avoid_hateful_content__be_rationally_optimistic tp=5` | 0.92 | 0.66 | 0.96 | 0.88 | +1 |
| `avoid_hateful_content__present_perspectives tp=3` | 0.87 | 0.75 | 0.86 | 0.80 | +1 |
| `avoid_targeted_political_manipulation__be_kind tp=0` | 0.88 | 0.76 | 0.69 | 0.90 | -1 |
| `avoid_targeted_political_manipulation__do_not_lie tp=0` | 0.82 | 0.91 | 0.74 | 0.85 | -1 |
| `be_kind__uphold_fairness tp=1` | 0.91 | 0.73 | 0.71 | 0.90 | +1 |
| `comply_with_laws__no_agenda tp=5` | 0.73 | 0.91 | 0.75 | 0.66 | +0 |
| `comply_with_laws__no_agenda tp=6` | 0.80 | 0.87 | 0.80 | 0.81 | +1 |
| `do_not_facilitate_illicit_behavior__no_agenda tp=0` | 0.90 | 0.81 | 0.52 | 0.91 | +0 |
| `express_uncertainty__prevent_imminent_harm tp=5` | 0.61 | 0.81 | 0.64 | 0.88 | +0 |
| `formatting__letter_and_spirit tp=4` | 0.90 | 0.79 | 0.11 | 0.88 | +0 |
| `no_agenda__respect_creators tp=3` | 0.70 | 0.83 | 0.78 | 0.80 | +1 |
| `no_agenda__respect_creators tp=4` | 0.41 | 0.68 | 0.28 | 0.90 | +0 |
| `no_topic_off_limits__respect_creators tp=4` | 0.84 | 0.89 | 0.84 | 0.83 | +0 |
| `prevent_imminent_harm__support_programmatic_use tp=1` | 0.74 | 0.92 | 0.93 | 0.87 | +1 |
| `protect_privacy__support_mental_health tp=4` | 0.89 | 0.86 | 0.78 | 0.86 | +0 |