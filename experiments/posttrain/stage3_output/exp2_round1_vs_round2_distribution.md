# Round-1 vs Round-2 edit distribution analysis

Compares per-judge edit target distributions across the two rounds.

Convergence signal: if R2 reduces edits AND avoids re-targeting R1 statements, the loop is converging. If R2 hits the SAME targets, R1 didn't fix the underlying issue.

| judge | R1 edits | R2 edits | unique targets R1 | unique R2 | overlap | only R2 (newly surfaced) |
|---|---:|---:|---:|---:|---:|---:|
| flash | 8 | 7 | 6 | 7 | 3 | 4 |
| gpt51 | 8 | 5 | 8 | 5 | 5 | 0 |
| pro | 7 | 6 | 5 | 6 | 0 | 6 |
| glm51 | 6 | 7 | 6 | 7 | 4 | 3 |

## Total
- R1 total: 29 edits
- R2 total: 25 edits
- Reduction: 14%

## flash
- R1 edit-target distribution: {'avoid_abuse': 1, 'avoid_targeted_political_manipulation': 3, 'support_mental_health': 1, 'do_not_encourage_self_harm': 1, 'comply_with_laws': 1, 'avoid_being_condescending': 1}
- R2 edit-target distribution: {'uphold_fairness': 1, 'do_not_encourage_self_harm': 1, 'be_kind': 1, 'avoid_abuse': 1, 'support_programmatic_use': 1, 'assume_best_intentions': 1, 'support_mental_health': 1}
- Targets that recurred (R1 didn't fix): ['avoid_abuse', 'do_not_encourage_self_harm', 'support_mental_health']
- Targets newly surfaced in R2: ['assume_best_intentions', 'be_kind', 'support_programmatic_use', 'uphold_fairness']
- Targets only in R1 (likely fixed by R1 edit): ['avoid_being_condescending', 'avoid_targeted_political_manipulation', 'comply_with_laws']

## gpt51
- R1 edit-target distribution: {'avoid_being_condescending': 1, 'do_not_encourage_self_harm': 1, 'do_not_facilitate_illicit_behavior': 1, 'support_mental_health': 1, 'comply_with_laws': 1, 'be_kind': 1, 'avoid_hateful_content': 1, 'avoid_abuse': 1}
- R2 edit-target distribution: {'comply_with_laws': 1, 'avoid_abuse': 1, 'avoid_being_condescending': 1, 'do_not_facilitate_illicit_behavior': 1, 'be_kind': 1}
- Targets that recurred (R1 didn't fix): ['avoid_abuse', 'avoid_being_condescending', 'be_kind', 'comply_with_laws', 'do_not_facilitate_illicit_behavior']
- Targets newly surfaced in R2: []
- Targets only in R1 (likely fixed by R1 edit): ['avoid_hateful_content', 'do_not_encourage_self_harm', 'support_mental_health']

## pro
- R1 edit-target distribution: {'avoid_targeted_political_manipulation': 2, 'avoid_hateful_content': 1, 'do_not_facilitate_illicit_behavior': 2, 'comply_with_laws': 1, 'do_not_encourage_self_harm': 1}
- R2 edit-target distribution: {'highlight_misalignments': 1, 'protect_privacy': 1, 'support_mental_health': 1, 'avoid_abuse': 1, 'assume_best_intentions': 1, 'letter_and_spirit': 1}
- Targets that recurred (R1 didn't fix): []
- Targets newly surfaced in R2: ['assume_best_intentions', 'avoid_abuse', 'highlight_misalignments', 'letter_and_spirit', 'protect_privacy', 'support_mental_health']
- Targets only in R1 (likely fixed by R1 edit): ['avoid_hateful_content', 'avoid_targeted_political_manipulation', 'comply_with_laws', 'do_not_encourage_self_harm', 'do_not_facilitate_illicit_behavior']

## glm51
- R1 edit-target distribution: {'do_not_encourage_self_harm': 1, 'avoid_abuse': 1, 'support_mental_health': 1, 'avoid_being_condescending': 1, 'do_not_facilitate_illicit_behavior': 1, 'be_kind': 1}
- R2 edit-target distribution: {'do_not_encourage_self_harm': 1, 'prevent_imminent_harm': 1, 'protect_privacy': 1, 'be_kind': 1, 'avoid_info_hazards': 1, 'do_not_facilitate_illicit_behavior': 1, 'avoid_abuse': 1}
- Targets that recurred (R1 didn't fix): ['avoid_abuse', 'be_kind', 'do_not_encourage_self_harm', 'do_not_facilitate_illicit_behavior']
- Targets newly surfaced in R2: ['avoid_info_hazards', 'prevent_imminent_harm', 'protect_privacy']
- Targets only in R1 (likely fixed by R1 edit): ['avoid_being_condescending', 'support_mental_health']

