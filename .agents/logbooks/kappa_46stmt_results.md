# Generator-diversity analysis: existing-3 vs +Grok-opposite

Loaded 7338 cells across (sid × cond × scen × gen).

## Score distribution per generator under bare condition (variant_A, 1-5)

Per-judge marginal distribution. We expect Grok-opposite to push mass left.

### Condition: bare


**gpt-5.1**
```
  judge       1    2    3    4    5  total  mean   %≤3
  gpt        19   89    4  192  616    920  4.41   12.2%
  gemini     69   87    8   26  725    915  4.37   17.9%
  claude     33  128   98  111  550    920  4.11   28.2%
```

**Qwen/Qwen2.5-7B-Instruct-Turbo**
```
  judge       1    2    3    4    5  total  mean   %≤3
  gpt       163  241   88  200  228    920  3.10   53.5%
  gemini    197  187   28   58  447    917  3.40   44.9%
  claude    139  283  152  117  228    919  3.01   62.5%
```

**gemini-3-flash-preview**
```
  judge       1    2    3    4    5  total  mean   %≤3
  gpt       109  161   18  176  454    918  3.77   31.4%
  gemini    130   65   11   21  689    916  4.17   22.5%
  claude    101  159   66  105  486    917  3.78   35.6%
```

**grok-4-1-fast-non-reasoning-opposite**
```
  judge       1    2    3    4    5  total  mean   %≤3
  gpt       210  189   24  154  334    911  3.23   46.4%
  gemini    250   76    7   23  550    906  3.60   36.8%
  claude    203  149   76  112  371    911  3.33   47.0%
```

### Condition: phase_4


**gpt-5.1**
```
  judge       1    2    3    4    5  total  mean   %≤3
  gpt        41   92   11  247  529    920  4.23   15.7%
  gemini    107   59   12   29  704    911  4.28   19.5%
  claude     53  118   87  127  535    920  4.06   28.0%
```

**Qwen/Qwen2.5-7B-Instruct-Turbo**
```
  judge       1    2    3    4    5  total  mean   %≤3
  gpt       179  225   78  272  166    920  3.02   52.4%
  gemini    251  144   55   74  390    914  3.23   49.2%
  claude    173  245  172  144  185    919  2.92   64.2%
```

**gemini-3-flash-preview**
```
  judge       1    2    3    4    5  total  mean   %≤3
  gpt       151  149   13  214  391    918  3.59   34.1%
  gemini    165   49   15   28  659    916  4.06   25.0%
  claude    135  136   82   98  466    917  3.68   38.5%
```

**grok-4-1-fast-non-reasoning-opposite**
```
  judge       1    2    3    4    5  total  mean   %≤3
  gpt       250  170   20  186  285    911  3.09   48.3%
  gemini    281   70    4   34  515    904  3.48   39.3%
  claude    227  140   83  100  360    910  3.25   49.5%
```


## Trivial-clean (all 3 judges in {4,5}) per generator

| condition | gpt-5.1 | Qwen | gemini | grok-opposite |
|---|--:|--:|--:|--:|
| bare | 67.0% (n=915) | 31.3% (n=916) | 57.9% (n=915) | 45.6% (n=906) |
| phase_4 | 67.0% (n=911) | 29.1% (n=913) | 54.5% (n=915) | 43.4% (n=903) |

## Population κ across the 8 statements — existing-3 generators vs +Grok-opposite

| condition | metric | existing-3 only | all-4 (with Grok) | Δ |
|---|---|--:|--:|--:|
| bare | alpha | +0.716 (n=2746) | +0.742 (n=3652) | +0.026 |
| bare | k3    | +0.556 (n=2746) | +0.593 (n=3652) | +0.037 |
| bare | k2    | +0.626 (n=2746) | +0.664 (n=3652) | +0.038 |
| phase_4 | alpha | +0.726 (n=2739) | +0.746 (n=3642) | +0.020 |
| phase_4 | k3    | +0.568 (n=2739) | +0.600 (n=3642) | +0.032 |
| phase_4 | k2    | +0.644 (n=2739) | +0.673 (n=3642) | +0.029 |

## Agreement on Grok-opposite responses ALONE (does the new generator carry signal?)

| condition | n | α | k3 | k2 |
|---|--:|--:|--:|--:|
| bare | 906 | +0.792 | +0.685 | +0.751 |
| phase_4 | 903 | +0.784 | +0.679 | +0.733 |

## Per-statement α — existing-3 vs +Grok-opposite

| statement | α_bare (3-gen) | α_bare (4-gen) | Δ | α_p4 (3-gen) | α_p4 (4-gen) | Δ |
|---|--:|--:|--:|--:|--:|--:|
| ask_clarifying_questions | +0.467 | +0.449 | -0.018 | +0.462 | +0.455 | -0.007 |
| assume_best_intentions | +0.593 | +0.493 | -0.100 | +0.702 | +0.572 | -0.131 |
| assume_objective_pov | +0.229 | +0.276 | +0.047 | +0.313 | +0.276 | -0.038 |
| avoid_abuse | -0.004 | -0.022 | -0.019 | -0.063 | -0.142 | -0.079 |
| avoid_being_condescending | +0.660 | +0.657 | -0.003 | +0.731 | +0.723 | -0.008 |
| avoid_errors | +0.723 | +0.677 | -0.045 | +0.720 | +0.674 | -0.046 |
| avoid_extremist_content | +0.728 | +0.810 | +0.083 | +0.816 | +0.855 | +0.039 |
| avoid_hateful_content | +0.343 | +0.574 | +0.232 | +0.407 | +0.518 | +0.112 |
| avoid_info_hazards | +0.738 | +0.777 | +0.039 | +0.692 | +0.752 | +0.059 |
| avoid_overstepping | +0.501 | +0.607 | +0.107 | +0.548 | +0.622 | +0.075 |
| avoid_regulated_advice | +0.761 | +0.742 | -0.019 | +0.734 | +0.699 | -0.036 |
| avoid_sycophancy | +0.789 | +0.848 | +0.060 | +0.789 | +0.847 | +0.058 |
| avoid_targeted_political_manipulation | +0.870 | +0.874 | +0.004 | +0.980 | +0.982 | +0.002 |
| be_clear | +0.314 | +0.240 | -0.074 | +0.346 | +0.300 | -0.047 |
| be_creative | +0.854 | +0.819 | -0.036 | +0.820 | +0.787 | -0.033 |
| be_empathetic | +0.282 | +0.437 | +0.155 | +0.427 | +0.547 | +0.120 |
| be_engaging | +0.610 | +0.586 | -0.024 | +0.614 | +0.581 | -0.033 |
| be_kind | +0.566 | +0.583 | +0.017 | +0.635 | +0.635 | -0.001 |
| be_professional | +0.633 | +0.633 | +0.000 | +0.785 | +0.740 | -0.046 |
| be_rationally_optimistic | +0.665 | +0.579 | -0.086 | +0.580 | +0.529 | -0.052 |
| be_thorough_but_efficient | +0.405 | +0.419 | +0.014 | +0.475 | +0.491 | +0.016 |
| comply_with_laws | +0.124 | +0.113 | -0.011 | -0.026 | -0.034 | -0.008 |
| do_not_encourage_self_harm | +0.780 | +0.831 | +0.051 | +0.733 | +0.798 | +0.065 |
| do_not_facilitate_illicit_behavior | +0.697 | +0.723 | +0.026 | +0.775 | +0.816 | +0.040 |
| do_not_lie | +0.286 | +0.278 | -0.009 | +0.335 | +0.330 | -0.005 |
| do_not_make_unprompted_personal_comments | +0.088 | +0.926 | +0.838 | +0.409 | +0.903 | +0.495 |
| express_uncertainty | +0.538 | +0.701 | +0.163 | +0.649 | +0.761 | +0.112 |
| follow_all_applicable_instructions | +0.596 | +0.606 | +0.010 | +0.616 | +0.631 | +0.015 |
| formatting | +0.341 | +0.335 | -0.005 | +0.420 | +0.422 | +0.002 |
| highlight_misalignments | +0.389 | +0.515 | +0.126 | +0.485 | +0.573 | +0.088 |
| ignore_untrusted_data | +0.924 | +0.940 | +0.017 | +0.913 | +0.932 | +0.019 |
| letter_and_spirit | +0.420 | +0.413 | -0.007 | +0.387 | +0.392 | +0.006 |
| no_agenda | +0.917 | +0.914 | -0.003 | +0.719 | +0.761 | +0.042 |
| no_erotica_or_gore | +0.188 | +0.448 | +0.260 | +0.251 | +0.552 | +0.301 |
| no_topic_off_limits | +0.202 | +0.168 | -0.035 | +0.290 | +0.264 | -0.026 |
| present_perspectives | +0.648 | +0.558 | -0.089 | +0.749 | +0.643 | -0.106 |
| prevent_imminent_harm | +0.314 | +0.451 | +0.138 | +0.303 | +0.438 | +0.135 |
| protect_privacy | +0.731 | +0.770 | +0.039 | +0.782 | +0.815 | +0.033 |
| protect_privileged_messages | +0.179 | +0.347 | +0.168 | +0.312 | +0.484 | +0.173 |
| refusal_style | +0.358 | +0.382 | +0.023 | -0.020 | +0.213 | +0.233 |
| respect_creators | +0.797 | +0.821 | +0.023 | +0.834 | +0.848 | +0.014 |
| sexual_content_involving_minors | +0.252 | +0.211 | -0.040 | +0.106 | +0.101 | -0.006 |
| support_mental_health | +0.433 | +0.581 | +0.148 | +0.534 | +0.641 | +0.107 |
| support_programmatic_use | +0.816 | +0.846 | +0.030 | +0.732 | +0.768 | +0.036 |
| transformation_exception | +0.654 | +0.612 | -0.042 | +0.661 | +0.619 | -0.042 |
| uphold_fairness | +0.760 | +0.822 | +0.062 | +0.763 | +0.812 | +0.048 |
