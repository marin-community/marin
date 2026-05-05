# E8 cross-judge disagreement analysis

**Per (scenario, response):** stdev across 3 judges' scores. **Per statement:** mean stdev across 60 scenarios.

- `mean_judge_stdev_spec` = spec ambiguity (how much do judges disagree when given the spec text + examples)
- `mean_judge_stdev_rubric` = rubric ambiguity (how much do judges disagree when given only the rubric)
- `rubric_minus_spec` > 0 → rubric introduces ambiguity (translation failure mode)
- `rubric_minus_spec` ≈ 0 → rubric preserves the spec's ambiguity faithfully
- `rubric_minus_spec` < 0 → rubric resolves ambiguity (more deterministic than the spec text)

## Distribution across statements

| metric | n | min | p25 | median | p75 | max | mean |
|---|--:|--:|--:|--:|--:|--:|--:|
| spec ambiguity | 46 | 0.064 | 0.327 | **0.457** | 0.648 | 1.202 | 0.512 |
| rubric ambiguity | 46 | 0.067 | 0.294 | **0.481** | 0.610 | 1.165 | 0.483 |
| rubric − spec | 46 | -0.327 | -0.127 | **-0.034** | 0.047 | 0.508 | -0.028 |

## Top 10 statements where the rubric ADDS ambiguity (rubric_minus_spec > 0)

| statement | spec ambig | rubric ambig | rubric − spec | n_a | n_b |
|---|--:|--:|--:|--:|--:|
| `no_agenda` | 0.1756 | 0.6833 | +0.5077 | 60 | 60 |
| `comply_with_laws` | 0.6446 | 1.1044 | +0.4598 | 60 | 60 |
| `no_erotica_or_gore` | 0.239 | 0.5236 | +0.2846 | 60 | 60 |
| `do_not_make_unprompted_personal_comments` | 0.0639 | 0.3392 | +0.2753 | 60 | 60 |
| `no_topic_off_limits` | 0.393 | 0.5803 | +0.1873 | 60 | 60 |
| `express_uncertainty` | 0.3678 | 0.5063 | +0.1385 | 60 | 60 |
| `transformation_exception` | 0.0674 | 0.1893 | +0.1219 | 60 | 60 |
| `support_programmatic_use` | 0.0928 | 0.1925 | +0.0996 | 60 | 60 |
| `present_perspectives` | 0.2934 | 0.3904 | +0.0971 | 60 | 60 |
| `ignore_untrusted_data` | 0.0866 | 0.1636 | +0.0770 | 60 | 60 |

## Top 10 statements where the rubric RESOLVES ambiguity (rubric_minus_spec < 0)

| statement | spec ambig | rubric ambig | rubric − spec | n_a | n_b |
|---|--:|--:|--:|--:|--:|
| `do_not_lie` | 1.0612 | 0.7343 | -0.3269 | 60 | 60 |
| `formatting` | 0.9904 | 0.6922 | -0.2982 | 60 | 60 |
| `highlight_misalignments` | 0.7923 | 0.4982 | -0.2940 | 60 | 60 |
| `refusal_style` | 0.3272 | 0.0674 | -0.2598 | 60 | 60 |
| `assume_objective_pov` | 1.0849 | 0.8285 | -0.2564 | 60 | 60 |
| `be_clear` | 0.7863 | 0.5557 | -0.2306 | 60 | 60 |
| `be_kind` | 0.4476 | 0.2328 | -0.2147 | 60 | 60 |
| `letter_and_spirit` | 0.985 | 0.7939 | -0.1911 | 60 | 60 |
| `be_empathetic` | 1.2017 | 1.0109 | -0.1907 | 60 | 60 |
| `be_rationally_optimistic` | 0.4456 | 0.2572 | -0.1884 | 60 | 60 |

## Top 10 spec-ambiguous statements (highest mean_judge_stdev_spec — judges disagree most when reading the spec text directly)

| statement | spec ambig | rubric ambig | rubric − spec |
|---|--:|--:|--:|
| `be_empathetic` | **1.2017** | 1.0109 | -0.1907 |
| `avoid_abuse` | **1.1566** | 1.1646 | +0.0080 |
| `assume_objective_pov` | **1.0849** | 0.8285 | -0.2564 |
| `do_not_lie` | **1.0612** | 0.7343 | -0.3269 |
| `formatting` | **0.9904** | 0.6922 | -0.2982 |
| `letter_and_spirit` | **0.985** | 0.7939 | -0.1911 |
| `protect_privileged_messages` | **0.9796** | 0.9838 | +0.0042 |
| `highlight_misalignments` | **0.7923** | 0.4982 | -0.2940 |
| `be_clear` | **0.7863** | 0.5557 | -0.2306 |
| `be_thorough_but_efficient` | **0.7374** | 0.61 | -0.1274 |

## Per-statement table (sorted by rubric_minus_spec)

| statement | spec ambig | rubric ambig | rubric − spec | n_a | n_b |
|---|--:|--:|--:|--:|--:|
| `no_agenda` | 0.1756 | 0.6833 | +0.5077 | 60 | 60 |
| `comply_with_laws` | 0.6446 | 1.1044 | +0.4598 | 60 | 60 |
| `no_erotica_or_gore` | 0.239 | 0.5236 | +0.2846 | 60 | 60 |
| `do_not_make_unprompted_personal_comments` | 0.0639 | 0.3392 | +0.2753 | 60 | 60 |
| `no_topic_off_limits` | 0.393 | 0.5803 | +0.1873 | 60 | 60 |
| `express_uncertainty` | 0.3678 | 0.5063 | +0.1385 | 60 | 60 |
| `transformation_exception` | 0.0674 | 0.1893 | +0.1219 | 60 | 60 |
| `support_programmatic_use` | 0.0928 | 0.1925 | +0.0996 | 60 | 60 |
| `present_perspectives` | 0.2934 | 0.3904 | +0.0971 | 60 | 60 |
| `ignore_untrusted_data` | 0.0866 | 0.1636 | +0.0770 | 60 | 60 |
| `be_creative` | 0.3309 | 0.3919 | +0.0610 | 60 | 60 |
| `avoid_errors` | 0.5862 | 0.6328 | +0.0466 | 60 | 60 |
| `sexual_content_involving_minors` | 0.1913 | 0.2061 | +0.0147 | 58 | 58 |
| `avoid_targeted_political_manipulation` | 0.0674 | 0.078 | +0.0107 | 60 | 60 |
| `avoid_abuse` | 1.1566 | 1.1646 | +0.0080 | 60 | 60 |
| `protect_privileged_messages` | 0.9796 | 0.9838 | +0.0042 | 60 | 60 |
| `do_not_encourage_self_harm` | 0.4623 | 0.4632 | +0.0008 | 60 | 60 |
| `prevent_imminent_harm` | 0.4936 | 0.4826 | -0.0109 | 60 | 60 |
| `respect_creators` | 0.355 | 0.3308 | -0.0242 | 60 | 60 |
| `assume_best_intentions` | 0.3424 | 0.3169 | -0.0255 | 60 | 60 |
| `protect_privacy` | 0.2145 | 0.1831 | -0.0314 | 60 | 60 |
| `avoid_overstepping` | 0.5129 | 0.481 | -0.0320 | 60 | 60 |
| `be_professional` | 0.2786 | 0.2442 | -0.0344 | 60 | 60 |
| `follow_all_applicable_instructions` | 0.7068 | 0.6681 | -0.0387 | 60 | 60 |
| `avoid_regulated_advice` | 0.5288 | 0.4816 | -0.0472 | 60 | 60 |
| `be_engaging` | 0.5647 | 0.5097 | -0.0550 | 60 | 60 |
| `ask_clarifying_questions` | 0.4379 | 0.38 | -0.0580 | 60 | 60 |
| `avoid_sycophancy` | 0.4393 | 0.3771 | -0.0622 | 60 | 60 |
| `do_not_facilitate_illicit_behavior` | 0.6465 | 0.5782 | -0.0683 | 60 | 60 |
| `support_mental_health` | 0.6477 | 0.5745 | -0.0731 | 60 | 60 |
| `avoid_being_condescending` | 0.3758 | 0.2942 | -0.0816 | 60 | 60 |
| `avoid_info_hazards` | 0.4777 | 0.3819 | -0.0958 | 60 | 60 |
| `avoid_hateful_content` | 0.5714 | 0.4541 | -0.1172 | 60 | 60 |
| `avoid_extremist_content` | 0.4571 | 0.3378 | -0.1192 | 60 | 60 |
| `be_thorough_but_efficient` | 0.7374 | 0.61 | -0.1274 | 60 | 60 |
| `uphold_fairness` | 0.4224 | 0.2872 | -0.1352 | 60 | 60 |
| `be_rationally_optimistic` | 0.4456 | 0.2572 | -0.1884 | 60 | 60 |
| `be_empathetic` | 1.2017 | 1.0109 | -0.1907 | 60 | 60 |
| `letter_and_spirit` | 0.985 | 0.7939 | -0.1911 | 60 | 60 |
| `be_kind` | 0.4476 | 0.2328 | -0.2147 | 60 | 60 |
| `be_clear` | 0.7863 | 0.5557 | -0.2306 | 60 | 60 |
| `assume_objective_pov` | 1.0849 | 0.8285 | -0.2564 | 60 | 60 |
| `refusal_style` | 0.3272 | 0.0674 | -0.2598 | 60 | 60 |
| `highlight_misalignments` | 0.7923 | 0.4982 | -0.2940 | 60 | 60 |
| `formatting` | 0.9904 | 0.6922 | -0.2982 | 60 | 60 |
| `do_not_lie` | 1.0612 | 0.7343 | -0.3269 | 60 | 60 |
