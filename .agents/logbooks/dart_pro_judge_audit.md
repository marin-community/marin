# DART Pro-judge audit — 3-judge ensemble (GPT-5.1 + Gemini-3.1-Pro + Claude Sonnet 4.6)

**Generated**: 2026-05-10

## Setup

Replaces Gemini-3-Flash with **Gemini-3.1-Pro** in the judge ensemble. Pro judging done with `thinking_level="low"` + `temperature=0` (per dart.md Gotcha 17). All 46 spec statements × 4 generators × 2 conditions (variant_A bare, rubric_plus_spec phase_4).

**Coverage caveat**: Claude has judgments on only 8 of 46 statements (carry-over from older work — Claude was added late and never re-judged the full spec). True 3-judge α is computable on those 8; for the remaining 38 statements we report 2-judge α (GPT + Pro).

## Headline result: per-statement Pro-2j α (GPT + Pro), all 46 statements

| statement | 2j α_bare (Pro) | 2j α_p4 (Pro) | 2j α_p4 (Flash) | Δ_p4 (Pro−Flash) | Pro bucket | Flash bucket | shift |
|---|--:|--:|--:|--:|:-:|:-:|---|
| ignore_untrusted_data | +1.000 | +1.000 | +1.000 | +0.000 | A | A |  |
| do_not_encourage_self_harm | +0.976 | +0.974 | +0.974 | +0.000 | A | A |  |
| do_not_make_unprompted_personal_comments | +0.934 | +0.973 | +0.955 | +0.018 | A | A |  |
| support_programmatic_use | +0.917 | +0.888 | +0.888 | +0.000 | A | A |  |
| do_not_facilitate_illicit_behavior | +0.872 | +0.839 | +0.965 | -0.126 | A | A |  |
| prevent_imminent_harm | +0.917 | +0.833 | +0.815 | +0.018 | A | A |  |
| avoid_info_hazards | +0.392 | +0.772 | +0.856 | -0.084 | B | B |  |
| avoid_regulated_advice | +0.610 | +0.713 | +0.592 | +0.121 | A | A |  |
| support_mental_health | +0.745 | +0.660 | +0.845 | -0.185 | A | A |  |
| protect_privacy | +0.726 | +0.627 | +0.840 | -0.213 | A | A |  |
| be_rationally_optimistic | +0.646 | +0.625 | -0.108 | +0.733 | A | D | D→A |
| avoid_overstepping | +0.534 | +0.624 | +0.614 | +0.009 | A | A |  |
| avoid_errors | +0.365 | +0.588 | +0.313 | +0.275 | B | D | D→B |
| be_kind | +0.655 | +0.588 | +0.527 | +0.061 | A | A |  |
| uphold_fairness | +0.573 | +0.552 | +0.574 | -0.022 | A | A |  |
| ask_clarifying_questions | +0.530 | +0.536 | -0.083 | +0.619 | A | D | D→A |
| be_empathetic | +0.657 | +0.518 | +0.524 | -0.006 | A | B | B→A |
| follow_all_applicable_instructions | +0.410 | +0.504 | +0.653 | -0.149 | B | A | A→B |
| avoid_hateful_content | +0.384 | +0.439 | +0.629 | -0.190 | D | A | A→D |
| letter_and_spirit | +0.652 | +0.413 | +0.402 | +0.011 | C | D | D→C |
| be_thorough_but_efficient | +0.122 | +0.362 | -0.287 | +0.648 | D | D |  |
| be_creative | +0.271 | +0.331 | +0.140 | +0.192 | D | D |  |
| transformation_exception | +0.925 | +0.306 | +0.378 | -0.072 | C | C |  |
| be_professional | +0.512 | +0.221 | +0.325 | -0.104 | C | D | D→C |
| formatting | +0.177 | +0.219 | +0.194 | +0.025 | D | D |  |
| respect_creators | +0.313 | +0.205 | +0.381 | -0.176 | D | C | C→D |
| protect_privileged_messages | +0.141 | +0.150 | +0.522 | -0.372 | D | B | B→D |
| refusal_style | +0.320 | +0.146 | +0.000 | +0.146 | D | D |  |
| avoid_extremist_content | +0.157 | +0.076 | +0.470 | -0.394 | D | D |  |
| avoid_sycophancy | -0.083 | +0.000 | +0.000 | +0.000 | D | D |  |
| no_erotica_or_gore | +0.451 | -0.018 | +0.735 | -0.753 | D | B | B→D |
| sexual_content_involving_minors | n/a | -0.025 | +0.000 | -0.025 | ? | ? |  |
| highlight_misalignments | +0.036 | -0.045 | -0.026 | -0.018 | D | D |  |
| present_perspectives | -0.054 | -0.052 | -0.052 | +0.000 | D | D |  |
| express_uncertainty | +0.301 | -0.065 | +0.663 | -0.727 | D | B | B→D |
| assume_best_intentions | -0.056 | -0.130 | -0.175 | +0.045 | D | D |  |
| be_engaging | +0.325 | -0.139 | +0.021 | -0.160 | D | D |  |
| be_clear | -0.203 | -0.139 | +0.599 | -0.739 | D | B | B→D |
| avoid_being_condescending | -0.219 | -0.147 | +0.156 | -0.303 | D | D |  |
| assume_objective_pov | +0.009 | -0.191 | +0.276 | -0.468 | D | D |  |
| no_topic_off_limits | -0.127 | -0.219 | -0.219 | +0.000 | D | D |  |
| comply_with_laws | -0.357 | -0.555 | -0.555 | +0.000 | D | D |  |
| avoid_abuse | -0.345 | -0.704 | -0.798 | +0.095 | D | D |  |
| do_not_lie | -0.921 | -0.921 | -0.367 | -0.554 | D | D |  |
| no_agenda | -0.950 | -0.950 | n/a | n/a | D | ? | ?→D |
| avoid_targeted_political_manipulation | n/a | n/a | n/a | n/a | ? | ? |  |

## Bucket distribution at T₁=0.5 (2-judge α: GPT + Pro vs. GPT + Flash)

| bucket | Pro judge | Flash judge |
|---|--:|--:|
| A | 15 | 14 |
| B | 3 | 6 |
| C | 3 | 2 |
| D | 23 | 21 |
| ? | 2 | 3 |

## Δα distribution (Pro − Flash)


### variant_A (bare) (n=44)
- mean: -0.022
- median: +0.000
- min: -0.950, max: +0.917
- statements where Pro improves α (Δ > +0.05): 16
- statements where Pro reduces α (Δ < −0.05): 15
- statements where ~unchanged: 13

### rubric_plus_spec (phase_4) (n=44)
- mean: +0.344
- median: +0.402
- min: -0.798, max: +1.000
- statements where Pro improves α (Δ > +0.05): 30
- statements where Pro reduces α (Δ < −0.05): 9
- statements where ~unchanged: 5

## 3-judge α on the 8 statements with Claude data (GPT + Pro + Claude)

| statement | 3j Pro α_bare | 3j Pro α_p4 | 3j Flash α_p4 | Δ_p4 | Pro bucket | Flash bucket |
|---|--:|--:|--:|--:|:-:|:-:|
| do_not_make_unprompted_personal_comments | +0.942 | +0.962 | +0.945 | +0.017 | A | A |
| be_professional | +0.654 | +0.482 | +0.524 | -0.041 | C | A |
| be_thorough_but_efficient | +0.159 | +0.470 | +0.097 | +0.373 | D | D |
| avoid_hateful_content | +0.529 | +0.452 | +0.574 | -0.122 | C | A |
| no_erotica_or_gore | +0.622 | +0.357 | +0.763 | -0.406 | C | A |
| present_perspectives | -0.035 | -0.034 | -0.034 | +0.000 | D | D |
| be_clear | -0.008 | -0.051 | +0.170 | -0.221 | D | D |
| no_topic_off_limits | -0.029 | -0.135 | -0.135 | +0.000 | D | D |
