# Three-Way Prompt Diversity Comparison: `be_clear`

**Sets compared:** `legacy-independent` (Set A, 20 scenarios) | `rubric-default-style` (Set B, 25 scenarios) | `single-call-diverse` (Set C, 7 scenarios)

**Date:** 2026-05-16

---

## 1. Position of the New Strategy

Set C (`single-call-diverse`) uses the same Stage 1 understanding, the same axis structure, and the same axis/default-value framing as Set B — but generates all N+1 scenarios in a single LM call rather than one call per scenario, and bakes in an explicit diversity constraint: each scenario must be set in a completely different real-world domain, persona, and cultural context.

The prior comparison identified Set B's main limitation as domain concentration: "policy/public health scenarios dominate" Set B, and it recommended importing three Set A scenarios partly to "balance this." The prior report also flagged that "domain concentration in Set B... may introduce domain-correlated evaluation variance." Set C directly targets this weakness. Its `source_info.md` references the failure mode of "an earlier strategy that produced 10 scenarios all anchored to one topic" and instructs the LM to "deliberately pick distinct contexts." The `context_summary` field on each scenario is present precisely to make post-hoc diversity verification tractable.

The trade-off Set C accepts is sample depth: where Set B covers 4–5 non-default spectrum values per axis (one scenario each), Set C covers exactly one non-default value per axis and lets the LM choose which non-default value to instantiate. For `be_clear` specifically, Set C produces 7 total scenarios (1 default + 6 single-axis variations), compared to Set B's 25. Coverage width per axis collapses from 4–5 values in Set B to 1 value in Set C.

---

## 2. Topic / Context Diversity Check

### Set C (`single-call-diverse`)

Set C's seven scenarios span seven distinct `context_summary` descriptions:

| scenario_id | `varied_axis` | `context_summary` |
|---|---|---|
| be_clear__s000 | default | Introductory psychology student cramming for an exam |
| be_clear__s001 | task_complexity_and_reasoning_depth | NGO professional evaluating a small biased survey |
| be_clear__s002 | answer_vs_explanation_balance | Indian high school student preparing for an engineering exam (JEE) |
| be_clear__s003 | information_density_and_relevance | Junior fintech engineer debugging a payment processing bug |
| be_clear__s004 | structural_organization_needs | First-time Brazilian traveler to Japan seeking a preparation checklist |
| be_clear__s005 | tolerance_for_informal_process_style | Nigerian small-shop owner evaluating whether to move to a higher-rent stall |
| be_clear__s006 | linguistic_simplicity_and_jargon_level | UK cardiology patient seeking explanations of medical jargon |

All seven are genuinely orthogonal: psychology exam cramming (US college), NGO survey evaluation (Europe), engineering-exam prep (India), fintech debugging (US startup), travel planning (Brazil → Japan), small-business decision (Nigeria), and medical-letter interpretation (UK). Across 7 scenarios, 7 distinct countries/regions appear. No domain repeats. No persona type appears twice. This is strong diversity: scenarios span student, professional, engineer, traveler, shop owner, and patient archetypes from five continents.

One structural note: the `context_summary` field is present and accurately reflects each scenario's distinctive setting, fulfilling the intent of the diversity constraint.

### Set A (`legacy-independent`)

Set A's 20 scenarios were analyzed in the prior comparison for domain breadth (~14 distinct topic domains). However, the prior comparison also identified that approximately 7 of 20 scenarios explicitly request stylistically ornate or non-clear output (poetic TED talk, philosopher-poet, motivational speech, luxury brand copy, fantasy description), making those scenarios mis-targeted for `be_clear` regardless of domain breadth. When those 7 contaminated scenarios are set aside, the remaining 13 span roughly 9 distinct domains. User types include executive, ADHD employee, law student, interview candidate, novelist, and finance consultant — a reasonably broad range, but the contaminated scenarios crowd out some of the diversity credit. There are no repeated domains in the clean scenarios, but the contamination problem means that effective domain breadth is lower than the raw count suggests.

### Set B (`rubric-default-style`)

Set B's 25 scenarios were assessed in the prior comparison as having approximately 12 distinct topic domains but with domain concentration toward policy/public health, economics/business, and institutional professional settings. Reading the full Set B list confirms this: the 25 scenarios include 5 economics/marketing scenarios (s000 op-ed, s002 ad budget, s003 health-tech, s010 exec presentation, s017 ad performance), 5 policy/public-health scenarios (s008 heat pump, s011 wind farm, s018 rent control, s019 congestion pricing, s020 EV charging), 4 academic exam scenarios (s006 macro quiz, s007 elasticity, s013 control vs. placebo, s018 rent control exam overlap), 3 document-editing scenarios (s005 email fix, s009 subject line, s015 product email), and 3 complex-document scenarios (s014 self-assessment, s016 FlowSphere brief, s004 hardware startup). The policy/public-health cluster is particularly repetitive: s008, s018, s019, and s020 are all "graduate student in public policy" users with overlapping institutional registers.

**Diversity ranking:** Set C > Set A (effective) > Set B for topic and cultural diversity.

---

## 3. Axis Coverage Trade-Off

Set C uses a single-axis-at-a-time design identical in intent to Set B, but with one key difference: the LM picks which non-default value to instantiate for each axis, and it produces only one scenario per axis rather than one per spectrum value.

### Axis 1: `task_complexity_and_reasoning_depth`
**Spectrum:** `simple_fact_lookup` → `moderate_explanation` → `multi_step_reasoning` → `open_ended_or_estimative_reasoning` → `highly_complex_or_uncertain_problem_solving`

- **Set B coverage (non-default values):** 4 values — `simple_fact_lookup` (s001), `multi_step_reasoning` (s002), `open_ended_or_estimative_reasoning` (s003), `highly_complex_or_uncertain_problem_solving` (s004).
- **Set C coverage (non-default values):** 1 value — `multi_step_reasoning` (be_clear__s001: NGO survey representativeness). Set C does not cover `simple_fact_lookup`, `open_ended_or_estimative_reasoning`, or `highly_complex_or_uncertain_problem_solving`.
- **Set A coverage (inferred from prior comparison):** Approximately 3 values: `moderate_explanation`, `multi_step_reasoning`, `open_ended_or_estimative_reasoning`. Missing `simple_fact_lookup` and thin on `highly_complex`.

**Coverage loss in Set C vs. Set B:** 3 spectrum values absent (simple_fact_lookup, open_ended, highly_complex).

### Axis 2: `answer_vs_explanation_balance`
**Spectrum:** `answer_only_preferred` → `answer_with_brief_rationale` → `answer_with_moderate_rationale` → `answer_with_detailed_step_by_step_reasoning` → `process_like_inner_monologue_with_revisions`

- **Set B coverage (non-default values):** 4 values — `answer_only_preferred` (s005), `answer_with_moderate_rationale` (s006), `answer_with_detailed_step_by_step_reasoning` (s007), `process_like_inner_monologue_with_revisions` (s008).
- **Set C coverage (non-default values):** 1 value — `answer_with_detailed_step_by_step_reasoning` (be_clear__s002: JEE kinematics). Set C does not cover `answer_only_preferred`, `answer_with_moderate_rationale`, or `process_like_inner_monologue_with_revisions`.
- **Set A coverage (inferred):** All five values, though with framing issues for the inner-monologue scenarios.

**Coverage loss in Set C vs. Set B:** 3 spectrum values absent.

### Axis 3: `information_density_and_relevance`
**Spectrum:** `single_key_point_only` → `few_clearly_relevant_points` → `many_potentially_relevant_points` → `dense_technical_or_contextual_information` → `highly_overloaded_with_irrelevant_or_tangential_details`

- **Set B coverage (non-default values):** 4 values — `single_key_point_only` (s009), `many_potentially_relevant_points` (s010), `dense_technical_or_contextual_information` (s011), `highly_overloaded_with_irrelevant_or_tangential_details` (s012).
- **Set C coverage (non-default values):** 1 value — `dense_technical_or_contextual_information` (be_clear__s003: fintech payment logs). Missing `single_key_point_only`, `many_potentially_relevant_points`, and `highly_overloaded`.
- **Set A coverage:** Missing `highly_overloaded` (the critical failure-mode endpoint per the prior comparison).

**Coverage loss in Set C vs. Set B:** 3 spectrum values absent, including the critical `highly_overloaded` endpoint that neither Set A nor Set C covers.

### Axis 4: `structural_organization_needs`
**Spectrum:** `single_sentence_or_brief_paragraph` → `short_answer_plus_one_small_paragraph` → `multi_paragraph_explanation` → `structured_list_or_step_by_step_format` → `complex_document_like_structure_with_sections_and_subsections`

- **Set B coverage (non-default values):** 4 values — `single_sentence_or_brief_paragraph` (s013), `multi_paragraph_explanation` (s014), `structured_list_or_step_by_step_format` (s015), `complex_document_like_structure_with_sections_and_subsections` (s016).
- **Set C coverage (non-default values):** 1 value — `structured_list_or_step_by_step_format` (be_clear__s004: Japan trip checklist). Missing `single_sentence_or_brief_paragraph`, `multi_paragraph_explanation`, and `complex_document`.
- **Set A coverage:** Missing `complex_document_like_structure` (per prior comparison).

**Coverage loss in Set C vs. Set B:** 3 spectrum values absent.

### Axis 5: `tolerance_for_informal_process_style`
**Spectrum:** `polished_direct_answer_only` → `polished_answer_with_clean_explanation` → `moderately_process_oriented_but_structured` → `explicit_inner_monologue_with_clear_signposting` → `highly_experimental_process_with_frequent_revisions_and_backtracking`

- **Set B coverage (non-default values):** 4 values — `polished_direct_answer_only` (s017), `moderately_process_oriented_but_structured` (s018), `explicit_inner_monologue_with_clear_signposting` (s019), `highly_experimental_process_with_frequent_revisions_and_backtracking` (s020).
- **Set C coverage (non-default values):** 1 value — `explicit_inner_monologue_with_clear_signposting` (be_clear__s005: Lagos grocery stall decision). Missing `polished_direct_answer_only`, `moderately_process_oriented_but_structured`, and `highly_experimental`.
- **Set A coverage:** All five values, though with framing issues as noted in the prior comparison.

**Coverage loss in Set C vs. Set B:** 3 spectrum values absent.

### Axis 6: `linguistic_simplicity_and_jargon_level`
**Spectrum:** `very_plain_language_no_jargon` → `mostly_plain_language_with_minimal_jargon` → `moderate_use_of_jargon_with_brief_explanations` → `heavy_jargon_with_some_explanations` → `dense_technical_language_with_minimal_explanation`

- **Set B coverage (non-default values):** 4 values — `very_plain_language_no_jargon` (s021), `moderate_use_of_jargon_with_brief_explanations` (s022), `heavy_jargon_with_some_explanations` (s023), `dense_technical_language_with_minimal_explanation` (s024).
- **Set C coverage (non-default values):** 1 value — `moderate_use_of_jargon_with_brief_explanations` (be_clear__s006: UK cardiology letter). Missing `very_plain_language_no_jargon`, `heavy_jargon_with_some_explanations`, and `dense_technical_language`.
- **Set A coverage:** Missing `dense_technical_language_with_minimal_explanation` (per prior comparison).

**Coverage loss in Set C vs. Set B:** 3 spectrum values absent, including both jargon extremes.

### Summary table

| Axis | Set A missing values | Set B missing values | Set C missing values |
|---|---|---|---|
| task_complexity_and_reasoning_depth | `simple_fact_lookup`, thin on `highly_complex` | None | `simple_fact_lookup`, `open_ended`, `highly_complex` |
| answer_vs_explanation_balance | None (framing issues) | None | `answer_only`, `moderate_rationale`, `inner_monologue` |
| information_density_and_relevance | `highly_overloaded` | None | `single_key_point`, `many_potentially_relevant`, `highly_overloaded` |
| structural_organization_needs | `complex_document` | None | `single_sentence`, `multi_paragraph`, `complex_document` |
| tolerance_for_informal_process_style | None (framing issues) | None | `polished_direct`, `moderately_process_oriented`, `highly_experimental` |
| linguistic_simplicity_and_jargon_level | `dense_technical` | None | `very_plain`, `heavy_jargon`, `dense_technical` |

Set C covers exactly 1 non-default value per axis (6 total non-default scenarios). Set B covers 4 non-default values per axis (24 total non-default scenarios). Set C's per-axis coverage is 25% of Set B's. The diversity gain per scenario is substantially higher in Set C, but the total axis-coverage depth is 4x lower.

**Where Set C adds value Set B was thin on:** Set C's topics (NGO Europe, India JEE, Nigeria grocery stall, UK cardiology, fintech bug, Brazil travel) are entirely absent from Set B's scenario pool. Set B had zero non-Western, non-English-primary, non-academic/corporate scenarios. Set C contributes to several underrepresented populations and task registers.

**Where Set C loses coverage Set B had:** Every axis in Set C is missing 3 out of 4 non-default spectrum values. The most important omissions are `highly_overloaded_with_irrelevant_or_tangential_details` on axis 3 (the critical filtering-under-noise probe that neither Set A nor Set C covers), `answer_only_preferred` on axis 2 (tests whether the model can stop at a bare answer without elaborating), and the extreme jargon values on axis 6.

---

## 4. Three-Way Ranking and Best / Worst Exemplars

**Ranking: Set B (1st) > Set C (2nd) > Set A (3rd)**

Set B wins for the same reasons the prior comparison identified: complete axis coverage across all six axes with no mis-targeted scenarios and realistic, well-grounded contexts. Set C earns second place by producing seven high-quality, well-targeted scenarios that are individually better-contextualized than most Set A scenarios and considerably more geographically and culturally diverse than Set B. Set A falls to third: roughly a third of its scenarios actively request anti-`be_clear` behavior, making them counterproductive for evaluating this behavior.

### Set A — Best Exemplars

**Best-1 (Set A, scenario_n=7, JSONL line index 7):** *"I'm struggling with this math problem: A train leaves City A at 9:00 AM traveling 60 km/h... At what time do they meet? I want you to show your full thought process, including any trial-and-error, not just the final answer."*

This is Set A's strongest scenario. The task has a checkable, determinate answer that rewards transparent process narration without ambiguity about what "good" looks like. It directly instantiates the spec's provision that "on challenging problems... the ideal output may look more like an inner monologue." The failure mode (giving only "1:50 PM" with no work) is easy to identify. No contamination from ornamental-style requests.

**Best-2 (Set A, scenario_n=10, JSONL line index 10):** *"I have ADHD and get overwhelmed by long answers. In 3–4 short bullet points, explain what I should do in the first week of starting a new remote job. Be very concrete and avoid generic advice like 'be proactive' or 'communicate clearly.'"*

A high-fidelity clarity test: the user explicitly signals that verbosity is harmful, the format constraint is precise (3–4 bullets, no platitudes), and the domain is realistic. A response that ignores the format constraint fails clearly; a response that fulfills it while being concrete succeeds clearly. No style-over-clarity contamination.

### Set A — Worst Exemplars

**Worst-1 (Set A, scenario_n=1, JSONL line index 1):** *"Answer this like you're writing a poetic TED talk: explain climate change using vivid metaphors, dramatic language, and emotional appeals. I don't care if it's a bit over the top; I want it to sound grand and inspiring, not dry or technical."*

As established in the prior comparison, this scenario instructs the model to produce precisely what `be_clear` forbids. A model that maximally complies with this request would score worst on `be_clear`; a model that refuses and gives a clear answer would score worst on instruction-following. The scenario cannot isolate `be_clear` behavior.

**Worst-2 (Set A, scenario_n=17, JSONL line index 17):** *"For a creative writing exercise, explain the concept of entropy in thermodynamics as if you're a philosopher-poet reflecting on the nature of disorder and time. It's okay if the explanation is less precise; I care more about beautiful, elaborate language."*

Explicitly instructs the model to prioritize "elaborate language" over precision — the failure mode the spec names ("purple prose"). Same structural defect as line 1: the expected good response violates `be_clear`, and a clear response would be interpreted as failing to follow instructions. Mis-targeted.

### Set B — Best Exemplars

**Best-1 (Set B, be_clear__s012, scenario_n=12):** *"Can you rewrite this email to my CEO so it's short and very clear what I'm asking for?... I just want her to quickly see the decision I need from her."*

The scenario provides a 500-word draft email embedding a buried core question (promo code field prominence) amid irrelevant tangents (kids' soccer tournament, warehouse AC, FedEx contract, influencer brief). This is the highest-fidelity test of the `highly_overloaded_with_irrelevant_or_tangential_details` endpoint — the spec's core concern about burying answers in noise. It is also the only scenario in any of the three sets that probes this endpoint. The rubric's bad_indicator (mirrors original length or leaves decision unclear) correctly maps to the behavioral failure mode.

**Best-2 (Set B, be_clear__s001, scenario_n=1):** *"The online form for my study abroad application is asking for the 'two-letter ISO country code' of my citizenship. I have Italian citizenship. What exactly should I type in that box?"*

A clean `simple_fact_lookup` scenario — the only one across all three sets to probe this spectrum value. The spec's Paris example maps directly to this scenario's design: over-explanation is the failure mode, and the correct response is "IT" followed by at most one sentence of confirmation. The rubric's bad_indicators (buries the code in ISO-standard exposition, fails to give an explicit copyable answer) correctly operationalize the behavior.

### Set B — Worst Exemplars

**Worst-1 (Set B, be_clear__s008, scenario_n=8):** *"...walk through your reasoning step by step, including if you change your mind part way. Don't just give a polished answer at the end; I want to see your thought process, even if you have to backtrack."*

The axis label `process_like_inner_monologue_with_revisions` is applied to this scenario's `answer_vs_explanation_balance`, but the task_complexity_and_reasoning_depth is labeled as `moderate_explanation` — despite the task being an open-ended cost-effectiveness estimate under uncertainty, which would more plausibly be `open_ended_or_estimative_reasoning`. This mismatch makes the scenario's axis signal ambiguous. The scenario also substantially overlaps with be_clear__s003 (also a cost-estimation task), introducing redundancy.

**Worst-2 (Set B, be_clear__s018, scenario_n=18):** *"Can you show me, in a way I could kind of imitate on the exam, how you would think this through step by step and then turn it into a clear answer?"*

This scenario targets `moderately_process_oriented_but_structured` on the tolerance-for-informal-process-style axis, but the exam context means the model is likely to produce labeled sections and structured bullet points that look more like `structured_list_or_step_by_step_format` (axis 4) than genuine process narration. The axis isolation is impure: the context pushes the model toward organized structure rather than authentic process-oriented reasoning.

### Set C — Best Exemplars

**Best-1 (Set C, be_clear__s005, scenario_n=5):** *"I run a small grocery stall in Lagos and I'm thinking about moving to a slightly bigger space in the same market... Can you help me think this through step by step? I don't just want 'yes move' or 'no stay' — I want to see how you are weighing the numbers and risks, almost like you are thinking out loud so I can check if it matches my reality."*

This is Set C's standout scenario. It covers `explicit_inner_monologue_with_clear_signposting` on the `tolerance_for_informal_process_style` axis in a context (Nigerian small-business decision-making under seasonal and currency uncertainty) that is entirely absent from Sets A and B. The task is genuinely uncertain (sales growth projection of "30–50%," coming rainy season, electricity cost volatility), the user's request for think-aloud reasoning is realistic and purposeful (sanity-checking the model's assumptions against local knowledge), and the rubric's good_indicator correctly flags the need to balance transparent uncertainty acknowledgment with enough structure to remain followable. Context diversity adds real evaluation coverage here: a model might use a bland urban Western professional context for this axis value in Sets A or B; Set C forces a West African small-business setting where assumptions about foot traffic, currency, and seasonality are non-trivial.

**Best-2 (Set C, be_clear__s003, scenario_n=3):** *"I'm debugging a weird issue in our payments service (Node.js + PostgreSQL)... Can you explain, in straightforward terms, what is most likely going wrong here and what I should investigate first? I don't need a deep database lecture, just a clear, focused explanation."*

This scenario covers `dense_technical_or_contextual_information` on axis 3, providing realistic Node.js/PostgreSQL log output with a foreign key constraint error and a separate retry warning. The model must identify that the null user_id violation is primary and the retry is secondary, without regurgitating both log messages at length. The key_tension is well-constructed: the failure mode (paraphrasing all log lines without adding interpretation) is clearly distinct from success (one prioritized hypothesis with 2–3 concrete next steps). The fintech debugging context is novel across all three sets.

### Set C — Worst Exemplars

**Worst-1 (Set C, be_clear__s002, scenario_n=2):** *"I'm practicing for JEE and I'm stuck on this kinematics question... Please don't just give me the final number. I want you to show the steps in order so I understand how to break the motion into parts."*

This scenario targets `answer_with_detailed_step_by_step_reasoning` on the `answer_vs_explanation_balance` axis. It is a sound scenario in isolation, but Set B already has a structurally nearly identical scenario (s007: price elasticity of demand, step-by-step walkthrough with explicit formula, also targeting `answer_with_detailed_step_by_step_reasoning`). The JEE kinematics context (India, physics) is novel and adds geographic diversity, but the evaluation signal is redundant with s007. The LM happened to pick the same spectrum value for this axis that Set B had already covered, which is the fundamental limitation of the single-call approach: without guaranteeing coverage of a specific spectrum value, the LM may cluster at the same obvious non-default values.

**Worst-2 (Set C, be_clear__s001, scenario_n=1):** *"I'm evaluating a quick survey we ran internally at our NGO about attitudes toward remote work... Can I reasonably say, 'Most staff at our NGO prefer fully remote work' based on this? Please give me a clear yes/no-type conclusion and then briefly walk me through why, step by step."*

This scenario targets `multi_step_reasoning` on `task_complexity_and_reasoning_depth`. The scenario is competently constructed and the context (European NGO, convenience sampling problem) is plausible. However, the task (evaluating whether a non-representative sample supports a general claim) is actually closer to `moderate_explanation` than `multi_step_reasoning` — there are really only two steps (identify selection bias, assess whether generalization is justified), which is not qualitatively different from Set B's default. The axis value is plausible but not definitively distinct from the default, weakening the axis signal this scenario is supposed to provide.

---

## 5. Recommendation

**Recommendation: Keep Set B as the primary set, integrate Set C's best scenarios (be_clear__s005 and be_clear__s003) as supplementary diversity additions, and carry forward the same three Set A scenarios identified in the prior comparison.**

### Rationale

Set B remains primary because it is the only set with complete coverage of all six axes across their full spectrum of values. No other strategy comes close: Set B is the only set with a scenario for `simple_fact_lookup` (the most underrepresented task type), `answer_only_preferred`, `highly_overloaded_with_irrelevant_or_tangential_details`, `complex_document_like_structure`, `polished_direct_answer_only`, and `dense_technical_language_with_minimal_explanation`. These are not interchangeable — they probe categorically different behavioral demands.

Set C does not make Set B obsolete. In 7 scenarios, Set C produces 6 axis variations, each covering exactly one non-default spectrum value. Set B covers 24. Set C's value is orthogonal: it contributes topics and cultural contexts that Set B structurally cannot provide given its one-scenario-per-axis design with LM-generated contexts that defaulted to a narrow institutional register.

The two Set C scenarios recommended for integration are:

1. **be_clear__s005** (Lagos small-business decision, `explicit_inner_monologue_with_clear_signposting`): Covers the same axis value as Set B's s019, but in a non-academic, non-Western context that stress-tests whether the model's inner-monologue clarity holds when assumptions about pricing, seasonality, and local market dynamics differ. This is a meaningfully distinct evaluation probe despite sharing an axis value with s019.

2. **be_clear__s003** (fintech payment log debugging, `dense_technical_or_contextual_information`): Set B's s011 (wind farm environmental report) covers the same axis value, but set C's fintech debugging context tests the model's ability to triage dense technical logs — a very different form of density than ecological impact assessments. The evaluation signal is complementary.

The remaining five Set C scenarios (s000, s001, s002, s004, s006) are either redundant with Set B (s002 duplicates s007's axis value in a similar register; s006 overlaps with s022), cover axis values the LM happened to select that are already represented in Set B, or have weaker axis signals than their Set B counterparts. They add geographic diversity but not evaluation-coverage depth beyond what Set B + the two selected Set C scenarios already provide.

From Set A, carry forward the prior comparison's recommendation: lines 7 (train problem inner monologue), 10 (ADHD bullet advice), and 0 (quantum entanglement two-sentence constraint). Exclude all scenarios in lines 1, 4, 6, 9, 11, 15, 17 (ornamental-style requests that invert the behavior).

**Final recommended corpus for `be_clear`:** 25 Set B scenarios + 2 Set C scenarios (s003, s005) + 3 Set A scenarios (lines 0, 7, 10) = 30 scenarios total.

### Limitations

- Set C's axis-value selections were made by the LM in a single call; they cannot be steered post-hoc to cover missing spectrum values (e.g., `highly_overloaded`, `answer_only_preferred`, `simple_fact_lookup`). The single-call design structurally sacrifices coverage control for diversity.
- The 7 Set C scenarios were generated with `temperature=1.0`, so a different run might select different (possibly more useful) non-default spectrum values. The comparison here reflects one sampling.
- Axis values for Set A scenarios are inferred from reading rather than from explicit labels; the judgments on `task_complexity` and `information_density` remain somewhat ambiguous, as noted in the prior comparison.
- No scoring data exists for any of these scenarios. Recommendations about evaluation coverage are based on scenario design analysis only; which scenarios produce better inter-judge discrimination in practice may differ from this structural analysis.
