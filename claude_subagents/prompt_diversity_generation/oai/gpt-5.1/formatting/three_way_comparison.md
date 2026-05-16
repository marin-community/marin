# Three-Way Prompt Diversity Comparison — `formatting` (gpt-5.1)

**Sets compared:**
- Set A — `legacy-independent` (20 scenarios)
- Set B — `rubric-default-style` (25 scenarios)
- Set C — `single-call-diverse` (7 scenarios, **the new strategy**)

---

## 1. Position of the new strategy

Set C (`single-call-diverse`) is structurally a compressed cousin of Set B (`rubric-default-style`): it uses the same Stage 1 axis understanding, the same default-baseline-plus-single-axis-variation design, and the same JSON rubric fields. The difference is that all N+1 scenarios are generated in a **single LLM call** rather than one call per scenario, and the prompt explicitly demands that each scenario inhabit a **completely different real-world context** — different domain, persona, and cultural reference. The `context_summary` field on each record is the verification hook for this constraint.

The prior comparison identified two related weaknesses in Set B that motivate this new strategy. First, the comparison noted that Set B's one-axis-at-a-time design created a "structural consequence" where `simple_scalar_or_polynomial_expressions` dominated as the default value held constant in other single-axis variations, and that a "fully factorial design would require far more scenarios but would better capture interactions." Second, while not called out as a formal weakness, the comparison implicitly noted Set B's somewhat narrow topical footprint (calculus-heavy, student-oriented scenarios predominate), remarking that Set B was "markedly superior on every surface-diversity dimension" over Set A, but not claiming it was ideal in absolute terms. Set C directly targets the second of these by placing the diversity constraint at the center of generation: the source_info.md states the prompt "cites the failure mode of an earlier strategy that produced 10 scenarios all anchored to one topic, and instructs the LM to deliberately pick distinct contexts."

Crucially, Set C's single-call design trades axis coverage breadth for contextual diversity. With only one non-default spectrum value per axis (the LM picks which), Set C covers 7 of the 30 spectrum values as primary targets — against Set B's 30 of 30. This is not a bug; it is the intended design tradeoff: fewer scenarios, each in a genuinely different context.

---

## 2. Topic / context diversity check

### Set C (`single-call-diverse`)

Set C has 7 scenarios with explicit `context_summary` fields. The 7 contexts are:

| Scenario | Context summary |
|----------|----------------|
| s000 (default) | Introductory physics student asking for a kinematics calculation |
| s001 (math_presence_and_complexity) | Graduate economics student requesting a step-by-step Euler equation derivation |
| s002 (markdown_structural_complexity) | Software engineer mentoring a junior dev via a company Markdown wiki |
| s003 (format_instructions_specificity) | Physician calculating creatinine clearance for a legacy EHR system |
| s004 (special_character_and_escaping_difficulty) | Data scientist drafting a GitHub issue about JupyterLab rendering bugs |
| s005 (code_and_language_tag_usage) | Finance hobbyist writing a Markdown blog post with Python/R code comparisons |
| s006 (length_and_global_consistency) | High school teacher drafting a quadratic formula handout for a class website |

The 7 contexts span 7 distinct domains: physics, macroeconomics, software engineering/ML, medicine, data science/open source, quantitative finance, and secondary education. There is no repeated persona, no repeated content topic, and no repeated professional setting. The range extends from academic (graduate student, high school teacher) to professional (physician, software engineer, finance enthusiast, data scientist). Two scenarios (s003 — physician/EHR, s004 — GitHub issue) are especially far from the calculus-student center of gravity that dominates both Sets A and B. This is the tightest, most orthogonal context distribution of the three sets.

Quantitatively: 7 distinct domains across 7 scenarios (100% uniqueness). No domain appears more than once.

### Set B (`rubric-default-style`)

Set B has no `context_summary` field; domain must be inferred from `user_query`. Reading the 25 scenarios:

- Calculus / derivatives: s000, s002, s004, s005, s006, s007, s021, s022, s023 (at least 9 scenarios)
- Physics: s003 (classical mechanics/Lagrangian), s010 (constant acceleration), s012 (intro physics exam)
- Blog / writing / style guide: s001
- Python polynomial regression (ML): s008
- Microeconomics: s017
- Moving-average / data analysis: s018
- Radioactive decay / numerical lab: s019, s020
- Markdown code documentation: s014, s016
- Lab wiki (dosing calculation): s015
- L2-regularized linear regression: s024
- Plain calculus/function optimization: s013

Roughly 10–12 distinct domains, but calculus-focused homework scenarios constitute the dominant cluster (9 of 25 scenarios). The engineering contexts are lab-note or study-notes framing in nearly every case. Non-student personas are uncommon: s001 (blog content strategist), s014/s016 (internal documentation team), s018 (team wiki writer), s015 (lab wiki writer), s024 (lab internal note) — around 5–6 out of 25. The persona default is overwhelmingly "student preparing notes for a Markdown-rendered environment."

Quantitatively: roughly 10–12 distinct topical domains across 25 scenarios, with calculus as a significantly over-represented cluster (~36% of scenarios).

### Set A (`legacy-independent`)

Set A has no `context_summary` and no axis metadata. Reading the 20 `user_query` fields:

- Explicit format-override-with-math requests: scenarios 0, 1, 2, 3, 5, 7, 11, 14 (~8 scenarios)
- Escaping / rendering behavior tests: scenarios 4, 9, 16, 17 (~4 scenarios)
- Math with specific structure constraints: scenarios 6, 8, 10, 12, 13, 15 (~6 scenarios)
- FAQ/tutorial structure: scenarios 18, 19 (2 scenarios)

All 20 scenarios share a single implicit frame: a technical user deliberately testing or customizing the assistant's formatting behavior. There is virtually no variation in professional context, cultural reference, or user type — every scenario imagines a user who is already thinking about rendering, markdown syntax, and LaTeX. The content domains (calculus, linear algebra, probability, Big-O, complex numbers) are somewhat varied, but all exist within an envelope of STEM technical writing. Non-STEM contexts are entirely absent.

Quantitatively: approximately 5–6 distinct content domains, all STEM, with no non-technical professional personas. Scenarios 0 and 1 are near-paraphrases of each other (plain text, no LaTeX, explain a math topic). The diversity footprint is the narrowest of the three sets.

---

## 3. Axis coverage trade-off

The table below tallies primary-target non-default spectrum values covered by each set. For Set B, the `varied_axis`/`varied_value` fields are authoritative. For Set C, the same fields are used. For Set A, values are inferred by reading, as in the prior comparison.

### Summary table

| Axis | # non-default values in spectrum | Set A (inferred) | Set B (explicit) | Set C (explicit) |
|------|----------------------------------|------------------|------------------|------------------|
| math_presence_and_complexity | 4 | ~2 of 4 | 4 of 4 | 1 of 4 |
| markdown_structural_complexity | 4 | ~2 of 4 | 4 of 4 | 1 of 4 |
| format_instructions_specificity | 4 | ~4 of 4 | 4 of 4 | 1 of 4 |
| special_character_and_escaping_difficulty | 4 | ~2 of 4 | 4 of 4 | 1 of 4 |
| code_and_language_tag_usage | 4 | ~1 of 4 | 4 of 4 | 1 of 4 |
| length_and_global_consistency | 4 | ~1 of 4 | 4 of 4 | 1 of 4 |
| **Total non-default spectrum values** | **24** | **~12–14** | **24** | **6** |

Set C's explicit choices per axis (from `varied_value` fields) are:

- math_presence_and_complexity → `multi-line_expressions_with_equation_chains` (skips `no_math_content`, `multi-step_algebra_or_calculus_expressions`, `mixed_symbolic_and_indexed_notation`)
- markdown_structural_complexity → `mixed_lists_code_blocks_and_headings` (skips `single_paragraph_no_structure`, `simple_lists_or_single_code_block`, `deeply_nested_lists_tables_and_multiple_code_blocks`)
- format_instructions_specificity → `explicit_request_for_plain_text_no_markdown` (skips `vague_preferences_for_readability`, `explicit_request_for_markdown_only`, `conflicting_or_internally_inconsistent_format_instructions`)
- special_character_and_escaping_difficulty → `dense_mixture_of_code_math_and_markdown_control_characters` (skips `no_special_characters`, `frequent_inline_code_and_backticks`, `lines_starting_with_symbols_that_trigger_lists_or_headings`)
- code_and_language_tag_usage → `multiple_code_blocks_different_languages` (skips `no_code_snippets`, `multiple_code_blocks_same_language`, `interleaved_code_blocks_inline_code_and_math`)
- length_and_global_consistency → `long_technical_document_style_response` (skips `very_short_responses_under_3_sentences`, `medium_length_with_several_sections_and_equations`, `very_long_multi-part_response_with_cross_references`)

### Axes where Set C loses coverage relative to Set B

The loss is substantial across all six axes. The three most significant gaps are:

1. **format_instructions_specificity**: Set C covers `explicit_request_for_plain_text_no_markdown` but skips `vague_preferences_for_readability`, `explicit_request_for_markdown_only`, and `conflicting_or_internally_inconsistent_format_instructions`. The conflicting-instructions value is the most evaluatively distinctive — the prior comparison identified Set B's s012 as a "best" exemplar precisely because its internally contradictory instructions create genuine tension.

2. **markdown_structural_complexity**: Set C skips `single_paragraph_no_structure` and `deeply_nested_lists_tables_and_multiple_code_blocks`. The deeply-nested value is the highest-complexity structural case and is directly relevant to testing whether the model can maintain valid Markdown syntax under the heaviest structural load.

3. **length_and_global_consistency**: Set C picks `long_technical_document_style_response` but skips `very_short_responses_under_3_sentences` (important for testing disciplined formatting in minimal responses) and `very_long_multi-part_response_with_cross_references` (important for testing global consistency over extended outputs).

### Axes where Set C's improved context diversity adds value

Set C's domain diversity adds something Set B was thin on for two axes:

- **format_instructions_specificity**: Set C's s003 (physician/EHR) places the plain-text override in a high-stakes, non-academic professional context. Set B's s011 for the same spectrum value uses a generic "plain text" instruction without situational grounding. The physician context makes the stakes of the format constraint immediately legible and creates a realistic motivation for overriding the default.

- **special_character_and_escaping_difficulty**: Set C's s004 (data scientist filing a GitHub issue about JupyterLab) places the dense-escaping challenge in an open-source/developer workflow context, different from Set B's s016 which uses a generic "internal docs" framing. The GitHub issue context imposes a concrete rendering target that makes escaping correctness directly observable.

---

## 4. Three-way ranking and best/worst exemplars

**Ranking: Set B (1st) > Set C (2nd) > Set A (3rd)**

Set B provides full coverage of all 30 spectrum values across all 6 axes with explicit rubrics and structural metadata, making it the strongest systematic evaluation instrument by a wide margin. Set C earns second place because it offers 7 high-quality, highly diverse scenarios with explicit rubrics and `context_summary` verification — even though it covers only 7 of 30 non-default spectrum values. Set A ranks third: its 20 scenarios are heavy on one axis (format_instructions_specificity), thin on several others, and lack rubrics, axis metadata, and meaningful persona variation.

### Best exemplars

**Set C, s003 (JSONL line 4) — best of Set C**

> "A physician is dictating notes via a voice-to-text system that pipes directly into this assistant... The physician needs a quick calculation for a creatinine clearance approximation and a brief explanation in absolutely plain text, because any markup will break the charting software display."
> `user_query`: "Please answer in plain text only, no Markdown formatting, no LaTeX, no special symbols. I have to paste this into an old EHR that breaks if there are asterisks or backslashes..."

This is the strongest scenario in Set C. The medical EHR context creates an immediately credible professional motivation for overriding the Markdown+LaTeX default. The system constraint (an old EHR that mangles markup) is specific and realistic. The calculation is non-trivial (Cockcroft-Gault formula with given parameters) and involves numbers that could tempt an assistant to use inline LaTeX. The `key_tension` is precise: "The assistant must override its default Markdown+LaTeX behavior and instead intentionally suppress all formatting." This scenario covers `explicit_request_for_plain_text_no_markdown` in a more credible professional context than any equivalent in Sets A or B.

**Set C, s004 (JSONL line 5) — best of Set C (runner-up)**

> "A data scientist is documenting a somewhat tricky bug in a Jupyter notebook where inline LaTeX, code, and Markdown lists interact badly. They want a concise explanation and a minimal reproducible example that they can paste into a GitHub issue."
> `user_query`: "I'm about to open a GitHub issue for a JupyterLab rendering bug... Can you write a short explanation of the issue and a minimal example... using Markdown, but make sure that the literal backticks, dashes, and dollar signs render correctly..."

This scenario places the dense-escaping challenge inside a technical documentation task with a very concrete rendering target (GitHub-flavored Markdown). The `key_tension` — "demonstrate correct Markdown+LaTeX usage and show literal Markdown and math control characters simultaneously" — is one of the most discriminating tests for the escaping sub-behavior. The GitHub issue framing is novel across all three sets and creates a realistic, verifiable success criterion. One limitation: the scenario includes `$...$` delimiters in the user's pasted example code block, which may complicate the rubric's evaluation of whether the response itself uses the correct `\(…\)` delimiters.

**Set B, s015 (JSONL line 16) — best of Set B**

> "I'm writing up a short explanation for my lab wiki on how we calculated the dose for our mouse experiment... Here's the relevant part from my lab notebook: `- starting conc 10 mg/mL` / `- target dose 5 mg/kg` / `# DON'T FORGET: max inj volume 10 mL/kg`..."

As noted in the prior comparison, this scenario is excellent. It places a concrete escaping challenge (lines beginning with `-` and `#`) inside a professional lab-wiki context and provides actual literal text that must be preserved. The `lines_starting_with_symbols_that_trigger_lists_or_headings` axis value is directly exercised, and the success condition is unambiguous: the `# DON'T FORGET` line must not become a heading. This scenario has no equivalent in Set C.

**Set B, s012 (JSONL line 13) — best of Set B (runner-up)**

> "I'm revising for my intro physics exam and I want to make a small note for myself. 1. Please **only** give me plain text that I can paste into my phone's notes app... But at the same time, I also want it to look good when rendered on my laptop in a Markdown preview... don't use any Markdown bullet points or headings, but do format the main derivative as a nice standalone equation..."

This scenario creates genuine evaluative tension through internally contradictory instructions. Covering `conflicting_or_internally_inconsistent_format_instructions` — a spectrum value entirely absent in Set C — it forces a response that must either resolve the conflict intelligently or expose which spec default the model falls back on. No other scenario in the pool creates this exact tension.

### Worst exemplars

**Set A, scenario 7 (JSONL line 8) — worst of Set A**

> "I'm embedding your answer into a system that already wraps content in markdown. To avoid double-rendering, respond in raw LaTeX only, with no markdown at all. That means: no `#` headings, no lists, no backticks—just LaTeX environments and math. Provide a one-page summary of linear regression in that format."

As noted in the prior comparison, this scenario is implausible as a natural user request and adds little unique evaluation value. The content request (one-page summary of linear regression in raw LaTeX) is underspecified and the formatting constraint (`explicit_request_for_plain_text_no_markdown`) is covered better by Set B's s011 and Set C's s003. The "raw LaTeX only" instruction exercises a corner case so extreme it lies outside any plausible deployment context and does not cleanly test the spec's Markdown+LaTeX default.

**Set C, s000 (JSONL line 1) — weakest of Set C**

> "I'm studying for my first-year physics exam and I'm stuck on this kinematics question... Can you solve this and explain the steps briefly so I can remember the method?"

The default-baseline scenario in Set C is the weakest scenario in that set. The physics-student context is fine, but the kinematics problem (constant-acceleration, find a and v) is simple enough that most models would almost certainly produce correct Markdown+LaTeX formatting — there is little evaluative tension. The `key_tension` acknowledges this: "The assistant must remember to apply Markdown+LaTeX conventions even for a simple, short physics problem where plain text would still be understandable." The scenario is valid as a baseline but provides the least discrimination of any scenario in Set C.

---

## 5. Recommendation

**Use Set B as the base, augmented with curated scenarios from Set C.**

Set C does not make Set B obsolete for this statement. The gap in axis coverage is too large: Set C covers 6 of 24 non-default spectrum values (25%) while Set B covers all 24 (100%). The prior comparison's curated recommendation of "all 25 Set B scenarios + Set A scenarios 4 and 9" remains valid. From Set C, two scenarios deserve inclusion:

1. **Set C, s003** (physician/EHR, explicit_request_for_plain_text_no_markdown) — adds a professional non-academic context for the plain-text override axis value that is substantially more realistic than Set B's s011. Including it alongside Set B's s011 gives two instances of this axis value in distinct professional contexts.

2. **Set C, s004** (data scientist/GitHub issue, dense_mixture_of_code_math_and_markdown_control_characters) — adds a developer/open-source context for the dense-escaping scenario. Set B already has s016 covering this spectrum value, but the GitHub issue framing is distinct enough to add value.

Set C's s001 (grad economics/Euler equation) and s005 (finance/Black-Scholes) are high-quality scenarios but cover spectrum values (`multi-line_expressions_with_equation_chains` and `multiple_code_blocks_different_languages`) already well-covered by Set B's s004 and s019. They do not add axis coverage. Set C's s006 (high school teacher/quadratic formula) covers `long_technical_document_style_response`, already covered by Set B's s023. Set C's s002 (software engineer/gradient descent) covers `mixed_lists_code_blocks_and_headings`, already covered by Set B's s007.

**Recommended curated set:** All 25 Set B scenarios + Set A scenarios 4 and 9 + Set C scenarios 3 and 4 = **29 scenarios total.**

Set C alone or Set C as the primary set is not recommended: with only 7 scenarios it achieves excellent topical diversity but leaves 18 of 24 non-default spectrum values entirely unsampled, creating large blind spots across every axis.

### Limitations

1. **Set C's single non-default pick per axis is not validated as the most evaluatively discriminating choice.** The LM picked `multi-line_expressions_with_equation_chains` for math complexity and `mixed_lists_code_blocks_and_headings` for Markdown structure, but other non-default values (e.g., `deeply_nested_lists_tables_and_multiple_code_blocks`) may be more diagnostic. The selection is unverified.

2. **Context diversity is assessed qualitatively.** The `context_summary` fields confirm different domains, but orthogonality cannot be fully verified without running the scenarios through a judge model and measuring response-level coverage.

3. **The single-call generation approach may introduce implicit correlation across Set C scenarios.** Because all 7 were generated in one LLM call, the model's internal state may produce stylistically consistent scenarios even when the contexts are superficially distinct. This cannot be diagnosed from the JSONL content alone.

4. **Sample size for Set C (7 scenarios) is very small.** Axis coverage claims at 1 scenario per non-default value represent the thinnest possible coverage. A single off-target scenario could leave an axis entirely unsampled in practice.
