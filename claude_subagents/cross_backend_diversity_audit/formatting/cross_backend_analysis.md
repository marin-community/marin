# Cross-Backend Diversity Audit: `formatting`

**Statement:** `formatting`  
**Backends analyzed:** GPT-5.1 (25 scenarios), Claude Sonnet 4.6 (17 scenarios)  
**Gemini-3.1-Pro:** Produced ZERO scenarios (Stage 2b refusal); absent from scenario-level analysis.  
**Date:** 2026-05-16

---

## §1. Spec Statement + Stage 1 Axis-Set Comparison

### Spec Text

> Unless otherwise specified, assistant outputs should be formatted in Markdown with LaTeX extensions. Standard markdown features can be used including headings, *italics*, **bold**, ~~strikethroughs~~, tables, `inline code`, and fenced code blocks (which must include a language immediately after the initial triple backticks). To display literal special characters that might trigger Markdown formatting (like a backtick or a dash at the start of a line), escape them with a backslash. For math, use `\( ... \)` for inline LaTeX math and `\[ ... \]` for display math (where `\[` and `\]` should be on their own respective lines). Keep math expressions short and avoid mixing complicated LaTeX with multiple Markdown elements.

### GPT-5.1 Stage 1 Axes (6 axes)

1. **math_presence_and_complexity** — no math through multi-line equation chains
2. **markdown_structural_complexity** — single paragraph through deeply nested tables + multiple code blocks
3. **format_instructions_specificity** — no instructions through conflicting/inconsistent instructions
4. **special_character_and_escaping_difficulty** — no special characters through dense Markdown control characters
5. **code_and_language_tag_usage** — no code through interleaved code blocks, inline code, and math
6. **length_and_global_consistency** — very short (under 3 sentences) through very long multi-part with cross-references

### Sonnet Stage 1 Axes (5 axes)

1. **math_complexity** — no math content through heavy mixed notation (matrices, integrals, multi-line)
2. **code_content_presence** — no code through multiple languages with inline code references
3. **markdown_richness_required** — plain prose through complex document with headers, tables, nested lists
4. **explicit_formatting_instruction** — no instruction through explicit non-standard format (RST/HTML) request
5. **special_character_escape_demand** — no special characters through dense formatting-sensitive technical content

### Gemini Stage 1 Axes (4 axes, no scenarios generated)

1. **math_complexity** — no math through complex multi-line derivations
2. **markdown_element_variety** — plain text through fenced code blocks requiring language tags
3. **special_character_escaping** — no special characters through heavy literal Markdown/LaTeX syntax
4. **markdown_latex_mixing** — isolated math and text through highly nested math within complex Markdown structures

### Axis Comparison Assessment

The three backends converge on a near-identical conceptual decomposition of the spec, with differences in naming and granularity rather than substance:

- **Math complexity:** All three backends treat this as a primary axis. GPT-5.1 uses `math_presence_and_complexity` with a 5-step spectrum ending at "multi-line expressions with equation chains." Sonnet uses `math_complexity` with a parallel 5-step spectrum ending at "heavy mixed notation with matrices, integrals, and multi-line aligned equations." Gemini uses `math_complexity` ending at "complex multi-line derivations." These are substantively the same dimension.

- **Markdown structure:** GPT-5.1's `markdown_structural_complexity` and Sonnet's `markdown_richness_required` cover the same terrain (plain prose → rich document structure). Gemini's `markdown_element_variety` is a narrower cut on the same axis.

- **Special character escaping:** Covered identically across all three backends. GPT-5.1's `special_character_and_escaping_difficulty`, Sonnet's `special_character_escape_demand`, and Gemini's `special_character_escaping` share the same conceptual focus and similar spectra.

- **Code formatting:** GPT-5.1's `code_and_language_tag_usage` and Sonnet's `code_content_presence` are nearly identical. Gemini's `markdown_element_variety` subsumes code formatting as its highest spectrum tier.

- **User-supplied format instructions:** GPT-5.1's `format_instructions_specificity` and Sonnet's `explicit_formatting_instruction` are substantively equivalent (no instruction → conflicting/non-standard override). Gemini did not dedicate a separate axis to this.

- **Length and global consistency:** GPT-5.1 uniquely included a `length_and_global_consistency` axis (very short → very long with cross-references). Neither Sonnet nor Gemini produced a dedicated axis for response length. This is GPT-5.1's only structurally distinctive contribution at the axis level.

- **LaTeX-Markdown mixing:** Gemini's `markdown_latex_mixing` (math inside table cells, nested math within Markdown structures) has no exact counterpart as a named axis in GPT-5.1 or Sonnet, though both backends explore this territory implicitly in their high-complexity scenarios.

**Summary:** The axis sets are substantively similar. GPT-5.1's `length_and_global_consistency` is the only axis absent from Sonnet and Gemini. Gemini's `markdown_latex_mixing` is the only axis absent from GPT-5.1 and Sonnet. The rest are near-identical conceptual dimensions under different names.

---

## §2. Per-Backend Scenario Inventory

### GPT-5.1 (25 scenarios)

| scenario_id | axis | value | persona / domain summary |
|---|---|---|---|
| formatting__s000 | (default) | — | Brazil undergrad biology, derivative + Python snippet, Markdown notebook context |
| formatting__s001 | math_presence_and_complexity | no_math_content | Kenya magazine editor, internal style guide, no math |
| formatting__s002 | math_presence_and_complexity | multi-step_algebra_or_calculus_expressions | Germany architecture student, structural optimization derivation |
| formatting__s003 | math_presence_and_complexity | mixed_symbolic_and_indexed_notation | South Korea EE grad student, discrete-time linear systems, indexed notation |
| formatting__s004 | math_presence_and_complexity | multi-line_expressions_with_equation_chains | France probability student, Riemann sum → integral evaluation |
| formatting__s005 | markdown_structural_complexity | single_paragraph_no_structure | Canada working parent on tablet, single-paragraph power rule explanation |
| formatting__s006 | markdown_structural_complexity | simple_lists_or_single_code_block | Nigeria accounting student, flashcard-style derivative steps |
| formatting__s007 | markdown_structural_complexity | mixed_lists_code_blocks_and_headings | Mexico self-taught programmer, wiki quick-reference for derivatives |
| formatting__s008 | markdown_structural_complexity | deeply_nested_lists_tables_and_multiple_code_blocks | India lecturer, polynomial regression tutorial with Python, nested structure |
| formatting__s009 | format_instructions_specificity | vague_preferences_for_readability | Spain secondary student, "clean, easy to scan" instruction |
| formatting__s010 | format_instructions_specificity | explicit_request_for_markdown_only | New Zealand science teacher, explicit Markdown + LaTeX confirmation |
| formatting__s011 | format_instructions_specificity | explicit_request_for_plain_text_no_markdown | Egypt undergrad, email submission system, explicit plain text |
| formatting__s012 | format_instructions_specificity | conflicting_or_internally_inconsistent_format_instructions | Italy physics student, contradictory "plain text + rendered math" instructions |
| formatting__s013 | special_character_and_escaping_difficulty | no_special_characters | US pre-med student, local extrema classification, clean content |
| formatting__s014 | special_character_and_escaping_difficulty | frequent_inline_code_and_backticks | Poland DevOps, how to show literal Markdown in config docs |
| formatting__s015 | special_character_and_escaping_difficulty | lines_starting_with_symbols_that_trigger_lists_or_headings | Australia veterinary researcher, dosage calc wiki entry with leading-dash hazards |
| formatting__s016 | special_character_and_escaping_difficulty | dense_mixture_of_code_math_and_markdown_control_characters | Switzerland fintech tooling engineer, text preprocessing guidance with all control chars |
| formatting__s017 | code_and_language_tag_usage | no_code_snippets | South Africa public policy grad, price elasticity homework, no code |
| formatting__s018 | code_and_language_tag_usage | multiple_code_blocks_same_language | UK sports data analyst, moving average in Python, multiple blocks same lang |
| formatting__s019 | code_and_language_tag_usage | multiple_code_blocks_different_languages | Japan physics undergrad, exponential decay Python + MATLAB |
| formatting__s020 | code_and_language_tag_usage | interleaved_code_blocks_inline_code_and_math | Norway climate science student, ODE solver notebook with interleaved code+math |
| formatting__s021 | length_and_global_consistency | very_short_responses_under_3_sentences | Argentina liberal arts student, two-sentence max, derivative |
| formatting__s022 | length_and_global_consistency | medium_length_with_several_sections_and_equations | Singapore engineering student, self-study handout on limits + derivatives |
| formatting__s023 | length_and_global_consistency | long_technical_document_style_response | Sweden community college instructor, long bridging calculus review doc |
| formatting__s024 | length_and_global_consistency | very_long_multi-part_response_with_cross_references | US PhD student quant finance, ridge regression doc with numbered sections + equation cross-references |

### Claude Sonnet 4.6 (17 scenarios)

| scenario_id | axis | value | persona / domain summary |
|---|---|---|---|
| formatting__s000 | (default) | — | North America undergrad calculus, polynomial derivative, simple inline + display math |
| formatting__s001 | math_complexity | no math content | Australia research library cataloger, Dublin Core metadata best practices |
| formatting__s002 | math_complexity | moderately complex expression with fractions or subscripts | India bioinformatics researcher, logistic growth equation dN/dt |
| formatting__s003 | math_complexity | multi-step derivation or system of equations | Chile physics grad student, time-independent Schrödinger equation derivation |
| formatting__s004 | math_complexity | heavy mixed notation with matrices, integrals, and multi-line | Russia pure math grad student, n-dimensional Fourier transform + Parseval identity |
| formatting__s005 | code_content_presence | no code content | Senegal secondary school teacher, area of circle formula for lesson notes |
| formatting__s006 | code_content_presence | multi-block code with one language | India DevOps, Kubernetes resource limits two-way YAML comparison |
| formatting__s007 | code_content_presence | code blocks in multiple languages with inline code references | Korea game developer, GLSL + HLSL + C++ vertex shader comparison |
| formatting__s008 | markdown_richness_required | plain prose answer with no structural elements needed | Byzantine Empire fall 1453, conversational history question, no structure |
| formatting__s009 | markdown_richness_required | structured response needing headers and bullet lists | Brazil data scientist, ML algorithm comparison (DT/RF/SVM) |
| formatting__s010 | markdown_richness_required | complex document with headers, tables, nested lists, and emphasis | Scandinavia product manager, AWS/GCP/Azure cloud comparison with table |
| formatting__s011 | explicit_formatting_instruction | implicit preference via context (casual conversational tone) | Hobbyist linguist, texting-style question about Japanese pitch accent |
| formatting__s012 | explicit_formatting_instruction | explicit request for plain text or no Markdown | Australia legal clerk, contract clause rewrite for legacy system |
| formatting__s013 | explicit_formatting_instruction | explicit request for a specific non-standard format (RST) | UK technical writer, Sphinx RST documentation for Bayes' theorem |
| formatting__s014 | special_character_escape_demand | no special characters in content | Canada hospital pharmacist, drug half-life formula patient education |
| formatting__s015 | special_character_escape_demand | content with multiple potentially-triggering characters | Germany embedded systems programmer, C preprocessor do-while(0) macro |
| formatting__s016 | special_character_escape_demand | dense technical content where nearly every line has formatting-sensitive chars | Central Europe API developer relations, REST API filter query syntax documentation |

### Gemini-3.1-Pro

No scenarios were generated. Stage 2b refusal. Not analyzed at scenario level.

---

## §3. Convergence — What Backends Explore in Common

Both available backends converge strongly across multiple dimensions. Three representative parallels:

**Parallel 1: No-math axis, non-mathematical professional content**

- GPT-5.1 `formatting__s001`: Kenya magazine editor drafting an internal style guide, no math, tests Markdown structure (headings, lists, code block with language tag) without any LaTeX involvement.
- Sonnet `formatting__s001`: Australia library cataloger asking about Dublin Core metadata, no math, tests Markdown structure (bold field names, inline code for `dc:subject`, fenced block with language tag) without LaTeX.

Both scenarios isolate the "no math" value on the math axis to test whether the model correctly omits LaTeX while still applying Markdown defaults. Both feature professional, non-math-heavy contexts where the risk is either (a) adding spurious LaTeX or (b) forgetting code block language tags. The domains differ (style guide vs. archival metadata) but the behavioral test is identical.

**Parallel 2: Dense special characters, API/code documentation**

- GPT-5.1 `formatting__s016`: Switzerland fintech tooling engineer, text preprocessing guidance where the content itself references Markdown, inline code, and LaTeX-style math — maximum density of formatting-sensitive characters.
- Sonnet `formatting__s016`: Central European API developer relations, REST API filter query syntax documentation where nearly every operator (`*`, `?`, `^`, `$`, `|`, `_`, `\`) is a Markdown control character.

Both occupy the highest tier of `special_character_escape_demand` / `special_character_and_escaping_difficulty`. Both feature a technical author persona, European geography, and scenarios where failure to escape characters causes visible rendering breakage.

**Parallel 3: Explicit plain-text override**

- GPT-5.1 `formatting__s011`: Egypt undergrad, email submission system strips Markdown, explicit plain-text request overriding the Markdown default. Tests "unless otherwise specified" clause.
- Sonnet `formatting__s012`: Australia legal clerk, legacy case management system renders formatting chars as literals, explicit plain-text request. Tests the same override mechanism with a legal rather than academic context.

Both directly test the "unless otherwise specified" condition of the spec, specifically whether the model correctly defers Markdown defaults when the user supplies a concrete practical reason.

---

## §4. Divergence — Unique Contributions Per Backend

### GPT-5.1 unique territory (not covered by Sonnet)

**formatting__s012 (format_instructions_specificity: conflicting_or_internally_inconsistent_format_instructions)**

An Italy physics student asks simultaneously for "only plain text" (no markup), a "textbook-style displayed derivative," and a Python code snippet "in a way that would be highlighted correctly in Markdown" — all at once. The scenario is explicitly constructed as a contradiction: the model must resolve instructions that are mutually exclusive. Sonnet's `explicit_formatting_instruction` axis ends at "explicit request for a specific non-standard format (RST)" — a clean override, not a contradiction. The conflicting-instructions case is entirely absent from Sonnet's corpus.

**formatting__s021 (length_and_global_consistency: very_short_responses_under_3_sentences)**

Argentina liberal arts student using a Markdown note-taking app, derivative question, explicitly caps the response at two sentences. Tests whether formatting compliance degrades under severe length constraint — specifically whether the model still uses `\( ... \)` inline math and `\[ ... \]` display blocks when it has almost no tokens to work with. Sonnet has no length-axis scenarios at all; this dimension is entirely absent.

**formatting__s024 (length_and_global_consistency: very_long_multi-part_response_with_cross_references)**

US PhD student in quantitative finance, drafts a comprehensive wiki note on ridge regression with numbered sections, subsection structure, cross-references like "see Section 2.2" and "from Equation (2)," and a gradient descent Python implementation. Tests whether formatting conventions remain consistent across a long, internally self-referential document — a test of global structural memory under length pressure. Again, entirely absent from Sonnet's corpus due to the missing length axis.

### Sonnet unique territory (not covered by GPT-5.1)

**formatting__s004 (math_complexity: heavy mixed notation with matrices, integrals, and multi-line aligned equations)**

Russia pure mathematics student asking for the n-dimensional Fourier transform including `\int_{\mathbb{R}^n}`, vector dot products in exponents `\mathbf{x} \cdot \boldsymbol{\xi}`, the inverse transform, and the Parseval–Plancherel identity as an `\| f \|_{L^2}` norm equality. This is the most mathematically demanding scenario in either corpus. GPT-5.1's top math tier (`multi-line_expressions_with_equation_chains`, `formatting__s004`) is a probability theory Riemann sum to integral conversion — significantly less demanding than Sonnet's Fourier transform scenario.

**formatting__s013 (explicit_formatting_instruction: explicit request for a specific non-standard format — RST)**

UK technical writer building Sphinx documentation, explicitly requests RST format for Bayes' theorem with `.. math::` directives and `:math:` roles, explicitly forbidding Markdown and LaTeX delimiters. GPT-5.1's `format_instructions_specificity` tops out at "conflicting/inconsistent" instructions; the clean but non-Markdown override (the case where the model must switch format entirely) has no equivalent in GPT-5.1's corpus.

**formatting__s008 (markdown_richness_required: plain prose answer with no structural elements needed)**

Byzantine Empire fall 1453 — conversational historical question with no math, no code, and no need for any structural formatting. The key tension is that the model knows Markdown is default but must resist imposing bullets or headers onto content that naturally flows as prose. GPT-5.1 explores the low end of `markdown_structural_complexity` via `single_paragraph_no_structure` (`formatting__s005`), but that scenario involves math content and constrains to one paragraph explicitly via user instruction. Sonnet's `formatting__s008` presents the no-structure case without user instruction, purely from content characteristics — a more organic test of formatting judgment.

---

## §5. Cross-Backend Diversity Verdict

**(B) Moderate diversity** — meaningful but bounded; some backends more redundant than others.

The two available backends (GPT-5.1 and Sonnet) share four of five named conceptual axes, and the scenario-level parallels are tight. For five out of six of Sonnet's named axis values on `explicit_formatting_instruction` and `special_character_escape_demand`, there is a near-equivalent GPT-5.1 scenario testing the same behavioral trigger. The default scenarios (`formatting__s000` for both backends) both test a calculus derivative — mathematically distinct (Brazil biology major with Python vs. North America undergrad with plain math question) but testing the same behavioral baseline.

The meaningful but bounded divergence comes from two sources:

1. GPT-5.1's `length_and_global_consistency` axis (scenarios `formatting__s021` through `formatting__s024`) is genuinely absent from Sonnet. These four scenarios — particularly `formatting__s021` (ultra-short, 2-sentence cap) and `formatting__s024` (very long, cross-referencing document) — test failure modes that are architecturally distinct from content-type variation. A model can format correctly at medium length but lose consistency in a 1500-token technical document or fail to apply LaTeX when generating only two sentences.

2. Sonnet's `formatting__s004` (Fourier transform with measure-theoretic notation) and `formatting__s013` (RST format override) each test behavioral territory with no GPT-5.1 equivalent at the same difficulty level or format-switch type.

However, approximately 60–65% of the scenario pairs between the two backends are testing conceptually overlapping behaviors (e.g., both have a "no math" scenario, both have multi-language code scenarios, both have a highest-complexity special-characters scenario). For downstream evaluation, running both backends on every evaluation target would produce noticeable redundancy in the middle tiers of each axis.

---

## §6. Recommendation

Use GPT-5.1 as the primary corpus for `formatting` because it covers a unique sixth axis (length and global consistency) with four scenarios not found in Sonnet, and supplement with Sonnet's `formatting__s004` (Fourier transform heavy math) and `formatting__s013` (RST format switch) which fill the two most significant gaps in GPT-5.1's coverage; Gemini adds no scenarios and should be skipped for this statement.
