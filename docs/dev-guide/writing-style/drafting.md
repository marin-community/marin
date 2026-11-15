# Drafting Marin Prose

Read this guide and the [guide for your format](index.md#read-the-right-file) before writing.

## Writing principles

**Pick an audience.** Decide early who your target audience is. Typically the format guidance will supply a default choice.

**Lead with the substance.** Establish the problem, result, or technical contribution early. Avoid generic introductions about the importance of AI or the broader research landscape.

**Explain why, not just what.** Describe the constraints, alternatives, and tradeoffs behind technical decisions. Where possible, explain why one approach succeeded or failed relative to another.

**Be specific.** Use actual numbers, configurations, experimental results, and implementation details. Link to code, experiments, issues, and prior research. Don't replace technical details with high-level summaries.

**Explain things clearly.** Introduce concepts when readers need them. Build explanations in a logical order, connecting problems to solutions and observations to conclusions. Assume technical competence without assuming familiarity with every subsystem.

**Be empirically grounded.** Distinguish measured results from estimates, hypotheses, and intuitions. Explain important limitations and uncertainty, but don't hedge unnecessarily. Never invent results or rationales to make a narrative more complete.

**Be candid.** Discuss failed experiments, unexpected outcomes, shortcomings, and decisions we might revisit. Don't turn failures into contrived success stories.

**Use straightforward language.** Prefer ordinary words and direct sentences. Technical jargon is appropriate when precise. Contractions, first-person language, and occasional informality are welcome. Don't force jokes, enthusiasm, or personality.

**Respect individual voices** when editing human-authored prose.Different authors write differently. Preserve an author's terminology, emphasis, and natural phrasing. Don't impose a uniform house voice beyond clarity, precision, and directness.

## Avoid

- Corporate or promotional prose: "groundbreaking," "game-changing," "pushing the boundaries," "exciting milestone."
- Generic filler: "In the rapidly evolving landscape of AI," "It's important to note," "Let's dive in."
- Artificial rhetorical flourishes, especially repetitive "not just X, but Y" constructions.
- Excessive enthusiasm, self-congratulation, or performative humility.
- Overselling the generality or significance of narrow experimental results.
- Vague conclusions that merely restate that the work was valuable.
- Bulleted lists where a few coherent paragraphs would be clearer. Bulleted lists are sometimes correct though.
- Excessive bolding, dramatic one-sentence paragraphs, and unnecessary transitions.
- Inventing a tidy narrative around a messy research process.

## When drafting or editing with an LLM

Prioritize fidelity to the author's technical content over stylistic polish.

Preserve meaningful distinctions, caveats, and experimental details. Do not silently fill gaps with plausible-sounding explanations. Identify missing information or unsupported claims.

Prefer concise, natural prose over elaborate exposition. Remove repetition and filler without removing substance.

When editing an existing draft, retain the author's voice. Don't rewrite sentences merely to make them sound more polished or to adhere to the stylistic guidelines here.

**The writing should feel like a researcher explaining their actual work, not an AI producing a research blog post.**

## Write in plain Technical English

Use these practices from Orwell's rules of writing and Simplified Technical
English (ASD-STE100). Write short, active sentences with common, concrete
words. Give each sentence one action or claim. Name the person or system that
acts when that helps the reader follow the work.

Use one precise term for each concept throughout the prose and figures. Keep
technical terms that improve precision. Explain unfamiliar terms before using
them to explain something else. When a source term is unclear, check what it
means before using it.

Give each paragraph a job, such as explaining a mechanism, showing a result,
or stating a limit. Preserve the author's voice when revising human-written
prose. Use judgment when a rule would make a sentence less clear or accurate.

After drafting, use the [prose review](review-prose.md).
