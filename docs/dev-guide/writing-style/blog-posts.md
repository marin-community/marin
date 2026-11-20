# Blog Posts

Note that this guide is specifically for agent-authored research notes, which is the main style of blog that agents should be writing. Other kinds of blog posts need not adhere to these guidelines.

Consult the [blog exemplars](blog-exemplars.md) for examples of how Marin posts explain different kinds of work.

Give readers a concise account of what we did and learned. Research Notes make useful findings public with minimal additional work. A few paragraphs can be enough to share a measured result, observation, or engineering lesson. Keep the length proportional to the substance.

## Target Audience

Write for technically sophisticated readers with a working knowledge of machine learning and the basics of modern language models. Assume roughly the background needed to take [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/), but not necessarily familiarity with everything taught in the course.

Readers should be comfortable with concepts like transformers, gradient descent, and tokens. Do not assume deep knowledge of the specific technique being discussed or firsthand experience training large language models.

Introduce specialized concepts briefly and in context. Give readers enough background to understand what was done, why it matters, and how to interpret the results, without turning the post into a tutorial.

Link liberally to relevant papers, course materials, and high-quality blog posts for readers who want more depth. Prefer authoritative, directly relevant sources; never invent citations or links.

For example, a post about a new kernel for expert parallelism should briefly explain what expert parallelism distributes and why communication becomes expensive, then link to a more detailed explanation. There's no need to explain transformers or mixture-of-experts models from first principles.

## Explanatory Outline

**Construct the explanatory outline before drafting.** After reviewing the source material, identify the central result or question the post will address. Build an outline around the argument needed to explain that result. Identify what the reader needs to understand, what we tried, what the evidence shows, and what remains uncertain. Choose the order that makes the explanation easiest to follow. Treat the research chronology and source organization as reference material. Omit findings that don't contribute to the central argument, or propose a separate post for them.

### Hypothetical Example

Does aggressive deduplication improve pretraining efficiency?

1. **Main result.** More aggressive document-level deduplication reduced duplicate training tokens by 35%. Held-out loss stayed about the same at a fixed training compute budget.
2. **Problem.** Repeated documents consume training compute without necessarily exposing the model to new information. More aggressive deduplication might improve token efficiency, but it could also remove useful examples.
3. **Approach.** Train three 1B-parameter models on datasets with different deduplication thresholds, holding architecture, token budget, optimizer settings, and data mixture constant.
4. **Evidence.** The aggressive setting removed substantially more repeated content. Final held-out losses across the three settings were 2.91, 2.90, and 2.91. Each setting ran with one seed.
5. **Interpretation and limitations.** These runs showed no consistent held-out loss improvement from more aggressive deduplication. The value of the removed documents and the effects at larger scales remain open.
6. **Out of scope.** Changes in downstream benchmark performance and comparisons among deduplication algorithms are questions for separate experiments.

## Structure

Unless inappropriate for the content, use the following structure:

- **Lead-in (no heading).** Lead with the main result. State the question the work set out to answer (or goal to be met), what was tested to answer it, and why the result matters. The question need not be expressed as a literal syntactic question. Readers should be able to understand the question and the answer from the lead alone.
- **Background.** Explain the problem, why it matters, and the most relevant prior work. Keep it brief and link out for more detail.
- **Approach.** Explain what changed relative to the baseline and why. Include examples, small Python or pseudocode diffs, or diagrams where helpful. Link to the implementation.
- **Results.** Present the evidence with concrete numbers, experimental conditions, comparisons, and figures. Generally include at least one results figure. Give every figure a caption. Discuss limitations and unresolved questions where relevant.

Use straightforward, functional section headings such as "Background," "Approach," "Benchmark," "Results," or "Limitations." Add a qualifier when useful, as in "Single-TPU Benchmark." Borrow structures from the exemplars when they suit the material.

## Writing the article

Follow the [general drafting guidance](drafting.md).

- **Connect the result to the research question.** Name the prior result, observation, or constraint that prompted the work. Explain the choice of approaches when the sources give a reason, and say what outcome would have counted as an improvement. If a follow-up answers a different question, explain what prompted it and how its comparison changed.
- **Compare approaches on the relevant workload.** Name the baseline and explain how it handles the work being changed. State the conceptual difference before giving implementation details. Give the measured change and its conditions, and identify which effects the experiment measured separately.
- **Establish the problem before the technique.** Explain what needs to be computed, why it is needed, and where the cost or difficulty arises. Use a small example to isolate the mechanism, show the steps a reader needs, and connect them to the general technique. Link to the implementation for details beyond the explanation.
- **Link methods and evidence on first mention.** Link the phrase naming a method, run, diagnostic, or comparison to its specific source, including in the lead. Prefer an accessible Weights & Biases run page for run-level claims. Use derived analyses for the calculations they document and link code revisions directly.

## Review

Apply the [general prose review](review-prose.md) and the checks below.

### Background and comparison

Read the title and opening. State the headline result and the question it
answers in your own words. Before the results, check that the draft explains
what prompted this particular comparison, why the team chose those approaches
when the sources give a reason, what it measured, and what would count as an
improvement. If a follow-up takes over the story, check that the draft explains
what prompted it and which question it answers.

Read the background through the first paragraph of the approach. Check that
the reader can explain why the baseline does the work being changed and which
property or constraint the new approach uses. If the background jumps from
definitions or dimensions to the change, add the missing connection. Move
benchmark-specific details to where they help explain the mechanism or result.

### Section headings

Read the section headings on their own as a table of contents. For each
section, use the label that matches its purpose in the
[structure above](#structure), such as “Background,” “Approach,”
“Benchmark,” “Results,” or “Limitations.” Add a short qualifier when it
helps readers choose a section, as in “Single-device benchmark.” Rewrite
headings that describe a technique or state a finding as functional labels.
For example, change “Shard the computation” to “Approach.” Keep the
technical detail in the section itself.
