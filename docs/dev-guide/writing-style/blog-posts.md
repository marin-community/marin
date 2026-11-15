# Blog Posts

Use this guide for public technical explainers and research notes. Consult the [blog exemplars](blog-exemplars.md) for examples of how Marin posts explain different kinds of work.

Give readers a concise account of what we did and learned. Keep the length proportional to the substance.

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
- **Results.** Present the evidence with concrete numbers, experimental conditions, comparisons, and figures. Generally include at least one results figure. Discuss limitations and unresolved questions where relevant.

Use straightforward, functional section headings such as "Background," "Approach," "Benchmark," "Results," or "Limitations." Add a qualifier when useful, as in "Single-TPU Benchmark." Borrow structures from the exemplars when they suit the material.

## Writing Guidelines

- **Prioritize clarity and substance.** Use direct, natural, technically precise prose. Lead paragraphs with the main takeaway. Favor intuition, concrete examples, and observed results over abstract claims or lengthy derivations.
- **Keep the post focused on a coherent question or result.** Choose the main contribution before organizing the material. Include background experiments and follow-up results only when they help explain or interpret that contribution. Don't combine loosely related findings merely because they came from the same research effort.
- **Connect the result to the research question.** Name the prior result, observation, or constraint that prompted the work. Explain why the team chose the approaches it tested when the sources give a reason. Say what outcome would have counted as an improvement. When a follow-up answers a different question, explain what prompted it and how its comparison changed.
- **Make comparisons meaningful.** Name the baseline and explain how it handles the part of the problem being changed. Explain the conceptual difference between approaches before quantifying their implementation differences. Describe what changed, how much the measured outcome improved, and under what conditions. Distinguish effects measured separately from those that weren't.
- Build explanations from concepts to implementations. When introducing a technical technique, first establish the underlying computation, why it is needed, and what makes it expensive or difficult. Then explain how the implementation works and how it differs from the baseline. Don't begin with kernel-specific details, numerical accounting, or implementation comparisons before establishing what is being computed. Favor intuition before implementation detail.
- Organize information in the order that helps readers understand the result. Introduce concepts before relying on them and use specific referents. Keep the details that support the explanation or evidence. Link to sources for the full implementation and experiment records.
- Make examples do explanatory work. Choose examples that isolate the central idea, show enough intermediate steps to make the reasoning easy to follow, and explicitly connect the example to the general technique.
- Don't introduce solutions to unexplained problems. If you're explaining how a technique avoids a numerical instability, computational cost, or other difficulty, first establish why that difficulty arises.
- Make the links between ideas explicit. When two steps in an explanation depend on each other, say how. Give readers enough context to follow the reasoning without filling in a missing step themselves.
- **Keep technical detail focused.** Explain enough to make the contribution understandable to a strong ML engineer outside the subarea. Include implementation details when they're central to the result, but don't reproduce a paper or implementation guide.
- **Link to sources where readers first need them.** Link an unfamiliar named method to its paper, design note, documentation, or implementation on first mention. Link specific runs, diagnostics, and comparisons directly where the prose first uses them, including in the lead. For run-level claims, prefer an accessible Weights & Biases run page over derived JSON checked into the repository. Link derived analyses for calculations or comparisons they document. Put the link on the phrase that names the method or evidence so readers can tell what it opens. Link Git revisions to the relevant commit or file.
- **Be candid and precise.** Discuss negative results, limitations, and unresolved questions. Distinguish observations from interpretations. Don't invent motivations, causal explanations, experimental details, or broader implications.
- **Avoid unnecessary rhetoric.** No promotional language, generic introductions, formulaic conclusions, or exaggerated significance. Don't turn modest findings into sweeping claims or drown readers in internal terminology.

Figures should always have captions.

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
